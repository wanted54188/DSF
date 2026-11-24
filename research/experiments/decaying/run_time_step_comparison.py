"""Only complete IC101/2026 Trend/Hold and deterministic Clark4. Manual entry.

The frozen paper_dt.py and RUN.json remain unchanged so existing restart
identity is preserved. The selection is deliberately not configurable.
"""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PRIOR=ROOT/'research/full_aug_hold_audit_v1_20261117'
sys.path.insert(0,str(PRIOR))
import paper_dt as p
c,e,d,storage=p.c,p.e,p.d,p.storage
CASE=110240101;SEED=2026
METHODS=('Parameter-PC','Parameter-Hold')
FROZEN_ROOT=PRIOR/'paper_runs/dt_half01'
PLAN=dict(case=CASE,network_seed=SEED,continue_methods=list(METHODS),reuse_methods=['Aug','Fresh-C4'],
          clark_dt=[.005,.0025],window=[1,5],refresh_pair_times='1+.025k,1+.025k+.005',
          additional_call_limit={'CNN':3000,'RHS':24000},reference_quality=c.QUALITY,
          clark_start='pure Clark4 from shared t1 omega0; no neural startup; deterministic single sample')

def fingerprint():return {f.name:c.sha(f) for f in (Path(__file__),HERE/'test_limited_dt.py')}
def validate_identity(root):
    """Read-only preflight. Does not load a network or call inventory on other cases."""
    run=c.read(root/'RUN.json')
    c.require(run['code_identity']==c.local_identity(),'Historical fine driver changed; do not bypass SHA gate')
    c.require(run['protocol']==c.P,'Historical protocol changed')
    c.require(run['runtime']==d.runtime_identity('cuda'),'Resume runtime/GPU differs')
    inv=run['inventory'];row=next(r for r in inv['checkpoints'] if r['seed']==SEED)
    selected={row[k] for k in ('file','run','done')}
    desc=inv['references'][str(CASE)];selected.update(desc['fields'].values());selected.add(desc['time_path'])
    selected.update(k for k in inv['input_sha256'] if Path(k).suffix in ('.py','.json'))
    for name in sorted(selected):
        c.require(c.sha(ROOT/name)==inv['input_sha256'][name],'Frozen input changed: '+name)
    for method in ('Aug','Fresh-C4'):
        folder=root/'trajectories'/c.key(CASE,SEED,method);status=c.read(folder/'STATUS.json')
        c.require(status['status']=='VERIFIED' and status['last_accepted_step']==800,'Completed baseline missing')
        cp=c.read(folder/'CHECKPOINT.json');c.require(c.sha(folder/cp['file'])==cp['sha256'],'Completed baseline SHA mismatch')
    c.require((root/'startup'/f'{CASE}_{SEED}'/'DONE.json').exists(),'Missing shared startup; this entry never recomputes startup')
    return run,row,c.References(desc)

class LimitedJournal(c.Journal):
    def __init__(self,folder):
        super().__init__(folder);self.base=dict(self.used)
    def tick(self,kind):
        allowed={c.key(CASE,SEED,v) for v in METHODS}|{'Clark4_0.005','Clark4_0.0025'}
        c.require(self.scope in allowed,'Refusing unrelated trajectory '+self.scope)
        c.require(self.used[kind]-self.base[kind]<PLAN['additional_call_limit'][kind],'Limited invocation call budget exceeded')
        super().tick(kind)

def clark_solver(device):
    s=c.q.study.make_solver(c.q.study.AUGMENTED,SEED).double().to(device).eval().requires_grad_(False)
    def forbidden(*_):raise AssertionError('Clark4 must never call CNN')
    s.closure.register_forward_pre_hook(forbidden)
    def closure(st):
        qx,qy=c.q.study.common.prior.c4.flux4(s,st,s.closure.filter_width)
        return dict(qx=qx,qy=qy)
    s.closure_fields=closure
    return s

def clark_step(s,w,t,h,j,initial):
    """Native RK4 plus the same inherited finite/K/Z safety bounds at each stage."""
    native=s.rhs;values=[]
    def rhs(w,t):
        j.tick('RHS');f=native(w,t)
        stats=e.basic_metrics(s,w)
        if not bool(torch.isfinite(f['rhs']).all()) or stats['K']>100*initial['K'] or stats['Z']>100*initial['Z']:
            raise e.TrajectoryFailure('FAILED_NUMERICAL','Clark4 finite/K/Z stage gate',dict(omega=e.cpu(w),rhs=e.cpu(f['rhs'])),dict(time_s=t,stage=len(values)+1))
        values.append(f['rhs'].clone());return f
    s.rhs=rhs
    try:out=s.step(w,t,h)
    finally:s.rhs=native
    stats=e.basic_metrics(s,out)
    if not bool(torch.isfinite(out).all()) or stats['K']>100*initial['K'] or stats['Z']>100*initial['Z']:
        raise e.TrajectoryFailure('FAILED_NUMERICAL','Clark4 endpoint gate',dict(omega=e.cpu(out)),dict(time_s=t+h))
    return out,torch.stack(values)

def run_clark(root,start,h,j,device,stop=800):
    folder=root/'limited_completion'/f'Clark4_{h}';folder.mkdir(parents=True,exist_ok=True);j.scope=f'Clark4_{h}'
    if (folder/'STATUS.json').exists():
        status=c.read(folder/'STATUS.json')
        if status['status']=='VERIFIED':
            cp=c.read(folder/'CHECKPOINT.json')
            c.require(cp['accepted_step']==800 and c.sha(folder/cp['file'])==cp['sha256'],'Completed Clark4 SHA mismatch')
            return
        if status['status']=='FAILED_NUMERICAL':return
    s=clark_solver(device);op=c.operators(s);w=torch.tensor(start['omega0'],device=device);initial=e.basic_metrics(s,w)
    n=0;compute=0.;cpfile=folder/'CHECKPOINT.json'
    # Commit pointer stores the entire safe state. Immutable step transactions
    # can be recovered if the process died between transaction and pointer.
    files=sorted((folder/'steps').glob('*.npz'))
    for expected,file in enumerate(files,1):
        with np.load(file) as z:info=c.meta(z)
        c.require(info['accepted_step']==expected,'Clark4 restart gap')
    if cpfile.exists():
        cp=c.read(cpfile);c.require(cp['accepted_step']<=len(files) and c.sha(folder/cp['file'])==cp['sha256'],'Clark4 pointer mismatch')
    if files:
        z=c.load(files[-1]);n=c.meta(z)['accepted_step'];w=torch.tensor(z['omega'],device=device);compute=c.meta(z)['compute_seconds']
        c.dump(cpfile,dict(accepted_step=n,file=str(files[-1].relative_to(folder)),sha256=c.sha(files[-1])))
    def snapshot(w,n):
        native=s.rhs;j.tick('RHS');f=native(w,1+n*c.H)
        source=-f['div_q'];return dict(omega=e.cpu(w),rhs=e.cpu(f['rhs']),Source=e.cpu(source),qx=e.cpu(f['qx']),qy=e.cpu(f['qy']))
    if not (folder/'INITIAL.npz').exists():c.npz(folder/'INITIAL.npz',**snapshot(torch.tensor(start['omega0'],device=device),0))
    for step in range(n,stop):
        attempt=c.read(folder/'ATTEMPT.json') if (folder/'ATTEMPT.json').exists() else {}
        replay=attempt.get('replays',0)+int(attempt.get('step')==step)
        c.require(replay<=2,'Clark4 uncommitted replay limit exceeded')
        c.dump(folder/'ATTEMPT.json',dict(step=step,replays=replay))
        before=w.clone();c.sync(device);begin=time.perf_counter();ks=[]
        try:
            for half in range(round(c.H/h)):
                w,k=clark_step(s,w,1+step*c.H+half*h,h,j,initial);ks.extend(k)
        except e.TrajectoryFailure as err:
            c.npz(folder/'FAILED_STAGE.npz',**err.arrays,metadata=c.packed(dict(reason=str(err),details=err.details)))
            c.dump(folder/'STATUS.json',dict(status=err.status,reason=str(err),details=err.details));return
        c.sync(device);compute+=time.perf_counter()-begin
        rhs=e.cpu(torch.stack(ks));p.verify(e.cpu(before),e.cpu(w),rhs,op)
        observation=snapshot(w,step+1) if (step+1)%10==0 else {}
        file=folder/'steps'/f'{step+1:04d}.npz'
        c.npz(file,omega=e.cpu(w),rhs=rhs,metadata=c.packed(dict(accepted_step=step+1,compute_seconds=compute,h=h)),**{'observation__'+k:v for k,v in observation.items()})
        c.dump(cpfile,dict(accepted_step=step+1,file=str(file.relative_to(folder)),sha256=c.sha(file)))
        c.dump(folder/'STATUS.json',dict(status='VERIFIED' if step+1==800 else 'RUNNING',last_accepted_step=step+1,CNN=0,actual_calls=getattr(j,'per_key',{}).get(j.scope,{}),compute_seconds=compute,resource=c.resource_memory(device),reference_quality=c.QUALITY))
        if (step+1)%10==0:print(f'Clark4 h={h}: t={1+(step+1)*c.H:.2f}/5',flush=True)

def summarize(root,run):
    """81 frames only; never reconstruct all trajectories or run a CNN."""
    op=c.load(root/'OPERATORS.npz');ref=c.References(run['inventory']['references'][str(CASE)])
    coarse=ROOT/'research/full_aug_parameter_pc_full_v1_20261116/results/run01'
    latest=c.read(coarse/'LATEST_REPORT.json');report=coarse/latest['path'];manifest=c.read(report/'MANIFEST.json')
    c.require(c.sha(report/'MANIFEST.json')==latest['manifest_sha256'],'Coarse report identity changed')
    dest=root/'limited_completion/reports'/str(time.time_ns());dest.mkdir(parents=True)
    records=[];drift=[];spectra=[];budgets=[];statuses=[];checks=[]
    def score(w,snap,method,h,n):
        ctx=dict(case=CASE,network_seed=SEED if method!='Clark4' else '',method=method,h=h,step=n,time_s=1+n*c.H)
        records.extend(p.m.state_rows(w,ref.at(n)['reference_omega'],op,ctx))
        if snap:
            terms=dict(SGS=snap['Source'],RHS=snap['rhs'])
            st=p.m.fields(w,op);terms.update(advection=-p.m.project(st['u']*st['gx']+st['v']*st['gy'],op),viscosity=p.m.project(float(op['nu'])*st['lap'],op))
            if 'Sd' in snap:terms.update(Sd=snap['Sd'],Sb=snap['Sb'])
            masks,shell=p.m.geometry(op);hw=p.m.F(w)
            vals=dict(E=.5*abs(hw)**2*op['inverse'],Z=.5*abs(hw)**2)
            for name,v in terms.items():
                z=(hw.conj()*p.m.F(v)).real;vals['T_Z_'+name]=z;vals['T_K_'+name]=z*op['inverse']
                for band,mask in masks.items():budgets.append(dict(**ctx,band=band,term=name,Z_rate=float(z[mask].sum()),K_rate=float((z*op['inverse'])[mask].sum())))
            spectra.extend(dict(**ctx,shell=k,**{name:float(v[masks['all']&(shell==k)].sum()) for name,v in vals.items()}) for k in range(48))
    for method in ('Aug','Fresh-C4',*METHODS):
        folder=root/'trajectories'/c.key(CASE,SEED,method);status=c.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else dict(status='NOT_RUN')
        statuses.append(dict(method=method,h=.0025,**status))
        name=f'STATES_{CASE}_{SEED}_{method}.npy';c.require(c.sha(report/name)==manifest['outputs'][name],'Coarse states changed')
        old=np.load(report/name,mmap_mode='r');start=c.load(root/'startup'/f'{CASE}_{SEED}'/'STARTUP.npz')
        for n in range(0,801,10):
            score(old[n],None,method,.005,n)
            if n==0:w=start['omega0'];snap=storage.observation(start)
            else:
                file=folder/'steps'/f'{n:04d}.npz'
                if not file.exists():continue
                with np.load(file) as z:w=z['omega']
                obs=folder/'ENDPOINT.npz' if n==800 else folder/'steps'/f'{n+1:04d}.npz'
                snap=c.load(obs) if n==800 and obs.exists() else storage.observation(c.load(obs)) if obs.exists() else None
            score(w,snap,method,.0025,n)
            drift.append(dict(method=method,step=n,time_s=1+n*c.H,state_MSE=float(np.mean((w-old[n])**2))))
    for h in (.005,.0025):
        folder=root/'limited_completion'/f'Clark4_{h}';statuses.append(dict(method='Clark4',h=h,**(c.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else dict(status='NOT_RUN'))))
        for n in range(0,801,10):
            file=folder/'INITIAL.npz' if n==0 else folder/'steps'/f'{n:04d}.npz'
            if file.exists():
                z=c.load(file);snap=z if n==0 else storage.observation(z);score(z['omega'],snap,'Clark4',h,n)
    summaries=[]
    for method in ('Aug','Fresh-C4',*METHODS,'Clark4'):
        for h in (.005,.0025):
            for band in ('all','low','mid','high','corner'):
                for window,steps in dict(full=range(10,801,10),early=range(40,241,10),late=range(250,801,10),t5=[800]).items():
                    rr=[r for r in records if r['method']==method and r['h']==h and r['band']==band and r['step'] in steps]
                    ok=len(rr)==len(steps)
                    summaries.append(dict(method=method,h=h,band=band,window=window,status='VERIFIED' if ok else 'INCOMPLETE',samples=len(rr),**{k:float(np.mean([r[k] for r in rr])) if ok else None for k in ('omega_MSE','u_MSE','v_MSE','K','Z','K_relative_bias','Z_relative_bias')}))
    for name,rr in dict(STATE_METRICS=records,WINDOW_SUMMARY=summaries,DT_STATE_DRIFT=drift,SPECTRA=spectra,PHYSICAL_BUDGETS=budgets,STATUS=statuses).items():c.csvout(dest/(name+'.csv'),rr)
    costs=[dict(method=r['method'],h=r['h'],status=r['status'],actual_CNN=r.get('actual_calls',{}).get('CNN',r.get('CNN')),actual_RHS=r.get('actual_calls',{}).get('RHS'),solver_seconds=r.get('solver_wall_seconds',r.get('compute_seconds')),IO_seconds=r.get('IO_wall_seconds'),wall_scope='solver only; interrupted session wall not reconstructed') for r in statuses]
    c.csvout(dest/'COSTS.csv',costs)
    paired=[]
    for r in summaries:
        if r['status']!='VERIFIED':continue
        for b in summaries:
            if b['status']=='VERIFIED' and (r['band'],r['window'])==(b['band'],b['window']) and ((r['method']==b['method'] and r['h']!=b['h']) or (r['h']==b['h'] and r['method']!=b['method'])):
                paired.append(dict(candidate=r['method'],baseline=b['method'],candidate_h=r['h'],baseline_h=b['h'],band=r['band'],window=r['window'],candidate_MSE=r['omega_MSE'],baseline_MSE=b['omega_MSE'],improvement_pct=100*(1-r['omega_MSE']/b['omega_MSE']) if b['omega_MSE'] else None))
    c.csvout(dest/'PAIRED_MSE.csv',paired)
    (dest/'REPORT_CN.md').write_text('仅101/2026的两步长敏感性；非独立泛化或收敛阶证明。\nClark4从共同t1初态纯解析运行，不含共享Aug启动；Clark4不计网络seed样本。\nSUMMARY/PAIRED_MSE记录统一80帧主窗，早窗1.2..2.2、晚窗>2.2、终点5；不完整窗不评分。\n谱/预算来自细步保存Source；粗步神经方法这里只重评分状态，不伪造粗步Source。\nREFERENCE_NOT_CONVERGED。\n',encoding='utf-8')
    c.dump(root/'limited_completion/LATEST_REPORT.json',dict(path=str(dest.relative_to(root)),reference_quality=c.QUALITY))
    print('Report:',dest,flush=True)

@torch.inference_mode()
def execute(root):
    run,row,ref=validate_identity(root)
    acceptance=c.read(HERE/'LIMITED_DT_ACCEPTANCE.json');c.require(acceptance['code_identity']==fingerprint() and acceptance['status']=='PASS','Run CPU acceptance after code edits')
    dest=root/'limited_completion';dest.mkdir(exist_ok=True)
    identity=dict(plan=PLAN,code_identity=fingerprint(),original_RUN_sha256=c.sha(root/'RUN.json'))
    with storage.RunLock(root):
        if (dest/'PLAN.json').exists():c.require(c.read(dest/'PLAN.json')==identity,'Limited resume identity changed')
        else:c.dump(dest/'PLAN.json',identity)
        # No startup, Aug/Sym or unrelated case can be scheduled through this path.
        start,_=d.startup(root,CASE,SEED,row,ref,None,'cuda')
        j=LimitedJournal(root/'invocations')
        try:
            for method in METHODS:d.run_trajectory(root,CASE,SEED,method,row,start,j,'cuda')
            for h in (.005,.0025):run_clark(root,start,h,j,'cuda')
        except BaseException as ex:
            c.dump(dest/f'INTERRUPTION_{time.time_ns()}.json',dict(error=str(ex),traceback=traceback.format_exc()));raise
        finally:j.close()
        summarize(root,run)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute-limited',action='store_true');ap.add_argument('--rebuild-only',action='store_true');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    print(json.dumps(PLAN,ensure_ascii=False,indent=2),flush=True)
    if not (a.execute_limited or a.rebuild_only or a.check_only):print('NOT_RUN: manual --execute-limited required');return
    p.bind()
    if a.check_only:validate_identity(FROZEN_ROOT);print('PASS: existing run identity, selected inputs and finished baselines; 0 CNN/0 integration');return
    if a.rebuild_only:
        with storage.RunLock(FROZEN_ROOT):summarize(FROZEN_ROOT,c.read(FROZEN_ROOT/'RUN.json'))
        return
    c.require(torch.cuda.is_available(),'CUDA required');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    execute(FROZEN_ROOT)
if __name__=='__main__':main()
