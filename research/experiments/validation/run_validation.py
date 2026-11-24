"""Explicitly authorized frozen validation, isolated from production experiments.

No training, no parameter selection, no silent change to existing operators.
Repeat the same command to resume. Every terminal failure is retained.
"""
import argparse,copy,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'research/full_aug_parameter_pc_full_v1_20261116'
ENGINE=ROOT/'A0B1 full/jcp_dissipation_20261024/decisive/reference_platform_v2_20261025'
sys.path[:0]=[str(ROOT),str(OLD),str(ENGINE)]
CASES=(110240101,110240102,110240104)
SEEDS=(2026,2028)
NAMES={'Aug':'DSF-Aug','Fresh-C4':'DSF-Sym','Parameter-PC':'DSF-Trend','Parameter-Hold':'DSF-Hold','Parameter-Hold-Single':'DSF-SingleHold'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    with q.open('w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def save(p,**a):
    p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    with q.open('wb') as f:np.savez(f,**a);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def identity():
    files=[Path(__file__),HERE/'acceptance.py',ROOT/'research/full_aug_hold_audit_v1_20261117/PAPER_PROTOCOL.json',
           ROOT/'research/full_aug_hold_audit_v1_20261117/independent_ic105/FROZEN.json',
           ROOT/'research/full_aug_hold_audit_v1_20261117/independent_ic105/PROVENANCE_ATTESTATION.json',
           ROOT/'research/full_aug_hold_audit_v1_20261117/independent_ic105/INITIAL_FOURIER_COEFFICIENTS.npz',
           ROOT/'research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json',
           HERE.parent/'limited_dt.py',HERE.parent/'test_limited_dt.py',ROOT/'research/full_aug_hold_audit_v1_20261117/paper_dt.py',
           ENGINE/'rp_spectral.py',ROOT/'research/skew_full_validation_20261020/core.py',ROOT/'research/skew_full_validation_20261020/reference.py']
    files+=list(OLD.glob('*.py'))+[OLD/'PROTOCOL.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}
def freeze():
    p=HERE/'FREEZE.json'
    rec=dict(authorization='User 2026-11-18: 你直接帮我都验证了; supersedes earlier no-new-simulation constraint for these frozen validations.',
        code_inputs=identity(),cases=list(CASES)+( [110240105]),seeds=list(SEEDS),primary_period=5,period_candidates=[4,5,8],
        sensitivity_methods=['Parameter-Hold','Parameter-PC'],single_hold='One orientation, same physical refresh pair times; common Aug startup. Not an equal-total-CNN-budget comparison.',
        IC105_methods=list(NAMES)[:4]+['Clark4'],DNS_variants=[dict(n=1536,dt=.000125),dict(n=3072,dt=.000125),dict(n=3072,dt=.0000625)],
        reference_gates=read(ROOT/'research/full_aug_hold_audit_v1_20261117/PAPER_PROTOCOL.json')['reference_gates'],
        references='Reuse frozen old 1536 references for 101/102/104; IC105 starts from frozen t=0 coefficients. All new DNS from native bandlimited t=0, never an evolved padded field.',
        reporting='All complete, failed and partial trajectories retained. 80 equal scoring times 1.05..5. No interpolation, no tuning, no statistical-independence claim for network seeds.',training_calls=0)
    if p.exists():assert read(p)==rec,'Frozen campaign identity changed'
    else:write(p,rec)
    acc=read(HERE/'ACCEPTANCE.json');assert acc['status']=='PASS' and acc['code_inputs']==identity(),'Adapter acceptance changed'
    return rec
def setup():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()

def dns_folder(case,n,dt):return HERE/'DNS'/f'{case}_n{n}_h{dt:.8f}'
@torch.inference_mode()
def dns(case,n,dt):
    assert case in (*CASES,110240105) and (n,dt) in ((1536,.000125),(3072,.000125),(3072,.0000625))
    from rp_spectral import SpectralRK4,Grid,labels,raw_diagnostics
    import pc_storage
    folder=dns_folder(case,n,dt);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():print('REUSE DNS',folder.name,flush=True);return
    if (folder/'FAILED.json').exists():print('TERMINAL DNS FAILURE',folder.name,flush=True);return
    cfg=read(ROOT/'research/ssc_long_horizon_v1_20261030/results/run01/reference/110240101/n1536_half/DONE.json')['config']
    cfg={k:cfg[k] for k in ('length','nu','delta','initial')}
    total=round(5/dt);every=round(.05/dt)
    run=dict(case=case,n=n,dt=dt,config=cfg,total_steps=total,freeze_sha256=sha(HERE/'FREEZE.json'),
             torch=str(torch.__version__),GPU=torch.cuda.get_device_name(),saved_times='1+.05*i, i=0..80',reference_quality='REFERENCE_NOT_CONVERGED')
    if (folder/'RUN.json').exists():assert read(folder/'RUN.json')==run
    else:write(folder/'RUN.json',run)
    with pc_storage.RunLock(folder):
        s=SpectralRK4(n,cfg['length'],cfg['nu'],'cuda');low=Grid(100,cfg['length'],'cuda',torch.float64)
        h=s.initial(case,cfg['initial'])
        if case==110240105:
            with np.load(ROOT/'research/full_aug_hold_audit_v1_20261117/independent_ic105/INITIAL_FOURIER_COEFFICIENTS.npz') as z:
                modes=z['modes'];v=h.detach().cpu().numpy()[np.ix_(modes%n,modes%n)]
                assert np.linalg.norm(v-z['omega_hat'])/np.linalg.norm(z['omega_hat'])<1e-13,'Frozen IC105 mismatch'
        cpfile=folder/'CHECKPOINT.json';start=0;frames={};native={};observations=[];slot=1;actual=0
        if cpfile.exists():
            cp=read(cpfile);assert cp['run_sha256']==sha(folder/'RUN.json')
            state=folder/f'state{cp["slot"]}.npz';assert sha(state)==cp['state_sha256']
            with np.load(state) as z:h=torch.as_tensor(z['h'],device='cuda')
            start=cp['step'];slot=cp['slot'];frames=cp['frames'];native=cp['native'];observations=cp['observations'];actual=cp['actual_RHS']
            s.max_speed.fill_(cp['max_speed'])
            assert all(sha(folder/p)==v for p,v in {**frames,**native}.items())
        log=folder/'RHS_CALLS.jsonl'
        if log.exists():
            with log.open() as f:
                for line in f:
                    if line.strip():actual=max(actual,json.loads(line)['RHS'])
        journal=log.open('a',buffering=1);original=s.rhs
        def counted(v):
            nonlocal actual
            actual+=1;journal.write(json.dumps(dict(RHS=actual))+'\n')
            return original(v)
        s.rhs=counted;begin=time.perf_counter();z0=float(.5*h.abs().square().sum());last=start
        try:
            for step in range(start,total+1):
                if step>start:h,_,_=s.step(h,dt)
                last=step
                if step%every==0:
                    stats,spec=raw_diagnostics(s,h);speed=float(s.max_speed);dx=cfg['length']/n
                    cfl=dt*(speed/dx+4*cfg['nu']/dx**2)
                    if not np.isfinite(stats['Z']) or stats['Z']>100*z0 or not np.isfinite(cfl) or cfl>.5:
                        save(folder/'FAILED_STATE.npz',h=h.detach().cpu().numpy())
                        write(folder/'FAILED.json',dict(status='FAILED_NUMERICAL',step=step,time_s=step*dt,CFL=cfl,actual_RHS=actual,**stats));return
                    if step>=round(1/dt):
                        coarse=round((step*dt-1)/.005);name=f'frame_{coarse:04d}.npz'
                        if name not in frames:
                            lab=labels(s,low,h,cfg['delta'],cfg['nu'],legacy=False)
                            gap=lab['reference_source_rhs_gap'];flux=lab['reference_source'];scale=max(np.linalg.norm(gap),1e-12)
                            discrepancy=float(np.linalg.norm(gap-flux)/scale)
                            mean=float(abs(gap.mean())/max(np.sqrt(np.mean(gap**2)),1e-12))
                            assert discrepancy<=1e-10 and mean<=1e-10 and all(np.isfinite(a).all() for a in lab.values()),'Reference operator identity'
                            lab.update(reference_source=gap,flux_source=flux,time_s=np.array(step*dt))
                            save(folder/name,**lab);frames[name]=sha(folder/name)
                            cut=(n-1)//3;modes=np.arange(-cut,cut+1)
                            hn=h.index_select(0,torch.as_tensor(modes%n,device='cuda')).index_select(1,torch.as_tensor(modes%n,device='cuda')).cpu().numpy()
                            nn=f'native_{coarse:04d}.npz';save(folder/nn,h=hn,modes=modes,n=np.array(n),time_s=np.array(step*dt),spectrum=spec)
                            native[nn]=sha(folder/nn)
                            stats.update(source_flux_relative=discrepancy,source_mean_relative=mean)
                    if not observations or observations[-1]['step']!=step:observations.append(dict(step=step,time_s=step*dt,CFL=cfl,**stats))
                    if step>start or not cpfile.exists():
                        slot=1-slot;state=folder/f'state{slot}.npz';save(state,h=h.detach().cpu().numpy())
                        journal.flush();os.fsync(journal.fileno())
                        write(cpfile,dict(run_sha256=sha(folder/'RUN.json'),step=step,slot=slot,state_sha256=sha(state),frames=frames,native=native,observations=observations,actual_RHS=actual,max_speed=speed))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                    write(folder/'STATUS.json',dict(status='RUNNING',step=step,total_steps=total,time_s=step*dt,completed_t_end=step*dt,actual_RHS=actual,session_seconds=elapsed,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
                    print(f'DNS {case%1000}/{n}/{dt} t={step*dt:.2f}/5 RHS={actual} session={elapsed:.1f}s',flush=True)
            assert len(frames)==81 and len(native)==81
            write(folder/'DONE.json',dict(status='OPERATOR_CHECKS_PASS',reference_quality='REFERENCE_NOT_CONVERGED',completed_t_end=5.,frames=frames,native=native,observations=observations,actual_RHS=actual,
                max_native_tail_fraction=max(o['tail_fraction'] for o in observations),max_CFL=max(o['CFL'] for o in observations),run_sha256=sha(folder/'RUN.json'),session_seconds=time.perf_counter()-begin))
            write(folder/'STATUS.json',dict(status='COMPLETE_UNQUALIFIED',completed_t_end=5.,actual_RHS=actual))
        except BaseException as ex:
            write(folder/f'INTERRUPTION_{time.time_ns()}.json',dict(status='INTERRUPTED',last_attempted_step=last,actual_RHS=actual,error=str(ex),traceback=traceback.format_exc()));raise
        finally:journal.flush();os.fsync(journal.fileno());journal.close()

def bind(period=5,single=False):
    import pc_common as c,pc_engine as e,driver as d
    c.HERE=HERE;c.P=copy.deepcopy(c.P);c.P['hard_budget']={'CNN':1000000,'RHS':1000000}
    class Bank(e.Bank):
        def observe(self,n,a):
            c.require(n%period in (0,1),'Unscheduled refresh')
            if n%period==0:self.pending=a
            else:
                c.require(self.pending is not None and self.pending['step']==n-1,'Incomplete refresh pair')
                self.active=[self.pending,a];self.pending=None
        def predict(self,n,fraction,method):return super().predict(n,fraction,'Parameter-Hold' if 'Hold' in method else method)
    class Engine(e.Engine):
        def orbit(self,w):
            if not(single and self.method.startswith('Parameter')):return super().orbit(w)
            f=self.original(self.s.state(w));c.q.physical(self.s,self.s.state(w),f)
            return dict(original=f,heads=self.captured_raw[None].repeat(4,1,1,1),d=e.vec(f,'qd'),b=e.vec(f,'qb'))
        def prepare(self,n,w,bank):
            self.n=n;self.stage=0;self.bank=bank;self.records=[];self.values=[];self.snapshots=[];self.cache=None;self.fresh=None
            if self.method.startswith('Parameter') and n%period in (0,1) and not(n==1 and bank.active[-1]['step']==1):
                self.j.phase='refresh';f=self.orbit(w);a=e.save_anchor(self.root,self.folder/'anchors',n,f['heads']);bank.observe(n,a);self.cache=f['original'];self.fresh=f
            self.frozen_descriptor=bank.descriptor() if bank else None
        def flux(self,st):
            if not(single and self.method.startswith('Parameter')):return super().flux(st)
            h,tau=self.bank.predict(self.n,self.fractions[self.stage],self.method)
            orig=self.original;method=self.method
            def held(ss):self.captured_raw=h[0];return self.decode(ss,h[0])
            try:
                self.original=held;self.method='Aug';out=super().flux(st);self.last_check['tau']=tau
                return out
            finally:self.original=orig;self.method=method
    d.Engine=Engine;d.Bank=Bank
    def load_bank(root,desc,device):return Bank([e.load_anchor(root,a,device) for a in desc['active']],e.load_anchor(root,desc['pending'],device) if desc['pending'] else None)
    d.load_bank=load_bank
    return c,e,d,Engine,Bank

@torch.inference_mode()
def les(group):
    import pc_storage
    period=4 if group=='period4' else 8 if group=='period8' else 5;single=group=='single'
    c,e,d,Engine,Bank=bind(period,single);inventory=c.inventory();cases=CASES
    if group=='ic105':
        cases=(110240105,);f=dns_folder(110240105,1536,.000125);done=read(f/'DONE.json');assert done['completed_t_end']==5.
        assert all(sha(f/n)==v for n,v in done['frames'].items())
        inventory['references']['110240105']=dict(kind='frames',frames={str(n):str((f/f'frame_{n:04d}.npz').relative_to(ROOT)) for n in range(0,801,10)},steps=list(range(0,801,10)))
    root=HERE/'LES'/group;root.mkdir(parents=True,exist_ok=True)
    if not (root/'OPERATORS.npz').exists():shutil.copy2(OLD/'results/run01/OPERATORS.npz',root/'OPERATORS.npz')
    rec=dict(freeze_sha256=sha(HERE/'FREEZE.json'),group=group,period=period,single=single,inventory=inventory,runtime=d.runtime_identity('cuda'))
    if (root/'RUN.json').exists():assert read(root/'RUN.json')==rec
    else:write(root/'RUN.json',rec)
    methods=('Parameter-Hold-Single',) if single else ('Parameter-Hold','Parameter-PC') if group.startswith('period') else ('Aug','Fresh-C4','Parameter-PC','Parameter-Hold')
    with pc_storage.RunLock(root):
        j=c.Journal(root/'invocations')
        try:
            for case in cases:
                for seed in SEEDS:
                    if case in CASES:
                        src=OLD/'results/run01/startup'/f'{case}_{seed}';dst=root/'startup'/src.name
                        if not dst.exists():
                            assert src.is_dir() and dst.resolve().is_relative_to(root)
                            shutil.copytree(src,dst)
                            write(root/'IMPORTED_STARTUP'/f'{src.name}.json',dict(source=str(src.relative_to(ROOT)),files={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()}))
                    row=next(r for r in inventory['checkpoints'] if r['seed']==seed)
                    ref=c.References(inventory['references'][str(case)])
                    try:start,_=d.startup(root,case,seed,row,ref,j,'cuda')
                    except e.TrajectoryFailure as ex:
                        for method in methods:write(root/'trajectories'/c.key(case,seed,method)/'STATUS.json',dict(status=ex.status,reason='Common startup: '+str(ex),completed_t_end=1.))
                        continue
                    shift=(list(cases).index(case)+list(SEEDS).index(seed))%len(methods);order=methods[shift:]+methods[:shift]
                    for method in order:d.run_trajectory(root,case,seed,method,row,start,j,'cuda')
            write(root/'EXECUTION.json',dict(status='ALL_SCHEDULED_TRAJECTORIES_TERMINAL',actual_calls=j.used,per_scope=j.per_key,reference_quality=c.QUALITY))
        except BaseException as ex:
            write(root/f'INTERRUPTION_{time.time_ns()}.json',dict(error=str(ex),actual_calls=j.used,traceback=traceback.format_exc()));raise
        finally:j.close()
    if group=='ic105':clark((110240105,),inventory)

@torch.inference_mode()
def clark(cases=CASES,inventory=None):
    c,e,d,E,B=bind()
    sys.path.insert(0,str(HERE.parent))
    import limited_dt as limited
    inventory=inventory or c.inventory()
    for case in cases:
        if case==110240105:
            f=dns_folder(case,1536,.000125);assert (f/'DONE.json').exists()
            omega=c.load(f/'frame_0000.npz')['reference_omega']
        else:omega=c.References(inventory['references'][str(case)]).at(0)['reference_omega']
        root=HERE/'CLARK'/str(case);root.mkdir(parents=True,exist_ok=True)
        rec=dict(case=case,model='Clark4',dt=.005,network_seed=None,initial_sha256=hashlib.sha256(np.asarray(omega).tobytes()).hexdigest(),freeze_sha256=sha(HERE/'FREEZE.json'))
        if (root/'RUN.json').exists():assert read(root/'RUN.json')==rec
        else:write(root/'RUN.json',rec)
        import pc_storage
        with pc_storage.RunLock(root):
            j=c.Journal(root/'invocations')
            try:limited.run_clark(root,dict(omega0=omega),.005,j,'cuda')
            finally:j.close()

def all_tasks():
    # Separate processes prevent adapter state and GPU allocations leaking across jobs.
    tasks=[['clark'],['les','--group','single'],['les','--group','period4'],['les','--group','period8'],
           ['dns','--case','110240105','--n','1536','--dt','0.000125'],['les','--group','ic105']]
    for case in (110240105,*CASES):
        for dt in (.000125,.0000625):tasks.append(['dns','--case',str(case),'--n','3072','--dt',str(dt)])
    records=[]
    for task in tasks:
        write(HERE/'CAMPAIGN_STATUS.json',dict(status='RUNNING',current=task,completed=records))
        p=subprocess.run([sys.executable,'-B',str(Path(__file__)),*task,'--execute'])
        records.append(dict(task=task,returncode=p.returncode))
        if p.returncode:write(HERE/'CAMPAIGN_STATUS.json',dict(status='INTERRUPTED',current=task,completed=records));return p.returncode
    write(HERE/'CAMPAIGN_STATUS.json',dict(status='ALL_SCHEDULED_COMMANDS_TERMINAL',completed=records))
    return 0
def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['all','dns','les','clark']);ap.add_argument('--execute',action='store_true');ap.add_argument('--case',type=int);ap.add_argument('--n',type=int);ap.add_argument('--dt',type=float);ap.add_argument('--group',choices=['single','period4','period8','ic105'])
    a=ap.parse_args();assert a.execute,'Explicit execution flag required';freeze();setup()
    if a.cmd=='dns':dns(a.case,a.n,a.dt)
    elif a.cmd=='les':les(a.group)
    elif a.cmd=='clark':clark()
    else:sys.exit(all_tasks())
if __name__=='__main__':main()
