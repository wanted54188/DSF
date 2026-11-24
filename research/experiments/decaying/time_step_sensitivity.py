"""Explicit manual whole-window h/2 replay; old modules remain read-only."""
import sys,json,time,argparse,copy
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'research/full_aug_parameter_pc_full_v1_20261116'
sys.path.insert(0,str(OLD))
import pc_common as c
import pc_engine as e
import pc_storage as storage
import pc_metrics as m
import driver as d
BASE_VERIFY=m.verify_step
PLAN=json.loads((HERE/'PAPER_PROTOCOL.json').read_text(encoding='utf-8'))

class FineEngine(e.Engine):
    def prepare(self,*args):
        super().prepare(*args);self.fractions=(0.,.5,.5,1.);self.half_index=0
    def rhs(self,w,t):
        f=super().rhs(w,t);self.records[-1]['native_half']=self.half_index;return f
    def advance(self,n,w,bank):
        c.sync(w.device);begin=time.perf_counter();diag=self.diagnostic_seconds;nn=self.network_seconds
        self.prepare(n,w,bank);frozen=bank.descriptor() if bank else None
        for half,fracs in enumerate(((0.,.25,.25,.5),(.5,.75,.75,1.))):
            self.stage=0;self.half_index=half;self.fractions=fracs
            if half:self.fresh=None
            w=self.s.step(w,1+c.H*(n+half*.5),c.H/2)
            if not bool(torch.isfinite(w).all()):raise e.TrajectoryFailure('FAILED_NUMERICAL','Nonfinite half endpoint',dict(omega=e.cpu(w)))
            stats=e.basic_metrics(self.s,w)
            if stats['K']>100*self.initial['K'] or stats['Z']>100*self.initial['Z']:
                raise e.TrajectoryFailure('FAILED_NUMERICAL','Inherited K/Z bound at half endpoint',dict(omega=e.cpu(w)),stats)
        c.require(len(self.values)==8 and len(self.records)==8 and (not bank or bank.descriptor()==frozen),'Half integration/history mismatch')
        c.sync(w.device)
        return w,dict(solver_wall_seconds=time.perf_counter()-begin,
            rhs_wall_seconds_including_checks=self.diagnostic_seconds-diag,CNN_wall_seconds=self.network_seconds-nn)

def verify(w,new,ks,op,h=.005):
    if len(ks)==4:return BASE_VERIFY(w,new,ks,op,h=h)
    assert len(ks)==8
    hh=h/2;mid=m.project(w+hh/6*(ks[0]+2*ks[1]+2*ks[2]+ks[3]),op)
    a=BASE_VERIFY(w,mid,ks[:4],op,h=hh);b=BASE_VERIFY(mid,new,ks[4:],op,h=hh)
    return {**{'half1__'+k:v for k,v in a.items()},**{'half2__'+k:v for k,v in b.items()}}

def scan(folder):
    records=[]
    for expected,path in enumerate(sorted((folder/'steps').glob('*.npz')),2):
        z=c.load(path);info=c.meta(z)
        c.require(info['accepted_step']==expected and path.stem==f'{expected:04d}' and z['rhs'].shape==(8,100,100),'Invalid fine restart')
        records.append((path,info))
    cp=folder/'CHECKPOINT.json'
    if cp.exists():
        v=c.read(cp);c.require(v['accepted_step']<=len(records)+1,'Restart pointer ahead')
        c.require(c.sha(folder/'steps'/f'{v["accepted_step"]:04d}.npz')==v['sha256'],'Restart SHA mismatch')
    if records:
        p,v=records[-1];c.dump(cp,dict(accepted_step=v['accepted_step'],file=str(p.relative_to(folder)),sha256=c.sha(p)))
    return records

BASE_FAILURE=storage.write_failure
def failure(folder,error,engine,root,step):
    if engine.stage<4:
        error.details.setdefault('stage_override',f'half{engine.half_index+1}/stage{engine.stage+1}')
        error.details.setdefault('time_override',1+c.H*(step+engine.fractions[engine.stage]))
    BASE_FAILURE(folder,error,engine,root,step)

def bind():
    # In-memory adapter only; no original source/config writes.
    original_identity=c.local_identity()
    c.HERE=HERE;c.P=copy.deepcopy(c.P)
    c.P['hard_budget']={'CNN':PLAN['full_window_dt']['CNN_hard_limit'],'RHS':PLAN['full_window_dt']['RHS_hard_limit']}
    c.P['full_window_refinement']=PLAN['full_window_dt']
    def identity():
        return {**{'original/'+k:v for k,v in original_identity.items()},
            **{p.name:c.sha(p) for p in HERE.glob('*.py')},'PAPER_PROTOCOL.json':c.sha(HERE/'PAPER_PROTOCOL.json')}
    c.local_identity=identity;d.Engine=FineEngine;storage.scan=scan;storage.write_failure=failure;m.verify_step=verify

def comparison(root):
    import pc_report
    out=pc_report.build_report(root)
    coarse=ROOT/'research/full_aug_parameter_pc_full_v1_20261116/results/run01'
    oldreport=coarse/c.read(coarse/'LATEST_REPORT.json')['path'];oldmanifest=c.read(oldreport/'MANIFEST.json')
    fine=c.csvread(out/'WINDOW_SUMMARY.csv');old=c.csvread(oldreport/'WINDOW_SUMMARY.csv')
    idx={(r['case'],r['seed'],r['method'],r['band'],r['window']):r for r in old};rows=[]
    for r in fine:
        key=tuple(r[k] for k in ('case','seed','method','band','window'));v=idx.get(key)
        if r['status']!='VERIFIED' or not v or v['status']!='VERIFIED':continue
        rows.append(dict(zip(('case','seed','method','band','window'),key),
            coarse_MSE=float(v['mean_omega_MSE']),fine_MSE=float(r['mean_omega_MSE']),
            fine_minus_coarse_MSE=float(r['mean_omega_MSE'])-float(v['mean_omega_MSE']),
            reference_quality=c.QUALITY))
    c.csvout(out/'FULL_WINDOW_DT_SCORE_DRIFT.csv',rows)
    state=[]
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            for method in c.METHODS:
                name=f'STATES_{c.key(case,seed,method)}.npy'
                if not (out/name).exists() or not (oldreport/name).exists():continue
                c.require(c.sha(oldreport/name)==oldmanifest['outputs'][name],'Coarse states changed')
                a=np.load(out/name,mmap_mode='r');b=np.load(oldreport/name,mmap_mode='r')
                for n in range(0,min(len(a),len(b)),10):
                    state.append(dict(case=case,seed=seed,method=method,time_s=1+n*c.H,
                        coarse_fine_state_MSE=float(np.mean((a[n]-b[n])**2))))
    c.csvout(out/'FULL_WINDOW_DT_STATE_DRIFT.csv',state)
    (out/'FULL_WINDOW_DT_CN.md').write_text('全窗dt=.0025自主复算，四方法均使用自己的细步轨迹；物理刷新时刻与历史单位保持原.005协议。\n'
        'FULL_WINDOW_DT_SCORE_DRIFT.csv给出全窗、早/晚窗、终点绝对MSE漂移；PAIRED_LONG_SUMMARY.csv给出细步排名。\n'
        'FULL_WINDOW_DT_STATE_DRIFT.csv给出直接轨迹差异。不能只看排名不变，还须将分带K/Z和谱尾偏差与粗步对照。\n'
        '这是两步长敏感性，不是四阶收敛阶证明；旧报告要求的局部timechecks不在本次重复。REFERENCE_NOT_CONVERGED。\n',encoding='utf-8')
    c.dump(root/'LATEST_REPORT.json',dict(path=str(out.relative_to(root)),scope='WHOLE_WINDOW_DT_REFINEMENT',
        outputs={p.name:c.sha(p) for p in out.iterdir() if p.is_file()}))

@torch.inference_mode()
def execute(root):
    bind();inventory=c.inventory()
    accepted=c.read(HERE/'DT_CPU_ACCEPTANCE.json')
    c.require(accepted['status']=='PASS' and accepted['code_identity']==c.local_identity(),
              'Full-window adapter CPU acceptance/code identity changed')
    source=ROOT/'research/full_aug_parameter_pc_full_v1_20261116/results/run01/RUN.json'
    original=c.read(source)
    for name,digest in original['code_identity'].items():c.require(c.sha(OLD/name)==digest,'Historical code changed')
    ident=dict(protocol=c.P,code_identity=c.local_identity(),inventory=inventory,runtime=d.runtime_identity('cuda'),
        reference_quality=c.QUALITY,coarse_RUN_sha256=c.sha(source))
    if root.exists():c.require(c.read(root/'RUN.json')==ident,'Fine resume identity changed')
    else:root.mkdir(parents=True);c.dump(root/'RUN.json',ident)
    with storage.RunLock(root):
        journal=c.Journal(root/'invocations')
        try:
            for case in c.P['cases']:
                c.require(str(case) in inventory['references'],'Missing reference: do not substitute')
                ref=c.References(inventory['references'][str(case)])
                for seed in c.P['seeds']:
                    row=next(v for v in inventory['checkpoints'] if v['seed']==seed)
                    try:start,_=d.startup(root,case,seed,row,ref,journal,'cuda')
                    except (e.TrajectoryFailure,RuntimeError) as err:
                        if not isinstance(err,e.TrajectoryFailure) and 'HARD_BUDGET_EXHAUSTED' not in str(err):raise
                        for method in c.METHODS:c.dump(root/'trajectories'/c.key(case,seed,method)/'STATUS.json',
                            dict(status=err.status if isinstance(err,e.TrajectoryFailure) else 'NOT_RUN_BUDGET',reason=str(err),reference_quality=c.QUALITY))
                        continue
                    order=list(c.METHODS);shift=(c.P['cases'].index(case)+c.P['seeds'].index(seed))%4
                    for method in order[shift:]+order[:shift]:d.run_trajectory(root,case,seed,method,row,start,journal,'cuda')
            c.dump(root/'EXECUTION.json',dict(actual_calls=journal.used,per_scope=journal.per_key,reference_quality=c.QUALITY))
        finally:journal.close()
    comparison(root)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute-full-rollout',action='store_true');ap.add_argument('--run',default='dt_half01');ap.add_argument('--rebuild-only',action='store_true')
    args=ap.parse_args();c.require(Path(args.run).name==args.run and args.run not in ('','.','..'),'Invalid run name')
    if not args.execute_full_rollout and not args.rebuild_only:
        print('NOT_RUN: manual --execute-full-rollout required');print(PLAN['full_window_dt']);return
    root=HERE/'paper_runs'/args.run
    if args.rebuild_only:bind();comparison(root);return
    c.require(torch.cuda.is_available(),'CUDA required');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    begin=time.perf_counter()
    try:execute(root)
    finally:
        if root.exists():c.dump(root/f'COMMAND_WALL_{time.time_ns()}.json',dict(seconds=time.perf_counter()-begin,storage_bytes=storage.tree_bytes(root)))

if __name__=='__main__':main()
