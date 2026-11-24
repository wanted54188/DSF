"""Fixed local h-vs-two-h/2 sensitivity; never change the main run settings."""
import time
import numpy as np
import torch
import pc_common as c
from pc_engine import Engine,load_bank,TrajectoryFailure,cpu,basic_metrics


def run_checks(root,inventory,journal,device):
    for case in c.P['cases']:
        for seed in c.P['seeds']:
            for method in c.METHODS:
                traj=root/'trajectories'/c.key(case,seed,method)
                status=c.read(traj/'STATUS.json') if (traj/'STATUS.json').exists() else {}
                for n in c.P['timestep_diagnostic']['steps']:
                    folder=root/'timecheck'/f'{c.key(case,seed,method)}_{n:04d}';folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'DONE.json').exists():continue
                    # NPZ is the atomic transaction; recover its final JSON after
                    # a process interruption rather than repeat an accepted job.
                    if (folder/'STATES.npz').exists():
                        result=c.meta(c.load(folder/'STATES.npz'))['result']
                        c.require((result['case'],result['seed'],result['method'],result['step'])==(case,seed,method,n),
                                  'Recovered half-step identity changed')
                        c.dump(folder/'DONE.json',dict(result,recovered_atomic_result=True));continue
                    if (folder/'FAILED_STAGE.npz').exists():
                        info=c.meta(c.load(folder/'FAILED_STAGE.npz'))
                        c.dump(folder/'DONE.json',dict(status=info['status'],case=case,seed=seed,method=method,
                            step=n,time_s=1+n*c.H,reason=info['reason'],details=info['details'],
                            reference_quality=c.QUALITY,recovered_atomic_failure=True));continue
                    if status.get('status')!='VERIFIED':
                        c.dump(folder/'DONE.json',dict(status='NOT_RUN_MAIN_TRAJECTORY_INCOMPLETE',case=case,seed=seed,method=method,step=n))
                        continue
                    attempts=c.read(folder/'ATTEMPTS.json')['count'] if (folder/'ATTEMPTS.json').exists() else 0
                    if attempts>=3:
                        c.dump(folder/'DONE.json',dict(status='NOT_RUN_REPLAY_ALLOWANCE',case=case,seed=seed,method=method,step=n));continue
                    c.dump(folder/'ATTEMPTS.json',dict(count=attempts+1))
                    prev=c.load(traj/'steps'/f'{n:04d}.npz');coarse=c.load(traj/'steps'/f'{n+1:04d}.npz')
                    info=c.meta(coarse);initial=c.meta(c.load(root/'startup'/f'{case}_{seed}'/'STARTUP.npz'))['initial']
                    row=next(v for v in inventory['checkpoints'] if v['seed']==seed)
                    journal.scope=f'TIME_{c.key(case,seed,method)}_{n:04d}'
                    journal.note('SCOPE_BEGIN');begin=time.perf_counter()
                    s=c.model(row,device);e=Engine(s,method,journal,root,folder,initial)
                    e.n=n;e.bank=load_bank(root,info['bank'],device) if method.startswith('Parameter') else None
                    frozen=e.bank.descriptor() if e.bank else None
                    w=torch.tensor(prev['omega'],device=device)
                    e.records=[];e.values=[];e.snapshots=[];e.cache=None;e.fresh=None
                    details=dict(case=case,seed=seed,method=method,step=n,time_s=1+n*c.H)
                    try:
                        for first,fractions in ((0.,(0.,.25,.25,.5)),(.5,(.5,.75,.75,1.))):
                            e.stage=0;e.fractions=fractions
                            w=s.step(w,1+c.H*(n+first),c.H/2)
                        c.require(e.bank is None or e.bank.descriptor()==frozen,'Half step mutated history')
                        c.require(len(e.values)==8 and bool(torch.isfinite(w).all()),'Invalid half-step result')
                        inv=basic_metrics(s,w)
                        if inv['K']>100*initial['K'] or inv['Z']>100*initial['Z']:
                            raise TrajectoryFailure('FAILED_NUMERICAL','Time-check endpoint exceeds inherited safety bound',dict(omega=cpu(w)),inv)
                        diff=torch.fft.fft2(w-torch.tensor(coarse['omega'],device=device),norm='forward')
                        metrics={band+'_MSE_difference':float(diff[mask].abs().square().sum()) for band,mask in c.q.masks(s).items()}
                        result=dict(status='VERIFIED_LOCAL_DT_CHECK',**details,**metrics,
                            two_half_RMS_difference=metrics['all_MSE_difference']**.5,
                            wall_seconds=time.perf_counter()-begin,reference_quality=c.QUALITY,
                            meaning='Local one-step sensitivity, NOT whole-window time convergence or a replacement time step')
                        c.npz(folder/'STATES.npz',initial=prev['omega'],coarse=coarse['omega'],two_half=cpu(w),
                              rhs=cpu(torch.stack(e.values)),metadata=c.packed(dict(**details,stages=e.records,bank=frozen,result=result)))
                    except TrajectoryFailure as error:
                        c.npz(folder/'FAILED_STAGE.npz',metadata=c.packed(dict(**details,status=error.status,reason=str(error),details=error.details,
                            stage=e.stage+1,history=frozen)),**error.arrays)
                        result=dict(status=error.status,**details,reason=str(error),details=error.details,reference_quality=c.QUALITY)
                    except RuntimeError as error:
                        if 'HARD_BUDGET_EXHAUSTED' not in str(error):raise
                        result=dict(status='NOT_RUN_BUDGET',**details,reason=str(error),reference_quality=c.QUALITY)
                    finally:
                        e.close();journal.note('SCOPE_END',seconds=time.perf_counter()-begin)
                    c.dump(folder/'DONE.json',result)
                    print('Time sensitivity:',case%1000,seed,method,n,result['status'],flush=True)
