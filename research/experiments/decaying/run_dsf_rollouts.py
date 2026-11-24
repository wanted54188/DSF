"""One explicit command runs/resumes all 24 trajectories AND writes final reports."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
import pc_common as c
from pc_engine import Engine,Bank,TrajectoryFailure,save_anchor,load_bank,cpu,basic_metrics
import pc_storage as storage


def runtime_identity(device):
    return dict(torch_version=str(torch.__version__),numpy_version=np.__version__,device=str(device),
        GPU=torch.cuda.get_device_name() if str(device).startswith('cuda') else None)


def bind_scope(j,name): j.scope=name


def startup(root,case,seed,row,reference,j,device):
    folder=root/'startup'/f'{case}_{seed}'; folder.mkdir(parents=True,exist_ok=True)
    old=storage.recover_failure(folder)
    if old is not None:
        raise TrajectoryFailure(old['status'],old['reason'],details=old.get('details',{}))
    if (folder/'DONE.json').exists():
        rec=c.read(folder/'DONE.json')
        for name,h in rec['outputs'].items(): c.require(c.sha(folder/name)==h,'Startup SHA changed')
        return c.load(folder/'STARTUP.npz'),rec
    if (folder/'STARTUP.npz').exists():
        # Recover an atomic startup transaction even if its final receipt was
        # interrupted. No repeated warmup/CNN is needed.
        z=c.load(folder/'STARTUP.npz'); info=c.meta(z)
        load_bank(root,info['bank'],device)
        original=folder/'ORIGINAL_W1.npz'
        if not original.exists():
            raw={k.removeprefix('original_w1__'):v for k,v in z.items() if k.startswith('original_w1__')}
            c.require(bool(raw),'Incomplete startup transaction has no original fields')
            c.npz(original,**raw)
        rec=dict(status='VERIFIED_RECOVERED_ATOMIC_STARTUP',wall_seconds=None,
            counts=j.per_key.get(f'{case}_{seed}_STARTUP',dict(CNN=0,RHS=0)),
            outputs={p.name:c.sha(p) for p in (folder/'STARTUP.npz',original)},reference_quality=c.QUALITY)
        c.dump(folder/'DONE.json',rec)
        return z,rec
    attempts=c.read(folder/'ATTEMPTS.json')['count'] if (folder/'ATTEMPTS.json').exists() else 0
    c.require(attempts<3,'HARD_BUDGET_EXHAUSTED startup replay allowance')
    c.dump(folder/'ATTEMPTS.json',dict(count=attempts+1))
    bind_scope(j,f'{case}_{seed}_STARTUP')
    s=c.model(row,device); before=c.q.study.common.model_hash(s)
    if not (root/'OPERATORS.npz').exists(): c.npz(root/'OPERATORS.npz',**c.operators(s))
    else:
        op=c.load(root/'OPERATORS.npz')
        c.require(all(np.array_equal(value,op[k]) for k,value in c.operators(s).items()),'Production operator mismatch')
    w0=s.project(torch.tensor(reference.at(0)['reference_omega'],device=device))
    inv=basic_metrics(s,w0); e=Engine(s,'Aug',j,root,folder,inv)
    begin=time.perf_counter()
    j.note('SCOPE_BEGIN')
    location='COMMON_AUG_STARTUP'
    try:
        w1,timing=e.advance(0,w0,None)
        stage0=e.snapshots[0]; rhs=torch.stack(e.values); records=list(e.records)
        j.phase='startup_two_true_refresh_anchors'
        location='ANCHOR0'; old=e.orbit(w0)
        location='ANCHOR1'; now=e.orbit(w1)
        a0=save_anchor(root,folder/'anchors',0,old['heads'])
        a1=save_anchor(root,folder/'anchors',1,now['heads'])
        bank=Bank([a0,a1]).descriptor()
        origpath=folder/'ORIGINAL_W1.npz'
        c.npz(folder/'STARTUP.npz',omega0=cpu(w0),omega1=cpu(w1),rhs=cpu(rhs),
            metadata=c.packed(dict(bank=bank,initial=inv,stages=records,timing=timing)),
            **{'observation__'+k:v for k,v in stage0.items()},
            **{'original_w1__'+k:cpu(v) for k,v in now['original'].items()})
        c.npz(origpath,**{k:cpu(v) for k,v in now['original'].items()})
        c.require(c.q.study.common.model_hash(s)==before,'Startup model changed')
        rec=dict(status='VERIFIED',wall_seconds=time.perf_counter()-begin,counts=j.per_key[j.scope],
                 outputs={p.name:c.sha(p) for p in (folder/'STARTUP.npz',origpath)},reference_quality=c.QUALITY)
        c.dump(folder/'DONE.json',rec)
        return c.load(folder/'STARTUP.npz'),rec
    except TrajectoryFailure as error:
        error.details['startup_location']=location
        if location.startswith('ANCHOR'):
            error.details.update(stage_override=location,time_override=1. if location=='ANCHOR0' else 1.005)
        storage.write_failure(folder,error,e,root,0)
        raise
    finally:
        j.note('SCOPE_END',seconds=time.perf_counter()-begin)
        c.require(c.q.study.common.model_hash(s)==before,'Startup weights changed')
        e.close()


def run_trajectory(root,case,seed,method,row,start,j,device):
    folder=root/'trajectories'/c.key(case,seed,method); folder.mkdir(parents=True,exist_ok=True)
    if storage.recover_failure(folder) is not None:return
    if (folder/'STATUS.json').exists():
        state=c.read(folder/'STATUS.json')
        if state['status'] in ('VERIFIED','FAILED_PHYSICAL','FAILED_NUMERICAL','NOT_RUN_BUDGET'): return
    records=storage.scan(folder); n=records[-1][1]['accepted_step'] if records else 1
    description=records[-1][1]['bank'] if records else c.meta(start)['bank']
    previous=c.load(records[-1][0]) if records else None
    w_np=previous['omega'] if previous else start['omega1']
    bind_scope(j,c.key(case,seed,method)); s=c.model(row,device)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    before=c.q.study.common.model_hash(s); bank=load_bank(root,description,device) if method.startswith('Parameter') else None
    e=Engine(s,method,j,root,folder,c.meta(start)['initial'])
    w=torch.tensor(w_np,device=device); wall_begin=time.perf_counter()
    compute_sum=sum(m['timing']['solver_wall_seconds'] for _,m in records)
    import json
    io_sum=sum(json.loads(line)['seconds'] for path in folder.glob('IO*.jsonl') for line in path.read_text().splitlines())
    io_log=folder/f'IO_{time.time_ns()}.jsonl'
    j.note('SCOPE_BEGIN')
    # A durable attempt pointer distinguishes an interrupted uncommitted step.
    attempted=folder/'ATTEMPT.json'; replay_count=0
    if attempted.exists():
        old=c.read(attempted)
        replay_count=old.get('replay_count',0)
        if old['step']>=n:
            replay_count+=1
    if replay_count>2:
        c.dump(folder/'STATUS.json',dict(status='NOT_RUN_BUDGET',reason='Uncommitted-step replay allowance exhausted',last_accepted_step=n,
                                       actual_calls=j.per_key.get(j.scope,{}),reference_quality=c.QUALITY))
        j.note('SCOPE_END',seconds=time.perf_counter()-wall_begin); e.close(); return
    try:
        for step in range(n,800):
            c.dump(attempted,dict(step=step,replay_count=replay_count))
            try:
                out,timing=e.advance(step,w,bank)
            except TrajectoryFailure as error:
                storage.write_failure(folder,error,e,root,step); return
            # Every accepted native step is an immutable restart point.
            metadata=dict(step_start=step,bank=bank.descriptor() if bank else None,
                stages=e.records,timing=timing,calls=dict(j.per_key.get(j.scope,{})))
            obs=e.snapshots[0] if e.snapshots else None
            begin=time.perf_counter()
            storage.commit(folder,step+1,cpu(out),cpu(torch.stack(e.values)),metadata,obs)
            commit_wall=time.perf_counter()-begin
            # Commit IO has a separate append-only measurement receipt.
            with io_log.open('a',encoding='utf-8') as f:
                import json
                f.write(json.dumps(dict(accepted_step=step+1,seconds=commit_wall))+'\n')
            compute_sum+=timing['solver_wall_seconds']; io_sum+=commit_wall; w=out
            c.dump(folder/'STATUS.json',dict(status='RUNNING',last_accepted_step=step+1,
                actual_calls=j.per_key[j.scope],solver_wall_seconds=compute_sum,IO_wall_seconds=io_sum,
                resource=c.resource_memory(device),reference_quality=c.QUALITY))
            if (step+1)%10==0:
                print(f'{case%1000}/{seed}/{method} t={1+(step+1)*c.H:.2f}/5, total CNN={j.used["CNN"]}',flush=True)
        endpoint=folder/'ENDPOINT.npz'
        if not endpoint.exists():
            c.dump(attempted,dict(step=800,replay_count=replay_count))
            try:
                c.sync(device); begin=time.perf_counter(); obs=e.endpoint(800,w,bank); c.sync(device)
                endpoint_time=time.perf_counter()-begin
            except TrajectoryFailure as error:
                storage.write_failure(folder,error,e,root,800); return
            c.npz(endpoint,metadata=c.packed(dict(step=800,stages=e.records,wall_seconds=endpoint_time)),**obs)
        c.require(c.q.study.common.model_hash(s)==before,'Model weights changed')
        c.dump(folder/'STATUS.json',dict(status='VERIFIED',last_accepted_step=800,
            actual_calls=j.per_key[j.scope],solver_wall_seconds=compute_sum,
            endpoint_wall_seconds=c.meta(c.load(endpoint))['wall_seconds'],IO_wall_seconds=io_sum,
            current_session_wall_seconds=time.perf_counter()-wall_begin,resource=c.resource_memory(device),
            physical_envelope='VERIFIED_ALL_APPLIED_STAGES',bounded_head_reachability='NOT_RUN',
            reference_quality=c.QUALITY))
    except RuntimeError as error:
        if 'HARD_BUDGET_EXHAUSTED' in str(error):
            c.dump(folder/'STATUS.json',dict(status='NOT_RUN_BUDGET',reason=str(error),last_accepted_step=n,
                actual_calls=j.per_key.get(j.scope,{}),reference_quality=c.QUALITY))
            return
        raise
    finally:
        j.note('SCOPE_END',seconds=time.perf_counter()-wall_begin)
        c.require(c.q.study.common.model_hash(s)==before,'Trajectory weights changed')
        e.close()


def execute(root,device):
    from pc_audit import audit
    acceptance=audit(reuse=True)
    inventory=acceptance['inventory']; identity=c.local_identity()
    if root.exists():
        run=c.read(root/'RUN.json')
        c.require(run['code_identity']==identity and run['inventory']==inventory
                  and run['runtime']==runtime_identity(device),'Resume identity changed; do not mix runs')
    else:
        root.mkdir(parents=True)
        c.dump(root/'RUN.json',dict(protocol=c.P,code_identity=identity,inventory=inventory,runtime=runtime_identity(device),
            reference_quality=c.QUALITY))
    with storage.RunLock(root):
        j=c.Journal(root/'invocations'); begin=time.perf_counter()
        try:
            for case in c.P['cases']:
                if str(case) in inventory['missing']:
                    for seed in c.P['seeds']:
                        for method in c.METHODS:
                            c.dump(root/'trajectories'/c.key(case,seed,method)/'STATUS.json',
                                dict(status='MISSING_INPUT',reason=inventory['missing'][str(case)],reference_quality=c.QUALITY))
                    continue
                references=c.References(inventory['references'][str(case)])
                for seed in c.P['seeds']:
                    row=next(r for r in inventory['checkpoints'] if r['seed']==seed)
                    try: start,rec=startup(root,case,seed,row,references,j,device)
                    except TrajectoryFailure as error:
                        failed=c.read(root/'startup'/f'{case}_{seed}'/'STATUS.json')
                        for method in c.METHODS:
                            c.dump(root/'trajectories'/c.key(case,seed,method)/'STATUS.json',dict(
                                failed,reason='COMMON_STARTUP: '+str(error),reference_quality=c.QUALITY))
                        continue
                    except RuntimeError as error:
                        if 'HARD_BUDGET_EXHAUSTED' not in str(error):raise
                        for method in c.METHODS:
                            c.dump(root/'trajectories'/c.key(case,seed,method)/'STATUS.json',dict(
                                status='NOT_RUN_BUDGET',reason='COMMON_STARTUP: '+str(error),
                                reference_quality=c.QUALITY))
                        continue
                    # Rotate execution order by fixed seed/case index, not results.
                    order=list(c.METHODS); offset=(c.P['cases'].index(case)+c.P['seeds'].index(seed))%4
                    order=order[offset:]+order[:offset]
                    for method in order: run_trajectory(root,case,seed,method,row,start,j,device)
            from pc_timecheck import run_checks
            run_checks(root,inventory,j,device)
            c.dump(root/'EXECUTION.json',dict(status='ALL_SCHEDULED_TRAJECTORIES_TERMINAL',actual_calls=j.used,
                per_scope=j.per_key,this_session_seconds=time.perf_counter()-begin,
                reference_quality=c.QUALITY))
        except BaseException as error:
            c.dump(root/f'INTERRUPTION_{time.time_ns()}.json',dict(status='INTERRUPTED',error=str(error),
                actual_calls=j.used,traceback=traceback.format_exc(),resume='Repeat the identical manual command'))
            raise
        finally: j.close()
    from pc_report import build_report
    build_report(root)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute-full-rollout',action='store_true')
    ap.add_argument('--run',default='run01'); ap.add_argument('--rebuild-only',action='store_true')
    args=ap.parse_args()
    c.require(Path(args.run).name==args.run and args.run not in ('','.','..'),'Use a single run name')
    root=c.HERE/'results'/args.run
    if args.rebuild_only:
        from pc_report import build_report
        build_report(root); return
    if not args.execute_full_rollout:
        print('NOT_RUN: explicit --execute-full-rollout required. Same command resumes automatically.')
        print(c.P['hard_budget']); return
    c.require(torch.cuda.is_available(),'CUDA required by fixed protocol')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    try:
        with torch.inference_mode(): execute(root,'cuda')
    finally:
        if root.exists():
            c.dump(root/f'COMMAND_WALL_{time.time_ns()}.json',dict(seconds=time.perf_counter()-begin,
                includes='input audit, model setup, integrations, IO and automatic report in THIS invocation',
                total_storage_bytes=storage.tree_bytes(root),resource=c.resource_memory('cuda'),
                reference_quality=c.QUALITY))


if __name__=='__main__': main()
