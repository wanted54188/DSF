"""Immutable accepted steps, durable call accounting and process-safe resume."""
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
import pc_common as c


class RunLock:
    def __init__(self,root): self.path=root/'RUNNING.lock'
    def __enter__(self):
        self.f=self.path.open('a+b'); self.f.seek(0)
        if self.path.stat().st_size==0: self.f.write(b' '); self.f.flush()
        self.f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.f.close(); raise RuntimeError('Another process owns this run')
        return self
    def __exit__(self,*args):
        self.f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.f.fileno(),fcntl.LOCK_UN)
        self.f.close()


def scan(folder):
    """Recover even a completely written step whose mutable pointer was not updated."""
    paths=sorted((folder/'steps').glob('*.npz'))
    expected=2; records=[]
    for path in paths:
        z=c.load(path); m=c.meta(z)
        c.require(m['accepted_step']==expected and path.stem==f'{expected:04d}','Non-contiguous accepted steps')
        c.require(z['omega'].shape==(100,100) and z['rhs'].shape==(4,100,100),'Committed step shape')
        records.append((path,m)); expected+=1
    pointer=folder/'CHECKPOINT.json'
    if pointer.exists():
        cp=c.read(pointer)
        c.require(cp['accepted_step']<=expected-1,'Resume pointer ahead of accepted files')
        if cp['accepted_step']>=2:
            p=folder/'steps'/f'{cp["accepted_step"]:04d}.npz'
            c.require(c.sha(p)==cp['sha256'],'Committed checkpoint SHA mismatch')
    if records:
        p,m=records[-1]
        c.dump(pointer,dict(accepted_step=m['accepted_step'],file=str(p.relative_to(folder)),sha256=c.sha(p)))
    return records


def commit(folder,n,omega,rhs,metadata,observation=None):
    path=folder/'steps'/f'{n:04d}.npz'
    arrays=dict(omega=omega,rhs=rhs,metadata=c.packed(dict(metadata,accepted_step=n)))
    if observation:
        arrays.update({'observation__'+k:v for k,v in observation.items()})
    c.npz(path,**arrays)
    c.dump(folder/'CHECKPOINT.json',dict(accepted_step=n,file=str(path.relative_to(folder)),sha256=c.sha(path)))
    return path


def observation(z): return {k.removeprefix('observation__'):v for k,v in z.items() if k.startswith('observation__')}


def recover_failure(folder):
    """Complete an interrupted failure receipt without evaluating the model again."""
    path=folder/'FAILED_STAGE.npz'
    if not path.exists():return None
    info=c.meta(c.load(path));status=info['status']
    c.require(status in ('FAILED_PHYSICAL','FAILED_NUMERICAL'),'Invalid failure transaction')
    result=dict(status=status,first_failed_step=info['step'],first_failed_stage=info['stage'],
        first_failed_time=info['time_s'],reason=info['reason'],details=info['details'],
        last_accepted_step=info['step'],full_failure_file=path.name,failure_sha256=c.sha(path),
        reference_quality=c.QUALITY,recovered_atomic_failure=True)
    if (folder/'STATUS.json').exists():
        old=c.read(folder/'STATUS.json')
        if old.get('status') in ('FAILED_PHYSICAL','FAILED_NUMERICAL'):
            c.require(old['failure_sha256']==c.sha(path),'Failed-stage SHA changed');return old
    c.dump(folder/'STATUS.json',result);return result


def write_failure(folder,error,engine,root,step):
    from pc_engine import cpu,D
    arrays=dict(error.arrays)
    if hasattr(engine,'predicted_heads'): arrays['predicted_heads']=cpu(engine.predicted_heads)
    if 'qd' in arrays and 'qb' in arrays:
        import torch
        q=torch.as_tensor(arrays['qd']+arrays['qb'],device=engine.s.kx.device)
        if bool(torch.isfinite(q).all()): arrays['Source']=cpu(D(engine.s,q))
    history=engine.bank.descriptor() if engine.bank else None
    failed_stage=engine.stage+1 if engine.stage<4 else 'endpoint'
    failure_time=1+c.H*(step+((0.,.5,.5,1.)[engine.stage] if engine.stage<4 else 1.))
    failed_stage=error.details.get('stage_override',failed_stage)
    failure_time=error.details.get('time_override',failure_time)
    c.npz(folder/'FAILED_STAGE.npz',metadata=c.packed(dict(step=step,stage=failed_stage,
        time_s=failure_time,history=history,
        status=error.status,reason=str(error),details=error.details)),**arrays)
    c.dump(folder/'STATUS.json',dict(status=error.status,first_failed_step=step,
        first_failed_stage=failed_stage,first_failed_time=failure_time,
        reason=str(error),details=error.details,last_accepted_step=step,
        full_failure_file='FAILED_STAGE.npz',failure_sha256=c.sha(folder/'FAILED_STAGE.npz'),
        resource=c.resource_memory(engine.s.kx.device),
        reference_quality=c.QUALITY))


def tree_bytes(folder): return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
