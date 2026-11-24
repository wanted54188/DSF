"""Continue original complex128 Fourier checkpoints; bounded-memory dense labels."""
from pathlib import Path
import os,time,hashlib
import numpy as np
import torch
import common as c
from rp_spectral import SpectralRK4,Grid,labels,raw_diagnostics

def mapped_labels(solver,low,h,delta,nu):
    a=labels(solver,low,h,delta,nu,legacy=False)
    # Primary label is the RHS gap. Independent flux divergence remains testable.
    return dict(reference_omega=a['reference_omega'],reference_dot=a['reference_dot'],
        reference_source=a['reference_source_rhs_gap'],reference_qx=a['reference_qx'],reference_qy=a['reference_qy'],
        flux_source=a['reference_source'],reference_residual=a['reference_residual'])

def memmaps(d,count,n,mode):
    return {k:np.lib.format.open_memmap(d/(k+'.npy'),mode=mode,dtype=np.float64,shape=(count,n,n)) for k in c.FIELDS}

def commit(d,h,state):
    """Write the other slot before atomically publishing the journal pointer."""
    previous=c.read(d/'CHECKPOINT.json') if (d/'CHECKPOINT.json').exists() else {}
    slot=1-previous.get('slot',1);name=f'checkpoint_slot{slot}.npz'
    c.atomic_npz(d/name,h=h.detach().cpu().numpy())
    c.dump(d/'CHECKPOINT.json',dict(**state,slot=slot,file=name,sha256=c.sha(d/name)))

def integrate(run,case,v,device='cpu',resume=False):
    c.seed_guard(case['seed']);d=run/'reference'/str(case['seed'])/v['variant'];d.mkdir(parents=True,exist_ok=True)
    if (d/'DONE.json').exists():
        done=c.read(d/'DONE.json')
        for p,h in done['outputs'].items():
            if c.sha(d/p)!=h:raise ValueError('Completed reference output changed '+p)
        return done
    lock=d/'RUNNING.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:return _integrate(run,d,case,v,device,resume)
    except BaseException as exc:
        c.dump(d/f'ERROR_{time.time_ns()}.json',dict(type=type(exc).__name__,message=str(exc),resume='Last committed checkpoint is safe; use --resume; stale lock requires explicit --recover-lock'))
        raise
    finally:lock.unlink(missing_ok=True)

def _integrate(run,d,case,v,device,resume):
    p=c.policy();cfg=v['config'];dt=v['dt'];n=v['n'];old=Path(v['path']);record=cfg['record_dt']
    for t in (1.,1.2,5.,record,p['native_record_dt'],p['checkpoint_dt']):
        if abs(t/dt-round(t/dt))>1e-8:raise ValueError('Unaligned fixed schedule')
    if c.sha(old/'FREEZE.json')!=v['freeze_sha256'] or c.sha(old/'DONE.json')!=v['done_sha256']:raise ValueError('Original input identity changed')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Explicit CUDA reference requested but unavailable')
    torch.set_num_threads(4);solver=SpectralRK4(n,cfg['length'],cfg['nu'],device);low=Grid(cfg['les_n'],cfg['length'],device,torch.float64)
    times=1+np.arange(round(4/record)+1)*record;count=len(times);every=round(record/dt);native_every=round(.05/dt)
    final=round(5/dt);prefix_end=round(1.2/dt);cp_path=d/'CHECKPOINT.json';observations=[];native=[];overlap=[]
    freeze=c.sha(run/'FREEZE.json');started=time.perf_counter();elapsed0=0.;i_next=0;first=0
    iz=ie=0.;E0=Z0=0.;balance_start=1.2
    if cp_path.exists():
        if not resume:raise FileExistsError('Partial task exists; explicit --resume required')
        cp=c.read(cp_path)
        if cp['freeze_sha256']!=freeze or c.sha(d/cp['file'])!=cp['sha256']:raise ValueError('Local checkpoint identity mismatch')
        with c.archive(d/cp['file']) as z:h=torch.as_tensor(np.array(z['h']),device=device)
        if tuple(h.shape)!=(n,n) or h.dtype!=torch.complex128:raise ValueError('Wrong checkpoint tensor')
        arrays=memmaps(d,count,low.n,'r+');first=cp['step'];i_next=cp['next_record'];observations=cp['observations'];native=cp['native'];overlap=cp['overlap']
        for k,digest in cp.get('last_record_hashes',{}).items():
            if hashlib.sha256(arrays[k][i_next-1].tobytes()).hexdigest()!=digest:raise ValueError('Committed record changed')
        elapsed0=cp['elapsed_seconds'];E0=cp['E0'];Z0=cp['Z0'];iz=cp['integrated_Z'];ie=cp['integrated_E'];balance_start=cp['balance_start']
        solver.max_speed=torch.tensor(cp['max_speed'],device=device,dtype=torch.float64)
    else:
        if (d/'reference_omega.npy').exists():raise RuntimeError('Interrupted before first commit: use --restart-uncommitted, never guess a checkpoint')
        arrays=memmaps(d,count,low.n,'w+');np.save(d/'time_s.npy',times)
        # Reuse and independently check the SAME low-grid prefix, no interpolation.
        if c.sha(old/'FIELDS.npz')!=v['fields_sha256']:raise ValueError('Original prefix archive changed')
        with c.archive(old/'FIELDS.npz') as z:
            oldtimes=np.array(z['time_s']);prefix={k:np.array(z[k]) for k in ('reference_omega','reference_source','reference_dot','reference_qx','reference_qy')}
        if not np.allclose(oldtimes,times[:len(oldtimes)],atol=1e-12,rtol=0):raise ValueError('Prefix times mismatch')
        for i,t in enumerate(oldtimes):
            w=prefix['reference_omega'][i];rhs=low.rhs(torch.as_tensor(w,device=device),cfg['nu']).cpu().numpy()
            lab={k:prefix[k][i] for k in ('reference_omega','reference_dot','reference_qx','reference_qy')}
            lab.update(reference_source=prefix['reference_dot'][i]-rhs,flux_source=prefix['reference_source'][i],
                reference_residual=prefix['reference_dot'][i]-rhs-prefix['reference_source'][i])
            for k in c.FIELDS:arrays[k][i]=lab[k]
        i_next=len(oldtimes)
        # Native prefix snapshots are exact old high-grid fields, not low-grid padding.
        md=c.read(old/'REFERENCE_METADATA.json');done=c.read(old/'DONE.json')
        for row in md['history']:
            t=row['time_s']
            if abs((t-1)/.05-round((t-1)/.05))>1e-8:continue
            name=row['high_file']
            if not name or c.sha(old/name)!=done['outputs_sha256'][name]:raise ValueError('Native prefix snapshot identity')
            with c.archive(old/name) as z:w=torch.as_tensor(np.array(z['omega']),device=device,dtype=torch.float64)
            hh=solver.grid.fft(w)*solver.grid.mask;lab=mapped_labels(solver,low,hh,cfg['delta'],cfg['nu']);idx=c.index(oldtimes,t)
            checks={k:c.relative(lab[k],prefix[k][idx]) for k in ('reference_omega','reference_source','reference_dot')}
            checks['spectrum']=c.relative(abs(np.fft.fft2(lab['reference_omega']))**2,abs(np.fft.fft2(prefix['reference_omega'][idx]))**2)
            overlap.append(dict(time_s=t,kind='independent_native_prefix_relabel',**checks))
            dest=f'native_{round(t/.05):06d}.npy';np.save(d/dest,w.cpu().numpy());native.append(dict(time_s=t,file=dest,sha256=c.sha(d/dest)))
            _,spectrum=raw_diagnostics(solver,hh);np.save(d/f'spectrum_{round(t/.05):06d}.npy',spectrum)
        if max(x[k] for x in overlap for k in ('reference_omega','reference_source','reference_dot','spectrum'))>p['overlap_relative_limit']:
            c.csvout(d/'OVERLAP.csv',overlap);raise ValueError('Original native/READY prefix mismatch; no splicing allowed')
        if v['checkpoint']:
            if c.sha(v['checkpoint_path'])!=v['checkpoint']['sha256']:raise ValueError('Original checkpoint changed')
            with c.archive(v['checkpoint_path']) as z:h=torch.as_tensor(np.array(z['h']),device=device)
            first=prefix_end
            if h.dtype!=torch.complex128 or tuple(h.shape)!=(n,n):raise ValueError('Original Fourier checkpoint schema')
            lab=mapped_labels(solver,low,h,cfg['delta'],cfg['nu'])
            checks={k:c.relative(lab[k],prefix[k][-1]) for k in ('reference_omega','reference_source','reference_dot')}
            overlap.append(dict(time_s=1.2,kind='checkpoint_boundary',spectrum=c.relative(abs(np.fft.fft2(lab['reference_omega']))**2,abs(np.fft.fft2(prefix['reference_omega'][-1]))**2),**checks))
            if max(checks.values())>p['overlap_relative_limit']:raise ValueError('Checkpoint/prefix endpoint mismatch')
            c.log(run,f'CONTINUE {case["seed"]} {v["variant"]} exact original h at t=1.2')
        else:
            h=solver.initial(case['seed'],cfg['initial']);first=0;i_next=0;balance_start=0.
            c.log(run,f'COLD identical original seed/spectrum {case["seed"]} {v["variant"]}; no low-grid interpolation')
        # Preserve prefix low labels for cold-generation overlap checks.
        Eobs,_=raw_diagnostics(solver,h);E0=Eobs['E'];Z0=Eobs['Z']
        observations=[dict(**r,period='original_prefix') for r in md['history']]

    def checkpoint(step):
        for a in arrays.values():a.flush()
        c.csvout(d/'OVERLAP.csv',overlap)
        commit(d,h,dict(step=step,next_record=i_next,freeze_sha256=freeze,observations=observations,native=native,overlap=overlap,
            last_record_hashes={k:hashlib.sha256(a[i_next-1].tobytes()).hexdigest() for k,a in arrays.items()} if i_next else {},
            elapsed_seconds=elapsed0+time.perf_counter()-started,E0=E0,Z0=Z0,integrated_Z=iz,integrated_E=ie,balance_start=balance_start,max_speed=float(solver.max_speed)))
    checkpoint(first)
    with torch.inference_mode():
        for step in range(first,final+1):
            t=step*dt
            if t>=1-1e-12 and step%every==0:
                i=round((t-1)/record)
                if i>=i_next:
                    lab=mapped_labels(solver,low,h,cfg['delta'],cfg['nu'])
                    if t<=1.2+1e-12:
                        checks={k:c.relative(lab[k],arrays[k][i]) for k in ('reference_omega','reference_source','reference_dot')}
                        overlap.append(dict(time_s=t,kind='cold_replay',spectrum=0.,**checks))
                        if max(checks.values())>p['overlap_relative_limit']:raise ValueError('Cold replay differs from old prefix')
                    for k in c.FIELDS:arrays[k][i]=lab[k]
                    i_next=i+1
                    obs,spec=raw_diagnostics(solver,h);dx=cfg['length']/n
                    cfl=dt*(float(solver.max_speed)/dx+4*cfg['nu']/dx**2)
                    observations.append(dict(time_s=t,period='new_continuation',CFL=cfl,**obs,
                        E_balance_relative=abs(obs['E']-E0-ie)/max(E0,1e-30),Z_balance_relative=abs(obs['Z']-Z0-iz)/max(Z0,1e-30)))
                    if not np.isfinite(obs['Z']) or cfl>p['quality']['CFL'] or obs['Z']>100*Z0:
                        c.dump(d/'NATIVE_SAFETY_FAILURE.json',dict(time_s=t,CFL=cfl if np.isfinite(cfl) else None,Z=obs['Z'] if np.isfinite(obs['Z']) else None))
                        raise RuntimeError('Native CFL/nonfinite/100x safety stop; last SAFE checkpoint preserved')
                    if step%native_every==0:
                        name=f'native_{round(t/.05):06d}.npy';np.save(d/name,solver.grid.ifft(h).cpu().numpy())
                        native=[r for r in native if abs(r['time_s']-t)>1e-10];native.append(dict(time_s=t,file=name,sha256=c.sha(d/name)))
                        np.save(d/f'spectrum_{round(t/.05):06d}.npy',spec)
            if step%native_every==0 or step==final:
                checkpoint(step);elapsed=elapsed0+time.perf_counter()-started
                estimate=(time.perf_counter()-started)/max(step-first,1)*(final-step)
                c.dump(d/'PROGRESS.json',dict(step=step,total=final,time_s=t,next_record=i_next,seconds=elapsed,device=device,
                    remaining_seconds_from_current_invocation=estimate,remaining_estimate_not_guaranteed=True))
                c.log(run,f'REFERENCE {case["seed"]} {v["variant"]} t={t:.4f}/5 seconds={elapsed:.1f}')
                if time.perf_counter()-started>p['reference_resource_seconds_per_invocation']:raise TimeoutError('Reference invocation budget exhausted; checkpoint retained; explicit resume grants another invocation')
            if step==final:break
            h,dz,de=solver.step(h,dt);iz+=float(dz);ie+=float(de)
    if i_next!=count:raise ValueError('Incomplete time coverage')
    for a in arrays.values():a.flush()
    native.sort(key=lambda r:r['time_s'])
    c.csvout(d/'RAW_DIAGNOSTICS.csv',observations)
    done=dict(status='GENERATED_NOT_YET_QUALIFIED',seed=case['seed'],role='development',variant=v['variant'],n=n,dt=dt,
        config=cfg,record_dt=record,frames=count,native=native,seconds=elapsed0+time.perf_counter()-started,
        continuation_from=v['checkpoint_path'],balance_start=balance_start,device=device,
        device_name=torch.cuda.get_device_name() if device=='cuda' else 'CPU',origin='same_original_NS_checkpoint_or_identical_initial',
        checkpoint_sha256=c.sha(d/'CHECKPOINT.json'),outputs={f.name:c.sha(f) for f in d.iterdir() if f.suffix=='.npy' or f.name in ('OVERLAP.csv','RAW_DIAGNOSTICS.csv')})
    c.dump(d/'DONE.json',done);return done

def open_fields(d):
    return {**{k:np.load(d/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in c.FIELDS},'time_s':np.load(d/'time_s.npy',allow_pickle=False)}
