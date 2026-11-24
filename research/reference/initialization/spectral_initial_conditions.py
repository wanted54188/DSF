"""Independent no-SGS Navier--Stokes trajectories and exact projected moment labels."""
import math
import time
from pathlib import Path
import numpy as np
import torch
from .core import Grid, dump, load, rms, relative, scalar, sha


def initial(grid, seed, cfg):
    rng = np.random.default_rng(seed)
    h = np.zeros((grid.n, grid.n), dtype=np.complex128)
    energy = 0.
    for ky in range(-cfg['initial_max_mode'], cfg['initial_max_mode']+1):
        for kx in range(cfg['initial_max_mode']+1):
            r = math.hypot(kx,ky)
            if (kx==0 and ky<=0) or not 2<=r<=cfg['initial_max_mode']:
                continue
            amp = float(rng.uniform(.5,1.5)*r*np.exp(-(r/cfg['initial_peak_mode'])**4))
            z = amp*np.exp(1j*rng.uniform(0,2*math.pi))/2
            h[ky % grid.n,kx % grid.n] = z
            h[-ky % grid.n,-kx % grid.n] = z.conjugate()
            energy += amp*amp/2
    return grid.project(grid.ifft(torch.as_tensor(h, device=grid.device))*cfg['initial_RMS']/math.sqrt(energy))


def labels(high, low, w, delta):
    st = high.state(w)
    filt = torch.exp(-delta*delta*high.k2/24)
    # F=P_LES T G, applied identically to moments and resolved fields.
    def F(a):
        return low.project(high.transfer(high.ifft(high.fft(a)*filt), low))
    wb, ub, vb = F(w), F(st['u']), F(st['v'])
    uxw, vyw = F(st['u']*w), F(st['v']*w)
    pad = Grid(2*low.n, low.length, low.device, low.dtype)
    up = lambda a:low.transfer(a,pad)
    qx, qy = up(uxw)-up(ub)*up(wb), up(vyw)-up(vb)*up(wb)
    raw = -pad.div(qx,qy,project=False)
    source = low.project(pad.transfer(raw,low))
    # Independent projected identity F(-u.grad(w)) + ubar.grad(wbar).
    gx, gy = low.grad(wb)
    rhs_difference = F(-st['u']*st['omega_x']-st['v']*st['omega_y'])+low.project(ub*gx+vb*gy)
    check = relative(source,rhs_difference)
    return dict(omega=wb,source=source,source_raw=raw,qx_raw=qx,qy_raw=qy,
                raw_norm=scalar(rms(raw)),source_identity_relative_gap=check,
                reference_budget=scalar((wb*source).mean()))


@torch.no_grad()
def integrate(cfg, seed, n, dt, device, callback=None):
    grid=Grid(n,cfg['length'],device,torch.float64)
    low=Grid(cfg['les_n'],cfg['length'],device,torch.float64)
    steps=round(cfg['final_time']/dt); every=round(cfg['save_interval']/dt)
    if abs(steps*dt-cfg['final_time'])>1e-10 or steps%every:
        raise ValueError('unaligned reference times')
    w=initial(grid,seed,cfg); fields=[]; high_frames=[]; hist=[]
    Z0=scalar(.5*w.square().mean()); E0=grid.energy(w)[0]
    intZ=intE=0.; maxcfl=0.; max_tail=0.; previous=None
    start=time.perf_counter()
    for i in range(steps+1):
        st=grid.state(w)
        gx,gy=st['omega_x'],st['omega_y']
        dz=-cfg['nu']*scalar((gx.square()+gy.square()).mean())
        de=-cfg['nu']*scalar(w.square().mean())
        if previous is not None:
            intZ+=dt*(dz+previous[0])/2;intE+=dt*(de+previous[1])/2
        previous=(dz,de)
        if i%every==0:
            lab=labels(grid,low,w,cfg['delta'])
            fields.append({k:(v.cpu().numpy() if torch.is_tensor(v) else v) for k,v in lab.items()})
            high_frames.append(w.cpu().numpy())
            power=grid.fft(w).abs().square()
            tail=((grid.kx.abs().maximum(grid.ky.abs()))>=.8*grid.cut*2*math.pi/grid.length)&grid.mask
            tailfrac=scalar(power[tail].sum()/power.sum().clamp_min(1e-30));max_tail=max(max_tail,tailfrac)
            E,Z=grid.energy(w)
            hist.append(dict(time=i*dt,E=E,Z=Z,Z_balance_relative=abs(Z-Z0-intZ)/Z0,
                             E_balance_relative=abs(E-E0-intE)/E0,tail_fraction=tailfrac))
            if callback:callback(i,steps,seed,scalar(rms(w)))
        if i==steps:break
        speed=scalar(st['u'].abs().max()+st['v'].abs().max())
        cfl=dt*(speed/(grid.length/n)+4*cfg['nu']/(grid.length/n)**2)
        maxcfl=max(maxcfl,cfl)
        if cfl>cfg['reference_CFL'] or not torch.isfinite(w).all():
            raise RuntimeError(f'reference CFL/nonfinite: {cfl}')
        w=grid.step(w,dt,cfg['nu'])
    arrays={k:np.stack([f[k] for f in fields]) for k in fields[0]}
    arrays['high_omega']=np.stack(high_frames)
    arrays['times']=np.arange(len(fields))*cfg['save_interval']
    meta=dict(seed=seed,n=n,dt=dt,seconds=time.perf_counter()-start,max_CFL=maxcfl,
              max_tail=max_tail,history=hist,
              max_source_identity_gap=float(arrays['source_identity_relative_gap'].max()),
              max_source_mean=float(np.abs(arrays['source'].mean((-2,-1))).max()),
              initial_RMS=cfg['initial_RMS'],purpose='independent_NS_decaying',DNS_claim=False)
    return arrays,meta


def save_trajectory(path, arrays, meta):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    # Uncompressed preserves exact float64 arrays and enables fast inspection.
    np.savez(path,**arrays)
    dump(path.with_suffix('.json'),dict(**meta,archive=str(path.resolve()),sha256=sha(path)))


def generate_all(protocol, out, device, log):
    cfg=protocol['physics']; data=Path(out)/'DATA';data.mkdir(exist_ok=True)
    first=protocol['split']['train'][0]
    manifests={}
    for role,seeds in protocol['split'].items():
        for seed in seeds:
            p=data/f'{role}_{seed}.npz'
            if p.exists():
                m=load(p.with_suffix('.json'))
                if sha(p)!=m['sha256']:raise RuntimeError('reference archive hash drift')
                manifests[str(seed)]=m;continue
            log('reference',f'{role} seed={seed} n={cfg["reference_n"]}')
            arrays,meta=integrate(cfg,seed,cfg['reference_n'],cfg['reference_dt'],device)
            meta['role']=role
            save_trajectory(p,arrays,meta);manifests[str(seed)]=load(p.with_suffix('.json'))
    # Calibration is train trajectory only. Never use test data to select resolution.
    basepath=data/f'train_{first}.npz'
    checks={}
    for tag,n,dt in [('half_dt',cfg['reference_n'],cfg['reference_dt']/2),
                     ('fine_space',cfg['refinement_n'],cfg['reference_dt']/2)]:
        p=data/f'calibration_{tag}_{first}.npz'
        if not p.exists():
            log('resolution',tag)
            arrays,meta=integrate(cfg,first,n,dt,device)
            meta['role']='resolution_calibration_train_only';save_trajectory(p,arrays,meta)
    with np.load(basepath) as base,np.load(data/f'calibration_half_dt_{first}.npz') as half,np.load(data/f'calibration_fine_space_{first}.npz') as fine:
        for key in ['omega','source','source_raw']:
            def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
            checks[key+'_time']=rel(base[key],half[key]);checks[key+'_space']=rel(half[key],fine[key])
    checks['max_tail_all']=max(m['max_tail'] for m in manifests.values())
    checks['max_Z_budget_error']=max(max(h['Z_balance_relative'] for h in m['history']) for m in manifests.values())
    checks['max_E_budget_error']=max(max(h['E_balance_relative'] for h in m['history']) for m in manifests.values())
    checks['max_source_identity_gap']=max(m['max_source_identity_gap'] for m in manifests.values())
    passed=(all(v<=protocol['checks']['time_relative'] for k,v in checks.items() if k.endswith('_time')) and
            all(v<=protocol['checks']['space_relative'] for k,v in checks.items() if k.endswith('_space')) and
            checks['max_tail_all']<=protocol['checks']['tail_fraction'] and
            max(checks['max_Z_budget_error'],checks['max_E_budget_error'])<=protocol['checks']['budget_relative'] and
            checks['max_source_identity_gap']<=protocol['checks']['source_identity'])
    result=dict(status='RESOLUTION_CHECKS_PASS' if passed else 'REFERENCE_UNVERIFIED',checks=checks,
                manifests=manifests,DNS_claim=False,test_generated_but_not_used_for_model_selection=True)
    dump(Path(out)/'reference_checks.json',result)
    return result
