"""Frozen CPU LES reuse/recovery and matched long-reference metrics, no fitting."""
from pathlib import Path
import hashlib,time
import numpy as np
import torch
import common as c
from reference import open_fields
from quality import band

def state_metrics(op,w,r,t):
    psi=op.psi(w);pr=op.psi(r);E=.5*np.mean(w*psi);Z=.5*np.mean(w*w);Er=.5*np.mean(r*pr);Zr=.5*np.mean(r*r)
    row=dict(time_s=float(t),E=float(E),Z=float(Z),reference_E=float(Er),reference_Z=float(Zr),
        E_bias=float(E-Er),Z_bias=float(Z-Zr),E_relative_bias=float((E-Er)/max(abs(Er),1e-30)),Z_relative_bias=float((Z-Zr)/max(abs(Zr),1e-30)))
    for name in ('all','low','high'):
        err=band(op,w-r,name);ref=band(op,r,name);floor=1e-12*c.rms(r)
        row['omega_relative_L2_'+name]=c.rms(err)/max(c.rms(ref),floor,1e-30)
        row['error_power_'+name]=c.rms(err)**2
    power=abs(np.fft.fft2(w)/w.size)**2
    row['radial_tail_fraction']=float(power[(op.shell>=30)&op.mask].sum()/max(power.sum(),1e-30))
    return row

def horizons(rows,thresholds,status,failure_time=None):
    result=[]
    for threshold in thresholds:
        j=next((i for i,r in enumerate(rows) if r['omega_relative_L2_all']>threshold),None)
        if j is None:
            result.append(dict(threshold=threshold,first_exceedance=None,last_confirmed_time=rows[-1]['time_s'],
                duration_lower_bound=rows[-1]['time_s']-1.,censor='RIGHT_CENSORED_AT_END' if status=='COMPLETED' else 'CENSORED_BY_NUMERICAL_FAILURE',failure_time=failure_time))
        else:
            last=rows[max(j-1,0)]['time_s'];now=rows[j]['time_s']
            result.append(dict(threshold=threshold,first_exceedance=now,last_confirmed_time=last,duration_lower_bound=max(0.,last-1.),duration_upper_bound=now-1.,
                censor='OBSERVED_BRACKET_NO_INTERPOLATION',failure_time=failure_time))
    return result

def summary(rows,status):
    t=np.array([r['time_s'] for r in rows]);span=t[-1]-t[0];full=status=='COMPLETED' and abs(t[-1]-5)<1e-9
    result=dict(status=status,last_valid_time=float(t[-1]),accuracy_window_complete=full)
    for key in ('omega_relative_L2_all','omega_relative_L2_low','omega_relative_L2_high','E_relative_bias','Z_relative_bias'):
        vals=np.abs([r[key] for r in rows])
        result[key+'_valid_prefix_window']=float(np.trapezoid(vals,t)/span) if span else float(vals[0])
        result[key+'_full_window']=result[key+'_valid_prefix_window'] if full else None
        result[key+'_terminal5']=float(vals[-1]) if full else None
        result[key+'_maximum_valid']=float(max(vals))
    return result

def validate_history():
    run=c.read(c.OLD/'results/campaign03/RUN.json')
    if not run['status'].startswith('COMPLETED'):raise ValueError('Old campaign incomplete')
    if run['independent_flows']!=2 or run['locked_access'] or run['final_test_access']:raise ValueError('Unexpected old scope')
    if run['manifest_sha256']!=c.sha(c.MANIFEST):raise ValueError('Historical reference manifest identity mismatch')
    for name,digest in run['code_sha256'].items():
        if c.sha(c.OLD/name)!=digest:raise ValueError('Old numerical code changed; unsafe reuse')
    for name,digest in run['core_dependencies'].items():
        if c.sha(c.ROOT/name)!=digest:raise ValueError('Original operator changed; unsafe reuse')
    for r in run['models']:
        if c.sha(r['checkpoint'])!=r['actual_sha256'] or c.sha(r['config'])!=r['effective_config_sha256']:raise ValueError('Frozen model changed')
    if c.read(c.OLD/'results/campaign03/SELECTION.json')['winner']['method']!='SSC_1_16':raise ValueError('SSC parameter identity changed')
    return run

def load_history(seed,weight,method,dt,ref0,registry):
    e=c.engine();weight=e.canonical_weight(method,weight);tid=e.taskid(seed,weight,method,dt);base=c.OLD/'results/campaign03'
    short=base/'short'/tid/'SUMMARY.json';long=base/'long'/tid/'SUMMARY.json'
    if not short.is_file() or not long.is_file():return None
    sm,lm=c.read(short),c.read(long)
    for item in (sm,lm):
        if (item['initial_seed'],item['weight_seed'],item['method'],item['dt_s'])!=(seed,weight,method,dt):raise ValueError('Cached condition metadata mismatch')
    if sm['status']!='COMPLETED':return None
    if lm['status'] not in ('COMPLETED','SAFETY_STOP_100X','NONFINITE'):return None
    pp,lp=c.array_guard(sm['state_archive']),c.array_guard(lm['state_archive'])
    if not pp.exists() or not lp.exists():return None
    with c.archive(pp) as z:sw=np.array(z['omega']);st=np.array(z['time_s'])
    with c.archive(lp) as z:lw=np.array(z['omega']);lt=np.array(z['time_s'])
    if np.any(np.diff(st)<=0) or np.any(np.diff(lt)<=0) or not np.isfinite(sw).all() or not np.isfinite(lw).all():raise ValueError('Invalid cached state/time')
    if lm['status']=='COMPLETED' and abs(lt[-1]-5)>1e-10:raise ValueError('Incomplete cached horizon')
    if st[0]!=1 or abs(st[-1]-1.2)>1e-10 or abs(lt[0]-st[-1])>1e-10 or not np.array_equal(sw[-1],lw[0]):raise ValueError('Autonomous continuation mismatch')
    initial=hashlib.sha256(sw[0].tobytes()).hexdigest()
    if initial!=sm['initial_sha256']:raise ValueError('Saved LES initial hash mismatch')
    if c.relative(sw[0],ref0)>1e-6:raise ValueError('New reference/old LES initial mismatch')
    # Original manifest and actual READY prefix identity have been verified by quality.
    registry.append(dict(seed=seed,weight=weight,method=method,dt=dt,initial_sha256=initial,
        short_summary=str(short),short_summary_sha256=c.sha(short),long_summary=str(long),long_summary_sha256=c.sha(long),
        short_states=str(pp),short_sha256=c.sha(pp),long_states=str(lp),long_sha256=c.sha(lp),
        original_manifest_sha256=c.read(base/'RUN.json')['manifest_sha256'],continuation_max_difference=0.,new_neural_calls=0))
    return np.r_[st,lt[1:]],np.concatenate((sw,lw[1:])),lm

def make_solver(weight,method):
    from spnsgs.config import load_config
    e=c.engine();r=next(r for r in c.read(c.OLD/'results/campaign03/RUN.json')['models'] if r['seed']==weight)
    cfg=load_config(r['config']);solver=e.build(cfg,r['checkpoint'],method);return solver

def fresh_les(run,seed,weight,method,dt,ref0,op,resume):
    """Only missing/uncertified cache: CPU RK4 from the same initial, no reference forcing."""
    e=c.engine();dest=run/'new_les'/e.taskid(seed,weight,method,dt);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'SUMMARY.json').exists():
        done=c.read(dest/'SUMMARY.json')
        if done['status'] in ('COMPLETED','SAFETY_STOP_100X','NONFINITE'):
            with c.archive(dest/'STATES.npz') as z:return np.array(z['time_s']),np.array(z['omega']),done
    solver=make_solver(weight,method);torch.set_num_threads(2);t0=1.;start=time.perf_counter();freeze=c.sha(run/'FREEZE.json')
    E0=.5*np.mean(ref0*op.psi(ref0));Z0=.5*np.mean(ref0**2)
    cp=dest/'CHECKPOINT.json';times=[];names=[];calls=0
    if cp.exists():
        if not resume:raise RuntimeError('Partial LES exists: --resume required')
        zc=c.read(cp)
        if zc['freeze']!=freeze or c.sha(dest/zc['file'])!=zc['sha256']:raise ValueError('LES checkpoint mismatch')
        with c.archive(dest/zc['file']) as z:field=torch.as_tensor(np.array(z['omega']),dtype=torch.float32)
        j0=zc['step'];times=zc['times'];names=zc['names'];calls=zc['calls']
    else:
        field=solver.project(torch.as_tensor(op.project(ref0),dtype=torch.float32));j0=0
        np.save(dest/'state_000000.npy',field.numpy());times=[1.];names=['state_000000.npy']
    def checkpoint(j):
        old=c.read(cp) if cp.exists() else {};slot=1-old.get('slot',1);file=f'checkpoint{slot}.npz'
        c.atomic_npz(dest/file,omega=field.numpy());c.dump(cp,dict(slot=slot,file=file,sha256=c.sha(dest/file),freeze=freeze,step=j,times=times,names=names,calls=calls))
    steps=round(4/dt);stride=round(.05/dt);status='COMPLETED';failure_time=None
    try:
        with torch.inference_mode():
            for j in range(j0,steps):
                now=1+(j+1)*dt;new=solver.step(field,1+j*dt,dt);calls+=0 if method=='Clark' else 4
                if not torch.isfinite(new).all():status='NONFINITE';failure_time=now;break
                field=new;w=field.numpy().astype(float);E=.5*np.mean(w*op.psi(w));Z=.5*np.mean(w*w)
                if E/E0>100 or Z/Z0>100:status='SAFETY_STOP_100X';failure_time=now
                if (j+1)%stride==0 or failure_time or j+1==steps:
                    name=f'state_{j+1:06d}.npy';np.save(dest/name,w);times.append(now);names.append(name);checkpoint(j+1)
                if failure_time:break
    except BaseException as exc:
        c.dump(dest/f'ERROR_{time.time_ns()}.json',dict(error=repr(exc)));raise
    states=np.stack([np.load(dest/n,allow_pickle=False) for n in names]);c.atomic_npz(dest/'STATES.npz',time_s=np.array(times),omega=states)
    done=dict(status=status,first_failure_time_s=failure_time,new_neural_calls=calls,elapsed_seconds=time.perf_counter()-start,device='cpu')
    c.dump(dest/'SUMMARY.json',done)
    if failure_time:c.dump(dest/'FAILURE.json',done)
    return np.array(times),states,done

def run(run,resume=False):
    p=c.policy();old=validate_history();e=c.engine();registry=[];allrows=[];spectra=[];summaries=[];horizon=[];failures=[];paths={};raw={}
    for seed in c.SEEDS:
        q=c.read(run/'quality'/str(seed)/'QUALITY.json')
        if q['status']!='PASS_LONG_REFERENCE_QUALITY':raise RuntimeError('Reference quality not qualified')
        d=Path(q['production'])
        if c.sha(d/'DONE.json')!=q['production_done_sha256']:raise ValueError('Qualified production changed')
        if q['protocol_sha256']!=c.sha(run/'FREEZE.json'):raise ValueError('Reference was qualified under a different freeze')
        record=c.read(d/'DONE.json')
        for name in [*(k+'.npy' for k in c.FIELDS),'time_s.npy']:
            if c.sha(d/name)!=record['outputs'][name]:raise ValueError('Qualified low-grid array changed '+name)
        a=open_fields(d);cfg=c.read(d/'DONE.json')['config'];op=c.op(cfg['les_n'],cfg['length'])
        for dt in p['les_dt']:
            for weight in p['weights']:
                for method in c.METHODS:
                    key=(seed,weight,method,dt);cw=e.canonical_weight(method,weight)
                    alias=(seed,cw,method,dt)
                    if alias in raw:times,states,done=raw[alias]
                    else:
                        loaded=load_history(seed,cw,method,dt,a['reference_omega'][0],registry)
                        times,states,done=loaded if loaded else fresh_les(run,seed,cw,method,dt,a['reference_omega'][0],op,resume)
                        raw[alias]=times,states,done
                    dest=run/'analysis'/e.taskid(seed,weight,method,dt);dest.mkdir(parents=True,exist_ok=True)
                    rows=[];sp=[]
                    for t,w in zip(times,states):
                        ref=a['reference_omega'][c.index(a['time_s'],t)];r=state_metrics(op,w,ref,t);rows.append(r)
                        if abs((t-1)/.05-round((t-1)/.05))<1e-8:
                            hp=abs(np.fft.fft2(w)/w.size)**2;rh=abs(np.fft.fft2(ref)/ref.size)**2;ep=op.shell_power(w-ref)
                            inv=np.divide(1.,op.k2,out=np.zeros_like(op.k2),where=op.k2>0)
                            Ew=np.bincount(op.shell.ravel(),weights=(.5*hp*inv).ravel());Er=np.bincount(op.shell.ravel(),weights=(.5*rh*inv).ravel())
                            Zw=.5*op.shell_power(w);Zr=.5*op.shell_power(ref)
                            sp.extend(dict(time_s=float(t),shell=k,error_power=float(ep[k]),model_E=float(Ew[k]),reference_E=float(Er[k]),model_Z=float(Zw[k]),reference_Z=float(Zr[k])) for k in range(len(ep)))
                    context=dict(seed=seed,weight=weight,method=method,dt=dt,analytic_weight_alias=method=='Clark' and weight==2028)
                    sm=dict(**context,**summary(rows,done['status']),failure_time=done.get('first_failure_time_s'),
                        saved_time_cadence='historical short native LES step, long .05s plus failure endpoint; no interpolation')
                    c.csvout(dest/'ERROR_TIMEWISE.csv',rows);c.csvout(dest/'SPECTRA.csv',sp);c.dump(dest/'SUMMARY.json',sm)
                    summaries.append(sm);allrows.extend({**context,**r} for r in rows);spectra.extend({**context,**r} for r in sp)
                    horizon.extend({**context,**r} for r in horizons(rows,p['prediction_error_thresholds'],done['status'],done.get('first_failure_time_s')))
                    if done['status']!='COMPLETED':failures.append({**context,**done})
                    paths[key]=(times,states,rows,op,a)
                    c.log(run,f'ACCURACY {seed} {weight} {method} dt={dt} status={done["status"]}')
    # Common-time scoring has identical nodes for EVERY compared method in a case.
    common=[]
    for seed in c.SEEDS:
        for weight in p['weights']:
            for dt in p['les_dt']:
                rr=[paths[seed,weight,m,dt][2] for m in c.METHODS]
                nodes=set(round(r['time_s'],10) for r in rr[0])
                for r in rr[1:]:nodes&={round(x['time_s'],10) for x in r}
                nodes=sorted(nodes)
                for m,rows in zip(c.METHODS,rr):
                    sub=[r for r in rows if round(r['time_s'],10) in nodes]
                    common.append(dict(seed=seed,weight=weight,method=m,dt=dt,common_end=nodes[-1],**summary(sub,'COMMON_INTERVAL_ONLY')))
    # Fine/coarse comparison uses exact shared timestamps, never failure filling.
    dtrows=[]
    for seed in c.SEEDS:
        for w in p['weights']:
            for m in c.METHODS:
                ta,wa,ra,op,a=paths[seed,w,m,.005];tb,wb,rb,_,_=paths[seed,w,m,.0025]
                for i,t in enumerate(ta):
                    candidates=np.where(abs(tb-t)<1e-9)[0]
                    if len(candidates):dtrows.append(dict(seed=seed,weight=w,method=m,time_s=float(t),state_relative_dt_difference=c.relative(wa[i],wb[candidates[0]])))
    for name,rr in [('SUMMARY',summaries),('ERROR_TIMEWISE',allrows),('SPECTRA',spectra),('PREDICTION_HORIZONS',horizon),('COMMON_INTERVAL',common),('DT_COMPARISON',dtrows)]:c.csvout(run/(name+'.csv'),rr)
    c.dump(run/'TRAJECTORY_REUSE.json',registry);c.dump(run/'FAILURES.json',failures)
    source(run,paths)
    # Existing randomized matched-state benchmark is the fair cost comparator.
    cost=c.csvread(c.OLD/'results/campaign03/RHS_COST.csv');c.csvout(run/'RHS_COST.csv',[r for r in cost if r['method'] in c.METHODS])
    c.dump(run/'RHS_COST_PROVENANCE.json',dict(path=str(c.OLD/'results/campaign03/RHS_COST.csv'),sha256=c.sha(c.OLD/'results/campaign03/RHS_COST.csv'),
        scope='Historical CPU 2-thread same101/t1.1 state, 3 shuffled batches x8, preserved matched original prefix; excludes all I/O'))
    from reporting import render
    render(run,summaries,allrows,paths,failures)
    c.dump(run/'EVALUATION_DONE.json',dict(status='COMPLETED_WITH_FAILURES_RECORDED' if failures else 'COMPLETED',
        rows=len(summaries),independent_flows=2,reused_actual_trajectories=len(registry),locked_access=False,training=False))
    return summaries

def source(run,paths):
    e=c.engine();torch.set_num_threads(2);rows=[];spectra=[];start=time.perf_counter();calls=0
    for weight in c.policy()['weights']:
        solver=make_solver(weight,'FULL')
        with torch.inference_mode():
            for seed in c.SEEDS:
                _,_,_,op,a=paths[seed,weight,'FULL',.005]
                for t in 1+np.arange(81)*.05:
                    i=c.index(a['time_s'],t);w=a['reference_omega'][i];target=a['reference_source'][i]
                    out=solver.rhs(torch.tensor(np.array(w,copy=True),dtype=torch.float32),float(t));calls+=1
                    ss={k:v.numpy().astype(float) for k,v in selected_sources(solver,out).items()}
                    rb=ss['RemoveD']-target
                    for m,s in ss.items():
                        for b in ('all','low','high'):
                            err=band(op,s-target,b);r=band(op,rb,b);delta=band(op,s-ss['RemoveD'],b);y=band(op,target,b)
                            cross=2*float(np.mean(r*delta));power=c.rms(delta)**2
                            rows.append(dict(seed=seed,weight=weight,method=m,time_s=float(t),band=b,relative_L2=c.rms(err)/max(c.rms(y),1e-12*c.rms(target),1e-30),
                                base_error_power=c.rms(r)**2,error_power=c.rms(err)**2,cross_term=cross,correction_power=power,
                                identity_abs=abs(c.rms(err)**2-c.rms(r)**2-cross-power),
                                source_E=float(np.mean(op.psi(w)*s)),source_Z=float(np.mean(w*s)),reference_E=float(np.mean(op.psi(w)*target)),reference_Z=float(np.mean(w*target))))
                        ep=op.shell_power(s-target);rp=op.shell_power(target)
                        spectra.extend(dict(seed=seed,weight=weight,method=m,time_s=float(t),shell=k,error_power=float(x),reference_power=float(y)) for k,(x,y) in enumerate(zip(ep,rp)))
    c.csvout(run/'SOURCE_TIMEWISE.csv',rows);c.csvout(run/'SOURCE_SPECTRA.csv',spectra)
    c.dump(run/'SOURCE_COST.json',dict(device='cpu',actual_forward_calls=calls,seconds=time.perf_counter()-start,
        identity_max=max(r['identity_abs'] for r in rows),scope='a-priori reference-state, separate from autonomous trajectories'))

def selected_sources(solver,out):
    """Exactly seven frozen methods; do not recompute the old 12-point scan."""
    e=c.engine();sf=-out['div_q'];sb=-solver.project(solver.derivative(out['qb_x'],'x')+solver.derivative(out['qb_y'],'y'))
    qx,qy=e.native.clark_flux(out,solver.config.data.filter_width_m)
    sc=-solver.project(solver.derivative(qx,'x')+solver.derivative(qy,'y'))
    radius=torch.sqrt(solver.k_squared);high,low=e.native.responses(radius)
    optimized=-torch.expm1(-(radius/16).pow(4))
    return {
        'FULL':sf,'RemoveD':sb,'Clark':sc,
        'SSC_0.5_24':solver.project(sb+e.native.multiply(solver,sc-sb,high)),
        'SSC_1_16':solver.project(sb+e.native.multiply(solver,sc-sb,optimized)),
        'MixAll':solver.project(sb+.5*(sc-sb)),
        'P1':solver.project(sf+e.native.multiply(solver,sc-sf,low))}
