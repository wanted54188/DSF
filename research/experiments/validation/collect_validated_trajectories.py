"""Read-only independent scoring; no model factory, CNN or integration calls."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import campaign as v
import pc_common as c,pc_metrics as m
def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def main():
    out=v.HERE/'REPORTS'/str(time.time_ns());out.mkdir(parents=True)
    oldrun=v.read(v.OLD/'results/run01/RUN.json');op=c.load(v.OLD/'results/run01/OPERATORS.npz')
    refs={int(k):c.References(desc) for k,desc in oldrun['inventory']['references'].items()}
    newref=v.dns_folder(110240105,1536,.000125)
    if (newref/'DONE.json').exists():
        done=v.read(newref/'DONE.json');assert all(v.sha(newref/n)==s for n,s in done['frames'].items())
        refs[110240105]=c.References(dict(kind='frames',frames={str(n):str((newref/f'frame_{n:04d}.npz').relative_to(v.ROOT)) for n in range(0,801,10)},steps=list(range(0,801,10))))
    audit=[];states=[];checks=[];inputs={};summary=[];cost=[];spectra=[];snap_checks=[];fluxes=[]
    def track(p):inputs[str(p.relative_to(v.ROOT))]=v.sha(p)
    def score(folder,case,seed,model,group,startw,status,isclark=False,initialw0=None):
        accepted=status.get('last_accepted_step',0)
        audit.append(dict(model=model,case=case,seed=seed,dt=.005,reference_version='new IC105 1536 h0.000125' if case==110240105 else 'frozen1536_half',experiment_status=status['status'],completed_t_end=1+.005*accepted,
            checkpoint_sha='' if isclark else next(r['sha256'] for r in oldrun['inventory']['checkpoints'] if r['seed']==seed),source_file=str(folder.relative_to(v.ROOT)),validation_status='NOT_SCORED_INCOMPLETE' if status['status']!='VERIFIED' else 'COMPLETE_IDENTITY_AND_RK4_CHECK_PENDING',group=group))
        if status['status']!='VERIFIED':return
        assert accepted==800
        paths=sorted((folder/'steps').glob('*.npz'));assert len(paths)==(800 if isclark else 799)
        w=startw;expected=1 if isclark else 2;maxerr=0.;frames={0:startw} if isclark else {0:initialw0} if initialw0 is not None else {}
        for p in paths:
            z=c.load(p);meta=c.meta(z);assert meta['accepted_step']==expected
            assert np.isfinite(z['omega']).all() and z['rhs'].shape==(4,100,100)
            metrics=m.verify_step(w,z['omega'],z['rhs'],op)
            checks.append(dict(case=case,seed=seed,model=model,group=group,step=expected,**metrics))
            w=z['omega'];expected+=1;track(p)
            if (expected-1)%10==0:frames[expected-1]=w
        audit[-1]['validation_status']='COMPLETE_IDENTITY_AND_NATIVE_RK4_VERIFIED'
        assert all(n in frames for n in range(10,801,10))
        rr=[]
        for n in ([0] if 0 in frames else [])+list(range(10,801,10)):
            ctx=dict(case=case,seed=seed,model=model,dt=.005,group=group,step=n,time=1+n*.005)
            scored=m.state_rows(frames[n],refs[case].at(n)['reference_omega'],op,ctx);states.extend(scored);rr.extend(scored)
            if not isclark and n>0:
                source=folder/'ENDPOINT.npz' if n==800 else folder/'steps'/f'{n+1:04d}.npz';z=c.load(source)
                snap=z if n==800 else {k.removeprefix('observation__'):a for k,a in z.items() if k.startswith('observation__')}
                assert np.max(abs(snap['omega']-frames[n]))<1e-12
                ss,ff,ch,cs=m.snapshot(snap,op,float(op['nu']),ctx);spectra.extend(ss);fluxes.append(ff);snap_checks.append(ch)
        full=[r for r in rr if r['band']=='all' and r['step']>0];high=[r for r in rr if r['band']=='high' and r['step']>0]
        summary.append(dict(case=case,seed=seed,model=model,group=group,dt=.005,samples=80,full_omega_MSE=float(np.mean([r['omega_MSE'] for r in full])),t5_omega_relative_L2=full[-1]['omega_relative_L2'],
            full_u_MSE=float(np.mean([r['u_MSE'] for r in full])),full_v_MSE=float(np.mean([r['v_MSE'] for r in full])),mean_abs_K_bias=float(np.mean([abs(r['K_relative_bias']) for r in full])),mean_abs_Z_bias=float(np.mean([abs(r['Z_relative_bias']) for r in full])),
            mean_high_Z_ratio=float(np.mean([r['Z']/r['reference_Z'] for r in high])),reference_quality='REFERENCE_NOT_CONVERGED'))
        calls=status.get('actual_calls',{})
        cost.append(dict(case=case,seed=seed,model=model,group=group,CNN=calls.get('CNN',0 if isclark else None),RHS=calls.get('RHS'),solver_seconds=status.get('solver_wall_seconds',status.get('compute_seconds')),IO_seconds=status.get('IO_wall_seconds'),status=status['status'],timing_scope='solver including guards; endpoint/IO separately recorded'))
        track(folder/'STATUS.json')
    for group in ('single','period4','period8','ic105'):
        root=v.HERE/'LES'/group
        for p in sorted((root/'trajectories').glob('*/STATUS.json')):
            case,seed,method=p.parent.name.split('_',2);case=int(case);seed=int(seed)
            st=c.load(root/'startup'/f'{case}_{seed}'/'STARTUP.npz');track(root/'RUN.json')
            score(p.parent,case,seed,v.NAMES[method],group,st['omega1'],v.read(p),initialw0=st['omega0'])
    for p in sorted((v.HERE/'CLARK').glob('*/limited_completion/Clark4_0.005/STATUS.json')):
        case=int(p.parents[2].name);folder=p.parent;z=c.load(folder/'INITIAL.npz')
        score(folder,case,'','Clark4','primary',z['omega'],v.read(p),True)
    dns=[]
    for p in sorted((v.HERE/'DNS').glob('*/RUN.json')):
        run=v.read(p);f=p.parent
        terminal=v.read(f/'DONE.json') if (f/'DONE.json').exists() else v.read(f/'FAILED.json') if (f/'FAILED.json').exists() else v.read(f/'STATUS.json') if (f/'STATUS.json').exists() else dict(status='NOT_RUN')
        dns.append(dict(case=run['case'],n=run['n'],dt=run['dt'],status=terminal['status'],completed_t_end=terminal.get('completed_t_end',terminal.get('time_s',0)),actual_RHS=terminal.get('actual_RHS'),max_tail=terminal.get('max_native_tail_fraction'),reference_quality='REFERENCE_NOT_CONVERGED',source=str(f.relative_to(v.ROOT))))
    for name,rr in [('RESULT_AUDIT',audit),('STATE_METRICS',states),('PAIRED_ACCURACY',summary),('COSTS',cost),('RK4_CHECKS',checks),('SPECTRA',spectra),('SNAPSHOT_CHECKS',snap_checks),('FLUX_BUDGETS',fluxes),('DNS_STATUS',dns)]:csvout(out/(name+'.csv'),rr)
    v.write(out/'INPUT_SHA256.json',inputs)
    v.write(out/'RECEIPT.json',dict(status='PASS_AVAILABLE_COMPLETE_SCOPE',complete_trajectories=len(summary),audited_trajectories=len(audit),checks=len(checks),training=0,CNN=0,new_RHS=0,reference_quality='REFERENCE_NOT_CONVERGED',note='Complete trajectories independently scored; ongoing/missing trajectories never filled. DNS qualification remains separate.'))
    v.write(v.HERE/'LATEST_REPORT.json',dict(path=str(out.relative_to(v.HERE)),sha256=v.sha(out/'RECEIPT.json')))
    print(out,flush=True)
if __name__=='__main__':main()
