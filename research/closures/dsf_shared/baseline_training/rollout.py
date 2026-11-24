"""New weights, original RK4, same initial states, no historical trajectory reuse."""
import json
import os
import time
import numpy as np
import torch
import common as c
import evaluation
import metrics


def matrix(a,stage):
    rows=[]
    for initial in c.CASES:
        for seed in a['protocol']['training_seeds']:
            for method in evaluation.METHODS:
                for dt in a['protocol']['rollout_dt']:
                    alias=method in ('Clark4','NoSGS') and seed!=2026
                    rows.append(dict(stage=stage,initial=initial,training_seed=seed,method=method,dt=dt,
                        analytic_alias=alias,steps=round((.2 if stage=='short' else 3.8)/dt),
                        expected_CNN_forwards=0 if alias or method in ('Clark4','NoSGS') else 4*round((.2 if stage=='short' else 3.8)/dt)))
    return rows


def name(row):return f'{row["initial"]}_{row["training_seed"]}_{row["method"]}_{row["dt"]:g}'


def references(a):
    result={}
    for r in a['references']:
        entry=r['fields']['reference_omega']
        if c.sha(entry['path'])!=entry['expected_sha256']:raise ValueError('Evaluation reference changed')
        result[r['seed'],r['variant']]=dict(time_s=np.load(r['time_path'],allow_pickle=False),
            omega=np.load(entry['path'],mmap_mode='r',allow_pickle=False))
    return result


def stage_power(stages,dt):
    g=metrics.geometry();out={}
    for weight,st in zip((1,2,2,1),stages):
        h=metrics.fft(st['omega'].detach().cpu().numpy())
        for label,field in [('SGS',st['source_applied']),('RHS',st['rhs'])]:
            sh=metrics.fft(field.detach().cpu().numpy());co=np.real(np.conj(h)*sh)
            for band,mask in metrics.bands(g).items():
                for q,v in [('Z',co),('E',co*g['inv'])]:
                    key=f'{label}_{band}_{q}_integral'
                    out[key]=out.get(key,0.)+float(weight*dt*v[mask].sum()/6)
    return out


def evaluate(dest,a,row,t,w,status,failure,refs,cost):
    records=[];summary=[];spectra=[]
    complete=status=='COMPLETED' and abs(t[-1]-(1.2 if row['stage']=='short' else 5))<1e-9
    for (initial,variant),ref in refs.items():
        if initial!=row['initial']:continue
        local=[]
        for tj,wj in zip(t,w):
            i=int(np.argmin(abs(ref['time_s']-tj)))
            if abs(ref['time_s'][i]-tj)>1e-9:raise ValueError('Exact reference time absent; interpolation forbidden')
            y=ref['omega'][i];m=metrics.state_row(wj,y,tj)
            failed_endpoint=failure is not None and tj>=failure-1e-9
            m.update(**row,reference=variant,quality='REFERENCE_NOT_CONVERGED',failure_endpoint=failed_endpoint,
                role=a['protocol']['evaluation_labels'][str(initial)])
            local.append(m)
            gp=metrics.geometry();mp=abs(metrics.fft(wj))**2;rp=abs(metrics.fft(y))**2;ep=abs(metrics.fft(wj-y))**2
            for k in range(48):
                mask=gp['mask']&(gp['shell']==k)
                spectra.append(dict(**row,reference=variant,time_s=float(tj),shell=k,model_power=float(mp[mask].sum()),
                    reference_power=float(rp[mask].sum()),error_power=float(ep[mask].sum()),failure_endpoint=failed_endpoint))
        records+=local;valid=[r for r in local if not r['failure_endpoint']]
        summary.append({**row,'reference':variant,'status':status,'completed_window':complete,'failure_time_s':failure,
            'quality':'REFERENCE_NOT_CONVERGED','role':a['protocol']['evaluation_labels'][str(initial)],**cost,
            **metrics.summarize(valid,complete)})
    c.csvout(dest/'ERROR_TIMEWISE.csv',records);c.csvout(dest/'SPECTRA.csv',spectra);c.dump(dest/'SUMMARY.json',summary)


def task(a,out,row,refs,device,resume=False):
    dest=out/row['stage']/name(row)
    checkpoint_identity=dict(code_sha256=a['code_sha256'],registry_sha256=c.sha(out/'BASELINE_REGISTRY.json'),row=row,device=device)
    if (dest/'DONE.json').exists():
        if not resume:raise FileExistsError(dest)
        done=c.read(dest/'DONE.json')
        if done['identity']!=checkpoint_identity:raise ValueError('Trajectory identity changed')
        for fname,h in done['outputs'].items():
            if c.sha(dest/fname)!=h:raise ValueError('Trajectory output changed')
        return
    s=evaluation.runtime(a,out,row['method'],row['training_seed'],device)
    initial=refs[row['initial'],a['protocol']['primary_variant']]['omega'][0]
    w0=s.project(torch.tensor(np.array(initial),dtype=torch.float32,device=device))
    prefix=None;begin,end=(1.,1.2) if row['stage']=='short' else (1.2,5.)
    if row['stage']=='long':
        short=out/'short'/name(row);receipt=c.read(short/'DONE.json')
        if receipt['status']!='COMPLETED' or c.sha(short/'STATES.npz')!=receipt['outputs']['STATES.npz']:
            raise ValueError('Matching completed short prefix required')
        with np.load(short/'STATES.npz',allow_pickle=False) as z:prefix=(z['time_s'].copy(),z['omega'].copy())
        if not np.array_equal(prefix[1][0],w0.cpu().numpy()):raise ValueError('Short initial state changed')
    w=w0 if prefix is None else torch.tensor(prefix[1][-1],device=device)
    times=[begin];states=[w.cpu().numpy().copy()];budgets=[];step=0;elapsed0=0.;rhs0=0.;calls0=0;nn0=0
    if dest.exists():
        if not resume or not (dest/'last.pt').exists():raise ValueError('Explicit legal resume required')
        old=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        if old['identity']!=checkpoint_identity:raise ValueError('Cross-method/device/checkpoint resume forbidden')
        w=old['omega'].to(device);times=old['times'];states=old['states'];budgets=old['budgets'];step=old['step']
        elapsed0=old['wall'];rhs0=old['rhs_seconds'];calls0=old['rhs_calls'];nn0=old['CNN_forwards']
    else:dest.mkdir(parents=True)
    t0=time.perf_counter();failure=None;status='COMPLETED';steps=round((end-begin)/row['dt'])
    m0=metrics.state_row(w0.cpu().numpy(),w0.cpu().numpy(),1.);e0=m0['E'];z0=m0['Z']
    print(f'LES START {row["stage"]} {name(row)}',flush=True)
    try:
        with torch.inference_mode():
            for j in range(step,steps):
                t=begin+j*row['dt'];before=w.cpu().numpy().copy();s.stages.clear()
                new=s.step(w,t,row['dt']);step=j+1;now=begin+step*row['dt']
                if not bool(torch.isfinite(new).all()):status='NONFINITE';failure=now;break
                after=new.cpu().numpy().copy();power=stage_power(s.stages,row['dt'])
                dw=after-before;power['Z_increment']=float(np.mean(before*dw+.5*dw*dw))
                power['Z_increment_minus_RK_stage']=power['Z_increment']-power['RHS_all_Z_integral']
                power['E_increment']=float(np.mean(metrics.psi(before,metrics.geometry())*dw+.5*metrics.psi(dw,metrics.geometry())*dw))
                power['E_increment_minus_RK_stage']=power['E_increment']-power['RHS_all_E_integral']
                budgets.append(dict(time_s=now,**power));w=new
                zh=.5*float(w.square().mean());eh=.5*float((w*s.state(w)['psi']).mean())
                if zh/max(z0,1e-24)>100 or eh/max(e0,1e-24)>100:status='SAFETY_STOP_100X';failure=now
                stride=1 if row['stage']=='short' else round(.05/row['dt'])
                if step%stride==0 or step==steps or failure is not None:times.append(now);states.append(after)
                if step%round(.1/row['dt'])==0 or step==steps or failure is not None:
                    c.save_pt(dest/'last.pt',dict(identity=checkpoint_identity,omega=w.cpu(),times=times,states=states,
                        budgets=budgets,step=step,wall=elapsed0+time.perf_counter()-t0,rhs_seconds=rhs0+s.rhs_seconds,
                        rhs_calls=calls0+s.rhs_calls,CNN_forwards=nn0+s.calls))
                if step%round(.5/row['dt'])==0:print(f'LES {name(row)} t={now:.3f}/{end}',flush=True)
                if failure is not None:break
        path=dest/'STATES.npz';tmp=dest/'STATES.tmp'
        with tmp.open('wb') as f:np.savez(f,time_s=np.array(times),omega=np.stack(states))
        os.replace(tmp,path);c.csvout(dest/'STAGE_BUDGET.csv',budgets)
        cost=dict(wall_seconds=elapsed0+time.perf_counter()-t0,RHS_seconds=rhs0+s.rhs_seconds,
            RHS_calls=calls0+s.rhs_calls,CNN_forwards=nn0+s.calls,device=device)
        tt,ww=np.array(times),np.stack(states)
        if prefix is not None:
            if not np.array_equal(prefix[1][-1],ww[0]):raise ValueError('Autonomous prefix mismatch')
            tt=np.r_[prefix[0],tt[1:]];ww=np.concatenate((prefix[1],ww[1:]))
        evaluate(dest,a,row,tt,ww,status,failure,refs,cost)
        c.dump(dest/'DONE.json',dict(status=status,identity=checkpoint_identity,**cost,
            outputs={p.name:c.sha(p) for p in dest.iterdir() if p.name in ('STATES.npz','STAGE_BUDGET.csv','SUMMARY.json','ERROR_TIMEWISE.csv','SPECTRA.csv')}))
    except BaseException as e:c.dump(dest/'FAILURE.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:s.capture_handle.remove()


def run(a,out,stage,device,resume=False):
    c.verify(a);evaluation.registry(a,out);torch.set_num_threads(a['protocol']['cpu_threads'])
    if stage=='long':
        for r in matrix(a,'short'):
            if not r['analytic_alias'] and c.read(out/'short'/name(r)/'DONE.json')['status']!='COMPLETED':
                raise RuntimeError('Long blocked by failed short trajectory')
    refs=references(a)
    for row in matrix(a,stage):
        if not row['analytic_alias']:task(a,out,row,refs,device,resume)
