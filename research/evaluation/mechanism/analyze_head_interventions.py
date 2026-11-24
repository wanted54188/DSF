"""Bounded, read-only evidence audit; no CNN forward or time integration."""
import sys,json,hashlib,time,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'research/full_aug_parameter_pc_full_v1_20261116'
sys.path.insert(0,str(OLD))
import pc_common as c
import pc_metrics as m
from pc_engine import vec,D,cpu
P=json.loads((HERE/'PROTOCOL.json').read_text(encoding='utf-8'))

def sha(p):return c.sha(p)
def read(p):return c.read(p)
def dump(p,v):p.write_text(json.dumps(c.clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(p,rows):
    import csv as cc
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=cc.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(c.clean(rows))
def rms(a):return float(np.sqrt(np.mean(a*a)))
def pct(a,b):return 100*(1-a/b) if b else None

class Inputs:
    def __init__(self,run,out):
        self.run=run;self.out=out;self.hashes={};self.record=read(run/'RUN.json')
        self.verify(run/'RUN.json');self.verify(run/'LATEST_REPORT.json')
        for name,digest in self.record['code_identity'].items():self.verify(OLD/name,digest)
        for name,digest in self.record['inventory']['input_sha256'].items():
            if Path(name).suffix=='.py':self.verify(ROOT/name,digest)
        last=read(run/'LATEST_REPORT.json');self.report=run/last['path']
        assert sha(self.report/'MANIFEST.json')==last['manifest_sha256']
        self.manifest=read(self.report/'MANIFEST.json');self.verify(self.report/'MANIFEST.json')
    def verify(self,p,expected=None):
        p=p.resolve();key=str(p.relative_to(ROOT))
        if key not in self.hashes:
            h=sha(p)
            if expected is not None:assert h==expected,('input SHA changed',key)
            self.hashes[key]=h
        return p
    def artifact(self,name):return self.verify(self.report/name,self.manifest['outputs'][name])
    def reference(self,case,n):
        desc=self.record['inventory']['references'][str(case)]
        if desc['kind']=='dense':
            p=ROOT/desc['fields']['reference_omega'];key=str(p.relative_to(ROOT))
            self.verify(p,self.record['inventory']['input_sha256'][key])
            return np.array(np.load(p,mmap_mode='r')[4*n])
        p=ROOT/desc['frames'][str(n)];key=str(p.relative_to(ROOT))
        self.verify(p,self.record['inventory']['input_sha256'][key])
        with np.load(p,allow_pickle=False) as z:return np.array(z['reference_omega'])

def main(out):
    out.mkdir(parents=True,exist_ok=False);a=Inputs(ROOT/P['source'],out);begin=time.perf_counter()
    op=c.load(a.verify(a.run/'OPERATORS.npz'));masks,shell=m.geometry(op)
    rows=c.csvread(a.artifact('LONG_ROLLOUT_METRICS.csv'))
    idx={(int(r['case']),int(r['seed']),r['method'],int(r['step']),r['band']):r for r in rows}
    paired=[];checks=[];curves=[]
    allsummary={}
    for case in P['cases']:
        for seed in P['seeds']:
            for method in P['methods']:
                states=np.load(a.artifact(f'STATES_{case}_{seed}_{method}.npy'),mmap_mode='r')
                assert states.shape==(801,100,100)
                raw=[]
                for n in range(0,801,10):
                    ref=m.project(a.reference(case,n),op);w=states[n];st=m.fields(w,op);rt=m.fields(ref,op)
                    err=abs(m.F(w-ref))**2;power=abs(m.F(w))**2;rp=abs(m.F(ref))**2
                    for band,mask in masks.items():
                        mse=float(err[mask].sum());prev=float(idx[case,seed,method,n,band]['omega_MSE'])
                        assert abs(mse-prev)<1e-12*max(mse,1e-15)
                        K=.5*float((power*op['inverse'])[mask].sum());Kr=.5*float((rp*op['inverse'])[mask].sum())
                        Z=.5*float(power[mask].sum());Zr=.5*float(rp[mask].sum())
                        row=dict(case=case,seed=seed,method=method,step=n,time_s=1+n*.005,band=band,
                            omega_MSE=mse,K=K,Z=Z,ref_K=Kr,ref_Z=Zr,K_relative_bias=(K-Kr)/Kr,Z_relative_bias=(Z-Zr)/Zr,
                            u_relative_L2=rms(st['u']-rt['u'])/rms(rt['u']),v_relative_L2=rms(st['v']-rt['v'])/rms(rt['v']))
                        raw.append(row);curves.append(row)
                for band in masks:
                    vals=[r for r in raw if r['band']==band];assert len(vals)==81
                    for window,selected in [('full',vals[1:]),('early',[r for r in vals if 1.2-1e-10<=r['time_s']<=2.2+1e-10]),
                        ('late',[r for r in vals if r['time_s']>2.2+1e-10]),('t5',vals[-1:])]:
                        mean=lambda k:float(np.mean([r[k] for r in selected]))
                        v=dict(case=case,seed=seed,method=method,band=band,window=window,n=len(selected),MSE=mean('omega_MSE'),
                            abs_K_bias=float(np.mean([abs(r['K_relative_bias']) for r in selected])),
                            abs_Z_bias=float(np.mean([abs(r['Z_relative_bias']) for r in selected])),
                            u_relative_L2=mean('u_relative_L2'),v_relative_L2=mean('v_relative_L2'),Z=mean('Z'),ref_Z=mean('ref_Z'))
                        allsummary[case,seed,method,band,window]=v
                    v=allsummary[case,seed,method,band,'full']
                    v['time_integral_mean_MSE']=float(np.trapz([r['omega_MSE'] for r in vals],dx=.05)/4)
                checks.append(dict(case=case,seed=seed,method=method,states=801,uniform_frames=81,absolute_MSE_reconstruction='PASS'))
                print('Independent state MSE:',case%1000,seed,method,flush=True)
            for baseline in ('Fresh-C4','Parameter-PC','Aug'):
                for band in masks:
                    for window in ('full','early','late','t5'):
                        h=allsummary[case,seed,'Parameter-Hold',band,window];b=allsummary[case,seed,baseline,band,window]
                        paired.append(dict(case=case,seed=seed,baseline=baseline,band=band,window=window,
                            Hold_MSE=h['MSE'],baseline_MSE=b['MSE'],MSE_improvement_pct=pct(h['MSE'],b['MSE']),
                            abs_K_bias_change=h['abs_K_bias']-b['abs_K_bias'],abs_Z_bias_change=h['abs_Z_bias']-b['abs_Z_bias'],
                            u_relative_L2_improvement_pct=pct(h['u_relative_L2'],b['u_relative_L2']),
                            v_relative_L2_improvement_pct=pct(h['v_relative_L2'],b['v_relative_L2']),
                            mean_Z_change=h['Z']-b['Z'],reference_Z=h['ref_Z']))
    csv(out/'INDEPENDENT_STATE_METRICS.csv',curves);csv(out/'ABSOLUTE_MSE_CHECKS.csv',checks)
    csv(out/'WINDOW_ABSOLUTES.csv',list(allsummary.values()));csv(out/'HOLD_DIRECT_PAIRED.csv',paired)
    costs=c.csvread(a.artifact('COSTS.csv'));costidx={(int(r['case']),int(r['seed']),r['method']):r for r in costs}
    speed=[]
    for case in P['cases']:
        for seed in P['seeds']:
            h=costidx[case,seed,'Parameter-Hold'];f=costidx[case,seed,'Fresh-C4'];pc=costidx[case,seed,'Parameter-PC']
            speed.append(dict(case=case,seed=seed,Hold_CNN=int(h['actual_CNN']),Fresh_CNN=int(f['actual_CNN']),
                PC_CNN=int(pc['actual_CNN']),Hold_loop_seconds=float(h['measured_loop_seconds']),
                Fresh_loop_seconds=float(f['measured_loop_seconds']),PC_loop_seconds=float(pc['measured_loop_seconds']),
                Hold_measured_speedup=float(f['measured_loop_seconds'])/float(h['measured_loop_seconds']),
                CNN_saved_pct=100*(1-int(h['actual_CNN'])/int(f['actual_CNN']))))
    csv(out/'HOLD_MEASURED_COST.csv',speed)
    hc=c.csvread(a.artifact('HEAD_CONSISTENCY.csv'));heads=[]
    for case in P['cases']:
        for seed in P['seeds']:
            for method in P['methods'][2:]:
                for ch in range(7):
                    v=[r for r in hc if (int(r['case']),int(r['seed']),r['method'],int(r['channel']))==(case,seed,method,ch)]
                    heads.append(dict(case=case,seed=seed,method=method,channel=ch,
                        pooled_MSE=float(np.mean([float(r['MSE']) for r in v])),
                        pooled_relative_L2=float(np.sqrt(sum(float(r['MSE']) for r in v)/sum(float(r['fresh_head_power']) for r in v)))))
    csv(out/'HEAD_FULL_WINDOW.csv',heads)
    # Runtime stage budgets are separate from same-state interventions below.
    # Read original accepted metadata instead of comparing Source on unrelated states to DNS.
    budgets=[]
    for case in P['cases']:
        for seed in P['seeds']:
            for method in P['methods']:
                for n in range(10,801,10):
                    if n==800:
                        z=c.load(a.verify(a.run/'trajectories'/c.key(case,seed,method)/'ENDPOINT.npz'))
                    else:z=c.load(a.verify(a.run/'trajectories'/c.key(case,seed,method)/'steps'/f'{n+1:04d}.npz'))
                    rec=c.meta(z)['stages'][0]
                    budgets.append(dict(case=case,seed=seed,method=method,step=n,time_s=1+n*.005,
                        **{k:rec[k] for k in ('pi_d_mean','pi_b_mean','backscatter_fraction','original_limiter_fraction','qd_RMS','qb_RMS')},
                        **rec['budget']))
                print('Saved branch budgets:',case%1000,seed,method,flush=True)
    csv(out/'AUTONOMOUS_BRANCH_BUDGETS.csv',budgets)
    torch.set_num_threads(2);decoded=0;cf=[];temporal=[];models={}
    for seed in P['seeds']:
        row=next(v for v in a.record['inventory']['checkpoints'] if v['seed']==seed)
        for k,sk in (('file','sha256'),('run','run_sha256'),('done','done_sha256')):a.verify(ROOT/row[k],row[sk])
        s=c.model(row,'cpu');s.closure.register_forward_pre_hook(lambda *_:(_ for _ in ()).throw(AssertionError('Forbidden CNN forward')))
        models[seed]=s
    with torch.inference_mode():
        for case in P['cases']:
            for seed in P['seeds']:
                s=models[seed]
                for path in P['mechanism_state_paths']:
                    for t in P['mechanism_samples']:
                        n=round((t-1)/.005);folder=a.run/'trajectories'/c.key(case,seed,path)
                        z=c.load(a.verify(folder/('ENDPOINT.npz' if n==800 else f'steps/{n+1:04d}.npz')))
                        snap={k:v for k,v in z.items() if k!='metadata'} if n==800 else {k.removeprefix('observation__'):v for k,v in z.items() if k.startswith('observation__')}
                        # Endpoint did not persist bank metadata: its first pending
                        # anchor cannot change active pair, inherited from step800.
                        desc=c.meta(z).get('bank')
                        if desc is None:desc=c.meta(c.load(a.verify(folder/'steps/0800.npz')))['bank']
                        hs=[];steps=[]
                        for anchor in desc['active']:
                            az=c.load(a.verify(a.run/anchor['file'],anchor['sha256']));hs.append(torch.from_numpy(az['heads']));steps.append(anchor['step'])
                        tau=n-steps[-1];hold=hs[-1];trend=(1+tau)*hs[-1]-tau*hs[0];fresh=torch.from_numpy(snap['fresh_heads'])
                        actual=trend if path=='Parameter-PC' else hold
                        assert np.array_equal(cpu(actual),snap['predicted_heads'])
                        for ch in range(7):
                            d=cpu(hs[-1][:,ch]-hs[0][:,ch]);fh=cpu(fresh[:,ch]);hp=cpu(hold[:,ch]);tp=cpu(trend[:,ch])
                            temporal.append(dict(case=case,seed=seed,state_path=path,time_s=t,channel=ch,tau=tau,
                                raw_slope_RMS_per_second=rms(d)/.005,hold_MSE_to_fresh=float(np.mean((hp-fh)**2)),
                                trend_MSE_to_fresh=float(np.mean((tp-fh)**2)),
                                lag_alignment=float(np.mean((fh-hp)*d)),
                                ideal_tau_in_head_direction=float(np.mean((fh-hp)*d)/np.mean(d*d)) if np.any(d) else None))
                        variants={'hold':hold,'trend':trend,'fresh':fresh}
                        for ch in range(7):
                            h=hold.clone();h[:,ch]=trend[:,ch];variants[f'hold_channel_{ch}_trend']=h
                        w=torch.from_numpy(snap['omega']);st=s.state(w);hw=m.F(cpu(w));g=torch.stack((st['omega_x'],st['omega_y']))
                        for name,raw in variants.items():
                            parts=[]
                            for turn in range(4):
                                ss=s.state(c.q.spatial(w,turn));parts.append(c.decoder.decode(s,ss,raw[turn]));decoded+=1
                            mean=c.q.mean_fields(s,st,parts);d,b=vec(mean,'qd'),vec(mean,'qb');sd,sb=cpu(D(s,d)),cpu(D(s,b));source=sd+sb
                            check,_=c.feasible.envelope(s,st,d,b,parts[0]);pb=cpu(-(b*g).sum(0))
                            for band,mask in masks.items():
                                hd,hb,hq=m.F(sd),m.F(sb),m.F(source)
                                cf.append(dict(case=case,seed=seed,state_path=path,time_s=t,variant=name,band=band,
                                    Source_power=float((abs(hq)**2)[mask].sum()),Sd_power=float((abs(hd)**2)[mask].sum()),
                                    Sb_power=float((abs(hb)**2)[mask].sum()),
                                    Z_rate_d=float((hw.conj()*hd).real[mask].sum()),Z_rate_b=float((hw.conj()*hb).real[mask].sum()),
                                    K_rate=float(((hw.conj()*hq).real*op['inverse'])[mask].sum()),
                                    signed_negative_pi_fraction=float(np.mean(pb<0)),signed_injection_mean=float(np.mean(np.maximum(-pb,0))),
                                    signed_positive_pi_mean=float(np.mean(np.maximum(pb,0))),physical_envelope_pass=check['passed']))
                            if name==('trend' if path=='Parameter-PC' else 'hold'):
                                assert np.max(abs(source-snap['Source']))<1e-11*max(rms(source),1.)
                        print('Same-state raw-head intervention:',case%1000,seed,path,t,'decodes',decoded,flush=True)
    assert decoded==P['mechanism_max_algebraic_decodes']
    csv(out/'HEAD_TEMPORAL_DIAGNOSTIC.csv',temporal);csv(out/'SAME_STATE_HEAD_INTERVENTIONS.csv',cf)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(11,10))
    for i,case in enumerate(P['cases']):
        for j,seed in enumerate(P['seeds']):
            ax=axs[i,j]
            for method in P['methods']:
                rr=[r for r in curves if (r['case'],r['seed'],r['method'],r['band'])==(case,seed,method,'all')]
                ax.semilogy([r['time_s'] for r in rr[1:]],[r['omega_MSE'] for r in rr[1:]],label=method)
            ax.set_title(f'{case%1000}/{seed}');ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_xlabel('t');ax.set_ylabel('absolute omega MSE')
    fig.suptitle('Independent absolute MSE audit; REFERENCE_NOT_CONVERGED');fig.tight_layout();fig.savefig(out/'HOLD_MSE.pdf');fig.savefig(out/'HOLD_MSE.png',dpi=150);plt.close(fig)
    dump(out/'RUN.json',dict(status='VERIFIED_BOUNDED_POSTPROCESS',real_CNN=0,new_RHS=0,algebraic_decodes=decoded,
        seconds=time.perf_counter()-begin,reference_quality=P['reference_quality'],
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},protocol_sha256=sha(HERE/'PROTOCOL.json')))
    dump(out/'INPUT_SHA256.json',a.hashes)
    dump(out/'OUTPUT_SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='OUTPUT_SHA256.json'})
    print('VERIFIED',out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='audit01');args=ap.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('','.','..')
    main(HERE/'results'/args.run)
