"""Existing reference variants only; no reference generation or CNN."""
import sys,json
from pathlib import Path
import numpy as np
from analyze import HERE,ROOT,OLD,sha,read,csv,dump
import pc_common as c
import pc_metrics as m

def main():
    out=HERE/'results/reference01';out.mkdir(parents=True,exist_ok=False)
    run=ROOT/'research/full_aug_parameter_pc_full_v1_20261116/results/run01'
    rep=run/read(run/'LATEST_REPORT.json')['path'];manifest=read(rep/'MANIFEST.json');files={}
    def use(p,expected=None):
        p=Path(p);h=sha(p)
        if expected:assert h==expected
        files[str(p.relative_to(ROOT))]=h;return p
    audit=read(use(c.q.study.inputs.RUN/'AUDIT.json'))
    op=c.load(use(run/'OPERATORS.npz'));masks,_=m.geometry(op);rows=[];quality=[]
    for case in (110240101,110240102):
        qpath=ROOT/f'research/ssc_long_horizon_v1_20261030/results/run01/quality/{case}/QUALITY.json'
        q=read(use(qpath));quality.append(dict(case=case,status=q['status'],**q['gates'],reference_quality=c.QUALITY))
        for v in ('n1024_main','n1536_main','n1536_half'):
            rec=next(x for x in audit['references'] if x['seed']==case and x['variant']==v)
            f=rec['fields']['reference_omega'];rr=np.load(use(Path(f['path']),f['expected_sha256']),mmap_mode='r')
            tt=np.load(use(Path(rec['time_path']),rec['time_sha256']))
            for seed in (2026,2028):
                states={name:np.load(use(rep/f'STATES_{case}_{seed}_{name}.npy',manifest['outputs'][f'STATES_{case}_{seed}_{name}.npy']),mmap_mode='r')
                    for name in ('Fresh-C4','Parameter-Hold')}
                by={band:[] for band in masks}
                for n in range(0,801,10):
                    ix=int(np.argmin(abs(tt-(1+n*.005))));assert abs(tt[ix]-(1+n*.005))<1e-10
                    ref=m.project(rr[ix],op)
                    ep={name:abs(m.F(a[n]-ref))**2 for name,a in states.items()}
                    for band,mask in masks.items():by[band].append({name:float(p[mask].sum()) for name,p in ep.items()})
                for band,series in by.items():
                    for window,values in (('full',series[1:]),('t5',series[-1:])):
                        h=float(np.mean([x['Parameter-Hold'] for x in values]));f=float(np.mean([x['Fresh-C4'] for x in values]))
                        rows.append(dict(case=case,seed=seed,reference_variant=v,band=band,window=window,
                            Hold_MSE=h,Fresh_MSE=f,Hold_improvement_pct=100*(1-h/f),reference_quality=c.QUALITY))
            print('Existing reference rescored:',case%1000,v,flush=True)
    csv(out/'EXISTING_REFERENCE_RANKING.csv',rows);csv(out/'EXISTING_REFERENCE_GATES.csv',quality)
    dump(out/'RUN.json',dict(status='VERIFIED_EXISTING_REFERENCE_SENSITIVITY',real_CNN=0,new_DNS=0,
        scope='Existing1024/1536 space comparison and1536 dt comparison. Does not resolve failed reference convergence. 104 lacks comparable variants.',reference_quality=c.QUALITY))
    dump(out/'INPUT_SHA256.json',files)

if __name__=='__main__':main()
