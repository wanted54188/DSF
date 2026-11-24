"""Fixed saved LES states scored against every complete reference variant."""
import json,time
import numpy as np
import campaign as v
import pc_common as c,pc_metrics as m
from qualify import members,output
def main():
    old=v.OLD/'results/run01';report=old/v.read(old/'LATEST_REPORT.json')['path'];manifest=v.read(report/'MANIFEST.json')
    op=c.load(old/'OPERATORS.npz');masks,_=m.geometry(op);rows=[];checks=[]
    for case in (*v.CASES,110240105):
        refs=members(case);complete=[(i,r) for i,r in enumerate(refs) if r is not None]
        if len(complete)<2:continue
        truths={i:{n:m.project(r['sample'](n)['reference_omega'],op) for n in range(10,801,10)} for i,r in complete}
        models=[]
        if case in v.CASES:
            for seed in v.SEEDS:
                for method in ('Aug','Fresh-C4','Parameter-PC','Parameter-Hold'):
                    name=f'STATES_{case}_{seed}_{method}.npy';p=report/name;assert v.sha(p)==manifest['outputs'][name]
                    aa=np.load(p,mmap_mode='r');models.append((v.NAMES[method],seed,{n:np.array(aa[n]) for n in range(10,801,10)}))
        else:
            root=v.HERE/'LES/ic105'
            for seed in v.SEEDS:
                for method in ('Aug','Fresh-C4','Parameter-PC','Parameter-Hold'):
                    folder=root/'trajectories'/c.key(case,seed,method)
                    if not (folder/'STATUS.json').exists() or v.read(folder/'STATUS.json')['status']!='VERIFIED':continue
                    models.append((v.NAMES[method],seed,{n:c.load(folder/'steps'/f'{n:04d}.npz')['omega'] for n in range(10,801,10)}))
        folder=v.HERE/'CLARK'/str(case)/'limited_completion/Clark4_0.005'
        if (folder/'STATUS.json').exists() and v.read(folder/'STATUS.json')['status']=='VERIFIED':models.append(('Clark4','',{n:c.load(folder/'steps'/f'{n:04d}.npz')['omega'] for n in range(10,801,10)}))
        for model,seed,states in models:
            metrics={}
            for i,r in complete:
                metrics[i]={band:[] for band in masks}
                for n in range(10,801,10):
                    for row in m.state_rows(states[n],truths[i][n],op,{}):metrics[i][row['band']].append(row)
            for band,mask in masks.items():
                scores={}
                for i,r in complete:
                    detail=metrics[i][band];vals=[a['omega_MSE'] for a in detail];scores[i]=vals
                    rows.append(dict(case=case,seed=seed,model=model,band=band,reference_n=r['n'],reference_dt=r['dt'],samples=80,full_omega_MSE=float(np.mean(vals)),t5_omega_MSE=vals[-1],t5_omega_relative_L2=detail[-1]['omega_relative_L2'],full_u_MSE=float(np.mean([a['u_MSE'] for a in detail])),full_v_MSE=float(np.mean([a['v_MSE'] for a in detail])),mean_abs_K_bias=float(np.mean([abs(a['K_relative_bias']) for a in detail])),mean_abs_Z_bias=float(np.mean([abs(a['Z_relative_bias']) for a in detail])),mean_Z_ratio=float(np.mean([a['Z']/a['reference_Z'] for a in detail])),reference_quality='SEE_SEPARATE_QUALIFICATION'))
                if 2 in scores:
                    for i,r in complete:
                        if i==2:continue
                        bounds=[]
                        for j,n in enumerate(range(10,801,10)):
                            dr=float((abs(m.F(truths[i][n]-truths[2][n]))**2)[mask].sum());bound=2*np.sqrt(scores[2][j]*dr)+dr;actual=abs(scores[i][j]-scores[2][j]);assert actual<=bound+1e-12
                            bounds.append(bound)
                        checks.append(dict(case=case,seed=seed,model=model,band=band,reference_n=r['n'],reference_dt=r['dt'],fine_n=refs[2]['n'],fine_dt=refs[2]['dt'],absolute_full_window_MSE_drift=abs(np.mean(scores[i])-np.mean(scores[2])),mean_framewise_uncertainty_bound=float(np.mean(bounds))))
    out=v.HERE/'REFERENCE_RESCORING'/str(time.time_ns());out.mkdir(parents=True)
    output(out/'MODEL_REFERENCE_SCORES.csv',rows);output(out/'UNCERTAINTY_BOUNDS.csv',checks)
    v.write(out/'RECEIPT.json',dict(status='PASS_AVAILABLE_COMPLETE_SCOPE',rows=len(rows),bounds=len(checks),CNN=0,new_RHS=0,no_interpolation=True))
    v.write(v.HERE/'LATEST_REFERENCE_RESCORING.json',dict(path=str(out.relative_to(v.HERE))))
    print('Reference rescoring rows',len(rows),'bounds',len(checks),flush=True)
if __name__=='__main__':main()
