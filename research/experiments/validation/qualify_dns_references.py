"""Fixed-gate reference qualification, including fine modes absent on coarse grid."""
import csv,json,time
from pathlib import Path
import numpy as np
import campaign as v
import pc_common as c
Q=v.read(v.ROOT/'research/full_aug_hold_audit_v1_20261117/PAPER_PROTOCOL.json')['reference_gates']
STEPS=tuple(range(0,801,10))
def output(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def members(case):
    old=v.read(v.OLD/'results/run01/RUN.json')['inventory']['references']
    ans=[]
    for n,dt in ((1536,.000125),(3072,.000125),(3072,.0000625)):
        f=v.dns_folder(case,n,dt)
        if (f/'DONE.json').exists():
            done=v.read(f/'DONE.json');assert done['completed_t_end']==5.
            for name,h in {**done['frames'],**done['native']}.items():assert v.sha(f/name)==h,'DNS output SHA changed'
            def sample(i,folder=f):
                z=c.load(folder/f'frame_{i:04d}.npz');assert abs(float(z['time_s'])-(1+.005*i))<1e-10,'Reference timestamp mismatch'
                return z
            def native(i,folder=f):
                with np.load(folder/f'native_{i:04d}.npz') as z:
                    assert abs(float(z['time_s'])-(1+.005*i))<1e-10,'Native timestamp mismatch'
                    h,md=np.array(z['h']),np.array(z['modes']);assert h.dtype==np.complex128 and h.shape==(len(md),len(md)) and np.isfinite(h).all()
                    return h,md
            ans.append(dict(n=n,dt=dt,sample=sample,native=native,done=done,source=str(f.relative_to(v.ROOT)),new=True))
        elif n==1536 and str(case) in old:
            ref=c.References(old[str(case)])
            if case in (110240101,110240102):
                f=v.ROOT/f'research/ssc_long_horizon_v1_20261030/results/run01/reference/{case}/n1536_half';done=v.read(f/'DONE.json')
                for key in ('reference_omega','reference_source','reference_dot','reference_qx','reference_qy'):
                    assert v.sha(f/(key+'.npy'))==done['outputs'][key+'.npy'],'Frozen coarse reference field changed'
                arrays={k:np.load(f/(k+'.npy'),mmap_mode='r') for k in ('reference_omega','reference_source','reference_dot','reference_qx','reference_qy')}
                def sample(i,aa=arrays):return {k:np.array(a[4*i]) for k,a in aa.items()}
                def native(i,folder=f,meta=done):
                    entry=next(a for a in meta['native'] if abs(a['time_s']-(1+.005*i))<1e-10);p=folder/entry['file'];assert v.sha(p)==entry['sha256']
                    w=np.load(p);h=np.fft.fft2(w,norm='forward');n=w.shape[-1];cut=(n-1)//3;md=np.arange(-cut,cut+1)
                    return h[np.ix_(md%n,md%n)],md
            else:
                f=v.ROOT/'research/full_c4_multi_ic_v1_20261110/results/paired01/reference'/str(case);done=v.read(f/'DONE.json')
                def sample(i,desc=old[str(case)],metadata=done):
                    p=v.ROOT/desc['frames'][str(i)];assert v.sha(p)==metadata['frames'][p.name],'104 reference field changed'
                    return c.load(p)
                native=None
            ans.append(dict(n=n,dt=dt,sample=sample,native=native,done=done,source=str(f.relative_to(v.ROOT)),new=False))
        else:ans.append(None)
    return ans
def main():
    rule=v.HERE/'QUALIFICATION_RULES_FREEZE.json'
    if rule.exists():
        frozen=v.read(rule);assert v.sha(v.HERE/'QUALIFICATION_RULES.json')==frozen['rules_sha256'] and v.sha(__file__)==frozen['qualification_code_sha256'],'Qualification rules changed after freeze'
    out=v.HERE/'QUALIFICATION'/str(time.time_ns());out.mkdir(parents=True)
    cases=[];convergence=[];native_rows=[];identities=[];tails=[];sources=[];reproductions=[]
    k=np.fft.fftfreq(100,d=.01).round();kx=k[None,:];ky=k[:,None];k2=kx*kx+ky*ky;inv=np.divide(1,k2,out=np.zeros_like(k2),where=k2>0)
    active=(abs(kx)<=33)&(abs(ky)<=33);active[0,0]=False;shell=np.round(np.sqrt(k2));masks={'all':active,'low':active&(shell>=1)&(shell<=16),'high_legacy':active&(shell>=17)&(shell<=71),'mid':active&(shell>=17)&(shell<=29),'high':active&(shell>=30)}
    F=lambda z:np.fft.fft2(z,norm='forward');I=lambda z:np.fft.ifft2(z,norm='forward').real;project=lambda z:I(F(z)*active)
    for case in (*v.CASES,110240105):
        mm=members(case);missing=[name for name,m in zip(('1536_main','3072_main','3072_half'),mm) if m is None];local_conv=[];local_native=[];local_id=[];local_tail=[]
        replay=True
        if case==110240104 and mm[0] is not None and mm[0]['new']:
            old=v.read(v.OLD/'results/run01/RUN.json')['inventory']['references'][str(case)]
            legacy=c.References(old)
            for n in STEPS:
                a=mm[0]['sample'](n);b=legacy.at(n)
                for field in ('reference_omega','reference_source','reference_dot'):
                    err=float(np.linalg.norm(a[field]-b[field])/max(np.linalg.norm(b[field]),1e-12));reproductions.append(dict(case=case,time_s=1+n*.005,field=field,relative_difference=err,passes=err<=1e-10));replay &= err<=1e-10
        for a in mm:
            if a is None:continue
            sources.append(dict(case=case,n=a['n'],dt=a['dt'],source=a['source']))
            obs=a['done'].get('observations')
            if not obs:
                p=v.ROOT/a['source']/'RAW_DIAGNOSTICS.csv'
                with p.open(encoding='utf-8-sig') as f:obs=list(csv.DictReader(f))
            tail=max(float(o['tail_fraction']) for o in obs)
            cfl=max(float(o.get('CFL') or 0) for o in obs)
            endpoint_cfl=None
            if a['native'] is not None:
                hh,md=a['native'](800);ng=a['n'];full=np.zeros((ng,ng),dtype=np.complex128);full[np.ix_(md%ng,md%ng)]=hh
                kg=np.fft.fftfreq(ng,d=1/ng).round();xx=kg[None,:];yy=kg[:,None];kk=xx*xx+yy*yy;iv=np.divide(1.,kk,out=np.zeros_like(kk),where=kk>0)
                uu=np.fft.ifft2(1j*yy*iv*full,norm='forward').real;vv=np.fft.ifft2(-1j*xx*iv*full,norm='forward').real;speed=float((abs(uu)+abs(vv)).max());dx=2*np.pi/ng
                endpoint_cfl=a['dt']*(speed/dx+4e-6/dx**2);cfl=max(cfl,endpoint_cfl)
                del full,hh,uu,vv,kk,iv
            row=dict(case=case,n=a['n'],dt=a['dt'],max_tail=tail,tail_limit=Q['tail_fraction'],tail_pass=tail<=Q['tail_fraction'],max_CFL=cfl,endpoint_CFL=endpoint_cfl,CFL_pass=cfl<=Q['CFL'],fine_candidate=a['n']==3072)
            local_tail.append(row);tails.append(row)
            for n in STEPS:
                z=a['sample'](n);w=z['reference_omega'];S=z['reference_source'];dot=z['reference_dot'];qx=z['reference_qx'];qy=z['reference_qy']
                hw=F(w);psi=hw*inv;u=I(1j*ky*psi);vv=I(-1j*kx*psi);gx=I(1j*kx*hw);gy=I(1j*ky*hw)
                resolved=project(-u*gx-vv*gy+1e-6*I(-k2*hw));flux=-I((1j*kx*F(qx)+1j*ky*F(qy))*active);scale=max(float(np.sqrt(np.mean(S*S))),1e-30)
                rr=dict(case=case,n=a['n'],dt=a['dt'],time_s=1+n*.005,RHS_relative=float(np.sqrt(np.mean((dot-resolved-S)**2))/scale),Flux_relative=float(np.sqrt(np.mean((flux-S)**2))/scale),mean_relative=float(abs(S.mean())/scale))
                rr['passes']=max(rr['RHS_relative'],rr['Flux_relative'])<=Q['source_flux_identity'] and rr['mean_relative']<=Q['mean_relative'];local_id.append(rr);identities.append(rr)
        for a,b,axis in ((mm[0],mm[1],'space'),(mm[1],mm[2],'time')):
            if a is None or b is None:continue
            accum={}
            for n in STEPS:
                za,zb=a['sample'](n),b['sample'](n)
                for field in ('reference_omega','reference_source','reference_dot'):
                    for band,mask in masks.items():
                        er=F(za[field]-zb[field])*mask;hr=F(zb[field]);ex=float(np.sum(abs(er)**2));den=float(np.sum(abs(hr*mask)**2));full=float(np.sum(abs(hr)**2));r=(ex/max(den,Q['normalization_floor']**2*full,1e-60))**.5
                        ac=accum.setdefault((field,band),[0.,0.,0.,0.,0.]);ac[0]+=ex;ac[1]+=den;ac[2]+=full
                        if r>ac[3]:ac[3]=r;ac[4]=1+n*.005
                if a['native'] is not None and b['native'] is not None:
                    ha,md=a['native'](n);hb,mf=b['native'](n);idx=md+len(mf)//2;common=hb[np.ix_(idx,idx)];outside=max(0.,float((abs(hb)**2).sum()-(abs(common)**2).sum()));den=float((abs(hb)**2).sum());error=float((abs(ha-common)**2).sum())+outside
                    def ez(h,modes):
                        kk=modes[:,None]**2+modes[None,:]**2;iv=np.divide(1.,kk,out=np.zeros_like(kk,dtype=float),where=kk>0);pp=abs(h)**2
                        return .5*float((pp*iv).sum()),.5*float(pp.sum())
                    ea,zz=ez(ha,md);eb,zt=ez(hb,mf);rel=(error/max(den,1e-60))**.5
                    nr=dict(case=case,axis=axis,time_s=1+n*.005,native_omega_relative=rel,native_E_relative=abs(ea-eb)/max(abs(eb),1e-30),native_Z_relative=abs(zz-zt)/max(abs(zt),1e-30),fine_discarded_power_fraction=outside/max(den,1e-60))
                    nr['passes']=max(nr['native_omega_relative'],nr['native_E_relative'],nr['native_Z_relative'])<=Q['native_raw_relative_limit'];local_native.append(nr);native_rows.append(nr)
            for (field,band),(ex,den,full,worst,peak) in accum.items():
                pooled=(ex/max(den,Q['normalization_floor']**2*full,1e-60))**.5;rr=dict(case=case,axis=axis,field=field,band=band,relative_RMS=pooled,max_frame_relative_RMS=worst,peak_time_s=peak,passes=max(pooled,worst)<=Q['relative_time_space']);local_conv.append(rr);convergence.append(rr)
        native_complete=len(local_native)==162
        fine_tails=[r for r in local_tail if r['fine_candidate']]
        gates=dict(complete_three_references=not missing,filtered_time_space=bool(local_conv) and all(r['passes'] for r in local_conv if r['band'] in ('all','low','high_legacy')),native_coverage=native_complete,native_time_space=native_complete and all(r['passes'] for r in local_native),operator_identities=bool(local_id) and all(r['passes'] for r in local_id),fine_tail=len(fine_tails)==2 and all(r['tail_pass'] for r in fine_tails),CFL=bool(local_tail) and all(r['CFL_pass'] for r in local_tail))
        if case==110240104 and mm[0] is not None and mm[0]['new']:gates['coarse_104_archival_reproduction']=replay
        status='PASS_REFERENCE_QUALITY' if all(gates.values()) else 'REFERENCE_NOT_CONVERGED'
        cases.append(dict(case=case,status=status,missing_variants=missing,gates=gates,thresholds=Q,native_frames=len(local_native),note='Additional mid/high diagnostics reported without replacing the legacy inherited gate. Fine tail applies to 3072 candidates; 1536 tail is explicitly reported.'))
    for name,rows in [('FILTERED_CONVERGENCE',convergence),('NATIVE_CONVERGENCE',native_rows),('OPERATOR_IDENTITIES',identities),('TAIL_CFL',tails),('SOURCES',sources),('REPRODUCTION_104',reproductions)]:output(out/(name+'.csv'),rows)
    v.write(out/'QUALIFICATION.json',dict(cases=cases,training=0,CNN=0,new_RHS=0,scope='Empirical frozen-grid/time/archived-frame qualification; not a continuum theorem'))
    v.write(v.HERE/'LATEST_QUALIFICATION.json',dict(path=str(out.relative_to(v.HERE)),sha256=v.sha(out/'QUALIFICATION.json')))
    print(json.dumps(cases,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
