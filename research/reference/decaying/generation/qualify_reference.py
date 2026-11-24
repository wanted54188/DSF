"""Empirical spatial/time reference qualification, independent of SGS identity."""
from pathlib import Path
import numpy as np
import common as c
from reference import open_fields

def band(op,a,name):
    k=op.shell
    mask=op.mask & ((k>=1)&(k<=16) if name=='low' else (k>=17)&(k<=71) if name=='high' else True)
    return np.fft.ifft2(np.fft.fft2(a)*mask).real

def convergence(op,a,b,q,axis):
    if not np.allclose(a['time_s'],b['time_s'],rtol=0,atol=1e-11):raise ValueError('Nonpaired convergence times')
    rows=[]
    for key in ('reference_omega','reference_source','reference_dot'):
        for name in ('all','low','high'):
            sd=sy=sf=0.;worst=(-1.,None,0.,0.);floor_frames=0
            for i,t in enumerate(a['time_s']):
                target=band(op,b[key][i],name);err=band(op,a[key][i]-b[key][i],name)
                full=c.rms(b[key][i]);x=c.rms(err);y=c.rms(target);floor=q['normalization_floor']*full
                r=x/max(y,floor,1e-30);floor_frames+=y<floor;sd+=x*x;sy+=y*y;sf+=full*full
                if r>worst[0]:worst=(r,float(t),x,y)
            glob=np.sqrt(sd)/max(np.sqrt(sy),q['normalization_floor']*np.sqrt(sf),1e-30)
            rows.append(dict(axis=axis,field=key,band=name,relative_RMS=float(glob),max_frame_relative_RMS=worst[0],
                peak_time_s=worst[1],peak_difference_RMS=worst[2],peak_reference_RMS=worst[3],floor_active_frames=int(floor_frames),
                threshold=q['relative_time_space'],passes=bool(max(glob,worst[0])<=q['relative_time_space'])))
    return rows

def native_difference(a,b,length):
    """Full fine-space norm: common-mode error PLUS discarded fine modes."""
    na,nb=a.shape[-1],b.shape[-1]
    if na>nb:raise ValueError('Coarse/fine ordering')
    ha=np.fft.fft2(a)/na**2;hb=np.fft.fft2(b)/nb**2
    ka=(na-1)//3;ks=np.arange(-ka,ka+1)
    aa=ha[np.ix_(ks%na,ks%na)];bb=hb[np.ix_(ks%nb,ks%nb)]
    outside=max(0.,float(np.sum(abs(hb)**2)-np.sum(abs(bb)**2)))
    error=float(np.sum(abs(aa-bb)**2))+outside;den=float(np.sum(abs(hb)**2))
    def ez(h,n):
        modes=np.fft.fftfreq(n,d=1/n);k2=(modes[:,None]**2+modes[None,:]**2)*(2*np.pi/length)**2
        inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0);p=abs(h)**2
        return .5*np.sum(p*inv),.5*np.sum(p)
    ea,za=ez(ha,na);eb,zb=ez(hb,nb)
    return dict(native_omega_relative=float(np.sqrt(error/max(den,1e-60))),fine_discarded_power_fraction=outside/max(den,1e-60),
        native_E_relative=float(abs(ea-eb)/max(abs(eb),1e-30)),native_Z_relative=float(abs(za-zb)/max(abs(zb),1e-30)))

def qualify(run,case):
    seed=case['seed'];c.seed_guard(seed);p=c.policy();q=p['quality'];d=run/'quality'/str(seed);d.mkdir(parents=True,exist_ok=True)
    base=run/'reference'/str(seed);arrays={v:open_fields(base/v) for v in c.VARIANTS}
    op=c.op(case['variants'][0]['config']['les_n'],case['variants'][0]['config']['length']);nu=case['variants'][0]['config']['nu']
    conv=[];identities=[];native=[];tails=[];overlap=[];reference_metrics=[];reference_spectra=[]
    for left,right,axis in ((c.VARIANTS[0],c.VARIANTS[1],'space'),(c.VARIANTS[1],c.VARIANTS[2],'time')):
        conv+=convergence(op,arrays[left],arrays[right],q,axis)
        la=c.read(base/left/'DONE.json');rb=c.read(base/right/'DONE.json')
        for row in rb['native']:
            old=next((x for x in la['native'] if abs(x['time_s']-row['time_s'])<1e-9),None)
            if old is None:raise ValueError('Missing paired native snapshot')
            pa,pb=base/left/old['file'],base/right/row['file']
            if c.sha(pa)!=old['sha256'] or c.sha(pb)!=row['sha256']:raise ValueError('Native snapshot changed')
            a=np.load(c.array_guard(pa),allow_pickle=False);b=np.load(c.array_guard(pb),allow_pickle=False)
            diff=native_difference(a,b,case['variants'][0]['config']['length'])
            native.append(dict(axis=axis,time_s=row['time_s'],**diff,threshold=p['native_raw_relative_limit'],
                passes=max(diff[k] for k in ('native_omega_relative','native_E_relative','native_Z_relative'))<=p['native_raw_relative_limit']))
    for v,a in arrays.items():
        cfg=case['variants'][0]['config'];count=round(4/cfg['record_dt'])+1;n=cfg['les_n']
        if a['reference_omega'].shape!=(count,n,n) or not np.allclose(a['time_s'],1+np.arange(count)*cfg['record_dt'],atol=1e-10,rtol=0):
            raise ValueError('Wrong low grid/time coverage')
        for i,t in enumerate(a['time_s']):
            if any(not np.isfinite(a[k][i]).all() for k in c.FIELDS):raise ValueError('Nonfinite reference')
            w=a['reference_omega'][i];s=a['reference_source'][i];scale=max(c.rms(s),1e-30)
            rhs=op.project(-np.sum(np.stack((op.derivative(op.psi(w),'y'),-op.derivative(op.psi(w),'x')))*op.grad(w),axis=0))+nu*op.project(np.fft.ifft2(-op.k2*np.fft.fft2(w)).real)
            flux=op.source(np.stack((a['reference_qx'][i],a['reference_qy'][i])))
            identities.append(dict(variant=v,time_s=float(t),RHS_relative=c.rms(a['reference_dot'][i]-rhs-s)/scale,
                flux_relative=c.rms(flux-s)/scale,mean_relative=abs(float(s.mean()))/scale))
            if v==p['production']:
                psi=op.psi(w);dot=a['reference_dot'][i];visc=nu*np.fft.ifft2(-op.k2*np.fft.fft2(w)).real
                reference_metrics.append(dict(time_s=float(t),E=float(.5*np.mean(w*psi)),Z=float(.5*np.mean(w*w)),
                    E_dot=float(np.mean(psi*dot)),Z_dot=float(np.mean(w*dot)),source_E=float(np.mean(psi*s)),source_Z=float(np.mean(w*s)),
                    viscosity_E=float(np.mean(psi*visc)),viscosity_Z=float(np.mean(w*visc)),omega_mean=float(w.mean()),source_mean=float(s.mean())))
                if abs((float(t)-1)/.05-round((float(t)-1)/.05))<1e-8:
                    ww=op.shell_power(w);ss=op.shell_power(s);dd=op.shell_power(dot);ee=.5*op.shell_cospectrum(w,psi)
                    reference_spectra.extend(dict(time_s=float(t),shell=j,energy=float(ee[j]),enstrophy=float(.5*ww[j]),source_power=float(ss[j]),dot_power=float(dd[j])) for j in range(len(ww)))
        obs=c.csvread(base/v/'RAW_DIAGNOSTICS.csv')
        maxcfl=max([float(x.get('CFL') or 0) for x in obs]+[0.])
        # Original prefix CFL is already qualified but is retained explicitly.
        old=next(x for x in case['variants'] if x['variant']==v)
        maxcfl=max(maxcfl,c.read(Path(old['path'])/'REFERENCE_METADATA.json')['max_CFL'])
        maxtail=max(float(x['tail_fraction']) for x in obs)
        tails.append(dict(variant=v,max_CFL=maxcfl,max_raw_tail=maxtail,CFL_pass=maxcfl<=q['CFL'],
            candidate=v!='n1024_main',tail_pass=maxtail<=q['tail_fraction'],tail_limit=q['tail_fraction'],CFL_limit=q['CFL']))
        overlap+=c.csvread(base/v/'OVERLAP.csv')
    # Official READY identity is checked before prefix comparison; no external STAR field.
    if c.sha(case['ready_path'])!=case['ready_sha256']:raise ValueError('Original READY identity changed')
    with c.archive(case['ready_path']) as z:
        prod=arrays[p['production']];tt=np.array(z['time_s'])
        if not np.allclose(tt,prod['time_s'][:len(tt)],atol=1e-12,rtol=0):raise ValueError('READY overlap time mismatch')
        for i,t in enumerate(tt):
            overlap.append(dict(time_s=float(t),kind='official_READY_dense_prefix',**{k:c.relative(prod[k][i],z[k][i]) for k in ('reference_omega','reference_source','reference_dot')},
                spectrum=c.relative(abs(np.fft.fft2(prod['reference_omega'][i]))**2,abs(np.fft.fft2(z['reference_omega'][i]))**2)))
    gates=dict(filtered_time_space=bool(conv) and all(r['passes'] for r in conv),native_time_space=bool(native) and all(r['passes'] for r in native),
        operator_identities=all(max(r['RHS_relative'],r['flux_relative'])<=q['source_flux_identity'] and r['mean_relative']<=q['mean_relative'] for r in identities),
        candidate_tail=all(r['tail_pass'] for r in tails if r['candidate']),CFL=all(r['CFL_pass'] for r in tails),
        prefix_identity=all(max(float(r[k]) for k in ('reference_omega','reference_source','reference_dot','spectrum'))<=p['overlap_relative_limit'] for r in overlap))
    result=dict(status='PASS_LONG_REFERENCE_QUALITY' if all(gates.values()) else 'INCONCLUSIVE_BLOCKED_REFERENCE_QUALITY',seed=seed,
        gates=gates,production=str(base/p['production']),production_done_sha256=c.sha(base/p['production']/'DONE.json'),
        protocol_sha256=c.sha(run/'FREEZE.json'),quality=q,native_raw_limit=p['native_raw_relative_limit'],DNS_claim=False,
        uncertainty='Empirical whole-window/sparse-native convergence evidence, not rigorous continuum certification',
        failed_filtered=[r for r in conv if not r['passes']],failed_native=[r for r in native if not r['passes']])
    for name,rows in [('CONVERGENCE',conv),('NATIVE_CONVERGENCE',native),('OPERATOR_IDENTITIES',identities),('TAIL_CFL',tails),('OVERLAP',overlap)]:c.csvout(d/(name+'.csv'),rows)
    c.csvout(d/'REFERENCE_METRICS.csv',reference_metrics);c.csvout(d/'REFERENCE_SPECTRA.csv',reference_spectra)
    c.dump(d/'QUALITY.json',result)
    return result
