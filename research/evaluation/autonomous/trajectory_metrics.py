"""Independent NumPy scoring: exact saved states, production symbols, no CNN."""
import numpy as np

F=lambda v:np.fft.fft2(v,norm='forward')
I=lambda v:np.fft.ifft2(v,norm='forward').real


def geometry(op):
    k=np.fft.fftfreq(100,d=.01).round(); radius=np.hypot(k[:,None],k[None,:]); shell=radius.round().astype(int)
    active=op['dealias'] & (shell>0)
    return dict(all=active,low=active&(shell<=16),mid=active&(shell>=17)&(shell<=29),
                high=active&(shell>=30),corner=active&(radius>33)),shell


def project(w,op):
    h=F(w)*op['dealias']; h[0,0]=0; return I(h)
def D(q,op): return -project(I(1j*op['kx']*F(q[0])+1j*op['ky']*F(q[1])),op)
def fields(w,op):
    h=F(w)*op['dealias']; hp=h*op['inverse']; hp[0,0]=0
    return dict(psi=I(hp),u=I(1j*op['ky']*hp),v=I(-1j*op['kx']*hp),
                gx=I(1j*op['kx']*h),gy=I(1j*op['ky']*h),lap=I(-op['k_squared']*h))
def ratio(a,b): return float(a/b) if b!=0 else None


def state_rows(w,reference,op,ctx):
    masks,_=geometry(op); h=F(w); power=abs(h)**2; st=fields(w,op)
    ref=project(reference,op) if reference is not None else None
    rh=F(ref) if ref is not None else None; rt=fields(ref,op) if ref is not None else None
    uerr=float(np.mean((st['u']-rt['u'])**2)) if rt else None
    verr=float(np.mean((st['v']-rt['v'])**2)) if rt else None
    ur=float(np.mean(rt['u']**2)) if rt else None; vr=float(np.mean(rt['v']**2)) if rt else None
    rows=[]
    Kvelocity=.5*float(np.mean(st['u']**2+st['v']**2)); K=.5*float((power*op['inverse']).sum())
    assert abs(Kvelocity-K)<=5e-5*max(K,1.)
    for band,mask in masks.items():
        k=.5*float((power*op['inverse'])[mask].sum()); z=.5*float(power[mask].sum())
        mse=float((abs(h-rh)**2)[mask].sum()) if rh is not None else None
        kr=.5*float((abs(rh)**2*op['inverse'])[mask].sum()) if rh is not None else None
        zr=.5*float((abs(rh)**2)[mask].sum()) if rh is not None else None
        row=dict(**ctx,band=band,reference_available=ref is not None,omega_MSE=mse,
            omega_relative_L2=(mse/(2*zr))**.5 if zr else None,K=k,Z=z,reference_K=kr,reference_Z=zr,
            K_relative_bias=(k-kr)/kr if kr else None,Z_relative_bias=(z-zr)/zr if zr else None,
            u_MSE=uerr,v_MSE=verr,u_relative_L2=(uerr/ur)**.5 if ur else None,v_relative_L2=(verr/vr)**.5 if vr else None,
            velocity_relative_L2=((uerr+verr)/(ur+vr))**.5 if rt and ur+vr else None,
            K_from_velocity=Kvelocity if band=='all' else None,K_velocity_minus_frozen=Kvelocity-K if band=='all' else None)
        rows.append(row)
    assert abs(rows[0]['Z']-sum(r['Z'] for r in rows[1:4]))<=1e-12*max(rows[0]['Z'],1.)
    if ref is not None:
        assert abs(rows[0]['omega_MSE']-sum(r['omega_MSE'] for r in rows[1:4]))<=1e-12*max(rows[0]['omega_MSE'],1.)
    return rows


def snapshot(snap,op,nu,ctx):
    masks,shell=geometry(op); w=snap['omega']; h=F(w); st=fields(w,op)
    d=np.stack((snap['qd_x'],snap['qd_y'])); b=np.stack((snap['qb_x'],snap['qb_y']))
    sd,sb,source=D(d,op),D(b,op),D(d+b,op)
    adv=-project(st['u']*st['gx']+st['v']*st['gy'],op)
    visc=project(nu*st['lap'],op)
    terms=dict(advection=adv,viscosity=visc,forcing=np.zeros_like(w),Sd=sd,Sb=sb,SGS=source,RHS=snap['rhs'])
    checks=dict(**ctx,Source_max_abs_reconstruction=float(np.max(abs(source-snap['Source']))),
        branch_Source_max_abs_reconstruction=float(np.max(abs(source-sd-sb))),
        RHS_max_abs_reconstruction=float(np.max(abs(adv+visc+source-snap['rhs']))),
        Z_Parseval_residual=abs(.5*float(np.mean(w*w))-.5*float((abs(h)**2).sum())))
    pi=-np.sum((d+b)*np.stack((st['gx'],st['gy'])),axis=0)
    # Independent witness validation at every full-flux observation. All other
    # stage envelope results remain explicit runtime receipts, not a fake replay.
    A=snap['witness_A']; coeff=snap['witness_c']; g=np.stack((st['gx'],st['gy']))
    gn=np.sqrt(np.sum(g*g,axis=0)+1e-16)
    reconstructed=-coeff*float(op['filter_width'])**3*gn*np.stack((A[0]*g[0]+A[1]*g[1],A[1]*g[0]+A[2]*g[1]))
    half=.5*(A[0]+A[2]); rad=np.hypot(.5*(A[0]-A[2]),A[1])
    assert np.min(half-rad)>0 and np.max((half+rad)/(half-rad))<=16*(1+1e-12)
    assert np.max(abs(A[0]+A[2]-2))<1e-12 and np.min(coeff)>0 and np.max(coeff)<=float(op['coefficient_max'])*(1+1e-12)
    assert np.max(abs(reconstructed-d))<=1e-12*max(float(np.max(abs(d))),1.)
    support=np.sqrt(np.sum(d*d,axis=0)+snap['gradient_support_norm']**2+2e-16)
    amplitude=np.linalg.norm(b,axis=0)/(float(op['beta'])*support)
    allowed=float(op['transfer_max'])*(np.maximum(-np.sum(d*g,axis=0),0)+snap['gradient_support_norm']*gn)
    transfer=np.maximum(np.sum(b*g,axis=0),0)/np.maximum(allowed,1e-300)
    assert np.max(amplitude)<=1+1e-12 and np.max(transfer)<=1+1e-12
    checks.update(witness_qd_max_abs_error=float(np.max(abs(reconstructed-d))),
                  signed_amplitude_ratio=float(np.max(amplitude)),signed_transfer_ratio=float(np.max(transfer)))
    checks['SGS_enstrophy_ibp_residual']=abs(float(np.mean(w*source)+np.mean(pi)))
    scale=max(float(np.sqrt(np.mean(snap['rhs']**2))),1.)
    assert max(checks[k] for k in ('Source_max_abs_reconstruction','branch_Source_max_abs_reconstruction',
                                  'RHS_max_abs_reconstruction','Z_Parseval_residual','SGS_enstrophy_ibp_residual'))<=1e-12*scale
    spectra=dict(E=.5*abs(h)**2*op['inverse'],Z=.5*abs(h)**2)
    for name,v in terms.items():
        transfer=(h.conj()*F(v)).real
        spectra['T_Z_'+name]=transfer; spectra['T_K_'+name]=transfer*op['inverse']
    spectral=[dict(**ctx,shell=k,**{name:float(value[masks['all']&(shell==k)].sum()) for name,value in spectra.items()}) for k in range(48)]
    flux=dict(**ctx,qd_RMS=float(np.sqrt(np.mean(d*d))),qb_RMS=float(np.sqrt(np.mean(b*b))),
        pi_mean=float(pi.mean()),pi_RMS=float(np.sqrt(np.mean(pi*pi))),backscatter_fraction=float(np.mean(pi<0)),
        SGS_K_rate=float(np.mean(st['psi']*source)),SGS_Z_rate=float(np.mean(w*source)))
    consistency=[]
    if 'fresh_delta_d' in snap:
        original=snap['original_qd']+snap['original_qb']
        actual=source-D(original,op); fresh=D(snap['fresh_delta_d']+snap['fresh_delta_b'],op)
        err=F(actual-fresh); hf=F(fresh)
        for band,mask in masks.items():
            error=float((abs(err)**2)[mask].sum()); power=float((abs(hf)**2)[mask].sum())
            row=dict(**ctx,band=band,correction_MSE_to_fresh=error,fresh_correction_power=power,
                correction_relative_L2=(error/power)**.5 if power else None,
                target_kind='SAME_AUTONOMOUS_STATE_MODEL_FRESH_NOT_DNS')
            for name,weight in (('K',st['psi']),('Z',w)):
                hh=F(weight)
                row[name+'_rate_error_to_fresh']=float((hh.conj()*err).real[mask].sum())
                row[name+'_fresh_correction_rate']=float((hh.conj()*hf).real[mask].sum())
            consistency.append(row)
    return spectral,flux,checks,consistency


def verify_step(w,new,ks,op,h=.005):
    """Reconstruct all native stages and exact quadratic RK defect, without CNN."""
    weights=np.array([1,2,2,1])/6
    unprojected=w+(h/6)*(ks[0]+2*ks[1]+2*ks[2]+ks[3])
    replay=project(unprojected,op)
    maxerr=float(np.max(abs(replay-new)))
    assert maxerr<=1e-12*max(float(np.max(abs(new))),1.)
    Y=np.stack((w,w+.5*h*ks[0],w+.5*h*ks[1],w+h*ks[2]))
    hy=F(Y); hk=F(ks); hw=F(w); hn=F(new)
    a=np.array([[0,0,0,0],[.5,0,0,0],[0,.5,0,0],[0,0,1,0]])
    coeff=np.outer(weights,weights)-weights[:,None]*a-weights[None,:]*a.T
    result=dict(native_RK4_max_abs_replay_error=maxerr)
    masks,_=geometry(op)
    for band,mask in masks.items():
        for name,mult in (('K',op['inverse']),('Z',np.ones_like(op['inverse']))):
            delta=.5*float(((abs(hn)**2-abs(hw)**2)*mult)[mask].sum())
            quad=h*sum(weights[i]*float(((hy[i].conj()*hk[i]).real*mult)[mask].sum()) for i in range(4))
            defect=.5*h*h*sum(coeff[i,j]*float(((hk[i].conj()*hk[j]).real*mult)[mask].sum()) for i in range(4) for j in range(4))
            residual=delta-quad-defect
            assert abs(residual)<=1e-12*max(abs(delta),abs(quad),1.)
            result[band+'__'+name+'_change']=delta; result[band+'__'+name+'_RK_quadrature']=quad
            result[band+'__'+name+'_quadratic_defect']=defect; result[band+'__'+name+'_algebra_residual']=residual
    return result


def reference_spectra(sample,op,ctx):
    w=project(sample['reference_omega'],op); st=fields(w,op); h=F(w); masks,shell=geometry(op)
    terms=dict(advection=-project(st['u']*st['gx']+st['v']*st['gy'],op),
        viscosity=project(float(op['nu'])*st['lap'],op),forcing=np.zeros_like(w),
        SGS=sample['reference_source'],RHS=sample['reference_dot'])
    values=dict(E=.5*abs(h)**2*op['inverse'],Z=.5*abs(h)**2)
    for name,v in terms.items():
        z=(h.conj()*F(v)).real; values['T_Z_'+name]=z; values['T_K_'+name]=z*op['inverse']
    return [dict(**ctx,shell=k,**{name:float(v[masks['all']&(shell==k)].sum()) for name,v in values.items()}) for k in range(48)]
