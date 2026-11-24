"""Consistent source/state metrics on the 100^2 rectangular K33 grid."""
import numpy as np


def geometry(n=100):
    k=np.fft.fftfreq(n,d=1/n);ky,kx=np.meshgrid(k,k,indexing='ij');r=np.hypot(kx,ky)
    mask=(abs(kx)<=n//3)&(abs(ky)<=n//3);mask[0,0]=False
    inv=np.divide(1.,kx*kx+ky*ky,out=np.zeros_like(kx),where=r>0)
    shell=np.rint(r).astype(int)
    return dict(mask=mask,inv=inv,shell=shell)


def bands(g):
    return dict(all=g['mask'],low=g['mask']&(g['shell']<=16),high=g['mask']&(g['shell']>=17))


def fft(x):return np.fft.fft2(x,norm='forward')
def psi(x,g):return np.fft.ifft2(fft(x)*g['inv'],norm='forward').real


def source_rows(w,target,sources,ctx):
    g=geometry(w.shape[-1]);wh=fft(w);yh=fft(target);rows=[];spectra=[]
    for method,source in sources.items():
        h=fft(source);e=h-yh
        for b,mask in bands(g).items():
            power=float((abs(yh)**2)[mask].sum());mse=float((abs(e)**2)[mask].sum())
            denom=float(np.sqrt((abs(h)**2)[mask].sum()*max(power,1e-30)))
            rows.append(dict(**ctx,method=method,band=b,MSE=mse,source_relative_L2=np.sqrt(mse/max(power,1e-24)),
                correlation=float(np.real(np.conj(yh)*h)[mask].sum()/max(denom,1e-30)),
                SGS_E=float((np.real(np.conj(wh)*h)*g['inv'])[mask].sum()),
                SGS_Z=float(np.real(np.conj(wh)*h)[mask].sum()),
                reference_E=float((np.real(np.conj(wh)*yh)*g['inv'])[mask].sum()),
                reference_Z=float(np.real(np.conj(wh)*yh)[mask].sum())))
        powers={k:np.bincount(g['shell'].ravel(),weights=(v*g['mask']).ravel()) for k,v in
                dict(source_power=abs(h)**2,error_power=abs(e)**2).items()}
        for k in range(len(powers['source_power'])):spectra.append(dict(**ctx,method=method,shell=k,**{name:float(v[k]) for name,v in powers.items()}))
    return rows,spectra


def state_row(w,y,t):
    g=geometry(w.shape[-1]);h=fft(w);yh=fft(y);e=h-yh;row=dict(time_s=float(t))
    for b,mask in bands(g).items():
        ep=float((abs(e)**2)[mask].sum());yp=float((abs(yh)**2)[mask].sum())
        row['omega_relative_L2_'+b]=np.sqrt(ep/max(yp,1e-24));row['error_power_'+b]=ep
    for label,v in [('E',g['inv']),('Z',np.ones_like(g['inv']))]:
        model=.5*float((abs(h)**2*v*g['mask']).sum());ref=.5*float((abs(yh)**2*v*g['mask']).sum())
        row[label]=model;row['reference_'+label]=ref
        row[label+'_signed_relative_bias']=(model-ref)/max(abs(ref),1e-24)
        row[label+'_absolute_relative_bias']=abs(row[label+'_signed_relative_bias'])
    row['tail_fraction']=float((abs(h)**2)[g['mask']&(g['shell']>=30)].sum()/max((abs(h)**2)[g['mask']].sum(),1e-24))
    return row


def summarize(rows,complete):
    result={}
    for key in ('omega_relative_L2_all','omega_relative_L2_low','omega_relative_L2_high',
                'E_absolute_relative_bias','Z_absolute_relative_bias','tail_fraction'):
        result[key+'_endpoint']=rows[-1][key] if complete else None
        result[key+'_window']=float(np.trapezoid([r[key] for r in rows],[r['time_s'] for r in rows])/
            (rows[-1]['time_s']-rows[0]['time_s'])) if complete and len(rows)>1 else None
    return result
