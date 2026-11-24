"""Fixed rotation-consistency directional pilot, strictly capped at 96 CNN."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode=True
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research/full_c4_equivariant_training_v1_20261109'))
import study
from model import spatial,vector

COUNTS=dict(CNN=0,native_RHS=0,endpoint_closure_only=0,independent_RK4_steps=0,Source_FFT=0,synthetic=0)


def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p).resolve();require(p.is_relative_to(HERE),'Output escaped new version directory')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
def rms(v):return float(torch.linalg.vector_norm(v)/math.sqrt(v.numel()))
def relative(a,b):return rms(a-b)/max(rms(b),1e-30)
def vv(f,prefix):return torch.stack((f[prefix+'x'],f[prefix+'y']))
def iv(v,turn):return torch.stack(vector(v[0],v[1],-turn))
def tensor(a,turn):
    a,b,d=(spatial(x,-turn) for x in (a['A11'],a['A12'],a['A22']))
    return torch.stack((d,-b,a) if turn%2 else (a,b,d))
def source(s,q):
    COUNTS['Source_FFT']+=1
    return -s.project(s.derivative(q[0],'x')+s.derivative(q[1],'y'))


def masks(s):
    k=torch.fft.fftfreq(100,d=.01,device=s.kx.device).round()
    radius=torch.hypot(k[:,None],k[None,:]);shell=radius.round()
    band=s.dealias&(shell>0)
    return dict(all=band,low=band&(shell<=16),mid=band&(shell>=17)&(shell<=29),
                high=band&(shell>=30),corner=band&(radius>33))


def physical(s,st,f):
    g=torch.stack((st['omega_x'],st['omega_y']))
    gn=(g.square().sum(0)+1e-16).sqrt()
    d,b=vv(f,'qd_'),vv(f,'qb_')
    coeff=f['c_theta'];a11,a12,a22=(f[k] for k in ('A11','A12','A22'))
    nu=coeff*s.closure.filter_width**3*gn
    reconstructed=-nu*torch.stack((a11*g[0]+a12*g[1],a12*g[0]+a22*g[1]))
    half=(a11+a22)/2;rad=torch.hypot((a11-a22)/2,a12)
    mineig=half-rad;condition=(half+rad)/mineig
    pd=-(d*g).sum(0);pb=-(b*g).sum(0)
    prior=f['gradient_support_norm']
    support=(d.square().sum(0)+prior.square()+2e-16).sqrt()
    beta=s.closure.backscatter_factor*s.closure.maximum_backscatter_ratio
    transfer=s.closure.maximum_local_backscatter_transfer_ratio*(pd.relu()+prior*gn)
    ratio=float((torch.linalg.vector_norm(b,dim=0)/(beta*support).clamp_min(1e-24)).max())
    tratio=float(((-pb).relu()/transfer.clamp_min(1e-24)).max())
    leith=relative(reconstructed,d)
    measured=[float(mineig.min()),float(condition.max()),float((a11+a22-2).abs().max()),ratio,tratio,leith]
    require(np.isfinite(measured).all() and measured[0]>0 and measured[1]<=16+1e-12
            and max(measured[2],leith)<=1e-12 and ratio<=1+1e-12 and tratio<=1+1e-12,
            'Original physical envelope violated: '+str(measured))
    require(float(pd.min())>=-1e-12*max(rms(pd),1e-24),'Negative qd transfer')
    require(bool((coeff>0).all()) and float(coeff.max())<=s.closure.dissipation_coefficient_soft_limit+1e-12,
            'Dissipation coefficient bound')
    return dict(min_eigenvalue=measured[0],max_condition=measured[1],trace_error=measured[2],
                signed_amplitude_ratio=ratio,signed_transfer_ratio=tratio,leith_relative=leith,
                pi_d_mean=float(pd.mean()),pi_b_mean=float(pb.mean()))


def mean_fields(s,st,branches):
    d=torch.stack([iv(vv(f,'qd_'),j) for j,f in enumerate(branches)]).mean(0)
    b=torch.stack([iv(vv(f,'qb_'),j) for j,f in enumerate(branches)]).mean(0)
    coeffs=torch.stack([spatial(f['c_theta'],-j) for j,f in enumerate(branches)])
    coeff=coeffs.mean(0)
    a=(torch.stack([coeffs[j]*tensor(f,j) for j,f in enumerate(branches)]).mean(0)/coeff)
    return dict(qx=d[0]+b[0],qy=d[1]+b[1],qd_x=d[0],qd_y=d[1],qb_x=b[0],qb_y=b[1],
                c_theta=coeff,A11=a[0],A12=a[1],A22=a[2],
                gradient_support_norm=branches[0]['gradient_support_norm'])


@torch.inference_mode()
def synthetic():
    """No network forward: exercise actual P33 symbols, vector/tensor rules."""
    s=study.make_solver(study.AUGMENTED,2026).double().eval().requires_grad_(False)
    x=torch.arange(100,dtype=torch.float64)*2*math.pi/100
    y,x=torch.meshgrid(x,x,indexing='ij')
    w=s.project(torch.sin(3*x+2*y)+.3*torch.cos(11*x-9*y))
    q=torch.stack((torch.cos(4*x+y)+.2*torch.sin(35*x-2*y),torch.sin(x-7*y)))
    src=source(s,q); errors=[]
    for j in (1,2,3):
        qr=torch.stack(vector(q[0],q[1],j));sr=source(s,qr)
        errors.append(relative(sr,spatial(src,j)))
    require(max(errors)<1e-10,'Synthetic discrete divergence rotation failed')
    # Orbit variance / directional gain identities use fixed manufactured data.
    orbit=torch.stack((src,src+.13*w,src-.07*w,src+.04*w));mean=orbit.mean(0)
    truth=src+.03*w
    lhs=((orbit-truth)**2).mean();rhs=((mean-truth)**2).mean()+((orbit-mean)**2).mean()
    require(abs(float(lhs-rhs))<1e-12*max(float(lhs),1.),'Orbit MSE identity')
    r=.2*w;d=.005*(mean-orbit[0]);gain=2*(r*d).mean()-d.square().mean()
    require(abs(float(gain-(r.square().mean()-(r-d).square().mean())))<1e-12,'Step gain identity')
    # Test convex qd tensor reconstruction using prescribed admissible fields.
    st=s.state(w);g=torch.stack((st['omega_x'],st['omega_y']));gn=(g.square().sum(0)+1e-16).sqrt()
    branches=[]
    for j in range(4):
        wr=spatial(w,j);sr=s.state(wr);gg=torch.stack((sr['omega_x'],sr['omega_y']))
        c=torch.ones_like(w)*s.closure.dissipation_coefficient_soft_limit*(.2+.1*j)
        A11=1+.15*torch.cos(x);A12=.08*torch.sin(y);A22=2-A11
        nu=c*s.closure.filter_width**3*(gg.square().sum(0)+1e-16).sqrt()
        qd=-nu*torch.stack((A11*gg[0]+A12*gg[1],A12*gg[0]+A22*gg[1]))
        qb=.2*qd
        branches.append(dict(c_theta=c,A11=A11,A12=A12,A22=A22,qd_x=qd[0],qd_y=qd[1],
                             qb_x=qb[0],qb_y=qb[1],gradient_support_norm=torch.ones_like(w)))
    result=physical(s,st,mean_fields(s,st,branches))
    COUNTS['synthetic']=1
    return dict(status='PASS',divergence_C4_max_relative=max(errors),physical_mean=result,
                real_CNN=0,real_RHS=0,identities=['orbit_squared_error','endpoint_gain','Leith_SPD_reconstruction'])


def inputs(files):
    frozen_path=ROOT/'research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json'
    frozen=read(frozen_path);files[str(frozen_path)]=sha(frozen_path)
    require(frozen['quality']=='REFERENCE_NOT_CONVERGED','Quality changed')
    for section in ('training_code_sha256','baseline_code_sha256'):
        for name,h in frozen[section].items():
            require(sha(ROOT/name)==h,'Historical implementation changed: '+name);files[name]=h
    models=[]
    for r in frozen['checkpoints']:
        if r['mode']!=study.AUGMENTED:continue
        for k,h in (('file','sha256'),('run','run_sha256'),('done','done_sha256')):
            require(sha(ROOT/r[k])==r[h],'Frozen checkpoint identity changed');files[r[k]]=r[h]
        models.append(r)
    require({r['seed'] for r in models}=={2026,2028},'Two frozen augmented seeds required')
    audit=read(study.inputs.RUN/'AUDIT.json');selected={}
    for case in (110240101,110240102):
        row=next(r for r in audit['references'] if r['seed']==case and r['variant']=='n1536_half')
        meta=row['fields']['reference_omega'];path=study.inputs.safe(meta['path'])
        require(sha(path)==meta['expected_sha256'],'Reference omega identity');files[str(path)]=meta['expected_sha256']
        require(sha(row['time_path'])==row['time_sha256'],'Reference times identity');files[row['time_path']]=row['time_sha256']
        times=np.load(row['time_path'],allow_pickle=False);values=np.load(path,mmap_mode='r',allow_pickle=False)
        require(values.shape==(3201,100,100) and values.dtype==np.float64,'Reference shape/dtype')
        for t in (1.2,1.5,2.):
            for h in (0,.005):
                index=round((t+h-1)/.00125);require(abs(times[index]-t-h)<1e-10,'Exact reference timestamp')
                selected[case,round(t+h,8)]=np.array(values[index])
    return models,selected


def make_model(row,cap):
    payload=torch.load(ROOT/row['file'],map_location='cpu',weights_only=True)
    require(payload['identity']==read(ROOT/row['run']) and payload['epoch']==row['epoch']
            and payload['backscatter_factor']==1.,'Payload training identity')
    s=study.make_solver(study.AUGMENTED,row['seed']).double().cuda().eval().requires_grad_(False)
    s.load_state_dict(payload['model'],strict=True)
    native=s.rhs
    def counted_rhs(w,t):
        require(COUNTS['native_RHS']<cap['native_RHS'],'RHS budget exhausted; remaining NOT_RUN')
        COUNTS['native_RHS']+=1
        return native(w,t)
    def hook(*_):
        require(COUNTS['CNN']<cap['CNN'],'CNN budget exhausted; remaining NOT_RUN')
        COUNTS['CNN']+=1
    handle=s.closure.register_forward_pre_hook(hook);s.rhs=counted_rhs
    return s,handle


def score(s,p,truth,S,ctx,tolerance):
    band=masks(s);fft=lambda a:torch.fft.fft2(a,norm='forward')
    mean=S.mean(0);crot=mean-S[0];d=.005*crot;r=truth-p
    fields=dict(baseline=p,symmetry_kick=p+d,opposite_kick=p-d)
    hr,hd=fft(r),fft(d);q=fft(S);hqmean=q.mean(0)
    power=hd.abs().square();variance=((q-hqmean).abs().square()).mean(0)
    rows=[];states=[]
    for name,mask in band.items():
        A=float(power[mask].sum());B=float(hr[mask].abs().square().sum())
        C=float((hr.conj()*hd).real[mask].sum());gain=2*C-A
        actual=float((fft(fields['symmetry_kick']-truth).abs().square())[mask].sum())
        require(abs(gain-(B-actual))<=1e-12*max(B,actual,1e-30),'Spectral gain identity')
        # RMS error induced by measured commuting/reconstruction discrepancies.
        eps=tolerance*.005*max(rms(mean),rms(S[0]))
        resolution=4*eps*B**.5+6*eps*A**.5+8*eps**2
        v=float(variance[mask].sum());normalizer=float(q[0].abs().square()[mask].sum())
        rows.append(dict(**ctx,band=name,baseline_MSE=B,corrected_MSE=actual,
                         opposite_MSE=B+2*C+A,gain=gain,pairing=C,correction_power=A,
                         improvement_pct=100*gain/B if B else None,cosine=C/(A*B)**.5 if A*B else None,
                         oracle_alpha=C/A if A else None,variance_Source=v,normalized_variance=v/normalizer if normalizer else None,
                         degeneracy=bool(A==0 or B==0 or normalizer==0),empirical_resolution=resolution))
    for label,state in fields.items():
        st=s.state(state);target=s.state(truth)
        hh,hy=fft(state),fft(truth);error=hh-hy
        uv_err=float(((st['u']-target['u'])**2+(st['v']-target['v'])**2).mean())
        uv_pow=float((target['u']**2+target['v']**2).mean())
        for name,mask in band.items():
            Z=.5*float(hh.abs().square()[mask].sum());Zr=.5*float(hy.abs().square()[mask].sum())
            K=.5*float((hh.abs().square()*s.inverse_k_squared)[mask].sum());Kr=.5*float((hy.abs().square()*s.inverse_k_squared)[mask].sum())
            mse=float(error.abs().square()[mask].sum());zp=float(hy.abs().square()[mask].sum())
            states.append(dict(**ctx,variant=label,band=name,MSE=mse,omega_relative_L2=(mse/zp)**.5 if zp else None,
                               velocity_MSE=uv_err,velocity_relative_L2=(uv_err/uv_pow)**.5 if uv_pow else None,
                               K=K,K_ref=Kr,K_relative_bias=(K-Kr)/Kr if Kr else None,
                               Z=Z,Z_ref=Zr,Z_relative_bias=(Z-Zr)/Zr if Zr else None))
    # Squared-error identity over four endpoint counterfactuals, not 4 RK4s.
    orbit_states=p+.005*(S-S[0]);lhs=float((orbit_states-truth).square().mean())
    rhs=float((fields['symmetry_kick']-truth).square().mean())+.005**2*float((S-mean).square().mean())
    require(abs(lhs-rhs)<=1e-12*max(lhs,1e-30),'State orbit variance identity')
    mask=band['all'];require(abs(rows[0]['gain']-sum(v['gain'] for v in rows[1:4]))<=1e-12,'Band partition')
    return rows,states,dict(state_orbit_identity_residual=abs(lhs-rhs),correction_RMS=rms(d))


@torch.inference_mode()
def execute(out):
    protocol=read(HERE/'PROTOCOL.json');cap=protocol['hard_budget']
    out.mkdir(parents=True,exist_ok=False);files={};started=time.perf_counter()
    dump(out/'RUN.json',dict(protocol=protocol,protocol_sha256=sha(HERE/'PROTOCOL.json'),
                            code_sha256={p.name:sha(p) for p in HERE.glob('*.py')}))
    test=synthetic();dump(out/'SYNTHETIC.json',test)
    models,refs=inputs(files);metrics=[];states=[];checks=[];physics=[]
    for model in models:
        s,handle=make_model(model,cap);before=study.common.model_hash(s)
        for case in protocol['cases']:
            for t in protocol['times']:
                ctx=dict(case=case,seed=model['seed'],time_s=t,h=.005)
                # Future reference is intentionally accessed below, after the
                # predictor and every rotation closure have been evaluated.
                w0=s.project(torch.as_tensor(refs[case,t],device='cuda'))
                wp=s.step(w0,t,.005);COUNTS['independent_RK4_steps']+=1
                require(bool(torch.isfinite(wp).all()),'Nonfinite predictor')
                branches=[];sources=[]
                for j in range(4):
                    require(COUNTS['endpoint_closure_only']<cap['endpoint_closure_only'],'Endpoint budget exhausted')
                    COUNTS['endpoint_closure_only']+=1
                    st=s.state(spatial(wp,j));f=s.closure_fields(st)
                    chk=physical(s,st,f);physics.append(dict(**ctx,orientation=j,**chk))
                    src=source(s,vv(f,'q'))
                    sources.append(spatial(src,-j));branches.append(f)
                S=torch.stack(sources);st=s.state(wp);mean=mean_fields(s,st,branches)
                chk=physical(s,st,mean);physics.append(dict(**ctx,orientation='mean',**chk))
                sm=source(s,vv(mean,'q'));error=relative(sm,S.mean(0))
                require(error<=1e-12,'Mean flux/Source inconsistency')
                # Exact reindexing of the cached four members verifies the
                # Reynolds average without any extra neural forward.
                cached_covariance=0.
                for turn in (1,2,3):
                    transformed=torch.stack([spatial(S[(j+turn)%4],turn) for j in range(4)]).mean(0)
                    cached_covariance=max(cached_covariance,relative(transformed,spatial(S.mean(0),turn)))
                require(cached_covariance<=1e-10,'Cached group reindexing failure')
                truth=s.project(torch.as_tensor(refs[case,round(t+.005,8)],device='cuda'))
                eps=max(error,cached_covariance,test['divergence_C4_max_relative'],torch.finfo(wp.dtype).eps)
                mr,sr,identity=score(s,wp,truth,S,ctx,eps);metrics.extend(mr);states.extend(sr)
                checks.append(dict(**ctx,mean_flux_source_relative=error,
                                   cached_reindex_relative=cached_covariance,**identity))
                np.savez_compressed(out/f'CACHE_{case}_{model["seed"]}_{t:.3f}.npz',
                    initial=w0.cpu().numpy(),predicted=wp.cpu().numpy(),reference=truth.cpu().numpy(),
                    sources=S.cpu().numpy(),mean_qd=vv(mean,'qd_').cpu().numpy(),mean_qb=vv(mean,'qb_').cpu().numpy())
                csvout(out/'DIRECTIONAL.csv',metrics);csvout(out/'STATE_DIAGNOSTICS.csv',states)
                csvout(out/'CHECKS.csv',checks);csvout(out/'PHYSICAL.csv',physics)
                dump(out/'STATUS.json',dict(status='RUNNING',last=ctx,counts=COUNTS))
                print(f'IC{case%1000}/{model["seed"]} t={t}: CNN {COUNTS["CNN"]}/96',flush=True)
        require(before==study.common.model_hash(s),'Frozen parameters changed');handle.remove()
    require(all(COUNTS[k]==v for k,v in cap.items()),'Hard budget completion mismatch')
    dump(out/'DONE.json',dict(status='VERIFIED_EXECUTION',counts=COUNTS,input_sha256=files,
        reference_quality=protocol['reference_quality'],wall_seconds=time.perf_counter()-started,
        corrected_autonomous_rollout='NOT_RUN',training_steps=0,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--run',default='run01');a=p.parse_args()
    if not a.execute:print('NOT_RUN: --execute required; 96 CNN, 48 native RHS cap');return
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=HERE/'results'/a.run
    try:execute(out)
    except BaseException as error:
        dump(out/'FAILURE.json',dict(status='FAIL',error=str(error),traceback=traceback.format_exc(),
                                   counts=COUNTS,remaining='NOT_RUN',reference_quality='REFERENCE_NOT_CONVERGED'))
        raise


if __name__=='__main__':main()
