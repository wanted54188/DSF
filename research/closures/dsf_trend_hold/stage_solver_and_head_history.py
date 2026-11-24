"""Native RK4 adapter, immutable within-step history and strict physical guards."""
import time
import uuid
import numpy as np
import torch
import pc_common as c


class TrajectoryFailure(RuntimeError):
    def __init__(self,status,reason,arrays=None,details=None):
        super().__init__(reason); self.status=status; self.arrays=arrays or {}; self.details=details or {}


def D(s,v): return -s.project(s.derivative(v[0],'x')+s.derivative(v[1],'y'))
def vec(f,tag): return torch.stack((f[tag+'_x'],f[tag+'_y']))
def cpu(v): return v.detach().cpu().numpy()
def rms(v): return float(v.square().mean().sqrt())
def relative(a,b): return rms(a-b)/max(rms(b),1e-300)


class Bank:
    """Promote only COMPLETE fresh pairs; pending first sample is never active."""
    def __init__(self,active,pending=None): self.active=active; self.pending=pending
    def observe(self,n,anchor):
        c.require(n%5 in (0,1),'Unscheduled history observation')
        if n%5 == 0:
            self.pending=anchor
        else:
            c.require(self.pending is not None and self.pending['step']==n-1,'Missing pending first anchor')
            self.active=[self.pending,anchor]; self.pending=None
    def descriptor(self):
        strip=lambda a:{k:v for k,v in a.items() if k!='heads'}
        return dict(active=[strip(a) for a in self.active],pending=strip(self.pending) if self.pending else None)
    def predict(self,n,fraction,method):
        first,last=self.active
        c.require(last['step']-first['step']==1,'Anchors must be consecutive true refresh states')
        tau=n+fraction-last['step']
        c.require(tau>=0,'Future history leak')
        if method=='Parameter-Hold': return last['heads'],float(tau)
        return (1+tau)*last['heads']-tau*first['heads'],float(tau)


def load_anchor(root,desc,device):
    path=root/desc['file']; c.require(c.sha(path)==desc['sha256'],'Anchor SHA changed')
    z=c.load(path); m=c.meta(z)
    c.require(m['step']==desc['step'],'Anchor timestamp changed')
    c.require(z['heads'].shape==(4,7,100,100) and z['heads'].dtype==np.float64,'Raw head anchor shape/dtype')
    return dict(**desc,heads=torch.as_tensor(z['heads'],device=device))


def load_bank(root,description,device):
    return Bank([load_anchor(root,a,device) for a in description['active']],
                load_anchor(root,description['pending'],device) if description['pending'] else None)


def save_anchor(root,folder,n,heads):
    path=folder/f'{n:04d}_{uuid.uuid4().hex}.npz'
    c.npz(path,heads=cpu(heads),metadata=c.packed(dict(step=n,time_s=1+n*c.H)))
    return dict(file=str(path.relative_to(root)),sha256=c.sha(path),step=n,heads=heads.clone())


def verify_physical(s,st,d,b,original):
    if not all(bool(torch.isfinite(v).all()) for v in (st['omega'],d,b)):
        raise TrajectoryFailure('FAILED_NUMERICAL','NaN/Inf at stage',dict(omega=cpu(st['omega']),qd=cpu(d),qb=cpu(b)))
    check,witness=c.feasible.envelope(s,st,d,b,original)
    if check['qd_envelope_pass']:
        A=witness['A']; coeff=witness['c']; half=.5*(A[0]+A[2]); rad=torch.hypot(.5*(A[0]-A[2]),A[1])
        check.update(witness_eigenvalue_min=float((half-rad).min()),
            witness_condition_max=float(((half+rad)/(half-rad)).max()),
            witness_trace_error=float((A[0]+A[2]-2).abs().max()),
            witness_coefficient_max=float(coeff.max()))
        c.require(check['witness_eigenvalue_min']>0 and check['witness_condition_max']<=16*(1+1e-12)
                  and check['witness_trace_error']<=1e-12,'Certificate implementation failed')
    if not check['passed']:
        raise TrajectoryFailure('FAILED_PHYSICAL','Original closed physical envelope violated',
            dict(omega=cpu(st['omega']),gradient=cpu(torch.stack((st['omega_x'],st['omega_y']))),
                 qd=cpu(d),qb=cpu(b),current_C4_support=cpu(original['gradient_support_norm']),
                 qd_bad_mask=cpu(witness['bad']),witness_c=cpu(witness['c']),witness_A=cpu(witness['A'])),check)
    return check,witness


def basic_metrics(s,w):
    hw=torch.fft.fft2(w,norm='forward'); power=hw.abs().square(); masks=c.q.masks(s)
    K=.5*float((power*s.inverse_k_squared).sum()); Z=.5*float(power.sum())
    outside=float(power[~masks['all']].sum())
    c.require(outside<=1e-24*max(float(power.sum()),1e-300),'State left real zero-mean rectangular P33')
    c.require(abs(Z-.5*float(w.square().mean()))<=1e-12*max(Z,1.),'State Parseval identity')
    return dict(K=K,Z=Z,high_Z=.5*float(power[masks['high']].sum()),corner_Z=.5*float(power[masks['corner']].sum()),
                outside_power=outside,mean=float(w.mean()),max_abs_omega=float(w.abs().max()))


class Engine:
    def __init__(self,s,method,journal,root,folder,initial_invariants,diagnostics=True):
        self.s=s; self.method=method; self.j=journal; self.root=root; self.folder=folder
        self.native_closure=s.closure_fields; self.native_rhs=s.rhs
        s.closure_fields=self.flux; s.rhs=self.rhs
        self.handle=s.closure.register_forward_pre_hook(lambda *_:self.j.tick('CNN'))
        self.captured_raw=None
        self.raw_handle=s.closure.output.register_forward_hook(lambda _m,_i,value:setattr(self,'captured_raw',value[0].detach().clone()))
        self.decoder_verified=False;self.decoder_calls=0;self.fractions=(0.,.5,.5,1.)
        self.initial=initial_invariants; self.n=0; self.stage=0; self.bank=None
        self.cache=None; self.fresh=None; self.records=[]; self.values=[]; self.snapshots=[]
        self.diagnostics=diagnostics; self.last_original=None; self.last_check=None
        self.diagnostic_seconds=0.; self.network_seconds=0.; self.closed=False
        self.bootstrap_original=None
    def original(self,st):
        c.sync(st['omega'].device); begin=time.perf_counter()
        self.captured_raw=None
        out=self.native_closure(st)
        if self.captured_raw is None and hasattr(self.s,'_synthetic_head_provider'):
            self.captured_raw=self.s._synthetic_head_provider(st).clone()
        c.require(self.captured_raw is not None,'Original raw head capture missing')
        c.sync(st['omega'].device); self.network_seconds+=time.perf_counter()-begin
        if not bool(torch.isfinite(torch.stack([out[k] for k in ('qd_x','qd_y','qb_x','qb_y','A11','A12','A22','c_theta')])).all()):
            raise TrajectoryFailure('FAILED_NUMERICAL','Nonfinite original network decoder',
                dict(omega=cpu(st['omega']),qd=cpu(vec(out,'qd')),qb=cpu(vec(out,'qb'))))
        if not self.decoder_verified:
            replay=self.decode(st,self.captured_raw)
            error=max(relative(out[k],replay[k]) for k in ('qd_x','qd_y','qb_x','qb_y','c_theta','A11','A12','A22','backscatter_transfer_limiter'))
            c.require(error<=1e-12,'Original decoder replay mismatch');self.decoder_verified=True
        return out
    def decode(self,st,heads):
        self.decoder_calls+=1;self.j.note('ALGEBRAIC_DECODE')
        if not bool(torch.isfinite(heads).all()):
            raise TrajectoryFailure('FAILED_NUMERICAL','Nonfinite extrapolated logits',dict(omega=cpu(st['omega']),raw=cpu(heads)))
        return c.decoder.decode(self.s,st,heads)
    def close(self):
        if not self.closed:
            self.handle.remove();self.raw_handle.remove(); self.s.closure_fields=self.native_closure; self.s.rhs=self.native_rhs; self.closed=True
    def orbit(self,w):
        s=self.s; branches=[]; sources=[];heads=[]
        for turn in range(4):
            st=s.state(c.q.spatial(w,turn)); f=self.original(st)
            heads.append(self.captured_raw.clone())
            try: c.q.physical(s,st,f)
            except RuntimeError as exc:
                details,_=c.feasible.envelope(s,st,vec(f,'qd'),vec(f,'qb'),f)
                raise TrajectoryFailure('FAILED_PHYSICAL','Original rotated member failed '+str(exc),
                    dict(omega=cpu(w),rotated_omega=cpu(st['omega']),qd=cpu(vec(f,'qd')),qb=cpu(vec(f,'qb'))),
                    dict(orientation=90*turn,**details)) from exc
            branches.append(f); sources.append(c.q.spatial(D(s,vec(f,'qd')+vec(f,'qb')),-turn))
        mean=c.q.mean_fields(s,s.state(w),branches)
        d,b=vec(mean,'qd'),vec(mean,'qb'); src=D(s,d+b)
        c.require(relative(src,torch.stack(sources).mean(0))<=1e-12,'Group mean Source/flux divergence')
        # Group-mean envelope is checked when applied as Fresh-C4. A diagnostic
        # fresh target for History is not silently projected or used to replace PC.
        return dict(original=branches[0],branches=branches,heads=torch.stack(heads),d=d,b=b,delta_d=d-vec(branches[0],'qd'),delta_b=b-vec(branches[0],'qb'),
                    limiter_fraction=float(torch.stack([(f['backscatter_transfer_limiter']<1).double().mean() for f in branches]).mean()))
    def prepare(self,n,w,bank):
        self.n=n; self.stage=0; self.bank=bank; self.records=[]; self.values=[]; self.snapshots=[]
        self.cache=None; self.fresh=None
        if self.method.startswith('Parameter') and n%5 in (0,1):
            if n==1 and bank.active[-1]['step']==1:
                # Startup supplies the two true anchors, but reevaluate current
                # original Aug once. Startup raw network tensors are not shared.
                return
            self.j.phase='refresh'; fresh=self.orbit(w)
            anchor=save_anchor(self.root,self.folder/'anchors',n,fresh['heads'])
            bank.observe(n,anchor); self.cache=fresh['original']; self.fresh=fresh
        self.frozen_descriptor=bank.descriptor() if bank else None
    def flux(self,st):
        s=self.s; fraction=self.fractions[self.stage] if self.stage<4 else 0.
        tau=None; self.current_fresh=None
        if self.method=='Aug':
            f=self.original(st);members=[f];d,b=vec(f,'qd'),vec(f,'qb')
            c.q.physical(s,st,f)
            self.predicted_heads=self.captured_raw[None]
        elif self.method=='Fresh-C4':
            fresh=self.orbit(st['omega']);members=fresh['branches'];f=members[0]
            d,b=fresh['d'],fresh['b'];self.current_fresh=fresh;self.predicted_heads=fresh['heads']
        else:
            heads,tau=self.bank.predict(self.n,fraction,self.method)
            self.predicted_heads=heads;members=[]
            for turn in range(4):
                ss=s.state(c.q.spatial(st['omega'],turn))
                member=self.decode(ss,heads[turn])
                if not all(bool(torch.isfinite(member[k]).all()) for k in ('qd_x','qd_y','qb_x','qb_y','c_theta','A11','A12','A22')):
                    raise TrajectoryFailure('FAILED_NUMERICAL','Nonfinite original parameter decoding',dict(omega=cpu(st['omega']),raw=cpu(heads)))
                try:c.q.physical(s,ss,member)
                except RuntimeError as exc:
                    details,_=c.feasible.envelope(s,ss,vec(member,'qd'),vec(member,'qb'),member)
                    raise TrajectoryFailure('FAILED_PHYSICAL','Original parameter member outside envelope',
                        dict(omega=cpu(st['omega']),rotated_omega=cpu(ss['omega']),raw=cpu(heads),
                             qd=cpu(vec(member,'qd')),qb=cpu(vec(member,'qb'))),dict(orientation=90*turn,**details)) from exc
                members.append(member)
            mean=c.q.mean_fields(s,st,members);d,b=vec(mean,'qd'),vec(mean,'qb');f=members[0]
            c.q.physical(s,st,dict(mean,qb_x=torch.zeros_like(st['omega']),qb_y=torch.zeros_like(st['omega'])))
            self.current_fresh=self.fresh if self.stage==0 else None
        self.last_original=self.current_fresh['original'] if self.current_fresh is not None else f
        check,cert=verify_physical(s,st,d,b,f)
        src=D(s,d+b);sd=D(s,d);sb=D(s,b)
        expected=torch.stack([c.q.spatial(D(s,vec(v,'qd')+vec(v,'qb')),-j) for j,v in enumerate(members)]).mean(0)
        source_identity=relative(src,expected)
        c.require(source_identity<=1e-12,'Group decoder flux/Source identity')
        g=torch.stack((st['omega_x'],st['omega_y']));pd=-(d*g).sum(0);pb=-(b*g).sum(0);pi=pd+pb
        ibp=float((st['omega']*src).mean()+pi.mean())
        c.require(abs(ibp)<=1e-12*max(rms(st['omega'])*rms(src),rms(pi),1e-30),'SGS integration by parts')
        limiter=float(torch.stack([(v['backscatter_transfer_limiter']<1).double().mean() for v in members]).mean())
        check.update(tau=tau,Source_identity_relative=source_identity,SGS_Z_ibp_residual=ibp,
            qd_RMS=rms(d),qb_RMS=rms(b),pi_d_mean=float(pd.mean()),pi_b_mean=float(pb.mean()),
            pi_RMS=rms(pi),backscatter_fraction=float((pi<0).double().mean()),original_limiter_fraction=limiter,
            raw_abs_max=float(self.predicted_heads.abs().max()),algebraic_decoder_calls=self.decoder_calls)
        self.last_check=check;self.last_cert=cert
        return dict(qx=d[0]+b[0],qy=d[1]+b[1],qd_x=d[0],qd_y=d[1],qb_x=b[0],qb_y=b[1],
                    pi_d=pd,pi_b=pb,pi=pi,Source=src,Sd=sd,Sb=sb,gradient_support_norm=f['gradient_support_norm'])
    def rhs(self,w,t):
        self.j.phase=f'step{self.n}_stage{self.stage+1}'; self.j.tick('RHS')
        fraction=self.fractions[self.stage] if self.stage<4 else 0.
        c.require(abs(t-(1+c.H*(self.n+fraction)))<1e-10,'Native RK4 stage time mismatch')
        before=time.perf_counter(); fields=self.native_rhs(w,t)
        if not bool(torch.isfinite(fields['rhs']).all()):
            raise TrajectoryFailure('FAILED_NUMERICAL','NaN/Inf RHS',dict(omega=cpu(w),rhs=cpu(fields['rhs'])))
        stats=basic_metrics(self.s,w)
        if stats['K']>100*self.initial['K'] or stats['Z']>100*self.initial['Z']:
            raise TrajectoryFailure('FAILED_NUMERICAL','Inherited 100x total K/Z safety bound',dict(omega=cpu(w)),stats)
        adv=-self.s.project(fields['u']*fields['omega_x']+fields['v']*fields['omega_y'])
        visc=self.s.project(self.s.config.physics.kinematic_viscosity_m2_s*fields['laplacian_omega'])
        force=self.s.project(self.s.vorticity_forcing(t,w))
        reconstruction=relative(fields['rhs'],adv+visc+force+fields['Source'])
        c.require(reconstruction<=1e-12,'Full RHS reconstruction')
        fft=lambda v:torch.fft.fft2(v,norm='forward')
        hw=fft(w); inv=self.s.inverse_k_squared; masks=c.q.masks(self.s)
        budget={}
        for name,v in dict(advection=adv,viscosity=visc,forcing=force,Sd=fields['Sd'],Sb=fields['Sb'],SGS=fields['Source'],RHS=fields['rhs']).items():
            z=(hw.conj()*fft(v)).real
            for band,mask in masks.items():
                budget[band+'__Z_'+name]=float(z[mask].sum()); budget[band+'__K_'+name]=float((z*inv)[mask].sum())
        scale=max(stats['Z'],rms(fields['rhs']),1.)
        c.require(max(abs(budget['all__Z_advection']),abs(budget['all__K_advection']))<=5e-5*scale,'Native advection leakage')
        for name in ('RHS','SGS'):
            for quantity in ('K','Z'):
                full=budget['all__'+quantity+'_'+name]
                parts=sum(budget[b+'__'+quantity+'_'+name] for b in ('low','mid','high'))
                c.require(abs(full-parts)<=1e-12*max(abs(full),scale),'FFT band budget partition')
        record=dict(step=self.n,stage=self.stage+1,time_s=float(t),**stats,**self.last_check,
                    rhs_reconstruction_relative=reconstruction,budget=budget)
        self.records.append(record); self.values.append(fields['rhs'].clone())
        if self.stage==0 and self.n%10==0:
            self.snapshots.append(self.snapshot(fields))
        self.stage+=1
        self.diagnostic_seconds+=time.perf_counter()-before
        return fields
    def snapshot(self,f):
        out={k:cpu(f[k]) for k in ('omega','psi','u','v','omega_x','omega_y','rhs','Source','Sd','Sb','qx','qy','qd_x','qd_y','qb_x','qb_y','gradient_support_norm')}
        original=self.last_original
        out.update(original_qd=cpu(vec(original,'qd')),original_qb=cpu(vec(original,'qb')),
            witness_c=cpu(self.last_cert['c']),witness_A=cpu(self.last_cert['A']))
        if self.current_fresh is not None:
            out.update(fresh_delta_d=cpu(self.current_fresh['delta_d']),fresh_delta_b=cpu(self.current_fresh['delta_b']))
            out['fresh_heads']=cpu(self.current_fresh['heads'])
        out['predicted_heads']=cpu(self.predicted_heads)
        return out
    def advance(self,n,w,bank):
        c.sync(w.device); before=time.perf_counter(); d0=self.diagnostic_seconds; nn0=self.network_seconds
        self.prepare(n,w,bank)
        # No predictor mutation is permitted inside the original RK4 step.
        frozen=bank.descriptor() if bank else None
        out=self.s.step(w,1+n*c.H,c.H)
        c.sync(w.device); elapsed=time.perf_counter()-before
        c.require(self.stage==4 and (bank is None or bank.descriptor()==frozen),'History mutated within RK4')
        if not bool(torch.isfinite(out).all()): raise TrajectoryFailure('FAILED_NUMERICAL','Nonfinite endpoint',dict(omega=cpu(out)))
        endpoint_stats=basic_metrics(self.s,out)
        if endpoint_stats['K']>100*self.initial['K'] or endpoint_stats['Z']>100*self.initial['Z']:
            raise TrajectoryFailure('FAILED_NUMERICAL','Inherited 100x K/Z endpoint safety bound',dict(omega=cpu(out)),endpoint_stats)
        return out,dict(solver_wall_seconds=elapsed,rhs_wall_seconds_including_checks=self.diagnostic_seconds-d0,
                        CNN_wall_seconds=self.network_seconds-nn0)
    def endpoint(self,n,w,bank):
        self.prepare(n,w,bank); f=self.rhs(w,1+n*c.H)
        # prepare(n=800) already performs the scheduled History refresh; that
        # first pending anchor does not replace the completed pair.
        return self.snapshot(f)
