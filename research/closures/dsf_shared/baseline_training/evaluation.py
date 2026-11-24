"""Only newly trained models define this benchmark; analytic methods fixed."""
from types import MethodType
import time
import numpy as np
import torch
import common as c
import data
import metrics

METHODS=(*c.MODES,'Clark4','P1','P1-C4','NoSGS')


def registry(a,out):
    r=c.read(out/'BASELINE_REGISTRY.json')
    if r['code_sha256']!=a['code_sha256'] or len(r['models'])!=8:raise ValueError('Complete matched 8-run registry required')
    for row in r['models']:
        if c.sha(row['path'])!=row['sha256']:raise ValueError('Selected trained checkpoint changed')
    return r


def model(a,out,mode,seed,device='cpu',cached=False):
    r=registry(a,out);record=next(x for x in r['models'] if (x['mode'],x['seed'])==(mode,seed))
    payload=torch.load(record['path'],map_location=device,weights_only=False)
    if payload['identity']['mode']!=mode or payload['identity']['code_sha256']!=a['code_sha256']:
        raise ValueError('Checkpoint prior identity mismatch')
    s=c.make_solver(mode,seed,device);s.load_state_dict(payload['model']);s.closure.set_backscatter_factor(1.)
    s.eval();s.requires_grad_(False)
    if cached:data.enable_cached_prior(s)
    return s


def weights(s):
    r=torch.sqrt(s.k_squared)
    z=(r-8)/8
    return torch.where(r<=8,torch.ones_like(r),torch.where(r>=16,torch.zeros_like(r),.5*(1+torch.cos(torch.pi*z))))


def runtime(a,out,method,seed,device):
    if method not in METHODS:raise ValueError(method)
    s=(c.make_solver('FULL-C2',2026,device) if method in ('Clark4','NoSGS') else
       model(a,out,method if method in c.MODES else 'FULL-C2',seed,device))
    s.calls=0;s.rhs_calls=0;s.rhs_seconds=0.;s.stages=[]
    handle=s.closure.output.register_forward_hook(lambda module,args,value:setattr(s,'calls',s.calls+1))
    s.capture_handle=handle
    if method in ('Clark4','NoSGS'):
        def closure(self,st):
            q=c.prior.c4.flux4(self,st,self.config.data.filter_width_m) if method=='Clark4' else (st['omega']*0,st['omega']*0)
            return dict(qx=q[0],qy=q[1])
        s.closure_fields=MethodType(closure,s)
    native=s.rhs
    def rhs(self,w,t):
        start=time.perf_counter();st=native(w,t);applied=-st['div_q']
        if method in ('P1','P1-C4'):
            q=c.prior.c4.flux2(st,self.config.data.filter_width_m) if method=='P1' else c.prior.c4.flux4(self,st,self.config.data.filter_width_m)
            analytic=c.prior.c4.source(self,q)
            applied=self.project(applied+c.prior.c4.multiply(self,analytic-applied,weights(self)))
            st['rhs']=self.project(st['rhs']+(applied+st['div_q']))
        st['source_applied']=applied;self.stages.append(st);self.rhs_calls+=1
        if device.startswith('cuda'):torch.cuda.synchronize()
        self.rhs_seconds+=time.perf_counter()-start
        return st
    s.rhs=MethodType(rhs,s);return s


def apriori(a,out,device='cpu',resume=False):
    c.verify(a);r=registry(a,out);dest=out/'apriori'
    if (dest/'DONE.json').exists():
        if not resume:raise FileExistsError(dest)
        rec=c.read(dest/'DONE.json')
        if rec['registry_sha256']!=c.sha(out/'BASELINE_REGISTRY.json'):raise ValueError('A-priori registry changed')
        for name,h in rec['outputs'].items():
            if c.sha(dest/name)!=h:raise ValueError('A-priori result changed')
        return
    if dest.exists():raise FileExistsError('Incomplete a-priori outputs retained; use new run or inspect failure')
    dest.mkdir();rows=[];spectra=[];internal=[];calls=0;started=time.perf_counter()
    try:
        for initial in c.CASES:
            cache=data.load_cache(a,out,initial,device)
            for seed in a['protocol']['training_seeds']:
                solvers={m:model(a,out,m,seed,device,True) for m in c.MODES}
                with torch.inference_mode():
                    for i,t in enumerate(cache['time_s']):
                        sources={};parts={}
                        for m,s in solvers.items():
                            st,f,source=data.forward_frame(s,cache,i);sources[m]=source;calls+=1
                            v=c.prior.violations(s,{**st,**f})
                            sd=c.prior.c4.source(s,(f['qd_x'],f['qd_y']));sb=c.prior.c4.source(s,(f['qb_x'],f['qb_y']))
                            internal.append(dict(initial=initial,seed=seed,method=m,time_s=float(t),
                                limiter_fraction=float((f['backscatter_transfer_limiter']<1-1e-6).float().mean()),
                                **v,Sd_Z=float((-f['pi_d']).mean()),Sb_Z=float((-f['pi_b']).mean()),
                                Sd_E=float((st['psi']*sd).mean()),Sb_E=float((st['psi']*sb).mean())))
                        s=solvers['FULL-C2'];q2=c.prior.c4.flux2(st,s.config.data.filter_width_m)
                        q4=(q2[0]+cache['fourth_x'][i],q2[1]+cache['fourth_y'][i]);sc4=c.prior.c4.source(s,q4)
                        sf=sources['FULL-C2'];sc2=c.prior.c4.source(s,q2)
                        sources.update(Clark4=sc4,P1=s.project(sf+c.prior.c4.multiply(s,sc2-sf,weights(s))),
                            **{'P1-C4':s.project(sf+c.prior.c4.multiply(s,sc4-sf,weights(s))),'NoSGS':sf*0})
                        ctx=dict(initial=initial,training_seed=seed,time_s=float(t),
                            role=a['protocol']['evaluation_labels'][str(initial)],quality='REFERENCE_NOT_CONVERGED')
                        rr,ss=metrics.source_rows(cache['omega'][i].cpu().numpy(),cache['source'][i].cpu().numpy(),
                            {k:v.cpu().numpy() for k,v in sources.items()},ctx)
                        rows+=rr;spectra+=ss
                del solvers
        for name,rr in [('SOURCE.csv',rows),('SOURCE_SPECTRA.csv',spectra),('CONSTITUENTS.csv',internal)]:c.csvout(dest/name,rr)
        c.dump(dest/'DONE.json',dict(status='COMPLETED',code_sha256=a['code_sha256'],registry_sha256=c.sha(out/'BASELINE_REGISTRY.json'),
            CNN_forwards=calls,wall_seconds=time.perf_counter()-started,device=device,
            outputs={p.name:c.sha(p) for p in dest.glob('*.csv')}))
    except BaseException as e:c.dump(dest/'FAILURE.json',dict(type=type(e).__name__,message=str(e)));raise
