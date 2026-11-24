"""Manual selected-frame preprocessing; no high-grid arrays or DNS generation."""
import time
import numpy as np
import torch
import common as c

STATE=('omega','psi','omega_x','omega_y','strain','laplacian_omega','u_x','u_y','v_x','v_y')


def prepare(a,out,resume=False):
    c.verify(a);p=a['protocol'];torch.set_num_threads(p['cpu_threads'])
    for seed in c.CASES:
        dest=out/'data'/str(seed)
        if (dest/'DONE.json').exists():
            if not resume:raise FileExistsError(dest)
            load_cache(a,out,seed,'cpu');continue
        if dest.exists():raise FileExistsError('Incomplete data cache retained; choose new run')
        dest.mkdir(parents=True);tic=time.perf_counter()
        row=next(r for r in a['references'] if r['seed']==seed and r['variant']==p['primary_variant'])
        arrays={};verified={}
        for name,rec in row['fields'].items():
            digest=c.sha(rec['path'])
            if digest!=rec['expected_sha256']:raise ValueError('Reference array SHA mismatch: '+name)
            verified[rec['path']]=digest
            arrays[name]=np.load(rec['path'],mmap_mode='r',allow_pickle=False)
        t=np.load(row['time_path'],allow_pickle=False)
        selected=np.arange(0,3201,round(p['frame_interval']/.00125),dtype=int)
        if len(selected)!=81 or not np.allclose(t[selected],1+np.arange(81)*.05,atol=1e-12,rtol=0):raise ValueError('Frame schedule mismatch')
        s=c.make_solver('FULL-C2',2026);bundle={k:[] for k in (*STATE,'fourth_x','fourth_y','source','qx','qy','pi')}
        checks=[]
        with torch.no_grad():
            for i in selected:
                original=torch.tensor(np.array(arrays['reference_omega'][i]),dtype=torch.float32)
                w=s.project(original);st=s.state(w)
                target=torch.tensor(np.array(arrays['reference_source'][i]),dtype=torch.float32)
                qx=torch.tensor(np.array(arrays['reference_qx'][i]),dtype=torch.float32)
                qy=torch.tensor(np.array(arrays['reference_qy'][i]),dtype=torch.float32)
                dot=torch.tensor(np.array(arrays['reference_dot'][i]),dtype=torch.float32)
                flux_source=c.prior.c4.source(s,(qx,qy))
                resolved=s.project(-(st['u']*st['omega_x']+st['v']*st['omega_y'])+1e-6*st['laplacian_omega'])
                norm=lambda x:float(x.square().mean().sqrt())
                mismatch=max(norm(flux_source-target)/max(norm(target),1e-12),norm(resolved+target-dot)/max(norm(dot),1e-12))
                if mismatch>p['reference_adapter_relative_tolerance']:raise ValueError(f'Reference adapter mismatch t={t[i]} value={mismatch}')
                inc=c.prior.c4.fourth_increment(s,st,s.config.data.filter_width_m)
                for k in STATE:bundle[k].append(st[k].cpu())
                for k,v in [('fourth_x',inc[0]),('fourth_y',inc[1]),('source',target),('qx',qx),('qy',qy),
                            ('pi',-(qx*st['omega_x']+qy*st['omega_y']))]:bundle[k].append(v.cpu())
                checks.append(dict(seed=seed,time_s=float(t[i]),operator_relative_mismatch=mismatch,quality='REFERENCE_NOT_CONVERGED'))
        packed={k:torch.stack(v) for k,v in bundle.items()}
        packed['time_s']=torch.tensor(t[selected],dtype=torch.float64)
        if not all(torch.isfinite(v).all() for v in packed.values()):raise FloatingPointError('Nonfinite prepared labels')
        c.save_pt(dest/'CACHE.pt',packed);c.csvout(dest/'OPERATOR_CHECKS.csv',checks)
        c.dump(dest/'DONE.json',dict(seed=seed,role=row['role'],frames=81,time_range=[1.,5.],
            code_sha256=a['code_sha256'],cache_sha256=c.sha(dest/'CACHE.pt'),verified_arrays=verified,
            quality='REFERENCE_NOT_CONVERGED',wall_seconds=time.perf_counter()-tic,CNN_forwards=0))
        print(f'PREPARED {seed}: 81 frames, no CNN, {time.perf_counter()-tic:.1f}s',flush=True)


def load_cache(a,out,seed,device):
    if seed not in c.CASES:raise PermissionError('Only authorized 101/102')
    d=out/'data'/str(seed);done=c.read(d/'DONE.json')
    if done['seed']!=seed or done['code_sha256']!=a['code_sha256'] or c.sha(d/'CACHE.pt')!=done['cache_sha256']:
        raise ValueError('Cache identity mismatch')
    x=torch.load(d/'CACHE.pt',map_location='cpu',weights_only=False)
    if x['omega'].shape!=(81,100,100) or len(x['time_s'])!=81:raise ValueError('Cache schema mismatch')
    return {k:v.to(device) for k,v in x.items()}


def enable_cached_prior(s):
    """Reference-state training only; reuse the SAME checked C4 increment.

    Dynamic LES uses the original installed prior and recomputes every stage.
    """
    from types import MethodType
    cl=s.closure
    def vectors(self,x,y,*derivatives):
        f,p=c.prior.MODES[self._integration_mode]
        q2=(x,y);q4=(x+self._reference_fourth[0],y+self._reference_fourth[1])
        return (*(q2 if f=='c2' else q4),*(q2 if p=='c2' else q4))
    cl._prior_vectors=MethodType(vectors,cl)


def forward_frame(s,cache,i):
    s.closure._reference_fourth=(cache['fourth_x'][i],cache['fourth_y'][i])
    st={k:cache[k][i] for k in STATE}
    fields=s.closure_fields(st)
    return st,fields,c.prior.c4.source(s,(fields['qx'],fields['qy']))
