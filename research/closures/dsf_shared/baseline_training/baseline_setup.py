"""Versioned new NS benchmark; original research artifacts are read-only."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
PRIOR=ROOT/'research/full_c4_prior_integration_v1_20261103'
sys.path.insert(0,str(PRIOR))
import fc4_model as prior
sys.path.insert(0,str(HERE))
CASES=(110240101,110240102)
MODES=tuple(prior.MODES)
CONFIG=ROOT/'results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json'


def safe(path,write=False):
    p=Path(path).resolve()
    if not p.is_relative_to(HERE if write else ROOT):raise PermissionError('Path outside permitted study/project')
    if any(str(x) in str(p) for x in (110240201,110240202,110240301,110240302)):
        raise PermissionError('Locked/reserved access forbidden')
    return p


def read(p):return json.loads(safe(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with safe(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def dump(p,x):
    p=safe(p,True);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8');os.replace(tmp,p)


def csvout(p,rows):
    rows=list(rows);p=safe(p,True);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def save_pt(p,x):
    import torch
    p=safe(p,True);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    torch.save(x,tmp);os.replace(tmp,p)


def protocol():return read(HERE/'PROTOCOL.json')
def code():
    paths=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))+[HERE/'PROTOCOL.json',PRIOR/'fc4_model.py',prior.C4_PATH,ROOT/'spnsgs/solver.py',ROOT/'spnsgs/config.py',ROOT/'spnsgs/paper.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def identity():
    p=protocol();rows=[];root=ROOT/p['reference_root']
    manifest=root/'COMPLETE_LONG_REFERENCE_MANIFEST.json';m=read(manifest)
    files={str(manifest):sha(manifest),str(CONFIG):sha(CONFIG)}
    for seed in CASES:
        q=root/'quality'/str(seed)/'QUALITY.json';files[str(q)]=sha(q)
        qv=read(q)
        entry=next(x for x in m['cases'] if x['seed']==seed)
        production=safe(entry['production'])
        if production!=safe(root/'reference'/str(seed)/p['primary_variant']):raise ValueError('Manifest production mismatch')
        if sha(production/'DONE.json')!=entry['production_done_sha256']:raise ValueError('Primary manifest/DONE SHA mismatch')
        for variant in (p['primary_variant'],*p['comparison_variants']):
            d=safe(root/'reference'/str(seed)/variant);done=read(d/'DONE.json')
            if done['seed']!=seed or done['role']!='development' or done['frames']!=3201:raise ValueError('Reference identity/schema mismatch')
            cfg=done['config']
            if cfg['nu']!=1e-6 or cfg['les_n']!=100 or not np.isclose(cfg['delta'],.25132741228718347):raise ValueError('Reference physics mismatch')
            files[str(d/'DONE.json')]=sha(d/'DONE.json')
            t=np.load(d/'time_s.npy',allow_pickle=False)
            if not np.allclose(t,1+np.arange(3201)*.00125,rtol=0,atol=1e-12):raise ValueError('Expected actual reference times 1..5, not 0..5')
            if sha(d/'time_s.npy')!=done['outputs']['time_s.npy']:raise ValueError('Time SHA mismatch')
            files[str(d/'time_s.npy')]=done['outputs']['time_s.npy']
            fields={}
            for name in ('reference_omega','reference_source','reference_dot','reference_qx','reference_qy'):
                f=d/(name+'.npy')
                with f.open('rb') as stream:
                    version=np.lib.format.read_magic(stream);shape,order,dtype=np.lib.format._read_array_header(stream,version)
                if shape!=(3201,100,100) or order or dtype!=np.dtype('float64'):raise ValueError('Invalid field header '+name)
                fields[name]=dict(path=str(f),expected_sha256=done['outputs'][f.name],shape=list(shape),dtype=str(dtype))
            rows.append(dict(seed=seed,role='train' if seed==p['train_seed'] else 'development_validation',variant=variant,
                directory=str(d),fields=fields,quality=qv['status'],gates=qv['gates'],times=[float(t[0]),float(t[-1])],
                time_path=str(d/'time_s.npy'),time_sha256=done['outputs']['time_s.npy']))
    return dict(version=p['version'],code_sha256=code(),protocol=p,files=files,references=rows,
        role_amendment='New user-authorized training benchmark; historical development roles preserved in source metadata',
        quality='REFERENCE_NOT_CONVERGED',arrays_decoded=False,models_loaded=False,training='NOT_RUN')


def verify(a):
    if a['code_sha256']!=code():raise ValueError('Code/protocol changed: new run required')
    for f,h in a['files'].items():
        if sha(f)!=h:raise ValueError('Input metadata changed: '+f)


def matrix(a):
    p=a['protocol']
    return [dict(mode=m,seed=s,train_initial=p['train_seed'],validation_initial=p['validation_seed'],
        epochs=p['epochs'],train_frames=81,batch_frames=p['batch_frames'],optimizer_updates=21*p['epochs'],
        initialization='FRESH_PAIRED_NOT_POF_CHECKPOINT',quality='REFERENCE_NOT_CONVERGED')
        for s in p['training_seeds'] for m in MODES]


def model_hash(solver):
    h=hashlib.sha256()
    for k,v in solver.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_solver(mode,seed,device='cpu',n=100):
    import torch
    from spnsgs.config import load_config
    from spnsgs.solver import SpectralVorticitySolver
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cfg=load_config(CONFIG)
    s=SpectralVorticitySolver(cfg,n,n,cfg.domain.lx/n,cfg.domain.ly/n,torch.device(device),torch.float32)
    prior.install(s,mode);s.closure.set_backscatter_factor(1.);return s
