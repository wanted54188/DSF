"""Read-only Aug inputs, exact mean-flux closure and bounded diagnostics."""
import ast
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PRIOR=ROOT/'research/full_aug_rotation_pc_pilot_v1_20261114'
spec=importlib.util.spec_from_file_location('previous_aug_rotation_direction',PRIOR/'experiment.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
study=previous.study
spatial,vector=previous.spatial,previous.vector
OLD=PRIOR/'results/run02'
COUNTS=dict(CNN=0,group_RHS=0,Aug_RHS=0,Clark4_RHS=0,group_steps=0,Aug_steps=0,Clark4_steps=0,
            analytic_same_state_Clark4=0,algebraic_closure_calls=0,source_evaluations=0)

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return previous.sha(p)
def dump(p,x):
    p=Path(p).resolve();require(p.is_relative_to(HERE),'Output outside new experiment')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(p,rows):
    rows=list(rows)
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
def rms(x):return previous.rms(x)
def rel(x,y):return previous.relative(x,y)
def source(s,q):
    COUNTS['source_evaluations']+=1
    return -s.project(s.derivative(q[0],'x')+s.derivative(q[1],'y'))
def qvec(f,prefix='q'):return previous.vv(f,prefix)
def geometry(s):return previous.masks(s)


def decoder():
    """Extract original Aug constitutive arithmetic; no CNN or Eq decoder."""
    tree=ast.parse(study.common.prior.TRANSFORMED_SOURCE);fn=tree.body[0]
    def index(name):
        return next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
    fn.body=copy.deepcopy(fn.body[:index('feature_fields')])+copy.deepcopy(fn.body[index('c_theta_unbounded'):])
    fn.args.args.append(ast.arg('raw'));fn.name='aug_constitutive_only'
    namespace=dict(study.common.prior.FORWARD.__globals__)
    exec(compile(ast.fix_missing_locations(tree),str(HERE/'original_aug_decoder_AST'),'exec'),namespace)
    return namespace[fn.name],ast.unparse(tree)


DECODER,DECODER_SOURCE=decoder()
def decode(s,st,raw):
    COUNTS['algebraic_closure_calls']+=1
    return DECODER(s.closure,*(st[k] for k in study.data.STATE),raw)


class Orbit:
    """Every call reconstructs all four current states; never modifies bounds."""
    def __init__(self,s,native,strict=True):self.s=s;self.native=native;self.strict=strict;self.rows=[];self.last=None
    def __call__(self,st):
        s=self.s;branches=[];rotated_sources=[];checks=[]
        for j in range(4):
            ss=st if j==0 else s.state(spatial(st['omega'],j))
            f=self.native(ss)
            checks.append(previous.physical(s,ss,f))
            branches.append(f)
            rotated_sources.append(spatial(source(s,qvec(f)),-j))
        mean=previous.mean_fields(s,st,branches)
        try:meancheck=previous.physical(s,st,mean)
        except RuntimeError as error:
            raise RuntimeError('FAIL_GATE: mean signed/physical envelope; no repair: '+str(error)) from error
        g=torch.stack((st['omega_x'],st['omega_y']))
        mean['pi_d']=-(qvec(mean,'qd_')*g).sum(0);mean['pi_b']=-(qvec(mean,'qb_')*g).sum(0)
        mean['pi']=mean['pi_d']+mean['pi_b']
        S=source(s,qvec(mean));members=torch.stack(rotated_sources)
        residual=rel(S,members.mean(0))
        require(residual<=1e-12,'Mean Source differs from mean flux divergence')
        self.last=dict(mean=meancheck,branches=checks,source_mean_relative=residual,
                       S_aug=members[0],S_sym=S,rotation_variance=float((members-members.mean(0)).square().mean()),
                       limiter_active_fraction=float(torch.stack([(f['backscatter_transfer_limiter']<1).double().mean()
                                                                 for f in branches if 'backscatter_transfer_limiter' in f]).mean())
                       if 'backscatter_transfer_limiter' in branches[0] else None)
        self.rows.append({k:v for k,v in self.last.items() if k not in ('S_aug','S_sym')})
        return mean


def frozen_inputs():
    files={};freeze_path=ROOT/'research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json'
    freeze=read(freeze_path);files[str(freeze_path.relative_to(ROOT))]=sha(freeze_path)
    require(freeze['quality']=='REFERENCE_NOT_CONVERGED','Reference flag changed')
    for section in ('training_code_sha256','baseline_code_sha256'):
        for name,digest in freeze[section].items():
            require(sha(ROOT/name)==digest,'Historical code changed: '+name);files[name]=digest
    require(sha(study.common.CONFIG)==freeze['baseline_config_sha256'],'Original config changed')
    files[str(study.common.CONFIG.relative_to(ROOT))]=sha(study.common.CONFIG)
    models=[]
    for row in freeze['checkpoints']:
        if row['mode']!=study.AUGMENTED:continue
        for key,expected in (('file','sha256'),('run','run_sha256'),('done','done_sha256')):
            require(sha(ROOT/row[key])==row[expected],'Frozen identity mismatch');files[row[key]]=row[expected]
        models.append(row)
    require({r['seed'] for r in models}=={2026,2028},'Wrong Aug checkpoints')
    receipt=read(OLD/'DONE.json');oldrun=read(OLD/'RUN.json')
    require(receipt['status']=='VERIFIED_EXECUTION' and oldrun['protocol']['model']=='FULL-C4-Augmented (101-only)',
            'Wrong predecessor run')
    for name,digest in receipt['output_sha256'].items():
        require(sha(OLD/name)==digest,'Predecessor output changed: '+name);files[str((OLD/name).relative_to(ROOT))]=digest
    for name,digest in oldrun['code_sha256'].items():
        require(sha(PRIOR/name)==digest,'Predecessor implementation changed');files[str((PRIOR/name).relative_to(ROOT))]=digest
    for p in (OLD/'DONE.json',PRIOR/'PROTOCOL.json',OLD/'report01/DECISION.json'):
        files[str(p.relative_to(ROOT))]=sha(p)
    require(read(OLD/'report01/DECISION.json')['joint_physical_gate']=='FAIL','Previous FAIL must be retained')
    for row in models:
        require(receipt['input_sha256'][row['file']]==row['sha256'],'Pilot used other weights')
    return models,files


def load_model(row,device='cpu'):
    payload=torch.load(ROOT/row['file'],map_location='cpu',weights_only=True)
    require(payload['identity']==read(ROOT/row['run']) and payload['epoch']==row['epoch']
            and payload['backscatter_factor']==1.,'Payload source mismatch')
    s=study.make_solver(study.AUGMENTED,row['seed']).double().to(device).eval().requires_grad_(False)
    s.load_state_dict(payload['model'],strict=True)
    return s


def spectra(s,f):
    fft=lambda v:torch.fft.fft2(v,norm='forward')
    w=fft(f['omega']);inv=s.inverse_k_squared
    terms=dict(advection=-s.project(f['u']*f['omega_x']+f['v']*f['omega_y']),
               viscosity=s.config.physics.kinematic_viscosity_m2_s*f['laplacian_omega'],SGS=-f['div_q'],RHS=f['rhs'])
    if 'qd_x' in f:
        terms['Sd']=source(s,qvec(f,'qd_'));terms['Sb']=source(s,qvec(f,'qb_'))
    out=dict(E=.5*w.abs().square()*inv,Z=.5*w.abs().square())
    for name,v in terms.items():
        transfer=(w.conj()*fft(v)).real
        out['T_Z_'+name]=transfer;out['T_K_'+name]=transfer*inv
    return out


def state_rows(s,w,ref,ctx):
    h=torch.fft.fft2(w,norm='forward');y=torch.fft.fft2(ref,norm='forward')
    st=s.state(w);tr=s.state(ref);masks=geometry(s);result=[]
    uerr=rms(st['u']-tr['u'])/max(rms(tr['u']),1e-30)
    verr=rms(st['v']-tr['v'])/max(rms(tr['v']),1e-30)
    for name,mask in masks.items():
        mse=float((h-y).abs().square()[mask].sum());rp=float(y.abs().square()[mask].sum())
        K=.5*float((h.abs().square()*s.inverse_k_squared)[mask].sum())
        Kr=.5*float((y.abs().square()*s.inverse_k_squared)[mask].sum())
        Z=.5*float(h.abs().square()[mask].sum());Zr=.5*rp
        result.append(dict(**ctx,band=name,MSE=mse,omega_relative_L2=(mse/rp)**.5 if rp else None,
                           u_relative_L2=uerr,v_relative_L2=verr,K=K,K_ref=Kr,K_bias=(K-Kr)/Kr if Kr else None,
                           Z=Z,Z_ref=Zr,Z_bias=(Z-Zr)/Zr if Zr else None))
    return result
