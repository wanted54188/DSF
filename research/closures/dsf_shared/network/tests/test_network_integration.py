import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE.parents[1]));sys.path.insert(0,str(HERE))
import fc4_model as model
import fc4_io as io
from spnsgs.config import load_config
from spnsgs.solver import SpectralVorticitySolver
torch.set_num_threads(2)


def make(n=24,mode=None):
    cfg=load_config(io.ROOT/'results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json')
    torch.manual_seed(2026)
    s=SpectralVorticitySolver(cfg,n,n,2*np.pi/n,2*np.pi/n,torch.device('cpu'),torch.float64)
    s.closure.set_backscatter_factor(1.)
    if mode:model.install(s,mode,True)
    x=torch.arange(n,dtype=torch.float64)*2*np.pi/n;y,x=torch.meshgrid(x,x,indexing='ij')
    w=s.project(torch.sin(2*x+3*y)+.4*torch.cos(4*x-y))
    return s,w


def test_c2_bitwise_all_fields_parameters_and_gradients(record_property):
    a,w=make();b,_=make(mode='FULL-C2')
    fa=a.rhs(w,1.);fb=b.rhs(w,1.)
    assert set(fa)==set(fb)
    assert all(torch.equal(v,fb[k]) for k,v in fa.items() if torch.is_tensor(v))
    assert a.state_dict().keys()==b.state_dict().keys()
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==80008
    fa['div_q'].square().mean().backward();fb['div_q'].square().mean().backward()
    for p,q in zip(a.parameters(),b.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.equal(p.grad,q.grad)
    record_property('C2_max_abs_error',0.);record_property('C2_bitwise',True)


@pytest.mark.parametrize('mode',model.MODES)
def test_modes_prior_input_and_bounds(mode,record_property):
    s,w=make(mode=mode);base,_=make();feat=[];raw=[];basefeat=[];baseraw=[]
    h=s.closure.input.register_forward_pre_hook(lambda m,args:feat.append(args[0].detach().clone()))
    r=s.closure.output.register_forward_hook(lambda m,args,value:raw.append(value.detach().clone()))
    hh=base.closure.input.register_forward_pre_hook(lambda m,args:basefeat.append(args[0].detach().clone()))
    rr=base.closure.output.register_forward_hook(lambda m,args,value:baseraw.append(value.detach().clone()))
    f=s.rhs(w,1.);b=base.rhs(w,1.)
    for hook in (h,r,hh,rr):hook.remove()
    assert feat[0].shape==(1,12,24,24) and torch.isfinite(feat[0]).all()
    assert torch.equal(feat[0][:,:10],basefeat[0][:,:10])
    p=s.closure._integration_last
    if mode!='FULL-C2':
        for got,expected in zip(p['c4'],model.c4.flux4(s,f,s.config.data.filter_width_m)):
            assert torch.equal(got,expected)
    if mode in ('FULL-C2','FULL-Prior4'):
        assert torch.equal(feat[0],basefeat[0]);assert torch.equal(raw[0],baseraw[0])
        assert torch.equal(f['qd_x'],b['qd_x']) and torch.equal(f['qd_y'],b['qd_y'])
    else:
        assert not torch.equal(feat[0][:,10:],basefeat[0][:,10:])
    if mode in ('FULL-C2','FULL-Input4'):
        assert all(torch.equal(x,y) for x,y in zip(p['constitutive'],p['c2']))
    else:assert all(torch.equal(x,y) for x,y in zip(p['constitutive'],p['c4']))
    v=model.violations(s,f)
    for k in ('signed_norm_violation','negative_transfer_violation','PSD_negative_eigenvalue','trace_error'):
        assert v[k]<1e-11
        record_property(k,v[k])
    sd=model.c4.source(s,(f['qd_x'],f['qd_y']));sb=model.c4.source(s,(f['qb_x'],f['qb_y']))
    assert float(abs(sd+sb+f['div_q']).max())<1e-12
    assert float(abs(f['div_q'].mean()))<1e-13
    assert float(abs((w*sd).mean()+f['pi_d'].mean()))<1e-12


def test_limiter_active_and_zero_fields(record_property):
    s,w=make(mode='FULL-C4')
    # Exercise original limiter at a constructed strongly injecting direction.
    with torch.no_grad():
        s.closure.output.weight.zero_();s.closure.output.bias.zero_()
        s.closure.output.bias[4]=5.;s.closure.output.bias[5]=-5.;s.closure.output.bias[6]=-5.
    f=s.rhs(w,1.)
    v=model.violations(s,f)
    assert max(v[k] for k in ('signed_norm_violation','negative_transfer_violation','PSD_negative_eigenvalue'))<1e-11
    z=s.rhs(w*0,1.)
    assert torch.isfinite(z['qx']).all() and torch.isfinite(z['rhs']).all()
    record_property('manufactured_limiter_active_fraction',float((f['backscatter_transfer_limiter']<1).double().mean()))


def test_backward_and_save_load_roundtrip(tmp_path):
    for mode in ('FULL-C2','FULL-C4'):
        s,w=make(mode=mode);opt=torch.optim.AdamW(s.parameters(),lr=1e-4)
        loss=s.rhs(w,1.)['div_q'].square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in s.parameters())
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in s.parameters())
        opt.step();path=tmp_path/(mode+'.pt')
        torch.save({'model':s.state_dict(),'optimizer':opt.state_dict(),'mode':mode},path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        other,_=make(mode=mode);other.load_state_dict(saved['model'])
        assert torch.equal(other.rhs(w,1.)['div_q'],s.rhs(w,1.)['div_q'])
        opt2=torch.optim.AdamW(other.parameters(),lr=1e-4);opt2.load_state_dict(saved['optimizer'])
        for net,optimizer in ((s,opt),(other,opt2)):
            optimizer.zero_grad();net.rhs(w,1.)['div_q'].square().mean().backward();optimizer.step()
        assert all(torch.equal(x,y) for x,y in zip(s.parameters(),other.parameters()))


def test_native_rk4_and_current_stage_prior():
    s,w=make(mode='FULL-C4');seen=[]
    native=s.rhs
    def rhs(x,t):seen.append(x.clone());return native(x,t)
    s.rhs=rhs;result=s.step(w,1.,.001)
    assert len(seen)==4 and not torch.equal(seen[0],seen[1]) and torch.isfinite(result).all()


def test_c4_stress_curl_and_flux_source(record_property):
    # Independent stress formulation, including two mixed-index contributions.
    s,w=make(n=100,mode='FULL-C4');state=s.state(w);d=s.derivative;delta=s.config.data.filter_width_m
    def tau(a,b):
        value=delta**2/12*sum(state[a+'_'+ax]*state[b+'_'+ax] for ax in ('x','y'))
        return value+delta**4/288*(d(state[a+'_x'],'x')*d(state[b+'_x'],'x')+
            2*d(state[a+'_x'],'y')*d(state[b+'_x'],'y')+d(state[a+'_y'],'y')*d(state[b+'_y'],'y'))
    # omega = dv/dx-du/dy in the original solver convention.
    curl_div=-s.project(d(d(tau('v','u'),'x')+d(tau('v','v'),'y'),'x')-
                        d(d(tau('u','u'),'x')+d(tau('u','v'),'y'),'y'))
    flux=model.c4.source(s,model.c4.flux4(s,state,delta))
    relative=float(torch.linalg.vector_norm(curl_div-flux)/torch.linalg.vector_norm(flux))
    # Native solver creates k and inverse-k^2 in float32 before converting to
    # requested float64. Do not change those buffers in production to force
    # an ideal identity. This is an audited operator precision floor.
    assert relative<2e-6
    record_property('native_C4_stress_vs_vorticity_source_relative',relative)
    record_property('native_tolerance_float32_constructed_buffers',2e-6)
    # Independent ideal Fourier control, used ONLY in this synthetic test.
    v=torch.fft.fftfreq(100,d=1/100,dtype=torch.float64)
    s.ky,s.kx=torch.meshgrid(v,v,indexing='ij');s.k_squared=s.kx**2+s.ky**2
    s.inverse_k_squared=torch.where(s.k_squared>0,1/s.k_squared,0)
    state=s.state(w)
    exact_stress=-s.project(d(d(tau('v','u'),'x')+d(tau('v','v'),'y'),'x')-
                            d(d(tau('u','u'),'x')+d(tau('u','v'),'y'),'y'))
    exact_flux=model.c4.source(s,model.c4.flux4(s,state,delta))
    ideal=float(torch.linalg.vector_norm(exact_stress-exact_flux)/torch.linalg.vector_norm(exact_flux))
    assert ideal<2e-11
    record_property('ideal_control_C4_stress_vs_vorticity_source_relative',ideal)


def test_cpu_float32_and_projection(record_property):
    s,w=make(n=100,mode='FULL-C4');s=s.float();w=w.float()
    f=s.rhs(w,1.);v=model.violations(s,f)
    assert max(v[k] for k in ('signed_norm_violation','negative_transfer_violation','PSD_negative_eigenvalue','trace_error'))<2e-5
    for k in ('signed_norm_violation','negative_transfer_violation','PSD_negative_eigenvalue','trace_error'):
        record_property('float32_'+k,v[k])
    record_property('float32_absolute_tolerance',2e-5)
    assert torch.isfinite(f['rhs']).all()
    # Mask fixed at 33 in each coordinate, source mean is numerical roundoff.
    assert not bool(s.dealias[0,34]) and bool(s.dealias[33,33])


def test_access_and_rollout_adaptation():
    import fc4_rollout
    for seed in (110240201,110240202,110240301,110240302):
        with pytest.raises(PermissionError):io.guarded(io.ROOT/str(seed)/'x.npy')
    with pytest.raises(PermissionError):io.output(io.ROOT/'original.pt')
    # Compile only: no reference access, model construction or LES.
    fn=fc4_rollout.adapted_task({},'FULL-C4')
    assert callable(fn)
    assert sum(r['CNN_forwards'] for r in fc4_rollout.matrix('short'))==5760


def test_cli_default_no_decode(monkeypatch,tmp_path,capsys):
    import validation
    monkeypatch.setattr(io,'reference',lambda *x:pytest.fail('Real reference decoded'))
    monkeypatch.setattr(io,'solver',lambda *x:pytest.fail('Real model loaded'))
    for stage in ('apriori','short','long','train'):
        validation.main([stage,'--output',str(tmp_path)])
    assert capsys.readouterr().out.count('DRY_RUN_NO_FIELDS_NO_WEIGHTS_NO_TRAINING')==4
    assert not list(tmp_path.iterdir())


def test_reused_cpu_harness_runs_synthetic_and_resumes(monkeypatch,tmp_path):
    import fc4_rollout
    from types import SimpleNamespace
    c=io.legacy();toy,w=make(n=100)
    toy.calls=0;toy.stages=[];toy.rhs_seconds=0.;toy.capture_handle=SimpleNamespace(remove=lambda:None)
    def rhs(x,t):
        toy.calls+=1;st=toy.state(x);z=x*0
        # A manufactured linear ODE, not a turbulence/model experiment.
        st.update(rhs=-.03*x,div_q=.03*x,qd_x=z,qd_y=z,qb_x=z,qb_y=z)
        toy.stages.append(st);return st
    toy.rhs=rhs
    times=1+np.arange(161)*.00125
    ref=dict(time_s=times,omega=np.stack([w.numpy()*np.exp(-.03*(t-1)) for t in times]))
    monkeypatch.setattr(c,'reference',lambda *args:ref)
    monkeypatch.setattr(io,'solver',lambda *args:toy)
    f=fc4_rollout.adapted_task({},'FULL-C4');freeze=dict(code_sha256={'synthetic':'v1'},inputs={})
    f(freeze,tmp_path,'short',110240101,2026,.005,False)
    dest=tmp_path/'short/110240101_2026_FULL-C4_0.005'
    done=io.read(dest/'DONE.json')
    assert done['status']=='COMPLETED' and done['RHS_calls']==160 and done['method']=='FULL-C4'
    assert (dest/'SUMMARY.json').exists()
    f(freeze,tmp_path,'short',110240101,2026,.005,True)
    # Copying a mode checkpoint must fail identity matching.
    with np.load(dest/'CHECKPOINT.npz',allow_pickle=False) as z:
        identity=__import__('json').loads(str(z['identity']))
    assert identity['method']=='FULL-C4'


def test_report_missing_results_never_GO(tmp_path):
    import fc4_report
    report=fc4_report.generate(dict(code_sha256=io.code_identity()),tmp_path)
    assert io.read(report/'DECISION.json')['status']=='INCONCLUSIVE'
    assert 'NOT_RUN' in (report/'FULL_C4_FROZEN_TRANSFER_CN.md').read_text(encoding='utf-8')


def test_paired_training_entry_same_initialization_and_resume(monkeypatch,tmp_path):
    import fc4_training
    import spnsgs.paper as paper
    import spnsgs.config as configs
    import spnsgs.data as data
    template=io.ROOT/'results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json'
    original_loader=configs.load_config
    raw=tmp_path/'raw';raw.mkdir();dummy=raw/'synthetic.csv';dummy.write_text('synthetic metadata only',encoding='utf-8')
    def loader(path):
        cfg=original_loader(path);cfg.data.data_dir=raw;return cfg
    monkeypatch.setattr(configs,'load_config',loader)
    monkeypatch.setattr(data,'discover_files',lambda directory,pattern:[(.1,dummy)])
    initial=[];observed_modes=[];resume_seen=[]
    def tiny_trainer(cfg,resume_checkpoint=None,stop_after_epoch=None):
        torch.manual_seed(cfg.training.seed)
        s=paper.SpectralVorticitySolver(cfg,24,24,2*np.pi/24,2*np.pi/24,torch.device('cpu'),torch.float64)
        initial.append([p.detach().clone() for p in s.parameters()]);observed_modes.append(s.closure._integration_mode)
        resume_seen.append(resume_checkpoint is not None)
        folder=Path(cfg.training.output_dir)
        torch.save({'synthetic_only':True},folder/'last_checkpoint.pt')
        torch.save({'synthetic_only':True},folder/'best_closure.pt')
    monkeypatch.setattr(paper,'train_paper',tiny_trainer)
    a=dict(parent=dict(models=[dict(seed=2026,config=str(template))]),code_sha256=io.code_identity(),input_sha256={})
    fc4_training.execute(a,tmp_path,[2026],'cpu')
    assert observed_modes==['FULL-C2','FULL-C4']
    assert all(torch.equal(p,q) for p,q in zip(*initial))
    fc4_training.execute(a,tmp_path,[2026],'cpu',True)
    assert resume_seen==[False,False,True,True]
    assert io.read(tmp_path/'PAIRED_MODELS.json')['models'][0]['method']=='Retrained-FULL-C2'

