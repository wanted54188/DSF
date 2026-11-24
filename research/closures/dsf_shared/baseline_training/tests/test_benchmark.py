import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parents[1]))
import common as c
import data
import training
import objective
import evaluation
import rollout
import metrics
torch.set_num_threads(2)


def manufactured(n=12,count=3):
    s=c.make_solver('FULL-C2',2026,n=n)
    x=torch.arange(n,dtype=torch.float32)*2*torch.pi/n;y,x=torch.meshgrid(x,x,indexing='ij')
    cache={k:[] for k in (*data.STATE,'fourth_x','fourth_y','source','qx','qy','pi')}
    for i in range(count):
        w=s.project(torch.sin(x+2*y)+(.3+.01*i)*torch.cos(3*x-y));st=s.state(w)
        inc=c.prior.c4.fourth_increment(s,st,s.config.data.filter_width_m)
        q=c.prior.c4.flux4(s,st,s.config.data.filter_width_m);source=c.prior.c4.source(s,q)
        for k in data.STATE:cache[k].append(st[k].detach())
        for k,v in [('fourth_x',inc[0]),('fourth_y',inc[1]),('source',source),('qx',q[0]),('qy',q[1]),
                    ('pi',-(q[0]*st['omega_x']+q[1]*st['omega_y']))]:cache[k].append(v.detach())
    return {**{k:torch.stack(v) for k,v in cache.items()},'time_s':torch.arange(count,dtype=torch.float64)*.05+1}


def test_same_initialization_and_parameter_count():
    hashes=[]
    for m in c.MODES:
        s=c.make_solver(m,2026,n=12);hashes.append(c.model_hash(s))
        assert sum(p.numel() for p in s.parameters())==80008
    assert len(set(hashes))==1


@pytest.mark.parametrize('mode',c.MODES)
def test_cached_prior_matches_live_values_and_parameter_gradients(mode,record_property):
    cache=manufactured();live=c.make_solver(mode,2026,n=12);cached=c.make_solver(mode,2026,n=12)
    data.enable_cached_prior(cached);w=cache['omega'][0]
    f=live.rhs(w,1.);st,g,source=data.forward_frame(cached,cache,0)
    assert torch.equal(-f['div_q'],source)
    f['div_q'].square().mean().backward();source.square().mean().backward()
    for p,q in zip(live.parameters(),cached.parameters()):
        if p.grad is not None:assert torch.equal(p.grad,q.grad)
    record_property('cached_vs_live_source_max_abs_error',0.)


def test_loss_exact_labels_and_source_budget_sign():
    cache=manufactured();s=c.make_solver('FULL-C4',2026,n=12);st={k:cache[k][0] for k in data.STATE}
    f=s.closure_fields(st)
    f.update(qx=cache['qx'][0],qy=cache['qy'][0],pi=cache['pi'][0])
    q=objective.quantities(s,cache,0,st,f,cache['source'][0])
    for key in ('source','flux','signed_transfer','signed_mean','budget'):assert abs(float(q[key]))<1e-10
    assert all(torch.isfinite(x) for x in q.values())
    pair=(cache['omega'][0]*cache['source'][0]).mean()
    assert abs(float(pair+cache['pi'][0].mean()))<1e-6


def test_matched_eight_runs_and_explicit_roles():
    a=dict(protocol=c.protocol());rows=c.matrix(a)
    assert len(rows)==8 and {r['train_initial'] for r in rows}=={110240101}
    assert {r['validation_initial'] for r in rows}=={110240102}
    assert sum(r['optimizer_updates'] for r in rows)==16800
    r=rollout.matrix(a,'long')
    assert sum(not x['analytic_alias'] for x in r)==56
    assert sum(x['expected_CNN_forwards'] for x in r)==218880
    assert all(x['method'] in evaluation.METHODS for x in r)


def test_synthetic_training_epoch_resume_exact(monkeypatch,tmp_path):
    original=c.make_solver
    monkeypatch.setattr(c,'make_solver',lambda mode,seed,device='cpu',n=100:original(mode,seed,device,n=12))
    cache=manufactured();p=copy.deepcopy(c.protocol());p.update(epochs=4,batch_frames=2,
        signed_warmup_epochs=0,signed_ramp_epochs=1,ema_start_epoch=2,validation_every_epochs=1)
    a=dict(protocol=p,code_sha256=c.code())
    for name in ('continuous','resumed'):
        for seed in c.CASES:c.dump(tmp_path/name/'data'/str(seed)/'DONE.json',dict(cache_sha256='manufactured_fixture'))
    training.train_one(a,tmp_path/'continuous','FULL-C4',2026,cache,cache,'cpu')
    training.train_one(a,tmp_path/'resumed','FULL-C4',2026,cache,cache,'cpu',stop_after_epoch=2)
    training.train_one(a,tmp_path/'resumed','FULL-C4',2026,cache,cache,'cpu',resume=True)
    path=lambda run:tmp_path/run/'training/FULL-C4_2026'
    x=torch.load(path('continuous')/'last.pt',weights_only=False);y=torch.load(path('resumed')/'last.pt',weights_only=False)
    assert x['updates']==y['updates']==8
    assert all(torch.equal(x['model'][k],y['model'][k]) for k in x['model'])
    assert x['best_epoch']==y['best_epoch']
    assert all(torch.equal(x['ema'][k],y['ema'][k]) for k in x['ema'])
    xx=torch.load(path('continuous')/'best.pt',weights_only=False);yy=torch.load(path('resumed')/'best.pt',weights_only=False)
    assert all(torch.equal(xx['model'][k],yy['model'][k]) for k in xx['model'])


def test_masks_parseval_and_no_failed_full_window():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,100));g=metrics.geometry()
    h=metrics.fft(x)*g['mask'];x=np.fft.ifft2(h,norm='forward').real
    r=metrics.state_row(x,.9*x,2.)
    assert abs(r['error_power_all']-r['error_power_low']-r['error_power_high'])<1e-12
    assert metrics.summarize([r],False)['omega_relative_L2_all_endpoint'] is None


def test_p1_weight_and_new_c2_anchor(monkeypatch):
    # No .pt is loaded: a manufactured native solver checks the actual wrapper.
    original=c.make_solver
    monkeypatch.setattr(evaluation,'model',lambda a,out,mode,seed,device='cpu',cached=False:original(mode,seed,device,n=100))
    s=evaluation.runtime({},None,'P1-C4',2026,'cpu');w=manufactured(n=100,count=1)['omega'][0]
    with torch.no_grad():st=s.rhs(w,1.)
    sf=-st['div_q'];sc4=c.prior.c4.source(s,c.prior.c4.flux4(s,st,s.config.data.filter_width_m))
    expected=s.project(sf+c.prior.c4.multiply(s,sc4-sf,evaluation.weights(s)))
    assert torch.equal(st['source_applied'],expected) and s.calls==1
    assert float(evaluation.weights(s)[0,8])==1 and float(evaluation.weights(s)[0,16])==0
    s.capture_handle.remove()


def test_default_cli_does_not_decode_train_or_evolve(monkeypatch,tmp_path,capsys):
    import pipeline
    monkeypatch.setattr(data,'prepare',lambda *x:pytest.fail('Forbidden real preparation'))
    monkeypatch.setattr(training,'run',lambda *x:pytest.fail('Forbidden training'))
    monkeypatch.setattr(evaluation,'apriori',lambda *x:pytest.fail('Forbidden inference'))
    monkeypatch.setattr(rollout,'run',lambda *x:pytest.fail('Forbidden evolution'))
    for stage in ('prepare','train','evaluate','short','long'):pipeline.main([stage,'--output',str(tmp_path)])
    assert capsys.readouterr().out.count('DRY_RUN_NO_DATA_DECODE_NO_TRAINING_NO_LES')==5
    assert not list(tmp_path.iterdir())
    for seed in (110240201,110240202,110240301,110240302):
        with pytest.raises(PermissionError):c.safe(c.ROOT/str(seed)/'x.npy')


def test_prepare_uses_only_selected_synthetic_labels(monkeypatch,tmp_path):
    native=c.make_solver
    monkeypatch.setattr(c,'make_solver',lambda mode,seed,device='cpu',n=100:native(mode,seed,device,n=12))
    monkeypatch.setattr(c,'verify',lambda a:None)
    s=c.make_solver('FULL-C2',2026);x=torch.arange(12,dtype=torch.float32)*2*torch.pi/12
    w=s.project(torch.sin(x)[None,:].expand(12,12));st=s.state(w)
    dot=s.project(1e-6*st['laplacian_omega']).numpy();references=[]
    for seed in c.CASES:
        d=tmp_path/'raw'/str(seed);d.mkdir(parents=True)
        t=1+np.arange(3201)*.00125;np.save(d/'time_s.npy',t);fields={}
        for name,value in [('reference_omega',w.numpy()),('reference_source',w.numpy()*0),
                           ('reference_dot',dot),('reference_qx',w.numpy()*0),('reference_qy',w.numpy()*0)]:
            f=d/(name+'.npy');np.save(f,np.broadcast_to(value.astype(np.float64),(3201,12,12)))
            fields[name]=dict(path=str(f),expected_sha256=c.sha(f))
        references.append(dict(seed=seed,variant='n1536_half',role='train' if seed==110240101 else 'development_validation',
                               fields=fields,time_path=str(d/'time_s.npy')))
    a=dict(protocol=c.protocol(),code_sha256=c.code(),references=references)
    data.prepare(a,tmp_path/'run')
    saved=torch.load(tmp_path/'run/data/110240101/CACHE.pt',weights_only=False)
    assert saved['omega'].shape==(81,12,12) and float(saved['time_s'][0])==1 and float(saved['time_s'][-1])==5
    assert c.read(tmp_path/'run/data/110240102/DONE.json')['CNN_forwards']==0


def test_synthetic_rollout_and_completed_resume(monkeypatch,tmp_path):
    from types import SimpleNamespace
    a=dict(protocol=c.protocol(),code_sha256=c.code())
    c.dump(tmp_path/'BASELINE_REGISTRY.json',dict(synthetic_only=True))
    s=c.make_solver('FULL-C2',2026);cache=manufactured(n=100,count=1);w=cache['omega'][0]
    s.calls=0;s.rhs_calls=0;s.rhs_seconds=0.;s.stages=[];s.capture_handle=SimpleNamespace(remove=lambda:None)
    def rhs(x,t):
        st=s.state(x);st.update(rhs=-.03*x,source_applied=-.03*x)
        s.stages.append(st);s.rhs_calls+=1;return st
    s.rhs=rhs
    monkeypatch.setattr(evaluation,'runtime',lambda *args:s)
    times=1+np.arange(161)*.00125
    refs={(110240102,'n1536_half'):dict(time_s=times,
        omega=np.stack([w.numpy()*np.exp(-.03*(t-1)) for t in times]))}
    row=dict(stage='short',initial=110240102,training_seed=2026,method='FULL-C4',dt=.005,analytic_alias=False,steps=40,CNN_forwards=160)
    rollout.task(a,tmp_path,row,refs,'cpu')
    done=c.read(tmp_path/'short'/rollout.name(row)/'DONE.json')
    assert done['status']=='COMPLETED' and done['RHS_calls']==160
    rollout.task(a,tmp_path,row,refs,'cpu',True)
    summary=c.read(tmp_path/'short'/rollout.name(row)/'SUMMARY.json')[0]
    assert summary['completed_window'] and summary['role']=='DEVELOPMENT_VALIDATION_NOT_FINAL_TEST'


def test_report_missing_runs_explicit(tmp_path):
    import report
    d=report.generate(dict(protocol=c.protocol()),tmp_path)
    assert c.read(d/'STATUS.json')['training_completed']==0
    text=(d/'NS_BASELINE_REPORT_CN.md').read_text(encoding='utf-8')
    assert 'NOT_RUN' in text and 'REFERENCE_NOT_CONVERGED' in text


def test_eight_group_driver_and_registry_synthetic_only(monkeypatch,tmp_path):
    # A tiny fixture verifies the real all-eight driver. Its baseline pointer
    # stays under this pytest temp directory, never the actual study root.
    fixture=tmp_path/'fixture_study';fixture.mkdir();out=fixture/'results/toy'
    p=copy.deepcopy(c.protocol());p.update(epochs=1,batch_frames=2,signed_warmup_epochs=0,
        signed_ramp_epochs=1,ema_start_epoch=1,validation_every_epochs=1)
    a=dict(protocol=p,code_sha256=c.code());cache=manufactured()
    native=c.make_solver
    monkeypatch.setattr(c,'HERE',fixture)
    monkeypatch.setattr(c,'verify',lambda a:None)
    monkeypatch.setattr(c,'make_solver',lambda mode,seed,device='cpu',n=100:native(mode,seed,device,n=12))
    monkeypatch.setattr(data,'load_cache',lambda a,out,seed,device:cache)
    for seed in c.CASES:c.dump(out/'data'/str(seed)/'DONE.json',dict(cache_sha256='synthetic_fixture'))
    training.run(a,out,'cpu')
    r=c.read(out/'BASELINE_REGISTRY.json')
    assert len(r['models'])==8
    assert len({x['initial_parameter_sha256'] for x in r['models'] if x['seed']==2026})==1
    assert c.read(fixture/'LATEST_BASELINE.json')['status']=='EIGHT_MATCHED_RUNS_COMPLETE'
    assert evaluation.registry(a,out)['primary_full_baseline']=='NEW_TRAINED_FULL-C2'
