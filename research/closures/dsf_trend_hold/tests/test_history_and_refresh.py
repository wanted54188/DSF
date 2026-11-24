"""CPU-only acceptance of parameter-history RK4, promotion, resume and report."""
from pathlib import Path
import numpy as np
import torch
import pc_common as c
import pc_engine as e
import pc_metrics as m
import pc_storage as storage


class MockJournal:
    def __init__(self):self.used=dict(CNN=0,RHS=0);self.phase='synthetic';self.decodes=0
    def tick(self,k):self.used[k]+=1
    def note(self,k,**kwargs):self.decodes+=int(k=='ALGEBRAIC_DECODE')


def model():
    s=c.q.study.make_solver(c.q.study.AUGMENTED,2026).double().eval().requires_grad_(False)
    s.closure.register_forward_pre_hook(lambda *_:(_ for _ in ()).throw(AssertionError('CPU fixture called real CNN')))
    def heads(st):
        raw=torch.zeros((7,100,100),dtype=torch.float64)
        raw[0]=.2+.02*st['omega'];raw[1]=.15;raw[2]=.1;raw[3]=-.1
        raw[4]=.02;raw[5]=.1+.02*st['omega'];raw[6]=-.1
        return raw
    s._synthetic_head_provider=heads
    s.closure_fields=lambda st:c.decoder.decode(s,st,heads(st))
    return s


def bank_test():
    def a(n):return dict(step=n,heads=torch.full((4,7,8,8),float(n)),file=str(n),sha256='software_test')
    b=e.Bank([a(0),a(1)]);b.observe(5,a(5))
    assert b.active[-1]['step']==1 and b.predict(5,1.,'Parameter-PC')[1]==5.
    assert torch.equal(b.predict(5,.5,'Parameter-PC')[0],torch.full_like(a(0)['heads'],5.5))
    assert torch.equal(b.predict(5,1.,'Parameter-Hold')[0],a(1)['heads'])
    b.observe(6,a(6));assert b.pending is None and b.predict(6,0.,'Parameter-PC')[1]==0.
    assert 'heads' not in b.descriptor()['active'][0]


def budget_test():
    aug=4*799+1;fresh=16*799+4
    par=4*sum(n%5 in (0,1) for n in range(2,800))+4
    assert (aug,fresh,par)==(3197,12788,1276)
    assert (12+aug+fresh+2*par)*6==c.P['hard_budget']['uninterrupted_main_CNN']
    assert (4+4*(4*799+1))*6==c.P['hard_budget']['uninterrupted_main_RHS']
    assert 3*6*(8+32)==c.P['hard_budget']['uninterrupted_dt_CNN']


def nonconvex_test(s,x):
    """Manufactured original-decoder pair; does NOT assert frozen-CNN reachability."""
    import math
    st=s.state(s.project(torch.sin(x)));cl=s.closure
    raw=torch.zeros((7,100,100),dtype=torch.float64);r=min(.7,.5*cl.cholesky_offdiagonal_limit)
    invsoft=lambda z:torch.log(torch.expm1(torch.as_tensor(z,dtype=torch.float64)))
    raw[1]=invsoft(1-cl.cholesky_diagonal_minimum)
    raw[3]=invsoft(math.sqrt(1-r*r)-cl.cholesky_diagonal_minimum)
    raw[4]=9.;raw[6]=9.;raw[2]=math.atanh(r/cl.cholesky_offdiagonal_limit)
    plus=c.decoder.decode(s,st,raw)
    raw[2]=-math.atanh(r/cl.cholesky_offdiagonal_limit);minus=c.decoder.decode(s,st,raw)
    for f in (plus,minus):c.q.physical(s,st,f)
    mean={k:.5*(plus[k]+minus[k]) for k in plus}
    check,_=c.feasible.envelope(s,st,e.vec(mean,'qd'),e.vec(mean,'qb'),mean)
    assert check['qd_envelope_pass'] and check['signed_transfer_pass'] and not check['signed_amplitude_pass']
    return dict(status='VERIFIED_COUNTEREXAMPLE',members_admissible=True,mean_admissible=False,
        mean_amplitude_ratio=check['max_signed_amplitude_ratio'],
        frozen_CNN_reachability='NOT_PROVED',real_CNN=0)


@torch.inference_mode()
def run(folder):
    folder.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);bank_test();budget_test()
    journal=MockJournal();s=model();x=torch.arange(100,dtype=torch.float64)*2*torch.pi/100
    y,x=torch.meshgrid(x,x,indexing='ij');w0=s.project(torch.sin(3*x+2*y)+.2*torch.cos(7*x-4*y))
    nonconvex=nonconvex_test(s,x)
    initial=e.basic_metrics(s,w0);warm=e.Engine(s,'Aug',journal,folder,folder/'warm',initial)
    w1,_=warm.advance(0,w0,None);old=warm.orbit(w0);now=warm.orbit(w1)
    a0=e.save_anchor(folder,folder/'anchors',0,old['heads']);a1=e.save_anchor(folder,folder/'anchors',1,now['heads'])
    desc=e.Bank([a0,a1]).descriptor();op=c.operators(s);warm.close()
    continuous={};method_rows=[]
    for method in c.METHODS:
        ss=model();bank=e.load_bank(folder,desc,'cpu') if method.startswith('Parameter') else None
        engine=e.Engine(ss,method,journal,folder,folder/method,initial);w=w1.clone()
        for n in range(1,7):
            out,_=engine.advance(n,w,bank)
            m.verify_step(e.cpu(w),e.cpu(out),e.cpu(torch.stack(engine.values)),op)
            if bank and n==5:assert bank.active[-1]['step']==1 and engine.records[-1]['tau']==5.
            if bank and n==6:assert bank.active[-1]['step']==6 and bank.pending is None
            w=out
        continuous[method]=e.cpu(w).copy();method_rows.append(dict(method=method,steps=6,status='PASS'))
        engine.prepare(10,w,bank);f=engine.rhs(w,1.05);snap=engine.snapshot(f)
        m.snapshot(snap,op,float(op['nu']),dict(case=110240101,seed=2026,method=method,step=10,time_s=1.05))
        if bank:assert np.array_equal(snap['predicted_heads'],e.cpu(bank.predict(10,0.,method)[0]))
        engine.close()
    # Joint C4 action must permute raw head orientations, NOT rotate Cholesky logits.
    equiv=[];base=None;base_rhs=None
    for turn in range(4):
        ss=model();active=[]
        for a in (a0,a1):
            active.append(dict(a,heads=a['heads'][[(j+turn)%4 for j in range(4)]].clone()))
        bank=e.Bank(active);engine=e.Engine(ss,'Parameter-PC',journal,folder,folder/'rotation',initial)
        out,_=engine.advance(2,c.q.spatial(w1,turn),bank)
        if turn==0:base=out.clone();base_rhs=engine.values[0].clone()
        else:
            error=max(e.relative(out,c.q.spatial(base,turn)),e.relative(engine.values[0],c.q.spatial(base_rhs,turn)))
            assert error<1e-12;equiv.append(error)
        engine.close()
    # Half-step stage timestamps and cumulative history time unit stay correct.
    ss=model();engine=e.Engine(ss,'Parameter-PC',journal,folder,folder/'half',initial)
    bank=e.load_bank(folder,desc,'cpu');engine.n=2;engine.bank=bank;w=w1.clone()
    for first,fracs in ((0.,(0.,.25,.25,.5)),(.5,(.5,.75,.75,1.))):
        engine.stage=0;engine.fractions=fracs;engine.values=[];before=w.clone()
        w=ss.step(w,1+c.H*(2+first),c.H/2)
        m.verify_step(e.cpu(before),e.cpu(w),e.cpu(torch.stack(engine.values)),op,h=c.H/2)
    engine.close()
    # ACTUAL driver interruption/resume with read-only synthetic producer.
    import driver
    startup=folder/'startup/110240101_2026'
    c.npz(startup/'ORIGINAL_W1.npz',**{k:e.cpu(v) for k,v in now['original'].items()})
    start=dict(omega0=e.cpu(w0),omega1=e.cpu(w1),rhs=e.cpu(torch.stack(warm.values)),
        metadata=c.packed(dict(bank=desc,initial=initial)),**{'observation__'+k:v for k,v in warm.snapshots[0].items()})
    old_model=c.model;old_engine=driver.Engine;interrupted=[False]
    class Interrupted(e.Engine):
        def rhs(self,w,t):
            if self.n==3 and self.stage==1 and not interrupted[0]:
                interrupted[0]=True;raise KeyboardInterrupt('SYNTHETIC_INTERRUPT')
            return super().rhs(w,t)
        def advance(self,n,w,bank):
            if n==7:raise e.TrajectoryFailure('FAILED_NUMERICAL','SYNTHETIC_END',dict(omega=e.cpu(w)))
            return super().advance(n,w,bank)
    j=c.Journal(folder/'test_invocations')
    try:
        c.model=lambda *a,**k:model();driver.Engine=Interrupted
        try:driver.run_trajectory(folder,110240101,2026,'Parameter-PC',{},start,j,'cpu')
        except KeyboardInterrupt:pass
        else:raise AssertionError('No interruption')
        driver.run_trajectory(folder,110240101,2026,'Parameter-PC',{},start,j,'cpu')
        last=c.load(folder/'trajectories/110240101_2026_Parameter-PC/steps/0007.npz')['omega']
        assert np.array_equal(last,continuous['Parameter-PC']) and j.used==dict(CNN=0,RHS=25)
    finally:j.close();c.model=old_model;driver.Engine=old_engine
    # Execute the actual half-step diagnostic driver on a synthetic accepted pair.
    import shutil
    from pc_timecheck import run_checks
    dtroot=folder/'timecheck_driver';dtroot.mkdir()
    src=folder/'trajectories/110240101_2026_Parameter-PC'
    target=dtroot/'trajectories/110240101_2026_Parameter-PC'
    (target/'steps').mkdir(parents=True)
    for k in (2,3):shutil.copyfile(src/'steps'/f'{k:04d}.npz',target/'steps'/f'{k:04d}.npz')
    for a in (a0,a1):
        path=dtroot/a['file'];path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/a['file'],path)
    c.npz(dtroot/'startup/110240101_2026/STARTUP.npz',metadata=c.packed(dict(initial=initial)))
    c.dump(target/'STATUS.json',dict(status='VERIFIED',software_fixture_only=True))
    saved=(c.P['cases'],c.P['seeds'],c.P['timestep_diagnostic']['steps'],c.METHODS)
    dtjournal=c.Journal(dtroot/'invocations')
    try:
        c.P['cases']=[110240101];c.P['seeds']=[2026];c.P['timestep_diagnostic']['steps']=[2];c.METHODS=('Parameter-PC',)
        c.model=lambda *a,**k:model()
        run_checks(dtroot,dict(checkpoints=[dict(seed=2026)]),dtjournal,'cpu')
        dtfolder=dtroot/'timecheck/110240101_2026_Parameter-PC_0002'
        assert c.read(dtfolder/'DONE.json')['status']=='VERIFIED_LOCAL_DT_CHECK'
        dz=c.load(dtfolder/'STATES.npz');h=c.H/2;ks=dz['rhs']
        mid=m.project(dz['initial']+(h/6)*(ks[0]+2*ks[1]+2*ks[2]+ks[3]),op)
        m.verify_step(dz['initial'],mid,ks[:4],op,h=h);m.verify_step(mid,dz['two_half'],ks[4:],op,h=h)
        assert dtjournal.used==dict(CNN=0,RHS=8)
        recovery=folder/'timecheck_recovery'
        recovered=recovery/'timecheck/110240101_2026_Parameter-PC_0002'
        recovered.mkdir(parents=True);shutil.copyfile(dtfolder/'STATES.npz',recovered/'STATES.npz')
        run_checks(recovery,dict(checkpoints=[]),dtjournal,'cpu')
        assert c.read(recovered/'DONE.json')['recovered_atomic_result'] and dtjournal.used==dict(CNN=0,RHS=8)
    finally:
        dtjournal.close();c.model=old_model
        c.P['cases'],c.P['seeds'],c.P['timestep_diagnostic']['steps'],c.METHODS=saved
    recovered_fail=folder/'failure_recovery';recovered_fail.mkdir()
    shutil.copyfile(src/'FAILED_STAGE.npz',recovered_fail/'FAILED_STAGE.npz')
    assert storage.recover_failure(recovered_fail)['recovered_atomic_failure']
    assert storage.recover_failure(recovered_fail)['status']=='FAILED_NUMERICAL'
    with storage.RunLock(folder):
        try:
            with storage.RunLock(folder):pass
        except RuntimeError:pass
        else:raise AssertionError('Concurrent writer accepted')
    # Strict mean gate remains a gate, never a limiter/fallback.
    ss=model();engine=e.Engine(ss,'Aug',journal,folder,folder/'bad',initial)
    try:e.verify_physical(ss,ss.state(w1),torch.zeros((2,100,100)),torch.ones((2,100,100))*1e5,now['original'])
    except e.TrajectoryFailure as error:assert error.status=='FAILED_PHYSICAL'
    else:raise AssertionError('Mean physical gate ignored')
    engine.close()
    c.npz(startup/'STARTUP.npz',**start);c.npz(folder/'OPERATORS.npz',**op)
    ref=folder/'SYNTHETIC_REFERENCE.npz'
    c.npz(ref,reference_omega=e.cpu(w0),reference_dot=np.zeros((100,100)),reference_source=np.zeros((100,100)))
    c.dump(folder/'RUN.json',dict(code_identity=c.local_identity(),synthetic_software_test=True,
        inventory=dict(input_sha256={str(ref.relative_to(c.ROOT)):c.sha(ref)},
            references={'110240101':dict(kind='frames',frames={'0':str(ref.relative_to(c.ROOT))},steps=[0])},source_reuse=[])))
    from pc_report import build_report
    r1=build_report(folder);r2=build_report(folder)
    assert c.read(r2/'DECISION.json')['decision']=='SOFTWARE_TEST_ONLY'
    assert all(r['omega_MSE_improvement_pct']=='' for r in c.csvread(r2/'PAIRED_LONG_SUMMARY.csv'))
    assert journal.used['CNN']==0 and j.used['CNN']==0
    result=dict(status='PASS',real_checkpoint_CNN=0,real_full_LES='NOT_RUN',
        synthetic_total_RHS=journal.used['RHS']+j.used['RHS']+dtjournal.used['RHS'],algebraic_decodes=journal.decodes,
        pending_pair_tau_to5='PASS',four_method_RK4=method_rows,joint_history_C4_max_relative=max(equiv),
        original_decoder_identity='PASS',current_state_physical_members_and_mean='PASS',
        strict_mean_failure_no_fallback='PASS',half_step_history_and_RK_identity='PASS',
        actual_half_step_diagnostic_driver='PASS',
        atomic_failure_and_half_step_recovery='PASS',universal_signed_closure=nonconvex,
        actual_driver_resume_bitwise='PASS',discarded_RHS_counted=1,concurrent_lock='PASS',
        automatic_report_plots_rebuild='PASS',partial_window_NOT_RUN='PASS')
    c.dump(folder/'SYNTHETIC.json',result);return result


if __name__=='__main__':
    import time,json
    print(json.dumps(run(c.HERE/'synthetic_runs'/str(time.time_ns()))))
