"""Guarded CPU acceptance of full-window adapter; no checkpoint CNN."""
import time
import numpy as np
import torch
import paper_dt as p
import pc_tests
c,e,d=p.c,p.e,p.d

@torch.inference_mode()
def main():
    torch.set_num_threads(2);p.bind();root=p.HERE/'cpu_runs'/str(time.time_ns());root.mkdir(parents=True)
    journal=pc_tests.MockJournal();s=pc_tests.model();x=torch.arange(100,dtype=torch.float64)*2*torch.pi/100
    y,x=torch.meshgrid(x,x,indexing='ij');w0=s.project(torch.sin(3*x+2*y)+.2*torch.cos(7*x-4*y))
    initial=e.basic_metrics(s,w0);eng=p.FineEngine(s,'Aug',journal,root,root/'warm',initial)
    w1,_=eng.advance(0,w0,None);op=c.operators(s)
    p.verify(e.cpu(w0),e.cpu(w1),e.cpu(torch.stack(eng.values)),op)
    old,now=eng.orbit(w0),eng.orbit(w1)
    a0=e.save_anchor(root,root/'anchors',0,old['heads']);a1=e.save_anchor(root,root/'anchors',1,now['heads'])
    desc=e.Bank([a0,a1]).descriptor();warm_rhs=e.cpu(torch.stack(eng.values));snap=eng.snapshots[0];eng.close()
    continuous={}
    for method in c.METHODS:
        ss=pc_tests.model();bank=e.load_bank(root,desc,'cpu') if method.startswith('Parameter') else None
        ee=p.FineEngine(ss,method,journal,root,root/method,initial);w=w1.clone()
        for n in range(1,7):
            before=w.clone();w,_=ee.advance(n,w,bank)
            assert len(ee.values)==8
            p.verify(e.cpu(before),e.cpu(w),e.cpu(torch.stack(ee.values)),op)
            if bank and n==5:assert bank.active[-1]['step']==1 and ee.records[-1]['tau']==5
            if bank and n==6:assert bank.active[-1]['step']==6
        continuous[method]=e.cpu(w).copy();ee.close()
    startup=root/'startup/110240101_2026';c.npz(startup/'ORIGINAL_W1.npz',**{k:e.cpu(v) for k,v in now['original'].items()})
    start=dict(omega0=e.cpu(w0),omega1=e.cpu(w1),rhs=warm_rhs,metadata=c.packed(dict(bank=desc,initial=initial)),
        **{'observation__'+k:v for k,v in snap.items()})
    original=c.model;originalengine=d.Engine;interrupted=[False]
    class Interrupt(p.FineEngine):
        def rhs(self,w,t):
            if self.n==3 and self.half_index==1 and self.stage==1 and not interrupted[0]:
                interrupted[0]=True;raise KeyboardInterrupt('SYNTHETIC_HALF_INTERRUPT')
            return super().rhs(w,t)
        def advance(self,n,w,bank):
            if n==7:raise e.TrajectoryFailure('FAILED_NUMERICAL','SYNTHETIC_END',dict(omega=e.cpu(w)))
            return super().advance(n,w,bank)
    j=c.Journal(root/'invocations')
    try:
        c.model=lambda *a,**k:pc_tests.model();d.Engine=Interrupt
        try:d.run_trajectory(root,110240101,2026,'Parameter-PC',{},start,j,'cpu')
        except KeyboardInterrupt:pass
        else:raise AssertionError('No half-step interruption')
        d.run_trajectory(root,110240101,2026,'Parameter-PC',{},start,j,'cpu')
        folder=root/'trajectories/110240101_2026_Parameter-PC'
        last=c.load(folder/'steps/0007.npz')['omega']
        assert np.array_equal(last,continuous['Parameter-PC']) and j.used['RHS']==53 and j.used['CNN']==0
        assert len(p.scan(folder))==6
    finally:j.close();c.model=original;d.Engine=originalengine
    c.npz(startup/'STARTUP.npz',**start);c.npz(root/'OPERATORS.npz',**op)
    ref=root/'SYNTHETIC_REFERENCE.npz';c.npz(ref,reference_omega=e.cpu(w0),reference_dot=np.zeros((100,100)),reference_source=np.zeros((100,100)))
    c.dump(root/'RUN.json',dict(code_identity=c.local_identity(),synthetic_software_test=True,
        inventory=dict(input_sha256={str(ref.relative_to(c.ROOT)):c.sha(ref)},
            references={'110240101':dict(kind='frames',frames={'0':str(ref.relative_to(c.ROOT))},steps=[0])},source_reuse=[])))
    import pc_report
    report=pc_report.build_report(root)
    assert c.read(report/'DECISION.json')['decision']=='SOFTWARE_TEST_ONLY'
    assert journal.used['CNN']==0
    assert 6*(16+6393+25572+2*1276)==p.PLAN['full_window_dt']['CNN_without_replay']
    assert 6*(8+4*6393)==p.PLAN['full_window_dt']['RHS_without_replay']
    result=dict(status='PASS',real_CNN=0,full_window_real_LES='NOT_RUN',
        eight_stage_four_method='PASS',quarter_stage_timestamps='PASS',fixed_physical_refresh_tau5='PASS',
        half_RK_reconstruction='PASS',actual_driver_half_step_interrupt_resume_bitwise='PASS',
        automatic_eight_RHS_report_rebuild='PASS',
        budget_arithmetic='PASS',synthetic_RHS=journal.used['RHS']+j.used['RHS'],code_identity=c.local_identity())
    c.dump(root/'TEST.json',result);c.dump(p.HERE/'DT_CPU_ACCEPTANCE.json',dict(result,fixture=str(root.relative_to(p.HERE))))
    print(result)

if __name__=='__main__':main()
