"""CPU-only integration/restart tests. A real CNN forward is forbidden."""
import json,time
import numpy as np
import torch
import limited_dt as l
import pc_tests as fixture
c,e,d,p=l.c,l.e,l.d,l.p

@torch.inference_mode()
def main():
    torch.set_num_threads(2);p.bind()
    root=p.HERE/'cpu_runs'/('limited_'+str(time.time_ns()));root.mkdir(parents=True)
    x=torch.arange(100,dtype=torch.float64)*2*torch.pi/100;y,x=torch.meshgrid(x,x,indexing='ij')
    s=fixture.model();w=s.project(torch.sin(3*x+2*y)+.2*torch.cos(7*x-4*y));initial=e.basic_metrics(s,w)
    j=fixture.MockJournal();warm=p.FineEngine(s,'Aug',j,root,root/'warm',initial)
    w1,_=warm.advance(0,w,None);a=warm.orbit(w);b=warm.orbit(w1)
    a0=e.save_anchor(root,root/'anchors',0,a['heads']);a1=e.save_anchor(root,root/'anchors',1,b['heads']);desc=e.Bank([a0,a1]).descriptor()
    start=dict(omega0=e.cpu(w),omega1=e.cpu(w1),rhs=e.cpu(torch.stack(warm.values)),metadata=c.packed(dict(bank=desc,initial=initial)))
    warm.close();continuous={}
    for method in l.METHODS:
        ss=fixture.model();bank=e.load_bank(root,desc,'cpu');eng=p.FineEngine(ss,method,j,root,root/'expected',initial);z=w1.clone()
        for n in range(1,7):
            before=z.clone();z,_=eng.advance(n,z,bank);p.verify(e.cpu(before),e.cpu(z),e.cpu(torch.stack(eng.values)),c.operators(ss))
            if n==5:assert bank.active[-1]['step']==1 and eng.records[-1]['tau']==5
            if n==6:assert bank.active[-1]['step']==6
        continuous[method]=e.cpu(z);eng.close()
    original_model=c.model;original_engine=d.Engine
    journal=c.Journal(root/'invocations')
    class Interrupt(p.FineEngine):
        def rhs(self,w,t):
            if self.n==3 and self.half_index==1 and self.stage==1 and not flag[0]:flag[0]=True;raise KeyboardInterrupt('synthetic interruption')
            return super().rhs(w,t)
        def advance(self,n,w,bank):
            if n==7:raise e.TrajectoryFailure('FAILED_NUMERICAL','synthetic test end',dict(omega=e.cpu(w)))
            return super().advance(n,w,bank)
    try:
        c.model=lambda *a,**k:fixture.model();d.Engine=Interrupt
        for method in l.METHODS:
            flag=[False]
            try:d.run_trajectory(root,l.CASE,l.SEED,method,{},start,journal,'cpu')
            except KeyboardInterrupt:pass
            else:raise AssertionError('missing interruption')
            d.run_trajectory(root,l.CASE,l.SEED,method,{},start,journal,'cpu')
            folder=root/'trajectories'/c.key(l.CASE,l.SEED,method)
            assert np.array_equal(c.load(folder/'steps/0007.npz')['omega'],continuous[method])
            before=dict(journal.used);d.run_trajectory(root,l.CASE,l.SEED,method,{},start,journal,'cpu');assert before==journal.used
    finally:c.model=original_model;d.Engine=original_engine;journal.close()
    assert journal.used['CNN']==0
    # Pure Clark at both dt: real production FFT/RK4; no CNN permitted.
    old_solver=l.clark_solver;l.clark_solver=lambda device:old_solver(device)
    clark=[]
    try:
        for h in (.005,.0025):
            aroot=root/f'continuous_{h}';aroot.mkdir();broot=root/f'resumed_{h}';broot.mkdir()
            jj=fixture.MockJournal();l.run_clark(aroot,start,h,jj,'cpu',stop=8)
            class Break(fixture.MockJournal):
                def tick(self,k):
                    if self.used['RHS']==20:raise KeyboardInterrupt('Clark uncommitted step')
                    super().tick(k)
            try:l.run_clark(broot,start,h,Break(),'cpu',stop=8)
            except KeyboardInterrupt:pass
            else:raise AssertionError('Clark interruption missing')
            l.run_clark(broot,start,h,fixture.MockJournal(),'cpu',stop=8)
            folder=f'limited_completion/Clark4_{h}/steps/0008.npz'
            assert np.array_equal(c.load(aroot/folder)['omega'],c.load(broot/folder)['omega'])
            clark.append(dict(dt=h,bitwise_resume='PASS',CNN=jj.used['CNN']))
            assert jj.used['CNN']==0
    finally:l.clark_solver=old_solver
    # Scheduler guards reject broad/unrelated scopes before model evaluation.
    jj=l.LimitedJournal(root/'scope_test');jj.scope=c.key(110240102,2026,'Parameter-Hold')
    try:
        try:jj.tick('CNN')
        except RuntimeError as err:assert 'unrelated' in str(err)
        else:raise AssertionError('unrelated case scheduled')
        jj.scope=c.key(l.CASE,l.SEED,'Aug')
        try:jj.tick('CNN')
        except RuntimeError:pass
        else:raise AssertionError('baseline rerun permitted')
    finally:jj.close()
    result=dict(status='PASS',code_identity=l.fingerprint(),real_CNN=0,
        Trend_Hold_eight_stage_resume='BITWISE_PASS',physical_refresh_pair_tau5='PASS',
        clark=clark,unrelated_case_and_baseline_guard='PASS',fixture=str(root.relative_to(c.ROOT)),
        real_long_LES='NOT_RUN')
    (l.HERE/'LIMITED_DT_ACCEPTANCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
