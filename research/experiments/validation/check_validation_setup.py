"""Zero-real-CNN software acceptance before frozen expensive execution."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
import campaign as v
import pc_common as c,pc_engine as e,pc_tests as t,pc_metrics as m
torch.set_num_threads(2)
@torch.inference_mode()
def main():
    root=v.HERE/'CPU_ACCEPTANCE'/str(time.time_ns());root.mkdir(parents=True)
    c.HERE=v.HERE
    x=torch.arange(100,dtype=torch.float64)*2*torch.pi/100;y,x=torch.meshgrid(x,x,indexing='ij')
    w=torch.sin(3*x+2*y)+.2*torch.cos(7*x-4*y)
    s=t.model();w=s.project(w);initial=e.basic_metrics(s,w);j=t.MockJournal();base=e.Engine(s,'Aug',j,root,root/'base',initial)
    w1,_=base.advance(0,w,None);h0=base.orbit(w)['heads'];h1=base.orbit(w1)['heads'];base.close()
    a0=e.save_anchor(root,root/'anchors',0,h0);a1=e.save_anchor(root,root/'anchors',1,h1)
    rows=[]
    # The generalized period=5 adapter must reproduce the historical native adapter.
    for method in ('Parameter-PC','Parameter-Hold'):
        native=e.Engine(t.model(),method,t.MockJournal(),root,root/f'native_{method}',initial);bn=e.Bank([a0,a1]);wn=w1.clone()
        c1,e1,d,E,B=v.bind(5,False);custom=E(t.model(),method,t.MockJournal(),root,root/f'custom_{method}',initial);bc=B([a0,a1]);wc=w1.clone()
        for n in range(1,12):
            wn,_=native.advance(n,wn,bn);wc,_=custom.advance(n,wc,bc)
            assert torch.equal(wn,wc),'Primary period5 numerical behavior changed'
        native.close();custom.close();rows.append(dict(check='period5 exact native parity',method=method,max_difference=0.))
    for period in (4,5,8):
        c1,e1,d,E,B=v.bind(period,False);eng=E(t.model(),'Parameter-Hold',t.MockJournal(),root,root/f'p{period}',initial);b=B([a0,a1]);ww=w1.clone()
        for n in range(1,2*period+2):
            previous=ww.clone();ww,_=eng.advance(n,ww,b)
            m.verify_step(e.cpu(previous),e.cpu(ww),e.cpu(torch.stack(eng.values)),c.operators(eng.s))
            if n%period==0:assert b.pending['step']==n and b.active[-1]['step']==n-period+1
            if n%period==1 and n>1:assert b.pending is None and b.active[-1]['step']==n
        eng.close();rows.append(dict(check='complete-pair promotion and native RK4 reconstruction',period=period,status='PASS'))
    c1,e1,d,E,B=v.bind(5,True);journal=t.MockJournal();eng=E(t.model(),'Parameter-Hold-Single',journal,root,root/'single',initial);b=B([a0,a1]);ww=w1.clone()
    for n in range(1,8):
        eng.prepare(n,ww,b);before=eng.decoder_calls
        hh=b.predict(n,0.,'Parameter-Hold-Single')[0][0]
        ss=eng.s.state(ww);f=c.decoder.decode(eng.s,ss,hh)
        reference=e.D(eng.s,e.vec(f,'qd')+e.vec(f,'qb'))
        got=eng.flux(ss)['Source'];assert torch.equal(reference,got)
        assert eng.decoder_calls-before==1,'Single-Hold decoded more than one orientation'
        # An entirely separate manual four-stage evaluation of the held raw head.
        def rhs(z):
            st=eng.s.state(z);ff=c.decoder.decode(eng.s,st,hh)
            return eng.s.project(-st['u']*st['omega_x']-st['v']*st['omega_y']+eng.s.config.physics.kinematic_viscosity_m2_s*st['laplacian_omega']+e.D(eng.s,e.vec(ff,'qd')+e.vec(ff,'qb')))
        k1=rhs(ww);k2=rhs(eng.s.project(ww+.0025*k1));k3=rhs(eng.s.project(ww+.0025*k2));k4=rhs(eng.s.project(ww+.005*k3))
        target=eng.s.project(ww+.005/6*(k1+2*k2+2*k3+k4))
        # prepare has already stored any refresh; directly invoke native RK4 to avoid duplicate immutable refresh files.
        eng.stage=0;out=eng.s.step(ww,1+n*.005,.005)
        assert e.relative(out,target)<1e-14,'Single-Hold manual trajectory discrepancy'
        ww=out
    eng.close();assert journal.used['CNN']==0
    rows.append(dict(check='single orientation current-state decoding and manual RK4',steps=7,real_CNN=0,status='PASS'))
    from rp_spectral import SpectralRK4,Grid,labels
    cfg=v.read(v.ROOT/'research/ssc_long_horizon_v1_20261030/results/run01/reference/110240101/n1536_half/DONE.json')['config']
    high=SpectralRK4(128,cfg['length'],cfg['nu'],'cpu');low=Grid(100,cfg['length'],'cpu',torch.float64);h=high.initial(110240105,cfg['initial'])
    with np.load(v.ROOT/'research/full_aug_hold_audit_v1_20261117/independent_ic105/INITIAL_FOURIER_COEFFICIENTS.npz') as z:
        md=z['modes'];assert np.linalg.norm(h.numpy()[np.ix_(md%128,md%128)]-z['omega_hat'])/np.linalg.norm(z['omega_hat'])<1e-13
    for _ in range(4):h,_,_=high.step(h,.000125)
    lab=labels(high,low,h,cfg['delta'],cfg['nu'],legacy=False)
    discrepancy=np.linalg.norm(lab['reference_source']-lab['reference_source_rhs_gap'])/np.linalg.norm(lab['reference_source'])
    assert discrepancy<1e-10
    rows.append(dict(check='frozen Fourier initial condition, DNS RK4 and low-grid Source/flux identity',relative_discrepancy=float(discrepancy),n=128,scope='software only; not reference qualification'))
    v.write(v.HERE/'ACCEPTANCE.json',dict(status='PASS',code_inputs=v.identity(),checks=rows,real_CNN=0,low_resolution_test_DNS_RHS=17,expensive_campaign_RHS=0))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
