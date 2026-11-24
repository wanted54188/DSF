"""CPU operator checks. Only manufactured single RK4 steps are advanced."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import forced_run as run
from forced_physics import ForcedDNS, Grid, closure, configure_les, embed, fdns_labels, forcing, rms_reynolds
from forced_run import PROTOCOL, Counter, c, check_protocol, checkpoint_rows, clark_solver, field_spectra, index, screen_rows
from forced_evaluate import long_statistics, scores, transfer


CFG = c.read(PROTOCOL)
torch.set_num_threads(2)


def close(a,b,atol=2e-12):
    torch.testing.assert_close(a,b,atol=atol,rtol=2e-13)


def test_reynolds_number_and_beta_convention():
    check_protocol(CFG)
    assert CFG["beta"] == 0
    assert CFG["reynolds"]["nominal"] == 20000
    observed = rms_reynolds(.5,CFG)
    assert observed["Re_rms_unit_length"] == pytest.approx(20000)
    assert observed["Re_rms_forcing_length"] == pytest.approx(5000)
    with pytest.raises(ValueError,match="beta=0"):
        check_protocol(dict(CFG,beta=1))


def test_force_sign_modes_filter_and_curl():
    high = Grid(128)
    low = Grid(100)
    f = forcing(high,CFG)
    g = high.fft(f)
    truth = torch.zeros_like(g)
    truth[[4,-4],0] = -.25
    close(g,truth)
    delta = CFG["les"]["delta"]
    f_bar = low.project(high.transfer(high.ifft(g*torch.exp(-delta**2*high.k2/24)),low))
    close(f_bar,forcing(low,CFG,delta))
    y = torch.arange(high.n,dtype=high.dtype)*high.length/high.n
    fx = (CFG["F0"]*torch.sin(4*y))[:,None].expand_as(f)
    close(-high.grad(fx)[1],f)
    assert int(low.cut) == 33 and not low.mask[0,0]


def test_drag_and_one_native_step_budget():
    s = ForcedDNS(64,CFG,"cpu")
    g = s.grid
    h = .8*s.f_hat
    rhs,z,k = s.rhs(h)
    rate = CFG["alpha"]+CFG["nu"]*16
    close(rhs,s.f_hat-rate*h)
    assert float(z) == pytest.approx(16*float(k))
    dt = CFG["dns"]["primary"]["dt"]
    out, integral = s.step_with_budget(h,dt)
    coeff = .8*np.exp(-rate*dt)+(1-np.exp(-rate*dt))/rate
    close(out,coeff*s.f_hat)
    now,_ = s.diagnostics(out,dt)
    old,_ = s.diagnostics(h,dt)
    assert float(integral[2,0]) < 0 and float(integral[2,1]) < 0
    np.testing.assert_allclose(closure.cpu(integral).sum(0),
                               [now[x]-old[x] for x in ("K","Z")],rtol=1e-9,atol=2e-15)


def test_fdns_two_source_paths_and_parseval():
    s = ForcedDNS(128,CFG,"cpu")
    g = s.grid
    low = Grid(100)
    random = torch.Generator().manual_seed(123)
    h = g.fft(torch.randn((128,128),generator=random,dtype=g.dtype))*g.mask
    bar = closure.cpu(forcing(low,CFG,CFG["les"]["delta"]))
    field, err, mean = fdns_labels(s,low,h,CFG,bar)
    assert max(err,mean) < 1e-11
    z,src = field["reference_omega"],field["reference_source"]
    omega = torch.as_tensor(z)
    st = low.state(omega)
    close(low.grad(st["v"])[0]-low.grad(st["u"])[1],omega)
    src_gap = field["reference_dot"]-closure.cpu(low.rhs(omega,CFG["nu"]))-bar+CFG["alpha"]*z
    np.testing.assert_allclose(src_gap,src,atol=2e-12,rtol=0)
    budget = transfer(z,src,field["reference_qx"],field["reference_qy"])
    assert budget["T_Z"] == pytest.approx(-budget["forward_Z"]+budget["backscatter_Z"],abs=1e-12)
    assert field_spectra(z)[1].sum() == pytest.approx(.5*np.mean(z*z))
    assert scores(z,z)["CC"] == pytest.approx(1)


def test_frozen_coefficients_embed_at_identical_modes():
    s = ForcedDNS(64,CFG,"cpu")
    g = ForcedDNS(128,CFG,"cpu")
    n = CFG["initial"]["initial_max_mode"]
    modes = np.arange(-n,n+1)
    h = closure.cpu(s.initial(CFG["flow_seed"],CFG["initial"]))[np.ix_(modes%64,modes%64)]
    broad = closure.cpu(embed(g,modes,h))
    np.testing.assert_array_equal(broad[np.ix_(modes%128,modes%128)],h)
    assert np.linalg.norm(broad) == pytest.approx(CFG["initial"]["initial_RMS"])


def test_production_les_rhs_uses_filtered_force_and_drag():
    grid = Grid(100)
    s,bar = clark_solver(CFG,grid)
    random = torch.Generator().manual_seed(42)
    w = grid.project(torch.randn((100,100),generator=random,dtype=grid.dtype))
    f = s.rhs(w,21.0)
    adv = -s.project(f["u"]*f["omega_x"]+f["v"]*f["omega_y"])
    terms = adv+CFG["nu"]*f["laplacian_omega"]+bar-CFG["alpha"]*w-f["div_q"]
    close(f["rhs"],terms)
    assert s.config.physics.kinematic_viscosity_m2_s == CFG["nu"]
    s2 = c.q.study.make_solver(c.q.study.AUGMENTED,2026).double()
    assert s2.config.physics.kinematic_viscosity_m2_s == pytest.approx(1e-6)
    close(configure_les(s2,CFG,grid),bar)
    close(s2.vorticity_forcing(40.0,w),bar-CFG["alpha"]*w)


def test_frozen_aug_rhs_on_manufactured_state(tmp_path):
    grid = Grid(100)
    s = c.model(checkpoint_rows(CFG)[2026])
    bar = configure_les(s,CFG,grid)
    y = torch.arange(grid.n,dtype=grid.dtype)*grid.length/grid.n
    w = grid.project(3*torch.cos(3*y[:,None])*torch.sin(5*y[None,:])
                     + torch.cos(2*y[:,None]+y[None,:]))
    counter = Counter()
    e = closure.Engine(s,"Aug",counter,tmp_path,tmp_path,closure.basic_metrics(s,w))
    with torch.inference_mode():
        try:
            e.prepare(0,w,None)
            field = s.rhs(w,1.)
            assert e.records[0]["rhs_reconstruction_relative"] < 1e-12
            close(s.vorticity_forcing(1.,w),bar-CFG["alpha"]*w)
            close(field["Source"],-field["div_q"])
        finally:
            e.close()
    assert counter.summary() == dict(CNN=1,RHS=1,ALGEBRAIC_DECODE=1,full_decode=2)


@pytest.mark.parametrize("mode",["Parameter-Hold","Parameter-PC"])
def test_history_pair_activation_and_frozen_stage_cache(monkeypatch,tmp_path,mode):
    def anchor(_root,_path,n,heads):
        return {"step":n,"heads":heads}
    monkeypatch.setattr(closure,"save_anchor",anchor)
    monkeypatch.setattr(closure,"basic_metrics",lambda *_:dict(K=1.,Z=1.))
    bank = closure.Bank([anchor(None,None,0,torch.tensor(0.)),anchor(None,None,1,torch.tensor(1.))])
    e = closure.Engine.__new__(closure.Engine)
    e.root,e.folder,e.method,e.j = tmp_path,tmp_path,mode,Counter()
    e.initial = dict(K=1.,Z=1.)
    e.diagnostic_seconds = e.network_seconds = 0.0
    e.orbit = lambda w:dict(original={},heads=w)
    observed = []
    def staged(w,t,h):
        assert t == pytest.approx(1+e.n*.005) and h == .005
        for fraction in (0,.5,.5,1):
            observed.append(bank.predict(e.n,fraction,mode))
            e.stage += 1
        return w
    e.s = SimpleNamespace(step=staged)
    e.advance(5,torch.tensor(5.),bank)
    assert bank.pending["step"] == 5 and bank.active[-1]["step"] == 1
    assert [v[1] for v in observed] == [4,4.5,4.5,5]
    assert float(observed[-1][0]) == (1 if mode=="Parameter-Hold" else 6)
    observed.clear()
    e.advance(6,torch.tensor(6.),bank)
    assert bank.pending is None and [r["step"] for r in bank.active] == [5,6]
    assert float(observed[-1][0]) == (6 if mode=="Parameter-Hold" else 7)


def test_warmup_screen_rejects_shear_and_accepts_unsteady_fixture():
    dt = CFG["les"]["sample_dt"]
    times = np.arange(1,index(20,dt)+1)*dt
    rows = [dict(time_s=float(t), K=1.,Z=8.+.3*np.sin(2*np.pi*t),
                 off_axis_Z=.6,CFL=.1,budget_increment=[[.1,0],[-.05,0],[-.05,0]]) for t in times]
    assert screen_rows(rows,20,CFG)["pass_screen"]
    for r in rows:
        r["off_axis_Z"] = 0
    assert not screen_rows(rows,20,CFG)["pass_screen"]


def test_time_averaged_spectrum_and_pdf_keep_amplitude_and_mass():
    y = np.arange(100)*2*np.pi/100
    w = np.cos(4*y[:,None])+np.zeros((100,100))
    z = np.zeros_like(w)
    archive = [dict(time_s=t,omega=a*w,Source=z,qx=z,qy=z) for t,a in ((0,1),(1,2))]
    edges = np.linspace(-1.5,1.5,40)
    stats,spec,pdf = long_statistics(archive,0,{"raw":edges,"reference_scaled":edges},1.)
    # Z = amplitude squared / 4 on this manufactured shear.
    assert stats["Z"] == pytest.approx(.625)
    assert spec[1,3] == pytest.approx(.625)
    assert stats["K"] == pytest.approx(.625/16)
    for density,below,above in pdf.values():
        assert below>0 and above>0
        assert np.dot(density,np.diff(edges))+below+above == pytest.approx(1.)


def test_primary_receipt_stays_separate_from_refinement_receipt(monkeypatch,tmp_path):
    monkeypatch.setattr(c,"HERE",tmp_path)
    monkeypatch.setattr(run,"frozen",lambda *_: None)
    c.dump(tmp_path/"WINDOW.json",CFG["window"])
    first,last = CFG["window"]["start"], CFG["window"]["end"]
    dt = CFG["les"]["sample_dt"]
    times = np.arange(index(last,dt)+1)*dt
    rows = [dict(time_s=t,K=1.,Z=8+.3*np.sin(t),off_axis_Z=.6,CFL=.1,
                 budget_increment=[[.1,0],[-.05,0],[-.05,0]],
                 tail_Z=0,tail_dissipation=0,source_identity_relative=0.,source_mean_relative=0.,
                 frame_sha256=str(i)) for i,t in enumerate(times)]
    y = np.arange(100)*2*np.pi/100
    field = np.cos(4*y[:,None])+np.zeros((100,100))
    def archive(_out,_cfg,_variant,_window):
        for i in range(index(first,dt),index(last,dt)+1):
            yield i,rows[i],dict(reference_omega=field,reference_source=field)
    monkeypatch.setattr(run,"reference_archive",archive)
    monkeypatch.setattr(run,"block_records",lambda *_:rows)
    run.qualify(tmp_path,CFG,primary_only=True)
    pilot_path = run.reference_receipt(tmp_path)
    pilot = c.read(pilot_path)
    assert pilot_path.name == "REFERENCE_PRIMARY_ONLY.json"
    assert pilot["status"] == "PRIMARY_REFERENCE_ONLY"
    assert pilot["strict_full_match"] is None
    assert len(pilot["primary_frames"]) == 81
    run.qualify(tmp_path,CFG)
    full_path = run.reference_receipt(tmp_path)
    full = c.read(full_path)
    assert full_path.name == "REFERENCE_QUALIFICATION.json"
    assert full["strict_full_match"] is True
    assert pilot["primary_frames"] == full["primary_frames"]
    assert c.read(pilot_path) == pilot
