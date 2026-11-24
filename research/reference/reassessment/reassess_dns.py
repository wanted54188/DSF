"""Read-only postprocessing of the frozen DSF references and LES snapshots."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESEARCH = ROOT / "research"
JCP = RESEARCH / "JCP_DSF_TemporalSym_20261117"
VALID = JCP / "VALIDATION_20261118"
PRODUCTION = RESEARCH / "full_aug_parameter_pc_full_v1_20261116"
OLD = RESEARCH / "ssc_long_horizon_v1_20261030/results/run01"
OTHER = RESEARCH / "full_c4_multi_ic_v1_20261110/results/paired01/reference/110240104"
IC105 = VALID / "DNS/110240105_n1536_h0.00012500"
CASES = (110240101, 110240102, 110240104, 110240105)
VARIANTS = ("n1024_main", "n1536_main", "n1536_half")
COMPARISONS = (("space", 0, 1), ("time", 1, 2), ("space_and_time", 0, 2))
METHODS = dict(zip(("Aug", "Fresh-C4", "Parameter-PC", "Parameter-Hold"),
                   ("DSF-Aug", "DSF-Sym", "DSF-Trend", "DSF-Hold")))
SEEDS = (2026, 2028)
STEPS = np.arange(0, 801, 10)
TIMES = 1 + .005 * STEPS
FIELDS = ("reference_omega", "reference_source", "reference_dot", "reference_qx", "reference_qy")

sys.path.insert(0, str(PRODUCTION))
from pc_metrics import F, I, geometry  # original normalized FFT and bands


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def rms(a, axis=(-2, -1)):
    return np.sqrt(np.mean(a*a, axis=axis))


def spatial_cc(a, b):
    a, b = a-a.mean(), b-b.mean()
    den = rms(a)*rms(b)
    return float(np.mean(a*b)/den) if den > 0 else None


def archive(path, expected=None):
    if expected is not None:
        assert sha(path) == expected, f"SHA mismatch: {path}"
    return np.load(path, allow_pickle=False)


def initial_coefficients(n, case, cfg):
    """Reconstruct only the prescribed t=0 modes using the archived generator."""
    rng = np.random.default_rng(case)
    cut = cfg["initial_max_mode"]
    h = np.zeros((n, n), complex)
    variance = 0.
    for ky in range(-cut, cut+1):
        for kx in range(cut+1):
            r = math.hypot(kx, ky)
            if (kx == 0 and ky <= 0) or not 2 <= r <= cut:
                continue
            amp = float(rng.uniform(.5, 1.5)*r*np.exp(-(r/cfg["initial_peak_mode"])**4))
            z = amp*np.exp(1j*rng.uniform(0, 2*math.pi)) / 2
            h[ky % n, kx % n] = z
            h[-ky % n, -kx % n] = z.conjugate()
            variance += amp*amp/2
    modes = np.arange(-cut, cut+1)
    recovered = F(I(h) * cfg["initial_RMS"]/math.sqrt(variance))
    return recovered[np.ix_(modes % n, modes % n)]


def checked_data(values, step_count=81):
    for value in values.values():
        assert value.shape == (step_count, 100, 100) and np.isfinite(value).all()
    return values


def dense_reference(case, tag, inventory):
    folder = OLD / "reference" / str(case) / tag
    receipt = folder / "DONE.json"
    done = read_json(receipt)
    idx = (STEPS*4).astype(int)
    assert done["seed"] == case and done["frames"] == 3201
    times = archive(folder / "time_s.npy", done["outputs"]["time_s.npy"])
    np.testing.assert_allclose(times[idx], TIMES, atol=1e-11, rtol=0)
    values = {}
    for field in FIELDS:
        p = folder / (field+".npy")
        assert sha(p) == done["outputs"][p.name]
        values[field] = np.load(p, mmap_mode="r", allow_pickle=False)[idx]
    diag = folder / "RAW_DIAGNOSTICS.csv"
    assert sha(diag) == done["outputs"][diag.name]
    original = Path(done["continuation_from"]).parent
    config = read_json(original / "FREEZE.json")
    assert config["seed"] == case
    inventory.append(dict(case=case, role="training" if case == CASES[0] else "development",
                          kind="DNS", reference=tag, source=rel(folder), receipt=rel(receipt),
                          receipt_sha256=sha(receipt), execution_status=done["status"],
                          original_quality="REFERENCE_NOT_CONVERGED", grid=done["n"], dt=done["dt"],
                          original_prefix=rel(original), samples=len(times)))
    return checked_data(values), done, folder


def snapshot_reference(case, inventory):
    folder = OTHER if case == CASES[2] else IC105
    receipt = folder / "DONE.json"
    done = read_json(receipt)
    names = [f"frame_{(8000+40*n if folder == OTHER else n):0{5 if folder == OTHER else 4}d}.npz"
             for n in STEPS]
    values = {k: [] for k in FIELDS}
    for t, name in zip(TIMES, names):
        with archive(folder / name, done["frames"][name]) as z:
            assert abs(float(z["time_s"])-t) < 1e-10
            for k in FIELDS:
                values[k].append(z[k])
    inventory.append(dict(case=case, role="development" if folder == OTHER else "independent_candidate",
                          kind="DNS", reference="n1536_half", source=rel(folder), receipt=rel(receipt),
                          receipt_sha256=sha(receipt), execution_status=done["status"],
                          original_quality="REFERENCE_NOT_CONVERGED", grid=1536, dt=.000125,
                          samples=len(done["frames"])))
    return checked_data({k: np.stack(v) for k,v in values.items()}), done, folder


def load_les(case, report, manifest, inventory):
    run = PRODUCTION / "results/run01"
    checkpoints = {c["seed"]:c for c in read_json(run/"RUN.json")["inventory"]["checkpoints"]}
    result = {}
    for seed, method in itertools.product(SEEDS, METHODS):
        p = report / f"STATES_{case}_{seed}_{method}.npy"
        q = run / "trajectories" / f"{case}_{seed}_{method}"
        status = read_json(q / "STATUS.json")
        assert status["status"] == "VERIFIED" and status["last_accepted_step"] == 800
        h = manifest["outputs"][p.name]
        assert sha(p) == h
        states = np.load(p, mmap_mode="r", allow_pickle=False)
        assert states.shape == (801, 100, 100)
        for step in (10, 400, 800):
            with archive(q / "steps" / f"{step:04d}.npz") as z:
                np.testing.assert_array_equal(z["omega"], states[step])
                assert json.loads(str(z["metadata"]))["accepted_step"] == step
        model = METHODS[method]
        checkpoint = checkpoints[seed]
        result[model, seed] = states[STEPS]
        inventory.append(dict(case=case, kind="LES", model=model, seed=seed, source=rel(p),
                              source_sha256=h, receipt=rel(q / "STATUS.json"),
                              receipt_sha256=sha(q / "STATUS.json"), execution_status=status["status"],
                              samples=80, dt=.005, original_quality=status["reference_quality"],
                              checkpoint_sha256=checkpoint["sha256"], checkpoint=checkpoint["file"],
                              physical_envelope=status.get("physical_envelope")))
    q = VALID / "CLARK" / str(case) / "limited_completion/Clark4_0.005"
    status = read_json(q / "STATUS.json")
    assert status["status"] == "VERIFIED" and status["last_accepted_step"] == 800
    with archive(q / "INITIAL.npz") as z:
        states = [z["omega"]]
    checkpoint = read_json(q / "CHECKPOINT.json")
    assert checkpoint["accepted_step"] == 800
    for step in STEPS[1:]:
        expected = checkpoint["sha256"] if step == 800 else None
        with archive(q / "steps" / f"{step:04d}.npz", expected) as z:
            meta = json.loads(str(z["metadata"]))
            assert meta["accepted_step"] == step and meta["h"] == .005
            states.append(z["omega"])
    result["Clark4", None] = np.stack(states)
    inventory.append(dict(case=case, kind="LES", model="Clark4", source=rel(q / "steps"),
                          receipt=rel(q / "STATUS.json"), receipt_sha256=sha(q / "STATUS.json"),
                          execution_status=status["status"], samples=80, dt=.005,
                          original_quality=status["reference_quality"],checkpoint=rel(q/"CHECKPOINT.json"),
                          source_sha256=checkpoint["sha256"],
                          physical_envelope=status.get("physical_envelope")))
    checked_data(result)
    return result


def shell_sum(x, shell):
    return np.bincount(shell.ravel(), weights=x.ravel())


def transfers(ref, op):
    w, S = ref["reference_omega"], ref["reference_source"]
    # Original reference operators use exact integer modes on [0,2pi)^2.
    # The LES production operator archive is used separately for trajectory scoring.
    k = np.fft.fftfreq(100,d=.01).round()
    kx,ky = k[None,:],k[:,None]
    k2 = kx*kx+ky*ky
    inv = np.divide(1,k2,out=np.zeros_like(k2),where=k2!=0)
    h = F(w)*op["dealias"]
    psi = h*inv
    gx,gy = I(1j*kx*h),I(1j*ky*h)
    u,v = I(1j*ky*psi),I(-1j*kx*psi)
    pi = -ref["reference_qx"]*gx - ref["reference_qy"]*gy
    means = dict(TK=np.mean(I(psi)*S, axis=(-2,-1)), TZ=np.mean(w*S, axis=(-2,-1)))
    flux = I(-(1j*kx*F(ref["reference_qx"])+1j*ky*F(ref["reference_qy"]))*op["dealias"])
    resolved = F(-u*gx-v*gy + 1e-6*I(-k2*h))*op["dealias"]
    resolved[:,0,0]=0
    adv_visc = I(resolved)
    error = dict(flux_identity=rms(flux-S)/rms(S), RHS_identity=rms(ref["reference_dot"]-adv_visc-S)/rms(S),
                 source_mean_fraction=abs(S.mean(axis=(-2,-1)))/rms(S),
                 Pi_Z_integral_residual=np.abs(pi.mean(axis=(-2,-1))+means["TZ"]))
    return pi, {**means, **error}


def f1(pred, obs):
    tp = (pred & obs).sum()
    return float(2*tp/(pred.sum()+obs.sum())) if pred.any() or obs.any() else None


def filtered(case, refs, op, masks, shell, out):
    details, spectra, summary = [], [], []
    pi_data = {tag: transfers(a,op) for tag,a in refs.items()}
    h = {tag: F(a["reference_omega"]) for tag,a in refs.items()}
    k2 = np.fft.fftfreq(100,d=.01).round()**2
    k2 = k2[:,None] + k2[None,:]
    inv = np.divide(1,k2,out=np.zeros_like(k2),where=k2!=0)
    for tag, v in h.items():
        p = abs(v)**2
        zz = .5*p.sum(axis=(-2,-1))
        kk = .5*(p*inv).sum(axis=(-2,-1))
        pi, diag = pi_data[tag]
        power = (v.conj()*F(refs[tag]["reference_source"])).real
        for j,t in enumerate(TIMES):
            details.append(dict(case=case, reference=tag, time=t, K=kk[j], Z=zz[j],
                                forward_magnitude=np.maximum(pi[j],0).mean(),
                                backscatter_magnitude=np.maximum(-pi[j],0).mean(),
                                **{k: val[j] for k,val in diag.items()}))
            if j in (0,40,80):
                data = [shell_sum(x,shell) for x in (.5*p[j]*inv, .5*p[j], power[j]*inv,power[j])]
                for k in range(1,48):
                    spectra.append(dict(case=case, reference=tag, time=t, k=k,
                                        **dict(zip(("E","Z","TK","TZ"), (a[k] for a in data)))))
    for axis,i,j in COMPARISONS[:2]:
        a,b = (VARIANTS[i], VARIANTS[j])
        if a not in refs or b not in refs:
            continue
        ctx = dict(case=case, axis=axis, reference_a=a, reference_b=b, samples=81,
                   original_quality="REFERENCE_NOT_CONVERGED")
        for f in (*FIELDS[:3], "Pi_Z", "flux", "TK", "TZ"):
            if f in ("TK","TZ"):
                va,vb = pi_data[a][1][f],pi_data[b][1][f]
                summary.append(dict(**ctx, metric=f, band="integrated",
                                    rms_absolute=np.sqrt(np.mean((va-vb)**2)),
                                    mean_a=va.mean(), mean_b=vb.mean(), endpoint_a=va[-1], endpoint_b=vb[-1]))
                continue
            va,vb = (pi_data[a][0],pi_data[b][0]) if f == "Pi_Z" else (refs[a].get(f),refs[b].get(f))
            if f == "flux":
                va,vb = (np.stack([refs[tag][k] for k in FIELDS[3:]],axis=1) for tag in (a,b))
            d, y = F(va-vb),F(vb)
            bandmasks = masks if f in FIELDS[:3] else {"all_native_LES_grid": np.ones((100,100), bool)}
            for band, mask in bandmasks.items():
                dr, yr = (abs(z[..., mask])**2 for z in (d,y))
                while dr.ndim > 2:
                    dr,yr = dr.sum(axis=1),yr.sum(axis=1)
                x, z = dr.sum(axis=1), yr.sum(axis=1)
                r = np.sqrt(x/np.maximum(z,1e-60))
                entry = dict(**ctx, metric=f, band=band, modes=int(mask.sum()),
                             pooled_relative=np.sqrt(x.sum()/z.sum()), worst_relative=max(r),
                             worst_time=TIMES[np.argmax(r)], endpoint_relative=r[-1],
                             rms_absolute=np.sqrt(x.mean()))
                if f in ("reference_source","Pi_Z") and band in ("all","all_native_LES_grid"):
                    entry["mean_spatial_CC"] = np.mean([spatial_cc(x,y) for x,y in zip(va,vb)])
                    if f == "Pi_Z":
                        entry["forward_F1"] = f1(va>0,vb>0)
                        entry["backscatter_F1"] = f1(va<0,vb<0)
                summary.append(entry)
                out["filtered_difference"].extend(dict(**ctx, metric=f, band=band,time=t,relative=y)
                                                   for t,y in zip(TIMES,r))
        xa, xb = ([r for r in details if r["reference"] == tag] for tag in (a,b))
        for k in ("K","Z","forward_magnitude","backscatter_magnitude"):
            va,vb = (np.array([row[k] for row in rows]) for rows in (xa,xb))
            rr=abs(va-vb)/abs(vb)
            summary.append(dict(**ctx,metric=k,band="integrated",pooled_relative=np.linalg.norm(va-vb)/np.linalg.norm(vb),
                                worst_relative=max(rr),worst_time=TIMES[np.argmax(rr)],endpoint_relative=rr[-1]))
        for t in (1.,3.,5.):
            sa,sb=([r for r in spectra if r["time"]==t and r["reference"]==tag] for tag in (a,b))
            for metric in ("E","Z","TK","TZ"):
                for band,first,last in (("all",1,47),("low",1,16),("mid",17,29),("high",30,47)):
                    x,y=(np.array([r[metric] for r in s if first<=r["k"]<=last]) for s in (sa,sb))
                    summary.append(dict(case=case,axis=axis+"_spectrum",reference_a=a,reference_b=b,time=t,
                        samples=1,metric=metric,band=band,shell_first=first,shell_last=last,
                        relative_shell_L1=np.sum(abs(x-y))/np.sum(abs(y))))
    for tag in refs:
        for k in ("flux_identity","RHS_identity","source_mean_fraction","Pi_Z_integral_residual"):
            series = pi_data[tag][1][k]
            summary.append(dict(case=case,axis="operator",reference_a=tag, metric=k,band="all",
                                samples=81,maximum=max(series),worst_time=TIMES[np.argmax(series)]))
    out["filtered_values"].extend(details)
    out["filtered_spectra"].extend(spectra)
    out["dns_diagnostics"].extend(summary)
    return pi_data


def score_les(case, states, refs, op, mask, out):
    ref_fields = {tag: r["reference_omega"][1:] for tag,r in refs.items()}
    errors = {}
    info = {}
    for tag, r in ref_fields.items():
        for (model,seed), w in states.items():
            d = w[1:] - r
            # Same retained Fourier norm as pc_metrics.state_rows.
            hd = F(d)
            lost = float(np.max(np.sum(abs(hd[:,~mask])**2, axis=1)))
            # Scores use P33; independently compare with the physical grid MSE.
            mse = float(np.mean(np.sum(abs(hd[:,mask])**2,axis=1)))
            np.testing.assert_allclose(mse, np.mean(d*d), atol=2e-12, rtol=2e-6)
            errors[tag,model,seed] = mse
            hd *= mask
            hu,hv = hd*op["ky"]*op["inverse"], hd*op["kx"]*op["inverse"]
            info[tag,model,seed] = dict(case=case,model=model,network_seed=seed,reference=tag,
                                       samples=80,t_first=TIMES[1],t_last=TIMES[-1],LES_dt=.005,
                                       omega_MSE=mse,u_MSE=np.mean(np.sum(abs(hu)**2,axis=(-2,-1))),
                                       v_MSE=np.mean(np.sum(abs(hv)**2,axis=(-2,-1))),
                                       t5_omega_relative_L2=np.linalg.norm(hd[-1])/np.linalg.norm(F(r[-1])*mask),
                                       max_error_outside_P33=lost, original_quality="REFERENCE_NOT_CONVERGED")
    for seed in SEEDS:
        keys = [("Clark4", None), *((m, seed) for m in METHODS.values())]
        by_ref = {}
        for tag in refs:
            sorted_keys = sorted(keys,key=lambda k:errors[(tag,*k)])
            by_ref[tag] = sorted_keys
            for rank,k in enumerate(sorted_keys,1):
                d = info[(tag,*k)]
                if k[1] is None:
                    d[f"rank_in_seed_{seed}_comparison"] = rank
                    d["improvement_vs_Clark4_pct"] = 0.
                    for base,col in (("DSF-Aug","Aug"),("DSF-Sym","Sym")):
                        d[f"improvement_vs_{col}_pct_seed_{seed}"] = 100*(1-d["omega_MSE"]/errors[tag,base,seed])
                else:
                    for basem, bases, col in (("Clark4",None,"Clark4"),("DSF-Aug",seed,"Aug"),("DSF-Sym",seed,"Sym")):
                        base = errors[tag,basem,bases]
                        d[f"improvement_vs_{col}_pct"] = 100*(base-d["omega_MSE"])/base
                    d["rank"] = rank
        if len(refs) < 2:
            continue
        for axis,i,j in COMPARISONS:
            a,b=VARIANTS[i],VARIANTS[j]
            dr = I(F(ref_fields[b]-ref_fields[a])*mask)
            for A,B in itertools.combinations(reversed(keys),2):
                dw = I(F(states[A][1:]-states[B][1:])*mask)
                da,db = (errors[(tag,*A)]-errors[(tag,*B)] for tag in (a,b))
                shift = 2*np.mean(dw*dr)
                tight = 2*np.mean(rms(dw)*rms(dr))
                pooled = 2*np.sqrt(np.mean(dw**2)*np.mean(dr**2))
                np.testing.assert_allclose(da-db,shift,rtol=0,atol=2e-12)
                assert abs(da-db) <= tight+2e-12 and tight <= pooled+2e-12
                out["pair_bounds"].append(dict(case=case,network_seed=seed, axis=axis,
                    model_A=A[0],model_B=B[0],reference_a=a,reference_b=b,samples=80,
                    D_a=da,D_b=db,D_a_minus_D_b=da-db,drift_inner_product=shift,
                    tighter_mean_framewise_bound=tight,pooled_bound=pooled,
                    no_reversal=np.sign(da)==np.sign(db), bound_smaller_than_abs_D_a=pooled<abs(da)))
        velocity = {}
        for metric in ("u_MSE","v_MSE"):
            for tag in refs:
                this_order=sorted(keys,key=lambda k:info[(tag,*k)][metric])
                velocity[f"{tag}_{metric}_order"] = " < ".join(k[0] for k in this_order)
        out["ordering"].append(dict(case=case,network_seed=seed,**velocity,
            **{tag:" < ".join(k[0] for k in ordered) for tag,ordered in by_ref.items()},
            all_three_orderings_equal=len({tuple(v) for v in by_ref.values()})==1))
    out["ranking"].extend(info.values())


def native(case, refs, out, times):
    coeffs = {}
    for tag,(filtered_fields,done,folder) in refs.items():
        n = done.get("n", done.get("grid", 1536))
        if case in CASES[:2]:
            diag = read_csv(folder / "RAW_DIAGNOSTICS.csv")
        else:
            diag = done["observations"]
        for k in ("tail_fraction","CFL","E","Z","E_balance_relative","Z_balance_relative"):
            samples = [(float(r["time_s"]) if "time_s" in r else r["step"]*.000125,float(r[k]))
                       for r in diag if r.get(k) not in (None,"")]
            if samples:
                t,v=max(samples,key=lambda p:p[1])
                out["dns_diagnostics"].append(dict(case=case,axis="native_archive",reference_a=tag,metric=k,
                    samples=len(samples),maximum=v,worst_time=t,endpoint_a=samples[-1][1],
                    source=rel(folder/("RAW_DIAGNOSTICS.csv" if case in CASES[:2] else "DONE.json"))))
        if case == CASES[2]:
            continue
        for t in times:
            if case == CASES[3]:
                name = f"native_{round((t-1)/.005):04d}.npz"
                with archive(folder/name,done["native"][name]) as z:
                    hh, m = z["h"],z["modes"]
                    assert abs(float(z["time_s"])-t)<1e-10
            else:
                d = next(r for r in done["native"] if abs(r["time_s"]-t)<1e-10)
                v = archive(folder/d["file"],d["sha256"])
                cut = (n-1)//3
                m = np.arange(-cut,cut+1)
                assert m[0] == -m[-1]
                hh = F(v)[np.ix_(m%n,m%n)]
                del v
            r2 = m[:,None]**2+m[None,:]**2
            shell = np.rint(np.sqrt(r2)).astype(int)
            pow = abs(hh)**2
            zs=.5*pow
            es=np.divide(zs,r2,out=np.zeros_like(zs),where=r2>0)
            visc=1e-6*r2*pow
            tail = np.maximum(abs(m[:,None]),abs(m[None,:])) >= .8*m[-1]
            filtered_z=.5*np.mean(filtered_fields["reference_omega"][round((t-1)/.05)]**2)
            row = dict(case=case,axis="native_FFT",time=t,reference_a=tag,samples=1,metric="native_resolved_Z",
                       K_native=es.sum(),Z_native=zs.sum(),Z_dissipation=visc.sum(),
                       Z_FDNS_over_native=filtered_z/zs.sum(),
                       tail_Z_fraction=zs[tail].sum()/zs.sum(),tail_dissipation_fraction=visc[tail].sum()/visc.sum())
            out["dns_diagnostics"].append(row)
            sh = [shell_sum(a,shell) for a in (es,zs,visc)]
            out["native_spectra"].extend(dict(case=case,reference=tag,time=t,k=k,E=sh[0][k],
                                             Z=sh[1][k],viscous_Z_dissipation=sh[2][k])
                                        for k in range(1,len(sh[0])))
            if t in (1.,3.,5.):
                coeffs[tag,t] = hh
    if case not in CASES[:2]:
        return
    source = OLD/"quality"/str(case)/"NATIVE_CONVERGENCE.csv"
    archived = read_csv(source)
    for axis,i,j in COMPARISONS[:2]:
        a,b=VARIANTS[i],VARIANTS[j]
        records = [row for row in archived if row["axis"]==axis]
        for t in (1.,3.,5.):
            x,y=coeffs[a,t],coeffs[b,t]
            offset=(len(y)-len(x))//2
            common=y[offset:offset+len(x),offset:offset+len(x)]
            outside = 0.
            if offset:
                other = np.ones(y.shape,dtype=bool)
                other[offset:offset+len(x),offset:offset+len(x)] = False
                outside = np.sum(abs(y[other])**2)
            value=np.sqrt((np.sum(abs(x-common)**2)+outside)/np.sum(abs(y)**2))
            orig=next(float(row["native_omega_relative"]) for row in records if abs(float(row["time_s"])-t)<1e-10)
            # Compare squared norms: the legacy total-minus-common calculation
            # can create a relative RMS floor of order sqrt(machine epsilon).
            np.testing.assert_allclose(value**2,orig**2,atol=2e-15,rtol=1e-8)
            out["dns_diagnostics"].append(dict(case=case,axis="native_"+axis,reference_a=a,reference_b=b,
                time=t,metric="native_omega_FFT_recheck",pooled_relative=value,original_value=orig))
        for k in ("native_omega_relative","native_E_relative","native_Z_relative","fine_discarded_power_fraction"):
            row=max(records,key=lambda r:float(r[k]))
            out["dns_diagnostics"].append(dict(case=case,axis="native_"+axis,reference_a=a,reference_b=b,
                metric=k,samples=len(records),maximum=float(row[k]),worst_time=float(row["time_s"]),
                endpoint_relative=float(records[-1][k]),source=rel(source)))


def replay_checks(case, refs, pis, out, replay):
    if case not in CASES[:2]:
        return
    with archive(replay/"FIELD_EXAMPLES.npz") as z:
        for step in (10,400,800):
            index=step//10
            for key in ("Source","Pi_Z"):
                pred=z[f"case{case}_step{step}_{key}"]
                truth={tag:(a["reference_source"][index] if key=="Source" else pis[tag][0][index])
                       for tag,a in refs.items()}
                np.testing.assert_allclose(pred[0],truth["n1536_half"],atol=3e-6,rtol=8e-7)
                for tag, q in truth.items():
                    for name,y in zip(z["models"][1:],pred[1:]):
                        out["fixed_replay"].append(dict(case=case,time=TIMES[index],metric=key,
                            model=str(name), label_reference=tag,prediction_input="n1536_half",
                            network_seed=2026,MSE=np.mean((y-q)**2),spatial_CC=spatial_cc(y,q),
                            forward_F1=f1(y>0,q>0) if key=="Pi_Z" else None,
                            backscatter_F1=f1(y<0,q<0) if key=="Pi_Z" else None))


def main(output, all_native, replay):
    run=PRODUCTION/"results/run01"
    pointer=read_json(run/"LATEST_REPORT.json")
    report=run/pointer["path"]
    assert sha(report/"MANIFEST.json")==pointer["manifest_sha256"]
    manifest=read_json(report/"MANIFEST.json")
    op=dict(archive(run/"OPERATORS.npz"))
    masks,shell=geometry(op)
    active=masks["all"]
    k=np.fft.fftfreq(100,d=.01).round().astype(int)
    expected=(abs(k[:,None])<=33)&(abs(k[None,:])<=33)
    expected[0,0]=False
    np.testing.assert_array_equal(active,expected)
    masks["legacy_high"]=active&(shell>=17)&(shell<=71)
    inventory=[]
    files=("filtered_values","filtered_spectra","filtered_difference","dns_diagnostics","native_spectra",
           "ranking","pair_bounds","ordering","fixed_replay","les_statistics","les_pdf","legacy_gates")
    out={f:[] for f in files}
    qual=read_json(VALID/"LATEST_QUALIFICATION.json")
    original=VALID/qual["path"]/"QUALIFICATION.json"
    assert sha(original)==qual["sha256"]
    for q in read_json(original)["cases"]:
        out["legacy_gates"].append(dict(case=q["case"],protocol="frozen_3072",status=q["status"],
                 failed_gates="|".join(k for k,v in q["gates"].items() if not v),
                 missing="|".join(q["missing_variants"]),source=rel(original)))
    for case in CASES:
        print(f"Reading archived flow {case}",flush=True)
        if case in CASES[:2]:
            bundles={tag:dense_reference(case,tag,inventory) for tag in VARIANTS}
            cfg=[b[1]["config"] for b in bundles.values()]
            assert all(c==cfg[0] for c in cfg)
            a,b=(initial_coefficients(n,case,cfg[0]["initial"]) for n in (1024,1536))
            diff=np.linalg.norm(a-b)/np.linalg.norm(b)
            assert diff<1e-13
            out["dns_diagnostics"].append(dict(case=case,axis="static_initial",metric="common_initial_Fourier_relative",
                band="abs(kx),abs(ky)<=12",reference_a=VARIANTS[0],reference_b=VARIANTS[1],pooled_relative=diff,
                source="analytic reconstruction from frozen seed/config (no advance)"))
            quality=OLD/"quality"/str(case)
            q=read_json(quality/"QUALITY.json")
            out["legacy_gates"].append(dict(case=case,protocol="original_1024_1536",status=q["status"],
                failed_gates="|".join(k for k,v in q["gates"].items() if not v),source=rel(quality/"QUALITY.json")))
            out["dns_diagnostics"].extend(dict(case=case,axis="legacy_"+r["axis"],metric=r["field"],band=r["band"],
                samples=3201,pooled_relative=float(r["relative_RMS"]),worst_relative=float(r["max_frame_relative_RMS"]),
                worst_time=float(r["peak_time_s"]),original_threshold=float(r["threshold"]),original_pass=r["passes"],
                source=rel(quality/"CONVERGENCE.csv")) for r in read_csv(quality/"CONVERGENCE.csv"))
        else:
            bundles={VARIANTS[-1]:snapshot_reference(case,inventory)}
        refs={tag:x[0] for tag,x in bundles.items()}
        pis=filtered(case,refs,op,masks,shell,out)
        native(case,bundles,out,tuple(TIMES) if all_native else (1.,3.,5.))
        replay_checks(case,refs,pis,out,replay)
        if case != CASES[-1]:
            states=load_les(case,report,manifest,inventory)
            score_les(case,states,refs,op,active,out)
            native_reference=refs[VARIANTS[-1]]["reference_omega"]
            for (name,seed),w in {**states,("FDNS",None):native_reference}.items():
                h=F(w)*active
                p=abs(h)**2
                for j,t in enumerate(TIMES):
                    out["les_statistics"].append(dict(case=case,model=name,seed=seed,time=t,
                        K=.5*np.sum(p[j]*op["inverse"]),Z=.5*p[j].sum(),CC=spatial_cc(w[j],native_reference[j]),
                        maximum_omega=abs(w[j]).max()))
                # Same normalization and bins for all models, fixed by FDNS samples.
                sigma=rms(native_reference[1:])
                bins=np.linspace(-8,8,129)
                normalized=w[1:]/sigma[:,None,None]
                count,_=np.histogram(normalized,bins)
                for left,right,value in zip(bins[:-1],bins[1:],count):
                    out["les_pdf"].append(dict(case=case,model=name,seed=seed,left=left,right=right,
                        density=value/normalized.size/(right-left),outside_fraction=1-count.sum()/normalized.size))
    output.mkdir(parents=True,exist_ok=True)
    special={"ranking":"REFERENCE_RANKING_ROBUSTNESS","dns_diagnostics":"DNS_REFERENCE_DIAGNOSTICS"}
    for k, rows in out.items():
        write_csv(output/(special.get(k,k.upper())+".csv"), rows)
    write_csv(output/"SOURCE_INVENTORY.csv", inventory)
    receipt=read_json(replay/"RECEIPT.json")
    assert receipt["status"]=="COMPLETE" and receipt["reference_tag"]=="n1536_half" and receipt["new_RHS_calls"]==0
    replay_rows=read_csv(replay/"STATE_METRICS.csv")
    output_replay=[]
    for (case,model), rr in itertools.groupby(sorted(replay_rows,key=lambda r:(r["case"],r["model"])),
                                              key=lambda r:(r["case"],r["model"])):
        rows=list(rr)
        assert [int(r["step"]) for r in rows]==list(range(1,801))
        metrics=("Source_MSE_all","Pi_Z_MSE","Source_CC","Pi_Z_CC","Pi_Z_positive_ref_CC",
                 "Pi_Z_negative_ref_CC","net_TK","net_TZ")
        output_replay.append(dict(case=case,model=model,samples=len(rows),
                             **{k:np.mean([float(r[k]) for r in rows]) for k in metrics}))
    write_csv(output/"APRIORI_ARCHIVED_REPLAY.csv",output_replay)
    (output/"REASSESSMENT_RECEIPT.json").write_text(json.dumps(dict(
        DNS_steps=0,LES_steps=0,CNN_calls=0,new_model_evaluations=0,
        output_rows={k:len(v) for k,v in out.items()}, mask_modes=int(active.sum()),
        band_modes={k:int(v.sum()) for k,v in masks.items()}, max_shell=int(shell[active].max()),
        max_radius=float(np.sqrt(2)*33), native_fft_times=81 if all_native else 3,
        prior_replay_source=replay.name, prior_replay_sha256={p.name:sha(p) for p in replay.iterdir() if p.is_file()},
        source_production_manifest=rel(report/"MANIFEST.json")
    ),ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--replay", type=Path, required=True, help="External FDNS_REPLAY archive")
    parser.add_argument("--all-native",action="store_true",help="Manual opt-in: transform every archived native frame")
    args=parser.parse_args()
    main(args.output.resolve(),args.all_native,args.replay.resolve())
