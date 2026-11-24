"""Manual DNS/LES driver and reference screening for one frozen forced case."""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from forced_physics import (HERE, ROOT, ENGINE, PRODUCTION, Grid, ForcedDNS,
                            checkpoint_rows, closure, common as c, configure_les,
                            embed, fdns_labels, forcing)
from pc_storage import RunLock


PROTOCOL = HERE / "PROTOCOL.json"
DEFAULT_RUN = "forced01_tail_pilot"
ORIGINAL = HERE.parent/"FORCED_01_20261122/runs/forced01"
PILOT_SOURCE = HERE.parent/"FORCED_01_SHORT_16_20_20261122/runs/forced01_short"
PILOT_QUALITY = "DISSIPATION_TAIL_FAILED_EXPLORATORY"
MODELS = {"Clark4": "NGM4", "Aug": "DSF-Aug", "Fresh-C4": "DSF-Sym",
          "Parameter-PC": "DSF-Trend", "Parameter-Hold": "DSF-Hold"}


def index(t, dt):
    i = round(t/dt)
    if abs(i*dt-t) > 1e-10:
        raise ValueError(f"Time {t} is off the locked grid {dt}")
    return i


def sample_path(folder, i):
    return folder / "frames" / f"frame_{i:06d}.npz"


def code_identity():
    helpers = [c.q, c.decoder, c.feasible, c.atomic_io]
    paths = [*HERE.glob("*.py"), PROTOCOL, ENGINE/"rp_spectral.py",
             ROOT/"research/skew_full_validation_20261020/core.py",
             ROOT/"research/skew_full_validation_20261020/reference.py",
             ROOT/"research/full_response_20261020/operators.py",
             *(Path(m.__file__) for m in helpers),
             *(PRODUCTION/f for f in ("pc_common.py", "pc_engine.py", "pc_storage.py", "PROTOCOL.json"))]
    return {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths}


def pinned(path, data):
    if path.exists():
        if c.read(path) != data:
            raise ValueError(f"Existing run identity differs: {path}")
    else:
        c.dump(path, data)


def archive(path, **arrays):
    if path.exists():  # May be an orphan frame from a DNS interruption.
        old = c.load(path)
        if old.keys() != arrays.keys() or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
            raise ValueError(f"An existing archive differs: {path}")
    else:
        c.npz(path, **arrays)
    return c.sha(path)


def environment(device):
    return {"device": device, "torch": str(torch.__version__), "numpy": np.__version__,
            "GPU": torch.cuda.get_device_name(device) if device.startswith("cuda") else None}


def check_protocol(cfg):
    les = cfg["les"]
    if cfg["window"] != {"start": 16, "end": 20} or les["duration"] != 4.0 or les["statistics_skip"] != 0.0:
        raise ValueError("Expected the fixed 16–20 forecast window")
    re = cfg["reynolds"]
    if not (cfg["nu"] > 0 and cfg["alpha"] >= 0 and cfg["beta"] == 0):
        raise ValueError("Expected positive kinematic viscosity, nonnegative friction and beta=0")
    if not math.isclose(re["nominal"], re["velocity_scale"]*re["length_scale"]/cfg["nu"]):
        raise ValueError("Nominal Reynolds number does not match its velocity and length scales")
    if (cfg["forcing_velocity_direction"], cfg["forcing_variation_direction"], cfg["forcing_mode"]) != ("x", "y", 4):
        raise ValueError("Expected the fixed x-directed Kolmogorov force varying in y at mode 4")
    if (les["dt"], les["anchor_spacing"], les["refresh_period"], les["sample_dt"]) != (.005, .005, .025, .05):
        raise ValueError("This adapter requires the frozen production clock and sampling interval")
    if set(les["methods"]) != set(MODELS) or les["seeds"] != [2026, 2028]:
        raise ValueError("Five models and the two original network seeds are required")
    if (les["n"] != 100 or not math.isclose(les["delta"], 4*cfg["length"]/100)
            or not math.isclose(cfg["length"], 2*math.pi)):
        raise ValueError("Expected the original [0,2pi)^2 Gaussian/P33 LES geometry")
    for variant in cfg["dns"].values():
        if variant["n"] < 2*les["n"]:
            raise ValueError("DNS/FDNS grid mismatch")
        index(les["sample_dt"], variant["dt"])


def initialize(out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        identity = {"case": cfg["case"], "protocol": cfg, "inputs": code_identity()}
        pinned(out/"FREEZE.json", identity)
        # Initializer only: generates low Fourier modes without advancing time.
        dns = ForcedDNS(64, cfg, "cpu")
        h = dns.initial(cfg["flow_seed"], cfg["initial"])
        cut = cfg["initial"]["initial_max_mode"]
        modes = np.arange(-cut, cut+1)
        h = closure.cpu(h)[np.ix_(modes % 64, modes % 64)]
        digest = archive(out/"INITIAL_FOURIER.npz", modes=modes, omega_hat=h)
        pinned(out/"INITIAL.json", {"flow_seed": cfg["flow_seed"], "file_sha256": digest,
                                    "freeze_sha256": c.sha(out/"FREEZE.json"), "status": "INITIAL_ONLY"})
        pinned(out/"WINDOW.json", dict(**cfg["window"], selection="fixed_before_evaluation",
                                      freeze_sha256=c.sha(out/"FREEZE.json")))


def frozen(out, cfg):
    expected = {"case": cfg["case"], "protocol": cfg, "inputs": code_identity()}
    if c.read(out/"FREEZE.json") != expected:
        raise ValueError("Run is not initialized or the frozen protocol/dependency changed")
    receipt = c.read(out/"INITIAL.json")
    if (receipt["file_sha256"] != c.sha(out/"INITIAL_FOURIER.npz")
            or receipt["freeze_sha256"] != c.sha(out/"FREEZE.json")):
        raise ValueError("Frozen initial condition integrity failure")
    window = dict(**cfg["window"], selection="fixed_before_evaluation",
                  freeze_sha256=receipt["freeze_sha256"])
    if c.read(out/"WINDOW.json") != window:
        raise ValueError("Frozen evaluation window changed")
    return receipt


def block_records(folder):
    records = []
    for path in sorted((folder/"blocks").glob("*.npz")):
        with np.load(path,allow_pickle=False) as block:
            records.extend(c.meta(block)["observations"])
    return records


def adopt_warmup(out, folder, cfg, solver, low, bar_f, device):
    """Start the short archive from the original run's committed t=16 state."""
    source = ORIGINAL/"DNS/primary"
    original_freeze = c.read(ORIGINAL/"FREEZE.json")
    original_cfg = original_freeze["protocol"]
    physical_keys = ("case", "flow_seed", "length", "nu", "alpha", "beta", "F0", "forcing_mode",
                     "forcing_velocity_direction", "forcing_variation_direction", "initial", "dns")
    if any(original_cfg[k] != cfg[k] for k in physical_keys):
        raise ValueError("Original warmup physics differ")
    if any(original_cfg["les"][k] != cfg["les"][k] for k in ("n", "delta", "sample_dt")):
        raise ValueError("Original warmup filter or cadence differs")
    for name,digest in original_freeze["inputs"].items():
        if c.sha(ROOT/name) != digest:
            raise ValueError(f"Original warmup input SHA changed: {name}")
    original_ic = c.read(ORIGINAL/"INITIAL.json")
    new_ic = c.read(out/"INITIAL.json")
    if (original_ic["freeze_sha256"] != c.sha(ORIGINAL/"FREEZE.json")
            or original_ic["file_sha256"] != c.sha(ORIGINAL/"INITIAL_FOURIER.npz")
            or original_ic["file_sha256"] != new_ic["file_sha256"]):
        raise ValueError("Original warmup initial condition differs")
    config = c.read(source/"RUN.json")
    opts = cfg["dns"]["primary"]
    if (config["variant"] != "primary" or config["initial"] != original_ic
            or any(config[k] != opts[k] for k in opts) or config["environment"] != environment(device)):
        raise ValueError("Original primary grid, time step or device identity differs")
    t = cfg["window"]["start"]
    step = index(t,opts["dt"])
    block_name = f"{step:08d}.npz"
    checkpoint = source/"blocks"/block_name
    if not checkpoint.exists():
        raise ValueError("Original primary has not committed t=16. Finish that segment or run the short case from t=0.")
    digest = c.sha(checkpoint)
    for p in (source/"segments").glob("*.json"):
        recorded = c.read(p)["blocks"].get(block_name)
        if recorded is not None and recorded != digest:
            raise ValueError(f"Original t=16 checkpoint does not match {p.name}")
    original = c.load(checkpoint)
    meta = c.meta(original)
    if meta["step"] != step or original["omega_hat"].shape != (opts["n"],opts["n"]):
        raise ValueError("Original t=16 checkpoint does not match its declared step or grid")
    rows = [r for r in block_records(source) if r["time_s"] <= t+1e-10]
    sample_dt = cfg["les"]["sample_dt"]
    if [index(r["time_s"],sample_dt) for r in rows] != list(range(index(t,sample_dt)+1)):
        raise ValueError("Original warmup diagnostics do not cover 0–16")
    state = torch.as_tensor(original["omega_hat"],device=device)
    solver.max_speed.fill_(meta["running_speed_max"])
    _, spectra = solver.diagnostics(state,opts["dt"])
    lab, err, mean = fdns_labels(solver,low,state,cfg,bar_f)
    rows[-1].update(source_identity_relative=err, source_mean_relative=mean,
                    frame_sha256=archive(sample_path(folder,index(t,sample_dt)), **lab,
                                         time_s=np.array(float(t)),native_spectra=spectra))
    provenance = dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), sha256=digest,
                      original_run_sha256=c.sha(source/"RUN.json"), original_rhs_calls=meta["rhs_calls"])
    c.npz(folder/"blocks"/block_name, omega_hat=original["omega_hat"],
          metadata=c.packed(dict(step=step, observations=rows, rhs_calls=solver.rhs_calls,
                                 integration_seconds=0.0, running_speed_max=float(solver.max_speed),
                                 adopted_warmup=provenance)))


@torch.inference_mode()
def run_dns(out, cfg, variant, until, device, reuse_warmup=False):
    freeze = frozen(out, cfg)
    les, gates = cfg["les"], cfg["gates"]
    opts = cfg["dns"][variant]
    dt = opts["dt"]
    ends = list(cfg["window"].values())
    if until not in ends:
        raise ValueError("DNS segment must end on a registered warmup or evaluation boundary")
    limit = index(until, dt)
    folder = out/"DNS"/variant
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder):
        pinned(folder/"RUN.json", {"variant": variant, **opts, "initial": freeze, "environment": environment(device)})
        if (folder/"FAILED.json").exists():
            raise ValueError("This DNS has a recorded numerical failure")
        s = ForcedDNS(opts["n"], cfg, device)
        low = Grid(les["n"], cfg["length"], device)
        bar_f = closure.cpu(forcing(low, cfg, les["delta"]))
        files = sorted((folder/"blocks").glob("*.npz"))
        if reuse_warmup and variant != "primary":
            raise ValueError("--reuse-warmup applies to the primary grid only")
        if reuse_warmup and not files:
            adopt_warmup(out,folder,cfg,s,low,bar_f,device)
            files = sorted((folder/"blocks").glob("*.npz"))
        first = int(files[0].stem) if files else 0
        if reuse_warmup and first == 0:
            raise ValueError("This short primary already started at t=0. Resume it without --reuse-warmup.")
        if first:
            with np.load(files[0],allow_pickle=False) as source_block:
                if first != index(cfg["window"]["start"],dt) or not c.meta(source_block).get("adopted_warmup"):
                    raise ValueError("DNS prefix identity mismatch")
        begin, sample = 0, index(les["sample_dt"], dt)
        block = index(1, dt)
        elapsed = 0.0
        if files:
            last = c.load(files[-1])
            meta = c.meta(last)
            begin, elapsed = meta["step"], meta["integration_seconds"]
            if [p.stem for p in files] != [f"{n:08d}" for n in range(first, begin+1, block)]:
                raise ValueError("DNS block sequence has gaps")
            s.max_speed.fill_(meta["running_speed_max"])
            s.rhs_calls = meta["rhs_calls"]
            state = torch.as_tensor(last["omega_hat"], device=device)
        else:
            initial = c.load(out/"INITIAL_FOURIER.npz")
            state = embed(s, initial["modes"], initial["omega_hat"])

        old, _ = s.diagnostics(state, dt)
        budget = torch.zeros((3, 2), dtype=torch.float64, device=device)
        observations = []
        attempt = time.time_ns()
        try:
            if not files:
                initial_stats = dict(step=0, time_s=0.0, **old)
                c.npz(folder/"blocks/00000000.npz", omega_hat=closure.cpu(state),
                      metadata=c.packed(dict(step=0, observations=[initial_stats], rhs_calls=0,
                                             running_speed_max=0.0, integration_seconds=0.0)))
            for n in range(begin+1, limit+1):
                start = time.perf_counter() if (n-1)%sample == 0 else start
                state, add = s.step_with_budget(state, dt)
                budget += add
                if n%sample:
                    continue
                c.sync(device)
                elapsed += time.perf_counter()-start
                diag, native = s.diagnostics(state, dt)
                change = np.array([diag["K"]-old["K"], diag["Z"]-old["Z"]])
                rates = closure.cpu(budget)
                defect = abs(change-rates.sum(0))/np.maximum(abs(rates).sum(0), 1e-12)
                if (not all(math.isfinite(v) for v in diag.values()) or not np.isfinite(defect).all()
                        or diag["CFL"] > gates["max_CFL"]):
                    c.dump(folder/"FAILED.json", {"status": "FAILED_NUMERICAL", "attempted_step": n,
                                                   "time_s": n*dt, **diag})
                    raise ValueError("Nonfinite DNS or CFL cap exceeded")
                row = dict(step=n, time_s=n*dt, **diag,
                           budget_defect_relative=defect.tolist(), budget_increment=rates.tolist())
                if n*dt >= cfg["window"]["start"]-1e-10:
                    lab, err, mean = fdns_labels(s, low, state, cfg, bar_f)
                    row["frame_sha256"] = archive(sample_path(folder, n//sample), **lab,
                                                    time_s=np.array(n*dt), native_spectra=native)
                    row.update(source_identity_relative=err, source_mean_relative=mean)
                observations.append(row)
                old, budget = diag, torch.zeros_like(budget)
                if n%block == 0:
                    meta = dict(step=n, observations=observations, rhs_calls=s.rhs_calls,
                                running_speed_max=float(s.max_speed), integration_seconds=elapsed)
                    c.npz(folder/"blocks"/f"{n:08d}.npz", omega_hat=closure.cpu(state), metadata=c.packed(meta))
                    observations = []
                    c.dump(folder/"STATUS.json", dict(status="SEGMENT_RUNNING", completed_t_end=n*dt,
                                                       rhs_calls=s.rhs_calls))
                    print(f"{variant} t={n*dt:.1f} RHS={s.rhs_calls} CFL={diag['CFL']:.4f}", flush=True)
            completed = max(begin, limit)*dt
            segment = folder/"segments"/f"{limit:08d}.json"
            committed = sorted((folder/"blocks").glob("*.npz"))
            pinned(segment, dict(status="SEGMENT_COMPLETE", through=until,
                                 run_sha256=c.sha(folder/"RUN.json"),
                                 blocks={p.name:c.sha(p) for p in committed if int(p.stem) <= limit}))
            c.dump(folder/"STATUS.json", dict(status="SEGMENT_COMPLETE", completed_t_end=completed,
                                               rhs_calls=s.rhs_calls, integration_seconds=elapsed))
        except BaseException as ex:
            c.dump(folder/f"INTERRUPTION_{attempt}.json", dict(error=str(ex), rhs_calls=s.rhs_calls,
                                                               last_committed_step=max(int(p.stem) for p in (folder/"blocks").glob("*.npz"))))
            raise


def screen_rows(rows, start, cfg):
    p = cfg["screen"]
    w = p["block_duration"]
    gates = cfg["gates"]
    dt = cfg["les"]["sample_dt"]
    hit = {index(r["time_s"], dt): r for r in rows}
    end = index(start, dt)
    length = index(w, dt)
    halves = [[hit[i] for i in range(k-length+1, k+1)] for k in (end-length, end)]
    q = np.array([[[(r["K"]), (r["Z"])] for r in half] for half in halves])
    means = q.mean(axis=1)
    drift = abs(means[1]-means[0])/means.mean(axis=0)
    turnovers = np.sqrt(2*q[:,:,1]).sum(axis=1)*dt
    z_cv = q[:,:,1].std(axis=1)/means[:,1]
    off = np.array([[r["off_axis_Z"] for r in half] for half in halves]).mean(axis=1)
    increments = np.array([[r["budget_increment"] for r in half] for half in halves])
    imbalance = abs(increments[:,:,:,0].sum(axis=(1,2)))/np.maximum(abs(increments[:,:,:,0]).sum(axis=(1,2)), 1e-12)
    passed = (max(drift) <= p["max_half_mean_drift"] and min(turnovers) >= p["min_turnovers"]
              and min(z_cv) >= p["min_Z_cv"] and min(off) >= p["min_off_axis_Z_fraction"]
              and max(imbalance) <= p["max_K_budget_imbalance"]
              and max(r["CFL"] for half in halves for r in half) <= gates["max_CFL"])
    return dict(start=start, pass_screen=bool(passed), K_Z_half_drift=drift.tolist(),
                turnovers=turnovers.tolist(), Z_cv=z_cv.tolist(), off_axis_Z=off.tolist(),
                net_over_gross_K_budget=imbalance.tolist())


def describe_window(out, cfg):
    frozen(out, cfg)
    print(c.read(out/"WINDOW.json"))


def reference_archive(out, cfg, variant, window):
    root = out/"DNS"/variant
    if (root/"FAILED.json").exists():
        raise ValueError(f"DNS {variant} has a numerical failure")
    identity = c.read(root/"RUN.json")
    if (identity["variant"] != variant or identity["initial"] != c.read(out/"INITIAL.json")
            or any(identity[k] != v for k,v in cfg["dns"][variant].items())):
        raise ValueError("DNS RUN identity mismatch")
    stop = index(window["end"],identity["dt"])
    segment = c.read(root/"segments"/f"{stop:08d}.json")
    if (segment["through"] != window["end"] or segment["run_sha256"] != c.sha(root/"RUN.json")
            or any(c.sha(root/"blocks"/p) != digest for p,digest in segment["blocks"].items())):
        raise ValueError("DNS block receipt mismatch")
    delta = cfg["les"]["sample_dt"]
    series = block_records(root)
    i0, i1 = (index(window[k], delta) for k in ("start", "end"))
    rows = {index(r["time_s"], delta): r for r in series}
    for i in range(i0, i1+1):
        meta = rows[i]
        frame = sample_path(root, i)
        if c.sha(frame) != meta["frame_sha256"]:
            raise ValueError(f"FDNS hash mismatch: {frame}")
        yield i, meta, c.load(frame)


def field_spectra(w):
    k = np.fft.fftfreq(w.shape[-1], d=1/w.shape[-1]).round()
    k2 = k[:, None]**2+k[None,:]**2
    shell = np.rint(np.sqrt(k2)).astype(int)
    q = .5*abs(np.fft.fft2(w, norm="forward"))**2
    e = np.divide(q, k2, out=np.zeros_like(q), where=k2>0)
    return np.array([np.bincount(shell.ravel(), weights=p.ravel())[1:48] for p in (e, q)])


def weighted_relative(a, b, floor):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), floor))


def reference_receipt(out):
    """Prefer the refinement receipt when both reference evaluations exist."""
    full = out/"REFERENCE_QUALIFICATION.json"
    return full if full.is_file() else out/"REFERENCE_PRIMARY_ONLY.json"


def verify_pilot_input(cfg):
    """Verify all checks except the explicitly unresolved dissipation spectrum."""
    source = PILOT_SOURCE
    ref_file = source/"REFERENCE_PRIMARY_ONLY.json"
    src_freeze = c.read(source/"FREEZE.json")
    if src_freeze["protocol"] != cfg:
        raise ValueError("The reference physics, time window or closure configuration changed")
    for name,digest in src_freeze["inputs"].items():
        if c.sha(ROOT/name) != digest:
            raise ValueError(f"Short-window reference input changed: {name}")
    ref = c.read(ref_file)
    if (ref["status"] != "PRIMARY_REFERENCE_CHECKS_FAILED" or ref["physics_operator_pass"]
            or ref["window"] != c.read(source/"WINDOW.json")
            or ref["window"]["freeze_sha256"] != c.sha(source/"FREEZE.json")):
        raise ValueError("This pilot requires the original failed short-window reference receipt")
    variant = ref["variants"]["primary"]
    g = cfg["gates"]
    comparisons = {"max_CFL":"max_CFL", "max_tail_Z":"max_native_Z_tail",
                   "max_source_identity":"max_source_identity_relative",
                   "max_RK4_budget_defect":"max_budget_defect_relative"}
    if any(not math.isfinite(variant[k]) or variant[k] > g[limit] for k,limit in comparisons.items()):
        raise ValueError("An additional numerical check failed. The dissipation-tail pilot cannot be used.")
    tail = variant["max_tail_dissipation"]
    if not math.isfinite(tail) or tail <= g["max_native_dissipation_tail"]:
        raise ValueError("Original receipt does not document the expected dissipation-tail failure")
    t = cfg["window"]
    expected = {str(i) for i in range(index(t["start"],cfg["les"]["sample_dt"]),
                                      index(t["end"],cfg["les"]["sample_dt"])+1)}
    if set(ref["primary_frames"]) != expected:
        raise ValueError("Expected a complete 81-frame reference archive")
    return ref_file, ref


def qualify(out, cfg, primary_only=False):
    frozen(out, cfg)
    window = c.read(out/"WINDOW.json")
    gate, les = cfg["gates"], cfg["les"]
    floor = gate["normalization_floor"]
    grids, frames, receipt = {}, {}, {}
    variants = ("primary",) if primary_only else ("primary", "space", "time")
    for variant in variants:
        grids[variant] = list(reference_archive(out, cfg, variant, window))
        frames[variant] = {i: f for i, _, f in grids[variant]}
        rows = block_records(out/"DNS"/variant)
        scope = [r for r in rows if r["time_s"] <= window["end"]+1e-10]
        rest = [r for _,r,_ in grids[variant]]
        stats = dict(max_CFL=max(r["CFL"] for r in scope),
                     max_tail_Z=max(r["tail_Z"] for r in rest),
                     max_tail_dissipation=max(r["tail_dissipation"] for r in rest),
                     max_source_identity=max(max(r["source_identity_relative"],r["source_mean_relative"]) for r in rest),
                     max_RK4_budget_defect=max(max(r.get("budget_defect_relative", [0.])) for r in scope),
                     warmup=screen_rows(rows, window["start"], cfg),
                     evaluation=screen_rows(rows, window["end"], cfg))
        stats["physical_operator_pass"] = bool(
            stats["max_CFL"] <= gate["max_CFL"] and stats["max_tail_Z"] <= gate["max_native_Z_tail"]
            and stats["max_tail_dissipation"] <= gate["max_native_dissipation_tail"]
            and stats["max_source_identity"] <= gate["max_source_identity_relative"]
            and stats["max_RK4_budget_defect"] <= gate["max_budget_defect_relative"])
        receipt[variant] = stats
    i0 = index(window["start"], les["sample_dt"])
    short = index(les["forecast_duration"], les["sample_dt"])
    stat = 1 + index(les["statistics_skip"], les["sample_dt"])
    base = frames["primary"]
    k = np.fft.fftfreq(les["n"], d=1/les["n"]).round()
    shell = np.rint(np.hypot(k[:,None], k[None,:])).astype(int)
    bands = dict(all=(1,47), low=(1,16), mid=(17,29), high=(30,47))
    def spectrum_window(fs):
        return np.mean([field_spectra(f["reference_omega"]) for i,f in fs.items() if i >= i0+stat], axis=0)
    base_spec = spectrum_window(base)
    checks = {}
    for name in variants[1:]:
        others = frames[name]
        fields = {}
        for key in ("reference_omega","reference_source"):
            fft = [(np.fft.fft2(others[i][key],norm="forward"),
                    np.fft.fft2(f[key],norm="forward")) for i,f in base.items()]
            for band, (lo,hi) in bands.items():
                mask = (shell >= lo) & (shell <= hi)
                err = [weighted_relative(a[mask], b[mask], floor) for a,b in fft]
                fields[f"{key}_{band}"] = dict(start=err[0], forecast_max=max(err[:short+1]), full_max=max(err))
        comp = spectrum_window(others)
        mean_spectrum = abs(comp-base_spec).sum(1)/np.maximum(base_spec.sum(1), floor)
        means = abs(comp.sum(1)-base_spec.sum(1))/np.maximum(base_spec.sum(1),floor)
        checks[name] = dict(fields=fields, mean_K_Z_relative=means.tolist(),
                            mean_E_Z_spectrum_L1_relative=mean_spectrum.tolist(),
                            short_match=all(v["forecast_max"] <= gate["legacy_field_relative"] for v in fields.values()),
                            strict_full_match=all(v["full_max"] <= gate["legacy_field_relative"] for v in fields.values()),
                            mean_match=bool(max(*mean_spectrum, *means) <= gate["max_stat_relative"]))
    physics = all(r["physical_operator_pass"] for r in receipt.values())
    sampled = all(r["warmup"]["pass_screen"] and r["evaluation"]["pass_screen"] for r in receipt.values())
    short_pass = None if primary_only else physics and all(r["short_match"] for r in checks.values())
    means_pass = None if primary_only else physics and all(r["mean_match"] for r in checks.values())
    strict = None if primary_only else physics and all(r["strict_full_match"] for r in checks.values())
    filename = "REFERENCE_PRIMARY_ONLY.json" if primary_only else "REFERENCE_QUALIFICATION.json"
    status = ("PRIMARY_REFERENCE_ONLY" if physics else "PRIMARY_REFERENCE_CHECKS_FAILED") if primary_only else (
        "FINITE_GRID_CHECKS_PASS" if strict else "REFERENCE_NOT_CONVERGED")
    pinned(out/filename, dict(
        status=status, window_kind="fixed_transient_forecast", strict_full_match=strict,
        window=window, physics_operator_pass=physics, statistics_screen_pass=sampled,
        short_phase_checks_pass=short_pass, statistical_reference_checks_pass=means_pass,
        variants=receipt, comparisons=checks,
        primary_frames={str(i):row["frame_sha256"] for i,row,_ in grids["primary"]},
        variant_frames={tag:{str(i):row["frame_sha256"] for i,row,_ in records}
                        for tag,records in grids.items()}))
    print(f"{filename}: {status}, physics_operator_pass={physics}, short_phase_checks_pass={short_pass}, "
          f"finite_window_mean_checks_pass={means_pass}, activity_screen_pass={sampled}")


class Counter:
    """Count complete decodes via actual forwards plus algebraic decoder calls."""
    def __init__(self):
        self.calls = dict(CNN=0, RHS=0, ALGEBRAIC_DECODE=0)
        self.phase = ""

    def tick(self, kind):
        self.calls[kind] += 1

    def note(self, kind):
        self.tick(kind)

    def summary(self):
        return {**self.calls, "full_decode": self.calls["CNN"]+self.calls["ALGEBRAIC_DECODE"]}


def start_neural(out, cfg, row, w, grid, ref_key):
    folder = out/"LES"/"startup"/str(row["seed"])
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder):
        file = folder/"STARTUP.npz"
        if not file.exists():
            counter = Counter()
            s = c.model(row, str(w.device))
            configure_les(s, cfg, grid)
            e = closure.Engine(s, "Aug", counter, out, folder, closure.basic_metrics(s, w))
            wall = time.perf_counter()
            try:
                w1, _ = e.advance(0,w,None)
                first = {k: e.snapshots[0][k] for k in ("Source", "qx", "qy", "rhs")}
                anchors = [closure.save_anchor(out, folder/"anchors", n, e.orbit(v)["heads"])
                           for n,v in enumerate((w,w1))]
                c.sync(w.device)
                meta = dict(reference=ref_key, bank=closure.Bank(anchors).descriptor(),
                            costs=counter.summary(), seconds=time.perf_counter()-wall)
                c.npz(file, omega1=closure.cpu(w1), metadata=c.packed(meta), **first)
            except Exception as ex:
                c.dump(folder/f"FAILURE_{time.time_ns()}.json", dict(error=str(ex), calls=counter.summary()))
                raise
            finally:
                e.close()
        z = c.load(file)
        if c.meta(z)["reference"] != ref_key:
            raise ValueError("Shared startup reference changed")
        return z


def clark_solver(cfg, grid):
    seed = cfg["les"]["seeds"][0]
    s = c.q.study.make_solver(c.q.study.AUGMENTED, seed, device=str(grid.device)).double().eval()
    bar_f = configure_les(s, cfg, grid)
    def forbid(*_):
        raise ValueError("NGM4 attempted a CNN forward")
    s.closure.register_forward_pre_hook(forbid)
    def flux(st):
        qx,qy = c.q.study.common.prior.c4.flux4(s, st, s.closure.filter_width)
        return dict(qx=qx,qy=qy)
    s.closure_fields = flux
    return s, bar_f


@torch.inference_mode()
def run_les(out, cfg, method, seed, device, allow_unqualified, attempt):
    frozen(out, cfg)
    if (method == "Clark4" and seed is not None) or (method != "Clark4" and seed not in cfg["les"]["seeds"]):
        raise ValueError("Specify a frozen network seed for DSF only")
    if not allow_unqualified:
        raise ValueError("Explicit --allow-dissipation-tail-fail required for this separate pilot")
    ref_file, ref = verify_pilot_input(cfg)
    if c.read(PILOT_SOURCE/"INITIAL.json")["file_sha256"] != c.read(out/"INITIAL.json")["file_sha256"]:
        raise ValueError("Pilot Fourier initial condition differs from the reference")
    pinned(out/"PILOT_REFERENCE.json", dict(
        status=PILOT_QUALITY, physics_operator_pass=False,
        failed_metric="max_native_dissipation_tail",
        value=ref["variants"]["primary"]["max_tail_dissipation"],
        threshold=cfg["gates"]["max_native_dissipation_tail"],
        source=ref_file.relative_to(ROOT).as_posix(), sha256=c.sha(ref_file),
        window=ref["window"]))
    les = cfg["les"]
    t0 = ref["window"]["start"]
    initial = index(t0, les["sample_dt"])
    init_file = sample_path(PILOT_SOURCE/"DNS/primary",initial)
    sha = ref["primary_frames"][str(initial)]
    if c.sha(init_file) != sha:
        raise ValueError("FDNS initial frame SHA mismatch")
    key = dict(initial=sha, t0=t0, window_end=ref["window"]["end"])
    folder = out/"LES"/f"{MODELS[method]}_{seed or 'analytic'}"/attempt
    folder.mkdir(parents=True, exist_ok=True)
    with RunLock(folder):
        if (folder/"RUN.json").exists():
            raise ValueError("Existing LES attempt retained. Choose a new --attempt for a deliberate rerun.")
        rows = checkpoint_rows(cfg)
        record = dict(model=MODELS[method], network_seed=seed, method=method, reference=key,
                      reference_receipt=ref_file.name, reference_receipt_sha256=c.sha(ref_file),
                      reference_quality=PILOT_QUALITY,
                      checkpoint_sha256=rows[seed]["sha256"] if seed else None,
                      h=les["dt"], runtime=environment(device), exploratory=bool(allow_unqualified))
        pinned(folder/"RUN.json", record)
        counter = Counter()
        actual, diagnostic, hashes = 0.0, 0.0, {}
        max_sampled_cfl = 0.0
        end = index(les["duration"], les["dt"])
        stride = index(les["sample_dt"], les["dt"])
        grid = Grid(les["n"],cfg["length"],device)
        w0 = torch.as_tensor(c.load(init_file)["reference_omega"], device=device)
        e, bank, startup = None, None, None
        loop_start = time.perf_counter()
        n = 0

        def save_frame(step, state, snap):
            nonlocal max_sampled_cfl
            t = t0+step*les["dt"]
            elapsed = step*les["dt"]
            fields = {k: snap[k] for k in ("Source","qx","qy","rhs")}
            arr = closure.cpu(state)
            resolved = grid.state(state)
            speed = float((resolved["u"].abs()+resolved["v"].abs()).max())
            cfl = les["dt"] * (speed/(grid.length/grid.n)+cfg["nu"]*2*grid.cut**2+cfg["alpha"])
            max_sampled_cfl = max(max_sampled_cfl,cfl)
            a = dict(omega=arr, forcing=closure.cpu(bar_f), drag=-cfg["alpha"]*arr,
                     time_s=np.array(t), lead_time=np.array(elapsed), CFL=np.array(cfl), **fields)
            name = sample_path(folder,step//stride)
            hashes[str(step//stride)] = archive(name, **a)
            if step % index(1.,les["dt"]) == 0:
                c.dump(folder/"STATUS.json",dict(status="RUNNING",step=step,time_s=t,
                                                 costs=counter.summary()))
                print(f"{record['model']} seed={seed} t={t:.1f} calls={counter.summary()}",flush=True)

        try:
            w = w0
            if seed is None:
                s,bar_f = clark_solver(cfg,grid)
                native = s.rhs
                inv = closure.basic_metrics(s,w)
                latest = {}
                def rhs(x,t):
                    counter.tick("RHS")
                    f = native(x,t)
                    stats = closure.basic_metrics(s,x)
                    if (not bool(torch.isfinite(f["rhs"]).all())
                            or any(stats[k] > 100*inv[k] for k in ("K","Z"))):
                        raise closure.TrajectoryFailure("FAILED_NUMERICAL","NGM4 stage safety bound")
                    if not latest:
                        latest.update(f)
                    return f
                s.rhs = rhs
                for n in range(end):
                    latest.clear()
                    old = w
                    c.sync(device)
                    start = time.perf_counter()
                    w = s.step(w, t0+n*les["dt"], les["dt"])
                    c.sync(device)
                    actual += time.perf_counter()-start
                    if n%stride == 0:
                        snap = {k:closure.cpu(latest[k]) for k in ("qx","qy","rhs")}
                        save_frame(n,old,dict(snap,Source=closure.cpu(-latest["div_q"])))
                before_diagnostic = counter.summary()
                start = time.perf_counter()
                f = s.rhs(w,t0+les["duration"])
                c.sync(device)
                diagnostic = time.perf_counter()-start
                snap = {k:closure.cpu(f[k]) for k in ("qx","qy","rhs")}
                save_frame(end,w,dict(snap,Source=closure.cpu(-f["div_q"])))
            else:
                s = c.model(rows[seed],device)
                weights_before = c.q.study.common.model_hash(s)
                bar_f = configure_les(s,cfg,grid)
                startup = start_neural(out,cfg,rows[seed],w,grid,key)
                # Production Engine clock is labeled 1+n*h internally. The
                # physical clock is t0+n*h and the forcing is time-independent.
                save_frame(0,w,startup)
                if method.startswith("Parameter"):
                    bank = closure.load_bank(out,c.meta(startup)["bank"],device)
                w = torch.as_tensor(startup["omega1"],device=device)
                e = closure.Engine(s,method,counter,out,folder,closure.basic_metrics(s,w0))
                for n in range(1,end):
                    old = w
                    w,timings = e.advance(n,w,bank)
                    actual += timings["solver_wall_seconds"]
                    if n%stride == 0:
                        save_frame(n,old,e.snapshots[0])
                c.sync(device)
                before_diagnostic = counter.summary()
                start = time.perf_counter()
                final = e.endpoint(end,w,bank)
                c.sync(device)
                diagnostic = time.perf_counter()-start
                save_frame(end,w,final)
                if c.q.study.common.model_hash(s) != weights_before:
                    raise ValueError("Model weights changed")
            if len(hashes) != end//stride+1:
                raise ValueError("Missing LES archive frame")
            c.dump(folder/"DONE.json",dict(status="COMPLETE", frames=hashes,
                   completed_t_end=t0+les["duration"], run_sha256=c.sha(folder/"RUN.json"),
                   costs=counter.summary(), startup=c.meta(startup) if startup else None,
                   endpoint_calls={k:v-before_diagnostic[k] for k,v in counter.summary().items()},
                   solver_seconds=actual, endpoint_seconds=diagnostic,
                   max_sampled_CFL=max_sampled_cfl,
                   complete_session_seconds=time.perf_counter()-loop_start,
                   resources=c.resource_memory(device)))
            c.dump(folder/"STATUS.json",dict(status="COMPLETE",step=end,time_s=t0+les["duration"]))
        except BaseException as ex:
            status = ex.status if isinstance(ex,closure.TrajectoryFailure) else "INTERRUPTED_OR_RUNTIME_FAILURE"
            c.dump(folder/"FAILED.json",dict(status=status,error=str(ex),last_attempted_step=n,
                 frames=hashes,costs=counter.summary(),solver_seconds=actual))
            raise
        finally:
            if e is not None:
                e.close()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight","init"):
        sub.add_parser(action)
    l = sub.add_parser("les")
    l.add_argument("--model",choices=MODELS,required=True)
    l.add_argument("--seed",type=int)
    l.add_argument("--attempt",default="primary")
    l.add_argument("--allow-dissipation-tail-fail",action="store_true")
    l.add_argument("--execute",action="store_true")
    l.add_argument("--device",default="cuda",choices=("cpu","cuda"))
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run) or not re.fullmatch(r"[a-zA-Z0-9_-]+", getattr(args,"attempt","ok")):
        parser.error("Run and attempt must be single safe directory names")
    c.HERE = HERE
    cfg = c.read(PROTOCOL)
    check_protocol(cfg)
    out = HERE/"runs"/args.run
    if args.action in ("preflight","init"):
        rows = checkpoint_rows(cfg)
        if args.action == "init":
            initialize(out,cfg)
        print(json.dumps(dict(protocol=cfg,checkpoints={k:v["file"] for k,v in rows.items()},
                              status="PASS_STATIC_ONLY" if args.action=="preflight" else "INITIAL_ONLY"),
                         ensure_ascii=False,indent=2))
    else:
        if not args.execute:
            parser.error("Manual integration requires --execute")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable on this interpreter/device")
        run_les(out,cfg,args.model,args.seed,args.device,args.allow_dissipation_tail_fail,args.attempt)


if __name__ == "__main__":
    cli()
