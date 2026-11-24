"""Freeze paired checkpoints and run a budgeted staged RK4 comparison."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAIN = ROOT / "research/full_c4_equivariant_training_v1_20261109"
sys.path.insert(0, str(TRAIN))
import study

read, sha, require = study.inputs.read, study.inputs.sha, study.inputs.require
BASELINE = study.inputs.RUN
CASES = study.CASES
SEEDS = tuple(study.P["seeds"])
C4, BASE, AUG, EQ, ORBIT = "Clark4", "FULL-C4", study.AUGMENTED, study.EQUIVARIANT, "FULL-C4-rotation-mean"
MODELS = (BASE, AUG, EQ)
DT = .005
FRAMES = (0, 10, 30, 50, 70)
CAPS = {"one_step": dict(cnn=240, rhs=280, clark4=40),
        "short": dict(cnn=432, rhs=504, clark4=72),
        "extension": dict(cnn=1440, rhs=1680, clark4=240)}
SNAPSHOT = HERE / "frozen"
OLD = ROOT / "research/full_c4_covariant_rk4_pair_v1_20261107/results/run01"


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def csvout(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def path_name(path):
    return str(path.relative_to(ROOT))


def freeze():
    audit, registry = study.verify_baseline()
    report_dir = TRAIN / "results/reports/paired01/report01"
    report = read(report_dir / "MANIFEST.json")
    run = TRAIN / "results/training/paired01"
    receipt = read(run / "STUDY.json")
    require(report["status"] == "PASS" and report["run"] == "paired01"
            and receipt["code_sha256"] == study.code_identity()
            and receipt["baseline_code_sha256"] == audit["code_sha256"]
            and receipt["cache_sha256"] == registry["cache_sha256"], "Paired training identity")
    rows = []
    for mode in MODELS:
        for seed in SEEDS:
            folder = (BASELINE if mode == BASE else run) / "training" / f"{mode}_{seed}"
            ckpt, meta, done = (folder / name for name in ("best.pt", "RUN.json", "DONE.json"))
            for path in (ckpt, meta, done):
                require(sha(path) == report["input_sha256"][path_name(path)], "Input differs from training report")
            record, identity = read(done), read(meta)
            require(record["status"] == "TRAINING_COMPLETE" and record["optimizer_updates"] == 2100
                    and record["best_sha256"] == sha(ckpt) and identity["cache_sha256"] == registry["cache_sha256"],
                    "Checkpoint completion identity")
            target = SNAPSHOT / f"{mode}_{seed}.pt" if mode != BASE else ckpt
            rows.append(dict(mode=mode, seed=seed, epoch=record["best_epoch"], sha256=record["best_sha256"],
                             original=path_name(ckpt), file=path_name(target), run=path_name(meta),
                             run_sha256=sha(meta), done=path_name(done), done_sha256=sha(done)))
    SNAPSHOT.mkdir(exist_ok=False)
    for row in rows:
        if row["mode"] != BASE:
            path = ROOT / row["file"]
            shutil.copyfile(ROOT / row["original"], path)
            require(sha(path) == row["sha256"], "Snapshot checksum")
    dump(SNAPSHOT / "CHECKPOINTS.json", dict(
        status="FROZEN", quality=study.P["reference_quality"], source_run="paired01",
        selection="Original best checkpoint by 102 development Source NMSE",
        training_code_sha256=receipt["code_sha256"], baseline_code_sha256=audit["code_sha256"],
        baseline_config_sha256=sha(study.common.CONFIG), cache_sha256=registry["cache_sha256"],
        training_report_sha256=sha(report_dir / "MANIFEST.json"), checkpoints=rows,
    ))
    print(SNAPSHOT / "CHECKPOINTS.json", flush=True)


class Budget:
    def __init__(self):
        self.phase = "one_step"
        self.used = {phase: dict.fromkeys(cap, 0) for phase, cap in CAPS.items()}

    def count(self, key):
        if self.used[self.phase][key] >= CAPS[self.phase][key]:
            raise RuntimeError(f"Hard {key} budget exhausted: {self.phase}")
        self.used[self.phase][key] += 1

    def complete(self):
        require(self.used[self.phase] == CAPS[self.phase], f"Wrong call count: {self.phase}")


def solvers(snapshot, device, budget):
    def setup(solver, is_clark=False):
        solver = solver.to(device=device, dtype=torch.float64).eval().requires_grad_(False)
        if is_clark:
            def closure(state):
                budget.count("clark4")
                qx, qy = study.common.prior.c4.flux4(solver, state, solver.config.data.filter_width_m)
                return dict(qx=qx, qy=qy)
            solver.closure_fields = closure
        else:
            solver.closure.register_forward_pre_hook(lambda *_: budget.count("cnn"))
        native = solver.rhs

        def rhs(w, t):
            budget.count("rhs")
            value = native(w, t)
            require(bool(torch.isfinite(value["rhs"]).all()), "Nonfinite RHS")
            return value

        solver.rhs = rhs
        return solver

    analytic = setup(study.common.make_solver(BASE, 2026, "cpu"), is_clark=True)
    by_model = {(C4, None): analytic}
    for row in snapshot["checkpoints"]:
        mode, seed = row["mode"], row["seed"]
        for field, digest in (("file", "sha256"), ("run", "run_sha256"), ("done", "done_sha256")):
            require(sha(ROOT / row[field]) == row[digest], "Frozen checkpoint or receipt changed")
        payload = torch.load(study.inputs.safe(ROOT / row["file"]), map_location="cpu", weights_only=True)
        require(payload["identity"] == read(ROOT / row["run"]) and payload["epoch"] == row["epoch"]
                and payload["backscatter_factor"] == 1., "Frozen payload mismatch")
        solver = study.make_solver(AUG if mode == BASE else mode, seed, "cpu")
        solver.load_state_dict(payload["model"], strict=True)
        solver = setup(solver)
        require(all(torch.equal(getattr(analytic, key), getattr(solver, key))
                    for key in ("kx", "ky", "inverse_k_squared", "dealias")), "Paired operator mismatch")
        by_model[mode, seed] = solver
    return by_model


def reference(case, audit, projection, files):
    row, = [r for r in audit["references"] if r["seed"] == case and r["variant"] == "n1536_half"]
    field = row["fields"]["reference_omega"]
    for name, expected in ((field["path"], field["expected_sha256"]), (row["time_path"], row["time_sha256"])):
        require(sha(name) == expected, "Reference SHA mismatch")
        files[path_name(Path(name))] = expected
    values = np.load(study.inputs.safe(field["path"]), mmap_mode="r", allow_pickle=False)
    times = np.load(study.inputs.safe(row["time_path"]), allow_pickle=False)
    require(values.shape == (3201, 100, 100) and values.dtype == np.float64
            and np.allclose(times, 1+.00125*np.arange(3201), rtol=0, atol=1e-10), "Reference schedule mismatch")
    sample_times = sorted(set([1+step*DT for step in range(41)]
                              + [1+.05*i+j*DT for i in FRAMES for j in (0, 1)]))
    refs = {}
    for t in sample_times:
        i = round((t-1)/.00125)
        require(abs(times[i]-t) < 1e-10, "Exact-time archive required")
        refs[round(t, 8)] = projection(torch.from_numpy(np.array(values[i]))).numpy()
    return refs


def fft(w):
    return np.fft.fft2(w, norm="forward")


def geometry(analytic):
    k = np.fft.fftfreq(100, d=.01).round()
    shell = np.hypot(k[:, None], k[None, :]).round()
    active = analytic.dealias.cpu().numpy() & (shell > 0)
    return dict(all=active, low=active & (shell <= 16), middle=active & (shell >= 17) & (shell <= 29),
                tail=active & (shell >= 30)), analytic.inverse_k_squared.cpu().numpy()


def metrics(pred, ref, control, masks, inverse, context, initial_gap=None):
    w, y, c = map(fft, (pred, ref, control))
    w_power, y_power = abs(w)**2, abs(y)**2
    require(w_power[~masks["all"]].sum() <= 1e-24*w_power.sum(), "State left K33")
    error = w-y
    defect = error if initial_gap is None else error-fft(initial_gap)
    rows = []
    for band, mask in masks.items():
        mse = float((abs(error[mask])**2).sum())
        local = float((abs(defect[mask])**2).sum()) if initial_gap is not None else None
        k, kref = (.5*float((p*inverse)[mask].sum()) for p in (w_power, y_power))
        z, zref = (.5*float(p[mask].sum()) for p in (w_power, y_power))
        rows.append(dict(**context, band=band, MSE=mse, omega_L2=(mse/(2*zref))**.5,
                         defect_MSE=local, defect_RMS_per_dt=None if local is None else local**.5/DT,
                         difference_vs_Clark4_RMS=float(np.linalg.norm((w-c)[mask])),
                         K=k, reference_K=kref, K_bias=(k-kref)/kref,
                         Z=z, reference_Z=zref, Z_bias=(z-zref)/zref,
                         Z_excess=z-zref))
    require(abs(rows[0]["MSE"]-sum(r["MSE"] for r in rows[1:])) < 1e-12, "Band MSE accounting")
    return rows


def advance(solver, state, t):
    out = solver.step(state, t, DT)
    require(bool(torch.isfinite(out).all()), "Nonfinite RK4 state")
    require(float(out.square().mean()) <= 100*float(state.square().mean()), "RK4 enstrophy safety stop")
    return out


def one_step(by_model, data, refs, starts, masks, inverse, device):
    rows, trajectories, initial_sources = [], {}, []
    for case in CASES:
        for frame in FRAMES:
            t0 = 1+.05*frame
            t1 = round(t0+DT, 8)
            w0 = starts[case, frame]
            gap = w0-refs[case][round(t0, 8)]
            require(np.linalg.norm(gap) < 5e-6*np.linalg.norm(w0), "Cached/float64 reference mismatch")
            w = torch.from_numpy(w0).to(device)
            predictions = {}
            for model_key, s in by_model.items():
                actual_rhs = s.rhs
                captured = []
                if frame == 0:
                    def capture(state, time_s, rhs=actual_rhs, saved=captured):
                        result = rhs(state, time_s)
                        if not saved:
                            saved.append((-result["div_q"]).cpu().numpy())
                        return result
                    s.rhs = capture
                try:
                    predicted = advance(s, w, t0).cpu().numpy()
                finally:
                    s.rhs = actual_rhs
                predictions[model_key] = predicted
                if frame == 0:
                    trajectories[(case, *model_key)] = [w0, predicted]
                    model, seed = model_key
                    ref_source = data[case]["source"][0].double().numpy()
                    difference = abs(fft(captured[0] - ref_source))**2
                    for band, mask in masks.items():
                        initial_sources.append(dict(case=case, model=model, seed=seed, band=band,
                                                    source_MSE=float(difference[mask].sum())))
            control = predictions[C4, None]
            for seed in SEEDS:
                for model in (C4, *MODELS):
                    value = predictions[model, None if model == C4 else seed]
                    rows.extend(metrics(value, refs[case][t1], control, masks, inverse,
                                        dict(case=case, seed=seed, model=model, t0=t0, time_s=t1), gap))
        print(f"One-step defects completed for {case}", flush=True)
    return rows, trajectories, initial_sources


def extend(trajectories, by_model, steps, device):
    for (case, model, seed), states in trajectories.items():
        if model == ORBIT:
            continue
        s = by_model[model, seed]
        w = torch.from_numpy(states[-1]).to(device)
        for step in range(len(states), steps+1):
            w = advance(s, w, 1+(step-1)*DT)
            states.append(w.cpu().numpy())
        print(f"{case} {model}/{seed} reached {1+steps*DT:.2f}", flush=True)


def reuse_orbit(trajectories, snapshot, files):
    receipt = read(OLD / "MANIFEST.json")
    path = OLD / "STATES.npz"
    require(receipt["status"] == "PASS" and receipt["reference_quality"] == study.P["reference_quality"]
            and sha(path) == receipt["output_sha256"][path.name], "Old orbit archive identity")
    original = next(row for row in snapshot["checkpoints"] if row["mode"] == BASE and row["seed"] == 2026)
    require(receipt["input_sha256"][original["file"]] == original["sha256"]
            and receipt["input_sha256"][path_name(BASELINE / "data/110240102/CACHE.pt")]
            == snapshot["cache_sha256"]["110240102"], "Old orbit uses different checkpoint/cache")
    for file in (path, OLD / "MANIFEST.json"):
        files[path_name(file)] = sha(file)
    comparison = {}
    with np.load(path, allow_pickle=False) as old:
        require(np.allclose(old["time_s"], 1+DT*np.arange(11), rtol=0, atol=1e-12), "Old orbit time grid")
        for model, seed in ((C4, None), (BASE, 2026)):
            current = np.stack(trajectories[110240102, model, seed])
            previous = old[model]
            require(np.array_equal(current[0], previous[0]), "Old common initial state differs")
            relative = float(np.linalg.norm(current-previous)/np.linalg.norm(previous))
            require(relative < 1e-10, f"Old {model} trajectory replay mismatch")
            comparison[model] = relative
        trajectory = old[ORBIT]
        require(np.array_equal(trajectory[0], trajectories[110240102, C4, None][0]), "Old orbit initial state differs")
        trajectories[110240102, ORBIT, 2026] = list(trajectory)
    return comparison


def trajectory_rows(trajectories, refs, masks, inverse, steps):
    rows = []
    for case in CASES:
        control = trajectories[case, C4, None]
        for seed in SEEDS:
            for model in (C4, *MODELS, ORBIT):
                states = trajectories.get((case, model, None if model == C4 else seed), ())
                if len(states) != steps+1:
                    continue
                for j, state in enumerate(states):
                    t = round(1+j*DT, 8)
                    rows.extend(metrics(state, refs[case][t], control[j], masks, inverse,
                                        dict(case=case, seed=seed, model=model, step=j, time_s=t)))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["case"], row["seed"], row["model"], row["band"])
        groups.setdefault(key, []).append(row)
    summary = []
    for group in groups.values():
        end = group[-1]
        summary.append(dict(**end, window_mean_MSE=float(np.mean([r["MSE"] for r in group[1:]])),
                            peak_positive_Z_excess=max(0., max(r["Z_excess"] for r in group)),
                            peak_abs_Z_bias=max(abs(r["Z_bias"]) for r in group),
                            peak_abs_K_bias=max(abs(r["K_bias"]) for r in group)))
    return summary


def gate(summary, long=False):
    rows = {(r["case"], r["seed"], r["model"], r["band"]): r for r in summary}
    checks = []
    for case in CASES:
        for seed in SEEDS:
            get = lambda model, band: rows[case, seed, model, band]
            eq, c4, base = (get(m, "all") for m in (EQ, C4, BASE))
            tail = {m: abs(get(m, "tail")["Z_bias"]) for m in (EQ, C4, BASE)}
            row = dict(case=case, seed=seed, endpoint=eq["MSE"] < min(c4["MSE"], base["MSE"]),
                       window=eq["window_mean_MSE"] < min(c4["window_mean_MSE"], base["window_mean_MSE"]),
                       tail=tail[EQ] <= (tail[C4] if long else max(tail[C4], tail[BASE])))
            if long:
                row.update(middle=get(EQ, "middle")["MSE"] <= get(C4, "middle")["MSE"],
                           K=abs(eq["K_bias"]) <= abs(c4["K_bias"]), Z=abs(eq["Z_bias"]) <= abs(c4["Z_bias"]))
            row["pass"] = all(v for k, v in row.items() if k not in ("case", "seed"))
            checks.append(row)
    return dict(status="PASS" if all(row["pass"] for row in checks) else "FAIL", pairs=checks)


def save_window(out, trajectories, refs, masks, inverse, steps, name):
    rows = trajectory_rows(trajectories, refs, masks, inverse, steps)
    summary = summarize(rows)
    csvout(out / f"{name}_TIMEWISE.csv", rows)
    csvout(out / f"{name}_ENDPOINT.csv", summary)
    np.savez_compressed(out / f"{name}_STATES.npz", time_s=1+DT*np.arange(steps+1),
                        **{f"{c}_{m}_{s}": np.stack(w) for (c, m, s), w in trajectories.items() if len(w) == steps+1})
    return summary


@torch.inference_mode()
def run(out, device, budget):
    audit, registry = study.verify_baseline()
    snapshot_path = SNAPSHOT / "CHECKPOINTS.json"
    snapshot = read(snapshot_path)
    require(snapshot["status"] == "FROZEN" and snapshot["quality"] == study.P["reference_quality"]
            and snapshot["cache_sha256"] == registry["cache_sha256"]
            and snapshot["baseline_code_sha256"] == audit["code_sha256"]
            and snapshot["baseline_config_sha256"] == sha(study.common.CONFIG)
            and snapshot["training_code_sha256"] == study.code_identity(), "Frozen study identity")
    files = {path_name(snapshot_path): sha(snapshot_path)}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    by_model = solvers(snapshot, device, budget)
    projector = study.common.make_solver(BASE, 2026, "cpu").double()
    masks, inverse = geometry(by_model[C4, None])
    data = {case: study.inputs.cache(case) for case in CASES}
    refs = {case: reference(case, audit, projector.project, files) for case in CASES}
    starts = {(case, i): projector.project(data[case]["omega"][i].double()).numpy()
              for case in CASES for i in FRAMES}
    one, trajectories, source_rows = one_step(by_model, data, refs, starts, masks, inverse, device)
    budget.complete()
    csvout(out / "ONE_STEP.csv", one)
    csvout(out / "INITIAL_SOURCE.csv", source_rows)
    budget.phase = "short"
    extend(trajectories, by_model, 10, device)
    budget.complete()
    old_replay = reuse_orbit(trajectories, snapshot, files)
    short = save_window(out, trajectories, refs, masks, inverse, 10, "SHORT")
    short_gate = gate(short)
    dump(out / "SHORT_GATE.json", short_gate)
    print(f"Extension gate: {short_gate['status']}", flush=True)
    if short_gate["status"] == "PASS":
        budget.phase = "extension"
        extend(trajectories, by_model, 40, device)
        budget.complete()
        extension = save_window(out, trajectories, refs, masks, inverse, 40, "EXTENSION")
        long_gate = gate(extension, long=True)
        dump(out / "LONG_GATE.json", long_gate)
    else:
        long_gate = dict(status="NOT_RUN", reason="1.05 gate did not pass")
    return dict(input_sha256=files, orbit_comparison_replay_relative=old_replay,
                short_gate=short_gate, extension_status="COMPLETED" if short_gate["status"] == "PASS" else "NOT_RUN",
                long_gate=long_gate, t5="NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pair"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run", default="run01")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
        return
    if not args.execute:
        print(json.dumps(CAPS, indent=2))
        return
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    require(Path(args.run).name == args.run and args.run not in ("", ".", ".."), "Invalid output name")
    out = HERE / "results" / args.run
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget()
    start = time.perf_counter()
    try:
        result = run(out, args.device, budget)
        dump(out / "MANIFEST.json", dict(
            status="PASS", reference_quality=study.P["reference_quality"], device=args.device, dtype="float64",
            budgets=CAPS, actual=budget.used, wall_seconds=time.perf_counter()-start,
            code_sha256=sha(Path(__file__)), protocol_sha256=sha(HERE / "PROTOCOL.md"),
            output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}, **result))
    except Exception as error:
        dump(out / "FAILURE.json", dict(status="FAIL", error=str(error), actual=budget.used))
        raise


if __name__ == "__main__":
    main()
