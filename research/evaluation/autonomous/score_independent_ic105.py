"""Read-only scoring of the frozen IC105 run against 80 saved FDNS frames."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CASE = 110240105
SEEDS = (2026, 2028)
MODELS = {"Aug": "DSF-Aug", "Fresh-C4": "DSF-Sym", "Parameter-PC": "DSF-Trend", "Parameter-Hold": "DSF-Hold"}
STEPS = range(10, 801, 10)
QUALITY = "REFERENCE_NOT_CONVERGED"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(path, *keys):
    with np.load(path, allow_pickle=False) as saved:
        return tuple(saved[key] for key in keys)


def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tex_table(result, audits):
    completed = len(result)
    lines = [
        r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{IC105自主LES相对于共同离散FDNS的误差。全窗为$t=1.05,1.10,\ldots,5$的80个等权评分时刻。NGM4只计算一条确定性轨迹。}",
        r"\label{tab:ic105-accuracy}", r"\begin{tabular}{llrrr}\toprule",
        r"模型 & 网络种子 & 全窗涡量MSE & $t=5$涡量相对$L_2$ & 高频$Z$比值\\\midrule",
    ]
    results = {(r["model"], r["seed"]): r for r in result}
    for audit in audits:
        model, seed = audit["model"], audit["seed"]
        shown_seed = seed if seed is not None else r"\textemdash"
        if (model, seed) in results:
            r = results[model, seed]
            lines.append(f"{model} & {shown_seed} & {r['full_omega_MSE']:.6f} & {r['t5_omega_relative_L2']:.5f} & {r['mean_high_Z_ratio']:.3f}" + r"\\")
        else:
            status = audit["experiment_status"].replace("_", r"\_")
            lines.append(f"{model} & {shown_seed} & " + rf"\multicolumn{{3}}{{l}}{{\texttt{{{status}}}}}\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    lines.append(f"表~\\ref{{tab:ic105-accuracy}}报告{completed}条达到$t=5$并通过存档校验的轨迹。")
    if completed == 9:
        baseline = results["NGM4", None]["full_omega_MSE"]
        for seed in SEEDS:
            ordered = sorted([(model, results[model, seed]["full_omega_MSE"]) for model in MODELS.values()]
                             + [("NGM4", baseline)], key=lambda item: item[1])
            hold = results["DSF-Hold", seed]["full_omega_MSE"]
            changes = [100 * (1-hold/base) for base in (baseline, results["DSF-Trend", seed]["full_omega_MSE"],
                                                        results["DSF-Sym", seed]["full_omega_MSE"])]
            ordering = r"$<$".join(model for model, _ in ordered)
            lines.append(f"种子{seed}下，全窗涡量MSE由低到高为{ordering}。Hold相对NGM4、Trend和Sym的有符号MSE降幅分别为"
                         + "、".join(f"{x:.2f}\\%" for x in changes) + "。")
    lines.append(r"$Z$比值为80帧逐帧频带比值的均值，高频带为径向壳30--47。")
    return "\n".join(lines) + "\n"


def score(study, out):
    repo = study.parents[1]
    sys.path.insert(0, str(repo / "research/full_aug_parameter_pc_full_v1_20261116"))
    import pc_metrics as metrics

    validation = study / "VALIDATION_20261118"
    root = validation / "LES/ic105"
    clark = validation / f"CLARK/{CASE}"
    execution = root / "EXECUTION.json"
    provenance = repo / "research/full_aug_hold_audit_v1_20261117/independent_ic105"
    rec = read(provenance / "PROVENANCE_ATTESTATION.json")
    assert rec["case"] == CASE and rec["status"] == "USER_ATTESTED_UNUSED_BEFORE_FREEZE"
    assert rec["initial_freeze_sha256"] == sha(provenance/"FROZEN.json")
    assert rec["initial_coefficients_sha256"] == sha(provenance/"INITIAL_FOURIER_COEFFICIENTS.npz")
    assert read(execution)["status"] == "ALL_SCHEDULED_TRAJECTORIES_TERMINAL"
    run = read(root / "RUN.json")
    freeze = sha(validation / "FREEZE.json")
    assert run["group"] == "ic105" and run["period"] == 5 and not run["single"] and run["freeze_sha256"] == freeze
    clark_run = read(clark/"RUN.json")
    assert clark_run["case"] == CASE and clark_run["dt"] == .005 and clark_run["network_seed"] is None
    assert clark_run["freeze_sha256"] == freeze

    entries = [("NGM4", None, clark/"limited_completion/Clark4_0.005")]
    entries.extend((model, seed, root/"trajectories"/f"{CASE}_{seed}_{old}") for seed in SEEDS for old, model in MODELS.items())
    statuses = [read(folder/"STATUS.json") for _, _, folder in entries]
    terminal = {"VERIFIED", "FAILED_PHYSICAL", "FAILED_NUMERICAL", "NOT_RUN_BUDGET"}
    if any(status["status"] not in terminal for status in statuses):
        raise RuntimeError("IC105 has nonterminal trajectories; no paper table generated")

    fdns = validation / f"DNS/{CASE}_n1536_h0.00012500"
    done = read(fdns/"DONE.json")
    assert done["status"] == "OPERATOR_CHECKS_PASS" and done["completed_t_end"] == 5
    assert done["reference_quality"] == QUALITY
    references = {}
    used_inputs = {str(p.relative_to(repo)): sha(p) for p in
                   (provenance/"PROVENANCE_ATTESTATION.json", root/"RUN.json", execution, root/"OPERATORS.npz",
                    clark/"RUN.json", fdns/"DONE.json", validation/"FREEZE.json")}
    for step in (0, *STEPS):
        frame = fdns/f"frame_{step:04d}.npz"
        digest = sha(frame)
        assert done["frames"][frame.name] == digest
        w, time = arrays(frame, "reference_omega", "time_s")
        assert w.shape == (100, 100) and w.dtype == np.float64 and np.isfinite(w).all()
        assert abs(float(time) - (1 + step*.005)) <= 1e-11
        references[step] = w
        used_inputs[str(frame.relative_to(repo))] = digest

    with np.load(root/"OPERATORS.npz", allow_pickle=False) as saved:
        op = dict(saved)
    _, shell = metrics.geometry(op)
    assert shell[op["dealias"]].max() == 47
    ckpt = {r["seed"]: r for r in run["inventory"]["checkpoints"]}
    for seed in SEEDS:
        assert sha(repo/ckpt[seed]["file"]) == ckpt[seed]["sha256"]
    summary, audit, frames = [], [], []
    for (model, seed, folder), status in zip(entries, statuses):
        accepted = status.get("last_accepted_step")
        row = dict(case=CASE, model=model, seed=seed, dt=.005, experiment_status=status["status"],
                   t_end=1+.005*accepted if accepted is not None else None, checkpoint_sha256=ckpt[seed]["sha256"] if seed else None,
                   reference_quality=QUALITY, reference_n=1536, reference_dt=.000125,
                   reason=status.get("reason"), archive=str(folder.relative_to(repo)))
        audit.append(row)
        used_inputs[str((folder/"STATUS.json").relative_to(repo))] = sha(folder/"STATUS.json")
        if status["status"] != "VERIFIED":
            continue
        assert accepted == 800
        source = folder/"INITIAL.npz" if seed is None else root/"startup"/f"{CASE}_{seed}"/"STARTUP.npz"
        initial_key = "omega" if seed is None else "omega0"
        w, = arrays(source, initial_key)
        np.testing.assert_allclose(w, metrics.project(references[0], op), rtol=0, atol=1e-11)
        files = sorted((folder/"steps").glob("*.npz"))
        assert [int(f.stem) for f in files] == list(range(1 if seed is None else 2, 801))
        scored = []
        maximum_replay_error = 0.
        for step in STEPS:
            path = folder/"steps"/f"{step:04d}.npz"
            previous = folder/"steps"/f"{step - 1:04d}.npz"
            w, rhs, raw_meta = arrays(path, "omega", "rhs", "metadata")
            w_prev, = arrays(previous, "omega")
            assert json.loads(str(raw_meta.item()))["accepted_step"] == step
            assert np.isfinite(w).all() and rhs.shape == (4, 100, 100)
            check = metrics.verify_step(w_prev, w, rhs, op)
            maximum_replay_error = max(maximum_replay_error, check["native_RK4_max_abs_replay_error"])
            context = dict(case=CASE, seed=seed, model=model, dt=.005, step=step, time=1+step*.005)
            scored.extend(metrics.state_rows(w, references[step], op, context))
            used_inputs[str(path.relative_to(repo))] = sha(path)
            used_inputs[str(previous.relative_to(repo))] = sha(previous)
        frames.extend(scored)
        full = [r for r in scored if r["band"]=="all"]
        high = [r for r in scored if r["band"]=="high"]
        assert len(full) == len(high) == 80
        row.update(scoring_frames=80, sampled_rk4_replay_max_error=maximum_replay_error,
                   physical_envelope=status.get("physical_envelope"))
        summary.append(dict(**row, full_omega_MSE=np.mean([r["omega_MSE"] for r in full]),
                            t5_omega_relative_L2=full[-1]["omega_relative_L2"],
                            full_u_MSE=np.mean([r["u_MSE"] for r in full]),
                            full_v_MSE=np.mean([r["v_MSE"] for r in full]),
                            mean_abs_K_bias=np.mean([abs(r["K_relative_bias"]) for r in full]),
                            mean_abs_Z_bias=np.mean([abs(r["Z_relative_bias"]) for r in full]),
                            mean_high_Z_ratio=np.mean([r["Z"]/r["reference_Z"] for r in high])))
    destination = out/"tables"
    destination.mkdir(parents=True, exist_ok=True)
    table(destination/"ic105_audit.csv", audit)
    table(destination/"ic105_accuracy.csv", summary)
    table(destination/"ic105_state_metrics.csv", frames)
    (destination/"ic105_receipt.json").write_text(json.dumps(dict(scored=len(summary), scheduled=9, reference_quality=QUALITY,
                                                                  inputs_sha256=used_inputs, new_CNN_calls=0), indent=2), encoding="utf-8")
    pending = destination/"ic105_results.tex.tmp"
    pending.write_text(tex_table(summary, audit), encoding="utf-8")
    pending.replace(destination/"ic105_results.tex")
    print(f"IC105: {len(summary)}/9 scored; FDNS reference quality: {QUALITY}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    score(args.study.resolve(), args.output.resolve())
