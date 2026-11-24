"""Score archived same-state head substitutions against matched-time FDNS Source."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def read_archive(repo):
    source = repo / "research/full_aug_hold_audit_v1_20261117/analyze.py"
    spec = importlib.util.spec_from_file_location("archived_analysis", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(repo, output):
    audit = read_archive(repo)
    c, m, protocol = audit.c, audit.m, audit.P
    inputs = audit.Inputs(repo / protocol["source"], output)
    operators = c.load(inputs.verify(inputs.run / "OPERATORS.npz"))
    masks, _ = m.geometry(operators)
    reference_inventory = inputs.record["inventory"]
    history_file = repo / "research/full_aug_hold_audit_v1_20261117/results/audit01/SAME_STATE_HEAD_INTERVENTIONS.csv"
    with history_file.open(encoding="utf-8-sig", newline="") as file:
        old = {
            (int(r["case"]), int(r["seed"]), r["state_path"], float(r["time_s"]), r["variant"], r["band"]): r
            for r in csv.DictReader(file)
        }

    def forbid_cnn(*_args):
        raise AssertionError("CNN forward is forbidden in the archived analysis")

    torch.set_num_threads(2)
    models = {}
    for checkpoint in reference_inventory["checkpoints"]:
        for name, field in (("file", "sha256"), ("run", "run_sha256"), ("done", "done_sha256")):
            inputs.verify(repo / checkpoint[name], checkpoint[field])
        solver = c.model(checkpoint, "cpu")
        solver.closure.register_forward_pre_hook(forbid_cnn)
        models[checkpoint["seed"]] = solver

    results = []
    with torch.inference_mode():
        for case in protocol["cases"]:
            desc = reference_inventory["references"][str(case)]
            for seed in protocol["seeds"]:
                solver = models[seed]
                for path in protocol["mechanism_state_paths"]:
                    for time in protocol["mechanism_samples"]:
                        step = round((time - 1) / c.H)
                        assert step in desc["steps"]
                        refs = c.References(desc)
                        name = desc["fields"]["reference_source"] if desc["kind"] == "dense" else desc["frames"][str(step)]
                        inputs.verify(repo / name, reference_inventory["input_sha256"][str(Path(name))])
                        target = refs.at(step)["reference_source"]
                        target_hat = m.F(target)
                        folder = inputs.run / "trajectories" / c.key(case, seed, path)
                        saved = c.load(inputs.verify(folder / ("ENDPOINT.npz" if step == 800 else f"steps/{step + 1:04d}.npz")))
                        snap = saved if step == 800 else {
                            k.removeprefix("observation__"): v for k, v in saved.items() if k.startswith("observation__")
                        }
                        bank = c.meta(saved).get("bank")
                        if bank is None:
                            bank = c.meta(c.load(inputs.verify(folder / "steps/0800.npz")))["bank"]
                        history = [torch.from_numpy(c.load(inputs.verify(inputs.run / a["file"], a["sha256"]))["heads"]) for a in bank["active"]]
                        tau = step - bank["active"][-1]["step"]
                        hold = history[-1]
                        trend = (1 + tau) * hold - tau * history[0]
                        used = trend if path == "Parameter-PC" else hold
                        np.testing.assert_array_equal(audit.cpu(used), snap["predicted_heads"])
                        w = torch.from_numpy(snap["omega"])
                        stage = solver.state(w)
                        rotated = [solver.state(c.q.spatial(w, turn)) for turn in range(4)]
                        for variant, head in (("hold", hold), ("trend", trend), ("fresh", torch.from_numpy(snap["fresh_heads"]))):
                            members = [c.decoder.decode(solver, s, h) for s, h in zip(rotated, head)]
                            flux = c.q.mean_fields(solver, stage, members)
                            source = audit.cpu(audit.D(solver, audit.vec(flux, "qd"))) + audit.cpu(audit.D(solver, audit.vec(flux, "qb")))
                            if head is used:
                                np.testing.assert_allclose(source, snap["Source"], rtol=0, atol=1e-11 * max(audit.rms(source), 1))
                            source_hat = m.F(source)
                            diff_hat = source_hat - target_hat
                            for band in ("all", "low", "mid", "high", "corner"):
                                mask = masks[band]
                                power = float(np.sum(np.abs(source_hat[mask]) ** 2))
                                old_power = float(old[case, seed, path, time, variant, band]["Source_power"])
                                np.testing.assert_allclose(power, old_power, rtol=1e-10, atol=1e-12)
                                results.append(dict(case=case, seed=seed, state_path=path, time_s=time, variant=variant, band=band,
                                                    tau=tau, source_power=power, archived_source_power=old_power,
                                                    reference_source_power=float(np.sum(np.abs(target_hat[mask]) ** 2)),
                                                    source_MSE_to_FDNS=float(np.sum(np.abs(diff_hat[mask]) ** 2)),
                                                    reference_file=name, reference_quality=c.QUALITY))
    assert len(results) == 36 * 3 * 5
    output.mkdir(parents=True, exist_ok=True)
    audit.csv(output / "same_state_source_vs_fdns.csv", results)
    (output / "same_state_source_receipt.json").write_text(json.dumps(dict(
        states=36, rows=len(results), new_CNN=0, new_integration=0,
        scope="Same autonomous LES state across substitutions; target is the same-time FDNS Source on the FDNS state",
        archive_sha256=hashlib.sha256(history_file.read_bytes()).hexdigest(), inputs=inputs.hashes,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.repo.resolve(), args.output.resolve())
