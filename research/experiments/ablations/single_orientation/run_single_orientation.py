"""Two single-orientation history methods, using the original unaugmented FULL-C4 weights."""

import argparse
import copy
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DRIVER = ROOT / "research/full_aug_parameter_pc_full_v1_20261116"
sys.path[:0] = [str(ROOT), str(DRIVER)]

import driver
import pc_common as common
import pc_engine as native
import pc_storage as storage

CASE = 110240104
SEEDS = (2026, 2028)
METHODS = ("Parameter-Hold", "Parameter-PC")
NAMES = {"Parameter-Hold": "Single-Hold", "Parameter-PC": "Single-Trend"}
RESULT = HERE / "run01"
OLD = DRIVER / "results/run01"
FROZEN = ROOT / "research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json"
ORIGINAL_TRAINING = ROOT / "research/full_c4_ns_retrain_v1_20261103/training.py"


class SingleEngine(native.Engine):
    def orbit(self, omega):
        """The driver calls this at a refresh. Evaluate at orientation zero only."""
        state = self.s.state(omega)
        fields = self.original(state)
        self.check_member(state, fields)
        return {"original": fields, "heads": self.captured_raw.clone()}

    def check_member(self, state, fields):
        try:
            common.q.physical(self.s, state, fields)
        except RuntimeError as exc:
            raise native.TrajectoryFailure(
                "FAILED_PHYSICAL",
                str(exc),
                {"omega": native.cpu(state["omega"]),
                 "qd": native.cpu(native.vec(fields, "qd")),
                 "qb": native.cpu(native.vec(fields, "qb"))},
            ) from exc

    def flux(self, state):
        if self.method == "Aug":
            # Legacy driver label for the shared current-state startup. The
            # solver has already loaded the original FULL-C4 checkpoint.
            return super().flux(state)
        method = self.method
        fraction = self.fractions[self.stage]
        head, tau = self.bank.predict(self.n, fraction, method)
        original = self.original

        def decode_one(current_state):
            self.captured_raw = head
            fields = self.decode(current_state, head)
            self.check_member(current_state, fields)
            return fields

        try:
            self.method = "Aug"
            self.original = decode_one
            flux = super().flux(state)
            self.last_check["tau"] = tau
            return flux
        finally:
            self.method = method
            self.original = original


def single_bank(root, descriptor, device):
    def load(entry):
        file = root / entry["file"]
        assert common.sha(file) == entry["sha256"], file
        saved = common.load(file)
        assert common.meta(saved)["step"] == entry["step"]
        head = saved["heads"]
        assert head.shape == (7, 100, 100) and head.dtype == np.float64
        assert np.isfinite(head).all()
        return dict(entry, heads=torch.as_tensor(head, device=device))

    return native.Bank(
        [load(entry) for entry in descriptor["active"]],
        load(descriptor["pending"]) if descriptor["pending"] else None,
    )


def checked(path, expected, files):
    actual = common.sha(path)
    assert actual == expected, f"Input SHA changed: {path}"
    files[str(path.relative_to(ROOT))] = actual


def inputs():
    inventory = common.read(OLD / "RUN.json")["inventory"]
    archive = common.read(OLD / "LATEST_REPORT.json")
    report = OLD / archive["path"]
    manifest = common.read(report / "MANIFEST.json")
    files = {}
    checked(report / "MANIFEST.json", archive["manifest_sha256"], files)
    old_run = common.read(OLD / "RUN.json")
    for name in ("pc_common.py", "pc_engine.py", "pc_metrics.py", "pc_storage.py", "driver.py"):
        checked(DRIVER / name, old_run["code_identity"][name], files)
    files[str(FROZEN.relative_to(ROOT))] = common.sha(FROZEN)
    original = [row for row in common.read(FROZEN)["checkpoints"]
                if row["mode"] == "FULL-C4" and row["seed"] in SEEDS]
    assert {row["seed"] for row in original} == set(SEEDS)
    for row in original:
        for key, sha_key in (("file", "sha256"), ("run", "run_sha256"), ("done", "done_sha256")):
            checked(ROOT / row[key], row[sha_key], files)
        training = common.read(ROOT / row["run"])
        done = common.read(ROOT / row["done"])
        protocol = training["protocol"]
        assert training["mode"] == "FULL-C4" and protocol.get("augmentation", "none") == "none"
        assert protocol["train_seed"] == 110240101 and protocol["validation_seed"] == 110240102
        assert done["status"] == "TRAINING_COMPLETE" and done["best_sha256"] == row["sha256"]
        checked(ORIGINAL_TRAINING, training["code_sha256"][str(ORIGINAL_TRAINING.relative_to(ROOT))], files)
        for method in METHODS:
            name = f"STATES_{CASE}_{row['seed']}_{method}.npy"
            checked(report / name, manifest["outputs"][name], files)
    for name in ("COSTS.csv",):
        checked(report / name, manifest["outputs"][name], files)
    references = inventory["references"][str(CASE)]
    for step in range(0, 801, 10):
        name = references["frames"][str(step)]
        checked(ROOT / name, inventory["input_sha256"][name], files)
        frame = common.load(ROOT / name)
        assert abs(float(frame["time_s"]) - (1 + step * common.H)) < 1e-10
    return original, references, report, files


def load_original(row, device="cpu"):
    assert row["mode"] == "FULL-C4"
    file = ROOT / row["file"]
    assert common.sha(file) == row["sha256"]
    payload = torch.load(file, map_location="cpu", weights_only=True)
    assert payload["identity"] == common.read(ROOT / row["run"])
    assert payload["epoch"] == row["epoch"] and payload["backscatter_factor"] == 1.0
    solver = common.q.study.common.make_solver("FULL-C4", row["seed"])
    solver = solver.double().to(device).eval().requires_grad_(False)
    solver.load_state_dict(payload["model"], strict=True)
    with np.load(OLD / "OPERATORS.npz", allow_pickle=False) as archived:
        for key, value in common.operators(solver).items():
            assert np.array_equal(value, archived[key]), f"Operator mismatch: {key}"
    return solver


def bind():
    common.HERE = HERE
    common.P = copy.deepcopy(common.P)
    common.P["hard_budget"] = {"CNN": 2400, "RHS": 15000}
    common.model = load_original
    driver.Engine = SingleEngine
    driver.load_bank = single_bank


@torch.inference_mode()
def preflight():
    """Check real checkpoint/decoder parity and stage extrapolation without advancing LES."""
    bind()
    rows, reference, _, files = inputs()
    solver = load_original(rows[0])
    omega = solver.project(torch.as_tensor(common.References(reference).at(0)["reference_omega"]))
    import pc_tests

    journal = pc_tests.MockJournal()
    engine = SingleEngine(solver, "Parameter-Hold", journal, HERE, HERE, native.basic_metrics(solver, omega))
    try:
        a = engine.orbit(omega)["heads"]
        b = a + .001
        bank = native.Bank([{"step": 0, "heads": a}, {"step": 1, "heads": b}])
        engine.n, engine.stage, engine.bank = 2, 1, bank
        state = solver.state(omega)
        for method, expected in zip(METHODS, [b, 2.5 * b - 1.5 * a]):
            engine.method = method
            actual = engine.flux(state)
            decoded = common.decoder.decode(solver, state, expected)
            assert torch.equal(actual["qx"], decoded["qx"]) and torch.equal(actual["qy"], decoded["qy"])
            assert engine.last_check["tau"] == 1.5
        assert journal.used == {"CNN": 1, "RHS": 0}, journal.used
    finally:
        engine.close()
    common.dump(HERE / "PREFLIGHT.json", {
        "status": "PASS", "script_sha256": common.sha(__file__), "input_count": len(files),
        "one_orientation_shape": list(a.shape), "CNN": 1, "RHS": 0,
        "checks": ["original checkpoint and operator identity", "real decoder parity", "hold and trend stage arithmetic"],
    })
    print("Preflight passed: one original CNN evaluation, zero LES steps.", flush=True)


@torch.inference_mode()
def run():
    bind()
    assert common.read(HERE / "PREFLIGHT.json")["script_sha256"] == common.sha(__file__)
    rows, refs, report, files = inputs()
    assert torch.cuda.is_available(), "CUDA required for this pilot"
    device = "cuda"
    common.safe(RESULT).mkdir(exist_ok=True)
    record = {
        "case": CASE, "flow_role": "development, absent from training and checkpoint selection",
        "network_seeds": list(SEEDS), "history_methods": NAMES, "checkpoint_rows": rows,
        "rotational_augmentation": False, "inference_orientations": [0],
        "start": 1.0, "end": 5.0, "dt": common.H, "refresh_pair_period": .025,
        "refresh_pair_spacing": .005, "scoring_times": "1.05:0.05:5 inclusive, 80 equally weighted frames",
        "startup": "One current-state step with original unaugmented weights. Two unrotated anchor evaluations.",
        "reference": refs, "reference_quality": common.QUALITY,
        "input_sha256": files, "script_sha256": common.sha(__file__),
        "runtime": driver.runtime_identity(device), "training_updates": 0, "new_DNS_steps": 0,
    }
    with storage.RunLock(RESULT):
        path = RESULT / "RUN.json"
        if path.exists():
            assert common.read(path) == record, "Run identity changed"
        else:
            common.dump(path, record)
        journal = common.Journal(RESULT / "invocations")
        try:
            reference = common.References(refs)
            for row in rows:
                seed = row["seed"]
                try:
                    start, receipt = driver.startup(RESULT, CASE, seed, row, reference, journal, device)
                except native.TrajectoryFailure as exc:
                    for method in METHODS:
                        target = RESULT / "trajectories" / common.key(CASE, seed, method) / "STATUS.json"
                        common.dump(target, dict(status=exc.status, reason=str(exc), last_accepted_step=0,
                                                scope="unaugmented current-state startup"))
                    continue
                assert receipt["counts"] == {"CNN": 6, "RHS": 4}
                for method in METHODS if seed == 2026 else METHODS[::-1]:
                    archived = np.load(report / f"STATES_{CASE}_{seed}_{method}.npy", mmap_mode="r")
                    np.testing.assert_allclose(start["omega0"], archived[0], rtol=0, atol=1e-12)
                    col = NAMES[method]
                    print(f"Running {col}, IC104, seed {seed}, original FULL-C4", flush=True)
                    driver.run_trajectory(RESULT, CASE, seed, method, row, start, journal, device)
            common.dump(RESULT / "EXECUTION.json", dict(status="TERMINAL", actual_calls=journal.used,
                                                        training_updates=0, new_DNS_steps=0))
        except BaseException as exc:
            common.dump(RESULT / "INTERRUPTION.json", dict(error=str(exc), detail=traceback.format_exc(),
                                                           actual_calls=journal.used))
            raise
        finally:
            journal.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "run"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (preflight if args.action == "check" else run)()


if __name__ == "__main__":
    main()
