"""Read-only baseline identity, rotation of cached samples, and model factory."""

import json
from pathlib import Path
import sys

import torch

from model import install, spatial, vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research/full_c4_mechanism_v1_20261104"))
import inputs

common, data, objective = inputs.benchmark_modules()
import training as native_training

P = inputs.read(HERE / "PROTOCOL.json")
AUGMENTED, EQUIVARIANT = P["modes"]
CASES = (P["train_case"], P["development_case"])


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE / "results"):
        raise PermissionError(f"Output outside the new experiment: {path}")
    return path


def write_json(path, payload):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def save_pt(path, payload):
    path = safe_output(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def code_identity():
    paths = [*HERE.glob("*.py"), HERE / "PROTOCOL.json"]
    return {str(path.relative_to(ROOT)): inputs.sha(path) for path in sorted(paths)}


def verify_baseline(cases=CASES):
    audit = inputs.read(inputs.RUN / "AUDIT.json")
    registry = inputs.read(inputs.RUN / "BASELINE_REGISTRY.json")
    protocol = common.protocol()
    inputs.require(audit["protocol"] == protocol and audit["quality"] == P["reference_quality"]
                   and registry["code_sha256"] == audit["code_sha256"]
                   and registry["reference_quality"] == P["reference_quality"],
                   "Original benchmark provenance changed")
    inputs.require((protocol["train_seed"], protocol["validation_seed"]) == CASES
                   and protocol["training_seeds"] == P["seeds"]
                   and protocol["epochs"] == 100 and protocol["batch_frames"] == 4
                   and protocol["optimizer"] == "AdamW" and not protocol["TF32"],
                   "Fixed original training protocol changed")
    for name, digest in audit["code_sha256"].items():
        inputs.require(inputs.sha(ROOT / name) == digest, f"Original code changed: {name}")
    inputs.require(inputs.sha(common.CONFIG) == audit["files"][str(common.CONFIG)],
                   "Original solver configuration changed")
    inputs.require(Path(native_training.__file__).resolve() == inputs.BENCH / "training.py",
                   "Wrong original training module")
    for case in cases:
        if case not in CASES:
            raise PermissionError("Only 101 and 102 caches are permitted")
        directory = inputs.RUN / "data" / str(case)
        receipt = inputs.read(directory / "DONE.json")
        expected = registry["cache_sha256"][str(case)]
        inputs.require(receipt["seed"] == case and receipt["frames"] == 81
                       and receipt["quality"] == P["reference_quality"]
                       and receipt["cache_sha256"] == expected
                       and inputs.sha(directory / "CACHE.pt") == expected, "Original cache changed")
    return audit, registry


def source(solver, fields):
    return common.prior.c4.source(solver, (fields["qx"], fields["qy"]))


def make_solver(mode, seed, device="cpu"):
    if mode not in P["modes"] or seed not in P["seeds"]:
        raise ValueError("Choose a fixed architecture and seed")
    solver = common.make_solver("FULL-C4", seed, device=device)
    if mode == EQUIVARIANT:
        install(solver, common.prior, P["orientation_width"])
    expected = torch.fft.fftfreq(P["grid"], d=1/P["grid"], device=device).abs() <= P["cutoff"]
    inputs.require(torch.equal(solver.dealias, expected[:, None] & expected[None, :])
                   and solver.config.domain.lx == solver.config.domain.ly
                   and not solver.config.physics.forcing_enabled
                   and solver.config.paper.closure_channels == 32,
                   "Expected unforced square production K33 solver")
    return solver


def augmentation_turn(frame, epoch):
    return (frame + epoch - 1) % 4


def rotate_frame(cache, index, turn):
    """Transform the cached inputs and original labels together, at one index."""
    scalar_keys = ("omega", "psi", "strain", "laplacian_omega", "source", "pi")
    sample = {name: spatial(cache[name][index], turn) for name in scalar_keys}
    for x, y in (("omega_x", "omega_y"), ("fourth_x", "fourth_y"), ("qx", "qy")):
        sample[x], sample[y] = vector(cache[x][index], cache[y][index], turn)
    ux, uy, vx, vy = (spatial(cache[k][index], turn) for k in ("u_x", "u_y", "v_x", "v_y"))
    # Under a proper quarter-turn, R (grad u) R^T = [[vy,-vx],[-uy,ux]].
    tensor = (ux, uy, vx, vy) if turn % 2 == 0 else (vy, -vx, -uy, ux)
    sample.update(zip(("u_x", "u_y", "v_x", "v_y"), tensor))
    return {name: field.unsqueeze(0) for name, field in sample.items()}


def training_quantities(solver, cache, index, epoch, mode):
    if mode == AUGMENTED:
        turn = augmentation_turn(index, epoch)
        if turn:
            cache = rotate_frame(cache, index, turn)
            index = 0
    state, fields, prediction = data.forward_frame(solver, cache, index)
    return objective.quantities(solver, cache, index, state, fields, prediction)


def effective_protocol(base, mode):
    # train_one only compares initial hashes across p["modes"]. These modes
    # have different architectures; the augmented init is checked separately.
    return dict(base, version=P["version"], modes=[mode],
                augmentation=P["augmentation"] if mode == AUGMENTED else "none",
                architecture=mode, orientation_width=P["orientation_width"] if mode == EQUIVARIANT else None)
