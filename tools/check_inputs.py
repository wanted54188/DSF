"""Check packaged code and optional external frozen inputs without running a solver."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from workspace import ROOT, source_paths


CHECKPOINTS = Path("research/full_c4_equivariant_dynamics_v1_20261109/frozen/CHECKPOINTS.json")
CHECK_CONFIG = Path("results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json")
DECODER_RECEIPT = Path("research/full_aug_orbit_rk4_one_step_v1_20261114/PHASE_ONE_MANIFEST.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(file):
    checksum = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify(root, inputs, paths=None):
    failures = []
    for name, expected in inputs.items():
        relative = Path(str(name).replace("\\", "/"))
        file = root / (paths or {}).get(relative.as_posix(), relative)
        if not file.is_file():
            failures.append(f"missing {relative}")
        elif expected is not None and digest(file) != expected:
            failures.append(f"SHA-256 mismatch {relative}")
    return failures


def frozen_inputs(manifest, include_unaugmented):
    inputs = {CHECK_CONFIG: manifest["baseline_config_sha256"], DECODER_RECEIPT: None}
    modes = {"FULL-C4-Augmented"}
    if include_unaugmented:
        modes.add("FULL-C4")
    for row in manifest["checkpoints"]:
        if row["mode"] in modes and row["seed"] in (2026, 2028):
            for resource, checksum in (("file", "sha256"), ("run", "run_sha256"), ("done", "done_sha256")):
                inputs[row[resource]] = row[checksum]
    return inputs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive-root", type=Path, help="Directory containing the original asset tree")
    cli.add_argument("--include-unaugmented", action="store_true", help="Also check the ablation checkpoints")
    args = cli.parse_args()
    paths = source_paths()
    manifest = read_json(ROOT / paths[CHECKPOINTS.as_posix()])
    frozen_code = manifest["training_code_sha256"] | manifest["baseline_code_sha256"]
    failures = verify(ROOT, frozen_code, paths)
    if failures:
        sys.exit("Packaged frozen code check failed:\n  " + "\n  ".join(failures))
    print(f"Verified SHA-256 of {len(frozen_code)} frozen source/config files.")
    if args.archive_root:
        inputs = frozen_inputs(manifest, args.include_unaugmented)
        failures = verify(args.archive_root.resolve(), inputs)
        if failures:
            sys.exit("External frozen inputs check failed:\n  " + "\n  ".join(failures))
        print(f"Verified {len(inputs)} external checkpoint, configuration and identity files.")


if __name__ == "__main__":
    main()
