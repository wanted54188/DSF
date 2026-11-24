"""Run the DSF-SGS paper code using manuscript names."""

import argparse
from pathlib import Path
import subprocess
import sys

from tools.workspace import DEFAULT_WORKSPACE, ROOT, prepare_workspace, source_paths


ENTRIES = {
    "train-dsf-aug": "research/closures/dsf_aug/training/train_dsf.py",
    "decaying-rollouts": "research/experiments/decaying/run_dsf_rollouts.py",
    "validate-decaying": "research/experiments/validation/run_validation.py",
    "time-step-comparison": "research/experiments/decaying/run_time_step_comparison.py",
    "freeze-ic105": "research/experiments/independent_ic105/freeze_initial_condition.py",
    "score-ic105": "research/evaluation/autonomous/score_independent_ic105.py",
    "forced-reference": "research/experiments/forced_01/reference_window/run_forced_case.py",
    "forced-rollouts": "research/experiments/forced_01/archived_reference/run_forced_case.py",
    "score-forced": "research/experiments/forced_01/archived_reference/score_forced_rollouts.py",
    "orientation-ablation": "research/experiments/ablations/single_orientation/run_single_orientation.py",
    "score-orientation-ablation": "research/experiments/ablations/single_orientation/score_single_orientation.py",
    "audit-initial-conditions": "research/reference/initialization/audit_flow_conditions.py",
    "reassess-dns": "research/reference/reassessment/reassess_dns.py",
    "a-priori-replay": "research/evaluation/a_priori/replay_common_fdns_path.py",
    "head-interventions": "research/evaluation/mechanism/analyze_head_interventions.py",
    "score-interventions": "research/evaluation/mechanism/compare_intervention_sources.py",
    "verify-ngm4": "research/closures/ngm4/verify_stress_equivalence.py",
}
MODELS = {
    "ngm4": "Clark4", "dsf-aug": "Aug", "dsf-sym": "Fresh-C4",
    "dsf-trend": "Parameter-PC", "dsf-hold": "Parameter-Hold",
}
FORCED_OPERATOR_TEST = (
    "research/JCP_DSF_TemporalSym_20261117/FORCED_01_SHORT_16_20_20261122/test_forced.py"
)


def normalize_model(args):
    normalized = args.copy()
    for index, arg in enumerate(args):
        if arg == "--model" and index + 1 < len(args):
            value = args[index + 1]
            normalized[index + 1] = MODELS.get(value.lower(), value)
        elif arg.startswith("--model="):
            value = arg.partition("=")[2]
            normalized[index] = "--model=" + MODELS.get(value.lower(), value)
    return normalized


def run_python(args, directory):
    return subprocess.run([sys.executable, "-B", *args], cwd=directory, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the experimental entry points and source files")
    for name, help_text in (("prepare", "Create the isolated runtime source tree"),
                            ("run", "Run an experiment or a scoring script"),
                            ("test", "Run the history and Forced-01 operator tests")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
        if name == "run":
            command.add_argument("entry", choices=ENTRIES)
            command.add_argument("options", nargs=argparse.REMAINDER)
        elif name == "test":
            command.add_argument("--with-assets", action="store_true",
                                 help="Also test with frozen configuration and weights in the workspace")
    args = parser.parse_args()
    if args.command == "list":
        for entry, source in ENTRIES.items():
            print(f"{entry:28} {source}")
        return 0

    try:
        workspace = prepare_workspace(args.workspace)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "prepare":
        print(workspace)
        return 0
    if args.command == "test":
        status = run_python([str(ROOT / "tools/smoke_test.py"), "--workspace", str(workspace)], workspace)
        if status:
            return status
        tests = ["-m", "pytest", "-p", "no:cacheprovider", "-q", FORCED_OPERATOR_TEST]
        if not args.with_assets:
            tests += ["-k", "not production_les_rhs_uses_filtered_force_and_drag "
                      "and not frozen_aug_rhs_on_manufactured_state"]
        return run_python(tests, workspace)

    old_paths = {new: old for old, new in source_paths().items()}
    options = args.options[1:] if args.options[:1] == ["--"] else args.options
    if args.entry in ("forced-reference", "forced-rollouts"):
        options = normalize_model(options)
    return run_python([old_paths[ENTRIES[args.entry]], *options], workspace)


if __name__ == "__main__":
    sys.exit(main())
