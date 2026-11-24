"""Exercise paired-anchor promotion and reference-free budget counts."""

import argparse
from pathlib import Path
import sys
from workspace import DEFAULT_WORKSPACE, prepare_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    args = parser.parse_args()
    root = prepare_workspace(args.workspace)
    sys.path.insert(0, str(root / "research/full_aug_parameter_pc_full_v1_20261116"))
    from pc_tests import bank_test, budget_test

    bank_test()
    budget_test()
    print("Anchor promotion, stage history and invocation counts passed.")


if __name__ == "__main__":
    main()
