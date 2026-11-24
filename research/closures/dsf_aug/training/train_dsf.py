"""Manual-only entry for the four fixed training runs."""

import argparse
import ast
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

sys.dont_write_bytecode = True

import torch

import study


def trainer():
    """Bind the original training loop to local outputs and the paired sample adapter."""
    original = study.native_training.train_one
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    target_forward = ast.dump(ast.parse("st,f,source=data.forward_frame(s,cache,int(i))").body[0])
    target_quantity = ast.dump(ast.parse("q=objective.quantities(s,cache,int(i),st,f,source)").body[0])

    class SampleAdapter(ast.NodeTransformer):
        def __init__(self):
            self.forward = 0
            self.quantity = 0

        def visit_Assign(self, node):
            if ast.dump(node) == target_forward:
                self.forward += 1
                return None
            if ast.dump(node) == target_quantity:
                self.quantity += 1
                return ast.parse("q=training_quantities(s,cache,int(i),epoch,mode)").body[0]
            return node

    adapter = SampleAdapter()
    tree = adapter.visit(tree)
    study.inputs.require((adapter.forward, adapter.quantity) == (1, 1), "Original training loop changed")
    writer = SimpleNamespace(
        make_solver=study.make_solver, model_hash=study.common.model_hash, CASES=study.CASES,
        read=study.inputs.read, sha=study.inputs.sha, dump=study.write_json, save_pt=study.save_pt,
    )
    scope = dict(original.__globals__, c=writer, training_quantities=study.training_quantities)
    exec(compile(ast.fix_missing_locations(tree), __file__ + ":original_train_one", "exec"), scope)
    return scope["train_one"]


def require_acceptance(path, device, code):
    path = study.safe_output(path)
    receipt = study.inputs.read(path)
    if (receipt["status"] != "PASS" or device not in receipt["accepted_training_devices"]
            or receipt["code_sha256"] != code or receipt["torch_version"] != str(torch.__version__)):
        raise RuntimeError("A matching PASS acceptance receipt is required on the requested device")
    if device == "cuda" and receipt["cuda_device"] != torch.cuda.get_device_name(0):
        raise RuntimeError("Run acceptance on this CUDA device before training")
    return study.inputs.sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=study.P["modes"], required=True)
    parser.add_argument("--seed", type=int, choices=study.P["seeds"], required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run", default="paired01")
    parser.add_argument("--acceptance", type=Path,
                        default=study.HERE / "results/accept01/TEST_ACCEPTANCE.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    effective = study.effective_protocol(study.common.protocol(), args.mode)
    if not args.execute:
        print(json.dumps(dict(status="PLAN_ONLY", mode=args.mode, seed=args.seed,
                              expected_updates=study.P["expected_updates"], protocol=effective), indent=2))
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on this host")
    code = study.code_identity()
    acceptance_sha = require_acceptance(args.acceptance, args.device, code)
    audit, registry = study.verify_baseline()
    study.inputs.require(Path(args.run).name == args.run and args.run not in ("", ".", ".."), "Invalid run name")
    out = study.safe_output(study.HERE / "results/training" / args.run)
    torch.set_num_threads(effective["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = effective["TF32"]
    torch.backends.cudnn.allow_tf32 = effective["TF32"]
    train_one = trainer()
    train_cache, val_cache = (
        study.data.load_cache(audit, study.inputs.RUN, case, args.device) for case in study.CASES
    )
    if (out / "training" / f"{args.mode}_{args.seed}").exists() and not args.resume:
        raise FileExistsError("Existing run. Use --resume or choose a new run name.")
    run_identity = dict(
        study=study.P["version"], code_sha256=code, baseline_code_sha256=audit["code_sha256"],
        baseline_config_sha256=study.inputs.sha(study.common.CONFIG),
        baseline_registry_sha256=study.inputs.sha(study.inputs.RUN / "BASELINE_REGISTRY.json"),
        cache_sha256=registry["cache_sha256"], acceptance_sha256=acceptance_sha, device=args.device,
    )
    receipt_path = out / "STUDY.json"
    if receipt_path.exists():
        study.inputs.require(study.inputs.read(receipt_path) == run_identity, "Study run identity changed")
    else:
        study.write_json(receipt_path, run_identity)
    for case in study.CASES:
        local = out / "data" / str(case) / "DONE.json"
        reference = dict(cache_sha256=registry["cache_sha256"][str(case)],
                         source=str(study.inputs.RUN / "data" / str(case) / "CACHE.pt"))
        if local.exists():
            study.inputs.require(study.inputs.read(local) == reference, "Local cache receipt changed")
        else:
            study.write_json(local, reference)
    identity = dict(protocol=effective, code_sha256=code)
    train_one(identity, out, args.mode, args.seed, train_cache, val_cache, args.device, resume=args.resume)
    result = study.inputs.read(out / "training" / f"{args.mode}_{args.seed}" / "DONE.json")
    study.inputs.require(result["optimizer_updates"] == study.P["expected_updates"]
                         and result["training_forwards"] == study.P["expected_train_forwards"],
                         "Completed run differs from fixed training budget")


if __name__ == "__main__":
    main()
