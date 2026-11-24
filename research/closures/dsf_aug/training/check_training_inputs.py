"""Full-grid acceptance and bounded RHS / single-sample backward profiling."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True

import torch
from torch import nn

from model import C4Conv, spatial, vector
import study
from train import trainer

P = study.P
require = study.inputs.require


def magnitude(field):
    return float(field.double().square().mean().sqrt())


def relative(actual, expected):
    return magnitude(actual - expected) / max(magnitude(expected), 1e-15)


def projected_source(solver, fields):
    hx = torch.fft.fft2(fields["qx"])
    hy = torch.fft.fft2(fields["qy"])
    h = -1j * (solver.kx * hx + solver.ky * hy) * solver.dealias
    h[0, 0] = 0
    return torch.fft.ifft2(h).real


def physical(solver, state, fields, tolerance):
    require(all(bool(torch.isfinite(x).all()) for x in fields.values()), "Nonfinite FULL field")
    cl = solver.closure
    gx, gy = state["omega_x"], state["omega_y"]
    gnorm = (gx.square() + gy.square() + 1e-16).sqrt()
    leith = fields["c_theta"] * cl.filter_width**3 * gnorm
    a, b, d = (fields[k].double() for k in ("A11", "A12", "A22"))
    half_trace = (a + d) / 2
    radius = torch.hypot((a - d) / 2, b)
    smallest, largest = half_trace - radius, half_trace + radius
    qbnorm = torch.hypot(fields["qb_x"], fields["qb_y"])
    limit = cl.backscatter_factor * cl.maximum_backscatter_ratio * fields["backscatter_support_norm"]
    transfer_limit = cl.maximum_local_backscatter_transfer_ratio * (
        fields["pi_d"].relu() + fields["gradient_support_norm"] * gnorm)
    errors = dict(
        leith_relative=relative(fields["nu_d"], leith),
        trace_max_abs=float((2*half_trace - 2).abs().max()),
        qd_x_relative=relative(fields["qd_x"], -leith * (fields["A11"] * gx + fields["A12"] * gy)),
        qd_y_relative=relative(fields["qd_y"], -leith * (fields["A12"] * gx + fields["A22"] * gy)),
        flux_reconstruction_x_relative=relative(fields["qx"], fields["qd_x"] + fields["qb_x"]),
        flux_reconstruction_y_relative=relative(fields["qy"], fields["qd_y"] + fields["qb_y"]),
        signed_norm_excess_relative=float(((qbnorm - limit).relu() / limit.clamp_min(1e-24)).max()),
        signed_transfer_excess_relative=float(((-fields["pi_b"] - transfer_limit).relu()
                                              / transfer_limit.clamp_min(1e-24)).max()),
        source_relative=relative(study.source(solver, fields), projected_source(solver, fields)),
    )
    require(max(errors.values()) <= tolerance, f"Constitutive identity or envelope failure: {errors}")
    require(float(smallest.min()) > 0 and float((largest/smallest).max()) <= 16.0, "SPD condition bound failed")
    require(float(fields["pi_d"].min()) >= -tolerance*float(fields["pi_d"].abs().max())
            and float(fields["nu_d"].min()) >= 0, "Negative dissipative transfer")
    require(bool(((fields["backscatter_transfer_limiter"] >= 0)
                  & (fields["backscatter_transfer_limiter"] <= 1)).all()), "Invalid limiter")
    return dict(**errors, minimum_eigenvalue=float(smallest.min()),
                max_condition=float((largest/smallest).max()),
                minimum_dissipative_transfer=float(fields["pi_d"].min()),
                limiter_active=int((fields["backscatter_transfer_limiter"] < 1).sum()))


def initialise_test_heads(solver):
    # The trainable output layer normally starts at zero. Exercise nonzero
    # tensor and vector heads with a deterministic, test-only perturbation.
    generator = torch.Generator(device="cpu").manual_seed(7109)
    head = solver.closure.output.conv.weight
    with torch.no_grad():
        head.copy_(.035 * torch.randn(head.shape, generator=generator, dtype=head.dtype).to(head.device))


@torch.no_grad()
def covariance(solver, w, tolerance, physical_tolerance):
    cl = solver.closure
    initialise_test_heads(solver)
    cl.set_backscatter_factor(1)
    state = solver.state(w)
    base = solver.closure_fields(state)
    check = physical(solver, state, base, physical_tolerance)
    src = study.source(solver, base)
    errors = []
    for turn in range(4):
        transformed = spatial(w, turn)
        new_state = solver.state(transformed)
        out = solver.closure_fields(new_state)
        row = {"degrees": 90 * turn}
        for name in ("c_theta", "nu_d", "backscatter_gate", "backscatter_support_norm",
                     "gradient_support_norm", "backscatter_transfer_limiter", "pi_d", "pi_b", "pi"):
            row[name] = relative(out[name], spatial(base[name], turn))
        a, b, d = (spatial(base[name], turn) for name in ("A11", "A12", "A22"))
        expected = torch.stack((a, b, d) if turn % 2 == 0 else (d, -b, a))
        row["SPD"] = relative(torch.stack((out["A11"], out["A12"], out["A22"])), expected)
        for name, x, y in (("qd", "qd_x", "qd_y"), ("qb", "qb_x", "qb_y"),
                           ("q", "qx", "qy"), ("analytic_C4_support", "gradient_support_x", "gradient_support_y")):
            row[name] = relative(torch.stack((out[x], out[y])), torch.stack(vector(base[x], base[y], turn)))
        row["Source"] = relative(study.source(solver, out), spatial(src, turn))
        require(max(v for k, v in row.items() if k != "degrees") <= tolerance, f"Equivariance failure: {row}")
        errors.append(row)
    require(magnitude(base["qd_x"]) > 0 and magnitude(base["qb_x"]) > 0
            and magnitude(base["A12"]) > 0, "Degenerate covariance test")
    cl.set_backscatter_factor(0)
    warm = solver.closure_fields(state)
    require(torch.count_nonzero(warm["qb_x"]) == 0 and torch.count_nonzero(warm["qb_y"]) == 0,
            "Original signed warmup disabled neither component")
    cl.set_backscatter_factor(1)
    zero = solver.closure_fields(solver.state(torch.zeros_like(w)))
    require(all(bool(torch.isfinite(x).all()) for x in zero.values())
            and magnitude(study.source(solver, zero)) <= physical_tolerance, "Zero-state Source")
    h = torch.fft.fft2(src, norm="forward")
    mask = solver.dealias.clone()
    mask[0, 0] = False
    require(float(torch.linalg.vector_norm(h[~mask])) <= tolerance*magnitude(src), "Source outside K33")
    return dict(turns=errors, physical=check, zero_source_rms=magnitude(study.source(solver, zero)))


@torch.no_grad()
def limiter_stress(solver, w, tolerance):
    # Anisotropic dissipation nearly transverse to g, with signed flux along g.
    # Only this pointwise test uses prescribed local gradients and head outputs.
    state = solver.state(w)
    for key in ("u_x", "u_y", "v_x", "v_y", "omega_y"):
        state[key] = torch.zeros_like(w)
    state["omega_x"] = torch.ones_like(w)
    raw = w.new_tensor([20., -20., 12., 0., 12., 12., 0.]).view(1, 7, 1, 1)
    hook = solver.closure.output.register_forward_hook(lambda _m, _i, value: raw.expand_as(value))
    try:
        out = solver.closure_fields(state)
    finally:
        hook.remove()
    check = physical(solver, state, out, tolerance)
    require(check["limiter_active"] == w.numel() and check["max_condition"] > 15.99,
            "Stress case did not exercise limiter and SPD cap")
    return check


def jvp_fd(solver, w):
    solver.requires_grad_(False)
    direction = solver.project(torch.roll(w, shifts=(7, 13), dims=(-2, -1)) - .37 * w)
    direction = direction * (magnitude(w) / magnitude(direction))

    def forward(state):
        fields = solver.closure_fields(solver.state(state))
        q = [fields[key] for key in ("qd_x", "qd_y", "qb_x", "qb_y")]
        return torch.stack((*q, study.source(solver, fields)))

    rows = []
    with torch.no_grad():
        for name, tangent in (("amplitude", w), ("mixed", direction)):
            value, derivative = torch.func.jvp(forward, (w,), (tangent,))
            require(relative(value, forward(w)) < 1e-12, "JVP primal mismatch")
            _, rotated = torch.func.jvp(forward, (spatial(w, 1),), (spatial(tangent, 1),))
            expected = torch.stack((*vector(derivative[0], derivative[1], 1),
                                    *vector(derivative[2], derivative[3], 1), spatial(derivative[4], 1)))
            rotation_error = relative(rotated, expected)
            require(rotation_error < P["equivariance_relative_fp64"], "Rotated state JVP mismatch")
            scales = []
            for epsilon in (1e-3, 1e-4, 1e-5, 1e-6):
                plus, minus = w + epsilon*tangent, w - epsilon*tangent
                fd = (forward(plus) - forward(minus)) / (2*epsilon)
                plus_fields, minus_fields = (solver.closure_fields(solver.state(x)) for x in (plus, minus))
                limiter_switches = (plus_fields["backscatter_transfer_limiter"] < 1) != (
                    minus_fields["backscatter_transfer_limiter"] < 1)
                relu_switches = (plus_fields["pi_b"] < 0) != (minus_fields["pi_b"] < 0)
                scales.append(dict(epsilon=epsilon, qd=relative(fd[:2], derivative[:2]),
                                   qb=relative(fd[2:4], derivative[2:4]),
                                   Source=relative(fd[4], derivative[4]),
                                   limiter_switches=int(limiter_switches.sum()), relu_sign_switches=int(relu_switches.sum())))
            require(all(min(row[key] for row in scales) < P["jvp_relative"] for key in ("qd", "qb", "Source")),
                    f"State JVP/FD failure: {scales}")
            rows.append(dict(direction=name, rotated_jvp_relative=rotation_error, ladder=scales))
    return rows


@torch.no_grad()
def data_check(cache, registry):
    solver = study.make_solver(study.AUGMENTED, 2026)
    w = cache["omega"][0]
    reference = study.common.prior.c4
    tolerance = study.common.protocol()["reference_adapter_relative_tolerance"]
    checks = []
    for turn in range(4):
        sample = study.rotate_frame(cache, 0, turn)
        inverse = study.rotate_frame(sample, 0, -turn)
        require(all(torch.equal(inverse[key][0], cache[key][0]) for key in sample), "Rotation round trip")
        state = solver.state(sample["omega"][0])
        fourth = reference.fourth_increment(solver, state, solver.closure.filter_width)
        prepared = reference.source(solver, (sample["qx"][0], sample["qy"][0]))
        pi = -(sample["qx"] * sample["omega_x"] + sample["qy"] * sample["omega_y"])
        errors = {k: relative(sample[k][0], state[k]) for k in study.data.STATE if k != "omega"}
        errors.update(fourth=relative(torch.stack(fourth), torch.stack((sample["fourth_x"][0], sample["fourth_y"][0]))),
                      reference_source=relative(prepared, sample["source"][0]), signed_transfer=relative(pi, sample["pi"]))
        require(max(errors.values()) < tolerance, f"Cached rotation does not match production operators: {errors}")
        checks.append(dict(degrees=90*turn, **errors))
    for frame in range(len(cache["time_s"])):
        histogram = [study.augmentation_turn(frame, epoch) for epoch in range(1, 101)]
        require(all(histogram.count(turn) == 25 for turn in range(4)), "Unbalanced augmentation exposure")
    initial_hashes = {}
    for seed in P["seeds"]:
        model = study.make_solver(study.AUGMENTED, seed)
        done = study.inputs.read(study.inputs.RUN / "training" / f"FULL-C4_{seed}" / "DONE.json")
        row = next(r for r in registry["models"] if (r["mode"], r["seed"]) == ("FULL-C4", seed))
        digest = study.common.model_hash(model)
        require(done["best_sha256"] == row["sha256"] and digest == done["initial_parameter_sha256"],
                "Augmented initialization differs from original FULL-C4")
        initial_hashes[str(seed)] = digest
    return dict(operator_replay=checks, original_initial_parameter_sha256=initial_hashes)


def check_training_graph(cache, mode, seed, device):
    solver = study.make_solver(mode, seed, device)
    frame = {name: data[:1].to(device) for name, data in cache.items()}
    with torch.no_grad():
        live = study.source(solver, solver.closure_fields(solver.state(frame["omega"][0])))
    study.data.enable_cached_prior(solver)
    with torch.no_grad():
        _, _, fast = study.data.forward_frame(solver, frame, 0)
        error = relative(live, fast)
    tolerance = study.common.protocol()["reference_adapter_relative_tolerance"]
    require(error <= tolerance, f"Training adapter replay failure on {device}: {error}")
    solver.train()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    terms = study.training_quantities(solver, frame, 0, 12, mode)
    weights = study.common.protocol()["loss_weights"]
    require(set(terms) == set(weights), "Changed original loss terms")
    loss = study.objective.total(terms, weights)
    require(bool(torch.isfinite(loss)), "Nonfinite original training objective")
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in solver.parameters()),
            "Original objective did not produce finite gradients for every parameter")
    return dict(mode=mode, seed=seed, device=device, cached_live_relative=error,
                loss=float(loss.detach()), optimizer_steps=0,
                single_sample_backward_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)


@torch.inference_mode()
def profile(cache, mode, device):
    if device == "cuda":
        torch.cuda.empty_cache()
    solver = study.make_solver(mode, 2026, device).eval()
    w = solver.project(cache["omega"][0].to(device))
    n = P["grid"]
    macs = 0
    for module in solver.modules():
        if isinstance(module, nn.Conv2d):
            macs += module.weight.numel()*n*n
        if isinstance(module, C4Conv):
            macs += 4*module.weight.numel()*n*n
    for _ in range(P["profile_warmup_rhs"]):
        solver.rhs(w, 1.)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(P["profile_measured_rhs"]):
        start = time.perf_counter()
        result = solver.rhs(w, 1.)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed.append(1e3*(time.perf_counter()-start))
        require(bool(torch.isfinite(result["rhs"]).all()), "Nonfinite profiled RHS")
        del result
    return dict(mode=mode, device=device, trainable_parameters=sum(p.numel() for p in solver.parameters()),
                convolution_MAC_per_rhs=macs, convolution_FLOP_per_rhs=2*macs,
                measured_rhs=P["profile_measured_rhs"], warmup_rhs=P["profile_warmup_rhs"],
                rhs_median_ms=statistics.median(elapsed), rhs_min_ms=min(elapsed), rhs_max_ms=max(elapsed),
                rhs_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                rhs_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="accept01")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    require(Path(args.run).name == args.run and args.run not in ("", ".", ".."), "Invalid run name")
    out = study.safe_output(study.HERE / "results" / args.run)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAIL", torch_version=str(torch.__version__), code_sha256=study.code_identity(),
                  reference_quality=P["reference_quality"], optimizer_steps=0, rollout_steps=0)
    try:
        require(not args.cuda or torch.cuda.is_available(), "CUDA acceptance requested but unavailable")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        audit, registry = study.verify_baseline(cases=(study.CASES[0],))
        cache = study.inputs.cache(study.CASES[0])
        trainer()
        report["training_data"] = data_check(cache, registry)
        report["baseline"] = dict(code_sha256=audit["code_sha256"], cache_sha256=registry["cache_sha256"],
                                  config_sha256=study.inputs.sha(study.common.CONFIG))
        tests = []
        for seed in P["seeds"]:
            solver = study.make_solver(study.EQUIVARIANT, seed).double()
            w = solver.project(cache["omega"][0].double())
            tests.append(dict(device="cpu", dtype="float64", seed=seed,
                              covariance=covariance(solver, w, P["equivariance_relative_fp64"],
                                                    P["constraint_relative_fp64"]),
                              stress=limiter_stress(solver, w, P["constraint_relative_fp64"]),
                              jvp_fd=jvp_fd(solver, w)))
        report["float64_tests"] = tests
        devices = ["cpu", "cuda"] if args.cuda else ["cpu"]
        tests = []
        for device in devices:
            for seed in P["seeds"]:
                solver = study.make_solver(study.EQUIVARIANT, seed, device)
                w = solver.project(cache["omega"][0].to(device))
                tests.append(dict(device=device, dtype="float32", seed=seed,
                                  covariance=covariance(solver, w, P["equivariance_relative_fp32"],
                                                        P["constraint_relative_fp32"]),
                                  stress=limiter_stress(solver, w, P["constraint_relative_fp32"])))
        report["float32_tests"] = tests
        del solver, w
        print("PASS full-grid covariance, Source, limiter and state JVP/FD", flush=True)
        report["training_graph"] = [
            check_training_graph(cache, mode, seed, device)
            for device in devices for mode in P["modes"] for seed in P["seeds"]
        ]
        report["profiles"] = [profile(cache, mode, device) for device in devices for mode in P["modes"]]
        report.update(status="PASS", accepted_training_devices=devices,
                      cuda_device=torch.cuda.get_device_name(0) if args.cuda else None,
                      single_original_cache_bytes=sum(x.numel()*x.element_size() for x in cache.values()))
        study.write_json(out / "TEST_ACCEPTANCE.json", report)
        study.write_json(out / "TRAINING_CONFIGS.json", [
            dict(mode=mode, seed=seed, protocol=study.effective_protocol(audit["protocol"], mode),
                 cache_sha256=registry["cache_sha256"], updates=P["expected_updates"])
            for mode in P["modes"] for seed in P["seeds"]
        ])
        print(json.dumps(report["profiles"], indent=2), flush=True)
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        study.write_json(out / "FAILURE.json", report)
        raise


if __name__ == "__main__":
    main()
