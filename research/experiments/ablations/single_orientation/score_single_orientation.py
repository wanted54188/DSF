"""Score the completed pilot against matched four-orientation archives."""

import numpy as np

import pilot
import pc_common as common
import pc_metrics as metrics

STEPS = range(10, 801, 10)
ARCHIVED_NAMES = {"Parameter-Hold": "DSF-Hold", "Parameter-PC": "DSF-Trend"}
OUT = pilot.RESULT / "analysis"


def write_table(filename, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    common.csvout(OUT / filename, rows, keys)


def verify_single(folder, startup, operator, method):
    """Verify native updates, frozen stage clocks, single-head cache, and snapshots."""
    steps = sorted((folder / "steps").glob("*.npz"))
    assert len(steps) == 799 and steps[-1].stem == "0800"
    result = {}
    previous = startup["omega1"]
    max_rk = max_budget = max_source = 0.0
    assert metrics.verify_step(startup["omega0"], previous, startup["rhs"], operator)[
        "native_RK4_max_abs_replay_error"] < 1e-11
    checked_anchors = {}
    expected_CNN = 0
    for step, file in enumerate(steps, start=2):
        saved = common.load(file)
        info = common.meta(saved)
        step_start = step - 1
        assert file.stem == f"{step:04d}"
        assert (info["accepted_step"], info["step_start"]) == (step, step_start)
        assert saved["rhs"].shape == (4, 100, 100) and np.isfinite(saved["omega"]).all()
        stages = info["stages"]
        assert len(stages) == 4
        history = info["bank"]
        expected_latest = 1 + ((step_start - 1) // 5) * 5
        assert [a["step"] for a in history["active"]] == [expected_latest - 1, expected_latest]
        pending = history["pending"]
        if step_start % 5 == 0:
            assert pending["step"] == step_start
        else:
            assert pending is None
        expected_CNN += int(step_start > 1 and step_start % 5 in (0, 1))
        assert info["calls"]["CNN"] == expected_CNN
        for stage, fraction in zip(stages, (0., .5, .5, 1.)):
            assert abs(stage["time_s"] - (1 + common.H * (step_start + fraction))) < 1e-12
            assert stage["tau"] == step_start + fraction - expected_latest
            assert stage["passed"]
        for anchor in history["active"] + ([pending] if pending else []):
            if anchor["file"] in checked_anchors:
                continue
            path = pilot.RESULT / anchor["file"]
            assert common.sha(path) == anchor["sha256"]
            head = common.load(path)["heads"]
            assert head.shape == (7, 100, 100) and np.isfinite(head).all()
            checked_anchors[anchor["file"]] = head
        check = metrics.verify_step(previous, saved["omega"], saved["rhs"], operator)
        max_rk = max(max_rk, check["native_RK4_max_abs_replay_error"])
        max_budget = max(max_budget, *(abs(v) for k, v in check.items() if k.endswith("_algebra_residual")))
        if step_start in STEPS:
            prefix = "observation__"
            snapshot = {k.removeprefix(prefix): value for k, value in saved.items() if k.startswith(prefix)}
            assert snapshot["predicted_heads"].shape == (1, 7, 100, 100)
            assert np.array_equal(snapshot["omega"], previous)
            first, last = [checked_anchors[anchor["file"]] for anchor in history["active"]]
            tau = step_start - expected_latest
            target = last if method == "Parameter-Hold" else (1 + tau) * last - tau * first
            np.testing.assert_allclose(snapshot["predicted_heads"][0], target, rtol=0, atol=1e-13)
            _, _, checks, _ = metrics.snapshot(snapshot, operator, float(operator["nu"]), {})
            max_source = max(max_source, checks["Source_max_abs_reconstruction"])
        previous = saved["omega"]
        if step in STEPS:
            result[step] = previous
    endpoint = common.load(folder / "ENDPOINT.npz")
    assert np.array_equal(endpoint["omega"], result[800])
    assert endpoint["predicted_heads"].shape == (1, 7, 100, 100)
    _, _, checks, _ = metrics.snapshot(endpoint, operator, float(operator["nu"]), {})
    max_source = max(max_source, checks["Source_max_abs_reconstruction"])
    return result, dict(native_steps_checked=len(steps) + 1, snapshots_checked=80,
                        max_rk_replay_error=max_rk, max_budget_residual=max_budget,
                        max_source_reconstruction=max_source)


def scoring_rows(states, references, operator, seed, label):
    rows = []
    for step in STEPS:
        rows.extend(metrics.state_rows(states[step], references[step], operator, dict(
            case=pilot.CASE, seed=seed, model=label, step=step, time=1 + step * common.H,
        )))
    return rows


def summary(rows):
    whole = [r for r in rows if r["band"] == "all"]
    high = [r for r in rows if r["band"] == "high"]
    assert len(whole) == len(high) == 80
    return dict(
        full_omega_MSE=float(np.mean([r["omega_MSE"] for r in whole])),
        t5_omega_MSE=whole[-1]["omega_MSE"],
        t5_omega_relative_L2=whole[-1]["omega_relative_L2"],
        full_u_MSE=float(np.mean([r["u_MSE"] for r in whole])),
        full_v_MSE=float(np.mean([r["v_MSE"] for r in whole])),
        mean_abs_K_bias=float(np.mean([abs(r["K_relative_bias"]) for r in whole])),
        mean_abs_Z_bias=float(np.mean([abs(r["Z_relative_bias"]) for r in whole])),
        mean_high_Z_ratio=float(np.mean([r["Z"] / r["reference_Z"] for r in high])),
    )


def report_table(summary_rows, comparisons):
    lines = [
        "# IC104 单朝向历史模型小实验", "",
        "对照使用 IC104 的同一 100² 涡量初态、RK4、h=0.005、相同 K33 投影和存档参考。"
        "主指标为 t=1.05 至 5 的 80 帧等权绝对涡量 MSE。现用参考状态为 REFERENCE_NOT_CONVERGED。", "",
        "Single-Hold 和 Single-Trend 均使用原始未旋转增强的 FULL-C4 检查点。"
        "每对历史以 0.025 为周期、0.005 为锚点间隔，第二帧到达后更新。"
        "启动和刷新只计算原朝向 head。阶段内用当前状态重算梯度、Clark4 支撑和限制器。", "",
        "|网络种子|模型|全窗涡量 MSE|t=5 相对 L2|模型特异 CNN 次数|执行状态|",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        mse = f'{row["full_omega_MSE"]:.8f}' if "full_omega_MSE" in row else "—"
        rel = f'{row["t5_omega_relative_L2"]:.6f}' if "t5_omega_relative_L2" in row else "—"
        lines.append(f'|{row["seed"]}|{row["model"]}|{mse}|{rel}|{row["CNN"]}|{row["status"]}|')
    lines.extend(["", "|网络种子|同规则配对|单朝向 MSE 相对变化|相对该种子最佳四朝向方案|",
                  "|---:|---|---:|---:|"])
    for row in comparisons:
        change_from_best = row.get("MSE_change_from_best_four_pct")
        best = (f'{row["best_four_orientation"]} ({change_from_best:+.2f}%)'
                if change_from_best is not None else "—")
        lines.append(f'|{row["seed"]}|{row["candidate"]} 对 {row["baseline"]}|{row["MSE_change_pct"]:+.2f}%|{best}|')
    wins = sum(r["MSE_change_pct"] < 0 for r in comparisons)
    lines += [
        "", f"全窗 MSE 更低的单朝向方案出现于 {wins}/{len(comparisons)} 组完整同规则配对。"
        "正百分比表示单朝向误差升高。两个网络种子共享同一个初态，结果范围限于 IC104。", "",
        "两套部署同时改变检查点与推理朝向数，启动首步也分别用各自权重。"
        "因此上述差异评价这两套完整部署的表现，单独的旋转平均效应需使用同权重对照。"
        "表中 CNN 次数为模型特异循环与终点诊断所用调用数。"
        "每种子公共启动在单朝向部署和原四朝向部署中分别消耗 6 和 12 次 CNN。"
        "各批次的实测时间记录于 CSV，受机器负载及诊断成本差异影响。", "",
        "运行身份及冻结检查点 SHA 见上一级 RUN.json。"
        "TIME_SERIES.csv、COMPARISONS.csv 和 INDEPENDENT_CHECKS.csv 提供误差序列、配对差额和重构检查。",
    ]
    (OUT / "REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    common.HERE = pilot.HERE
    run = common.read(pilot.RESULT / "RUN.json")
    assert not run["rotational_augmentation"] and run["inference_orientations"] == [0]
    checked_files = {}
    for path, sha in run["input_sha256"].items():
        pilot.checked(pilot.ROOT / path, sha, checked_files)
    archive = common.read(pilot.OLD / "LATEST_REPORT.json")
    report = pilot.OLD / archive["path"]
    assert common.sha(report / "MANIFEST.json") == archive["manifest_sha256"]
    manifest = common.read(report / "MANIFEST.json")
    pilot.checked(report / "WINDOW_SUMMARY.csv", manifest["outputs"]["WINDOW_SUMMARY.csv"], checked_files)
    operator = common.load(pilot.RESULT / "OPERATORS.npz")
    archived_operator = common.load(pilot.OLD / "OPERATORS.npz")
    assert all(np.array_equal(operator[key], archived_operator[key]) for key in archived_operator)
    reference = common.References(run["reference"])
    truth = {step: reference.at(step)["reference_omega"] for step in STEPS}
    costs = common.csvread(report / "COSTS.csv")
    expected = common.csvread(report / "WINDOW_SUMMARY.csv")
    raw = []
    summaries = []
    audits = []
    comparisons = []
    for weight in run["checkpoint_rows"]:
        seed = weight["seed"]
        startup_dir = pilot.RESULT / "startup" / f"{pilot.CASE}_{seed}"
        if (startup_dir / "DONE.json").exists():
            startup_receipt = common.read(startup_dir / "DONE.json")
            for key, sha in startup_receipt["outputs"].items():
                assert common.sha(startup_dir / key) == sha
            startup = common.load(startup_dir / "STARTUP.npz")
        else:
            startup = None
        for method in pilot.METHODS:
            folder = pilot.RESULT / "trajectories" / common.key(pilot.CASE, seed, method)
            if not (folder / "STATUS.json").exists():
                continue
            archived_file = report / f"STATES_{pilot.CASE}_{seed}_{method}.npy"
            archived = np.load(archived_file, mmap_mode="r", allow_pickle=False)
            if startup:
                np.testing.assert_allclose(archived[0], startup["omega0"], rtol=0, atol=1e-12)
            baseline = ARCHIVED_NAMES[method]
            rows = scoring_rows(archived, truth, operator, seed, baseline)
            raw.extend(rows)
            old_stats = summary(rows)
            archived_mse = next(float(r["mean_omega_MSE"]) for r in expected
                                if (r["case"], r["seed"], r["method"], r["band"], r["window"], r["status"])
                                == (str(pilot.CASE), str(seed), method, "all", "full", "VERIFIED"))
            assert abs(old_stats["full_omega_MSE"] - archived_mse) <= 1e-12
            base_cost = next(r for r in costs if (r["case"], r["seed"], r["method"])
                             == (str(pilot.CASE), str(seed), method))
            summaries.append(dict(case=pilot.CASE, seed=seed, model=baseline,
                                  checkpoint="FULL-C4-Augmented", status="VERIFIED_ARCHIVED",
                                  CNN=int(base_cost["actual_CNN"]), RHS=int(base_cost["actual_RHS"]),
                                  solver_seconds=float(base_cost["solver_seconds"]),
                                  loop_wall_seconds=float(base_cost["measured_loop_seconds"]),
                                  reference_quality=common.QUALITY, **old_stats))
            receipt = common.read(folder / "STATUS.json")
            name = pilot.NAMES[method]
            calls = receipt.get("actual_calls", {})
            candidate = dict(case=pilot.CASE, seed=seed, model=name, checkpoint="FULL-C4 unaugmented",
                             checkpoint_sha=weight["sha256"], status=receipt["status"],
                             CNN=calls.get("CNN"), RHS=calls.get("RHS"),
                             solver_seconds=receipt.get("solver_wall_seconds"),
                             loop_wall_seconds=receipt.get("current_session_wall_seconds"),
                             completed_t_end=1 + common.H * receipt["last_accepted_step"],
                             reference_quality=common.QUALITY)
            summaries.append(candidate)
            if receipt["status"] != "VERIFIED":
                continue
            assert startup is not None
            assert calls == {"CNN": 319, "RHS": 3197}, calls
            fields, checks = verify_single(folder, startup, operator, method)
            audits.append(dict(case=pilot.CASE, seed=seed, model=name, **checks))
            rows = scoring_rows(fields, truth, operator, seed, name)
            raw.extend(rows)
            candidate.update(summary(rows))
            comparisons.append(dict(case=pilot.CASE, seed=seed, candidate=name, baseline=baseline,
                                    candidate_MSE=candidate["full_omega_MSE"],
                                    baseline_MSE=old_stats["full_omega_MSE"],
                                    MSE_change_pct=100 * (candidate["full_omega_MSE"] / old_stats["full_omega_MSE"] - 1),
                                    CNN_change_pct=100 * (candidate["CNN"] / int(base_cost["actual_CNN"]) - 1)))
    for pair in comparisons:
        choices = [row for row in summaries if row["seed"] == pair["seed"] and row["status"] == "VERIFIED_ARCHIVED"]
        if len(choices) == len(pilot.METHODS):
            best = min(choices, key=lambda row: row["full_omega_MSE"])
            pair["best_four_orientation"] = best["model"]
            pair["best_four_MSE"] = best["full_omega_MSE"]
            pair["MSE_change_from_best_four_pct"] = 100 * (pair["candidate_MSE"] / best["full_omega_MSE"] - 1)
    write_table("SUMMARY.csv", summaries)
    write_table("COMPARISONS.csv", comparisons)
    write_table("TIME_SERIES.csv", raw)
    write_table("INDEPENDENT_CHECKS.csv", audits)
    report_table(summaries, comparisons)
    common.dump(OUT / "MANIFEST.json", dict(
        completed_pairs=len(comparisons), qualified_reference=False, models_refit=0, new_CNN=0, new_RHS=0,
        analysis_script_sha256=common.sha(__file__), source_run_sha256=common.sha(pilot.RESULT / "RUN.json"),
        files={p.name: common.sha(p) for p in OUT.iterdir()
               if p.is_file() and p.name != "MANIFEST.json"},
    ))
    print(f"Independently verified and scored {len(comparisons)} completed pairs: {OUT}")


if __name__ == "__main__":
    main()
