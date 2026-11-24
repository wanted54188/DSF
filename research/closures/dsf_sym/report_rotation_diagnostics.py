"""Postprocess the committed RK4 results without model calls."""

import argparse
import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pair import HERE, SNAPSHOT, CASES, SEEDS, C4, BASE, AUG, EQ, ORBIT, read, sha, require, csvout, dump

LABELS = {C4: "Clark4", BASE: "Original FULL", AUG: "Augmented", EQ: "Equivariant", ORBIT: "Four-rotation mean"}
COLORS = {C4: "#333333", BASE: "#758195", AUG: "#AE641E", EQ: "#166F8A", ORBIT: "#78479C"}
MODELS = (C4, BASE, AUG, EQ, ORBIT)
BANDS = ("all", "low", "middle", "tail")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    for row in table:
        for key, value in row.items():
            if key not in ("model", "band"):
                row[key] = float(value) if value else None
        row["case"], row["seed"] = int(row["case"]), int(row["seed"])
    return table


def gain(value, baseline):
    return 100*(1-value/baseline)


def one_step_summary(table):
    groups = {}
    for row in table:
        key = tuple(row[k] for k in ("case", "seed", "model", "band"))
        groups.setdefault(key, []).append(row)
    means = {k: float(np.mean([r["defect_MSE"] for r in v])) for k, v in groups.items()}
    return [dict(case=c, seed=s, model=m, band=b, frames=len(groups[c, s, m, b]),
                 mean_defect_MSE=value, gain_vs_Clark4_pct=gain(value, means[c, s, C4, b]),
                 gain_vs_original_pct=gain(value, means[c, s, BASE, b]))
            for (c, s, m, b), value in means.items()]


def plot_window(table, out):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.6), sharex=True, layout="constrained")
    panels = (("all", "omega_L2", "All-band vorticity L2 (%)"),
              ("middle", "omega_L2", "Mid-band vorticity L2 (%)"),
              ("tail", "Z_bias", "Tail enstrophy bias (%)"))
    for col, case in enumerate(CASES):
        axes[0, col].set_title(f"{str(case)[-3:]} reference flow")
        for row, (band, key, label) in enumerate(panels):
            ax = axes[row, col]
            if col == 0:
                ax.set_ylabel(label)
            if key == "Z_bias":
                ax.axhline(0, color="#999999", lw=.8)
            for model in MODELS:
                for i, seed in enumerate(SEEDS[:1] if model == C4 else SEEDS):
                    samples = [r for r in table if (r["case"], r["seed"], r["model"], r["band"])
                               == (case, seed, model, band)]
                    if samples:
                        ax.plot([r["time_s"] for r in samples], [100*r[key] for r in samples],
                                color=COLORS[model], lw=1.4, ls=":" if model == ORBIT else ("-" if i == 0 else "--"),
                                label=LABELS[model] if i == 0 else None)
            ax.grid(axis="y", alpha=.22)
        axes[-1, col].set_xlabel("Time")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    axes[0, -1].legend(handles, labels, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Common-state RK4 · solid: seed 2026 · dashed: seed 2028", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"PAIRED_TRAJECTORIES.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def endpoint_table(lines, data, name):
    lines.extend(["", f"## {name}", "", "| 流动 | 种子 | 模型 | 全频 L2 | 中频 L2 | 谱尾 L2 | "
                  "全频 MSE 改善 | 谱尾 Z 偏差 | 谱尾 Z 最大超额 | K 偏差 | Z 偏差 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    lookup = {(r["case"], r["seed"], r["model"], r["band"]): r for r in data}
    for case in CASES:
        for seed in SEEDS:
            for model in MODELS:
                if (case, seed, model, "all") not in lookup:
                    continue
                full, mid, tail = (lookup[case, seed, model, band] for band in ("all", "middle", "tail"))
                base = lookup[case, seed, C4, "all"]
                lines.append(f"| {str(case)[-3:]} | {seed} | {LABELS[model]} | {100*full['omega_L2']:.4f}% | "
                             f"{100*mid['omega_L2']:.3f}% | {100*tail['omega_L2']:.2f}% | "
                             f"{gain(full['MSE'], base['MSE']):+.2f}% | {100*tail['Z_bias']:+.2f}% | "
                             f"{tail['peak_positive_Z_excess']:.3e} | {100*full['K_bias']:+.4f}% | "
                             f"{100*full['Z_bias']:+.4f}% |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run01")
    parser.add_argument("--report-id", default="report01")
    args = parser.parse_args()
    require(all(Path(x).name == x and x not in ("", ".", "..") for x in (args.run, args.report_id)), "Invalid name")
    folder = HERE / "results" / args.run
    manifest = read(folder / "MANIFEST.json")
    require(manifest["status"] == "PASS", "Only a completed run can be reported")
    for file, expected in manifest["output_sha256"].items():
        require(sha(folder / file) == expected, f"Result changed: {file}")
    snapshot = read(SNAPSHOT / "CHECKPOINTS.json")
    for entry in snapshot["checkpoints"]:
        require(sha(HERE.parents[1] / entry["file"]) == entry["sha256"], "Frozen checkpoint changed")
    out = HERE / "reports" / args.run / args.report_id
    out.mkdir(parents=True, exist_ok=False)
    step = one_step_summary(rows(folder / "ONE_STEP.csv"))
    csvout(out / "ONE_STEP_SUMMARY.csv", step)
    short = rows(folder / "SHORT_ENDPOINT.csv")
    extended = manifest["extension_status"] == "COMPLETED"
    extension = rows(folder / "EXTENSION_ENDPOINT.csv") if extended else []
    plot_data = rows(folder / ("EXTENSION_TIMEWISE.csv" if extended else "SHORT_TIMEWISE.csv"))
    if extended:
        plot_data += [r for r in rows(folder / "SHORT_TIMEWISE.csv") if r["model"] == ORBIT]
    plot_window(plot_data, out)

    lines = ["# 冻结 checkpoint 的配对动力学验证", "",
             "101、102 各用两个种子，初态从同一保存缓存投影。所有新比较为生产 RK4、float64 和 dt=0.005。"
             "参考为原 n1536_half 精确时刻归档，质量保留 **REFERENCE_NOT_CONVERGED**。", "",
             "四个新 checkpoint 由原验证规则选出，使用独立 SHA 快照。", "",
             "| 模型 | 种子 | 固定 epoch | SHA-256 前 12 位 |", "|---|---:|---:|---|"]
    for r in snapshot["checkpoints"]:
        if r["mode"] != BASE:
            lines.append(f"| {r['mode']} | {r['seed']} | {r['epoch']} | {r['sha256'][:12]} |")
    lines.extend(["", "## 参考态 RK4 一步缺陷", "",
                  "在 t=1、1.5、2.5、3.5、4.5 启动一步，模型增量减同一时段参考增量。"
                  "以下为五帧等权汇总的缺陷 MSE 相对 Clark4 改善。每格为 2026／2028。", "",
                  "| 流动 | 频带 | 原 FULL | 增强版 | 等变版 |", "|---|---|---:|---:|---:|"])
    lookup = {(r["case"], r["seed"], r["model"], r["band"]): r for r in step}
    for case in CASES:
        for band, label in zip(BANDS, ("全频", "k≤16", "17≤k≤29", "k≥30")):
            values = ["／".join(f"{lookup[case, s, model, band]['gain_vs_Clark4_pct']:+.2f}%" for s in SEEDS)
                      for model in (BASE, AUG, EQ)]
            lines.append(f"| {str(case)[-3:]} | {label} | " + " | ".join(values) + " |")
    lines.extend(["", "终点表中的 MSE 改善均相对 Clark4。谱尾 Z 最大超额为窗口内"
                  "模型谱尾 enstrophy 减参考谱尾 enstrophy 的最大正值。"])
    endpoint_table(lines, short, "共同初态 t=1→1.05")
    lines.extend(["", f"短窗延长门槛为 **{manifest['short_gate']['status']}**。"
                  "旧四旋转集成使用 102／2026 的相同初态与时间网格，止于 1.05。"])
    if extension:
        endpoint_table(lines, extension, "续算至 t=1.20")
    long_gate = manifest["long_gate"]
    if long_gate["status"] == "FAIL":
        failures = [f"{str(r['case'])[-3:]}/{r['seed']}（"
                    + "、".join(k for k, v in r.items() if v is False and k != "pass") + "）"
                    for r in long_gate["pairs"] if not r["pass"]]
        decision = "暂缓至 5 的长窗。1.20 门槛未通过的配对为 " + "，".join(failures) + "。"
    elif long_gate["status"] == "PASS":
        decision = "1.20 的配对通过预定长窗条件，可将 t=5 作为下一阶段。"
    else:
        decision = "1.05 门槛未通过，暂缓更长窗口。"
    lines.extend(["", decision, "t=5 状态为 **NOT_RUN**。", "",
                  "[逐步轨迹图](PAIRED_TRAJECTORIES.png) · [一步汇总](ONE_STEP_SUMMARY.csv)", ""])
    for phase, counts in manifest["actual"].items():
        lines.append(f"- {phase} 共 {counts['cnn']} 次 CNN、{counts['rhs']} 次 RHS、{counts['clark4']} 次解析 Clark4。")
    (out / "REPORT_CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    dump(out / "MANIFEST.json", dict(status="PASS", model_calls=0, source_sha256=sha(folder / "MANIFEST.json"),
                                     outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(out / "REPORT_CN.md")


if __name__ == "__main__":
    main()
