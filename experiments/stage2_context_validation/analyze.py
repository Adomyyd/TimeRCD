import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.common.artifacts import write_json
from experiments.stage2_context_validation.statistics import (
    impact_radii_by_sequence,
    summarize_a_rows,
    summarize_b_rows,
)


def _read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty summary: {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_b(rows, radii, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(radius) for radius in radii]
    x = np.arange(len(radii))
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for target_kind, color in (("normal", "#2878b5"), ("anomaly", "#d9534f")):
        selected = [row for row in rows if row["target_kind"] == target_kind]
        selected.sort(key=lambda row: radii.index(row["radius"]))
        means = np.asarray([row["logit_mean"] for row in selected])
        low = np.asarray([row["ci_low"] for row in selected])
        high = np.asarray([row["ci_high"] for row in selected])
        axis.plot(x, means, marker="o", label=target_kind, color=color)
        axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Additional visible patch radius (full = all valid patches)")
    axis.set_ylabel("Mean target logit (z1 − z0)")
    axis.set_title("Fixed-target context-range curve")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Target")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(run_directory):
    run_directory = Path(run_directory).resolve()
    manifest = json.loads((run_directory / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"Run status must be completed, got {manifest.get('status')}")
    config = manifest["config"]
    repetitions, seed = config["bootstrap_repetitions"], config["seed"]
    radii = list(config["radii"])
    a_input = _read_csv(run_directory / "a_sequence_rings.csv")
    a_rows = [{
        **row,
        "seed": int(row["seed"]),
        "point_count": int(row["point_count"]),
        "delta_mean": float(row["delta_mean"]) if row["delta_mean"] else None,
        "mean_absolute_delta": float(row["mean_absolute_delta"]) if row["mean_absolute_delta"] else None,
    } for row in a_input]
    a_summary = summarize_a_rows(a_rows, repetitions, seed)
    ring0 = {
        (row["location"], row["event_pattern"]): abs(row["delta_mean"])
        for row in a_summary if row["ring"] == "ring0_anomaly"
    }
    for row in a_summary:
        baseline = ring0.get((row["location"], row["event_pattern"]))
        row["relative_to_ring0"] = None if not baseline else abs(row["delta_mean"]) / baseline
    individual_radii = impact_radii_by_sequence(a_rows)
    all_rings = {row["ring"]: row for row in a_summary
                 if row["location"] == "all" and row["event_pattern"] == "all"}
    ring0_magnitude = abs(all_rings["ring0_anomaly"]["delta_mean"])
    combined_rings = [row for name, row in all_rings.items()
                      if not name.startswith(("left_", "right_", "between_events_"))]
    peak = max(abs(row["delta_mean"]) for row in combined_rings)
    ring_order = (("ring1", 1), ("ring2", 2), ("ring3", 3), ("ring4_5", 4), ("ring6_10", 6),
                  ("ring11_20", 11), ("ring21_39", 21), ("ring40_plus", 40))
    impact_radius = next(
        (lower for name, lower in ring_order
         if abs(all_rings[name]["delta_mean"]) < 0.05 * peak), None
    )
    far_rings = (all_rings["ring21_39"], all_rings["ring40_plus"])
    extends_to_20 = ring0_magnitude > 0 and any(abs(row["delta_mean"]) >= 0.05 * ring0_magnitude for row in far_rings)
    local_condition = ring0_magnitude > 0 and all(
        abs(all_rings[name]["delta_mean"]) < 0.05 * ring0_magnitude
        and all_rings[name]["ci_low"] <= 0 <= all_rings[name]["ci_high"]
        for name in ("ring3", "ring4_5", "ring6_10", "ring11_20", "ring21_39", "ring40_plus")
    )
    b_input = _read_csv(run_directory / "b_target_scores.csv")
    b_rows = [{
        **row,
        "seed": int(row["seed"]),
        "radius": row["radius"] if row["radius"] == "full" else int(row["radius"]),
        "target_logit_mean": float(row["target_logit_mean"]),
    } for row in b_input]
    b_summary = summarize_b_rows(b_rows, radii, repetitions, seed)
    analysis_dir = run_directory / "analysis"
    analysis_dir.mkdir(exist_ok=False)
    _write_csv(analysis_dir / "a_summary.csv", a_summary)
    _write_csv(analysis_dir / "a_impact_radii.csv", individual_radii)
    _write_csv(analysis_dir / "b_summary.csv", b_summary)
    _plot_b(b_summary, radii, analysis_dir / "b_context_curve.png")
    write_json(analysis_dir / "summary.json", {
        "a_rows": len(a_rows), "a_summary_rows": len(a_summary),
        "a_sequence_impact_radii": len(individual_radii),
        "b_rows": len(b_rows), "b_summary_rows": len(b_summary),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "a_impact_radius_lower_bin_boundary": impact_radius,
        "a_protocol_locality_condition_met": local_condition,
        "a_effect_at_least_five_percent_at_20plus": extends_to_20,
    })

    report = [
        "# 交接验证A/B：异常影响半径与上下文范围",
        "",
        f"运行目录：`{run_directory}`。状态：`{manifest['status']}`。",
        "",
        "## A：异常段还原",
        "",
        f"{len({row['name'] for row in a_rows})}条派生配对序列；共享基准的布局在种子内平均，按独立基准序列重采样，95% percentile区间。",
        "逐序列环均值、分层汇总见 `a_sequence_rings.csv` 与 `analysis/a_summary.csv`。",
        "`relative_to_ring0`为signed环均值取绝对值后相对异常点环0均值绝对值的比值；同时检查signed Δ区间是否包含0。点级平均绝对变化是辅助读出，不替代主判据。",
        f"总体影响半径首个低于跨环峰值5%的环下界：{impact_radius if impact_radius is not None else '未达到'} patch；不对缺失环插值。",
        f"冻结局部判据（所有ring≥3低于5%且signed区间含0）：{local_condition}；20 patch以上仍达ring0的5%：{extends_to_20}。",
        "",
        "## B：远处上下文截断",
        "",
        f"{len({row['seed'] for row in b_rows})}个配对生成种子；固定32点目标的平均logit与配对变化见 `analysis/b_summary.csv`。",
        "饱和半径：" + "; ".join(
            f"{kind}={next(row['saturation_radius'] for row in b_summary if row['target_kind'] == kind)}"
            for kind in ("normal", "anomaly")
        ) + " patch（未达到或总变化为0时不适用）。",
        "曲线图：`analysis/b_context_curve.png`。饱和半径为从r到full的差小于总变化5%的最小r；总变化为0时不适用。",
        "截断会改变softmax分母，因此曲线描述上下文范围，不是单个patch的因果效应。只比较固定目标分数，不比较AP。",
        "",
        "## 解释边界",
        "",
        "结果仅约束本协议中的合成信号、冻结权重、窗口和patch掩码。区间跨零不证明等效；不从分数下降直接推断参照污染或网络内部因果机制。",
        "",
    ]
    (analysis_dir / "report.md").write_text("\n".join(report))
    return analysis_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    args = parser.parse_args()
    print(analyze(args.run_directory))


if __name__ == "__main__":
    main()
