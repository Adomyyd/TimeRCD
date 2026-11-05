"""Summarize paired padding interventions and within-series window sensitivity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json


def bootstrap_mean(values, seed=2024, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite, nonempty series-level observations")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def window_ranges(frame, windows):
    rows = []
    for (series, kind, variant), group in frame.groupby(["series", "score_kind", "variant"], sort=True):
        if variant not in ("current", "zero", "compact"):
            continue
        group = group.set_index("configured_window")
        if set(group.index) != set(windows) or group.index.duplicated().any():
            raise ValueError(f"Incomplete or duplicate windows for {series}/{variant}")
        ap = group.loc[windows, "average_precision"]
        if ap.isna().any():
            raise ValueError(f"Undefined AP for {series}")
        rows.append({"series": series, "score_kind": kind, "variant": variant,
                     "ap_range": float(ap.max() - ap.min()),
                     "best_window": int(ap.idxmax()), "worst_window": int(ap.idxmin()),
                     **{f"ap_w{window}": float(ap.loc[window]) for window in windows}})
    return pd.DataFrame(rows)


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] not in ("completed", "checks_failed"):
        raise ValueError("The run is not finished")
    config = manifest["config"]
    frame = pd.read_csv(directory / "condition_metrics.csv")
    if frame.duplicated(["case", "variant", "score_kind"]).any():
        raise ValueError("Duplicate condition rows")
    if not frame.metric_status.eq("ok").all():
        raise ValueError("Analysis requires defined metrics for every selected series")
    primary = frame[frame.score_kind == "scores"]
    ranges = window_ranges(frame, config["window_sizes"])
    current = ranges[(ranges.variant == "current") & (ranges.score_kind == "scores")].set_index("series")
    compact = ranges[(ranges.variant == "compact") & (ranges.score_kind == "scores")].set_index("series")
    zero = ranges[(ranges.variant == "zero") & (ranges.score_kind == "scores")].set_index("series")
    base = primary[primary.variant == "current"]
    expected = len(current) * len(config["window_sizes"])
    if len(base) != expected:
        raise ValueError("Missing baseline cases")
    paired = compact.ap_range - current.ap_range
    stats = {
        "run_status": manifest["status"], "series": len(current), "cases": len(base),
        "primary_score": "scores", "primary_metric": "average_precision",
        "material_ap_difference": 0.01,
        "bootstrap_unit": "series", "bootstrap_repetitions": 2000, "bootstrap_seed": config["seed"],
        "condition_metrics_sha256": sha256(directory / "condition_metrics.csv"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "mean_window_ap_range_current": float(current.ap_range.mean()),
        "mean_window_ap_range_zero": float(zero.ap_range.mean()),
        "mean_window_ap_range_compact": float(compact.ap_range.mean()),
        "range_change_compact_minus_current": bootstrap_mean(paired.to_numpy(), config["seed"]),
        "series_with_current_range_gt_001": int((current.ap_range > 0.01).sum()),
        "series_with_compact_range_gt_001": int((compact.ap_range > 0.01).sum()),
        "series_with_range_reduction_gt_001": int((paired < -0.01).sum()),
        "series_with_changed_best_window": int((current.best_window != compact.best_window).sum()),
        "conditions": {},
    }
    aggregates = []
    for (kind, variant, window), group in frame.groupby(["score_kind", "variant", "configured_window"], sort=True):
        aggregates.append({
            "score_kind": kind, "variant": variant, "window": int(window), "series": len(group),
            "mean_delta_ap": float(group.delta_ap.mean()),
            "max_abs_delta_ap": float(group.delta_ap.abs().max()),
            "cases_abs_delta_ap_gt_001": int((group.delta_ap.abs() > 0.01).sum()),
            "max_abs_prediction_difference": float(group.prediction_max_abs.max()),
        })
    for variant, group in primary.groupby("variant", sort=True):
        per_series = group.groupby("series").delta_ap.mean()
        stats["conditions"][variant] = {
            "cases": len(group), "series": len(per_series),
            "series_mean_delta_ap": bootstrap_mean(per_series.to_numpy(), config["seed"]),
            "max_abs_delta_ap": float(group.delta_ap.abs().max()),
            "mean_abs_delta_ap": float(group.delta_ap.abs().mean()),
            "cases_improved_gt_001": int((group.delta_ap > 0.01).sum()),
            "cases_worsened_gt_001": int((group.delta_ap < -0.01).sum()),
            "max_abs_prediction_difference": float(group.prediction_max_abs.max()),
        }
    return stats, pd.DataFrame(aggregates), ranges, frame


def render_report(directory, stats, aggregates, ranges, frame):
    lines = [
        "# 阶段一：真实基准序列补齐诊断", "",
        f"- 运行目录：`{directory.name}`；状态：`{stats['run_status']}`。",
        f"- {stats['series']} 条开发诊断序列 × 4 个窗口，共 {stats['cases']} 组。",
        "- 主指标为概率分数的 Average Precision（AP），补充报告 logits；差值均为干预减当前基线。",
        "- 0.01 为预先指定的描述性 AP 变化阈值，不是显著性检验。", "",
        "## 整序列补齐效应", "",
        "| 条件 | 组数 | 平均绝对 ΔAP | 最大绝对 ΔAP | 改善 >0.01 | 恶化 >0.01 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats["conditions"].items():
        lines.append(f"| {name} | {row['cases']} | {row['mean_abs_delta_ap']:.6f} | {row['max_abs_delta_ap']:.6f} | {row['cases_improved_gt_001']} | {row['cases_worsened_gt_001']} |")
    lines += ["", "## 窗口敏感性", "",
              "每条序列在预定四个窗口上的 AP 最大值减最小值，仅用于诊断。",
              f"- 当前补齐的平均范围：{stats['mean_window_ap_range_current']:.6f}。",
              f"- 零补齐后的平均范围：{stats['mean_window_ap_range_zero']:.6f}。",
              f"- compact 尾窗后的平均范围：{stats['mean_window_ap_range_compact']:.6f}。",
              f"- 范围 >0.01 的序列数：当前 {stats['series_with_current_range_gt_001']}，compact {stats['series_with_compact_range_gt_001']}。",
              f"- compact 后范围缩小超过 0.01：{stats['series_with_range_reduction_gt_001']} 条；最佳窗口变化：{stats['series_with_changed_best_window']} 条。", "",
              "## 证据边界", "",
              "本批是预先冻结的分层开发诊断集，来源于已有 700 条评测元数据，长度限定为 256–20000；不代表完整 TSB-AD-U。",
              "两类补齐干预不改变有效输入的标准化统计量。zero/compact 仅替换尾窗输出；无外层补齐时为严格 no-op。short_forced_* 仅用于短于配置窗口的序列。",
              "JSON 中的区间按序列重采样，而不是把多个窗口视为独立样本；只有一次确定性模型推理，区间不代表跨训练种子不确定性。",
              "窗口差异仍可能来自窗口边界、上下文范围、异常位置和数值饱和等因素，本实验不能将其归因为参照污染。",
              "详见 condition_effects_summary.csv、window_sensitivity.csv、analysis.json 以及原始逐例表。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    output = args.run_directory / "analysis"
    stats, aggregates, ranges, frame = summarize(args.run_directory)
    output.mkdir(exist_ok=False)
    write_json(output / "analysis.json", stats)
    aggregates.to_csv(output / "condition_effects_summary.csv", index=False)
    ranges.to_csv(output / "window_sensitivity.csv", index=False)
    (output / "report.md").write_text(render_report(args.run_directory, stats, aggregates, ranges, frame))
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
