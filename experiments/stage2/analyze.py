import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.stage1.analyze import bootstrap_mean


def paired_cases(frame, seeds):
    keys = ["case", "scope", "score_kind"]
    if frame.duplicated(keys + ["variant"]).any():
        raise ValueError("Duplicate condition rows")
    expected = {"baseline", "oracle"} | {f"random_{seed}" for seed in seeds}
    rows = []
    metadata = ["series", "window", "kind", "source", "density_bin", "generator_seed",
                "placement", "anomaly_count", "anomaly_ratio", "eligible_points", "length"]
    for identity, group in frame.groupby(keys, sort=True):
        if set(group.variant) != expected:
            raise ValueError(f"Incomplete conditions for {identity}")
        row = {**dict(zip(keys, identity)), **group.iloc[0][metadata].to_dict()}
        for metric, suffix in (("average_precision", "ap"), ("roc_auc", "roc")):
            scores = group.set_index("variant")[metric]
            if scores.isna().any():
                row.update({f"{name}_{suffix}": np.nan for name in (
                    "baseline", "oracle", "random_mean", "random_min", "random_max",
                    "oracle_minus_baseline", "oracle_minus_random")})
                continue
            random = scores[[f"random_{seed}" for seed in seeds]]
            row.update({f"baseline_{suffix}": scores.baseline, f"oracle_{suffix}": scores.oracle,
                        f"random_mean_{suffix}": random.mean(), f"random_min_{suffix}": random.min(),
                        f"random_max_{suffix}": random.max(),
                        f"oracle_minus_baseline_{suffix}": scores.oracle - scores.baseline,
                        f"oracle_minus_random_{suffix}": scores.oracle - random.mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(group, unit, seed, repetitions):
    group = group.dropna(subset=["oracle_minus_baseline_ap", "oracle_minus_random_ap"])
    if group.empty:
        return {"cases": 0, "units": 0}
    result = {"cases": len(group), "units": int(group[unit].nunique()), "bootstrap_unit": unit}
    for metric in ("baseline_ap", "oracle_ap", "random_mean_ap", "baseline_roc", "oracle_roc", "random_mean_roc"):
        result[metric] = float(group[metric].mean())
    for metric in ("oracle_minus_baseline_ap", "oracle_minus_random_ap"):
        values = group.groupby(unit)[metric].mean().to_numpy()
        result[metric] = bootstrap_mean(values, seed, repetitions)
    result["oracle_improved_gt_001"] = int((group.oracle_minus_baseline_ap > .01).sum())
    result["oracle_worsened_gt_001"] = int((group.oracle_minus_baseline_ap < -.01).sum())
    result["oracle_beats_random_mean_gt_001"] = int((group.oracle_minus_random_ap > .01).sum())
    result["oracle_beats_all_random_gt_001"] = int((group.oracle_ap - group.random_max_ap > .01).sum())
    return result


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] not in ("completed", "checks_failed"):
        raise ValueError("Run has not finished")
    config = manifest["config"]
    frame = pd.read_csv(directory / "metrics.csv")
    paired = paired_cases(frame, config["random_seeds"])
    full = paired[(paired.scope == "full_fallback") & (paired.score_kind == "scores")]
    expected = len(config["data"]["files"]) * len(config["real_windows"])
    expected += len(config["synthetic"]["seeds"]) * len(config["synthetic"]["placements"]) * len(config["synthetic"]["anomaly_counts"]) * len(config["synthetic_windows"])
    if len(full) != expected or full.baseline_ap.isna().any():
        raise ValueError("Missing cases or undefined full-series AP")
    windows = pd.read_csv(directory / "windows.csv")
    config_protocol = config["protocol"]
    args = (config_protocol["bootstrap_seed"], config_protocol["bootstrap_repetitions"])
    summary = {"status": manifest["status"], "cases": len(full),
               "metrics_sha256": sha256(directory / "metrics.csv"),
               "analysis_sha256": sha256(Path(__file__)), "groups": {}}
    for (scope, kind), primary in paired[paired.score_kind == "scores"].groupby(["scope", "kind"]):
        real = kind == "tsb"
        strata = ["window", "density_bin"] if real else ["window", "anomaly_count"]
        for identity, group in primary.groupby(strata):
            key = f"{scope}/{kind}/w{identity[0]}/{identity[1]}"
            summary["groups"][key] = aggregate(group, "series" if real else "generator_seed", *args)
        for window, group in primary.groupby("window"):
            key = f"{scope}/{kind}/w{window}/all"
            summary["groups"][key] = aggregate(group, "series" if real else "generator_seed", *args)
    per_source = []
    for (window, source), group in full[full.kind == "tsb"].groupby(["window", "source"]):
        per_source.append({"window": int(window), "source": source, **aggregate(group, "series", *args)})
    summary["per_source"] = per_source
    summary["coverage"] = []
    merged = windows.merge(full[["case", "kind", "window"]], on="case", validate="many_to_one")
    for (kind, window), group in merged.groupby(["kind", "window"]):
        summary["coverage"].append({
            "kind": kind, "window": int(window), "windows": len(group),
            "no_clean_windows": int((~group.eligible).sum()),
            "fallback_points": int(group.loc[~group.eligible, "valid_points"].sum()),
            "total_points": int(group.valid_points.sum()),
            "blocked_patches": int(group.blocked_patches.sum()),
            "mixed_patches": int(group.mixed_patches.sum()),
        })
    random = frame[(frame.scope == "full_fallback") & (frame.score_kind == "scores") & frame.variant.str.startswith("random_")]
    random_summary = random.groupby(["kind", "window", "variant"])[["average_precision", "delta_ap"]].mean().reset_index()
    synth_random = random[random.kind == "controlled_synthetic"].groupby(
        ["anomaly_count", "variant"])[["average_precision", "delta_ap"]].mean().reset_index()
    return summary, paired, random_summary, synth_random


def render(summary):
    lines = ["# 阶段二：全层干净参照 Oracle", "",
             f"运行状态：`{summary['status']}`；共 {summary['cases']} 组。",
             "随机列为五个预定屏蔽种子的指标均值，不是分数集成。Δ 均为 AP 绝对差。", "",
             "| 分组（整序列，空参照回退基线） | 原模型 AP | Oracle AP | 随机 AP | Oracle−原模型 | Oracle−随机 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, row in summary["groups"].items():
        if not key.startswith("full_fallback/") or not row["cases"]:
            continue
        lines.append(f"| {key} | {row['baseline_ap']:.6f} | {row['oracle_ap']:.6f} | {row['random_mean_ap']:.6f} | {row['oracle_minus_baseline_ap']['mean']:+.6f} | {row['oracle_minus_random_ap']['mean']:+.6f} |")
    lines += ["", "## 解释范围", "",
              "真实开发序列沿用官方预处理；合成序列固定干净生成过程的均值和标准差，避免异常数量改变归一化统计量。两者分开解释。",
              "Oracle 使用标签筛选全正常 Patch，并在每层只屏蔽 key/value；混合 Patch 一并排除，查询、残差和输出保留。",
              "没有干净参照时 Oracle 未定义，各组统一保留基线输出；eligible 作用域另行计算相同可干预点上的指标，可能出现单类标签。",
              "真实序列的区间按序列 bootstrap；合成数据先在同一生成种子内平均三个位置，再按三个生成种子 bootstrap，样本很少，区间仅为探索性描述。",
              "随机对照也使用标签决定屏蔽数量，且不匹配被删 Patch 的空间分布；不能把 Oracle 当作可部署方法或保证有效的性能上界。",
              "全层掩码切断的是注意力参照路径，不消除全序列归一化或混合 Patch 内的信息。结果不能直接推出可信度头可学习或会改善。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze paired Oracle and random reference controls")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    summary, paired, random, synth_random = summarize(args.run_directory)
    output = args.run_directory / "analysis"
    output.mkdir(exist_ok=False)
    write_json(output / "analysis.json", summary)
    paired.to_csv(output / "paired_cases.csv", index=False)
    random.to_csv(output / "random_seed_means.csv", index=False)
    synth_random.to_csv(output / "synthetic_random_seed_means.csv", index=False)
    shutil.copyfile(__file__, output / "analyze.py")
    (output / "report.md").write_text(render(summary))
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
