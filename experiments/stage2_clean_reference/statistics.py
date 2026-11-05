import numpy as np
import pandas as pd

from experiments.stage1.analyze import bootstrap_mean


CONTRASTS = (("C", "A"), ("V", "A"), ("KV", "V"), ("V", "C"),
             ("KV", "C"), ("KV", "A"), ("C_all", "A"))
IDENTITY = ["case", "seed", "shape", "event_length", "offset", "M", "P", "score_kind"]


def paired_contrasts(metrics):
    if metrics.duplicated(IDENTITY + ["variant"]).any():
        raise ValueError("Duplicate metric rows")
    columns = [k for k in metrics if k not in IDENTITY + ["variant", "metric_status"]]
    rows = []
    for identity, block in metrics.groupby(IDENTITY, sort=True):
        if set(block.variant) != {"A", "C", "V", "KV", "C_all"} or not block.metric_status.eq("ok").all():
            raise ValueError("Incomplete or undefined condition metrics")
        indexed = block.set_index("variant")
        for candidate, baseline in CONTRASTS:
            row = {**dict(zip(IDENTITY, identity)), "contrast": f"{candidate}_minus_{baseline}"}
            for column in columns:
                data = indexed[column]
                if data.isna().any() and not data.isna().all():
                    raise ValueError("Condition-dependent missingness")
                row[column] = data[candidate] - data[baseline]
            rows.append(row)
    return pd.DataFrame(rows), columns


def seed_summary(frame, dimensions, columns, config):
    keys = dimensions + ["score_kind", "contrast"]
    per_seed = frame.groupby(keys + ["seed"], dropna=False)[columns].mean().reset_index()
    rows = []
    for identity, block in per_seed.groupby(keys, dropna=False, sort=True):
        if set(block.seed) != set(config["synthetic"]["seeds"]):
            raise ValueError("Missing generation seed")
        for column in columns:
            values = block[column].dropna().to_numpy()
            result = bootstrap_mean(values, config["bootstrap_seed"], config["bootstrap_repetitions"]) if len(values) else {
                "mean": None, "ci95_low": None, "ci95_high": None}
            rows.append({**dict(zip(keys, identity)), "metric": column, "seeds": len(values), **result})
    return per_seed, pd.DataFrame(rows)


def summarize(directory, config):
    metrics = pd.read_csv(directory / "metrics.csv")
    if len(metrics) != 144 * 5 * 2 or metrics.case.nunique() != 144:
        raise ValueError("Unexpected full factorial metrics")
    contrasts, columns = paired_contrasts(metrics)
    contrasts.to_csv(directory / "contrasts.csv", index=False)
    long = contrasts[contrasts.event_length > 5]
    groupings = {"long_overall": [], "shape": ["shape"], "length": ["event_length"],
                 "offset": ["offset"], "cells": ["shape", "event_length", "offset"]}
    for name, dimensions in groupings.items():
        per_seed, summary = seed_summary(long, dimensions, columns, config)
        per_seed.to_csv(directory / f"seed_{name}.csv", index=False)
        summary.to_csv(directory / f"summary_{name}.csv", index=False)
    keys = ["seed", "shape", "event_length", "score_kind", "contrast"]
    if long.duplicated(keys + ["offset"]).any():
        raise ValueError("Duplicate alignment cells")
    left, right = [long[long.offset == offset].set_index(keys)[columns].sort_index() for offset in (0, 8)]
    if not left.index.equals(right.index):
        raise ValueError("Unpaired alignment contrast")
    alignment = (right - left).reset_index()
    seeds, summary = seed_summary(alignment, ["shape", "event_length"], columns, config)
    seeds.to_csv(directory / "seed_alignment_interaction.csv", index=False)
    summary.to_csv(directory / "summary_alignment_interaction.csv", index=False)
    output = pd.read_csv(directory / "summary_long_overall.csv")
    primary = output[(output.score_kind == "scores") & output.metric.isin([
        "average_precision", "P_anomaly__N_normal_auc", "P_anomaly__N_normal_contribution", "P_anomaly_mean"])]
    lines = ["# 最后一层干净参照替换：长事件种子配对汇总", "", "| 比较 | 读出 | 均值差 | 95%区间 |", "|---|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['metric']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    lines += ["", "每个种子内先平均形态/长度/对齐，再按8个生成种子bootstrap。受控合成诊断，非基准结论。", ""]
    (directory / "report.md").write_text("\n".join(lines))
    return "\n".join(lines)
