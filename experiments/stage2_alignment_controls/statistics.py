import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary


FACTORS = ["seed", "phase", "shape", "event_length"]
VARIANTS = {"A", "full_last", "full_all", "common_last", "common_all"}


def paired_effects(metrics):
    identity = FACTORS + ["offset", "score_kind"]
    if metrics.duplicated(identity + ["variant"]).any():
        raise ValueError("Duplicate observations")
    excluded = identity + ["case", "M", "P", "common_patches", "variant", "metric_status"]
    columns = [c for c in metrics if c not in excluded]
    rows = []
    contrasts = [(v, "A") for v in sorted(VARIANTS - {"A"})] + [
        ("full_all", "full_last"), ("common_all", "common_last"),
        ("common_last", "full_last"), ("common_all", "full_all")]
    for keys, group in metrics.groupby(identity, sort=True):
        if set(group.variant) != VARIANTS or not group.metric_status.eq("ok").all():
            raise ValueError("Incomplete condition coverage")
        values = group.set_index("variant")[columns]
        if (values.isna().any() != values.isna().all()).any():
            raise ValueError("Condition-dependent missingness")
        for left, right in contrasts:
            rows.append({**dict(zip(identity, keys)), "contrast": f"{left}_minus_{right}",
                         **(values.loc[left] - values.loc[right]).to_dict()})
    return pd.DataFrame(rows), columns


def alignment_interactions(effects, columns):
    keys = FACTORS + ["score_kind", "contrast"]
    if effects.duplicated(keys + ["offset"]).any() or set(effects.offset) != {0, 8}:
        raise ValueError("Invalid offset coverage")
    left, right = [effects[effects.offset == offset].set_index(keys)[columns].sort_index() for offset in (0, 8)]
    if not left.index.equals(right.index):
        raise ValueError("Unpaired offsets")
    delta = (right - left).reset_index()
    identity = FACTORS + ["score_kind"]
    indexed = delta.set_index(identity + ["contrast"])
    full = indexed.xs("full_all_minus_full_last", level="contrast")
    common = indexed.xs("common_all_minus_common_last", level="contrast")
    if not full.index.equals(common.index):
        raise ValueError("Unpaired selection controls")
    extra = []
    for name, values in (("I_full", full), ("I_common", common), ("J_common_minus_full", common - full)):
        block = values.reset_index()
        block["contrast"] = name
        extra.append(block)
    return delta, pd.concat(extra, ignore_index=True)


def save_summaries(directory, frame, columns, config, prefix, groupings):
    for name, dimensions in groupings.items():
        seeds, stats = seed_summary(frame, dimensions, columns, config)
        seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
        stats.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)


def summarize(directory, config):
    metrics = pd.read_csv(directory / "metrics.csv")
    if len(metrics) != 384 * 5 * 2 or metrics.case.nunique() != 384:
        raise ValueError("Incomplete frozen design")
    expected = {(s, p, h, length, offset) for s in config["synthetic"]["seeds"]
                for p in config["synthetic"]["phases"] for h in config["synthetic"]["shapes"]
                for length in config["synthetic"]["event_lengths"] for offset in (0, 8)}
    if set(map(tuple, metrics[FACTORS + ["offset"]].to_numpy())) != expected:
        raise ValueError("Factorial identity mismatch")
    effects, columns = paired_effects(metrics)
    delta, interactions = alignment_interactions(effects, columns)
    effects.to_csv(directory / "effects.csv", index=False)
    delta.to_csv(directory / "offset_differences.csv", index=False)
    interactions.to_csv(directory / "interactions.csv", index=False)
    groups = {"overall": [], "phase": ["phase"], "shape": ["shape"], "length": ["event_length"],
              "cells": ["phase", "shape", "event_length"]}
    save_summaries(directory, interactions, columns, config, "interaction", groups)
    save_summaries(directory, delta, columns, config, "offset", groups)
    effect_groups = {**groups, "offset": ["offset"], "cells": ["phase", "shape", "event_length", "offset"]}
    save_summaries(directory, effects, columns, config, "effect", effect_groups)
    absolute = metrics.rename(columns={"variant": "contrast"})
    save_summaries(directory, absolute, columns, config, "absolute", effect_groups)
    summary = pd.read_csv(directory / "summary_interaction_overall.csv")
    chosen = summary[(summary.score_kind == "scores") & summary.metric.isin(["average_precision", "roc_auc", "P_anomaly__N_normal_auc"])]
    lines = ["# E08：背景配对和共同P位置控制", "", "| 交互 | 读出 | 均值 | 95%区间 |", "|---|---|---:|---|"]
    for row in chosen.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['metric']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    lines += ["", "I=(all−last)在offset8减offset0；J=I_common−I_full。先种子内平均，再8种子配对bootstrap。", ""]
    report = "\n".join(lines)
    (directory / "report.md").write_text(report)
    return report
