import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary


IDENTITY = ["case", "seed", "phase", "shape", "event_length", "offset", "M", "P", "common_patches", "score_kind"]


def paired_effects(metrics, random_seeds):
    selectors = ["full", "keep_first", "keep_last"] + [f"random_{s}" for s in random_seeds]
    variants = {"A"} | {f"{s}_{scope}" for s in selectors for scope in ("last", "all")}
    if metrics.duplicated(IDENTITY + ["variant"]).any():
        raise ValueError("Duplicate metric observations")
    columns = [c for c in metrics if c not in IDENTITY + ["variant", "metric_status"]]
    effects, absolute, ranges = [], [], []
    for identity, group in metrics.groupby(IDENTITY, sort=True):
        if set(group.variant) != variants or not group.metric_status.eq("ok").all():
            raise ValueError("Incomplete condition coverage")
        values = group.set_index("variant")[columns]
        if (values.isna().any() != values.isna().all()).any():
            raise ValueError("Condition-dependent missingness")
        base = dict(zip(IDENTITY, identity))
        for scope in ("last", "all"):
            block = values.loc[[f"random_{s}_{scope}" for s in random_seeds]]
            values.loc[f"random_mean_{scope}"] = block.mean()
            for column in columns:
                ranges.append({**base, "scope": scope, "metric": column, "random_selections": len(block),
                               "mean": block[column].mean(), "min": block[column].min(),
                               "max": block[column].max(), "range": block[column].max() - block[column].min()})
        for variant, row in values.iterrows():
            absolute.append({**base, "contrast": variant, **row.to_dict()})
        contrasts = [(v, "A") for v in values.index if v != "A"]
        for selector in selectors + ["random_mean"]:
            contrasts.append((f"{selector}_all", f"{selector}_last"))
            if selector != "full":
                contrasts.extend((f"{selector}_{scope}", f"full_{scope}") for scope in ("last", "all"))
        for scope in ("last", "all"):
            contrasts.extend([(f"keep_last_{scope}", f"keep_first_{scope}"),
                              (f"keep_first_{scope}", f"random_mean_{scope}"),
                              (f"keep_last_{scope}", f"random_mean_{scope}")])
        for left, right in contrasts:
            effects.append({**base, "contrast": f"{left}_minus_{right}", **(values.loc[left] - values.loc[right]).to_dict()})
        interaction = ((values.loc["keep_last_all"] - values.loc["keep_first_all"])
                       - (values.loc["keep_last_last"] - values.loc["keep_first_last"]))
        effects.append({**base, "contrast": "position_by_scope", **interaction.to_dict()})
    return pd.DataFrame(effects), pd.DataFrame(absolute), pd.DataFrame(ranges), columns


def summarize(directory, config):
    metrics = pd.read_csv(directory / "metrics.csv")
    expected = {(s, p, h, length, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for length in (32, 128)}
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    if len(metrics) != 192 * 17 * 2 or metrics.case.nunique() != 192 or set(map(tuple, metrics[factors].to_numpy())) != expected:
        raise ValueError("Incomplete frozen factorial design")
    if metrics.duplicated(factors + ["score_kind", "variant"]).any() or set(metrics.score_kind) != {"scores", "logits"}:
        raise ValueError("Invalid factor or score identity")
    effects, absolute, ranges, columns = paired_effects(metrics, config["random_seeds"])
    effects.to_csv(directory / "effects.csv", index=False)
    absolute.to_csv(directory / "absolute_with_random_mean.csv", index=False)
    ranges.to_csv(directory / "random_ranges.csv", index=False)
    groups = {"overall": [], "phase": ["phase"], "shape": ["shape"], "length": ["event_length"],
              "cells": ["phase", "shape", "event_length"]}
    for prefix, frame in (("effect", effects), ("absolute", absolute)):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions, columns, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    primary = summary[(summary.score_kind == "scores") & (summary.metric == "average_precision") & summary.contrast.isin([
        "keep_last_all_minus_keep_first_all", "keep_last_last_minus_keep_first_last", "position_by_scope"])]
    lines = ["# E09：同数量P位置选择", "", "| AP比较 | 均值 | 95%区间 |", "|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    lines += ["", "五随机先输入内指标平均；生成种子内等权平均因素格，再8种子bootstrap。", ""]
    report = "\n".join(lines)
    (directory / "report.md").write_text(report)
    return report
