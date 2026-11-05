import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary


IDENTITY = ["case", "seed", "phase", "shape", "event_length", "offset", "score_kind"]


def paired_readouts(metrics, random_seeds):
    selectors = ["full", "keep_first", "keep_last"] + [f"random_{s}" for s in random_seeds]
    variants = {"A"} | {f"{s}_{scope}" for s in selectors for scope in ("last", "all")}
    if metrics.duplicated(IDENTITY + ["variant"]).any():
        raise ValueError("Duplicate observations")
    columns = [c for c in metrics if c not in IDENTITY + ["variant", "metric_status"]]
    absolute, effects = [], []
    for identity, group in metrics.groupby(IDENTITY, sort=True):
        if set(group.variant) != variants or not group.metric_status.eq("ok").all():
            raise ValueError("Missing or invalid condition")
        base = dict(zip(IDENTITY, identity))
        values = group.set_index("variant")[columns].copy()
        for scope in ("last", "all"):
            random = values.loc[[f"random_{s}_{scope}" for s in random_seeds]]
            if (random.isna().any() != random.isna().all()).any():
                raise ValueError("Selection-dependent missingness among random replicates")
            values.loc[f"random_mean_{scope}"] = random.mean()
            matched = values.loc[[f"{s}_{scope}" for s in ("keep_first", "keep_last", "random_mean")]]
            if (matched.isna().any() != matched.isna().all()).any():
                raise ValueError("Unexpected missingness among matched selections")
        selected = ["A"] + [f"{s}_{scope}" for s in ("full", "keep_first", "keep_last", "random_mean") for scope in ("last", "all")]
        for name in selected:
            absolute.append({**base, "contrast": name, **values.loc[name].to_dict()})
        contrasts = [(f"{s}_all", f"{s}_last") for s in ("full", "keep_first", "keep_last", "random_mean")]
        for scope in ("last", "all"):
            contrasts.append((f"keep_last_{scope}", f"keep_first_{scope}"))
            contrasts.extend((f"{s}_{scope}", "A") for s in ("keep_first", "keep_last", "random_mean"))
            contrasts.extend((f"{s}_{scope}", f"{b}_{scope}") for s in ("keep_first", "keep_last") for b in ("full", "random_mean"))
        for left, right in contrasts:
            effects.append({**base, "contrast": f"{left}_minus_{right}", **(values.loc[left] - values.loc[right]).to_dict()})
        interaction = ((values.loc["keep_last_all"] - values.loc["keep_first_all"])
                       - (values.loc["keep_last_last"] - values.loc["keep_first_last"]))
        effects.append({**base, "contrast": "position_by_scope", **interaction.to_dict()})
    return pd.DataFrame(absolute), pd.DataFrame(effects), columns


def summarize(directory, metrics, config):
    expected = {(s, p, h, length, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for length in (32, 128)}
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    if len(metrics) != 192 * 17 * 2 or metrics.case.nunique() != 192 or set(map(tuple, metrics[factors].to_numpy())) != expected:
        raise ValueError("Incomplete design")
    if metrics.duplicated(factors + ["variant", "score_kind"]).any() or set(metrics.score_kind) != {"scores", "logits"}:
        raise ValueError("Invalid factor identity")
    absolute, effects, columns = paired_readouts(metrics, config["random_seeds"])
    absolute.to_csv(directory / "absolute.csv", index=False)
    effects.to_csv(directory / "effects.csv", index=False)
    groups = {"overall": [], "shape": ["shape"], "length": ["event_length"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    for prefix, frame in (("absolute", absolute), ("effect", effects)):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions, columns, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    primary = summary[(summary.score_kind == "scores") & summary.contrast.isin([
        "keep_last_all_minus_keep_first_all", "position_by_scope"]) & summary.metric.isin([
        "average_precision", "r050_precision", "r050_fpr", "r050_recall"])]
    lines = ["# E10：高分区排序定位", "", "| 比较 | 指标 | 均值差 | 95%区间 |", "|---|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['metric']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
