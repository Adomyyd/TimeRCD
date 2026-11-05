import numpy as np
import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_layer_factorial.statistics import IDENTITY, VARIANTS, paired_effects as parent_effects


PRIMARY = ["J_add_keep_first", "J_add_keep_last", "R_keep_first", "R_keep_last", "K_add", "K_R"]


def paired_effects(metrics, audit=None):
    expected = VARIANTS | {"keep_first_add", "keep_last_add"}
    if metrics.empty or metrics[IDENTITY + ["variant"]].isna().any().any() or metrics.duplicated(IDENTITY + ["variant"]).any():
        raise ValueError("Missing or duplicated identities")
    if not np.isfinite(metrics.average_precision).all() or not metrics.metric_status.eq("ok").all():
        raise ValueError("Undefined metric")
    originals, columns = parent_effects(metrics[metrics.variant.isin(VARIANTS)])
    originals = originals.set_index(IDENTITY + ["contrast"])
    rows = []
    for identity, block in metrics.groupby(IDENTITY, sort=True):
        if set(block.variant) != expected:
            raise ValueError("Missing additive condition")
        values = block.set_index("variant")[columns]
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError("Infinite metric")
        effects = {}
        for selector in ("keep_first", "keep_last"):
            vadd, v11, v10, v01 = [values.loc[f"{selector}_{b}"] for b in ("add", "11", "10", "01")]
            if not vadd.isna().equals(v11.isna()):
                raise ValueError("Condition-dependent missingness")
            effects[f"J_add_{selector}"] = vadd - v10 - v01 + values.loc["A"]
            effects[f"R_{selector}"] = v11 - vadd
            effects[f"J_total_{selector}"] = originals.loc[(*identity, f"J_{selector}")]
        for part in ("add", "total"):
            effects[f"K_{part}"] = effects[f"J_{part}_keep_last"] - effects[f"J_{part}_keep_first"]
        effects["K_R"] = effects["R_keep_last"] - effects["R_keep_first"]
        for part in ("add", "11"):
            effects[f"position_{part}"] = values.loc[f"keep_last_{part}"] - values.loc[f"keep_first_{part}"]
        for name, total, components in [
                *[(s, effects[f"J_total_{s}"], effects[f"J_add_{s}"] + effects[f"R_{s}"])
                  for s in ("keep_first", "keep_last")],
                ("K", effects["K_total"], effects["K_add"] + effects["K_R"])]:
            a, b = total.to_numpy(dtype=float), components.to_numpy(dtype=float)
            if not np.allclose(a, b, atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError("Additivity identity failed")
            if audit:
                audit.check(f"{identity}/{name}/missing_identity", np.isnan(a), np.isnan(b), exact=True)
                mask = ~np.isnan(a)
                audit.check(f"{identity}/{name}/identity", a[mask], b[mask])
        for contrast, result in effects.items():
            rows.append({**dict(zip(IDENTITY, identity)), "contrast": contrast, **result.to_dict()})
    return pd.DataFrame(rows), columns


def summarize(directory, metrics, residuals, config, audit):
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(metrics) != 3456 or metrics.case.nunique() != 192 or set(map(tuple, metrics[factors].to_numpy())) != expected:
        raise ValueError("Incomplete factorial coverage")
    if metrics.duplicated(factors + ["variant", "score_kind"]).any() or set(metrics.score_kind) != {"scores", "logits"}:
        raise ValueError("Invalid factor/score identities")
    if len(residuals) != 384 or residuals.duplicated(factors + ["contrast"]).any():
        raise ValueError("Invalid residual coverage")
    effects, columns = paired_effects(metrics, audit)
    effects.to_csv(directory / "effects.csv", index=False)
    parent = pd.read_csv(config["parent_run"] + "/effects.csv").set_index(IDENTITY + ["contrast"])
    for row in effects[effects.contrast.isin(["J_total_keep_first", "J_total_keep_last", "K_total"])].to_dict("records"):
        name = "K" if row["contrast"] == "K_total" else row["contrast"].replace("J_total_", "J_")
        key = tuple(row[k] for k in IDENTITY)
        old = parent.loc[(*key, name), columns].to_numpy(dtype=float)
        actual = np.array([row[c] for c in columns], dtype=float)
        audit.check(f"{key}/{name}/parent_missing", np.isnan(actual), np.isnan(old), exact=True)
        mask = ~np.isnan(actual)
        audit.check(f"{key}/{name}/parent_effect", actual[mask], old[mask])
    groups = {"overall": [], "length": ["event_length"], "shape": ["shape"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    residual_columns = [c for c in residuals if c not in IDENTITY + ["contrast"]]
    for prefix, frame, selected in (("effect", effects, columns),
            ("absolute", metrics.rename(columns={"variant": "contrast"}), columns), ("residual", residuals, residual_columns)):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions, selected, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    primary = summary[(summary.score_kind == "logits") & summary.contrast.isin(PRIMARY) & (summary.metric == "average_precision")]
    lines = ["# E13：logit加性替代离线读出", "", "| 主要logits AP读出 | 均值 | 95%区间 |", "|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
