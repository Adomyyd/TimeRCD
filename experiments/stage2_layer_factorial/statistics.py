import numpy as np
import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary


IDENTITY = ["case", "seed", "phase", "shape", "event_length", "offset", "score_kind"]
VARIANTS = {"A"} | {f"{s}_{b}" for s in ("keep_first", "keep_last") for b in ("10", "01", "11")}


def assert_identity(left, right):
    a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.allclose(a[valid], b[valid], atol=1e-12, rtol=0):
        raise ValueError("Factorial identity failed")


def paired_effects(metrics):
    if metrics.empty or metrics[IDENTITY + ["variant"]].isna().any().any():
        raise ValueError("Missing observation identities")
    if metrics.duplicated(IDENTITY + ["variant"]).any():
        raise ValueError("Duplicate factorial observations")
    columns = [c for c in metrics if c not in IDENTITY + ["variant", "metric_status"]]
    if not np.isfinite(metrics["average_precision"]).all():
        raise ValueError("Undefined primary metric")
    rows = []
    for identity, group in metrics.groupby(IDENTITY, sort=True):
        if set(group.variant) != VARIANTS or not group.metric_status.eq("ok").all():
            raise ValueError("Missing or undefined factorial condition")
        values = group.set_index("variant")[columns]
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError("Infinite readout")
        matched = values.drop("A")
        if (matched.isna().any() != matched.isna().all()).any():
            raise ValueError("Unexpected missingness across matched masks")
        for variant, row in values.iterrows():
            for column in columns:
                if pd.isna(row[column]):
                    parts = column.split("_")
                    point_group = parts[1] if parts[0].startswith("r") else parts[0]
                    count_column = f"{point_group}_points"
                    if count_column in columns and row[count_column] != 0:
                        raise ValueError("Missing readout for nonempty point group")
        effects, interactions = {}, {}
        for selector in ("keep_first", "keep_last"):
            v10, v01, v11 = [values.loc[f"{selector}_{b}"] for b in ("10", "01", "11")]
            e70, e71 = v10 - values.loc["A"], v11 - v01
            e80, e81 = v01 - values.loc["A"], v11 - v10
            interaction = e71 - e70
            assert_identity(interaction, e81 - e80)
            interactions[selector] = interaction
            effects.update({f"{selector}_E7_given0": e70, f"{selector}_E7_given1": e71,
                            f"{selector}_E8_given0": e80, f"{selector}_E8_given1": e81,
                            f"{selector}_11_minus_A": v11 - values.loc["A"], f"J_{selector}": interaction})
        positions = {b: values.loc[f"keep_last_{b}"] - values.loc[f"keep_first_{b}"] for b in ("10", "01", "11")}
        effects.update({f"position_{b}": v for b, v in positions.items()})
        effects["K"] = interactions["keep_last"] - interactions["keep_first"]
        effects["position_11_minus_01"] = positions["11"] - positions["01"]
        assert_identity(effects["position_11_minus_01"], effects["K"] + positions["10"])
        for name, effect in effects.items():
            rows.append({**dict(zip(IDENTITY, identity)), "contrast": name, **effect.to_dict()})
    return pd.DataFrame(rows), columns


def summarize(directory, config):
    metrics = pd.read_csv(directory / "metrics.csv")
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(metrics) != 192 * 7 * 2 or metrics.case.nunique() != 192 or set(map(tuple, metrics[factors].to_numpy())) != expected:
        raise ValueError("Incomplete factorial coverage")
    if metrics.duplicated(factors + ["variant", "score_kind"]).any() or set(metrics.score_kind) != {"scores", "logits"}:
        raise ValueError("Invalid score/factor identities")
    effects, columns = paired_effects(metrics)
    effects.to_csv(directory / "effects.csv", index=False)
    groups = {"overall": [], "length": ["event_length"], "shape": ["shape"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    for prefix, frame in (("effect", effects), ("absolute", metrics.rename(columns={"variant": "contrast"}))):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions, columns, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    primary = summary[(summary.score_kind == "scores") & summary.contrast.isin(["J_keep_first", "J_keep_last", "K"])
                      & (summary.metric == "average_precision")]
    lines = ["# E12：最后两层2×2屏蔽开关", "", "| 主要AP交互 | 均值 | 95%区间 |", "|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
