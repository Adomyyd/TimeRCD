import numpy as np
import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary


IDENTITY = ["case", "seed", "phase", "shape", "event_length", "offset", "score_kind"]


def paired_effects(metrics, depths):
    if len(depths) < 2 or depths != sorted(set(depths)):
        raise ValueError("Depths must be unique and increasing")
    expected = {"A"} | {f"{s}_d{d}" for s in ("keep_first", "keep_last") for d in depths}
    if metrics.duplicated(IDENTITY + ["variant"]).any():
        raise ValueError("Duplicate depth observations")
    columns = [c for c in metrics if c not in IDENTITY + ["variant", "metric_status"]]
    rows = []
    for identity, group in metrics.groupby(IDENTITY, sort=True):
        if set(group.variant) != expected or not group.metric_status.eq("ok").all():
            raise ValueError("Missing or undefined depth condition")
        base = dict(zip(IDENTITY, identity))
        values = group.set_index("variant")[columns]
        matched = values.drop("A")
        if (matched.isna().any() != matched.isna().all()).any():
            raise ValueError("Unexpected missingness across matched masks")
        positions = {}
        for depth in depths:
            positions[depth] = values.loc[f"keep_last_d{depth}"] - values.loc[f"keep_first_d{depth}"]
            rows.append({**base, "contrast": f"position_d{depth}", **positions[depth].to_dict()})
            for selector in ("keep_first", "keep_last"):
                name = f"{selector}_d{depth}"
                rows.append({**base, "contrast": f"{name}_minus_A", **(values.loc[name] - values.loc["A"]).to_dict()})
        increments = []
        for left, right in zip(depths[:-1], depths[1:]):
            increment = positions[right] - positions[left]
            increments.append(increment)
            rows.append({**base, "contrast": f"position_d{right}_minus_d{left}", **increment.to_dict()})
            for selector in ("keep_first", "keep_last"):
                difference = values.loc[f"{selector}_d{right}"] - values.loc[f"{selector}_d{left}"]
                rows.append({**base, "contrast": f"{selector}_d{right}_minus_d{left}", **difference.to_dict()})
        total = positions[depths[-1]] - positions[depths[0]]
        if not np.allclose(sum(increments).to_numpy(dtype=float), total.to_numpy(dtype=float), atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError("Depth interaction telescoping identity failed")
        rows.append({**base, "contrast": "position_endpoint_interaction", **total.to_dict()})
    return pd.DataFrame(rows), columns


def summarize(directory, config):
    metrics = pd.read_csv(directory / "metrics.csv")
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(metrics) != 192 * 9 * 2 or metrics.case.nunique() != 192 or set(map(tuple, metrics[factors].to_numpy())) != expected:
        raise ValueError("Incomplete factorial coverage")
    if metrics.duplicated(factors + ["variant", "score_kind"]).any() or set(metrics.score_kind) != {"scores", "logits"}:
        raise ValueError("Invalid score/factor identities")
    effects, columns = paired_effects(metrics, config["depths"])
    effects.to_csv(directory / "effects.csv", index=False)
    groups = {"overall": [], "length": ["event_length"], "shape": ["shape"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    for prefix, frame in (("effect", effects), ("absolute", metrics.rename(columns={"variant": "contrast"}))):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions, columns, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    names = [f"position_d{b}_minus_d{a}" for a, b in zip(config["depths"][:-1], config["depths"][1:])]
    primary = summary[(summary.score_kind == "scores") & summary.contrast.isin(names) & (summary.metric == "average_precision")]
    lines = ["# E11：累积屏蔽深度", "", "| 位置AP差的相邻变化 | 均值 | 95%区间 |", "|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
