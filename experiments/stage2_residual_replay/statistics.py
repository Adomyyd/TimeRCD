import numpy as np
import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary
from .core import GROUPS, permutation_shapley, shapley


IDENTITY = ["case", "seed", "phase", "shape", "event_length", "offset", "score_kind"]
SELECTORS = ("keep_first", "keep_last")


def metric_columns(recalls):
    return ["average_precision", "roc_auc"] + [f"r{round(100 * r):03d}_{kind}" for r in recalls
            for kind in ("precision", "fpr")] + [f"{g}_rank_mean" for g in GROUPS]


def compare(audit, name, a, b, exact=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError(f"Unexpected missingness: {name}")
    mask = ~np.isnan(a)
    if audit:
        audit.check(name + "/missing", np.isnan(a), np.isnan(b), exact=True)
        audit.check(name, a[mask], b[mask], exact=exact)
    elif not (np.array_equal(a[mask], b[mask]) if exact else np.allclose(a[mask], b[mask], atol=1e-12, rtol=0)):
        raise ValueError(f"Failed identity: {name}")


def paired_effects(metrics, columns, audit=None):
    keys = IDENTITY + ["selector", "subset", "variant"]
    if metrics.empty or metrics[keys].isna().any().any() or metrics.duplicated(IDENTITY + ["selector", "subset"]).any():
        raise ValueError("Missing or duplicated subset identities")
    if not metrics.metric_status.eq("ok").all() or not np.isfinite(metrics[["average_precision", "roc_auc"]]).all().all():
        raise ValueError("Undefined primary readouts")
    if not metrics.score_kind.eq("logits").all():
        raise ValueError("Expected logits readouts")
    rows, marginal_rows, context_rows = [], [], []
    for identity, block in metrics.groupby(IDENTITY, sort=True):
        if set(block.selector) != set(SELECTORS):
            raise ValueError("Missing selector")
        base = dict(zip(IDENTITY, identity))
        phis, totals = {}, {}
        for selector in SELECTORS:
            group = block[block.selector == selector].sort_values("subset")
            if group.subset.tolist() != list(range(16)) or group.variant.tolist() != [f"{selector}_{i:02d}" for i in range(16)]:
                raise ValueError("Incomplete subset coverage")
            if "I_points" in group:
                if group.I_points.nunique() != 1 or (group.I_points.iloc[0] == 0) != (base["event_length"] == 32):
                    raise ValueError("Inconsistent empty I group")
            required = [c for c in columns if c != "I_rank_mean" or base["event_length"] != 32]
            if not np.isfinite(group[required].to_numpy(dtype=float)).all():
                raise ValueError("Undefined nonempty-group or detection readout")
            values = group[columns].to_numpy(dtype=float)
            phi, margins = shapley(values)
            compare(audit, f"{identity}/{selector}/permutations", phi, permutation_shapley(values))
            total = values[15] - values[0]
            compare(audit, f"{identity}/{selector}/efficiency", phi.sum(axis=0), total)
            if base["event_length"] == 32:
                compare(audit, f"{identity}/{selector}/empty_I_phi", phi[2, :2], np.zeros(2), exact=True)
                for subset in range(16):
                    if not subset & 4:
                        compare(audit, f"{identity}/{selector}/{subset}/empty_I_metrics", values[subset], values[subset | 4], exact=True)
            phis[selector], totals[selector] = phi, total
            for bit, subset, weight, delta in margins:
                marginal_rows.append({**base, "selector": selector, "group": GROUPS[bit], "context": subset,
                                      "weight": weight, **dict(zip(columns, delta))})
            for bit, name in enumerate(GROUPS):
                rows.append({**base, "contrast": f"phi_{selector}_{name}", **dict(zip(columns, phi[bit]))})
                context = {**base, "contrast": f"{selector}_{name}"}
                deltas = np.stack([d for b, _, _, d in margins if b == bit])
                for index, metric in enumerate(columns[:2]):
                    context.update({f"{metric}_min": deltas[:, index].min(), f"{metric}_max": deltas[:, index].max(),
                                    f"{metric}_alone": values[1 << bit, index] - values[0, index],
                                    f"{metric}_last": values[15, index] - values[15 ^ (1 << bit), index]})
                context_rows.append(context)
            rows.append({**base, "contrast": f"R_{selector}", **dict(zip(columns, total))})
        difference = phis["keep_last"] - phis["keep_first"]
        total = totals["keep_last"] - totals["keep_first"]
        compare(audit, f"{identity}/difference_efficiency", difference.sum(axis=0), total)
        for bit, name in enumerate(GROUPS):
            rows.append({**base, "contrast": f"D_{name}", **dict(zip(columns, difference[bit]))})
        rows.append({**base, "contrast": "K_R", **dict(zip(columns, total))})
    return pd.DataFrame(rows), pd.DataFrame(marginal_rows), pd.DataFrame(context_rows)


def summarize(directory, metrics, config, audit):
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(metrics) != 6144 or metrics.case.nunique() != 192 or set(map(tuple, metrics[factors].to_numpy())) != expected:
        raise ValueError("Incomplete factorial coverage")
    if metrics.duplicated(factors + ["selector", "subset"]).any():
        raise ValueError("Duplicate factor identities")
    columns = metric_columns(config["recalls"])
    effects, margins, contexts = paired_effects(metrics, columns, audit)
    effects.to_csv(directory / "effects.csv", index=False)
    margins.to_csv(directory / "marginals.csv", index=False)
    contexts.to_csv(directory / "contexts.csv", index=False)
    parent = pd.read_csv(config["parent_run"] + "/effects.csv").set_index(IDENTITY + ["contrast"])
    if not parent.index.is_unique:
        raise ValueError("Duplicate parent effects")
    for row in effects[effects.contrast.isin(["R_keep_first", "R_keep_last", "K_R"])].to_dict("records"):
        key = tuple(row[k] for k in IDENTITY) + (row["contrast"],)
        compare(audit, f"{key}/E13_effect", [row[c] for c in columns], parent.loc[key, columns].to_numpy(dtype=float))
    groups = {"overall": [], "length": ["event_length"], "shape": ["shape"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    context_columns = [c for c in contexts if c not in IDENTITY + ["contrast"]]
    for prefix, frame, selected in (("effect", effects, columns),
            ("absolute", metrics.rename(columns={"variant": "contrast"}), columns), ("context", contexts, context_columns)):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions, selected, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    primary = summary[(summary.metric == "average_precision") & summary.contrast.str.match("phi_|D_")]
    lines = ["# E14：固定点组残差回放", "", "| 主要AP归因 | 均值 | 95%区间 |", "|---|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['contrast']} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
