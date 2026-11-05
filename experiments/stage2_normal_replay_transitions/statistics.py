import numpy as np
import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_residual_replay.statistics import compare
from .data import CONDITIONS, SELECTORS
from .readouts import BASE_KEYS, GROUPS


DETECTION = ["tp", "fn", "fp", "tn", "selected", "precision", "recall", "fpr", "selected_fraction", "threshold"]


def validate_factor_coverage(thresholds, config):
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    factors = ["seed", "phase", "shape", "event_length", "offset"]
    identities = thresholds[BASE_KEYS].drop_duplicates()
    if (len(identities) != len(expected) or identities.case.nunique() != len(expected) or identities.duplicated(factors).any()
            or set(map(tuple, identities[factors].to_numpy())) != expected):
        raise ValueError("Incomplete factorial coverage")


def paired_effects(thresholds, transitions, recalls, audit=None):
    keys = BASE_KEYS + ["score_kind", "selector", "target_recall"]
    for frame, extra in ((thresholds, ["condition"]), (transitions, ["condition", "group"])):
        if frame.empty or frame[keys + extra].isna().any().any() or frame.duplicated(keys + extra).any():
            raise ValueError("Missing or duplicate transition identities")
        if not frame.score_kind.eq("logits").all():
            raise ValueError("Expected logits")
    if not np.isfinite(thresholds[DETECTION]).all().all():
        raise ValueError("Undefined threshold diagnostics")
    count_columns = ["points", "entered", "exited", "retained", "absent", "before_selected", "after_selected", "net_selected", "changed"]
    if not np.isfinite(transitions[count_columns]).all().all():
        raise ValueError("Undefined transition counts")
    counts = transitions[count_columns].to_numpy()
    if not np.equal(counts, np.floor(counts)).all() or (transitions[count_columns[:-2]] < 0).any().any():
        raise ValueError("Invalid integer counts")
    identity_sets = []
    for frame in (thresholds, transitions):
        identity_sets.append(set(map(tuple, frame[keys].to_numpy())))
        for _, block in frame.groupby(BASE_KEYS, sort=False):
            if set(zip(block.selector, block.target_recall)) != {(s, r) for s in SELECTORS for r in recalls}:
                raise ValueError("Incomplete selectors or recall targets")
    if identity_sets[0] != identity_sets[1]:
        raise ValueError("Unpaired threshold and transition identities")
    indexed = transitions.set_index(keys).sort_index()
    rows = []
    for identity, block in thresholds.groupby(keys, sort=True):
        if len(block) != 3 or set(block.condition) != set(CONDITIONS):
            raise ValueError("Missing threshold condition")
        baseline = block.set_index("condition").loc["add"]
        group = indexed.loc[[identity]]
        if len(group) != 14 or set(zip(group.condition, group["group"])) != {(c, g) for c in ("normal", "full") for g in GROUPS}:
            raise ValueError("Incomplete group transitions")
        for condition in ("normal", "full"):
            after = block.set_index("condition").loc[condition]
            selected = group[group.condition == condition].set_index("group").loc[list(GROUPS)]
            row = {**dict(zip(keys, identity)), "contrast": f"{condition}_minus_add"}
            row.update({f"delta_{k}": float(after[k] - baseline[k]) for k in DETECTION})
            for name, item in selected.iterrows():
                compare(audit, f"{identity}/{condition}/{name}/identities",
                        [item.points, item.before_selected, item.after_selected, item.net_selected, item.changed],
                        [item.entered + item.exited + item.retained + item.absent, item.retained + item.exited,
                         item.retained + item.entered, item.after_selected - item.before_selected, item.entered + item.exited], exact=True)
                for metric in ("entered", "exited", "retained", "absent", "net_selected"):
                    row[f"{name}_{metric}"] = int(item[metric])
                if name == "ALL":
                    row["jaccard"] = item.jaccard
                else:
                    compare(audit, f"{identity}/{condition}/{name}/selected",
                            [item.before_selected, item.after_selected], [baseline[f"{name}_selected"], after[f"{name}_selected"]], exact=True)
            for names in (("F", "L", "I", "N"), ("K", "B", "N")):
                for metric in count_columns:
                    compare(audit, f"{identity}/{condition}/{names}/{metric}", selected.loc[list(names), metric].sum(), selected.loc["ALL", metric], exact=True)
            compare(audit, f"{identity}/{condition}/global_deltas", [row["delta_selected"], row["delta_fp"], row["delta_fn"]],
                    [row["ALL_net_selected"], row["N_net_selected"], -sum(row[f"{g}_net_selected"] for g in ("F", "L", "I"))], exact=True)
            if condition == "normal":
                compare(audit, f"{identity}/normal_positive_invariance",
                        [row[f"{g}_{m}"] for g in ("F", "L", "I", "K", "B") for m in ("entered", "exited")]
                        + [row["delta_threshold"], row["delta_tp"], row["delta_fn"], row["delta_recall"]], np.zeros(14), exact=True)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(directory, thresholds, transitions, config, audit):
    validate_factor_coverage(thresholds, config)
    effects = paired_effects(thresholds, transitions, config["recalls"], audit)
    effects.to_csv(directory / "effects.csv", index=False)
    effect_columns = [c for c in effects if c not in BASE_KEYS + ["score_kind", "selector", "target_recall", "contrast"]]
    absolute_columns = DETECTION + [f"{g}_{m}" for g in GROUPS[1:] for m in ("selected", "capture")]
    groups = {"overall": [], "length": ["event_length"], "shape": ["shape"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    for prefix, frame, columns in (("effect", effects, effect_columns),
            ("absolute", thresholds.rename(columns={"condition": "contrast"}), absolute_columns)):
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions + ["selector", "target_recall"], columns, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    primary = summary[(summary.contrast == "normal_minus_add") & (summary.metric == "N_net_selected")]
    lines = ["# E15：正常点残差阈值集合转移", "", "| 选择 | 目标召回 | 净误报变化/序列 | 95%区间 |", "|---|---:|---:|---|"]
    for row in primary.to_dict("records"):
        lines.append(f"| {row['selector']} | {row['target_recall']:.2f} | {row['mean']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
