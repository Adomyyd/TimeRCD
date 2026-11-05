import numpy as np
import pandas as pd

from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_normal_replay_transitions.readouts import BASE_KEYS
from experiments.stage2_normal_replay_transitions.statistics import validate_factor_coverage
from experiments.stage2_residual_replay.statistics import compare
from .core import CELLS, EDGES, factorial_effects


KEYS = BASE_KEYS + ["score_kind", "selector", "target_recall"]
ABSOLUTE = ["tp", "fn", "fp", "tn", "selected", "precision", "recall", "fpr", "selected_fraction", "threshold", "strict_recall"]
TRANSITIONS = ["points", "entered", "exited", "retained", "absent", "before_selected", "after_selected", "net_selected", "changed", "jaccard"]


def paired_effects(readouts, transitions, interactions, recalls, audit=None):
    identity_sets = []
    for frame, extra in ((readouts, ["cell"]), (transitions, ["contrast"]), (interactions, [])):
        if frame.empty or frame[KEYS + extra].isna().any().any() or frame.duplicated(KEYS + extra).any():
            raise ValueError("Missing or duplicate cross identities")
        if not frame.score_kind.eq("logits").all():
            raise ValueError("Expected logits")
        for _, block in frame.groupby(BASE_KEYS, sort=False):
            if set(zip(block.selector, block.target_recall)) != {(s, r) for s in ("keep_first", "keep_last") for r in recalls}:
                raise ValueError("Incomplete cross selectors or recalls")
        identity_sets.append(set(map(tuple, frame[KEYS].to_numpy())))
    if not identity_sets[0] == identity_sets[1] == identity_sets[2]:
        raise ValueError("Unpaired cross tables")
    if not np.isfinite(readouts[ABSOLUTE]).all().all():
        raise ValueError("Undefined cross readout")
    indexed_t = transitions.set_index(KEYS).sort_index()
    indexed_i = interactions.set_index(KEYS).sort_index()
    rows = []
    for identity, block in readouts.groupby(KEYS, sort=True):
        if len(block) != 4 or set(block.cell) != set(CELLS):
            raise ValueError("Incomplete four-cell cross")
        b = block.set_index("cell").loc[list(CELLS)]
        n = b.fp + b.tn
        p = b.tp + b.fn
        counts = b[["tp", "fn", "fp", "tn", "selected"]].to_numpy()
        if not np.equal(counts, np.floor(counts)).all() or (counts < 0).any() or n.min() <= 0 or p.min() <= 0:
            raise ValueError("Invalid confusion counts")
        compare(audit, f"{identity}/classes", np.r_[n, p], np.r_[np.repeat(n.iloc[0], 4), np.repeat(p.iloc[0], 4)], exact=True)
        compare(audit, f"{identity}/rates", np.r_[b.fpr, b.recall, b.precision, b.selected_fraction],
                np.r_[b.fp / n, b.tp / p, b.tp / b.selected, b.selected / (p + n)])
        compare(audit, f"{identity}/selected", b.selected, b.tp + b.fp, exact=True)
        compare(audit, f"{identity}/source_codes", b[["normal_source", "threshold_source"]].to_numpy(),
                np.array([CELLS[c][:2] for c in CELLS]), exact=True)
        for before, after in (("00", "10"), ("01", "11")):
            compare(audit, f"{identity}/{before}/{after}/positive", b.loc[before, ["threshold", "tp", "fn", "recall"]],
                    b.loc[after, ["threshold", "tp", "fn", "recall"]], exact=True)
        effects = {metric: factorial_effects(b[metric].to_dict()) for metric in ("fp", "fpr")}
        for metric, e in effects.items():
            compare(audit, f"{identity}/{metric}/decomposition", [e["total"]] * 4 + [e["interaction"]] * 2,
                    [e["score_at_add"] + e["threshold_at_add"] + e["interaction"],
                     e["score_at_add"] + e["threshold_at_full"], e["threshold_at_add"] + e["score_at_full"],
                     e["score_average"] + e["threshold_average"],
                     e["score_at_full"] - e["score_at_add"], e["threshold_at_full"] - e["threshold_at_add"]], exact=metric == "fp")
        for name in effects["fp"]:
            rows.append({**dict(zip(KEYS, identity)), "contrast": name,
                         "fp": effects["fp"][name], "fpr": effects["fpr"][name]})
        group = indexed_t.loc[[identity]]
        if len(group) != len(EDGES) or set(group.contrast) != set(EDGES):
            raise ValueError("Incomplete transition edges")
        for name, (before, after) in EDGES.items():
            item = group[group.contrast == name].iloc[0]
            values = item[TRANSITIONS[:-1]].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values[[0, 1, 2, 3, 4, 5, 6, 8]] < 0).any():
                raise ValueError("Invalid transition counts")
            compare(audit, f"{identity}/{name}/transition", values,
                    [n.iloc[0], item.entered, item.exited, item.retained, n.iloc[0] - item.entered - item.exited - item.retained,
                     b.loc[before, "fp"], b.loc[after, "fp"], effects["fp"][name], item.entered + item.exited], exact=True)
            compare(audit, f"{identity}/{name}/states", [item.before_selected, item.after_selected, item.net_selected],
                    [item.retained + item.exited, item.retained + item.entered, item.entered - item.exited], exact=True)
            union = item.entered + item.exited + item.retained
            compare(audit, f"{identity}/{name}/jaccard", item.jaccard, item.retained / union if union else np.nan)
            if name.startswith("threshold_"):
                delta = b.loc[after, "threshold"] - b.loc[before, "threshold"]
                compare(audit, f"{identity}/{name}/nested", item.entered if delta >= 0 else item.exited, 0, exact=True)
        item = indexed_i.loc[identity]
        v = item[["positive", "negative", "zero", "net", "points"]].to_numpy(dtype=float)
        if not np.isfinite(v).all() or not np.equal(v, np.floor(v)).all() or (v[[0, 1, 2, 4]] < 0).any():
            raise ValueError("Invalid point interaction counts")
        compare(audit, f"{identity}/point_interaction", [item.positive + item.negative + item.zero, item.points, item.net],
                [n.iloc[0], n.iloc[0], item.positive - item.negative], exact=True)
        compare(audit, f"{identity}/interaction_net", item.net, effects["fp"]["interaction"], exact=True)
    return pd.DataFrame(rows)


def summarize(directory, readouts, transitions, interactions, config, audit):
    validate_factor_coverage(readouts, config)
    effects = paired_effects(readouts, transitions, interactions, config["recalls"], audit)
    effects.to_csv(directory / "effects.csv", index=False)
    groups = {"overall": [], "length": ["event_length"], "shape": ["shape"], "phase": ["phase"],
              "cells": ["phase", "shape", "event_length"]}
    frames = (("effect", effects, ["fp", "fpr"]),
              ("absolute", readouts.rename(columns={"cell": "contrast"}), ABSOLUTE),
              ("transition", transitions, TRANSITIONS),
              ("point_interaction", interactions.assign(contrast="interaction"), ["positive", "negative", "zero", "net", "points"]))
    for prefix, frame, columns in frames:
        for name, dimensions in groups.items():
            seeds, summary = seed_summary(frame, dimensions + ["selector", "target_recall"], columns, config)
            seeds.to_csv(directory / f"seed_{prefix}_{name}.csv", index=False)
            summary.to_csv(directory / f"summary_{prefix}_{name}.csv", index=False)
    summary = pd.read_csv(directory / "summary_effect_overall.csv")
    table = summary[summary.metric.eq("fp")].pivot(index=["selector", "target_recall"], columns="contrast", values="mean")
    lines = ["# E16：正常分数来源×正例阈值来源", "", "| 选择 | q | S0 | T0 | I | 总FP差 |", "|---|---:|---:|---:|---:|---:|"]
    for (selector, recall), row in table.iterrows():
        lines.append(f"| {selector} | {recall:.2f} | {row.score_at_add:+.3f} | {row.threshold_at_add:+.3f} | {row.interaction:+.3f} | {row.total:+.3f} |")
    report = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(report)
    return report
