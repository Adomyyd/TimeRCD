import numpy as np

from experiments.stage2_logit_additivity.core import additive_output
from experiments.stage2_layer_factorial.statistics import VARIANTS
from experiments.stage2_rank_localization.diagnostics import readout
from experiments.stage2_residual_replay.core import replay
from experiments.stage2_residual_replay.readouts import compare_readouts
from .core import threshold_membership, transition_counts
from .data import CONDITIONS, SELECTORS


BASE_KEYS = ["case", "seed", "phase", "shape", "event_length", "offset"]
GROUPS = ("ALL", "F", "L", "I", "N", "K", "B")


def analyze_case(meta, labels, fixed, roles, predictions, derived, outputs, config, parent_metrics, originals, audit):
    if set(predictions) != {f"{v}_{k}" for v in VARIANTS for k in ("logits", "scores")}:
        raise ValueError("Incomplete original E13 conditions")
    if set(outputs) != set(SELECTORS) or any(set(b) != set(CONDITIONS) for b in outputs.values()):
        raise ValueError("Incomplete E15 outputs")
    base = {k: meta[k] for k in BASE_KEYS}
    rows, transitions, memberships = [], [], {}
    # Recompute every E13 original/derived readout, preserving original float32 means.
    all_outputs = {**predictions, **{k: derived[k] for s in SELECTORS for k in (f"{s}_add_logits", f"{s}_add_scores")}}
    for name, values in all_outputs.items():
        variant, kind = name.rsplit("_", 1)
        selector = "A" if variant == "A" else variant.rsplit("_", 1)[0]
        result = readout(labels, values, {**fixed, **roles[selector]}, config["recalls"])
        compare_readouts(audit, f"{meta['case']}/{name}/E13", result, originals.loc[(meta["case"], variant, kind)])
    for selector in SELECTORS:
        groups = {**fixed, **roles[selector]}
        groups_all = {"ALL": np.ones(len(labels), dtype=bool), **groups}
        block = outputs[selector]
        identity = f"{meta['case']}/{selector}"
        additive, residual = additive_output(*[predictions[f"{v}_logits"] for v in
                                              ("A", f"{selector}_10", f"{selector}_01", f"{selector}_11")])
        for name, actual, expected in (("add", block["add"], derived[f"{selector}_add_logits"]),
                ("reconstructed_add", additive, block["add"]), ("residual", residual, derived[f"{selector}_residual"]),
                ("normal", block["normal"], replay(additive, residual, fixed, 8)),
                ("full", block["full"], predictions[f"{selector}_11_logits"].astype(np.float64))):
            audit.check(f"{identity}/{name}/endpoint", actual, expected, exact=True)
        audit.check(f"{identity}/normal_positive_unchanged", block["normal"][labels == 1], block["add"][labels == 1], exact=True)
        selected_by_condition, diagnostics = {}, {}
        for condition, subset in CONDITIONS.items():
            scores = block[condition]
            scoring = scores.astype(predictions[f"{selector}_11_logits"].dtype) if condition == "full" else scores
            result = readout(labels, scoring, groups, config["recalls"])
            compare_readouts(audit, f"{identity}/{condition}/E14", result,
                             parent_metrics.loc[(meta["case"], f"{selector}_{subset:02d}", "logits")])
            for recall in config["recalls"]:
                prefix = f"r{round(100 * recall):03d}"
                selected, row = threshold_membership(labels, scores, recall)
                key = f"{selector}_{condition}_{prefix}"
                memberships[key] = selected
                selected_by_condition[condition, recall] = selected
                for group, mask in groups.items():
                    count = int(np.sum(selected & mask))
                    row.update({f"{group}_selected": count, f"{group}_capture": count / mask.sum() if mask.any() else np.nan})
                check_id = f"{identity}/{condition}/{prefix}"
                fields = ("threshold", "precision", "recall", "fpr", "strict_recall", "selected_fraction")
                audit.check(check_id + "/parent_threshold", [row[f] for f in fields], [result[f"{prefix}_{f}"] for f in fields])
                audit.check(check_id + "/counts", [row["tp"] + row["fn"], row["fp"] + row["tn"],
                            row["selected"], sum(row[f"{g}_selected"] for g in fixed)],
                            [int(np.sum(labels == 1)), int(np.sum(labels == 0)), row["tp"] + row["fp"], row["selected"]], exact=True)
                for g in groups:
                    capture = result[f"{prefix}_{g}_capture"]
                    audit.check(check_id + f"/{g}/missing", np.isnan(row[f"{g}_capture"]), np.isnan(capture), exact=True)
                    if not np.isnan(capture):
                        audit.check(check_id + f"/{g}/capture", row[f"{g}_capture"], capture)
                diagnostics[condition, recall] = row
                rows.append({**base, "score_kind": "logits", "selector": selector, "condition": condition,
                             "target_recall": recall, **row})
        for condition in ("normal", "full"):
            for recall in config["recalls"]:
                before, after = [selected_by_condition[c, recall] for c in ("add", condition)]
                old, new = [diagnostics[c, recall] for c in ("add", condition)]
                if condition == "normal":
                    audit.check(f"{identity}/{recall}/normal_threshold", new["threshold"], old["threshold"], exact=True)
                    audit.check(f"{identity}/{recall}/normal_positive_set", after[labels == 1], before[labels == 1], exact=True)
                for group, mask in groups_all.items():
                    counts = transition_counts(before, after, mask)
                    check_id = f"{identity}/{condition}/{recall}/{group}"
                    audit.check(check_id + "/partition", sum(counts[k] for k in ("entered", "exited", "retained", "absent")), mask.sum(), exact=True)
                    audit.check(check_id + "/net", counts["net_selected"], int(after[mask].sum()) - int(before[mask].sum()), exact=True)
                    audit.check(check_id + "/xor", counts["changed"], np.sum((before ^ after) & mask), exact=True)
                    transitions.append({**base, "score_kind": "logits", "selector": selector,
                                        "condition": condition, "target_recall": recall, "group": group, **counts})
    return rows, transitions, memberships
