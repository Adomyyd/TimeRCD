import numpy as np

from experiments.stage2_normal_replay_transitions.core import transition_counts
from experiments.stage2_normal_replay_transitions.data import SELECTORS
from experiments.stage2_normal_replay_transitions.readouts import BASE_KEYS
from experiments.stage2_rank_localization.diagnostics import readout
from experiments.stage2_residual_replay.readouts import compare_readouts
from .core import CELLS, EDGES, cross_outputs, cross_memberships, factorial_effects, point_interaction


def analyze_case(meta, labels, fixed, roles, predictions, outputs, endpoints, old_memberships, config, metrics, audit):
    base = {**{k: meta[k] for k in BASE_KEYS}, "score_kind": "logits"}
    rows, transitions, interactions, arrays = [], [], [], {}
    if set(endpoints) != set(SELECTORS):
        raise ValueError("Missing cross selector")
    for selector in SELECTORS:
        identity = f"{meta['case']}/{selector}"
        crossed = cross_outputs(labels, outputs[selector]["add"], outputs[selector]["full"])
        if set(endpoints[selector]) != set(CELLS):
            raise ValueError("Missing replay endpoint")
        for cell, (_, _, subset) in CELLS.items():
            audit.check(f"{identity}/{cell}/endpoint", crossed[cell], endpoints[selector][cell], exact=True)
            values = crossed[cell]
            if cell == "11":
                values = values.astype(predictions[f"{selector}_11_logits"].dtype)
            compare_readouts(audit, f"{identity}/{cell}/E14", readout(labels, values, {**fixed, **roles[selector]}, config["recalls"]),
                             metrics.loc[(meta["case"], f"{selector}_{subset:02d}", "logits")])
        for recall in config["recalls"]:
            prefix = f"{selector}_r{round(recall * 100):03d}"
            masks, diagnostics = cross_memberships(labels, crossed, recall)
            for cell, row in diagnostics.items():
                s, t, _ = CELLS[cell]
                audit.check(f"{identity}/{recall}/{cell}/threshold_source", row["threshold"], diagnostics[f"0{t}"]["threshold"], exact=True)
                audit.check(f"{identity}/{recall}/{cell}/positive_source", masks[cell][labels == 1], masks[f"0{t}"][labels == 1], exact=True)
                if cell in ("00", "10", "11"):
                    old = {"00": "add", "10": "normal", "11": "full"}[cell]
                    audit.check(f"{identity}/{recall}/{cell}/E15", masks[cell],
                                old_memberships[f"{selector}_{old}_r{round(recall * 100):03d}"], exact=True)
                rows.append({**base, "selector": selector, "target_recall": recall, "cell": cell,
                             "normal_source": s, "threshold_source": t, **row})
                arrays[f"{prefix}_{cell}"] = masks[cell]
            for name, (before, after) in EDGES.items():
                counts = transition_counts(masks[before], masks[after], fixed["N"])
                audit.check(f"{identity}/{recall}/{name}/net", counts["net_selected"],
                            diagnostics[after]["fp"] - diagnostics[before]["fp"], exact=True)
                audit.check(f"{identity}/{recall}/{name}/partition", sum(counts[k] for k in ("entered", "exited", "retained", "absent")), int(fixed["N"].sum()), exact=True)
                audit.check(f"{identity}/{recall}/{name}/xor", counts["changed"], np.sum((masks[before] ^ masks[after]) & fixed["N"]), exact=True)
                if name.startswith("threshold_"):
                    delta = diagnostics[after]["threshold"] - diagnostics[before]["threshold"]
                    audit.check(f"{identity}/{recall}/{name}/nested", counts["entered"] if delta >= 0 else counts["exited"], 0, exact=True)
                transitions.append({**base, "selector": selector, "target_recall": recall, "contrast": name, **counts})
            z, counts = point_interaction(masks, fixed["N"])
            effects = factorial_effects({c: d["fp"] for c, d in diagnostics.items()})
            audit.check(f"{identity}/{recall}/interaction_sum", counts["net"], effects["interaction"], exact=True)
            arrays[f"{prefix}_interaction"] = z
            interactions.append({**base, "selector": selector, "target_recall": recall, **counts})
    return rows, transitions, interactions, arrays
