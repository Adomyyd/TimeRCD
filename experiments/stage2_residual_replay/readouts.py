import numpy as np
import pandas as pd

from experiments.stage2_rank_localization.diagnostics import readout
from .core import GROUPS, replay


def compare_readouts(audit, identity, actual, expected):
    columns = [c for c in actual if c != "metric_status"]
    if isinstance(expected, pd.DataFrame) or expected.metric_status != "ok":
        raise ValueError("Invalid parent endpoint metric")
    a = np.array([actual[c] for c in columns], dtype=float)
    b = expected[columns].to_numpy(dtype=float)
    audit.check(f"{identity}/missing", np.isnan(a), np.isnan(b), exact=True)
    mask = ~np.isnan(a)
    audit.check(f"{identity}/readouts", a[mask], b[mask])


def analyze_case(meta, labels, fixed, roles, predictions, derived, config, parent_metrics, audit):
    base = {k: meta[k] for k in ("case", "seed", "phase", "shape", "event_length", "offset")}
    rows, arrays = [], {}
    for selector in ("keep_first", "keep_last"):
        additive, residual = [derived[f"{selector}_{k}"] for k in ("add_logits", "residual")]
        true_original = predictions[f"{selector}_11_logits"]
        true = true_original.astype(np.float64)
        audit.check(f"{meta['case']}/{selector}/residual", residual, true - additive, exact=True)
        for subset in range(16):
            identity = f"{meta['case']}/{selector}/{subset}"
            output = replay(additive, residual, fixed, subset)
            selected = np.zeros(len(labels), dtype=bool)
            for bit, name in enumerate(GROUPS):
                if subset & (1 << bit):
                    selected |= fixed[name]
            audit.check(f"{identity}/outside_unchanged", output[~selected], additive[~selected], exact=True)
            audit.check(f"{identity}/inside_replayed", output[selected], true[selected], exact=True)
            arrays[f"{selector}_{subset:02d}"] = output
            scoring_output = output.astype(true_original.dtype) if subset == 15 else output
            result = readout(labels, scoring_output, {**fixed, **roles[selector]}, config["recalls"], audit, identity)
            if subset in (0, 15):
                endpoint = f"{selector}_{'add' if subset == 0 else '11'}"
                compare_readouts(audit, identity + "/parent", result, parent_metrics.loc[(meta["case"], endpoint, "logits")])
            rows.append({**base, "score_kind": "logits", "selector": selector, "subset": subset,
                         "variant": f"{selector}_{subset:02d}", **result})
        if not fixed["I"].any():
            for subset in range(16):
                if not subset & 4:
                    audit.check(f"{meta['case']}/{selector}/{subset}/empty_I",
                                arrays[f"{selector}_{subset:02d}"], arrays[f"{selector}_{subset | 4:02d}"], exact=True)
    return rows, arrays
