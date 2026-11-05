import numpy as np
import pandas as pd

from experiments.stage2_layer_factorial.statistics import VARIANTS
from experiments.stage2_rank_localization.diagnostics import readout
from .core import additive_output, residual_readout, sigmoid


def analyze_case(meta, labels, fixed, roles, predictions, config, parent_metrics, audit):
    expected = {f"{v}_{k}" for v in VARIANTS for k in ("scores", "logits")}
    if set(predictions) != expected:
        raise ValueError("Missing or unexpected saved conditions")
    if len(labels) != config["window"] or any(
            np.asarray(a).shape != labels.shape or not np.isfinite(a).all() for a in predictions.values()):
        raise ValueError("Invalid saved output shape or values")
    for variant in sorted(VARIANTS):
        error = float(np.max(np.abs(sigmoid(predictions[f"{variant}_logits"]) - predictions[f"{variant}_scores"])))
        audit.check(f"{meta['case']}/{variant}/sigmoid_error", error <= config["sigmoid_tolerance"], True, exact=True)
    derived, residuals = {}, []
    base = {k: meta[k] for k in ("case", "seed", "phase", "shape", "event_length", "offset")}
    for selector in ("keep_first", "keep_last"):
        l00, l10, l01, l11 = [np.asarray(predictions[f"{v}_logits"], dtype=np.float64)
                             for v in ("A", f"{selector}_10", f"{selector}_01", f"{selector}_11")]
        additive, residual = additive_output(l00, l10, l01, l11)
        audit.check(f"{meta['case']}/{selector}/residual_identity", residual, (l11 - l01) - (l10 - l00))
        audit.check(f"{meta['case']}/{selector}/reconstruction", additive + residual, l11)
        derived.update({f"{selector}_add_logits": additive, f"{selector}_add_scores": sigmoid(additive),
                        f"{selector}_residual": residual})
        residuals.append({**base, "score_kind": "logits", "contrast": selector,
                          **residual_readout(residual, l10 - l00, l01 - l00, fixed)})
    rows = []
    for variant in sorted(VARIANTS) + ["keep_first_add", "keep_last_add"]:
        selector = "A" if variant == "A" else variant.rsplit("_", 1)[0]
        for kind in ("logits", "scores"):
            scores = (predictions if variant in VARIANTS else derived)[f"{variant}_{kind}"]
            identity = f"{meta['case']}/{variant}/{kind}"
            result = readout(labels, scores, {**fixed, **roles[selector]}, config["recalls"], audit, identity)
            if variant in VARIANTS:
                previous = parent_metrics.loc[(meta["case"], variant, kind)]
                if isinstance(previous, pd.DataFrame) or previous.metric_status != "ok":
                    raise ValueError("Invalid parent metric identity")
                columns = [c for c in result if c != "metric_status"]
                actual = np.array([result[c] for c in columns], dtype=float)
                old = previous[columns].to_numpy(dtype=float)
                audit.check(f"{identity}/parent_missing", np.isnan(actual), np.isnan(old), exact=True)
                mask = ~np.isnan(actual)
                audit.check(f"{identity}/parent_readouts", actual[mask], old[mask])
            rows.append({**base, "score_kind": kind, "variant": variant, **result})
    return rows, residuals, derived
