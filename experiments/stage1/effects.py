import numpy as np

from experiments.common.artifacts import append_row
from experiments.common.metrics import detection_metrics, difference


def assemble_conditions(arrays, tail_length, has_outer_padding):
    current = {key: arrays[f"public_{key}"] for key in ("scores", "logits")}
    length = len(current["scores"])
    if not 0 < tail_length <= length:
        raise ValueError("Invalid tail length")
    conditions = {"current": current}
    for variant, source in (("tail_replay", "current"), ("zero", "zero"), ("compact", "compact")):
        condition = {}
        for key in ("scores", "logits"):
            candidate = current[key].copy()
            tail = arrays[f"tail_{source}_{key}"]
            if tail.shape != (tail_length,):
                raise ValueError("Tail prediction length mismatch")
            # A no-op intervention must not introduce batch-shape roundoff.
            if has_outer_padding or variant == "tail_replay":
                candidate[-tail_length:] = tail
            condition[key] = candidate
        conditions[variant] = condition
    return conditions


def record_effects(directory, config, base, labels, conditions):
    valid = base["tail_valid_length"]
    tail_labels = labels[-valid:]
    for kind in ("scores", "logits"):
        reference = conditions["current"][kind]
        baseline = detection_metrics(labels, reference)
        for variant, prediction in conditions.items():
            values = prediction[kind]
            metrics = detection_metrics(labels, values)
            delta = difference(reference, values, config["atol"], config["rtol"])
            row = {
                **base, "variant": variant, "score_kind": kind,
                "tail_anomaly_count": int(tail_labels.sum()),
                "tail_anomaly_ratio": float(tail_labels.mean()),
                "anomaly_ratio": float(labels.mean()), **metrics,
                "delta_ap": None if metrics["average_precision"] is None else metrics["average_precision"] - baseline["average_precision"],
                "delta_roc_auc": None if metrics["roc_auc"] is None else metrics["roc_auc"] - baseline["roc_auc"],
                **{f"prediction_{key}": value for key, value in delta.items()},
                "changed_points": int((~np.isclose(values, reference, atol=config["atol"], rtol=config["rtol"])).sum()),
            }
            append_row(directory / "condition_metrics.csv", row)
