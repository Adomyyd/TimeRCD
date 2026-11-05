import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def detection_metrics(labels, scores):
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Expected aligned finite scores and labels")
    if len(np.unique(labels)) < 2:
        return {"metric_status": "single_class", "roc_auc": None, "average_precision": None}
    return {
        "metric_status": "ok", "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def difference(reference, candidate, atol, rtol):
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite prediction")
    delta = np.abs(reference - candidate)
    return {
        "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
        "close": bool(np.allclose(candidate, reference, atol=atol, rtol=rtol)),
    }
