import numpy as np


def concordance(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected aligned binary labels and finite scores")
    negative = np.sort(scores[labels == 0])
    positive = scores[labels == 1]
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    return float((left + (right - left) * .5).sum()), len(negative) * len(positive)


def ranking_decomposition(labels, scores, groups):
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    if groups.shape != labels.shape:
        raise ValueError("Window membership must align with labels")
    total_correct, total_pairs = concordance(labels, scores)
    within_correct, within_pairs = 0.0, 0
    for group in np.unique(groups):
        chosen = groups == group
        correct, pairs = concordance(labels[chosen], scores[chosen])
        within_correct += correct
        within_pairs += pairs
    across_pairs = total_pairs - within_pairs
    return {
        "total_pairs": total_pairs, "within_pairs": within_pairs, "across_pairs": across_pairs,
        "auc": total_correct / total_pairs if total_pairs else None,
        "within_auc": within_correct / within_pairs if within_pairs else None,
        "across_auc": (total_correct - within_correct) / across_pairs if across_pairs else None,
        "within_contribution": within_correct / total_pairs if total_pairs else None,
        "across_contribution": (total_correct - within_correct) / total_pairs if total_pairs else None,
    }


def shift_statistics(reference, candidate, selected):
    reference, candidate, selected = np.asarray(reference), np.asarray(candidate), np.asarray(selected)
    if reference.shape != candidate.shape or selected.shape != reference.shape or selected.dtype != bool:
        raise ValueError("Invalid score selection")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite score")
    if not selected.any():
        return {"points": 0, "baseline_mean": None, "candidate_mean": None,
                "mean_shift": None, "median_shift": None, "fraction_decreased": None}
    delta = candidate[selected].astype(float) - reference[selected].astype(float)
    return {"points": int(selected.sum()), "baseline_mean": float(reference[selected].mean()),
            "candidate_mean": float(candidate[selected].mean()), "mean_shift": float(delta.mean()),
            "median_shift": float(np.median(delta)), "fraction_decreased": float((delta < 0).mean())}
