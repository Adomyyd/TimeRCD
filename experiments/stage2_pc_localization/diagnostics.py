import numpy as np

from experiments.stage2_followup.diagnostics import concordance
from experiments.stage2_patch_types.selections import patch_types


GROUPS = ("P_anomaly", "M_anomaly", "M_normal", "N_normal")


def point_partition(labels, window, patch_size):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected nonempty binary labels")
    if type(window) is not int or window <= 0:
        raise ValueError("Invalid window")
    width = min(len(labels), window)
    membership = np.arange(len(labels)) // width
    points = {t: np.zeros(len(labels), dtype=bool) for t in "MPN"}
    windows = []
    for start in range(0, len(labels), width):
        size = min(width, len(labels) - start)
        valid = np.arange(width) < size
        local = np.pad(labels[start:start + size], (0, width - size))
        types = patch_types(local, valid, patch_size)
        windows.append((start, size, types))
        for t in points:
            points[t][start:start + size] = np.repeat(types[t], patch_size)[:size]
    groups = {"P_anomaly": points["P"], "M_anomaly": points["M"] & (labels == 1),
              "M_normal": points["M"] & (labels == 0), "N_normal": points["N"]}
    if not np.all(np.stack(list(groups.values())).sum(axis=0) == 1):
        raise ValueError("Point partition is not exclusive and exhaustive")
    return groups, membership, windows


def pair_counts(positive, negative):
    scores = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    return concordance(labels, scores)


def pair_decomposition(labels, scores, groups, membership):
    labels, scores, membership = map(np.asarray, (labels, scores, membership))
    if labels.ndim != 1 or scores.shape != labels.shape or membership.shape != labels.shape:
        raise ValueError("Unaligned arrays")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid scores or labels")
    if set(groups) != set(GROUPS):
        raise ValueError("Unexpected point groups")
    for name, mask in groups.items():
        if mask.shape != labels.shape or mask.dtype != bool:
            raise ValueError("Invalid group mask")
        if not np.all(labels[mask] == int(name.endswith("anomaly"))):
            raise ValueError("Group label mismatch")
    if not np.all(np.stack(list(groups.values())).sum(axis=0) == 1):
        raise ValueError("Incomplete or overlapping groups")
    total_pairs = int((labels == 1).sum()) * int((labels == 0).sum())
    if not total_pairs:
        raise ValueError("Full sequence must have both classes")
    rows = []
    for positive in GROUPS[:2]:
        for negative in GROUPS[2:]:
            p, n = groups[positive], groups[negative]
            correct, pairs = pair_counts(scores[p], scores[n])
            within_correct, within_pairs = 0.0, 0
            for w in np.unique(membership):
                wc, wp = pair_counts(scores[p & (membership == w)], scores[n & (membership == w)])
                within_correct += wc
                within_pairs += wp
            for scope, c, count in (("all", correct, pairs), ("within", within_correct, within_pairs),
                                    ("across", correct - within_correct, pairs - within_pairs)):
                rows.append({"pair": f"{positive}__{negative}", "scope": scope, "pairs": count,
                             "correct": c, "weight": count / total_pairs,
                             "auc": c / count if count else None, "contribution": c / total_pairs})
    return rows
