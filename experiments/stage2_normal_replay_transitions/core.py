import numpy as np


def threshold_membership(labels, scores, recall):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold requires aligned binary labels with both classes")
    if not np.isfinite(scores).all() or not np.isfinite(recall) or not 0 < recall <= 1:
        raise ValueError("Invalid scores or recall target")
    positives = np.sort(scores[labels == 1])[::-1]
    count = int(np.ceil(recall * len(positives)))
    threshold = positives[count - 1]
    selected = scores >= threshold
    tp = int(np.sum(selected & (labels == 1)))
    fp = int(np.sum(selected & (labels == 0)))
    strict = int(np.sum((scores > threshold) & (labels == 1)))
    if not strict < count <= tp:
        raise ValueError("Threshold does not bracket target")
    return selected, {"threshold": float(threshold), "tp": tp, "fn": len(positives) - tp,
                      "fp": fp, "tn": int(np.sum(labels == 0)) - fp, "selected": int(selected.sum()),
                      "recall": tp / len(positives), "precision": tp / int(selected.sum()),
                      "fpr": fp / int(np.sum(labels == 0)), "selected_fraction": float(selected.mean()),
                      "strict_recall": strict / len(positives)}


def transition_counts(before, after, group):
    before, after, group = [np.asarray(x) for x in (before, after, group)]
    if before.ndim != 1 or any(x.shape != before.shape or x.dtype != bool for x in (before, after, group)):
        raise ValueError("Transitions require aligned boolean masks")
    entered = int(np.sum(group & ~before & after))
    exited = int(np.sum(group & before & ~after))
    retained = int(np.sum(group & before & after))
    absent = int(np.sum(group & ~before & ~after))
    union = entered + exited + retained
    return {"points": int(group.sum()), "entered": entered, "exited": exited, "retained": retained,
            "absent": absent, "before_selected": retained + exited, "after_selected": retained + entered,
            "net_selected": entered - exited, "changed": entered + exited,
            "jaccard": retained / union if union else np.nan}
