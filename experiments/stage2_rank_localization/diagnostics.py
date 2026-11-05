import numpy as np

from experiments.common.metrics import detection_metrics
from experiments.stage2_p_positions.data import select_positions


def point_groups(meta, arrays, selections, patch_size, random_seeds):
    expected, _ = select_positions(meta, arrays, patch_size, random_seeds)
    for name, mask in expected.items():
        if name not in selections or not np.array_equal(mask, selections[name]):
            raise ValueError("Saved mask differs from frozen selector")
    labels = np.asarray(arrays["labels"])
    first, last = np.zeros(len(labels), dtype=bool), np.zeros(len(labels), dtype=bool)
    for start in arrays["starts"]:
        first[start:start + patch_size] = True
        end = start + meta["event_length"]
        last[end - patch_size:end] = True
    fixed = {"F": first, "L": last, "I": labels.astype(bool) & ~first & ~last, "N": labels == 0}
    if not np.all(np.sum(list(fixed.values()), axis=0) == 1):
        raise ValueError("Fixed point groups do not partition input")
    roles = {}
    for selector, mask in {"A": np.zeros_like(selections["full"]), **expected}.items():
        blocked = np.repeat(mask, patch_size)[:len(labels)]
        kept = labels.astype(bool) & ~blocked
        if (blocked & ~labels.astype(bool)).any() or not np.all(kept.astype(int) + blocked + fixed["N"] == 1):
            raise ValueError("Role groups do not partition input")
        roles[selector] = {"K": kept, "B": blocked}
    return fixed, roles


def relative_ranks(scores):
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Ranks require nonempty finite one-dimensional scores")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_descending = len(scores) - np.cumsum(counts) + (counts + 1) / 2
    return average_descending[inverse] / len(scores)


def readout(labels, scores, groups, recalls, audit=None, identity=""):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid labels or score shape")
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Readout requires both classes and finite scores")
    if set(groups) != {"F", "L", "I", "N", "K", "B"}:
        raise ValueError("Missing point groups")
    for group in groups.values():
        if group.dtype != bool or group.shape != labels.shape:
            raise ValueError("Invalid group mask")
    for names in (("F", "L", "I", "N"), ("K", "B", "N")):
        if not np.all(np.sum([groups[name] for name in names], axis=0) == 1):
            raise ValueError("Nonpartitioning groups")
    if not np.array_equal(groups["N"], labels == 0):
        raise ValueError("Normal group differs from labels")
    if len(set(recalls)) != len(recalls) or any(r <= 0 or r > 1 for r in recalls):
        raise ValueError("Invalid recall targets")
    result = detection_metrics(labels, scores)
    ranks = relative_ranks(scores)
    for name, mask in groups.items():
        count = int(mask.sum())
        result.update({f"{name}_points": count, f"{name}_score_mean": float(scores[mask].mean()) if count else np.nan,
                       f"{name}_rank_mean": float(ranks[mask].mean()) if count else np.nan})
    positives = np.sort(scores[labels == 1])[::-1]
    normal_count = int((labels == 0).sum())
    for recall in recalls:
        target_count = int(np.ceil(recall * len(positives)))
        threshold = positives[target_count - 1]
        selected = scores >= threshold
        actual = int(np.sum(selected & (labels == 1)))
        strict = int(np.sum((scores > threshold) & (labels == 1)))
        if actual < target_count or strict >= target_count:
            raise ValueError("Threshold does not bracket target recall")
        prefix = f"r{int(round(100 * recall)):03d}"
        result.update({f"{prefix}_precision": actual / selected.sum(), f"{prefix}_recall": actual / len(positives),
                       f"{prefix}_fpr": np.sum(selected & (labels == 0)) / normal_count,
                       f"{prefix}_selected_fraction": float(selected.mean()), f"{prefix}_threshold": float(threshold),
                       f"{prefix}_strict_recall": strict / len(positives)})
        for name, mask in groups.items():
            count = int(np.sum(selected & mask))
            result[f"{prefix}_{name}_share"] = count / selected.sum()
            result[f"{prefix}_{name}_capture"] = count / mask.sum() if mask.any() else np.nan
        if audit is not None:
            audit.check(f"{identity}/{prefix}/recall_bracket", actual >= target_count and strict < target_count, True, exact=True)
            for name, parts in (("fixed", ("F", "L", "I", "N")), ("role", ("K", "B", "N"))):
                audit.check(f"{identity}/{prefix}/{name}_share", sum(result[f"{prefix}_{p}_share"] for p in parts), 1)
    return result
