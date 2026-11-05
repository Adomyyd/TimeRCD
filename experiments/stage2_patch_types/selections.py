import hashlib

import numpy as np


def patch_types(labels, valid, patch_size):
    labels, valid = np.asarray(labels), np.asarray(valid)
    if labels.ndim != 1 or labels.shape != valid.shape or valid.dtype != bool:
        raise ValueError("Expected aligned labels and boolean point validity")
    if not np.isin(labels, [0, 1]).all() or type(patch_size) is not int or patch_size <= 0:
        raise ValueError("Invalid binary labels or Patch size")
    padding = (-len(valid)) % patch_size
    counts = np.pad(valid, (0, padding)).reshape(-1, patch_size).sum(axis=1)
    bad = np.pad(valid & (labels == 1), (0, padding)).reshape(-1, patch_size).sum(axis=1)
    present = counts > 0
    mixed = (bad > 0) & (bad < counts)
    pure = present & (bad == counts)
    return {"valid": present, "M": mixed, "P": pure, "U": mixed | pure,
            "N": present & (bad == 0), "valid_counts": counts, "anomaly_counts": bad,
            "partial": present & (counts < patch_size)}


def split_random(union, mixed_count, seed, series_name, window_index):
    union = np.asarray(union)
    if union.ndim != 1 or union.dtype != bool or not 0 <= mixed_count <= union.sum():
        raise ValueError("Invalid random union or split size")
    digest = hashlib.sha256(f"partition_v1:{seed}:{series_name}:{window_index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    order = rng.permutation(np.flatnonzero(union))
    mixed = np.zeros_like(union)
    mixed[order[:mixed_count]] = True
    return {"M": mixed, "P": union & ~mixed, "U": union.copy()}


def make_selections(types, saved, number, seeds, name):
    valid = types["valid"]
    if not np.array_equal(valid, saved[f"w{number}_valid"]) or not np.array_equal(valid & ~types["U"], saved[f"w{number}_clean"]):
        raise ValueError("Patch labels or validity changed from the parent")
    if not (valid & ~types["U"]).any():
        raise ValueError("Unexpected empty reference window: stop before inference")
    result = {f"{kind}_label": types[kind].copy() for kind in ("M", "P", "U")}
    for seed in seeds:
        keep = saved[f"w{number}_random_{seed}"]
        if keep.shape != valid.shape or keep.dtype != bool or (keep & ~valid).any():
            raise ValueError("Invalid saved random mask")
        union = valid & ~keep
        if int(union.sum()) != int(types["U"].sum()):
            raise ValueError("Saved random union count mismatch")
        parts = split_random(union, int(types["M"].sum()), seed, name, number)
        for kind, selected in parts.items():
            if selected.sum() != types[kind].sum():
                raise ValueError("Random type count mismatch")
            result[f"{kind}_random_{seed}"] = selected
    return result
