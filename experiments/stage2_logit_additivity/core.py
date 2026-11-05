import numpy as np


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite logits")
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def additive_output(l00, l10, l01, l11):
    arrays = [np.asarray(a, dtype=np.float64) for a in (l00, l10, l01, l11)]
    if arrays[0].ndim != 1 or not len(arrays[0]) or any(
            a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected aligned finite one-dimensional logits")
    baseline, first, last, both = arrays
    with np.errstate(over="raise", invalid="raise"):
        additive = (first + last) - baseline
        residual = both - additive
    return additive, residual


def residual_readout(residual, first_delta, last_delta, groups):
    arrays = [np.asarray(a, dtype=np.float64) for a in (residual, first_delta, last_delta)]
    if arrays[0].ndim != 1 or not len(arrays[0]) or any(
            a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid residual or increment vectors")
    if set(groups) != {"F", "L", "I", "N"} or any(
            m.dtype != bool or m.shape != arrays[0].shape for m in groups.values()):
        raise ValueError("Invalid fixed groups")
    if not np.all(np.sum(list(groups.values()), axis=0) == 1):
        raise ValueError("Fixed groups must partition input")
    result = {}
    for name, mask in {"all": np.ones(len(arrays[0]), dtype=bool), **groups}.items():
        r, d7, d8 = [a[mask] for a in arrays]
        result[f"{name}_points"] = len(r)
        result.update({f"{name}_{k}": float(v) if len(r) else np.nan for k, v in {
            "mean": r.mean() if len(r) else np.nan,
            "mae": np.abs(r).mean() if len(r) else np.nan,
            "rms": np.sqrt(np.mean(r ** 2)) if len(r) else np.nan,
            "max_abs": np.max(np.abs(r)) if len(r) else np.nan,
            "E7_rms": np.sqrt(np.mean(d7 ** 2)) if len(r) else np.nan,
            "E8_rms": np.sqrt(np.mean(d8 ** 2)) if len(r) else np.nan,
        }.items()})
    return result
