import numpy as np
import pandas as pd


def patch_correlation(segments, minimum=3):
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 2 or segments.shape[1] != 16 or not np.isfinite(segments).all():
        raise ValueError("Expected finite 16-point P segments")
    if len(segments) < minimum:
        return None, "fewer_than_3_complete_P"
    centered = segments - segments.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms == 0):
        return None, "constant_P_segment"
    unit = centered / norms[:, None]
    pairs = (unit @ unit.T)[np.triu_indices(len(unit), 1)]
    return float(pairs.mean()), "ok"


def rank_correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3:
        raise ValueError("Expected at least three aligned observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite observations")
    a, b = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Undefined rank correlation")
    return float(np.corrcoef(a, b)[0, 1])
