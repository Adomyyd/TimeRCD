import numpy as np

from experiments.stage2_normal_replay_transitions.core import threshold_membership


CELLS = {"00": (0, 0, 0), "10": (1, 0, 8), "01": (0, 1, 7), "11": (1, 1, 15)}
EDGES = {"score_at_add": ("00", "10"), "threshold_at_add": ("00", "01"),
         "score_at_full": ("01", "11"), "threshold_at_full": ("10", "11"), "total": ("00", "11")}
WEIGHTS = {
    "score_at_add": [-1, 1, 0, 0], "threshold_at_add": [-1, 0, 1, 0],
    "score_at_full": [0, 0, -1, 1], "threshold_at_full": [0, -1, 0, 1],
    "total": [-1, 0, 0, 1], "interaction": [1, -1, -1, 1],
    "score_average": [-.5, .5, -.5, .5], "threshold_average": [-.5, -.5, .5, .5],
}


def cross_outputs(labels, additive, full):
    labels, additive, full = [np.asarray(a) for a in (labels, additive, full)]
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Cross requires both binary classes")
    if any(a.shape != labels.shape or not np.isfinite(a).all() for a in (additive, full)):
        raise ValueError("Cross requires aligned finite scores")
    sources = [additive.astype(np.float64), full.astype(np.float64)]
    return {cell: np.where(labels == 0, sources[s], sources[t]) for cell, (s, t, _) in CELLS.items()}


def cross_memberships(labels, outputs, recall):
    if set(outputs) != set(CELLS):
        raise ValueError("Incomplete cross")
    memberships, diagnostics = {}, {}
    for cell in CELLS:
        memberships[cell], diagnostics[cell] = threshold_membership(labels, outputs[cell], recall)
    return memberships, diagnostics


def point_interaction(memberships, normal):
    normal = np.asarray(normal)
    if set(memberships) != set(CELLS) or normal.ndim != 1 or normal.dtype != bool:
        raise ValueError("Invalid interaction masks")
    if any(np.asarray(m).dtype != bool or np.shape(m) != normal.shape for m in memberships.values()):
        raise ValueError("Invalid interaction membership")
    m = {c: a.astype(np.int8) for c, a in memberships.items()}
    z = (m["11"] - m["10"] - m["01"] + m["00"]) * normal
    if np.any(np.abs(z) > 1):
        raise ValueError("Cross violates common-threshold nesting")
    counts = {"positive": int(np.sum(z[normal] == 1)), "negative": int(np.sum(z[normal] == -1)),
              "zero": int(np.sum(z[normal] == 0)), "net": int(z.sum()), "points": int(normal.sum())}
    return z, counts


def factorial_effects(values):
    if set(values) != set(CELLS) or not np.isfinite(list(values.values())).all():
        raise ValueError("Invalid factorial values")
    array = np.array([values[c] for c in CELLS], dtype=float)
    return {name: float(np.dot(w, array)) for name, w in WEIGHTS.items()}
