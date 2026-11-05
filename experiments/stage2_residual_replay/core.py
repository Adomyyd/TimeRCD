from itertools import permutations
from math import factorial

import numpy as np


GROUPS = ("F", "L", "I", "N")


def replay(additive, residual, groups, subset):
    additive, residual = [np.asarray(a, dtype=np.float64) for a in (additive, residual)]
    if additive.ndim != 1 or not len(additive) or residual.shape != additive.shape or not all(
            np.isfinite(a).all() for a in (additive, residual)):
        raise ValueError("Expected aligned finite nonempty logit vectors")
    if type(subset) is not int or not 0 <= subset < 16:
        raise ValueError("Subset must be an integer in [0, 15]")
    if set(groups) != set(GROUPS) or any(m.dtype != bool or m.shape != additive.shape for m in groups.values()):
        raise ValueError("Invalid fixed group masks")
    if not np.all(np.sum(list(groups.values()), axis=0) == 1):
        raise ValueError("Fixed groups must partition all points")
    selected = np.zeros(len(additive), dtype=bool)
    for bit, name in enumerate(GROUPS):
        if subset & (1 << bit):
            selected |= groups[name]
    output = additive.copy()
    with np.errstate(over="raise", invalid="raise"):
        output[selected] += residual[selected]
    return output


def shapley(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 16 or values.shape[1] == 0 or np.isinf(values).any():
        raise ValueError("Expected 16 complete subset readouts")
    if np.any(np.isnan(values).any(axis=0) != np.isnan(values).all(axis=0)):
        raise ValueError("Condition-dependent missingness")
    result = np.zeros((4, values.shape[1]))
    margins = []
    for bit in range(4):
        for subset in range(16):
            if subset & (1 << bit):
                continue
            size = subset.bit_count()
            weight = factorial(size) * factorial(3 - size) / factorial(4)
            delta = values[subset | (1 << bit)] - values[subset]
            result[bit] += weight * delta
            margins.append((bit, subset, weight, delta))
    return result, margins


def permutation_shapley(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros((4, values.shape[1]))
    for order in permutations(range(4)):
        subset = 0
        for bit in order:
            following = subset | (1 << bit)
            result[bit] += (values[following] - values[subset]) / 24
            subset = following
    return result
