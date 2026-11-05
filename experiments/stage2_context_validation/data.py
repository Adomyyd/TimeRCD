import numpy as np


PATCH_SIZE = 16
LENGTH = 5000
TARGET = (2480, 2512)
RADIUS_VALUES = (0, 1, 2, 5, 10, 20, 40)


def generate_base(seed, length=LENGTH, period=64, noise_std=0.1):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return np.sin(2 * np.pi * t / period) + rng.normal(0, noise_std, length)


def apply_events(base, intervals, amplitude=4.0):
    base = np.asarray(base, dtype=np.float64)
    values = base.copy()
    labels = np.zeros(len(base), dtype=np.int64)
    for start, end in intervals:
        if not 0 <= start < end <= len(base):
            raise ValueError(f"Invalid event interval [{start}, {end})")
        if labels[start:end].any():
            raise ValueError("Event intervals must not overlap")
        values[start:end] += amplitude
        labels[start:end] = 1
    return values, labels


def restore_events(values, base, intervals):
    values = np.asarray(values, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if values.shape != base.shape or values.ndim != 1:
        raise ValueError("values and base must be equally sized vectors")
    restored = values.copy()
    occupied = np.zeros(len(values), dtype=bool)
    for start, end in intervals:
        if not 0 <= start < end <= len(values) or occupied[start:end].any():
            raise ValueError("Invalid or overlapping event intervals")
        occupied[start:end] = True
        restored[start:end] = base[start:end]
    if not np.array_equal(values[~occupied], base[~occupied]):
        raise ValueError("Values differ from base outside the event intervals")
    if not np.array_equal(restored, base):
        raise ValueError("Restored series does not exactly recover the base")
    return restored


def experiment_a_cases(seed, base=None):
    if base is None:
        base = generate_base(seed)
    layouts = {
        "center_isolated": ((2496, 2512),),
        "center_clustered": ((2464, 2480), (2496, 2512), (2528, 2544)),
        "left_isolated": ((0, 16),),
        "left_clustered": ((0, 16), (32, 48), (64, 80)),
        "right_isolated": ((4984, 5000),),
        "right_clustered": ((4920, 4936), (4952, 4968), (4984, 5000)),
    }
    for name, intervals in layouts.items():
        values, labels = apply_events(base, intervals)
        restored = restore_events(values, base, intervals)
        yield {
            "name": f"a_seed{seed}_{name}", "seed": seed, "layout": name,
            "base": np.asarray(base, dtype=np.float64).copy(), "values": values,
            "restored": restored, "labels": labels, "intervals": intervals,
        }


def experiment_b_cases(seed, base=None):
    if base is None:
        base = generate_base(seed)
    intervals = (TARGET,)
    anomaly, labels = apply_events(base, intervals)
    mean = float(anomaly.mean())
    std = float(anomaly.std())
    if not np.isfinite(std) or std == 0:
        raise ValueError("Invalid shared normalization statistics")
    for target_kind, values in (("normal", base), ("anomaly", anomaly)):
        yield {
            "name": f"b_seed{seed}_{target_kind}", "seed": seed,
            "target_kind": target_kind, "values": np.asarray(values).copy(),
            "labels": labels.copy() if target_kind == "anomaly" else np.zeros(len(base), dtype=np.int64),
            "intervals": intervals if target_kind == "anomaly" else (),
            "mean": mean, "std": std,
        }


def normalize(values, mean=None, std=None):
    values = np.asarray(values, dtype=np.float64)
    if mean is None:
        mean = float(values.mean())
    if std is None:
        std = float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Normalization statistics must be finite with positive std")
    return ((values - mean) / std).astype(np.float32), float(mean), float(std)


def patch_distance_to_target(length=LENGTH, target=TARGET, patch_size=PATCH_SIZE):
    point_patches = np.arange(length) // patch_size
    first, last = target[0] // patch_size, (target[1] - 1) // patch_size
    return np.maximum(np.maximum(first - point_patches, point_patches - last), 0)


def context_mask(length=LENGTH, radius=0, target=TARGET, patch_size=PATCH_SIZE):
    if radius != "full" and (type(radius) is not int or radius < 0):
        raise ValueError("radius must be a nonnegative integer or 'full'")
    distances = patch_distance_to_target(length, target, patch_size)
    if radius == "full":
        return np.ones(length, dtype=bool)
    # The target and five surrounding patches on each side are always visible.
    return distances <= 5 + radius


def ring_masks(labels, intervals, patch_size=PATCH_SIZE):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a binary vector")
    event_patches = sorted({point // patch_size for start, end in intervals
                            for point in range(start, end)})
    if not event_patches:
        raise ValueError("At least one event interval is required")
    point_patches = np.arange(len(labels)) // patch_size
    distances = np.min(np.abs(point_patches[:, None] - event_patches), axis=1)
    masks = {"ring0_anomaly": labels == 1}
    normal = labels == 0
    masks["same_event_patch_normal"] = normal & (distances == 0)
    bins = (("ring1", 1, 1), ("ring2", 2, 2), ("ring3", 3, 3),
            ("ring4_5", 4, 5), ("ring6_10", 6, 10),
            ("ring11_20", 11, 20), ("ring21_39", 21, 39),
            ("ring40_plus", 40, None))
    point_patches = np.arange(len(labels)) // patch_size
    left = point_patches < min(event_patches)
    right = point_patches > max(event_patches)
    between = ~(left | right)
    for name, lower, upper in bins:
        within = distances >= lower if upper is None else (distances >= lower) & (distances <= upper)
        masks[name] = normal & within
        masks[f"left_{name}"] = normal & within & left
        masks[f"right_{name}"] = normal & within & right
        masks[f"between_events_{name}"] = normal & within & between
    return masks, distances
