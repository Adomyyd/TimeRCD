import numpy as np


def bootstrap_mean(values, seed=2024, repetitions=10000, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires a finite nonempty one-dimensional sample")
    if type(repetitions) is not int or repetitions <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap repetition count or confidence")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(values.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def summarize_a_rows(rows, repetitions=10000, seed=2024):
    groups = {}
    for row in rows:
        if int(row["point_count"]) == 0:
            continue
        for location, pattern in ((row["location"], row["event_pattern"]), ("all", "all")):
            key = (location, pattern, row["ring"])
            groups.setdefault(key, []).append(row)
    output = []
    for index, ((location, pattern, ring), samples) in enumerate(sorted(groups.items())):
        by_seed = {}
        for row in samples:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        summary = {
            "location": location, "event_pattern": pattern, "ring": ring,
            "sequence_count": len(samples), "independent_base_count": len(by_seed),
        }
        metrics = (
            ("delta_mean", "ci_low", "ci_high"),
            ("mean_absolute_delta", "absolute_ci_low", "absolute_ci_high"),
            ("original_logit_mean", "original_logit_ci_low", "original_logit_ci_high"),
            ("restored_logit_mean", "restored_logit_ci_low", "restored_logit_ci_high"),
            ("original_probability_mean", "original_probability_ci_low", "original_probability_ci_high"),
            ("restored_probability_mean", "restored_probability_ci_low", "restored_probability_ci_high"),
        )
        for metric, low_key, high_key in metrics:
            # Derived layouts share a base; bootstrap independent sequences, not layout replicas.
            values = [float(np.mean([float(row[metric]) for row in by_seed[s]]))
                      for s in sorted(by_seed)]
            mean, low, high = bootstrap_mean(values, seed + index, repetitions)
            summary.update({metric: mean, low_key: low, high_key: high})
        output.append(summary)
    return output


def impact_radii_by_sequence(rows):
    bins = (("ring1", 1), ("ring2", 2), ("ring3", 3), ("ring4_5", 4), ("ring6_10", 6),
            ("ring11_20", 11), ("ring21_39", 21), ("ring40_plus", 40))
    sequences = {}
    for row in rows:
        if row["ring"].startswith(("left_", "right_", "between_events_")):
            continue
        if int(row["point_count"]) == 0:
            continue
        sequences.setdefault(row["name"], {})[row["ring"]] = row
    output = []
    for name, sequence in sorted(sequences.items()):
        peak = max(abs(float(row["delta_mean"])) for row in sequence.values())
        radius = next(
            (lower for ring, lower in bins
             if ring in sequence and abs(float(sequence[ring]["delta_mean"])) < 0.05 * peak), None
        )
        first = sequence["ring0_anomaly"]
        output.append({
            "name": name, "seed": int(first["seed"]), "layout": first["layout"],
            "impact_radius_lower_bin_boundary": radius,
        })
    return output


def saturation_radius(radius_values, scores):
    radii = list(radius_values)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(radii),) or len(radii) < 2:
        raise ValueError("scores must match a radius list with at least two entries")
    if not np.isfinite(scores).all() or radii[0] != 0 or radii[-1] != "full":
        raise ValueError("Expected finite scores, radius 0 first, and full last")
    full_index = radii.index("full")
    total = abs(scores[full_index] - scores[0])
    if total == 0:
        return None
    for index, radius in enumerate(radii):
        if radius == "full":
            continue
        if abs(scores[full_index] - scores[index]) < 0.05 * total:
            return int(radius)
    return None


def summarize_b_rows(rows, radii, repetitions=10000, seed=2024):
    groups = {}
    for row in rows:
        radius = row["radius"]
        if radius != "full":
            radius = int(radius)
        group = groups.setdefault((row["target_kind"], radius), {})
        identity = int(row["seed"])
        if identity in group:
            raise ValueError("Duplicate B seed/target/radius record")
        group[identity] = float(row["target_logit_mean"])
    expected = {(kind, radius) for kind in ("normal", "anomaly") for radius in radii}
    if set(groups) != expected:
        raise ValueError("B target/radius grid is incomplete or unexpected")
    expected_seeds = set(groups[("normal", 0)])
    if any(set(values) != expected_seeds for values in groups.values()):
        raise ValueError("B conditions do not contain identical paired seeds")
    output = []
    for target_kind in ("normal", "anomaly"):
        baseline = groups[(target_kind, 0)]
        for radius_index, radius in enumerate(radii):
            values = groups[(target_kind, radius)]
            ordered_seeds = sorted(values)
            means, low, high = bootstrap_mean(
                [values[s] for s in ordered_seeds], seed + radius_index, repetitions
            )
            changes = [values[s] - baseline[s] for s in ordered_seeds]
            change, change_low, change_high = bootstrap_mean(
                changes, seed + 100 + radius_index, repetitions
            )
            full = groups[(target_kind, "full")]
            gap, gap_low, gap_high = bootstrap_mean(
                [full[s] - values[s] for s in ordered_seeds], seed + radius_index, repetitions
            )
            means_by_radius = [
                float(np.mean([groups[(target_kind, r)][s] for s in sorted(groups[(target_kind, r)])]))
                for r in radii
            ]
            output.append({
                "target_kind": target_kind, "radius": radius,
                "seed_count": len(ordered_seeds), "logit_mean": means,
                "ci_low": low, "ci_high": high,
                "change_from_r0": change,
                "change_ci_low": change_low, "change_ci_high": change_high,
                "full_minus_r": gap, "full_minus_r_ci_low": gap_low,
                "full_minus_r_ci_high": gap_high,
                "saturation_radius": saturation_radius(radii, means_by_radius),
            })
    return output
