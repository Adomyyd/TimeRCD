import numpy as np

from experiments.common.data import Series


def generate(settings):
    length = settings["length"]
    event_length = settings["event_length"]
    for seed in settings["seeds"]:
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        normal = np.sin(2 * np.pi * t / settings["period"])
        normal += rng.normal(0, settings["noise_std"], length)
        mean, std = float(normal.mean()), float(normal.std())
        starts = np.arange(0, length - event_length + 1, 2 * event_length)
        for placement in settings["placements"]:
            if placement == "early":
                ordered = starts
            elif placement == "late":
                ordered = starts[::-1]
            elif placement == "spread":
                ordered = np.random.default_rng(seed + 100000).permutation(starts)
            else:
                raise ValueError(f"Unknown placement: {placement}")
            for count in settings["anomaly_counts"]:
                if count % event_length or count // event_length > len(starts):
                    raise ValueError("Anomaly count must fit complete separated events")
                labels = np.zeros(length, dtype=int)
                for start in ordered[:count // event_length]:
                    labels[start:start + event_length] = 1
                values = normal + labels * settings["shift"]
                source = {"kind": "controlled_synthetic", "seed": seed, "placement": placement,
                          "anomaly_count": count, "settings": settings,
                          "normalization": "fixed_clean_generator_statistics",
                          "normal_mean": mean, "normal_std": std}
                yield Series(f"syn_seed{seed}_{placement}_a{count}", values[:, None], labels, source)
