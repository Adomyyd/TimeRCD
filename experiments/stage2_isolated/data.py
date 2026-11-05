import hashlib

import numpy as np

from experiments.stage2_clean_reference.data import generate, shared_normalize
from experiments.stage2_patch_types.selections import patch_types


def random_selection(types, seed, position, repetition, count=1):
    eligible = np.flatnonzero(types["N"] & ~types["partial"])
    if not 0 < count <= len(eligible):
        raise ValueError("Invalid normal Patch selection count")
    digest = hashlib.sha256(f"stage2_isolated_v1:{seed}:{position}:{repetition}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected = np.zeros_like(types["valid"])
    selected[rng.choice(eligible, count, replace=False)] = True
    return selected


def generate_isolated(config):
    settings = {"length": config["window"], "anomaly_count": 16, "seeds": config["seeds"],
                "shapes": ["positive"], "event_lengths": [16], "offsets": [0],
                "period": config["period"], "noise_std": config["noise_std"], "amplitude": config["amplitude"]}
    for meta, base in generate(settings, config["patch_size"]):
        clean = base["clean"]
        for position in config["positions"]:
            start = config["patch_size"] * position
            if start < 32 or start + 48 > len(clean):
                raise ValueError("Event or adjacent Patch readouts too close to boundary")
            labels = np.zeros(len(clean), dtype=np.int64)
            labels[start:start + 16] = 1
            corrupt = clean + labels[:, None] * config["amplitude"]
            clean_norm, values, mean, std = shared_normalize(clean, corrupt)
            types = patch_types(labels, np.ones(len(clean), dtype=bool), config["patch_size"])
            if types["P"].sum() != 1 or not types["P"][position] or types["M"].any():
                raise ValueError("Isolated event does not occupy exactly one pure Patch")
            selections = {"C_anom": types["P"], "D_anom": types["P"]}
            for repetition in range(1, config["random_repetitions"] + 1):
                selections[f"D_rand_{repetition}"] = random_selection(types, meta["seed"], position, repetition)
            record = {"case": f"seed{meta['seed']}_p{position}", "seed": meta["seed"], "position": position,
                      "normalization_mean": float(mean[0]), "normalization_std": float(std[0])}
            arrays = {"clean": clean, "corrupt": corrupt, "clean_normalized": clean_norm,
                      "corrupt_normalized": values, "labels": labels,
                      **{f"type_{k}": v for k, v in types.items()},
                      **{f"selection_{k}": v for k, v in selections.items()}}
            yield record, arrays
