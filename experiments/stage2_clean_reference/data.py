import numpy as np
import torch

from experiments.common.artifacts import sha256, write_json
from experiments.stage2_patch_types.selections import patch_types


def shared_normalize(clean, corrupt):
    clean, corrupt = np.asarray(clean), np.asarray(corrupt)
    if clean.shape != corrupt.shape or clean.ndim != 2 or clean.shape[1] != 1:
        raise ValueError("Expected aligned univariate arrays")
    if not np.isfinite(clean).all() or not np.isfinite(corrupt).all():
        raise ValueError("Nonfinite input")
    mean, std = corrupt.mean(axis=0), corrupt.std(axis=0)
    std = np.where(std == 0, 1e-8, std)
    return (clean - mean) / std, (corrupt - mean) / std, mean, std


def generate(settings, patch_size):
    length, count = settings["length"], settings["anomaly_count"]
    if not 0 < count < length:
        raise ValueError("Invalid anomaly count")
    for seed in settings["seeds"]:
        rng = np.random.default_rng(seed)
        clean = (np.sin(2 * np.pi * np.arange(length) / settings["period"]) +
                 rng.normal(0, settings["noise_std"], length))[:, None]
        for shape in settings["shapes"]:
            for duration in settings["event_lengths"]:
                if count % duration:
                    raise ValueError("Anomaly count must be divisible by duration")
                events = count // duration
                aligned = (np.floor(((np.arange(events) + .5) * length / events) / patch_size) * patch_size).astype(int)
                for offset in settings["offsets"]:
                    starts = aligned + offset
                    if starts[0] < 0 or starts[-1] + duration > length or np.any(np.diff(starts) <= duration):
                        raise ValueError("Overlapping, adjacent or truncated events")
                    labels = np.zeros(length, dtype=np.int64)
                    perturbation = np.zeros(length)
                    for start in starts:
                        labels[start:start + duration] = 1
                        if shape == "positive":
                            pattern = np.ones(duration)
                        elif shape == "negative":
                            pattern = -np.ones(duration)
                        elif shape == "alternating":
                            pattern = np.where(np.arange(duration) % 2 == 0, 1, -1)
                        else:
                            raise ValueError(f"Unknown shape: {shape}")
                        perturbation[start:start + duration] = settings["amplitude"] * pattern
                    corrupt = clean + perturbation[:, None]
                    clean_norm, corrupt_norm, mean, std = shared_normalize(clean, corrupt)
                    types = patch_types(labels, np.ones(length, dtype=bool), patch_size)
                    meta = {"case": f"seed{seed}_{shape}_l{duration}_o{offset}", "seed": seed, "shape": shape,
                            "event_length": duration, "offset": offset, "event_count": events,
                            "anomaly_count": int(labels.sum()), "M": int(types["M"].sum()), "P": int(types["P"].sum()),
                            "normalization_mean": float(mean[0]), "normalization_std": float(std[0]),
                            "perturbation_energy": float(np.square(perturbation).sum())}
                    yield meta, {"clean": clean, "corrupt": corrupt, "clean_normalized": clean_norm,
                                 "corrupt_normalized": corrupt_norm, "labels": labels, "starts": starts,
                                 **{f"type_{k}": v for k, v in types.items()}}


def prepare(directory, config):
    records, gates = [], []
    for index, (meta, arrays) in enumerate(generate(config["synthetic"], config["patch_size"])):
        path = directory / "arrays" / f"input_{meta['case']}.npz"
        np.savez_compressed(path, **arrays)
        records.append({**meta, "input_snapshot": str(path.relative_to(directory)), "input_sha256": sha256(path)})
        if meta["seed"] == config["synthetic"]["seeds"][0] and meta["event_length"] in (5, 32):
            gates.append(index)
    if len(records) != 144 or len(gates) != 12 or len({r["case"] for r in records}) != 144:
        raise ValueError("Unexpected frozen factorial design")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "preflight.json", {"cases": len(records), "gate_indices": gates, "model_called": False})
    return records, gates


def load_input(directory, record):
    path = directory / record["input_snapshot"]
    if sha256(path) != record["input_sha256"]:
        raise ValueError("Input snapshot changed")
    with np.load(path) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    values = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32)
    clean = torch.tensor(arrays["clean_normalized"], dtype=torch.float32)
    valid = torch.ones(len(values), dtype=torch.bool)
    return arrays, values, clean, valid
