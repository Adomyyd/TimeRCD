import numpy as np

from experiments.common.artifacts import append_row, sha256, write_json
from experiments.stage2_clean_reference.data import generate as base_generate, shared_normalize
from experiments.stage2_patch_types.selections import patch_types


def generate(settings, patch_size):
    if settings["offsets"] != [0, 8]:
        raise ValueError("This protocol requires paired offsets 0 and 8")
    for meta, base in base_generate({**settings, "offsets": [0]}, patch_size):
        time = np.arange(settings["length"])
        noise = base["clean"][:, 0] - np.sin(2 * np.pi * time / settings["period"])
        perturbation = base["corrupt"] - base["clean"]
        for phase in settings["phases"]:
            clean = (np.sin(2 * np.pi * (time + phase) / settings["period"]) + noise)[:, None]
            corrupt = clean + perturbation
            pair = []
            for offset in settings["offsets"]:
                normal, abnormal, labels = [np.roll(x, offset, axis=0) for x in (clean, corrupt, base["labels"])]
                cn, an, mean, std = shared_normalize(normal, abnormal)
                types = patch_types(labels, np.ones(len(labels), dtype=bool), patch_size)
                record = {**meta, "case": f"seed{meta['seed']}_p{phase}_{meta['shape']}_l{meta['event_length']}_o{offset}",
                          "phase": phase, "offset": offset, "M": int(types["M"].sum()), "P": int(types["P"].sum()),
                          "normalization_mean": float(mean[0]), "normalization_std": float(std[0])}
                arrays = {"clean": normal, "corrupt": abnormal, "clean_normalized": cn, "corrupt_normalized": an,
                          "labels": labels, "starts": base["starts"] + offset,
                          **{f"type_{k}": v for k, v in types.items()}}
                pair.append((record, arrays))
            common = pair[0][1]["type_P"] & pair[1][1]["type_P"]
            if not common.any():
                raise ValueError("Empty common P set")
            for record, arrays in pair:
                arrays["common"] = common.copy()
                record["common_patches"] = int(common.sum())
            yield pair


def verify_pair(pair, settings, patch_size):
    (left, a), (right, b) = pair
    for name in ("clean", "corrupt", "labels"):
        if not np.array_equal(np.roll(a[name], 8, axis=0), b[name]):
            raise ValueError(f"Broken paired translation: {name}")
    if not np.array_equal(a["common"], b["common"]) or not np.array_equal(a["common"], a["type_P"] & b["type_P"]):
        raise ValueError("Common set is not the fixed-index intersection")
    for meta, arrays in pair:
        expected = patch_types(arrays["labels"], np.ones(len(arrays["labels"]), dtype=bool), patch_size)
        for key, value in expected.items():
            if not np.array_equal(value, arrays[f"type_{key}"]):
                raise ValueError("Saved Patch type mismatch")
        if arrays["labels"].sum() != settings["anomaly_count"] or (arrays["common"] & ~arrays["type_P"]).any():
            raise ValueError("Invalid anomaly count or common selection")
        energy = np.square(arrays["corrupt"] - arrays["clean"]).sum()
        if not np.isclose(energy, settings["anomaly_count"] * settings["amplitude"] ** 2, atol=1e-10, rtol=0):
            raise ValueError("Perturbation energy changed")
        for start in arrays["starts"]:
            if start < 0 or start + meta["event_length"] > len(arrays["labels"]):
                raise ValueError("Wrapped event")
    scale_error = max(abs(left[k] - right[k]) for k in ("normalization_mean", "normalization_std"))
    normalized_error = float(np.max(np.abs(np.roll(a["corrupt_normalized"], 8, axis=0) - b["corrupt_normalized"])))
    if max(scale_error, normalized_error) > 1e-12:
        raise ValueError("Shared distribution or normalization pairing failed")
    if not np.array_equal(b["common"], b["type_P"]):
        raise ValueError("Expected offset8 common=full")
    return {"normalization_stat_max_abs": scale_error, "normalized_translation_max_abs": normalized_error}


def prepare(directory, config):
    records, gates = [], []
    for pair in generate(config["synthetic"], config["patch_size"]):
        checks = verify_pair(pair, config["synthetic"], config["patch_size"])
        append_row(directory / "preflight_checks.csv", {"pair": pair[0][0]["case"], "close": True, **checks})
        for meta, arrays in pair:
            duration, offset = meta["event_length"], meta["offset"]
            expected_p = 40 if offset == 0 else (20 if duration == 32 else 35)
            expected_m = 0 if offset == 0 else (40 if duration == 32 else 10)
            if meta["P"] != expected_p or meta["M"] != expected_m or meta["common_patches"] != (20 if duration == 32 else 35):
                raise ValueError("Frozen coverage mismatch")
            path = directory / "arrays" / f"input_{meta['case']}.npz"
            np.savez_compressed(path, **arrays)
            if meta["seed"] == config["synthetic"]["seeds"][0] and meta["phase"] in (0, 32):
                gates.append(len(records))
            records.append({**meta, "input_snapshot": str(path.relative_to(directory)), "input_sha256": sha256(path)})
    if len(records) != 384 or len(gates) != 24 or len({r["case"] for r in records}) != 384:
        raise ValueError("Unexpected factorial coverage")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "preflight.json", {"cases": len(records), "gate_indices": gates, "model_called": False})
    return records, gates
