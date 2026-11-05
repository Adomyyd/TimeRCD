import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.stage2_patch_types.selections import patch_types


def select_positions(meta, arrays, patch_size, random_seeds):
    labels = arrays["labels"]
    starts = arrays["starts"]
    length = meta["event_length"]
    if meta["offset"] != 0 or length % patch_size or length < 2 * patch_size:
        raise ValueError("Expected aligned events containing at least two P patches")
    if len(starts) == 0 or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("Invalid event starts")
    expected = np.zeros_like(labels)
    events = []
    for start in starts:
        if start % patch_size or start < 0 or start + length > len(labels) or expected[start:start + length].any():
            raise ValueError("Unaligned, overlapping or out-of-bounds event")
        expected[start:start + length] = 1
        events.append(np.arange(start // patch_size, (start + length) // patch_size))
    if not np.array_equal(labels, expected):
        raise ValueError("Event labels mismatch")
    types = patch_types(labels, np.ones(len(labels), dtype=bool), patch_size)
    for name, mask in types.items():
        if not np.array_equal(mask, arrays[f"type_{name}"]):
            raise ValueError("Invalid saved Patch types")
    full = types["P"]
    if int(full.sum()) != sum(map(len, events)):
        raise ValueError("P set differs from event union")
    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError("Duplicate selection seeds")
    selections, kept = {"full": full.copy()}, {}
    for name in ["keep_first", "keep_last"] + [f"random_{seed}" for seed in random_seeds]:
        selected = full.copy()
        indices = []
        for index, patches in enumerate(events):
            if name == "keep_first":
                keep = patches[0]
            elif name == "keep_last":
                keep = patches[-1]
            else:
                seed = int(name.split("_")[1])
                digest = hashlib.sha256(f"p_positions_v1:{seed}:{meta['case']}:{index}".encode()).digest()
                keep = np.random.default_rng(int.from_bytes(digest, "big")).choice(patches)
            selected[keep] = False
            indices.append(int(keep))
        if (selected & ~full).any() or any(selected[event].sum() != len(event) - 1 for event in events):
            raise ValueError("Invalid matched per-event selection")
        selections[name], kept[name] = selected, indices
    if not np.array_equal(selections["keep_first"], arrays["common"]):
        raise ValueError("keep_first must equal E08 common deletion mask")
    return selections, kept


def verify_parents(config, checkpoint_sha256):
    parent, recovery = [Path(config[key]) for key in ("parent_run", "recovery_run")]
    original = json.loads((parent / "manifest.json").read_text())
    restored = json.loads((recovery / "manifest.json").read_text())
    if original["status"] != "interrupted" or restored["status"] != "completed":
        raise ValueError("Parent execution/recovery status mismatch")
    if Path(restored["recovery_parent"]).resolve() != parent.resolve():
        raise ValueError("Recovery belongs to a different parent")
    if any(item["checkpoint"]["sha256"] != checkpoint_sha256 for item in (original, restored)):
        raise ValueError("Parent checkpoint mismatch")
    fingerprints = {}
    for name, digest in json.loads((recovery / "parent_artifacts.sha256.json").read_text()).items():
        path = parent / name
        if sha256(path) != digest:
            raise ValueError(f"Recovered parent artifact changed: {name}")
        fingerprints[str(path)] = digest
    for root, names in ((parent, ("manifest.json", "data_manifest.json", "checks.csv", "gates/checks.csv", "regression/checks.csv")),
                        (recovery, ("manifest.json", "summary.json", "parent_artifacts.sha256.json", "checks.csv"))):
        for name in names:
            path = root / name
            fingerprints[str(path)] = sha256(path)
            if name.endswith("checks.csv"):
                checks = pd.read_csv(path)
                if checks.empty or not checks.close.eq(True).all():
                    raise ValueError(f"Failed parent checks: {path}")
    return fingerprints


def prepare(directory, config):
    parent = Path(config["parent_run"])
    records = [r for r in json.loads((parent / "data_manifest.json").read_text()) if r["offset"] == 0]
    expected = {(s, p, h, length) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for length in (32, 128)}
    if len(records) != 192 or len({r["case"] for r in records}) != 192 or {
            (r["seed"], r["phase"], r["shape"], r["event_length"]) for r in records} != expected:
        raise ValueError("Unexpected frozen parent factorial design")
    selections_manifest, gates = [], []
    for index, record in enumerate(records):
        source = parent / record["input_snapshot"]
        if sha256(source) != record["input_sha256"]:
            raise ValueError("Parent input hash mismatch")
        with np.load(source) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        selections, kept = select_positions(record, arrays, config["patch_size"], config["random_seeds"])
        count = 20 if record["event_length"] == 32 else 35
        if len(arrays["labels"]) != 5000 or arrays["labels"].sum() != 640 or selections["full"].sum() != 40:
            raise ValueError("Unexpected frozen input coverage")
        if any(mask.sum() != count for name, mask in selections.items() if name != "full"):
            raise ValueError("Unexpected matched selection count")
        target = directory / record["input_snapshot"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        path = directory / "arrays" / f"selection_{record['case']}.npz"
        np.savez_compressed(path, **selections, **{f"kept_{k}": np.array(v) for k, v in kept.items()})
        record.update(selection_snapshot=str(path.relative_to(directory)), selection_sha256=sha256(path))
        selections_manifest.append({"case": record["case"], "selected_patches": count,
                                    "kept_per_event": kept, "events": len(arrays["starts"])})
        if record["seed"] == config["synthetic"]["seeds"][0]:
            gates.append(index)
    if len(gates) != 24:
        raise ValueError("Missing gate cells")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "selections.json", selections_manifest)
    write_json(directory / "preflight.json", {"cases": len(records), "gate_indices": gates, "model_called": False})
    return records, gates
