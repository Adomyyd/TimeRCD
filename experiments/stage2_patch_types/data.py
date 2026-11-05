import json
from pathlib import Path

import numpy as np

from experiments.common.artifacts import append_row, sha256, write_json
from experiments.common.data import Series
from experiments.stage2.inference import windows
from .selections import make_selections, patch_types


def prepare(directory, config):
    parent = Path(config["parent_run"])
    sources = json.loads((parent / "data_manifest.json").read_text())
    if len(sources) != 51 or [s["index"] for s in sources] != list(range(51)):
        raise ValueError("Expected 51 frozen inputs in original order")
    fingerprints = {name: sha256(parent / name) for name in ("manifest.json", "data_manifest.json", "summary.json")}
    inputs, records = [], []
    candidates = {"first_no_P": None, "first_with_P": None, "first_partial": None, "first_synthetic": None}
    for source in sources:
        index = source["index"]
        case = f"s{index:04d}_w5000"
        for filename in (source["input_snapshot"], f"arrays/{case}_selections.npz", f"arrays/{case}.npz"):
            fingerprints[filename] = sha256(parent / filename)
        if fingerprints[source["input_snapshot"]] != source["input_sha256"]:
            raise ValueError("Parent input hash mismatch")
        with np.load(parent / source["input_snapshot"]) as saved:
            series = Series(source["name"], saved["values"].copy(), saved["labels"].copy(), source["source"])
        if series.source["kind"] == "controlled_synthetic" and series.source["normalization"] != "official_series":
            raise ValueError("Synthetic normalization must remain official_series")
        with np.load(parent / f"arrays/{case}_selections.npz") as saved:
            old_masks = {k: saved[k].copy() for k in saved.files}
        selections, offset, pure_count, mixed_count, partial_count = {}, 0, 0, 0, 0
        coverage = []
        for number, (_, valid) in enumerate(windows(series, 5000)):
            size = int(valid.sum())
            labels = np.pad(series.labels[offset:offset + size], (0, len(valid) - size))
            types = patch_types(labels, valid.numpy(), 16)
            selected = make_selections(types, old_masks, number, config["random_seeds"], series.name)
            selections[number] = (types, selected)
            pure_count += int(types["P"].sum())
            mixed_count += int(types["M"].sum())
            partial_count += int(types["partial"].sum())
            for key, mask in selected.items():
                row = {"case": case, "series": series.name, "kind": series.source["kind"],
                       "window_index": number, "offset": offset, "valid_points": size,
                       "selection": key, "selected_patches": int(mask.sum()),
                       "normal_points_covered": int((types["valid_counts"] - types["anomaly_counts"])[mask].sum()),
                       "anomaly_points_covered": int(types["anomaly_counts"][mask].sum()),
                       "partial_patches": int((types["partial"] & mask).sum()),
                       "valid_patches": int(types["valid"].sum())}
                append_row(directory / "coverage.csv", row)
                coverage.append(row)
            offset += size
        if offset != len(series.values):
            raise ValueError("Incomplete window coverage")
        for reason, matches in (
            ("first_no_P", index < 24 and pure_count == 0),
            ("first_with_P", index < 24 and pure_count > 0),
            ("first_partial", index < 24 and partial_count > 0),
            ("first_synthetic", index >= 24),
        ):
            if matches and candidates[reason] is None:
                candidates[reason] = index
        np.savez_compressed(directory / source["input_snapshot"], values=series.values, labels=series.labels)
        arrays = {}
        for number, (types, selected) in selections.items():
            arrays.update({f"w{number}_type_{k}": v for k, v in types.items()})
            arrays.update({f"w{number}_selection_{k}": v for k, v in selected.items()})
        mask_path = f"arrays/{case}_selections.npz"
        np.savez_compressed(directory / mask_path, **arrays)
        records.append({**source, "parent": str(parent), "input_sha256": sha256(directory / source["input_snapshot"]),
                        "selection_snapshot": mask_path, "selection_sha256": sha256(directory / mask_path),
                        "mixed_patches": mixed_count, "pure_patches": pure_count, "partial_patches": partial_count})
        inputs.append((index, series, selections, parent / f"arrays/{case}.npz"))
    if any(v is None for v in candidates.values()):
        raise ValueError(f"Missing planned gate coverage: {candidates}")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "parent_artifacts.sha256.json", fingerprints)
    write_json(directory / "preflight.json", {"gate_reasons": candidates, "gate_indices": sorted(set(candidates.values())),
               "cases": len(inputs), "model_called": False, "coverage_sha256": sha256(directory / "coverage.csv")})
    return inputs, sorted(set(candidates.values()))
