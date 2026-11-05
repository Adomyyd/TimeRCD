import json
from pathlib import Path

import numpy as np

from experiments.common.artifacts import sha256, write_json
from experiments.common.data import Series


def load_inputs(directory, config):
    real_parent, synthetic_parent = Path(config["real_parent"]), Path(config["synthetic_parent"])
    real = json.loads((real_parent / "data_manifest.json").read_text())[:24]
    synthetic = json.loads((synthetic_parent / "data_manifest.json").read_text())
    records, inputs, fingerprints = [], [], {}
    for parent, sources in ((real_parent, real), (synthetic_parent, synthetic)):
        for filename in ("manifest.json", "data_manifest.json"):
            fingerprints[str(parent / filename)] = sha256(parent / filename)
        for source in sources:
            index = source["index"]
            case = f"s{index:04d}_w5000"
            path = parent / source["input_snapshot"]
            if sha256(path) != source["input_sha256"]:
                raise ValueError(f"Parent input fingerprint mismatch: {path}")
            with np.load(path) as saved:
                series = Series(source["name"], saved["values"].copy(), saved["labels"].copy(), source["source"])
            if parent == synthetic_parent and series.source.get("normalization") != "official_series":
                raise ValueError("Synthetic inputs must use official normalization")
            with np.load(parent / f"arrays/{case}_masks.npz") as saved:
                masks = {k: saved[k].copy() for k in saved.files}
            with np.load(parent / f"arrays/{case}.npz") as saved:
                predictions = {k: saved[k].copy() for k in saved.files}
            for filename in (source["input_snapshot"], f"arrays/{case}.npz", f"arrays/{case}_masks.npz"):
                fingerprints[str(parent / filename)] = sha256(parent / filename)
            relative = f"arrays/input_s{index:04d}.npz"
            np.savez_compressed(directory / relative, values=series.values, labels=series.labels)
            np.savez_compressed(directory / f"arrays/{case}_selections.npz", **masks)
            records.append({**source, "parent": str(parent), "input_snapshot": relative,
                            "input_sha256": sha256(directory / relative)})
            inputs.append((index, series, masks, predictions))
    if len(inputs) != 51 or [x[0] for x in inputs] != list(range(51)):
        raise ValueError("Expected the 51 frozen inputs in original order")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "parents.sha256.json", fingerprints)
    return inputs
