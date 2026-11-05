import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from .diagnostics import point_groups


def load_parent(config):
    parent = Path(config["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    validation = json.loads((parent / "output_validation/summary.json").read_text())
    if previous["status"] != "completed" or validation["status"] != "passed":
        raise ValueError("Parent has not completed all validations")
    if previous["config"]["random_seeds"] != config["random_seeds"] or previous["config"]["patch_size"] != config["patch_size"]:
        raise ValueError("Parent selector configuration mismatch")
    fingerprints = json.loads((parent / "output_validation/artifacts.sha256.json").read_text())
    for name, digest in fingerprints.items():
        if sha256(name) != digest:
            raise ValueError(f"Previously validated artifact changed: {name}")
    for name in ("manifest.json", "summary.json", "inference_summary.json", "data_manifest.json", "metrics.csv",
                 "checks.csv", "gates/checks.csv", "source.sha256.json", "output_validation/checks.csv",
                 "output_validation/artifacts.sha256.json", "output_validation/summary.json"):
        fingerprints[str(parent / name)] = sha256(parent / name)
    for name, digest in json.loads((parent / "source.sha256.json").read_text()).items():
        fingerprints[str(parent / "source" / name)] = digest
    for name, digest in fingerprints.items():
        if sha256(name) != digest:
            raise ValueError(f"Parent artifact changed: {name}")
    for name in ("checks.csv", "gates/checks.csv", "output_validation/checks.csv"):
        frame = pd.read_csv(parent / name)
        if frame.empty or not frame.close.eq(True).all():
            raise ValueError("Failed parent checks")
    records = json.loads((parent / "data_manifest.json").read_text())
    expected = {(s, p, h, length, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for length in (32, 128)}
    if len(records) != 192 or len({r["case"] for r in records}) != 192 or {
            (r["seed"], r["phase"], r["shape"], r["event_length"], r["offset"]) for r in records} != expected:
        raise ValueError("Missing parent factorial cells")
    return previous, records, fingerprints


def load_case(parent, record, config):
    saved_arrays = []
    for kind in ("input", "selection"):
        path = parent / record[f"{kind}_snapshot"]
        if sha256(path) != record[f"{kind}_sha256"]:
            raise ValueError("Frozen input or selection changed")
        with np.load(path) as saved:
            saved_arrays.append({k: saved[k].copy() for k in saved.files})
    arrays, selections = saved_arrays
    fixed, roles = point_groups(record, arrays, selections, config["patch_size"], config["random_seeds"])
    with np.load(parent / f"arrays/{record['case']}.npz") as saved:
        predictions = {k: saved[k].copy() for k in saved.files}
    return arrays["labels"], fixed, roles, predictions
