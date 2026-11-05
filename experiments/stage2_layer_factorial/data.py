import json
from pathlib import Path
import shutil

import pandas as pd

from experiments.common.artifacts import sha256, write_json


def prepare(directory, config, checkpoint_sha256):
    parent = Path(config["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if previous["status"] != "completed" or not summary["model_state_unchanged"] or not summary["parent_artifacts_unchanged"]:
        raise ValueError("E11 parent not fully validated")
    if previous["checkpoint"]["sha256"] != checkpoint_sha256:
        raise ValueError("Frozen checkpoint mismatch")
    for key in ("random_seeds", "patch_size", "window", "recalls", "expected_layers", "synthetic"):
        if previous["config"][key] != config[key]:
            raise ValueError(f"Parent protocol mismatch: {key}")
    fingerprints = json.loads((parent / "parent_artifacts.sha256.json").read_text())
    for name, digest in json.loads((parent / "source.sha256.json").read_text()).items():
        fingerprints[str(parent / "source" / name)] = digest
    for name, digest in fingerprints.items():
        if sha256(name) != digest:
            raise ValueError(f"Parent chain or source changed: {name}")
    for name, expected in summary["checks"].items():
        checks = pd.read_csv(parent / name)
        if len(checks) != expected["count"] or not checks.close.eq(True).all():
            raise ValueError("Failed E11 checks")
    records = json.loads((parent / "data_manifest.json").read_text())
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(records) != 192 or len({r["case"] for r in records}) != 192 or {
            (r["seed"], r["phase"], r["shape"], r["event_length"], r["offset"]) for r in records} != expected:
        raise ValueError("Missing parent factorial cells")
    for path in parent.rglob("*"):
        if path.is_file():
            fingerprints[str(path)] = sha256(path)
    for record in records:
        for kind in ("input", "selection"):
            source = parent / record[f"{kind}_snapshot"]
            if sha256(source) != record[f"{kind}_sha256"]:
                raise ValueError("Frozen input or selection changed")
            target = directory / record[f"{kind}_snapshot"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    gates = [i for i, r in enumerate(records) if r["seed"] == config["synthetic"]["seeds"][0]]
    if len(gates) != 24:
        raise ValueError("Missing gate cells")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "parent_artifacts.sha256.json", fingerprints)
    return records, gates, fingerprints
