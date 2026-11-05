import json
from pathlib import Path

import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_rank_localization.data import load_case


def verify_hashes(fingerprints):
    for name, digest in fingerprints.items():
        if sha256(name) != digest:
            raise ValueError(f"Previously recorded artifact changed: {name}")


def load_parent(config):
    parent = Path(config["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if previous["status"] != "completed" or not summary["model_state_unchanged"] or not summary["parent_artifacts_unchanged"]:
        raise ValueError("E12 parent not fully validated")
    if sha256(config["checkpoint"]) != previous["checkpoint"]["sha256"]:
        raise ValueError("Official checkpoint changed")
    for key in ("random_seeds", "patch_size", "window", "recalls", "expected_layers", "switch_layers", "synthetic"):
        if previous["config"][key] != config[key]:
            raise ValueError(f"Parent configuration mismatch: {key}")
    fingerprints = json.loads((parent / "parent_artifacts.sha256.json").read_text())
    fingerprints.update({str(parent / "source" / k): v for k, v in json.loads((parent / "source.sha256.json").read_text()).items()})
    verify_hashes(fingerprints)
    for name, expected in summary["checks"].items():
        checks = pd.read_csv(parent / name)
        if checks.empty or len(checks) != expected["count"] or not checks.close.eq(True).all():
            raise ValueError("Invalid E12 parent checks")
    records = json.loads((parent / "data_manifest.json").read_text())
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(records) != 192 or len({r["case"] for r in records}) != 192 or {
            (r["seed"], r["phase"], r["shape"], r["event_length"], r["offset"]) for r in records} != expected:
        raise ValueError("Incomplete E12 factor cells")
    for path in parent.rglob("*"):
        if path.is_file():
            fingerprints[str(path)] = sha256(path)
    fingerprints[config["checkpoint"]] = previous["checkpoint"]["sha256"]
    return previous, records, fingerprints
