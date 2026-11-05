import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_logit_additivity.data import verify_hashes
from experiments.stage2_residual_replay.data import load_case as load_derived_case, load_parent as load_e13


SELECTORS = ("keep_first", "keep_last")
CONDITIONS = {"add": 0, "normal": 8, "full": 15}


def metric_table(path, expected_rows):
    frame = pd.read_csv(path)
    keys = ["case", "variant", "score_kind"]
    if len(frame) != expected_rows or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Invalid parent metric identities")
    if not frame.metric_status.eq("ok").all():
        raise ValueError("Undefined parent metrics")
    return frame.set_index(keys)


def load_parent(config):
    parent = Path(config["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if (previous["status"] != "completed" or not summary["parent_artifacts_unchanged"]
            or not summary["checkpoint_unchanged"] or summary["inference_calls"] != 0):
        raise ValueError("E14 parent not fully validated")
    for key in ("random_seeds", "window", "patch_size", "recalls", "synthetic", "checkpoint"):
        if previous["config"][key] != config[key]:
            raise ValueError(f"Parent configuration mismatch: {key}")
    checks = pd.read_csv(parent / "checks.csv")
    if len(checks) != summary["checks_passed"] or checks.empty or not checks.close.eq(True).all():
        raise ValueError("Invalid E14 checks")
    e13, records, derived, fingerprints = load_e13(previous["config"])
    if e13["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint chain mismatch")
    fingerprints.update(json.loads((parent / "parent_artifacts.sha256.json").read_text()))
    fingerprints.update({str(parent / "source" / k): v for k, v in
                         json.loads((parent / "source.sha256.json").read_text()).items()})
    snapshots = json.loads((parent / "replay_manifest.json").read_text())
    if len(snapshots) != len(records) or {r["case"] for r in snapshots} != {r["case"] for r in records}:
        raise ValueError("Missing or duplicate replay cases")
    if records != json.loads((parent / "data_manifest.json").read_text()):
        raise ValueError("E14 and E13 inputs differ")
    fingerprints.update({str(parent / r["snapshot"]): r["sha256"] for r in snapshots})
    verify_hashes(fingerprints)
    for path in parent.rglob("*"):
        if path.is_file():
            fingerprints[str(path)] = sha256(path)
    e13_path = Path(previous["config"]["parent_run"])
    metrics = metric_table(parent / "metrics.csv", 6144)
    expected = {(r["case"], f"{s}_{i:02d}", "logits") for r in records for s in SELECTORS for i in range(16)}
    if set(metrics.index) != expected:
        raise ValueError("Incomplete E14 subset coverage")
    originals = metric_table(e13_path / "metrics.csv", 3456)
    return previous, e13, records, derived, {r["case"]: r for r in snapshots}, fingerprints, metrics, originals


def load_case(config, previous, e13, meta, derived, snapshot):
    parent = Path(config["parent_run"])
    source = Path(previous["config"]["parent_run"])
    labels, fixed, roles, predictions, saved = load_derived_case(
        source, Path(e13["config"]["parent_run"]), meta, derived, config)
    path = parent / snapshot["snapshot"]
    if sha256(path) != snapshot["sha256"]:
        raise ValueError("Replay arrays changed")
    with np.load(path) as archive:
        expected = {f"{s}_{i:02d}" for s in SELECTORS for i in range(16)} | {f"group_{g}" for g in fixed}
        if set(archive.files) != expected:
            raise ValueError("Invalid replay array coverage")
        for group, mask in fixed.items():
            if not np.array_equal(archive[f"group_{group}"], mask):
                raise ValueError("Replay group mismatch")
        outputs = {s: {name: archive[f"{s}_{i:02d}"].copy() for name, i in CONDITIONS.items()} for s in SELECTORS}
    for block in outputs.values():
        if any(a.dtype != np.float64 or a.shape != labels.shape or not np.isfinite(a).all() for a in block.values()):
            raise ValueError("Invalid replay output")
    return labels, fixed, roles, predictions, saved, outputs
