import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_logit_additivity.data import verify_hashes
from experiments.stage2_rank_localization.data import load_case as load_original_case


def load_parent(config):
    parent = Path(config["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if previous["status"] != "completed" or not summary["parent_artifacts_unchanged"] or not summary["checkpoint_unchanged"]:
        raise ValueError("E13 parent not fully validated")
    for key in ("random_seeds", "patch_size", "window", "recalls", "synthetic", "primary_score_kind", "derived_dtype"):
        if previous["config"][key] != config[key]:
            raise ValueError(f"Parent configuration mismatch: {key}")
    fingerprints = json.loads((parent / "parent_artifacts.sha256.json").read_text())
    fingerprints.update({str(parent / "source" / k): v for k, v in json.loads((parent / "source.sha256.json").read_text()).items()})
    derived = json.loads((parent / "derived_manifest.json").read_text())
    if len(derived) != 192 or len({r["case"] for r in derived}) != 192:
        raise ValueError("Incomplete derived output identities")
    fingerprints.update({str(parent / r["snapshot"]): r["sha256"] for r in derived})
    fingerprints[config["checkpoint"]] = previous["checkpoint"]["sha256"]
    verify_hashes(fingerprints)
    checks = pd.read_csv(parent / "checks.csv")
    if len(checks) != summary["checks_passed"] or not checks.close.eq(True).all():
        raise ValueError("Invalid E13 checks")
    records = json.loads((parent / "data_manifest.json").read_text())
    expected = {(s, p, h, n, 0) for s in config["synthetic"]["seeds"] for p in (0, 16, 32, 48)
                for h in ("positive", "negative", "alternating") for n in (32, 128)}
    if len(records) != 192 or len({r["case"] for r in records}) != 192 or {
            (r["seed"], r["phase"], r["shape"], r["event_length"], r["offset"]) for r in records} != expected:
        raise ValueError("Incomplete parent factor cells")
    if {r["case"] for r in records} != {r["case"] for r in derived}:
        raise ValueError("Derived case mismatch")
    for path in parent.rglob("*"):
        if path.is_file():
            fingerprints[str(path)] = sha256(path)
    return previous, records, {r["case"]: r for r in derived}, fingerprints


def load_case(parent, original, meta, snapshot, config):
    labels, fixed, roles, predictions = load_original_case(original, meta, config)
    path = parent / snapshot["snapshot"]
    if sha256(path) != snapshot["sha256"]:
        raise ValueError("Derived arrays changed")
    with np.load(path) as saved:
        derived = {k: saved[k].copy() for k in saved.files}
    expected = {f"{s}_{k}" for s in ("keep_first", "keep_last") for k in ("add_logits", "add_scores", "residual")}
    expected |= {f"group_{g}" for g in fixed}
    if set(derived) != expected or len(labels) != config["window"]:
        raise ValueError("Unexpected derived output coverage")
    for name, group in fixed.items():
        if not np.array_equal(derived[f"group_{name}"], group):
            raise ValueError("Derived fixed group changed")
    for selector in ("keep_first", "keep_last"):
        for kind in ("add_logits", "residual"):
            values = derived[f"{selector}_{kind}"]
            if values.dtype != np.float64 or values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError("Invalid frozen derived vector")
    return labels, fixed, roles, predictions, derived
