import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_logit_additivity.data import verify_hashes
from experiments.stage2_normal_replay_transitions.data import load_parent as load_e14, load_case as load_replay_case
from experiments.stage2_normal_replay_transitions.readouts import analyze_case as validate_e15
from experiments.stage2_residual_replay.statistics import compare


def compare_table(audit, name, actual, expected, keys):
    a, b = [f.set_index(keys).sort_index() for f in (actual, expected)]
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index) or set(a) != set(b):
        raise ValueError(f"Parent table coverage mismatch: {name}")
    numeric = list(a.select_dtypes(include=np.number).columns)
    categorical = [c for c in a if c not in numeric]
    if not a[categorical].equals(b[categorical]):
        raise ValueError(f"Parent categorical mismatch: {name}")
    compare(audit, name, a[numeric].to_numpy(), b[numeric].to_numpy())


def load_parent(config):
    parent = Path(config["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if (previous["status"] != "completed" or summary["inference_calls"] != 0
            or not all(summary[k] for k in ("checkpoint_unchanged", "parent_artifacts_unchanged", "local_artifacts_unchanged"))):
        raise ValueError("E15 parent not fully validated")
    for key in ("checkpoint", "window", "patch_size", "recalls", "random_seeds", "synthetic", "tolerance"):
        if config[key] != previous["config"][key]:
            raise ValueError(f"Parent configuration mismatch: {key}")
    checks = pd.read_csv(parent / "checks.csv")
    if checks.empty or len(checks) != summary["checks_passed"] or not checks.close.eq(True).all():
        raise ValueError("Invalid E15 checks")
    fingerprints = {str(parent / k): v for k, v in json.loads((parent / "artifacts.sha256.json").read_text()).items()}
    fingerprints.update(json.loads((parent / "parent_artifacts.sha256.json").read_text()))
    fingerprints.update({str(parent / "source" / k): v for k, v in json.loads((parent / "source.sha256.json").read_text()).items()})
    members = json.loads((parent / "membership_manifest.json").read_text())
    fingerprints.update({str(parent / r["snapshot"]): r["sha256"] for r in members})
    verify_hashes(fingerprints)
    e14, e13, records, derived, replayed, older, metrics, originals = load_e14(previous["config"])
    if previous["checkpoint"] != e14["checkpoint"] or records != json.loads((parent / "data_manifest.json").read_text()):
        raise ValueError("E15 chain mismatch")
    if len(members) != len(records) or {r["case"] for r in members} != {r["case"] for r in records}:
        raise ValueError("Missing or duplicated E15 membership case")
    fingerprints.update(older)
    for path in parent.rglob("*"):
        if path.is_file():
            fingerprints[str(path)] = sha256(path)
    thresholds, transitions = [pd.read_csv(parent / f"{name}.csv") for name in ("thresholds", "transitions")]
    for frame, columns, count in ((thresholds, ["condition"], 6912), (transitions, ["condition", "group"], 32256)):
        keys = ["case", "selector", "target_recall"] + columns
        if len(frame) != count or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Invalid E15 table coverage")
    return {"previous": previous, "e14": e14, "e13": e13, "records": records, "derived": derived,
            "replayed": replayed, "fingerprints": fingerprints, "metrics": metrics, "originals": originals,
            "members": {r["case"]: r for r in members}, "thresholds": thresholds, "transitions": transitions}


def load_case(config, context, meta, audit):
    case = meta["case"]
    previous = context["previous"]
    labels, fixed, roles, predictions, derived, outputs = load_replay_case(
        previous["config"], context["e14"], context["e13"], meta,
        context["derived"][case], context["replayed"][case])
    rows, transitions, memberships = validate_e15(meta, labels, fixed, roles, predictions, derived, outputs,
            config, context["metrics"], context["originals"], audit)
    keys = ["case", "selector", "target_recall", "condition"]
    for name, actual, extra in (("thresholds", rows, []), ("transitions", transitions, ["group"])):
        expected = context[name][context[name].case == case]
        compare_table(audit, f"{case}/E15/{name}", pd.DataFrame(actual), expected, keys + extra)
    path = Path(config["parent_run"]) / context["members"][case]["snapshot"]
    if sha256(path) != context["members"][case]["sha256"]:
        raise ValueError("E15 membership changed")
    expected_arrays = {**memberships, "labels": labels, **{f"group_{g}": v for g, v in fixed.items()}}
    expected_arrays.update({f"{s}_role_{g}": v for s in ("keep_first", "keep_last") for g, v in roles[s].items()})
    with np.load(path) as saved:
        if set(saved.files) != set(expected_arrays):
            raise ValueError("Unexpected E15 membership arrays")
        for name, values in expected_arrays.items():
            audit.check(f"{case}/E15/membership/{name}", saved[name], values, exact=True)
    replay_path = Path(previous["config"]["parent_run"]) / context["replayed"][case]["snapshot"]
    with np.load(replay_path) as saved:
        endpoint_arrays = {s: {c: saved[f"{s}_{subset:02d}"].copy() for c, subset in
                             (("00", 0), ("10", 8), ("01", 7), ("11", 15))}
                           for s in ("keep_first", "keep_last")}
    return labels, fixed, roles, predictions, outputs, endpoint_arrays, memberships
