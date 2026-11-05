import json
from pathlib import Path
import shutil

import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.stage2_rank_localization.data import load_parent as load_position_parent


def prepare(directory, config, checkpoint_sha256):
    previous, records, fingerprints = load_position_parent(config)
    analysis = Path(config["analysis_parent"])
    analyzed = json.loads((analysis / "manifest.json").read_text())
    if analyzed["status"] != "completed" or Path(analyzed["config"]["parent_run"]).resolve() != Path(config["parent_run"]).resolve():
        raise ValueError("Invalid E10 parent identity or status")
    if analyzed["config"]["recalls"] != config["recalls"]:
        raise ValueError("Recall protocol differs from parent")
    if any(item["checkpoint"]["sha256"] != checkpoint_sha256 for item in (previous, analyzed)):
        raise ValueError("Frozen checkpoint mismatch")
    for name, digest in json.loads((analysis / "parent_artifacts.sha256.json").read_text()).items():
        if sha256(name) != digest:
            raise ValueError("E10 parent artifact changed")
        fingerprints[name] = digest
    for name in ("manifest.json", "summary.json", "metrics.csv", "checks.csv", "source.sha256.json", "parent_artifacts.sha256.json"):
        fingerprints[str(analysis / name)] = sha256(analysis / name)
    for name, digest in json.loads((analysis / "source.sha256.json").read_text()).items():
        path = analysis / "source" / name
        if sha256(path) != digest:
            raise ValueError("E10 source snapshot changed")
        fingerprints[str(path)] = digest
    checks = pd.read_csv(analysis / "checks.csv")
    if checks.empty or not checks.close.eq(True).all():
        raise ValueError("Failed E10 checks")
    parent = Path(config["parent_run"])
    for record in records:
        for kind in ("input", "selection"):
            source = parent / record[f"{kind}_snapshot"]
            if sha256(source) != record[f"{kind}_sha256"]:
                raise ValueError("Input or selection mismatch")
            target = directory / record[f"{kind}_snapshot"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    gates = [i for i, r in enumerate(records) if r["seed"] == config["synthetic"]["seeds"][0]]
    if len(gates) != 24:
        raise ValueError("Missing gate cells")
    write_json(directory / "data_manifest.json", records)
    write_json(directory / "parent_artifacts.sha256.json", fingerprints)
    return records, gates, fingerprints
