import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd

from experiments.common.artifacts import create_run, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage2_logit_additivity.data import verify_hashes
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.analyze import provenance
from .data import load_case, load_parent
from .readouts import analyze_case
from .statistics import summarize


def main():
    parser = argparse.ArgumentParser(description="Offline E16 normal-score / positive-threshold cross")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "protocol_path", "output_root", "checkpoint"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        if config["recalls"] != [.1, .25, .5, .75, .9, 1.] or config["tolerance"] != 1e-12:
            raise ValueError("Unexpected frozen recalls or tolerance")
        provenance(directory, manifest, args.config)
        hashes = json.loads((directory / "source.sha256.json").read_text())
        for path in (ROOT / "experiments").glob("*/*.py"):
            relative = path.relative_to(ROOT)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = sha256(target)
        write_json(directory / "source.sha256.json", hashes)
        context = load_parent(config)
        manifest.update(checkpoint=context["previous"]["checkpoint"], inference_calls=0, status="running_readouts")
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "parent_artifacts.sha256.json", context["fingerprints"])
        write_json(directory / "data_manifest.json", context["records"])
        audit = Audit(directory, config["tolerance"])
        rows, transitions, interactions, snapshots = [], [], [], []
        for meta in context["records"]:
            inputs = load_case(config, context, meta, audit)
            current, changes, points, arrays = analyze_case(meta, *inputs, config, context["metrics"], audit)
            rows.extend(current)
            transitions.extend(changes)
            interactions.extend(points)
            labels, fixed = inputs[:2]
            arrays.update(labels=labels, **{f"group_{g}": v for g, v in fixed.items()})
            path = directory / "arrays" / f"{meta['case']}.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path) as saved:
                for name, values in arrays.items():
                    audit.check(f"{meta['case']}/saved/{name}", saved[name], values, exact=True)
            snapshots.append({"case": meta["case"], "snapshot": str(path.relative_to(directory)), "sha256": sha256(path)})
        readouts, transitions, interactions = map(pd.DataFrame, (rows, transitions, interactions))
        for frame, name in ((readouts, "readouts"), (transitions, "transitions"), (interactions, "point_interactions")):
            frame.to_csv(directory / f"{name}.csv", index=False)
        write_json(directory / "membership_manifest.json", snapshots)
        write_json(directory / "readout_summary.json", {"cases": len(snapshots), "readout_rows": len(readouts),
                   "transition_rows": len(transitions), "interaction_rows": len(interactions), "checks_passed": audit.count, "inference_calls": 0})
        local = {str(p): sha256(p) for p in directory.rglob("*") if p.is_file() and p.name not in ("manifest.json", "checks.csv")}
        manifest["status"] = "running_statistics"
        write_json(directory / "manifest.json", manifest)
        report = summarize(directory, readouts, transitions, interactions, config, audit)
        verify_hashes(context["fingerprints"])
        verify_hashes(local)
        audit.check("parent_and_local_hashes_unchanged", True, True, exact=True)
        checks = pd.read_csv(directory / "checks.csv")
        if len(checks) != audit.count or not checks.close.eq(True).all():
            raise ValueError("Incomplete or failed audit")
        write_json(directory / "artifacts.sha256.json", {str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*")
                   if p.is_file() and p.name not in ("manifest.json", "summary.json", "artifacts.sha256.json")})
        summary = {"cases": len(snapshots), "selectors": 2, "cells": 4, "recall_targets": 6,
                   "readout_rows": len(readouts), "transition_rows": len(transitions), "interaction_rows": len(interactions),
                   "effect_rows": len(readouts) * 2, "checks_passed": audit.count, "max_abs": float(checks.max_abs.max()),
                   "inference_calls": 0, "parent_artifacts_unchanged": True, "checkpoint_unchanged": True,
                   "local_artifacts_unchanged": True, "model_state_check": "No model loaded; checkpoint hash and E13/E14/E15 chain verified."}
        write_json(directory / "summary.json", summary)
        manifest["status"] = "completed"
        print(report)
        print(json.dumps(summary, ensure_ascii=False))
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
