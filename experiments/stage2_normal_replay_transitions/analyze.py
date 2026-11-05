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
    parser = argparse.ArgumentParser(description="Offline E15 threshold membership transitions")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "protocol_path", "output_root", "checkpoint"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        if config["recalls"] != [.1, .25, .5, .75, .9, 1.] or config["tolerance"] != 1e-12:
            raise ValueError("Unexpected frozen recall targets or tolerance")
        provenance(directory, manifest, args.config)
        hashes = json.loads((directory / "source.sha256.json").read_text())
        sources = list((ROOT / "experiments/stage2_normal_replay_transitions").glob("*.py"))
        sources += list((ROOT / "experiments/stage2_residual_replay").glob("*.py"))
        sources += list((ROOT / "experiments/stage2_logit_additivity").glob("*.py"))
        sources += [ROOT / "experiments/tests/test_stage2_normal_replay_transitions.py"]
        for path in sources:
            relative = path.relative_to(ROOT)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = sha256(target)
        write_json(directory / "source.sha256.json", hashes)
        previous, e13, records, derived, replayed, fingerprints, metrics, originals = load_parent(config)
        manifest.update(checkpoint=previous["checkpoint"], inference_calls=0, status="running_readouts")
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        write_json(directory / "data_manifest.json", records)
        audit = Audit(directory, config["tolerance"])
        rows, transitions, snapshots = [], [], []
        for meta in records:
            case = meta["case"]
            labels, fixed, roles, predictions, saved, outputs = load_case(config, previous, e13, meta, derived[case], replayed[case])
            readouts, changes, memberships = analyze_case(meta, labels, fixed, roles, predictions, saved, outputs,
                                                        config, metrics, originals, audit)
            rows.extend(readouts)
            transitions.extend(changes)
            arrays = {**memberships, "labels": labels, **{f"group_{g}": a for g, a in fixed.items()}}
            arrays.update({f"{s}_role_{g}": a for s in ("keep_first", "keep_last") for g, a in roles[s].items()})
            path = directory / "arrays" / f"{case}.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path) as archive:
                for key, values in arrays.items():
                    audit.check(f"{case}/saved/{key}", archive[key], values, exact=True)
            snapshots.append({"case": case, "snapshot": str(path.relative_to(directory)), "sha256": sha256(path)})
        thresholds, transitions = pd.DataFrame(rows), pd.DataFrame(transitions)
        thresholds.to_csv(directory / "thresholds.csv", index=False)
        transitions.to_csv(directory / "transitions.csv", index=False)
        write_json(directory / "membership_manifest.json", snapshots)
        local = {str(directory / name): sha256(directory / name) for name in
                 ("thresholds.csv", "transitions.csv", "membership_manifest.json", "data_manifest.json", "source.sha256.json", "protocol.md")}
        local.update({str(directory / r["snapshot"]): r["sha256"] for r in snapshots})
        local.update({str(directory / "source" / k): v for k, v in hashes.items()})
        write_json(directory / "readout_summary.json", {"cases": len(records), "threshold_rows": len(thresholds),
                   "transition_rows": len(transitions), "checks_passed": audit.count, "inference_calls": 0})
        manifest["status"] = "running_statistics"
        write_json(directory / "manifest.json", manifest)
        report = summarize(directory, thresholds, transitions, config, audit)
        verify_hashes(fingerprints)
        verify_hashes(local)
        audit.check("parent_and_local_hashes_unchanged", True, True, exact=True)
        write_json(directory / "artifacts.sha256.json", {str(p.relative_to(directory)): sha256(p) for p in directory.iterdir()
                   if p.is_file() and p.name not in {"manifest.json", "summary.json", "artifacts.sha256.json"}})
        checks = pd.read_csv(directory / "checks.csv")
        if len(checks) != audit.count or not checks.close.eq(True).all():
            raise ValueError("Incomplete or failed audit")
        summary = {"cases": len(records), "selectors": 2, "conditions": 3, "recall_targets": 6,
                   "threshold_rows": len(thresholds), "transition_rows": len(transitions),
                   "checks_passed": audit.count, "max_abs": float(checks.max_abs.max()), "inference_calls": 0,
                   "parent_artifacts_unchanged": True, "checkpoint_unchanged": True,
                   "local_artifacts_unchanged": True, "model_state_check": "No model loaded; checkpoint hash and E13/E14 chain verified."}
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
