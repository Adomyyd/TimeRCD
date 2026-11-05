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
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.analyze import provenance as parent_provenance
from .core import GROUPS
from .data import load_case, load_parent
from .readouts import analyze_case
from .statistics import summarize


def main():
    parser = argparse.ArgumentParser(description="Offline fixed-group residual replay of E13")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "protocol_path", "output_root", "checkpoint"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        if config["groups"] != list(GROUPS) or config["primary_score_kind"] != "logits" or config["derived_dtype"] != "float64":
            raise ValueError("Unexpected group order, scale or precision")
        parent_provenance(directory, manifest, args.config)
        hashes = json.loads((directory / "source.sha256.json").read_text())
        sources = list((ROOT / "experiments/stage2_residual_replay").glob("*.py"))
        sources += [ROOT / p for p in ("experiments/stage2_logit_additivity/data.py",
                                      "experiments/tests/test_stage2_residual_replay.py")]
        for path in sources:
            relative = path.relative_to(ROOT)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = sha256(target)
        write_json(directory / "source.sha256.json", hashes)
        previous, records, derived, fingerprints = load_parent(config)
        manifest.update(checkpoint=previous["checkpoint"], inference_calls=0, status="running_readouts")
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        write_json(directory / "data_manifest.json", records)
        audit = Audit(directory, config["tolerance"])
        parent = Path(config["parent_run"])
        original = Path(previous["config"]["parent_run"])
        parent_metrics = pd.read_csv(parent / "metrics.csv").set_index(["case", "variant", "score_kind"])
        if not parent_metrics.index.is_unique or len(parent_metrics) != 3456:
            raise ValueError("Invalid E13 metric coverage")
        rows, snapshots = [], []
        for meta in records:
            labels, fixed, roles, predictions, saved = load_case(parent, original, meta, derived[meta["case"]], config)
            readouts, arrays = analyze_case(meta, labels, fixed, roles, predictions, saved, config, parent_metrics, audit)
            rows.extend(readouts)
            arrays.update({f"group_{g}": m for g, m in fixed.items()})
            path = directory / "arrays" / f"{meta['case']}.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path) as loaded:
                for key, values in arrays.items():
                    audit.check(f"{meta['case']}/saved/{key}", loaded[key], values, exact=True)
            snapshots.append({"case": meta["case"], "snapshot": str(path.relative_to(directory)), "sha256": sha256(path)})
        metrics = pd.DataFrame(rows)
        metrics.to_csv(directory / "metrics.csv", index=False)
        write_json(directory / "replay_manifest.json", snapshots)
        write_json(directory / "readout_summary.json", {"cases": len(records), "rows": len(metrics),
                   "checks_passed": audit.count, "inference_calls": 0})
        manifest["status"] = "running_statistics"
        write_json(directory / "manifest.json", manifest)
        report = summarize(directory, metrics, config, audit)
        for name, digest in fingerprints.items():
            audit.check(f"unchanged/{name}", sha256(name) == digest, True, exact=True)
        for record in snapshots:
            audit.check(f"replay_unchanged/{record['case']}", sha256(directory / record["snapshot"]) == record["sha256"], True, exact=True)
        checks = pd.read_csv(directory / "checks.csv")
        if len(checks) != audit.count or not checks.close.eq(True).all():
            raise ValueError("Incomplete or failed audit")
        summary = {"cases": len(records), "selectors": 2, "subsets": 16, "rows": len(metrics), "marginal_rows": 12288,
                   "checks_passed": audit.count, "max_abs": float(checks.max_abs.max()), "inference_calls": 0,
                   "parent_artifacts_unchanged": True, "checkpoint_unchanged": True,
                   "model_state_check": "Offline run did not load a model; parent artifacts and weight file verified."}
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
