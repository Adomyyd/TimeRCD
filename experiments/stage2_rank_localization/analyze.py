import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from experiments.common.artifacts import create_run, git, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage2_pc_localization.data import Audit
from .data import load_case, load_parent
from .diagnostics import readout
from .statistics import summarize


def analyze_case(meta, labels, fixed, roles, predictions, config, parent_metrics, audit):
    variants = ["A"] + [f"{selector}_{scope}" for selector in roles if selector != "A" for scope in ("last", "all")]
    if set(predictions) != {f"{variant}_{kind}" for variant in variants for kind in ("scores", "logits")}:
        raise ValueError("Saved output conditions differ from protocol")
    rows = []
    for variant in variants:
        selector = "A" if variant == "A" else variant.rsplit("_", 1)[0]
        groups = {**fixed, **roles[selector]}
        for kind in ("scores", "logits"):
            identity = f"{meta['case']}/{variant}/{kind}"
            result = readout(labels, predictions[f"{variant}_{kind}"], groups, config["recalls"], audit, identity)
            old = parent_metrics.loc[(meta["case"], variant, kind)]
            if isinstance(old, pd.DataFrame) or old.metric_status != "ok":
                raise ValueError("Duplicated or undefined parent metric")
            for name in ("average_precision", "roc_auc"):
                audit.check(f"{identity}/{name}/parent", result[name], old[name])
            base = {key: meta[key] for key in ("case", "seed", "phase", "shape", "event_length", "offset")}
            rows.append({**base, "score_kind": kind, "variant": variant, **result})
    return rows


def provenance(directory, manifest, config_path):
    shutil.copyfile(config_path, directory / "config.input.json")
    write_json(directory / "config.resolved.json", manifest["config"])
    shutil.copyfile(manifest["config"]["protocol_path"], directory / "protocol.md")
    manifest["protocol_sha256"] = sha256(directory / "protocol.md")
    manifest["git"] = {"commit": git("rev-parse", "HEAD").strip(), "status": git("status", "--short")}
    manifest["environment"] = {"python": sys.version, "executable": sys.executable,
                               "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")}}
    files = list((ROOT / "experiments/stage2_rank_localization").glob("*.py"))
    files += [ROOT / f for f in ("experiments/common/artifacts.py", "experiments/common/config.py", "experiments/common/metrics.py",
              "experiments/stage1/analyze.py", "experiments/stage2_clean_reference/statistics.py",
              "experiments/stage2_pc_localization/data.py", "experiments/stage2_pc_localization/diagnostics.py",
              "experiments/stage2_p_positions/data.py", "experiments/stage2_patch_types/selections.py",
              "experiments/tests/test_stage2_rank_localization.py", "EXPERIMENT_LOG.md")]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(target)
    write_json(directory / "source.sha256.json", hashes)


def main():
    parser = argparse.ArgumentParser(description="Offline high-score rank localization of E09")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "protocol_path", "output_root"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        provenance(directory, manifest, args.config)
        previous, records, fingerprints = load_parent(config)
        manifest.update(checkpoint=previous["checkpoint"], inference_calls=0, status="running_readouts")
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        write_json(directory / "data_manifest.json", records)
        audit = Audit(directory, config["tolerance"])
        parent = Path(config["parent_run"])
        parent_metrics = pd.read_csv(parent / "metrics.csv").set_index(["case", "variant", "score_kind"])
        if not parent_metrics.index.is_unique or len(parent_metrics) != 192 * 17 * 2:
            raise ValueError("Invalid parent metric coverage")
        rows, group_records = [], []
        for meta in records:
            labels, fixed, roles, predictions = load_case(parent, meta, config)
            rows.extend(analyze_case(meta, labels, fixed, roles, predictions, config, parent_metrics, audit))
            path = directory / "arrays" / f"groups_{meta['case']}.npz"
            np.savez_compressed(path, **fixed, **{f"{s}_{g}": mask for s, groups in roles.items() for g, mask in groups.items()})
            group_records.append({"case": meta["case"], "snapshot": str(path.relative_to(directory)), "sha256": sha256(path),
                                  "fixed_counts": {g: int(mask.sum()) for g, mask in fixed.items()},
                                  "role_counts": {s: {g: int(mask.sum()) for g, mask in groups.items()} for s, groups in roles.items()}})
        metrics = pd.DataFrame(rows)
        metrics.to_csv(directory / "metrics.csv", index=False)
        write_json(directory / "groups.json", group_records)
        write_json(directory / "readout_summary.json", {"cases": len(records), "rows": len(metrics), "checks_passed": audit.count,
                                                       "inference_calls": 0})
        manifest["status"] = "running_statistics"
        write_json(directory / "manifest.json", manifest)
        report = summarize(directory, metrics, config)
        for name, digest in fingerprints.items():
            audit.check(f"unchanged/{name}", sha256(name) == digest, True, exact=True)
        write_json(directory / "summary.json", {"cases": len(records), "conditions": 17, "rows": len(metrics),
                    "checks_passed": audit.count, "inference_calls": 0, "parent_artifacts_unchanged": True,
                    "model_state_check": "Offline run did not load a model; validated parent artifacts only."})
        manifest["status"] = "completed"
        print(report)
        print(f"Checks passed: {audit.count}; inference calls: 0")
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
