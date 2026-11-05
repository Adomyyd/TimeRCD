import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd

from experiments.common.artifacts import create_run, sha256, write_json
from experiments.common.metrics import detection_metrics
from experiments.stage2_pc_localization.data import Audit
from .statistics import summarize


def main():
    parser = argparse.ArgumentParser(description="Recover offline E08 summaries in a new directory")
    parser.add_argument("interrupted_run", type=Path)
    args = parser.parse_args()
    parent = args.interrupted_run.resolve()
    old = json.loads((parent / "manifest.json").read_text())
    if old["status"] != "interrupted":
        raise ValueError("Expected an explicitly recorded interrupted run")
    config = {**old["config"], "name": old["config"]["name"] + "_offline_recovery"}
    directory, manifest = create_run(config, parent / "config.input.json")
    print(f"Recovery directory: {directory}", flush=True)
    try:
        manifest.update(recovery_parent=str(parent), inference_calls=0, checkpoint=old["checkpoint"],
                        environment=old["environment"], source_execution_status="interrupted")
        audit = Audit(directory, 1e-12)
        source_hashes = json.loads((parent / "source.sha256.json").read_text())
        for name, digest in source_hashes.items():
            if sha256(parent / "source" / name) != digest:
                raise ValueError("Frozen source snapshot changed")
        original_statistics = parent / "source/experiments/stage2_alignment_controls/statistics.py"
        if sha256(original_statistics) != sha256(Path(__file__).with_name("statistics.py")):
            raise ValueError("Recovery statistics differ from executed protocol implementation")
        shutil.copytree(parent / "source", directory / "source")
        shutil.copyfile(__file__, directory / "recover.py")
        manifest["recovery_code_sha256"] = sha256(directory / "recover.py")
        for name in ("config.input.json", "config.resolved.json", "protocol.md", "data_manifest.json",
                     "source.sha256.json", "environment.packages.json", "preflight.json", "preflight_checks.csv",
                     "metrics.csv", "pairs.csv", "shifts.csv"):
            shutil.copyfile(parent / name, directory / name)
        input_records = json.loads((parent / "data_manifest.json").read_text())
        metrics = pd.read_csv(parent / "metrics.csv")
        if len(input_records) != 384 or metrics.duplicated(["case", "variant", "score_kind"]).any():
            raise ValueError("Invalid completed prediction coverage")
        fingerprints = {name: sha256(parent / name) for name in (
            "manifest.json", "metrics.csv", "pairs.csv", "shifts.csv", "data_manifest.json", "protocol.md",
            "source.sha256.json", "preflight_checks.csv", "parent_artifacts.sha256.json", "run.log")}
        audits = {}
        for name in ("checks.csv", "gates/checks.csv", "regression/checks.csv"):
            frame = pd.read_csv(parent / name)
            if frame.empty or not frame.close.eq(True).all():
                raise ValueError(f"Failed source checks: {name}")
            fingerprints[name] = sha256(parent / name)
            audits[name] = {"count": len(frame), "max_abs": float(frame.max_abs.max())}
        for record in input_records:
            case = record["case"]
            input_path = parent / record["input_snapshot"]
            if sha256(input_path) != record["input_sha256"]:
                raise ValueError("Frozen input changed")
            fingerprints[record["input_snapshot"]] = sha256(input_path)
            score_name = f"arrays/{case}.npz"
            fingerprints[score_name] = sha256(parent / score_name)
            with np.load(input_path) as saved:
                labels = saved["labels"].copy()
            with np.load(parent / score_name) as saved:
                if set(saved.files) != {f"{v}_{k}" for v in ("A", "full_last", "full_all", "common_last", "common_all") for k in ("scores", "logits")}:
                    raise ValueError("Incomplete output arrays")
                for variant in ("A", "full_last", "full_all", "common_last", "common_all"):
                    for kind in ("scores", "logits"):
                        scores = saved[f"{variant}_{kind}"]
                        new = detection_metrics(labels, scores)
                        row = metrics[(metrics.case == case) & (metrics.variant == variant) & (metrics.score_kind == kind)]
                        if len(row) != 1 or new["metric_status"] != "ok":
                            raise ValueError("Missing metric or undefined output")
                        for metric in ("average_precision", "roc_auc"):
                            audit.check(f"{case}/{variant}/{kind}/{metric}", new[metric], row.iloc[0][metric])
                if record["offset"] == 8:
                    for scope in ("last", "all"):
                        for kind in ("scores", "logits"):
                            audit.check(f"{case}/common_full/{scope}/{kind}", saved[f"common_{scope}_{kind}"], saved[f"full_{scope}_{kind}"], exact=True)
        original_parent = Path(old["config"]["parent_run"])
        for name, digest in json.loads((parent / "parent_artifacts.sha256.json").read_text()).items():
            if sha256(original_parent / name) != digest:
                raise ValueError("E07 parent artifact changed")
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        manifest["status"] = "running_analysis"
        write_json(directory / "manifest.json", manifest)
        report = summarize(directory, config)
        for name, digest in fingerprints.items():
            if sha256(parent / name) != digest:
                raise ValueError("Interrupted source changed during recovery")
        summary = {"cases": 384, "source_predictions": 1920, "inference_calls": 0,
                   "source_status": "interrupted", "source_checks": audits, "offline_checks_passed": audit.count,
                   "source_model_check": "Original run checks model and parent fingerprints before entering summarize(); partial summary files demonstrate that code path was reached. No independent model reload in recovery.",
                   "model_sha256": json.loads((original_parent / "summary.json").read_text())["model_sha256"],
                   "parent_artifacts_unchanged": True}
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
