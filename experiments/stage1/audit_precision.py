"""Recheck one saved short-sequence padding case in float32 and float64."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback

from experiments.common.artifacts import capture_provenance, create_run, sha256, write_json
from experiments.stage1.run import configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    match = re.fullmatch(r"s(\d+)_w(\d+)", args.case)
    if match is None:
        raise ValueError("case must have the form s0003_w5000")
    index, window = map(int, match.groups())
    parent = args.run_directory.resolve()
    previous = json.loads((parent / "manifest.json").read_text())
    config = deepcopy(previous["config"])
    config.update(name=f"precision_{args.case}", device=args.device)
    directory, manifest = create_run(config, parent / "config.resolved.json")
    print(f"Precision audit: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        if manifest["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint changed since the parent run")
        sources = json.loads((parent / "data_manifest.json").read_text())
        source = next(row for row in sources if row["index"] == index)
        input_path = parent / source["input_snapshot"]
        if sha256(input_path) != source["input_sha256"]:
            raise ValueError("Input snapshot hash mismatch")
        manifest["audit"] = {"parent_run": str(parent), "case": args.case,
                             "parent_manifest_sha256": sha256(parent / "manifest.json"),
                             "input_sha256": source["input_sha256"],
                             "scope": "Numerical follow-up; original failure and tolerances are retained"}
        import numpy as np
        import torch
        from experiments.common.metrics import detection_metrics, difference
        from experiments.stage1.inference import make_detector, probe_short, score_window
        from time_rcd._inference import _WindowDataset

        with np.load(input_path) as inputs:
            values, labels = inputs["values"], inputs["labels"]
        if len(values) >= window:
            raise ValueError("This audit applies only to short sequences")
        detector = make_detector(config, window)
        detector._ensure_tester(1)
        normalized, mask = _WindowDataset(values, len(values))[0]
        rows, arrays = [], {}
        for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
            model = detector._tester.model.to(dtype=dtype)
            reference = score_window(model, normalized, mask, detector._tester.device)
            candidate = probe_short(detector, values)["short_forced_zero"]
            for kind in ("scores", "logits"):
                delta = difference(reference[kind], candidate[kind], config["atol"], config["rtol"])
                metrics = detection_metrics(labels, candidate[kind])
                baseline = detection_metrics(labels, reference[kind])
                rows.append({"dtype": name, "score_kind": kind, **delta,
                             "delta_ap": metrics["average_precision"] - baseline["average_precision"],
                             "delta_roc_auc": metrics["roc_auc"] - baseline["roc_auc"]})
                arrays[f"{name}_reference_{kind}"] = reference[kind]
                arrays[f"{name}_zero_{kind}"] = candidate[kind]
        np.savez_compressed(directory / "arrays" / "precision.npz", **arrays)
        write_json(directory / "precision.json", rows)
        manifest["status"] = "completed"
        print(json.dumps(rows, indent=2))
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
