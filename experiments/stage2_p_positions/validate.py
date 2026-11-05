import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.common.metrics import detection_metrics
from experiments.stage2_pc_localization.data import Audit


def validate(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Expected a completed run")
    config = manifest["config"]
    output = directory / "output_validation"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "validate.py")
    audit = Audit(output, 1e-12)
    records = json.loads((directory / "data_manifest.json").read_text())
    gates = json.loads((directory / "preflight.json").read_text())["gate_indices"]
    metrics = pd.read_csv(directory / "metrics.csv").set_index(["case", "variant", "score_kind"])
    variants = ["A"] + [f"{s}_{scope}" for s in ["full", "keep_first", "keep_last"] + [
        f"random_{seed}" for seed in config["random_seeds"]] for scope in ("last", "all")]
    fingerprints = {str(directory / name): sha256(directory / name) for name in (
        "manifest.json", "metrics.csv", "data_manifest.json", "inference_summary.json", "summary.json")}
    for index, record in enumerate(records):
        case = record["case"]
        for kind in ("input", "selection"):
            path = directory / record[f"{kind}_snapshot"]
            if sha256(path) != record[f"{kind}_sha256"]:
                raise ValueError("Input or selection changed")
            fingerprints[str(path)] = sha256(path)
        with np.load(directory / record["input_snapshot"]) as saved:
            labels = saved["labels"].copy()
        for scope in (["full", "gate"] if index in gates else ["full"]):
            path = directory / ("gates" if scope == "gate" else "") / f"arrays/{case}.npz"
            fingerprints[str(path)] = sha256(path)
            with np.load(path) as saved:
                if set(saved.files) != {f"{v}_{k}" for v in variants for k in ("scores", "logits")}:
                    raise ValueError("Incomplete saved prediction keys")
                for variant in variants:
                    for kind in ("scores", "logits"):
                        prediction = saved[f"{variant}_{kind}"]
                        audit.check(f"{scope}/{case}/{variant}/{kind}/finite_length",
                                    prediction.shape == labels.shape and np.isfinite(prediction).all(), True, exact=True)
                        if scope == "full":
                            values = detection_metrics(labels, prediction)
                            for metric in ("average_precision", "roc_auc"):
                                audit.check(f"{case}/{variant}/{kind}/{metric}", values[metric],
                                            metrics.loc[(case, variant, kind), metric])
    absolute = pd.read_csv(directory / "absolute_with_random_mean.csv").set_index(["case", "score_kind", "contrast"])
    effects = pd.read_csv(directory / "effects.csv").set_index(["case", "score_kind", "contrast"])
    for record in records:
        case = record["case"]
        for kind in ("scores", "logits"):
            for metric in ("average_precision", "roc_auc"):
                differences = {}
                for scope in ("last", "all"):
                    random_mean = np.mean([metrics.loc[(case, f"random_{seed}_{scope}", kind), metric]
                                           for seed in config["random_seeds"]])
                    audit.check(f"{case}/{kind}/{scope}/{metric}/random_metric_mean", random_mean,
                                absolute.loc[(case, kind, f"random_mean_{scope}"), metric])
                    differences[scope] = (metrics.loc[(case, f"keep_last_{scope}", kind), metric]
                                          - metrics.loc[(case, f"keep_first_{scope}", kind), metric])
                    audit.check(f"{case}/{kind}/{scope}/{metric}/position_difference", differences[scope],
                                effects.loc[(case, kind, f"keep_last_{scope}_minus_keep_first_{scope}"), metric])
                audit.check(f"{case}/{kind}/{metric}/position_by_scope", differences["all"] - differences["last"],
                            effects.loc[(case, kind, "position_by_scope"), metric])
    fingerprints.update(json.loads((directory / "parent_artifacts.sha256.json").read_text()))
    for name, digest in fingerprints.items():
        if sha256(name) != digest:
            raise ValueError(f"Artifact changed during validation: {name}")
    write_json(output / "artifacts.sha256.json", fingerprints)
    summary = {"status": "passed", "checks": audit.count, "inference_calls": 0,
               "formal_cases": len(records), "gate_cases": len(gates), "artifacts_unchanged": True,
               "validation_code_sha256": sha256(output / "validate.py")}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify saved E09 predictions and primary metric identities")
    parser.add_argument("run", type=Path)
    print(json.dumps(validate(parser.parse_args().run.resolve()), ensure_ascii=False))
