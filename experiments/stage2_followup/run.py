import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from experiments.common.artifacts import append_row, capture_provenance, create_run, sha256, write_json
from experiments.common.config import project_path
from experiments.stage1.run import configure_runtime


def crossover(directory, parent, config):
    import numpy as np
    from experiments.common.data import Series
    from experiments.common.metrics import difference
    from experiments.stage1.inference import make_detector
    from experiments.stage2.inference import run_case

    sources = json.loads((parent / "data_manifest.json").read_text())
    detector = make_detector(config, 5000)
    checks, outputs = [], []
    for source in sources:
        if source["source"]["kind"] != "controlled_synthetic":
            continue
        path = parent / source["input_snapshot"]
        if sha256(path) != source["input_sha256"]:
            raise ValueError(f"Changed parent input: {path}")
        with np.load(path) as raw:
            values, labels = raw["values"].copy(), raw["labels"].copy()
        origin = deepcopy(source["source"])
        origin["normalization"] = "official_series"
        series = Series(source["name"], values, labels, origin)
        index = source["index"]
        print(f"Official normalization crossover: {series.name}", flush=True)
        checked = run_case(directory, config, detector, series, index, 5000)
        case = f"s{index:04d}_w5000"
        original_masks = parent / f"arrays/{case}_masks.npz"
        current_masks = directory / f"arrays/{case}_masks.npz"
        with np.load(original_masks) as old, np.load(current_masks) as new:
            if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
                raise ValueError("Normalization crossover changed reference masks")
        public = detector.predict(values, return_logits=True)
        with np.load(directory / f"arrays/{case}.npz") as arrays:
            parity = {"case": case, "masks_identical": True}
            for kind, reference in zip(("scores", "logits"), public):
                delta = difference(reference, arrays[f"baseline_{kind}"], config["atol"], config["rtol"])
                parity.update({f"public_{kind}_{k}": v for k, v in delta.items()})
                checked = checked and delta["close"]
        append_row(directory / "crossover_checks.csv", parity)
        np.savez_compressed(directory / source["input_snapshot"], values=values, labels=labels)
        outputs.append({**source, "source": origin,
                        "input_sha256": sha256(directory / source["input_snapshot"]),
                        "parent_input_sha256": source["input_sha256"],
                        "parent_masks_sha256": sha256(original_masks)})
        checks.append(checked)
    write_json(directory / "data_manifest.json", outputs)
    if len(checks) != 27:
        raise ValueError(f"Expected the frozen 27 synthetic inputs, got {len(checks)}")
    return {"crossover_cases": len(checks), "failed": len(checks) - sum(checks)}


def main():
    parser = argparse.ArgumentParser(description="Hard-mask score diagnostics and normalization crossover")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    parent = project_path(settings["parent_run"])
    previous = json.loads((parent / "manifest.json").read_text())
    if previous["status"] != "completed":
        raise ValueError("Follow-up requires a completed parent with passed checks")
    config = deepcopy(previous["config"])
    config.update(settings)
    config["parent_run"] = str(parent)
    if config["random_seeds"] != settings["protocol"]["random_seeds"]:
        raise ValueError("Random seeds differ from parent")
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        if manifest["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint changed")
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        from .offline import audit_saved
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            cases = audit_saved(directory, parent)
            result = crossover(directory, parent, config)
        result["offline_cases"] = cases
        write_json(directory / "summary.json", result)
        manifest["status"] = "completed" if result["failed"] == 0 else "checks_failed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)
    print(json.dumps(result))
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
