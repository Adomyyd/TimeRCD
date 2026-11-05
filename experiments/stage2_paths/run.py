import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

from experiments.common.artifacts import capture_provenance, create_run, sha256, write_json
from experiments.common.config import project_path
from experiments.stage1.run import configure_runtime


def main():
    parser = argparse.ArgumentParser(description="Pre-specified attention-edge ablations")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    for key in ("real_parent", "synthetic_parent", "protocol_path"):
        settings[key] = str(project_path(settings[key]))
    parents = [json.loads((Path(settings[key]) / "manifest.json").read_text()) for key in ("real_parent", "synthetic_parent")]
    if any(p["status"] != "completed" for p in parents):
        raise ValueError("Both parent runs must have passed their checks")
    if parents[0]["checkpoint"]["sha256"] != parents[1]["checkpoint"]["sha256"]:
        raise ValueError("Parent weights differ")
    config = deepcopy(parents[0]["config"])
    config.update(settings)
    if config["window"] != 5000 or config["random_seeds"] != parents[0]["config"]["random_seeds"]:
        raise ValueError("Window or random selections differ from protocol")
    config["batch_size"] = 1
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        manifest["protocol_sha256"] = sha256(directory / "protocol.md")
        if manifest["checkpoint"]["sha256"] != parents[0]["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint changed from parents")
        from experiments.stage1.inference import make_detector
        from .attention import model_fingerprint
        from .data import load_inputs
        from .inference import run_case
        manifest["status"] = "running_gates"
        write_json(directory / "manifest.json", manifest)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            inputs = load_inputs(directory, config)
            detector = make_detector(config, 5000)
            detector._ensure_tester(1)
            fingerprint = model_fingerprint(detector._tester.model)
            gate = directory / "gates"
            (gate / "arrays").mkdir(parents=True)
            for index in config["gate_indices"]:
                print(f"GATE {index}", flush=True)
                run_case(gate, config, detector, inputs[index], gate=True)
            if fingerprint != model_fingerprint(detector._tester.model):
                raise ValueError("Model state changed during gates")
            write_json(gate / "summary.json", {"status": "passed", "indices": config["gate_indices"], "model_sha256": fingerprint})
            manifest["status"] = "running"
            write_json(directory / "manifest.json", manifest)
            count = 0
            for item in inputs:
                print(f"FULL {item[0]} {item[1].name}", flush=True)
                count += run_case(directory, config, detector, item)
            if fingerprint != model_fingerprint(detector._tester.model):
                raise ValueError("Model state changed during experiment")
            summary = {"cases": len(inputs), "conditions": count, "model_state_unchanged": True,
                       "model_sha256": fingerprint, "gates": "passed"}
            write_json(directory / "summary.json", summary)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
