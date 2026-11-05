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
    parser = argparse.ArgumentParser(description="Frozen Patch-type reference interventions")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    settings["parent_run"] = str(project_path(settings["parent_run"]))
    settings["protocol_path"] = str(project_path(settings["protocol_path"]))
    previous = json.loads((Path(settings["parent_run"]) / "manifest.json").read_text())
    if previous["status"] != "completed":
        raise ValueError("Parent attention-path run must have passed")
    if settings["window"] != 5000 or settings["random_seeds"] != previous["config"]["random_seeds"]:
        raise ValueError("Frozen window/seeds mismatch")
    config = deepcopy(previous["config"])
    config.update(settings)
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        manifest["protocol_sha256"] = sha256(directory / "protocol.md")
        if manifest["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint changed")
        from experiments.stage1.inference import make_detector
        from experiments.stage2_paths.attention import model_fingerprint
        from .data import prepare
        from .inference import run_case
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            inputs, indices = prepare(directory, config)
            manifest["status"] = "running_gates"
            write_json(directory / "manifest.json", manifest)
            detector = make_detector(config, 5000)
            detector._ensure_tester(1)
            fingerprint = model_fingerprint(detector._tester.model)
            previous_summary = json.loads((Path(config["parent_run"]) / "summary.json").read_text())
            if fingerprint != previous_summary["model_sha256"]:
                raise ValueError("Loaded model state differs from parent")
            gate = directory / "gates"
            (gate / "arrays").mkdir(parents=True)
            for index in indices:
                print(f"GATE {index}", flush=True)
                run_case(gate, config, detector, inputs[index], gate=True)
            if fingerprint != model_fingerprint(detector._tester.model):
                raise ValueError("Gate changed model state")
            write_json(gate / "summary.json", {"status": "passed", "indices": indices})
            manifest["status"] = "running"
            write_json(directory / "manifest.json", manifest)
            conditions = 0
            for item in inputs:
                print(f"FULL {item[0]} {item[1].name}", flush=True)
                conditions += run_case(directory, config, detector, item)
            if fingerprint != model_fingerprint(detector._tester.model):
                raise ValueError("Model state changed")
            summary = {"cases": len(inputs), "conditions": conditions, "model_sha256": fingerprint,
                       "model_state_unchanged": True, "gates": "passed"}
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
