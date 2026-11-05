import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

from experiments.common.artifacts import capture_provenance, create_run, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage1.run import configure_runtime


def main():
    parser = argparse.ArgumentParser(description="Fixed-input, fixed-count P position controls")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "recovery_run", "output_root", "checkpoint", "protocol_path"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        hashes = json.loads((directory / "source.sha256.json").read_text())
        for folder in ("stage2_p_positions", "stage2_alignment_controls", "stage2_clean_reference", "stage2_pc_localization"):
            for path in (ROOT / "experiments" / folder).glob("*.py"):
                relative = path.relative_to(ROOT)
                target = directory / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                hashes[str(relative)] = sha256(target)
        write_json(directory / "source.sha256.json", hashes)
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        manifest["protocol_sha256"] = sha256(directory / "protocol.md")
        import pandas as pd
        from experiments.stage1.inference import make_detector
        from experiments.stage2_paths.attention import model_fingerprint
        from .data import prepare, verify_parents
        from .inference import run_case
        from .statistics import summarize

        fingerprints = verify_parents(config, manifest["checkpoint"]["sha256"])
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            records, gates = prepare(directory, config)
            manifest["status"] = "running_gates"
            write_json(directory / "manifest.json", manifest)
            detector = make_detector(config, config["window"])
            detector._ensure_tester(1)
            model, device = detector._tester.model, detector._tester.device
            fingerprint = model_fingerprint(model)
            previous = json.loads((Path(config["recovery_run"]) / "summary.json").read_text())
            if fingerprint != previous["model_sha256"]:
                raise ValueError("Model differs from frozen parent")
            gate = directory / "gates"
            (gate / "arrays").mkdir(parents=True)
            for index in gates:
                print(f"GATE {records[index]['case']}", flush=True)
                run_case(directory, gate, config, model, device, records[index], gate=True)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Gate changed model")
            write_json(gate / "summary.json", {"status": "passed", "indices": gates, "model_sha256": fingerprint})
            manifest["status"] = "running_inference"
            write_json(directory / "manifest.json", manifest)
            count = 0
            for record in records:
                print(f"FULL {record['case']}", flush=True)
                count += run_case(directory, directory, config, model, device, record)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Execution changed model")
            for name, digest in fingerprints.items():
                if sha256(name) != digest:
                    raise ValueError(f"Parent artifact changed: {name}")
            for record in records:
                for kind in ("input", "selection"):
                    if sha256(directory / record[f"{kind}_snapshot"]) != record[f"{kind}_sha256"]:
                        raise ValueError("Frozen input or selection changed")
            audits = {}
            for name in ("checks.csv", "gates/checks.csv"):
                frame = pd.read_csv(directory / name)
                if frame.empty or not frame.close.eq(True).all():
                    raise ValueError("Failed execution checks")
                audits[name] = {"count": len(frame), "max_abs": float(frame.max_abs.max())}
            summary = {"cases": len(records), "predictions": count, "gate_cases": len(gates), "checks": audits,
                       "model_sha256": fingerprint, "model_state_unchanged": True, "parent_artifacts_unchanged": True}
            write_json(directory / "inference_summary.json", summary)
            manifest["status"] = "running_analysis"
            write_json(directory / "manifest.json", manifest)
            report = summarize(directory, config)
            for name, digest in fingerprints.items():
                if sha256(name) != digest:
                    raise ValueError("Parent changed during analysis")
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
