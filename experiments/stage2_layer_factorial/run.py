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
    parser = argparse.ArgumentParser(description="Final two attention layers factorial controls")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "output_root", "checkpoint", "protocol_path"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        hashes = json.loads((directory / "source.sha256.json").read_text())
        for folder in ("stage2_layer_factorial", "stage2_depth_controls", "stage2_rank_localization", "stage2_p_positions",
                       "stage2_alignment_controls", "stage2_clean_reference", "stage2_pc_localization"):
            for path in (ROOT / "experiments" / folder).glob("*.py"):
                target = directory / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                hashes[str(path.relative_to(ROOT))] = sha256(target)
        write_json(directory / "source.sha256.json", hashes)
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        manifest["protocol_sha256"] = sha256(directory / "protocol.md")
        import pandas as pd
        from experiments.stage1.inference import make_detector
        from experiments.stage2_paths.attention import model_fingerprint
        from experiments.stage2_pc_localization.data import Audit
        from .attention import switches
        from .data import prepare
        from .inference import run_case
        from .statistics import summarize

        conditions = switches(config["switch_layers"], config["expected_layers"])
        records, gates, fingerprints = prepare(directory, config, manifest["checkpoint"]["sha256"])
        parent_metrics = pd.read_csv(Path(config["parent_run"]) / "metrics.csv").set_index(["case", "variant", "score_kind"])
        if not parent_metrics.index.is_unique or len(parent_metrics) != 3456:
            raise ValueError("Invalid E11 metric coverage")
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            detector = make_detector(config, config["window"])
            detector._ensure_tester(1)
            model, device = detector._tester.model, detector._tester.device
            fingerprint = model_fingerprint(model)
            previous = json.loads((Path(config["parent_run"]) / "summary.json").read_text())
            if fingerprint != previous["model_sha256"] or len(model.ts_encoder.transformer_encoder.layers) != config["expected_layers"]:
                raise ValueError("Unexpected model state or depth")
            write_json(directory / "preflight.json", {"cases": len(records), "gate_indices": gates,
                       "layers": config["expected_layers"], "active_layers": conditions, "model_sha256": fingerprint})
            gate = directory / "gates"
            (gate / "arrays").mkdir(parents=True)
            (gate / "diagnostics").mkdir()
            (directory / "diagnostics").mkdir()
            gate_audit, audit = Audit(gate / "diagnostics", config["tolerance"]), Audit(directory / "diagnostics", config["tolerance"])
            manifest["status"] = "running_gates"
            write_json(directory / "manifest.json", manifest)
            for index in gates:
                print(f"GATE {records[index]['case']}", flush=True)
                run_case(directory, gate, config, model, device, records[index], parent_metrics, gate_audit, gate=True)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Gate changed model")
            write_json(gate / "summary.json", {"status": "passed", "cases": len(gates), "model_sha256": fingerprint})
            manifest["status"] = "running_inference"
            write_json(directory / "manifest.json", manifest)
            count = 0
            for record in records:
                print(f"FULL {record['case']}", flush=True)
                count += run_case(directory, directory, config, model, device, record, parent_metrics, audit)
            if model_fingerprint(model) != fingerprint or sha256(config["checkpoint"]) != manifest["checkpoint"]["sha256"]:
                raise ValueError("Model or checkpoint changed")
            for record in records:
                for kind in ("input", "selection"):
                    if sha256(directory / record[f"{kind}_snapshot"]) != record[f"{kind}_sha256"]:
                        raise ValueError("Frozen input or selection changed")
            for name, digest in fingerprints.items():
                if sha256(name) != digest:
                    raise ValueError(f"Parent artifact changed: {name}")
            audits = {}
            for name in ("checks.csv", "gates/checks.csv", "diagnostics/checks.csv", "gates/diagnostics/checks.csv"):
                frame = pd.read_csv(directory / name)
                if frame.empty or not frame.close.eq(True).all():
                    raise ValueError("Failed run checks")
                audits[name] = {"count": len(frame), "max_abs": float(frame.max_abs.max())}
            summary = {"cases": len(records), "predictions": count, "gate_cases": len(gates), "checks": audits,
                       "model_sha256": fingerprint, "model_state_unchanged": True, "parent_artifacts_unchanged": True}
            write_json(directory / "inference_summary.json", summary)
            manifest["status"] = "running_statistics"
            write_json(directory / "manifest.json", manifest)
            report = summarize(directory, config)
            for name, digest in fingerprints.items():
                if sha256(name) != digest:
                    raise ValueError("Parent changed during statistics")
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
