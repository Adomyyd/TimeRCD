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


def regression(directory, parent, records, config, model, device):
    import numpy as np
    from experiments.stage2_clean_reference.data import load_input
    from experiments.stage2_paths.inference import check_array
    from .inference import predict

    output = directory / "regression"
    (output / "arrays").mkdir(parents=True)
    for record in records:
        arrays, values, _, valid = load_input(parent, record)
        with np.load(parent / f"arrays/{record['case']}.npz") as saved:
            old = {k: saved[k].copy() for k in saved.files}
        results = {}
        for variant, condition, scope in (("A", "A", "all"), ("C", "C", "last"), ("C_all", "C", "all")):
            pred, _ = predict(model, values, valid, arrays["type_valid"], arrays["type_P"], condition, scope, device)
            for kind, scores in pred.items():
                check_array(output, config, {"case": record["case"], "test": f"{variant}_{kind}_parent"}, old[f"{variant}_{kind}"], scores)
                results[f"{variant}_{kind}"] = scores
        np.savez_compressed(output / "arrays" / f"{record['case']}.npz", **results)


def main():
    parser = argparse.ArgumentParser(description="Paired background and common-P alignment controls")
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
        for folder in ("stage2_alignment_controls", "stage2_clean_reference", "stage2_pc_localization"):
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
        from .data import prepare
        from .inference import run_case
        from .statistics import summarize

        parent = Path(config["parent_run"])
        previous = json.loads((parent / "manifest.json").read_text())
        if previous["status"] != "completed" or manifest["checkpoint"]["sha256"] != previous["checkpoint"]["sha256"]:
            raise ValueError("Parent status or checkpoint mismatch")
        for name in ("checks.csv", "gates/checks.csv", "regression/checks.csv"):
            checks = pd.read_csv(parent / name)
            if checks.empty or not checks.close.eq(True).all():
                raise ValueError("Failed parent checks")
        old = json.loads((parent / "data_manifest.json").read_text())
        old = [r for r in old if r["seed"] == 3101 and r["event_length"] > 5]
        if len(old) != 12:
            raise ValueError("Missing regression inputs")
        names = ["manifest.json", "summary.json", "data_manifest.json", "checks.csv", "gates/checks.csv", "regression/checks.csv"]
        for r in old:
            names.extend([r["input_snapshot"], f"arrays/{r['case']}.npz"])
        fingerprints = {name: sha256(parent / name) for name in names}
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            records, gates = prepare(directory, config)
            manifest["status"] = "running_gates"
            write_json(directory / "manifest.json", manifest)
            detector = make_detector(config, config["window"])
            detector._ensure_tester(1)
            model, device = detector._tester.model, detector._tester.device
            fingerprint = model_fingerprint(model)
            if fingerprint != json.loads((parent / "summary.json").read_text())["model_sha256"]:
                raise ValueError("Model differs from parent")
            gate = directory / "gates"
            (gate / "arrays").mkdir(parents=True)
            for index in gates:
                print(f"GATE {records[index]['case']}", flush=True)
                run_case(directory, gate, config, model, device, records[index], gate=True)
            regression(directory, parent, old, config, model, device)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Gate changed model")
            write_json(gate / "summary.json", {"status": "passed", "indices": gates, "regression_cases": len(old)})
            manifest["status"] = "running"
            write_json(directory / "manifest.json", manifest)
            count = 0
            for record in records:
                print(f"FULL {record['case']}", flush=True)
                count += run_case(directory, directory, config, model, device, record)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Execution changed model")
            for name, digest in fingerprints.items():
                if sha256(parent / name) != digest:
                    raise ValueError("Parent artifact changed")
            for record in records:
                if sha256(directory / record["input_snapshot"]) != record["input_sha256"]:
                    raise ValueError("Frozen input changed")
            report = summarize(directory, config)
            audits = {}
            for name in ("checks.csv", "gates/checks.csv", "regression/checks.csv"):
                frame = pd.read_csv(directory / name)
                if frame.empty or not frame.close.eq(True).all():
                    raise ValueError("Failed execution checks")
                audits[name] = {"count": len(frame), "max_abs": float(frame.max_abs.max())}
            summary = {"cases": len(records), "predictions": count, "gate_cases": len(gates), "regression_cases": len(old),
                       "preflight_pairs": len(pd.read_csv(directory / "preflight_checks.csv")), "checks": audits,
                       "model_sha256": fingerprint, "model_state_unchanged": True, "parent_artifacts_unchanged": True}
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
