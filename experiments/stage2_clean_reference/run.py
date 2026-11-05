import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

from experiments.common.artifacts import append_row, capture_provenance, create_run, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage1.run import configure_runtime


def regression(directory, parent, records, config, model, device):
    import numpy as np
    from time_rcd._inference import _WindowDataset
    from experiments.stage2_patch_types.selections import patch_types
    from experiments.stage2_paths.inference import check_array
    from .inference import predict

    output = directory / "regression"
    (output / "arrays").mkdir(parents=True)
    for record in records:
        case = f"s{record['index']:04d}_w5000"
        path = parent / record["input_snapshot"]
        if sha256(path) != record["input_sha256"]:
            raise ValueError("Old synthetic input changed")
        with np.load(path) as saved:
            values, valid = _WindowDataset(saved["values"], 5000)[0]
            types = patch_types(saved["labels"], valid.numpy(), 16)
        if types["P"].any():
            raise ValueError("Old short events unexpectedly contain P")
        with np.load(parent / f"arrays/{case}.npz") as saved:
            baseline = {k: saved[f"A_{k}"].copy() for k in ("scores", "logits")}
            old_c = {k: saved[f"P_label__C_{k}"].copy() for k in ("scores", "logits")}
        predictions = {}
        for mode in ("A", "C", "V", "KV"):
            predictions[mode], _ = predict(model, values, valid, types["valid"], types["P"], mode, device, "self")
            for kind, scores in predictions[mode].items():
                for name, expected in (("A", baseline[kind]), ("P_C", old_c[kind])):
                    check_array(output, config, {"case": case, "test": f"{mode}_{kind}_parent_{name}"}, expected, scores)
        np.savez_compressed(output / "arrays" / f"{case}.npz",
                            **{f"{mode}_{k}": x for mode, pred in predictions.items() for k, x in pred.items()})
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Frozen-model final-layer clean K/V counterfactuals")
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
        for folder in ("stage2_clean_reference", "stage2_pc_localization"):
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
        for name in ("checks.csv", "gates/checks.csv"):
            checks = pd.read_csv(parent / name)
            if checks.empty or not checks.close.eq(True).all():
                raise ValueError("Failed parent checks")
        old = json.loads((parent / "data_manifest.json").read_text())
        old = [r for r in old if r["source"]["kind"] == "controlled_synthetic"]
        if len(old) != 27 or [r["index"] for r in old] != list(range(24, 51)):
            raise ValueError("Missing frozen regression cases")
        names = ["manifest.json", "summary.json", "data_manifest.json", "checks.csv", "gates/checks.csv"]
        for record in old:
            names.extend([record["input_snapshot"], f"arrays/s{record['index']:04d}_w5000.npz"])
        fingerprints = {name: sha256(parent / name) for name in names}
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            records, gates = prepare(directory, config)
            manifest["status"] = "running_gates"
            write_json(directory / "manifest.json", manifest)
            detector = make_detector(config, 5000)
            detector._ensure_tester(1)
            model, device = detector._tester.model, detector._tester.device
            fingerprint = model_fingerprint(model)
            if fingerprint != json.loads((parent / "summary.json").read_text())["model_sha256"]:
                raise ValueError("Model state differs from parent")
            gate_dir = directory / "gates"
            (gate_dir / "arrays").mkdir(parents=True)
            for index in gates:
                print(f"GATE {records[index]['case']}", flush=True)
                run_case(directory, gate_dir, config, model, device, records[index], gate=True)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Gate mutated model")
            count = regression(directory, parent, old, config, model, device)
            write_json(gate_dir / "summary.json", {"status": "passed", "indices": gates, "old_regression_cases": count})
            manifest["status"] = "running"
            write_json(directory / "manifest.json", manifest)
            predictions = 0
            for record in records:
                print(f"FULL {record['case']}", flush=True)
                predictions += run_case(directory, directory, config, model, device, record)
            if model_fingerprint(model) != fingerprint:
                raise ValueError("Execution mutated model")
            for name, digest in fingerprints.items():
                if sha256(parent / name) != digest:
                    raise ValueError(f"Parent artifact changed: {name}")
            report = summarize(directory, config)
            check_counts = {}
            for name in ("checks.csv", "gates/checks.csv", "regression/checks.csv"):
                frame = pd.read_csv(directory / name)
                if frame.empty or not frame.close.eq(True).all():
                    raise ValueError("Failed checks before summary")
                check_counts[name] = {"count": len(frame), "max_abs": float(frame.max_abs.max())}
            summary = {"cases": len(records), "predictions": predictions, "gate_cases": len(gates),
                       "regression_cases": count, "checks": check_counts, "model_sha256": fingerprint,
                       "model_state_unchanged": True, "parent_artifacts_unchanged": True}
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
