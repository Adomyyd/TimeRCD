import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import traceback

from experiments.common.artifacts import append_row, capture_provenance, create_run, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage1.run import configure_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("output_root", "protocol_path", "checkpoint", "r_run"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint fingerprint mismatch")
        previous = json.loads((Path(config["r_run"]) / "manifest.json").read_text())
        if previous["status"] != "completed":
            raise RuntimeError("R not complete")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"]).decode()
        (directory / "gpu_preflight.csv").write_text(gpu)
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
            raise RuntimeError("Expected CUDA_VISIBLE_DEVICES=2")
        line = [line for line in gpu.splitlines()[1:] if line.split(",")[0].strip() == "2"][0]
        used, utilization = int(line.split(",")[1].split()[0]), int(line.split(",")[3].split()[0])
        if used > 512 or utilization > 10:
            raise RuntimeError("GPU2 is busy; stop rather than interfere")
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        sources = json.loads((directory / "source.sha256.json").read_text())
        for folder in ("stage2_isolated", "stage2_cluster_probe", "stage2_clean_reference"):
            for path in (ROOT / "experiments" / folder).rglob("*.py"):
                target = directory / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                sources[str(path.relative_to(ROOT))] = sha256(target)
        write_json(directory / "source.sha256.json", sources)
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        protect = {config["checkpoint"]: sha256(config["checkpoint"])}
        for name in ("manifest.json", "summary.json", "per_series.csv"):
            path = Path(config["r_run"]) / name
            protect[str(path)] = sha256(path)
        write_json(directory / "parent_artifacts.sha256.json", protect)
        import numpy as np
        import torch
        from experiments.stage1.inference import make_detector
        from experiments.stage2_paths.attention import model_fingerprint
        from .data import generate_isolated
        from .inference import Auditor, gates
        from .statistics import readout, summarize

        records = []
        for record, arrays in generate_isolated(config):
            path = directory / f"arrays/input_{record['case']}.npz"
            np.savez_compressed(path, **arrays)
            records.append({**record, "input_snapshot": str(path.relative_to(directory)), "input_sha256": sha256(path)})
        if len(records) != 40:
            raise ValueError("Wrong factorial size")
        write_json(directory / "data_manifest.json", records)
        detector = make_detector(config, config["window"])
        detector._ensure_tester(1)
        model, device = detector._tester.model, detector._tester.device
        initial = model_fingerprint(model)
        if initial != config["model_sha256"] or any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("Model fingerprint or dtype mismatch")
        gate_dir = directory / "gates"
        (gate_dir / "arrays").mkdir(parents=True)
        gate_auditor = Auditor(gate_dir, model, device)
        manifest["status"] = "running_gates"
        write_json(directory / "manifest.json", manifest)
        for record in records[:5]:
            with np.load(directory / record["input_snapshot"]) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            gates(gate_auditor, record["case"], arrays)
        write_json(gate_dir / "summary.json", {"checks": gate_auditor.checks, "forwards": gate_auditor.forwards, "cases": 5})
        auditor = Auditor(directory, model, device)
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        for record in records:
            case = record["case"]
            auditor.check(case, "input_sha256", sha256(directory / record["input_snapshot"]) == record["input_sha256"])
            with np.load(directory / record["input_snapshot"]) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            values = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32)
            valid = torch.ones(len(values), dtype=torch.bool)
            official = auditor.public(values, valid)
            predictions = {"A": auditor.predict(case, "A", values, valid, arrays["type_valid"], arrays["type_P"], "A")}
            for kind in official:
                auditor.equal(case, f"A_public_{kind}", official[kind], predictions["A"][kind])
            for variant in ("C_anom", "D_anom") + tuple(f"D_rand_{k}" for k in range(1, 6)):
                predictions[variant] = auditor.predict(case, variant, values, valid, arrays["type_valid"],
                                                       arrays[f"selection_{variant}"], variant[0])
            np.savez_compressed(directory / f"arrays/{case}.npz",
                                **{f"{v}_{k}": x for v, pred in predictions.items() for k, x in pred.items()})
            for variant, pred in predictions.items():
                for kind, scores in pred.items():
                    append_row(directory / "metrics.csv", {"case": case, "seed": record["seed"], "position": record["position"],
                               "variant": variant, "score_kind": kind, **readout(arrays["labels"], scores, record["position"])})
            print(case, flush=True)
        final = model_fingerprint(model)
        auditor.check("final", "model_unchanged", final == initial == config["model_sha256"])
        for path, digest in protect.items():
            auditor.check("final", f"protected_{path}", sha256(path) == digest)
        for record in records:
            auditor.check(record["case"], "input_unchanged", sha256(directory / record["input_snapshot"]) == record["input_sha256"])
        write_json(directory / "inference_summary.json", {"cases": 40, "formal_predictions": 320,
                   "formal_and_parity_forwards": auditor.forwards, "gate_forwards": gate_auditor.forwards,
                   "checks": auditor.checks, "gate_checks": gate_auditor.checks, "model_sha256": final,
                   "model_unchanged": True, "protected_artifacts_unchanged": True})
        result = summarize(directory, config)
        write_json(directory / "summary.json", {**json.loads((directory / "inference_summary.json").read_text()), **result})
        manifest["status"] = "completed"
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
