import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

from experiments.common.artifacts import append_row, capture_provenance, create_run, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage1.run import configure_runtime


def summarize_dose(directory, config):
    import numpy as np
    import pandas as pd
    from .statistics import interval
    frame = pd.read_csv(directory / "metrics.csv")
    case_frame = frame.groupby(["case", "seed", "position", "count"], as_index=False)[["ap_loss", "anomaly_shift"]].mean()
    case_frame.to_csv(directory / "paired.csv", index=False)
    seeds = case_frame.groupby(["seed", "count"], as_index=False)[["ap_loss", "anomaly_shift"]].mean()
    seeds.to_csv(directory / "seed_effects.csv", index=False)
    rows = []
    for count, part in seeds.groupby("count"):
        for measure in ("ap_loss", "anomaly_shift"):
            rows.append({"count": int(count), "measure": measure, "n_seeds": len(part),
                         **interval(part[measure], config["bootstrap_seed"], config["bootstrap_repetitions"])})
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / "bootstrap.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    plot = summary[summary.measure == "ap_loss"]
    ax.errorbar(plot["count"], plot["mean"], yerr=np.array([plot["mean"] - plot.low, plot.high - plot["mean"]]),
                marker="o", capsize=4)
    ax.axhline(0, color="0.6", linewidth=.8)
    ax.set(xlabel="Number of deleted normal Patches", ylabel="AP(A) - AP(D_random)", xticks=[1, 16, 64])
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"dose_ap_loss.{suffix}", dpi=160)
    plt.close(fig)
    return rows


def execute_dose(directory, config, manifest):
    import numpy as np
    import torch
    from experiments.stage1.inference import make_detector
    from experiments.stage2.masking import reference_intervention
    from experiments.stage2_paths.attention import model_fingerprint
    from .data import random_selection
    from .inference import Auditor
    from .statistics import readout

    parent = Path(config["s_run"])
    if json.loads((parent / "manifest.json").read_text())["status"] != "completed":
        raise RuntimeError("S incomplete")
    if json.loads((parent / "decision.json").read_text())["decision"] != "pause_H_isolated_specific_harm":
        raise RuntimeError("S did not enter this branch")
    records = json.loads((parent / "data_manifest.json").read_text())
    paths = [parent / name for name in ("manifest.json", "summary.json", "decision.json", "data_manifest.json")]
    paths += [parent / r["input_snapshot"] for r in records]
    paths += [parent / f"arrays/{r['case']}.npz" for r in records]
    paths += [Path(config["checkpoint"])]
    protected = {str(path): sha256(path) for path in paths}
    write_json(directory / "parent_artifacts.sha256.json", protected)
    write_json(directory / "data_manifest.json", records)
    detector = make_detector(config, 5000)
    detector._ensure_tester(1)
    model, device = detector._tester.model, detector._tester.device
    if model_fingerprint(model) != config["model_sha256"]:
        raise RuntimeError("Wrong model")
    auditor = Auditor(directory, model, device)
    for record in records:
        case = record["case"]
        with np.load(parent / record["input_snapshot"]) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        with np.load(parent / f"arrays/{case}.npz") as saved:
            previous = {k: saved[k].copy() for k in saved.files}
        values = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32)
        valid = torch.ones(len(values), dtype=torch.bool)
        baseline = auditor.public(values, valid)
        for kind in baseline:
            auditor.equal(case, f"A_parent_{kind}", baseline[kind], previous[f"A_{kind}"])
        base_metrics = readout(arrays["labels"], baseline["scores"], record["position"])
        output = {f"A_{k}": v for k, v in baseline.items()}
        types = {k.removeprefix("type_"): v for k, v in arrays.items() if k.startswith("type_")}
        for count in config["counts"]:
            for repetition in range(1, 6):
                selected = random_selection(types, record["seed"], record["position"], repetition, count)
                variant = f"q{count}_k{repetition}"
                auditor.check(case, f"{variant}_normal_complete_count", selected.sum() == count and
                              not (selected & (~types["N"] | types["partial"])).any())
                prediction = auditor.predict(case, variant, values, valid, types["valid"], selected, "D")
                if count == 1:
                    for kind in prediction:
                        auditor.equal(case, f"{variant}_{kind}_S_replay", prediction[kind], previous[f"D_rand_{repetition}_{kind}"])
                if case == records[0]["case"] and count in (16, 64) and repetition == 1:
                    with reference_intervention(model, types["valid"] & ~selected):
                        reference = auditor.public(values, valid)
                    for kind in prediction:
                        auditor.equal(case, f"{variant}_{kind}_reference_intervention", prediction[kind], reference[kind])
                result = readout(arrays["labels"], prediction["scores"], record["position"])
                append_row(directory / "metrics.csv", {"case": case, "seed": record["seed"], "position": record["position"],
                           "count": count, "repetition": repetition, "ap_A": base_metrics["average_precision"],
                           "ap_D": result["average_precision"], "ap_loss": base_metrics["average_precision"] - result["average_precision"],
                           "anomaly_shift": result["anomaly_mean"] - base_metrics["anomaly_mean"]})
                output.update({f"{variant}_{k}": v for k, v in prediction.items()})
                output[f"{variant}_selected"] = selected
        np.savez_compressed(directory / f"arrays/{case}.npz", **output)
    auditor.check("final", "model_unchanged", model_fingerprint(model) == config["model_sha256"])
    for path, digest in protected.items():
        auditor.check("final", f"protected_{path}", sha256(path) == digest)
    summary = {"cases": len(records), "predictions": 600, "forwards": auditor.forwards, "checks_passed": auditor.checks,
               "model_unchanged": True, "parent_artifacts_unchanged": True}
    write_json(directory / "inference_summary.json", summary)
    summary["dose"] = summarize_dose(directory, config)
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("output_root", "protocol_path", "checkpoint", "s_run"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        if sha256(config["checkpoint"]) != config["checkpoint_sha256"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
            raise RuntimeError("Checkpoint/GPU identity mismatch")
        gpu = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]).decode()
        (directory / "gpu_preflight.csv").write_text(gpu)
        used, utilization = [int(v.strip()) for v in gpu.split(",")]
        if used > 512 or utilization > 10:
            raise RuntimeError("GPU2 busy")
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        fingerprints = json.loads((directory / "source.sha256.json").read_text())
        for path in (ROOT / "experiments/stage2_isolated").rglob("*.py"):
            target = directory / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            fingerprints[str(path.relative_to(ROOT))] = sha256(target)
        write_json(directory / "source.sha256.json", fingerprints)
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        print(json.dumps(execute_dose(directory, config, manifest), ensure_ascii=False), flush=True)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
