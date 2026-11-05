import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np
import torch

from experiments.common.artifacts import capture_provenance, create_run, sha256, write_json
from experiments.common.config import project_path
from experiments.stage1.inference import make_detector
from experiments.stage2_context_validation.data import (
    LENGTH,
    RADIUS_VALUES,
    TARGET,
    context_mask,
    experiment_a_cases,
    experiment_b_cases,
    generate_base,
    normalize,
    ring_masks,
)
from experiments.stage2_context_validation.inference import (
    disable_tf32,
    score_condition,
    target_logits,
)


def _write_rows(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gpu_gate():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    records = []
    for line in result.stdout.splitlines():
        index, utilization, free_memory = [int(part.strip()) for part in line.split(",")]
        records.append((index, utilization, free_memory))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if "CUDA_VISIBLE_DEVICES" in os.environ and not visible.isdigit():
        raise RuntimeError("Select exactly one GPU using its physical index in CUDA_VISIBLE_DEVICES")
    selected = int(visible) if visible else 0
    status = next((item for item in records if item[0] == selected), None)
    if status is None:
        raise RuntimeError(f"Cannot identify selected physical GPU {selected}: {records}")
    index, utilization, free_memory = status
    if utilization > 25 or free_memory < 6000:
        raise RuntimeError(
            f"GPU {index} is not idle enough for inference: utilization={utilization}%, "
            f"free_memory={free_memory} MiB; select an idle GPU and retry"
        )
    return {"index": index, "utilization_percent": utilization, "free_memory_mib": free_memory}


def _resolve_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    required = {
        "name", "protocol_path", "output_root", "checkpoint", "device", "seed",
        "bootstrap_repetitions", "generation_seeds", "length", "period", "noise_std",
        "amplitude", "window", "patch_size", "target", "radii", "batch_size", "threads",
        "atol", "rtol",
    }
    if set(config) != required:
        raise ValueError(f"Config keys differ from required set: {sorted(set(config) ^ required)}")
    if config["device"] != "cuda":
        raise ValueError("Official checkpoint inference is gated to CUDA")
    if config["seed"] != 2024 or config["bootstrap_repetitions"] != 10000:
        raise ValueError("Bootstrap seed/repetition count differs from the frozen protocol")
    if config["batch_size"] != 1 or config["threads"] <= 0:
        raise ValueError("Inference batch size/threads differ from the frozen protocol")
    if (config["length"], config["window"], config["patch_size"], tuple(config["target"])) != (
        LENGTH, LENGTH, 16, TARGET
    ):
        raise ValueError("Config geometry differs from the frozen protocol")
    if (config["period"], config["noise_std"], config["amplitude"]) != (64, 0.1, 4.0):
        raise ValueError("Synthetic generator settings differ from the frozen protocol")
    if config["radii"] != [*RADIUS_VALUES, "full"] or config["generation_seeds"] != list(range(3101, 3119)):
        raise ValueError("Seed count or context radii differ from the frozen protocol")
    for key in ("checkpoint", "output_root", "protocol_path"):
        config[key] = str(project_path(config[key]))
    config["data"] = {"kind": "controlled_synthetic_context"}
    config["config_source"] = str(path)
    if not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError(config["checkpoint"])
    if not Path(config["protocol_path"]).is_file():
        raise FileNotFoundError(config["protocol_path"])
    return config


def _save_inputs(directory, config):
    manifest = []
    a_cases = []
    b_cases = []
    for seed in config["generation_seeds"]:
        base = generate_base(seed, config["length"], config["period"], config["noise_std"])
        a_cases.extend(experiment_a_cases(seed, base))
        b_cases.extend(experiment_b_cases(seed, base))
    arrays_dir = directory / "arrays"
    for case in a_cases:
        path = arrays_dir / f"{case['name']}.npz"
        normalized, mean, std = normalize(case["values"])
        restored_normalized, _, _ = normalize(case["restored"], mean, std)
        np.savez_compressed(
            path, base=case["base"], values=case["values"], restored=case["restored"],
            labels=case["labels"], intervals=np.asarray(case["intervals"], dtype=np.int64),
            mean=mean, std=std, normalized=normalized, restored_normalized=restored_normalized,
        )
        manifest.append({"experiment": "A", "name": case["name"], "seed": case["seed"],
                         "layout": case["layout"], "target_kind": "",
                         "file": str(path.relative_to(directory)),
                         "sha256": sha256(path)})
    for case in b_cases:
        path = arrays_dir / f"{case['name']}.npz"
        np.savez_compressed(
            path, values=case["values"], labels=case["labels"],
            intervals=np.asarray(case["intervals"], dtype=np.int64).reshape(-1, 2),
            mean=np.asarray(case["mean"]), std=np.asarray(case["std"]),
        )
        manifest.append({"experiment": "B", "name": case["name"], "seed": case["seed"],
                         "layout": "", "target_kind": case["target_kind"],
                         "file": str(path.relative_to(directory)),
                         "sha256": sha256(path)})
    _write_rows(directory / "input_manifest.csv", manifest)
    return a_cases, b_cases


def _gate(detector, a_case, b_case, config):
    model = detector._tester.model
    device = detector._tester.device
    raw = a_case["values"][:, None]
    normalized, _, _ = normalize(a_case["values"])
    all_visible = np.ones(config["length"], dtype=bool)
    first = score_condition(model, normalized, all_visible, device)
    repeat = score_condition(model, normalized, all_visible, device)
    if not all(np.array_equal(first[key], repeat[key]) for key in ("logits", "scores")):
        raise RuntimeError("Repeated A gate inference was not bitwise identical")
    _, public_logits = detector.predict(raw, return_logits=True)
    public_logits = np.asarray(public_logits).reshape(-1)
    difference = float(np.max(np.abs(first["logits"] - public_logits)))
    if not np.allclose(first["logits"], public_logits, atol=config["atol"], rtol=config["rtol"]):
        raise RuntimeError(f"A gate differs from public inference: max_abs={difference}")

    normalized_b, _, _ = normalize(b_case["values"], b_case["mean"], b_case["std"])
    full_mask = context_mask(config["length"], "full", tuple(config["target"]))
    full = score_condition(model, normalized_b, full_mask, device)
    full_target = target_logits(full, full_mask, tuple(config["target"]))
    if full_target.shape != (32,) or not np.isfinite(full_target).all():
        raise RuntimeError("B full-context gate returned invalid target logits")
    _, public_b_logits = detector.predict(b_case["values"][:, None], return_logits=True)
    public_b_target = np.asarray(public_b_logits).reshape(-1)[slice(*config["target"])]
    b_difference = float(np.max(np.abs(full_target - public_b_target)))
    if not np.allclose(full_target, public_b_target, atol=config["atol"], rtol=config["rtol"]):
        raise RuntimeError(f"B full-context gate differs from public inference: max_abs={b_difference}")
    masked = context_mask(config["length"], 0, tuple(config["target"]))
    restricted = score_condition(model, normalized_b, masked, device)
    restricted_target = target_logits(restricted, masked, tuple(config["target"]))
    if restricted_target.shape != (32,) or not np.isfinite(restricted_target).all():
        raise RuntimeError("B restricted-context gate returned invalid target logits")
    return {
        "a_repeat_bitwise_equal": True,
        "a_public_max_abs_difference": difference,
        "b_public_max_abs_difference": b_difference,
        "b_target_points": int(len(full_target)),
        "b_restricted_visible_points": int(masked.sum()),
    }


def _run_a(detector, cases, config, directory):
    rows = []
    device = detector._tester.device
    model = detector._tester.model
    for case in cases:
        normalized, mean, std = normalize(case["values"])
        normalized_restored, mean_restored, std_restored = normalize(
            case["restored"], mean, std
        )
        if (mean, std) != (mean_restored, std_restored):
            raise RuntimeError("A conditions did not use shared normalization")
        valid = np.ones(config["length"], dtype=bool)
        original = score_condition(model, normalized, valid, device)
        restored = score_condition(model, normalized_restored, valid, device)
        delta = original["logits"] - restored["logits"]
        rings, _ = ring_masks(case["labels"], case["intervals"])
        for ring, mask in rings.items():
            if not mask.any():
                rows.append({
                    "name": case["name"], "seed": case["seed"], "layout": case["layout"],
                    "location": "center" if case["layout"].startswith("center") else "boundary",
                    "event_pattern": "isolated" if "isolated" in case["layout"] else "clustered",
                    "ring": ring, "point_count": 0, "original_logit_mean": "",
                    "restored_logit_mean": "", "delta_mean": "", "mean_absolute_delta": "",
                    "original_probability_mean": "", "restored_probability_mean": "",
                    "mean": mean, "std": std,
                })
                continue
            rows.append({
                "name": case["name"], "seed": case["seed"], "layout": case["layout"],
                "location": "center" if case["layout"].startswith("center") else "boundary",
                "event_pattern": "isolated" if "isolated" in case["layout"] else "clustered",
                "ring": ring, "point_count": int(mask.sum()),
                "original_logit_mean": float(original["logits"][mask].mean()),
                "restored_logit_mean": float(restored["logits"][mask].mean()),
                "delta_mean": float(delta[mask].mean()),
                "mean_absolute_delta": float(np.abs(delta[mask]).mean()),
                "original_probability_mean": float(original["scores"][mask].mean()),
                "restored_probability_mean": float(restored["scores"][mask].mean()),
                "mean": mean, "std": std,
            })
        np.savez_compressed(
            directory / "arrays" / f"scores_{case['name']}.npz",
            original_logits=original["logits"], restored_logits=restored["logits"],
            original_scores=original["scores"], restored_scores=restored["scores"],
        )
    _write_rows(directory / "a_sequence_rings.csv", rows)
    return rows


def _run_b(detector, cases, config, directory):
    radii = list(config["radii"])
    rows = []
    device = detector._tester.device
    model = detector._tester.model
    for case in cases:
        normalized, mean, std = normalize(case["values"], case["mean"], case["std"])
        for radius in radii:
            mask = context_mask(config["length"], radius, tuple(config["target"]), config["patch_size"])
            prediction = score_condition(model, normalized, mask, device)
            target = target_logits(prediction, mask, tuple(config["target"]))
            rows.append({
                "name": case["name"], "seed": case["seed"],
                "target_kind": case["target_kind"], "radius": radius,
                "visible_points": int(mask.sum()), "visible_patches": int(np.unique(np.flatnonzero(mask) // config["patch_size"]).size),
                "target_logit_mean": float(target.mean()), "target_logit_std": float(target.std()),
                "target_probability_mean": float(prediction["scores"][np.searchsorted(np.flatnonzero(mask), np.arange(*config["target"]))].mean()),
                "mean": mean, "std": std,
            })
            np.savez_compressed(
                directory / "arrays" / f"scores_{case['name']}_r{radius}.npz",
                point_mask=mask, target_logits=target,
                target_logit_mean=np.asarray(target.mean()),
            )
    _write_rows(directory / "b_target_scores.csv", rows)
    return rows


def run(config_path):
    config = _resolve_config(config_path)
    directory, manifest = create_run(config, config["config_source"])
    try:
        capture_provenance(directory, manifest)
        (directory / "protocol.md").write_text(Path(config["protocol_path"]).read_text())
        a_cases, b_cases = _save_inputs(directory, config)
        if len(a_cases) != 108 or len(b_cases) != 36:
            raise RuntimeError("Generated input counts differ from the frozen design")
        manifest["status"] = "inputs_saved"
        manifest["prepared_inputs"] = {"A": len(a_cases), "B": len(b_cases)}
        write_json(directory / "manifest.json", manifest)

        gpu = _gpu_gate()
        manifest["gpu_preflight"] = gpu
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        disable_tf32()
        detector = make_detector(config, config["window"])
        detector._ensure_tester(1)
        detector._tester.model.eval()
        manifest["status"] = "gate_running"
        write_json(directory / "manifest.json", manifest)
        before = _model_fingerprint(detector._tester.model)
        gate = _gate(detector, a_cases[0], next(c for c in b_cases if c["target_kind"] == "anomaly"), config)
        write_json(directory / "gate.json", gate)
        manifest["status"] = "running"
        manifest["model_sha256_before"] = before
        write_json(directory / "manifest.json", manifest)

        a_rows = _run_a(detector, a_cases, config, directory)
        b_rows = _run_b(detector, b_cases, config, directory)
        if len(a_cases) != 108 or len(b_cases) != 36 or len(b_rows) != 288:
            raise RuntimeError("Formal input/condition counts differ from the frozen design")
        after = _model_fingerprint(detector._tester.model)
        if before != after:
            raise RuntimeError("Model state changed during inference")
        if sha256(config["checkpoint"]) != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint file changed during inference")
        write_json(directory / "outputs.sha256.json", {
            str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and ("arrays" in path.relative_to(directory).parts
                                   or path.name in ("input_manifest.csv", "a_sequence_rings.csv", "b_target_scores.csv", "gate.json"))
        })
        manifest.update({
            "status": "completed", "model_sha256_after": after,
            "a_sequence_ring_rows": len(a_rows), "b_target_condition_rows": len(b_rows),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json(directory / "manifest.json", manifest)
        print(directory)
        return directory
    except Exception as error:
        manifest["status"] = "blocked_gpu_busy" if "not idle enough" in str(error) else "failed"
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        write_json(directory / "manifest.json", manifest)
        (directory / "error.txt").write_text(manifest["traceback"])
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
