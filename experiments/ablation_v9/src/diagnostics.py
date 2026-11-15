"""Optional observers only: no forward/backward, sampling, or optimizer updates.

All integration sites in train.py are marked OPTIONAL DIAGNOSTICS. Removing those
blocks restores the ordinary loop. Thresholds select evidence, never alter loss.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .parallel import unwrap_model


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def compare_states(actual, expected, prefix=""):
    """Exact recursive comparison; includes Adam moments and step counters."""
    mismatches = []
    def compare(a, b, key):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
                mismatches.append(key)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                mismatches.append(key + ".keys")
            for k in a.keys() & b.keys():
                compare(a[k], b[k], f"{key}.{k}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                mismatches.append(key + ".length")
            for i, (av, bv) in enumerate(zip(a, b)):
                compare(av, bv, f"{key}.{i}")
        elif a != b:
            mismatches.append(key)
    compare(actual, expected, prefix)
    return mismatches


@torch.no_grad()
def tensor_stats(tensor):
    # CPU NumPy reductions avoid diagnostic CUDA allocations and Torch thread-pool overhead.
    value = tensor.detach().cpu().numpy()
    finite = np.isfinite(value)
    good = value if finite.all() else value[finite]
    return {
        "shape": list(value.shape), "nonfinite": int(value.size - finite.sum()),
        "min": float(good.min()) if good.size else None,
        "max": float(good.max()) if good.size else None,
        "abs_max": float(np.abs(good).max()) if good.size else None,
    }


def gradient_l2(tensor):
    value = tensor.detach().cpu().numpy()
    return float(np.sqrt(np.square(value, dtype=np.float64).sum()))


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class BatchDiagnostics:
    def __init__(self, run_dir, config):
        self.config = config
        self.options = config["diagnostics"]
        self.directory = Path(run_dir) / "diagnostics"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.event_saved = False
        self.started = time.time()
        self.pre_forward_rng = None
        self.sample_batches = []
        root = Path(__file__).resolve().parents[3]
        sources = {}
        for directory in (root / "experiments/ablation_v9/src", root / "time_rcd"):
            if directory.exists():
                for path in sorted(directory.rglob("*.py")):
                    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = root / "experiments/ablation_v9/run_experiment.py"
        sources[str(entry.relative_to(root))] = hashlib.sha256(entry.read_bytes()).hexdigest()
        self.write_json("settings.json", {
            "options": self.options, "source_sha256": sources,
            "batch_index_origin": 0, "epoch_origin": 0,
            "snapshot_phase": "after backward, before optimizer.step",
            "rng_phase": "before forward",
            "threshold_action": "save first event; stop only on nonfinite",
        })

    def write_json(self, name, value):
        (self.directory / name).write_text(
            json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def begin_epoch(self, epoch, sampler, train_indices):
        self.epoch = epoch
        self.total = 0.0
        if sampler is None:
            size = int(self.config["optimization"]["batch_size"])
            self.sample_batches = [train_indices[i:i + size] for i in range(0, len(train_indices), size)]
        else:
            # Independent iterator uses the sampler's LOCAL random.Random only.
            self.sample_batches = [[train_indices[i] for i in positions] for positions in sampler]

    def before_forward(self):
        self.pre_forward_rng = rng_state()

    @torch.no_grad()
    def after_backward(self, model, optimizer, batch_index, batch, x_input,
                       encoder_mask, mask, output, losses, dataset):
        loss_values = {k: float(losses[k].detach().cpu()) for k in
                       ("total", "reconstruction", "classification")}
        if "normal_reference" in output:
            loss_values["normal_reference"] = float(losses["normal_reference"].detach().cpu())
        grad_norms = {
            name: gradient_l2(p.grad)
            for name, p in unwrap_model(model).named_parameters() if p.grad is not None
        }
        grad_norm = math.sqrt(sum(value * value for value in grad_norms.values()))
        largest = max(grad_norms, key=grad_norms.get, default=None)
        ids = self.sample_batches[batch_index]
        raw = torch.cat([torch.as_tensor(dataset.data[i]["time_series"], dtype=torch.float32).reshape(
            -1, batch["x"].shape[-1]) for i in ids], dim=0)
        prediction = output["reconstruction"].detach().cpu()
        target = batch["x"]
        errors = (prediction - target).square()
        per_sample_reconstruction = [
            float(errors[i][mask[i]].mean()) if mask[i].any() else None
            for i in range(len(ids))
        ]
        self.total += loss_values["total"]
        statistics = {
            "raw_x": tensor_stats(raw),
            "normalized_x": tensor_stats(batch["x"][batch["valid"]]),
            "masked_input": tensor_stats(x_input),
            "hidden": tensor_stats(output["h"]),
            "prediction": tensor_stats(output["reconstruction"]),
            "logits": tensor_stats(output["logits"]),
            "masked_squared_error": tensor_stats(errors[mask]),
        }
        if "normal_reference" in output:
            statistics["normal_reference_prediction"] = tensor_stats(output["normal_reference"])
            reference_errors = (output["normal_reference"].detach().cpu() - batch["x0"]).square()
            statistics["normal_reference_masked_squared_error"] = tensor_stats(reference_errors[mask])
        nonfinite = (not all(math.isfinite(v) for v in loss_values.values())
                     or not math.isfinite(grad_norm)
                     or any(s["nonfinite"] for s in statistics.values()))
        eligible = self.epoch >= int(self.options.get("start_epoch", 1))
        reasons = []
        if nonfinite:
            reasons.append("nonfinite")
        if eligible and loss_values["total"] >= float(self.options.get("loss_threshold", 10)):
            reasons.append("loss_threshold")
        if eligible and grad_norm >= float(self.options.get("grad_threshold", 1000)):
            reasons.append("grad_threshold")
        record = {
            "epoch": self.epoch, "batch": batch_index, "elapsed_seconds": time.time() - self.started,
            "loss": loss_values, "running_mean_total": self.total / (batch_index + 1),
            "sample_ids": ids, "lengths": batch["lengths"],
            "per_sample_masked_mse": per_sample_reconstruction,
            "raw_mean": raw.mean(dim=0).tolist(), "raw_std_plus_eps": (raw.std(dim=0) + 1e-4).tolist(),
            "mask_count": int(mask.sum()), "classification_count": int(losses["classification_count"]),
            "valid_points": int(batch["valid"].sum()), "padding_points": int((~batch["valid"]).sum()),
            "tensor_stats": statistics, "gradient_l2": grad_norm,
            "largest_gradient_parameter": largest,
            "largest_gradient_l2": grad_norms.get(largest), "triggers": reasons,
        }
        with (self.directory / "batches.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        if reasons and not self.event_saved:
            payload = {
                "phase": "pre_optimizer_step", "epoch": self.epoch, "batch_index": batch_index,
                "model_state_dict": cpu_copy(unwrap_model(model).state_dict()),
                "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
                "rng_before_forward": self.pre_forward_rng,
                "batch": cpu_copy(batch), "sample_ids": ids,
                "x_input": cpu_copy(x_input), "encoder_mask": cpu_copy(encoder_mask), "mask": cpu_copy(mask),
                "losses": loss_values, "gradient_norms": grad_norms, "config": self.config,
            }
            torch.save(payload, self.directory / "first_event.pth")
            self.write_json("first_event.json", record)
            self.event_saved = True
            print(f"[diagnostics] saved first event: epoch={self.epoch} batch={batch_index} "
                  f"loss={loss_values['total']:.8g} grad={grad_norm:.8g} triggers={reasons}", flush=True)
        if nonfinite:
            raise FloatingPointError("Nonfinite observed; diagnostic evidence saved before optimizer.step")

    def end_epoch(self, epoch, checkpoint_path):
        # Includes validation; ordinary checkpoint serialization does not consume RNG.
        torch.save(rng_state(), self.directory / f"rng_after_epoch_{epoch:04d}.pth")
        reference = self.options.get("reference_epoch0")
        if epoch != 0 or not reference:
            return
        actual = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = torch.load(reference, map_location="cpu", weights_only=False)
        mismatches = []
        for key in ("model_state_dict", "optimizer_state_dict"):
            mismatches.extend(compare_states(actual[key], expected[key], key))
        report = {"epoch": 0, "exact_match": not mismatches,
                  "mismatch_count": len(mismatches), "mismatches": mismatches,
                  "reference_epoch": expected["epoch"],
                  "trajectory_label": "epoch0_exact_match" if not mismatches else "same_config_rerun_not_exact_replay",
                  "stop_on_mismatch": self.options.get("stop_on_mismatch", False),
                  "actual_validation": actual["validation"], "reference_validation": expected["validation"]}
        self.write_json("epoch0_comparison.json", report)
        print(f"[diagnostics] epoch0 exact model/optimizer match: {not mismatches}; "
              f"stop_on_mismatch={self.options.get('stop_on_mismatch', False)}", flush=True)
        if mismatches and self.options.get("stop_on_mismatch", False):
            raise RuntimeError("epoch0 differs from reference; stopped before epoch1. "
                               "See diagnostics/epoch0_comparison.json")
