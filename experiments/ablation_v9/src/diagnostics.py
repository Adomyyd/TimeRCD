from __future__ import annotations
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from .models import unwrap_model
from .losses import normal_reference_mask, reference_patch_size
import os
import threading


"""Optional observers only: no forward/backward, sampling, or optimizer updates.

All integration sites in train.py are marked OPTIONAL DIAGNOSTICS. Removing those
blocks restores the ordinary loop. Thresholds select evidence, never alter loss.
"""





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


@torch.no_grad()
def tensor_l2_values(named_tensors):
    """Reduce on the source device; copy only scalars, never full CUDA gradients."""
    result, devices = {}, {}
    for name, tensor in named_tensors:
        if tensor.device.type == "cpu":
            result[name] = gradient_l2(tensor)
        else:
            devices.setdefault(tensor.device, []).append(
                (name, torch.linalg.vector_norm(tensor.detach(), dtype=torch.float64)))
    for values in devices.values():
        norms = torch.stack([value for _, value in values]).cpu().tolist()
        result.update((name, value) for (name, _), value in zip(values, norms))
    return result


@torch.no_grad()
def compact_tensor_stats(tensor):
    """Exact full-tensor extrema and RMS without copying the full hidden state."""
    value = tensor.detach()
    if value.device.type == "cpu":
        stats = tensor_stats(value)
        stats["rms"] = gradient_l2(value) / math.sqrt(max(1, value.numel()))
        return stats
    low, high = torch.aminmax(value)
    norm = torch.linalg.vector_norm(value, dtype=torch.float64)
    low, high, norm = torch.stack((low.double(), high.double(), norm)).cpu().tolist()
    if not all(math.isfinite(x) for x in (low, high, norm)):
        # Rare failure path keeps the original finite-only statistics and count.
        stats = tensor_stats(value)
        stats["rms"] = None
        return stats
    return dict(shape=list(value.shape), nonfinite=0, min=low, max=high,
                abs_max=max(abs(low), abs(high)), rms=norm / math.sqrt(max(1, value.numel())))


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
        self.detail = None
        if self.options.get("detailed", False):
            self.detail = DetailedObserver(self)
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
            "diagnostic_schema_version": 3,
            "detail_interval": int(self.options.get("detail_interval", 100)),
            "reduction_backend": "source_device_float64_l2_cpu_scalar_transfer",
            "batch_index_origin": 0, "epoch_origin": 0,
            "snapshot_phase": "after backward, before gradient clipping and optimizer.step",
            "rng_phase": "before forward",
            "threshold_action": "save first event or bounded detailed escalations; stop on nonfinite",
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

    def attach(self, model):
        if self.detail is not None:
            self.detail.attach(model)

    def before_forward(self, batch_index=0):
        self.pre_forward_rng = rng_state()
        if self.detail is not None:
            self.detail.begin_forward(batch_index)

    def after_forward(self):
        if self.detail is not None:
            self.detail.end_forward()

    def after_step(self, model, optimizer, batch_index, optimization_record=None):
        self.optimization_record = optimization_record
        if self.detail is not None:
            self.detail.after_step(model, optimizer, batch_index)

    def close(self):
        if self.detail is not None:
            self.detail.close()

    @torch.no_grad()
    def after_backward(self, model, optimizer, batch_index, batch, x_input,
                       encoder_mask, mask, output, losses, dataset, *, optimization_record=None, raw_norms=None):
        loss_values = {k: float(losses[k].detach().cpu()) for k in
                       ("total", "reconstruction", "classification")}
        if "normal_reference" in output:
            loss_values["normal_reference"] = float(losses["normal_reference"].detach().cpu())
        grad_norms = raw_norms if raw_norms is not None else tensor_l2_values(
            (name, p.grad) for name, p in unwrap_model(model).named_parameters() if p.grad is not None)
        grad_norm = math.hypot(*grad_norms.values())
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
            "normalized_x0": tensor_stats(batch["x0"][batch["valid"]]),
            "masked_input": tensor_stats(x_input),
            "hidden": compact_tensor_stats(output["h"]),
            "prediction": tensor_stats(output["reconstruction"]),
            "logits": tensor_stats(output["logits"]),
            "masked_squared_error": tensor_stats(errors[mask]),
        }
        reference_mask = None
        if "normal_reference" in output:
            reference_mask = normal_reference_mask(batch, mask & batch["valid"], reference_patch_size(self.config))
            statistics["normal_reference_prediction"] = tensor_stats(output["normal_reference"])
            reference_errors = (output["normal_reference"].detach().cpu() - batch["x0"]).square()
            statistics["normal_reference_masked_squared_error"] = tensor_stats(reference_errors[reference_mask])
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
        hidden = statistics["hidden"]
        if self.detail is not None and eligible and hidden["rms"] is not None and hidden["rms"] >= 1000:
            reasons.append("hidden_scale")
        detail_record = {}
        if self.detail is not None:
            reasons.extend(r for r in self.detail.activation_triggers()
                           if (eligible or r == "nonfinite") and r not in reasons)
            nonfinite = nonfinite or "nonfinite" in reasons
            candidate = dict(loss=loss_values, gradient_l2=grad_norm,
                             hidden_moments=hidden, triggers=reasons)
            detailed = (self.detail.scheduled(batch_index)
                        or self.detail.wants_event(candidate) or nonfinite)
            detail_record = self.detail.observe(
                model, optimizer, batch, mask, output, grad_norms, reference_mask=reference_mask, detailed=detailed)
            reasons.extend(r for r in self.detail.extra_triggers
                           if (eligible or r == "nonfinite") and r not in reasons)
            nonfinite = nonfinite or "nonfinite" in reasons
        record = {
            **detail_record,
            "optimization_control": optimization_record,
            "epoch": self.epoch, "batch": batch_index, "elapsed_seconds": time.time() - self.started,
            "loss": loss_values, "running_mean_total": self.total / (batch_index + 1),
            "sample_ids": ids, "lengths": batch["lengths"],
            "per_sample_masked_mse": per_sample_reconstruction,
            "raw_mean": raw.mean(dim=0).tolist(), "raw_std_plus_eps": (raw.std(dim=0) + 1e-4).tolist(),
            "mask_seed": int(self.config["runtime"]["seed"]) + self.epoch * 1000003 + batch_index,
            "normal_reference_count": losses.get("normal_reference_count", 0),
            "mask_count": int(mask.sum()), "classification_count": int(losses["classification_count"]),
            "valid_points": int(batch["valid"].sum()), "padding_points": int((~batch["valid"]).sum()),
            "tensor_stats": statistics, "gradient_l2": grad_norm,
            "hidden_moments": detail_record.get("hidden_moments", hidden),
            "largest_gradient_parameter": largest,
            "largest_gradient_l2": grad_norms.get(largest), "triggers": reasons,
        }
        with (self.directory / "batches.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        if self.detail is not None:
            self.detail.capture(model, optimizer, record, batch, x_input, encoder_mask, mask, output)
        elif reasons and (not self.event_saved or nonfinite):
            payload = {
                "phase": "pre_optimizer_step", "epoch": self.epoch, "batch_index": batch_index,
                "model_state_dict": cpu_copy(unwrap_model(model).state_dict()),
                "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
                "rng_before_forward": self.pre_forward_rng,
                "batch": cpu_copy(batch), "sample_ids": ids,
                "x_input": cpu_copy(x_input), "encoder_mask": cpu_copy(encoder_mask), "mask": cpu_copy(mask),
                "losses": loss_values, "gradient_norms": grad_norms, "config": self.config,
            }
            event_name = "nonfinite_event" if self.event_saved and nonfinite else "first_event"
            torch.save(payload, self.directory / (event_name + ".pth"))
            self.write_json(event_name + ".json", record)
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


"""Periodic CPU detail plus device scalar sentinels; never changes training tensors."""


def array_stats(value):
    # Float64 accumulation avoids diagnostic overflow on large finite FP32 values.
    a = value.detach().cpu().numpy()
    finite = np.isfinite(a)
    good = a[finite] if not finite.all() else a.reshape(-1)
    if not good.size:
        return dict(count=int(a.size), nonfinite=int(a.size), mean=None, rms=None,
                    std=None, abs_max=None, zero_fraction=None)
    square = np.square(good, dtype=np.float64).sum()
    mean = float(np.mean(good, dtype=np.float64))
    rms = math.sqrt(float(square / good.size))
    return dict(count=int(a.size), nonfinite=int(a.size - finite.sum()), mean=mean,
                rms=rms, std=math.sqrt(max(0., rms*rms-mean*mean)),
                abs_max=float(np.abs(good).max()),
                zero_fraction=float(np.count_nonzero(good == 0)/good.size))


def digest_tensor(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def group_name(name):
    if name.startswith("encoder.transformer_encoder.layers."):
        return ".".join(name.split(".")[:4])
    return ".".join(name.split(".")[:2]) if name.startswith("encoder.") else name.split(".")[0]


def state_groups(model, optimizer):
    groups = {}
    for name, p in unwrap_model(model).named_parameters():
        g = groups.setdefault(group_name(name), dict(parameter_sq=0., moment_sq=0.,
                            second_moment_sum=0., nonfinite=0, elements=0))
        a = p.detach().cpu().numpy()
        g["elements"] += a.size
        g["nonfinite"] += int((~np.isfinite(a)).sum())
        g["parameter_sq"] += float(np.square(a, dtype=np.float64).sum())
        state = optimizer.state.get(p, {})
        if "exp_avg" in state:
            m = state["exp_avg"].detach().cpu().numpy()
            v = state["exp_avg_sq"].detach().cpu().numpy()
            g["nonfinite"] += int((~np.isfinite(m)).sum() + (~np.isfinite(v)).sum())
            g["moment_sq"] += float(np.square(m, dtype=np.float64).sum())
            g["second_moment_sum"] += float(np.sum(v, dtype=np.float64))
    return groups


class DetailedObserver:
    def __init__(self, owner):
        self.owner = owner
        self.directory = owner.directory
        self.interval = int(owner.options.get("snapshot_interval", 0))
        self.detail_interval = int(owner.options.get("detail_interval", 100))
        self.sampled = False
        self.max_events = int(owner.options.get("max_events", 6))
        self.handles = []
        self.lock = threading.Lock()
        self.active = False
        self.activations = {}
        self.events = 0
        self.seen_levels = {}
        self.latest_periodic = None
        self.pending = None
        self.before_parameters = None
        self.extra_triggers = []

    def attach(self, model):
        for name, module in unwrap_model(model).named_modules():
            is_layer = name.startswith("encoder.transformer_encoder.layers.") and name.count(".") == 3
            is_head = name.startswith(("reconstruction_head.", "normal_reference_head.", "anomaly_head."))
            if is_layer or name == "encoder.projection_layer" or (is_head and isinstance(module, (torch.nn.Linear, torch.nn.GELU))):
                def observe(module, args, output, name=name):
                    if not self.active or (not self.sampled and name != "anomaly_head.1"):
                        return
                    # Views only on CUDA; summaries are on CPU, no random sampling.
                    # Up to 8 fixed time positions, ALL rows/channels; not full extrema.
                    step = max(1, math.ceil(output.shape[1] / 8))
                    sample = output.detach()[:, ::step].cpu()
                    s = array_stats(sample)
                    a = sample.numpy()
                    s.update(full_shape=list(output.shape), sample_shape=list(sample.shape),
                             time_stride=step, below_minus5=float(np.mean(a < -5)),
                             near_zero=float(np.mean(np.abs(a) < 1e-8)))
                    with self.lock:
                        self.activations[name + "@" + str(output.device)] = s
                self.handles.append(module.register_forward_hook(observe))

    def scheduled(self, batch_index):
        return (batch_index % self.detail_interval == 0
                or (self.interval > 0 and batch_index % self.interval == 0))

    def begin_forward(self, batch_index=0):
        self.activations = {}
        self.sampled = self.scheduled(batch_index)
        self.active = True

    def activation_triggers(self):
        reasons = []
        if any(k.startswith("anomaly_head.1@") and v["near_zero"] == 1
               for k, v in self.activations.items()):
            reasons.append("classifier_saturated")
        if any(v["nonfinite"] for v in self.activations.values()):
            reasons.append("nonfinite")
        return reasons

    def end_forward(self):
        # Disable BEFORE backward; activation-checkpoint recomputation is not sampled.
        self.active = False

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def observe(self, model, optimizer, batch, mask, output, grad_norms, *, reference_mask=None, detailed=True):
        self.extra_triggers = []
        groups = {}
        for name, norm in grad_norms.items():
            g = groups.setdefault(group_name(name), dict(l2_squared=0., max_parameter_l2=0.))
            g["l2_squared"] += norm*norm
            g["max_parameter_l2"] = max(g["max_parameter_l2"], norm)
        for g in groups.values():
            g["l2"] = math.sqrt(g.pop("l2_squared"))
        common = dict(observation_level="detailed" if detailed else "light",
                      activations_complete_for_schedule=self.sampled,
                      gradient_by_parameter=grad_norms, gradient_by_module=groups,
                      sampled_activations=dict(self.activations))
        if not detailed:
            return common
        self.before_parameters = {n: cpu_copy(p) for n, p in unwrap_model(model).named_parameters()}
        self.before_state = state_groups(model, optimizer)
        active = mask & batch["valid"]
        x = batch["x"].detach().cpu()
        x0 = batch["x0"].detach().cpu()
        pred = output["reconstruction"].detach().cpu()
        normal_pred = output.get("normal_reference")
        if normal_pred is not None:
            normal_pred = normal_pred.detach().cpu()
        regions = {"selected": active, "selected_normal": active & (batch["y"] == 0),
                   "selected_anomalous": active & (batch["y"] > 0),
                   "selected_changed": active & (x != x0).any(-1)}
        stats = {}
        for name, region in regions.items():
            stats[name] = dict(points=int(region.sum()), target_x=array_stats(x[region]),
                target_x0=array_stats(x0[region]), prediction_x=array_stats(pred[region]),
                mse_x=float((pred[region]-x[region]).square().mean()) if region.any() else None,
                zero_mse_x=float(x[region].square().mean()) if region.any() else None,
                zero_mse_x0=float(x0[region].square().mean()) if region.any() else None)
            if normal_pred is not None:
                stats[name].update(prediction_x0=array_stats(normal_pred[region]),
                    mse_x0=float((normal_pred[region]-x0[region]).square().mean()) if region.any() else None)
        logits = output["logits"].detach().cpu().numpy()
        shifted = logits - logits.max(axis=-1, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            exp = np.exp(shifted)
            probability = exp[..., 1] / exp.sum(-1)
        valid = batch["valid"].numpy()
        y = batch["y"].numpy() > 0
        predicted = probability >= 0.5
        cls = dict(tp=int((predicted & y & valid).sum()),
                   fp=int((predicted & ~y & valid).sum()),
                   fn=int((~predicted & y & valid).sum()),
                   valid=int(valid.sum()), positives=int((y & valid).sum()),
                   probability=array_stats(torch.from_numpy(probability[valid])))
        hidden = array_stats(output["h"])
        self.extra_triggers = []
        if hidden["rms"] is not None and hidden["rms"] >= 1000:
            self.extra_triggers.append("hidden_scale")
        if any(k.startswith("anomaly_head.1@") and v["near_zero"] == 1 for k,v in self.activations.items()):
            self.extra_triggers.append("classifier_saturated")
        if any(g["nonfinite"] for g in self.before_state.values()) or any(v["nonfinite"] for v in self.activations.values()):
            self.extra_triggers.append("nonfinite")
        reference_errors = (normal_pred-x0).square() if normal_pred is not None else None
        return dict(**common, hidden_moments=hidden, regions=stats, classification=cls,
                    per_sample_normal_reference_mse=[
                        float(reference_errors[i][reference_mask[i]].mean()) if reference_mask[i].any() else None
                        for i in range(len(x))] if reference_errors is not None else None,
                    state_before=self.before_state,
                    batch_sha256={k:digest_tensor(batch[k]) for k in ("x","x0","y","valid")},
                    mask_sha256=digest_tensor(mask),
                    observation_phase="forward and backward before optimizer.step")

    def save(self, name, payload):
        path = self.directory / (name + ".pth")
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def event_levels(self, record):
        levels = {}
        for reason in record["triggers"]:
            ratio = (record["loss"]["total"] / self.owner.options.get("loss_threshold", 10)
                     if reason == "loss_threshold" else
                     record["gradient_l2"] / self.owner.options.get("grad_threshold", 1000)
                     if reason == "grad_threshold" else
                     record["hidden_moments"]["rms"] / 1000 if reason == "hidden_scale" else 1)
            levels[reason] = int(math.log10(max(1, ratio))) if math.isfinite(ratio) else 99
        return levels

    def wants_event(self, record):
        levels = self.event_levels(record)
        # Every nonfinite occurrence gets evidence, even after a prior nonfinite event.
        return ("nonfinite" in levels or
                (bool(levels) and self.events < self.max_events and any(
                    level > self.seen_levels.get(reason, -1) for reason, level in levels.items())))

    def capture(self, model, optimizer, record, batch, x_input, encoder_mask, mask, output):
        owner = self.owner
        batch_index = record["batch"]
        reasons = record["triggers"]
        levels = self.event_levels(record)
        event = self.wants_event(record)
        periodic = self.interval > 0 and batch_index % self.interval == 0
        if not event and not periodic:
            return
        payload = dict(phase="pre_optimizer_step", epoch=owner.epoch, batch_index=batch_index,
            model_state_dict=cpu_copy(unwrap_model(model).state_dict()),
            optimizer_state_dict=cpu_copy(optimizer.state_dict()), rng_before_forward=owner.pre_forward_rng,
            rng_after_backward=rng_state(),
            gradients={n:cpu_copy(p.grad) if p.grad is not None else None for n,p in unwrap_model(model).named_parameters()},
            batch=cpu_copy(batch), x_input=cpu_copy(x_input), encoder_mask=cpu_copy(encoder_mask),
            mask=cpu_copy(mask), sample_ids=record["sample_ids"], losses=record["loss"],
            gradient_norms=record["gradient_by_parameter"], config=owner.config,
            diagnostic_record=record, output=cpu_copy(output), model_training=bool(model.training))
        if event:
            name = "first_event" if self.events == 0 else f"event_{self.events:03d}"
            self.save(name, payload)
            owner.write_json(name + ".json", record)
            # Hard link pins the old inode while rolling slots are atomically replaced.
            if self.latest_periodic is not None:
                previous = self.directory / (name + "_previous.pth")
                os.link(self.latest_periodic, previous)
            self.pending = (name, record)
            for reason, level in levels.items():
                self.seen_levels[reason] = max(level, self.seen_levels.get(reason, -1))
            self.events += 1
            owner.event_saved = True
            print(f"[diagnostics] {name}: epoch={owner.epoch} batch={batch_index} triggers={reasons}", flush=True)
        if periodic:
            slot = (batch_index // self.interval) % 2
            self.latest_periodic = self.save(f"rolling_{slot}", payload)
            owner.write_json(f"rolling_{slot}.json", dict(epoch=owner.epoch, batch=batch_index,
                             phase="pre_optimizer_step"))
        del payload

    @torch.no_grad()
    def after_step(self, model, optimizer, batch_index):
        if self.before_parameters is None:
            # Check every update for nonfinite parameters/Adam without host copies.
            tensors = []
            for n, p in unwrap_model(model).named_parameters():
                tensors.append((n, p))
                for key in ("exp_avg", "exp_avg_sq"):
                    value = optimizer.state.get(p, {}).get(key)
                    if value is not None:
                        tensors.append((n + "." + key, value))
            norms = tensor_l2_values(tensors)
            bad = [n for n, value in norms.items() if not math.isfinite(value)]
            record = dict(epoch=self.owner.epoch, batch=batch_index,
                          observation_level="light", phase="after_optimizer_step",
                          nonfinite_state=bad, optimization_control=getattr(self.owner, "optimization_record", None))
            if bad:
                self.save("nonfinite_update_post", dict(
                    phase="after_optimizer_step", epoch=self.owner.epoch, batch_index=batch_index,
                    model_state_dict=cpu_copy(unwrap_model(model).state_dict()),
                    optimizer_state_dict=cpu_copy(optimizer.state_dict()), rng_after_step=rng_state(),
                    config=self.owner.config, diagnostic_record=record))
                self.owner.write_json("nonfinite_update_post.json", record)
            with (self.directory / "updates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
            if bad:
                raise FloatingPointError("Nonfinite optimizer update; post-step evidence saved")
            return
        groups = {}
        for n,p in unwrap_model(model).named_parameters():
            a = p.detach().cpu().numpy()
            before = self.before_parameters[n].numpy()
            delta = a.astype(np.float64)-before
            g = groups.setdefault(group_name(n), dict(delta_sq=0., before_sq=0., abs_max=0.))
            g["delta_sq"] += float(np.square(delta).sum())
            g["before_sq"] += float(np.square(before, dtype=np.float64).sum())
            g["abs_max"] = max(g["abs_max"],float(np.abs(delta).max()))
        for g in groups.values():
            g["l2"] = math.sqrt(g.pop("delta_sq"))
            g["relative_l2"] = g["l2"]/max(1e-30,math.sqrt(g.pop("before_sq")))
        after = state_groups(model, optimizer)
        record = dict(epoch=self.owner.epoch,batch=batch_index,update_by_module=groups,
                      state_after=after,phase="after_optimizer_step",observation_level="detailed",
                      optimization_control=getattr(self.owner, "optimization_record", None))
        nonfinite = any(g["nonfinite"] for g in after.values())
        if self.pending or nonfinite:
            name = self.pending[0] if self.pending else "nonfinite_update"
            self.save(name + "_post",dict(phase="after_optimizer_step",epoch=self.owner.epoch,
                batch_index=batch_index,model_state_dict=cpu_copy(unwrap_model(model).state_dict()),
                optimizer_state_dict=cpu_copy(optimizer.state_dict()),rng_after_step=rng_state(),
                config=self.owner.config,diagnostic_record=record))
            self.owner.write_json(name + "_post.json",record)
        with (self.directory/"updates.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record),allow_nan=False)+"\n")
        self.pending = None
        self.before_parameters = None
        if nonfinite:
            raise FloatingPointError("Nonfinite optimizer update; post-step evidence saved")
