"""Periodic CPU detail plus device scalar sentinels; never changes training tensors."""
from __future__ import annotations
import hashlib
import math
import os
import threading
from pathlib import Path
import numpy as np
import torch
from .parallel import unwrap_model
from .diagnostics import cpu_copy, rng_state, json_safe, tensor_l2_values
import json


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

    def observe(self, model, optimizer, batch, mask, output, grad_norms, *, detailed=True):
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
                        float(reference_errors[i][active[i]].mean()) if active[i].any() else None
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
                          nonfinite_state=bad)
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
                      state_after=after,phase="after_optimizer_step",observation_level="detailed")
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
