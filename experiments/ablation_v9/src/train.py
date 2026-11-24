from __future__ import annotations
import json
from .config import validate_optimization
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import (
    LengthBucketBatchSampler,
    PairedTimeSeriesDataset,
    collate_samples,
    load_indices,
    sample_lengths,
)
from .evaluate import evaluate_model
from .losses import compute_losses, reference_patch_size, NORMAL_REFERENCE_VARIANTS
from .models import parallel_record, resolve_device, unwrap_model
from .data import apply_input_mask, sample_patch_mask
import copy
import hashlib
import math
import re
import shutil
from .models import unwrap_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(config: dict) -> torch.device:
    return resolve_device(config)


def make_batch_inputs(batch, config: dict, generator: torch.Generator):
    mask = sample_patch_mask(
        batch["valid"],
        int(config["input"]["mask_patch_size"]),
        float(config["input"]["mask_ratio"]),
        generator,
    )
    x_input = apply_input_mask(batch["x"], mask)
    encoder_mask = batch["valid"] & ~mask
    return x_input, encoder_mask, mask


def run_smoke(model, config: dict) -> dict:
    device = next(model.parameters()).device
    gpu_count = len(model.device_ids) if isinstance(model, torch.nn.DataParallel) else 1
    batch_size = min(max(2, gpu_count), int(config["optimization"]["batch_size"]))
    length = 320
    batch = {
        "x": torch.randn(batch_size, length, 1),
        "x0": torch.randn(batch_size, length, 1),
        "y": torch.zeros(batch_size, length, dtype=torch.long),
        "valid": torch.ones(batch_size, length, dtype=torch.bool),
    }
    batch["y"][:, 120:136] = 1
    generator = torch.Generator(device="cpu").manual_seed(int(config["runtime"]["seed"]))
    x_input, encoder_mask, mask = make_batch_inputs(batch, config, generator)
    output = model(x_input.to(device), encoder_mask.to(device))
    batch_device = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    losses = compute_losses(
        output,
        batch_device,
        config["input"]["target"],
        float(config["loss"]["reconstruction_weight"]),
        float(config["loss"]["classification_weight"]),
        mask.to(device),
        classification_padding=config["loss"].get("classification_padding", "ignore"),
        normal_reference_weight=float(config["loss"].get("normal_reference_weight", 0)),
        normal_reference_patch_size=reference_patch_size(config),
    )
    losses["total"].backward()
    return {
        "status": "passed",
        "losses": {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in losses.items()
        },
        "mask_count": int(mask.sum().item()),
        "classification_padding": config["loss"].get("classification_padding", "ignore"),
        "encoder_patch_size": unwrap_model(model).encoder.patch_size,
        **parallel_record(model, config),
        "mask_patch_size": int(config["input"]["mask_patch_size"]),
        "mask_ratio": float(config["input"]["mask_ratio"]),
    }


def _build_train_loader(config: dict, train_indices: list[int]):
    dataset = PairedTimeSeriesDataset(config["data"]["path"], train_indices)
    kind = str(config["optimization"].get("batch_sampler", "sequential"))
    if kind not in {"sequential", "length_bucket"}:
        raise ValueError("optimization.batch_sampler must be 'sequential' or 'length_bucket'")
    common = {
        "num_workers": int(config["optimization"]["num_workers"]),
        "collate_fn": collate_samples,
    }
    if kind == "length_bucket":
        lengths = sample_lengths(config["data"]["path"])
        # The dataset indexes its own position list, so the sampler must return
        # positions 0..N-1 rather than dataset sample ids.
        positions = list(range(len(train_indices)))
        sampler = LengthBucketBatchSampler(
            positions,
            [lengths[train_indices[position]] for position in positions],
            int(config["optimization"]["batch_size"]),
            int(config["runtime"]["seed"]),
            int(config["optimization"].get("bucket_multiplier", 50)),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common), kind, sampler
    loader = DataLoader(
        dataset,
        batch_size=int(config["optimization"]["batch_size"]),
        shuffle=False,
        **common,
    )
    return loader, kind, None


def _normal_reference_log(label: str, value: float, enabled: bool) -> str:
    return f"  {label}: {value:.4f}\n" if enabled else ""


def train_model(model, config: dict, run_dir: Path, *, resume_from: Path | None = None,
                checkpoint_keep_last: int | None = None) -> dict:
    set_seed(int(config["runtime"]["seed"]))
    device = _device(config)
    data_start = time.time()
    train_indices, test_indices = load_indices(
        config["data"]["path"],
        float(config["split"]["train_ratio"]),
        int(config["split"]["seed"]),
    )
    data_seconds = time.time() - data_start
    train_loader, batch_sampler_kind, batch_sampler = _build_train_loader(config, train_indices)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    control = OptimizationControl(optimizer, config["optimization"], len(train_loader))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_freq = int(config["optimization"]["log_freq"])
    has_normal_reference = config["experiment_id"] in NORMAL_REFERENCE_VARIANTS
    normal_weight_log = (
        f"normal_reference={config['loss']['normal_reference_weight']}, "
        if has_normal_reference else ""
    )
    patch_log = (
        f"Parallel settings: {parallel_record(model, config)}\n"
        f"Loss settings: classification_padding={config['loss'].get('classification_padding', 'ignore')}, "
        f"checkpoint_selection=valid_total\n"
        f"Loss weights: observation={config['loss']['reconstruction_weight']}, "
        f"{normal_weight_log}"
        f"classification={config['loss']['classification_weight']}\n"
        f"Patch settings: encoder_patch_size={unwrap_model(model).encoder.patch_size}, "
        f"mask_patch_size={int(config['input']['mask_patch_size'])}, "
        f"mask_ratio={float(config['input']['mask_ratio'])}, "
        f"input_mode={config['input']['input_mode']}\n"
        f"Sampler settings: batch_sampler={batch_sampler_kind}, "
        f"batch_size={int(config['optimization']['batch_size'])}, "
        f"bucket_multiplier={int(config['optimization'].get('bucket_multiplier', 50))}\n"
        f"Data ready: train_samples={len(train_indices)}, val_samples={len(test_indices)}, "
        f"batches_per_epoch={len(train_loader)}, load_seconds={data_seconds:.1f}\n"
        f"Optimization control: {config['optimization']}\n"
        f"Start: local={datetime.now().isoformat(timespec='seconds')}\n"
    )
    print(patch_log, end="", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(patch_log)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    run_start = time.time()

    # OPTIONAL DIAGNOSTICS: remove this block and the other marked blocks freely.
    diagnostics = None
    if config.get("diagnostics", {}).get("enabled", False):
        from .diagnostics import BatchDiagnostics
        diagnostics = BatchDiagnostics(run_dir, config)
        diagnostics.attach(model)
    # END OPTIONAL DIAGNOSTICS

    start_epoch = 0
    if resume_from is not None:
        restored = restore_training(resume_from, model, optimizer, config, run_dir)
        start_epoch = int(restored["next_epoch"])
        if control.enabled:
            control.verify_resume(restored.get("optimization_control"), start_epoch)
        best_loss = float(restored["best_loss"])
        best_epoch = int(restored["best_epoch"])
        stale = int(restored["stale"])
        # Restore AFTER model/optimizer/loader/diagnostic setup, immediately before iteration.
        restore_rng(restored["rng"])
        del restored
        message = f"Resumed: next_epoch={start_epoch} best_epoch={best_epoch} stale={stale}\n"
        print(message, end="", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)

    for epoch in range(start_epoch, int(config["optimization"]["epochs"])):
        model.train()
        epoch_start = time.time()
        last_log_time = epoch_start
        last_log_batch = 0
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        # OPTIONAL DIAGNOSTICS
        if diagnostics is not None:
            diagnostics.begin_epoch(epoch, batch_sampler, train_indices)
        # END OPTIONAL DIAGNOSTICS
        total = 0.0
        total_reconstruction = 0.0
        total_normal_reference = 0.0
        total_classification = 0.0
        batches = 0
        padded_points = 0
        valid_points = 0
        for batch_index, batch in enumerate(train_loader):
            control.begin_step(epoch * len(train_loader) + batch_index)
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device="cpu").manual_seed(
                int(config["runtime"]["seed"]) + epoch * 1000003 + batch_index
            )
            x_input, encoder_mask, mask = make_batch_inputs(batch, config, generator)
            batch_device = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            # OPTIONAL DIAGNOSTICS
            if diagnostics is not None:
                diagnostics.before_forward(batch_index)
            # END OPTIONAL DIAGNOSTICS
            output = model(x_input.to(device), encoder_mask.to(device))
            # OPTIONAL DIAGNOSTICS: exclude checkpoint recomputation from hook samples.
            if diagnostics is not None:
                diagnostics.after_forward()
            # END OPTIONAL DIAGNOSTICS
            losses = compute_losses(
                output,
                batch_device,
                config["input"]["target"],
                float(config["loss"]["reconstruction_weight"]),
                float(config["loss"]["classification_weight"]),
                mask.to(device),
                classification_padding=config["loss"].get("classification_padding", "ignore"),
                normal_reference_weight=float(config["loss"].get("normal_reference_weight", 0)),
                normal_reference_patch_size=reference_patch_size(config),
            )
            losses["total"].backward()
            optimization_record, raw_norms = control.inspect(unwrap_model(model), losses["total"])
            # OPTIONAL DIAGNOSTICS: observe gradients and snapshot BEFORE the update.
            if diagnostics is not None:
                diagnostics.after_backward(model, optimizer, batch_index, batch, x_input,
                                           encoder_mask, mask, output, losses, train_loader.dataset,
                                           optimization_record=optimization_record, raw_norms=raw_norms)
            # END OPTIONAL DIAGNOSTICS
            applied_control = control.apply(unwrap_model(model), optimization_record)
            if applied_control is not None:
                with (run_dir / "optimization.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(epoch=epoch, batch=batch_index, **applied_control)) + "\n")
            optimizer.step()
            # OPTIONAL DIAGNOSTICS: actual update, with no extra forward/backward.
            if diagnostics is not None:
                diagnostics.after_step(model, optimizer, batch_index, optimization_record=applied_control)
            # END OPTIONAL DIAGNOSTICS

            batch_total = float(losses["total"].detach().cpu())
            batch_reconstruction = float(losses["reconstruction"].detach().cpu())
            batch_classification = float(losses["classification"].detach().cpu())
            batch_normal_reference = float(losses["normal_reference"].detach().cpu())
            total += batch_total
            total_reconstruction += batch_reconstruction
            total_normal_reference += batch_normal_reference
            total_classification += batch_classification
            batches += 1
            padded_points += int((~batch["valid"]).sum())
            valid_points += int(batch["valid"].sum())

            if batch_index % log_freq == 0:
                now = time.time()
                seen = batch_index + 1 - last_log_batch
                window = now - last_log_time
                batch_log = (
                    f"Epoch {epoch}, Batch {batch_index}/{len(train_loader)}\n"
                    f"  Timing: local={datetime.now().strftime('%H:%M:%S')} "
                    f"elapsed={now - run_start:.1f}s window={window:.1f}s "
                    f"per_batch={window / max(1, seen):.4f}s "
                    f"samples_per_s={seen * len(batch['valid']) / max(1e-9, window):.1f}\n"
                    f"  Learning Rate: {optimizer.param_groups[0]['lr']:.9g}\n"
                    f"  Gradient Control: {applied_control}\n"
                    f"  Total Loss: {batch_total:.4f}\n"
                    f"  Recon Loss: {batch_reconstruction:.4f}\n"
                    f"{_normal_reference_log('Normal Reference Loss', batch_normal_reference, has_normal_reference)}"
                    f"  Anomaly Loss: {batch_classification:.4f}\n"
                )
                print(batch_log, end="", flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(batch_log)
                last_log_time = now
                last_log_batch = batch_index + 1

        average_total = total / max(1, batches)
        average_reconstruction = total_reconstruction / max(1, batches)
        average_normal_reference = total_normal_reference / max(1, batches)
        average_classification = total_classification / max(1, batches)
        padding_fraction = padded_points / max(1, padded_points + valid_points)
        epoch_seconds = time.time() - epoch_start
        validation = evaluate_model(model, config, config["data"]["path"], test_indices)
        validation_seconds = time.time() - epoch_start - epoch_seconds
        validation_loss = float(validation["valid_total"])
        epoch_log = (
            f"epoch={epoch} train_loss={average_total:.8f} "
            f"validation_loss={validation_loss:.8f} padding_fraction={padding_fraction:.4f}\n"
            f"Epoch {epoch} completed:\n"
            f"  Average Total Loss: {average_total:.4f}\n"
            f"  Average Recon Loss: {average_reconstruction:.4f}\n"
            f"{_normal_reference_log('Average Normal Reference Loss', average_normal_reference, has_normal_reference)}"
            f"  Average Anomaly Loss: {average_classification:.4f}\n"
            f"  Validation Recon Loss: {float(validation.get('reconstruction', 0)):.4f}\n"
            f"{_normal_reference_log('Validation Normal Reference Loss', float(validation.get('normal_reference', 0)), has_normal_reference)}"
            f"  Validation Anomaly Loss (valid): {float(validation.get('valid_classification', 0)):.4f}\n"
            f"  Validation Objective Loss: {float(validation['total']):.4f}\n"
            f"  Validation Valid Total Loss (selection): {validation_loss:.4f}\n"
            f"  Padding Fraction: {padding_fraction:.4f}\n"
            f"  Epoch Time: train={epoch_seconds:.1f}s validation={validation_seconds:.1f}s "
            f"per_batch={epoch_seconds / max(1, batches):.4f}s\n"
        )
        print(epoch_log, end="", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(epoch_log)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation": validation,
            "config": config,
        }
        if (config["experiment_id"] in NORMAL_REFERENCE_VARIANTS
                or config.get("diagnostics", {}).get("enabled", False) or resume_from is not None or control.enabled):
            checkpoint["training_state"] = checkpoint_training_state(
                epoch, validation_loss, best_loss, best_epoch, stale)
        if control.enabled:
            checkpoint["training_state"]["optimization_control"] = control.state_dict(epoch + 1)
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        if checkpoint_keep_last is not None:
            prune_epoch_checkpoints(run_dir, checkpoint_keep_last)
        # OPTIONAL DIAGNOSTICS
        if diagnostics is not None:
            diagnostics.end_epoch(epoch, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        # END OPTIONAL DIAGNOSTICS
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, checkpoint_dir / "best.pth")
        else:
            stale += 1
            if stale >= int(config["optimization"]["early_stopping_patience"]):
                break

    # OPTIONAL DIAGNOSTICS
    if diagnostics is not None:
        diagnostics.close()
    # END OPTIONAL DIAGNOSTICS
    return {"status": "completed", "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "selection_metric": "valid_total",
            "batch_sampler": batch_sampler_kind,
            "classification_padding": config["loss"].get("classification_padding", "ignore")}


def _sync_devices() -> None:
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def benchmark(model, config: dict, run_dir: Path, max_batches: int, warmup: int = 20) -> dict:
    """Measure throughput on the configured device; no validation, no checkpoints."""
    set_seed(int(config["runtime"]["seed"]))
    device = _device(config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    data_start = time.time()
    train_indices, _ = load_indices(
        config["data"]["path"],
        float(config["split"]["train_ratio"]),
        int(config["split"]["seed"]),
    )
    data_seconds = time.time() - data_start
    train_loader, kind, batch_sampler = _build_train_loader(config, train_indices)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    control = OptimizationControl(optimizer, config["optimization"], len(train_loader))
    model.train()
    if batch_sampler is not None:
        batch_sampler.set_epoch(0)
    limit = max(1, int(max_batches))
    warmup = max(0, min(int(warmup), limit - 1))
    timed_batches = 0
    padded_points = 0
    valid_points = 0
    timing_start = None
    last_total = float("nan")
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= limit:
            break
        control.begin_step(batch_index)
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["runtime"]["seed"]) + batch_index
        )
        x_input, encoder_mask, mask = make_batch_inputs(batch, config, generator)
        batch_device = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        if batch_index == warmup:
            _sync_devices()
            timing_start = time.time()
        output = model(x_input.to(device), encoder_mask.to(device))
        losses = compute_losses(
            output,
            batch_device,
            config["input"]["target"],
            float(config["loss"]["reconstruction_weight"]),
            float(config["loss"]["classification_weight"]),
            mask.to(device),
            classification_padding=config["loss"].get("classification_padding", "ignore"),
            normal_reference_weight=float(config["loss"].get("normal_reference_weight", 0)),
            normal_reference_patch_size=reference_patch_size(config),
        )
        losses["total"].backward()
        optimization_record, _ = control.inspect(unwrap_model(model), losses["total"])
        control.apply(unwrap_model(model), optimization_record)
        optimizer.step()
        last_total = float(losses["total"].detach().cpu())
        if timing_start is not None:
            timed_batches += 1
            padded_points += int((~batch["valid"]).sum())
            valid_points += int(batch["valid"].sum())
    _sync_devices()
    timed_seconds = time.time() - timing_start if timing_start is not None else 0.0
    per_batch = timed_seconds / max(1, timed_batches)
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    result = {
        "mode": "bench",
        "parallel": config["runtime"].get("parallel", "none"),
        "gpu_ids": config["runtime"].get("gpu_ids"),
        "visible_gpus": [torch.cuda.get_device_name(index) for index in range(cuda_devices)],
        "batch_size": int(config["optimization"]["batch_size"]),
        "batch_sampler": kind,
        "bucket_multiplier": int(config["optimization"].get("bucket_multiplier", 50)),
        "max_batches": limit,
        "warmup_batches": warmup,
        "timed_batches": timed_batches,
        "data_load_seconds": round(data_seconds, 2),
        "timed_seconds": round(timed_seconds, 2),
        "s_per_batch": round(per_batch, 4),
        "samples_per_s": round(
            timed_batches * int(config["optimization"]["batch_size"]) / max(1e-9, timed_seconds), 2
        ),
        "epoch_estimate_hours": round(per_batch * len(train_loader) / 3600.0, 3),
        "padding_fraction": round(padded_points / max(1, padded_points + valid_points), 4),
        "last_total_loss": last_total,
        "peak_memory_mb": [
            round(torch.cuda.max_memory_allocated(index) / 1e6, 1) for index in range(cuda_devices)
        ],
    }
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics" / "bench.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


"""Epoch-boundary continuation and bounded checkpoint retention.

Training remains in train.py; this module only validates/restores saved state.
"""





def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resume_config(previous, current):
    def comparable(config):
        value = copy.deepcopy(config)
        value.pop("diagnostics", None)
        for field in ("epochs", "log_freq"):
            value["optimization"].pop(field, None)
        return value
    if comparable(previous) != comparable(current):
        raise ValueError("Resume requires identical data/split/model/input/loss/optimizer/sampler/runtime; "
                         "only epochs, log_freq and diagnostics may differ")


def restore_rng(state):
    states = state["torch_cuda"]
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(states) != count:
        raise ValueError(f"Saved CUDA RNG count {len(states)} differs from current {count}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if states:
        torch.cuda.set_rng_state_all([item.cpu() for item in states])


def checkpoint_training_state(epoch, validation_loss, best_loss, best_epoch, stale):
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }
    improved = validation_loss < best_loss
    return {
        "next_epoch": epoch + 1,
        "best_loss": validation_loss if improved else best_loss,
        "best_epoch": epoch if improved else best_epoch,
        "stale": 0 if improved else stale + 1,
        "rng": state,
    }


def restore_training(checkpoint_path, model, optimizer, config, run_dir):
    checkpoint_path = Path(checkpoint_path).resolve()
    source_run = checkpoint_path.parent.parent
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    verify_resume_config(payload["config"], config)
    epoch = int(payload["epoch"])
    start_epoch = epoch + 1
    if start_epoch >= int(config["optimization"]["epochs"]):
        raise ValueError("epochs is the TOTAL epoch limit and must exceed the resumed next epoch")
    state = payload.get("training_state")
    rng_path = None
    if state is None:
        # Compatibility with the diagnostic run created before resume existed.
        # Its epoch RNG was captured after validation and before best serialization.
        rng_path = source_run / "diagnostics" / f"rng_after_epoch_{epoch:04d}.pth"
        if not rng_path.is_file():
            raise ValueError("Checkpoint has no complete RNG; matching epoch diagnostic RNG is required")
        history = {
            int(e): float(loss) for e, loss in re.findall(
                r"^epoch=(\d+) train_loss=\S+ validation_loss=(\S+)",
                (source_run / "train.log").read_text(), re.M
            )
        }
        if set(range(epoch + 1)) - set(history):
            raise ValueError("Legacy resume requires complete epoch history for early stopping")
        best_payload = torch.load(source_run / "checkpoints/best.pth",
                                  map_location="cpu", weights_only=False)
        best_epoch = int(best_payload["epoch"])
        if best_epoch > epoch:
            raise ValueError("Legacy resume requires the latest completed source checkpoint")
        best_loss = float(best_payload["validation"]["valid_total"])
        # Verify the history-selected best agrees despite its eight-decimal rounding.
        if abs(history[best_epoch] - min(history[e] for e in range(epoch + 1))) > 1e-7:
            raise ValueError("Source best checkpoint conflicts with validation history")
        state = {
            "next_epoch": start_epoch, "best_loss": best_loss, "best_epoch": best_epoch,
            "stale": epoch - best_epoch,
            "rng": torch.load(rng_path, map_location="cpu", weights_only=False),
        }
        del best_payload
    if int(state["next_epoch"]) != start_epoch:
        raise ValueError("Checkpoint epoch and training_state.next_epoch disagree")
    if not math.isfinite(float(state["best_loss"])):
        raise ValueError("Resume requires a finite best validation loss")
    if int(state["stale"]) >= int(config["optimization"]["early_stopping_patience"]):
        raise ValueError("Source already reached early stopping; continuation would change that rule")
    source_best = source_run / "checkpoints/best.pth"
    best_payload = torch.load(source_best, map_location="cpu", weights_only=False)
    if int(best_payload["epoch"]) != int(state["best_epoch"]):
        raise ValueError("Source best.pth does not match saved early-stopping state")
    del best_payload
    unwrap_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    destination = Path(run_dir) / "checkpoints/best.pth"
    shutil.copyfile(source_best, destination)
    provenance = {
        "source_run": source_run.name, "source_checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "rng_source": "checkpoint.training_state.rng" if rng_path is None else rng_path.name,
        "rng_sha256": None if rng_path is None else _sha256(rng_path),
        "best_checkpoint_sha256": _sha256(source_best),
        "source_epoch": epoch, "next_epoch": start_epoch,
        "best_loss": state["best_loss"], "best_epoch": state["best_epoch"], "stale": state["stale"],
        "total_epoch_limit": int(config["optimization"]["epochs"]),
        "numerical_limit": "Restores RNG and optimizer; CUDA execution need not be bitwise identical.",
    }
    (Path(run_dir) / "resume.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return state


def prune_epoch_checkpoints(run_dir, keep_last):
    if keep_last is None:
        return
    run_dir = Path(run_dir)
    files = sorted((run_dir / "checkpoints").glob("epoch_*.pth"))
    retained = set(files[-int(keep_last):])
    event = run_dir / "diagnostics/first_event.json"
    if event.exists():
        epoch = int(json.loads(event.read_text())["epoch"])
        retained.add(run_dir / "checkpoints" / f"epoch_{epoch - 1:04d}.pth")
    for path in files:
        if path not in retained:
            size = path.stat().st_size
            path.unlink()
            with (run_dir / "checkpoint_retention.jsonl").open("a") as handle:
                handle.write(json.dumps({"removed": path.name, "bytes": size,
                                         "keep_last": keep_last}) + "\n")


"""Optional optimizer safeguards; no model, loss, sampling or RNG changes."""


def gradient_norms(model):
    """Float64 reductions on source devices; transfer only per-parameter scalars."""
    values = [(name, torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64))
              for name, p in model.named_parameters() if p.grad is not None]
    if not values:
        return {}
    # DataParallel gathers gradients on its primary device.
    return dict(zip((n for n, _ in values), torch.stack([v for _, v in values]).cpu().tolist()))


class OptimizationControl:
    def __init__(self, optimizer, options, steps_per_epoch):
        validate_optimization(options)
        self.optimizer = optimizer
        self.options = options
        self.steps_per_epoch = int(steps_per_epoch)
        self.enabled = options.get("gradient_clip_norm") is not None or options.get("lr_schedule") is not None
        self.update_index = 0

    def learning_rate(self, update_index):
        peak = float(self.options["learning_rate"])
        schedule = self.options.get("lr_schedule")
        if schedule is None:
            return peak
        warm = int(schedule["warmup_epochs"]) * self.steps_per_epoch
        total = int(schedule["total_epochs"]) * self.steps_per_epoch
        if not 0 <= update_index < total:
            raise ValueError("Update lies outside fixed schedule horizon")
        if update_index < warm:
            fraction = update_index / max(1, warm - 1)
            return peak * (float(schedule["start_factor"]) + (1-float(schedule["start_factor"]))*fraction)
        fraction = (update_index - warm + 1) / (total - warm)
        minimum = float(schedule["min_lr"])
        return minimum + (peak-minimum)*0.5*(1+math.cos(math.pi*fraction))

    def begin_step(self, update_index):
        self.update_index = int(update_index)
        if self.enabled:
            lr = self.learning_rate(self.update_index)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def inspect(self, model, loss):
        if not self.enabled:
            return None, None
        norms = gradient_norms(model)
        raw = math.hypot(*norms.values())
        finite = math.isfinite(float(loss.detach())) and math.isfinite(raw)
        limit = self.options.get("gradient_clip_norm")
        coefficient = min(1., float(limit)/(raw+1e-6)) if finite and limit is not None else 1.
        record = dict(update_index=self.update_index,
                      learning_rate=[float(g["lr"]) for g in self.optimizer.param_groups],
                      gradient_l2_before=raw, clip_max_norm=limit,
                      clip_coefficient=coefficient, clipped=coefficient < 1.,
                      finite=finite, phase="before_gradient_clip")
        return record, norms

    @torch.no_grad()
    def apply(self, model, record):
        if record is None:
            return None
        if not record["finite"]:
            raise FloatingPointError("Nonfinite loss/gradient; optimizer.step was not executed")
        coefficient = record["clip_coefficient"]
        if coefficient < 1.:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(coefficient)
        result = dict(record, phase="after_gradient_clip_before_optimizer_step",
                      gradient_l2_after=math.hypot(*gradient_norms(model).values()))
        if not math.isfinite(result["gradient_l2_after"]):
            raise FloatingPointError("Nonfinite clipped gradient; optimizer.step was not executed")
        return result

    def state_dict(self, next_epoch):
        return dict(next_update=int(next_epoch)*self.steps_per_epoch,
                    steps_per_epoch=self.steps_per_epoch,
                    lr_schedule=self.options.get("lr_schedule"),
                    gradient_clip_norm=self.options.get("gradient_clip_norm"))

    def verify_resume(self, state, next_epoch):
        if state != self.state_dict(next_epoch):
            raise ValueError("Optimization state/horizon/loader length differs from checkpoint")
