from __future__ import annotations

import json
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
from .losses import compute_losses
from .parallel import parallel_record, resolve_device, unwrap_model
from .sampling import apply_input_mask, sample_patch_mask


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
    if config["input"]["input_mode"] == "masked_x":
        x_input = apply_input_mask(batch["x"], mask)
        encoder_mask = batch["valid"] & ~mask
    else:
        x_input = batch["x"]
        encoder_mask = batch["valid"]
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


def train_model(model, config: dict, run_dir: Path) -> dict:
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
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_freq = int(config["optimization"]["log_freq"])
    patch_log = (
        f"Parallel settings: {parallel_record(model, config)}\n"
        f"Loss settings: classification_padding={config['loss'].get('classification_padding', 'ignore')}, "
        f"checkpoint_selection=valid_total\n"
        f"Patch settings: encoder_patch_size={unwrap_model(model).encoder.patch_size}, "
        f"mask_patch_size={int(config['input']['mask_patch_size'])}, "
        f"mask_ratio={float(config['input']['mask_ratio'])}, "
        f"input_mode={config['input']['input_mode']}\n"
        f"Sampler settings: batch_sampler={batch_sampler_kind}, "
        f"batch_size={int(config['optimization']['batch_size'])}, "
        f"bucket_multiplier={int(config['optimization'].get('bucket_multiplier', 50))}\n"
        f"Data ready: train_samples={len(train_indices)}, val_samples={len(test_indices)}, "
        f"batches_per_epoch={len(train_loader)}, load_seconds={data_seconds:.1f}\n"
        f"Start: local={datetime.now().isoformat(timespec='seconds')}\n"
    )
    print(patch_log, end="", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(patch_log)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    run_start = time.time()

    for epoch in range(int(config["optimization"]["epochs"])):
        model.train()
        epoch_start = time.time()
        last_log_time = epoch_start
        last_log_batch = 0
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        total = 0.0
        total_reconstruction = 0.0
        total_classification = 0.0
        batches = 0
        padded_points = 0
        valid_points = 0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device="cpu").manual_seed(
                int(config["runtime"]["seed"]) + epoch * 1000003 + batch_index
            )
            x_input, encoder_mask, mask = make_batch_inputs(batch, config, generator)
            batch_device = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            output = model(x_input.to(device), encoder_mask.to(device))
            losses = compute_losses(
                output,
                batch_device,
                config["input"]["target"],
                float(config["loss"]["reconstruction_weight"]),
                float(config["loss"]["classification_weight"]),
                mask.to(device),
                classification_padding=config["loss"].get("classification_padding", "ignore"),
            )
            losses["total"].backward()
            optimizer.step()

            batch_total = float(losses["total"].detach().cpu())
            batch_reconstruction = float(losses["reconstruction"].detach().cpu())
            batch_classification = float(losses["classification"].detach().cpu())
            total += batch_total
            total_reconstruction += batch_reconstruction
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
                    f"  Total Loss: {batch_total:.4f}\n"
                    f"  Recon Loss: {batch_reconstruction:.4f}\n"
                    f"  Anomaly Loss: {batch_classification:.4f}\n"
                    f"  Classification points: loss={losses['classification_count']}, "
                    f"valid={int((batch['valid'] & (batch['y'] >= 0)).sum())}, "
                    f"padding={int((~batch['valid']).sum())}\n"
                )
                print(batch_log, end="", flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(batch_log)
                last_log_time = now
                last_log_batch = batch_index + 1

        average_total = total / max(1, batches)
        average_reconstruction = total_reconstruction / max(1, batches)
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
            f"  Average Anomaly Loss: {average_classification:.4f}\n"
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
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint, checkpoint_dir / "best.pth")
        else:
            stale += 1
            if stale >= int(config["optimization"]["early_stopping_patience"]):
                break

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
        )
        losses["total"].backward()
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
