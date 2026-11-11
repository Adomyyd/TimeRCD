from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedTimeSeriesDataset, collate_samples, load_indices
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


def train_model(model, config: dict, run_dir: Path) -> dict:
    set_seed(int(config["runtime"]["seed"]))
    device = _device(config)
    train_indices, test_indices = load_indices(
        config["data"]["path"],
        float(config["split"]["train_ratio"]),
        int(config["split"]["seed"]),
    )
    train_loader = DataLoader(
        PairedTimeSeriesDataset(config["data"]["path"], train_indices),
        batch_size=int(config["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["optimization"]["num_workers"]),
        collate_fn=collate_samples,
    )
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
    )
    print(patch_log, end="", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(patch_log)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(int(config["optimization"]["epochs"])):
        model.train()
        total = 0.0
        total_reconstruction = 0.0
        total_classification = 0.0
        batches = 0
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

            if batch_index % log_freq == 0:
                batch_log = (
                    f"Epoch {epoch}, Batch {batch_index}/{len(train_loader)}\n"
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

        average_total = total / max(1, batches)
        average_reconstruction = total_reconstruction / max(1, batches)
        average_classification = total_classification / max(1, batches)
        validation = evaluate_model(model, config, config["data"]["path"], test_indices)
        validation_loss = float(validation["valid_total"])
        epoch_log = (
            f"epoch={epoch} train_loss={average_total:.8f} "
            f"validation_loss={validation_loss:.8f}\n"
            f"Epoch {epoch} completed:\n"
            f"  Average Total Loss: {average_total:.4f}\n"
            f"  Average Recon Loss: {average_reconstruction:.4f}\n"
            f"  Average Anomaly Loss: {average_classification:.4f}\n"
            f"  Validation Objective Loss: {float(validation['total']):.4f}\n"
            f"  Validation Valid Total Loss (selection): {validation_loss:.4f}\n"
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
            "classification_padding": config["loss"].get("classification_padding", "ignore")}
