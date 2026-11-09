from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data import PairedTimeSeriesDataset, collate_samples
from .losses import compute_losses
from .sampling import apply_input_mask, sample_patch_mask


def evaluate_model(model, config: dict, dataset_path: str, indices: list[int]) -> dict:
    dataset = PairedTimeSeriesDataset(dataset_path, indices)
    loader = DataLoader(
        dataset,
        batch_size=int(config["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["optimization"]["num_workers"]),
        collate_fn=collate_samples,
    )
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["evaluation"]["validation_mask_seed"])
    )
    sums = {
        "total": 0.0,
        "reconstruction": 0.0,
        "classification": 0.0,
        "selected_normal_reconstruction": 0.0,
        "selected_anomalous_reconstruction": 0.0,
        "full_normal_reconstruction": 0.0,
        "full_anomalous_reconstruction": 0.0,
        "copy_full_normal_error": 0.0,
        "copy_full_anomalous_error": 0.0,
    }
    counts = {
        "batches": 0,
        "selected_normal_points": 0,
        "selected_anomalous_points": 0,
        "full_normal_points": 0,
        "full_anomalous_points": 0,
        "classification_points": 0,
        "classification_correct": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    with torch.no_grad():
        for batch in loader:
            x_input, encoder_mask, mask = _make_eval_inputs(batch, config, generator)
            batch_device = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            output = model(x_input.to(device), encoder_mask.to(device))
            losses = compute_losses(
                output,
                batch_device,
                config["input"]["target"],
                float(config["loss"]["reconstruction_weight"]),
                float(config["loss"]["classification_weight"]),
                mask.to(device),
            )
            sums["total"] += float(losses["total"])
            sums["reconstruction"] += float(losses["reconstruction"])
            sums["classification"] += float(losses["classification"])

            target = (
                batch_device["x0"]
                if config["input"]["target"] == "x0"
                else batch_device["x"]
            )
            error = (output["reconstruction"] - target).square()
            selected = mask.to(device) & batch_device["valid"]
            full = batch_device["valid"]
            selected_normal = selected & (batch_device["y"] == 0)
            selected_anomalous = selected & (batch_device["y"] > 0)
            full_normal = full & (batch_device["y"] == 0)
            full_anomalous = full & (batch_device["y"] > 0)

            for name, active in (
                ("selected_normal_reconstruction", selected_normal),
                ("selected_anomalous_reconstruction", selected_anomalous),
                ("full_normal_reconstruction", full_normal),
                ("full_anomalous_reconstruction", full_anomalous),
            ):
                if active.any():
                    sums[name] += float(error[active].sum())

            copy_error = (batch_device["x"] - batch_device["x0"]).square()
            if full_normal.any():
                sums["copy_full_normal_error"] += float(copy_error[full_normal].sum())
            if full_anomalous.any():
                sums["copy_full_anomalous_error"] += float(copy_error[full_anomalous].sum())

            counts["selected_normal_points"] += int(selected_normal.sum())
            counts["selected_anomalous_points"] += int(selected_anomalous.sum())
            counts["full_normal_points"] += int(full_normal.sum())
            counts["full_anomalous_points"] += int(full_anomalous.sum())

            active_cls = full & (batch_device["y"] >= 0)
            if active_cls.any():
                predicted = output["logits"].argmax(dim=-1)
                truth = (batch_device["y"] > 0).long()
                pred_active = predicted[active_cls]
                truth_active = truth[active_cls]
                counts["classification_points"] += int(active_cls.sum())
                counts["classification_correct"] += int((pred_active == truth_active).sum())
                counts["true_positive"] += int(((pred_active == 1) & (truth_active == 1)).sum())
                counts["false_positive"] += int(((pred_active == 1) & (truth_active == 0)).sum())
                counts["false_negative"] += int(((pred_active == 0) & (truth_active == 1)).sum())
            counts["batches"] += 1

    def mean(sum_key: str, count_key: str) -> float:
        return sums[sum_key] / max(1, counts[count_key])

    precision_den = counts["true_positive"] + counts["false_positive"]
    recall_den = counts["true_positive"] + counts["false_negative"]
    precision = counts["true_positive"] / max(1, precision_den)
    recall = counts["true_positive"] / max(1, recall_den)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "total": sums["total"] / max(1, counts["batches"]),
        "reconstruction": sums["reconstruction"] / max(1, counts["batches"]),
        "classification": sums["classification"] / max(1, counts["batches"]),
        "selected_normal_reconstruction": mean(
            "selected_normal_reconstruction", "selected_normal_points"
        ),
        "selected_anomalous_reconstruction": mean(
            "selected_anomalous_reconstruction", "selected_anomalous_points"
        ),
        "full_normal_reconstruction": mean(
            "full_normal_reconstruction", "full_normal_points"
        ),
        "full_anomalous_reconstruction": mean(
            "full_anomalous_reconstruction", "full_anomalous_points"
        ),
        "copy_full_normal_error": mean("copy_full_normal_error", "full_normal_points"),
        "copy_full_anomalous_error": mean(
            "copy_full_anomalous_error", "full_anomalous_points"
        ),
        "classification_accuracy": counts["classification_correct"]
        / max(1, counts["classification_points"]),
        "classification_precision": precision,
        "classification_recall": recall,
        "classification_f1": f1,
        **counts,
    }


def _make_eval_inputs(batch, config: dict, generator: torch.Generator):
    mask = sample_patch_mask(
        batch["valid"],
        int(config["model"]["patch_size"]),
        float(config["input"]["mask_ratio"]),
        generator,
    )
    if config["input"]["input_mode"] == "masked_x":
        return apply_input_mask(batch["x"], mask), batch["valid"] & ~mask, mask
    return batch["x"], batch["valid"], mask
