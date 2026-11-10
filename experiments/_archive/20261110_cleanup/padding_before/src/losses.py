from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    active = mask.bool()
    count = int(active.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    values = (prediction - target)[active]
    return (values.square().mean()), count


def classification_ce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or labels.ndim != 2 or valid.shape != labels.shape:
        raise ValueError("classification shapes must be logits=(B,T,2), labels=(B,T), valid=(B,T)")
    active = valid & (labels >= 0)
    count = int(active.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    clean_labels = (labels > 0).long()
    return F.cross_entropy(logits[active], clean_labels[active]), count


def compute_losses(output, batch, target: str, reconstruction_weight: float, classification_weight: float, reconstruction_mask: torch.Tensor):
    reconstruction_target = batch["x0"] if target == "x0" else batch["x"]
    recon_loss, recon_count = masked_mse(output["reconstruction"], reconstruction_target, reconstruction_mask)
    cls_loss, cls_count = classification_ce(output["logits"], batch["y"], batch["valid"])
    total = reconstruction_weight * recon_loss + classification_weight * cls_loss
    return {"total": total, "reconstruction": recon_loss, "classification": cls_loss, "reconstruction_count": recon_count, "classification_count": cls_count}
