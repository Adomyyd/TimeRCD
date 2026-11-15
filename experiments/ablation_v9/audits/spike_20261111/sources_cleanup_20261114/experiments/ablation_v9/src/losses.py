from __future__ import annotations

import torch
import torch.nn.functional as F


CLASSIFICATION_PADDING_POLICIES = ("ignore", "official")


def validate_classification_padding(policy: str) -> None:
    if policy not in CLASSIFICATION_PADDING_POLICIES:
        raise ValueError("loss.classification_padding must be 'ignore' or 'official'")


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    active = mask.bool()
    count = int(active.sum().item())
    if count == 0:
        return prediction.sum() * 0.0, 0
    values = (prediction - target)[active]
    return (values.square().mean()), count


def classification_ce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor,
                      padding_policy: str = "ignore") -> tuple[torch.Tensor, int]:
    validate_classification_padding(padding_policy)
    if logits.ndim != 3 or labels.ndim != 2 or valid.shape != labels.shape:
        raise ValueError("classification shapes must be logits=(B,T,2), labels=(B,T), valid=(B,T)")
    if padding_policy == "official":
        # Match the official order: binarization destroys the padding sentinel -1.
        clean_labels = (labels > 0.5).long()
        active = clean_labels != -1
    else:
        active = valid & (labels >= 0)
        clean_labels = (labels > 0).long()
    count = int(active.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[active], clean_labels[active]), count


def compute_losses(output, batch, target: str, reconstruction_weight: float, classification_weight: float,
                   reconstruction_mask: torch.Tensor, classification_padding: str = "ignore"):
    reconstruction_target = batch["x0"] if target == "x0" else batch["x"]
    recon_loss, recon_count = masked_mse(output["reconstruction"], reconstruction_target, reconstruction_mask)
    cls_loss, cls_count = classification_ce(output["logits"], batch["y"], batch["valid"], classification_padding)
    total = reconstruction_weight * recon_loss + classification_weight * cls_loss
    return {"total": total, "reconstruction": recon_loss, "classification": cls_loss, "reconstruction_count": recon_count, "classification_count": cls_count}
