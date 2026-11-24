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


P2_VARIANT = "p2_mask_x_normal_anomaly_patch"
NORMAL_REFERENCE_VARIANTS = {"p1_mask_x_normal", P2_VARIANT}


def reference_patch_size(config):
    if config["experiment_id"] == P2_VARIANT:
        return int(config["model"]["patch_size"])
    return None


def normal_reference_mask(batch, active, patch_size):
    if patch_size is None:
        return active
    anomalous = (batch["y"] > 0) & batch["valid"]
    length = anomalous.size(1)
    padded = F.pad(anomalous, (0, (-length) % patch_size))
    anomaly_patches = padded.unfold(1, patch_size, patch_size).any(dim=-1)
    return active & anomaly_patches.repeat_interleave(patch_size, dim=1)[:, :length]


def compute_losses(output, batch, target: str, reconstruction_weight: float, classification_weight: float,
                   reconstruction_mask: torch.Tensor, classification_padding: str = "ignore",
                   normal_reference_weight: float = 1.0, normal_reference_patch_size: int | None = None):
    if target != "x":
        raise ValueError("Observation reconstruction target must be x; old P1/P2 are retired")
    active = reconstruction_mask & batch["valid"]
    recon_loss, recon_count = masked_mse(output["reconstruction"], batch["x"], active)
    cls_loss, cls_count = classification_ce(output["logits"], batch["y"], batch["valid"], classification_padding)
    total = reconstruction_weight * recon_loss + classification_weight * cls_loss
    normal_loss, normal_count = recon_loss.new_zeros(()), 0
    if "normal_reference" in output:
        reference_mask = normal_reference_mask(batch, active, normal_reference_patch_size)
        normal_loss, normal_count = masked_mse(output["normal_reference"], batch["x0"], reference_mask)
        total = total + normal_reference_weight * normal_loss
    return {"total": total, "reconstruction": recon_loss, "classification": cls_loss,
            "normal_reference": normal_loss, "normal_reference_count": normal_count,
            "reconstruction_count": recon_count, "classification_count": cls_count}
