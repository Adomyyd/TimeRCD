from __future__ import annotations

import torch


def sample_patch_mask(valid: torch.Tensor, patch_size: int, ratio: float, generator: torch.Generator) -> torch.Tensor:
    if valid.ndim != 2:
        raise ValueError("valid must have shape (batch, time)")
    batch, max_len = valid.shape
    mask = torch.zeros_like(valid, dtype=torch.bool)
    for row in range(batch):
        length = int(valid[row].sum().item())
        num_patches = (length + patch_size - 1) // patch_size
        count = int(num_patches * ratio)
        if count <= 0:
            continue
        selected = torch.randperm(num_patches, generator=generator)[:count]
        for patch in selected.tolist():
            start = patch * patch_size
            end = min(start + patch_size, length)
            mask[row, start:end] = True
    return mask & valid


def apply_input_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError("x must be (batch,time,features), mask must be (batch,time)")
    output = x.clone()
    output[mask.unsqueeze(-1).expand_as(output)] = 0.0
    return output
