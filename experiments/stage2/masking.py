from contextlib import contextmanager

import numpy as np
import torch


def reference_masks(labels, valid, patch_size):
    labels, valid = np.asarray(labels), np.asarray(valid)
    if labels.shape != valid.shape or labels.ndim != 1 or valid.dtype != bool:
        raise ValueError("Expected aligned one-dimensional labels and boolean validity")
    if not np.isin(labels, [0, 1]).all() or patch_size <= 0:
        raise ValueError("Invalid labels or patch size")
    padding = (-len(labels)) % patch_size
    point_valid = np.pad(valid, (0, padding)).reshape(-1, patch_size)
    point_bad = np.pad((labels == 1) & valid, (0, padding)).reshape(-1, patch_size)
    present = point_valid.any(axis=1)
    bad = point_bad.any(axis=1)
    mixed = bad & ((point_valid.sum(axis=1) - point_bad.sum(axis=1)) > 0)
    return present, present & ~bad, mixed


def random_reference_mask(present, clean, rng):
    if present.shape != clean.shape or np.any(clean & ~present):
        raise ValueError("Invalid clean reference mask")
    if not clean.any():
        raise ValueError("No clean references: intervention undefined")
    keep = np.zeros_like(present)
    keep[rng.choice(np.flatnonzero(present), int(clean.sum()), replace=False)] = True
    return keep


@contextmanager
def reference_intervention(model, keep):
    """Replace only the key/value mask, checking every attention layer receives it."""
    encoder = model.ts_encoder
    if encoder.num_features != 1 or not encoder.use_rope:
        raise ValueError("This diagnostic requires the univariate RoPE encoder")
    keep = torch.as_tensor(keep, dtype=torch.bool, device=next(model.parameters()).device)[None]
    if keep.ndim != 2 or not keep.any():
        raise ValueError("At least one reference must remain")
    layers = encoder.transformer_encoder.layers
    calls = [0] * len(layers)

    def replace(module, args, kwargs):
        original = kwargs["attn_mask"]
        if original.shape != keep.shape or torch.any(keep & ~original):
            raise ValueError("Intervention must be a subset of the valid Patch mask")
        return args, {**kwargs, "attn_mask": keep}

    def check(index):
        def audit(module, args, kwargs):
            if not torch.equal(kwargs["attn_mask"], keep):
                raise RuntimeError("Reference mask did not reach an attention layer")
            calls[index] += 1
        return audit

    handles = []
    try:
        handles.append(encoder.transformer_encoder.register_forward_pre_hook(replace, with_kwargs=True))
        for index, layer in enumerate(layers):
            handles.append(layer.self_attn.register_forward_pre_hook(check(index), with_kwargs=True))
        yield
        if calls != [1] * len(layers):
            raise RuntimeError(f"Expected one call at every layer, got {calls}")
    finally:
        for handle in handles:
            handle.remove()
