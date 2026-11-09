import torch

from src.sampling import apply_input_mask, sample_patch_mask


def test_patch_sampling_is_valid_and_input_mask_is_p0_only_operation():
    valid = torch.ones(2, 32, dtype=torch.bool)
    generator = torch.Generator().manual_seed(72)
    mask = sample_patch_mask(valid, patch_size=16, ratio=0.15, generator=generator)
    assert mask.shape == valid.shape
    assert torch.all(mask <= valid)
    x = torch.ones(2, 32, 1)
    masked = apply_input_mask(x, mask)
    assert torch.all(masked[mask] == 0)
    assert torch.all(masked[~mask] == 1)
