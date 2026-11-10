import torch

from src.sampling import apply_input_mask, sample_patch_mask


def test_mask_granularity_is_independent_of_encoder_patch_size():
    encoder_patch_size = 16
    mask_patch_size = 4
    valid = torch.ones(2, 1024, dtype=torch.bool)
    generator = torch.Generator().manual_seed(72)
    mask = sample_patch_mask(
        valid,
        mask_patch_size=mask_patch_size,
        ratio=0.15,
        generator=generator,
    )

    assert encoder_patch_size == 16
    assert mask.shape == valid.shape
    assert torch.all(mask <= valid)
    assert torch.all(mask.sum(dim=1) == 152)
    for row in mask:
        for start in range(0, row.numel(), mask_patch_size):
            assert int(row[start : start + mask_patch_size].sum()) in (0, mask_patch_size)


def test_input_mask_zeroes_only_selected_values():
    valid = torch.ones(2, 32, dtype=torch.bool)
    generator = torch.Generator().manual_seed(72)
    mask = sample_patch_mask(valid, mask_patch_size=4, ratio=0.15, generator=generator)
    x = torch.ones(2, 32, 1)
    masked = apply_input_mask(x, mask)
    assert torch.all(masked[mask] == 0)
    assert torch.all(masked[~mask] == 1)
