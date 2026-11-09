import torch

from src.losses import classification_ce, masked_mse


def test_padding_is_excluded_from_classification():
    logits = torch.zeros(1, 3, 2)
    labels = torch.tensor([[0, 1, -1]])
    valid = torch.tensor([[True, True, False]])
    loss, count = classification_ce(logits, labels, valid)
    assert torch.isfinite(loss)
    assert count == 2


def test_zero_reconstruction_mask_is_explicitly_zero():
    prediction = torch.ones(1, 2, 1)
    target = torch.zeros(1, 2, 1)
    loss, count = masked_mse(prediction, target, torch.zeros(1, 2, dtype=torch.bool))
    assert loss.item() == 0.0
    assert count == 0
