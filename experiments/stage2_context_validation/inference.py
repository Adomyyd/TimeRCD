import numpy as np
import torch

from experiments.stage1.inference import score_window


def score_condition(model, normalized_values, point_mask, device):
    values = np.asarray(normalized_values, dtype=np.float32)
    point_mask = np.asarray(point_mask, dtype=bool)
    if values.ndim != 1 or point_mask.shape != values.shape:
        raise ValueError("values and point_mask must be equally sized vectors")
    if not np.isfinite(values).all() or not point_mask.any():
        raise ValueError("Inference input must be finite and contain visible points")
    result = score_window(
        model,
        torch.from_numpy(values[:, None]),
        torch.from_numpy(point_mask),
        device,
    )
    if any(result[key].shape != (int(point_mask.sum()),) for key in ("logits", "scores")):
        raise ValueError("Model returned an unexpected number of valid logits")
    if not np.isfinite(result["logits"]).all() or not np.isfinite(result["scores"]).all():
        raise ValueError("Model returned nonfinite scores or logits")
    return result


def target_logits(result, point_mask, target):
    visible_indices = np.flatnonzero(point_mask)
    target_indices = np.arange(target[0], target[1])
    positions = np.searchsorted(visible_indices, target_indices)
    if (positions >= len(visible_indices)).any() or not np.array_equal(visible_indices[positions], target_indices):
        raise ValueError("Target points are not all visible")
    return result["logits"][positions]


def disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
