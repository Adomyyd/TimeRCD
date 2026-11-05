from copy import deepcopy

import numpy as np
import torch

from models.TimeRCD import TimeRCDPretrainTester as LegacyTester
from time_rcd import TimeRCDDetector
from time_rcd._inference import _WindowDataset


class _SharedLegacyTester(LegacyTester):
    """Run the legacy forward path with the exact same loaded weights."""

    def __init__(self, public_tester):
        self.config = deepcopy(public_tester.config)
        self.device = public_tester.device
        self.win_size = public_tester.win_size
        self.batch_size = public_tester.batch_size
        self.model = public_tester.model


def make_detector(config, window):
    return TimeRCDDetector.from_local(
        config["checkpoint"], variant="uni", win_size=window,
        batch_size=config["batch_size"], device=config["device"],
    )


def compare_backends(detector, values):
    public_scores, public_logits = detector.predict(values, return_logits=True)
    # The benchmark creates a fresh tester for each series; its short-series
    # window mutation must not carry over to the next sequence.
    legacy = _SharedLegacyTester(detector._tester)
    scores, logits = legacy.zero_shot(values)
    legacy_scores = np.concatenate([chunk.reshape(-1) for chunk in scores])
    legacy_logits = np.concatenate([chunk.reshape(-1) for chunk in logits])
    actual = min(len(values), detector.win_size)
    expected = ((len(values) + actual - 1) // actual) * actual
    if len(legacy_scores) != expected or len(legacy_logits) != expected:
        raise ValueError("Legacy output has unexpected padded length")
    if public_scores.shape != (len(values),) or public_logits.shape != (len(values),):
        raise ValueError("Public API did not return exactly one score per input point")
    return {
        "public_scores": public_scores, "public_logits": public_logits,
        "legacy_scores": legacy_scores[:len(values)], "legacy_logits": legacy_logits[:len(values)],
    }, expected


def tail_inputs(values, window):
    dataset = _WindowDataset(values, min(len(values), window))
    current, mask = dataset[len(dataset) - 1]
    valid = int(mask.sum())
    zero = current.clone()
    zero[~mask] = 0
    return {
        "current": (current, mask),
        "zero": (zero, mask),
        "compact": (current[:valid], mask[:valid]),
    }, valid


def score_window(model, values, mask, device):
    with torch.inference_mode():
        dtype = next(model.parameters()).dtype
        embeddings = model(values[None].to(device=device, dtype=dtype), mask=mask[None].to(device))
        logits = model.anomaly_head(embeddings).mean(dim=-2)[0]
        logits = logits[mask.to(device)]
        return {
            "scores": logits.softmax(dim=-1)[:, 1].cpu().numpy(),
            "logits": (logits[:, 1] - logits[:, 0]).cpu().numpy(),
        }


def probe_tail(detector, values):
    inputs, valid = tail_inputs(values, detector.win_size)
    results = {}
    for name, (window, mask) in inputs.items():
        prediction = score_window(detector._tester.model, window, mask, detector._tester.device)
        results.update({f"tail_{name}_{key}": value for key, value in prediction.items()})
    return results, valid


def probe_short(detector, values):
    if len(values) >= detector.win_size:
        return {}
    normalized, _ = _WindowDataset(values, len(values))[0]
    padding = detector.win_size - len(values)
    mask = torch.arange(detector.win_size) < len(values)
    results = {}
    for name, fill in (("short_forced_zero", torch.zeros(padding, 1)),
                       ("short_forced_edge", normalized[-1:].expand(padding, -1))):
        padded = torch.cat((normalized, fill))
        results[name] = score_window(detector._tester.model, padded, mask, detector._tester.device)
    return results
