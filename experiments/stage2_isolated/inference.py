import hashlib

import numpy as np
import torch

from experiments.common.artifacts import append_row
from experiments.stage1.inference import score_window
from experiments.stage2.masking import reference_intervention
from experiments.stage2_paths.attention import attention_edges, edge_mask


class Auditor:
    def __init__(self, directory, model, device):
        self.directory, self.model, self.device = directory, model, device
        self.forwards, self.checks = 0, 0

    def check(self, case, name, passed, max_abs=0.0):
        append_row(self.directory / "checks.csv", {"case": case, "test": name, "passed": bool(passed), "max_abs": max_abs})
        self.checks += 1
        if not passed:
            raise RuntimeError(f"Gate failed: {case} {name}")

    def equal(self, case, name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        same_shape = a.shape == b.shape
        maximum = float(np.abs(a.astype(float) - b.astype(float)).max()) if same_shape and a.size else 0.0
        self.check(case, name, same_shape and np.array_equal(a, b), maximum)

    def public(self, values, valid):
        self.forwards += 1
        return score_window(self.model, values, valid, self.device)

    def predict(self, case, variant, values, valid, present, selected, condition):
        with attention_edges(self.model, present, selected, condition, "all") as records:
            prediction = self.public(values, valid)
        expected = edge_mask(present, selected, condition)
        digest = hashlib.sha256(expected.tobytes()).hexdigest()
        self.check(case, f"{variant}_all_layers", len(records) == len(self.model.ts_encoder.transformer_encoder.layers))
        for record in records:
            self.check(case, f"{variant}_layer{record['layer']}_mask", record["mask_sha256"] == digest)
            append_row(self.directory / "layer_masks.csv", {"case": case, "variant": variant, **record})
        for kind, scores in prediction.items():
            self.check(case, f"{variant}_{kind}_finite_length", scores.shape == (int(valid.sum()),) and np.isfinite(scores).all())
        return prediction


def gates(auditor, case, arrays):
    values = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32)
    valid = torch.ones(len(values), dtype=torch.bool)
    present, selected = arrays["type_valid"], arrays["type_P"]
    baseline = auditor.predict(case, "A", values, valid, present, selected, "A")
    predictions = {"A": baseline}
    for condition in "CD":
        predictions[condition] = auditor.predict(case, condition, values, valid, present, selected, condition)
    chosen = np.repeat(selected, 16)[:len(values)]
    changed = values.clone()
    changed[chosen] += torch.linspace(.5, 2.5, int(chosen.sum()))[:, None]
    for condition in "CD":
        perturbed = auditor.predict(case, f"perturb_{condition}", changed, valid, present, selected, condition)
        empty = auditor.predict(case, f"empty_{condition}", values, valid, present, np.zeros_like(selected), condition)
        for kind in baseline:
            auditor.equal(case, f"{condition}_{kind}_perturb_nonselected", predictions[condition][kind][~chosen], perturbed[kind][~chosen])
            auditor.equal(case, f"{condition}_{kind}_empty_noop", baseline[kind], empty[kind])
    with reference_intervention(auditor.model, present & ~selected):
        old = auditor.public(values, valid)
    for kind in baseline:
        auditor.equal(case, f"D_{kind}_reference_intervention", predictions["D"][kind], old[kind])
        auditor.equal(case, f"CD_{kind}_nonselected", predictions["C"][kind][~chosen], predictions["D"][kind][~chosen])
    np.savez_compressed(auditor.directory / f"arrays/{case}.npz",
                        **{f"{v}_{k}": x for v, pred in predictions.items() for k, x in pred.items()})
