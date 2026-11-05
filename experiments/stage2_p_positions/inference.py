from pathlib import Path

import numpy as np
import torch

from experiments.common.artifacts import sha256
from experiments.stage1.inference import score_window
from experiments.stage2_alignment_controls.inference import predict, record_metrics
from experiments.stage2_clean_reference.data import load_input
from experiments.stage2_paths.inference import check_array
from time_rcd._inference import _WindowDataset
from .data import select_positions


def run_case(input_directory, output_directory, config, model, device, meta, gate=False):
    arrays, values, _, valid = load_input(input_directory, meta)
    path = input_directory / meta["selection_snapshot"]
    if sha256(path) != meta["selection_sha256"]:
        raise ValueError("Selection snapshot changed")
    expected, kept = select_positions(meta, arrays, config["patch_size"], config["random_seeds"])
    with np.load(path) as saved:
        selections = {name: saved[name].copy() for name in expected}
        for name, mask in expected.items():
            if not np.array_equal(mask, selections[name]):
                raise ValueError("Frozen selection differs from protocol")
        for name, indices in kept.items():
            if not np.array_equal(saved[f"kept_{name}"], indices):
                raise ValueError("Saved retained indices mismatch")

    def compare(name, a, b, exact=False):
        def convert(value):
            result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            return result.astype(np.int8) if result.dtype == bool else result
        check_array(output_directory, {**config, "atol": 0, "rtol": 0} if exact else config,
                    {"case": meta["case"], "test": name}, convert(a), convert(b))

    public_values, public_valid = _WindowDataset(arrays["corrupt"], config["window"])[0]
    compare("official_normalization", public_values, values, exact=True)
    compare("official_valid", public_valid, valid, exact=True)
    official = score_window(model, values, valid, device)
    present = arrays["type_valid"]
    baseline, original = predict(model, values, valid, present, selections["full"], "A", "all", device, capture=True)
    for kind in baseline:
        compare(f"A_official_{kind}", official[kind], baseline[kind])
    predictions = {"A": baseline}
    for selector, selected in selections.items():
        for scope in ("last", "all"):
            name = f"{selector}_{scope}"
            prediction, records = predict(model, values, valid, present, selected, "C", scope, device, capture=scope == "last")
            predictions[name] = prediction
            if scope == "last":
                if len(records) != len(original) or not records:
                    raise ValueError("Missing layer captures")
                for layer, (a, b) in enumerate(zip(original, records)):
                    for key in ("input", "q", "k", "v"):
                        compare(f"{name}/layer{layer}/{key}", a["capture"][key], b["capture"][key], exact=True)
    if gate:
        for scope in ("last", "all"):
            empty, _ = predict(model, values, valid, present, np.zeros_like(present), "C", scope, device)
            for kind in baseline:
                compare(f"empty/{scope}/{kind}", baseline[kind], empty[kind], exact=True)
    with np.load(Path(config["parent_run"]) / f"arrays/{meta['case']}.npz") as previous:
        for new, old in (("A", "A"), ("full_last", "full_last"), ("full_all", "full_all"),
                         ("keep_first_last", "common_last"), ("keep_first_all", "common_all")):
            for kind in baseline:
                compare(f"parent/{new}/{kind}", predictions[new][kind], previous[f"{old}_{kind}"])
    np.savez_compressed(output_directory / "arrays" / f"{meta['case']}.npz",
                        **{f"{v}_{k}": x for v, p in predictions.items() for k, x in p.items()})
    if not gate:
        record_metrics(output_directory, meta, arrays["labels"], predictions, config["patch_size"])
    return len(predictions)
