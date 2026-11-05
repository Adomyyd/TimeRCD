import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.common.artifacts import append_row, sha256
from experiments.stage1.inference import score_window
from experiments.stage2_alignment_controls.inference import predict as legacy_predict
from experiments.stage2_clean_reference.data import load_input
from experiments.stage2_paths.attention import edge_mask
from experiments.stage2_paths.inference import check_array
from experiments.stage2_rank_localization.diagnostics import point_groups, readout
from time_rcd._inference import _WindowDataset
from .attention import attention_depth


def predict(model, values, valid, present, selected, depth, device, capture=False):
    with attention_depth(model, present, selected, depth, capture=capture) as records:
        prediction = score_window(model, values, valid, device)
    return prediction, records


def run_case(input_directory, output, config, model, device, meta, parent_metrics, audit, gate=False):
    arrays, values, _, valid = load_input(input_directory, meta)
    path = input_directory / meta["selection_snapshot"]
    if sha256(path) != meta["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    with np.load(path) as saved:
        selections = {k: saved[k].copy() for k in saved.files}
    fixed, roles = point_groups(meta, arrays, selections, config["patch_size"], config["random_seeds"])
    present = arrays["type_valid"]
    layers = len(model.ts_encoder.transformer_encoder.layers)
    if layers != config["expected_layers"] or config["depths"][-1] != layers:
        raise ValueError("Model depth does not match protocol")

    def compare(name, a, b, exact=False):
        def convert(x):
            x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            return x.astype(np.int8) if x.dtype == bool else x
        check_array(output, {**config, "atol": 0, "rtol": 0} if exact else config,
                    {"case": meta["case"], "test": name}, convert(a), convert(b))

    public_values, public_valid = _WindowDataset(arrays["corrupt"], config["window"])[0]
    compare("official_normalization", values, public_values, True)
    compare("official_valid", valid, public_valid, True)
    official = score_window(model, values, valid, device)
    baseline, original = predict(model, values, valid, present, selections["full"], 0, device, capture=True)
    for kind in baseline:
        compare(f"A_official/{kind}", baseline[kind], official[kind])
    predictions = {"A": baseline}
    for selector in ("keep_first", "keep_last"):
        selected = selections[selector]
        for depth in config["depths"]:
            name = f"{selector}_d{depth}"
            prediction, records = predict(model, values, valid, present, selected, depth, device, capture=True)
            predictions[name] = prediction
            if len(records) != layers:
                raise ValueError("Missing layer records")
            for index, record in enumerate(records):
                condition = "C" if index >= layers - depth else "A"
                expected_mask = hashlib.sha256(edge_mask(present, selected, condition).tobytes()).hexdigest()
                audit.check(f"{meta['case']}/{name}/layer{index}/mask",
                            record["condition"] == condition and record["mask_sha256"] == expected_mask, True, exact=True)
                append_row(output / "layers.csv", {"case": meta["case"], "variant": name,
                           **{k: v for k, v in record.items() if k != "capture"}})
                if index <= layers - depth:
                    for key in ("input", "q", "k", "v"):
                        compare(f"{name}/layer{index}/{key}", original[index]["capture"][key], record["capture"][key], True)
            if gate and depth in (1, layers):
                legacy, _ = legacy_predict(model, values, valid, present, selected, "C", "last" if depth == 1 else "all", device)
                for kind in baseline:
                    compare(f"legacy/{name}/{kind}", prediction[kind], legacy[kind], True)
    if gate:
        for depth in config["depths"]:
            empty, _ = predict(model, values, valid, present, np.zeros_like(present), depth, device)
            for kind in baseline:
                compare(f"empty/d{depth}/{kind}", baseline[kind], empty[kind], True)
    endpoints = {"A": "A", **{f"{selector}_d{depth}": f"{selector}_{scope}" for selector in ("keep_first", "keep_last")
                              for depth, scope in ((1, "last"), (layers, "all"))}}
    with np.load(Path(config["parent_run"]) / f"arrays/{meta['case']}.npz") as old:
        for new, previous in endpoints.items():
            for kind in baseline:
                compare(f"parent/{new}/{kind}", predictions[new][kind], old[f"{previous}_{kind}"])
    np.savez_compressed(output / "arrays" / f"{meta['case']}.npz",
                        **{f"{v}_{k}": p for v, pred in predictions.items() for k, p in pred.items()})
    for variant, prediction in predictions.items():
        selector = "A" if variant == "A" else variant.rsplit("_d", 1)[0]
        for kind, scores in prediction.items():
            result = readout(arrays["labels"], scores, {**fixed, **roles[selector]}, config["recalls"], audit,
                             f"{meta['case']}/{variant}/{kind}")
            if variant in endpoints:
                previous = parent_metrics.loc[(meta["case"], endpoints[variant], kind)]
                columns = [c for c in result if c != "metric_status"]
                actual = np.array([result[c] for c in columns], dtype=float)
                expected = previous[columns].to_numpy(dtype=float)
                audit.check(f"{meta['case']}/{variant}/{kind}/E10_missing", np.isnan(actual), np.isnan(expected), exact=True)
                mask = ~np.isnan(actual)
                audit.check(f"{meta['case']}/{variant}/{kind}/E10_readouts", actual[mask], expected[mask])
            base = {k: meta[k] for k in ("case", "seed", "phase", "shape", "event_length", "offset")}
            append_row(output / "metrics.csv", {**base, "score_kind": kind, "variant": variant, **result})
    return len(predictions)
