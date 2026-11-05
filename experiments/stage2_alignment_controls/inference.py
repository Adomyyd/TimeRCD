import numpy as np
import torch

from experiments.common.artifacts import append_row
from experiments.common.metrics import detection_metrics
from experiments.stage1.inference import score_window
from experiments.stage2_clean_reference.data import load_input
from experiments.stage2_followup.diagnostics import shift_statistics
from experiments.stage2_paths.attention import attention_edges
from experiments.stage2_paths.inference import check_array
from experiments.stage2_pc_localization.diagnostics import pair_decomposition, point_partition
from time_rcd._inference import _WindowDataset


def predict(model, values, valid, present, selected, condition, scope, device, capture=False):
    with attention_edges(model, present, selected, condition, scope, capture=capture) as records:
        prediction = score_window(model, values, valid, device)
    return prediction, records


def record_metrics(directory, meta, labels, predictions, patch_size):
    groups, membership, _ = point_partition(labels, len(labels), patch_size)
    identity = {k: meta[k] for k in ("case", "seed", "phase", "shape", "event_length", "offset", "M", "P", "common_patches")}
    for variant, prediction in predictions.items():
        for kind, scores in prediction.items():
            base = {**identity, "variant": variant, "score_kind": kind}
            metrics = detection_metrics(labels, scores)
            pairs = [p for p in pair_decomposition(labels, scores, groups, membership) if p["scope"] == "all"]
            if abs(sum(p["contribution"] for p in pairs) - metrics["roc_auc"]) > 1e-12:
                raise ValueError("AUC decomposition failed")
            for pair in pairs:
                append_row(directory / "pairs.csv", {**base, **pair})
                metrics[f"{pair['pair']}_auc"] = pair["auc"]
                metrics[f"{pair['pair']}_contribution"] = pair["contribution"]
            for name, mask in groups.items():
                shift = shift_statistics(predictions["A"][kind], scores, mask)
                append_row(directory / "shifts.csv", {**base, "group": name, **shift})
                metrics[f"{name}_mean"] = shift["candidate_mean"]
            append_row(directory / "metrics.csv", {**base, **metrics})


def run_case(input_directory, output_directory, config, model, device, meta, gate=False):
    arrays, values, _, valid = load_input(input_directory, meta)
    present = arrays["type_valid"]
    def compare(name, a, b, exact=False):
        def convert(x):
            result = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            return result.astype(np.int8) if result.dtype == bool else result
        check_array(output_directory, {**config, "atol": 0, "rtol": 0} if exact else config,
                    {"case": meta["case"], "test": name}, convert(a), convert(b))
    public_values, public_valid = _WindowDataset(arrays["corrupt"], config["window"])[0]
    compare("official_normalization", public_values, values, exact=True)
    compare("official_valid", public_valid, valid, exact=True)
    official = score_window(model, values, valid, device)
    baseline, original = predict(model, values, valid, present, arrays["type_P"], "A", "all", device, capture=True)
    for kind in baseline:
        compare(f"A_official_{kind}", official[kind], baseline[kind])
    predictions = {"A": baseline}
    for selector, selected in (("full", arrays["type_P"]), ("common", arrays["common"])):
        for scope in ("last", "all"):
            name = f"{selector}_{scope}"
            prediction, records = predict(model, values, valid, present, selected, "C", scope, device, capture=scope == "last")
            predictions[name] = prediction
            if scope == "last":
                for layer, (a, b) in enumerate(zip(original, records)):
                    for key in ("input", "q", "k", "v"):
                        compare(f"{name}/layer{layer}/{key}", a["capture"][key], b["capture"][key], exact=True)
    if meta["offset"] == 8:
        for scope in ("last", "all"):
            for kind in baseline:
                compare(f"offset8_full_common/{scope}/{kind}", predictions[f"full_{scope}"][kind], predictions[f"common_{scope}"][kind], exact=True)
    if gate:
        for scope in ("last", "all"):
            empty, _ = predict(model, values, valid, present, np.zeros_like(present), "C", scope, device)
            for kind in baseline:
                compare(f"empty/{scope}/{kind}", baseline[kind], empty[kind], exact=True)
    np.savez_compressed(output_directory / "arrays" / f"{meta['case']}.npz",
                        **{f"{v}_{k}": x for v, p in predictions.items() for k, x in p.items()})
    if not gate:
        record_metrics(output_directory, meta, arrays["labels"], predictions, config["patch_size"])
    return len(predictions)
