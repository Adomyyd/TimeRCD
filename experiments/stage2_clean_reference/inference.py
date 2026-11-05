import numpy as np
import torch

from experiments.common.artifacts import append_row
from experiments.common.metrics import detection_metrics
from experiments.stage1.inference import score_window
from experiments.stage2_followup.diagnostics import shift_statistics
from experiments.stage2_paths.attention import attention_edges
from experiments.stage2_paths.inference import check_array
from experiments.stage2_pc_localization.diagnostics import pair_decomposition, point_partition
from time_rcd._inference import _WindowDataset
from .attention import last_reference
from .data import load_input


def predict(model, values, valid, present, selected, mode, device, donor=None):
    with last_reference(model, present, selected, mode, donor) as record:
        result = score_window(model, values, valid, device)
    return result, record


def record_metrics(directory, meta, labels, predictions, patch_size):
    groups, membership, _ = point_partition(labels, len(labels), patch_size)
    identity = {k: meta[k] for k in ("case", "seed", "shape", "event_length", "offset", "M", "P")}
    for variant, prediction in predictions.items():
        for kind, scores in prediction.items():
            base = {**identity, "variant": variant, "score_kind": kind}
            metrics = detection_metrics(labels, scores)
            decomposition = pair_decomposition(labels, scores, groups, membership)
            selected = [p for p in decomposition if p["scope"] == "all"]
            if abs(sum(p["contribution"] for p in selected) - metrics["roc_auc"]) > 1e-12:
                raise ValueError("AUC decomposition failed")
            for pair in selected:
                append_row(directory / "pairs.csv", {**base, **pair})
                metrics[f"{pair['pair']}_auc"] = pair["auc"]
                metrics[f"{pair['pair']}_contribution"] = pair["contribution"]
            for name, chosen in groups.items():
                shift = shift_statistics(predictions["A"][kind], scores, chosen)
                append_row(directory / "shifts.csv", {**base, "group": name, **shift})
                metrics[f"{name}_mean"] = shift["candidate_mean"]
            append_row(directory / "metrics.csv", {**base, **metrics})


def run_case(input_directory, output_directory, config, model, device, meta, gate=False):
    arrays, values, clean, valid = load_input(input_directory, meta)
    present, selected = arrays["type_valid"], arrays["type_P"]
    def compare(name, original, candidate, exact=False):
        def array(x):
            result = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            return result.astype(np.int8) if result.dtype == bool else result
        settings = {**config, "atol": 0, "rtol": 0} if exact else config
        check_array(output_directory, settings, {"case": meta["case"], "test": name}, array(original), array(candidate))
    official_values, official_valid = _WindowDataset(arrays["corrupt"], config["window"])[0]
    compare("official_normalized_input", official_values, values, exact=True)
    compare("official_point_valid", official_valid, valid, exact=True)
    official = score_window(model, values, valid, device)
    baseline, reference = predict(model, values, valid, present, selected, "A", device)
    _, donor = predict(model, clean, valid, present, selected, "A", device)
    predictions = {"A": baseline}
    for kind in baseline:
        compare(f"A_official_{kind}", official[kind], baseline[kind])
    for mode in ("C", "V", "KV"):
        prediction, record = predict(model, values, valid, present, selected, mode, device, donor)
        predictions[mode] = prediction
        for name in ("input", "q", "k", "v"):
            compare(f"{mode}_fixed_{name}", reference[name], record[name], exact=True)
        if mode == "V":
            compare("V_fixed_weights", reference["weights"], record["weights"], exact=True)
        if mode == "KV":
            compare("KV_diagonal_logits", torch.diagonal(reference["logits"], dim1=-2, dim2=-1),
                    torch.diagonal(record["logits"], dim1=-2, dim2=-1), exact=True)
            untouched = ~record["edges"].expand_as(record["logits"])
            compare("KV_untouched_logits", reference["logits"][untouched], record["logits"][untouched], exact=True)
        if not selected.any():
            for kind in baseline:
                compare(f"{mode}_empty_P_{kind}", baseline[kind], prediction[kind], exact=True)
    with attention_edges(model, present, selected, "C", "last"):
        old_c = score_window(model, values, valid, device)
    with attention_edges(model, present, selected, "C", "all"):
        predictions["C_all"] = score_window(model, values, valid, device)
    for kind in baseline:
        compare(f"C_legacy_{kind}", old_c[kind], predictions["C"][kind])
        if not selected.any():
            compare(f"C_all_empty_P_{kind}", official[kind], predictions["C_all"][kind])
    for mode in ("V", "KV"):
        replay, _ = predict(model, values, valid, present, selected, mode, device, "self")
        for kind in baseline:
            compare(f"{mode}_self_replay_{kind}", baseline[kind], replay[kind], exact=True)
        if gate:
            empty, _ = predict(model, values, valid, present, np.zeros_like(selected), mode, device, donor)
            for kind in baseline:
                compare(f"{mode}_explicit_empty_{kind}", baseline[kind], empty[kind], exact=True)
    if gate:
        np.savez_compressed(output_directory / "arrays" / f"{meta['case']}_last_layer.npz",
                            **{f"{branch}_{key}": record[key].cpu().numpy()
                               for branch, record in (("corrupt", reference), ("clean", donor))
                               for key in ("input", "q", "k", "v")})
    for variant, prediction in predictions.items():
        for kind, output in prediction.items():
            if not np.isfinite(output).all() or output.shape != arrays["labels"].shape:
                raise ValueError(f"Invalid {variant}/{kind} output")
    np.savez_compressed(output_directory / "arrays" / f"{meta['case']}.npz",
                        **{f"{variant}_{kind}": output for variant, prediction in predictions.items() for kind, output in prediction.items()})
    if not gate:
        record_metrics(output_directory, meta, arrays["labels"], predictions, config["patch_size"])
    return len(predictions)
