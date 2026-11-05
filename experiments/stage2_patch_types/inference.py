import numpy as np
import torch

from experiments.common.artifacts import append_row
from experiments.common.metrics import detection_metrics
from experiments.stage1.inference import score_window
from experiments.stage2.inference import windows
from experiments.stage2.masking import reference_intervention
from experiments.stage2_paths.inference import check_array, predict_edges
from experiments.stage2_followup.diagnostics import ranking_decomposition, shift_statistics


def run_case(directory, config, detector, item, gate=False):
    index, series, planned, parent_path = item
    case = f"s{index:04d}_w5000"
    model, device = detector._tester.model, detector._tester.device
    if model.ts_encoder.patch_size != 16:
        raise ValueError("Frozen Patch size changed")
    with np.load(parent_path) as saved:
        parent = {k: saved[k].copy() for k in saved.files if k == "A_scores" or k == "A_logits" or "__all__C_" in k or "__all__D_" in k}
    selectors = list(planned[0][1])
    names = ["A"] + [f"{s}__{c}" for s in selectors for c in "CD"]
    chunks = {v: {k: [] for k in ("scores", "logits")} for v in names}
    membership, point_types = [], {t: [] for t in ("M", "P", "N")}
    offset = 0
    for number, (values, valid) in enumerate(windows(series, 5000)):
        types, selections = planned[number]
        size, present = int(valid.sum()), types["valid"]
        identity = {"case": case, "window_index": number}
        def check(name, a, b):
            check_array(directory, config, {**identity, "test": name}, a, b)
        baseline, _ = predict_edges(model, values, valid, present, types["U"], "A", "all", device,
                                    directory, {**identity, "variant": "A"})
        official = score_window(model, values, valid, device)
        for kind in baseline:
            check(f"A_public_{kind}", official[kind], baseline[kind])
            check(f"A_parent_{kind}", parent[f"A_{kind}"][offset:offset + size], baseline[kind])
        predictions = {"A": baseline}
        for selector, selected in selections.items():
            selected_points = np.repeat(selected, 16)[:size]
            for condition in "CD":
                name = f"{selector}__{condition}"
                if not selected.any():
                    prediction = baseline
                    append_row(directory / "noops.csv", {**identity, "variant": name, "reason": "empty_selection"})
                else:
                    prediction, _ = predict_edges(model, values, valid, present, selected, condition, "all", device,
                                                   directory, {**identity, "variant": name})
                predictions[name] = prediction
                for kind in prediction:
                    if not selected.any():
                        check(f"{name}_noop_{kind}", baseline[kind], prediction[kind])
                    if selector.startswith("U_"):
                        old_selector = selector[2:]
                        check(f"{name}_parent_{kind}", parent[f"{old_selector}__all__{condition}_{kind}"][offset:offset + size], prediction[kind])
            for kind in baseline:
                check(f"{selector}_CD_nonselected_{kind}", predictions[f"{selector}__C"][kind][~selected_points],
                      predictions[f"{selector}__D"][kind][~selected_points])
            if gate:
                with reference_intervention(model, present & ~selected):
                    old = score_window(model, values, valid, device)
                for kind in old:
                    check(f"{selector}_D_legacy_{kind}", old[kind], predictions[f"{selector}__D"][kind])
                changed = values.clone()
                selected_valid = np.repeat(selected, 16)[:len(valid)] & valid.numpy()
                changed[selected_valid] += torch.linspace(.5, 2.5, int(selected_valid.sum()))[:, None]
                for condition in "CD":
                    pred, _ = predict_edges(model, changed, valid, present, selected, condition, "all", device,
                                            directory, {**identity, "variant": f"{selector}_perturb_{condition}"})
                    for kind in pred:
                        check(f"{selector}_{condition}_perturb_nonselected_{kind}",
                              predictions[f"{selector}__{condition}"][kind][~selected_points], pred[kind][~selected_points])
        for variant, pred in predictions.items():
            for kind, scores in pred.items():
                chunks[variant][kind].append(scores)
        for t in point_types:
            point_types[t].append(np.repeat(types[t], 16)[:size])
        membership.append(np.full(size, number))
        offset += size
    if offset != len(series.values):
        raise ValueError("Window assembly lost input points")
    predictions = {v: {k: np.concatenate(x) for k, x in p.items()} for v, p in chunks.items()}
    np.savez_compressed(directory / "arrays" / f"{case}.npz", **{f"{v}_{k}": x for v, p in predictions.items() for k, x in p.items()})
    if not gate:
        record_metrics(directory, case, series, predictions, np.concatenate(membership),
                       {t: np.concatenate(x) for t, x in point_types.items()}, planned)
    return len(predictions)


def record_metrics(directory, case, series, predictions, membership, point_types, planned):
    base = {"case": case, "series": series.name, "kind": series.source["kind"],
            "source": series.name.split("_")[1] if series.source["kind"] == "tsb" else "controlled",
            "generator_seed": series.source.get("seed"), "placement": series.source.get("placement"),
            "anomaly_count": int(series.labels.sum()), "length": len(series.values),
            "mixed_patches": sum(int(t["M"].sum()) for t, _ in planned.values()),
            "pure_patches": sum(int(t["P"].sum()) for t, _ in planned.values())}
    point_groups = {"anomaly": series.labels == 1, "normal": series.labels == 0,
                    "M_anomaly": point_types["M"] & (series.labels == 1),
                    "M_normal": point_types["M"] & (series.labels == 0),
                    "P_anomaly": point_types["P"], "N_normal": point_types["N"]}
    for variant, prediction in predictions.items():
        for kind, scores in prediction.items():
            append_row(directory / "metrics.csv", {**base, "variant": variant, "score_kind": kind,
                       **detection_metrics(series.labels, scores)})
            append_row(directory / "ranking.csv", {**base, "variant": variant, "score_kind": kind,
                       **ranking_decomposition(series.labels, scores, membership)})
            if variant != "A":
                for name, chosen in point_groups.items():
                    append_row(directory / "score_shifts.csv", {**base, "variant": variant, "score_kind": kind,
                               "group": name, **shift_statistics(predictions["A"][kind], scores, chosen)})
