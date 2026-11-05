import numpy as np
import torch

from experiments.common.artifacts import append_row
from experiments.common.metrics import detection_metrics, difference
from experiments.stage1.inference import score_window
from experiments.stage2.inference import windows
from experiments.stage2.masking import reference_intervention, reference_masks
from experiments.stage2_followup.diagnostics import ranking_decomposition, shift_statistics
from .attention import attention_edges


class CheckFailure(RuntimeError):
    pass


def check_array(directory, config, identity, reference, candidate):
    result = difference(np.asarray(reference), np.asarray(candidate), config["atol"], config["rtol"])
    append_row(directory / "checks.csv", {**identity, **result})
    if not result["close"]:
        np.savez_compressed(directory / "failed_check.npz", reference=reference, candidate=candidate)
        raise CheckFailure(f"Numerical gate failed: {identity}; {result}")


def predict_edges(model, values, point_valid, present, selected, condition, scope, device,
                  directory, identity, capture=False):
    with attention_edges(model, present, selected, condition, scope, capture=capture) as records:
        prediction = score_window(model, values, point_valid, device)
    for record in records:
        append_row(directory / "layer_masks.csv", {**identity, **{k: v for k, v in record.items() if k != "capture"}})
    return prediction, records


def record_metrics(directory, series, case, predictions, membership, groups, eligible):
    base = {"case": case, "series": series.name, "kind": series.source["kind"],
            "source": series.name.split("_")[1] if series.source["kind"] == "tsb" else "controlled",
            "generator_seed": series.source.get("seed"), "placement": series.source.get("placement"),
            "anomaly_count": int(series.labels.sum()), "length": len(series.values)}
    for variant, prediction in predictions.items():
        for kind, values in prediction.items():
            for scope, chosen in (("full", np.ones(len(values), dtype=bool)), ("eligible", eligible)):
                if not chosen.any():
                    continue
                append_row(directory / "metrics.csv", {**base, "variant": variant, "score_kind": kind,
                           "scope": scope, **detection_metrics(series.labels[chosen], values[chosen])})
            append_row(directory / "ranking.csv", {**base, "variant": variant, "score_kind": kind,
                       **ranking_decomposition(series.labels, values, membership)})
            if variant == "A":
                continue
            selection = variant.split("__")[0]
            selected_points = groups[selection]
            for name, chosen in (
                ("anomaly", series.labels == 1), ("normal", series.labels == 0),
                ("selected_anomaly", selected_points & (series.labels == 1)),
                ("selected_normal", selected_points & (series.labels == 0)),
                ("unselected_normal", ~selected_points & (series.labels == 0)),
            ):
                append_row(directory / "score_shifts.csv", {**base, "variant": variant, "score_kind": kind,
                           "group": name, **shift_statistics(predictions["A"][kind], values, chosen)})


def run_case(directory, config, detector, item, gate=False):
    index, series, saved_masks, parent_predictions = item
    model, device = detector._tester.model, detector._tester.device
    case = f"s{index:04d}_w5000"
    selectors = ["label"] + [f"random_{seed}" for seed in config["random_seeds"]]
    variants = ["A"] + [f"{s}__{scope}__{c}" for s in selectors for scope in ("last", "all") for c in "BCD"]
    chunks = {v: {k: [] for k in ("scores", "logits")} for v in variants}
    memberships, eligibility = [], []
    groups = {s: [] for s in selectors}
    offset = 0
    for number, (values, valid) in enumerate(windows(series, config["window"])):
        size = int(valid.sum())
        labels = np.pad(series.labels[offset:offset + size], (0, len(valid) - size))
        present, clean, mixed = reference_masks(labels, valid.numpy(), model.ts_encoder.patch_size)
        if not np.array_equal(present, saved_masks[f"w{number}_valid"]) or not np.array_equal(clean, saved_masks[f"w{number}_clean"]):
            raise ValueError("Label/valid Patch masks differ from frozen parent")
        selections = {"label": present & ~clean}
        for name in selectors[1:]:
            keep = saved_masks[f"w{number}_{name}"]
            if keep.dtype != bool or keep.shape != present.shape or (keep & ~present).any():
                raise ValueError("Invalid saved random selection")
            selections[name] = present & ~keep
            if selections[name].sum() != selections["label"].sum():
                raise ValueError("Random selection count differs from label selection")
        if not clean.any():
            raise CheckFailure("Unexpected empty-reference window absent in the parent 5000 run; inspect before fallback")
        identity = {"case": case, "window_index": number}
        def compare(test, reference, candidate):
            check_array(directory, config, {**identity, "test": test}, reference, candidate)
        baseline, a_records = predict_edges(model, values, valid, present, selections["label"], "A", "all",
                                            device, directory, {**identity, "variant": "A"}, capture=gate)
        official = score_window(model, values, valid, device)
        for kind in baseline:
            compare(f"A_public_{kind}", official[kind], baseline[kind])
            compare(f"A_parent_{kind}", parent_predictions[f"baseline_{kind}"][offset:offset + size], baseline[kind])
        predictions = {"A": baseline}
        for selector, selected in selections.items():
            point_selected = np.repeat(selected, model.ts_encoder.patch_size)[:size]
            groups[selector].append(point_selected)
            for scope in ("last", "all"):
                for condition in "BCD":
                    variant = f"{selector}__{scope}__{condition}"
                    pred, records = predict_edges(model, values, valid, present, selected, condition, scope,
                                                  device, directory, {**identity, "variant": variant}, capture=gate)
                    predictions[variant] = pred
                    if gate and scope == "last":
                        for layer, (a, b) in enumerate(zip(a_records, records)):
                            for name in ("input", "q", "k", "v"):
                                compare(f"{variant}_layer{layer}_{name}", a["capture"][name].cpu().numpy(),
                                        b["capture"][name].cpu().numpy())
                c, d = predictions[f"{selector}__{scope}__C"], predictions[f"{selector}__{scope}__D"]
                for kind in c:
                    compare(f"{selector}_{scope}_CD_nonselected_{kind}", c[kind][~point_selected], d[kind][~point_selected])
                    if scope == "last":
                        compare(f"{selector}_last_AB_nonselected_{kind}", baseline[kind][~point_selected],
                                predictions[f"{selector}__last__B"][kind][~point_selected])
            old_name = "oracle" if selector == "label" else selector
            for kind in baseline:
                compare(f"{selector}_D_parent_{kind}", parent_predictions[f"{old_name}_{kind}"][offset:offset + size],
                        predictions[f"{selector}__all__D"][kind])
            if gate:
                with reference_intervention(model, present & ~selected):
                    old = score_window(model, values, valid, device)
                for kind in old:
                    compare(f"{selector}_D_old_forward_{kind}", old[kind], predictions[f"{selector}__all__D"][kind])
                changed = values.clone()
                changed_points = np.repeat(selected, model.ts_encoder.patch_size)[:len(valid)] & valid.numpy()
                changed[changed_points] += torch.linspace(.5, 2.5, int(changed_points.sum()))[:, None]
                for condition in "CD":
                    perturbed, _ = predict_edges(model, changed, valid, present, selected, condition, "all", device,
                                                 directory, {**identity, "variant": f"{selector}_perturb_{condition}"})
                    for kind in baseline:
                        compare(f"{selector}_{condition}_perturb_nonselected_{kind}",
                                predictions[f"{selector}__all__{condition}"][kind][~point_selected], perturbed[kind][~point_selected])
        for variant, pred in predictions.items():
            for kind, output in pred.items():
                chunks[variant][kind].append(output)
        append_row(directory / "windows.csv", {**identity, "offset": offset, "valid_points": size,
                   "valid_patches": int(present.sum()), "selected_patches": int(selections["label"].sum()),
                   "mixed_patches": int(mixed.sum()), "fallback": False})
        memberships.append(np.full(size, number))
        eligibility.append(np.ones(size, dtype=bool))
        offset += size
    if offset != len(series.values):
        raise ValueError("Lost input points")
    predictions = {v: {k: np.concatenate(x) for k, x in p.items()} for v, p in chunks.items()}
    eligible = np.concatenate(eligibility)
    np.savez_compressed(directory / "arrays" / f"{case}.npz", eligible=eligible,
                        **{f"{v}_{k}": x for v, p in predictions.items() for k, x in p.items()})
    if not gate:
        record_metrics(directory, series, case, predictions, np.concatenate(memberships),
                       {s: np.concatenate(parts) for s, parts in groups.items()}, eligible)
    return len(predictions)
