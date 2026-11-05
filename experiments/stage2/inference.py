import hashlib

import numpy as np
import torch

from experiments.common.artifacts import append_row
from experiments.common.metrics import detection_metrics, difference
from experiments.stage1.inference import score_window
from time_rcd._inference import _WindowDataset

from .masking import reference_intervention, reference_masks, random_reference_mask


def windows(series, window):
    actual = min(len(series.values), window)
    normalization = series.source.get("normalization", "official_series")
    if normalization not in ("official_series", "fixed_clean_generator_statistics"):
        raise ValueError(f"Unknown normalization: {normalization}")
    if series.source["kind"] != "controlled_synthetic" or normalization == "official_series":
        dataset = _WindowDataset(series.values, actual)
        for index in range(len(dataset)):
            yield dataset[index]
        return
    # The generator's clean statistics are fixed across anomaly counts.
    values = (series.values - series.source["normal_mean"]) / series.source["normal_std"]
    for start in range(0, len(values), actual):
        chunk = values[start:start + actual]
        valid = len(chunk)
        chunk = np.pad(chunk, ((0, actual - valid), (0, 0)), mode="edge")
        yield torch.tensor(chunk, dtype=torch.float32), torch.arange(actual) < valid


def run_case(directory, config, detector, series, index, window):
    detector._ensure_tester(1)
    model, device = detector._tester.model, detector._tester.device
    patch_size = model.ts_encoder.patch_size
    case = f"s{index:04d}_w{window}"
    names = ["baseline", "oracle"] + [f"random_{seed}" for seed in config["random_seeds"]]
    chunks = {name: {kind: [] for kind in ("scores", "logits")} for name in names}
    masks, eligible_chunks = {}, []
    offset = 0
    no_op_max, no_op_pass = 0.0, True
    for number, (values, valid) in enumerate(windows(series, window)):
        size = int(valid.sum())
        labels = np.pad(series.labels[offset:offset + size], (0, len(valid) - size))
        present, clean, mixed = reference_masks(labels, valid.numpy(), patch_size)
        baseline = score_window(model, values, valid, device)
        with reference_intervention(model, present):
            replay = score_window(model, values, valid, device)
        for kind in baseline:
            delta = difference(baseline[kind], replay[kind], config["atol"], config["rtol"])
            no_op_max = max(no_op_max, delta["max_abs"])
            no_op_pass = no_op_pass and delta["close"]
        eligible = bool(clean.any())
        eligible_chunks.append(np.full(size, eligible))
        masks[f"w{number}_valid"] = present
        masks[f"w{number}_clean"] = clean
        predictions = {"baseline": baseline}
        for name in names[1:]:
            keep = clean
            if name.startswith("random_") and eligible:
                digest = hashlib.sha256(f"{name}:{series.name}:{window}:{number}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                keep = random_reference_mask(present, clean, rng)
            masks[f"w{number}_{name}"] = keep if eligible else present
            # All-anomalous Patch windows have no defined clean-reference oracle.
            # Keep baseline there in every arm and report matched eligible points separately.
            if not eligible or np.array_equal(keep, present):
                predictions[name] = baseline
            else:
                with reference_intervention(model, keep):
                    predictions[name] = score_window(model, values, valid, device)
        for name, prediction in predictions.items():
            for kind, result in prediction.items():
                chunks[name][kind].append(result)
        append_row(directory / "windows.csv", {
            "case": case, "window_index": number, "offset": offset, "valid_points": size,
            "anomaly_points": int(labels.sum()), "valid_patches": int(present.sum()),
            "clean_patches": int(clean.sum()), "blocked_patches": int((present & ~clean).sum()),
            "mixed_patches": int(mixed.sum()), "eligible": eligible,
        })
        offset += size
    predictions = {name: {kind: np.concatenate(values) for kind, values in parts.items()}
                   for name, parts in chunks.items()}
    eligible = np.concatenate(eligible_chunks)
    if offset != len(series.values):
        raise RuntimeError("Window assembly lost points")
    arrays = {f"{name}_{kind}": value for name, pred in predictions.items() for kind, value in pred.items()}
    arrays["eligible"] = eligible
    np.savez_compressed(directory / "arrays" / f"{case}.npz", **arrays)
    np.savez_compressed(directory / "arrays" / f"{case}_masks.npz", **masks)
    parity = {"case": case, "noop_max_abs": no_op_max, "noop_close": bool(no_op_pass)}
    if series.source["kind"] != "controlled_synthetic":
        scores, logits = detector.predict(series.values, return_logits=True)
        for kind, reference in (("scores", scores), ("logits", logits)):
            delta = difference(reference, predictions["baseline"][kind], config["atol"], config["rtol"])
            parity[f"public_{kind}_max_abs"] = delta["max_abs"]
            parity[f"public_{kind}_close"] = delta["close"]
            no_op_pass = no_op_pass and delta["close"]
    else:
        for kind in ("scores", "logits"):
            parity[f"public_{kind}_max_abs"] = None
            parity[f"public_{kind}_close"] = None
    append_row(directory / "checks.csv", parity)
    ratio = float(series.labels.mean())
    base = {"case": case, "series": series.name, "window": window,
            "kind": series.source["kind"], "length": len(series.values), "anomaly_ratio": ratio,
            "density_bin": "low" if ratio <= .01 else ("medium" if ratio <= .1 else "high"),
            "source": series.name.split("_")[1] if series.source["kind"] == "tsb" else "controlled",
            "generator_seed": series.source.get("seed"), "placement": series.source.get("placement"),
            "anomaly_count": int(series.labels.sum()), "eligible_points": int(eligible.sum())}
    for scope, selected in (("full_fallback", np.ones(len(eligible), dtype=bool)), ("eligible", eligible)):
        if not selected.any():
            continue
        for kind in ("scores", "logits"):
            baseline_metrics = detection_metrics(series.labels[selected], predictions["baseline"][kind][selected])
            for name, prediction in predictions.items():
                metrics = detection_metrics(series.labels[selected], prediction[kind][selected])
                append_row(directory / "metrics.csv", {
                    **base, "scope": scope, "score_kind": kind, "variant": name, **metrics,
                    "delta_ap": None if metrics["average_precision"] is None else
                    metrics["average_precision"] - baseline_metrics["average_precision"],
                })
    return bool(no_op_pass)
