import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.artifacts import append_row, sha256, write_json
from experiments.common.metrics import detection_metrics
from .diagnostics import ranking_decomposition, shift_statistics


def audit_saved(directory, parent):
    sources = json.loads((parent / "data_manifest.json").read_text())
    metadata = pd.read_csv(parent / "metrics.csv")
    cases = metadata[(metadata.scope == "full_fallback") & (metadata.score_kind == "scores") &
                     (metadata.variant == "baseline")]
    windows = pd.read_csv(parent / "windows.csv")
    fingerprints = {}
    for filename in ("manifest.json", "data_manifest.json", "metrics.csv", "windows.csv"):
        fingerprints[filename] = sha256(parent / filename)
    for source in sources:
        path = parent / source["input_snapshot"]
        if sha256(path) != source["input_sha256"]:
            raise ValueError(f"Changed input: {path}")
        fingerprints[source["input_snapshot"]] = source["input_sha256"]
    for row in cases.itertuples(index=False):
        index = int(row.case.split("_")[0][1:])
        source = next(s for s in sources if s["index"] == index)
        with np.load(parent / source["input_snapshot"]) as raw:
            labels = raw["labels"].copy()
        membership = np.full(len(labels), -1, dtype=int)
        case_windows = windows[windows.case == row.case].sort_values("window_index")
        for win in case_windows.itertuples(index=False):
            membership[win.offset:win.offset + win.valid_points] = win.window_index
        if (membership < 0).any():
            raise ValueError("Incomplete saved window coverage")
        base = {"case": row.case, "series": row.series, "kind": row.kind, "window": row.window,
                "anomaly_count": row.anomaly_count, "generator_seed": source["source"].get("seed"),
                "placement": source["source"].get("placement"), "density_bin": row.density_bin}
        filename = f"arrays/{row.case}.npz"
        fingerprints[filename] = sha256(parent / filename)
        with np.load(parent / filename) as arrays:
            for kind in ("scores", "logits"):
                reference = arrays[f"baseline_{kind}"]
                candidate = arrays[f"oracle_{kind}"]
                for name, selected in (("normal", labels == 0), ("anomaly", labels == 1)):
                    append_row(directory / "original_class_shifts.csv", {
                        **base, "score_kind": kind, "class": name,
                        **shift_statistics(reference, candidate, selected)})
                decomposed = {}
                for name, scores in (("baseline", reference), ("oracle", candidate)):
                    result = ranking_decomposition(labels, scores, membership)
                    decomposed[name] = result
                    append_row(directory / "original_ranking.csv", {
                        **base, "score_kind": kind, "variant": name, **result})
                append_row(directory / "original_ranking_changes.csv", {
                    **base, "score_kind": kind,
                    "delta_auc": decomposed["oracle"]["auc"] - decomposed["baseline"]["auc"],
                    "within_delta_contribution": decomposed["oracle"]["within_contribution"] - decomposed["baseline"]["within_contribution"],
                    "across_delta_contribution": decomposed["oracle"]["across_contribution"] - decomposed["baseline"]["across_contribution"],
                    "within_pairs": decomposed["baseline"]["within_pairs"],
                    "across_pairs": decomposed["baseline"]["across_pairs"],
                })
            for win in case_windows.itertuples(index=False):
                chosen = membership == win.window_index
                for name in ("baseline", "oracle"):
                    append_row(directory / "original_window_metrics.csv", {
                        **base, "window_index": win.window_index, "eligible": win.eligible,
                        "points": int(chosen.sum()), "anomaly_points": int(labels[chosen].sum()),
                        "variant": name, **detection_metrics(labels[chosen], arrays[f"{name}_scores"][chosen])})
    write_json(directory / "parent_artifacts.sha256.json", fingerprints)
    return len(cases)
