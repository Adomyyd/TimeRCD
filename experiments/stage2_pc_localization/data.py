import json

import numpy as np
import pandas as pd

from experiments.common.artifacts import append_row, sha256
from .diagnostics import point_partition


class Audit:
    def __init__(self, directory, tolerance):
        self.path = directory / "checks.csv"
        self.tolerance = tolerance
        self.count = 0

    def check(self, name, actual, expected, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        same_shape = actual.shape == expected.shape
        delta = float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0)) if same_shape else None
        close = same_shape and (np.array_equal(actual, expected) if exact else
                                np.allclose(actual, expected, atol=self.tolerance, rtol=0))
        append_row(self.path, {"test": name, "close": bool(close), "max_abs": delta})
        self.count += 1
        if not close:
            raise ValueError(f"Failed check: {name}, max_abs={delta}")


def load_parent(parent, config, audit):
    manifest = json.loads((parent / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Parent did not complete")
    audit.check("parent_window", manifest["config"]["window"], config["window"], exact=True)
    for name in ("checks.csv", "gates/checks.csv"):
        checks = pd.read_csv(parent / name)
        if checks.empty or not checks.close.eq(True).all():
            raise ValueError(f"Invalid parent checks: {name}")
        audit.check(f"parent_{name}", checks.close.to_numpy(), np.ones(len(checks), dtype=bool), exact=True)
    records = json.loads((parent / "data_manifest.json").read_text())
    records = [r for r in records if r["source"]["kind"] == "tsb"]
    audit.check("case_indices", [r["index"] for r in records], np.arange(config["expected_cases"]), exact=True)
    audit.check("with_P_count", sum(r["pure_patches"] > 0 for r in records), config["expected_with_P"], exact=True)
    if len({r["name"] for r in records}) != len(records):
        raise ValueError("Duplicate series identities")
    return manifest, records


def load_case(parent, record, config, seeds, coverage, audit):
    case = f"s{record['index']:04d}_w5000"
    for key in ("input", "selection"):
        if sha256(parent / record[f"{key}_snapshot"]) != record[f"{key}_sha256"]:
            raise ValueError(f"{case}: {key} hash mismatch")
    with np.load(parent / record["input_snapshot"]) as saved:
        labels = saved["labels"].copy()
        if len(saved["values"]) != len(labels) or not np.isfinite(saved["values"]).all():
            raise ValueError("Invalid saved input")
    groups, membership, windows = point_partition(labels, config["window"], config["patch_size"])
    selectors = ["label"] + [f"random_{s}" for s in seeds]
    with np.load(parent / record["selection_snapshot"]) as saved:
        for number, (start, size, types) in enumerate(windows):
            for key, value in types.items():
                audit.check(f"{case}/w{number}/type_{key}", saved[f"w{number}_type_{key}"], value, exact=True)
            for selector in selectors:
                for t in "MPU":
                    selection = f"{t}_{selector}"
                    mask = saved[f"w{number}_selection_{selection}"]
                    if mask.dtype != bool or mask.shape != types[t].shape or (mask & ~types["valid"]).any():
                        raise ValueError("Invalid selection")
                    audit.check(f"{case}/w{number}/{selection}/count", mask.sum(), types[t].sum(), exact=True)
                    if selector == "label":
                        audit.check(f"{case}/w{number}/{selection}/mask", mask, types[t], exact=True)
                    rows = coverage[(coverage.case == case) & (coverage.window_index == number) &
                                    (coverage.selection == selection)]
                    if len(rows) != 1 or rows.iloc[0]["series"] != record["name"]:
                        raise ValueError("Missing, duplicate or misidentified coverage")
                    row = rows.iloc[0]
                    expected = [start, size, int(mask.sum()), int(types["anomaly_counts"][mask].sum()),
                                int((types["valid_counts"] - types["anomaly_counts"])[mask].sum()),
                                int((types["partial"] & mask).sum()), int(types["valid"].sum())]
                    columns = ["offset", "valid_points", "selected_patches", "anomaly_points_covered",
                               "normal_points_covered", "partial_patches", "valid_patches"]
                    audit.check(f"{case}/w{number}/{selection}/coverage", row[columns].to_numpy(), expected, exact=True)
    for t, name in (("M", "mixed_patches"), ("P", "pure_patches"), ("partial", "partial_patches")):
        audit.check(f"{case}/{name}", sum(int(w[2][t].sum()) for w in windows), record[name], exact=True)
    with np.load(parent / f"arrays/{case}.npz") as saved:
        names = ["A"] + [f"P_{s}__C" for s in selectors]
        predictions = {name: {kind: saved[f"{name}_{kind}"].copy() for kind in ("scores", "logits")} for name in names}
    for name, prediction in predictions.items():
        for kind, scores in prediction.items():
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise ValueError("Invalid prediction")
            if not record["pure_patches"]:
                audit.check(f"{case}/{name}/{kind}/noop", scores, predictions["A"][kind], exact=True)
    meta = {"case": case, "series": record["name"], "source": record["name"].split("_")[1],
            "has_P": record["pure_patches"] > 0}
    return meta, labels, groups, membership, predictions
