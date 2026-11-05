from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from .artifacts import sha256


@dataclass
class Series:
    name: str
    values: np.ndarray
    labels: np.ndarray
    source: dict

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        self.labels = np.asarray(self.labels)
        if self.values.ndim != 2 or self.values.shape[1] != 1 or not len(self.values):
            raise ValueError(f"{self.name}: expected nonempty (T, 1) values")
        if self.labels.shape != (len(self.values),):
            raise ValueError(f"{self.name}: labels must have shape (T,)")
        if not np.isfinite(self.values).all() or not np.isin(self.labels, [0, 1]).all():
            raise ValueError(f"{self.name}: nonfinite values or nonbinary labels")
        self.labels = self.labels.astype(np.int64)


def load_series(settings, seed):
    if settings["kind"] == "synthetic":
        for index, length in enumerate(settings["lengths"]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            t = np.arange(length)
            normal = np.sin(2 * np.pi * t / settings["period"])
            normal += rng.normal(0, settings["noise_std"], length)
            labels = np.zeros(length, dtype=np.int64)
            start = length // 2
            labels[start:min(length, start + settings["anomaly_length"])] = 1
            values = normal + labels * settings["anomaly_shift"]
            yield Series(f"synthetic_{index:03d}_n{length}", values[:, None], labels, {
                "kind": "synthetic", "seed": seed, "sample_index": index,
                "original_length": length, "offset": 0, "settings": settings,
            })
        return
    selected = None
    if "selection_manifest" in settings:
        manifest = json.loads(Path(settings["selection_manifest"]).read_text())
        selected = {row["filename"]: row for row in manifest["selected"]}
        if set(selected) != set(settings["files"]):
            raise ValueError("TSB file list differs from the frozen selection manifest")
    for filename in settings["files"]:
        path = Path(settings["root"]) / filename
        fingerprint = sha256(path)
        if selected is not None and fingerprint != selected[filename]["sha256"]:
            raise ValueError(f"{path}: data changed after sample selection")
        frame = pd.read_csv(path)
        if len(frame.columns) != 2 or frame.columns[-1] != "Label":
            raise ValueError(f"{path}: expected one value column followed by Label")
        match = re.search(r"_tr_(\d+)_", path.stem)
        if match is None:
            raise ValueError(f"{path}: missing _tr_<length>_ in filename")
        train_length = int(match.group(1))
        if not 0 <= train_length < len(frame):
            raise ValueError(f"{path}: invalid training split {train_length}")
        offset = train_length if settings["split"] == "test" else 0
        if selected is not None and len(frame) - offset != selected[filename]["length"]:
            raise ValueError(f"{path}: split length differs from frozen selection")
        # Do not drop missing rows: that would silently change temporal coordinates.
        yield Series(path.stem, frame.iloc[offset:, :1].to_numpy(), frame.Label.to_numpy()[offset:], {
            "kind": "tsb", "path": str(path), "sha256": fingerprint,
            "original_length": len(frame), "train_length": train_length,
            "offset": offset, "split": settings["split"],
        })


def geometry(length, window_size, patch_size=16):
    actual = min(length, window_size)
    count = (length + actual - 1) // actual
    tail = length - (count - 1) * actual
    return {
        "length": length, "configured_window": window_size, "actual_window": actual,
        "window_count": count, "tail_valid_length": tail,
        "tail_valid_ratio": tail / actual,
        "patch_size": patch_size, "patches_per_window": (actual + patch_size - 1) // patch_size,
        "tail_valid_patches": (tail + patch_size - 1) // patch_size,
        "valid_patch_count": (count - 1) * ((actual + patch_size - 1) // patch_size)
        + (tail + patch_size - 1) // patch_size,
        "tail_partial_patch": bool(tail % patch_size),
        "window_patch_aligned": actual % patch_size == 0,
        "length_patch_aligned": length % patch_size == 0,
        "length_group": "short" if length < window_size else (
            "one_window" if length == window_size else (
                "multiple_full" if tail == actual else "multiple_with_tail"
            )
        ),
    }
