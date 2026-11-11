from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


_DATA_CACHE: dict[str, list[dict[str, Any]]] = {}


@dataclass
class Sample:
    x: torch.Tensor
    x0: torch.Tensor
    y: torch.Tensor
    attribute: Any


def load_data(data_path: str | Path) -> list[dict[str, Any]]:
    key = str(Path(data_path).resolve())
    if key not in _DATA_CACHE:
        with Path(key).open("rb") as handle:
            _DATA_CACHE[key] = pickle.load(handle)
    return _DATA_CACHE[key]


class PairedTimeSeriesDataset(Dataset):
    def __init__(self, data_path: str | Path, indices: list[int]):
        self.data_path = Path(data_path)
        self.data = load_data(self.data_path)
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        sample = self.data[self.indices[index]]
        x = torch.as_tensor(sample["time_series"], dtype=torch.float32)
        x0 = torch.as_tensor(sample["normal_time_series"], dtype=torch.float32)
        y = torch.as_tensor(sample["labels"], dtype=torch.long)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(-1)
        if x.shape != x0.shape or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Paired sample shape mismatch: x={tuple(x.shape)}, "
                f"x0={tuple(x0.shape)}, y={tuple(y.shape)}"
            )
        return Sample(x=x, x0=x0, y=y, attribute=sample.get("attribute"))


def collate_samples(batch: list[Sample]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    features = batch[0].x.shape[-1]
    if any(item.x.shape[-1] != features for item in batch):
        raise ValueError("inconsistent feature count")

    # Match original preprocessing: batch statistics come from observed X and
    # are applied to both X and paired X0.
    observed = torch.cat([item.x for item in batch], dim=0)
    mean = observed.mean(dim=0, keepdim=True)
    std = observed.std(dim=0, keepdim=True) + 1e-4
    xs = [(item.x - mean) / std for item in batch]
    x0s = [(item.x0 - mean) / std for item in batch]
    ys = [item.y for item in batch]
    lengths = [item.x.shape[0] for item in batch]

    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    x0 = torch.nn.utils.rnn.pad_sequence(x0s, batch_first=True, padding_value=0.0)
    y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    valid = torch.zeros(len(batch), max(lengths), dtype=torch.bool)
    for row, length in enumerate(lengths):
        valid[row, :length] = True
    return {
        "x": x,
        "x0": x0,
        "y": y,
        "valid": valid,
        "lengths": lengths,
        "attribute": [item.attribute for item in batch],
    }


def load_indices(data_path: str | Path, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    data = load_data(data_path)
    permutation = list(range(len(data)))
    random.Random(seed).shuffle(permutation)
    train_size = int(len(data) * train_ratio)
    return permutation[:train_size], permutation[train_size:]


def file_manifest(data_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    record: dict[str, Any] = {
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
    }
    for path, key in ((data_path, "data"), (manifest_path, "manifest")):
        stat = path.stat()
        record[f"{key}_bytes"] = stat.st_size
    with manifest_path.open("r", encoding="utf-8") as handle:
        declared = json.load(handle)
    record["declared"] = declared
    record["data_sha256_declared"] = declared.get("pickle_sha256")
    record["manifest_sha256"] = sha256(manifest_path)
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
