from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parallel import validate_parallel_config
from .losses import validate_classification_padding


VARIANTS = {"p0_mask_x", "p1_full_x", "p2_full_x0"}
MODES = {"audit", "smoke", "train", "eval", "report", "all"}


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment_id"])

    @property
    def data_path(self) -> Path:
        return Path(self.raw["data"]["path"])

    @property
    def manifest_path(self) -> Path:
        return Path(self.raw["data"]["manifest_path"])

    @property
    def seed(self) -> int:
        return int(self.raw["runtime"]["seed"])

    @property
    def device(self) -> str:
        return str(self.raw["runtime"]["device"])

    def as_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    validate_config(raw)
    raw["loss"].setdefault("classification_padding", "ignore")
    return ExperimentConfig(raw=raw, path=config_path)


def validate_config(raw: dict[str, Any]) -> None:
    required = {
        "schema_version", "experiment_id", "data", "split", "input", "model",
        "loss", "optimization", "runtime", "evaluation",
    }
    missing = required - set(raw)
    unknown = set(raw) - required
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    experiment_id = raw["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    if experiment_id not in VARIANTS:
        raise ValueError(f"Unsupported experiment_id: {experiment_id}")
    input_cfg = raw["input"]
    expected = {
        "p0_mask_x": {"input_mode": "masked_x", "target": "x"},
        "p1_full_x": {"input_mode": "full_x", "target": "x"},
        "p2_full_x0": {"input_mode": "full_x", "target": "x0"},
    }[experiment_id]
    for key, value in expected.items():
        if input_cfg.get(key) != value:
            raise ValueError(f"{experiment_id}: input.{key} must be {value!r}")
    if input_cfg.get("mask_unit") != "patch":
        raise ValueError("input.mask_unit must be 'patch'")
    ratio = float(input_cfg.get("mask_ratio", 0.15))
    if not 0.0 < ratio < 1.0:
        raise ValueError("input.mask_ratio must be between 0 and 1")
    if int(input_cfg["mask_patch_size"]) <= 0:
        raise ValueError("input.mask_patch_size must be positive")
    if int(raw["model"]["num_features"]) != 1:
        raise ValueError("Current v9 implementation only supports num_features=1")
    if int(raw["model"]["patch_size"]) <= 0:
        raise ValueError("model.patch_size must be positive")
    if float(raw["loss"]["reconstruction_weight"]) < 0 or float(raw["loss"]["classification_weight"]) < 0:
        raise ValueError("loss weights must be non-negative")
    validate_classification_padding(raw["loss"].get("classification_padding", "ignore"))
    if float(raw["loss"]["reconstruction_weight"]) + float(raw["loss"]["classification_weight"]) <= 0:
        raise ValueError("at least one loss weight must be positive")
    if int(raw["optimization"]["batch_size"]) <= 0 or int(raw["optimization"]["epochs"]) <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if int(raw["optimization"]["log_freq"]) <= 0:
        raise ValueError("optimization.log_freq must be positive")
    if not raw["data"]["path"] or not raw["data"]["manifest_path"]:
        raise ValueError("data paths must be provided")
    validate_parallel_config(raw)


def config_to_json(config: ExperimentConfig) -> str:
    return json.dumps(config.raw, ensure_ascii=False, indent=2) + "\n"
