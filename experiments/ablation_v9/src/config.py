from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parallel import validate_parallel_config
from .losses import validate_classification_padding


VARIANTS = {"p0_mask_x", "p1_mask_x_normal"}
BATCH_SAMPLERS = {"sequential", "length_bucket"}


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
    unknown = set(raw) - required - {"diagnostics"}
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
        "p1_mask_x_normal": {"input_mode": "masked_x", "target": "x"},
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
    import math
    normal_weight = float(raw["loss"].get("normal_reference_weight", 0))
    weights = [float(raw["loss"][key]) for key in ("reconstruction_weight", "classification_weight")] + [normal_weight]
    if not all(math.isfinite(w) and w >= 0 for w in weights):
        raise ValueError("loss weights must be finite and non-negative")
    if experiment_id == "p1_mask_x_normal" and normal_weight <= 0:
        raise ValueError("P1 requires positive loss.normal_reference_weight")
    if experiment_id == "p0_mask_x" and normal_weight != 0:
        raise ValueError("P0 has no normal reference head")
    validate_classification_padding(raw["loss"].get("classification_padding", "ignore"))
    if float(raw["loss"]["reconstruction_weight"]) + float(raw["loss"]["classification_weight"]) <= 0:
        raise ValueError("at least one loss weight must be positive")
    if int(raw["optimization"]["batch_size"]) <= 0 or int(raw["optimization"]["epochs"]) <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if int(raw["optimization"]["log_freq"]) <= 0:
        raise ValueError("optimization.log_freq must be positive")
    batch_sampler = str(raw["optimization"].get("batch_sampler", "sequential"))
    if batch_sampler not in BATCH_SAMPLERS:
        raise ValueError("optimization.batch_sampler must be 'sequential' or 'length_bucket'")
    if int(raw["optimization"].get("bucket_multiplier", 50)) <= 0:
        raise ValueError("optimization.bucket_multiplier must be positive")
    if not raw["data"]["path"] or not raw["data"]["manifest_path"]:
        raise ValueError("data paths must be provided")
    if "diagnostics" in raw:
        import math
        diagnostic = raw["diagnostics"]
        allowed = {"enabled", "every_batch", "reference_epoch0", "loss_threshold", "grad_threshold", "start_epoch", "stop_on_mismatch"}
        if not isinstance(diagnostic, dict) or set(diagnostic) - allowed:
            raise ValueError("Invalid diagnostics fields")
        if not isinstance(diagnostic.get("enabled", False), bool):
            raise ValueError("diagnostics.enabled must be boolean")
        if not isinstance(diagnostic.get("stop_on_mismatch", False), bool):
            raise ValueError("diagnostics.stop_on_mismatch must be boolean")
        if diagnostic.get("every_batch", True) is not True:
            raise ValueError("diagnostics records every batch")
        if int(diagnostic.get("start_epoch", 1)) < 0:
            raise ValueError("diagnostics.start_epoch must be nonnegative")
        for key, default in (("loss_threshold", 10), ("grad_threshold", 1000)):
            value = float(diagnostic.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"diagnostics.{key} must be finite and positive")
    validate_parallel_config(raw)

