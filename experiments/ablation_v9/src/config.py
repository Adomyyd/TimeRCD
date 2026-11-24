from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .models import validate_parallel_config
from .losses import NORMAL_REFERENCE_VARIANTS, reference_patch_size, validate_classification_padding
from .data import file_manifest
from .models import build_model


VARIANTS = {"p0_mask_x"} | NORMAL_REFERENCE_VARIANTS
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
    expected = {"input_mode": "masked_x", "target": "x"}
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
    if experiment_id in NORMAL_REFERENCE_VARIANTS and normal_weight <= 0:
        raise ValueError(f"{experiment_id} requires positive loss.normal_reference_weight")
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
        allowed = {"enabled", "every_batch", "reference_epoch0", "loss_threshold", "grad_threshold", "start_epoch", "stop_on_mismatch", "detailed", "detail_interval", "snapshot_interval", "max_events"}
        if not isinstance(diagnostic, dict) or set(diagnostic) - allowed:
            raise ValueError("Invalid diagnostics fields")
        if not isinstance(diagnostic.get("detailed", False), bool):
            raise ValueError("diagnostics.detailed must be boolean")
        if int(diagnostic.get("snapshot_interval", 0)) < 0:
            raise ValueError("diagnostics.snapshot_interval must be nonnegative")
        if int(diagnostic.get("detail_interval", 100)) < 1:
            raise ValueError("diagnostics.detail_interval must be positive")
        if int(diagnostic.get("max_events", 6)) < 1:
            raise ValueError("diagnostics.max_events must be positive")
        if diagnostic.get("snapshot_interval", 0) and not diagnostic.get("detailed", False):
            raise ValueError("diagnostics.snapshot_interval requires detailed diagnostics")
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
    validate_optimization(raw["optimization"])
    validate_parallel_config(raw)


def run_audit(config: ExperimentConfig) -> dict:
    if not config.data_path.is_file():
        raise FileNotFoundError(config.data_path)
    if not config.manifest_path.is_file():
        raise FileNotFoundError(config.manifest_path)

    manifest = file_manifest(config.data_path, config.manifest_path)
    declared = manifest["declared"]
    declared_config = declared.get("config", {})
    if declared_config.get("is_multivariate") is not False:
        raise ValueError("Current v9 ablation expects the existing univariate dataset")
    if declared_config.get("num_features") != 1:
        raise ValueError("Manifest must declare num_features=1")

    actual_samples = int(declared.get("actual_samples", 0))
    train_ratio = float(config.raw["split"]["train_ratio"])
    train_samples = int(actual_samples * train_ratio)
    test_samples = actual_samples - train_samples
    if train_samples <= 0 or test_samples <= 0:
        raise ValueError("train/test split must both be non-empty")

    model = build_model(config.raw)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    padding_policy = config.raw["loss"].get("classification_padding", "ignore")
    return {
        "status": "passed",
        "experiment_id": config.experiment_id,
        "data_manifest": manifest,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "parameter_count": parameter_count,
        "protocol_checks": {
            "input_mode": config.raw["input"]["input_mode"],
            "target": config.raw["input"]["target"],
            "normal_reference_anomaly_patch_size": reference_patch_size(config.raw),
            "normal_reference_target": "x0" if model.normal_reference_head is not None else None,
            "normal_reference_weight": float(config.raw["loss"].get("normal_reference_weight", 0)),
            "encoder_patch_size": model.encoder.patch_size,
            "mask_patch_size": int(config.raw["input"]["mask_patch_size"]),
            "mask_ratio": float(config.raw["input"]["mask_ratio"]),
            "batch_sampler": str(config.raw["optimization"].get("batch_sampler", "sequential")),
            "batch_size": int(config.raw["optimization"]["batch_size"]),
            "encoder_token_visible_if_any_point_visible": True,
            "classification_padding": padding_policy,
            "classification_uses_valid_mask": padding_policy == "ignore",
            "padding_label_is_not_used_as_normal": padding_policy == "ignore",
        },
    }
def validate_optimization(options):
    clip = options.get("gradient_clip_norm")
    if clip is not None and (not math.isfinite(float(clip)) or float(clip) <= 0):
        raise ValueError("gradient_clip_norm must be finite and positive")
    schedule = options.get("lr_schedule")
    if schedule is None:
        return
    if set(schedule) != {"kind", "warmup_epochs", "total_epochs", "min_lr", "start_factor"}:
        raise ValueError("Invalid lr_schedule fields")
    if schedule["kind"] != "warmup_cosine":
        raise ValueError("lr_schedule.kind must be warmup_cosine")
    peak = float(options["learning_rate"])
    minimum, factor = float(schedule["min_lr"]), float(schedule["start_factor"])
    warm, total = int(schedule["warmup_epochs"]), int(schedule["total_epochs"])
    if not (math.isfinite(peak) and 0 <= minimum <= peak and 0 < factor <= 1):
        raise ValueError("Invalid schedule learning rates or start_factor")
    if not (1 <= warm < total and int(options["epochs"]) <= total):
        raise ValueError("Require 1 <= warmup_epochs < total_epochs and epochs <= total_epochs")


