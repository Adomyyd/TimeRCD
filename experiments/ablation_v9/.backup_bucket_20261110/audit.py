from __future__ import annotations

from .config import ExperimentConfig
from .data import file_manifest
from .models import build_model


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
            "encoder_patch_size": model.encoder.patch_size,
            "mask_patch_size": int(config.raw["input"]["mask_patch_size"]),
            "mask_ratio": float(config.raw["input"]["mask_ratio"]),
            "encoder_token_visible_if_any_point_visible": True,
            "classification_padding": padding_policy,
            "classification_uses_valid_mask": padding_policy == "ignore",
            "padding_label_is_not_used_as_normal": padding_policy == "ignore",
        },
    }
