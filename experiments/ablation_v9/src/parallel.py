from __future__ import annotations

import torch
from torch import nn


def validate_parallel_config(config: dict) -> None:
    runtime = config["runtime"]
    mode = runtime.get("parallel", "none")
    ids = runtime.get("gpu_ids")
    if mode not in {"none", "data_parallel"}:
        raise ValueError("runtime.parallel must be 'none' or 'data_parallel'")
    if ids is not None and (
        not isinstance(ids, list) or not ids
        or any(type(index) is not int or index < 0 for index in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("runtime.gpu_ids must be null or a non-empty list of unique non-negative integers")
    if mode == "none" and ids is not None:
        raise ValueError("runtime.gpu_ids requires runtime.parallel='data_parallel'; use runtime.device for single GPU")
    if mode == "data_parallel":
        if str(runtime["device"]) != "cuda":
            raise ValueError("data_parallel requires runtime.device='cuda'; select devices through gpu_ids")
        if ids is None or len(ids) < 2:
            raise ValueError("data_parallel requires at least two explicit runtime.gpu_ids")
        if int(config["optimization"]["batch_size"]) < len(ids):
            raise ValueError("global batch_size must be at least the number of selected GPUs")


def resolve_device(config: dict) -> torch.device:
    validate_parallel_config(config)
    runtime = config["runtime"]
    device = torch.device(str(runtime["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if runtime.get("parallel", "none") == "data_parallel":
            ids = runtime["gpu_ids"]
        else:
            ids = [device.index if device.index is not None else torch.cuda.current_device()]
        count = torch.cuda.device_count()
        if any(index >= count for index in ids):
            raise ValueError(f"Requested GPU ids {ids}, but only {count} CUDA devices are visible; ids are relative to CUDA_VISIBLE_DEVICES")
        device = torch.device("cuda", ids[0])
    return device


def prepare_model(model: nn.Module, config: dict) -> nn.Module:
    device = resolve_device(config)
    model = model.to(device)
    if config["runtime"].get("parallel", "none") == "data_parallel":
        ids = config["runtime"]["gpu_ids"]
        return nn.DataParallel(model, device_ids=ids, output_device=ids[0], dim=0)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def parallel_record(model: nn.Module, config: dict) -> dict:
    return {
        "parallel": config["runtime"].get("parallel", "none"),
        "gpu_ids": list(model.device_ids) if isinstance(model, nn.DataParallel) else [],
        "primary_device": str(next(model.parameters()).device),
        "global_batch_size": int(config["optimization"]["batch_size"]),
    }
