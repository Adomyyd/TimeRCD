from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from time_rcd._core.ts_encoder_bi_bias import TimeSeriesEncoder
from .losses import NORMAL_REFERENCE_VARIANTS


class AblationPretrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_proj = int(config["model"]["d_proj"])
        self.encoder = TimeSeriesEncoder(
            d_model=int(config["model"]["d_model"]),
            d_proj=d_proj,
            patch_size=int(config["model"]["patch_size"]),
            num_layers=int(config["model"]["num_layers"]),
            num_heads=int(config["model"]["num_heads"]),
            d_ff_dropout=float(config["model"]["d_ff_dropout"]),
            use_rope=bool(config["model"]["use_rope"]),
            num_features=int(config["model"]["num_features"]),
            activation=str(config["model"]["activation"]),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_proj, d_proj * 4),
            nn.GELU(),
            nn.Dropout(float(config["model"]["dropout"])),
            nn.Linear(d_proj * 4, d_proj * 4),
            nn.GELU(),
            nn.Dropout(float(config["model"]["dropout"])),
            nn.Linear(d_proj * 4, 1),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.GELU(),
            nn.Dropout(float(config["model"]["dropout"])),
            nn.Linear(d_proj // 2, 2),
        )
        self._init_heads()
        self.normal_reference_head = None
        if config["experiment_id"] in NORMAL_REFERENCE_VARIANTS:
            # Preserve P0's shared parameter initialization and CPU RNG position.
            with torch.random.fork_rng(devices=[]):
                self.normal_reference_head = nn.Sequential(
                    nn.Linear(d_proj, d_proj * 4), nn.GELU(),
                    nn.Dropout(float(config["model"]["dropout"])),
                    nn.Linear(d_proj * 4, d_proj * 4), nn.GELU(),
                    nn.Dropout(float(config["model"]["dropout"])),
                    nn.Linear(d_proj * 4, 1),
                )
                for layer in self.normal_reference_head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        nn.init.zeros_(layer.bias)

    def _init_heads(self):
        for module in (self.reconstruction_head, self.anomaly_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, encoder_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x, encoder_mask)
        # Two full-length MLP heads otherwise retain twice the large activations.
        # Recompute P1 heads in backward, preserving their dropout RNG state.
        checkpoint_heads = self.normal_reference_head is not None and self.training and torch.is_grad_enabled()
        if checkpoint_heads:
            reconstruction = checkpoint(self.reconstruction_head, h, use_reentrant=False,
                                        preserve_rng_state=True).squeeze(-1)
        else:
            reconstruction = self.reconstruction_head(h).squeeze(-1)
        logits = self.anomaly_head(h).mean(dim=-2)
        output = {"h": h, "reconstruction": reconstruction, "logits": logits}
        if self.normal_reference_head is not None:
            if checkpoint_heads:
                output["normal_reference"] = checkpoint(self.normal_reference_head, h, use_reentrant=False,
                                                        preserve_rng_state=True).squeeze(-1)
            else:
                output["normal_reference"] = self.normal_reference_head(h).squeeze(-1)
        return output


def build_model(config):
    return AblationPretrainModel(config)


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
