from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from time_rcd._core.ts_encoder_bi_bias import TimeSeriesEncoder


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

    def _init_heads(self):
        for module in (self.reconstruction_head, self.anomaly_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, encoder_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x, encoder_mask)
        reconstruction = self.reconstruction_head(h).squeeze(-1)
        logits = self.anomaly_head(h).mean(dim=-2)
        return {"h": h, "reconstruction": reconstruction, "logits": logits}


def build_model(config):
    return AblationPretrainModel(config)
