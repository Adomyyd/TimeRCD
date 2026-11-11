"""Throughput benchmark contract: fixed batch limit, no checkpoints."""
import pickle
from pathlib import Path

import torch

from src.config import load_config
from src.models import build_model
from src.train import benchmark, set_seed


ROOT = Path(__file__).resolve().parents[1]


def config_for():
    config = load_config(ROOT / "configs" / "p1_full_x.json").raw
    config["runtime"].update(device="cpu", parallel="none", gpu_ids=None)
    config["model"].update(d_model=32, d_proj=8, num_layers=1, num_heads=4,
                           dropout=0.0, d_ff_dropout=0.0)
    config["optimization"].update(batch_size=3, epochs=1, log_freq=1)
    return config


def synthetic_data(tmp_path):
    rows = []
    for length in (129, 157, 145, 193, 171, 163, 181, 149, 167, 155, 139, 175):
        x = torch.linspace(-1, 1, length).unsqueeze(-1)
        rows.append(dict(time_series=x, normal_time_series=x * 0.9,
                         labels=(torch.arange(length) % 3 == 0).long()))
    path = tmp_path / "synthetic.pkl"
    with path.open("wb") as handle:
        pickle.dump(rows, handle)
    return path


def test_benchmark_reports_throughput_and_respects_batch_limit(tmp_path):
    config = config_for()
    config["data"]["path"] = str(synthetic_data(tmp_path))
    config["split"]["train_ratio"] = 0.75
    set_seed(72)
    result = benchmark(build_model(config), config, tmp_path, max_batches=3, warmup=1)
    assert result["timed_batches"] == 2
    assert result["max_batches"] == 3
    assert result["batch_size"] == 3
    assert result["batch_sampler"] == "length_bucket"
    assert result["s_per_batch"] > 0
    assert result["samples_per_s"] > 0
    assert 0.0 <= result["padding_fraction"] <= 1.0
    assert result["epoch_estimate_hours"] >= 0
    assert (tmp_path / "metrics" / "bench.json").is_file()
