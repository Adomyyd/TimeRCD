"""Small synthetic checks; never loads the 400k dataset."""
import argparse
import copy
import json
from pathlib import Path
import pickle

import pytest
import torch

from src.config import load_config, validate_config
from src.data import Sample, collate_samples
from src.losses import compute_losses
from src.models import build_model
from src.parallel import prepare_model, resolve_device, unwrap_model
from src.train import make_batch_inputs, run_smoke, set_seed, train_model
from run_experiment import apply_runtime_overrides
import run_experiment as entry

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["p0_mask_x", "p1_full_x", "p2_full_x0"]


def config_for(variant="p0_mask_x"):
    config = load_config(ROOT / "configs" / f"{variant}.json").raw
    config["runtime"].update(device="cpu", parallel="none", gpu_ids=None)
    config["model"].update(d_model=32, d_proj=8, num_layers=1, num_heads=4,
                           dropout=0.0, d_ff_dropout=0.0)
    return config


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch_for_test():
    generator = torch.Generator().manual_seed(13)
    return collate_samples([
        Sample(torch.randn(length, 1, generator=generator),
               torch.randn(length, 1, generator=generator),
               (torch.arange(length) % 3 == 0).long(), None)
        for length in [113, 167, 129, 181, 155]
    ])


@pytest.mark.parametrize("runtime,batch_size", [
    ({"parallel": "ddp"}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [0]}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [0, 0]}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [-1, 0]}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [False, 1]}, 16),
    ({"parallel": "data_parallel", "gpu_ids": "0,1"}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [0, 1], "device": "cpu"}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [0, 1], "device": "cuda:0"}, 16),
    ({"parallel": "data_parallel", "gpu_ids": [0, 1]}, 1),
    ({"parallel": "none", "gpu_ids": [0, 1]}, 16),
])
def test_invalid_settings(runtime, batch_size):
    config = config_for()
    config["runtime"].update(device="cuda", **{k: v for k, v in runtime.items() if k != "device"})
    if "device" in runtime:
        config["runtime"]["device"] = runtime["device"]
    config["optimization"]["batch_size"] = batch_size
    with pytest.raises(ValueError):
        validate_config(config)


def test_old_configs_and_cli_override():
    config = load_config(ROOT / "configs/p0_mask_x.json")
    config.raw["runtime"].pop("parallel", None)
    config.raw["runtime"].pop("gpu_ids", None)
    validate_config(config.raw)
    apply_runtime_overrides(config, argparse.Namespace(parallel="data_parallel", gpu_ids=[1, 0]))
    assert config.raw["runtime"]["gpu_ids"] == [1, 0]
    apply_runtime_overrides(config, argparse.Namespace(parallel="none", gpu_ids=None))
    assert config.raw["runtime"]["gpu_ids"] is None


def test_device_selection_and_unavailable_ids(monkeypatch):
    config = config_for()
    config["runtime"].update(device="cuda", parallel="data_parallel", gpu_ids=[1, 0])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device(config) == torch.device("cuda:1")
    config["runtime"]["gpu_ids"] = [0, 2]
    with pytest.raises(ValueError, match="visible"):
        resolve_device(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device(config)


@pytest.mark.parametrize("variant", VARIANTS)
def test_global_loss_and_gradients_after_uneven_split(variant):
    config = config_for(variant)
    set_seed(72)
    single = build_model(config).eval()
    split = copy.deepcopy(single)
    batch = batch_for_test()
    x, visible, selected = make_batch_inputs(batch, config, torch.Generator().manual_seed(72))
    expected = single(x, visible)
    # Global normalization and padding happen before scatter, as in DataParallel.
    pieces = [split(a, b) for a, b in zip(x.split([3, 2]), visible.split([3, 2]))]
    gathered = {key: torch.cat([part[key] for part in pieces], dim=0) for key in expected}
    for key in expected:
        torch.testing.assert_close(gathered[key], expected[key], atol=2e-6, rtol=2e-5)
    reference_loss = compute_losses(expected, batch, config["input"]["target"], 1, 1, selected)
    gathered_loss = compute_losses(gathered, batch, config["input"]["target"], 1, 1, selected)
    reference_loss["total"].backward()
    gathered_loss["total"].backward()
    torch.testing.assert_close(reference_loss["total"], gathered_loss["total"])
    for a, b in zip(single.parameters(), split.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4)


def test_wrapped_training_saves_portable_checkpoint(tmp_path, monkeypatch):
    config = config_for()
    samples = []
    for length in [129, 157, 145, 193, 171, 163]:
        x = torch.linspace(-1, 1, length).unsqueeze(-1)
        samples.append({"time_series": x, "normal_time_series": x * 0.9,
                        "labels": (torch.arange(length) % 3 == 0).long()})
    path = tmp_path / "synthetic.pkl"
    with path.open("wb") as handle:
        pickle.dump(samples, handle)
    config["data"]["path"] = str(path)
    config["split"]["train_ratio"] = 0.67
    config["optimization"].update(batch_size=3, epochs=1, log_freq=1)
    # Force CPU fallback even on a CUDA host; CUDA execution is checked below.
    with monkeypatch.context() as context:
        context.setattr(torch.cuda, "is_available", lambda: False)
        model = torch.nn.DataParallel(prepare_model(build_model(config), config))
    assert run_smoke(model, config)["status"] == "passed"
    result = train_model(model, config, tmp_path)
    assert result["status"] == "completed"
    payload = torch.load(tmp_path / "checkpoints/best.pth", weights_only=False)
    assert all(not name.startswith("module.") for name in payload["model_state_dict"])
    unwrapped = build_model(config)
    unwrapped.load_state_dict(payload["model_state_dict"], strict=True)
    unwrap_model(model).load_state_dict(unwrapped.state_dict(), strict=True)
    assert "Parallel settings:" in (tmp_path / "train.log").read_text()


def test_entry_smoke_records_effective_config_and_status(tmp_path, monkeypatch):
    config = config_for()
    data_path = tmp_path / "not_loaded.pkl"
    data_path.write_bytes(b"not a pickle; smoke must not load data")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"config": {"is_multivariate": False, "num_features": 1},
                                         "actual_samples": 10}), encoding="utf-8")
    config["data"].update(path=str(data_path), manifest_path=str(manifest_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(entry, "HERE", tmp_path)
    args = argparse.Namespace(config=str(config_path), run_id="test_smoke", mode="smoke",
                              parallel="none", gpu_ids=None, checkpoint=None)
    entry.run_one(args)
    run = tmp_path / "runs/test_smoke"
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    assert json.loads((run / "config.json").read_text())["runtime"]["parallel"] == "none"
    assert json.loads((run / "smoke.json").read_text())["global_batch_size"] == 16
    with pytest.raises(FileExistsError):
        entry.run_one(args)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA GPUs")
@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_two_gpu_forward_gradients_and_tail(variant):
    config = config_for(variant)
    set_seed(72)
    reference = build_model(config).to("cuda:0").eval()
    config["runtime"].update(device="cuda", parallel="data_parallel", gpu_ids=[0, 1])
    parallel = prepare_model(copy.deepcopy(reference), config).eval()
    batch = batch_for_test()
    x, visible, selected = make_batch_inputs(batch, config, torch.Generator().manual_seed(72))
    device_batch = {key: value.to("cuda:0") for key, value in batch.items() if isinstance(value, torch.Tensor)}
    x, visible, selected = x.to("cuda:0"), visible.to("cuda:0"), selected.to("cuda:0")
    a, b = reference(x, visible), parallel(x, visible)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=2e-5, rtol=2e-4)
    loss_a = compute_losses(a, device_batch, config["input"]["target"], 1, 1, selected)
    loss_b = compute_losses(b, device_batch, config["input"]["target"], 1, 1, selected)
    loss_a["total"].backward()
    loss_b["total"].backward()
    for first, second in zip(reference.parameters(), unwrap_model(parallel).parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, atol=3e-5, rtol=5e-4)
    optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-4)
    before = unwrap_model(parallel).encoder.embedding_layer.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, unwrap_model(parallel).encoder.embedding_layer.weight)
    parallel.zero_grad(set_to_none=True)
    tail = parallel(x[:1], visible[:1])
    assert tail["logits"].shape[0] == 1
    (tail["reconstruction"].square().mean() + tail["logits"].square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in parallel.parameters() if p.grad is not None)
    # Exercise dropout and wrapper attribute access in training mode too.
    unwrap_model(parallel).load_state_dict(reference.state_dict())
    for module in parallel.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.1
    parallel.train()
    assert run_smoke(parallel, config)["status"] == "passed"
