"""Diagnostic observation must preserve updates and RNG, including dropout/DP."""
import copy
import importlib.util
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.diagnostics import BatchDiagnostics, compare_states, rng_state
from src.models import build_model
from src.models import prepare_model
from src.train import make_batch_inputs, run_smoke, set_seed, train_model
from src.losses import compute_losses
from src.data import collate_samples, Sample


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "audits/spike_20261111/sources_cleanup_20261114/experiments/ablation_v9/src/train.py"


def config_for(tmp_path, parallel="none"):
    config = load_config(ROOT / "configs/p0_mask_x_padding_official.json").raw
    config["runtime"].update(device="cpu" if parallel == "none" else "cuda",
                             parallel=parallel, gpu_ids=None if parallel == "none" else [0, 1])
    config["model"].update(d_model=32, d_proj=8, num_layers=1, num_heads=4,
                          dropout=0.1, d_ff_dropout=0.1)
    config["optimization"].update(batch_size=4, epochs=2, log_freq=1)
    config["split"]["train_ratio"] = 0.75
    rows = []
    for length in range(129, 153, 2):
        x = torch.linspace(-2, 2, length).unsqueeze(-1)
        rows.append(dict(time_series=x, normal_time_series=x * 0.9,
                         labels=(torch.arange(length) % 5 == 0).long()))
    path = tmp_path / "data.pkl"
    with path.open("wb") as handle:
        pickle.dump(rows, handle)
    config["data"]["path"] = str(path)
    return config


def assert_rng_equal(a, b):
    assert a["python"] == b["python"]
    assert a["numpy"][0] == b["numpy"][0]
    np.testing.assert_array_equal(a["numpy"][1], b["numpy"][1])
    assert a["numpy"][2:] == b["numpy"][2:]
    assert torch.equal(a["torch_cpu"], b["torch_cpu"])
    assert len(a["torch_cuda"]) == len(b["torch_cuda"])
    assert all(torch.equal(x, y) for x, y in zip(a["torch_cuda"], b["torch_cuda"]))


@pytest.mark.parametrize("parallel", ["none", "data_parallel"])
def test_full_loop_observation_is_numerically_inert(tmp_path, parallel, monkeypatch):
    if parallel != "none" and torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    assert SNAPSHOT.is_file(), "required historical training source snapshot is missing"
    config = config_for(tmp_path, parallel)
    import sys
    import types
    from src import data, evaluate, losses, models
    legacy = types.ModuleType("_v9_historical_test")
    legacy.__path__ = []
    monkeypatch.setitem(sys.modules, "_v9_historical_test", legacy)
    for name, module in dict(data=data, evaluate=evaluate, losses=losses,
                             parallel=models, sampling=data).items():
        monkeypatch.setitem(sys.modules, "_v9_historical_test." + name, module)
    spec = importlib.util.spec_from_file_location("_v9_historical_test.train", SNAPSHOT)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    results = []
    for name, fn, diagnostic in (("original", original.train_model, False),
                                 ("disabled", train_model, False),
                                 ("enabled", train_model, True)):
        run = tmp_path / name
        run.mkdir()
        raw = copy.deepcopy(config)
        if diagnostic:
            raw["diagnostics"] = dict(enabled=True, every_batch=True, start_epoch=1,
                                      loss_threshold=1e20, grad_threshold=1e20,
                                      reference_epoch0=str(tmp_path / "original/checkpoints/epoch_0000.pth"))
        set_seed(72)
        model = prepare_model(build_model(raw), raw)
        run_smoke(model, raw)
        fn(model, raw, run)
        results.append((torch.load(run / "checkpoints/epoch_0001.pth", weights_only=False,
                                   map_location="cpu"), rng_state()))
    for actual, state in results[1:]:
        for key in ("model_state_dict", "optimizer_state_dict", "validation"):
            assert not compare_states(actual[key], results[0][0][key], key)
        assert_rng_equal(state, results[0][1])
    rows = [json.loads(line) for line in (tmp_path / "enabled/diagnostics/batches.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert [row["batch"] for row in rows] == [0, 1, 2, 0, 1, 2]
    assert all(len(row["sample_ids"]) == len(row["lengths"]) for row in rows)
    assert not (tmp_path / "disabled/diagnostics").exists()
    comparison = json.loads((tmp_path / "enabled/diagnostics/epoch0_comparison.json").read_text())
    assert comparison["exact_match"]


def test_event_snapshot_is_before_update_and_replays_dropout(tmp_path):
    config = config_for(tmp_path)
    config["diagnostics"] = dict(enabled=True, start_epoch=0,
                                 loss_threshold=1e-12, grad_threshold=1e20)
    set_seed(72)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    x = torch.linspace(-2, 2, 145).unsqueeze(-1)
    batch = collate_samples([Sample(x, x, torch.zeros(145, dtype=torch.long), None)] * 2)
    dataset = type("Dataset", (), {"data": [{"time_series": x}, {"time_series": x}]})()
    diagnostic = BatchDiagnostics(tmp_path, config)
    diagnostic.begin_epoch(0, None, [0, 1])
    inputs, encoder_mask, mask = make_batch_inputs(batch, config, torch.Generator().manual_seed(72))
    diagnostic.before_forward()
    output = model(inputs, encoder_mask)
    losses = compute_losses(output, batch, "x", 1, 1, mask, classification_padding="official")
    losses["total"].backward()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    diagnostic.after_backward(model, optimizer, 0, batch, inputs, encoder_mask, mask, output, losses, dataset)
    optimizer.step()
    payload = torch.load(tmp_path / "diagnostics/first_event.pth", weights_only=False)
    assert not compare_states(before, payload["model_state_dict"])
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert payload["optimizer_state_dict"]["state"] == {}
    # Constructing the model consumes RNG; restore the captured state AFTER loading it.
    replay = build_model(config)
    replay.load_state_dict(payload["model_state_dict"])
    state = payload["rng_before_forward"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    replay_output = replay(payload["x_input"], payload["encoder_mask"])
    replay_losses = compute_losses(replay_output, payload["batch"], "x", 1, 1, payload["mask"],
                                   classification_padding="official")
    assert replay_losses["total"].item() == payload["losses"]["total"]
    replay_losses["total"].backward()
    for name, parameter in replay.named_parameters():
        assert torch.linalg.vector_norm(parameter.grad.double()).item() == pytest.approx(payload["gradient_norms"][name], rel=1e-12)
    # A later event cannot replace the first one.
    first = (tmp_path / "diagnostics/first_event.pth").read_bytes()
    diagnostic.after_backward(model, optimizer, 0, batch, inputs, encoder_mask, mask, output, losses, dataset)
    assert (tmp_path / "diagnostics/first_event.pth").read_bytes() == first


def test_epoch0_mismatch_stops_before_next_epoch(tmp_path):
    config = config_for(tmp_path)
    reference = tmp_path / "reference.pth"
    actual = tmp_path / "actual.pth"
    payload = dict(epoch=0, model_state_dict={"w": torch.ones(1)},
                   optimizer_state_dict={}, validation={})
    torch.save(payload, reference)
    payload["model_state_dict"]["w"] = torch.zeros(1)
    torch.save(payload, actual)
    config["diagnostics"] = dict(enabled=True, reference_epoch0=str(reference), stop_on_mismatch=True)
    diagnostic = BatchDiagnostics(tmp_path, config)
    with pytest.raises(RuntimeError, match="epoch0 differs"):
        diagnostic.end_epoch(0, actual)
    report = json.loads((tmp_path / "diagnostics/epoch0_comparison.json").read_text())
    assert report["mismatch_count"] == 1
    diagnostic.options["stop_on_mismatch"] = False
    diagnostic.end_epoch(0, actual)
    report = json.loads((tmp_path / "diagnostics/epoch0_comparison.json").read_text())
    assert report["trajectory_label"] == "same_config_rerun_not_exact_replay"


def test_cli_overrides_are_validated_and_saved(monkeypatch, tmp_path):
    from run_experiment import parse_args, apply_runtime_overrides
    config = load_config(ROOT / "configs/p0_mask_x_padding_official.json")
    original = copy.deepcopy(config.raw)
    checkpoint = tmp_path / "reference.pth"
    checkpoint.touch()
    monkeypatch.setattr("sys.argv", [
        "run_experiment.py", "--config", str(config.path), "--run-id", "test-diagnostic",
        "--mode", "train", "--diagnostics", "--epochs", "2", "--log-freq", "1",
        "--diagnostic-reference-epoch0", str(checkpoint),
    ])
    apply_runtime_overrides(config, parse_args())
    assert config.raw["optimization"]["epochs"] == 2
    assert config.raw["optimization"]["log_freq"] == 1
    assert config.raw["diagnostics"]["enabled"] is True
    for key in ("model", "input", "loss", "data", "split", "runtime"):
        assert config.raw[key] == original[key]
    monkeypatch.setattr("sys.argv", [
        "run_experiment.py", "--config", str(config.path), "--run-id", "test-diagnostic",
        "--mode", "train", "--epochs", "0",
    ])
    with pytest.raises(SystemExit):
        parse_args()
    # A diagnostic training snapshot can still be used for ordinary evaluation.
    monkeypatch.setattr("sys.argv", [
        "run_experiment.py", "--config", str(config.path), "--run-id", "test-eval",
        "--mode", "eval", "--checkpoint", str(checkpoint),
    ])
    apply_runtime_overrides(config, parse_args())
    assert config.raw["diagnostics"]["enabled"] is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_full_size_two_gpu_observation_is_numerically_inert(tmp_path, monkeypatch):
    import sys
    original_factory = config_for
    def full_size_config(path, parallel):
        config = original_factory(path, parallel)
        config["model"] = load_config(ROOT / "configs/p0_mask_x_padding_official.json").raw["model"]
        config["optimization"]["batch_size"] = 32
        rows = []
        for i in range(96):
            length = 1000 + i * 10
            x = torch.linspace(-2, 2, length).unsqueeze(-1)
            rows.append(dict(time_series=x, normal_time_series=x * 0.9,
                             labels=(torch.arange(length) % 5 == 0).long()))
        with Path(config["data"]["path"]).open("wb") as handle:
            pickle.dump(rows, handle)
        return config
    monkeypatch.setattr(sys.modules[__name__], "config_for", full_size_config)
    test_full_loop_observation_is_numerically_inert(tmp_path, "data_parallel", monkeypatch)
