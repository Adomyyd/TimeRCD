"""Safeguards must preserve raw evidence, resume schedules and legacy behavior."""
import copy
import json
import math
from pathlib import Path
import pytest
import torch
from src.train import OptimizationControl, validate_optimization
from src.config import load_config
from src.diagnostics import compare_states
from src.models import build_model
from src.models import prepare_model
from src.train import train_model, run_smoke, set_seed
from test_diagnostics import config_for

ROOT = Path(__file__).resolve().parents[1]

def options():
    return dict(learning_rate=5e-4, gradient_clip_norm=5., epochs=3,
                lr_schedule=dict(kind="warmup_cosine", warmup_epochs=1,
                                 total_epochs=3, min_lr=5e-5, start_factor=.1))

def test_schedule_endpoints_and_resume_horizon():
    model = torch.nn.Linear(2, 1)
    c = OptimizationControl(torch.optim.AdamW(model.parameters()), options(), 4)
    rates = [c.learning_rate(i) for i in range(12)]
    assert rates[0] == pytest.approx(5e-5)
    assert rates[3] == pytest.approx(5e-4)
    assert rates[-1] == pytest.approx(5e-5)
    assert rates[:4] == sorted(rates[:4])
    assert rates[3:] == sorted(rates[3:], reverse=True)
    c.verify_resume(c.state_dict(2), 2)
    with pytest.raises(ValueError):
        c.verify_resume(c.state_dict(1), 2)
    with pytest.raises(ValueError):
        OptimizationControl(c.optimizer, options(), 5).verify_resume(c.state_dict(2), 2)
    bad = options()
    bad["epochs"] = 4
    with pytest.raises(ValueError):
        validate_optimization(bad)

def test_clipping_direction_nonfinite_and_disabled():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    c = OptimizationControl(optimizer, options(), 4)
    model.weight.grad = torch.tensor([[300., 400.]])
    c.begin_step(0)
    record, norms = c.inspect(model, torch.tensor(1.))
    result = c.apply(model, record)
    assert record["gradient_l2_before"] == 500
    assert result["gradient_l2_after"] == pytest.approx(5., rel=1e-6)
    assert torch.allclose(model.weight.grad, torch.tensor([[3., 4.]]))
    assert not optimizer.state
    model.weight.grad[0, 0] = float("nan")
    record, _ = c.inspect(model, torch.tensor(1.))
    with pytest.raises(FloatingPointError):
        c.apply(model, record)
    assert not optimizer.state
    d = OptimizationControl(optimizer, dict(learning_rate=.001), 4)
    before = model.weight.grad.clone()
    assert d.inspect(model, torch.tensor(1.)) == (None, None)
    d.begin_step(0)
    assert d.apply(model, None) is None
    assert torch.allclose(before, model.weight.grad, equal_nan=True)

@pytest.mark.parametrize("variant", ["p0_mask_x", "p1_mask_x_normal"])
@pytest.mark.parametrize("parallel", ["none", "data_parallel"])
def test_stable_training_diagnostics_and_resume(tmp_path, variant, parallel):
    if parallel != "none" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    config = config_for(tmp_path, parallel)
    config["experiment_id"] = variant
    if variant == "p1_mask_x_normal":
        config["loss"]["normal_reference_weight"] = 1
    config["optimization"].update(options())
    config["diagnostics"] = dict(enabled=True, every_batch=True, detailed=True,
                                start_epoch=0, loss_threshold=1e-8,
                                grad_threshold=1e20, detail_interval=2,
                                snapshot_interval=2, max_events=2)
    def run(name, cfg, resume=None):
        path = tmp_path / name
        path.mkdir()
        set_seed(72)
        m = prepare_model(build_model(cfg), cfg)
        run_smoke(m, cfg)
        train_model(m, cfg, path, resume_from=resume)
        return path
    full = run("full", config)
    short = copy.deepcopy(config)
    short["optimization"]["epochs"] = 2
    partial = run("partial", short)
    resumed = run("resumed", config, partial / "checkpoints/epoch_0001.pth")
    a = torch.load(full / "checkpoints/epoch_0002.pth", map_location="cpu", weights_only=False)
    b = torch.load(resumed / "checkpoints/epoch_0002.pth", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "validation", "training_state"):
        # NumPy RNG arrays need the existing dedicated helper.
        if key == "training_state":
            from test_diagnostics import assert_rng_equal
            assert_rng_equal(a[key]["rng"], b[key]["rng"])
            assert a[key]["optimization_control"] == b[key]["optimization_control"]
        else:
            assert not compare_states(a[key], b[key], key)
    rows = [json.loads(x) for x in (full / "optimization.jsonl").read_text().splitlines()]
    batches = [json.loads(x) for x in (full / "diagnostics/batches.jsonl").read_text().splitlines()]
    updates = [json.loads(x) for x in (full / "diagnostics/updates.jsonl").read_text().splitlines()]
    assert len(rows) == len(batches) == len(updates) == 9
    for row, batch, update in zip(rows, batches, updates):
        assert batch["gradient_l2"] == row["gradient_l2_before"]
        assert batch["optimization_control"]["phase"] == "before_gradient_clip"
        assert row["gradient_l2_after"] <= 5.00001
        assert update["optimization_control"] == {k:v for k,v in row.items() if k not in ("epoch","batch")}
    resumed_rows = [json.loads(x) for x in (resumed / "optimization.jsonl").read_text().splitlines()]
    assert resumed_rows == rows[-3:]
    payload = torch.load(full / "diagnostics/first_event.pth", map_location="cpu", weights_only=False)
    assert payload["diagnostic_record"]["optimization_control"]["phase"] == "before_gradient_clip"
    raw_norm = math.hypot(*(float(torch.linalg.vector_norm(g.double()))
                            for g in payload["gradients"].values()))
    assert raw_norm == pytest.approx(rows[0]["gradient_l2_before"], rel=1e-6)

def test_paired_source_configs():
    a = load_config(ROOT / "configs/p0_mask_x_official_stable.json").raw
    b = load_config(ROOT / "configs/p1_mask_x_normal_official_stable.json").raw
    a.pop("experiment_id"); b.pop("experiment_id")
    b["loss"].pop("normal_reference_weight")
    assert a == b
