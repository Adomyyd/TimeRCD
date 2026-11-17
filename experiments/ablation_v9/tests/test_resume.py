"""Epoch-boundary resume: compare with uninterrupted dropout training."""
import copy
import json
from pathlib import Path

import pytest
import torch

from src.diagnostics import compare_states, rng_state
from src.models import build_model
from src.models import prepare_model
from src.train import prune_epoch_checkpoints, verify_resume_config
from src.train import run_smoke, set_seed, train_model
from test_diagnostics import config_for, assert_rng_equal


@pytest.mark.parametrize("parallel", ["none", "data_parallel"])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_matches_uninterrupted_model_optimizer_rng_and_best(tmp_path, parallel, legacy):
    if parallel == "data_parallel" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    config = config_for(tmp_path, parallel)
    config["optimization"]["epochs"] = 3
    config["diagnostics"] = dict(enabled=True, every_batch=True, start_epoch=1,
                                 loss_threshold=1e20, grad_threshold=1e20)
    full = tmp_path / "full"
    full.mkdir()
    set_seed(72)
    model = prepare_model(build_model(config), config)
    run_smoke(model, config)
    expected_result = train_model(model, config, full)
    expected_rng = rng_state()
    expected = torch.load(full / "checkpoints/epoch_0002.pth", map_location="cpu", weights_only=False)
    del model

    partial = tmp_path / "partial"
    partial.mkdir()
    short = copy.deepcopy(config)
    short["optimization"]["epochs"] = 2
    set_seed(72)
    model = prepare_model(build_model(short), short)
    run_smoke(model, short)
    train_model(model, short, partial)
    del model
    checkpoint = partial / "checkpoints/epoch_0001.pth"
    if legacy:
        payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
        payload.pop("training_state")
        torch.save(payload, checkpoint)

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    # Mimic the real entry: initialize and smoke first, then restore all saved states.
    set_seed(999)
    model = prepare_model(build_model(config), config)
    run_smoke(model, config)
    actual_result = train_model(model, config, resumed, resume_from=checkpoint, checkpoint_keep_last=2)
    actual_rng = rng_state()
    actual = torch.load(resumed / "checkpoints/epoch_0002.pth", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "validation"):
        assert not compare_states(actual[key], expected[key], key)
    for key in ("best_loss", "best_epoch", "stale", "next_epoch"):
        assert actual["training_state"][key] == expected["training_state"][key]
    assert actual_result == expected_result
    assert_rng_equal(actual_rng, expected_rng)
    info = json.loads((resumed / "resume.json").read_text())
    assert info["source_epoch"] == 1 and info["next_epoch"] == 2
    rows = [json.loads(line) for line in (resumed / "diagnostics/batches.jsonl").read_text().splitlines()]
    assert {row["epoch"] for row in rows} == {2}
    for key in ("model_state_dict", "validation", "epoch"):
        best_a = torch.load(resumed / "checkpoints/best.pth", map_location="cpu", weights_only=False)
        best_b = torch.load(full / "checkpoints/best.pth", map_location="cpu", weights_only=False)
        assert not compare_states(best_a[key], best_b[key], key)


def test_resume_rejects_changed_training_conditions(tmp_path):
    previous = config_for(tmp_path)
    current = copy.deepcopy(previous)
    current["optimization"]["epochs"] = 50
    current["optimization"]["log_freq"] = 10
    current["diagnostics"] = dict(enabled=True)
    verify_resume_config(previous, current)
    current["optimization"]["batch_size"] += 1
    with pytest.raises(ValueError, match="identical"):
        verify_resume_config(previous, current)


def test_retention_keeps_best_last_two_and_first_event_predecessor(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for i in range(6):
        (checkpoints / f"epoch_{i:04d}.pth").write_bytes(b"synthetic")
    (checkpoints / "best.pth").write_bytes(b"best")
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "first_event.json").write_text(json.dumps({"epoch": 2}))
    prune_epoch_checkpoints(tmp_path, 2)
    assert sorted(p.name for p in checkpoints.iterdir()) == [
        "best.pth", "epoch_0001.pth", "epoch_0004.pth", "epoch_0005.pth"]
    assert len((tmp_path / "checkpoint_retention.jsonl").read_text().splitlines()) == 3
