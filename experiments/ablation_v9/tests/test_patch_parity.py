"""Check actual source behavior without loading a dataset or starting training."""
import ast
import importlib.util
import multiprocessing
from pathlib import Path
import sys
from typing import Tuple

import pytest
import torch

from src.config import load_config
from src.evaluate import _make_eval_inputs
from src.sampling import apply_input_mask, sample_patch_mask
from src.models import build_model
from src.train import make_batch_inputs, run_smoke, set_seed

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCES = ROOT / "audits" / "patch_20261110" / "sources"
sys.path.insert(0, str(SOURCES))
import time_rcd_config as original_config


def source_mask(filename):
    tree = ast.parse((SOURCES / filename).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "create_random_mask")
    namespace = {"torch": torch, "Tuple": Tuple,
                 "default_config": original_config.default_config}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace["create_random_mask"]


def source_entry_patch_assignments(filename):
    tree = ast.parse((SOURCES / filename).read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    assignments = [node for node in ast.walk(main) if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) in ("config", "config.ts_config.patch_size")]
    namespace = {"default_config": original_config.default_config}
    exec(compile(ast.Module(body=sorted(assignments, key=lambda node: node.lineno),
                            type_ignores=[]), filename, "exec"), namespace)
    return namespace["config"]


@pytest.mark.parametrize("filename", ["upstream_training.py", "local_training.py", "train_400k.py"])
def test_entry_mutates_same_default_object(filename, monkeypatch):
    monkeypatch.setattr(original_config, "default_config", original_config.TimeRCDConfig())
    assert original_config.default_config.ts_config.patch_size == 4
    config = source_entry_patch_assignments(filename)
    assert config is original_config.default_config
    assert original_config.default_config.ts_config.patch_size == 16
    x = torch.ones(1, 1024, 1)
    _, mask = source_mask("local_training.py")(x, torch.ones(1, 1024, dtype=torch.bool))
    assert mask.sum().item() == 144  # floor(64 * .15) * 16; size 4 would select 152.


@pytest.mark.parametrize("block_size", [4, 16])
@pytest.mark.parametrize("filename", ["upstream_training.py", "local_training.py"])
def test_sample_and_zero_fill_match_original(filename, block_size, monkeypatch):
    monkeypatch.setattr(original_config, "default_config", original_config.TimeRCDConfig())
    original_config.default_config.ts_config.patch_size = block_size
    lengths = torch.tensor([0, 1, 3, 15, 16, 27, 31, 33, 127, 1023, 1024])
    valid = torch.arange(1024)[None, :] < lengths[:, None]
    x = torch.arange(valid.numel() * 2, dtype=torch.float32).reshape(len(lengths), 1024, 2)
    torch.manual_seed(72)
    expected_x, expected_mask = source_mask(filename)(x, valid)
    actual = sample_patch_mask(valid, block_size, 0.15, torch.Generator().manual_seed(72))
    assert torch.equal(actual, expected_mask.bool())
    assert torch.equal(apply_input_mask(x, actual), expected_x)


@pytest.mark.parametrize("variant", ["p0_mask_x", "p1_full_x", "p2_full_x0"])
def test_train_and_eval_inputs_match_protocol(variant):
    config = load_config(ROOT / "configs" / f"{variant}.json").raw
    batch = {"x": torch.ones(2, 1024, 1), "valid": torch.ones(2, 1024, dtype=torch.bool)}
    train = make_batch_inputs(batch, config, torch.Generator().manual_seed(72))
    evaluation = _make_eval_inputs(batch, config, torch.Generator().manual_seed(72))
    assert all(torch.equal(a, b) for a, b in zip(train, evaluation))
    x_input, visible, selected = train
    assert selected.sum(1).tolist() == [144, 144]
    if variant == "p0_mask_x":
        assert torch.equal(visible, batch["valid"] & ~selected)
        assert (x_input[selected] == 0).all()
    else:
        assert torch.equal(x_input, batch["x"])
        assert torch.equal(visible, batch["valid"])


def test_encoder_keeps_partially_visible_token():
    spec = importlib.util.spec_from_file_location(
        "audit_encoder", REPO / "time_rcd" / "_core" / "ts_encoder_bi_bias.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.TimeSeriesEncoder(d_model=32, d_proj=8, patch_size=16,
                                      num_layers=1, num_heads=4, activation="gelu")
    captured = []
    def capture(module, args, kwargs):
        captured.append(kwargs["attn_mask"].clone())
    handle = encoder.transformer_encoder.register_forward_pre_hook(capture, with_kwargs=True)
    visible = torch.ones(1, 48, dtype=torch.bool)
    visible[:, :4] = False
    visible[:, 16:32] = False
    with torch.no_grad():
        output = encoder(torch.ones(1, 48, 1), visible)
    handle.remove()
    assert captured[0].tolist() == [[True, False, True]]
    assert output.shape == (1, 48, 1, 8)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("variant", ["p0_mask_x", "p1_full_x", "p2_full_x0"])
def test_default_model_cpu_smoke(variant):
    config = load_config(ROOT / "configs" / f"{variant}.json").raw
    config["runtime"]["device"] = "cpu"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        set_seed(72)
        model = build_model(config)
        result = run_smoke(model, config)
        assert result["status"] == "passed"
        assert result["mask_count"] == 96
        assert result["encoder_patch_size"] == result["mask_patch_size"] == 16
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.encoder.embedding_layer.weight.grad is not None
        assert model.reconstruction_head[0].weight.grad is not None
        assert model.anomaly_head[0].weight.grad is not None
    finally:
        torch.set_num_threads(previous_threads)


def spawn_probe(config, pipe):
    pipe.send((config.ts_config.patch_size, original_config.default_config.ts_config.patch_size))
    pipe.close()


def test_spawn_has_fresh_global_default(monkeypatch):
    monkeypatch.setattr(original_config, "default_config", original_config.TimeRCDConfig())
    config = source_entry_patch_assignments("upstream_training.py")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=spawn_probe, args=(config, child))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "spawn probe timed out"
        assert parent.recv() == (16, 4)
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
