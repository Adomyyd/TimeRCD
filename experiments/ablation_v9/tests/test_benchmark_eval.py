import ast
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src import benchmark_eval as bench
from src.config import load_config
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCES = ROOT / "audits" / "eval_20261112" / "sources"


@pytest.fixture
def protocol():
    result = bench.load_protocol()
    result.update(window_size=16, inference_batch_size=2, heavy_workers=1, light_workers=1)
    return result


@pytest.fixture
def model():
    config = load_config(ROOT / "configs" / "p0_mask_x.json").raw
    config = copy.deepcopy(config)
    config["model"].update(d_model=16, d_proj=8, num_layers=1, num_heads=2, patch_size=4, d_ff_dropout=0.0, dropout=0.0)
    torch.manual_seed(7)
    return build_model(config).cpu().eval()


def source_function(path, class_name, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    method = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("length", [13, 32, 35])
def test_scores_match_official_zero_shot(model, protocol, length):
    from utils.dataset import TimeRCDDataset
    core_source = REPO / "time_rcd" / "_core" / "TimeRCD_pretrain_multi.py"
    collate = source_function(core_source, None, "test_collate_fn", {"torch": torch})
    reference = source_function(SOURCES / "upstream_models_TimeRCD.py", "TimeRCDPretrainTester", "zero_shot",
                                {"torch": torch, "F": torch.nn.functional, "DataLoader": DataLoader,
                                 "TimeRCDDataset": TimeRCDDataset, "test_collate_fn": collate,
                                 "tqdm": types.SimpleNamespace(tqdm=lambda it, **kw: it)})

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anomaly_head = model.anomaly_head

        def forward(self, time_series, mask):
            return model.encoder(time_series, mask)

    tester = types.SimpleNamespace(win_size=16, batch_size=2, device=torch.device("cpu"), model=Adapter())
    values = np.sin(np.arange(length) * .31).reshape(-1, 1)
    expected_p, expected_l = reference(tester, values)
    actual_p, actual_l = bench.score_series(bench.detection_model(model), values, protocol)
    np.testing.assert_array_equal(actual_p, np.concatenate(expected_p).reshape(-1)[:length])
    np.testing.assert_array_equal(actual_l, np.concatenate(expected_l).reshape(-1)[:length])


def test_official_main_selection_and_hyperparameters(tmp_path):
    tree = ast.parse((SOURCES / "upstream_main.py").read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "filter_list" for t in n.targets)]
    assert tuple(ast.literal_eval(assignments[1].value)) == bench.EXCLUDED
    hp = {}
    exec(compile((SOURCES / "upstream_HP_list.py").read_text(), "HP_list.py", "exec"), hp)
    protocol = bench.load_protocol()
    assert protocol["window_size"] == hp["Optimal_Uni_algo_HP_dict"]["Time_RCD"]["win_size"] == 5000
    assert protocol["inference_batch_size"] == hp["Optimal_Uni_algo_HP_dict"]["Time_RCD"]["batch_size"] == 64
    for group, count in bench.EXPECTED_COUNTS.items():
        for index in range(count):
            (tmp_path / f"{index}_{group}_id_x_tr_5_1st_10.csv").touch()
    (tmp_path / "0_OPPORTUNITY_id_x_tr_5_1st_10.csv").touch()
    assert len(bench.real_targets(tmp_path)) == 700
    next(tmp_path.glob("*_IOPS_*.csv")).unlink()
    with pytest.raises(ValueError, match="inventory"):
        bench.real_targets(tmp_path)


def test_csv_dropna_then_filename_boundary(tmp_path):
    path = tmp_path / "0_IOPS_id_x_tr_2_1st_4.csv"
    pd.DataFrame({"value": [1, np.nan, 3, 5, 7], "Label": [0, 0, 0, 1, 0]}).to_csv(path, index=False)
    values, labels, boundary, full = bench.read_real_series(path)
    assert boundary == 2
    np.testing.assert_array_equal(values[:, 0], [5, 7])
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_array_equal(full[:, 0], [1, 3, 5, 7])


def test_all_metrics_match_pinned_official_fast(protocol):
    import evaluation
    import evaluation.metrics
    basic_name = "evaluation._v9_official_basic"
    metrics_name = "evaluation._v9_official_metrics"
    spec = importlib.util.spec_from_file_location(basic_name, SOURCES / "upstream_evaluation_basic_metrics.py")
    basic = importlib.util.module_from_spec(spec)
    sys.modules[basic_name] = basic
    spec.loader.exec_module(basic)
    official = types.ModuleType(metrics_name)
    official.__package__ = "evaluation"
    sys.modules[metrics_name] = official
    text = (SOURCES / "upstream_evaluation_metrics.py").read_text().replace("from .basic_metrics import", "from ._v9_official_basic import")
    exec(compile(text, str(SOURCES / "upstream_evaluation_metrics.py"), "exec"), official.__dict__)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        labels = np.zeros(64, dtype=int)
        labels[8:15] = 1
        labels[33:43] = 1
        scores = np.round(.5 + .23 * np.sin(np.arange(64) * .37) + labels * .18, 3)
        expected = official.get_metrics_optimized(scores, labels, slidingWindow=5,
                    pred=scores > scores.mean() + 3 * scores.std(), heavy_workers=1, light_workers=1,
                    f1_t_n_splits=800, f1_t_chunk_size=25)
        actual = bench.official_metrics(scores, labels, 5, protocol)
        for key in ("AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC", "Standard-F1", "PA-F1", "Affiliation-F", "F1_T"):
            assert float(expected[key]) > 0, key
        assert set(actual) == set(bench.METRIC_KEYS) == set(expected)
        for key in bench.METRIC_KEYS:
            assert actual[key] == pytest.approx(float(expected[key]), abs=1e-12), key
    finally:
        torch.set_num_threads(previous_threads)


def test_checkpoint_mismatch_rejected():
    config = load_config(ROOT / "configs" / "p0_mask_x.json").raw
    bench.verify_checkpoint_config({"config": copy.deepcopy(config)}, config)
    saved = copy.deepcopy(config)
    saved["experiment_id"] = "p1_full_x"
    with pytest.raises(ValueError, match="experiment_id"):
        bench.verify_checkpoint_config({"config": saved}, config)


def test_joint_pipeline_split_full_inputs_and_partial_status(tmp_path, model, protocol, monkeypatch):
    config = copy.deepcopy(load_config(ROOT / "configs" / "p0_mask_x.json").raw)
    samples = [{"time_series": np.arange(13) + i, "labels": [0] * 7 + [1] * 6} for i in range(6)]
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps(samples))
    config["data"]["path"] = str(path)
    config["split"]["train_ratio"] = .5
    real = tmp_path / "0_IOPS_id_x_tr_2_1st_6.csv"
    pd.DataFrame({"value": np.arange(13), "Label": [0] * 7 + [1] * 6}).to_csv(real, index=False)
    monkeypatch.setattr(bench, "real_targets", lambda _: [("IOPS", real)])
    calls = []

    def metrics(scores, labels, sw, p):
        calls.append(len(scores))
        return {key: .25 for key in bench.METRIC_KEYS}

    monkeypatch.setattr(bench, "official_metrics", metrics)
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"fixture checkpoint, never loaded")
    run = tmp_path / "run"
    result = bench.evaluate_benchmarks(model, config, run, checkpoint, protocol,
                                     checkpoint_epoch=31, max_synthetic=2)
    assert result["full_benchmark"] is False
    _, held_out = bench.load_indices(path, .5, config["split"]["seed"])
    assert json.loads((run / "metrics/synthetic_indices.json").read_text()) == held_out[:2]
    assert calls == [13, 13, 13, 13, 11, 11]
    assert len(list((run / "scores").glob("*.npz"))) == 3
    assert result["summaries"]["real"]["probability"]["series_count"] == 1
    assert len(pd.read_csv(run / "metrics/per_series.csv")) == 6
    assert len(pd.read_csv(run / "metrics/group_mean_metrics.csv")) == 4
    assert len(pd.read_csv(run / "metrics/domain_summary.csv")) == 8


def test_metrics_failure_preserves_partial_rows(tmp_path, model, protocol, monkeypatch):
    config = copy.deepcopy(load_config(ROOT / "configs" / "p0_mask_x.json").raw)
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps([{"time_series": np.arange(13), "labels": [0]*7+[1]*6}] * 2))
    config["data"]["path"] = str(path)
    config["split"]["train_ratio"] = .5
    monkeypatch.setattr(bench, "real_targets", lambda _: [("IOPS", path)])
    count = 0

    def fail_second(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("fixture metric failure")
        return {key: .25 for key in bench.METRIC_KEYS}

    monkeypatch.setattr(bench, "official_metrics", fail_second)
    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="fixture metric failure"):
        bench.evaluate_benchmarks(model, config, run, path, protocol, checkpoint_epoch=0)
    assert not (run / "metrics/eval.json").exists()
    assert json.loads((run / "metrics/eval_failure.json").read_text())["completed_score_rows"] == 1
    assert len(pd.read_csv(run / "metrics/per_series.csv")) == 1


def test_group_macro_is_not_pooled_series_macro():
    rows = [{"domain": "real", "score_kind": "probability", "group": group,
             **{key: value for key in bench.METRIC_KEYS}}
            for group, value in [("IOPS", 0), ("IOPS", 0), ("NAB", 1)]]
    result = bench.summarize(rows)["real"]["probability"]
    assert result["group_macro"]["AUC-PR"] == .5
    assert result["series_macro"]["AUC-PR"] == pytest.approx(1/3)


def test_unified_eval_entry_loads_checkpoint_and_runs_both_domains(tmp_path, protocol, monkeypatch):
    import run_experiment as entry
    config = copy.deepcopy(load_config(ROOT / "configs" / "p0_mask_x.json").raw)
    config["model"].update(d_model=16, d_proj=8, num_layers=1, num_heads=2, patch_size=4)
    config["runtime"].update(device="cpu", parallel="none", gpu_ids=None)
    config["split"]["train_ratio"] = .5
    data = tmp_path / "data.pkl"
    data.write_bytes(pickle.dumps([{"time_series": np.arange(13), "labels": [0]*7+[1]*6}] * 2))
    manifest = tmp_path / "data.manifest.json"
    manifest.write_text(json.dumps({"config": {"is_multivariate": False, "num_features": 1}, "actual_samples": 2}))
    config["data"].update(path=str(data), manifest_path=str(manifest))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    protocol["save_scores"] = False
    protocol_path = tmp_path / "eval.json"
    protocol_path.write_text(json.dumps(protocol))
    checkpoint = tmp_path / "best.pth"
    torch.save({"config": config, "epoch": 31, "model_state_dict": build_model(config).state_dict()}, checkpoint)
    csv_path = tmp_path / "0_IOPS_id_x_tr_2_1st_4.csv"
    pd.DataFrame({"value": np.arange(13), "Label": [0]*7+[1]*6}).to_csv(csv_path, index=False)
    monkeypatch.setattr(bench, "real_targets", lambda _: [("IOPS", csv_path)])
    monkeypatch.setattr(bench, "official_metrics", lambda *args: {key: .5 for key in bench.METRIC_KEYS})
    monkeypatch.setattr(entry, "HERE", tmp_path)
    args = argparse.Namespace(config=str(config_path), run_id="eval_fixture", mode="eval",
            checkpoint=str(checkpoint), eval_config=str(protocol_path), parallel=None, gpu_ids=None,
            batch_size=None, eval_max_synthetic=None, eval_max_real=None)
    entry.run_one(args)
    run = tmp_path / "runs/eval_fixture"
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    result = json.loads((run / "metrics/eval.json").read_text())
    assert result["checkpoint_epoch"] == 31
    assert result["summaries"]["synthetic"]["probability"]["series_count"] == 1
    assert result["summaries"]["real"]["probability"]["series_count"] == 1
    assert not (run / "metrics/train.json").exists()
