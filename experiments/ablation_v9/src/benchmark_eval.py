"""Detection evaluation; training-time masked validation remains in evaluate.py."""
from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import load_data, load_indices, sha256
from .parallel import unwrap_model


UPSTREAM_COMMIT = "d458549f6962029a5c5241c43a5baa1ec5eccbb2"
GROUPS = ("IOPS", "MGAB", "NAB", "NEK", "Power", "SED", "Stock", "TODS", "UCR", "WSD", "YAHOO")
EXPECTED_COUNTS = dict(zip(GROUPS, (17, 9, 28, 9, 1, 3, 20, 15, 228, 111, 259)))
EXCLUDED = ("Daphnet", "CATSv2", "SWaT", "LTDB", "TAO", "Exathlon", "MITDB", "MSL", "SMAP", "SMD", "SVDB", "OPP")
METRIC_KEYS = (
    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
    "Standard-F1", "Standard-Precision", "Standard-Recall",
    "PA-F1", "PA-Precision", "PA-Recall",
    "Affiliation-F", "Affiliation-P", "Affiliation-R",
    "F1_T", "Precision_T", "Recall_T",
)
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "eval_official.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"protocol_id", "real_data_dir", "window_size", "inference_batch_size", "metrics_mode",
                "heavy_workers", "light_workers", "cpu_threads", "f1_t_n_splits", "f1_t_chunk_size", "seed", "save_scores"}
    if set(protocol) != required:
        raise ValueError(f"Evaluation protocol fields must be {sorted(required)}")
    if protocol["protocol_id"] != "official_uni_d458549":
        raise ValueError("Unsupported evaluation protocol_id")
    for key in ("window_size", "inference_batch_size", "heavy_workers", "light_workers", "cpu_threads", "f1_t_chunk_size"):
        if type(protocol[key]) is not int or protocol[key] <= 0:
            raise ValueError(f"Evaluation {key} must be a positive integer")
    if type(protocol["f1_t_n_splits"]) is not int or protocol["f1_t_n_splits"] < 50:
        raise ValueError("Evaluation f1_t_n_splits must be an integer >= 50")
    if protocol["metrics_mode"] not in {"fast", "default"}:
        raise ValueError("Evaluation metrics_mode must be fast or default")
    if type(protocol["save_scores"]) is not bool or type(protocol["seed"]) is not int:
        raise ValueError("Evaluation save_scores/seed must be bool/int")
    if not 0 <= protocol["seed"] < 2 ** 32:
        raise ValueError("Evaluation seed must be in [0, 2**32)")
    return protocol


def real_targets(data_dir: str | Path) -> list[tuple[str, Path]]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"TSB-AD-U directory not found: {root}")
    targets = []
    for path in sorted(root.glob("*.csv")):
        if any(token in path.name for token in EXCLUDED):
            continue
        group = path.stem.split("_")[1] if "_" in path.stem else ""
        if group not in GROUPS:
            raise ValueError(f"Unexpected group admitted by official filter: {path.name}")
        targets.append((group, path))
    missing = set(GROUPS) - {group for group, _ in targets}
    if missing:
        raise FileNotFoundError(f"Missing official univariate benchmark groups: {sorted(missing)}")
    counts = {group: sum(current == group for current, _ in targets) for group in GROUPS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Incomplete or unexpected TSB-AD-U inventory: {counts}; expected {EXPECTED_COUNTS}")
    return targets


def read_real_series(path: Path) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    # The filename boundary is applied after dropna, exactly as in official main.py.
    frame = pd.read_csv(path).dropna()
    if frame.columns[-1] != "Label":
        raise ValueError(f"Expected Label as last column: {path}")
    values = frame.iloc[:, :-1].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy(dtype=int)
    boundary = int(path.stem.split("_")[-3])
    if values.ndim != 2 or values.shape[1] != 1 or not 0 <= boundary < len(values):
        raise ValueError(f"Invalid univariate series or train boundary: {path}")
    return values[boundary:], labels[boundary:], boundary, values


class DetectionHead(nn.Module):
    """Use only the trained encoder and anomaly head, as official zero_shot does."""
    def __init__(self, core: nn.Module):
        super().__init__()
        self.encoder = core.encoder
        self.anomaly_head = core.anomaly_head

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.anomaly_head(self.encoder(x, mask)).mean(dim=-2)


def detection_model(model: nn.Module) -> nn.Module:
    scorer = DetectionHead(unwrap_model(model))
    if isinstance(model, nn.DataParallel):
        scorer = nn.DataParallel(scorer, device_ids=model.device_ids, output_device=model.output_device)
    return scorer.eval()


def score_series(scorer: nn.Module, values: np.ndarray, protocol: dict) -> tuple[np.ndarray, np.ndarray]:
    from utils.dataset import TimeRCDDataset

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite univariate sequence")
    window = min(int(protocol["window_size"]), len(values))
    dataset = TimeRCDDataset(values, window_size=window, stride=window, normalize=True)
    loader = DataLoader(dataset, batch_size=int(protocol["inference_batch_size"]), shuffle=False, num_workers=0)
    device = next(scorer.parameters()).device
    probabilities, margins = [], []
    with torch.no_grad():
        for x, valid in loader:
            logits = scorer(x.to(device), valid.to(device))
            probabilities.append(torch.softmax(logits, dim=-1)[..., 1].cpu().numpy().reshape(-1))
            margins.append((logits[..., 1] - logits[..., 0]).cpu().numpy().reshape(-1))
    # Official flattens padded windows and then truncates to the original test length.
    probability = np.concatenate(probabilities)[:len(values)]
    margin = np.concatenate(margins)[:len(values)]
    if len(probability) != len(values) or not np.isfinite(probability).all() or not np.isfinite(margin).all():
        raise ValueError("Nonfinite scores or incomplete sequence coverage")
    return probability, margin


def official_metrics(scores: np.ndarray, labels: np.ndarray, sliding_window: int, protocol: dict) -> dict:
    from evaluation.metrics import get_metrics, get_metrics_optimized

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError("Metrics require aligned nonempty 1D scores and labels")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Metrics require binary labels and finite scores")
    pred = scores > (np.mean(scores) + 3 * np.std(scores))
    if protocol["metrics_mode"] == "fast":
        result = get_metrics_optimized(
            scores, labels, slidingWindow=sliding_window, pred=pred,
            heavy_workers=protocol["heavy_workers"], light_workers=protocol["light_workers"],
            f1_t_n_splits=protocol["f1_t_n_splits"], f1_t_chunk_size=protocol["f1_t_chunk_size"],
        )
    else:
        result = get_metrics(scores, labels, slidingWindow=sliding_window, pred=pred)
    # Do not silently aggregate absent/nonfinite results. Internal official zero
    # fallbacks remain part of the reused implementation and are recorded in the manifest.
    result = {key: float(result[key]) for key in METRIC_KEYS}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f"Official metric returned nonfinite results: {result}")
    return result


def verify_checkpoint_config(payload: dict, config: dict) -> None:
    saved = payload.get("config")
    if not isinstance(saved, dict):
        raise ValueError("v9 evaluation requires a checkpoint containing its training config")
    for field in ("experiment_id", "model", "input", "loss", "split"):
        if saved.get(field) != config[field]:
            raise ValueError(f"Checkpoint/config mismatch: {field}; use the training run config.json")
    if Path(saved["data"]["path"]).resolve() != Path(config["data"]["path"]).resolve():
        raise ValueError("Checkpoint/config mismatch: synthetic dataset path")


def summarize(rows: list[dict]) -> dict:
    result = {}
    for domain in ("synthetic", "real"):
        result[domain] = {}
        for score_kind in ("probability", "logit"):
            selected = [row for row in rows if row["domain"] == domain and row["score_kind"] == score_kind]
            groups = sorted({row["group"] for row in selected})
            per_group = {
                group: {key: float(np.mean([row[key] for row in selected if row["group"] == group]))
                        for key in METRIC_KEYS}
                for group in groups
            }
            result[domain][score_kind] = {
                "series_count": len(selected), "group_means": per_group,
                "series_macro": {key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS} if selected else {},
                "group_macro": {key: float(np.mean([per_group[group][key] for group in groups])) for key in METRIC_KEYS} if groups else {},
            }
    return result


def evaluate_benchmarks(model: nn.Module, config: dict, run_dir: Path, checkpoint: Path,
                        protocol: dict, *, checkpoint_epoch: int,
                        max_synthetic: int | None = None, max_real: int | None = None) -> dict:
    from utils.slidingWindows import find_length_rank

    for limit in (max_synthetic, max_real):
        if limit is not None and limit <= 0:
            raise ValueError("Evaluation limits must be positive; omit them for the full benchmark")
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    targets = real_targets(protocol["real_data_dir"])
    log(f"Preparing evaluation: {len(targets)} real series; loading synthetic pickle and reproducing held-out split")
    _, held_out = load_indices(config["data"]["path"], float(config["split"]["train_ratio"]), int(config["split"]["seed"]))
    if not held_out:
        raise ValueError("Empty synthetic held-out split")
    all_synthetic_count, all_real_count = len(held_out), len(targets)
    selected_indices = held_out[:max_synthetic]
    targets = targets[:max_real]
    full = len(selected_indices) == all_synthetic_count and len(targets) == all_real_count
    write_json(metrics_dir / "synthetic_indices.json", selected_indices)
    repo = Path(__file__).resolve().parents[3]
    source_paths = ("main.py", "HP_list.py", "models/TimeRCD.py", "utils/dataset.py", "utils/slidingWindows.py",
                    "evaluation/metrics.py", "evaluation/basic_metrics.py", "experiments/ablation_v9/src/benchmark_eval.py")
    official_defaults = all(protocol[key] == value for key, value in {
        "window_size": 5000, "inference_batch_size": 64, "metrics_mode": "fast",
        "f1_t_n_splits": 800, "f1_t_chunk_size": 25, "seed": 2024,
    }.items())
    manifest = {
        "protocol": protocol, "upstream_commit": UPSTREAM_COMMIT,
        "official_default_parameters": official_defaults,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "checkpoint_epoch": checkpoint_epoch,
        "source_sha256": {p: sha256(repo / p) for p in source_paths},
        "dependency_versions": {name: importlib.metadata.version(name) for name in
                                ("torch", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "tqdm")},
        "synthetic_available": all_synthetic_count, "synthetic_selected": len(selected_indices),
        "real_available": all_real_count, "real_selected": len(targets), "full_benchmark": full,
        "synthetic_indices_sha256": sha256(metrics_dir / "synthetic_indices.json"),
        "real_files": [{"group": group, "filename": path.name, "sha256": sha256(path)} for group, path in targets],
        "score_definitions": {"probability": "softmax(channel_mean_logits)[...,1]", "logit": "logits[...,1]-logits[...,0]"},
        "synthetic_protocol": "same held-out split used for validation; full raw sequence, official zero-shot inference and per-series metrics; not an independent test split",
        "aggregation": "per-series metrics; per-group mean; unweighted group macro and separate all-series macro; no concatenation across series",
        "thresholds": "official oracle benchmark search, not validation-fitted deployment thresholds; pred=mean+3std is passed exactly as main.py",
        "metric_caveat": "official metric functions catch some internal errors and return zeros; those zeros are retained; missing/nonfinite results fail this run",
        "local_metric_difference": "F1_T threshold execution is serial to avoid nested-pool stalls, with the same n_splits; see parity evidence in audits/eval_20261112",
    }
    write_json(run_dir / "evaluation_manifest.json", manifest)
    scorer = detection_model(model)
    raw = load_data(config["data"]["path"])
    rows: list[dict] = []
    fields = ["domain", "group", "series_id", "score_kind", "points", "anomaly_points", "sliding_window", "train_index", *METRIC_KEYS]
    score_dir = run_dir / "scores"
    if protocol["save_scores"]:
        score_dir.mkdir()
    with (metrics_dir / "per_series.csv").open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        csv_handle.flush()

        def evaluate_one(domain, group, series_id, values, labels, full_values, boundary):
            log(f"{domain} {series_id}: inference, points={len(labels)}")
            probability, margin = score_series(scorer, values, protocol)
            sliding_window = int(find_length_rank(full_values, rank=1))
            if protocol["save_scores"]:
                np.savez_compressed(score_dir / f"{domain}_{series_id}.npz", value=values,
                                    label=labels, anomaly_score=probability, logits=margin)
            for score_kind, scores in (("probability", probability), ("logit", margin)):
                log(f"{domain} {series_id}: {score_kind} metrics, sliding_window={sliding_window}")
                result = official_metrics(scores, labels, sliding_window, protocol)
                row = dict(domain=domain, group=group, series_id=series_id, score_kind=score_kind,
                           points=len(labels), anomaly_points=int(labels.sum()), sliding_window=sliding_window,
                           train_index=boundary, **result)
                writer.writerow(row)
                csv_handle.flush()
                rows.append(row)
                log(f"{domain} {series_id} {score_kind}: AUC-PR={result['AUC-PR']:.6f} F1_T={result['F1_T']:.6f}")

        try:
            log(f"Evaluation start: synthetic={len(selected_indices)}/{all_synthetic_count}, real={len(targets)}/{all_real_count}, full={full}")
            for index in selected_indices:
                sample = raw[index]
                values = np.asarray(sample["time_series"], dtype=float)
                if values.ndim == 1:
                    values = values[:, None]
                labels = (np.asarray(sample["labels"]).reshape(-1) > 0).astype(int)
                if len(values) != len(labels):
                    raise ValueError(f"Synthetic length mismatch at index {index}")
                evaluate_one("synthetic", "synthetic", str(index), values, labels, values, 0)
            for group, path in targets:
                values, labels, boundary, full_values = read_real_series(path)
                evaluate_one("real", group, path.stem, values, labels, full_values, boundary)
        except Exception as exc:
            write_json(metrics_dir / "eval_failure.json", {"error": repr(exc), "completed_score_rows": len(rows)})
            log(f"Evaluation failed: {exc!r}; completed per-series rows preserved")
            raise

    result = {"status": "completed", "full_benchmark": full, "protocol_id": protocol["protocol_id"],
              "official_default_parameters": official_defaults,
              "checkpoint_epoch": checkpoint_epoch, "summaries": summarize(rows)}
    group_rows, domain_rows = [], []
    for domain, score_summaries in result["summaries"].items():
        for score_kind, summary in score_summaries.items():
            for group, metrics in summary["group_means"].items():
                group_rows.append(dict(domain=domain, score_kind=score_kind, group=group, **metrics))
            for aggregation in ("series_macro", "group_macro"):
                domain_rows.append(dict(domain=domain, score_kind=score_kind, aggregation=aggregation,
                                        series_count=summary["series_count"], **summary[aggregation]))
    pd.DataFrame(group_rows).to_csv(metrics_dir / "group_mean_metrics.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(metrics_dir / "domain_summary.csv", index=False)
    write_json(metrics_dir / "eval.json", result)
    log("Evaluation completed: metrics/eval.json and metrics/per_series.csv")
    return result
