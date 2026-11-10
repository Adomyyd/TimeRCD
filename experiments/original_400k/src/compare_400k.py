from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO))
RUN_DIR = EXPERIMENT_DIR / "runs" / "time_rcd_400k_20261108"
NEW_CHECKPOINT = RUN_DIR / "checkpoints" / "pretrain_checkpoint_best.pth"
OFFICIAL_CHECKPOINT = REPO / "best_model" / "pretrain_checkpoint_best_uni.pth"
DATA_DIR = REPO / "datasets" / "TSB-AD-U"
OUT_DIR = RUN_DIR / "comparison"
TARGET_GROUPS = [
    "IOPS", "MGAB", "NAB", "NEK", "Power", "SED",
    "Stock", "TODS", "UCR", "WSD", "YAHOO",
]
WINDOW_SIZE = 5000
METRICS = [
    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
    "Standard-F1", "F1_T", "Affiliation-F",
]


def target_group(filename: str) -> str | None:
    stem = Path(filename).stem
    for group in TARGET_GROUPS:
        if group.lower() in stem.lower():
            return group
    return None


def load_targets() -> list[tuple[str, Path]]:
    targets = []
    found_groups = set()
    for path in sorted(DATA_DIR.glob("*.csv")):
        group = target_group(path.name)
        if group is not None:
            targets.append((group, path))
            found_groups.add(group)
    missing = [group for group in TARGET_GROUPS if group not in found_groups]
    if missing:
        raise FileNotFoundError(f"Missing paper benchmark groups in TSB-AD-U: {missing}")
    return targets


def make_tester(checkpoint: Path) -> TimeRCDPretrainTester:
    config = default_config
    config.ts_config.patch_size = 16
    config.ts_config.num_features = 1
    config.win_size = WINDOW_SIZE
    config.batch_size = 1
    config.cuda_devices = "0"
    return TimeRCDPretrainTester(str(checkpoint), config)


def main() -> None:
    if not NEW_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"New best checkpoint not found: {NEW_CHECKPOINT}. Finish training first."
        )
    if not OFFICIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(OFFICIAL_CHECKPOINT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "new_checkpoint": str(NEW_CHECKPOINT),
        "official_checkpoint": str(OFFICIAL_CHECKPOINT),
        "dataset_dir": str(DATA_DIR),
        "paper_univariate_groups": TARGET_GROUPS,
        "matched_file_count": len(targets),
        "matched_files_by_group": {
            group: sum(1 for current_group, _ in targets if current_group == group)
            for group in TARGET_GROUPS
        },
        "window_size": WINDOW_SIZE,
        "scoring": "anomaly-head softmax probability",
        "protocol": "all matched TSB-AD test series; repository TimeRCD preprocessing and optimized metrics",
        "metrics_workers": {"heavy": 4, "light": 4},
        "aggregation": "mean each metric within each paper dataset group, then unweighted mean across groups",
    }
    (OUT_DIR / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    results = []

    for model_name, checkpoint in [
        ("trained_400k", NEW_CHECKPOINT),
        ("official", OFFICIAL_CHECKPOINT),
    ]:
        print(f"Loading {model_name}: {checkpoint}", flush=True)
        tester = make_tester(checkpoint)
        for index, (group, csv_path) in enumerate(targets, start=1):
            try:
                frame = pd.read_csv(csv_path).dropna()
                values = frame.iloc[:, :-1].to_numpy(dtype=float)
                labels = frame["Label"].to_numpy(dtype=int)
                train_index = int(csv_path.stem.split("_")[-3])
                test_data = values[train_index:]
                test_labels = labels[train_index:]
                if test_data.ndim != 2 or test_data.shape[1] != 1:
                    raise ValueError(f"Expected univariate test data, got {test_data.shape}")

                score_batches, _ = tester.zero_shot(test_data)
                scores = np.concatenate(
                    [np.asarray(batch).reshape(-1) for batch in score_batches]
                )
                aligned = min(len(scores), len(test_labels))
                scores = scores[:aligned]
                aligned_labels = test_labels[:aligned]
                if aligned == 0:
                    raise ValueError("Empty aligned test output")

                sliding_window = find_length_rank(values, rank=1)
                metrics = get_metrics_optimized(
                    scores,
                    aligned_labels,
                    slidingWindow=sliding_window,
                    pred=scores > (np.mean(scores) + 3 * np.std(scores)),
                    heavy_workers=4,
                    light_workers=4,
                    f1_t_n_splits=800,
                    f1_t_chunk_size=25,
                )
                row = {
                    "model": model_name,
                    "group": group,
                    "filename": csv_path.name,
                    "test_points": len(test_labels),
                    "aligned_points": aligned,
                    **metrics,
                }
                results.append(row)
                pd.DataFrame(results).to_csv(OUT_DIR / "per_series_metrics.csv", index=False)
                print(
                    f"[{index}/{len(targets)}] {model_name} {group} {csv_path.name}: "
                    f"AUC-PR={metrics.get('AUC-PR')} "
                    f"F1_T={metrics.get('F1_T')} "
                    f"Standard-F1={metrics.get('Standard-F1')} "
                    f"VUS-PR={metrics.get('VUS-PR')}",
                    flush=True,
                )
            except Exception as exc:
                results.append({
                    "model": model_name,
                    "group": group,
                    "filename": csv_path.name,
                    "error": repr(exc),
                })
                pd.DataFrame(results).to_csv(OUT_DIR / "per_series_metrics.csv", index=False)
                traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del tester
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(results)
    valid = frame[frame["error"].isna()] if "error" in frame else frame
    grouped = valid.groupby(["model", "group"])[METRICS].mean(numeric_only=True)
    grouped.to_csv(OUT_DIR / "group_mean_metrics.csv")
    summary = grouped.groupby("model").mean(numeric_only=True)
    summary.to_csv(OUT_DIR / "mean_metrics_across_11_groups.csv")
    print("Mean metrics across the 11 paper groups:", flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    import torch
    from evaluation.metrics import get_metrics_optimized
    from models.TimeRCD import TimeRCDPretrainTester
    from time_rcd._core.time_rcd_config import default_config
    from utils.slidingWindows import find_length_rank

    main()
