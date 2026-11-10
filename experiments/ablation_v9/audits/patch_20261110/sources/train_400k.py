from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/diskA/YYD/TimeRCD")
RUN_DIR = REPO / "experiments" / "runs" / "time_rcd_400k_20261108"
DATA_DIR = REPO / "experiments"
DATASET = "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set.pkl"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(REPO))

import torch
from training import train_worker
from time_rcd._core.time_rcd_config import default_config


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume official TimeRCD 400k pretraining")
    parser.add_argument(
        "--resume",
        default=str(RUN_DIR / "checkpoints" / "pretrain_checkpoint_latest.pth"),
        help="Absolute checkpoint path; defaults to the latest checkpoint in this run",
    )
    args = parser.parse_args()
    resume_path = Path(args.resume).resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    dataset_path = DATA_DIR / DATASET
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "train.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(original_stdout, log)
        sys.stderr = Tee(original_stderr, log)
        try:
            config = default_config
            config.num_epochs = 50
            config.learning_rate = 5e-4
            config.weight_decay = 1e-5
            config.batch_size = 16
            config.ts_config.patch_size = 16
            config.ts_config.num_features = 1
            config.dropout = 0.1
            config.seed = 72
            config.cuda_devices = "0"
            config.dist_port = "16979"
            config.mixed_precision = False
            config.early_stopping_patience = 7
            config.num_workers = 0
            config.log_freq = 100
            config.save_freq = 5
            config.pretrain_data_path = str(DATA_DIR)
            config.checkpoint_dir = str(RUN_DIR / "checkpoints")
            config.continuation_checkpoint = str(resume_path)

            config_record = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(dataset_path),
                "dataset_samples": 400000,
                "dataset_time_points": 1576189006,
                "epochs_max": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "patch_size": config.ts_config.patch_size,
                "seed": config.seed,
                "gpu_visible": "0",
                "mixed_precision": config.mixed_precision,
                "early_stopping_patience": config.early_stopping_patience,
                "data_split": "official dataset class random 95/5 split, seed 42",
                "checkpoint_dir": config.checkpoint_dir,
                "resume_checkpoint": str(resume_path),
                "notes": [
                    "Official training.py model and loss implementation.",
                    "Batch size 16 is selected for a single 11 GB RTX 2080 Ti.",
                    "The official training code's label-padding behavior is preserved."
                ]
            }
            (RUN_DIR / "train_config.json").write_text(
                json.dumps(config_record, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(config_record, indent=2), flush=True)
            torch.cuda.set_device(0)
            train_worker(0, 1, config, DATASET)
            print("TRAINING_FINISHED", flush=True)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()