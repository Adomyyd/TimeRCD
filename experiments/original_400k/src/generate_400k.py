from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT_DIR.parents[1]
OUT_DIR = EXPERIMENT_DIR / "data"
OUT_PATH = OUT_DIR / "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set.pkl"
MANIFEST_PATH = OUT_DIR / "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set.manifest.json"
sys.path.insert(0, str(REPO / "dataset_gen" / "src"))
os.chdir(REPO / "dataset_gen")
from generate_dataset import generate_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "num_samples": 400000,
        "seq_len": None,
        "anomaly_sample_ratio": 1.0,
        "is_multivariate": False,
        "num_features": 1,
        "use_attribute_set": True,
        "num_workers": 8,
        "generator": "dataset_gen/src/generate_dataset.py",
        "seed": None,
    }
    print("Generation configuration:", json.dumps(config, ensure_ascii=False), flush=True)
    dataset = generate_dataset(**{k: v for k, v in config.items()
                                  if k in {
                                      "num_samples", "seq_len", "anomaly_sample_ratio",
                                      "is_multivariate", "num_features",
                                      "use_attribute_set", "num_workers"
                                  }})
    lengths = np.fromiter((len(item["time_series"]) for item in dataset), dtype=np.int64)
    anomaly_points = sum(int(np.count_nonzero(item["labels"])) for item in dataset)
    temp_path = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    with temp_path.open("wb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, OUT_PATH)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "actual_samples": len(dataset),
        "total_time_points": int(lengths.sum()),
        "min_sequence_length": int(lengths.min()),
        "max_sequence_length": int(lengths.max()),
        "mean_sequence_length": float(lengths.mean()),
        "samples_with_anomaly": sum(bool(np.any(item["labels"])) for item in dataset),
        "anomaly_labeled_points": anomaly_points,
        "sample_keys": sorted(dataset[0].keys()),
        "pickle_bytes": OUT_PATH.stat().st_size,
        "pickle_sha256": sha256_file(OUT_PATH),
        "data_path": str(OUT_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
