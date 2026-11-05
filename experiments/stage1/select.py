"""Freeze a metadata-stratified development set before running any model."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.common.data import geometry, load_series


QUOTAS = {"short": 8, "full": 1, "low_tail": 7, "other_tail": 8}


def stratum(length):
    if length < 5000:
        return "short"
    if length % 5000 == 0:
        return "full"
    return "low_tail" if length % 5000 <= 1000 else "other_tail"


def choose(candidates, seed, quotas=QUOTAS):
    chosen = []
    sources, densities, alignments = Counter(), Counter(), Counter()
    for group, count in quotas.items():
        pool = [row for row in candidates if row["stratum"] == group]
        if len(pool) < count:
            raise ValueError(f"Stratum {group} has {len(pool)} candidates, needs {count}")
        for _ in range(count):
            def priority(row):
                tie = hashlib.sha256(f"{seed}:{row['filename']}".encode()).hexdigest()
                return (sources[row["source"]], densities[row["density_bin"]],
                        alignments[row["patch_aligned"]], tie)
            row = min(pool, key=priority)
            pool.remove(row)
            chosen.append(row)
            sources[row["source"]] += 1
            densities[row["density_bin"]] += 1
            alignments[row["patch_aligned"]] += 1
    return chosen


def inventory(metadata, data_root, minimum, maximum):
    columns = ["filename", "data_shape", "train_index"]
    frame = pd.read_csv(metadata, usecols=columns)
    if frame.filename.duplicated().any():
        raise ValueError("Duplicate filenames in metadata")
    records, eligible = [], []
    for row in frame.sort_values("filename").itertuples(index=False):
        fields = dict(zip(frame.columns, row))
        filename = fields["filename"]
        total, channels = map(int, fields["data_shape"].split("x"))
        length = total - int(fields["train_index"])
        record = {"filename": filename, "metadata_length": length}
        if not minimum <= length <= maximum or channels != 1:
            records.append({**record, "status": "outside_length_or_channel_bounds"})
            continue
        settings = {"kind": "tsb", "root": str(data_root), "files": [filename], "split": "test"}
        try:
            series = next(load_series(settings, 0))
        except ValueError as exc:
            records.append({**record, "status": "invalid_data", "reason": str(exc)})
            continue
        if len(series.values) != length:
            records.append({**record, "status": "metadata_length_mismatch", "actual_length": len(series.values)})
            continue
        ratio = float(series.labels.mean())
        if ratio == 0 or ratio == 1:
            records.append({**record, "status": "single_class"})
            continue
        record.update({
            "status": "eligible", "length": length, "source": filename.split("_")[1],
            "stratum": stratum(length), "patch_aligned": length % 16 == 0,
            "anomaly_ratio": ratio, "anomaly_count": int(series.labels.sum()),
            "density_bin": "low" if ratio <= 0.01 else ("medium" if ratio <= 0.1 else "high"),
            "sha256": series.source["sha256"], "offset": series.source["offset"],
        })
        eligible.append(record)
        records.append(record)
    return records, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="Uni_Time_RCD_v4_wins-5000.csv")
    parser.add_argument("--data-root", default="datasets/TSB-AD-U")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--template", default="experiments/configs/stage1_tsb_sample.json")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    metadata, root = project_path(args.metadata), project_path(args.data_root)
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records, eligible = inventory(metadata, root, 256, 20000)
    selected = choose(eligible, args.seed)
    config = json.loads(project_path(args.template).read_text())
    config.update(name="stage1_real_v1", device=args.device, seed=args.seed,
                  force_short_padding=True, batch_size=1)
    config["data"] = {"kind": "tsb", "root": str(root), "split": "test",
                      "files": [row["filename"] for row in selected],
                      "selection_manifest": str(output / "selection.json")}
    protocol = {
        "role": "development_diagnostic", "seed": args.seed,
        "universe": "Only files listed in the existing 700-row evaluation metadata; not all TSB-AD-U",
        "metadata_path": str(metadata), "metadata_sha256": sha256(metadata),
        "metadata_columns_read": ["filename", "data_shape", "train_index"],
        "length_bounds_inclusive": [256, 20000], "stratification_window": 5000,
        "quotas": QUOTAS, "selection_order": "stratum order; greedy least represented source, density, alignment; SHA256(seed:filename) tie break",
        "density_bins": {"low": "(0, .01]", "medium": "(.01, .1]", "high": "(.1, 1)"},
        "exclusions": dict(Counter(row["status"] for row in records)),
        "primary_score": "scores", "primary_metric": "average_precision",
        "supplementary_score": "logits", "material_ap_difference": 0.01,
        "bootstrap": {"unit": "series", "repetitions": 2000, "seed": args.seed},
        "window_sizes": config["window_sizes"], "selected": selected,
    }
    write_json(output / "inventory.json", records)
    write_json(output / "selection.json", protocol)
    write_json(output / "config.json", config)
    coverage = [{"filename": row["filename"], **geometry(row["length"], window)}
                for row in selected for window in config["window_sizes"]]
    write_json(output / "coverage.json", coverage)
    print(f"Frozen {len(selected)} series from {len(eligible)} eligible candidates; config: {output / 'config.json'}")
    print("Sources:", dict(Counter(row["source"] for row in selected)))
    print("Density bins:", dict(Counter(row["density_bin"] for row in selected)))


if __name__ == "__main__":
    main()
