import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.stage1.analyze import bootstrap_mean
from experiments.stage2.analyze import paired_cases


def normalization_pairs(old, new):
    keys = ["case", "series", "anomaly_count", "generator_seed", "placement"]
    columns = keys + ["baseline_ap", "oracle_ap", "random_mean_ap", "oracle_minus_baseline_ap", "oracle_minus_random_ap"]
    paired = old[columns].merge(new[columns], on=keys, how="outer", suffixes=("_fixed", "_official"),
                                validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Normalization crossover has unmatched inputs")
    paired = paired.drop(columns="_merge")
    for comparison in ("oracle_minus_baseline_ap", "oracle_minus_random_ap"):
        paired[f"interaction_{comparison}"] = paired[f"{comparison}_official"] - paired[f"{comparison}_fixed"]
    return paired


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] not in ("completed", "checks_failed"):
        raise ValueError("Run has not finished")
    config = manifest["config"]
    parent = Path(config["parent_run"])
    fingerprints = json.loads((directory / "parent_artifacts.sha256.json").read_text())
    for filename in ("manifest.json", "metrics.csv"):
        if sha256(parent / filename) != fingerprints[filename]:
            raise ValueError(f"Changed parent artifact: {filename}")
    previous = paired_cases(pd.read_csv(parent / "metrics.csv"), config["random_seeds"])
    current = paired_cases(pd.read_csv(directory / "metrics.csv"), config["random_seeds"])

    def primary(frame):
        return frame[(frame.scope == "full_fallback") & (frame.score_kind == "scores") &
                     (frame.kind == "controlled_synthetic") & (frame.window == 5000)]

    paired = normalization_pairs(primary(previous), primary(current))
    if len(paired) != 27 or paired.filter(regex="_ap").isna().any().any():
        raise ValueError("Expected 27 defined paired normalization cases")
    protocol = config["protocol"]
    seed, repetitions = protocol["bootstrap_seed"], protocol["bootstrap_repetitions"]
    stats = {"status": manifest["status"], "analysis_sha256": sha256(Path(__file__)),
             "normalization": {}, "score_shifts": [], "ranking": [], "single_window": {}}
    for count, group in paired.groupby("anomaly_count"):
        row = {"cases": len(group), "generator_seeds": int(group.generator_seed.nunique())}
        for column in paired.columns:
            if column.endswith("_fixed") or column.endswith("_official"):
                row[column] = float(group[column].mean())
            if column.startswith("interaction_") or column.startswith("oracle_minus_"):
                values = group.groupby("generator_seed")[column].mean().to_numpy()
                row[f"{column}_ci"] = bootstrap_mean(values, seed, repetitions)
        stats["normalization"][str(count)] = row
    shifts = pd.read_csv(directory / "original_class_shifts.csv")
    for identity, group in shifts.groupby(["kind", "window", "score_kind", "class"]):
        row = dict(zip(["kind", "window", "score_kind", "class"], identity))
        row["window"] = int(row["window"])
        row.update(cases=len(group), series_mean_shift=float(group.mean_shift.mean()),
                   series_median_mean_shift=float(group.mean_shift.median()),
                   cases_mean_decreased=int((group.mean_shift < 0).sum()))
        stats["score_shifts"].append(row)
    ranking = pd.read_csv(directory / "original_ranking_changes.csv")
    for identity, group in ranking.groupby(["kind", "window", "score_kind"]):
        row = dict(zip(["kind", "window", "score_kind"], identity))
        row["window"] = int(row["window"])
        row.update(cases=len(group), mean_delta_auc=float(group.delta_auc.mean()),
                   mean_within_contribution=float(group.within_delta_contribution.mean()),
                   mean_across_contribution=float(group.across_delta_contribution.mean()))
        stats["ranking"].append(row)
    windows = pd.read_csv(parent / "windows.csv")
    one_window = windows.groupby("case").size().loc[lambda x: x == 1].index
    selected = previous[(previous.kind == "tsb") & (previous.window == 5000) &
                        (previous.scope == "full_fallback") & (previous.score_kind == "scores") &
                        previous.case.isin(one_window)]
    stats["single_window"] = {"cases": len(selected), "baseline_ap": float(selected.baseline_ap.mean()),
                              "oracle_ap": float(selected.oracle_ap.mean()),
                              "delta_ap": float(selected.oracle_minus_baseline_ap.mean()),
                              "worsened_gt_001": int((selected.oracle_minus_baseline_ap < -.01).sum())}
    return stats, paired


def main():
    parser = argparse.ArgumentParser(description="Summarize score-path and normalization follow-up")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    stats, paired = summarize(args.run_directory)
    output = args.run_directory / "analysis"
    output.mkdir(exist_ok=False)
    write_json(output / "analysis.json", stats)
    paired.to_csv(output / "normalization_pairs.csv", index=False)
    shutil.copyfile(__file__, output / "analyze.py")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
