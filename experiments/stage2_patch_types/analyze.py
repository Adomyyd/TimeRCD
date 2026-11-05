import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.stage1.analyze import bootstrap_mean


METADATA = ["case", "series", "kind", "source", "generator_seed", "placement",
            "anomaly_count", "length", "mixed_patches", "pure_patches"]
CONTRASTS = [f"{t}_{c}_minus_A" for t in "MPU" for c in "CD"] + [
    f"{t}_C_minus_D" for t in "MPU"] + [f"interaction_{c}" for c in "CD"]
ARMS = ["A"] + [f"{t}_{c}" for t in "MPU" for c in "CD"]


def contrasts(metrics, seeds):
    frame = metrics[metrics.score_kind == "scores"]
    if frame.duplicated(["case", "variant"]).any():
        raise ValueError("Duplicate probability metrics")
    selectors = ["label"] + [f"random_{s}" for s in seeds]
    expected = {"A"} | {f"{t}_{s}__{c}" for t in "MPU" for s in selectors for c in "CD"}
    rows = []
    for case, group in frame.groupby("case", sort=True):
        if set(group.variant) != expected or not group.metric_status.eq("ok").all():
            raise ValueError(f"Incomplete or undefined metrics: {case}")
        values = group.set_index("variant").average_precision
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite AP")
        base = group.iloc[0][METADATA].to_dict()
        for selector in selectors:
            row = {**base, "selector": selector, "A": float(values["A"])}
            for t in "MPU":
                for c in "CD":
                    row[f"{t}_{c}"] = float(values[f"{t}_{selector}__{c}"])
                    row[f"{t}_{c}_minus_A"] = row[f"{t}_{c}"] - row["A"]
                row[f"{t}_C_minus_D"] = row[f"{t}_C"] - row[f"{t}_D"]
            for c in "CD":
                row[f"interaction_{c}"] = row[f"U_{c}"] - row[f"M_{c}"] - row[f"P_{c}"] + row["A"]
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    paired = []
    for case, group in per_seed.groupby("case", sort=True):
        label = group[group.selector == "label"].iloc[0]
        random = group[group.selector != "label"]
        row = label[METADATA].to_dict()
        for key in ARMS + CONTRASTS:
            row[f"label_{key}"] = float(label[key])
            row[f"random_{key}"] = float(random[key].mean())
            row[f"label_minus_random_{key}"] = float(label[key] - random[key].mean())
        paired.append(row)
    return per_seed, pd.DataFrame(paired)


def aggregate(frame, unit, config):
    result = {"cases": len(frame), "units": int(frame[unit].nunique()), "bootstrap_unit": unit}
    for prefix in ("label", "random"):
        for arm in ARMS:
            result[f"{prefix}_{arm}"] = float(frame.groupby(unit)[f"{prefix}_{arm}"].mean().mean())
    for prefix in ("label", "random", "label_minus_random"):
        for contrast in CONTRASTS:
            key = f"{prefix}_{contrast}"
            result[key] = bootstrap_mean(frame.groupby(unit)[key].mean().to_numpy(),
                                         config["bootstrap_seed"], config["bootstrap_repetitions"])
    result["M_C_improves_gt_001"] = int((frame.label_M_C_minus_A > .01).sum())
    result["M_C_worsens_gt_001"] = int((frame.label_M_C_minus_A < -.01).sum())
    return result


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Execution must complete before analysis")
    config = manifest["config"]
    audit = {}
    for name in ("checks.csv", "gates/checks.csv"):
        checks = pd.read_csv(directory / name)
        if checks.empty or not checks.close.eq(True).all():
            raise ValueError(f"Failed checks: {name}")
        audit[name] = {"count": len(checks), "max_abs": float(checks.max_abs.max())}
    metrics = pd.read_csv(directory / "metrics.csv")
    per_seed, paired = contrasts(metrics, config["random_seeds"])
    if len(paired) != 51 or set(paired.case) != {f"s{i:04d}_w5000" for i in range(51)}:
        raise ValueError("Expected all 51 frozen cases")
    if len(metrics) != 51 * 37 * 2 or metrics.duplicated(["case", "variant", "score_kind"]).any():
        raise ValueError("Unexpected metrics coverage")
    real = paired[paired.kind == "tsb"]
    synthetic = paired[paired.kind == "controlled_synthetic"]
    if len(real) != 24 or len(synthetic) != 27 or synthetic.pure_patches.any():
        raise ValueError("Unexpected dataset or synthetic Patch coverage")
    # Empty P and M=U are regression identities, not a new synthetic treatment.
    for row in synthetic.itertuples():
        with np.load(directory / f"arrays/{row.case}.npz") as saved:
            for selector in ["label"] + [f"random_{s}" for s in config["random_seeds"]]:
                for c in "CD":
                    for kind in ("scores", "logits"):
                        if not np.array_equal(saved[f"P_{selector}__{c}_{kind}"], saved[f"A_{kind}"]):
                            raise ValueError("Synthetic empty P differs from A")
                        if not np.array_equal(saved[f"M_{selector}__{c}_{kind}"], saved[f"U_{selector}__{c}_{kind}"]):
                            raise ValueError("Synthetic M differs from U")
    groups = {"tsb/all": aggregate(real, "series", config)}
    for name, chosen in (("with_M", real.mixed_patches > 0), ("with_P", real.pure_patches > 0),
                         ("without_P", real.pure_patches == 0)):
        if chosen.any():
            groups[f"tsb/{name}"] = aggregate(real[chosen], "series", config)
    for count, group in synthetic.groupby("anomaly_count"):
        groups[f"synthetic/{int(count)}"] = aggregate(group, "generator_seed", config)
    keys = [f"{p}_{c}" for p in ("label", "label_minus_random") for c in CONTRASTS]
    sources, omitted = [], []
    for source, group in real.groupby("source"):
        sources.append({"source": source, "cases": len(group), **{k: float(group[k].mean()) for k in keys}})
        other = real[real.source != source]
        omitted.append({"omitted_source": source, "cases": len(other), **{k: float(other[k].mean()) for k in keys}})
    coverage = pd.read_csv(directory / "coverage.csv")
    coverage_summary = coverage.groupby(["kind", "selection"])[[
        "selected_patches", "normal_points_covered", "anomaly_points_covered", "partial_patches"]].sum().reset_index()
    stats = {"groups": groups, "sources": sources, "leave_one_source_out": omitted,
             "checks": audit, "synthetic_P_noop_M_equals_U": True,
             "metrics_sha256": sha256(directory / "metrics.csv"), "analysis_sha256": sha256(Path(__file__))}
    return stats, per_seed, paired, coverage_summary


def render(stats):
    lines = ["# Patch组成消融", "", "M：混合Patch；P：全异常有效点Patch；U：两者并集。全层C保留自身连接，D删除整列。", "",
             "| 数据 | A | M_C | M_D | P_C | P_D | U_C | U_D |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in stats["groups"].items():
        lines.append("| " + name + " | " + " | ".join(f"{group['label_' + arm]:.6f}" for arm in ARMS) + " |")
    lines += ["", "主比较：基准all的M_C−A及其减数量匹配随机均值。区间及辅助比较见analysis.json。",
              "合成P为空，M=U，仅作为无操作和父结果回归验证。开发数据复用，结果为探索性诊断。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    stats, seeds, paired, coverage = summarize(args.run_directory)
    output = args.run_directory / "analysis"
    output.mkdir(exist_ok=False)
    write_json(output / "analysis.json", stats)
    seeds.to_csv(output / "per_seed_contrasts.csv", index=False)
    paired.to_csv(output / "paired_contrasts.csv", index=False)
    coverage.to_csv(output / "coverage_summary.csv", index=False)
    shutil.copyfile(__file__, output / "analyze.py")
    (output / "report.md").write_text(render(stats))
    print(render(stats))


if __name__ == "__main__":
    main()
