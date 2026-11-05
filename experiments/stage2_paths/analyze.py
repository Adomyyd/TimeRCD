import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256, write_json
from experiments.stage1.analyze import bootstrap_mean


CONTRASTS = ("C_minus_D", "C_minus_A", "B_minus_A", "D_minus_B", "interaction")


def contrasts(frame, seeds):
    frame = frame[(frame.scope == "full") & (frame.score_kind == "scores")]
    if frame.duplicated(["case", "variant"]).any():
        raise ValueError("Duplicate full-series probability metrics")
    selectors = ["label"] + [f"random_{seed}" for seed in seeds]
    expected = {"A"} | {f"{s}__{l}__{c}" for s in selectors for l in ("last", "all") for c in "BCD"}
    rows = []
    metadata = ["case", "series", "kind", "source", "generator_seed", "placement", "anomaly_count", "length"]
    for case, group in frame.groupby("case", sort=True):
        if set(group.variant) != expected or not group.metric_status.eq("ok").all():
            raise ValueError(f"Incomplete conditions or undefined metrics: {case}")
        values = group.set_index("variant").average_precision
        base = group.iloc[0][metadata].to_dict()
        for scope in ("last", "all"):
            for selector in selectors:
                a = float(values["A"])
                b, c, d = (float(values[f"{selector}__{scope}__{arm}"]) for arm in "BCD")
                rows.append({**base, "layer_scope": scope, "selector": selector,
                             "A": a, "B": b, "C": c, "D": d,
                             "C_minus_D": c - d, "C_minus_A": c - a,
                             "B_minus_A": b - a, "D_minus_B": d - b,
                             "interaction": d - c - b + a})
    all_rows = pd.DataFrame(rows)
    paired = []
    for (case, scope), group in all_rows.groupby(["case", "layer_scope"], sort=True):
        label = group[group.selector == "label"].iloc[0]
        random = group[group.selector != "label"]
        if len(random) != len(seeds):
            raise ValueError("Incomplete random seeds")
        row = {**label[metadata].to_dict(), "layer_scope": scope}
        for key in ("A", "B", "C", "D", *CONTRASTS):
            row[f"label_{key}"] = float(label[key])
            row[f"random_{key}"] = float(random[key].mean())
            row[f"label_minus_random_{key}"] = float(label[key] - random[key].mean())
        paired.append(row)
    return all_rows, pd.DataFrame(paired)


def aggregate(frame, unit, config):
    result = {"cases": len(frame), "units": int(frame[unit].nunique()), "bootstrap_unit": unit}
    for name in ("A", "B", "C", "D"):
        for selector in ("label", "random"):
            result[f"{selector}_{name}"] = float(frame[f"{selector}_{name}"].mean())
    for selector in ("label", "random", "label_minus_random"):
        for name in CONTRASTS:
            key = f"{selector}_{name}"
            values = frame.groupby(unit)[key].mean().to_numpy()
            result[key] = bootstrap_mean(values, config["bootstrap_seed"], config["bootstrap_repetitions"])
    result["C_beats_A_gt_001"] = int((frame.label_C_minus_A > .01).sum())
    result["C_worse_A_gt_001"] = int((frame.label_C_minus_A < -.01).sum())
    result["C_beats_D_gt_001"] = int((frame.label_C_minus_D > .01).sum())
    result["C_worse_D_gt_001"] = int((frame.label_C_minus_D < -.01).sum())
    return result


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Mechanism analysis requires all execution checks to pass")
    config = manifest["config"]
    for name in ("checks.csv", "gates/checks.csv"):
        checks = pd.read_csv(directory / name)
        if not checks.close.all():
            raise ValueError(f"Failed checks in {name}")
    metrics = pd.read_csv(directory / "metrics.csv")
    per_seed, paired = contrasts(metrics, config["random_seeds"])
    if paired.case.nunique() != 51 or len(paired) != 102:
        raise ValueError("Expected all 51 frozen inputs and both scopes")
    stats = {"status": manifest["status"], "metrics_sha256": sha256(directory / "metrics.csv"),
             "analysis_sha256": sha256(Path(__file__)), "groups": {}, "sources": [], "leave_one_source_out": []}
    for scope, group in paired.groupby("layer_scope"):
        real = group[group.kind == "tsb"]
        stats["groups"][f"tsb/{scope}"] = aggregate(real, "series", config)
        for count, synthetic in group[group.kind == "controlled_synthetic"].groupby("anomaly_count"):
            stats["groups"][f"synthetic/{int(count)}/{scope}"] = aggregate(synthetic, "generator_seed", config)
        for source, source_rows in real.groupby("source"):
            stats["sources"].append({"scope": scope, "source": source, "cases": len(source_rows),
                **{key: float(source_rows[key].mean()) for key in
                   ("label_C_minus_D", "label_C_minus_A", "label_B_minus_A", "label_minus_random_C_minus_D")}})
            retained = real[real.source != source]
            stats["leave_one_source_out"].append({"scope": scope, "omitted_source": source, "cases": len(retained),
                **{key: float(retained[key].mean()) for key in
                   ("label_C_minus_D", "label_C_minus_A", "label_minus_random_C_minus_D")}})
    return stats, per_seed, paired


def render(stats):
    lines = ["# 四条件注意力连接消融", "",
             "A：原模型；B：只删除选中Patch自身连接；C：保留自身、删除被其他位置读取的连接；D：删除整列。",
             "所有值为序列宏平均概率AP；合成数据按异常点数分组。", "",
             "| 数据/层范围 | A | B | C | D | C−D | C−A | 标签−随机的(C−D) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in stats["groups"].items():
        lines.append(f"| {name} | {row['label_A']:.6f} | {row['label_B']:.6f} | {row['label_C']:.6f} | {row['label_D']:.6f} | {row['label_C_minus_D']['mean']:+.6f} | {row['label_C_minus_A']['mean']:+.6f} | {row['label_minus_random_C_minus_D']['mean']:+.6f} |")
    lines += ["", "主比较为基准all条件的C−D及其标签−随机差；必须同时看C−A。区间见analysis.json。",
              "随机五次指标在序列内部平均；合成先在同一生成种子内平均布局。开发数据已多次使用，结果为后续诊断，不是独立确认性结论。",
              "恢复自身连接同时改变注意力分母及可用参照数量，差值不能唯一归因于自身语义。标签选择也未控制参照内容、位置与原始注意力质量。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pre-specified attention-path contrasts")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    stats, per_seed, paired = summarize(args.run_directory)
    output = args.run_directory / "analysis"
    output.mkdir(exist_ok=False)
    write_json(output / "analysis.json", stats)
    per_seed.to_csv(output / "per_seed_contrasts.csv", index=False)
    paired.to_csv(output / "paired_contrasts.csv", index=False)
    shutil.copyfile(__file__, output / "analyze.py")
    (output / "report.md").write_text(render(stats))
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
