import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from experiments.common.artifacts import create_run, git, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.common.metrics import detection_metrics
from experiments.stage2_followup.diagnostics import shift_statistics
from .data import Audit, load_case, load_parent
from .diagnostics import pair_decomposition
from .statistics import paired_statistics, source_sensitivity, summarize


def analyze_case(meta, labels, groups, membership, predictions, selectors, parent_metrics, audit):
    metrics, shifts, pairs = [], [], []
    for kind in ("scores", "logits"):
        measured, decomposed = {}, {}
        for variant, prediction in predictions.items():
            scores = prediction[kind]
            measured[variant] = detection_metrics(labels, scores)
            old = parent_metrics[(parent_metrics.case == meta["case"]) & (parent_metrics.variant == variant) &
                                 (parent_metrics.score_kind == kind)]
            if len(old) != 1 or old.iloc[0]["series"] != meta["series"] or old.iloc[0].metric_status != "ok":
                raise ValueError("Missing or misidentified parent metric")
            for metric in ("average_precision", "roc_auc"):
                audit.check(f"{meta['case']}/{variant}/{kind}/{metric}", measured[variant][metric], old.iloc[0][metric])
            decomposed[variant] = pair_decomposition(labels, scores, groups, membership)
            for scopes in (("all",), ("within", "across")):
                selected = [p for p in decomposed[variant] if p["scope"] in scopes]
                audit.check(f"{meta['case']}/{variant}/{kind}/{scopes}/auc_sum",
                            sum(p["contribution"] for p in selected), measured[variant]["roc_auc"])
                audit.check(f"{meta['case']}/{variant}/{kind}/{scopes}/weight_sum", sum(p["weight"] for p in selected), 1)
        for selector in selectors:
            variant = f"P_{selector}__C"
            base = {**meta, "selector": selector, "score_kind": kind}
            row = dict(base)
            for metric in ("average_precision", "roc_auc"):
                row[f"baseline_{metric}"] = measured["A"][metric]
                row[f"candidate_{metric}"] = measured[variant][metric]
                row[f"delta_{metric}"] = measured[variant][metric] - measured["A"][metric]
            metrics.append(row)
            for group, mask in groups.items():
                shifts.append({**base, "group": group, **shift_statistics(predictions["A"][kind], predictions[variant][kind], mask)})
            for original, changed in zip(decomposed["A"], decomposed[variant]):
                if (original["pair"], original["scope"], original["pairs"]) != (changed["pair"], changed["scope"], changed["pairs"]):
                    raise ValueError("Pair identity mismatch")
                pairs.append({**base, **{k: original[k] for k in ("pair", "scope", "pairs", "weight")},
                              "baseline_auc": original["auc"], "candidate_auc": changed["auc"],
                              "delta_auc": changed["auc"] - original["auc"] if original["auc"] is not None else None,
                              "baseline_contribution": original["contribution"], "candidate_contribution": changed["contribution"],
                              "delta_contribution": changed["contribution"] - original["contribution"]})
            for scopes in (("all",), ("within", "across")):
                contribution = sum(p["delta_contribution"] for p in pairs if p["selector"] == selector and
                                   p["score_kind"] == kind and p["scope"] in scopes)
                audit.check(f"{meta['case']}/{selector}/{kind}/{scopes}/delta_sum", contribution, row["delta_roc_auc"])
    return metrics, shifts, pairs


def save_statistics(directory, tables, selectors, config, audit):
    definitions = {
        "metrics": ([], [f"{p}_{m}" for m in ("average_precision", "roc_auc") for p in ("baseline", "candidate", "delta")]),
        "shifts": (["group"], ["points", "baseline_mean", "candidate_mean", "mean_shift", "median_shift", "fraction_decreased"]),
        "pairs": (["pair", "scope"], ["pairs", "weight", "baseline_auc", "candidate_auc", "delta_auc",
                                       "baseline_contribution", "candidate_contribution", "delta_contribution"]),
    }
    paired_tables, summaries = {}, {}
    for name, rows in tables.items():
        dimensions, values = definitions[name]
        frame = pd.DataFrame(rows)
        frame.to_csv(directory / f"{name}.csv", index=False)
        paired = paired_statistics(frame, dimensions, values, selectors)
        paired.to_csv(directory / f"paired_{name}.csv", index=False)
        columns = [f"{prefix}_{v}" for prefix in ("label", "random", "label_minus_random") for v in values]
        summary = summarize(paired, dimensions, columns, config)
        summary.to_csv(directory / f"summary_{name}.csv", index=False)
        source_sensitivity(paired, dimensions, columns).to_csv(directory / f"sources_{name}.csv", index=False)
        paired_tables[name], summaries[name] = paired, summary
    for prefix in ("label", "random", "label_minus_random"):
        for scopes in (("all",), ("within", "across")):
            pair = paired_tables["pairs"]
            total = pair[pair.scope.isin(scopes)].groupby(["case", "score_kind"])[f"{prefix}_delta_contribution"].sum()
            expected = paired_tables["metrics"].set_index(["case", "score_kind"])[f"{prefix}_delta_roc_auc"]
            if set(total.index) != set(expected.index):
                raise ValueError("Paired identities differ")
            audit.check(f"paired/{prefix}/{scopes}/sum", total.loc[expected.index].to_numpy(), expected.to_numpy())
    return summaries


def render(summaries):
    lines = ["# P_C离线损失定位", "", "概率分数；序列等权、配对bootstrap 10000次。完整输出见CSV。", "",
             "| 集合 | 指标/贡献 | 平均差 | 95%区间 | 有数据序列 |", "|---|---|---:|---|---:|"]
    for name in ("metrics", "pairs", "shifts"):
        frame = summaries[name]
        chosen = frame[(frame.score_kind == "scores") & frame.subset.isin(["all", "with_P"])]
        if name == "metrics":
            chosen = chosen[chosen.statistic.isin(["label_delta_average_precision", "label_delta_roc_auc", "label_minus_random_delta_roc_auc"])]
        elif name == "pairs":
            chosen = chosen[(chosen.scope == "all") & (chosen.statistic == "label_delta_contribution")]
        else:
            chosen = chosen[chosen.statistic == "label_mean_shift"]
        for row in chosen.to_dict("records"):
            label = row.get("pair", row.get("group", row["statistic"]))
            lines.append(f"| {row['subset']} | {label} | {row['mean']:+.6f} | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['present_cases']} |")
    lines += ["", "贡献是ROC-AUC样本对记账，不是AP分解或因果归因。空组条件指标缺失、加权贡献为0；开发数据反复复用。", ""]
    return "\n".join(lines)


def provenance(directory, manifest, config_path):
    shutil.copyfile(config_path, directory / "config.input.json")
    shutil.copyfile(manifest["config"]["protocol_path"], directory / "protocol.md")
    manifest["protocol_sha256"] = sha256(directory / "protocol.md")
    manifest["git"] = {"commit": git("rev-parse", "HEAD").strip(), "status": git("status", "--short")}
    manifest["environment"] = {"python": sys.version, "executable": sys.executable,
                               "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")}}
    files = list((ROOT / "experiments/stage2_pc_localization").glob("*.py"))
    files += [ROOT / f for f in ("experiments/common/artifacts.py", "experiments/common/config.py",
                               "experiments/common/metrics.py", "experiments/stage1/analyze.py",
                               "experiments/stage2_patch_types/selections.py", "experiments/stage2_followup/diagnostics.py",
                               "experiments/tests/test_stage2_pc_localization.py", "EXPERIMENT_LOG.md")]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(target)
    write_json(directory / "source.sha256.json", hashes)


def main():
    parser = argparse.ArgumentParser(description="Offline localization of P_C ranking losses")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "protocol_path", "output_root"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        provenance(directory, manifest, args.config)
        parent = Path(config["parent_run"])
        audit = Audit(directory, config["tolerance"])
        previous, records = load_parent(parent, config, audit)
        manifest["checkpoint"] = previous["checkpoint"]
        manifest["inference_calls"] = 0
        filenames = ["manifest.json", "data_manifest.json", "summary.json", "checks.csv", "gates/checks.csv", "metrics.csv", "coverage.csv"]
        for record in records:
            filenames.extend([record["input_snapshot"], record["selection_snapshot"], f"arrays/s{record['index']:04d}_w5000.npz"])
        fingerprints = {name: sha256(parent / name) for name in filenames}
        write_json(directory / "parent_artifacts.sha256.json", fingerprints)
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        seeds = previous["config"]["random_seeds"]
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError("Expected five unique frozen random selectors")
        selectors = ["label"] + [f"random_{s}" for s in seeds]
        coverage, metrics = pd.read_csv(parent / "coverage.csv"), pd.read_csv(parent / "metrics.csv")
        tables = {"metrics": [], "shifts": [], "pairs": []}
        for record in records:
            case = load_case(parent, record, config, seeds, coverage, audit)
            rows = analyze_case(*case, selectors, metrics, audit)
            for name, block in zip(tables, rows):
                tables[name].extend(block)
        summaries = save_statistics(directory, tables, selectors, config, audit)
        report = render(summaries)
        (directory / "report.md").write_text(report)
        for name, digest in fingerprints.items():
            audit.check(f"parent_unchanged/{name}", sha256(parent / name) == digest, True, exact=True)
        write_json(directory / "summary.json", {"cases": len(records), "with_P": sum(r["pure_patches"] > 0 for r in records),
                   "checks_passed": audit.count, "rows": {k: len(v) for k, v in tables.items()}, "inference_calls": 0})
        manifest["status"] = "completed"
        print(report)
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
