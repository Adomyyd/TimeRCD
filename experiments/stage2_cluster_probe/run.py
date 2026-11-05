import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common.artifacts import append_row, create_run, sha256, write_json
from experiments.common.config import ROOT, project_path
from experiments.stage2_patch_types.selections import patch_types
from .analysis import patch_correlation, rank_correlation


def analyze(directory, config):
    parent = Path(config["parent_run"])
    hashes = {}
    checks = 0

    def protect(path, expected=None):
        path = Path(path)
        digest = sha256(path)
        hashes[str(path)] = digest
        if expected is not None:
            check("sha256", digest == expected, str(path))

    def check(name, passed, case="parent"):
        nonlocal checks
        append_row(directory / "checks.csv", {"case": case, "test": name, "passed": bool(passed)})
        checks += 1
        if not passed:
            raise ValueError(f"Failed check {case}: {name}")

    for name in ("manifest.json", "data_manifest.json", "metrics.csv"):
        protect(parent / name)
    check("parent_completed", json.loads((parent / "manifest.json").read_text())["status"] == "completed")
    records = json.loads((parent / "data_manifest.json").read_text())
    records = [r for r in records if r["source"]["kind"] == "tsb" and r["pure_patches"] > 0]
    check("13_prespecified_series", len(records) == 13)
    metrics = pd.read_csv(parent / "metrics.csv")
    rows = []
    for record in records:
        case = f"s{record['index']:04d}_w5000"
        raw_path = Path(record["source"]["path"])
        protect(raw_path, record["source"]["sha256"])
        protect(parent / record["input_snapshot"], record["input_sha256"])
        protect(parent / record["selection_snapshot"], record["selection_sha256"])
        protect(parent / f"arrays/{case}.npz")
        raw = pd.read_csv(raw_path).iloc[record["source"]["offset"]:]
        values, labels = raw.iloc[:, :1].to_numpy(), raw.Label.to_numpy()
        with np.load(parent / record["input_snapshot"]) as saved:
            check("raw_values_match_snapshot", np.array_equal(values, saved["values"]), case)
            check("raw_labels_match_snapshot", np.array_equal(labels, saved["labels"]), case)
        std = np.where(values.std(axis=0) == 0, 1e-8, values.std(axis=0))
        normalized = (values - values.mean(axis=0)) / std
        segments, locations, incomplete = [], [], 0
        window = min(len(labels), config["window"])
        with np.load(parent / record["selection_snapshot"]) as saved:
            for number, start in enumerate(range(0, len(values), window)):
                size = min(window, len(values) - start)
                local_labels = np.pad(labels[start:start + size], (0, window - size))
                valid = np.arange(window) < size
                types = patch_types(local_labels, valid, config["patch_size"])
                check(f"P_mapping_w{number}", np.array_equal(types["P"], saved[f"w{number}_selection_P_label"]), case)
                for index in np.flatnonzero(types["P"]):
                    if types["valid_counts"][index] != 16:
                        incomplete += 1
                        continue
                    absolute = start + 16 * index
                    segments.append(normalized[absolute:absolute + 16, 0])
                    locations.append([number, int(index), int(absolute)])
        check("P_count", len(segments) + incomplete == record["pure_patches"], case)
        segments = np.asarray(segments).reshape(-1, 16)
        correlation, status = patch_correlation(segments, config["minimum_patches"])
        np.savez_compressed(directory / f"arrays/{case}_p_segments.npz", segments=segments,
                            locations=np.asarray(locations, dtype=int).reshape(-1, 3), labels=labels)
        aps = {}
        with np.load(parent / f"arrays/{case}.npz") as saved:
            for variant in ("A", "P_label__C"):
                scores = saved[f"{variant}_scores"]
                check(f"{variant}_finite_aligned", scores.shape == labels.shape and np.isfinite(scores).all(), case)
                ap = float(average_precision_score(labels, scores))
                original = metrics[(metrics.case == case) & (metrics.variant == variant) & (metrics.score_kind == "scores")]
                check(f"{variant}_unique_metric", len(original) == 1, case)
                check(f"{variant}_AP_recomputed", abs(ap - original.average_precision.iloc[0]) <= 1e-12, case)
                aps[variant] = ap
        rows.append({"case": case, "series": record["name"], "n_P": record["pure_patches"],
                     "n_complete_P": len(segments), "incomplete_P_excluded": incomplete,
                     "rho_bar": correlation, "status": status, "ap_A": aps["A"],
                     "ap_P_C": aps["P_label__C"], "delta_ap": aps["P_label__C"] - aps["A"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "per_series.csv", index=False)
    eligible = frame[frame.status == "ok"]
    rho = rank_correlation(eligible.rho_bar, eligible.delta_ap)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(eligible.rho_bar, eligible.delta_ap)
    for row in eligible.itertuples():
        ax.annotate(row.case.split("_")[0], (row.rho_bar, row.delta_ap), xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="0.6", linewidth=.8)
    ax.set(xlabel="Mean pairwise Pearson correlation of P segments", ylabel="AP(P_C) - AP(A)",
           title=f"Real development sequences: Spearman rho={rho:.3f}, n={len(eligible)}")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"similarity_vs_benefit.{suffix}", dpi=160)
    plt.close(fig)
    for path, digest in list(hashes.items()):
        check("source_unchanged", sha256(path) == digest, path)
    write_json(directory / "parent_artifacts.sha256.json", hashes)
    summary = {"rho_spearman": rho, "n": len(eligible), "total_P_series": len(rows),
               "missing": frame[frame.status != "ok"].where(pd.notna(frame), None).to_dict("records"),
               "checks_passed": checks, "model_loads": 0, "model_forwards": 0,
               "torch_imported": "torch" in sys.modules, "parent_artifacts_unchanged": True}
    # Pandas float columns keep NaN after where(None); JSON must contain real missing values.
    summary["missing"] = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                          for r in summary["missing"]]
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("parent_run", "output_root", "protocol_path"):
        config[key] = str(project_path(config[key]))
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        shutil.copyfile(args.config, directory / "config.input.json")
        shutil.copyfile(config["protocol_path"], directory / "protocol.md")
        sources = list(Path(__file__).parent.rglob("*.py")) + [ROOT / "experiments/common/artifacts.py",
                   ROOT / "experiments/stage2_patch_types/selections.py"]
        fingerprints = {}
        for path in sources:
            relative = path.relative_to(ROOT)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            fingerprints[str(relative)] = sha256(target)
        write_json(directory / "source.sha256.json", fingerprints)
        write_json(directory / "environment.json", {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__})
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        print(json.dumps(analyze(directory, config), ensure_ascii=False))
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
