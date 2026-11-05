import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.common.artifacts import append_row, create_run, sha256, write_json
from experiments.common.config import ROOT


class Validation:
    def __init__(self, directory):
        self.directory, self.count = directory, 0

    def check(self, name, passed):
        self.count += 1
        append_row(self.directory / "checks.csv", {"test": name, "passed": bool(passed)})
        if not passed:
            raise RuntimeError(f"Output validation failed: {name}")

    def equal(self, name, a, b, tolerance=1e-12):
        self.check(name, np.asarray(a).shape == np.asarray(b).shape and np.allclose(a, b, atol=tolerance, rtol=0))

    def provenance(self, path):
        self.check(f"{path.name}_completed", json.loads((path / "manifest.json").read_text())["status"] == "completed")
        for check_path in [path / "checks.csv", path / "gates/checks.csv"]:
            if check_path.exists():
                checks = pd.read_csv(check_path)
                self.check(f"{check_path}_all_pass", len(checks) > 0 and checks.passed.eq(True).all())
        for relative, digest in json.loads((path / "source.sha256.json").read_text()).items():
            self.check(f"{path.name}_source_{relative}", sha256(path / "source" / relative) == digest)
        protected = path / "parent_artifacts.sha256.json"
        if protected.exists():
            for relative, digest in json.loads(protected.read_text()).items():
                target = Path(relative) if Path(relative).is_absolute() else ROOT / relative
                self.check(f"{path.name}_protected_{relative}", sha256(target) == digest)


def validate_r(v, path):
    frame = pd.read_csv(path / "per_series.csv")
    for row in frame.itertuples():
        with np.load(path / f"arrays/{row.case}_p_segments.npz") as saved:
            segments = saved["segments"]
        if len(segments) < 3:
            v.check(f"R_{row.case}_missing", pd.isna(row.rho_bar) and row.status == "fewer_than_3_complete_P")
        else:
            pairs = [np.corrcoef(segments[i], segments[j])[0, 1]
                     for i in range(len(segments)) for j in range(i + 1, len(segments))]
            v.equal(f"R_{row.case}_rho", row.rho_bar, np.mean(pairs))
        v.equal(f"R_{row.case}_delta", row.delta_ap, row.ap_P_C - row.ap_A)
    eligible = frame.dropna(subset=["rho_bar"])
    rho = np.corrcoef(rankdata(eligible.rho_bar), rankdata(eligible.delta_ap))[0, 1]
    v.equal("R_rank_correlation", rho, json.loads((path / "summary.json").read_text())["rho_spearman"])


def validate_s(v, path):
    frame = pd.read_csv(path / "metrics.csv")
    records = json.loads((path / "data_manifest.json").read_text())
    for record in records:
        case = record["case"]
        with np.load(path / record["input_snapshot"]) as data, np.load(path / f"arrays/{case}.npz") as scores:
            labels = data["labels"]
            for row in frame[frame.case == case].itertuples():
                x = scores[f"{row.variant}_{row.score_kind}"]
                v.check(f"S_{case}_{row.variant}_{row.score_kind}_finite", x.shape == labels.shape and np.isfinite(x).all())
                actual = [x[labels == 1].mean(), roc_auc_score(labels, x), average_precision_score(labels, x),
                          ((rankdata(x) - 1) / (len(x) - 1))[labels == 1].max(),
                          x[(record["position"] - 2) * 16:record["position"] * 16].mean(),
                          x[(record["position"] + 1) * 16:(record["position"] + 3) * 16].mean()]
                expected = [row.anomaly_mean, row.roc_auc, row.average_precision, row.max_anomaly_percentile,
                            row.left_normal_mean, row.right_normal_mean]
                v.equal(f"S_{case}_{row.variant}_{row.score_kind}_readouts", actual, expected)
    paired = pd.read_csv(path / "paired.csv")
    for row in paired.itertuples():
        part = frame[(frame.case == row.case) & (frame.score_kind == row.score_kind)].set_index("variant")
        base = part.loc["A", row.measure]
        c = part.loc["C_anom", row.measure] - base
        d = part.loc["D_anom", row.measure] - base
        random = np.array([part.loc[f"D_rand_{k}", row.measure] - base for k in range(1, 6)])
        expected = {"C_minus_A": c, "D_minus_A": d, "random_mean_minus_A": random.mean(),
                    "C_minus_random": c - random.mean(), "C_abs_minus_random_abs": abs(c) - np.abs(random).mean()}
        v.equal(f"S_pair_{row.Index}", row.value, expected[row.contrast])
    seeds = pd.read_csv(path / "seed_effects.csv")
    for row in seeds.itertuples():
        values = paired[(paired.seed == row.seed) & (paired.score_kind == row.score_kind) &
                        (paired.measure == row.measure) & (paired.contrast == row.contrast)].value
        v.check(f"S_seed_count_{row.Index}", len(values) == 5)
        v.equal(f"S_seed_mean_{row.Index}", row.value, values.mean())
    for row in pd.read_csv(path / "bootstrap.csv").itertuples():
        x = seeds[(seeds.score_kind == row.score_kind) & (seeds.measure == row.measure) &
                  (seeds.contrast == row.contrast)].value.to_numpy()
        draws = np.random.default_rng(2024).integers(0, 8, (10000, 8))
        low, high = np.quantile(x[draws].mean(axis=1), [.025, .975])
        v.equal(f"S_bootstrap_{row.Index}", [row.mean, row.low, row.high], [x.mean(), low, high])


def validate_dose(v, path, s_path):
    rows = []
    for record in json.loads((s_path / "data_manifest.json").read_text()):
        case = record["case"]
        with np.load(s_path / record["input_snapshot"]) as data, np.load(path / f"arrays/{case}.npz") as scores, \
                np.load(s_path / f"arrays/{case}.npz") as previous:
            labels = data["labels"]
            base = average_precision_score(labels, scores["A_scores"])
            v.equal(f"dose_{case}_A", scores["A_scores"], previous["A_scores"], 0)
            for count in (1, 16, 64):
                for repetition in range(1, 6):
                    prefix = f"q{count}_k{repetition}"
                    selected = scores[f"{prefix}_selected"]
                    v.check(f"dose_{case}_{prefix}_selection", selected.sum() == count and
                            not (selected & (~data["type_N"] | data["type_partial"])).any())
                    x = scores[f"{prefix}_scores"]
                    v.check(f"dose_{case}_{prefix}_finite", x.shape == labels.shape and np.isfinite(x).all())
                    if count == 1:
                        for kind in ("scores", "logits"):
                            v.equal(f"dose_{case}_{prefix}_{kind}_replay", scores[f"{prefix}_{kind}"], previous[f"D_rand_{repetition}_{kind}"], 0)
                    rows.append({"case": case, "count": count, "seed": record["seed"],
                                 "ap_loss": base - average_precision_score(labels, x),
                                 "anomaly_shift": float(x[labels == 1].mean()) - float(scores["A_scores"][labels == 1].mean())})
    frame = pd.DataFrame(rows)
    case_means = frame.groupby(["case", "seed", "count"])[["ap_loss", "anomaly_shift"]].mean().reset_index()
    seeds = case_means.groupby(["seed", "count"])[["ap_loss", "anomaly_shift"]].mean().reset_index()
    for row in pd.read_csv(path / "bootstrap.csv").itertuples():
        x = seeds[seeds["count"] == row.count][row.measure].to_numpy()
        draws = np.random.default_rng(2024).integers(0, 8, (10000, 8))
        low, high = np.quantile(x[draws].mean(axis=1), [.025, .975])
        v.equal(f"dose_bootstrap_{row.count}_{row.measure}", [row.mean, row.low, row.high], [x.mean(), low, high])


def validate_tiny(v, micro, boundary):
    with np.load(micro / "arrays/micro.npz") as saved:
        for condition in "ACD":
            for layer in range(2):
                v.equal(f"micro_{condition}_{layer}", saved[f"{condition}_layer{layer}_actual"], saved[f"{condition}_layer{layer}_manual"])
    for name, expected in (("point_16", [0]), ("point_17", [1]), ("point_16_17", [0, 1])):
        with np.load(boundary / f"arrays/{name}.npz") as saved:
            v.check(f"boundary_{name}_selection", np.flatnonzero(saved["selected"]).tolist() == expected)
            v.equal(f"boundary_{name}_patches", saved["patches"], np.pad(saved["values"][:, 0], (0, 8)).reshape(1, 313, 16), 0)


def main():
    parser = argparse.ArgumentParser()
    for key in ("r", "s", "dose", "micro", "boundary"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    config = {"name": "stage2_RS_output_validation_v1", "output_root": str(ROOT / "experiments/runs"),
              "parents": {key: str(path.resolve()) for key, path in vars(args).items()}, "model_forwards": 0}
    directory, manifest = create_run(config, __file__)
    print(f"Validation directory: {directory}", flush=True)
    try:
        shutil.copyfile(__file__, directory / "validator.py")
        shutil.copyfile(ROOT / "experiments/stage2_isolated/protocols/output_validation_v1.md", directory / "protocol.md")
        v = Validation(directory)
        for path in vars(args).values():
            v.provenance(path)
        validate_r(v, args.r)
        validate_s(v, args.s)
        validate_dose(v, args.dose, args.s)
        validate_tiny(v, args.micro, args.boundary)
        summary = {"checks_passed": v.count, "model_loads": 0, "model_forwards": 0,
                   "torch_imported": "torch" in sys.modules, "validator_sha256": sha256(directory / "validator.py")}
        write_json(directory / "summary.json", summary)
        manifest["status"] = "completed"
        print(json.dumps(summary))
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
