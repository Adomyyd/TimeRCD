import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.common.artifacts import write_json


def readout(labels, scores, position):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.shape != scores.shape or not np.isfinite(scores).all() or labels.sum() != 16:
        raise ValueError("Invalid isolated readout")
    result = {"anomaly_mean": float(scores[labels == 1].mean()),
              "roc_auc": float(roc_auc_score(labels, scores)),
              "average_precision": float(average_precision_score(labels, scores)),
              "max_anomaly_percentile": float(((rankdata(scores) - 1) / (len(scores) - 1))[labels == 1].max())}
    for direction, start in (("left", 16 * (position - 2)), ("right", 16 * (position + 1))):
        if labels[start:start + 32].any() or len(scores[start:start + 32]) != 32:
            raise ValueError("Propagation readout overlaps event or boundary")
        result[f"{direction}_normal_mean"] = float(scores[start:start + 32].mean())
    return result


def interval(seed_values, seed=2024, repetitions=10000):
    seed_values = np.asarray(seed_values, dtype=float)
    if seed_values.ndim != 1 or len(seed_values) < 2 or not np.isfinite(seed_values).all():
        raise ValueError("Expected finite seed-level values")
    rng = np.random.default_rng(seed)
    boot = seed_values[rng.integers(len(seed_values), size=(repetitions, len(seed_values)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(seed_values.mean()), "low": float(low), "high": float(high)}


def decision(main, extra, magnitude):
    if main["high"] < 0 and extra["high"] < 0 and magnitude["low"] > 0:
        return "pause_H_isolated_specific_harm"
    if extra["low"] <= 0 <= extra["high"] and magnitude["low"] <= 0 <= magnitude["high"]:
        return "no_extra_harm_detected_not_equivalence"
    return "inconclusive_H_remains_locked"


def summarize(directory, config):
    frame = pd.read_csv(directory / "metrics.csv")
    measures = ["anomaly_mean", "roc_auc", "average_precision", "max_anomaly_percentile",
                "left_normal_mean", "right_normal_mean"]
    paired = []
    for (case, kind), part in frame.groupby(["case", "score_kind"], sort=False):
        if len(part) != 8 or part.variant.nunique() != 8:
            raise ValueError("Missing isolated conditions")
        part = part.set_index("variant")
        for measure in measures:
            a = float(part.loc["A", measure])
            c, d = float(part.loc["C_anom", measure]) - a, float(part.loc["D_anom", measure]) - a
            random = np.array([float(part.loc[f"D_rand_{k}", measure]) - a for k in range(1, 6)])
            effects = {"C_minus_A": c, "D_minus_A": d, "random_mean_minus_A": float(random.mean()),
                       "C_minus_random": float(c - random.mean()),
                       "C_abs_minus_random_abs": float(abs(c) - np.abs(random).mean())}
            for contrast, value in effects.items():
                paired.append({"case": case, "seed": int(part.seed.iloc[0]), "position": int(part.position.iloc[0]),
                               "score_kind": kind, "measure": measure, "contrast": contrast, "value": value})
    paired = pd.DataFrame(paired)
    paired.to_csv(directory / "paired.csv", index=False)
    seed_frame = paired.groupby(["seed", "score_kind", "measure", "contrast"], as_index=False).value.mean()
    seed_frame.to_csv(directory / "seed_effects.csv", index=False)
    summaries = []
    for (kind, measure, contrast), part in seed_frame.groupby(["score_kind", "measure", "contrast"], sort=False):
        if sorted(part.seed) != config["seeds"]:
            raise ValueError("Seed coverage mismatch")
        summaries.append({"score_kind": kind, "measure": measure, "contrast": contrast, "n_seeds": len(part),
                          **interval(part.value, config["bootstrap_seed"], config["bootstrap_repetitions"])})
    summary = pd.DataFrame(summaries)
    summary.to_csv(directory / "bootstrap.csv", index=False)
    primary = {r["contrast"]: {k: r[k] for k in ("mean", "low", "high")}
               for r in summaries if r["score_kind"] == "scores" and r["measure"] == "anomaly_mean"}
    result = {"primary": primary, "decision": decision(primary["C_minus_A"], primary["C_minus_random"],
                                                       primary["C_abs_minus_random_abs"])}
    write_json(directory / "decision.json", result)
    return result
