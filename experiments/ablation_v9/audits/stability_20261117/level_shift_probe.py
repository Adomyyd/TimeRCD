"""Compare stable detectors around isolated, persistent synthetic level shifts."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE), str(ROOT)]
from src.benchmark_eval import detection_model, load_protocol, score_series
from src.data import load_data, load_indices
from src.models import build_model

MODELS = ("p0", "p1")
BANDS = ("pre_far", "pre_near", "event", "post_near", "post_mid", "post_far")
OUTPUT = Path(__file__).with_suffix(".json")


def candidates(sample):
    y = np.asarray(sample["labels"]).reshape(-1)
    d = np.asarray(sample["time_series"]).reshape(-1) - np.asarray(sample["normal_time_series"]).reshape(-1)
    for name, (start, end) in sample["attribute"]["anomalies"].items():
        kind = name.split("_", 1)[-1]
        if kind not in ("sudden increase", "sudden decrease"):
            continue
        a, b = int(start), int(end)
        if a < 128 or b + 128 > len(y) or a >= b:
            continue
        if np.any(y[a-128:a]) or np.any(y[b:b+128]) or not np.all(y[a:b] > 0):
            continue
        before, after = d[a-128:a], d[b:b+128]
        change = float(np.median(after) - np.median(before))
        if change * (1 if kind == "sudden increase" else -1) <= 0:
            continue
        if max(np.ptp(before), np.ptp(after)) > max(1e-7, abs(change)*0.01):
            continue
        yield kind, a, b


def summary(rows):
    if not rows:
        return {"n": 0}
    rng = np.random.default_rng(72)
    idx = rng.integers(len(rows), size=(2000, len(rows)))
    def paired(d):
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        return [float(d.mean()), float(lo), float(hi)]
    result = {"n": len(rows)}
    for metric in ("prob", "over_half", "official_alarm"):
        bands = {}
        delta = {}
        for band in BANDS:
            v = np.array([[r[m][band][metric] for m in MODELS] for r in rows])
            d = v[:, 1]-v[:, 0]
            delta[band] = d
            bands[band] = {"p0": float(v[:,0].mean()), "p1": float(v[:,1].mean()),
                           "difference_ci": paired(d)}
        bands["post_far_minus_pre_far_difference_ci"] = paired(delta["post_far"]-delta["pre_far"])
        bands["post_near_minus_pre_near_difference_ci"] = paired(delta["post_near"]-delta["pre_near"])
        result[metric] = bands
    return result


def main():
    torch.set_num_threads(2)
    protocol = load_protocol()
    scorers = {}
    for tag in MODELS:
        run = BASE / "runs" / f"{tag}_official_stable_20261117_130401"
        config = json.loads((run/"config.json").read_text())
        core = build_model(config)
        payload = torch.load(run/"checkpoints/best.pth", map_location="cpu", weights_only=False)
        core.load_state_dict(payload["model_state_dict"])
        del payload
        scorers[tag] = detection_model(core).cuda()
    path = ROOT / config["data"]["path"]
    data = load_data(path)
    _, ids = load_indices(path, config["split"]["train_ratio"], config["split"]["seed"])
    rows = []
    for scanned, sid in enumerate(ids, 1):
        sample = data[sid]
        events = list(candidates(sample))
        if not events:
            continue
        kind, a, b = random.Random(sid).choice(events)
        values = np.asarray(sample["time_series"]).reshape(-1)
        w = min(protocol["window_size"], len(values))
        row = {"series": sid, "kind": kind, "start": a, "end": b,
               "same_window": (a-128)//w == (b+127)//w}
        slices = (slice(a-128,a-32), slice(a-16,a), slice(a,b),
                  slice(b,b+16), slice(b+16,b+32), slice(b+32,b+128))
        for tag, scorer in scorers.items():
            p, _ = score_series(scorer, values, protocol)
            t = p.mean() + 3*p.std()
            row[tag] = {band: {"prob": float(p[s].mean()), "over_half": float((p[s]>0.5).mean()),
                               "official_alarm": float((p[s]>t).mean())} for band,s in zip(BANDS,slices)}
        rows.append(row)
        if len(rows)%128 == 0:
            print("events", len(rows), "scanned", scanned, flush=True)
        if len(rows) == 512:
            break
    if not rows:
        raise ValueError("No eligible shifts in validation split")
    report = {"scanned": scanned, "window": protocol["window_size"], "all": summary(rows),
              "same_window": summary([r for r in rows if r["same_window"]]),
              "increase": summary([r for r in rows if r["kind"]=="sudden increase"]),
              "decrease": summary([r for r in rows if r["kind"]=="sudden decrease"]), "events": rows}
    OUTPUT.write_text(json.dumps(report, allow_nan=False)+"\n")
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
