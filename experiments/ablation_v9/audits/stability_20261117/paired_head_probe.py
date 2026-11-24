"""Paired read-only probe of the stable checkpoints on the validation prefix."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE), str(ROOT)]
from src.data import PairedTimeSeriesDataset, collate_samples, load_indices
from src.models import build_model
from src.train import make_batch_inputs
from torch.utils.data import DataLoader

def main():
    torch.set_num_threads(2)
    models = []
    for tag in ("p0", "p1"):
        run = BASE / "runs" / f"{tag}_official_stable_20261117_130401"
        config = json.loads((run / "config.json").read_text())
        model = build_model(config)
        state = torch.load(run / "checkpoints/best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        del state
        models.append(model.cuda().eval())
    path = ROOT / config["data"]["path"]
    _, valid = load_indices(path, config["split"]["train_ratio"], config["split"]["seed"])
    dataset = PairedTimeSeriesDataset(path, valid[:1024])
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_samples)
    rng = torch.Generator().manual_seed(config["evaluation"]["validation_mask_seed"])
    rows = []
    with torch.inference_mode():
        for batch in loader:
            masked, visible, mask = make_batch_inputs(batch, config, rng)
            # Preserve full-batch preprocessing and split only the GPU forward.
            outs = []
            for model in models:
                chunks = []
                for x, v in zip(masked.split(4), visible.split(4)):
                    output = model(x.cuda(), v.cuda())
                    chunks.append({k: t.cpu() for k, t in output.items() if k != "h"})
                outs.append({k: torch.cat([c[k] for c in chunks]) for k in chunks[0]})
            for i in range(len(mask)):
                sel = mask[i]
                x, x0 = batch["x"][i, sel, 0], batch["x0"][i, sel, 0]
                rows.append(np.stack([a.numpy() for a in (
                    batch["y"][i, sel], x, x0, outs[0]["reconstruction"][i, sel, 0],
                    outs[1]["reconstruction"][i, sel, 0], outs[1]["normal_reference"][i, sel, 0]
                )], axis=1))
    data = np.concatenate(rows).astype("float64")
    assert np.isfinite(data).all()
    y, x, x0, p0, obs, ref = data.T
    d, h = x-x0, obs-ref
    result = {"series": len(rows), "masked": len(y), "groups": {}}
    for name, m in {"normal": y==0, "anomaly": y>0, "changed": d!=0,
                    "normal_changed": (y==0)&(d!=0), "anomaly_changed": (y>0)&(d!=0)}.items():
        n = int(m.sum())
        if not n:
            result["groups"][name] = {"n": n}
            continue
        def mse(a):
            return float(np.mean(a[m]**2))
        real = m & (d!=0)
        corr = float(np.corrcoef(d[m],h[m])[0,1]) if np.std(d[m]) and np.std(h[m]) else None
        result["groups"][name] = dict(n=n, p0_x=mse(p0-x), obs_x=mse(obs-x), ref_x=mse(ref-x),
            obs_x0=mse(obs-x0), ref_x0=mse(ref-x0), copy_x0=mse(d), delta_mse=mse(h-d),
            delta_corr=corr, pred_delta_rms=mse(h)**0.5, sign=float(np.mean(
            np.sign(h[real])==np.sign(d[real]))) if real.any() else None)
    wins = []
    for r in rows:
        sub = r[r[:,0]>0].astype("float64")
        if len(sub):
            wins.append(np.mean((sub[:,4]-sub[:,2])**2)-np.mean((sub[:,5]-sub[:,2])**2))
    result["series_with_masked_anomaly"] = len(wins)
    result["series_ref_wins_x0"] = int(np.sum(np.array(wins)>0))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
