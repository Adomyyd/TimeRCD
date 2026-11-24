"""Small paired frozen-encoder readout, independent of the training entry point."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE), str(ROOT)]
from src.data import PairedTimeSeriesDataset, collate_samples, load_indices
from src.models import build_model
from src.train import make_batch_inputs


def cache(models, config, path, ids, training):
    loader = DataLoader(PairedTimeSeriesDataset(path, ids), batch_size=32, collate_fn=collate_samples)
    rng = torch.Generator().manual_seed(72 if training else config["evaluation"]["validation_mask_seed"])
    sample_rng = torch.Generator().manual_seed(112)
    features = [[], []]
    targets, labels, groups = [], [], []
    for k, batch in enumerate(loader):
        x, visible, mask = make_batch_inputs(batch, config, rng)
        if training:
            for row in mask:
                points = row.nonzero().flatten()
                row[points[torch.randperm(len(points), generator=sample_rng)[128:]]] = False
        groups.append(mask.nonzero()[:,0] + k*32)
        targets.append(batch["x0"][mask].reshape(-1))
        labels.append(batch["y"][mask].reshape(-1))
        for j, encoder in enumerate(models):
            with torch.no_grad():
                for a, v, m in zip(x.split(4), visible.split(4), mask.split(4)):
                    h = encoder(a.cuda(), v.cuda())
                    features[j].append(h[m.cuda()].reshape(-1, 256).cpu())
        if (k+1)%16 == 0:
            print("cached", "train" if training else "holdout", (k+1)*32, flush=True)
    return [torch.cat(f) for f in features], torch.cat(targets), torch.cat(labels), torch.cat(groups)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(72)
    encoders = []
    epochs = []
    for tag in ("p0","p1"):
        run = BASE / "runs" / (tag+"_official_stable_20261117_130401")
        cfg = json.loads((run/"config.json").read_text())
        model = build_model(cfg)
        ckpt = torch.load(run/"checkpoints/best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        encoders.append(model.encoder.eval().requires_grad_(False).cuda())
        epochs.append(ckpt["epoch"])
        del model, ckpt
    path = ROOT / cfg["data"]["path"]
    print("loading data", flush=True)
    train_ids, valid_ids = load_indices(path, cfg["split"]["train_ratio"], cfg["split"]["seed"])
    train_ids, valid_ids = train_ids[:2048], valid_ids[:512]
    assert set(train_ids).isdisjoint(valid_ids)
    train, y, labels, _ = cache(encoders, cfg, path, train_ids, True)
    val, truth, valid_labels, groups = cache(encoders, cfg, path, valid_ids, False)
    torch.manual_seed(127)
    initial = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Linear(128,1))
    errors, history = [], []
    for j in range(2):
        X, V = train[j].cuda(), val[j].cuda()
        mean, std = X.mean(0), X.std(0).clamp_min(1e-5)
        X = (X-mean)/std
        V = (V-mean)/std
        head = copy.deepcopy(initial).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=1e-5)
        rng = torch.Generator().manual_seed(83)
        target = y.cuda()
        trace = []
        for epoch in range(15):
            for ix in torch.randperm(len(y), generator=rng).split(1024):
                ix = ix.cuda()
                loss = (head(X[ix]).flatten()-target[ix]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                err = (torch.cat([head(v).flatten().cpu() for v in V.split(4096)])-truth).double().square()
            assert torch.isfinite(err).all()
            trace.append([float(err.mean()), float(err[valid_labels>0].mean())])
        errors.append(err.numpy())
        history.append(trace)
        print("finished probe", j, trace[-1], flush=True)
    result = dict(best_epochs=epochs, train_series=len(train_ids), holdout_series=len(valid_ids),
                  train_points=len(y), train_anomaly=int((labels>0).sum()), history_all_anomaly=history)
    for name, mask in (("all", np.ones(len(truth),dtype=bool)), ("normal",valid_labels.numpy()==0),
                       ("anomaly",valid_labels.numpy()>0)):
        idx = groups.numpy()[mask]
        counts = np.bincount(idx,minlength=len(valid_ids))
        a,b = [np.bincount(idx,weights=e[mask],minlength=len(valid_ids)) for e in errors]
        active = counts>0
        rng = np.random.default_rng(72)
        sample = rng.choice(np.flatnonzero(active),(2000,active.sum()),replace=True)
        delta = (b-a)[sample].sum(1)/counts[sample].sum(1)
        result[name] = dict(points=int(counts.sum()),series=int(active.sum()),
                            p0=float(a.sum()/counts.sum()), p1=float(b.sum()/counts.sum()),
                            paired_ci=np.percentile(delta,[2.5,97.5]).tolist(),p1_wins=int((b[active]<a[active]).sum()))
    dest = OUT/"result.json"
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result["anomaly"],flush=True)


if __name__ == "__main__":
    main()
