"""Paired pointwise classification probes on frozen full-observation representations."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE), str(ROOT)]
from src.benchmark_eval import load_protocol, read_real_series, real_targets
from src.data import load_data, load_indices
from src.models import build_model
from utils.dataset import TimeRCDDataset

SEEDS = (127, 128, 129)
TAGS = ("p0", "p1")
PROTOCOL = load_protocol()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def load_models():
    models = {}
    configs = []
    epochs = {}
    for tag in TAGS:
        run = BASE / "runs" / f"{tag}_official_stable_20261117_130401"
        config = json.loads((run / "config.json").read_text())
        model = build_model(config)
        payload = torch.load(run / "checkpoints/best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        epochs[tag] = int(payload["epoch"])
        models[tag] = model.requires_grad_(False).eval().cuda()
        configs.append(config)
    assert all(configs[0][k] == configs[1][k] for k in ("split", "data"))
    return models, configs[0], epochs


def windows(values):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    assert len(values) and np.isfinite(values).all()
    width = min(len(values), PROTOCOL["window_size"])
    ds = TimeRCDDataset(values, width, stride=width, normalize=True)
    for i in range(len(ds)):
        x, valid = ds[i]
        yield i * width, x[None].cuda(), valid[None].cuda(), int(valid.sum())


def cache(models, data, ids):
    rng = np.random.default_rng(112)
    feats = {tag: [] for tag in TAGS}
    labels = []
    for count, sid in enumerate(ids, 1):
        sample = data[sid]
        y = np.asarray(sample["labels"]).reshape(-1)
        picked = np.sort(rng.choice(len(y), min(128, len(y)), replace=False))
        labels.append(torch.tensor(y[picked], dtype=torch.float32))
        for offset, x, mask, n in windows(sample["time_series"]):
            ix = picked[(picked >= offset) & (picked < offset + n)] - offset
            if len(ix):
                for tag, model in models.items():
                    h = model.encoder(x, mask)[0, :, 0]
                    feats[tag].append(h[ix].cpu())
        if count % 512 == 0:
            print("cached", count, flush=True)
    return {tag: torch.cat(parts) for tag, parts in feats.items()}, torch.cat(labels)


def train(features, labels):
    assert torch.all((labels == 0) | (labels == 1))
    y = labels.cuda()
    probes = {}
    for tag in TAGS:
        x = features.pop(tag).cuda()
        mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
        x.sub_(mean).div_(std)
        heads = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 1)).cuda()
            opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=1e-5)
            rng = torch.Generator().manual_seed(seed + 1000)
            for epoch in range(15):
                for batch in torch.randperm(len(y), generator=rng).split(1024):
                    idx = batch.cuda()
                    logits = head(x[idx]).flatten()
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            assert torch.isfinite(loss)
            heads[str(seed)] = head.eval().requires_grad_(False)
            print("trained", tag, seed, flush=True)
        probes[tag] = (mean, std, heads)
        torch.save(dict(mean=mean.cpu(), std=std.cpu(), heads={s: {k:v.cpu() for k,v in h.state_dict().items()}
                   for s,h in heads.items()}), OUT / f"{tag}_heads.pt")
    return probes


@torch.no_grad()
def score(models, probes, values):
    scores = {}
    for _, x, mask, n in windows(values):
        for tag, model in models.items():
            h = model.encoder(x, mask)[0, :n, 0]
            original = model.anomaly_head(h).softmax(-1)[:, 1]
            mean, std, heads = probes[tag]
            z = (h - mean) / std
            preds = {seed: head(z).flatten().sigmoid() for seed, head in heads.items()}
            preds["original"] = original
            for seed, p in preds.items():
                scores.setdefault((tag, seed), []).append(p.cpu().numpy())
    return {k: np.concatenate(v).astype(float) for k,v in scores.items()}


def metrics(y, p):
    assert len(y) == len(p) and np.isfinite(p).all()
    alarm = p > p.mean() + 3*p.std()
    return dict(ap=float(average_precision_score(y, p)) if y.any() else None,
                recall=float(alarm[y].mean()) if y.any() else None,
                fpr=float(alarm[~y].mean()) if (~y).any() else None)


def evaluate(models, probes, items, domain, events=None, primary=None):
    rows, plateaus = [], []
    for i, (sid, group, x, labels) in enumerate(items, 1):
        y = np.asarray(labels).reshape(-1)
        assert np.isin(y, [0, 1]).all()
        y = y.astype(bool)
        predictions = score(models, probes, x)
        if primary is None or sid in primary:
            for (tag, seed), p in predictions.items():
                rows.append(dict(series=sid, group=group, tag=tag, seed=seed, **metrics(y,p)))
        if events and sid in events:
            a,b = events[sid]
            assert not y[a-128:a].any() and not y[b:b+128].any()
            for (tag, seed), p in predictions.items():
                alarm = p > p.mean() + 3*p.std()
                plateaus.append(dict(series=sid, tag=tag, seed=seed, pre=float(alarm[a-128:a].mean()),
                                     post=float(alarm[b:b+128].mean()), event=float(alarm[a:b].mean())))
        if i % 128 == 0:
            print(domain, i, flush=True)
    save(domain+"_rows.json", rows)
    if plateaus:
        save("platform_rows.json", plateaus)
    return rows, plateaus


def summarize(rows, keys):
    result = {}
    for seed in ("original", *map(str, SEEDS)):
        sets = [{r["series"]: r for r in rows if r["tag"] == tag and r["seed"] == seed} for tag in TAGS]
        assert sets[0].keys() == sets[1].keys()
        result[seed] = {}
        for key in keys:
            vals = np.array([[s[sid][key] for s in sets] for sid in sets[0] if sets[0][sid][key] is not None])
            delta = vals[:,1] - vals[:,0]
            rng = np.random.default_rng(72)
            sample = rng.choice(delta, (2000,len(delta)), replace=True).mean(1)
            result[seed][key] = dict(n=len(delta), p0=float(vals[:,0].mean()), p1=float(vals[:,1].mean()),
                                     delta=float(delta.mean()), ci=np.percentile(sample,[2.5,97.5]).tolist())
    return result


def main():
    torch.set_num_threads(2)
    models, cfg, epochs = load_models()
    path = ROOT / cfg["data"]["path"]
    print("loading dataset", flush=True)
    data = load_data(path)
    train_ids, valid = load_indices(path, cfg["split"]["train_ratio"], cfg["split"]["seed"])
    train_ids, val_ids = train_ids[:2048], valid[:512]
    previous = json.loads((OUT.parent / "stability_20261117/level_shift_probe.json").read_text())
    events = {r["series"]:(r["start"],r["end"]) for r in previous["events"]}
    selected = sorted(set(val_ids) | events.keys())
    assert len(events) == 512 and set(selected).issubset(valid) and set(selected).isdisjoint(train_ids)
    features, y = cache(models, data, train_ids)
    save("design.json", dict(train=train_ids, validation=val_ids, platform=list(events), 
         seeds=SEEDS, best_epochs=epochs, training_points=len(y), positive_points=int(y.sum()),
         primary_seed=SEEDS[0], gate="All seed AP differences positive and primary paired 95% lower bound > 0"))
    probes = train(features, y)
    items = ((sid, "synthetic", data[sid]["time_series"], data[sid]["labels"]) for sid in selected)
    rows, platforms = evaluate(models, probes, items, "synthetic", events, set(val_ids))
    summary = summarize(rows, ("ap","recall","fpr"))
    save("synthetic.json", summary)
    save("platform.json", summarize(platforms, ("pre","post","event")))
    passed = all(summary[str(s)]["ap"]["delta"] > 0 for s in SEEDS) and summary[str(SEEDS[0])]["ap"]["ci"][0] > 0
    print("synthetic primary", summary[str(SEEDS[0])], "real gate", passed, flush=True)
    if passed:
        targets = real_targets(ROOT / PROTOCOL["real_data_dir"])
        items = ((path.name, group, *read_real_series(path)[:2]) for group, path in targets)
        real, _ = evaluate(models, probes, items, "real")
        save("real.json", summarize(real, ("ap","recall","fpr")))
    print("complete", flush=True)


if __name__ == "__main__":
    main()
