"""Paired context ablation for fixed anomaly and frozen-probe readouts."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / "frozen_classifier_20261121"
sys.path.insert(0, str(PREV))
import probe as prev

P = 16
HEADS = ("original", "probe127")
KINDS = ("anomaly", "normal")
LABELS = ("baseline", "near", "far")


def load_heads():
    heads = {}
    for tag in prev.TAGS:
        state = torch.load(PREV / f"{tag}_heads.pt", map_location="cpu", weights_only=True)
        head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        head.load_state_dict(state["heads"]["127"])
        heads[tag] = (state["mean"].cuda(), state["std"].cuda(), head.cuda().eval())
    return heads


def pick(y, width, rng):
    # Each target needs valid symmetric offsets in its own model window.
    candidates = []
    for offset in range(0, len(y), width):
        n = min(len(y) - offset, width) // P
        if n < 21:
            continue
        counts = y[offset:offset+n*P].reshape(n, P).sum(1)
        ii = np.arange(10, n-10)
        a, normal = ii[counts[ii] > 0], ii[counts[ii] == 0]
        if len(a) and len(normal):
            candidates.append((offset, a, normal))
    if not candidates:
        return None
    # Uniform over eligible anomalous patches, then same-window normal control.
    sizes = np.array([len(a) for _, a, _ in candidates])
    k = rng.choice(len(sizes), p=sizes/sizes.sum())
    offset, a, normal = candidates[k]
    return offset, [int(rng.choice(a)), int(rng.choice(normal))]


def blocks(t, distance):
    return [t-distance-1, t-distance, t+distance, t+distance+1]


@torch.inference_mode()
def series_rows(sid, group, values, labels, rng, models, heads):
    y = np.asarray(labels).reshape(-1)
    assert np.isin(y, [0, 1]).all()
    width = min(len(y), prev.PROTOCOL["window_size"])
    targets = pick(y, width, rng)
    if not targets:
        return []
    offset, chosen = targets
    ds = prev.TimeRCDDataset(np.asarray(values, dtype=float).reshape(-1, 1), width, stride=width, normalize=True)
    x, valid = ds[offset // width]
    n = int(valid.sum())
    masks = valid.repeat(5, 1)
    meta = []
    for i, t in enumerate(chosen):
        for j, distance in enumerate((1, 9)):
            idx = np.concatenate([np.arange(b*P,(b+1)*P) for b in blocks(t,distance)])
            assert valid[idx].all() and not np.any(idx//P == t)
            masks[1 + i*2+j, idx] = False
            meta.append(int(y[offset+idx].sum()))
    x = x[None].expand(5, -1, -1).cuda()
    masks = masks.cuda()
    rows = []
    for tag, model in models.items():
        h = model.encoder(x, masks)[:, :n, 0]
        mean, std, head = heads[tag]
        outputs = [model.anomaly_head(h).softmax(-1)[...,1], head((h-mean)/std).squeeze(-1).sigmoid()]
        for name, out in zip(HEADS, outputs):
            probs = out.cpu().numpy()
            assert np.isfinite(probs).all()
            for i, t in enumerate(chosen):
                points = np.arange(t*P, (t+1)*P)
                if i == 0:
                    points = points[y[offset+points] > 0]
                row = dict(series=sid, group=group, target=KINDS[i], model=tag, head=name,
                           offset=offset, patch=t, points=len(points), near_positive=meta[2*i],
                           far_positive=meta[2*i+1])
                # Identical unperturbed background scores anchor percentile changes.
                background = np.sort(probs[0])
                for label, k in zip(LABELS, (0, 2*i+1, 2*i+2)):
                    p = probs[k, points]
                    left = background.searchsorted(p)
                    right = background.searchsorted(p, side="right")
                    row[label] = [float(p.mean()), float(((left+right)/(2*n)).mean())]
                rows.append(row)
    return rows


def interval(d):
    rng = np.random.default_rng(72)
    lo, hi = np.percentile(d[rng.integers(len(d), size=(3000,len(d)))].mean(1), [2.5,97.5])
    return [float(d.mean()), float(lo), float(hi)]


def summarize(rows):
    summary = {}
    for name in HEADS:
        for kind in KINDS:
            groups = [[r for r in rows if r["head"]==name and r["target"]==kind and r["model"]==tag]
                      for tag in prev.TAGS]
            assert [r["series"] for r in groups[0]] == [r["series"] for r in groups[1]]
            for subset, use in (("all", lambda r: True), ("both_normal", lambda r: r["near_positive"]==r["far_positive"]==0),
                                ("near_more_anomaly", lambda r: r["near_positive"]>r["far_positive"])):
                results, effects = {}, []
                for tag, group in zip(prev.TAGS, groups):
                    g = [r for r in group if use(r)]
                    if not g:
                        break
                    v = np.array([[r[l] for l in LABELS] for r in g])
                    diff = v[:,2]-v[:,1]
                    results[tag] = dict(n=len(g), means=v.mean(0).tolist(), probability=interval(diff[:,0]),
                                        percentile=interval(diff[:,1]))
                    effects.append(diff)
                if len(effects)==2:
                    d = effects[1]-effects[0]
                    results["p1_minus_p0"] = [interval(d[:,j]) for j in range(2)]
                    summary[f"{name}/{kind}/{subset}"] = results
    return summary


def evaluate(domain, items, models, heads):
    rows, skipped = [], []
    for i, (sid, group, values, labels) in enumerate(items, 1):
        rng = np.random.default_rng([72, i])
        sub = series_rows(sid, group, values, labels, rng, models, heads)
        if sub:
            rows.extend(sub)
        else:
            skipped.append(sid)
        if i % 128 == 0:
            print(domain, i, flush=True)
    summary = summarize(rows)
    result = dict(seen=i, eligible=i-len(skipped), skipped=skipped, summary=summary)
    for filename, content in ((domain+"_rows",rows),(domain,result)):
        (OUT/(filename+".json")).write_text(json.dumps(content,allow_nan=False))
    print(domain, i, "eligible", i-len(skipped), flush=True)


def main():
    torch.set_num_threads(2)
    models, cfg, _ = prev.load_models()
    assert all(m.encoder.patch_size==P for m in models.values())
    heads = load_heads()
    targets = prev.real_targets(prev.ROOT / prev.PROTOCOL["real_data_dir"])
    evaluate("real", ((p.name,g,*prev.read_real_series(p)[:2]) for g,p in targets), models, heads)
    ids = json.loads((PREV/"design.json").read_text())["validation"]
    print("loading synthetic validation samples", flush=True)
    data = prev.load_data(prev.ROOT/cfg["data"]["path"])
    evaluate("synthetic", ((i,"synthetic",data[i]["time_series"],data[i]["labels"]) for i in ids), models, heads)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
