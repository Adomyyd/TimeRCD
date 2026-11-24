"""Frozen P1 residual readout. Run from this audit directory on the experiment host."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "frozen_classifier_20261121"))
import probe as prev
from evaluation.basic_metrics import basic_metricor
from src.data import apply_input_mask

metric = basic_metricor().metric_PR


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, allow_nan=False, indent=2))


@torch.inference_mode()
def features(model, values):
    result = []
    for _, x, valid, n in prev.windows(values):
        hidden = model.encoder(x, valid)[0, :n, 0]
        logits = model.anomaly_head(hidden)
        margin = logits[:, 1] - logits[:, 0]
        indices = torch.arange(x.shape[1], device=x.device)
        phase = indices // model.encoder.patch_size % 4
        hide = phase[None, :] == torch.arange(4, device=x.device)[:, None]
        h = model.encoder(apply_input_mask(x.expand(4, -1, -1), hide),
                          valid & ~hide)[phase[:n], indices[:n], 0]
        normal = model.normal_reference_head(h).flatten()
        observed = model.reconstruction_head(h).flatten()
        result.append(torch.stack([margin, (x[0, :n, 0] - normal).abs(),
                                   (observed - normal).abs()], 1).cpu().numpy())
    arr = np.concatenate(result)
    assert len(arr) == len(values) and np.isfinite(arr).all()
    return arr


def evaluate(model, fits, items, name):
    rows = []
    for i, (sid, group, x, labels) in enumerate(items, 1):
        y = np.asarray(labels).reshape(-1)
        assert np.isin(y, [0, 1]).all()
        z = features(model, x).astype(float)
        scores = [expit((z[:, :len(w)] - mean) / scale @ w + bias)
                  for w, bias, mean, scale in fits]
        a, b = [float(metric(y, s)) for s in scores]
        row = dict(series=sid, group=group, a=a, b=b, delta=b-a)
        if name == 'real':
            ranks = [rankdata(s) / len(y) for s in scores]
            dr = ranks[1] - ranks[0]
            for key, target, sign in [('positive_up', 1, 1), ('negative_down', 0, -1),
                                      ('positive_down', 1, -1), ('negative_up', 0, 1)]:
                choices = np.flatnonzero(y == target)
                if len(choices):
                    k = int(choices[np.argmax(sign * dr[choices])])
                    row[key] = dict(index=k, rank_a=ranks[0][k], rank_b=ranks[1][k],
                                    features=z[k].tolist())
        rows.append(row)
        if i % 100 == 0:
            print(name, i, flush=True)
    save(name + '_rows.json', rows)
    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(72)
    grouped, boot = {}, []
    values = np.array([[r['a'], r['b']] for r in rows])
    for g in groups:
        v = values[[r['group'] == g for r in rows]]
        d = v[:, 1] - v[:, 0]
        grouped[g] = [len(v), *v.mean(axis=0).tolist()]
        boot.append(rng.choice(d, (3000, len(d))).mean(1))
    d = values[:, 1] - values[:, 0]
    bounds = lambda samples: np.percentile(samples, [2.5, 97.5]).tolist()
    result = dict(groups=grouped, series_macro=values.mean(0).tolist(),
                  series_ci=bounds(rng.choice(d, (3000, len(d))).mean(1)),
                  group_macro=np.mean([v[1:] for v in grouped.values()],axis=0).tolist(),
                  group_ci=bounds(np.mean(boot,axis=0)),
                  wins=int((d>1e-12).sum()), losses=int((d < -1e-12).sum()), n=len(d))
    save(name+'.json', result)
    print(name, result, flush=True)


def main():
    torch.set_num_threads(2)
    models, cfg, epochs = prev.load_models()
    model = models.pop('p1')
    del models
    data_path = prev.ROOT / cfg['data']['path']
    data = prev.load_data(data_path)
    train, valid = prev.load_indices(data_path, cfg['split']['train_ratio'],cfg['split']['seed'])
    train, valid = train[:2048], valid[:512]
    assert not set(train).intersection(valid)
    parts, labels = [], []
    rng = np.random.default_rng(112)
    for i, sid in enumerate(train,1):
        sample = data[sid]
        y = np.asarray(sample['labels']).reshape(-1)
        ix = rng.choice(len(y),min(128,len(y)),replace=False)
        parts.append(features(model, sample['time_series'])[ix])
        labels.append(y[ix])
        if i % 256 == 0:
            print('fit features', i, flush=True)
    x, y = np.concatenate(parts).astype(float), np.concatenate(labels)
    fits = []
    for n in (1,3):
        scaler = StandardScaler().fit(x[:,:n])
        lr = LogisticRegression(C=1, max_iter=200, tol=1e-8).fit(scaler.transform(x[:,:n]),y)
        fits.append((lr.coef_[0],lr.intercept_[0],scaler.mean_,scaler.scale_))
    assert fits[0][0][0] > 0
    save('fit.json',dict(train=train,validation=valid,epoch=epochs['p1'],points=len(y),
         positive=int(y.sum()), parameters=[[np.asarray(v).tolist() for v in fit] for fit in fits]))
    evaluate(model,fits,((sid,'synthetic',data[sid]['time_series'],data[sid]['labels']) for sid in valid),'synthetic')
    targets = prev.real_targets(prev.ROOT / prev.PROTOCOL['real_data_dir'])
    assert len(targets)==700
    evaluate(model,fits,((p.name,g,*prev.read_real_series(p)[:2]) for g,p in targets),'real')
    print('complete',flush=True)


if __name__ == '__main__':
    main()
