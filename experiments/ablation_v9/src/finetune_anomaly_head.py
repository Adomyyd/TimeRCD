"""Tune the original P1 classifier on full-observation windows, with frozen features."""
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence

from .benchmark_eval import detection_model, load_protocol, read_real_series, real_targets, score_series, write_json
from .data import LengthBucketBatchSampler, load_data, load_indices


def windows(sample, width):
    x = np.asarray(sample['time_series'], dtype=float).reshape(-1, 1)
    y = torch.as_tensor(np.asarray(sample['labels']).reshape(-1), dtype=torch.long)
    assert len(x) == len(y) and len(y) > 0
    # Match TimeRCDDataset: normalize the series before non-overlapping windows;
    # repeat the last value for the final padded window, exclude padding from CE.
    std = x.std(axis=0)
    x = torch.from_numpy(((x - x.mean(axis=0)) / np.where(std == 0, 1e-8, std)).astype('float32'))
    w = min(len(x), width)
    for a in range(0, len(x), w):
        part, label = x[a:a+w], y[a:a+w]
        if len(part) < w:
            n = w - len(part)
            part = torch.cat((part, part[-1:].expand(n, -1)))
            label = torch.cat((label, label.new_full((n,), -1)))
        yield part, label


def epoch_run(model, data, batches, micro, width, optimizer=None):
    head = model.anomaly_head
    head.train(optimizer is not None)
    ce, points = 0., 0
    for step, ids in enumerate(batches, 1):
        pieces = [w for i in ids for w in windows(data[i], width)]
        n = sum(len(data[i]['labels']) for i in ids)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for pos in range(0, len(pieces), micro):
            xs, ys = zip(*pieces[pos:pos + micro])
            x = pad_sequence(xs, batch_first=True).cuda()
            y = pad_sequence(ys, batch_first=True, padding_value=-1).cuda()
            valid = y >= 0
            with torch.no_grad():
                hidden = model.encoder(x, valid)
            with torch.set_grad_enabled(optimizer is not None):
                logits = head(hidden).mean(dim=-2)
                loss = cross_entropy(logits[valid], y[valid], reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite full-window CE')
                if optimizer is not None:
                    (loss / n).backward()
            ce += float(loss.detach())
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        points += n
        if step == 1 or step % 250 == 0:
            print('train' if optimizer else 'validation', step, '/', len(batches), 'CE', ce/points, flush=True)
    return ce / points


def run(model, cfg, run_dir, plan):
    protocol = load_protocol()
    baseline = Path(plan['p1_baseline'])
    checks = []
    scorer = detection_model(model)
    for _, p in real_targets(protocol['real_data_dir'])[:3]:
        x, labels, _, _ = read_real_series(p)
        prob, margin = score_series(scorer, x, protocol)
        with np.load(baseline / 'scores' / ('real_' + p.stem + '.npz')) as old:
            np.testing.assert_array_equal(labels, old['label'])
            np.testing.assert_allclose(prob, old['anomaly_score'], atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(margin, old['logits'], atol=2e-4, rtol=1e-5)
            checks.append({'series': p.name, 'max_probability_difference': float(abs(prob-old['anomaly_score']).max())})
    write_json(run_dir / 'metrics/zero_step.json', checks)
    model.eval().requires_grad_(False)
    model.anomaly_head.requires_grad_(True)
    assert all(p.requires_grad == n.startswith('anomaly_head.') for n, p in model.named_parameters())
    optimizer = torch.optim.AdamW(model.anomaly_head.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
    path = cfg['data']['path']
    print('zero-step passed; loading training data', flush=True)
    data = load_data(path)
    train, val = load_indices(path, cfg['split']['train_ratio'], cfg['split']['seed'])
    def sampler(ids):
        return LengthBucketBatchSampler(ids, [len(data[i]['labels']) for i in ids], plan['batch_windows'], cfg['runtime']['seed'])
    batches, validation = sampler(train), sampler(val)
    best, history = float('inf'), []
    assert 1 <= plan['epochs'] <= 2
    for epoch in range(1, plan['epochs']+1):
        batches.set_epoch(epoch-1)
        loss = epoch_run(model, data, batches, plan['microbatch'], protocol['window_size'], optimizer)
        v = epoch_run(model, data, validation, plan['microbatch'], protocol['window_size'])
        history.append({'epoch': epoch, 'train_ce': loss, 'validation_ce': v})
        write_json(run_dir / 'metrics/head_epochs.json', history)
        if v < best:
            best = v
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, config=cfg), run_dir / 'checkpoints/best.pth')
        torch.save(dict(epoch=epoch, state_dict=model.anomaly_head.state_dict()), run_dir / 'checkpoints' / f'head_epoch_{epoch}.pt')
        print(history[-1], flush=True)
