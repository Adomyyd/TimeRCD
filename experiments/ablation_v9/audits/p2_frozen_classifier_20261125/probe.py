"""P2 extension of the archived, paired P0/P1 frozen-probe protocol."""
import gzip
import json
from pathlib import Path

from audits.frozen_classifier_20261121 import probe as old


def run(model, cfg, run_dir, plan):
    prev = Path(plan['reference_probe'])
    design = json.loads((prev / 'design.json').read_text())
    assert list(old.SEEDS) == design['seeds']
    old.TAGS, old.OUT = ('p2',), run_dir
    path = cfg['data']['path']
    print('loading P2 probe data', flush=True)
    data = old.load_data(path)
    train, valid = old.load_indices(path, cfg['split']['train_ratio'], cfg['split']['seed'])
    assert train[:2048] == design['train'] and valid[:512] == design['validation']
    models = {'p2': model}
    features, y = old.cache(models, data, train[:2048])
    assert len(y) == design['training_points'] and int(y.sum()) == design['positive_points']
    old.save('sample_ids.json', {'train': train[:2048], 'validation': valid[:512]})
    probes = old.train(features, y)
    targets = old.real_targets(old.ROOT / old.PROTOCOL['real_data_dir'])
    domains = [('synthetic', ((i, 'synthetic', data[i]['time_series'], data[i]['labels']) for i in valid[:512])),
               ('real', ((p.name, g, *old.read_real_series(p)[:2]) for g, p in targets))]
    for domain, items in domains:
        old.TAGS = ('p2',)
        rows, _ = old.evaluate(models, probes, items, domain)
        with gzip.open(prev / (domain + '_rows.json.gz'), 'rt') as f:
            baseline = json.load(f)
        result = {}
        for group in (['all', 'UCR'] if domain == 'real' else ['all']):
            selected = [r for r in baseline + rows if group == 'all' or r['group'] == group]
            result[group] = {}
            for tag in ('p0', 'p1'):
                old.TAGS = (tag, 'p2')
                pair = old.summarize(selected, ('ap', 'recall', 'fpr'))
                # Historical helper names its two columns p0 and p1, irrespective of TAGS.
                for measures in pair.values():
                    for m in measures.values():
                        a, b = m.pop('p0'), m.pop('p1')
                        m[tag], m['p2'] = a, b
                result[group]['p2_minus_' + tag] = pair
        old.save('metrics/' + domain + '_paired.json', result)
