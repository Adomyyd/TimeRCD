"""Official group-macro and per-series paired differences against both fixed baselines."""
import json
import sys
from pathlib import Path

import pandas as pd

out, plan_file = map(Path, sys.argv[1:])
plan = json.loads(plan_file.read_text())
cols = ['Affiliation-F', 'F1_T', 'Standard-F1', 'VUS-PR']


def read(folder):
    df = pd.read_csv(folder / 'metrics/per_series.csv')
    df = df.query("domain == 'real' and score_kind == 'probability'").set_index(['group', 'series_id'])[cols].sort_index()
    assert len(df) == 700 and df.index.is_unique and df.notna().all().all()
    return df


new = read(out)
summary = {}
for tag in ['p0', 'p1']:
    old = read(Path(plan[tag + '_baseline']))
    assert old.index.equals(new.index)
    delta = 100 * (new-old)
    delta.to_csv(out / 'metrics' / f'paired_minus_{tag}.csv')
    summary[tag] = delta.groupby(level='group').mean().mean().to_dict()
(out / 'metrics/paired_group_macro_pp.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
