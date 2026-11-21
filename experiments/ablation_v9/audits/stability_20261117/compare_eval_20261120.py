"""Verify and report the stable P0/P1 pair; --check-history validates tables without writing."""
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V9 = HERE.parents[1]
RUNS = V9 / 'runs'
history = sys.argv[1:] == ['--check-history']
assert history or not sys.argv[1:]
methods = ['P0 official', 'P0 ignore'] if history else ['P0 stable', 'P1 stable']
names = ['p0_official_dp_20261111_003608_eval_20261114_restart2_s200_r700','p0_ignore_dp_20261112_170107_eval_20261114_restart2_s200_r700'] if history else [f'p{i}_official_stable_20261117_130401_eval_20261120_s200_r700' for i in range(2)]
main = ['Affiliation-F','F1_T','Standard-F1','VUS-PR']
extra = ['AUC-PR','AUC-ROC','VUS-ROC','PA-F1']
groups = 'IOPS MGAB NAB NEK Power SED Stock TODS UCR WSD YAHOO'.split()
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
fmt = lambda x: f'{100*x:.2f}'
row = lambda label, values: '| ' + ' | '.join([label,*values]) + ' |'
manifest, ids, score, epochs, tables, missing = [], [], [], [], [], []
for name, label in zip(names,methods):
    r = RUNS/name
    m = read(r/'evaluation_manifest.json'); e = read(r/'metrics/eval.json'); s = e['summaries']
    assert read(r/'status.json')['status'] == e['status'] == 'completed'
    assert m['synthetic_selected'] == 200 and m['real_selected'] == 700
    train = name.split('_eval_')[0]
    assert Path(m['checkpoint']).parts[-3:] == (train,'checkpoints','best.pth')
    epoch = e['checkpoint_epoch']; assert epoch == m['checkpoint_epoch'] == read(RUNS/train/'metrics/train.json')['best_epoch']
    manifest.append([m[k] for k in ('protocol','upstream_commit','source_sha256','dependency_versions','synthetic_indices_sha256','real_files')]); epochs.append(epoch)
    with (r/'metrics/per_series.csv').open() as f: records = list(csv.DictReader(f))
    keys = [tuple(t[k] for k in ('domain','group','series_id','score_kind','points','anomaly_points','train_index','sliding_window')) for t in records]
    assert len(keys) == len(set(keys)) == 1800; ids.append(sorted(keys))
    count = Counter((t['domain'],t['score_kind']) for t in records)
    assert count == {(d,k):n for d,n in [('synthetic',200),('real',700)] for k in ('probability','logit')}
    for d,k in count:
        block = s[d][k]; rows = [t for t in records if (t['domain'],t['score_kind']) == (d,k)]
        for field,value in block['group_macro'].items():
            means = []
            for g,summary in block['group_means'].items():
                finite = [float(t[field]) for t in rows if t['group']==g and t[field] and math.isfinite(float(t[field]))]; assert len(finite) == block['valid_metric_counts']['groups'][g][field]
                if finite:
                    gm = statistics.mean(finite); assert math.isclose(gm,summary[field],abs_tol=1e-10); means.append(gm)
            assert math.isclose(statistics.mean(means),value,abs_tol=1e-10)
    S = s['real']['probability']; G = S['group_means']; assert set(G) == set(groups)
    miss = {k:n for k,n in S['valid_metric_counts']['series_macro'].items() if n != 700}; missing.append(miss)
    score.append(s); values = lambda branch, fields: row(label,[fmt(branch[k]) for k in fields]); tab = [values(S['group_macro'], main)]
    for k in main: tab.append(row(label,[fmt(G[g][k]) for g in groups]+[fmt(S['group_macro'][k])]))
    for d in ('real','synthetic'): tab.append(values(s[d]['probability']['series_macro'], main+extra))
    for d,avg in [('real','group_macro'),('real','series_macro'),('synthetic','series_macro')]: tab.append(values(s[d]['logit'][avg],main))
    for fields in [extra+['Standard-Precision','Standard-Recall'],['PA-Precision','PA-Recall','Affiliation-P','Affiliation-R','Precision_T','Recall_T']]: tab.append(values(S['group_macro'],fields))
    tab.append(row(label,['400k，official padding，稳定训练', '72', str(epoch),f'[评测 run](runs/{name}/)'])); tables.append(tab)
assert manifest[0] == manifest[1] and ids[0] == ids[1]
path = V9/'RESULTS.md'; original = path.read_text(encoding='utf-8'); lines = original.splitlines()
if history:
    for tab in tables:
        for expected in tab[:-1]: assert expected in lines, expected
    print('Historical pair: all 24 numeric table rows and paired records verified.'); sys.exit()
assert epochs == [19,23]
report = ['# 稳定版 P0 与 P1 检测性能对照','','固定训练种子 72，分别使用 best epoch 19、23。每组 200 条合成留出、700 条真实序列；两组序列及评测协议已核对。以下为 probability 百分制；差值按未舍入数值计算，单位为百分点。','','| 数据及口径 | 指标 | P0 stable | P1 stable | P1 − P0 |','|---|---|---:|---:|---:|']
for domain,avg,title in [('real','group_macro','真实，11 组等权'),('real','series_macro','真实，700 条等权'),('synthetic','series_macro','合成，200 条等权')]:
    for k in main:
        a,b = [s[domain]['probability'][avg][k]*100 for s in score]; report.append(row(title,[k,f'{a:.2f}',f'{b:.2f}',f'{b-a:+.2f}']))
report.extend(['','真实数据各组 P1 − P0（百分点）：','',row('数据集',main),row('---',['---:']*4)])
for g in groups:
    delta = [(score[1]['real']['probability']['group_means'][g][k]-score[0]['real']['probability']['group_means'][g][k])*100 for k in main]; report.append(row(g,[f'{v:+.2f}' for v in delta]))
report += ['','真实 probability 有效数不足 700 的指标：'+ '；'.join(f'{method}: {miss}' for method,miss in zip(methods,missing))+'。其他指标各有 700 个有效值。','', '合成留出参与选 best；F1 含测试标签阈值搜索。单种子配对只描述这两份权重，不能据此证明上下文利用。','']
if not any(x.startswith('| P0 stable |') for x in lines):
    anchors = [i for i,x in enumerate(lines) if x.startswith('| P0 ignore |')]; assert len(anchors) == len(tables[0])
    for j,i in reversed(list(enumerate(anchors))): lines[i+1:i+1] = [tab[j] for tab in tables]
    text = '\n'.join(lines)+'\n'; text = text.replace('稳定版 P0/P1 的同协议检测结果待登记。','稳定版 P0/P1 的同协议检测及差值见[配对报告](audits/stability_20261117/EVAL_PAIR_20261120.md)。').replace('当前 P0 使用','当前 P0/P1 使用').replace('两组均完成，每组 1800 行','历史 padding 对照两组均完成，每组 1800 行').replace('NaN 沿用官方返回值','以下计数针对历史 padding 对照。NaN 沿用官方返回值'); assert path.read_text(encoding='utf-8') == original
    tmp = path.with_suffix('.stable-eval.tmp'); tmp.write_text(text,encoding='utf-8'); tmp.replace(path)
else:
    for tab in tables:
        for item in tab: assert item in lines
(HERE/'EVAL_PAIR_20261120.md').write_text('\n'.join(report),encoding='utf-8')
print('\n'.join(report))
