"""Run the stable pair on real data only; log progress every half hour."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import os
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
v9 = here.parents[1]
root = v9.parents[1]
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(v9), os.environ.get("PYTHONPATH","")]), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0,1")
ids = [f"p{i}_official_stable_20261117_130401" for i in range(2)]
runs = [v9/"runs"/(s+"_eval_real700_20261121") for s in ids]
def read(p): return json.loads(p.read_text())
def snapshot(label):
    out = []
    for run in runs:
        f = run/"metrics/per_series.csv"
        with f.open() if f.exists() else open(os.devnull) as h:
            rows = list(csv.DictReader(h))
        status = run/"status.json"
        out.append(dict(run=run.name,state=read(status)["status"] if status.exists() else "queued",
                        rows=len(rows),complete_series=sum(x["score_kind"]=="logit" for x in rows),
                        age_seconds=round(time.time()-f.stat().st_mtime) if f.exists() else None))
    for x in out:
        if x["state"]=="running" and x["age_seconds"] is not None and x["age_seconds"]>=1800:
            x["health"]="stalled"
    record=dict(utc=datetime.now(timezone.utc).isoformat(),event=label,progress=out)
    with (here/"real700_health.jsonl").open("a") as h: h.write(json.dumps(record)+"\n")
    print(record,flush=True)
next_check=time.monotonic()+1800
for name,run in zip(ids,runs):
    src=v9/"runs"/name
    with (here/(run.name+".log")).open("x") as out:
        p=subprocess.Popen([sys.executable,"-u",str(v9/"run_experiment.py"),"--config",str(src/"config.json"),"--checkpoint",str(src/"checkpoints/best.pth"),"--mode","eval","--eval-real-only","--parallel","data_parallel","--gpu-ids","0","1","--run-id",run.name],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
        while True:
            try:
                rc=p.wait(timeout=max(.1,next_check-time.monotonic()))
                break
            except subprocess.TimeoutExpired:
                snapshot("30min")
                next_check+=1800
        snapshot("exit")
        if rc: raise SystemExit(rc)
man=[read(r/"evaluation_manifest.json") for r in runs]
summ=[read(r/"metrics/eval.json") for r in runs]
for m,e,k in zip(man,summ,[19,23]):
    assert m["checkpoint_epoch"]==k and m["synthetic_selected"]==0 and m["real_selected"]==700
    assert e["status"]=="completed" and set(e["summaries"])=={"real"}
for key in ["protocol","real_files","source_sha256","dependency_versions"]:
    assert man[0][key]==man[1][key]
sets=[]
for r in runs:
    with (r/"metrics/per_series.csv").open() as h: rows=list(csv.DictReader(h))
    identity={(x["domain"],x["group"],x["series_id"],x["score_kind"],x["points"],x["anomaly_points"],x["train_index"]) for x in rows}
    assert len(identity)==len(rows)==1400 and all(x[0]=="real" for x in identity)
    sets.append(identity)
assert sets[0]==sets[1]
keys=["Affiliation-F","F1_T","Standard-F1","VUS-PR"]
s=[e["summaries"]["real"]["probability"] for e in summ]
hdr="| Model | "+" | ".join(keys)+" |"
sep="|---|---:|---:|---:|---:|"
def row(label,values): return "| "+label+" | "+" | ".join(f"{x:.2f}" for x in values)+" |"
table=[hdr,sep]+[row(label,[100*y["group_macro"][k] for k in keys]) for label,y in zip(["P0 stable","P1 stable"],s)]
table += [row("P1 minus P0",[100*(s[1]["group_macro"][k]-s[0]["group_macro"][k]) for k in keys])]
report=["# Stable pair: real data only","","700 paired series, 11 equally weighted groups; probability scores, percent. Best epochs 19 / 23.","",*table,"","Per-dataset P1 minus P0 (percentage points):","",hdr.replace("Model","Dataset"),sep]
for g in sorted(s[0]["group_means"]):
    report.append(row(g,[100*(s[1]["group_means"][g][k]-s[0]["group_means"][g][k]) for k in keys]))
report += ["","Source runs:"]+[f"- [{r.name}](../../runs/{r.name}/)" for r in runs]
(here/"REAL700_RESULTS.md").write_text("\n".join(report)+"\n")
path=v9/"RESULTS.md"
doc=path.read_text()
anchor="| P0 ignore | 80.98 | 43.21 | 40.63 | 46.40 |"
assert doc.count(anchor)==1 and table[2] not in doc
doc=doc.replace(anchor,anchor+"\n"+"\n".join(table[2:4])).replace("稳定版 P0/P1 的同协议检测结果待登记。","稳定版 P0/P1 仅评测全部真实序列，见[配对差值及分项](audits/stability_20261117/REAL700_RESULTS.md)。")
path.write_text(doc)
print("\n".join(report),flush=True)
