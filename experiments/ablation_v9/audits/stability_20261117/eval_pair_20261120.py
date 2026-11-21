"""Evaluate the completed stable pair sequentially, then verify and tabulate it."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V9 = HERE.parents[1]
ROOT = V9.parents[1]
RUNS = V9 / 'runs'
os.chdir(ROOT)
for method in ('p0', 'p1'):
    name = f'{method}_official_stable_20261117_130401'
    source = RUNS / name
    assert json.loads((source/'status.json').read_text())['status'] == 'completed'
    run = name + '_eval_20261120_s200_r700'
    args = [sys.executable, '-u', str(V9/'run_experiment.py'), '--config', str(source/'config.json'), '--checkpoint', str(source/'checkpoints/best.pth'), '--eval-config', str(V9/'configs/eval_official.json'), '--run-id', run, '--parallel', 'data_parallel', '--gpu-ids', '0', '1', '--eval-max-synthetic', '200', '--mode', 'eval']
    print('Starting', run, flush=True)
    with (HERE/f'{run}.log').open('x') as stream:
        subprocess.run(args, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1'}, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert json.loads((RUNS/run/'status.json').read_text())['status'] == 'completed'
subprocess.run([sys.executable, str(HERE/'compare_eval_20261120.py')], check=True)
