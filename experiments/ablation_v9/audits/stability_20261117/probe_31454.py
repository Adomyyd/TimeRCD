from pathlib import Path
import sys, subprocess, os, signal, json, time
ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__' and sys.argv[1:] == ['serial']:
    sys.path[:0] = [str(ROOT),str(ROOT.parents[1])]
    import numpy as np
    from evaluation.basic_metrics import basic_metricor
    file = ROOT/'runs/p0_official_stable_20261117_130401_eval_20261120_s200_r700/scores/synthetic_31454.npz'
    with np.load(file) as d:
        start=time.monotonic()
        rows=basic_metricor._process_pointf1pa_chunk((np.arange(.7,.99,.001),d['label'].astype(int),d['anomaly_score'].astype(float)))
        print('SERIAL',len(rows),round(time.monotonic()-start,3),max(x['F1_PA'] for x in rows))
elif __name__ == '__main__' and sys.argv[1:] == ['child']:
    sys.path[:0] = [str(ROOT),str(ROOT.parents[1])]
    import numpy as np
    from src.metric_worker import OfficialMetricWorker
    from src.benchmark_eval import load_protocol
    file = ROOT/'runs/p0_official_stable_20261117_130401_eval_20261120_s200_r700/scores/synthetic_31454.npz'
    with np.load(file) as d, OfficialMetricWorker(load_protocol()) as w:
        for i in range(3):
            start=time.monotonic()
            m=w.compute(d['anomaly_score'],d['label'],125)
            print('PROBE',i,round(time.monotonic()-start,3),len(m),m['PA-F1'],flush=True)
elif __name__ == '__main__':
    with Path(__file__).with_suffix('.log').open('w') as f:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'child'],stdout=f,stderr=f,start_new_session=True)
        try: print('exit',p.wait(timeout=45))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();print('timed out')
    print([x for x in Path(__file__).with_suffix('.log').read_text().splitlines() if x.startswith('PROBE') or 'Error:' in x])
