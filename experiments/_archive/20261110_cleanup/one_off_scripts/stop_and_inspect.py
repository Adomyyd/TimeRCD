"""One-shot, PID-scoped stop and log audit; does not load data or weights."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time
from datetime import datetime, timezone

ROOT = Path("/diskA/YYD/TimeRCD")
RUN = ROOT / "experiments/original_400k/runs/time_rcd_400k_20261108"
AUDIT = Path(__file__).resolve().parent
PID = 408588

def identity():
    proc = Path(f"/proc/{PID}")
    try:
        stat = (proc / "stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        return {"state": tail[0], "start_ticks": tail[19],
                "argv": (proc / "cmdline").read_bytes().decode().strip("\0").split("\0"),
                "cwd": str((proc / "cwd").resolve())}
    except (FileNotFoundError, ProcessLookupError):
        return None

parser = argparse.ArgumentParser()
parser.add_argument("--stop", action="store_true")
args = parser.parse_args()
before = identity()
print(json.dumps({"pid": PID, "identity": before}, ensure_ascii=False), flush=True)
events = []
if args.stop and before and before["state"] != "Z":
    allowed = {str(ROOT / "experiments/train_400k.py"),
               str(ROOT / "experiments/original_400k/src/train_400k.py")}
    if not any(arg in allowed for arg in before["argv"]):
        raise RuntimeError("PID command is not the expected 400k trainer; no signal sent")
    if not any("time-rcd/bin/python" in arg for arg in before["argv"][:1]):
        raise RuntimeError("Unexpected interpreter; no signal sent")
    current = identity()
    if current != before:
        raise RuntimeError("Process identity changed; no signal sent")
    os.kill(PID, signal.SIGTERM)
    events.append("SIGTERM")
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        current = identity()
        if not current or current["state"] == "Z":
            break
        time.sleep(.2)
    current = identity()
    if current and current["state"] != "Z" and current["start_ticks"] == before["start_ticks"]:
        os.kill(PID, signal.SIGKILL)
        events.append("SIGKILL")
        time.sleep(1)
after = identity()
summary = {"time_utc": datetime.now(timezone.utc).isoformat(),
           "pid": PID, "before": before, "signals": events, "after": after,
           "checkpoints": [], "logs": {}}
for path in sorted((RUN / "checkpoints").glob("*")):
    st = path.stat()
    summary["checkpoints"].append({"name": path.name, "bytes": st.st_size,
                                   "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()})
pattern = re.compile(r"Epoch (\d+), Batch (\d+)/(\d+)\s+Total Loss: ([\d.eE+\-]+)\s+Recon Loss: ([\d.eE+\-]+)\s+Anomaly Loss: ([\d.eE+\-]+)")
for name in ("resume_console.log", "train.log"):
    text = (RUN / name).read_text(errors="replace")
    rows = [{"epoch": int(m[1]), "batch": int(m[2]), "batches": int(m[3]),
             "total": float(m[4]), "reconstruction": float(m[5]), "classification": float(m[6]),
             "line": text.count("\n", 0, m.start()) + 1} for m in pattern.finditer(text)]
    averages = [{"epoch": int(m[1]), "total": float(m[2]), "reconstruction": float(m[3]),
                 "classification": float(m[4]), "line": text.count("\n", 0, m.start()) + 1}
                for m in re.finditer(r"Epoch (\d+) completed:\s+Average Total Loss: ([\d.eE+\-]+)\s+Average Recon Loss: ([\d.eE+\-]+)\s+Average Anomaly Loss: ([\d.eE+\-]+)", text)]
    summary["logs"][name] = {"bytes": (RUN / name).stat().st_size, "averages": averages,
                             "epoch11_top_logged": sorted([r for r in rows if r["epoch"] == 11],
                                                          key=lambda r: r["total"], reverse=True)[:12],
                             "last_logged_batch": rows[-1] if rows else None,
                             "tail": text.splitlines()[-15:]}
if args.stop:
    (AUDIT / "stop_and_log_evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if args.stop and after and after["state"] != "Z" and before and after["start_ticks"] == before["start_ticks"]:
    raise RuntimeError("Target process still alive")
