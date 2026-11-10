"""Bounded synthetic regression and entry checks; never starts formal training."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/diskA/YYD/TimeRCD")
OUT = ROOT / "experiments/_archive/20261110_cleanup"
PYTHON = "/home/omnisky/anaconda3/envs/time-rcd/bin/python3"
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
results = {}
for route in ("original_400k", "ablation_v9"):
    cmd = [PYTHON, "-B", str(ROOT / "experiments" / route / "run_experiment.py"), "--help"]
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    results[route] = {"command": cmd, "returncode": completed.returncode,
                      "stdout": completed.stdout, "stderr": completed.stderr}
(OUT / "entry_help.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
if any(item["returncode"] for item in results.values()):
    print(json.dumps(results, ensure_ascii=False))
    raise SystemExit(1)
cmd = [PYTHON, "-B", str(ROOT / "experiments/ablation_v9/tests/run_checks.py"),
       "-p", "no:cacheprovider", "--junitxml", str(OUT / "pytest_remote.xml")]
completed = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
(OUT / "pytest_remote.log").write_text(completed.stdout + completed.stderr)
results["regression"] = {"command": cmd, "returncode": completed.returncode}
(OUT / "validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print("Both experiment entries: --help passed")
print(completed.stdout)
print(completed.stderr)
raise SystemExit(completed.returncode)
