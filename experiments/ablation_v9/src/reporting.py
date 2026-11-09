from __future__ import annotations

import json
from pathlib import Path


def write_report(run_dirs: list[Path], output_path: Path) -> None:
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics" / "eval.json"
        if metrics_path.is_file():
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
