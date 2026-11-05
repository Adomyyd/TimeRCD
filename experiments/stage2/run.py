import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from experiments.common.artifacts import capture_provenance, create_run, sha256, write_json
from experiments.common.config import load_config, project_path
from experiments.stage1.run import configure_runtime


def run_cases(directory, config):
    import numpy as np
    from experiments.common.data import load_series
    from experiments.stage1.inference import make_detector
    from .inference import run_case
    from .synthetic import generate

    real = list(load_series(config["data"], config["seed"]))
    synthetic = list(generate(config["synthetic"]))
    series_list = real + synthetic
    sources = []
    for index, series in enumerate(series_list):
        relative = f"arrays/input_s{index:04d}.npz"
        np.savez_compressed(directory / relative, values=series.values, labels=series.labels)
        sources.append({"index": index, "name": series.name, "source": series.source,
                        "input_snapshot": relative, "input_sha256": sha256(directory / relative)})
    write_json(directory / "data_manifest.json", sources)
    checks = []
    for window in sorted(set(config["real_windows"] + config["synthetic_windows"])):
        detector = make_detector(config, window)
        for index, series in enumerate(series_list):
            allowed = config["real_windows"] if index < len(real) else config["synthetic_windows"]
            if window not in allowed:
                continue
            print(f"series={series.name} window={window}", flush=True)
            checks.append(run_case(directory, config, detector, series, index, window))
        del detector
    return {"cases": len(checks), "failed": len(checks) - sum(checks),
            "real_series": len(real), "synthetic_series": len(synthetic)}


def main():
    parser = argparse.ArgumentParser(description="Stage 2: all-layer clean-reference Oracle")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    config = load_config(project_path(settings["base_config"]))
    config.update(settings)
    for key in ("random_seeds", "real_windows", "synthetic_windows"):
        values = config[key]
        if not values or len(values) != len(set(values)) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError(f"Invalid {key}")
        if key != "random_seeds" and 0 in values:
            raise ValueError("Windows must be positive")
    config["batch_size"] = 1
    directory, manifest = create_run(config, args.config)
    print(f"Run directory: {directory}", flush=True)
    try:
        manifest["runtime"] = configure_runtime(config)
        capture_provenance(directory, manifest)
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            summary = run_cases(directory, config)
        write_json(directory / "summary.json", summary)
        manifest["status"] = "completed" if summary["failed"] == 0 else "checks_failed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)
    print(json.dumps(summary))
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
