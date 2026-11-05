import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import traceback

from experiments.common.artifacts import append_row, capture_provenance, create_run, write_json
from experiments.common.config import load_config


def configure_runtime(config):
    # The official loader selects CUDA before honoring the detector's device.
    # Hide GPUs before importing torch to make CPU runs genuinely CPU-only.
    if config["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    return {
        "device": config["device"], "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "cuda_version": torch.version.cuda, "threads": torch.get_num_threads(),
        "deterministic_algorithms": True, "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "gpu": torch.cuda.get_device_name(0) if config["device"] == "cuda" else None,
    }


def record_case(directory, config, detector, series, series_index, window):
    import numpy as np
    from experiments.common.data import geometry
    from experiments.common.metrics import detection_metrics, difference
    from .inference import compare_backends, probe_tail, probe_short
    from .effects import assemble_conditions, record_effects

    arrays, raw_length = compare_backends(detector, series.values)
    tails, valid = probe_tail(detector, series.values)
    arrays.update(tails)
    case = f"s{series_index:04d}_w{window}"
    base = {"case": case, "series": series.name, **geometry(len(series.values), window)}
    conditions = assemble_conditions(arrays, valid, valid < base["actual_window"])
    if config.get("force_short_padding", False):
        conditions.update(probe_short(detector, series.values))
    for name, prediction in conditions.items():
        arrays.update({f"{name}_{key}": value for key, value in prediction.items()})
    np.savez_compressed(directory / "arrays" / f"{case}.npz", **arrays)
    record_effects(directory, config, base, series.labels, conditions)
    backend = {}
    for key in ("scores", "logits"):
        delta = difference(arrays[f"public_{key}"], arrays[f"legacy_{key}"], config["atol"], config["rtol"])
        backend.update({f"{key}_{k}": v for k, v in delta.items()})
    append_row(directory / "backend_parity.csv", {
        **base, "legacy_raw_length": raw_length, **backend,
    })
    append_row(directory / "metrics.csv", {
        **base, "anomaly_count": int(series.labels.sum()),
        "anomaly_ratio": float(series.labels.mean()),
        **detection_metrics(series.labels, arrays["public_scores"]),
    })
    passed = backend["scores_close"] and backend["logits_close"]
    for reference, candidate in (("compact", "zero"), ("compact", "current"), ("zero", "current")):
        deltas = {}
        for key in ("scores", "logits"):
            delta = difference(arrays[f"tail_{reference}_{key}"], arrays[f"tail_{candidate}_{key}"],
                               config["atol"], config["rtol"])
            deltas.update({f"{key}_{k}": v for k, v in delta.items()})
        append_row(directory / "padding.csv", {
            **base, "reference": reference, "candidate": candidate, **deltas,
        })
        if candidate == "zero":
            passed = passed and deltas["scores_close"] and deltas["logits_close"]
    # The tail probe must reproduce the corresponding segment of the full run.
    alignment = {}
    for key in ("scores", "logits"):
        delta = difference(arrays[f"public_{key}"][-valid:], arrays[f"tail_current_{key}"],
                           config["atol"], config["rtol"])
        alignment.update({f"{key}_{k}": v for k, v in delta.items()})
        passed = passed and delta["close"]
    append_row(directory / "tail_alignment.csv", {**base, **alignment})
    for variant in ("short_forced_zero", "short_forced_edge"):
        if variant not in conditions:
            continue
        deltas = {}
        for key in ("scores", "logits"):
            delta = difference(conditions["current"][key], conditions[variant][key], config["atol"], config["rtol"])
            deltas.update({f"{key}_{k}": value for k, value in delta.items()})
            if variant == "short_forced_zero":
                passed = passed and delta["close"]
        append_row(directory / "short_padding.csv", {
            **base, "variant": variant, "forced_input_length": window,
            "forced_total_patches": (window + 15) // 16, **deltas,
        })
    return bool(passed)


def run_cases(directory, config):
    import numpy as np
    from experiments.common.artifacts import sha256
    from experiments.common.data import load_series
    from .inference import make_detector

    sources = []
    series_list = list(load_series(config["data"], config["seed"]))
    for index, series in enumerate(series_list):
        relative = f"arrays/input_s{index:04d}.npz"
        np.savez_compressed(directory / relative, values=series.values, labels=series.labels)
        sources.append({"index": index, "name": series.name, "length": len(series.values),
                        "source": series.source, "input_snapshot": relative,
                        "input_sha256": sha256(directory / relative)})
    write_json(directory / "data_manifest.json", sources)
    checks = []
    for window in config["window_sizes"]:
        detector = make_detector(config, window)
        for index, series in enumerate(series_list):
            print(f"series={series.name} window={window}", flush=True)
            checks.append(record_case(directory, config, detector, series, index, window))
        del detector
    return {"cases": len(checks), "passed": sum(checks), "failed": len(checks) - sum(checks),
            "numerical_checks_passed": all(checks),
            "scope": "Implementation diagnostic; not a benchmark or mechanism conclusion"}


def main():
    parser = argparse.ArgumentParser(description="Stage 1: TimeRCD inference/padding diagnostics")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
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
        manifest["status"] = "completed" if summary["numerical_checks_passed"] else "checks_failed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)
    print(f"Cases: {summary['cases']}; numerical checks failed: {summary['failed']}")
    return 0 if summary["numerical_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
