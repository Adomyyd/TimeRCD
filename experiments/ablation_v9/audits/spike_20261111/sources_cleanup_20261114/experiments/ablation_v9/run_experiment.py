from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch

from src.audit import run_audit
from src.config import ExperimentConfig, load_config, validate_config
from src.models import build_model
from src.parallel import prepare_model, unwrap_model
from src.reporting import write_report
from src.train import benchmark, run_smoke, set_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TimeRCD v9 ablation experiments")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument("--run-id", required=True, help="Unique output directory name")
    parser.add_argument("--mode", choices=["audit", "smoke", "bench", "train", "eval", "report", "all"], required=True)
    parser.add_argument("--max-batches", type=int, default=300,
                        help="Batch limit for --mode bench (default 300)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override optimization.batch_size before the config snapshot")
    parser.add_argument("--checkpoint", help="Checkpoint path for eval mode")
    parser.add_argument("--eval-config", default=str(HERE / "configs" / "eval_official.json"),
                        help="Detection evaluation protocol (synthetic held-out + official TSB-AD-U)")
    parser.add_argument("--eval-max-synthetic", type=int, default=None,
                        help="Optional synthetic series limit; results are marked as a subset")
    parser.add_argument("--eval-max-real", type=int, default=None,
                        help="Optional real series limit; results are marked as a subset")
    parser.add_argument("--parallel", choices=["none", "data_parallel"], default=None,
                        help="Override runtime.parallel; default uses JSON config")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                        help="Visible CUDA device ids for data_parallel; first id gathers outputs")
    args = parser.parse_args()
    if args.mode == "eval" and not args.checkpoint:
        parser.error("--mode eval requires --checkpoint; use the training run's config.json")
    for name in ("eval_max_synthetic", "eval_max_real"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error("evaluation limits must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def environment_record(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": config.device,
        "parallel": config.raw["runtime"].get("parallel", "none"),
        "gpu_ids": config.raw["runtime"].get("gpu_ids"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpus": [
            {"id": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ],
        "git_revision": git_revision(),
    }


def append_registry(run_id: str, config: ExperimentConfig, run_dir: Path, status: str, mode: str) -> None:
    registry = HERE / "registry" / "experiments.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "mode": mode,
        "status": status,
        "run_dir": str(run_dir),
        "updated_utc": utc_now(),
    }
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", run_id):
        raise ValueError("run-id must be 3-128 characters: letters, digits, '.', '_' or '-'")
    run_dir = HERE / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def save_run_metadata(run_dir: Path, config: ExperimentConfig, command: list[str]) -> None:
    (run_dir / "metrics").mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "checkpoints").mkdir()
    write_json(run_dir / "config.json", config.as_dict())
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    write_json(run_dir / "environment.json", environment_record(config))


def load_model(config: ExperimentConfig) -> torch.nn.Module:
    set_seed(config.seed)
    model = build_model(config.raw)
    return prepare_model(model, config.raw)


def apply_runtime_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.parallel is not None:
        config.raw["runtime"]["parallel"] = args.parallel
        if args.parallel == "none":
            config.raw["runtime"]["gpu_ids"] = None
    if args.gpu_ids is not None:
        config.raw["runtime"]["gpu_ids"] = args.gpu_ids
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        config.raw["optimization"]["batch_size"] = int(batch_size)
    validate_config(config.raw)


def run_one(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    apply_runtime_overrides(config, args)
    run_dir = create_run_dir(args.run_id)
    save_run_metadata(run_dir, config, sys.argv)
    status = {"run_id": args.run_id, "mode": args.mode, "status": "running", "started_utc": utc_now()}
    write_json(run_dir / "status.json", status)
    append_registry(args.run_id, config, run_dir, "running", args.mode)

    try:
        audit = run_audit(config)
        write_json(run_dir / "data_manifest.json", audit["data_manifest"])
        write_json(run_dir / "audit.json", audit)

        if args.mode == "audit":
            status["status"] = "completed"
            status["completed_utc"] = utc_now()
            write_json(run_dir / "status.json", status)
            append_registry(args.run_id, config, run_dir, "completed", args.mode)
            return

        if args.mode == "smoke":
            model = load_model(config)
            smoke = run_smoke(model, config.raw)
            write_json(run_dir / "smoke.json", smoke)
            status["status"] = "completed"
            status["completed_utc"] = utc_now()
            write_json(run_dir / "status.json", status)
            append_registry(args.run_id, config, run_dir, "completed", args.mode)
            return

        if args.mode == "bench":
            model = load_model(config)
            benchmark(model, config.raw, run_dir, args.max_batches)
            status["status"] = "completed"
            status["completed_utc"] = utc_now()
            write_json(run_dir / "status.json", status)
            append_registry(args.run_id, config, run_dir, "completed", args.mode)
            return

        if args.mode in {"eval", "all"}:
            from src.benchmark_eval import DEFAULT_PROTOCOL, load_protocol, real_targets
            # Check evaluation dependencies before all-mode can start training.
            from evaluation.metrics import get_metrics, get_metrics_optimized
            from utils.slidingWindows import find_length_rank
            protocol = load_protocol(getattr(args, "eval_config", DEFAULT_PROTOCOL))
            real_targets(protocol["real_data_dir"])
            write_json(run_dir / "eval_config.json", protocol)

        if args.mode in {"train", "all"}:
            model = load_model(config)
            smoke = run_smoke(model, config.raw)
            write_json(run_dir / "smoke.json", smoke)
            result = train_model(model, config.raw, run_dir)
            write_json(run_dir / "metrics" / "train.json", result)

        if args.mode in {"eval", "all"}:
            from src.benchmark_eval import evaluate_benchmarks, verify_checkpoint_config
            checkpoint = Path(args.checkpoint) if args.mode == "eval" else run_dir / "checkpoints" / "best.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            if args.mode == "all":
                del model
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            verify_checkpoint_config(payload, config.raw)
            model = load_model(config)
            unwrap_model(model).load_state_dict(payload["model_state_dict"])
            checkpoint_epoch = int(payload["epoch"])
            del payload
            set_seed(protocol["seed"])
            torch.set_num_threads(protocol["cpu_threads"])
            evaluate_benchmarks(model, config.raw, run_dir, checkpoint, protocol,
                                checkpoint_epoch=checkpoint_epoch,
                                max_synthetic=getattr(args, "eval_max_synthetic", None),
                                max_real=getattr(args, "eval_max_real", None))

        if args.mode == "report":
            run_dirs = sorted(path for path in (HERE / "runs").iterdir() if path.is_dir())
            write_report(run_dirs, run_dir / "metrics" / "report.json")

        status["status"] = "completed"
        status["completed_utc"] = utc_now()
        write_json(run_dir / "status.json", status)
        append_registry(args.run_id, config, run_dir, "completed", args.mode)
    except BaseException as exc:
        final_status = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["status"] = final_status
        status["error"] = repr(exc)
        status["completed_utc"] = utc_now()
        write_json(run_dir / "status.json", status)
        append_registry(args.run_id, config, run_dir, final_status, args.mode)
        raise


if __name__ == "__main__":
    run_one(parse_args())
