"""Epoch-boundary continuation and bounded checkpoint retention.

Training remains in train.py; this module only validates/restores saved state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .parallel import unwrap_model


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resume_config(previous, current):
    def comparable(config):
        value = copy.deepcopy(config)
        value.pop("diagnostics", None)
        for field in ("epochs", "log_freq"):
            value["optimization"].pop(field, None)
        return value
    if comparable(previous) != comparable(current):
        raise ValueError("Resume requires identical data/split/model/input/loss/optimizer/sampler/runtime; "
                         "only epochs, log_freq and diagnostics may differ")


def restore_rng(state):
    states = state["torch_cuda"]
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(states) != count:
        raise ValueError(f"Saved CUDA RNG count {len(states)} differs from current {count}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if states:
        torch.cuda.set_rng_state_all([item.cpu() for item in states])


def checkpoint_training_state(epoch, validation_loss, best_loss, best_epoch, stale):
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }
    improved = validation_loss < best_loss
    return {
        "next_epoch": epoch + 1,
        "best_loss": validation_loss if improved else best_loss,
        "best_epoch": epoch if improved else best_epoch,
        "stale": 0 if improved else stale + 1,
        "rng": state,
    }


def restore_training(checkpoint_path, model, optimizer, config, run_dir):
    checkpoint_path = Path(checkpoint_path).resolve()
    source_run = checkpoint_path.parent.parent
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    verify_resume_config(payload["config"], config)
    epoch = int(payload["epoch"])
    start_epoch = epoch + 1
    if start_epoch >= int(config["optimization"]["epochs"]):
        raise ValueError("epochs is the TOTAL epoch limit and must exceed the resumed next epoch")
    state = payload.get("training_state")
    rng_path = None
    if state is None:
        # Compatibility with the diagnostic run created before resume existed.
        # Its epoch RNG was captured after validation and before best serialization.
        rng_path = source_run / "diagnostics" / f"rng_after_epoch_{epoch:04d}.pth"
        if not rng_path.is_file():
            raise ValueError("Checkpoint has no complete RNG; matching epoch diagnostic RNG is required")
        history = {
            int(e): float(loss) for e, loss in re.findall(
                r"^epoch=(\d+) train_loss=\S+ validation_loss=(\S+)",
                (source_run / "train.log").read_text(), re.M
            )
        }
        if set(range(epoch + 1)) - set(history):
            raise ValueError("Legacy resume requires complete epoch history for early stopping")
        best_payload = torch.load(source_run / "checkpoints/best.pth",
                                  map_location="cpu", weights_only=False)
        best_epoch = int(best_payload["epoch"])
        if best_epoch > epoch:
            raise ValueError("Legacy resume requires the latest completed source checkpoint")
        best_loss = float(best_payload["validation"]["valid_total"])
        # Verify the history-selected best agrees despite its eight-decimal rounding.
        if abs(history[best_epoch] - min(history[e] for e in range(epoch + 1))) > 1e-7:
            raise ValueError("Source best checkpoint conflicts with validation history")
        state = {
            "next_epoch": start_epoch, "best_loss": best_loss, "best_epoch": best_epoch,
            "stale": epoch - best_epoch,
            "rng": torch.load(rng_path, map_location="cpu", weights_only=False),
        }
        del best_payload
    if int(state["next_epoch"]) != start_epoch:
        raise ValueError("Checkpoint epoch and training_state.next_epoch disagree")
    if not math.isfinite(float(state["best_loss"])):
        raise ValueError("Resume requires a finite best validation loss")
    if int(state["stale"]) >= int(config["optimization"]["early_stopping_patience"]):
        raise ValueError("Source already reached early stopping; continuation would change that rule")
    source_best = source_run / "checkpoints/best.pth"
    best_payload = torch.load(source_best, map_location="cpu", weights_only=False)
    if int(best_payload["epoch"]) != int(state["best_epoch"]):
        raise ValueError("Source best.pth does not match saved early-stopping state")
    del best_payload
    unwrap_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    destination = Path(run_dir) / "checkpoints/best.pth"
    shutil.copyfile(source_best, destination)
    provenance = {
        "source_run": source_run.name, "source_checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "rng_source": "checkpoint.training_state.rng" if rng_path is None else rng_path.name,
        "rng_sha256": None if rng_path is None else _sha256(rng_path),
        "best_checkpoint_sha256": _sha256(source_best),
        "source_epoch": epoch, "next_epoch": start_epoch,
        "best_loss": state["best_loss"], "best_epoch": state["best_epoch"], "stale": state["stale"],
        "total_epoch_limit": int(config["optimization"]["epochs"]),
        "numerical_limit": "Restores RNG and optimizer; CUDA execution need not be bitwise identical.",
    }
    (Path(run_dir) / "resume.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return state


def prune_epoch_checkpoints(run_dir, keep_last):
    if keep_last is None:
        return
    run_dir = Path(run_dir)
    files = sorted((run_dir / "checkpoints").glob("epoch_*.pth"))
    retained = set(files[-int(keep_last):])
    event = run_dir / "diagnostics/first_event.json"
    if event.exists():
        epoch = int(json.loads(event.read_text())["epoch"])
        retained.add(run_dir / "checkpoints" / f"epoch_{epoch - 1:04d}.pth")
    for path in files:
        if path not in retained:
            size = path.stat().st_size
            path.unlink()
            with (run_dir / "checkpoint_retention.jsonl").open("a") as handle:
                handle.write(json.dumps({"removed": path.name, "bytes": size,
                                         "keep_last": keep_last}) + "\n")
