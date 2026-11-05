import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
from torch import nn

from experiments.common.artifacts import append_row, create_run, sha256, write_json
from experiments.common.config import ROOT
from experiments.stage2.masking import reference_masks
from experiments.stage2_patch_types.selections import patch_types
from experiments.stage2_paths.attention import attention_edges
from time_rcd._core.ts_encoder_bi_bias import TimeSeriesEncoder


def boundary_cases():
    for points, expected in [([16], [0]), ([17], [1]), ([16, 17], [0, 1])]:
        labels = np.zeros(5000, dtype=np.int64)
        labels[np.array(points) - 1] = 1
        values = np.arange(5000, dtype=float)[:, None] + 10000 * labels[:, None]
        yield points, expected, values, labels


def audit(directory):
    torch.manual_seed(2024)
    torch.set_num_threads(1)
    model = nn.Module()
    model.ts_encoder = TimeSeriesEncoder(d_model=8, d_proj=4, patch_size=16, num_layers=2, num_heads=2,
                                        d_ff_dropout=0., num_features=1).eval()
    model.eval()
    checks, max_abs = 0, 0.

    def check(name, passed, error=0.):
        nonlocal checks, max_abs
        checks += 1
        max_abs = max(max_abs, error)
        append_row(directory / "checks.csv", {"test": name, "passed": bool(passed), "max_abs": error})
        if not passed:
            raise RuntimeError(f"Boundary mapping failed: {name}")

    for points, expected, values, labels in boundary_cases():
        name = "point_" + "_".join(map(str, points))
        valid = np.ones(5000, dtype=bool)
        types = patch_types(labels, valid, 16)
        present, clean, mixed = reference_masks(labels, valid, 16)
        selected = present & ~clean
        for kind, actual in (("reference_selection", selected), ("U", types["U"]), ("M", types["M"]), ("mixed", mixed)):
            check(f"{name}_{kind}", np.flatnonzero(actual).tolist() == expected)
        check(f"{name}_P_empty", not types["P"].any())
        check(f"{name}_valid_patch_count", int(present.sum()) == 313 and present.all())
        check(f"{name}_tail_valid_points", types["valid_counts"][-1] == 8)
        captured = []

        def capture(module, args):
            captured.append(args[0].detach().numpy().copy())

        handle = model.ts_encoder.embedding_layer.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode(), attention_edges(model, present, selected, "C", "all") as records:
                output = model.ts_encoder(torch.tensor(values[None], dtype=torch.float32), torch.tensor(valid[None]))
        finally:
            handle.remove()
        check(f"{name}_embedding_calls", len(captured) == 1)
        patches = captured[0]
        manual = np.pad(values[:, 0], (0, 8)).reshape(1, 313, 16)
        error = float(np.abs(patches - manual).max())
        check(f"{name}_actual_patch_values", np.array_equal(patches, manual), error)
        check(f"{name}_point16_slot", patches[0, 0, 15] == values[15, 0])
        check(f"{name}_point17_slot", patches[0, 1, 0] == values[16, 0])
        check(f"{name}_outputs_finite_aligned", tuple(output.shape) == (1, 5000, 1, 4) and bool(torch.isfinite(output).all()))
        allowed = np.ones((313, 313), dtype=bool)
        for index in expected:
            allowed[:, index] = False
            allowed[index, index] = True
        check(f"{name}_two_layers", len(records) == 2)
        for record in records:
            check(f"{name}_layer{record['layer']}_selected_mask",
                  record["mask_sha256"] == hashlib.sha256(allowed.tobytes()).hexdigest())
            append_row(directory / "layer_masks.csv", {"case": name, **record})
        np.savez_compressed(directory / f"arrays/{name}.npz", values=values, labels=labels, patches=patches,
                            selected=selected, allowed=allowed, valid_counts=types["valid_counts"])
    summary = {"checks_passed": checks, "max_abs": max_abs, "cases": 3, "mapping_correct": True,
               "official_model_loads": 0, "official_model_forwards": 0, "tiny_encoder_forwards": 3,
               "point16_patch": 0, "point17_patch": 1}
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-run", type=Path, required=True)
    args = parser.parse_args()
    if json.loads((args.micro_run / "manifest.json").read_text())["status"] != "completed":
        raise RuntimeError("Micro audit must pass first")
    parent = args.micro_run / "summary.json"
    config = {"name": "stage2_isolated_boundary_v1", "output_root": str(ROOT / "experiments/runs"),
              "micro_run": str(args.micro_run.resolve()), "parent_sha256": sha256(parent),
              "device": "cpu", "seed": 2024, "patch_size": 16, "point_coordinate": "1-based"}
    directory, manifest = create_run(config, parent)
    print(f"Run directory: {directory}", flush=True)
    try:
        shutil.copyfile(ROOT / "experiments/stage2_isolated/protocols/boundary_mapping_v1.md", directory / "protocol.md")
        write_json(directory / "config.resolved.json", config)
        fingerprints = {}
        for relative in ("experiments/stage2_isolated/boundary_audit.py", "experiments/stage2/masking.py",
                         "experiments/stage2_patch_types/selections.py", "experiments/stage2_paths/attention.py",
                         "time_rcd/_core/ts_encoder_bi_bias.py"):
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            fingerprints[relative] = sha256(target)
        write_json(directory / "source.sha256.json", fingerprints)
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        print(json.dumps(audit(directory)))
        if sha256(parent) != config["parent_sha256"]:
            raise RuntimeError("Micro parent changed")
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        (directory / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)


if __name__ == "__main__":
    main()
