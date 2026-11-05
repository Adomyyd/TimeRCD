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
from experiments.common.config import ROOT, project_path
from experiments.stage2_paths.attention import attention_edges, edge_attention, edge_mask


class TinyAttention(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.num_heads, self.head_dim = 1, 2
        weights = ([[[.7, .2], [-.1, .6]], [[.4, -.3], [.5, .8]], [[.9, .1], [-.2, .7]], [[.6, .1], [.2, .5]]],
                   [[[.5, -.1], [.3, .9]], [[.8, .2], [-.4, .6]], [[.7, -.2], [.1, .8]], [[.9, -.1], [.1, .6]]])
        for name, weight in zip(("q_proj", "k_proj", "v_proj", "out_proj"), weights[layer]):
            module = nn.Linear(2, 2, bias=False, dtype=torch.float64)
            module.weight.data.copy_(torch.tensor(weight, dtype=torch.float64))
            setattr(self, name, module)

    def apply_rope(self, x, freqs):
        return x

    def forward(self, query, key, value, freqs, attn_mask=None):
        allowed = attn_mask[:, None, None, :].expand(1, 1, 3, 3)
        return edge_attention(self, query, key, value, freqs, allowed)


class TinyLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attn = TinyAttention(index)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts_encoder = nn.Module()
        self.ts_encoder.num_features, self.ts_encoder.use_rope = 1, True
        self.ts_encoder.transformer_encoder = nn.Module()
        self.ts_encoder.transformer_encoder.layers = nn.ModuleList([TinyLayer(i) for i in range(2)])

    def forward(self, x, valid):
        outputs = []
        for layer in self.ts_encoder.transformer_encoder.layers:
            x = x + layer.self_attn(x, x, x, None, attn_mask=valid)
            outputs.append(x.detach().numpy()[0].copy())
        return outputs


def manual_layer(x, module, allowed):
    weights = [getattr(module, name).weight.detach().numpy() for name in ("q_proj", "k_proj", "v_proj", "out_proj")]
    q, k, v = [x @ weight.T for weight in weights[:3]]
    attention = np.zeros((3, 3))
    output = np.zeros_like(x)
    for i in range(3):
        columns = [j for j in range(3) if allowed[i, j]]
        logits = np.array([sum(q[i, d] * k[j, d] for d in range(2)) / np.sqrt(2) for j in columns])
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        for j, probability in zip(columns, probabilities):
            attention[i, j] = probability
            for d in range(2):
                output[i, d] += probability * v[j, d]
    return x + output @ weights[3].T, attention


def audit(directory):
    model = TinyModel().eval()
    valid, selected = np.ones(3, dtype=bool), np.array([False, True, False])
    matrices = {"A": np.ones((3, 3), dtype=bool), "C": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
                "D": np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool)}
    x = np.array([[.4, -.7], [1.3, .2], [-.5, .8]])
    arrays, outputs, checks = {"input": x}, {}, []

    def check(name, expected, actual, exact=False):
        expected, actual = np.asarray(expected), np.asarray(actual)
        error = float(np.abs(expected.astype(float) - actual.astype(float)).max())
        passed = np.array_equal(expected, actual) if exact else error <= 1e-12
        row = {"test": name, "passed": bool(passed), "max_abs": error}
        append_row(directory / "checks.csv", row)
        checks.append(row)
        if not passed:
            raise RuntimeError(f"Micro mask audit failed: {name}")

    for condition, allowed in matrices.items():
        check(f"{condition}_mask", allowed, edge_mask(valid, selected, condition), exact=True)
        with torch.inference_mode(), attention_edges(model, valid, selected, condition, "all") as records:
            actual = model(torch.tensor(x[None]), torch.tensor(valid[None]))
        current = x.copy()
        arrays[f"{condition}_mask"] = allowed
        for layer, module in enumerate(model.ts_encoder.transformer_encoder.layers):
            current, probabilities = manual_layer(current, module.self_attn, allowed)
            check(f"{condition}_layer{layer}_output", current, actual[layer])
            check(f"{condition}_layer{layer}_mask_sha256", True,
                  records[layer]["mask_sha256"] == hashlib.sha256(allowed.tobytes()).hexdigest(), exact=True)
            arrays[f"{condition}_layer{layer}_manual"] = current.copy()
            arrays[f"{condition}_layer{layer}_actual"] = actual[layer]
            arrays[f"{condition}_layer{layer}_attention"] = probabilities
        outputs[condition] = actual
        with torch.inference_mode(), attention_edges(model, valid, selected, condition, "all"):
            changed = x.copy()
            changed[1] += [2.3, -.8]
            changed_outputs = model(torch.tensor(changed[None]), torch.tensor(valid[None]))
        if condition in "CD":
            for layer in range(2):
                check(f"{condition}_layer{layer}_perturb_nonselected", actual[layer][[0, 2]], changed_outputs[layer][[0, 2]])
    for layer in range(2):
        check(f"CD_layer{layer}_nonselected", outputs["C"][layer][[0, 2]], outputs["D"][layer][[0, 2]])
    for selection in ([True, False, False], [False, True, False], [False, False, True], [True, False, True]):
        expected = np.ones((3, 3), dtype=bool)
        for i in range(3):
            for j in range(3):
                if selection[j] and i != j:
                    expected[i, j] = False
        check(f"C_selection_{selection}", expected, edge_mask(valid, np.array(selection), "C"), exact=True)
    np.savez_compressed(directory / "arrays/micro.npz", **arrays)
    summary = {"checks_passed": len(checks), "max_abs": max(r["max_abs"] for r in checks), "layers": 2, "tokens": 3,
               "official_model_loads": 0, "official_model_forwards": 0, "tiny_forwards": 6, "C_diagonal_correct": True}
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose-run", required=True, type=Path)
    args = parser.parse_args()
    if json.loads((args.dose_run / "manifest.json").read_text())["status"] != "completed":
        raise RuntimeError("Dose diagnostic must complete first")
    config_path = args.dose_run / "summary.json"
    config = {"name": "stage2_isolated_micro_v1", "output_root": str(ROOT / "experiments/runs"),
              "dose_run": str(args.dose_run.resolve()), "dose_summary_sha256": sha256(config_path),
              "dtype": "float64", "device": "cpu", "atol": 1e-12}
    directory, manifest = create_run(config, config_path)
    print(f"Run directory: {directory}", flush=True)
    try:
        protocol = project_path("experiments/stage2_isolated/protocols/micro_mask_v1.md")
        shutil.copyfile(protocol, directory / "protocol.md")
        fingerprints = {}
        for path in [Path(__file__), ROOT / "experiments/stage2_paths/attention.py"]:
            target = directory / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            fingerprints[str(path.relative_to(ROOT))] = sha256(target)
        write_json(directory / "source.sha256.json", fingerprints)
        write_json(directory / "config.resolved.json", config)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        print(json.dumps(audit(directory)))
        if sha256(config_path) != config["dose_summary_sha256"]:
            raise RuntimeError("Dose parent changed")
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
