import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.common.artifacts import sha256
from experiments.stage2_alignment_controls.data import generate
from experiments.stage2_alignment_controls.inference import predict as legacy_predict
from experiments.stage2_depth_controls.attention import attention_depth
from experiments.stage2_depth_controls.inference import predict, run_case
from experiments.stage2_depth_controls.statistics import paired_effects
from experiments.stage2_p_positions.data import select_positions
from experiments.stage2_paths.attention import edge_mask, model_fingerprint
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.diagnostics import point_groups, readout
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig


def model_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    config = TimeRCDConfig()
    config.ts_config.d_model = 16
    config.ts_config.d_proj = 8
    config.ts_config.patch_size = 16
    config.ts_config.num_heads = 2
    config.ts_config.num_layers = 4
    return TimeSeriesPretrainModel(config).eval()


class DepthAttentionTests(unittest.TestCase):
    def test_masks_prefix_inputs_endpoints_and_empty_selection(self):
        model = model_fixture()
        values = torch.randn(80, 1)
        valid = torch.arange(80) < 65
        present = np.ones(5, dtype=bool)
        selected = np.array([False, True, False, True, False])
        fingerprint = model_fingerprint(model)
        baseline, original = predict(model, values, valid, present, selected, 0, "cpu", capture=True)
        for depth in (1, 2, 4):
            prediction, records = predict(model, values, valid, present, selected, depth, "cpu", capture=True)
            for index, record in enumerate(records):
                condition = "C" if index >= 4 - depth else "A"
                self.assertEqual(record["condition"], condition)
                self.assertEqual(record["mask_sha256"], hashlib.sha256(edge_mask(present, selected, condition).tobytes()).hexdigest())
                if index <= 4 - depth:
                    for key in ("input", "q", "k", "v"):
                        self.assertTrue(torch.equal(original[index]["capture"][key], record["capture"][key]))
            empty, _ = predict(model, values, valid, present, np.zeros_like(present), depth, "cpu")
            for kind in baseline:
                np.testing.assert_array_equal(empty[kind], baseline[kind])
            if depth in (1, 4):
                legacy, _ = legacy_predict(model, values, valid, present, selected, "C", "last" if depth == 1 else "all", "cpu")
                for kind in prediction:
                    np.testing.assert_array_equal(prediction[kind], legacy[kind])
        self.assertEqual(model_fingerprint(model), fingerprint)

    def test_invalid_depth_calls_and_exception_restore(self):
        model = model_fixture()
        present, selected = np.ones(3, dtype=bool), np.array([True, False, False])
        for depth in (-1, 5, True, 1.5):
            with self.assertRaises(ValueError):
                with attention_depth(model, present, selected, depth):
                    pass
        with self.assertRaises(RuntimeError):
            with attention_depth(model, present, selected, 2):
                pass
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with attention_depth(model, present, selected, 2):
                raise RuntimeError("injected")
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in model.ts_encoder.transformer_encoder.layers))


class DepthStatisticsTests(unittest.TestCase):
    def frame(self):
        values = {"A": .1, "keep_first_d1": .2, "keep_last_d1": .3,
                  "keep_first_d2": .4, "keep_last_d2": .2, "keep_first_d4": .5, "keep_last_d4": .8}
        return pd.DataFrame([{"case": "one", "seed": 1, "phase": 0, "shape": "positive", "event_length": 32,
                              "offset": 0, "score_kind": "scores", "variant": v, "metric_status": "ok",
                              "average_precision": value, "I_rank_mean": np.nan} for v, value in values.items()])

    def test_adjacent_position_interactions_and_telescoping(self):
        effects, _ = paired_effects(self.frame().sample(frac=1, random_state=2), [1, 2, 4])
        indexed = effects.set_index("contrast")
        self.assertAlmostEqual(indexed.loc["position_d2_minus_d1", "average_precision"], -.3)
        self.assertAlmostEqual(indexed.loc["position_d4_minus_d2", "average_precision"], .5)
        self.assertAlmostEqual(indexed.loc["position_endpoint_interaction", "average_precision"], .2)
        self.assertTrue(effects.I_rank_mean.isna().all())

    def test_missing_duplicate_and_invalid_depths_fail(self):
        frame = self.frame()
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                paired_effects(bad, [1, 2, 4])
        with self.assertRaises(ValueError):
            paired_effects(frame, [2, 1, 4])


class DepthPipelineTests(unittest.TestCase):
    def test_miniature_gate_full_readouts_parent_and_state(self):
        model = model_fixture()
        fingerprint = model_fingerprint(model)
        source = json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())
        settings = {**source["synthetic"], "length": 512, "anomaly_count": 128, "seeds": [1],
                    "phases": [0], "event_lengths": [32], "shapes": ["positive"]}
        meta, arrays = next(generate(settings, 16))[0]
        config = json.loads(Path("experiments/configs/stage2_depth_controls_v1.json").read_text())
        config.update(window=512, expected_layers=4, depths=[1, 2, 4])
        masks, kept = select_positions(meta, arrays, 16, config["random_seeds"])
        fixed, roles = point_groups(meta, arrays, masks, 16, config["random_seeds"])
        values, valid = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32), torch.ones(512, dtype=torch.bool)
        old, rows = {}, []
        for variant, selector, condition, scope in [("A", "A", "A", "all")] + [
                (f"{selector}_{scope}", selector, "C", scope) for selector in ("keep_first", "keep_last") for scope in ("last", "all")]:
            pred, _ = legacy_predict(model, values, valid, arrays["type_valid"], masks["full" if selector == "A" else selector], condition, scope, "cpu")
            for kind, scores in pred.items():
                old[f"{variant}_{kind}"] = scores
                result = readout(arrays["labels"], scores, {**fixed, **roles[selector]}, config["recalls"])
                rows.append({"case": meta["case"], "variant": variant, "score_kind": kind, **result})
        parent_metrics = pd.DataFrame(rows).set_index(["case", "variant", "score_kind"])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "arrays").mkdir()
            parent = directory / "parent"
            (parent / "arrays").mkdir(parents=True)
            config["parent_run"] = str(parent)
            np.savez_compressed(parent / f"arrays/{meta['case']}.npz", **old)
            for kind, data in (("input", arrays), ("selection", {**masks, **{f"kept_{k}": v for k, v in kept.items()}})):
                path = directory / f"arrays/{kind}.npz"
                np.savez_compressed(path, **data)
                meta.update({f"{kind}_snapshot": str(path.relative_to(directory)), f"{kind}_sha256": sha256(path)})
            for gate in (True, False):
                output = directory / ("gate" if gate else "full")
                (output / "arrays").mkdir(parents=True)
                (output / "diagnostics").mkdir()
                audit = Audit(output / "diagnostics", 1e-12)
                self.assertEqual(run_case(directory, output, config, model, "cpu", meta, parent_metrics, audit, gate=gate), 7)
                self.assertTrue(pd.read_csv(output / "checks.csv").close.all())
                self.assertTrue(pd.read_csv(output / "diagnostics/checks.csv").close.all())
                self.assertEqual(len(pd.read_csv(output / "metrics.csv")), 14)
        self.assertEqual(model_fingerprint(model), fingerprint)


if __name__ == "__main__":
    unittest.main()
