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
from experiments.stage2_depth_controls.inference import predict as depth_predict
from experiments.stage2_layer_factorial.attention import attention_layers, switches
from experiments.stage2_layer_factorial.inference import predict, run_case
from experiments.stage2_layer_factorial.statistics import paired_effects
from experiments.stage2_p_positions.data import select_positions
from experiments.stage2_paths.attention import edge_mask, model_fingerprint
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.diagnostics import point_groups, readout
from experiments.tests.test_stage2_depth_controls import model_fixture


class LayerAttentionTests(unittest.TestCase):
    def test_masks_common_prefix_depth_endpoints_and_empty(self):
        model = model_fixture()
        values = torch.randn(80, 1)
        valid = torch.arange(80) < 65
        present = np.ones(5, dtype=bool)
        selected = np.array([False, True, False, True, False])
        fingerprint = model_fingerprint(model)
        baseline, original = predict(model, values, valid, present, selected, [], "cpu", capture=True)
        traces = {}
        for switch, active in switches([2, 3], 4).items():
            prediction, records = predict(model, values, valid, present, selected, active, "cpu", capture=True)
            traces[switch] = records
            for index, record in enumerate(records):
                condition = "C" if index in active else "A"
                self.assertEqual(record["condition"], condition)
                self.assertEqual(record["mask_sha256"], hashlib.sha256(edge_mask(present, selected, condition).tobytes()).hexdigest())
                if index <= min(active, default=4):
                    for key in ("input", "q", "k", "v"):
                        self.assertTrue(torch.equal(original[index]["capture"][key], record["capture"][key]))
            empty, _ = predict(model, values, valid, present, np.zeros_like(present), active, "cpu")
            for kind in baseline:
                np.testing.assert_array_equal(empty[kind], baseline[kind])
            if switch in ("00", "01", "11"):
                legacy, _ = depth_predict(model, values, valid, present, selected, len(active), "cpu")
                for kind in prediction:
                    np.testing.assert_array_equal(prediction[kind], legacy[kind])
        for index in (2, 3):
            for key in ("input", "q", "k", "v"):
                self.assertTrue(torch.equal(traces["10"][index]["capture"][key], traces["11"][index]["capture"][key]))
        self.assertEqual([a["mask_sha256"] == b["mask_sha256"] for a, b in zip(traces["10"], traces["11"])],
                         [True, True, True, False])
        self.assertEqual(model_fingerprint(model), fingerprint)

    def test_invalid_layers_calls_and_exception_restore(self):
        model = model_fixture()
        present, selected = np.ones(3, dtype=bool), np.array([True, False, False])
        for active in ([-1], [4], [True], [1.5], [2, 2]):
            with self.assertRaises(ValueError):
                with attention_layers(model, present, selected, active):
                    pass
        for layers in ([3, 2], [1, 3], [2, 2], [2.0, 3]):
            with self.assertRaises(ValueError):
                switches(layers, 4)
        with self.assertRaises(RuntimeError):
            with attention_layers(model, present, selected, [2]):
                pass
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with attention_layers(model, present, selected, [2, 3]):
                raise RuntimeError("injected")
        self.assertTrue(all("forward" not in layer.self_attn.__dict__ for layer in model.ts_encoder.transformer_encoder.layers))

    def test_existing_adapter_partial_restore(self):
        model = model_fixture()
        module = model.ts_encoder.transformer_encoder.layers[2].self_attn
        original = module.forward
        module.forward = original
        with self.assertRaises(ValueError):
            with attention_layers(model, np.ones(3, dtype=bool), np.zeros(3, dtype=bool), [2]):
                pass
        self.assertIs(module.__dict__["forward"], original)
        for index in (0, 1, 3):
            self.assertNotIn("forward", model.ts_encoder.transformer_encoder.layers[index].self_attn.__dict__)
        del module.forward


class LayerStatisticsTests(unittest.TestCase):
    def frame(self):
        values = {"A": .1, "keep_first_10": .2, "keep_first_01": .3, "keep_first_11": .5,
                  "keep_last_10": .4, "keep_last_01": .2, "keep_last_11": .8}
        return pd.DataFrame([{"case": "one", "seed": 1, "phase": 0, "shape": "positive", "event_length": 32,
                              "offset": 0, "score_kind": "scores", "variant": v, "metric_status": "ok",
                              "average_precision": value, "I_points": 0, "I_rank_mean": np.nan} for v, value in values.items()])

    def test_interaction_signs_conditional_effects_and_identity(self):
        effects, _ = paired_effects(self.frame().sample(frac=1, random_state=2))
        indexed = effects.set_index("contrast")
        expected = {"J_keep_first": .1, "J_keep_last": .3, "K": .2, "position_10": .2,
                    "position_01": -.1, "position_11": .3, "position_11_minus_01": .4,
                    "keep_first_E7_given0": .1, "keep_first_E7_given1": .2,
                    "keep_last_E8_given0": .1, "keep_last_E8_given1": .4}
        for name, value in expected.items():
            self.assertAlmostEqual(indexed.loc[name, "average_precision"], value)
        self.assertTrue(effects.I_rank_mean.isna().all())

    def test_missing_duplicate_undefined_and_partial_missingness_fail(self):
        frame = self.frame()
        missing = frame.copy()
        missing.loc[2, "average_precision"] = np.nan
        identity = frame.copy()
        identity.loc[2, "seed"] = np.nan
        nonempty = frame.copy()
        nonempty["I_points"] = 10
        inconsistent = frame.copy()
        inconsistent.loc[2, "I_rank_mean"] = .1
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), missing, identity, nonempty, inconsistent):
            with self.assertRaises(ValueError):
                paired_effects(bad)


class LayerPipelineTests(unittest.TestCase):
    def test_miniature_gate_full_readouts_parent_and_state(self):
        model = model_fixture()
        fingerprint = model_fingerprint(model)
        source = json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())
        settings = {**source["synthetic"], "length": 512, "anomaly_count": 128, "seeds": [1],
                    "phases": [0], "event_lengths": [32], "shapes": ["positive"]}
        meta, arrays = next(generate(settings, 16))[0]
        config = json.loads(Path("experiments/configs/stage2_layer_factorial_v1.json").read_text())
        config.update(window=512, expected_layers=4, switch_layers=[2, 3])
        masks, kept = select_positions(meta, arrays, 16, config["random_seeds"])
        fixed, roles = point_groups(meta, arrays, masks, 16, config["random_seeds"])
        values, valid = torch.tensor(arrays["corrupt_normalized"], dtype=torch.float32), torch.ones(512, dtype=torch.bool)
        old, rows = {}, []
        for variant, selector, depth in [("A", "A", 0)] + [
                (f"{s}_d{d}", s, d) for s in ("keep_first", "keep_last") for d in (1, 2)]:
            pred, _ = depth_predict(model, values, valid, arrays["type_valid"], masks["full" if selector == "A" else selector], depth, "cpu")
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
                metrics = pd.read_csv(output / "metrics.csv")
                self.assertEqual(len(metrics), 14)
                effects, _ = paired_effects(metrics)
                self.assertEqual(len(effects), 34)
        self.assertEqual(model_fingerprint(model), fingerprint)


if __name__ == "__main__":
    unittest.main()
