import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.common.metrics import detection_metrics
from experiments.stage2_alignment_controls.data import generate
from experiments.stage2_layer_factorial.statistics import VARIANTS
from experiments.stage2_logit_additivity.core import additive_output, residual_readout, sigmoid
from experiments.stage2_logit_additivity.data import verify_hashes
from experiments.stage2_logit_additivity.readouts import analyze_case
from experiments.stage2_logit_additivity.statistics import paired_effects
from experiments.stage2_p_positions.data import select_positions
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.diagnostics import point_groups, readout


class AdditivityCoreTests(unittest.TestCase):
    def test_hand_computed_residuals_groups_empty_and_invalid_inputs(self):
        additive, residual = additive_output([0, 1], [1, 2], [3, 0], [5, 0])
        np.testing.assert_array_equal(additive, [4, 1])
        np.testing.assert_array_equal(residual, [1, -1])
        self.assertEqual(additive.dtype, np.float64)
        groups = {"F": np.array([True, False]), "N": np.array([False, True]),
                  "L": np.zeros(2, dtype=bool), "I": np.zeros(2, dtype=bool)}
        result = residual_readout(residual, np.ones(2), np.array([3, -1]), groups)
        self.assertEqual(result["all_mean"], 0)
        self.assertEqual(result["all_mae"], 1)
        self.assertEqual(result["all_rms"], 1)
        self.assertEqual(result["F_mean"], 1)
        self.assertTrue(np.isnan(result["I_mae"]))
        for bad in ([], [np.nan, 1], [np.inf, 0], [[1, 2]], [1]):
            with self.assertRaises(ValueError):
                additive_output(bad, [1, 2], [1, 2], [1, 2])
        with self.assertRaises(ValueError):
            residual_readout(residual, np.ones(2), np.ones(2), {**groups, "I": np.ones(2, dtype=bool)})

    def test_additive_logits_can_have_metric_interaction_and_shift_can_have_zero_remainder(self):
        labels = np.array([1, 0])
        l00, l10, l01, l11 = [np.array(a, dtype=float) for a in ([0, 1], [2, 1], [2, 1], [4, 1])]
        add, residual = additive_output(l00, l10, l01, l11)
        ap = lambda x: detection_metrics(labels, x)["average_precision"]
        self.assertEqual(ap(add) - ap(l10) - ap(l01) + ap(l00), -.5)
        np.testing.assert_array_equal(residual, [0, 0])
        self.assertEqual(ap(l11) - ap(add), 0)
        add, residual = additive_output(l00, l10, l01, l11 + 5)
        np.testing.assert_array_equal(residual, [5, 5])
        self.assertEqual(ap(l11 + 5) - ap(add), 0)

    def test_extreme_sigmoid_and_ties(self):
        with np.errstate(over="raise", invalid="raise"):
            result = sigmoid(np.array([-1000, 0, 1000]))
        np.testing.assert_array_equal(result, [0, .5, 1])
        self.assertEqual(detection_metrics(np.array([1, 0]), np.array([0., 0.]))["average_precision"], .5)
        with self.assertRaises(ValueError):
            sigmoid([np.inf])


class AdditivityStatisticsTests(unittest.TestCase):
    def frame(self):
        values = {"A": .1, "keep_first_10": .2, "keep_first_01": .3, "keep_first_11": .5,
                  "keep_last_10": .4, "keep_last_01": .2, "keep_last_11": .8,
                  "keep_first_add": .45, "keep_last_add": .6}
        return pd.DataFrame([{"case": "one", "seed": 1, "phase": 0, "shape": "positive", "event_length": 32,
                              "offset": 0, "score_kind": "logits", "variant": v, "metric_status": "ok",
                              "average_precision": value, "I_points": 0, "I_rank_mean": np.nan} for v, value in values.items()])

    def test_signs_interaction_and_remainder_identities(self):
        effects, _ = paired_effects(self.frame().sample(frac=1, random_state=2))
        indexed = effects.set_index("contrast")
        expected = {"J_add_keep_first": .05, "J_add_keep_last": .1, "R_keep_first": .05,
                    "R_keep_last": .2, "K_add": .05, "K_R": .15, "K_total": .2}
        for name, value in expected.items():
            self.assertAlmostEqual(indexed.loc[name, "average_precision"], value)
        self.assertTrue(effects.I_rank_mean.isna().all())

    def test_missing_duplicate_nonfinite_and_changed_hash_rejected(self):
        frame = self.frame()
        undefined = frame.copy()
        undefined.loc[7, "average_precision"] = np.nan
        missing = frame.copy()
        missing.loc[7, "I_rank_mean"] = .1
        for bad in (frame.iloc[1:], frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), undefined, missing):
            with self.assertRaises(ValueError):
                paired_effects(bad)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "parent.txt"
            path.write_text("original")
            fingerprints = {str(path): sha256(path)}
            verify_hashes(fingerprints)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                verify_hashes(fingerprints)


class AdditivityPipelineTests(unittest.TestCase):
    def test_miniature_parent_readouts_arrays_and_statistics(self):
        source = json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())
        settings = {**source["synthetic"], "length": 512, "anomaly_count": 128, "seeds": [1],
                    "phases": [0], "event_lengths": [32], "shapes": ["positive"]}
        meta, arrays = next(generate(settings, 16))[0]
        config = json.loads(Path("experiments/configs/stage2_logit_additivity_v1.json").read_text())
        config["window"] = 512
        selections, _ = select_positions(meta, arrays, 16, config["random_seeds"])
        fixed, roles = point_groups(meta, arrays, selections, 16, config["random_seeds"])
        rng = np.random.default_rng(42)
        predictions, rows = {}, []
        for variant in sorted(VARIANTS):
            selector = "A" if variant == "A" else variant.rsplit("_", 1)[0]
            logits = rng.normal(size=512).astype(np.float32)
            for kind, scores in (("logits", logits), ("scores", sigmoid(logits))):
                predictions[f"{variant}_{kind}"] = scores
                rows.append({"case": meta["case"], "variant": variant, "score_kind": kind,
                             **readout(arrays["labels"], scores, {**fixed, **roles[selector]}, config["recalls"])})
        parent = pd.DataFrame(rows).set_index(["case", "variant", "score_kind"])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            audit = Audit(directory, 1e-12)
            rows, residuals, derived = analyze_case(meta, arrays["labels"], fixed, roles, predictions, config, parent, audit)
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(residuals), 2)
            self.assertEqual(len(derived), 6)
            path = directory / "derived.npz"
            np.savez_compressed(path, **derived)
            with np.load(path) as saved:
                for key, array in derived.items():
                    np.testing.assert_array_equal(saved[key], array)
            effects, _ = paired_effects(pd.DataFrame(rows), audit)
            self.assertEqual(len(effects), 22)
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())
            with self.assertRaises(ValueError):
                analyze_case(meta, arrays["labels"], fixed, roles, {k: v for k, v in predictions.items() if k != "A_logits"}, config, parent, audit)


if __name__ == "__main__":
    unittest.main()
