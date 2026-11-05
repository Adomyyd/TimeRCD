import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_alignment_controls.data import generate
from experiments.stage2_logit_additivity.core import additive_output
from experiments.stage2_p_positions.data import select_positions
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.diagnostics import point_groups, readout
from experiments.stage2_residual_replay.core import GROUPS, permutation_shapley, replay, shapley
from experiments.stage2_residual_replay.data import load_case
from experiments.stage2_residual_replay.readouts import analyze_case
from experiments.stage2_residual_replay.statistics import metric_columns, paired_effects


class ReplayCoreTests(unittest.TestCase):
    def test_partition_subset_truth_table_empty_group_and_invalid_inputs(self):
        groups = {name: np.arange(4) == i for i, name in enumerate(GROUPS)}
        base, residual = np.arange(4, dtype=float), np.array([1, -2, 3, -4.])
        for subset in range(16):
            expected = base + residual * np.array([bool(subset & (1 << i)) for i in range(4)])
            np.testing.assert_array_equal(replay(base, residual, groups, subset), expected)
        empty = {**groups, "I": np.zeros(4, dtype=bool), "N": groups["I"] | groups["N"]}
        for subset in range(16):
            np.testing.assert_array_equal(replay(base, residual, empty, subset), replay(base, residual, empty, subset ^ 4))
        for subset in (-1, 16, True, 2.0):
            with self.assertRaises(ValueError):
                replay(base, residual, groups, subset)
        for bad in ({**groups, "F": np.ones(4, dtype=bool)}, {k: v for k, v in groups.items() if k != "I"},
                    {**groups, "F": np.ones(4, dtype=int)}):
            with self.assertRaises(ValueError):
                replay(base, residual, bad, 0)
        with self.assertRaises(ValueError):
            replay(base, [1, 2, np.nan, 4], groups, 0)

    def test_additive_negative_and_symmetric_interaction_games(self):
        coefficients = np.array([1., -2., 3., -4.])
        values = np.array([[7 + sum(coefficients[i] for i in range(4) if s & (1 << i))] for s in range(16)])
        phi, margins = shapley(values)
        np.testing.assert_allclose(phi[:, 0], coefficients, atol=1e-14)
        self.assertEqual(len(margins), 32)
        np.testing.assert_allclose(phi, permutation_shapley(values), atol=1e-14)
        interaction = np.zeros((16, 1))
        interaction[-1] = 1
        np.testing.assert_allclose(shapley(interaction)[0], np.full((4, 1), .25), atol=1e-14)
        pair = np.array([[float(s & 1 != 0 and s & 2 != 0)] for s in range(16)])
        np.testing.assert_allclose(shapley(pair)[0][:, 0], [.5, .5, 0, 0], atol=1e-14)

    def test_permutation_equivariance_missingness_and_completeness(self):
        values = np.random.default_rng(12).normal(size=(16, 3))
        order = [2, 0, 3, 1]
        remapped = np.array([values[sum(1 << order[i] for i in range(4) if s & (1 << i))] for s in range(16)])
        np.testing.assert_allclose(shapley(remapped)[0], shapley(values)[0][order], atol=1e-14)
        np.testing.assert_allclose(shapley(values)[0].sum(axis=0), values[-1] - values[0], atol=1e-14)
        absent = np.column_stack([values, np.full(16, np.nan)])
        self.assertTrue(np.isnan(shapley(absent)[0][:, -1]).all())
        partial = values.copy()
        partial[1, 1] = np.nan
        for bad in (values[:-1], partial, np.full((16, 1), np.inf)):
            with self.assertRaises(ValueError):
                shapley(bad)


class ReplayStatisticsTests(unittest.TestCase):
    def frame(self):
        rows = []
        for selector, coefficients in (("keep_first", [.02, -.01, 0, .03]), ("keep_last", [.01, .04, 0, -.02])):
            for subset in range(16):
                value = .4 + sum(coefficients[i] for i in range(4) if subset & (1 << i))
                rows.append({"case": "one", "seed": 1, "phase": 0, "shape": "positive", "event_length": 32,
                             "offset": 0, "score_kind": "logits", "selector": selector, "subset": subset,
                             "variant": f"{selector}_{subset:02d}", "metric_status": "ok",
                             "average_precision": value, "roc_auc": value, "I_points": 0, "I_rank_mean": np.nan})
        return pd.DataFrame(rows)

    def test_pairing_sign_efficiency_context_marginals_and_dummy(self):
        effects, margins, contexts = paired_effects(self.frame().sample(frac=1, random_state=2), ["average_precision", "roc_auc", "I_rank_mean"])
        self.assertEqual((len(effects), len(margins), len(contexts)), (15, 64, 8))
        values = effects.set_index("contrast").average_precision
        for name, expected in {"D_F": -.01, "D_L": .05, "D_I": 0., "D_N": -.05,
                               "R_keep_first": .04, "R_keep_last": .03, "K_R": -.01}.items():
            self.assertAlmostEqual(values[name], expected)
        self.assertEqual(values["phi_keep_first_I"], 0)
        self.assertTrue(effects.I_rank_mean.isna().all())

    def test_missing_duplicate_undefined_and_variant_mismatch_rejected(self):
        frame = self.frame()
        bad_variant = frame.copy()
        bad_variant.loc[1, "variant"] = "wrong"
        undefined = frame.copy()
        undefined.loc[1, "average_precision"] = np.nan
        for bad in (frame.iloc[1:], frame.iloc[:16], pd.concat([frame, frame.iloc[:1]]), bad_variant, undefined):
            with self.assertRaises(ValueError):
                paired_effects(bad, ["average_precision", "roc_auc"])


class ReplayPipelineTests(unittest.TestCase):
    def fixture(self):
        source = json.loads(Path("experiments/configs/stage2_alignment_controls_v1.json").read_text())
        settings = {**source["synthetic"], "length": 512, "anomaly_count": 128, "seeds": [1],
                    "phases": [0], "event_lengths": [32], "shapes": ["positive"]}
        meta, arrays = next(generate(settings, 16))[0]
        config = json.loads(Path("experiments/configs/stage2_residual_replay_v1.json").read_text())
        config["window"] = 512
        selections, _ = select_positions(meta, arrays, 16, config["random_seeds"])
        fixed, roles = point_groups(meta, arrays, selections, 16, config["random_seeds"])
        rng = np.random.default_rng(12)
        predictions, derived, rows = {}, {f"group_{g}": m for g, m in fixed.items()}, []
        baseline = rng.normal(size=512).astype(np.float32)
        for selector in ("keep_first", "keep_last"):
            l10, l01, l11 = [rng.integers(-4, 5, size=512).astype(np.float32) for _ in range(3)]
            additive, residual = additive_output(baseline, l10, l01, l11)
            predictions[f"{selector}_11_logits"] = l11
            derived.update({f"{selector}_add_logits": additive, f"{selector}_residual": residual,
                            f"{selector}_add_scores": np.zeros(512)})
            for kind, values in (("add", additive), ("11", l11)):
                rows.append({"case": meta["case"], "variant": f"{selector}_{kind}", "score_kind": "logits",
                             **readout(arrays["labels"], values, {**fixed, **roles[selector]}, config["recalls"])})
        return meta, arrays, config, selections, fixed, roles, predictions, derived, pd.DataFrame(rows).set_index(["case", "variant", "score_kind"])

    def test_miniature_all_subsets_ties_parent_readouts_and_statistics(self):
        meta, arrays, config, _, fixed, roles, predictions, derived, parent = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            audit = Audit(directory, 1e-12)
            rows, replayed = analyze_case(meta, arrays["labels"], fixed, roles, predictions, derived, config, parent, audit)
            self.assertEqual((len(rows), len(replayed)), (32, 32))
            effects, margins, _ = paired_effects(pd.DataFrame(rows), metric_columns(config["recalls"]), audit)
            self.assertEqual((len(effects), len(margins)), (15, 64))
            path = directory / "replay.npz"
            np.savez_compressed(path, **replayed)
            with np.load(path) as saved:
                for name, values in replayed.items():
                    np.testing.assert_array_equal(saved[name], values)
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())

    def test_frozen_derived_hash_and_group_mismatch_rejected(self):
        meta, arrays, config, selections, fixed, _, predictions, derived, _ = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original, parent = directory / "original", directory / "parent"
            (original / "arrays").mkdir(parents=True)
            parent.mkdir()
            for kind, data in (("input", arrays), ("selection", selections)):
                path = original / f"{kind}.npz"
                np.savez_compressed(path, **data)
                meta.update({f"{kind}_snapshot": path.name, f"{kind}_sha256": sha256(path)})
            np.savez_compressed(original / f"arrays/{meta['case']}.npz", **predictions)
            path = parent / "derived.npz"
            np.savez_compressed(path, **derived)
            snapshot = {"snapshot": path.name, "sha256": sha256(path)}
            load_case(parent, original, meta, snapshot, config)
            changed = {**derived, "group_F": ~fixed["F"]}
            np.savez_compressed(path, **changed)
            with self.assertRaisesRegex(ValueError, "arrays changed"):
                load_case(parent, original, meta, snapshot, config)
            snapshot["sha256"] = sha256(path)
            with self.assertRaisesRegex(ValueError, "group changed"):
                load_case(parent, original, meta, snapshot, config)


if __name__ == "__main__":
    unittest.main()
