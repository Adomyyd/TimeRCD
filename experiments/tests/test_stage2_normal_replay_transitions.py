import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_logit_additivity.core import additive_output, sigmoid
from experiments.stage2_normal_replay_transitions.core import threshold_membership, transition_counts
from experiments.stage2_normal_replay_transitions.data import CONDITIONS, load_case, metric_table
from experiments.stage2_normal_replay_transitions.readouts import analyze_case
from experiments.stage2_normal_replay_transitions.statistics import paired_effects, validate_factor_coverage
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.diagnostics import readout
from experiments.stage2_residual_replay.core import replay
from experiments.tests import test_stage2_residual_replay as replay_tests


def fixture():
    meta, arrays, config, _, fixed, roles, _, _, _ = replay_tests.ReplayPipelineTests().fixture()
    rng = np.random.default_rng(23)
    predictions = {"A_logits": rng.integers(-3, 4, 512).astype(np.float32)}
    derived = {f"group_{g}": m for g, m in fixed.items()}
    outputs, parent, originals = {}, [], []
    for selector in ("keep_first", "keep_last"):
        for condition in ("10", "01", "11"):
            predictions[f"{selector}_{condition}_logits"] = rng.integers(-4, 5, 512).astype(np.float32)
        additive, residual = additive_output(*[predictions[f"{v}_logits"] for v in
                                               ("A", f"{selector}_10", f"{selector}_01", f"{selector}_11")])
        derived.update({f"{selector}_add_logits": additive, f"{selector}_add_scores": sigmoid(additive),
                        f"{selector}_residual": residual})
        outputs[selector] = {c: replay(additive, residual, fixed, s) for c, s in CONDITIONS.items()}
        for c, subset in CONDITIONS.items():
            values = outputs[selector][c]
            if c == "full":
                values = values.astype(np.float32)
            parent.append({"case": meta["case"], "variant": f"{selector}_{subset:02d}", "score_kind": "logits",
                           **readout(arrays["labels"], values, {**fixed, **roles[selector]}, config["recalls"])})
    for name, values in list(predictions.items()):
        predictions[name.replace("_logits", "_scores")] = sigmoid(values).astype(np.float32)
    for name, values in {**predictions, **{k: v for k, v in derived.items() if "add_" in k}}.items():
        variant, kind = name.rsplit("_", 1)
        selector = "A" if variant == "A" else variant.rsplit("_", 1)[0]
        originals.append({"case": meta["case"], "variant": variant, "score_kind": kind,
                          **readout(arrays["labels"], values, {**fixed, **roles[selector]}, config["recalls"])})
    keys = ["case", "variant", "score_kind"]
    return meta, arrays["labels"], config, fixed, roles, predictions, derived, outputs, pd.DataFrame(parent).set_index(keys), pd.DataFrame(originals).set_index(keys)


class TransitionCoreTests(unittest.TestCase):
    def test_ties_recall_boundaries_and_invalid_inputs(self):
        labels, scores = np.array([1, 1, 1, 0, 0]), np.array([4., 2., 2., 2., 0.])
        selected, row = threshold_membership(labels, scores, .5)
        np.testing.assert_array_equal(selected, [True, True, True, True, False])
        self.assertEqual((row["threshold"], row["tp"], row["fp"], row["fn"], row["tn"]), (2, 3, 1, 0, 1))
        self.assertEqual((row["recall"], row["strict_recall"]), (1., 1 / 3))
        for recall in (.01, 1.):
            mask, result = threshold_membership(labels, scores, recall)
            self.assertGreaterEqual(result["recall"], recall)
            self.assertEqual(result["selected"], int(mask.sum()))
        for y, s, r in (([0, 0], [1, 2], .5), ([1, 2], [1, 2], .5), (labels, scores, 0),
                        (labels, scores, 1.1), (labels, scores, np.nan), (labels, [1, 2], .5),
                        (labels, [4, 2, 2, np.inf, 0], .5)):
            with self.assertRaises(ValueError):
                threshold_membership(y, s, r)

    def test_hand_count_all_states_empty_group_and_permutation(self):
        before, after = np.array([0, 1, 1, 0], bool), np.array([1, 0, 1, 0], bool)
        group = np.ones(4, bool)
        counts = transition_counts(before, after, group)
        self.assertEqual([counts[k] for k in ("entered", "exited", "retained", "absent", "net_selected", "changed")], [1, 1, 1, 1, 0, 2])
        self.assertEqual(counts["jaccard"], 1 / 3)
        self.assertEqual(transition_counts(before, after, group & False)["points"], 0)
        self.assertTrue(np.isnan(transition_counts(before, after, group & False)["jaccard"]))
        order = [2, 0, 3, 1]
        self.assertEqual(counts, transition_counts(before[order], after[order], group[order]))
        for bad in ((before.astype(int), after, group), (before, after[:-1], group)):
            with self.assertRaises(ValueError):
                transition_counts(*bad)


class TransitionPipelineTests(unittest.TestCase):
    def run_fixture(self, directory):
        meta, labels, config, fixed, roles, predictions, derived, outputs, parent, originals = fixture()
        audit = Audit(directory, 1e-12)
        rows, changes, memberships = analyze_case(meta, labels, fixed, roles, predictions, derived, outputs, config, parent, originals, audit)
        return pd.DataFrame(rows), pd.DataFrame(changes), memberships, config, audit

    def test_miniature_parent_endpoints_save_reload_pairing_and_seed_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            thresholds, transitions, arrays, config, audit = self.run_fixture(directory)
            self.assertEqual((len(thresholds), len(transitions), len(arrays)), (36, 168, 36))
            effects = paired_effects(thresholds, transitions, config["recalls"], audit)
            self.assertEqual(len(effects), 24)
            normal = effects[effects.contrast == "normal_minus_add"]
            self.assertTrue(normal[["delta_tp", "delta_fn", "delta_threshold", "F_entered", "L_exited", "I_entered"]].eq(0).all().all())
            np.testing.assert_array_equal(normal.delta_fp, normal.N_net_selected)
            self.assertTrue(transitions[transitions["group"] == "I"].points.eq(0).all())
            self.assertTrue(thresholds.I_capture.isna().all())
            path = directory / "membership.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path) as saved:
                for name, mask in arrays.items():
                    np.testing.assert_array_equal(mask, saved[name])
            shuffled = paired_effects(thresholds.sample(frac=1, random_state=3), transitions.sample(frac=1, random_state=4), config["recalls"])
            pd.testing.assert_frame_equal(effects, shuffled)
            settings = {**config, "synthetic": {"seeds": [1]}, "bootstrap_repetitions": 100}
            seeds, summary = seed_summary(effects, ["selector", "target_recall"], ["N_entered", "N_exited", "N_net_selected"], settings)
            self.assertEqual((len(seeds), len(summary)), (24, 72))
            self.assertTrue(summary.seeds.eq(1).all())
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())

    def test_missing_duplicate_mismatched_and_undefined_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            thresholds, transitions, _, config, _ = self.run_fixture(Path(temp))
            invalid = thresholds.copy()
            invalid.loc[0, "fp"] = np.nan
            wrong = transitions.copy()
            wrong.loc[0, "entered"] += 1
            for t, c in ((thresholds.iloc[1:], transitions), (pd.concat([thresholds, thresholds.iloc[:1]]), transitions),
                         (thresholds, transitions.iloc[1:]), (thresholds, pd.concat([transitions, transitions.iloc[:1]])),
                         (invalid, transitions), (thresholds, wrong), (thresholds[thresholds.selector == "keep_first"], transitions)):
                with self.assertRaises(ValueError):
                    paired_effects(t, c, config["recalls"])

    def test_parent_readout_and_endpoint_mutations_rejected(self):
        meta, labels, config, fixed, roles, predictions, derived, outputs, parent, originals = fixture()
        with tempfile.TemporaryDirectory() as temp:
            for arm in ("add", "normal", "full"):
                changed = {s: {c: a.copy() for c, a in b.items()} for s, b in outputs.items()}
                changed["keep_first"][arm][0] += 1
                with self.assertRaises(ValueError):
                    analyze_case(meta, labels, fixed, roles, predictions, derived, changed, config, parent, originals, Audit(Path(temp), 1e-12))
            bad_parent = parent.copy()
            bad_parent.iloc[0, bad_parent.columns.get_loc("average_precision")] += .1
            with self.assertRaises(ValueError):
                analyze_case(meta, labels, fixed, roles, predictions, derived, outputs, config, bad_parent, originals, Audit(Path(temp), 1e-12))

    def test_frozen_replay_hash_group_and_coverage_rejected(self):
        meta, labels, config, fixed, roles, predictions, derived, outputs, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "replay.npz"
            arrays = {f"{s}_{i:02d}": replay(derived[f"{s}_add_logits"], derived[f"{s}_residual"], fixed, i)
                      for s in outputs for i in range(16)}
            arrays.update({f"group_{g}": m for g, m in fixed.items()})
            np.savez_compressed(path, **arrays)
            snapshot = {"snapshot": path.name, "sha256": sha256(path)}
            previous = {"config": {"parent_run": str(directory)}}
            e13 = {"config": {"parent_run": str(directory)}}
            config = {**config, "parent_run": str(directory)}
            with patch("experiments.stage2_normal_replay_transitions.data.load_derived_case", return_value=(labels, fixed, roles, predictions, derived)):
                load_case(config, previous, e13, meta, {}, snapshot)
                for changed in ({**arrays, "group_F": ~fixed["F"]}, {k: v for k, v in arrays.items() if k != "keep_first_08"}):
                    np.savez_compressed(path, **changed)
                    with self.assertRaisesRegex(ValueError, "arrays changed"):
                        load_case(config, previous, e13, meta, {}, snapshot)
                    snapshot["sha256"] = sha256(path)
                    with self.assertRaises(ValueError):
                        load_case(config, previous, e13, meta, {}, snapshot)

    def test_parent_metric_duplicates_and_missing_rows_rejected(self):
        *_, parent, _ = fixture()
        frame = parent.reset_index()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.csv"
            for changed in (frame.iloc[:-1], pd.concat([frame.iloc[:-1], frame.iloc[:1]])):
                changed.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    metric_table(path, len(frame))

    def test_all_factor_cells_required_and_case_factor_bijection(self):
        config = {"synthetic": {"seeds": [1, 2]}}
        rows = [{"case": f"{s}_{p}_{h}_{n}", "seed": s, "phase": p, "shape": h, "event_length": n, "offset": 0}
                for s in (1, 2) for p in (0, 16, 32, 48) for h in ("positive", "negative", "alternating") for n in (32, 128)]
        frame = pd.DataFrame(rows)
        validate_factor_coverage(frame.sample(frac=1, random_state=5), config)
        duplicated_case = frame.copy()
        duplicated_case.loc[1, "case"] = duplicated_case.loc[0, "case"]
        duplicated_factor = pd.concat([frame, frame.iloc[:1].assign(case="extra")])
        for bad in (frame.iloc[1:], duplicated_case, duplicated_factor):
            with self.assertRaises(ValueError):
                validate_factor_coverage(bad, config)


if __name__ == "__main__":
    unittest.main()
