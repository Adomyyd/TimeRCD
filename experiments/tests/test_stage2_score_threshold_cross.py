import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.common.artifacts import sha256
from experiments.stage2_clean_reference.statistics import seed_summary
from experiments.stage2_normal_replay_transitions.readouts import analyze_case as e15_readouts
from experiments.stage2_pc_localization.data import Audit
from experiments.stage2_rank_localization.diagnostics import readout
from experiments.stage2_score_threshold_cross.core import (
    CELLS, WEIGHTS, cross_outputs, cross_memberships, factorial_effects, point_interaction)
from experiments.stage2_score_threshold_cross.data import compare_table, load_case, load_parent
from experiments.stage2_score_threshold_cross.readouts import analyze_case
from experiments.stage2_score_threshold_cross.statistics import paired_effects
from experiments.tests.test_stage2_normal_replay_transitions import fixture as e15_fixture


def fixture(directory):
    meta, labels, config, fixed, roles, predictions, derived, outputs, parent, originals = e15_fixture()
    audit = Audit(directory, 1e-12)
    old_rows, old_changes, memberships = e15_readouts(meta, labels, fixed, roles, predictions, derived,
                                                   outputs, config, parent, originals, audit)
    endpoints = {s: cross_outputs(labels, b["add"], b["full"]) for s, b in outputs.items()}
    additions = []
    for s in endpoints:
        additions.append({"case": meta["case"], "variant": f"{s}_07", "score_kind": "logits",
                          **readout(labels, endpoints[s]["01"], {**fixed, **roles[s]}, config["recalls"])})
    parent = pd.concat([parent, pd.DataFrame(additions).set_index(parent.index.names)])
    inputs = meta, labels, fixed, roles, predictions, outputs, endpoints, memberships, config, parent, audit
    return inputs, derived, originals, old_rows, old_changes


class CrossCoreTests(unittest.TestCase):
    def test_hand_count_cross_signed_point_interaction_and_paths(self):
        labels = np.array([1, 1, 0, 0, 0, 0])
        add, full = np.array([5, 3, 2, 4, 0, 1.]), np.array([4, 1, 4, 2, 1, 0.])
        outputs = cross_outputs(labels, add, full)
        masks, metrics = cross_memberships(labels, outputs, 1.)
        self.assertEqual([metrics[c]["fp"] for c in CELLS], [1, 1, 3, 3])
        effects = factorial_effects({c: metrics[c]["fp"] for c in CELLS})
        self.assertEqual([effects[k] for k in ("score_at_add", "threshold_at_add", "interaction", "total")], [0, 2, 0, 2])
        z, counts = point_interaction(masks, labels == 0)
        np.testing.assert_array_equal(z, [0, 0, -1, 1, 1, -1])
        self.assertEqual(counts, {"positive": 2, "negative": 2, "zero": 0, "net": 0, "points": 4})
        self.assertEqual(effects["score_average"] + effects["threshold_average"], effects["total"])

    def test_ties_allow_source_dependent_tp_and_preserve_fixed_threshold(self):
        y = np.array([1, 1, 1, 0, 0])
        outputs = cross_outputs(y, np.array([4, 2, 2, 2, 0.]), np.array([3, 3, 1, 3, 2.]))
        masks, metrics = cross_memberships(y, outputs, .5)
        self.assertEqual([metrics[c]["tp"] for c in CELLS], [3, 3, 2, 2])
        self.assertEqual([metrics[c]["threshold"] for c in CELLS], [2, 2, 3, 3])
        z, counts = point_interaction(masks, y == 0)
        self.assertEqual(counts["net"], factorial_effects({c: d["fp"] for c, d in metrics.items()})["interaction"])
        self.assertTrue(np.all(np.abs(z) <= 1))

    def test_equal_sources_noops_empty_selected_union_and_permutation(self):
        y, a = np.array([1, 1, 0, 0]), np.array([4., 3., 2., 1.])
        outputs = cross_outputs(y, a, a)
        masks, metrics = cross_memberships(y, outputs, .1)
        _, counts = point_interaction(masks, y == 0)
        self.assertEqual((counts["positive"], counts["negative"], counts["zero"]), (0, 0, 2))
        self.assertTrue(all(v == 0 for v in factorial_effects({c: d["fp"] for c, d in metrics.items()}).values()))
        order = [3, 1, 0, 2]
        permuted = cross_outputs(y[order], a[order], a[order])
        for c in CELLS:
            np.testing.assert_array_equal(outputs[c][order], permuted[c])

    def test_invalid_scores_labels_masks_and_values_rejected(self):
        y, a = np.array([1, 0]), np.array([1., 2.])
        for labels, add, full in (([0, 0], a, a), ([1, 2], a, a), (y, a[:1], a), (y, a, [1, np.nan])):
            with self.assertRaises(ValueError):
                cross_outputs(labels, add, full)
        with self.assertRaises(ValueError):
            cross_memberships(y, {"00": a}, .5)
        with self.assertRaises(ValueError):
            factorial_effects({"00": 1})
        masks = {c: np.zeros(2, bool) for c in CELLS}
        masks["00"][:] = True
        masks["11"][:] = True
        with self.assertRaises(ValueError):
            point_interaction(masks, np.ones(2, bool))
        with self.assertRaises(ValueError):
            point_interaction({c: a for c in CELLS}, y == 0)


class CrossPipelineTests(unittest.TestCase):
    def test_miniature_endpoints_identities_save_reload_and_seed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            inputs, *_ = fixture(directory)
            rows, changes, points, arrays = analyze_case(*inputs)
            self.assertEqual((len(rows), len(changes), len(points), len(arrays)), (48, 60, 12, 60))
            frames = list(map(pd.DataFrame, (rows, changes, points)))
            effects = paired_effects(*frames, inputs[-3]["recalls"], inputs[-1])
            self.assertEqual(len(effects), 96)
            shuffled = paired_effects(*[f.sample(frac=1, random_state=17) for f in frames], inputs[-3]["recalls"])
            pd.testing.assert_frame_equal(effects, shuffled)
            path = directory / "memberships.npz"
            np.savez_compressed(path, **arrays)
            with np.load(path) as saved:
                for k, v in arrays.items():
                    np.testing.assert_array_equal(saved[k], v)
            config = {**inputs[-3], "synthetic": {"seeds": [1]}, "bootstrap_repetitions": 100}
            seeds, summary = seed_summary(effects, ["selector", "target_recall"], ["fp", "fpr"], config)
            self.assertEqual((len(seeds), len(summary)), (96, 192))
            self.assertTrue(summary.seeds.eq(1).all())
            self.assertTrue(pd.read_csv(directory / "checks.csv").close.all())

    def test_missing_duplicate_nonfinite_and_modified_tables_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs, *_ = fixture(Path(temp))
            rows, changes, points, _ = analyze_case(*inputs)
            frames = list(map(pd.DataFrame, (rows, changes, points)))
            for position in range(3):
                for invalid in (frames[position].iloc[1:], pd.concat([frames[position], frames[position].iloc[:1]])):
                    bad = frames.copy()
                    bad[position] = invalid
                    with self.assertRaises(ValueError):
                        paired_effects(*bad, inputs[-3]["recalls"])
            for position, column, value in ((0, "fp", np.nan), (0, "normal_source", 9),
                                            (1, "entered", 1.5), (2, "net", 999)):
                bad = [f.copy() for f in frames]
                bad[position][column] = bad[position][column].astype(float)
                bad[position].loc[0, column] = value
                with self.assertRaises(ValueError):
                    paired_effects(*bad, inputs[-3]["recalls"])
            bad = [f[f.selector == "keep_first"] for f in frames]
            with self.assertRaises(ValueError):
                paired_effects(*bad, inputs[-3]["recalls"])

    def test_four_endpoint_and_parent_membership_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs, *_ = fixture(Path(temp))
            for cell in CELLS:
                changed = list(inputs)
                endpoints = {s: {c: a.copy() for c, a in b.items()} for s, b in inputs[6].items()}
                endpoints["keep_first"][cell][0] += 1
                changed[6] = endpoints
                with self.assertRaises(ValueError):
                    analyze_case(*changed)
            changed = list(inputs)
            members = {k: v.copy() for k, v in inputs[7].items()}
            key = next(iter(members))
            members[key][0] = ~members[key][0]
            changed[7] = members
            with self.assertRaises(ValueError):
                analyze_case(*changed)

    def test_parent_table_nan_coverage_and_categorical_checks(self):
        frame = pd.DataFrame({"case": ["a", "b"], "group": ["I", "N"], "value": [np.nan, 3.]})
        with tempfile.TemporaryDirectory() as temp:
            audit = Audit(Path(temp), 1e-12)
            compare_table(audit, "same", frame, frame.iloc[::-1], ["case"])
            for bad in (frame.iloc[:1], frame.assign(group="N"), frame.assign(value=[0., 3.]), pd.concat([frame, frame.iloc[:1]])):
                with self.assertRaises(ValueError):
                    compare_table(audit, "bad", frame, bad, ["case"])

    def test_parent_case_validation_and_recorded_membership_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            inputs, derived, originals, rows, changes = fixture(directory)
            meta, labels, fixed, roles, predictions, outputs, endpoints, memberships, config, metrics, audit = inputs
            replay_path = directory / "replay.npz"
            np.savez_compressed(replay_path, **{f"{s}_{CELLS[c][2]:02d}": a for s, b in endpoints.items() for c, a in b.items()})
            arrays = {**memberships, "labels": labels, **{f"group_{g}": a for g, a in fixed.items()}}
            arrays.update({f"{s}_role_{g}": a for s in outputs for g, a in roles[s].items()})
            path = directory / "membership.npz"
            np.savez_compressed(path, **arrays)
            config = {**config, "parent_run": str(directory)}
            context = {"previous": {"config": {"parent_run": str(directory)}}, "e14": {}, "e13": {},
                       "derived": {meta["case"]: {}}, "replayed": {meta["case"]: {"snapshot": replay_path.name}},
                       "members": {meta["case"]: {"snapshot": path.name, "sha256": sha256(path)}},
                       "metrics": metrics, "originals": originals, "thresholds": pd.DataFrame(rows), "transitions": pd.DataFrame(changes)}
            with patch("experiments.stage2_score_threshold_cross.data.load_replay_case", return_value=(labels, fixed, roles, predictions, derived, outputs)):
                result = load_case(config, context, meta, audit)
                np.testing.assert_array_equal(result[0], labels)
                arrays["labels"] = 1 - labels
                np.savez_compressed(path, **arrays)
                with self.assertRaisesRegex(ValueError, "membership changed"):
                    load_case(config, context, meta, audit)
                context["members"][meta["case"]]["sha256"] = sha256(path)
                with self.assertRaises(ValueError):
                    load_case(config, context, meta, audit)

    def test_incomplete_parent_run_and_changed_config_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            config = {"parent_run": str(path), "checkpoint": "fixed"}
            summary = {"inference_calls": 0, "checkpoint_unchanged": True,
                       "parent_artifacts_unchanged": True, "local_artifacts_unchanged": True}
            (path / "summary.json").write_text(json.dumps(summary))
            for manifest in ({"status": "failed"}, {"status": "completed", "config": {"checkpoint": "changed"}}):
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_parent(config)


if __name__ == "__main__":
    unittest.main()
