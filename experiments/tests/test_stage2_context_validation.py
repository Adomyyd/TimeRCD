import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.stage2_context_validation.data import (
    TARGET,
    context_mask,
    experiment_a_cases,
    experiment_b_cases,
    generate_base,
    normalize,
    patch_distance_to_target,
    ring_masks,
)
from experiments.stage2_context_validation.inference import score_condition, target_logits
from experiments.stage2_context_validation.analyze import analyze
from experiments.stage2_context_validation import run as runner
from experiments.stage2_context_validation.statistics import (
    bootstrap_mean,
    impact_radii_by_sequence,
    saturation_radius,
    summarize_a_rows,
    summarize_b_rows,
)


class ContextValidationDataTests(unittest.TestCase):
    def test_a_restores_exact_base_and_preserves_outside(self):
        for case in experiment_a_cases(3101):
            changed = case["labels"].astype(bool)
            self.assertTrue(np.array_equal(case["values"][~changed], case["base"][~changed]))
            self.assertTrue(np.array_equal(case["restored"], case["base"]))
            self.assertEqual(int(case["labels"].sum()), sum(end - start for start, end in case["intervals"]))

    def test_boundary_and_cluster_layouts_are_frozen(self):
        cases = {case["layout"]: case for case in experiment_a_cases(3101)}
        np.testing.assert_array_equal(cases["left_isolated"]["intervals"], [[0, 16]])
        np.testing.assert_array_equal(cases["right_isolated"]["intervals"], [[4984, 5000]])
        self.assertEqual(len(cases["center_clustered"]["intervals"]), 3)

    def test_b_pairs_share_anomaly_statistics(self):
        normal, anomaly = list(experiment_b_cases(3101))
        self.assertEqual(normal["target_kind"], "normal")
        self.assertEqual((normal["mean"], normal["std"]), (anomaly["mean"], anomaly["std"]))
        self.assertTrue(np.array_equal(normal["values"][:TARGET[0]], anomaly["values"][:TARGET[0]]))
        self.assertTrue(np.all(anomaly["labels"][TARGET[0]:TARGET[1]] == 1))
        normalized_normal, mean, std = normalize(normal["values"], normal["mean"], normal["std"])
        normalized_anomaly, mean2, std2 = normalize(anomaly["values"], anomaly["mean"], anomaly["std"])
        self.assertEqual((mean, std), (mean2, std2))
        np.testing.assert_array_equal(normalized_normal[:TARGET[0]], normalized_anomaly[:TARGET[0]])

    def test_context_masks_keep_base_radius_and_partial_tail_correct(self):
        distances = patch_distance_to_target()
        base = context_mask(radius=0)
        self.assertTrue(np.all(base[distances <= 5]))
        self.assertFalse(np.any(base[distances > 5]))
        self.assertTrue(np.all(context_mask(radius="full")))
        self.assertEqual(int(base.sum()), int((distances <= 5).sum()))
        self.assertEqual(int(context_mask(radius="full")[4992:].sum()), 8)
        self.assertLessEqual(int(base.sum()), int(context_mask(radius=10).sum()))

    def test_ring_masks_partition_all_points(self):
        case = next(experiment_a_cases(3101))
        masks, distances = ring_masks(case["labels"], case["intervals"])
        combined = [mask for name, mask in masks.items()
                    if not name.startswith(("left_", "right_", "between_events_"))]
        stacked = np.stack(combined)
        self.assertTrue(np.all(stacked.sum(axis=0) == 1))
        self.assertTrue(np.all(distances[case["labels"] == 1] == 0))
        self.assertTrue(masks["left_ring1"].any())
        self.assertTrue(masks["right_ring1"].any())


class ContextValidationInferenceTests(unittest.TestCase):
    def test_target_logits_require_visible_target(self):
        mask = np.ones(32, dtype=bool)
        logits = np.arange(32, dtype=np.float64)
        result = {"logits": logits}
        np.testing.assert_array_equal(target_logits(result, mask, (8, 16)), np.arange(8, 16))
        mask[8] = False
        with self.assertRaises(ValueError):
            target_logits(result, mask, (8, 16))

    def test_score_condition_applies_point_mask_and_extracts_logits(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(1, 1, bias=False)
                self.anomaly_head = nn.Linear(1, 2)

            def forward(self, values, mask):
                return self.projection(values).unsqueeze(-2)

        model = TinyModel().eval()
        values = np.linspace(-1, 1, 32, dtype=np.float32)
        mask = np.ones(32, dtype=bool)
        mask[:4] = False
        output = score_condition(model, values, mask, "cpu")
        self.assertEqual(output["logits"].shape, (28,))
        self.assertTrue(np.isfinite(output["logits"]).all())

    def test_score_condition_rejects_nonfinite_model_outputs(self):
        class NonfiniteModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(float("nan")))
                self.anomaly_head = nn.Linear(1, 2)

            def forward(self, values, mask):
                return values.unsqueeze(-2) * self.weight

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            score_condition(NonfiniteModel(), np.ones(16, dtype=np.float32),
                            np.ones(16, dtype=bool), "cpu")


class ContextValidationStatisticsTests(unittest.TestCase):
    def test_bootstrap_mean_is_deterministic(self):
        first = bootstrap_mean([1, 2, 3], seed=19, repetitions=200)
        second = bootstrap_mean([1, 2, 3], seed=19, repetitions=200)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.0)

    def test_bootstrap_rejects_nonfinite_and_multidimensional_values(self):
        for values in ([1, np.nan], [1, np.inf], [[1, 2], [3, 4]]):
            with self.assertRaises(ValueError):
                bootstrap_mean(values)

    def test_saturation_radius_respects_full_and_zero_total(self):
        self.assertEqual(saturation_radius([0, 1, 10, "full"], [1, 5, 9.9, 10]), 10)
        self.assertIsNone(saturation_radius([0, 1, "full"], [2, 2, 2]))

    def test_a_impact_radius_uses_binned_lower_boundary_and_skips_empty_groups(self):
        rows = []
        for ring, magnitude in (("ring0_anomaly", 1.0), ("ring3", 0.01), ("ring4_5", 0.02),
                                ("ring6_10", 0.03), ("ring11_20", 0.04),
                                ("ring21_39", 0.04), ("ring40_plus", 0.02)):
            rows.append({
                "name": "sample", "seed": 3101, "layout": "center_isolated",
                "location": "center", "event_pattern": "isolated", "ring": ring,
                "point_count": 1, "delta_mean": magnitude,
                "mean_absolute_delta": magnitude,
                "original_logit_mean": magnitude, "restored_logit_mean": 0.0,
                "original_probability_mean": 0.6, "restored_probability_mean": 0.5,
            })
        rows.append({**rows[1], "name": "sample", "ring": "left_ring1", "point_count": 0})
        self.assertEqual(impact_radii_by_sequence(rows)[0]["impact_radius_lower_bin_boundary"], 3)
        summary = summarize_a_rows(rows, repetitions=100, seed=2)
        self.assertTrue(any(row["ring"] == "ring3" for row in summary))

    def test_b_changes_are_paired_by_seed(self):
        rows = []
        for seed, base in ((1, 10.0), (2, 20.0)):
            rows.extend([
                {"seed": seed, "target_kind": "normal", "radius": 0, "target_logit_mean": base},
                {"seed": seed, "target_kind": "normal", "radius": 1, "target_logit_mean": base + seed},
                {"seed": seed, "target_kind": "normal", "radius": 10, "target_logit_mean": base + 2 * seed - 0.01},
                {"seed": seed, "target_kind": "normal", "radius": "full", "target_logit_mean": base + 2 * seed},
                {"seed": seed, "target_kind": "anomaly", "radius": 0, "target_logit_mean": base + 3},
                {"seed": seed, "target_kind": "anomaly", "radius": 1, "target_logit_mean": base + 3 + seed},
                {"seed": seed, "target_kind": "anomaly", "radius": 10, "target_logit_mean": base + 3 + 2 * seed - 0.01},
                {"seed": seed, "target_kind": "anomaly", "radius": "full", "target_logit_mean": base + 3 + 2 * seed},
            ])
        summary = summarize_b_rows(rows, [0, 1, 10, "full"], repetitions=100, seed=8)
        normal_r1 = next(row for row in summary if row["target_kind"] == "normal" and row["radius"] == 1)
        self.assertEqual(normal_r1["change_from_r0"], 1.5)
        self.assertEqual(normal_r1["saturation_radius"], 10)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_b_rows(rows + [rows[0]], [0, 1, 10, "full"], repetitions=100)
        with self.assertRaisesRegex(ValueError, "paired seeds"):
            summarize_b_rows(rows[1:], [0, 1, 10, "full"], repetitions=100)

    def test_a_derived_layouts_do_not_count_as_independent_bases(self):
        template = {
            "location": "boundary", "event_pattern": "isolated", "ring": "ring1",
            "point_count": 16, "delta_mean": 1.0, "mean_absolute_delta": 1.0,
            "original_logit_mean": 2.0, "restored_logit_mean": 1.0,
            "original_probability_mean": 0.7, "restored_probability_mean": 0.6,
        }
        rows = [{**template, "seed": seed, "name": f"{seed}_{side}"}
                for seed in (1, 2) for side in ("left", "right")]
        summary = summarize_a_rows(rows, repetitions=100)
        boundary = next(row for row in summary if row["location"] == "boundary")
        self.assertEqual(boundary["sequence_count"], 4)
        self.assertEqual(boundary["independent_base_count"], 2)

    def test_impact_radius_uses_absolute_signed_ring_mean_not_mean_absolute_points(self):
        template = {"name": "sample", "seed": 3101, "layout": "center_isolated", "point_count": 16}
        rows = [
            {**template, "ring": "ring0_anomaly", "delta_mean": 1.0, "mean_absolute_delta": 2.0},
            {**template, "ring": "ring1", "delta_mean": -0.04, "mean_absolute_delta": 10.0},
        ]
        self.assertEqual(impact_radii_by_sequence(rows)[0]["impact_radius_lower_bin_boundary"], 1)

    def test_saved_tables_produce_report_and_context_plot(self):
        radii = [0, 1, 2, 5, 10, 20, 40, "full"]
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            run_dir = Path(temporary)
            (run_dir / "manifest.json").write_text(json.dumps({
                "status": "completed",
                "config": {"bootstrap_repetitions": 100, "seed": 2024, "radii": radii},
            }))
            a_rows = []
            ring_names = ["ring0_anomaly", "same_event_patch_normal", "ring1", "ring2", "ring3",
                          "ring4_5", "ring6_10", "ring11_20", "ring21_39", "ring40_plus"]
            for ring in ring_names:
                magnitude = 1.0 if ring == "ring0_anomaly" else 0.01
                a_rows.append({
                    "name": "sample", "seed": 3101, "layout": "center_isolated",
                    "location": "center", "event_pattern": "isolated", "ring": ring,
                    "point_count": 1, "delta_mean": magnitude,
                    "mean_absolute_delta": magnitude,
                    "original_logit_mean": magnitude, "restored_logit_mean": 0.0,
                    "original_probability_mean": 0.6, "restored_probability_mean": 0.5,
                })
            with (run_dir / "a_sequence_rings.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(a_rows[0]))
                writer.writeheader()
                writer.writerows(a_rows)
            b_rows = []
            for target_kind, offset in (("normal", 0.0), ("anomaly", 1.0)):
                for radius in radii:
                    b_rows.append({
                        "name": f"b_{target_kind}", "seed": 3101,
                        "target_kind": target_kind, "radius": radius,
                        "target_logit_mean": offset + (0 if radius == 0 else 1),
                    })
            with (run_dir / "b_target_scores.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(b_rows[0]))
                writer.writeheader()
                writer.writerows(b_rows)
            output = analyze(run_dir)
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "b_context_curve.png").is_file())
            self.assertTrue((output / "a_impact_radii.csv").is_file())


class ContextValidationRunnerTests(unittest.TestCase):
    def test_gpu_gate_reads_selected_card_and_rejects_busy_resources(self):
        result = SimpleNamespace(stdout="0, 0, 10000\n2, 100, 4984\n")
        with patch.object(runner.subprocess, "run", return_value=result), \
             patch.dict(runner.os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            with self.assertRaisesRegex(RuntimeError, "not idle enough"):
                runner._gpu_gate()
        with patch.object(runner.subprocess, "run", return_value=result), \
             patch.dict(runner.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual(runner._gpu_gate()["index"], 0)

    def test_cpu_fixture_runs_input_snapshot_gate_predictions_and_report(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anomaly_head = nn.Linear(1, 2)

            def forward(self, values, mask):
                return values.unsqueeze(-2)

        class TinyDetector:
            def __init__(self):
                self._tester = None

            def _ensure_tester(self, channels):
                if channels != 1:
                    raise ValueError("Expected one channel")
                self._tester = SimpleNamespace(model=TinyModel().eval(), device="cpu")

            def predict(self, values, return_logits=False):
                normalized, _, _ = normalize(np.asarray(values).reshape(-1))
                result = score_condition(self._tester.model, normalized,
                                         np.ones(len(normalized), dtype=bool), "cpu")
                return result["scores"], result["logits"]

        config_path = Path(__file__).resolve().parents[1] / "configs/stage2_context_validation_v1.json"
        config = runner._resolve_config(config_path)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "fixture.weights"
            checkpoint.write_bytes(b"CPU fixture, not official weights")
            config.update(checkpoint=str(checkpoint), output_root=str(temporary / "runs"),
                          bootstrap_repetitions=100)
            with patch.object(runner, "_resolve_config", return_value=config), \
                 patch.object(runner, "_gpu_gate", return_value={"test_fixture": True}), \
                 patch.object(runner, "make_detector", return_value=TinyDetector()):
                directory = runner.run(config_path)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["b_target_condition_rows"], 288)
            with (directory / "input_manifest.csv").open() as stream:
                inputs = list(csv.DictReader(stream))
            self.assertEqual(len(inputs), 144)
            self.assertTrue(any(row["experiment"] == "B" for row in inputs))
            self.assertEqual(manifest["model_sha256_before"], manifest["model_sha256_after"])
            self.assertTrue((directory / "source/experiments/stage2_context_validation/run.py").is_file())
            self.assertTrue((analyze(directory) / "report.md").is_file())

    def test_busy_gpu_stops_before_model_initialization(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/stage2_context_validation_v1.json"
        config = runner._resolve_config(config_path)
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            config["output_root"] = temporary
            with patch.object(runner, "_resolve_config", return_value=config), \
                 patch.object(runner, "capture_provenance"), \
                 patch.object(runner, "_save_inputs", return_value=([None] * 108, [None] * 36)), \
                 patch.object(runner, "_gpu_gate", side_effect=RuntimeError("GPU not idle enough")), \
                 patch.object(runner, "make_detector") as loader:
                with self.assertRaisesRegex(RuntimeError, "not idle"):
                    runner.run(config_path)
                loader.assert_not_called()
            directories = list(Path(temporary).iterdir())
            manifest = json.loads((directories[0] / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "blocked_gpu_busy")


if __name__ == "__main__":
    unittest.main()
