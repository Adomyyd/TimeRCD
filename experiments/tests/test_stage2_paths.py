import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.stage1.inference import score_window
from experiments.stage2.masking import reference_intervention
from experiments.stage2_paths.attention import attention_edges, edge_attention, edge_mask, model_fingerprint
from experiments.stage2_paths.inference import CheckFailure, check_array
from experiments.stage2_paths.analyze import contrasts
from time_rcd._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from time_rcd._core.time_rcd_config import TimeRCDConfig


class EdgeMaskTests(unittest.TestCase):
    def test_predefined_paired_contrasts_average_within_series(self):
        base = {"case": "s0", "series": "test", "kind": "tsb", "source": "Test",
                "generator_seed": None, "placement": None, "anomaly_count": 2, "length": 64,
                "scope": "full", "score_kind": "scores", "metric_status": "ok"}
        rows = [{**base, "variant": "A", "average_precision": .5}]
        for selector, values in (("label", [.2, .6, .1]), ("random_1", [.4, .5, .3]),
                                 ("random_2", [.3, .4, .2])):
            for scope in ("last", "all"):
                for arm, value in zip("BCD", values):
                    rows.append({**base, "variant": f"{selector}__{scope}__{arm}", "average_precision": value})
        frame = pd.DataFrame(rows)
        per_seed, paired = contrasts(frame, [1, 2])
        self.assertEqual(len(per_seed), 6)
        self.assertEqual(len(paired), 2)
        row = paired.iloc[0]
        self.assertAlmostEqual(row.label_C_minus_D, .5)
        self.assertAlmostEqual(row.label_minus_random_C_minus_D, .3)
        self.assertAlmostEqual(row.label_C_minus_A, .1)
        self.assertAlmostEqual(row.label_interaction, -.2)
        with self.assertRaises(ValueError):
            contrasts(frame.iloc[:-1], [1, 2])
        with self.assertRaises(ValueError):
            contrasts(pd.concat([frame, frame.iloc[:1]]), [1, 2])

    def test_truth_table_direction_diagonal_and_padding(self):
        valid = np.array([True, True, True, False])
        selected = np.array([False, True, False, False])
        for condition in "ABCD":
            mask = edge_mask(valid, selected, condition)
            for i in range(4):
                for j in range(4):
                    deleted = ((condition == "B" and i == j) or
                               (condition == "C" and i != j) or condition == "D") and selected[j]
                    self.assertEqual(mask[i, j], bool(valid[j] and not deleted))
            self.assertFalse(mask[:, 3].any())
        self.assertTrue(edge_mask(valid, selected, "C")[1, 1])
        self.assertFalse(edge_mask(valid, selected, "C")[0, 1])
        self.assertTrue(edge_mask(valid, selected, "C")[1, 0])

    def test_noop_single_patch_empty_row_and_invalid_selection(self):
        valid = np.array([True, False])
        for c in "ABCD":
            np.testing.assert_array_equal(edge_mask(valid, np.zeros(2, dtype=bool), c), [[True, False], [True, False]])
        for c in "BD":
            with self.assertRaisesRegex(ValueError, "Empty attention row"):
                edge_mask(valid, valid, c)
        with self.assertRaises(ValueError):
            edge_mask(valid, np.array([False, True]), "A")
        with self.assertRaises(ValueError):
            edge_mask(valid, valid, "unknown")


class EdgeModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        config = TimeRCDConfig()
        config.ts_config.d_model = 16
        config.ts_config.d_proj = 8
        config.ts_config.patch_size = 16
        config.ts_config.num_layers = 3
        config.ts_config.num_heads = 2
        self.model = TimeSeriesPretrainModel(config).eval()
        self.values = torch.randn(49, 1)
        self.valid = torch.ones(49, dtype=torch.bool)
        self.present = np.ones(4, dtype=bool)
        self.selected = np.array([False, True, False, False])

    def predict(self, condition, scope="all", values=None, capture=False):
        with attention_edges(self.model, self.present, self.selected, condition, scope, capture) as records:
            result = score_window(self.model, self.values if values is None else values, self.valid, "cpu")
        return result, records

    def test_sdpa_matches_explicit_softmax_and_saved_mask_hash(self):
        module = self.model.ts_encoder.transformer_encoder.layers[0].self_attn
        values = torch.randn(1, 4, 16)
        freqs = self.model.ts_encoder.rope_embedder(4)
        for c in "ABCD":
            mask = torch.as_tensor(edge_mask(self.present, self.selected, c))[None, None]
            actual = edge_attention(module, values, values, values, freqs, mask)
            def heads(x):
                return x.view(1, 4, 2, 8).transpose(1, 2)
            q = heads(module.apply_rope(module.q_proj(values), freqs))
            k = heads(module.apply_rope(module.k_proj(values), freqs))
            v = heads(module.v_proj(values))
            weights = ((q @ k.transpose(-2, -1)) / math.sqrt(8)).masked_fill(~mask, -torch.inf).softmax(-1)
            expected = module.out_proj((weights @ v).transpose(1, 2).contiguous().view(1, 4, 16))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        _, records = self.predict("C")
        expected_hash = hashlib.sha256(edge_mask(self.present, self.selected, "C").tobytes()).hexdigest()
        self.assertTrue(all(r["mask_sha256"] == expected_hash for r in records))

    def test_official_and_legacy_column_masks_match(self):
        baseline = score_window(self.model, self.values, self.valid, "cpu")
        a, _ = self.predict("A")
        d, _ = self.predict("D")
        with reference_intervention(self.model, self.present & ~self.selected):
            old_d = score_window(self.model, self.values, self.valid, "cpu")
        for kind in a:
            np.testing.assert_allclose(a[kind], baseline[kind], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(d[kind], old_d[kind], atol=1e-6, rtol=1e-5)

    def test_last_layer_preserves_qkv_and_nonselected_outputs(self):
        a, a_records = self.predict("A", capture=True)
        predictions = {}
        for c in "BCD":
            predictions[c], records = self.predict(c, "last", capture=True)
            for left, right in zip(a_records, records):
                for name in ("input", "q", "k", "v"):
                    torch.testing.assert_close(left["capture"][name], right["capture"][name], atol=0, rtol=0)
        keep = ~np.repeat(self.selected, 16)[:49]
        for kind in a:
            np.testing.assert_array_equal(a[kind][keep], predictions["B"][kind][keep])
            np.testing.assert_array_equal(predictions["C"][kind][keep], predictions["D"][kind][keep])

    def test_all_layers_block_relay_and_preserve_state(self):
        before = model_fingerprint(self.model)
        changed = self.values.clone()
        changed[16:32] += torch.linspace(1, 4, 16)[:, None]
        keep = ~np.repeat(self.selected, 16)[:49]
        results = {}
        for c in "CD":
            results[c], _ = self.predict(c)
            other, _ = self.predict(c, values=changed)
            for kind in other:
                np.testing.assert_array_equal(other[kind][keep], results[c][kind][keep])
            self.assertGreater(np.max(np.abs(other["logits"][~keep] - results[c]["logits"][~keep])), 1e-4)
        for kind in results["C"]:
            np.testing.assert_array_equal(results["C"][kind][keep], results["D"][kind][keep])
        self.assertEqual(before, model_fingerprint(self.model))
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with attention_edges(self.model, self.present, self.selected, "C", "all"):
                raise RuntimeError("forced")
        self.assertTrue(all("forward" not in l.self_attn.__dict__ for l in self.model.ts_encoder.transformer_encoder.layers))
        with self.assertRaisesRegex(RuntimeError, "calls"):
            with attention_edges(self.model, self.present, self.selected, "C", "all"):
                pass
        self.assertEqual(before, model_fingerprint(self.model))

    def test_partial_and_padding_only_patches(self):
        values = torch.cat([self.values[:33], self.values[32:33].expand(31, -1)])
        valid = torch.arange(64) < 33
        present = np.array([True, True, True, False])
        with attention_edges(self.model, present, self.selected, "C", "all"):
            first = score_window(self.model, values, valid, "cpu")
        values[48:] = 900
        with attention_edges(self.model, present, self.selected, "C", "all"):
            second = score_window(self.model, values, valid, "cpu")
        self.assertEqual(first["scores"].shape, (33,))
        np.testing.assert_array_equal(first["logits"], second["logits"])

    def test_failed_check_retains_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(CheckFailure):
                check_array(path, {"atol": 1e-6, "rtol": 1e-5}, {"test": "forced"}, np.zeros(2), np.ones(2))
            self.assertTrue((path / "checks.csv").is_file())
            self.assertTrue((path / "failed_check.npz").is_file())


if __name__ == "__main__":
    unittest.main()
