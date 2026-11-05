import unittest

import numpy as np
import torch

from experiments.stage2_isolated.micro_audit import TinyAttention, TinyModel, manual_layer
from experiments.stage2_paths.attention import attention_edges


class MicroTests(unittest.TestCase):
    def test_manual_uniform_rows(self):
        module = TinyAttention(0)
        with torch.no_grad():
            module.q_proj.weight.zero_()
            module.k_proj.weight.zero_()
            module.v_proj.weight.copy_(torch.eye(2))
            module.out_proj.weight.copy_(torch.eye(2))
        x = np.array([[1., 2.], [3., 4.], [5., 6.]])
        mask = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool)
        output, probabilities = manual_layer(x, module, mask)
        np.testing.assert_allclose(output, x + [3., 4.], atol=1e-12)
        np.testing.assert_allclose(probabilities, [[.5, 0, .5], [1 / 3, 1 / 3, 1 / 3], [.5, 0, .5]], atol=1e-12)

    def test_adapter_restores_modules(self):
        model = TinyModel().eval()
        mask = np.ones(3, dtype=bool)
        with torch.inference_mode(), attention_edges(model, mask, np.array([False, True, False]), "C", "all"):
            result = model(torch.ones((1, 3, 2), dtype=torch.float64), torch.tensor(mask[None]))
        self.assertEqual(len(result), 2)
        for layer in model.ts_encoder.transformer_encoder.layers:
            self.assertNotIn("forward", layer.self_attn.__dict__)


if __name__ == "__main__":
    unittest.main()
