from contextlib import contextmanager
import hashlib
from types import MethodType

import torch

from experiments.stage2_paths.attention import edge_attention, edge_mask


@contextmanager
def attention_layers(model, valid, selected, active_layers, capture=False):
    encoder = model.ts_encoder
    layers = encoder.transformer_encoder.layers
    active = tuple(active_layers)
    if any(type(i) is not int or not 0 <= i < len(layers) for i in active) or len(set(active)) != len(active):
        raise ValueError("Active layers must be unique integer indices within the model")
    if model.training or encoder.num_features != 1 or not encoder.use_rope:
        raise ValueError("Requires an eval-mode univariate RoPE model")
    masks = {c: edge_mask(valid, selected, c) for c in ("A", "C")}
    device = next(model.parameters()).device
    original_valid = torch.as_tensor(valid, device=device)[None]
    records, restored = [], []
    calls = [0] * len(layers)

    def make_forward(index, edges, record):
        allowed = torch.as_tensor(edges, device=device)[None, None]
        def forward(module, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
            if query_id is not None or kv_id is not None or not torch.equal(attn_mask, original_valid):
                raise ValueError("Unexpected feature IDs or valid-Patch mask")
            calls[index] += 1
            return edge_attention(module, query, key, value, freqs, allowed, record["capture"] if capture else None)
        return forward

    try:
        for index, layer in enumerate(layers):
            condition = "C" if index in active else "A"
            edges = masks[condition]
            record = {"layer": index, "condition": condition,
                      "mask_sha256": hashlib.sha256(edges.tobytes()).hexdigest(),
                      "allowed_edges_valid_queries": int(edges[valid].sum())}
            if capture:
                record["capture"] = {}
            records.append(record)
            module = layer.self_attn
            if "forward" in module.__dict__:
                raise ValueError("Refusing to override an existing attention adapter")
            restored.append(module)
            module.forward = MethodType(make_forward(index, edges, record), module)
        yield records
        if calls != [1] * len(layers):
            raise RuntimeError(f"Unexpected attention calls: {calls}")
    finally:
        for module in restored:
            del module.forward


def switches(layers, expected_layers):
    if layers != [expected_layers - 2, expected_layers - 1] or any(type(i) is not int for i in layers):
        raise ValueError("Protocol requires the final two layers in ascending order")
    return {"00": [], "10": layers[:1], "01": layers[1:], "11": layers[:]}
