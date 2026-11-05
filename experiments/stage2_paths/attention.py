from contextlib import contextmanager
import hashlib
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F


def edge_mask(valid, selected, condition):
    valid, selected = np.asarray(valid), np.asarray(selected)
    if valid.ndim != 1 or valid.dtype != bool or selected.dtype != bool or valid.shape != selected.shape:
        raise ValueError("Expected aligned boolean Patch masks")
    if not valid.any() or np.any(selected & ~valid) or condition not in "ABCD" or len(condition) != 1:
        raise ValueError("Invalid Patch selection or condition")
    allowed = np.broadcast_to(valid[None], (len(valid), len(valid))).copy()
    diagonal = np.eye(len(valid), dtype=bool)
    if condition == "B":
        allowed &= ~(diagonal & selected[None])
    elif condition == "C":
        allowed &= ~(~diagonal & selected[None])
    elif condition == "D":
        allowed &= ~selected[None]
    if not allowed.any(axis=1).all():
        raise ValueError("Empty attention row: use the matched-window fallback before forward")
    return allowed


def edge_attention(module, query, key, value, freqs, allowed, capture=None):
    batch, tokens, channels = query.shape
    if batch != 1 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("This diagnostic requires one univariate window per batch")
    if allowed.shape != (1, 1, tokens, tokens) or allowed.dtype != torch.bool:
        raise ValueError("Expected explicit [1,1,query,key] boolean edges")
    q, k, v = module.q_proj(query), module.k_proj(key), module.v_proj(value)
    if capture is not None:
        capture.update({name: tensor.detach().clone() for name, tensor in
                        (("input", query), ("q", q), ("k", k), ("v", v))})
    q, k = module.apply_rope(q, freqs), module.apply_rope(k, freqs)
    def heads(x):
        return x.view(batch, tokens, module.num_heads, module.head_dim).transpose(1, 2)
    output = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=allowed, is_causal=False)
    return module.out_proj(output.transpose(1, 2).contiguous().view(batch, tokens, channels))


@contextmanager
def attention_edges(model, valid, selected, condition, scope, capture=False):
    encoder = model.ts_encoder
    if model.training or encoder.num_features != 1 or not encoder.use_rope or scope not in ("last", "all"):
        raise ValueError("Requires eval-mode univariate RoPE model and explicit layer scope")
    layers = encoder.transformer_encoder.layers
    baseline = edge_mask(valid, selected, "A")
    intervention = edge_mask(valid, selected, condition)
    device = next(model.parameters()).device
    original_valid = torch.as_tensor(valid, device=device)[None]
    records, restored = [], []
    calls = [0] * len(layers)

    def make_forward(index, edges, record):
        allowed = torch.as_tensor(edges, device=device)[None, None]
        def forward(module, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
            if query_id is not None or kv_id is not None or not torch.equal(attn_mask, original_valid):
                raise ValueError("Unexpected feature IDs or changed valid-Patch mask")
            calls[index] += 1
            return edge_attention(module, query, key, value, freqs, allowed,
                                  record["capture"] if capture else None)
        return forward

    try:
        for index, layer in enumerate(layers):
            active = scope == "all" or index == len(layers) - 1
            edges = intervention if active else baseline
            record = {"layer": index, "condition": condition if active else "A",
                      "mask_sha256": hashlib.sha256(edges.tobytes()).hexdigest(),
                      "tokens": len(valid), "allowed_edges_valid_queries": int(edges[valid].sum())}
            if capture:
                record["capture"] = {}
            records.append(record)
            module = layer.self_attn
            if "forward" in module.__dict__:
                raise ValueError("Refusing to override an existing per-instance attention adapter")
            restored.append(module)
            module.forward = MethodType(make_forward(index, edges, record), module)
        yield records
        if calls != [1] * len(layers):
            raise RuntimeError(f"Unexpected attention calls: {calls}")
    finally:
        for module in restored:
            del module.forward


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
