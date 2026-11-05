from contextlib import contextmanager
import math
from types import MethodType

import numpy as np
import torch

from experiments.stage2_paths.attention import edge_mask


def reference_attention(q, k, v, valid, selected, mode, donor_k=None, donor_v=None):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError("Expected [1, heads, tokens, head_dim] tensors")
    if mode not in ("A", "C", "V", "KV"):
        raise ValueError("Unknown intervention")
    allowed = edge_mask(valid, selected, "C" if mode == "C" else "A")
    if len(valid) != q.shape[-2]:
        raise ValueError("Patch count mismatch")
    mask = torch.as_tensor(allowed, device=q.device)[None, None]
    edges = torch.as_tensor(np.asarray(selected)[None, :] & ~np.eye(len(valid), dtype=bool), device=q.device)[None, None]
    if mode in ("V", "KV"):
        for donor in (donor_k, donor_v):
            if donor is None or donor.shape != q.shape or donor.device != q.device or donor.dtype != q.dtype or not torch.isfinite(donor).all():
                raise ValueError("Invalid donor tensor")
    logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    original_logits = logits
    if mode == "KV":
        donor_logits = q @ donor_k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        logits = torch.where(edges, donor_logits, logits)
    weights = logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)
    output = weights @ v
    if mode in ("V", "KV"):
        output = output + (weights * edges) @ (donor_v - v)
    return output, {"weights": weights, "logits": logits, "original_logits": original_logits,
                    "edges": edges, "allowed": mask}


@contextmanager
def last_reference(model, valid, selected, mode, donor=None):
    encoder = model.ts_encoder
    if model.training or encoder.num_features != 1 or not encoder.use_rope:
        raise ValueError("Requires eval-mode univariate RoPE model")
    module = encoder.transformer_encoder.layers[-1].self_attn
    if "forward" in module.__dict__:
        raise ValueError("Existing adapter must not be overwritten")
    expected = torch.as_tensor(valid, device=next(model.parameters()).device)[None]
    record, calls = {}, []

    def forward(module, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        if query_id is not None or kv_id is not None or not torch.equal(attn_mask, expected):
            raise ValueError("Unexpected IDs or validity mask")
        batch, tokens, channels = query.shape
        def heads(tensor):
            return tensor.view(batch, tokens, module.num_heads, module.head_dim).transpose(1, 2)
        q = heads(module.apply_rope(module.q_proj(query), freqs))
        k = heads(module.apply_rope(module.k_proj(key), freqs))
        v = heads(module.v_proj(value))
        source = {"k": k, "v": v} if donor == "self" else donor
        output, detail = reference_attention(q, k, v, valid, selected, mode,
                                             source["k"] if source is not None else None,
                                             source["v"] if source is not None else None)
        record.update({"input": query.detach().clone(), "q": q.detach().clone(), "k": k.detach().clone(),
                       "v": v.detach().clone(), **{name: tensor.detach().clone() for name, tensor in detail.items()}})
        calls.append(1)
        return module.out_proj(output.transpose(1, 2).contiguous().view(batch, tokens, channels))

    module.forward = MethodType(forward, module)
    try:
        yield record
        if len(calls) != 1:
            raise RuntimeError(f"Expected one final-layer call, got {len(calls)}")
    finally:
        del module.forward
