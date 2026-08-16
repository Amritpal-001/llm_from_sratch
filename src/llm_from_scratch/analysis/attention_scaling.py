import torch
from torch import Tensor

from llm_from_scratch.nn.attention import AdditiveAttention, dot_product_attention, scaled_dot_product_attention


def _entropy(scores: Tensor) -> float:
    """Mean softmax entropy (nats) over the last dim; low entropy = weight collapsed onto one key."""
    weights = torch.softmax(scores, dim=-1)
    return -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1).mean().item()


def _grad_norm(attn_fn, q: Tensor, k: Tensor, v: Tensor) -> float:
    q = q.clone().requires_grad_(True)
    attn_fn(q, k, v).sum().backward()
    return q.grad.norm().item()


def measure_attention_stats(d_k: int, seq_len: int = 64, num_trials: int = 2000) -> dict[str, dict[str, float]]:
    """Paper C S3.2.1: for large d_k, unscaled dot products grow large in magnitude, pushing softmax
    into small-gradient regions; scaling (or additive attention's bounded tanh) counteracts this.

    Samples random q ~ N(0, 1) (one query per trial) and k, v ~ N(0, 1) (seq_len keys/values per
    trial), then compares dot_product_attention, scaled_dot_product_attention, and AdditiveAttention
    on: pre-softmax score std, softmax entropy, and gradient norm of the output w.r.t. q.
    """
    q = torch.randn(num_trials, 1, d_k)
    k = torch.randn(num_trials, seq_len, d_k)
    v = torch.randn(num_trials, seq_len, d_k)
    additive = AdditiveAttention(d_k)

    raw_scores = (q @ k.transpose(-2, -1)).squeeze(1)
    energy = additive.v(torch.tanh(additive.w_q(q).unsqueeze(-2) + additive.w_k(k).unsqueeze(-3)))
    additive_scores = energy.squeeze(-1).squeeze(1)

    return {
        "dot_product": {
            "score_std": raw_scores.std().item(),
            "entropy": _entropy(raw_scores),
            "grad_norm": _grad_norm(dot_product_attention, q, k, v),
        },
        "scaled_dot_product": {
            "score_std": (raw_scores / d_k**0.5).std().item(),
            "entropy": _entropy(raw_scores / d_k**0.5),
            "grad_norm": _grad_norm(scaled_dot_product_attention, q, k, v),
        },
        "additive": {
            "score_std": additive_scores.std().item(),
            "entropy": _entropy(additive_scores),
            "grad_norm": _grad_norm(additive, q, k, v),
        },
    }
