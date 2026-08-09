import math

import torch
from torch import Tensor


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Paper A S3.2: additive mask M, M_ij = 0 if j <= i (visible) else -inf."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Paper A S2.1/S3.2: Y = softmax(QK^T / sqrt(d_k) + M) V."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
