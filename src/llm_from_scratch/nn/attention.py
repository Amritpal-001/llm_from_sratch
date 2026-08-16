import math

import torch
import torch.nn as nn
from torch import Tensor

from llm_from_scratch.nn.linear import Linear


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


def dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Paper C S3.2.1: dot-product attention without the 1/sqrt(d_k) scale.

    For large d_k the unscaled dot products grow large in magnitude, which pushes softmax into
    its small-gradient regions; scaled_dot_product_attention exists to counteract exactly this.
    """
    scores = q @ k.transpose(-2, -1)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


class AdditiveAttention(nn.Module):
    """Paper C S3.2.1: Bahdanau-style additive attention, e_ij = v^T tanh(W_q q_i + W_k k_j).

    Unlike the dot-product variants above, the compatibility function is a learned single-hidden-
    layer feed-forward network, so (unlike the pure functions above) this holds its own parameters.
    """

    def __init__(self, d_k: int, d_hidden: int | None = None) -> None:
        super().__init__()
        d_hidden = d_hidden or d_k
        self.w_q = Linear(d_k, d_hidden, bias=False)
        self.w_k = Linear(d_k, d_hidden, bias=False)
        self.v = Linear(d_hidden, 1, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        energy = torch.tanh(self.w_q(q).unsqueeze(-2) + self.w_k(k).unsqueeze(-3))
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        return weights @ v


def cross_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Paper C S3.2.3: encoder-decoder cross-attention, Q from the decoder, K/V from the encoder.

    Mechanically identical to scaled_dot_product_attention: q and k/v may come from different
    sequences (even different lengths), and no causal mask is needed since the decoder attends to
    the whole encoder output. Kept here for understanding only — this project is decoder-only, so
    there is no encoder stack to source K/V from and nothing below wires this in.
    """
    return scaled_dot_product_attention(q, k, v, mask=mask)