import torch.nn as nn
from torch import Tensor

from llm_from_scratch.nn.attention import AdditiveAttention, dot_product_attention, scaled_dot_product_attention
from llm_from_scratch.nn.linear import Linear


class MultiHeadAttention(nn.Module):
    """Paper A S2.2: per-head W^Q_h, W^K_h, W^V_h; heads concatenated through W^O."""

    def __init__(self, d_model: int, num_heads: int, attention_type: str = "scaled_dot_product") -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.in_proj = Linear(d_model, 3 * d_model)
        self.out_proj = Linear(d_model, d_model)

        if attention_type == "scaled_dot_product":
            self.attn_fn = scaled_dot_product_attention
        elif attention_type == "dot_product":
            self.attn_fn = dot_product_attention
        elif attention_type == "additive":
            self.attn_fn = AdditiveAttention(self.head_dim)
        else:
            raise ValueError(f"unknown attention_type: {attention_type!r}")

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, d_model = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = self.attn_fn(q, k, v, mask=mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn_out)
