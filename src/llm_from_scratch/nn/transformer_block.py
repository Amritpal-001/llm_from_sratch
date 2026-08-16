import torch.nn as nn
from torch import Tensor

from llm_from_scratch.nn.feed_forward import PositionwiseFeedForward
from llm_from_scratch.nn.layer_norm import LayerNorm
from llm_from_scratch.nn.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Paper A S3.2/S3.3: Pre-LN causal self-attention + residual, Pre-LN FFN + residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, attention_type: str = "scaled_dot_product") -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, attention_type=attention_type)
        self.ln2 = LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x
