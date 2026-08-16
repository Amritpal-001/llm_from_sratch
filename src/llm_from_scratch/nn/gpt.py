import math

import torch.nn as nn
from torch import Tensor

from llm_from_scratch.nn.attention import make_causal_mask
from llm_from_scratch.nn.layer_norm import LayerNorm
from llm_from_scratch.nn.linear import Linear
from llm_from_scratch.nn.positional_encoding import SinusoidalPositionalEncoding
from llm_from_scratch.nn.transformer_block import TransformerBlock
from llm_from_scratch.training.config import GPTConfig


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.num_heads, config.d_ff, attention_type=config.attention_type)
                for _ in range(config.num_layers)
            ]
        )
        self.ln_final = LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self.apply(self._init_embedding)
        # GPT-2-style residual-stream scaling (engineering choice, not from Paper A/B).
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_embedding(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        seq_len = idx.shape[1]
        mask = make_causal_mask(seq_len, device=idx.device)
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_final(x)
        return self.lm_head(x)
