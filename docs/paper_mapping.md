# Paper mapping

"Paper A" = Serret 2026, arXiv:2604.00965v1, *Understanding Transformers and Attention Mechanisms: An Introduction for Applied Mathematicians*.
"Paper B" = Soydaner 2022, *Neural Computing and Applications*, *Attention Mechanism in Neural Networks: Where It Comes and Where It Goes* — background/history only, not a source for any implementation below.

| Module | Paper section | Concept implemented |
|---|---|---|
| `nn/attention.py` | Paper A S2.1, S3.2 | `Y = softmax(QK^T / sqrt(d_k) + M) V`; causal additive mask `M_ij = 0` if visible else `-inf` |
| `nn/multi_head_attention.py` | Paper A S2.2 | Per-head `W^Q_h, W^K_h, W^V_h` via combined `in_proj`; heads concatenated through `W^O` (`out_proj`) |
| `nn/transformer_block.py` | Paper A S3.2, S3.3 | Decoder block: causal self-attention + residual, feed-forward + residual. Pre-LN variant (Paper A S3.3 lists this as an alternative to its primary Post-LN diagram) |
| `nn/linear.py` | Paper A S2.1 | Explicit `W^Q/W^K/W^V/W^O` linear projection, implemented via raw matmul (no `nn.Linear`) |
| `nn/layer_norm.py` | not in Paper A | Ba et al. 2016, *Layer Normalization* |
| `nn/positional_encoding.py` | not in Paper A | Vaswani et al. 2017, *Attention Is All You Need* — fixed sin/cos positional encoding |
| `nn/feed_forward.py` | Paper A S3 (block diagrams) | Position-wise feed-forward sublayer |
| `nn/gpt.py` | Paper A S3.2 (decoder-only) | Stacked decoder blocks; weight tying and GPT-2-style init are engineering choices not sourced from either paper |
| `attention_variants/kv_cache.py` (Phase 2) | Paper A S4 | KV caching: `O(N^2 d)` full recompute vs `O(N d)` incremental decode |
| `attention_variants/gqa.py` (Phase 2) | Paper A S4.1 | Grouped/Multi-Query Attention |
| `attention_variants/mla.py` (Phase 2) | Paper A S4.2 | Multi-Head Latent Attention (low-rank KV compression) |
