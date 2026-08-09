# LLM from Sratch

A GPT-style, decoder-only transformer built from scratch in PyTorch — attention, multi-head attention, layer norm, and linear projections are all implemented with raw tensor ops (no `nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention`, or `nn.Transformer*` inside the package itself). The model is trained on TinyShakespeare with a character-level tokenizer.

Every module is grounded in the papers under [context/](context/) — see [docs/paper_mapping.md](docs/paper_mapping.md) for exactly which paper section each file implements.

![Self-attention](docs/self_attention.png)

## Setup

Install `uv`, create the virtual environment, then install torch and this package into it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is already installed
uv venv .venv
uv pip install --python .venv/bin/python torch
uv pip install --python .venv/bin/python -e .
```

## Running the pipeline

Use `./run.sh <script>.py` for everything — it wraps the container invocation and fixes an SSL cert path issue that otherwise breaks HTTPS downloads inside the container.

1. **Prepare data** — downloads TinyShakespeare, builds the vocab, writes `data/*.pt`:
   ```bash
   ./run.sh 101_prepare_data.py
   ```
2. **Verify components** — numeric parity checks of each hand-implemented module against PyTorch's own built-ins:
   ```bash
   ./run.sh 201_verify_attention.py      # ...through 209_verify_mla.py
   ```
3. **Train** — trains the GPT model, writes `checkpoints/gpt.pt`:
   ```bash
   ./run.sh 102_train_model.py
   ```
   For a real (longer) run on a GPU node, submit it as a batch job instead of running interactively:
   ```bash
   sbatch 102_train_model.sbatch
   ```
4. **Generate** — loads the checkpoint and samples text:
   ```bash
   ./run.sh 103_generate_text.py
   ```

## Repo layout

- [src/llm_from_scratch/](src/llm_from_scratch/) — the package itself:
  - `nn/` — attention, layer norm, transformer block, GPT
  - `attention_variants/` — KV cache, GQA, MLA
  - `data/` — tokenizer, dataset
  - `training/` — config, training loop, generation
- `1xx_*.py` — pipeline entry points (data prep, train, generate)
- `2xx_verify_*.py` — smoke tests: each hand-implemented component checked against PyTorch's built-in equivalent (`nn.LayerNorm`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`) for numeric and gradient parity
- [docs/paper_mapping.md](docs/paper_mapping.md) — module-to-paper-section index

## Paper mapping

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
