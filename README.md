# llm_from_sratch

Attention, transformer, and GPT implemented from scratch in torch (raw tensor ops for `Linear`/`LayerNorm`/attention — no `nn.Linear`/`nn.LayerNorm`/`nn.MultiheadAttention`/`nn.Transformer*` in the package itself). Decoder-only, GPT-style, trained on TinyShakespeare with a char-level tokenizer. Grounded in the papers under `context/` — see [docs/paper_mapping.md](docs/paper_mapping.md) for which section implements what.

## Setup (PACE)

`module load pytorch/...` on this cluster wraps `python` in an Apptainer container — `torch` must **never** be added as a pip/uv dependency here, or it silently downloads a mismatched build into `~/.local` and can break unrelated projects. Instead this repo uses an isolated `.venv` with `--system-site-packages` so it inherits the container's own torch:

```bash
module load cuda/12.1.1 pytorch/2.1.0
python -m venv --system-site-packages --without-pip .venv   # ensurepip isn't available in the container
apptainer exec --nv /usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif \
    ~/.local/bin/uv pip install --python .venv/bin/python -e .
```

After that, use `./run.sh <script>.py` for everything — it wraps the container invocation (and fixes an SSL cert path issue that otherwise breaks HTTPS downloads inside the container):

```bash
./run.sh 101_prepare_data.py          # downloads TinyShakespeare, builds vocab, writes data/*.pt
./run.sh 201_verify_attention.py      # ...through 209_verify_mla.py: numeric parity checks vs PyTorch built-ins
./run.sh 102_train_model.py           # trains GPT, writes checkpoints/gpt.pt
./run.sh 103_generate_text.py         # loads the checkpoint, samples text
```

For a real (longer) training run on a GPU node, submit the batch job rather than running it interactively:

```bash
sbatch 102_train_model.sbatch
```

## Layout

- `src/llm_from_scratch/` — the package: `nn/` (attention, layer norm, transformer block, GPT), `attention_variants/` (KV cache, GQA, MLA), `data/` (tokenizer, dataset), `training/` (config, training loop, generation).
- `1xx_*.py` — pipeline entry points (data prep, train, generate).
- `2xx_verify_*.py` — smoke tests: each hand-implemented component checked against PyTorch's own built-in (`nn.LayerNorm`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`) for numeric and gradient parity.
- `docs/paper_mapping.md` — module-to-paper-section index.
