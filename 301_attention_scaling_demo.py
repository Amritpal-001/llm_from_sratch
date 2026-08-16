from pathlib import Path

import matplotlib.pyplot as plt

from src.llm_from_scratch.analysis.attention_scaling import measure_attention_stats
from src.llm_from_scratch.utils.seed import set_seed

set_seed(0)

D_K_VALUES = [4, 8, 16, 32, 64] #, 128, 256, 512, 1024]
OUT_PATH = Path("docs/attention_scaling_demo.png")
NAMES = ["dot_product", "scaled_dot_product", "additive"]

stats = {d_k: measure_attention_stats(d_k) for d_k in D_K_VALUES}

from tqdm import tqdm
for d_k, s in tqdm(stats.items(), desc="Attention scaling stats"):
    row = "  ".join(f"{name}: std={v['score_std']:.2f} entropy={v['entropy']:.2f} grad_norm={v['grad_norm']:.2e}" for name, v in s.items())
    print(f"d_k={d_k:>4}  {row}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name in NAMES:
    axes[0].plot(D_K_VALUES, [stats[d][name]["score_std"] for d in D_K_VALUES], marker="o", label=name)
    axes[1].plot(D_K_VALUES, [stats[d][name]["entropy"] for d in D_K_VALUES], marker="o", label=name)
    axes[2].plot(D_K_VALUES, [stats[d][name]["grad_norm"] for d in D_K_VALUES], marker="o", label=name)

axes[0].set(xscale="log", xlabel="d_k", ylabel="std of pre-softmax scores", title="Score magnitude")
axes[1].set(xscale="log", xlabel="d_k", ylabel="softmax entropy (nats)", title="Attention weight entropy")
axes[2].set(xscale="log", yscale="log", xlabel="d_k", ylabel="||d(output)/d(q)||", title="Gradient w.r.t. query")
for ax in axes:
    ax.legend()
fig.tight_layout()

OUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
print(f"saved plot to {OUT_PATH}")
