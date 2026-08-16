import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path("logs")
OUT_PATH = Path("docs/loss_comparison.png")

DATASETS = ["tinyshakespeare", "ecoli", "swissprot", "human_mito", "human_chr21"]
ATTENTION_TYPES = ["dot_product", "scaled_dot_product", "additive"]
COLORS = {
    "dot_product": "#2a78d6",
    "scaled_dot_product": "#eb6834",
    "additive": "#1baf7a",
}

STEP_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def parse_log(path: Path):
    steps, train_loss, val_loss = [], [], []
    for line in path.read_text().splitlines():
        match = STEP_RE.match(line)
        if match:
            steps.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
    return steps, train_loss, val_loss


fig, axes = plt.subplots(1, len(DATASETS), figsize=(4 * len(DATASETS), 4), sharey=False)

for ax, dataset in zip(axes, DATASETS):
    for attention_type in ATTENTION_TYPES:
        log_path = LOG_DIR / f"{attention_type}_{dataset}_5000.md"
        if not log_path.exists() or log_path.stat().st_size == 0:
            print(f"skipping missing/empty log: {log_path}")
            continue
        steps, train_loss, val_loss = parse_log(log_path)
        if not steps:
            print(f"no parsed steps in: {log_path}")
            continue
        color = COLORS[attention_type]
        ax.plot(steps, train_loss, color=color, linestyle="-", linewidth=2, alpha=0.5)
        ax.plot(steps, val_loss, color=color, linestyle="--", linewidth=2, alpha=0.5)
    ax.set_title(dataset)
    ax.set_xlabel("step")
    ax.set_yscale("log")

    
axes[0].set_ylabel("loss [log scale]")

color_handles = [
    plt.Line2D([0], [0], color=COLORS[attention_type], linewidth=2, label=attention_type)
    for attention_type in ATTENTION_TYPES
]
style_handles = [
    plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="train"),
    plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="val"),
]
fig.legend(handles=color_handles + style_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

fig.tight_layout()
OUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"saved plot to {OUT_PATH}")
