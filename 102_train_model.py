import argparse
import json
from pathlib import Path

import torch

from llm_from_scratch.data.dataset import CharDataset
from llm_from_scratch.nn.gpt import GPT
from llm_from_scratch.training.config import GPTConfig, TrainConfig
from llm_from_scratch.training.trainer import train_model
from llm_from_scratch.utils.device import get_device
from llm_from_scratch.utils.seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--attention-type",
    dest="attention_type",
    default="additive",
    choices=["scaled_dot_product", "additive", "dot_product"],
)
parser.add_argument(
    "--dataset",
    dest="dataset",
    default="ecoli",
    choices=["tinyshakespeare", "ecoli", "swissprot", "human_mito", "human_chr21"],
)
args = parser.parse_args()

DATA_DIR = Path("data") / args.dataset
CHECKPOINT_PATH = Path("checkpoints/gpt.pt")

set_seed(0)
device = get_device()
print(f"using device: {device}")

meta = json.loads((DATA_DIR / "meta.json").read_text())
train_data = torch.load(DATA_DIR / "train.pt")
val_data = torch.load(DATA_DIR / "val.pt")

train_config = TrainConfig()
train_dataset = CharDataset(train_data, train_config.block_size)
val_dataset = CharDataset(val_data, train_config.block_size)

gpt_config = GPTConfig(
    vocab_size=meta["vocab_size"], max_seq_len=train_config.block_size, attention_type=args.attention_type
)
model = GPT(gpt_config)
print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

train_model(model, train_dataset, val_dataset, train_config, device, checkpoint_path=CHECKPOINT_PATH)
print(f"saved checkpoint to {CHECKPOINT_PATH}")
