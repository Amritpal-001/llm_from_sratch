import json
from pathlib import Path

import torch

from llm_from_scratch.data.tokenizer import CharTokenizer
from llm_from_scratch.nn.gpt import GPT
from llm_from_scratch.training.generate import generate
from llm_from_scratch.utils.device import get_device

DATASET = "tinyshakespeare"  # tinyshakespeare | ecoli | swissprot | human_mito | human_chr21
DATA_DIR = Path("data") / DATASET
CHECKPOINT_PATH = Path("checkpoints2/gpt.pt")
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 20

device = get_device()
meta = json.loads((DATA_DIR / "meta.json").read_text())
tokenizer = CharTokenizer(meta["stoi"])

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model = GPT(checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

start_ids = torch.tensor([tokenizer.encode("\n")], device=device)
output_ids = generate(model, start_ids, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
print(tokenizer.decode(output_ids[0].tolist()))
