import json
from pathlib import Path

import torch

from llm_from_scratch.data.download import download_tinyshakespeare
from llm_from_scratch.data.tokenizer import CharTokenizer, build_vocab

DATA_DIR = Path("data")
VAL_FRACTION = 0.1

raw_path = DATA_DIR / "tinyshakespeare.txt"
print(f"downloading tinyshakespeare to {raw_path}")
download_tinyshakespeare(raw_path)

text = raw_path.read_text()
print(text[:1000])
print(f"corpus length: {len(text)} characters")

stoi = build_vocab(text)
tokenizer = CharTokenizer(stoi)
print(f"vocab size: {tokenizer.vocab_size}")

ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
split_idx = int(len(ids) * (1 - VAL_FRACTION))
train_ids, val_ids = ids[:split_idx], ids[split_idx:]
print(f"train tokens: {len(train_ids)}, val tokens: {len(val_ids)}")

torch.save(train_ids, DATA_DIR / "train.pt")
torch.save(val_ids, DATA_DIR / "val.pt")
(DATA_DIR / "meta.json").write_text(json.dumps({"stoi": stoi, "vocab_size": tokenizer.vocab_size}))
print(f"wrote train.pt, val.pt, meta.json to {DATA_DIR}")
