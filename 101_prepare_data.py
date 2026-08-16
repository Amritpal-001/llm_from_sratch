import argparse
import json
from pathlib import Path

import torch

from llm_from_scratch.data.download import (
    download_ecoli_genome,
    download_human_chr21,
    download_human_mitochondria,
    download_swissprot,
    download_tinyshakespeare,
)
from llm_from_scratch.data.fasta import parse_fasta
from llm_from_scratch.data.tokenizer import CharTokenizer, build_vocab

DATA_DIR = Path("data")
VAL_FRACTION = 0.1

# is_fasta: raw file needs FASTA header stripping/sequence concatenation.
# uppercase: normalize nucleotide case (drops soft-masked lowercase distinction).
# sequence_separator: joiner between records ("" for a single genome, "\n" between
# unrelated protein sequences so one doesn't run into the next).
DATASETS = {
    "tinyshakespeare": {
        "raw_filename": "tinyshakespeare.txt",
        "download": download_tinyshakespeare,
        "is_fasta": False,
    },
    "ecoli": {
        "raw_filename": "ecoli_genome.fasta",
        "download": download_ecoli_genome,
        "is_fasta": True,
        "uppercase": True,
        "sequence_separator": "",
    },
    "swissprot": {
        "raw_filename": "swissprot.fasta",
        "download": download_swissprot,
        "is_fasta": True,
        "uppercase": False,
        "sequence_separator": "\n",
    },
    "human_mito": {
        "raw_filename": "human_mitochondria.fasta",
        "download": download_human_mitochondria,
        "is_fasta": True,
        "uppercase": True,
        "sequence_separator": "",
    },
    "human_chr21": {
        "raw_filename": "human_chr21.fasta",
        "download": download_human_chr21,
        "is_fasta": True,
        "uppercase": True,
        "sequence_separator": "",
    },
}


def load_text(raw_path: Path, config: dict) -> str:
    if not config["is_fasta"]:
        return raw_path.read_text()
    sequences = parse_fasta(raw_path)
    text = config["sequence_separator"].join(sequences)
    if config["uppercase"]:
        text = text.upper()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tinyshakespeare")
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    out_dir = DATA_DIR / args.dataset
    raw_path = out_dir / config["raw_filename"]

    print(f"downloading {args.dataset} to {raw_path}")
    config["download"](raw_path)

    text = load_text(raw_path, config)
    print(text[:1000])
    print(f"corpus length: {len(text)} characters")

    stoi = build_vocab(text)
    tokenizer = CharTokenizer(stoi)
    print(f"vocab size: {tokenizer.vocab_size}")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
    split_idx = int(len(ids) * (1 - VAL_FRACTION))
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]
    print(f"train tokens: {len(train_ids)}, val tokens: {len(val_ids)}")

    torch.save(train_ids, out_dir / "train.pt")
    torch.save(val_ids, out_dir / "val.pt")
    (out_dir / "meta.json").write_text(json.dumps({"stoi": stoi, "vocab_size": tokenizer.vocab_size}))
    print(f"wrote train.pt, val.pt, meta.json to {out_dir}")


if __name__ == "__main__":
    main()
