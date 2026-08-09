from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256


@dataclass
class TrainConfig:
    batch_size: int = 64
    block_size: int = 256
    learning_rate: float = 3e-4
    max_steps: int = 500
    eval_interval: int = 50
    eval_steps: int = 50
