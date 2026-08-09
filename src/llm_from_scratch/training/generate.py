import torch
from torch import Tensor

from llm_from_scratch.nn.gpt import GPT


@torch.no_grad()
def generate(model: GPT, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
    model.eval()
    max_seq_len = model.config.max_seq_len
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits = model(idx_cond)[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
