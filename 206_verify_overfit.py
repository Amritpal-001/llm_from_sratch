import math

import torch

from llm_from_scratch.nn.gpt import GPT
from llm_from_scratch.training.config import GPTConfig
from llm_from_scratch.training.trainer import compute_loss
from llm_from_scratch.utils.seed import set_seed

set_seed(0)

vocab_size, block_size = 10, 8
config = GPTConfig(vocab_size=vocab_size, d_model=32, num_heads=4, num_layers=2, d_ff=64, max_seq_len=block_size)
model = GPT(config)

sequences = torch.randint(0, vocab_size, (6, block_size + 1))
x, y = sequences[:, :-1], sequences[:, 1:]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

initial_loss = compute_loss(model, x, y).item()
final_loss = initial_loss
for step in range(300):
    loss = compute_loss(model, x, y)
    optimizer.zero_grad()
    loss.backward()
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name} at step {step}"
    optimizer.step()
    final_loss = loss.item()
    assert math.isfinite(final_loss), f"NaN/Inf loss at step {step}"

relative_drop = (initial_loss - final_loss) / initial_loss
print(f"initial loss {initial_loss:.4f}, final loss {final_loss:.4f}, relative drop {relative_drop:.2%}")
assert relative_drop > 0.9, f"expected >90% relative loss drop, got {relative_drop:.2%}"

print("ALL PASS: 206_verify_overfit.py")
