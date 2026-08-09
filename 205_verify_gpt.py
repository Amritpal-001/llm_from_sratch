import torch

from llm_from_scratch.nn.gpt import GPT
from llm_from_scratch.training.config import GPTConfig

torch.manual_seed(0)

config = GPTConfig(vocab_size=65, d_model=32, num_heads=4, num_layers=2, d_ff=64, max_seq_len=32)
model = GPT(config)

batch, seq_len = 3, 16
idx = torch.randint(0, config.vocab_size, (batch, seq_len))
logits = model(idx)
assert logits.shape == (batch, seq_len, config.vocab_size)
print("PASS: forward output shape correct")

assert model.lm_head.weight is model.token_embedding.weight
assert id(model.lm_head.weight) == id(model.token_embedding.weight)
print("PASS: lm_head and token_embedding weights are tied (same object)")

logits.sum().backward()
assert torch.isfinite(logits).all()
for name, p in model.named_parameters():
    assert p.grad is not None, f"{name} has no gradient"
    assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"
print("PASS: no NaN/Inf in forward output or gradients")

print("ALL PASS: 205_verify_gpt.py")
