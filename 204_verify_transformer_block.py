import torch

from llm_from_scratch.nn.attention import make_causal_mask
from llm_from_scratch.nn.transformer_block import TransformerBlock

torch.manual_seed(0)

d_model, num_heads, d_ff, seq_len, batch = 32, 4, 64, 10, 3
block = TransformerBlock(d_model, num_heads, d_ff)

x = torch.randn(batch, seq_len, d_model)
mask = make_causal_mask(seq_len)
out = block(x, mask=mask)
assert out.shape == x.shape
print("PASS: shape preserved")

x2 = x.clone()
t = seq_len - 1
x2[:, t, :] += 100.0  # large perturbation to the last (future-most) position

out1 = block(x, mask=mask)
out2 = block(x2, mask=mask)
assert torch.allclose(out1[:, :t, :], out2[:, :t, :], atol=1e-5), "causal leakage: future token affected earlier positions"
assert not torch.allclose(out1[:, t, :], out2[:, t, :]), "perturbed position should change its own output"
print("PASS: causal property holds (future perturbation does not leak backward)")

x = torch.randn(batch, seq_len, d_model, requires_grad=True)
out = block(x, mask=mask)
out.sum().backward()
assert torch.isfinite(out).all(), "NaN/Inf in forward output"
assert torch.isfinite(x.grad).all(), "NaN/Inf in input gradient"
for name, p in block.named_parameters():
    assert torch.isfinite(p.grad).all(), f"NaN/Inf in gradient of {name}"
print("PASS: no NaN/Inf in forward output or gradients")

print("ALL PASS: 204_verify_transformer_block.py")
