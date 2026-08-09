import torch
import torch.nn as nn

from llm_from_scratch.nn.layer_norm import LayerNorm

torch.manual_seed(0)

d_model = 32
ln_ref = nn.LayerNorm(d_model, eps=1e-5).eval()
ln_ours = LayerNorm(d_model, eps=1e-5)

with torch.no_grad():
    ln_ours.gamma.copy_(ln_ref.weight)
    ln_ours.beta.copy_(ln_ref.bias)

x = torch.randn(4, 10, d_model)
out_ref = ln_ref(x)
out_ours = ln_ours(x)
assert torch.allclose(out_ours, out_ref, atol=1e-6, rtol=1e-4), f"mismatch: max diff {(out_ours - out_ref).abs().max()}"
print("PASS: LayerNorm matches nn.LayerNorm (float32)")

x64 = x.double()
ln_ref64 = ln_ref.double()
ln_ours64 = ln_ours.double()
out_ref64 = ln_ref64(x64)
out_ours64 = ln_ours64(x64)
assert torch.allclose(out_ours64, out_ref64, atol=1e-9), f"float64 mismatch: max diff {(out_ours64 - out_ref64).abs().max()}"
print("PASS: LayerNorm matches nn.LayerNorm (float64, tight tolerance)")

print("ALL PASS: 202_verify_layer_norm.py")
