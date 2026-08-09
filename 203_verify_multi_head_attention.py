import torch
import torch.nn as nn

from llm_from_scratch.nn.attention import make_causal_mask
from llm_from_scratch.nn.multi_head_attention import MultiHeadAttention

torch.manual_seed(0)

d_model, num_heads, seq_len, batch = 32, 4, 6, 3

mha_ref = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True).eval()
mha_ours = MultiHeadAttention(d_model, num_heads)

with torch.no_grad():
    mha_ours.in_proj.weight.copy_(mha_ref.in_proj_weight)
    mha_ours.in_proj.bias.copy_(mha_ref.in_proj_bias)
    mha_ours.out_proj.weight.copy_(mha_ref.out_proj.weight)
    mha_ours.out_proj.bias.copy_(mha_ref.out_proj.bias)

x = torch.randn(batch, seq_len, d_model)

out_ours = mha_ours(x)
out_ref, _ = mha_ref(x, x, x, need_weights=False)
assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-4), f"non-causal mismatch: {(out_ours - out_ref).abs().max()}"
print("PASS: forward matches nn.MultiheadAttention (non-causal)")

mask = make_causal_mask(seq_len)
out_ours_c = mha_ours(x, mask=mask)
out_ref_c, _ = mha_ref(x, x, x, attn_mask=mask, need_weights=False)
assert torch.allclose(out_ours_c, out_ref_c, atol=1e-5, rtol=1e-4), f"causal mismatch: {(out_ours_c - out_ref_c).abs().max()}"
print("PASS: forward matches nn.MultiheadAttention (causal)")

x64, mha_ours64, mha_ref64 = x.double(), mha_ours.double(), mha_ref.double()
out_ours64 = mha_ours64(x64)
out_ref64, _ = mha_ref64(x64, x64, x64, need_weights=False)
assert torch.allclose(out_ours64, out_ref64, atol=1e-9), f"float64 mismatch: {(out_ours64 - out_ref64).abs().max()}"
print("PASS: forward matches nn.MultiheadAttention (float64, tight tolerance)")

mha_ours, mha_ref = mha_ours.float(), mha_ref.float()
x_ours = x.clone().requires_grad_(True)
x_ref = x.clone().requires_grad_(True)
out_ours = mha_ours(x_ours)
out_ref, _ = mha_ref(x_ref, x_ref, x_ref, need_weights=False)
out_ours.sum().backward()
out_ref.sum().backward()

assert torch.allclose(x_ours.grad, x_ref.grad, atol=1e-4, rtol=1e-3), "input grad mismatch"
assert torch.allclose(mha_ours.in_proj.weight.grad, mha_ref.in_proj_weight.grad, atol=1e-4, rtol=1e-3), "in_proj weight grad mismatch"
assert torch.allclose(mha_ours.out_proj.weight.grad, mha_ref.out_proj.weight.grad, atol=1e-4, rtol=1e-3), "out_proj weight grad mismatch"
print("PASS: backward gradients match (input, in_proj, out_proj)")

print("ALL PASS: 203_verify_multi_head_attention.py")
