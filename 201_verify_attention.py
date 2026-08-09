import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_from_scratch.nn.attention import make_causal_mask, scaled_dot_product_attention

torch.manual_seed(0)

# 1. Hand-computed fixed example (independent derivation via math.exp, not torch.softmax).
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
out = scaled_dot_product_attention(q, k, v)

s = 1 / math.sqrt(2)


def softmax2(a: float, b: float) -> tuple[float, float]:
    ea, eb = math.exp(a), math.exp(b)
    return ea / (ea + eb), eb / (ea + eb)


w00, w01 = softmax2(s, 0.0)
w10, w11 = softmax2(0.0, s)
expected = torch.tensor([[[w00 * 1 + w01 * 3, w00 * 2 + w01 * 4], [w10 * 1 + w11 * 3, w10 * 2 + w11 * 4]]])
assert torch.allclose(out, expected, atol=1e-6), f"hand-computed mismatch: {out} vs {expected}"
print("PASS: hand-computed scaled_dot_product_attention")

# 2. vs F.scaled_dot_product_attention (oracle), non-causal and causal.
q = torch.randn(2, 3, 8, 16)
k = torch.randn(2, 3, 8, 16)
v = torch.randn(2, 3, 8, 16)

out_ours = scaled_dot_product_attention(q, k, v)
out_ref = F.scaled_dot_product_attention(q, k, v)
assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-4)
print("PASS: matches F.scaled_dot_product_attention (non-causal)")

mask = make_causal_mask(8)
out_ours_causal = scaled_dot_product_attention(q, k, v, mask=mask)
out_ref_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(out_ours_causal, out_ref_causal, atol=1e-5, rtol=1e-4)
print("PASS: matches F.scaled_dot_product_attention (causal)")

# 3. vs nn.MultiheadAttention(num_heads=1) via explicit weight copy.
d_model, seq_len, batch = 16, 5, 2
mha = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True).eval()
x = torch.randn(batch, seq_len, d_model)

Wq, Wk, Wv = mha.in_proj_weight.chunk(3, dim=0)
bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
q = x @ Wq.T + bq
k = x @ Wk.T + bk
v = x @ Wv.T + bv

attn_out = scaled_dot_product_attention(q, k, v)
out_ours = attn_out @ mha.out_proj.weight.T + mha.out_proj.bias
out_ref, _ = mha(x, x, x, need_weights=False)
assert torch.allclose(out_ours, out_ref, atol=1e-4, rtol=1e-4)
print("PASS: single-head matches nn.MultiheadAttention")

causal_mask = make_causal_mask(seq_len)
attn_out_masked = scaled_dot_product_attention(q, k, v, mask=causal_mask)
out_ours_masked = attn_out_masked @ mha.out_proj.weight.T + mha.out_proj.bias
out_ref_masked, _ = mha(x, x, x, attn_mask=causal_mask, need_weights=False)
assert torch.allclose(out_ours_masked, out_ref_masked, atol=1e-4, rtol=1e-4)
print("PASS: single-head causal-masked matches nn.MultiheadAttention")

# 4. make_causal_mask shape/values, broadcast across batch and heads.
cm = make_causal_mask(4)
assert cm.shape == (4, 4)
upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
assert torch.all(torch.isinf(cm[upper])) and torch.all(cm[upper] < 0)
assert torch.all(cm[~upper] == 0)
scores = torch.randn(2, 3, 4, 4)
masked = scores + cm
assert torch.all(torch.isinf(masked[:, :, 0, 1:]))
assert torch.all(torch.isfinite(masked[:, :, 3, :]))
print("PASS: make_causal_mask shape/values/broadcast (checked across all batches/heads)")

print("ALL PASS: 201_verify_attention.py")
