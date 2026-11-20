import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, max_len=256, causal=False):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.max_len = max_len
        self.causal = causal

        self.W_q = nn.Linear(d_in, d_out)
        self.W_k = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)

        if self.causal:
            causal_mask = ~torch.tril(torch.ones(max_len, max_len), diagonal=0).bool()
            self.register_buffer("causal_mask", causal_mask)
        else:
            self.register_buffer("causal_mask", None)


    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        logits = Q @ torch.transpose(K, -2, -1)

        if self.causal and self.causal_mask is not None:
            logits = logits.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))

        scaled_logits = torch.divide(logits, math.sqrt(self.d_out))
        attention = torch.softmax(scaled_logits, dim=-1)
        output = attention @ V

        return output

attn = SingleHeadSelfAttention(16, 16, causal=False)
x = torch.rand(2, 5, 16)
ref = F.scaled_dot_product_attention(attn.W_q(x), attn.W_k(x), attn.W_v(x), is_causal=False)
print(torch.allclose(attn(x), ref, atol=1e-6))

attn_causal = SingleHeadSelfAttention(16, 16, causal=True)
x = torch.rand(2, 5, 16)
ref = F.scaled_dot_product_attention(attn_causal.W_q(x), attn_causal.W_k(x), attn_causal.W_v(x), is_causal=True)
print(torch.allclose(attn_causal(x), ref, atol=1e-6))
