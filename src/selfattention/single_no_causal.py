import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttentionNoCausal(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.W_q = nn.Linear(d_in, d_out)
        self.W_k = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)

    def forward(self, x):
        # batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        logits = Q @ torch.transpose(K, -2, -1)
        scaled_logits = torch.divide(logits, math.sqrt(self.d_out))
        attention = torch.softmax(scaled_logits, dim=-1)
        output = attention @ V

        return output

attn = SingleHeadSelfAttention(16, 16)
x = torch.rand(2, 5, 16)
ref = F.scaled_dot_product_attention(attn.W_q(x), attn.W_k(x), attn.W_v(x), is_causal=False)
print(torch.allclose(attn(x), ref, atol=1e-6))
