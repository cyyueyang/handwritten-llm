import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super(MultiQueryAttention, self).__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = self.d_model // self.n_heads
        self.w_qkv = nn.Linear(d_model, (n_heads + 2) * self.head_dim, bias=False)

        self.w_out = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        mask = torch.tril(torch.ones(1, 1, max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer('mask', mask)

    def forward(self, x):
        q, k, v = self.w_qkv(x).split([self.n_heads * self.head_dim, self.head_dim, self.head_dim], dim=-1)

        bs, seq_len, dim = q.size()

        q = q.view(bs, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bs, seq_len, 1, self.head_dim).permute(0, 2, 1, 3).expand(-1, self.n_heads, -1, -1)
        v = v.view(bs, seq_len, 1, self.head_dim).permute(0, 2, 1, 3).expand(-1, self.n_heads, -1, -1)

        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float('-inf'))
        attn_score = F.softmax(attn, dim=-1)

        out = attn_score @ v
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        out = self.w_out(out)

        return out

if __name__ == "__main__":
    bs, seq_len, d_model = 8, 16, 64
    x = torch.randn(bs, seq_len, d_model)
    attention = MultiQueryAttention(d_model, n_heads=8)
    out = attention(x)
    print(out.shape)

