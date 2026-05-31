import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(1, 1, max_len, max_len, dtype=torch.bool))
        self.register_buffer('mask', mask)

    def forward(self, x):
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        bs, seq_len, d_model = q.size()
        q = q.view(bs, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) / math.sqrt(self.d_model // self.n_heads)
        attn = attn.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float('-inf'))
        attn_score = attn.softmax(dim=-1)

        out = attn_score @ v
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        out = self.w_out(out)
        return out


if __name__ == '__main__':
    bs, seq_len, d_model = 8, 16, 64

    attention = MultiHeadedAttention(d_model=64, n_heads=8)

    x = torch.randn(bs, seq_len, d_model)
    y = attention(x)
    print(y.size())

