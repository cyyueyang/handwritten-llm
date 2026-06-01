import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len=2048):
        super(GroupQueryAttention, self).__init__()

        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.w_qkv = nn.Linear(d_model, (n_heads + 2 * n_kv_heads) * self.head_dim, bias=False)
        self.w_out = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        mask = torch.tril(torch.ones(1, 1, max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer('mask', mask)

    def forward(self, x):
        q, k, v = self.w_qkv(x).split(
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1
        )
        bs, seq_len, _ = q.size()

        group = self.n_heads // self.n_kv_heads

        q = q.view(bs, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bs, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
        v = v.view(bs, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3).repeat_interleave(group, dim=1)

        attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float('-inf'))
        attn_score = F.softmax(attn, dim=-1)

        out = attn_score @ v
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        out = self.w_out(out)

        return out

if __name__ == '__main__':
    bs, seq_len, d_model = 8, 16, 64
    attention = GroupQueryAttention(d_model=64, n_heads=8, n_kv_heads=4)

    x = torch.ones(bs, seq_len, d_model)
    y = attention(x)
    print(y.size())