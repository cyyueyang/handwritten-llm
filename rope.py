import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(
            self,
            dim: int,
            max_seq_len: int,
            base: float = 10000.0
    ):
        super(RoPE, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.t = torch.arange(0, max_seq_len)

        freqs = torch.outer(self.t, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def rotate_half(self, x):
        dim = x.size(-1)
        half_dim = dim // 2
        x0 = x[..., :half_dim]
        x1 = x[..., half_dim:]
        return torch.cat([-x1, x0], dim=-1)

    def forward(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape

        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)

        return cos * x + sin * self.rotate_half(x)

if __name__ == '__main__':
    bs, num_heads, seq_len, head_dim = 4, 8, 16, 64
    rope = RoPE(64, 128, 10000.0)
    x = torch.randn(bs, num_heads, seq_len, head_dim)
    y = rope(x)
    print(y.shape)