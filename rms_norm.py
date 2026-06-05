import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()

        self.dim = dim
        self.eps = eps

        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_normed = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x_normed * self.w

if __name__ == "__main__":
    bs, seq_len, dim = 4, 8, 64
    x = torch.randn(bs, seq_len, dim)

    official_model = nn.RMSNorm(normalized_shape=64, eps=1e-6)
    self_model = RMSNorm(dim=64, eps=1e-6)

    out1 = official_model(x)
    out2 = self_model(x)

    forward_diff = (out1 - out2).abs().max().item()
    assert forward_diff < 1e-5, f"前向不一致！max diff = {forward_diff}"
    print("test passed")