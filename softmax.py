import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max, _ = x.max(dim=dim, keepdim=True)
    x = x - x_max
    x_exp = torch.exp(x)

    result = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return result

if __name__ == '__main__':
    bs, seq_len, dim = 4, 8, 64
    x = torch.randn(dim, bs, seq_len)
    y1 = softmax(x)
    y2 = F.softmax(x, dim=-1)

    assert torch.allclose(y1, y2, atol=1e-6), "Softmax 实现与 PyTorch 不一致"
    print("通过 torch.allclose 验证")

