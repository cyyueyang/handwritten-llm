import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super(SwiGLU, self).__init__()
        hidden_dim = int(d_model * 8 / 3)

        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.gate_proj = nn.Linear(d_model, hidden_dim)

        self.down_proj = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.up_proj(x) * F.silu(self.gate_proj(x))
        x = self.down_proj(x)

        return x

if __name__ == '__main__':
    bs, seq_len, d_model = 8, 16, 64
    ffn = FFN(d_model)
    swiglu = SwiGLU(d_model)

    x = torch.randn(bs, seq_len, d_model)

    print(f"ffn(x) shape: {ffn(x).shape}")
    print(f"swiglu(x) shape: {swiglu(x).shape}")
