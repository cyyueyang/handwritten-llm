import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from rms_norm import RMSNorm
from rope import RoPE

class MultiLatentAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            num_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            max_seq_len: int,
    ):
        super(MultiLatentAttention, self).__init__()

        self.d_model = d_model
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.max_seq_len = max_seq_len

        self.rope = RoPE(self.qk_rope_head_dim, self.max_seq_len)

        if self.q_lora_rank > 0:
            self.q_down_proj = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)
        else:
            self.q_proj = nn.Linear(self.d_model, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)

        self.k_rope_proj = nn.Linear(self.d_model, self.qk_rope_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(self.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, self.d_model, bias=False)

        mask = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.bool), diagonal=0)
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        bs, seq_len, d_model = x.shape

        if self.q_lora_rank > 0:
            q = self.q_down_proj(x)
            q = self.q_norm(q)
            q = self.q_up_proj(q)
        else:
            q = self.q_proj(x)

        q_nope, q_rope = q.split([self.num_heads * self.qk_nope_head_dim, self.num_heads * self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        q_rope = q_rope.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)

        latens = self.kv_down_proj(x)
        # kv cache 存 latents 但是还没写 kv cache 故不存

        latens = self.kv_norm(latens)
        k_nope, v = self.kv_up_proj(latens).split([self.num_heads * self.qk_nope_head_dim, self.num_heads * self.v_head_dim], dim=-1)
        k_nope = k_nope.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)

        k_rope = self.k_rope_proj(x).view(bs, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2).expand(-1, self.num_heads, -1, -1)

        v = v.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)

        q_rope = self.rope(q_rope)
        k_rope = self.rope(k_rope)
        # kv cache 存 k_rope 但是还没写 kv cache 故不存

        query = torch.cat([q_nope, q_rope], dim=-1)
        key = torch.cat([k_nope, k_rope], dim=-1)

        attn = query @ key.transpose(-2, -1) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        attn = attn.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float('-inf'))
        attn_score = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        out = self.out_proj(out)

        return out


if __name__ == "__main__":
    bs, seq_len, d_model = 4, 16, 64

    MLA = MultiLatentAttention(64, 8, 8, 4, 64, 32, 64, 1024)
    x = torch.randn(bs, seq_len, d_model)
    out = MLA(x)
    print(out.shape)




