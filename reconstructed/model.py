# Auto-generated TokenViT skeleton from reconstruct_tokenvit.py
# Edit as needed to exactly match original training architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: [B, T, D]
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, v, k = qkv[0], qkv[2], qkv[1]  # note: if original ordering differs adjust here
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TokenViT(nn.Module):
    def __init__(self, embed_dim: int, depth: int, n_heads: int, max_tokens: int, token_width: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        self.token_proj = nn.Linear(token_width, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # Two policy heads for 64 squares each
        self.head_from = nn.Linear(embed_dim, 64)
        self.head_to = nn.Linear(embed_dim, 64)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_tokens: torch.Tensor, pad_mask: torch.Tensor = None):
        # x_tokens: [B, T, 10]
        x = x_tokens.float()
        B, T, D_in = x.shape
        if T > self.max_tokens:
            raise ValueError(f"Sequence length {T} > max_tokens {self.max_tokens}")
        x = self.token_proj(x)
        x = x + self.pos_embed[:, :T]
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+T, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head_from(cls_out), self.head_to(cls_out)

