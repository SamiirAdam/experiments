
# model.py — checkpoint-aligned TokenViT (single 'head' like your base ckpt)
#
# Aligns remaining keys by defining a single Linear head named 'head' and
# splitting it into (from,to) logits in forward: head_out[:,:64], head_out[:,64:128].
#
from typing import Optional
import torch
import torch.nn as nn

class TokenViT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 8,
        n_heads: int = 6,
        max_tokens: int = 256,
        vocab_size: int = 512,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_cls_token: bool = True,  # your ckpt has 'cls'
        head_out_dim: int = 128,     # expect 128 (64 from + 64 to)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.use_cls_token = use_cls_token
        self.head_out_dim = head_out_dim

        # Shared embedding for each of the 10 integer columns per token
        self.embed_val = nn.Embedding(vocab_size, embed_dim)
        self.col_linear = nn.Linear(10 * embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.col_linear.weight)
        nn.init.zeros_(self.col_linear.bias)

        # Names aligned to ckpt ('cls', 'pos', 'enc')
        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls, std=0.02)

        self.pos = nn.Parameter(torch.zeros(1, max_tokens + (1 if use_cls_token else 0), embed_dim))
        nn.init.normal_(self.pos, std=0.02)

        self.dropout = nn.Dropout(drop)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Single head (to match your base checkpoint naming)
        self.head = nn.Linear(embed_dim, head_out_dim)

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)
        s = (x * m).sum(dim=1)
        d = m.sum(dim=1).clamp_min(1.0)
        return s / d

    def forward(self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        """
        tokens: Long [B,T,10]
        pad_mask: Bool [B,T] True for valid tokens
        Returns:
            logits_from [B,64], logits_to [B,64]
        """
        B, T, K = tokens.shape
        assert K == 10, f"Expected 10 columns, got {K}"

        emb = self.embed_val(tokens)                 # [B,T,10,E]
        emb = emb.reshape(B, T, 10 * self.embed_dim) # [B,T,10E]
        x = self.col_linear(emb)                     # [B,T,E]

        if self.use_cls_token:
            cls = self.cls.expand(B, -1, -1)         # [B,1,E]
            x = torch.cat([cls, x], dim=1)           # [B,1+T,E]
            if pad_mask is not None:
                pad_mask = torch.cat([torch.ones(B, 1, device=pad_mask.device, dtype=pad_mask.dtype), pad_mask], dim=1)

        x = x + self.pos[:, : x.size(1), :]
        x = self.dropout(x)

        src_key_padding_mask = (~pad_mask) if pad_mask is not None else None
        x = self.enc(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        pooled = x[:, 0] if self.use_cls_token else self._masked_mean(x, pad_mask)  # [B,E]

        out = self.head(pooled)  # [B,head_out_dim], expect 128
        if out.size(1) == 128:
            logits_from = out[:, :64]
            logits_to   = out[:, 64:128]
        elif out.size(1) == 64:
            # If your base model only had one 64-way head, duplicate as a fallback (not ideal).
            logits_from = out
            logits_to   = out
        else:
            raise RuntimeError(f"Unexpected head_out_dim={out.size(1)}; expected 64 or 128")
        return logits_from, logits_to
