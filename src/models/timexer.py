# src/models/timexer.py
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.patch_embed_1d import PatchEmbed1D


class PreNorm(nn.Module):
    """LayerNorm -> module."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class MultiheadSelfAttention(nn.Module):
    """
    Simple MSA with explicit q_proj/k_proj/v_proj/out_proj names
    so LoRA can hook them via apply_lora_to_attn(...).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # <<< IMPORTANT: keep these names for LoRA >>>
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # <<< ------------------------------------- >>>

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, d)
        attn_mask: (B, 1, L, L) or (1, 1, L, L) with True for masked positions (optional)
        """
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B, H, L, d)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        y = self.out_proj(y)
        return self.proj_drop(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeXerBlock(nn.Module):
    """PreNorm(MSA) + residual, then PreNorm(FFN) + residual."""
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.attn = PreNorm(d_model, MultiheadSelfAttention(d_model, n_heads, dropout))
        self.ffn  = PreNorm(d_model, FeedForward(d_model, ffn_hidden, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class TimeXer(nn.Module):
    """
    (B,T,C) -> PatchEmbed1D -> N x TimeXerBlock -> pooled -> head
    - If `return_tokens=True`, also returns per-token states (for masked-recon pretrain).
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_hidden: int = 2048,
        dropout: float = 0.1,
        patch_size: int = 20,
        stride: Optional[int] = None,
        out_dim: int = 1,
        use_cls: bool = False,
    ):
        super().__init__()
        self.use_cls = use_cls

        self.patcher = PatchEmbed1D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls else None
        self.blocks = nn.ModuleList([
            TimeXerBlock(d_model, n_heads, ffn_hidden, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim) if out_dim > 0 else nn.Identity()

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if hasattr(self.head, "bias") and self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        """
        x: (B, T, C)
        returns:
           y: (B, out_dim)
           (optionally) tokens: (B, L, d_model)
        """
        tokens = self.patcher(x)  # (B, L, d)
        if self.use_cls:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+L, d)

        h = tokens
        for blk in self.blocks:
            h = blk(h)

        h = self.norm(h)

        if self.use_cls:
            pooled = h[:, 0]                        # (B, d)
        else:
            pooled = h.mean(dim=1)                  # mean pool over tokens

        y = self.head(pooled)                       # (B, out_dim)
        if return_tokens:
            return y, h
        return y


def build_timexer(
    num_features: int,
    num_targets: int,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    ffn_hidden: int = 2048,
    dropout: float = 0.1,
    patch_size: int = 20,
    stride: Optional[int] = None,
    use_cls: bool = False,
) -> TimeXer:
    return TimeXer(
        in_channels=num_features,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        dropout=dropout,
        patch_size=patch_size,
        stride=stride,
        out_dim=num_targets,
        use_cls=use_cls,
    )
