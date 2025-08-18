# src/models/components/patch_embed_1d.py

"""
How to use: put this in front of your TimeXer. 
Feed (B, T, C) features → PatchEmbed1D → TimeXer encoder/decoder → predictions.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed1D(nn.Module):
    """
    Turn a (B, T, C) series into patches: (B, L, P*C) and project to d_model.
    - patch_size: number of timesteps per patch (e.g., 16 or 24)
    - stride: usually = patch_size (non-overlapping), but you can set < patch_size to overlap
    """
    def __init__(self, in_channels: int, d_model: int, patch_size: int = 16, stride: int | None = None):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = patch_size if stride is None else stride
        self.in_channels = in_channels
        self.proj = nn.Linear(patch_size * in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))  # max 1024 patches; resize if needed
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        returns: (B, L, d_model)
        """
        B, T, C = x.shape
        # (B, C, T) for unfold
        x_u = x.permute(0, 2, 1)
        # Unfold over time: kernel_size = patch_size, stride = stride
        # result: (B, C*patch_size, L)
        patches = F.unfold(
            x_u.unsqueeze(-1),  # (B, C, T, 1)
            kernel_size=(self.patch_size, 1),
            stride=(self.stride, 1)
        )  # -> (B, C*patch_size, L)
        L = patches.shape[-1]
        patches = patches.transpose(1, 2)             # (B, L, C*patch_size)
        tokens  = self.proj(patches)                  # (B, L, d_model)

        # positional embedding (trim or interpolate if needed)
        if self.pos_emb.shape[1] < L:
            # interpolate on the fly if longer than max
            pos = F.interpolate(self.pos_emb.transpose(1,2), size=L, mode="linear", align_corners=False).transpose(1,2)
        else:
            pos = self.pos_emb[:, :L, :]
        return tokens + pos
