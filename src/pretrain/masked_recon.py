# src/pretrain/masked_recon.py

"""Pretrain a TimeXer encoder with masked reconstruction.
This is a simple pretraining task that can be used to initialize the model before fine-tuning on a specific task.
It masks out random spans of the input time series and trains the model to reconstruct the masked values.

Hook: run this once, save weights, then fine‑tune with your supervised loss.

"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

def random_mask(x: torch.Tensor, mask_ratio: float = 0.2, span: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask contiguous spans along time for (B,T,C) tensors.
    Returns masked_x (zeros where masked) and mask (bool, True where masked).
    """
    B, T, C = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    num_mask = int(T * mask_ratio)

    for b in range(B):
        t = 0
        masked = 0
        while masked < num_mask:
            start = np.random.randint(0, T)
            length = min(span, T - start, num_mask - masked)
            mask[b, start:start+length] = True
            masked += length
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask

class ReconHead(nn.Module):
    def __init__(self, d_model: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_channels)
    def forward(self, h):  # h: (B,L,d_model) or (B,T,d_model)
        return self.proj(h)

def pretrain_masked_reconstruction(
    encoder: nn.Module,                # your TimeXer encoder that maps (B,T,C)->(B,L,d_model)
    recon_head: nn.Module,             # small linear head back to C
    dataloader,                        # yields dict with "x": (B,T,C)
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-3,
    mask_ratio: float = 0.2,
    span: int = 4,
):
    encoder.train(); recon_head.train()
    opt = optim.AdamW(list(encoder.parameters()) + list(recon_head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        total = 0.0; steps = 0
        for batch in dataloader:
            x = batch["x"].to(device)        # (B,T,C)
            x_m, m = random_mask(x, mask_ratio=mask_ratio, span=span)
            h = encoder(x_m)                  # (B,L,d)
            # OPTIONAL: if encoder outputs tokens per patch, map back to T by simple repeat
            if h.shape[1] != x.shape[1]:
                # naive upsample tokens back to timestep count
                factor = x.shape[1] // h.shape[1]
                h = h.repeat_interleave(factor, dim=1)[:, :x.shape[1], :]

            xr = recon_head(h)               # (B,T,C)
            mask3 = m.unsqueeze(-1).expand_as(xr)               # (B,T,C) boolean
            xr_sel = xr[mask3]
            x_sel  = x[mask3]
            finite = torch.isfinite(x_sel) & torch.isfinite(xr_sel)
            if finite.any():
                loss = loss_fn(xr_sel[finite], x_sel[finite])
            else:
                continue  # skip this batch if nothing valid (should be rare)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()); steps += 1
        print(f"[pretrain] epoch {ep}/{epochs} loss={total/steps:.5f}")
