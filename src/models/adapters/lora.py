"""
This adapter wraps Linear layers used inside attention (q/k/v/o). 
You can apply it to your TimeXer attention projections without touching their main weights.

How to use after TimeXer is built:

from src.models.adapters.lora import apply_lora_to_attn

model = build_timexer_model(cfg)                 # your constructor
if cfg.lora.enable:
    apply_lora_to_attn(model, r=cfg.lora.rank, alpha=2*cfg.lora.rank, freeze_base=True)

"""

# src/models/adapters/lora.py
from __future__ import annotations
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Drop-in replacement that augments a frozen Linear with trainable low-rank adapters.
    y = x @ W^T + alpha / r * ( x @ B @ A )  (B: in->r, A: r->out)
    If you want to keep the base layer trainable, set freeze_base=False.
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, freeze_base: bool = True):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.r            = r
        self.alpha        = alpha
        self.scale        = alpha / r if r > 0 else 0.0
        self.freeze_base  = freeze_base

        
        device = base.weight.device
        dtype  = base.weight.dtype

        # copy base params onto the same device/dtype
        self.weight = nn.Parameter(
            base.weight.detach().clone().to(device=device, dtype=dtype),
            requires_grad=not freeze_base
        )
        if base.bias is not None:
            self.bias = nn.Parameter(
                base.bias.detach().clone().to(device=device, dtype=dtype),
                requires_grad=not freeze_base
            )
        else:
            self.bias = None

        # LoRA params on same device/dtype
        if r > 0:
            self.A = nn.Parameter(torch.zeros(self.r, self.out_features, device=device, dtype=dtype))
            self.B = nn.Parameter(torch.zeros(self.in_features, self.r, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            nn.init.zeros_(self.A)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        # lora path
        if self.r > 0:
            y = y + self.scale * x.matmul(self.B).matmul(self.A)
        return y


def apply_lora_to_attn(module: nn.Module, r: int = 8, alpha: float = 16.0, freeze_base: bool = True, names=("q_proj","k_proj","v_proj","out_proj")):
    """
    Recursively replace attention linear projections named in `names` with LoRALinear.
    Works if your attention submodules expose q_proj/k_proj/v_proj/out_proj as nn.Linear.
    """
    for name, child in module.named_children():
        if name in names and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, freeze_base=freeze_base))
        else:
            apply_lora_to_attn(child, r=r, alpha=alpha, freeze_base=freeze_base, names=names)
