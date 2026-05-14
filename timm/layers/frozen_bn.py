"""Frozen BatchNorm2d fallback when torchvision is unavailable."""
from __future__ import annotations

import torch
from torch import nn

try:  # pragma: no cover - relies on external torchvision install
    from torchvision.ops.misc import FrozenBatchNorm2d as _TvFrozenBatchNorm2d
except Exception:  # pragma: no cover - fallback path for broken/missing torchvision

    class FrozenBatchNorm2d(nn.Module):
        """Minimal FrozenBatchNorm2d that matches torchvision semantics."""

        def __init__(self, num_features: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.register_buffer("weight", torch.ones(num_features))
            self.register_buffer("bias", torch.zeros(num_features))
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim < 2:
                raise ValueError("FrozenBatchNorm2d expects input with channel dimension")
            shape = (1, -1) + (1,) * (x.ndim - 2)
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            mean = self.running_mean.view(shape)
            inv_std = torch.rsqrt(self.running_var + self.eps).view(shape)
            return (x - mean) * inv_std * weight + bias

    FrozenBatchNorm2d.__name__ = "FrozenBatchNorm2d"  # type: ignore[attr-defined]

else:  # pragma: no cover - exercised when torchvision import works
    FrozenBatchNorm2d = _TvFrozenBatchNorm2d


__all__ = ["FrozenBatchNorm2d"]
