from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QuantumInspiredStateMapper(nn.Module):
    """Maps classical features to a normalized amplitude-like state vector."""

    def __init__(self, input_dim: int, state_dim: int, mode: str = "trig"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.mode = mode
        multiplier = self._multiplier(mode)
        self.proj = nn.Linear(self.input_dim * multiplier, self.state_dim)

    @staticmethod
    def _multiplier(mode: str) -> int:
        if mode == "trig":
            return 3
        if mode in {"linear", "identity", "low_rank", "hash", "normalized", "cluster_based", "none"}:
            return 1
        if mode == "poly2":
            return 2
        if mode == "compact_trig":
            return 2
        raise ValueError(f"Unsupported state mapper mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "trig":
            encoded = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        elif self.mode in {"linear", "identity", "low_rank", "hash", "normalized", "cluster_based", "none"}:
            encoded = x
        elif self.mode == "poly2":
            encoded = torch.cat([x, x.pow(2)], dim=-1)
        elif self.mode == "compact_trig":
            encoded = torch.cat([x, torch.sin(x)], dim=-1)
        else:
            raise ValueError(f"Unsupported state mapper mode: {self.mode}")
        state = torch.tanh(self.proj(encoded))
        return F.normalize(state, p=2, dim=-1, eps=1e-8)

    def estimated_ops_per_sample(self) -> int:
        multiplier = self._multiplier(self.mode)
        trig_extra = 16 * self.input_dim if self.mode == "trig" else (8 * self.input_dim if self.mode == "compact_trig" else 0)
        poly_extra = self.input_dim if self.mode == "poly2" else 0
        return int(trig_extra + poly_extra + 2 * (self.input_dim * multiplier) * self.state_dim)
