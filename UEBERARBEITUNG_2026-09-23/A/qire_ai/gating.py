from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QuantumInspiredGating(nn.Module):
    """Classical routing modules used for fair ablation.

    Modes:
    - softmax_state: ordinary softmax over a state vector. Control condition.
    - born: amplitudes are squared and normalized. Inspired by Born-rule probabilities.
    - interference_born: amplitudes are linearly mixed before squaring.
    - gumbel_softmax: differentiable stochastic-ish selector during training.
    - hard_threshold: sparse thresholded probability control.

    This is not quantum hardware and does not imply quantum speedup.
    """

    def __init__(self, state_dim: int, num_experts: int, top_k: int = 2, mode: str = "interference_born", temperature: float = 1.0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.mode = mode
        self.temperature = float(max(0.05, temperature))
        self.amplitude = nn.Linear(self.state_dim, self.num_experts)
        eye = torch.eye(self.num_experts)
        noise = 0.02 * torch.randn(self.num_experts, self.num_experts)
        self.interference = nn.Parameter(eye + noise)

    def forward(self, state: torch.Tensor):
        raw = self.amplitude(state) / self.temperature
        if self.mode in {"softmax", "softmax_state"}:
            probs = torch.softmax(raw, dim=-1)
        elif self.mode == "gumbel_softmax":
            probs = F.gumbel_softmax(raw, tau=self.temperature, hard=False, dim=-1) if self.training else torch.softmax(raw, dim=-1)
        elif self.mode == "hard_threshold":
            probs = torch.softmax(raw, dim=-1)
            thresh = probs.mean(dim=-1, keepdim=True)
            probs = torch.where(probs >= thresh, probs, torch.zeros_like(probs)) + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.mode == "entropy_adaptive":
            probs = torch.softmax(raw, dim=-1)
            entropy_probe = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            sharpen = torch.where(entropy_probe < 0.75 * torch.log(torch.tensor(float(self.num_experts), device=raw.device)), 0.75, 1.25)
            probs = torch.softmax(raw / sharpen, dim=-1)
        elif self.mode == "budget_aware":
            # Stronger sparsity pressure; scientific use requires comparison against softmax/born controls.
            probs = torch.softmax(raw / max(0.25, self.temperature * 0.65), dim=-1)
        elif self.mode == "hybrid_softmax_born":
            soft = torch.softmax(raw, dim=-1)
            amplitudes = torch.tanh(raw)
            born = amplitudes.pow(2) + 1e-8
            born = born / born.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            probs = 0.5 * soft + 0.5 * born
        else:
            amplitudes = torch.tanh(raw)
            if self.mode == "interference_born":
                amplitudes = amplitudes @ self.interference
            elif self.mode != "born":
                raise ValueError(f"Unsupported QIRE gate mode: {self.mode}")
            probs = amplitudes.pow(2) + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return probs, top_idx, top_vals, entropy


class BornRuleGating(QuantumInspiredGating):
    def __init__(self, state_dim: int, num_experts: int, top_k: int = 2):
        super().__init__(state_dim, num_experts, top_k=top_k, mode="interference_born")
