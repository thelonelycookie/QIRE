from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .sparse_experts import SparseExpertLayer


class PrunedMLP(nn.Module):
    """Magnitude-pruned MLP. Masks are enforced during and after fine-tuning."""

    model_name = "PrunedMLP"
    method = "magnitude_pruning"
    _method_prefix = "magnitude_pruning"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.method = f"{self._method_prefix}_h{self.hidden_dim}"
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        self.register_buffer("mask1", torch.ones_like(self.fc1.weight))
        self.register_buffer("mask2", torch.ones_like(self.fc2.weight))

    def forward(self, x: torch.Tensor):
        h = F.linear(x, self.fc1.weight * self.mask1, self.fc1.bias)
        h = F.relu(h)
        logits = F.linear(h, self.fc2.weight * self.mask2, self.fc2.bias)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "router_ops": 0, "expert_ops": self.estimated_ops_per_sample()}
        return logits, aux

    @torch.no_grad()
    def apply_magnitude_pruning(self, amount: float = 0.50) -> None:
        amount = float(max(0.0, min(0.95, amount)))
        for weight, mask in [(self.fc1.weight, self.mask1), (self.fc2.weight, self.mask2)]:
            flat = weight.detach().abs().flatten()
            threshold = torch.quantile(flat, amount)
            mask.copy_((weight.detach().abs() > threshold).float())
        self.enforce_masks()
        self.method = f"{self._method_prefix}_h{self.hidden_dim}_p{amount:g}"

    @torch.no_grad()
    def enforce_masks(self) -> None:
        self.fc1.weight.mul_(self.mask1)
        self.fc2.weight.mul_(self.mask2)

    def estimated_ops_per_sample(self) -> int:
        nonzero = int(self.mask1.sum().item() + self.mask2.sum().item())
        return int(2 * nonzero)

    def estimated_active_params(self) -> int:
        return int(self.mask1.sum().item() + self.mask2.sum().item())


class QuantizedMLP(nn.Module):
    """Quantization-aware comparison baseline.

    This approximates lower-precision inference by quantizing weights in forward.
    It is a CPU simulation, not proof of real hardware energy savings.
    """

    model_name = "QuantizedMLP"
    method = "simulated_weight_quantization"

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, bits: int = 8):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.bits = int(bits)
        self.method = f"simulated_weight_quantization_h{self.hidden_dim}_b{self.bits}"
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def _q(self, w: torch.Tensor) -> torch.Tensor:
        levels = max(2, 2 ** self.bits - 1)
        scale = w.detach().abs().max().clamp_min(1e-8)
        q = torch.round((w / scale + 1.0) * (levels / 2.0)) / (levels / 2.0) - 1.0
        return q * scale

    def forward(self, x: torch.Tensor):
        h = F.linear(x, self._q(self.fc1.weight), self.fc1.bias)
        h = F.relu(h)
        logits = F.linear(h, self._q(self.fc2.weight), self.fc2.bias)
        aux = {"active_model_parts": 1.0, "gate_entropy": 0.0, "quant_bits": self.bits}
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        base = 2 * (self.input_dim * self.hidden_dim + self.hidden_dim * self.num_classes)
        # Simulated effective cost proxy: 8-bit roughly 1/4 memory bandwidth vs fp32, not wall-time.
        return int(base * max(0.125, self.bits / 32.0))


class StandardMoE(nn.Module):
    """Sparse Mixture-of-Experts baseline with a classical softmax router."""

    model_name = "StandardMoE"
    method = "softmax_sparse_moe"

    def __init__(
        self,
        input_dim: int,
        expert_hidden_dim: int,
        num_classes: int,
        num_experts: int = 4,
        top_k: int = 2,
        full_moe: bool = False,
        random_gate: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.top_k = self.num_experts if full_moe else max(1, min(int(top_k), self.num_experts))
        self.full_moe = bool(full_moe)
        self.random_gate = bool(random_gate)
        if full_moe:
            self.method = f"full_moe_upper_bound_e{self.num_experts}_h{self.expert_hidden_dim}"
        elif random_gate:
            self.method = f"random_gate_negative_control_e{self.num_experts}_h{self.expert_hidden_dim}"
        else:
            self.method = f"softmax_sparse_moe_top{self.top_k}_e{self.num_experts}_h{self.expert_hidden_dim}"
        self.router = nn.Linear(self.input_dim, self.num_experts)
        self.experts = SparseExpertLayer(self.input_dim, self.expert_hidden_dim, self.num_classes, self.num_experts)

    def forward(self, x: torch.Tensor):
        if self.random_gate and not self.training:
            probs = torch.ones(x.shape[0], self.num_experts, dtype=x.dtype, device=x.device) / self.num_experts
        else:
            probs = torch.softmax(self.router(x), dim=-1)
        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = self.experts(x, top_idx, top_vals)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        aux = {
            "active_model_parts": float(self.top_k),
            "gate_entropy": float(entropy.detach().cpu()),
            "router_ops": 2 * self.input_dim * self.num_experts,
            "expert_ops": self.top_k * 2 * (self.input_dim * self.expert_hidden_dim + self.expert_hidden_dim * self.num_classes),
        }
        return logits, aux

    def estimated_ops_per_sample(self) -> int:
        router_ops = 2 * self.input_dim * self.num_experts
        expert_ops = self.top_k * 2 * (self.input_dim * self.expert_hidden_dim + self.expert_hidden_dim * self.num_classes)
        return int(router_ops + expert_ops)


class DistilledStudentMLP(PrunedMLP):
    """Small student placeholder baseline. Teacher distillation is optional; in this
    implementation it trains as a compact MLP and is explicitly logged as compact
    student, not as a proven distilled model unless teacher logits are added later.
    """

    model_name = "StudentMLP"
    method = "compact_student_control"
    _method_prefix = "compact_student_control"
