from __future__ import annotations

import torch
from torch import nn

from .gating import QuantumInspiredGating
from .quantum_state_mapper import QuantumInspiredStateMapper
from .sparse_experts import SparseExpertLayer


class QIREModel(nn.Module):
    """Quantum-inspired sparse expert model.

    Core path: input -> state mapper -> gate -> sparse experts. Gate variants are
    included for ablation; no quantum-speedup claim is implied.
    """

    model_name = "QIRE-AI"
    method = "quantum_inspired_sparse_experts"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        expert_hidden_dim: int,
        num_classes: int,
        num_experts: int = 4,
        top_k: int = 2,
        gate_type: str = "interference_born",
        state_mapper_mode: str = "trig",
        temperature: float = 1.0,
        shared_backbone: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.gate_type = gate_type
        self.state_mapper_mode = state_mapper_mode
        self.temperature = float(temperature)
        self.shared_backbone = bool(shared_backbone)
        self.method = (
            f"qire_{state_mapper_mode}_{gate_type}_top{self.top_k}_t{self.temperature:g}"
            f"_e{self.num_experts}_h{self.expert_hidden_dim}"
        )
        self.state_mapper = QuantumInspiredStateMapper(self.input_dim, self.state_dim, mode=state_mapper_mode)
        self.gate = QuantumInspiredGating(self.state_dim, self.num_experts, top_k=self.top_k, mode=gate_type, temperature=temperature)
        if self.shared_backbone:
            self.backbone = nn.Sequential(nn.Linear(self.input_dim, self.expert_hidden_dim), nn.ReLU())
            self.experts = SparseExpertLayer(self.expert_hidden_dim, max(2, self.expert_hidden_dim // 2), self.num_classes, self.num_experts)
            self.method += "_shared"
        else:
            self.experts = SparseExpertLayer(self.input_dim, self.expert_hidden_dim, self.num_classes, self.num_experts)

    def forward(self, x: torch.Tensor):
        state = self.state_mapper(x)
        probs, top_idx, top_vals, entropy = self.gate(state)
        expert_input = self.backbone(x) if self.shared_backbone else x
        logits = self.experts(expert_input, top_idx, top_vals)
        router_ops = self.state_mapper.estimated_ops_per_sample() + 2 * self.state_dim * self.num_experts
        if self.gate_type == "interference_born":
            router_ops += 2 * self.num_experts * self.num_experts
        expert_dim = self.expert_hidden_dim if self.shared_backbone else self.input_dim
        expert_ops = self.top_k * 2 * (expert_dim * self.experts.hidden_dim + self.experts.hidden_dim * self.num_classes)
        expert_ops = expert_ops * (1.0 - float(getattr(self, "_expert_prune_amount", 0.0)))
        aux = {
            "active_model_parts": float(self.top_k),
            "gate_entropy": float(entropy.detach().cpu()),
            "router_ops": int(router_ops),
            "expert_ops": int(expert_ops),
        }
        return logits, aux


    @torch.no_grad()
    def apply_magnitude_pruning(self, amount: float = 0.50) -> None:
        """Prune expert weights only. Used for QIRE + PrunedExperts proposals.

        This is an experimental CPU-side pruning proxy. It does not prove hardware
        energy savings; it only changes active weights/estimated operation proxies.
        """
        amount = float(max(0.0, min(0.95, amount)))
        self._expert_prune_amount = amount
        for module in self.experts.modules():
            if isinstance(module, nn.Linear):
                flat = module.weight.detach().abs().flatten()
                if flat.numel() == 0:
                    continue
                threshold = torch.quantile(flat, amount)
                mask = (module.weight.detach().abs() > threshold).float()
                module.register_buffer("_qire_prune_mask", mask) if not hasattr(module, "_qire_prune_mask") else None
                module._qire_prune_mask.copy_(mask)
                module.weight.mul_(module._qire_prune_mask)

    @torch.no_grad()
    def enforce_masks(self) -> None:
        for module in self.experts.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "_qire_prune_mask"):
                module.weight.mul_(module._qire_prune_mask)

    def estimated_ops_per_sample(self) -> int:
        state_map_ops = self.state_mapper.estimated_ops_per_sample()
        gate_linear_ops = 2 * self.state_dim * self.num_experts
        interference_ops = 2 * self.num_experts * self.num_experts if self.gate_type == "interference_born" else 0
        if self.shared_backbone:
            backbone_ops = 2 * self.input_dim * self.expert_hidden_dim
            expert_input_dim = self.expert_hidden_dim
            expert_hidden = max(2, self.expert_hidden_dim // 2)
        else:
            backbone_ops = 0
            expert_input_dim = self.input_dim
            expert_hidden = self.expert_hidden_dim
        expert_ops = self.top_k * 2 * (expert_input_dim * expert_hidden + expert_hidden * self.num_classes)
        expert_ops = expert_ops * (1.0 - float(getattr(self, "_expert_prune_amount", 0.0)))
        return int(state_map_ops + gate_linear_ops + interference_ops + backbone_ops + expert_ops)
