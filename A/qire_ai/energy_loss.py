from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EnergyAwareLoss(nn.Module):
    """Task loss plus resource/activation penalties.

    This is a research objective, not proof of real energy saving. Real claims must
    use measured or carefully estimated resource metrics after training.
    """

    def __init__(self, lambda_resource: float = 0.0, mu_activation: float = 0.0):
        super().__init__()
        self.lambda_resource = float(lambda_resource)
        self.mu_activation = float(mu_activation)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        resource_cost: torch.Tensor | float = 0.0,
        activation_cost: torch.Tensor | float = 0.0,
    ):
        task_loss = F.cross_entropy(logits, y)
        return task_loss + self.lambda_resource * resource_cost + self.mu_activation * activation_cost


def energy_aware_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    aux: dict | None,
    *,
    lambda_resource: float = 0.0,
    mu_activation: float = 0.0,
) -> torch.Tensor:
    task = F.cross_entropy(logits, y)
    if not lambda_resource and not mu_activation:
        return task
    # Normalize costs so the penalties act as gentle regularizers.
    ops = float(model.estimated_ops_per_sample()) if hasattr(model, "estimated_ops_per_sample") else float(sum(p.numel() for p in model.parameters()))
    active = float(aux.get("active_model_parts", 1.0)) if isinstance(aux, dict) else 1.0
    resource_cost = torch.tensor(ops / 10000.0, dtype=task.dtype, device=task.device)
    activation_cost = torch.tensor(active, dtype=task.dtype, device=task.device)
    return task + float(lambda_resource) * resource_cost + float(mu_activation) * activation_cost
