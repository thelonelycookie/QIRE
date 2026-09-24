from __future__ import annotations
from typing import Any


def job_cost_class(job: Any) -> str:
    dataset = getattr(job, "dataset", "")
    model = getattr(job, "model_key", "")
    experts = int(getattr(job, "num_experts", 0) or 0)
    exp_h = int(getattr(job, "expert_hidden_dim", 0) or 0)
    epochs = int(getattr(job, "epochs", 0) or 0)
    heavy_dataset = dataset in {"digits", "synthetic_high_dim", "synthetic_sparse_signal", "synthetic_many_classes", "synthetic_expert_specialization"}
    if model == "qire" and (experts >= 8 or exp_h >= 32 or epochs >= 120 or heavy_dataset):
        return "large"
    if model in {"random_forest", "svm_rbf", "full_moe"} or heavy_dataset:
        return "medium"
    if model in {"logreg", "svm_linear"} or epochs <= 25:
        return "tiny"
    return "small"


def job_priority(job: Any, scientific_value: float = 1.0) -> float:
    cost_penalty = {"tiny": 0.05, "small": 0.15, "medium": 0.35, "large": 0.70}.get(job_cost_class(job), 0.2)
    novelty = 0.25 if getattr(job, "model_key", "") == "qire" else 0.1
    claim_value = 0.25 if "reproduction" in str(getattr(job, "search_mode", "")) else 0.0
    return scientific_value + novelty + claim_value - cost_penalty
