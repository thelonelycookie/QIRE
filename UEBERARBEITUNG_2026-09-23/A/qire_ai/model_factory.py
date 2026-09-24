from __future__ import annotations

from .models_baseline import DenseMLP, DeepMLP, EarlyExitMLP, LogisticRegressionTorch, LowRankMLP, SklearnClassifierWrapper, TensorTrainMLP
from .models_efficient import DistilledStudentMLP, PrunedMLP, QuantizedMLP, StandardMoE
from .models_qire import QIREModel
from .search_spaces import ExperimentJob


def build_model_from_job(job: ExperimentJob, input_dim: int, num_classes: int):
    top_k = max(1, min(int(job.top_k), int(job.num_experts)))
    if job.model_key == "logreg":
        return LogisticRegressionTorch(input_dim, num_classes), None
    if job.model_key in {"random_forest", "svm_rbf", "svm_linear"}:
        return SklearnClassifierWrapper(job.model_key, input_dim, num_classes, seed=job.seed), None
    if job.model_key == "mlp":
        return DenseMLP(input_dim, int(job.hidden_dim), num_classes), None
    if job.model_key == "deep_mlp":
        return DeepMLP(input_dim, int(job.hidden_dim), num_classes), None
    if job.model_key == "lowrank_mlp":
        rank_dim = max(2, min(int(job.state_dim or max(2, input_dim // 2)), int(job.hidden_dim)))
        return LowRankMLP(input_dim, rank_dim, int(job.hidden_dim), num_classes), None
    if job.model_key == "tensor_train_mlp":
        return TensorTrainMLP(input_dim, int(job.hidden_dim), num_classes, tt_rank=int(job.tt_rank), num_cores=int(job.tt_cores)), None
    if job.model_key == "early_exit_mlp":
        return EarlyExitMLP(input_dim, int(job.hidden_dim), num_classes, confidence_threshold=float(job.confidence_threshold)), None
    if job.model_key == "pruned_mlp":
        return PrunedMLP(input_dim, int(job.hidden_dim), num_classes), float(job.prune_amount)
    if job.model_key == "quantized_mlp":
        return QuantizedMLP(input_dim, int(job.hidden_dim), num_classes, bits=int(job.quant_bits)), None
    if job.model_key == "student_mlp":
        return DistilledStudentMLP(input_dim, max(4, int(job.hidden_dim)), num_classes), None
    if job.model_key in {"standard_moe", "full_moe", "random_gate_moe"}:
        return StandardMoE(
            input_dim=input_dim,
            expert_hidden_dim=int(job.expert_hidden_dim),
            num_classes=num_classes,
            num_experts=int(job.num_experts),
            top_k=top_k,
            full_moe=job.model_key == "full_moe",
            random_gate=job.model_key == "random_gate_moe",
        ), None
    if job.model_key == "qire":
        state_dim = job.state_dim
        if state_dim is None:
            multiplier = 3 if job.state_mapper == "trig" else (2 if job.state_mapper in {"poly2", "compact_trig"} else 1)
            if job.state_mapper == "low_rank":
                state_dim = max(4, min(24, input_dim // 2))
            else:
                state_dim = max(4, min(96, input_dim * multiplier))
        return QIREModel(
            input_dim=input_dim,
            state_dim=int(state_dim),
            expert_hidden_dim=int(job.expert_hidden_dim),
            num_classes=num_classes,
            num_experts=int(job.num_experts),
            top_k=top_k,
            gate_type=job.gate_type,
            state_mapper_mode=job.state_mapper,
            temperature=float(job.temperature),
            shared_backbone=bool(job.shared_backbone),
        ), None
    raise ValueError(f"Unknown model_key: {job.model_key}")
