from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from .energy_loss import energy_aware_loss
from .metrics import (
    active_parameter_count,
    count_parameters,
    estimated_ops_per_sample,
    evaluate,
    gpu_info,
    measure_inference_stats,
    model_size_kb,
    process_memory_mb,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_iter(x: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int):
    n = x.shape[0]
    if batch_size <= 0 or batch_size >= n:
        yield x, y
        return
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    order = torch.randperm(n, generator=g, device=x.device)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]


def train_model(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    prune_amount: float | None = None,
    batch_size: int = 0,
    seed: int = 0,
    lambda_resource: float = 0.0,
    mu_activation: float = 0.0,
    grad_clip: float = 5.0,
    weight_decay: float = 0.0,
) -> Dict[str, float | str]:
    start = time.perf_counter()
    if hasattr(model, "fit_sklearn"):
        peak_memory = process_memory_mb()
        model.fit_sklearn(x_train, y_train)
        return {
            "training_time_sec": float(time.perf_counter() - start),
            "train_best_loss": 0.0,
            "nan_detected": "False",
            "peak_memory_mb": float(max(peak_memory, process_memory_mb())),
        }
    params = list(model.parameters())
    if not params:
        return {
            "training_time_sec": 0.0,
            "train_best_loss": 0.0,
            "nan_detected": "False",
            "peak_memory_mb": float(process_memory_mb()),
        }
    # weight_decay defaults to 0.0, which is exactly Adam's own default, so every
    # earlier experiment is unaffected and stays bit-for-bit reproducible.
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    prune_epoch = int(epochs * 0.60) if prune_amount is not None else None
    pruning_applied = False
    best_loss = float("inf")
    nan_detected = False
    peak_memory = process_memory_mb()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in _batch_iter(x_train, y_train, batch_size=batch_size, seed=seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(xb)
            if lambda_resource or mu_activation:
                loss = energy_aware_loss(logits, yb, model, aux, lambda_resource=lambda_resource, mu_activation=mu_activation)
            else:
                loss = F.cross_entropy(logits, yb)
                if isinstance(aux, dict) and "early_logits" in aux:
                    loss = 0.75 * loss + 0.25 * F.cross_entropy(aux["early_logits"], yb)
            if not torch.isfinite(loss):
                nan_detected = True
                break
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if hasattr(model, "enforce_masks"):
                model.enforce_masks()
            epoch_losses.append(float(loss.detach().cpu()))
        if nan_detected:
            break
        if epoch_losses:
            best_loss = min(best_loss, float(np.mean(epoch_losses)))
        peak_memory = max(peak_memory, process_memory_mb())
        if prune_epoch is not None and epoch == prune_epoch and hasattr(model, "apply_magnitude_pruning"):
            model.apply_magnitude_pruning(prune_amount)
            pruning_applied = True

    if prune_amount is not None and not pruning_applied and hasattr(model, "apply_magnitude_pruning"):
        model.apply_magnitude_pruning(prune_amount)
    return {
        "training_time_sec": float(time.perf_counter() - start),
        "train_best_loss": float(best_loss if math.isfinite(best_loss) else 0.0),
        "nan_detected": str(bool(nan_detected)),
        "peak_memory_mb": float(peak_memory),
    }


def run_single_experiment(
    *,
    run_id: str,
    experiment_id: str,
    hypothesis_id: str,
    dataset_name: str,
    model: torch.nn.Module,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    prune_amount: float | None,
    assumed_cpu_watts: float,
    inference_repeats: int,
    batch_size: int = 0,
    lambda_resource: float = 0.0,
    mu_activation: float = 0.0,
) -> Dict[str, object]:
    cpu_before = psutil.cpu_percent(interval=0.03)
    train_info = train_model(
        model, x_train, y_train, epochs, lr, prune_amount=prune_amount, batch_size=batch_size, seed=seed,
        lambda_resource=lambda_resource, mu_activation=mu_activation,
    )
    accuracy, loss, active_parts, extras = evaluate(model, x_test, y_test)
    inf_stats = measure_inference_stats(model, x_test, repeats=inference_repeats)
    memory_mb = process_memory_mb()
    ops = estimated_ops_per_sample(model)
    # Energy estimate: CPU wattage assumption only. Not laboratory energy measurement.
    energy_wh = (float(train_info["training_time_sec"]) * assumed_cpu_watts) / 3600.0
    cpu_after = psutil.cpu_percent(interval=0.03)
    ram = psutil.virtual_memory().percent
    gpu = gpu_info()

    row = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "hypothesis_id": hypothesis_id,
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "method": getattr(model, "method", model.__class__.__name__),
        "dataset": dataset_name,
        "seed": seed,
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(float(extras.get("balanced_accuracy", 0.0)), 6),
        "f1_macro": round(float(extras.get("f1_macro", 0.0)), 6),
        "precision_macro": round(float(extras.get("precision_macro", 0.0)), 6),
        "recall_macro": round(float(extras.get("recall_macro", 0.0)), 6),
        "f1_weighted": round(float(extras.get("f1_weighted", 0.0)), 6),
        "loss": round(loss, 6),
        "training_time_sec": round(float(train_info["training_time_sec"]), 6),
        "train_best_loss": round(float(train_info["train_best_loss"]), 6),
        "inference_time_ms": round(float(inf_stats["inference_time_ms"]), 8),
        "inference_time_ms_median": round(float(inf_stats["inference_time_ms_median"]), 8),
        "inference_time_ms_p95": round(float(inf_stats["inference_time_ms_p95"]), 8),
        "memory_mb": round(memory_mb, 3),
        "peak_memory_mb": round(float(train_info["peak_memory_mb"]), 3),
        "model_size_kb": round(model_size_kb(model), 3),
        "parameter_count": count_parameters(model),
        "active_parameter_count": active_parameter_count(model),
        "estimated_ops": ops,
        "router_ops": int(extras.get("router_ops", 0)),
        "expert_ops": int(extras.get("expert_ops", 0)),
        "estimated_energy_wh": round(energy_wh, 8),
        "active_model_parts": round(active_parts, 6),
        "gate_entropy": round(float(extras.get("gate_entropy", 0.0)), 6),
        "early_exit_rate": round(float(extras.get("early_exit_rate", 0.0)), 6),
        "confusion_matrix_json": extras.get("confusion_matrix_json", ""),
        "cpu_usage": round((cpu_before + cpu_after) / 2.0, 3),
        "gpu_usage": 0.0,
        "gpu_available": str(gpu.get("gpu_available", False)),
        "gpu_name": str(gpu.get("gpu_name", "none")),
        "gpu_memory_allocated_mb": round(float(gpu.get("gpu_memory_allocated_mb", 0.0)), 3),
        "ram_usage": round(ram, 3),
        "nan_detected": train_info.get("nan_detected", "False"),
        "notes": "Energy is an estimate from assumed CPU watts; not a direct hardware measurement.",
    }
    return row
