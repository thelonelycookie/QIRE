from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import psutil
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.nn import functional as F


def count_parameters(model: torch.nn.Module) -> int:
    try:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return 0


def active_parameter_count(model: torch.nn.Module) -> int:
    if hasattr(model, "estimated_active_params"):
        try:
            return int(model.estimated_active_params())
        except Exception:
            pass
    return count_parameters(model)


def process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def model_size_kb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        if hasattr(model, "estimator") and getattr(model, "estimator", None) is not None:
            import pickle
            path.write_bytes(pickle.dumps(model.estimator))
        else:
            torch.save(model.state_dict(), path)
        return float(path.stat().st_size / 1024.0)
    finally:
        try:
            path.unlink()
        except Exception:
            pass


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float, Dict[str, object]]:
    model.eval()
    logits, aux = model(x)
    loss = float(F.cross_entropy(logits, y).detach().cpu())
    pred = logits.argmax(dim=-1)
    y_np = y.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    accuracy = float((pred == y).float().mean().detach().cpu())
    active_parts = float(aux.get("active_model_parts", 1.0)) if isinstance(aux, dict) else 1.0
    avg = "binary" if len(np.unique(y_np)) == 2 else "macro"
    extras: Dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_np, pred_np)),
        "f1_macro": float(f1_score(y_np, pred_np, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_np, pred_np, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_np, pred_np, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_np, pred_np, average="weighted", zero_division=0)),
        "confusion_matrix_json": json.dumps(confusion_matrix(y_np, pred_np).tolist(), separators=(",", ":")),
        "gate_entropy": float(aux.get("gate_entropy", 0.0)) if isinstance(aux, dict) else 0.0,
        "router_ops": int(aux.get("router_ops", 0)) if isinstance(aux, dict) else 0,
        "expert_ops": int(aux.get("expert_ops", 0)) if isinstance(aux, dict) else 0,
        "early_exit_rate": float(aux.get("early_exit_rate", 0.0)) if isinstance(aux, dict) else 0.0,
    }
    return accuracy, loss, active_parts, extras


@torch.no_grad()
def measure_inference_stats(model: torch.nn.Module, x: torch.Tensor, repeats: int = 80) -> Dict[str, float]:
    model.eval()
    for _ in range(5):
        model(x)
    per_sample = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        elapsed = time.perf_counter() - start
        per_sample.append(1000.0 * elapsed / max(1, x.shape[0]))
    if not per_sample:
        return {"inference_time_ms": 0.0, "inference_time_ms_median": 0.0, "inference_time_ms_p95": 0.0}
    return {
        "inference_time_ms": float(statistics.mean(per_sample)),
        "inference_time_ms_median": float(statistics.median(per_sample)),
        "inference_time_ms_p95": float(np.percentile(np.asarray(per_sample, dtype=float), 95)),
    }


@torch.no_grad()
def measure_inference_ms_per_sample(model: torch.nn.Module, x: torch.Tensor, repeats: int = 80) -> float:
    return float(measure_inference_stats(model, x, repeats=repeats)["inference_time_ms"])


def estimated_ops_per_sample(model: torch.nn.Module) -> int:
    if hasattr(model, "estimated_ops_per_sample"):
        return int(model.estimated_ops_per_sample())
    return int(count_parameters(model) * 2)


def efficiency_per_accuracy(accuracy: float, ops: int) -> float:
    return float(accuracy / max(1, ops))


def gpu_info() -> Dict[str, object]:
    if torch.cuda.is_available():
        try:
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(0) / (1024 ** 2)),
                "gpu_memory_reserved_mb": float(torch.cuda.memory_reserved(0) / (1024 ** 2)),
            }
        except Exception:
            return {"gpu_available": True, "gpu_name": "unknown", "gpu_memory_allocated_mb": 0.0, "gpu_memory_reserved_mb": 0.0}
    return {"gpu_available": False, "gpu_name": "none", "gpu_memory_allocated_mb": 0.0, "gpu_memory_reserved_mb": 0.0}
