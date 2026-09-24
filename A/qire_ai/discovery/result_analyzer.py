from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def score_row(row: Dict[str, Any]) -> float:
    acc = _f(row, "balanced_accuracy", _f(row, "accuracy", 0.0))
    ops = max(_f(row, "estimated_ops", 1.0), 1.0)
    p95 = max(_f(row, "inference_time_ms_p95", _f(row, "inference_time_ms", 0.0)), 0.0)
    return acc - 0.000018 * (ops ** 0.5) - 0.001 * p95


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    failed = [r for r in rows if str(r.get("status")) == "failed"]
    by_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in ok:
        by_key[(r.get("dataset"), r.get("model_name"), r.get("method"))].append(r)
    groups = []
    for (dataset, model, method), rs in by_key.items():
        accs = [_f(r, "balanced_accuracy", _f(r, "accuracy")) for r in rs]
        ops = [_f(r, "estimated_ops") for r in rs]
        p95s = [_f(r, "inference_time_ms_p95", _f(r, "inference_time_ms")) for r in rs]
        groups.append({
            "dataset": dataset,
            "model_name": model,
            "method": method,
            "n": len(rs),
            "balanced_accuracy_mean": mean(accs) if accs else 0.0,
            "balanced_accuracy_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "estimated_ops_mean": mean(ops) if ops else 0.0,
            "inference_p95_mean": mean(p95s) if p95s else 0.0,
            "score": mean([score_row(r) for r in rs]) if rs else 0.0,
        })
    groups.sort(key=lambda g: g["score"], reverse=True)
    qire = [g for g in groups if str(g["model_name"]).lower().startswith("qire")]
    classical = [g for g in groups if g not in qire]
    return {
        "raw_count": len(rows),
        "ok_count": len(ok),
        "failed_count": len(failed),
        "groups": groups,
        "top_overall": groups[:15],
        "top_qire": qire[:15],
        "top_classical": classical[:15],
        "failed_examples": failed[-10:],
    }


def diagnose_qire(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    qires = [r for r in ok if str(r.get("model_name", "")).lower().startswith("qire")]
    baselines = [r for r in ok if r not in qires]
    if not qires:
        return [{"type": "missing_qire", "message": "No successful QIRE runs yet."}]
    # Compare approximate average ops with nearest dataset baselines.
    by_ds_base = defaultdict(list)
    for r in baselines:
        by_ds_base[r.get("dataset")].append(r)
    for q in qires:
        ds = q.get("dataset")
        base = by_ds_base.get(ds, [])
        if not base:
            continue
        best_base_ops = min(_f(b, "estimated_ops", 1e12) for b in base if _f(b, "estimated_ops", 0) > 0) or 1.0
        q_ops = _f(q, "estimated_ops", 0.0)
        if q_ops > 2.0 * best_base_ops:
            findings.append({"type": "qire_router_overhead", "dataset": ds, "method": q.get("method"), "message": "QIRE variant has substantially higher estimated ops than the lightest baseline on the same dataset."})
        if _f(q, "gate_entropy", 0.0) < 0.05 and _f(q, "active_model_parts", 0.0) <= 1.1:
            findings.append({"type": "router_collapse_possible", "dataset": ds, "method": q.get("method"), "message": "Very low gate entropy suggests possible router collapse; verify expert utilization."})
    return findings[:50]
