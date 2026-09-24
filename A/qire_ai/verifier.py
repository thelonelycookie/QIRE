from __future__ import annotations

import math
import warnings
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy ships transitively via scikit-learn
    _scipy_stats = None


def _f(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _seed_of(row: Dict[str, object]) -> Optional[int]:
    try:
        return int(row.get("seed"))  # type: ignore[arg-type]
    except Exception:
        return None


def paired_significance_test(
    baseline_rows: List[Dict[str, object]], new_rows: List[Dict[str, object]], metric: str = "balanced_accuracy"
) -> Dict[str, object]:
    """Paired comparison of a metric across matched seeds (same seed in both groups).

    Uses a two-sided Wilcoxon signed-rank test (primary, distribution-free) with a
    paired t-test reported alongside for context. Both require the same seed to be
    present in the baseline and the new-model rows; unmatched seeds are dropped.
    Mean/std across independent runs cannot show whether a difference is likely
    real versus sampling noise, so this is required before any claim of improvement.
    """
    base_by_seed = {_seed_of(r): _f(r, metric, _f(r, "accuracy")) for r in baseline_rows if _seed_of(r) is not None}
    new_by_seed = {_seed_of(r): _f(r, metric, _f(r, "accuracy")) for r in new_rows if _seed_of(r) is not None}
    common_seeds = sorted(set(base_by_seed) & set(new_by_seed))
    n_paired = len(common_seeds)
    result: Dict[str, object] = {
        "n_paired": n_paired,
        "wilcoxon_p_value": None,
        "ttest_p_value": None,
        "mean_paired_diff": None,
        "significant_p05": False,
        "note": "",
    }
    if n_paired < 2:
        result["note"] = "fewer than 2 matched seeds; cannot run a paired test"
        return result
    base_vals = [base_by_seed[s] for s in common_seeds]
    new_vals = [new_by_seed[s] for s in common_seeds]
    diffs = [b - a for a, b in zip(base_vals, new_vals)]
    result["mean_paired_diff"] = round(mean(diffs), 6)
    if n_paired < 5:
        result["note"] = "fewer than 5 matched seeds; test is underpowered, treat as preliminary"
    if _scipy_stats is None:
        result["note"] = (result["note"] + "; scipy unavailable, no p-value computed").strip("; ")
        return result
    # Near-identical paired values (common when a config barely changes a metric)
    # trigger scipy RuntimeWarnings about precision loss in its variance
    # calculation; the near-zero variance itself is the reason the test isn't
    # meaningful there, so it's reported via the note instead of a p-value.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            if any(d != 0 for d in diffs):
                _, p_wilcoxon = _scipy_stats.wilcoxon(new_vals, base_vals)
                p_wilcoxon = float(p_wilcoxon)
                if math.isnan(p_wilcoxon):
                    result["note"] = (result["note"] + "; wilcoxon p-value undefined (near-zero variance in paired differences)").strip("; ")
                else:
                    result["wilcoxon_p_value"] = round(p_wilcoxon, 6)
            else:
                result["wilcoxon_p_value"] = 1.0
        except Exception as exc:
            result["note"] = (result["note"] + f"; wilcoxon failed: {exc}").strip("; ")
        try:
            _, p_ttest = _scipy_stats.ttest_rel(new_vals, base_vals)
            p_ttest = float(p_ttest)
            if math.isnan(p_ttest):
                result["note"] = (result["note"] + "; t-test p-value undefined (near-zero variance in paired differences)").strip("; ")
            else:
                result["ttest_p_value"] = round(p_ttest, 6)
        except Exception as exc:
            result["note"] = (result["note"] + f"; paired t-test failed: {exc}").strip("; ")
    p_values = [p for p in (result["wilcoxon_p_value"], result["ttest_p_value"]) if p is not None]
    result["significant_p05"] = bool(p_values) and all(p < 0.05 for p in p_values)
    return result


def aggregate_by_model_dataset(raw_results: Iterable[Dict[str, object]]) -> Dict[tuple, Dict[str, float | str]]:
    grouped = defaultdict(list)
    for row in raw_results:
        if str(row.get("status", "ok")) != "ok":
            continue
        key = (row.get("dataset"), row.get("model_name"), row.get("method"))
        grouped[key].append(row)
    agg = {}
    for key, rows in grouped.items():
        accs = [_f(r, "accuracy") for r in rows]
        bals = [_f(r, "balanced_accuracy", _f(r, "accuracy")) for r in rows]
        agg[key] = {
            "dataset": str(key[0]),
            "model_name": str(key[1]),
            "method": str(key[2]),
            "accuracy_mean": mean(accs),
            "accuracy_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "balanced_accuracy_mean": mean(bals),
            "balanced_accuracy_std": pstdev(bals) if len(bals) > 1 else 0.0,
            "loss_mean": mean(_f(r, "loss") for r in rows),
            "training_time_mean": mean(_f(r, "training_time_sec") for r in rows),
            "inference_ms_mean": mean(_f(r, "inference_time_ms") for r in rows),
            "inference_p95_mean": mean(_f(r, "inference_time_ms_p95") for r in rows),
            "memory_mb_mean": mean(_f(r, "memory_mb") for r in rows),
            "peak_memory_mb_mean": mean(_f(r, "peak_memory_mb") for r in rows),
            "ops_mean": mean(_f(r, "estimated_ops") for r in rows),
            "router_ops_mean": mean(_f(r, "router_ops") for r in rows),
            "expert_ops_mean": mean(_f(r, "expert_ops") for r in rows),
            "energy_wh_mean": mean(_f(r, "estimated_energy_wh") for r in rows),
            "active_parts_mean": mean(_f(r, "active_model_parts") for r in rows),
            "n": len(rows),
            "rows": rows,
        }
    return agg


def _baseline_for(agg: Dict[tuple, Dict[str, float | str]], dataset: str) -> Dict[str, float | str] | None:
    # Dense MLP is primary scientific baseline. If absent, use logistic, then any classical model.
    priorities = ["MLP", "LogisticRegression", "PrunedMLP", "StandardMoE"]
    for target in priorities:
        candidates = [v for (ds, model, method), v in agg.items() if ds == dataset and model == target]
        if candidates:
            return sorted(candidates, key=lambda x: (float(x["ops_mean"]), -float(x["accuracy_mean"])))[0]
    candidates = [v for (ds, model, method), v in agg.items() if ds == dataset]
    return sorted(candidates, key=lambda x: (float(x["ops_mean"]), -float(x["accuracy_mean"])))[0] if candidates else None


def _pct_saving(base: float, new: float) -> float:
    return 100.0 * (base - new) / max(1e-12, base)


def build_comparisons(raw_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    agg = aggregate_by_model_dataset(raw_results)
    rows = []
    idx = 1
    for (dataset, model_name, method), metrics in sorted(agg.items()):
        baseline = _baseline_for(agg, str(dataset))
        if baseline is None or (model_name == baseline["model_name"] and method == baseline["method"]):
            continue
        acc_diff = float(metrics["accuracy_mean"]) - float(baseline["accuracy_mean"])
        bal_diff = float(metrics["balanced_accuracy_mean"]) - float(baseline["balanced_accuracy_mean"])
        loss_diff = float(metrics["loss_mean"]) - float(baseline["loss_mean"])
        time_saving = _pct_saving(float(baseline["inference_ms_mean"]), float(metrics["inference_ms_mean"]))
        p95_saving = _pct_saving(float(baseline["inference_p95_mean"]), float(metrics["inference_p95_mean"]))
        mem_saving = _pct_saving(float(baseline["memory_mb_mean"]), float(metrics["memory_mb_mean"]))
        ops_reduction = _pct_saving(float(baseline["ops_mean"]), float(metrics["ops_mean"]))
        energy_saving = _pct_saving(float(baseline["energy_wh_mean"]), float(metrics["energy_wh_mean"]))
        router_ops_percent = 100.0 * float(metrics["router_ops_mean"]) / max(1.0, float(metrics["ops_mean"]))
        efficiency = float(metrics["accuracy_mean"]) / max(1.0, float(metrics["ops_mean"]))
        sig = paired_significance_test(baseline["rows"], metrics["rows"], metric="balanced_accuracy")  # type: ignore[arg-type]
        significant_decline = bool(sig["significant_p05"]) and bal_diff < 0
        # estimated_ops is a theoretical FLOP-count proxy; it can and does diverge sharply from
        # measured wall-clock time on tiny CPU models, where Python/PyTorch per-call dispatch
        # overhead (e.g. the expert-routing loop) dominates any real compute saved by sparsity.
        ops_time_conflict = ops_reduction > 0 and time_saving < -20
        if acc_diff >= -0.02 and bal_diff >= -0.02 and ops_reduction > 15 and time_saving > -25 and not significant_decline:
            pareto = "strong preliminary"
            result = "partly confirmed"
        elif acc_diff >= -0.03 and ops_reduction > 0 and time_saving < -100 and not significant_decline:
            pareto = "ops-only (theoretical); measured wall-clock time much worse"
            result = "partly confirmed with limitation (ops proxy only, not a real speedup)"
        elif acc_diff >= -0.03 and ops_reduction > 0 and not significant_decline:
            pareto = "ops promising, runtime/accuracy limited"
            result = "partly confirmed with limitation"
        elif acc_diff >= -0.03 and time_saving > 0 and not significant_decline:
            pareto = "runtime promising, ops questionable"
            result = "partly confirmed with limitation"
        elif significant_decline:
            pareto = "not dominant"
            result = "not confirmed (significant accuracy loss)"
        else:
            pareto = "not dominant"
            result = "not confirmed"
        rows.append({
            "comparison_id": f"C{idx:03d}",
            "dataset": dataset,
            "baseline_model": f"{baseline['model_name']}:{baseline['method']}",
            "new_model": model_name,
            "new_method": method,
            "accuracy_difference": round(acc_diff, 6),
            "balanced_accuracy_difference": round(bal_diff, 6),
            "loss_difference": round(loss_diff, 6),
            "time_saving_percent": round(time_saving, 3),
            "p95_time_saving_percent": round(p95_saving, 3),
            "memory_saving_percent": round(mem_saving, 3),
            "ops_reduction_percent": round(ops_reduction, 3),
            "router_ops_percent": round(router_ops_percent, 3),
            "energy_saving_percent": round(energy_saving, 3),
            "efficiency_per_accuracy": round(efficiency, 10),
            "pareto_rating": pareto,
            "result": result,
            "n_baseline": int(baseline["n"]),
            "n_new": int(metrics["n"]),
            "paired_n": sig["n_paired"],
            "balanced_accuracy_wilcoxon_p": sig["wilcoxon_p_value"],
            "balanced_accuracy_ttest_p": sig["ttest_p_value"],
            "significant_at_p05": sig["significant_p05"],
            "paired_test_note": sig["note"],
            "ops_time_conflict": ops_time_conflict,
        })
        idx += 1
    return rows


def build_verification_reports(raw_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    comparisons = build_comparisons(raw_results)
    reports = []
    for i, c in enumerate(comparisons, start=1):
        claim = f"{c['new_model']} ({c['new_method']}) improves resource use versus {c['baseline_model']} on {c['dataset']} without major accuracy loss."
        paired_n = int(c.get("paired_n", 0))
        enough_seeds = paired_n >= 5
        significant_decline = bool(c.get("significant_at_p05")) and float(c["balanced_accuracy_difference"]) < 0
        ops_time_conflict = bool(c.get("ops_time_conflict"))
        if significant_decline:
            classification = "not confirmed: statistically significant accuracy loss (paired test)"
        elif ops_time_conflict:
            classification = "not confirmed as a real speedup: ops-count proxy improves but measured wall-clock time is much worse"
        elif c["accuracy_difference"] >= -0.02 and c["balanced_accuracy_difference"] >= -0.02 and c["ops_reduction_percent"] > 15:
            classification = "preliminarily supported" if enough_seeds else "weak preliminary support; needs more matched seeds"
        elif c["accuracy_difference"] < -0.03 or c["balanced_accuracy_difference"] < -0.03:
            classification = "not confirmed due to accuracy loss"
        else:
            classification = "unclear or not confirmed"
        p_wilcoxon = c.get("balanced_accuracy_wilcoxon_p")
        p_ttest = c.get("balanced_accuracy_ttest_p")
        significance_summary = (
            f"paired n={paired_n}; Wilcoxon p={p_wilcoxon if p_wilcoxon is not None else 'n/a'}; "
            f"paired t-test p={p_ttest if p_ttest is not None else 'n/a'}"
            + (f"; note: {c['paired_test_note']}" if c.get("paired_test_note") else "")
        )
        limitations = (
            "Energy is estimated; wall-time depends on Python/CPU overhead; router cost and real hardware energy "
            "need separate validation. Small sample sizes (<5 matched seeds) make p-values unreliable."
        )
        if ops_time_conflict:
            limitations += (
                f" estimated_ops fell by {c['ops_reduction_percent']}% but measured inference time changed by "
                f"{c['time_saving_percent']}% (negative = slower): the ops proxy is a theoretical FLOP count and "
                "does not reflect real speed here, likely due to Python-level per-expert routing overhead "
                "dominating at this model/batch scale."
            )
        reports.append({
            "claim_id": f"V{i:03d}",
            "claim": claim,
            "verification_method": "Automated repeated CPU experiment; paired Wilcoxon signed-rank and paired t-test on balanced accuracy across matched seeds versus selected baseline.",
            "data_used": f"Raw Results rows for {c['dataset']}: {c['baseline_model']} vs {c['new_model']}:{c['new_method']}",
            "result": c["result"],
            "classification": classification,
            "evidence_strength": "moderate" if enough_seeds else "weak/preliminary",
            "significance": significance_summary,
            "limitations": limitations,
            "next_required_test": "Repeat with >=5 matched seeds; compare QIRE gates against Softmax-State and StandardMoE; analyze router overhead separately.",
        })
    return reports


def build_run_quality_reports(raw_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    failed = defaultdict(int)
    for row in raw_results:
        key = (row.get("dataset"), row.get("model_name"), row.get("method"))
        if str(row.get("status", "ok")) == "ok":
            grouped[key].append(row)
        else:
            failed[key] += 1
    reports = []
    idx = 1
    for key, rows in sorted(grouped.items()):
        accs = [_f(r, "accuracy") for r in rows]
        bals = [_f(r, "balanced_accuracy", _f(r, "accuracy")) for r in rows]
        nan_count = sum(1 for r in rows if str(r.get("nan_detected", "False")) == "True")
        acc_std = pstdev(accs) if len(accs) > 1 else 0.0
        bal_std = pstdev(bals) if len(bals) > 1 else 0.0
        if nan_count or failed.get(key, 0):
            flag = "technical risk"
            rec = "Inspect errors and rerun after fixing instability."
        elif len(rows) < 3:
            flag = "too few seeds"
            rec = "Run at least 3 seeds; final claims should use >=5."
        elif acc_std > 0.06 or bal_std > 0.06:
            flag = "unstable across seeds"
            rec = "Increase seeds or simplify model/training."
        else:
            flag = "usable preliminary"
            rec = "Can be used for preliminary comparison; reproduce before final claim."
        reports.append({
            "quality_id": f"Q{idx:03d}",
            "dataset": str(key[0]),
            "model_name": str(key[1]),
            "method": str(key[2]),
            "n": len(rows),
            "accuracy_std": round(acc_std, 6),
            "balanced_accuracy_std": round(bal_std, 6),
            "loss_mean": round(mean(_f(r, "loss") for r in rows), 6),
            "nan_count": nan_count,
            "failed_count": failed.get(key, 0),
            "quality_flag": flag,
            "recommendation": rec,
        })
        idx += 1
    return reports
