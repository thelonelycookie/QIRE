from __future__ import annotations
from typing import Any, Dict, List
from .result_analyzer import _f
from .proposal_schema import ResearchProposal


def non_dominated(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    front = []
    for a in ok:
        dominated = False
        for b in ok:
            if a is b: continue
            if _f(b, "balanced_accuracy", _f(b, "accuracy")) >= _f(a, "balanced_accuracy", _f(a, "accuracy")) and _f(b, "estimated_ops", 1e9) <= _f(a, "estimated_ops", 1e9) and _f(b, "inference_time_ms_p95", 1e9) <= _f(a, "inference_time_ms_p95", 1e9):
                if _f(b, "balanced_accuracy", _f(b, "accuracy")) > _f(a, "balanced_accuracy", _f(a, "accuracy")) or _f(b, "estimated_ops") < _f(a, "estimated_ops"):
                    dominated = True; break
        if not dominated:
            front.append(a)
    return front


def find_pareto_gap_proposals(rows: List[Dict[str, Any]]) -> List[ResearchProposal]:
    front = non_dominated(rows)
    if len(front) < 2:
        return []
    front = sorted(front, key=lambda r: _f(r, "estimated_ops"))
    proposals: List[ResearchProposal] = []
    for lo, hi in zip(front, front[1:]):
        acc_gap = _f(hi, "balanced_accuracy", _f(hi, "accuracy")) - _f(lo, "balanced_accuracy", _f(lo, "accuracy"))
        ops_gap = _f(hi, "estimated_ops") - _f(lo, "estimated_ops")
        if ops_gap > 500 and acc_gap > 0.005:
            cfg = {"state_mapper": "low_rank", "gate_type": "hybrid_softmax_born", "top_k": 1, "num_experts": 4, "expert_hidden_dim": 12, "lambda_resource": 1e-5, "mu_activation": 1e-4, "temperature": 0.85}
            p = ResearchProposal(
                proposal_id=ResearchProposal.make_id({"gap": [lo.get("job_id"), hi.get("job_id")], **cfg}, prefix="GAP"),
                type="pareto_gap_candidate",
                observation="Pareto front has a gap between a low-cost lower-accuracy model and a higher-cost higher-accuracy model.",
                hypothesis="A top-1 low-rank hybrid QIRE variant may occupy the Pareto gap between cheap classical baselines and expensive high-accuracy models.",
                model_family="qire",
                config=cfg,
                baseline=[str(lo.get("model_name")), str(hi.get("model_name")), "StandardMoE"],
                datasets=list({str(lo.get("dataset")), str(hi.get("dataset")), "breast_cancer", "digits"}),
                success_criterion="non-dominated on accuracy/ops/inference p95 Pareto comparison",
                falsification_criterion="dominated by both neighboring Pareto models",
                risk="Pareto gap may be noise from insufficient seeds.",
                novelty_score=0.78,
                risk_score=0.38,
                scientific_value=0.9,
            )
            proposals.append(p)
    return proposals[:20]
