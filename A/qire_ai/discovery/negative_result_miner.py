from __future__ import annotations
from typing import Any, Dict, List
from .proposal_schema import ResearchProposal
from .result_analyzer import diagnose_qire


def mine_negative_results(rows: List[Dict[str, Any]]) -> List[ResearchProposal]:
    findings = diagnose_qire(rows)
    proposals: List[ResearchProposal] = []
    for f in findings:
        if f.get("type") == "qire_router_overhead":
            cfg = {"state_mapper": "low_rank", "gate_type": "budget_aware", "top_k": 1, "num_experts": 4, "expert_hidden_dim": 8, "lambda_resource": 3e-5, "mu_activation": 3e-4, "temperature": 0.65}
            p = ResearchProposal(
                proposal_id=ResearchProposal.make_id({"negative": f, **cfg}, prefix="NEG"),
                type="negative_result_followup",
                observation=f.get("message", "QIRE overhead detected"),
                hypothesis="A low-rank budget-aware top-1 QIRE variant may reduce router/expert overhead while preserving most accuracy.",
                model_family="qire",
                config=cfg,
                baseline=["StandardMoE", "PrunedMLP"],
                datasets=[str(f.get("dataset") or "wine"), "breast_cancer", "digits"],
                success_criterion=">=97% StandardMoE balanced accuracy and >=25% estimated-ops reduction versus StandardMoE",
                falsification_criterion="still higher ops than StandardMoE or >3% balanced accuracy loss",
                risk="May over-prune and lose accuracy; must be reproduced over seeds.",
                novelty_score=0.72,
                risk_score=0.45,
                scientific_value=0.85,
            )
            proposals.append(p)
        elif f.get("type") == "router_collapse_possible":
            cfg = {"state_mapper": "linear", "gate_type": "entropy_adaptive", "top_k": 2, "num_experts": 4, "expert_hidden_dim": 12, "lambda_resource": 0.0, "mu_activation": 1e-4, "temperature": 1.2}
            p = ResearchProposal(
                proposal_id=ResearchProposal.make_id({"collapse": f, **cfg}, prefix="NEG"),
                type="router_collapse_followup",
                observation=f.get("message", "Router collapse possible"),
                hypothesis="Entropy-adaptive routing with mild activation penalty may reduce expert collapse and improve specialization.",
                model_family="qire",
                config=cfg,
                baseline=["StandardMoE", "RandomGateMoE"],
                datasets=[str(f.get("dataset") or "synthetic_routing_challenge"), "synthetic_expert_specialization"],
                success_criterion="higher expert utilization and no balanced-accuracy drop versus born top-1 QIRE",
                falsification_criterion="expert utilization remains collapsed or accuracy decreases",
                risk="May increase active experts and cost.",
                novelty_score=0.65,
                risk_score=0.4,
                scientific_value=0.8,
            )
            proposals.append(p)
    return proposals[:30]
