from __future__ import annotations
from typing import Any, Dict, List
from .morphological_generator import generate_morphological_proposals
from .negative_result_miner import mine_negative_results
from .pareto_gap_finder import find_pareto_gap_proposals


def experimental_architecture_proposals():
    from .proposal_schema import ResearchProposal
    specs = [
        ("TWO_STAGE", "Two-stage QIRE uses a cheap coarse router and only invokes a stronger QIRE route for uncertain samples.", {"state_mapper":"linear","gate_type":"entropy_adaptive","top_k":1,"num_experts":4,"expert_hidden_dim":10,"temperature":1.2,"lambda_resource":1e-5,"mu_activation":1e-4}, "Two-stage behavior is approximated in v0.6 by entropy-adaptive top-1 routing; a later version may add a dedicated two-stage module."),
        ("ROUTER_CACHE", "Router-cache QIRE reuses routing decisions for similar feature clusters to reduce routing cost.", {"state_mapper":"cluster_based","gate_type":"budget_aware","top_k":1,"num_experts":4,"expert_hidden_dim":8,"lambda_resource":3e-5,"mu_activation":3e-4}, "Router cache is represented as a cluster_based mapper proposal; measured caching requires a later explicit cache implementation."),
        ("QUBO_FS", "QUBO-inspired feature selection maximizes feature value while penalizing feature cost and redundancy before QIRE routing.", {"state_mapper":"low_rank","gate_type":"budget_aware","top_k":1,"num_experts":4,"expert_hidden_dim":8,"lambda_resource":3e-5,"mu_activation":2e-4,"prune_amount":0.5}, "QUBO feature selection is approximated through low-rank/resource-aware proposals and literature-card tracking until a full QUBO selector is added."),
    ]
    out=[]
    for prefix, obs, cfg, risk in specs:
        out.append(ResearchProposal(
            proposal_id=ResearchProposal.make_id({"experimental":prefix, **cfg}, prefix=prefix),
            type="experimental_architecture_path",
            observation=obs,
            hypothesis=obs + " It may improve the accuracy/resource trade-off only if overhead stays below expert savings.",
            model_family="qire",
            config=cfg,
            baseline=["StandardMoE","PrunedMLP","MLP"],
            datasets=["synthetic_routing_challenge","feature_group_switching","budget_sensitive_task","digits"],
            success_criterion="non-dominated Pareto position versus StandardMoE and PrunedMLP",
            falsification_criterion="dominated by StandardMoE/PrunedMLP or unstable over seeds",
            risk=risk,
            expected_advantage="New route-generation path for future QIRE variants.",
            novelty_score=0.9,
            risk_score=0.65,
            scientific_value=0.75,
        ))
    return out


def generate_experiment_ideas(rows: List[Dict[str, Any]], limit: int = 100):
    proposals = []
    proposals.extend(mine_negative_results(rows))
    proposals.extend(find_pareto_gap_proposals(rows))
    proposals.extend(experimental_architecture_proposals())
    proposals.extend(generate_morphological_proposals(limit=limit))
    # Dedupe by proposal_id and sort by scientific value then novelty.
    seen = set(); out = []
    for p in sorted(proposals, key=lambda p: (p.scientific_value, p.novelty_score), reverse=True):
        if p.proposal_id in seen: continue
        seen.add(p.proposal_id); out.append(p)
        if len(out) >= limit: break
    return out
