from __future__ import annotations

import itertools
from typing import Dict, List

from .proposal_schema import ResearchProposal, validate_proposal

STATE_MAPPERS = ["linear", "low_rank", "compact_trig", "cluster_based"]
ROUTERS = ["born", "interference_born", "entropy_adaptive", "budget_aware", "hybrid_softmax_born"]
EXPERT_TYPES = ["mini", "low_rank", "pruned", "shared_backbone"]
TOP_KS = [1, 2]


def generate_morphological_proposals(limit: int = 60, datasets: List[str] | None = None) -> List[ResearchProposal]:
    datasets = datasets or ["wine", "breast_cancer", "digits", "synthetic_routing_challenge"]
    proposals: List[ResearchProposal] = []
    for mapper, gate, expert_type, top_k in itertools.product(STATE_MAPPERS, ROUTERS, EXPERT_TYPES, TOP_KS):
        experts = 4 if expert_type != "shared_backbone" else 6
        expert_hidden = 8 if expert_type == "mini" else (12 if expert_type in {"low_rank", "pruned"} else 16)
        cfg: Dict[str, object] = {
            "state_mapper": mapper,
            "gate_type": gate,
            "top_k": top_k,
            "num_experts": experts,
            "expert_hidden_dim": expert_hidden,
            "expert_type": expert_type,
            "lambda_resource": 1e-5 if gate in {"budget_aware", "entropy_adaptive"} else 0.0,
            "mu_activation": 1e-4 if top_k == 1 else 0.0,
            "shared_backbone": expert_type == "shared_backbone",
            "temperature": 0.75 if gate in {"budget_aware", "entropy_adaptive"} else 1.0,
        }
        payload = {"kind": "morph", **cfg}
        p = ResearchProposal(
            proposal_id=ResearchProposal.make_id(payload),
            type="qire_variant",
            observation="Morphological generator combines QIRE state mapper, router, activation and expert components.",
            hypothesis=f"{gate}+{mapper}+top{top_k}+{expert_type} may improve the accuracy/resource Pareto trade-off against StandardMoE.",
            model_family="qire",
            config=cfg,
            baseline=["StandardMoE", "PrunedMLP", "MLP"],
            datasets=datasets[:3] if expert_type != "shared_backbone" else datasets,
            success_criterion=">=97% of StandardMoE balanced accuracy and >=20% lower estimated ops than StandardMoE",
            falsification_criterion="lower balanced accuracy and higher estimated ops than StandardMoE on the same datasets",
            risk="May still be dominated by PrunedMLP or suffer router overhead.",
            expected_advantage="Structured sparse routing with smaller or resource-aware components.",
            novelty_score=0.55 + (0.1 if gate == "hybrid_softmax_born" else 0.0) + (0.1 if expert_type in {"pruned", "low_rank"} else 0.0),
            risk_score=0.35 + (0.2 if mapper == "compact_trig" else 0.0),
            scientific_value=0.7,
        )
        ok, _ = validate_proposal(p)
        if ok:
            proposals.append(p)
        if len(proposals) >= limit:
            break
    return proposals
