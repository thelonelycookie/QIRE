from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from qire_ai.search_spaces import ExperimentJob

QIRE_GATES = {"born", "softmax_state", "interference_born", "hard_threshold", "gumbel_softmax", "entropy_adaptive", "budget_aware", "hybrid_softmax_born"}
MOE_GATES = {"softmax", "random", "fixed", "top1_softmax", "top2_softmax"}
BASELINE_GATES = {"none", "baseline", ""}
STATE_MAPPERS = {"none", "linear", "low_rank", "trig", "normalized", "cluster_based", "compact_trig", "poly2"}
EXPERT_TYPES = {"mini", "standard", "low_rank", "pruned", "shared_backbone", "specialist"}
ACTIVATIONS = {"top_1", "top_2", "adaptive_top_k", "threshold_based", "budget_based"}

@dataclass
class ResearchProposal:
    proposal_id: str
    type: str
    observation: str
    hypothesis: str
    model_family: str
    config: Dict[str, Any]
    baseline: List[str]
    datasets: List[str]
    success_criterion: str
    falsification_criterion: str
    risk: str
    expected_advantage: str = ""
    novelty_score: float = 0.0
    risk_score: float = 0.0
    scientific_value: float = 0.0
    status: str = "proposed"
    source: str = "rule_based_discovery"
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def make_id(payload: Dict[str, Any], prefix: str = "DISC") -> str:
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
        return f"{prefix}-{digest}"


def validate_proposal(proposal: ResearchProposal) -> Tuple[bool, str]:
    if not proposal.hypothesis.strip():
        return False, "proposal has no hypothesis"
    if not proposal.baseline:
        return False, "proposal has no baseline"
    if not proposal.datasets:
        return False, "proposal has no datasets"
    family = proposal.model_family.lower()
    cfg = proposal.config or {}
    gate = str(cfg.get("gate_type", "none"))
    if family == "qire":
        if gate not in QIRE_GATES:
            return False, f"QIRE proposal uses invalid gate_type={gate!r}"
        mapper = str(cfg.get("state_mapper", "linear"))
        if mapper not in STATE_MAPPERS:
            return False, f"QIRE proposal uses invalid state_mapper={mapper!r}"
        top_k = int(cfg.get("top_k", 1))
        experts = int(cfg.get("num_experts", cfg.get("experts", 4)))
        if top_k < 1 or top_k > experts:
            return False, f"invalid top_k={top_k} for experts={experts}"
    elif family in {"moe", "standard_moe"}:
        if gate not in MOE_GATES:
            return False, f"MoE proposal uses invalid gate_type={gate!r}"
    elif family in {"baseline", "classical", "mlp", "pruned_mlp", "lowrank_mlp"}:
        if gate not in BASELINE_GATES:
            return False, f"baseline proposal must not use gate_type={gate!r}"
    else:
        return False, f"unknown model_family={proposal.model_family!r}"
    return True, "ok"


def proposal_to_jobs(proposal: ResearchProposal, seeds: List[int] | None = None, epochs: int = 80) -> List[ExperimentJob]:
    ok, reason = validate_proposal(proposal)
    if not ok:
        raise ValueError(reason)
    seeds = seeds or [1, 2, 3, 4, 5]
    cfg = proposal.config or {}
    jobs: List[ExperimentJob] = []
    family = proposal.model_family.lower()
    if family == "qire":
        for ds in proposal.datasets:
            for seed in seeds:
                jobs.append(ExperimentJob(
                    dataset=ds,
                    model_key="qire",
                    seed=seed,
                    epochs=int(cfg.get("epochs", epochs)),
                    lr=float(cfg.get("lr", 0.01)),
                    hidden_dim=int(cfg.get("hidden_dim", 32)),
                    expert_hidden_dim=int(cfg.get("expert_hidden_dim", cfg.get("expert_hidden", 12))),
                    num_experts=int(cfg.get("num_experts", cfg.get("experts", 4))),
                    top_k=int(cfg.get("top_k", 1)),
                    gate_type=str(cfg.get("gate_type", "born")),
                    state_mapper=str(cfg.get("state_mapper", "linear")),
                    lambda_resource=float(cfg.get("lambda_resource", 0.0)),
                    mu_activation=float(cfg.get("mu_activation", 0.0)),
                    temperature=float(cfg.get("temperature", 1.0)),
                    prune_amount=float(cfg.get("prune_amount", 0.0)),
                    shared_backbone=bool(cfg.get("shared_backbone", False)),
                    search_mode="discovery_proposal",
                    hypothesis_id=proposal.proposal_id,
                    experiment_id="DISCOVERY",
                    parent_job_id=proposal.proposal_id,
                    notes=f"Discovery proposal: {proposal.hypothesis} | risk={proposal.risk}",
                ))
    elif family in {"moe", "standard_moe"}:
        for ds in proposal.datasets:
            for seed in seeds:
                jobs.append(ExperimentJob(ds, "standard_moe", seed, epochs=epochs, expert_hidden_dim=int(cfg.get("expert_hidden_dim", 12)), num_experts=int(cfg.get("num_experts", 4)), top_k=int(cfg.get("top_k", 1)), gate_type="softmax", search_mode="discovery_proposal", hypothesis_id=proposal.proposal_id, experiment_id="DISCOVERY", parent_job_id=proposal.proposal_id))
    else:
        model_key = str(cfg.get("model_key", family if family not in {"baseline", "classical"} else "mlp"))
        if model_key == "baseline":
            model_key = "mlp"
        for ds in proposal.datasets:
            for seed in seeds:
                jobs.append(ExperimentJob(ds, model_key, seed, epochs=epochs, hidden_dim=int(cfg.get("hidden_dim", 32)), prune_amount=float(cfg.get("prune_amount", 0.5)), search_mode="discovery_proposal", hypothesis_id=proposal.proposal_id, experiment_id="DISCOVERY", parent_job_id=proposal.proposal_id))
    return jobs
