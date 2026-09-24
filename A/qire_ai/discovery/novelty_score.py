from __future__ import annotations
from typing import Any, Dict, Iterable


def config_signature(config: Dict[str, Any]) -> str:
    fields = ["state_mapper", "gate_type", "top_k", "num_experts", "expert_hidden_dim", "expert_type", "lambda_resource", "mu_activation"]
    return "|".join(str(config.get(f, "")) for f in fields)


def novelty_score(config: Dict[str, Any], tested_configs: Iterable[Dict[str, Any]]) -> float:
    sig = config_signature(config)
    tested = [config_signature(c) for c in tested_configs]
    if not tested:
        return 1.0
    min_dist = 1.0
    s_parts = sig.split("|")
    for t in tested:
        t_parts = t.split("|")
        diff = sum(1 for a, b in zip(s_parts, t_parts) if a != b) / max(1, len(s_parts))
        min_dist = min(min_dist, diff)
    return float(max(0.0, min(1.0, min_dist)))
