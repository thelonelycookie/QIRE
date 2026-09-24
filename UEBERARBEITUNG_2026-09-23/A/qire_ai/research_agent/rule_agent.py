from __future__ import annotations
from typing import Any, Dict, List
from qire_ai.discovery.negative_result_miner import mine_negative_results
from qire_ai.discovery.morphological_generator import generate_morphological_proposals


def run_rule_agent(rows: List[Dict[str, Any]], limit: int = 50):
    proposals = []
    proposals.extend(mine_negative_results(rows))
    proposals.extend(generate_morphological_proposals(limit=limit))
    # Pareto gaps are report objects in v0.6; convert only indirectly by keeping
    # morphological proposals as executable, validated objects.
    return proposals[:limit]
