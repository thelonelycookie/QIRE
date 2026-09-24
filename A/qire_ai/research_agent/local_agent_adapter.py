from __future__ import annotations
from typing import Any, Dict, List
from qire_ai.discovery.negative_result_miner import mine_negative_results
from qire_ai.discovery.morphological_generator import generate_morphological_proposals


def run_local_agent(rows: List[Dict[str, Any]], limit: int = 30):
    """Rule-based local fallback agent.

    It deliberately does not call a local LLM. This keeps v0.6 functional without
    Ollama/API setup while preserving the same proposal interface.
    """
    proposals = []
    proposals.extend(mine_negative_results(rows))
    proposals.extend(generate_morphological_proposals(limit=limit))
    return proposals[:limit]
