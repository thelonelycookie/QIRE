from __future__ import annotations
from typing import Any, Dict, List
from .result_analyzer import summarize_results


def generate_hypotheses(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = summarize_results(rows)
    hypotheses = []
    if summary["top_classical"]:
        best = summary["top_classical"][0]
        hypotheses.append({
            "hypothesis_id": "AUTO-H-CLASSICAL-001",
            "hypothesis": f"{best['model_name']} / {best['method']} is currently the strongest classical efficiency baseline on {best['dataset']}; QIRE variants must be compared against it before broad efficiency claims.",
            "needed_test": "reproduce on Wine, Breast Cancer and Digits with >=10 seeds",
            "evidence_level": "exploratory" if best.get("n", 0) < 5 else "moderate",
        })
    if summary["top_qire"]:
        bestq = summary["top_qire"][0]
        hypotheses.append({
            "hypothesis_id": "AUTO-H-QIRE-001",
            "hypothesis": f"The strongest observed QIRE variant is {bestq['method']} on {bestq['dataset']}; it should be tested first against StandardMoE, then MLP, then PrunedMLP.",
            "needed_test": "paired comparison against StandardMoE/MLP/PrunedMLP across matched seeds",
            "evidence_level": "exploratory" if bestq.get("n", 0) < 5 else "moderate",
        })
    if summary["failed_count"]:
        hypotheses.append({
            "hypothesis_id": "AUTO-H-FAIL-001",
            "hypothesis": "Failures should be mined as configuration constraints rather than ignored.",
            "needed_test": "inspect failures and add schema blocks for repeated invalid combinations",
            "evidence_level": "engineering",
        })
    if not hypotheses:
        hypotheses.append({
            "hypothesis_id": "AUTO-H-BOOTSTRAP-001",
            "hypothesis": "No data are available yet; start with baseline, router_lab and synthetic_benchmark before enabling strong discovery claims.",
            "needed_test": "quick + baseline + router_lab",
            "evidence_level": "planning",
        })
    return hypotheses
