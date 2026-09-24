from __future__ import annotations

def evidence_level(num_seeds: int, num_datasets: int, has_fair_baseline: bool) -> str:
    if num_seeds >= 10 and num_datasets >= 3 and has_fair_baseline:
        return "strong"
    if num_seeds >= 5 and num_datasets >= 2 and has_fair_baseline:
        return "moderate"
    if num_seeds >= 3 and num_datasets >= 1:
        return "weak"
    return "insufficient"
