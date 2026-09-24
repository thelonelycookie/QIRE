from __future__ import annotations
from collections import defaultdict
from typing import Sequence, Mapping, Any
from .evidence_levels import evidence_level


def gatekeeper_summary(rows: Sequence[Mapping[str, Any]]) -> dict:
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    by_model = defaultdict(lambda: {"seeds": set(), "datasets": set(), "n": 0})
    for r in ok:
        key = (str(r.get("model_name")), str(r.get("method")))
        by_model[key]["n"] += 1
        by_model[key]["seeds"].add(str(r.get("seed")))
        by_model[key]["datasets"].add(str(r.get("dataset")))
    out = []
    for (model, method), stats in by_model.items():
        has_baseline = any("MLP" in str(r.get("model_name")) or "Logistic" in str(r.get("model_name")) or "MoE" in str(r.get("model_name")) for r in ok)
        out.append({
            "model": model,
            "method": method,
            "n": stats["n"],
            "num_seeds": len(stats["seeds"]),
            "num_datasets": len(stats["datasets"]),
            "evidence_level": evidence_level(len(stats["seeds"]), len(stats["datasets"]), has_baseline),
        })
    return {"claim_gate": out, "rule": "No strong efficiency claim without >=10 seeds, >=3 datasets, and fair baselines."}
