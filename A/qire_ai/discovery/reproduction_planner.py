from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from .result_analyzer import score_row
from .proposal_schema import ResearchProposal, proposal_to_jobs
from qire_ai.config import REPORTS_DIR, LOGS_DIR


def plan_reproduction(rows: List[Dict[str, Any]], limit: int = 8) -> Dict[str, Any]:
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    ranked = sorted(ok, key=score_row, reverse=True)[:limit]
    proposals = []
    for r in ranked:
        model = str(r.get("model_name", ""))
        is_qire = model.lower().startswith("qire")
        cfg = {}
        try:
            cfg = json.loads(str(r.get("config_json", "{}")))
        except Exception:
            cfg = {}
        if is_qire:
            p = ResearchProposal(
                proposal_id=ResearchProposal.make_id({"reproduce": r.get("job_id"), "config": cfg}, prefix="REP"),
                type="reproduction_plan",
                observation=f"Promising QIRE row: {r.get('dataset')} {r.get('method')} score={score_row(r):.4f}",
                hypothesis="Promising QIRE result must be reproduced across seeds and compared against StandardMoE/PrunedMLP before any claim.",
                model_family="qire",
                config={
                    "gate_type": cfg.get("gate_type", r.get("gate_type", "born")),
                    "state_mapper": cfg.get("state_mapper", r.get("state_mapper", "linear")),
                    "top_k": int(float(cfg.get("top_k", r.get("top_k", 1)) or 1)),
                    "num_experts": int(float(cfg.get("num_experts", r.get("num_experts", 4)) or 4)),
                    "expert_hidden_dim": int(float(cfg.get("expert_hidden_dim", r.get("expert_hidden_dim", 12)) or 12)),
                    "lambda_resource": float(cfg.get("lambda_resource", r.get("lambda_resource", 0)) or 0),
                    "mu_activation": float(cfg.get("mu_activation", r.get("mu_activation", 0)) or 0),
                },
                baseline=["StandardMoE", "PrunedMLP", "MLP"],
                datasets=["wine", "breast_cancer", "digits"],
                success_criterion=">=97% baseline balanced accuracy and non-dominated on ops/time",
                falsification_criterion="dominated by PrunedMLP and StandardMoE across seeds",
                risk="Single-run result may be seed noise.",
                novelty_score=0.3,
                risk_score=0.25,
                scientific_value=0.95,
            )
            proposals.append(p)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True); LOGS_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# Reproduction Plan", "", "Good exploratory results are not final evidence. These proposals schedule matched-seed reproduction.", ""]
    for p in proposals:
        md.extend([f"## {p.proposal_id}", f"Hypothesis: {p.hypothesis}", f"Observation: {p.observation}", f"Datasets: {', '.join(p.datasets)}", f"Baselines: {', '.join(p.baseline)}", ""])
    (REPORTS_DIR / "reproduction_plan.md").write_text("\n".join(md), encoding="utf-8")
    with (LOGS_DIR / "reproduction_plan.jsonl").open("w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return {"proposal_count": len(proposals), "report": str(REPORTS_DIR / "reproduction_plan.md"), "proposals": [p.as_dict() for p in proposals]}


def jobs_from_reproduction_plan(rows: List[Dict[str, Any]], seeds: List[int] | None = None, epochs: int = 90):
    plan = plan_reproduction(rows)
    jobs = []
    for raw in plan["proposals"]:
        jobs.extend(proposal_to_jobs(ResearchProposal(**raw), seeds=seeds or list(range(1, 11)), epochs=epochs))
    return jobs
