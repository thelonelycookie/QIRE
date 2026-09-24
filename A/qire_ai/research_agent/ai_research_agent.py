from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from qire_ai.config import REPORTS_DIR, LOGS_DIR
from qire_ai.research_store import ResearchStore
from qire_ai.discovery.proposal_schema import ResearchProposal
from .result_summarizer import summarize_for_agent
from .safety_validator import validate_agent_proposals
from .local_agent_adapter import run_local_agent


def run_ai_research_agent(mode: str = "off", limit: int = 40) -> Dict[str, Any]:
    store = ResearchStore(); rows = store.load_raw_results(include_failed=True)
    summary = summarize_for_agent(rows)
    proposals: List[ResearchProposal] = []
    if mode == "off":
        pass
    elif mode in {"rules", "local"}:
        if mode == "rules":
            from .rule_agent import run_rule_agent
            proposals = run_rule_agent(rows, limit=limit)
        else:
            proposals = run_local_agent(rows, limit=limit)
    elif mode == "external":
        from .external_agent_adapter import run_external_agent
        proposals = run_external_agent(rows=rows, limit=limit)
    else:
        raise ValueError(f"Unsupported AI agent mode: {mode}. Use off/rules/local/external.")
    valid, blocked = validate_agent_proposals(proposals)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True); LOGS_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# AI Research Proposals", "", f"Agent mode: `{mode}`", "", "AI/agent proposals are hypotheses, not evidence.", "", "## Summary", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False)[:6000], "```", "", "## Validated Proposals"]
    for p in valid:
        md.extend([f"### {p.proposal_id}", p.hypothesis, f"Risk: {p.risk}", ""])
    if blocked:
        md.append("## Blocked Proposals")
        for b in blocked[:20]:
            md.append(f"- {b.get('proposal_id')}: {b.get('blocked_reason')}")
    (REPORTS_DIR / "ai_research_proposals.md").write_text("\n".join(md), encoding="utf-8")
    with (LOGS_DIR / "ai_research_proposals.jsonl").open("w", encoding="utf-8") as f:
        for p in valid:
            f.write(json.dumps(p.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return {"mode": mode, "valid_proposals": len(valid), "blocked_proposals": len(blocked), "report": str(REPORTS_DIR / "ai_research_proposals.md")}
