from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from qire_ai.config import REPORTS_DIR


def write_discovery_report(payload: Dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "discovery_report.md"
    lines: List[str] = []
    lines.append("# QIRE AI Console v1.0 Discovery Report")
    lines.append("")
    lines.append("This report is exploratory. It proposes testable next steps; it does not prove efficiency claims.")
    lines.append("")
    summary = payload.get("summary", {})
    lines.append("## Data Summary")
    lines.append(f"- raw rows: {summary.get('raw_count', 0)}")
    lines.append(f"- ok rows: {summary.get('ok_count', 0)}")
    lines.append(f"- failed rows: {summary.get('failed_count', 0)}")
    lines.append("")
    lines.append("## Top Overall Groups")
    for g in summary.get("top_overall", [])[:10]:
        lines.append(f"- {g.get('dataset')} | {g.get('model_name')} | {g.get('method')} | n={g.get('n')} | bal={g.get('balanced_accuracy_mean'):.4f} | ops={g.get('estimated_ops_mean'):.1f}")
    lines.append("")
    lines.append("## Auto Hypotheses")
    for h in payload.get("hypotheses", []):
        lines.append(f"- **{h.get('hypothesis_id')}**: {h.get('hypothesis')}  ")
        lines.append(f"  Needed test: {h.get('needed_test')}  ")
        lines.append(f"  Evidence: {h.get('evidence_level')}")
    lines.append("")
    lines.append("## Discovery Proposals")
    for p in payload.get("proposals", [])[:50]:
        lines.append(f"### {p.get('proposal_id')} — {p.get('type')}")
        lines.append(f"Observation: {p.get('observation')}")
        lines.append(f"Hypothesis: {p.get('hypothesis')}")
        lines.append(f"Config: `{p.get('config')}`")
        lines.append(f"Baselines: {', '.join(p.get('baseline', []))}")
        lines.append(f"Datasets: {', '.join(p.get('datasets', []))}")
        lines.append(f"Success: {p.get('success_criterion')}")
        lines.append(f"Falsification: {p.get('falsification_criterion')}")
        lines.append(f"Risk: {p.get('risk')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
