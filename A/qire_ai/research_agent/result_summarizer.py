from __future__ import annotations
from typing import Any, Dict, List
from qire_ai.discovery.result_analyzer import summarize_results


def summarize_for_agent(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_results(rows)
    return {
        "counts": {"raw": summary["raw_count"], "ok": summary["ok_count"], "failed": summary["failed_count"]},
        "top_overall": summary["top_overall"][:8],
        "top_qire": summary["top_qire"][:8],
        "top_classical": summary["top_classical"][:8],
        "warning": "Use as exploratory input only. Claims require verification over seeds and fair baselines.",
    }
