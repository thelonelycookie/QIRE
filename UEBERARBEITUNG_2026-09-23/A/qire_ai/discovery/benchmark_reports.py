from __future__ import annotations
from typing import Any, Dict, List
from qire_ai.config import REPORTS_DIR
from .result_analyzer import summarize_results


def _write(path, title: str, rows: List[Dict[str, Any]], filter_token: str):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filtered = [r for r in rows if filter_token in str(r.get('search_mode','')) or filter_token in str(r.get('dataset',''))]
    summary = summarize_results(filtered)
    lines = [f"# {title}", "", "This report is exploratory and does not by itself prove a scientific claim.", ""]
    lines.append(f"Rows: {summary['raw_count']} | ok: {summary['ok_count']} | failed: {summary['failed_count']}")
    lines.append("")
    for g in summary.get('top_overall', [])[:20]:
        lines.append(f"- {g.get('dataset')} | {g.get('model_name')} | {g.get('method')} | n={g.get('n')} | bal={g.get('balanced_accuracy_mean'):.4f} | ops={g.get('estimated_ops_mean'):.1f}")
    path.write_text("\n".join(lines), encoding='utf-8')
    return path


def write_router_lab_report(rows: List[Dict[str, Any]]):
    return _write(REPORTS_DIR / 'router_lab_report.md', 'Router Lab Report', rows, 'router_lab')


def write_synthetic_benchmark_report(rows: List[Dict[str, Any]]):
    return _write(REPORTS_DIR / 'synthetic_benchmark_report.md', 'Synthetic Benchmark Report', rows, 'synthetic')
