from __future__ import annotations
import json
from pathlib import Path
from qire_ai.config import LOGS_DIR, REPORTS_DIR, RAW_DIR, EXCEL_DIR, EXPORTS_DIR


def remote_status() -> dict:
    live = LOGS_DIR / "live_status.json"
    final = LOGS_DIR / "live_status_final.json"
    data = {}
    for p in [live, final]:
        if p.exists():
            try:
                data[p.name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                data[p.name] = {"error": str(exc)}
    files = {}
    for rel in [RAW_DIR/"raw_results.csv", RAW_DIR/"qire_runs.sqlite", REPORTS_DIR/"RUN_BOOK.md", REPORTS_DIR/"recommend_next.md", EXCEL_DIR/"qire_ai_console_results.xlsx"]:
        files[str(rel)] = {"exists": rel.exists(), "bytes": rel.stat().st_size if rel.exists() else 0}
    return {"live": data, "files": files, "exports": [p.name for p in EXPORTS_DIR.glob("*.zip")][-10:]}


def print_remote_status() -> None:
    print(json.dumps(remote_status(), indent=2, ensure_ascii=False))
