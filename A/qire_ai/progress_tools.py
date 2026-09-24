
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import LOGS_DIR


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def rank_rows(rows: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    def score(r: Dict[str, Any]) -> float:
        try:
            acc = float(r.get("balanced_accuracy") or r.get("accuracy") or 0.0)
            ops = max(1.0, float(r.get("estimated_ops") or 1.0))
            time_ms = max(1e-6, float(r.get("inference_time_ms_p95") or r.get("inference_time_ms") or 1e-6))
            return acc / (ops ** 0.65) / (time_ms ** 0.15)
        except Exception:
            return 0.0
    return sorted([r for r in rows if r.get("status", "ok") == "ok"], key=score, reverse=True)[:n]


def print_live_header(*, preset: str, executed: int, planned: int | str, failed: int, start: float, deadline: float | None, current: str = "") -> None:
    elapsed = time.perf_counter() - start
    if deadline:
        remaining = max(0.0, deadline - time.perf_counter())
        budget_text = f"elapsed {fmt_seconds(elapsed)} | remaining {fmt_seconds(remaining)}"
    else:
        budget_text = f"elapsed {fmt_seconds(elapsed)}"
    print("\n" + "=" * 78)
    print(f"QIRE AI Console v1.0 Remote Research Platform | preset={preset} | {budget_text}")
    print(f"completed={executed} | failed={failed} | planned={planned}")
    if current:
        print(current)
    print("=" * 78)


def write_hourly_summary(summary: Dict[str, Any], index: int) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"hourly_summary_{index:03d}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
