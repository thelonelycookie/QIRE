from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from qire_ai.config import RESULTS_DIR, LOGS_DIR, EXPORTS_DIR, RAW_DIR


def storage_report() -> dict:
    disk = shutil.disk_usage(str(RESULTS_DIR))
    largest = []
    for p in RESULTS_DIR.rglob("*"):
        if p.is_file():
            largest.append((p.stat().st_size, str(p.relative_to(RESULTS_DIR))))
    largest.sort(reverse=True)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disk_free_gb": round(disk.free/(1024**3), 3),
        "disk_total_gb": round(disk.total/(1024**3), 3),
        "largest_files": [{"path": p, "mb": round(s/(1024**2), 3)} for s,p in largest[:20]],
    }


def compress_jsonl_logs() -> dict:
    compressed = 0
    for p in LOGS_DIR.glob("*.jsonl"):
        gz = p.with_suffix(p.suffix + ".gz")
        if gz.exists() or p.stat().st_size < 2_000_000:
            continue
        with p.open("rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        compressed += 1
    return {"compressed_logs": compressed, "storage": storage_report()}
