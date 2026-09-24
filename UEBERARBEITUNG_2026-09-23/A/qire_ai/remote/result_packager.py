from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from qire_ai.config import RESULTS_DIR, EXPORTS_DIR, RAW_DIR, REPORTS_DIR, FIGURES_DIR, EXCEL_DIR, LOGS_DIR


def _iter_existing(paths: Iterable[Path]):
    for p in paths:
        if p.exists():
            yield p


def pack_results(label: str = "qire_results") -> dict:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = EXPORTS_DIR / f"{label}_{stamp}.zip"
    include_roots = [RAW_DIR, REPORTS_DIR, FIGURES_DIR, EXCEL_DIR, LOGS_DIR]
    count = 0
    bytes_total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root in include_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    # Avoid recursively adding previous huge export ZIPs or transient probes.
                    if path.name.startswith(".write_probe"):
                        continue
                    arc = path.relative_to(RESULTS_DIR.parent)
                    zf.write(path, arc.as_posix())
                    count += 1
                    bytes_total += path.stat().st_size
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_count": count,
            "uncompressed_bytes": bytes_total,
            "source_results_dir": str(RESULTS_DIR),
        }
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"archive": str(out), "file_count": count, "uncompressed_mb": round(bytes_total / (1024**2), 3), "archive_mb": round(out.stat().st_size / (1024**2), 3)}


def print_pack_results(label: str = "qire_results") -> None:
    print(json.dumps(pack_results(label=label), indent=2, ensure_ascii=False))
