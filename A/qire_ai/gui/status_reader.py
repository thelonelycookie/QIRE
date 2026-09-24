from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from qire_ai.config import LOGS_DIR


def read_live_status() -> Dict[str, Any]:
    path = LOGS_DIR / "live_status.json"
    if not path.exists():
        return {"status": "idle", "message": "No live_status.json yet"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
