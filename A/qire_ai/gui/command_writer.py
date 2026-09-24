from __future__ import annotations
import json
from qire_ai.config import CONTROL_DIR
from datetime import datetime, timezone


def write_command(command: str, source: str = "gui") -> None:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTROL_DIR / "command.json"
    if command == "clear":
        if path.exists():
            path.unlink()
        return
    path.write_text(json.dumps({"command": command, "source": source, "created_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
