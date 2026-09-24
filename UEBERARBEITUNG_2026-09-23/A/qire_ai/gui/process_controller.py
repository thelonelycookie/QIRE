from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

class ProcessController:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.process: Optional[subprocess.Popen] = None

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, args: List[str]):
        if self.running():
            raise RuntimeError("A background process is already running")
        cmd = [sys.executable, "qire_ai_console.py"] + args
        self.process = subprocess.Popen(cmd, cwd=str(self.project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        return self.process

    def poll_output(self) -> str:
        if not self.process or not self.process.stdout:
            return ""
        out = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            out.append(line.rstrip())
            if len(out) > 200:
                break
        return "\n".join(out)
