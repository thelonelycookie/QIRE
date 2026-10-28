"""llama.cpp-Server starten, Gesundheit pruefen, sauber beenden."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ORDNER = Path(__file__).resolve().parent.parent


def finde_exe(cfg_server: dict) -> str:
    """Sucht llama-server(.exe): Konfigpfad, ./llama.cpp/, PATH."""
    name = cfg_server["exe_windows"] if os.name == "nt" else cfg_server["exe_linux"]
    if cfg_server.get("exe_pfad"):
        p = Path(cfg_server["exe_pfad"])
        return str(p / name if p.is_dir() else p)
    for kand in (ORDNER / "llama.cpp" / name, ORDNER / name):
        if kand.exists():
            return str(kand)
    return shutil.which(name) or name


def baue_befehl(cfg: dict, modell: dict, port: Optional[int] = None, embedding: bool = False) -> list[str]:
    s = cfg["server"]
    datei = Path(modell["datei"])
    if not datei.is_absolute():
        datei = ORDNER / datei
    cmd = [finde_exe(s), "-m", str(datei), "--host", s["host"], "--port", str(port or s["port"]),
           "-c", str(modell.get("ctx", 8192)), "-ngl", str(modell.get("ngl", 99))]
    if embedding:
        cmd += ["--embedding"]
        return cmd + list(s.get("zusatz_args", []))
    if modell.get("n_cpu_moe"):
        cmd += ["--n-cpu-moe", str(modell["n_cpu_moe"])]
    cmd += ["-ctk", s["kv_cache_typ"], "-ctv", s["kv_cache_typ"]]
    cmd += list(s.get("flash_attn_args", []))
    cmd += list(s.get("zusatz_args", []))
    return cmd


class ServerProzess:
    def __init__(self, befehl: list[str], host: str, port: int, log: Optional[Path] = None):
        self.befehl, self.host, self.port, self.log = befehl, host, port, log
        self.proc: Optional[subprocess.Popen] = None
        self._logf = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def starten(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        out = subprocess.DEVNULL
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self._logf = open(self.log, "ab")
            out = self._logf
        flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(self.befehl, stdout=out, stderr=subprocess.STDOUT,
                                     creationflags=flags)

    def gesund(self, timeout: float = 3.0) -> bool:
        import requests
        if self.proc and self.proc.poll() is not None:
            return False
        try:
            return requests.get(self.url + "/health", timeout=timeout).status_code == 200
        except Exception:
            return False

    def warte_bereit(self, timeout_s: float = 240, stop=lambda: False) -> bool:
        """Wartet auf /health; bricht bei Stopp-Wunsch oder Prozessende ab."""
        ende = time.time() + timeout_s
        while time.time() < ende:
            if stop() or (self.proc and self.proc.poll() is not None):
                return False
            if self.gesund():
                return True
            time.sleep(2)
        return False

    def beenden(self, frist_s: float = 20.0) -> None:
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(frist_s)
            except subprocess.TimeoutExpired:
                p.kill()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    pass
        self.proc = None
        if self._logf:
            self._logf.close()
            self._logf = None
