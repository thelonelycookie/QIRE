from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from qire_ai.config import RESULTS_DIR, RAW_DIR, REPORTS_DIR, EXPORTS_DIR


def _check_import(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"name": name, "ok": True, "version": getattr(mod, "__version__", "installed")}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def _cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def detect_cloud_hint() -> str:
    # Conservative, privacy-preserving hints only; no metadata HTTP calls.
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "ARM/aarch64 detected; compatible with Oracle Ampere A1 style instances."
    return f"machine={platform.machine()}"


def server_health(deep: bool = False) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(RESULTS_DIR))
    packages = ["numpy", "pandas", "sklearn", "matplotlib", "openpyxl", "psutil", "torch"]
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cloud_hint": detect_cloud_hint(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / (1024**3), 3),
        "ram_available_gb": round(vm.available / (1024**3), 3),
        "ram_percent": vm.percent,
        "disk_total_gb": round(disk.total / (1024**3), 3),
        "disk_free_gb": round(disk.free / (1024**3), 3),
        "cwd": str(Path.cwd()),
        "results_dir": str(RESULTS_DIR),
        "tmux_installed": _cmd_exists("tmux"),
        "git_installed": _cmd_exists("git"),
        "rsync_installed": _cmd_exists("rsync"),
        "packages": [_check_import(p) for p in packages],
        "write_checks": {},
        "warnings": [],
    }
    for d in [RESULTS_DIR, RAW_DIR, REPORTS_DIR, EXPORTS_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            report["write_checks"][str(d)] = True
        except Exception as exc:
            report["write_checks"][str(d)] = False
            report["warnings"].append(f"Cannot write to {d}: {exc}")
    if report["disk_free_gb"] < 8:
        report["warnings"].append("Low disk space for long runs; pack/export and clean old artifacts.")
    if report["ram_total_gb"] < 8:
        report["warnings"].append("RAM below 8 GB; use remote_smoke/remote_baseline before long discovery.")
    if not report["tmux_installed"]:
        report["warnings"].append("tmux not installed; long remote runs should use tmux or systemd.")
    if deep:
        report["environment"] = {k: os.environ.get(k, "") for k in ["VIRTUAL_ENV", "PATH", "PYTHONPATH"]}
        try:
            report["pip_freeze_head"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=20).splitlines()[:80]
        except Exception as exc:
            report["pip_freeze_error"] = str(exc)
    return report


def print_server_health(deep: bool = False) -> None:
    print(json.dumps(server_health(deep=deep), indent=2, ensure_ascii=False))
