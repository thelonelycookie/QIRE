from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

import psutil

from .config import CONTROL_DIR, FIGURES_DIR, LOGS_DIR, RAW_DIR, REPORTS_DIR, SANDBOX_DIR
from .search_spaces import ExperimentJob

LIVE_STATUS_PATH = LOGS_DIR / "live_status.json"
FINAL_STATUS_PATH = LOGS_DIR / "live_status_final.json"
CONTROL_PATH = CONTROL_DIR / "command.json"
EVENT_LOG_PATH = LOGS_DIR / "events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_event(event_type: str, **payload: Any) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"created_at_utc": utc_now(), "event_type": event_type, **payload}
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class LiveStatusWriter:
    def __init__(self, preset: str, run_type: str, time_budget_sec: Optional[float] = None, total_jobs: Optional[int] = None):
        self.preset = preset
        self.run_type = run_type
        self.time_budget_sec = time_budget_sec
        self.total_jobs = total_jobs
        self.started_at = time.perf_counter()
        self.last_payload: Dict[str, Any] = {}

    def write(self, *, status: str, current_job: Optional[ExperimentJob] = None, completed_jobs: int = 0, failed_jobs: int = 0, skipped_jobs: int = 0, stage: str = "", message: str = "", best: Optional[Dict[str, Any]] = None, current_epoch: int = 0, max_epochs: int = 0) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        payload = {
            "created_at_utc": utc_now(),
            "status": status,
            "preset": self.preset,
            "run_type": self.run_type,
            "stage": stage,
            "elapsed_sec": round(time.perf_counter() - self.started_at, 3),
            "time_budget_sec": self.time_budget_sec,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "skipped_jobs": skipped_jobs,
            "planned_jobs": self.total_jobs,
            "ram_percent": round(vm.percent, 2),
            "ram_available_gb": round(vm.available / (1024**3), 3),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "message": message,
            "current_epoch": current_epoch,
            "max_epochs": max_epochs,
            "best_baseline": "",
            "best_qire": "",
            "pareto_best": "",
            "scientific_warning": "Do not claim efficiency until comparisons are stable over repeated seeds and fair baselines.",
        }
        if current_job is not None:
            payload.update({
                "current_job_id": current_job.job_id,
                "current_dataset": current_job.dataset,
                "current_model": current_job.model_key,
                "current_seed": current_job.seed,
                "current_gate": current_job.gate_type,
                "current_top_k": current_job.top_k,
                "current_epochs": current_job.epochs,
                "current_search_mode": current_job.search_mode,
            })
        if best:
            payload.update(best)
        safe_json_write(LIVE_STATUS_PATH, payload)
        self.last_payload = payload
        return payload

    def finalize(self, summary: Dict[str, Any]) -> None:
        payload = {**self.last_payload, "status": "finished", "final_summary": summary, "created_at_utc": utc_now()}
        safe_json_write(FINAL_STATUS_PATH, payload)
        safe_json_write(LIVE_STATUS_PATH, payload)


class CommandController:
    VALID = {"pause_after_current_job", "pause", "resume", "stop_safely", "clear"}

    def __init__(self, path: Path = CONTROL_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, command: str, source: str = "gui") -> None:
        if command not in self.VALID:
            raise ValueError(f"Unsupported control command: {command}")
        safe_json_write(self.path, {"command": command, "source": source, "created_at_utc": utc_now()})
        append_event("control_command", command=command, source=source)

    def read(self) -> str:
        if not self.path.exists():
            return ""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return str(data.get("command", ""))
        except Exception:
            return ""

    def clear(self) -> None:
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                safe_json_write(self.path, {"command": "clear", "created_at_utc": utc_now()})

    def should_stop(self) -> bool:
        return self.read() == "stop_safely"

    def should_pause(self) -> bool:
        return self.read() in {"pause", "pause_after_current_job"}

    def wait_if_paused(self, status: Optional[LiveStatusWriter] = None, *, completed_jobs: int = 0, failed_jobs: int = 0, skipped_jobs: int = 0, sleep_sec: float = 4.0) -> bool:
        """Returns True if execution should stop."""
        while True:
            command = self.read()
            if command == "stop_safely":
                return True
            if command == "resume":
                self.clear()
                append_event("resume_after_pause")
                return False
            if command in {"pause", "pause_after_current_job"}:
                if status:
                    status.write(status="paused", completed_jobs=completed_jobs, failed_jobs=failed_jobs, skipped_jobs=skipped_jobs, message="Paused by user. Waiting for resume or stop.")
                time.sleep(sleep_sec)
                continue
            return False


@dataclass
class RamGovernor:
    mode: str = "guard"
    soft_limit: float = 80.0
    caution_limit: float = 86.0
    throttle_limit: float = 88.0
    hard_limit: float = 92.0
    resume_limit: float = 84.0
    wait_seconds: float = 8.0
    max_wait_cycles: int = 999999

    def zone(self) -> str:
        pct = psutil.virtual_memory().percent
        if self.mode == "off":
            return "normal"
        if pct >= self.hard_limit:
            return "hard_pause"
        if pct >= self.throttle_limit:
            return "throttle"
        if pct >= self.caution_limit:
            return "caution"
        if pct >= self.soft_limit:
            return "soft"
        return "normal"

    def wait_if_needed(self, status: Optional[LiveStatusWriter] = None, *, completed_jobs: int = 0, failed_jobs: int = 0, skipped_jobs: int = 0) -> str:
        if self.mode == "off":
            return "normal"
        pct = psutil.virtual_memory().percent
        if self.mode == "guard" and pct >= self.hard_limit:
            raise RuntimeError(f"RAM guard triggered: {pct:.1f}% >= {self.hard_limit:.1f}%")
        if self.mode != "throttle":
            return self.zone()
        cycles = 0
        while psutil.virtual_memory().percent >= self.hard_limit and cycles < self.max_wait_cycles:
            cycles += 1
            pct = psutil.virtual_memory().percent
            cleanup_memory()
            if status:
                status.write(status="ram_paused", completed_jobs=completed_jobs, failed_jobs=failed_jobs, skipped_jobs=skipped_jobs, message=f"RAM {pct:.1f}% >= hard limit {self.hard_limit:.1f}%. Waiting until below resume limit {self.resume_limit:.1f}%.")
            append_event("ram_pause", ram_percent=pct, hard_limit=self.hard_limit, resume_limit=self.resume_limit)
            time.sleep(self.wait_seconds)
            if psutil.virtual_memory().percent <= self.resume_limit:
                append_event("ram_resume", ram_percent=psutil.virtual_memory().percent)
                break
        return self.zone()

    def job_allowed(self, job: ExperimentJob) -> tuple[bool, str]:
        zone = self.zone()
        if zone in {"normal", "soft"}:
            return True, zone
        heavy_dataset = job.dataset in {"synthetic_large", "synthetic_wide", "synthetic_high_dim", "synthetic_sparse_signal", "digits"}
        heavy_model = job.model_key == "qire" and (job.num_experts >= 8 or job.expert_hidden_dim >= 32 or job.state_mapper == "trig")
        if zone == "caution":
            return not (heavy_dataset and heavy_model), zone
        if zone == "throttle":
            small_model = job.model_key in {"logreg", "mlp", "pruned_mlp", "lowrank_mlp", "standard_moe"} or (job.model_key == "qire" and job.top_k <= 1 and job.expert_hidden_dim <= 16 and job.num_experts <= 4)
            small_dataset = job.dataset in {"iris", "wine", "breast_cancer"}
            return small_model and small_dataset, zone
        if zone == "hard_pause":
            return False, zone
        return True, zone


def cleanup_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def system_info() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / (1024**3), 3),
        "cwd": str(Path.cwd()),
    }


def best_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    if not ok:
        return {"best_baseline": "none", "best_qire": "none", "pareto_best": "none"}
    def f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try: return float(row.get(key, default) or default)
        except Exception: return default
    def score(row: Dict[str, Any]) -> float:
        return f(row, "balanced_accuracy", f(row, "accuracy")) - 0.00002 * (f(row, "estimated_ops") ** 0.5) - 0.001 * f(row, "inference_time_ms_p95")
    qires = [r for r in ok if str(r.get("model_name", "")).lower().startswith("qire") or str(r.get("model_name", "")).lower().startswith("qire-ai") or str(r.get("model_name", "")).lower() == "qire-ai"]
    baselines = [r for r in ok if r not in qires]
    best_base = max(baselines, key=score) if baselines else None
    best_qire = max(qires, key=score) if qires else None
    best_any = max(ok, key=score)
    def label(r: Optional[Dict[str, Any]]) -> str:
        if not r: return "none"
        return f"{r.get('dataset')} | {r.get('model_name')} | {r.get('method')} | bal={r.get('balanced_accuracy')} ops={r.get('estimated_ops')}"
    return {"best_baseline": label(best_base), "best_qire": label(best_qire), "pareto_best": label(best_any)}


QIRE_ALLOWED_GATES = {"born", "softmax_state", "interference_born", "hard_threshold", "gumbel_softmax", "entropy_adaptive", "budget_aware", "hybrid_softmax_born"}
MOE_ALLOWED_GATES = {"softmax", "random", "fixed", "top1_softmax", "top2_softmax", "baseline"}
BASELINE_MODELS = {"logreg", "random_forest", "svm_linear", "svm_rbf", "mlp", "deep_mlp", "lowrank_mlp", "early_exit_mlp", "pruned_mlp", "quantized_mlp", "student_mlp"}


def validate_job_config(job: ExperimentJob) -> tuple[bool, str]:
    if job.model_key == "qire" and job.gate_type not in QIRE_ALLOWED_GATES:
        return False, f"Invalid QIRE gate_type={job.gate_type}. Allowed: {sorted(QIRE_ALLOWED_GATES)}"
    if job.model_key in BASELINE_MODELS and job.gate_type not in {"baseline", "none", ""}:
        return False, f"Baseline model {job.model_key} must not use gate_type={job.gate_type}"
    if job.model_key in {"standard_moe", "full_moe", "random_gate_moe"} and job.gate_type not in MOE_ALLOWED_GATES:
        return False, f"MoE model {job.model_key} has invalid gate_type={job.gate_type}"
    if job.top_k < 1 or job.top_k > max(1, job.num_experts):
        return False, f"Invalid top_k={job.top_k} for num_experts={job.num_experts}"
    if job.epochs <= 0:
        return False, "epochs must be positive"
    return True, "ok"


def classify_failure(error: str) -> str:
    e = (error or "").lower()
    if "unsupported" in e or "invalid" in e or "valueerror" in e:
        return "Config Error"
    if "out of memory" in e or "ram" in e or "memoryerror" in e:
        return "RAM Error"
    if "timeout" in e:
        return "Timeout"
    if "nan" in e:
        return "NaN Loss"
    if "dataset" in e:
        return "Dataset Error"
    if "model" in e or "shape" in e or "mat1" in e:
        return "Model Error"
    return "Unknown Error"


def recommend_next(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    failed = [r for r in rows if str(r.get("status")) == "failed"]
    recs: List[Dict[str, str]] = []
    datasets = sorted({str(r.get("dataset")) for r in ok if r.get("dataset")})
    qire_rows = [r for r in ok if "QIRE" in str(r.get("model_name", "")) or str(r.get("method", "")).startswith("qire_")]
    if len(datasets) < 3:
        recs.append({"priority": "high", "action": "Run day_research or baseline on wine, breast_cancer and digits", "reason": "Current evidence uses too few datasets for strong claims."})
    if qire_rows:
        avg_router_share = mean([float(r.get("router_ops") or 0) / max(1.0, float(r.get("estimated_ops") or 1)) for r in qire_rows])
        if avg_router_share > 0.35:
            recs.append({"priority": "high", "action": "Run router_lab and QIRE top-1 low-rank variants", "reason": f"Average QIRE router share is high ({avg_router_share:.2f}); overhead likely dominates."})
    else:
        recs.append({"priority": "high", "action": "Run ablation with QIRE small/top-1 variants", "reason": "No successful QIRE rows are available yet."})
    model_counts: Dict[str, int] = {}
    for r in ok:
        key = f"{r.get('dataset')}::{r.get('model_name')}::{r.get('method')}"
        model_counts[key] = model_counts.get(key, 0) + 1
    unstable_or_low_seed = [k for k, n in model_counts.items() if n < 5]
    if unstable_or_low_seed:
        recs.append({"priority": "medium", "action": "Run seed_stability or verification on the best candidates with at least 5 seeds", "reason": "Many comparisons still have fewer than 5 seeds."})
    if failed:
        kinds = sorted({classify_failure(str(r.get("error", ""))) for r in failed})
        recs.append({"priority": "medium", "action": "Inspect and block repeated invalid configurations", "reason": "Failures detected: " + ", ".join(kinds)})
    if not recs:
        recs.append({"priority": "medium", "action": "Start innovation_day with conservative innovation enabled", "reason": "Existing data are broad enough for bounded follow-up proposals."})
    result = {"created_at_utc": utc_now(), "recommendations": recs, "raw_ok_count": len(ok), "failed_count": len(failed)}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_json_write(LOGS_DIR / "recommend_next.json", result)
    md = ["# QIRE AI Console — Recommended Next Actions", ""]
    for i, rec in enumerate(recs, 1):
        md.append(f"## {i}. {rec['action']}")
        md.append(f"- Priority: {rec['priority']}")
        md.append(f"- Reason: {rec['reason']}")
        md.append("")
    (REPORTS_DIR / "recommend_next.md").write_text("\n".join(md), encoding="utf-8")
    return result


def write_run_book(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], *, preset: str, run_type: str, started_at_utc: str, ended_at_utc: Optional[str] = None) -> Path:
    ended_at_utc = ended_at_utc or utc_now()
    best = best_summary(rows)
    failures = [r for r in rows if str(r.get("status")) == "failed"]
    rec = recommend_next(rows)
    lines = [
        "# QIRE AI Console Run Book", "",
        f"- Preset: {preset}",
        f"- Run type: {run_type}",
        f"- Started: {started_at_utc}",
        f"- Ended: {ended_at_utc}",
        f"- Raw ok rows: {summary.get('raw_result_count_ok')}",
        f"- Raw total rows: {summary.get('raw_result_count_total')}",
        f"- Failed rows in database: {len(failures)}", "",
        "## Best Current Results", "",
        f"- Best baseline: {best.get('best_baseline')}",
        f"- Best QIRE: {best.get('best_qire')}",
        f"- Pareto-best heuristic: {best.get('pareto_best')}", "",
        "## Scientific Warning", "",
        "Do not claim general QIRE efficiency until results are stable over repeated seeds, multiple datasets and fair baselines.", "",
        "## Recommended Next Actions", "",
    ]
    for item in rec.get("recommendations", []):
        lines.append(f"- [{item.get('priority')}] {item.get('action')} — {item.get('reason')}")
    path = REPORTS_DIR / "RUN_BOOK.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def deep_doctor() -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    def add(name: str, ok: bool, info: str):
        checks.append({"check": name, "ok": bool(ok), "info": info})
    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    for module in ["numpy", "pandas", "sklearn", "scipy", "torch", "matplotlib", "openpyxl", "psutil"]:
        try:
            m = __import__(module)
            add(module, True, getattr(m, "__version__", "installed"))
        except Exception as exc:
            add(module, False, str(exc))
    try:
        import customtkinter as ctk  # noqa: F401
        add("customtkinter", True, "installed")
    except Exception as exc:
        add("customtkinter", False, f"optional GUI dependency missing: {exc}")
    for d in [RAW_DIR, LOGS_DIR, CONTROL_DIR, REPORTS_DIR, FIGURES_DIR, SANDBOX_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            add(f"write:{d}", True, "writable")
        except Exception as exc:
            add(f"write:{d}", False, str(exc))
    add("system", True, json.dumps(system_info(), ensure_ascii=False))
    safe_json_write(LOGS_DIR / "doctor_deep.json", {"created_at_utc": utc_now(), "checks": checks})
    return checks
