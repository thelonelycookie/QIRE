from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from .config import EXCEL_DIR, FIGURES_DIR, LOGS_DIR, RAW_DIR
from .excel_export import optional_xlsx_export, write_csv
from .plotting import plot_all_figures
from .verifier import build_comparisons, build_verification_reports, build_run_quality_reports

RAW_HEADERS = [
    "run_id", "job_id", "experiment_id", "hypothesis_id", "model_name", "method", "dataset", "dataset_source", "dataset_notes", "n_train", "n_test", "seed",
    "accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "f1_weighted", "loss", "train_best_loss",
    "training_time_sec", "inference_time_ms", "inference_time_ms_median", "inference_time_ms_p95", "memory_mb", "peak_memory_mb", "model_size_kb",
    "parameter_count", "active_parameter_count", "estimated_ops", "router_ops", "expert_ops", "estimated_energy_wh", "active_model_parts", "gate_entropy", "early_exit_rate",
    "cpu_usage", "gpu_usage", "gpu_available", "gpu_name", "gpu_memory_allocated_mb", "ram_usage", "nan_detected", "confusion_matrix_json",
    "epochs", "lr", "hidden_dim", "expert_hidden_dim", "num_experts", "top_k", "prune_amount", "gate_type", "state_dim", "state_mapper",
    "lambda_resource", "mu_activation", "batch_size", "quant_bits", "temperature", "confidence_threshold", "shared_backbone", "tt_rank", "tt_cores", "max_samples",
    "search_mode", "parent_job_id", "status", "error", "notes", "created_at_utc", "config_json",
]

COMPARISON_HEADERS = [
    "comparison_id", "dataset", "baseline_model", "new_model", "new_method", "accuracy_difference", "balanced_accuracy_difference", "loss_difference",
    "time_saving_percent", "p95_time_saving_percent", "memory_saving_percent", "ops_reduction_percent", "router_ops_percent", "energy_saving_percent",
    "efficiency_per_accuracy", "pareto_rating", "result", "n_baseline", "n_new",
    "paired_n", "balanced_accuracy_wilcoxon_p", "balanced_accuracy_ttest_p", "significant_at_p05", "paired_test_note", "ops_time_conflict",
]

VERIFICATION_HEADERS = [
    "claim_id", "claim", "verification_method", "data_used", "result", "classification", "evidence_strength", "significance", "limitations", "next_required_test",
]

QUALITY_HEADERS = [
    "quality_id", "dataset", "model_name", "method", "n", "accuracy_std", "balanced_accuracy_std", "loss_mean", "nan_count", "failed_count", "quality_flag", "recommendation",
]


class ResearchStore:
    """SQLite is the durable live store. CSV/XLSX are export formats."""

    def __init__(self, root: Path | None = None):
        self.raw_csv = RAW_DIR / "raw_results.csv"
        self.comparisons_csv = RAW_DIR / "comparisons.csv"
        self.verification_csv = RAW_DIR / "verification_reports.csv"
        self.quality_csv = RAW_DIR / "run_quality_reports.csv"
        self.sqlite_path = RAW_DIR / "qire_runs.sqlite"
        self.queue_log = LOGS_DIR / "planned_jobs.jsonl"
        self.summary_json = LOGS_DIR / "run_summary.json"
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.sqlite_path)

    def _ensure_schema(self) -> None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_results (
                    job_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    dataset TEXT,
                    model_name TEXT,
                    method TEXT,
                    seed INTEGER,
                    status TEXT,
                    accuracy REAL,
                    balanced_accuracy REAL,
                    estimated_ops REAL,
                    inference_time_ms REAL,
                    created_at_utc TEXT,
                    row_json TEXT NOT NULL
                )
                """
            )
            existing_cols = {row[1] for row in con.execute("PRAGMA table_info(raw_results)").fetchall()}
            if "balanced_accuracy" not in existing_cols:
                con.execute("ALTER TABLE raw_results ADD COLUMN balanced_accuracy REAL DEFAULT 0")
            con.execute("CREATE INDEX IF NOT EXISTS idx_raw_dataset_model ON raw_results(dataset, model_name, method)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_results(status)")
            con.commit()
        self._import_legacy_csv_if_needed()

    def _import_legacy_csv_if_needed(self) -> None:
        if not self.raw_csv.exists():
            return
        with self._connect() as con:
            count = con.execute("SELECT COUNT(*) FROM raw_results").fetchone()[0]
        if count:
            return
        imported = 0
        with self.raw_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row:
                    continue
                row.setdefault("job_id", f"legacy_{row.get('run_id') or imported}")
                row.setdefault("status", "ok")
                row.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
                row.setdefault("config_json", json.dumps({"legacy_import": True, "run_id": row.get("run_id")}, sort_keys=True))
                normalized = {key: row.get(key, "") for key in RAW_HEADERS}
                self._insert_row(normalized)
                imported += 1

    def _insert_row(self, row: Dict[str, object]) -> None:
        row_json = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO raw_results
                (job_id, run_id, dataset, model_name, method, seed, status, accuracy, balanced_accuracy, estimated_ops, inference_time_ms, created_at_utc, row_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("job_id", "")), str(row.get("run_id", "")), str(row.get("dataset", "")), str(row.get("model_name", "")),
                    str(row.get("method", "")), int(float(row.get("seed") or 0)), str(row.get("status", "")), float(row.get("accuracy") or 0.0),
                    float(row.get("balanced_accuracy") or 0.0), float(row.get("estimated_ops") or 0.0), float(row.get("inference_time_ms") or 0.0),
                    str(row.get("created_at_utc", "")), row_json,
                ),
            )
            con.commit()

    def completed_job_ids(self) -> set[str]:
        with self._connect() as con:
            rows = con.execute("SELECT job_id FROM raw_results WHERE status = 'ok'").fetchall()
        return {r[0] for r in rows}

    def all_job_ids(self) -> set[str]:
        with self._connect() as con:
            rows = con.execute("SELECT job_id FROM raw_results").fetchall()
        return {r[0] for r in rows}

    def log_plan(self, jobs: Sequence[object], preset: str) -> None:
        # Very large time-budget presets can contain tens of thousands of candidate jobs.
        # Logging every planned job before the first run would delay visible work and create huge logs.
        # We therefore log all jobs for normal presets and a bounded preview for large presets.
        total = len(jobs)
        max_detail = 5000
        with self.queue_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"preset": preset, "planned_job_count": total, "log_mode": "full" if total <= max_detail else "preview", "created_at_utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True) + "\n")
            iterable = jobs if total <= max_detail else list(jobs[:max_detail])
            for job in iterable:
                f.write(json.dumps({"preset": preset, "job": job.as_dict(), "job_id": job.job_id}, sort_keys=True) + "\n")
            if total > max_detail:
                f.write(json.dumps({"preset": preset, "omitted_planned_jobs": total - max_detail, "reason": "large time-budget preset; jobs are still executable, only log detail is bounded"}, sort_keys=True) + "\n")

    def append_result(self, row: Dict[str, object]) -> None:
        row = {key: row.get(key, "") for key in RAW_HEADERS}
        self._insert_row(row)
        self._rewrite_raw_csv_from_sqlite()

    def _rewrite_raw_csv_from_sqlite(self) -> None:
        rows = self.load_raw_results(include_failed=True)
        write_csv(self.raw_csv, rows, RAW_HEADERS)

    def load_raw_results(self, include_failed: bool = False) -> List[Dict[str, object]]:
        with self._connect() as con:
            if include_failed:
                rows = con.execute("SELECT row_json FROM raw_results ORDER BY created_at_utc, job_id").fetchall()
            else:
                rows = con.execute("SELECT row_json FROM raw_results WHERE status = 'ok' ORDER BY created_at_utc, job_id").fetchall()
        out = []
        for (payload,) in rows:
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
        if not out and self.raw_csv.exists():
            with self.raw_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if include_failed or row.get("status", "ok") == "ok":
                        out.append(row)
        return out

    def export_analysis(self, scientific_status: str = "preliminary automated research run") -> Dict[str, object]:
        raw_ok = self.load_raw_results(include_failed=False)
        raw_total = self.load_raw_results(include_failed=True)
        comparisons = build_comparisons(raw_ok)
        verification_reports = build_verification_reports(raw_ok)
        quality_reports = build_run_quality_reports(raw_total)
        write_csv(self.comparisons_csv, comparisons, COMPARISON_HEADERS)
        write_csv(self.verification_csv, verification_reports, VERIFICATION_HEADERS)
        write_csv(self.quality_csv, quality_reports, QUALITY_HEADERS)
        try:
            optional_xlsx_export(
                EXCEL_DIR / "qire_ai_console_results.xlsx",
                {
                    "Raw Results": (RAW_HEADERS, raw_total),
                    "Comparisons": (COMPARISON_HEADERS, comparisons),
                    "Verification Reports": (VERIFICATION_HEADERS, verification_reports),
                    "Run Quality": (QUALITY_HEADERS, quality_reports),
                },
            )
        except Exception as exc:
            (LOGS_DIR / "xlsx_export_error.txt").write_text(str(exc), encoding="utf-8")
        plot_summary = {}
        try:
            plot_summary = plot_all_figures(self.raw_csv, FIGURES_DIR)
        except Exception as exc:
            (LOGS_DIR / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_result_count_ok": len(raw_ok),
            "raw_result_count_total": len(raw_total),
            "comparison_count": len(comparisons),
            "verification_report_count": len(verification_reports),
            "quality_report_count": len(quality_reports),
            "figure_summary": plot_summary,
            "scientific_status": scientific_status,
            "main_warning": "Do not claim efficiency until comparisons are stable over repeated seeds and fair baselines.",
        }
        self.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
