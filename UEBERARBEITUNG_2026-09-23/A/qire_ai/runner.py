from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import psutil

from .benchmark import run_single_experiment, set_seed
from .control_center import (
    CommandController,
    LiveStatusWriter,
    RamGovernor,
    append_event,
    best_summary,
    cleanup_memory,
    classify_failure,
    recommend_next,
    utc_now,
    validate_job_config,
    write_run_book,
)
from .data_loader import load_dataset
from .model_factory import build_model_from_job
from .research_store import ResearchStore
from .search_spaces import ExperimentJob
from .progress_tools import print_live_header, rank_rows, write_hourly_summary


def _job_metadata(job: ExperimentJob, state_dim: int | None = None) -> dict:
    return {
        "job_id": job.job_id,
        "epochs": job.epochs,
        "lr": job.lr,
        "hidden_dim": job.hidden_dim,
        "expert_hidden_dim": job.expert_hidden_dim,
        "num_experts": job.num_experts,
        "top_k": min(job.top_k, job.num_experts),
        "prune_amount": job.prune_amount,
        "gate_type": job.gate_type,
        "state_dim": state_dim if state_dim is not None else (job.state_dim or "auto"),
        "state_mapper": job.state_mapper,
        "lambda_resource": job.lambda_resource,
        "mu_activation": job.mu_activation,
        "batch_size": job.batch_size,
        "quant_bits": job.quant_bits,
        "temperature": job.temperature,
        "confidence_threshold": job.confidence_threshold,
        "shared_backbone": str(job.shared_backbone),
        "tt_rank": job.tt_rank,
        "tt_cores": job.tt_cores,
        "max_samples": job.max_samples or "",
        "search_mode": job.search_mode,
        "parent_job_id": job.parent_job_id,
        "config_json": json.dumps(job.as_dict(), ensure_ascii=False, sort_keys=True),
    }


def _failed_row(job: ExperimentJob, error: str, created_at: str) -> dict:
    return {
        "run_id": f"failed_{job.job_id}",
        "experiment_id": job.experiment_id,
        "hypothesis_id": job.hypothesis_id,
        "model_name": job.model_key,
        "method": job.gate_type if job.model_key == "qire" else job.model_key,
        "dataset": job.dataset,
        "seed": job.seed,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "f1_macro": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_weighted": 0.0,
        "loss": 0.0,
        "train_best_loss": 0.0,
        "training_time_sec": 0.0,
        "inference_time_ms": 0.0,
        "inference_time_ms_median": 0.0,
        "inference_time_ms_p95": 0.0,
        "memory_mb": 0.0,
        "peak_memory_mb": 0.0,
        "model_size_kb": 0.0,
        "parameter_count": 0,
        "active_parameter_count": 0,
        "estimated_ops": 0,
        "router_ops": 0,
        "expert_ops": 0,
        "estimated_energy_wh": 0.0,
        "active_model_parts": 0.0,
        "gate_entropy": 0.0,
        "early_exit_rate": 0.0,
        "cpu_usage": 0.0,
        "gpu_usage": 0.0,
        "gpu_available": "False",
        "gpu_name": "none",
        "gpu_memory_allocated_mb": 0.0,
        "ram_usage": psutil.virtual_memory().percent,
        "nan_detected": "False",
        "confusion_matrix_json": "",
        "notes": (job.notes + " | failure_category=" + classify_failure(error)).strip(),
        **_job_metadata(job),
        "status": "failed",
        "error": error,
        "created_at_utc": created_at,
    }


def execute_job(job: ExperimentJob, store: ResearchStore, assumed_cpu_watts: float = 15.0, inference_repeats: int = 80, ram_guard_percent: float = 92.0) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        ok, reason = validate_job_config(job)
        if not ok:
            raise ValueError(reason)
        current_ram = psutil.virtual_memory().percent
        if current_ram >= ram_guard_percent:
            raise RuntimeError(f"RAM guard triggered before run: {current_ram:.1f}% >= {ram_guard_percent:.1f}%")
        set_seed(job.seed)
        bundle = load_dataset(job.dataset, seed=job.seed, test_size=0.30, max_samples=job.max_samples)
        model, prune_amount = build_model_from_job(job, bundle.input_dim, bundle.num_classes)
        state_dim = getattr(model, "state_dim", None)
        run_id = f"{bundle.name}_{job.model_key}_{getattr(model, 'method', job.model_key)}_s{job.seed}_{job.job_id[-6:]}"
        row = run_single_experiment(
            run_id=run_id,
            experiment_id=job.experiment_id,
            hypothesis_id=job.hypothesis_id,
            dataset_name=bundle.name,
            model=model,
            seed=job.seed,
            x_train=bundle.x_train,
            y_train=bundle.y_train,
            x_test=bundle.x_test,
            y_test=bundle.y_test,
            epochs=job.epochs,
            lr=job.lr,
            prune_amount=prune_amount,
            assumed_cpu_watts=assumed_cpu_watts,
            inference_repeats=inference_repeats,
            batch_size=job.batch_size,
            lambda_resource=job.lambda_resource,
            mu_activation=job.mu_activation,
        )
        row.update({"dataset_source": bundle.source, "dataset_notes": bundle.notes, "n_train": bundle.n_train, "n_test": bundle.n_test})
        row.update(_job_metadata(job, state_dim=state_dim))
        row.update({"status": "ok", "error": "", "created_at_utc": created_at})
        if not row.get("notes"):
            row["notes"] = job.notes
        if str(row.get("nan_detected", "False")) == "True":
            row["status"] = "failed"
            row["error"] = "NaN or non-finite loss detected during training."
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        row = _failed_row(job, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}", created_at)
    store.append_result(row)
    cleanup_memory()
    return row


def _stage_for(preset: str, executed: int, planned: int) -> str:
    if preset != "day_research" or planned <= 0:
        return preset
    frac = executed / max(1, planned)
    if frac < 0.03: return "calibration"
    if frac < 0.18: return "baseline_establishment"
    if frac < 0.45: return "qire_exploration"
    if frac < 0.70: return "ablation_and_overhead"
    if frac < 0.90: return "reproduction"
    return "claim_evaluation_export"


def run_jobs(
    jobs: Sequence[ExperimentJob],
    *,
    preset_name: str,
    resume: bool = True,
    max_runs: int | None = None,
    time_budget_hours: float | None = None,
    assumed_cpu_watts: float = 15.0,
    inference_repeats: int = 80,
    export_every: int = 10,
    ram_guard_percent: float = 92.0,
    ram_mode: str = "guard",
    ram_soft_limit: float = 80.0,
    ram_caution_limit: float = 86.0,
    ram_throttle_limit: float = 88.0,
    ram_hard_limit: float | None = None,
    ram_resume_limit: float = 84.0,
    run_type: str = "exploration",
    live_status: bool = True,
) -> dict:
    store = ResearchStore()
    jobs = list(jobs)
    if max_runs is not None and max_runs > 0:
        jobs = jobs[:max_runs]
    store.log_plan(jobs, preset=preset_name)
    completed = store.completed_job_ids() if resume else set()
    planned_count = len(jobs)
    started_at_utc = utc_now()
    start = time.perf_counter()
    deadline = start + float(time_budget_hours) * 3600.0 if time_budget_hours else None
    ram_hard = ram_hard_limit if ram_hard_limit is not None else ram_guard_percent
    governor = RamGovernor(mode=ram_mode, soft_limit=ram_soft_limit, caution_limit=ram_caution_limit, throttle_limit=ram_throttle_limit, hard_limit=ram_hard, resume_limit=ram_resume_limit)
    commands = CommandController()
    # v0.6.1 fix: a previous GUI/CLI stop or pause command could remain in
    # results/control/command.json and immediately stop the next run before the
    # first job. Treat commands present at run start as stale and clear them.
    stale_command = commands.read()
    if stale_command in {"stop_safely", "pause", "pause_after_current_job", "resume", "clear"}:
        commands.clear()
        append_event("stale_control_command_cleared_at_run_start", command=stale_command, preset=preset_name)
    status_writer = LiveStatusWriter(preset_name, run_type, time_budget_sec=(float(time_budget_hours)*3600.0 if time_budget_hours else None), total_jobs=planned_count) if live_status else None
    executed = skipped = failed = 0
    planned_label = planned_count if planned_count else "open"
    print(f"QIRE AI Console v1.0 Remote Research Platform | preset={preset_name} | planned={planned_label} | resume={resume} | time_budget_hours={time_budget_hours} | ram_mode={ram_mode}")
    append_event("run_start", preset=preset_name, run_type=run_type, planned_jobs=planned_count)
    if status_writer:
        status_writer.write(status="running", completed_jobs=0, failed_jobs=0, skipped_jobs=0, stage="initializing", message="Run started")
    next_live = start
    next_hourly = start + 3600.0
    hourly_index = 1
    try:
        for i, job in enumerate(jobs, start=1):
            if deadline is not None and time.perf_counter() >= deadline:
                print("Time budget reached; stopping cleanly and exporting current results.")
                append_event("time_budget_reached", executed=executed)
                break
            if commands.should_stop():
                print("Stop-safely command received; exporting current results.")
                append_event("stop_safely_received", executed=executed)
                commands.clear()
                break
            if commands.should_pause() and commands.wait_if_paused(status_writer, completed_jobs=executed, failed_jobs=failed, skipped_jobs=skipped):
                print("Stop command received while paused; exporting current results.")
                commands.clear()
                break
            if resume and job.job_id in completed:
                skipped += 1
                continue
            try:
                zone = governor.wait_if_needed(status_writer, completed_jobs=executed, failed_jobs=failed, skipped_jobs=skipped)
            except RuntimeError as exc:
                print(str(exc))
                append_event("ram_guard_stop", error=str(exc))
                break
            allowed, reason = governor.job_allowed(job)
            if not allowed and ram_mode == "throttle":
                skipped += 1
                append_event("job_deferred_due_to_ram", job_id=job.job_id, zone=reason, dataset=job.dataset, model=job.model_key)
                if status_writer:
                    status_writer.write(status="throttling", current_job=job, completed_jobs=executed, failed_jobs=failed, skipped_jobs=skipped, stage=_stage_for(preset_name, executed, planned_count), message=f"Deferring RAM-heavy job in zone={reason}")
                continue
            ok, reason = validate_job_config(job)
            if not ok:
                row = _failed_row(job, f"ValueError: {reason}", utc_now())
                store.append_result(row)
                failed += 1
                append_event("invalid_job_blocked", job_id=job.job_id, reason=reason)
                continue
            if time.perf_counter() >= next_live:
                current = f"current job: {job.dataset} | {job.model_key} | seed={job.seed} | mode={job.search_mode} | gate={job.gate_type} | top_k={job.top_k} | epochs={job.epochs} | ram_zone={zone}"
                print_live_header(preset=preset_name, executed=executed, planned=planned_count, failed=failed, start=start, deadline=deadline, current=current)
                rows = store.load_raw_results(include_failed=False)
                best = best_summary(rows)
                if status_writer:
                    status_writer.write(status="running", current_job=job, completed_jobs=executed, failed_jobs=failed, skipped_jobs=skipped, stage=_stage_for(preset_name, executed, planned_count), best=best, message=current)
                next_live = time.perf_counter() + 10.0
            print(f"[{i}/{planned_count}] dataset={job.dataset} | model={job.model_key} | seed={job.seed} | mode={job.search_mode} | gate={job.gate_type} | top_k={job.top_k} | epochs={job.epochs} | hidden={job.hidden_dim} | experts={job.num_experts}")
            row = execute_job(job, store, assumed_cpu_watts=assumed_cpu_watts, inference_repeats=inference_repeats, ram_guard_percent=ram_hard)
            executed += 1
            if row.get("status") != "ok":
                failed += 1
                print(f"  FAILED: {str(row.get('error', ''))[:220]}")
                append_event("run_failed", job_id=job.job_id, failure_category=classify_failure(str(row.get("error", ""))))
            else:
                print(f"  ok acc={row['accuracy']} bal={row.get('balanced_accuracy')} ops={row['estimated_ops']} router_ops={row.get('router_ops')} infer_ms={row['inference_time_ms']}")
                append_event("run_ok", job_id=job.job_id, dataset=job.dataset, model=job.model_key, accuracy=row.get("accuracy"), ops=row.get("estimated_ops"))
            if status_writer:
                best = best_summary(store.load_raw_results(include_failed=False))
                status_writer.write(status="running", current_job=job, completed_jobs=executed, failed_jobs=failed, skipped_jobs=skipped, stage=_stage_for(preset_name, executed, planned_count), best=best, message="Finished one job")
            if export_every and executed % export_every == 0:
                partial = store.export_analysis(scientific_status=f"partial export after {executed} jobs in preset {preset_name}")
                rows = store.load_raw_results(include_failed=False)
                recommend_next(rows)
                best = rank_rows(rows, n=3)
                if best:
                    print("  current best efficiency/accuracy rows:")
                    for r in best:
                        print(f"    {r.get('dataset')} | {r.get('model_name')} | {r.get('method')} | bal={r.get('balanced_accuracy')} | ops={r.get('estimated_ops')} | p95={r.get('inference_time_ms_p95')}")
            if deadline is not None and time.perf_counter() >= next_hourly:
                hourly = store.export_analysis(scientific_status=f"hourly export after {executed} jobs in preset {preset_name}")
                hourly.update({"executed_jobs_this_run": executed, "failed_jobs_this_run": failed, "elapsed_sec": round(time.perf_counter() - start, 3)})
                write_hourly_summary(hourly, hourly_index)
                hourly_index += 1
                next_hourly += 3600.0
    except KeyboardInterrupt:
        print("Interrupted by user; exporting current results before exit.")
        append_event("keyboard_interrupt", executed=executed)
    summary = store.export_analysis(scientific_status=f"finished, interrupted, or stopped preset {preset_name}")
    summary.update({
        "preset": preset_name,
        "run_type": run_type,
        "planned_jobs": planned_count,
        "executed_jobs_this_run": executed,
        "skipped_existing_jobs": skipped,
        "failed_jobs_this_run": failed,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "ram_mode": ram_mode,
    })
    rows_total = store.load_raw_results(include_failed=True)
    write_run_book(summary, rows_total, preset=preset_name, run_type=run_type, started_at_utc=started_at_utc)
    store.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if status_writer:
        status_writer.finalize(summary)
    # Leave no stale stop/pause command behind for the next run.
    if commands.read() in {"stop_safely", "pause", "pause_after_current_job"}:
        commands.clear()
        append_event("control_command_cleared_at_run_finish", preset=preset_name)
    append_event("run_finished", preset=preset_name, executed=executed, failed=failed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
