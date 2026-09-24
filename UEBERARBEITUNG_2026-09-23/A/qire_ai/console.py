from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .auto_innovator import ensure_policy, innovation_jobs_if_enabled
from .control_center import CommandController, deep_doctor, recommend_next, system_info
from .data_loader import available_datasets
from .research_store import ResearchStore
from .runner import run_jobs
from .sandbox_mutator import run_sandbox_code_mutation
from .search_spaces import (
    ablation_jobs,
    baseline_jobs,
    classical_baseline_jobs,
    constrained_search_jobs,
    dataset_scaling_jobs,
    day_research_jobs,
    deep_research_jobs,
    discovery_proposal_jobs,
    grid_search_jobs,
    innovation_day_jobs,
    limit_jobs,
    overhead_jobs,
    quick_jobs,
    random_search_jobs,
    research8h_seed_jobs,
    router_lab_jobs,
    seed_stability_jobs,
    successive_halving_jobs,
    hyperband_discovery_jobs,
    synthetic_benchmark_jobs,
    dedupe_jobs as dedupe_jobs_safe,
)
from .progress_tools import rank_rows

from .discovery.discovery_engine import run_discovery, load_validated_proposals
from .discovery.proposal_schema import validate_proposal, proposal_to_jobs, ResearchProposal
from .discovery.reproduction_planner import plan_reproduction, jobs_from_reproduction_plan
from .discovery.literature_cards import ensure_literature_cards
from .research_agent.ai_research_agent import run_ai_research_agent
from .discovery.benchmark_reports import write_router_lab_report, write_synthetic_benchmark_report

from .remote.oracle_profile import apply_profile_defaults, profile_summary
from .remote.server_health import print_server_health, server_health
from .remote.remote_status import print_remote_status, remote_status
from .remote.result_packager import print_pack_results, pack_results
from .remote.storage_manager import storage_report, compress_jsonl_logs
from .scheduler_ext.adaptive_epochs import adapt_epochs_for_run_type
from .scheduler_ext.claim_driven_scheduler import order_jobs_for_oracle
from .verification_ext.scientific_gatekeeper import gatekeeper_summary

BANNER = r"""
============================================================
 QIRE AI Console v1.0 Remote Research Platform
 Local experiment-control system for QIRE-AI research
============================================================
"""

PRESETS = [
    "quick", "baseline", "classical", "ablation", "overhead", "router_lab", "synthetic_benchmark",
    "dataset_scaling", "seed_stability", "grid", "random", "halving", "constrained", "deep_research", "day_research", "research8h",
    "discovery", "hyperband_discovery", "innovation", "innovation_day", "sandbox_mutation",
    "remote_smoke", "remote_baseline", "remote_router_lab", "remote_synthetic_lab", "remote_discovery", "remote_innovation", "remote_reproduction", "remote_final_verification",
]

RUN_TYPES = ["smoke", "exploration", "verification", "final_evaluation"]


def _parse_seeds(seeds: list[int] | None, default):
    return tuple(seeds) if seeds else tuple(default)


def _parse_datasets(datasets: list[str] | None, default):
    return tuple(datasets) if datasets else tuple(default)


def default_run_type_for_preset(preset: str) -> str:
    if preset == "quick":
        return "smoke"
    if preset in {"seed_stability", "claims", "verification"}:
        return "verification"
    if preset == "final_evaluation":
        return "final_evaluation"
    return "exploration"


def build_jobs_for_preset(
    preset: str,
    *,
    epochs: int | None = None,
    seeds: list[int] | None = None,
    datasets: list[str] | None = None,
    enable_innovation: bool = False,
    max_innovation_jobs: int = 48,
):
    seeds3 = _parse_seeds(seeds, (1, 2, 3))
    datasets2 = _parse_datasets(datasets, ("wine", "breast_cancer"))
    if preset == "baseline":
        return baseline_jobs(datasets=datasets2, seeds=seeds3, epochs=epochs or 80)
    if preset == "quick":
        return quick_jobs(seeds=seeds3, epochs=epochs or 25)
    if preset == "classical":
        return classical_baseline_jobs(datasets=datasets2, seeds=_parse_seeds(seeds, (1, 2, 3, 4, 5)), epochs=epochs or 90)
    if preset == "ablation":
        return ablation_jobs(datasets=datasets2, seeds=_parse_seeds(seeds, (1, 2, 3, 4, 5)), epochs=epochs or 80)
    if preset == "overhead":
        return overhead_jobs(datasets=datasets2, seeds=seeds3, epochs=epochs or 70)
    if preset == "router_lab":
        return router_lab_jobs(datasets=_parse_datasets(datasets, ("synthetic_routing_challenge", "synthetic_expert_specialization", "wine")), seeds=seeds3, epochs=epochs or 60)
    if preset == "synthetic_benchmark":
        return synthetic_benchmark_jobs(seeds=seeds3, epochs=epochs or 70)
    if preset == "dataset_scaling":
        return dataset_scaling_jobs(seeds=seeds3, epochs=epochs or 70)
    if preset == "seed_stability":
        return seed_stability_jobs(datasets=datasets2, seeds=_parse_seeds(seeds, tuple(range(1, 11))), epochs=epochs or 80)
    if preset == "grid":
        return grid_search_jobs(datasets=datasets2, seeds=seeds3, epochs=epochs or 70)
    if preset == "random":
        return random_search_jobs(datasets=datasets2, seeds=_parse_seeds(seeds, tuple(range(1, 8))), epochs=epochs or 80, max_random=420)
    if preset == "halving":
        return successive_halving_jobs(datasets=datasets2, seeds=_parse_seeds(seeds, (1, 2, 3)), epochs=epochs or 35)
    if preset == "constrained":
        return constrained_search_jobs(datasets=datasets2, seeds=seeds3, epochs=epochs or 70)
    if preset == "research8h":
        return research8h_seed_jobs(seeds=_parse_seeds(seeds, tuple(range(1, 11))), epochs=epochs or 90, max_random=650)
    if preset == "deep_research":
        return deep_research_jobs(seeds=_parse_seeds(seeds, tuple(range(1, 8))), epochs=epochs or 100)
    if preset == "day_research":
        return day_research_jobs(seeds=_parse_seeds(seeds, tuple(range(1, 13))), epochs=epochs or 110)

    if preset == "discovery":
        return discovery_proposal_jobs(seeds=_parse_seeds(seeds, (1, 2, 3)), epochs=epochs or 80, max_jobs=max_innovation_jobs)
    if preset == "hyperband_discovery":
        return hyperband_discovery_jobs(datasets=datasets2, seeds=_parse_seeds(seeds, (1, 2, 3)), epochs=epochs or 12)
    if preset == "innovation":
        return innovation_jobs_if_enabled(enable_innovation, max_new_jobs=max_innovation_jobs)
    if preset == "innovation_day":
        if not enable_innovation:
            return []
        return innovation_day_jobs(max_new_jobs=max_innovation_jobs, epochs=epochs or 90)
    if preset == "sandbox_mutation":
        return []

    # v1.0 remote/Oracle-friendly presets. These are CPU-focused, reproducible,
    # and designed for tmux/server operation rather than desktop GUI workloads.
    if preset == "remote_smoke":
        return quick_jobs(seeds=_parse_seeds(seeds, (1,)), epochs=epochs or 6)
    if preset == "remote_baseline":
        return baseline_jobs(datasets=_parse_datasets(datasets, ("wine", "breast_cancer", "digits")), seeds=_parse_seeds(seeds, (1, 2, 3, 4, 5)), epochs=epochs or 70)
    if preset == "remote_router_lab":
        return router_lab_jobs(datasets=_parse_datasets(datasets, ("synthetic_routing_challenge", "synthetic_expert_specialization", "wine", "breast_cancer")), seeds=_parse_seeds(seeds, (1, 2, 3, 4)), epochs=epochs or 55)
    if preset == "remote_synthetic_lab":
        return synthetic_benchmark_jobs(seeds=_parse_seeds(seeds, (1, 2, 3, 4)), epochs=epochs or 60)
    if preset == "remote_discovery":
        jobs = []
        jobs.extend(discovery_proposal_jobs(seeds=_parse_seeds(seeds, (1, 2, 3, 4, 5)), epochs=epochs or 70, max_jobs=max_innovation_jobs))
        jobs.extend(constrained_search_jobs(datasets=_parse_datasets(datasets, ("wine", "breast_cancer", "digits")), seeds=_parse_seeds(seeds, (1, 2, 3, 4, 5)), epochs=epochs or 70))
        return limit_jobs(jobs, max_innovation_jobs if max_innovation_jobs else None)
    if preset == "remote_innovation":
        if not enable_innovation:
            return []
        return innovation_day_jobs(max_new_jobs=max_innovation_jobs, epochs=epochs or 75)
    if preset == "remote_reproduction":
        rows = ResearchStore().load_raw_results(include_failed=True)
        jobs = jobs_from_reproduction_plan(rows, epochs=epochs or 100)
        if not jobs:
            jobs = seed_stability_jobs(datasets=_parse_datasets(datasets, ("wine", "breast_cancer", "digits")), seeds=_parse_seeds(seeds, tuple(range(1, 11))), epochs=epochs or 100)
        return jobs
    if preset == "remote_final_verification":
        jobs = []
        jobs.extend(classical_baseline_jobs(datasets=_parse_datasets(datasets, ("wine", "breast_cancer", "digits")), seeds=_parse_seeds(seeds, tuple(range(1, 11))), epochs=epochs or 120))
        jobs.extend(seed_stability_jobs(datasets=_parse_datasets(datasets, ("wine", "breast_cancer", "digits")), seeds=_parse_seeds(seeds, tuple(range(1, 11))), epochs=epochs or 120))
        return dedupe_jobs_safe(jobs)
    raise ValueError(f"Unknown preset: {preset}")


def print_best() -> None:
    store = ResearchStore()
    rows = store.load_raw_results(include_failed=False)
    print(BANNER)
    if not rows:
        print("No successful runs yet.")
        return
    print("Top rows by conservative efficiency/accuracy score:")
    for i, r in enumerate(rank_rows(rows, n=20), start=1):
        print(f"{i:02d}. {r.get('dataset')} | {r.get('model_name')} | {r.get('method')} | bal={r.get('balanced_accuracy')} | acc={r.get('accuracy')} | ops={r.get('estimated_ops')} | p95={r.get('inference_time_ms_p95')} | seed={r.get('seed')}")


def print_claims() -> None:
    store = ResearchStore()
    summary = store.export_analysis(scientific_status="manual claim check from QIRE AI Console")
    print(BANNER)
    print("Claim check exported. Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    path = Path("results/raw/verification_reports.csv")
    if path.exists():
        print(f"Verification reports: {path}")
    print("Scientific rule: smoke runs and one-seed results are not claim-capable.")


def print_status() -> None:
    store = ResearchStore()
    rows = store.load_raw_results(include_failed=True)
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    failed_rows = [r for r in rows if r.get("status") == "failed"]
    print(BANNER)
    print(f"Stored rows total: {len(rows)} | ok: {len(ok_rows)} | failed: {len(failed_rows)}")
    live = Path("results/logs/live_status.json")
    if live.exists():
        try:
            data = json.loads(live.read_text(encoding="utf-8"))
            print(f"Live status: {data.get('status')} | preset={data.get('preset')} | stage={data.get('stage')} | elapsed={data.get('elapsed_sec')}s | RAM={data.get('ram_percent')}%")
        except Exception:
            pass
    if ok_rows:
        print("\nBest efficiency/accuracy rows:")
        for r in rank_rows(ok_rows, n=10):
            print(f"  {r.get('dataset')} | {r.get('model_name')} | {r.get('method')} | acc={r.get('accuracy')} | bal={r.get('balanced_accuracy')} | ops={r.get('estimated_ops')} | seed={r.get('seed')}")
    print("\nFiles:")
    for rel in [
        "results/raw/raw_results.csv",
        "results/raw/qire_runs.sqlite",
        "results/raw/comparisons.csv",
        "results/raw/verification_reports.csv",
        "results/reports/RUN_BOOK.md",
        "results/reports/recommend_next.md",
        "results/excel/qire_ai_console_results.xlsx",
        "results/figures/accuracy_vs_ops.png",
        "results/logs/live_status.json",
        "results/logs/run_summary.json",
        "results/sandbox/sandbox_mutation_report.jsonl",
        "results/reports/discovery_report.md",
        "results/reports/reproduction_plan.md",
        "results/reports/next_experiment_plan.md",
        "results/exports/",
    ]:
        marker = "exists" if Path(rel).exists() else "not yet exported"
        print(f"  {rel} [{marker}]")


def doctor(deep: bool = False) -> None:
    print(BANNER)
    if not deep:
        import importlib
        for name in ["torch", "sklearn", "scipy", "pandas", "openpyxl", "psutil", "matplotlib"]:
            try:
                mod = importlib.import_module(name)
                print(f"[OK] {name}: {getattr(mod, '__version__', 'installed')}")
            except Exception as exc:
                print(f"[FAIL] {name}: {exc}")
        print("\nUse --doctor --deep for path, GUI and registry checks.")
        return
    checks = deep_doctor()
    for c in checks:
        print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['check']}: {c['info']}")


def export_only() -> None:
    store = ResearchStore()
    summary = store.export_analysis(scientific_status="manual export from QIRE AI Console")
    from .control_center import write_run_book
    rows = store.load_raw_results(include_failed=True)
    write_run_book(summary, rows, preset="manual_export", run_type="export", started_at_utc=summary.get("created_at_utc", ""))
    print("Export finished:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def print_datasets() -> None:
    print("Available datasets:")
    for k, v in available_datasets().items():
        print(f"  {k}: {v}")


def print_presets() -> None:
    print("Available presets:")
    for p in PRESETS:
        print(f"  {p}")


def print_recommend_next() -> None:
    store = ResearchStore()
    result = recommend_next(store.load_raw_results(include_failed=True))
    print(BANNER)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Markdown report: results/reports/recommend_next.md")


def print_failures(limit: int = 20) -> None:
    store = ResearchStore()
    failed = [r for r in store.load_raw_results(include_failed=True) if str(r.get("status")) == "failed"]
    print(BANNER)
    if not failed:
        print("No failed runs recorded.")
        return
    for r in failed[-limit:]:
        print(f"- {r.get('dataset')} | {r.get('model_name')} | {r.get('method')} | seed={r.get('seed')} | {str(r.get('error'))[:220]}")


def print_last_runs(limit: int = 15) -> None:
    store = ResearchStore()
    rows = store.load_raw_results(include_failed=True)[-limit:]
    print(BANNER)
    for r in rows:
        print(f"- {r.get('created_at_utc')} | {r.get('status')} | {r.get('dataset')} | {r.get('model_name')} | {r.get('method')} | acc={r.get('accuracy')} | ops={r.get('estimated_ops')}")


def clean_smoke_tests() -> None:
    # Conservative: create backup and remove rows whose search_mode includes quick/smoke from CSV only; SQLite retained for traceability.
    path = Path("results/raw/raw_results.csv")
    if not path.exists():
        print("No raw_results.csv found.")
        return
    backup = path.with_suffix(".before_clean_smoke.csv")
    shutil.copy2(path, backup)
    import csv
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            if "quick" in str(row.get("search_mode", "")).lower() or "smoke" in str(row.get("search_mode", "")).lower():
                continue
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader(); w.writerows(rows)
    print(f"Cleaned CSV smoke/quick rows. Backup: {backup}. SQLite remains unchanged for audit trail.")



def run_discover(limit: int = 120) -> None:
    ensure_literature_cards()
    result = run_discovery(limit=limit)
    print(BANNER)
    print(json.dumps({
        "report": result.get("report"),
        "proposal_count": len(result.get("proposals", [])),
        "blocked_proposals": len(result.get("blocked_proposals", [])),
        "hypothesis_count": len(result.get("hypotheses", [])),
        "pareto_front_count": result.get("pareto_front_count"),
    }, indent=2, ensure_ascii=False))
    print("Discovery report: results/reports/discovery_report.md")


def generate_proposals(limit: int = 120) -> None:
    result = run_discovery(limit=limit)
    print(BANNER)
    print(f"Generated and validated {len(result.get('proposals', []))} proposals. Blocked: {len(result.get('blocked_proposals', []))}")
    print("Proposal queue: results/logs/discovery_proposals.jsonl")


def validate_proposals_cli() -> None:
    proposals = load_validated_proposals()
    ok = 0; blocked = 0
    for p in proposals:
        valid, reason = validate_proposal(p)
        if valid: ok += 1
        else:
            blocked += 1
            print(f"BLOCKED {p.proposal_id}: {reason}")
    print(BANNER)
    print(f"Validated proposals: ok={ok}, blocked={blocked}")


def schedule_reproduction_cli(epochs: int = 90) -> None:
    store = ResearchStore()
    rows = store.load_raw_results(include_failed=True)
    result = plan_reproduction(rows)
    jobs = jobs_from_reproduction_plan(rows, epochs=epochs)
    store.log_plan(jobs, preset="reproduction_plan")
    print(BANNER)
    print(json.dumps({"proposal_count": result.get("proposal_count"), "planned_jobs": len(jobs), "report": result.get("report")}, indent=2, ensure_ascii=False))
    print("Reproduction jobs have been logged; run --preset discovery or specific verification command to execute proposal jobs.")


def run_ai_agent_cli(mode: str = "off", limit: int = 40) -> None:
    result = run_ai_research_agent(mode=mode, limit=limit)
    print(BANNER)
    print(json.dumps(result, indent=2, ensure_ascii=False))

def interactive_main() -> None:
    ensure_policy()
    while True:
        print(BANNER)
        print("[1] Quick Test")
        print("[2] Baseline")
        print("[3] Classical Baselines")
        print("[4] QIRE Ablation Study")
        print("[5] Router Overhead Study")
        print("[6] Router Lab")
        print("[7] Synthetic Benchmark Lab")
        print("[8] Seed Stability")
        print("[9] Day Research")
        print("[10] Innovation Day: locked")
        print("[11] Sandbox Code Mutation: locked")
        print("[12] Show best results")
        print("[13] Claim check")
        print("[14] Recommend next")
        print("[15] Export CSV/XLSX/Charts/Run Book")
        print("[16] List datasets")
        print("[17] Doctor")
        print("[18] Discovery report")
        print("[19] Generate proposals")
        print("[20] Validate proposals")
        print("[0] Exit")
        choice = input("Select: ").strip()
        if choice == "0": return
        if choice == "12": print_best(); input("\nPress Enter..."); continue
        if choice == "13": print_claims(); input("\nPress Enter..."); continue
        if choice == "14": print_recommend_next(); input("\nPress Enter..."); continue
        if choice == "15": export_only(); input("\nPress Enter..."); continue
        if choice == "16": print_datasets(); input("\nPress Enter..."); continue
        if choice == "17": doctor(deep=True); input("\nPress Enter..."); continue
        if choice == "18": run_discover(); input("\nPress Enter..."); continue
        if choice == "19": generate_proposals(); input("\nPress Enter..."); continue
        if choice == "20": validate_proposals_cli(); input("\nPress Enter..."); continue
        mapping = {"1":"quick","2":"baseline","3":"classical","4":"ablation","5":"overhead","6":"router_lab","7":"synthetic_benchmark","8":"seed_stability","9":"day_research"}
        preset = mapping.get(choice)
        if not preset:
            print("Locked or unknown option. Use CLI flags for innovation/sandbox."); input("\nPress Enter..."); continue
        jobs = build_jobs_for_preset(preset)
        run_jobs(jobs, preset_name=preset, run_type=default_run_type_for_preset(preset), time_budget_hours=8 if preset=="day_research" else None, ram_mode="throttle", export_every=20)
        input("\nPress Enter...")


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QIRE AI Console v1.0 Remote Research Platform")
    parser.add_argument("--preset", choices=PRESETS, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--run-type", choices=RUN_TYPES, default=None)
    parser.add_argument("--enable-innovation", action="store_true", help="Unlock bounded innovation mode for this run only")
    parser.add_argument("--max-innovation-jobs", type=int, default=2000)
    parser.add_argument("--enable-code-mutation", action="store_true", help="Unlock sandbox-only candidate code mutation")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--claims", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--deep", action="store_true", help="Use with --doctor for deeper environment checks")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--recommend-next", action="store_true")
    parser.add_argument("--profile", choices=["local", "oracle_free"], default="local", help="Runtime profile. oracle_free enables server-friendly defaults.")
    parser.add_argument("--server-health", action="store_true", help="Print server/Oracle readiness diagnostics.")
    parser.add_argument("--remote-status", action="store_true", help="Print live status and key remote result files.")
    parser.add_argument("--pack-results", action="store_true", help="Create a ZIP archive of raw data, reports, figures, Excel and logs.")
    parser.add_argument("--storage-report", action="store_true", help="Print storage usage and largest result files.")
    parser.add_argument("--compress-logs", action="store_true", help="Compress large JSONL logs where safe.")

    parser.add_argument("--discover", action="store_true", help="Run Discovery Engine and write discovery_report.md")
    parser.add_argument("--generate-proposals", action="store_true", help="Generate proposal queue from previous results")
    parser.add_argument("--validate-proposals", action="store_true", help="Validate discovery proposal queue against model schemas")
    parser.add_argument("--schedule-reproduction", action="store_true", help="Create reproduction plan for promising results")
    parser.add_argument("--router-lab", action="store_true", help="Run Router Lab preset")
    parser.add_argument("--synthetic-lab", action="store_true", help="Run Synthetic Benchmark Lab preset")
    parser.add_argument("--ai-agent", choices=["off", "rules", "local", "external"], default=None, help="Run optional bounded AI/research-agent proposal generator. v1 default recommendation is rules.")
    parser.add_argument("--discovery-report", action="store_true", help="Print location of the Discovery report, generating it if needed")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--last-runs", action="store_true")
    parser.add_argument("--clean-smoke-tests", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--ram-guard-percent", type=float, default=92.0)
    parser.add_argument("--ram-mode", choices=["off", "guard", "throttle"], default="guard")
    parser.add_argument("--ram-soft-limit", type=float, default=80.0)
    parser.add_argument("--ram-caution-limit", type=float, default=86.0)
    parser.add_argument("--ram-throttle-limit", type=float, default=88.0)
    parser.add_argument("--ram-hard-limit", type=float, default=None)
    parser.add_argument("--ram-resume-limit", type=float, default=84.0)
    parser.add_argument("--run-isolation", choices=["inline", "subprocess"], default="inline", help="Prototype flag. Inline is default; subprocess isolation is reserved for later hardening.")
    parser.add_argument("--inference-repeats", type=int, default=120)
    parser.add_argument("--export-every", type=int, default=20)
    parser.add_argument("--control-command", choices=["pause_after_current_job", "pause", "resume", "stop_safely", "clear"], default=None)
    args = parser.parse_args(argv)
    args = apply_profile_defaults(args)

    ensure_policy()
    if args.server_health:
        print_server_health(deep=args.deep)
        return
    if args.remote_status:
        print_remote_status()
        return
    if args.pack_results:
        print_pack_results(label="qire_results")
        return
    if args.storage_report:
        print(json.dumps(storage_report(), indent=2, ensure_ascii=False))
        return
    if args.compress_logs:
        print(json.dumps(compress_jsonl_logs(), indent=2, ensure_ascii=False))
        return
    if args.control_command:
        cc = CommandController()
        if args.control_command == "clear": cc.clear()
        else: cc.write(args.control_command, source="cli")
        print(f"Control command applied: {args.control_command}")
        return
    if args.status: print_status(); return
    if args.best: print_best(); return
    if args.claims: print_claims(); return
    if args.doctor: doctor(deep=args.deep); return
    if args.export: export_only(); return
    if args.recommend_next: print_recommend_next(); return

    if args.discover: run_discover(); return
    if args.generate_proposals: generate_proposals(); return
    if args.validate_proposals: validate_proposals_cli(); return
    if args.schedule_reproduction: schedule_reproduction_cli(epochs=args.epochs or 90); return
    if args.discovery_report: run_discover(); return
    if args.ai_agent is not None: run_ai_agent_cli(mode=args.ai_agent); return
    if args.router_lab:
        jobs = build_jobs_for_preset("router_lab", epochs=args.epochs, seeds=args.seeds, datasets=args.datasets)
        run_jobs(jobs, preset_name="router_lab", resume=not args.no_resume, max_runs=args.max_runs, time_budget_hours=args.time_budget_hours, ram_guard_percent=args.ram_guard_percent, inference_repeats=args.inference_repeats, export_every=args.export_every, ram_mode=args.ram_mode, ram_soft_limit=args.ram_soft_limit, ram_caution_limit=args.ram_caution_limit, ram_throttle_limit=args.ram_throttle_limit, ram_hard_limit=args.ram_hard_limit, ram_resume_limit=args.ram_resume_limit, run_type=args.run_type or "exploration")
        write_router_lab_report(ResearchStore().load_raw_results(include_failed=True))
        return
    if args.synthetic_lab:
        jobs = build_jobs_for_preset("synthetic_benchmark", epochs=args.epochs, seeds=args.seeds, datasets=args.datasets)
        run_jobs(jobs, preset_name="synthetic_benchmark", resume=not args.no_resume, max_runs=args.max_runs, time_budget_hours=args.time_budget_hours, ram_guard_percent=args.ram_guard_percent, inference_repeats=args.inference_repeats, export_every=args.export_every, ram_mode=args.ram_mode, ram_soft_limit=args.ram_soft_limit, ram_caution_limit=args.ram_caution_limit, ram_throttle_limit=args.ram_throttle_limit, ram_hard_limit=args.ram_hard_limit, ram_resume_limit=args.ram_resume_limit, run_type=args.run_type or "exploration")
        write_synthetic_benchmark_report(ResearchStore().load_raw_results(include_failed=True))
        return
    if args.failures: print_failures(); return
    if args.last_runs: print_last_runs(); return
    if args.clean_smoke_tests: clean_smoke_tests(); return
    if args.list_datasets: print_datasets(); return
    if args.list_presets: print_presets(); return
    if not args.preset: interactive_main(); return
    if args.run_isolation == "subprocess":
        print("Note: --run-isolation subprocess is registered as a prototype flag; v1.0 keeps main execution inline for reliability; server workflow uses tmux for process persistence.")
    if args.preset == "sandbox_mutation":
        if not args.enable_code_mutation:
            print("Sandbox mutation requested, but --enable-code-mutation was not provided. Nothing will run.")
            return
        print(json.dumps(run_sandbox_code_mutation(max_candidates=args.max_candidates), indent=2, ensure_ascii=False))
        return
    if args.preset in {"innovation", "innovation_day"} and not args.enable_innovation:
        print(f"{args.preset} requested, but --enable-innovation was not provided. Nothing will run.")
        return
    jobs = build_jobs_for_preset(
        args.preset, epochs=args.epochs, seeds=args.seeds, datasets=args.datasets,
        enable_innovation=args.enable_innovation, max_innovation_jobs=args.max_innovation_jobs,
    )
    run_type = args.run_type or default_run_type_for_preset(args.preset)
    if args.profile == "oracle_free" or args.preset.startswith("remote_"):
        jobs = adapt_epochs_for_run_type(jobs, run_type=run_type, profile="oracle_free")
        jobs = order_jobs_for_oracle(jobs)
    jobs = limit_jobs(jobs, args.max_runs)
    run_jobs(
        jobs, preset_name=args.preset, resume=not args.no_resume, max_runs=None,
        time_budget_hours=args.time_budget_hours, ram_guard_percent=args.ram_guard_percent,
        inference_repeats=args.inference_repeats, export_every=args.export_every,
        ram_mode=args.ram_mode, ram_soft_limit=args.ram_soft_limit, ram_caution_limit=args.ram_caution_limit,
        ram_throttle_limit=args.ram_throttle_limit, ram_hard_limit=args.ram_hard_limit, ram_resume_limit=args.ram_resume_limit,
        run_type=run_type,
    )


if __name__ == "__main__":
    cli_main()
