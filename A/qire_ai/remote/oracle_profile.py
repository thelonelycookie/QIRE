from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    ram_mode: str = "throttle"
    ram_soft_limit: float = 76.0
    ram_caution_limit: float = 82.0
    ram_throttle_limit: float = 88.0
    ram_guard_percent: float = 92.0
    ram_resume_limit: float = 84.0
    inference_repeats: int = 60
    export_every: int = 50
    default_time_budget_hours: float = 12.0
    assumed_cpu_watts: float = 20.0
    max_heavy_jobs_per_stage: int = 80


LOCAL_PROFILE = RuntimeProfile(
    name="local",
    description="Default local workstation profile. Uses existing CLI defaults unless explicitly overridden.",
)

ORACLE_FREE_PROFILE = RuntimeProfile(
    name="oracle_free",
    description=(
        "CPU-only Oracle Cloud Always Free profile. Designed for Ampere A1/Ubuntu: "
        "RAM throttling, fewer inference repeats, periodic exports, and no GUI dependency."
    ),
)


def get_profile(name: str | None) -> RuntimeProfile:
    if not name or name == "local":
        return LOCAL_PROFILE
    if name == "oracle_free":
        return ORACLE_FREE_PROFILE
    raise ValueError(f"Unknown runtime profile: {name}")


def apply_profile_defaults(args: Any) -> Any:
    """Apply profile defaults without overriding explicit user choices too aggressively.

    The argparse defaults in v0.6 are already safe, but on Oracle we want a
    reproducible baseline: throttle RAM, conservative inference repeats, and a
    reasonable export cadence. This function mutates and returns args.
    """
    profile = get_profile(getattr(args, "profile", None))
    if profile.name != "oracle_free":
        return args
    # Use remote-safe settings if the user did not choose something stricter.
    if getattr(args, "ram_mode", None) == "guard":
        args.ram_mode = profile.ram_mode
    args.ram_soft_limit = min(float(getattr(args, "ram_soft_limit", profile.ram_soft_limit)), profile.ram_soft_limit)
    args.ram_caution_limit = min(float(getattr(args, "ram_caution_limit", profile.ram_caution_limit)), profile.ram_caution_limit)
    args.ram_throttle_limit = min(float(getattr(args, "ram_throttle_limit", profile.ram_throttle_limit)), profile.ram_throttle_limit)
    args.ram_guard_percent = min(float(getattr(args, "ram_guard_percent", profile.ram_guard_percent)), profile.ram_guard_percent)
    args.ram_resume_limit = min(float(getattr(args, "ram_resume_limit", profile.ram_resume_limit)), profile.ram_resume_limit)
    if int(getattr(args, "inference_repeats", profile.inference_repeats)) > profile.inference_repeats:
        args.inference_repeats = profile.inference_repeats
    if int(getattr(args, "export_every", profile.export_every)) < 1:
        args.export_every = profile.export_every
    elif getattr(args, "export_every", None) == 20:
        args.export_every = profile.export_every
    return args


def profile_summary(name: str | None) -> dict[str, Any]:
    p = get_profile(name)
    return p.__dict__.copy()
