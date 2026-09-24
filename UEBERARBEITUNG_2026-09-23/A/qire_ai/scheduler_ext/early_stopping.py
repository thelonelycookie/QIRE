from __future__ import annotations


def should_flag_early_stop(row: dict) -> tuple[bool, str]:
    """Post-run quality flag. Real training-level early stopping is implemented
    conservatively in future model loops; this protects reports now."""
    try:
        bal = float(row.get("balanced_accuracy", row.get("accuracy", 0)) or 0)
        ops = float(row.get("estimated_ops", 0) or 0)
    except Exception:
        return True, "invalid_metric"
    if str(row.get("nan_detected", "False")) == "True":
        return True, "nan_detected"
    if bal < 0.20:
        return True, "near_random_or_failed_learning"
    if ops > 1_000_000:
        return True, "excessive_ops"
    return False, "ok"
