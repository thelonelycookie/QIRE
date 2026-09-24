from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
EXCEL_DIR = RESULTS_DIR / "excel"
LOGS_DIR = RESULTS_DIR / "logs"
SANDBOX_DIR = RESULTS_DIR / "sandbox"
CONTROL_DIR = RESULTS_DIR / "control"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPORTS_DIR = RESULTS_DIR / "exports"
SQLITE_DIR = RESULTS_DIR / "sqlite"

for _path in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, EXCEL_DIR, LOGS_DIR, SANDBOX_DIR, CONTROL_DIR, REPORTS_DIR, EXPORTS_DIR, SQLITE_DIR]:
    _path.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class ExperimentConfig:
    epochs: int = 80
    lr: float = 0.01
    hidden_dim: int = 32
    expert_hidden_dim: int = 16
    num_experts: int = 4
    top_k: int = 2
    prune_amount: float = 0.50
    test_size: float = 0.30
    assumed_cpu_watts: float = 15.0
    inference_repeats: int = 80
