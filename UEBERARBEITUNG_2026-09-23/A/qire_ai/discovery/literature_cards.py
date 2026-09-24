from __future__ import annotations
from pathlib import Path
from qire_ai.config import PROJECT_ROOT

CARDS = {
    "mixture_of_experts.md": ("Mixture of Experts", "Sparse expert routing; QIRE should be compared first against MoE controls.", "needs_source"),
    "pruning.md": ("Pruning", "Remove low-importance weights or structures; strong classical baseline for resource efficiency.", "needs_source"),
    "low_rank.md": ("Low-Rank Compression", "Factorize dense mappings to reduce parameter and operation cost.", "needs_source"),
    "hyperband.md": ("Hyperband / Successive Halving", "Allocate more budget to promising candidates and stop weak candidates early.", "needs_source"),
    "pareto_optimization.md": ("Pareto Optimization", "Treat accuracy, operations, time and memory as competing objectives.", "needs_source"),
    "qaoa_qubo.md": ("QAOA / QUBO Inspiration", "Use cost functions over binary choices for feature selection, pruning and routing.", "needs_source"),
    "tensor_networks.md": ("Tensor Networks", "Low-rank structured representations; relevant to compact state mappers or experts.", "needs_source"),
    "energy_aware_training.md": ("Energy-Aware Training", "Optimize task loss with resource/activation penalties while enforcing accuracy constraints.", "needs_source"),
}


def ensure_literature_cards() -> Path:
    root = PROJECT_ROOT / "sources" / "literature_cards"
    root.mkdir(parents=True, exist_ok=True)
    for fname, (title, idea, source) in CARDS.items():
        path = root / fname
        if not path.exists():
            path.write_text(f"# {title}\n\nCore idea: {idea}\n\nTransfer to QIRE: to be tested experimentally.\n\nSources: {source}\n\nScientific note: do not use this card as evidence until sources are filled with real references.\n", encoding="utf-8")
    return root
