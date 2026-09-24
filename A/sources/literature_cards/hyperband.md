# Hyperband / Successive Halving

Core idea: Allocate more budget to promising candidates and stop weak candidates early.

Transfer to QIRE: already implemented as `discovery/hyperband_scheduler.py`
and exposed via the `hyperband_discovery` preset -- used for searching QIRE's
own hyperparameter grid (gate type, state mapper, num_experts, etc.)
efficiently, not for the QIRE-vs-baseline comparison itself. Relevant to cite
as the method behind how candidate QIRE configurations were selected for the
deeper multi-seed runs, if that search path is actually used before Phase 3's
"kernaussage schärfen" -- currently the ablation grid in `search_spaces.py`
is a brute-force full grid, not Hyperband-scheduled; switching the large
ablation sweep to Hyperband could make a full run (currently ~20-40 min for
the wine/breast_cancer grid) reach the same conclusion faster.

Candidate source (verify and read before citing): Li, Jamieson, DeSalvo,
Rostamizadeh, Talwalkar (2016/2018), "Hyperband: A Novel Bandit-Based Approach
to Hyperparameter Optimization", arXiv:1603.06560, JMLR 2018.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Do not cite until read.
