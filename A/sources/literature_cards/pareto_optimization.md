# Pareto Optimization

Core idea: Treat accuracy, operations, time and memory as competing objectives.

Transfer to QIRE: already the framing used by `verifier.py`'s `pareto_rating`
field ("strong preliminary", "ops promising, runtime/accuracy limited", etc.)
and by `discovery/pareto_gap_finder.py`. Directly relevant to the ROADMAP's
biggest finding so far: `estimated_ops` and measured wall-clock time are two
different objectives that can point in *opposite* directions (see
`ops_time_conflict` flag added to `verifier.py`) -- a genuinely Pareto-style
situation (a model can dominate on one axis and lose on another) rather than
a single-number "efficient or not" question. Framing the final report
explicitly as a multi-objective accuracy/ops/wall-time/memory trade-off,
with QIRE's actual position on that front, is more defensible than any
single-number efficiency claim.

Candidate source (verify and read before citing): "Searching Toward
Pareto-Optimal Device-Aware Neural Architectures", arXiv:1808.09830 -- uses
accuracy jointly with latency, energy, power, memory and FLOPs as competing
objectives, close to the metric set already tracked in `raw_results.csv`.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Do not cite until read.
