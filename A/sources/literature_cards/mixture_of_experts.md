# Mixture of Experts

Core idea: Sparse expert routing; QIRE should be compared first against MoE controls.

Transfer to QIRE: QIRE's `SparseExpertLayer` + top-k gate is structurally a
small sparsely-gated MoE layer; `StandardMoE` in this codebase is exactly the
classical-softmax-gate control condition this literature calls for. The open
question this project actually tests is narrower than "does MoE help" (long
established at scale) -- it is "does swapping the *gate* (Born-rule squared
amplitudes vs. softmax) change the accuracy/efficiency trade-off measurably".
Also relevant: the original paper reports gains at large model/cluster scale;
this project's own benchmark (`benchmark_sparse_dispatch.py`, see ROADMAP.md)
found the same pattern locally -- sparse routing only beats a dense baseline
in wall-clock time once model and batch size are large enough. Worth citing
that congruence explicitly in the discussion section.

Candidate source (verify and read before citing): Shazeer, Mirhoseini,
Maziarz, Davis, Le, Hinton, Dean (2017), "Outrageously Large Neural Networks:
The Sparsely-Gated Mixture-of-Experts Layer", arXiv:1701.06538.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Do not cite until read and the connection to QIRE above
is checked against what the paper actually claims.
