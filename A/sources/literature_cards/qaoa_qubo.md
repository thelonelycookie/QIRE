# QAOA / QUBO Inspiration

Core idea: Use cost functions over binary choices for feature selection, pruning and routing.

Transfer to QIRE: this is the weakest-connected card to the current codebase --
QIRE's Born-rule gate (`gating.py`) squares linear amplitudes to get routing
probabilities, which is inspired by the Born rule / quantum measurement
postulate, not by QAOA's variational cost-Hamiltonian approach to combinatorial
optimization. Be precise about this distinction in the write-up: QAOA is about
*solving* a discrete optimization problem with a parameterized quantum circuit;
QIRE's gate is about *classically simulating* an amplitude-squared probability
distribution for expert routing. They are both "quantum-inspired" but inspired
by different parts of quantum mechanics (variational circuits vs. the
measurement postulate). If this card stays in the final report, frame it as
"an adjacent quantum-inspired idea we considered but did not use", not as a
mechanism QIRE implements.

Candidate source (verify and read before citing): Farhi, Goldstone, Gutmann
(2014), "A Quantum Approximate Optimization Algorithm", arXiv:1411.4028.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Given the weak actual connection to this project's code,
consider whether this card belongs in the final related-work section at all.
