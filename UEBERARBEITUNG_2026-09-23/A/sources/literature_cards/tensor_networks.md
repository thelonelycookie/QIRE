# Tensor Networks

> **Veraltet (Hinweis vom 23.09.2026):** englisch, Stand 1. August, bezieht sich auf
> alte Module. Ersetzt durch `karte_tensornetze_ml.md` (Novikov 2015, Stoudenmire 2016)
> und `karte_butterfly_strukturierte_matrizen.md`. Übersicht: `../../LITERATUR_UND_ALLEINSTELLUNG.md`.

Core idea: Low-rank structured representations; relevant to compact state mappers or experts.

Transfer to QIRE: closest match is `QuantumInspiredStateMapper` (`quantum_state_mapper.py`),
which maps classical features into a normalized "amplitude-like" state vector
via a linear/trig projection -- conceptually similar to how tensor-network
methods represent quantum states compactly, though QIRE's mapper is a single
dense projection, not an actual tensor decomposition (no tensor-train/MPS
structure). A genuine tensor-network connection would require decomposing the
state-mapper or expert weights into a low-rank tensor-train form and measuring
whether that changes the ops/accuracy trade-off -- currently untested, listed
correctly as "to be tested experimentally."

Candidate source (verify and read before citing): survey paper "Tensor
Decomposition for Model Reduction in Neural Networks: A Review", arXiv:2304.13539
(2023) -- a review, good starting point before chasing primary sources.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Do not cite until read.
