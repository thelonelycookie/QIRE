# Low-Rank Compression

> **Veraltet (Hinweis vom 23.09.2026):** englisch, alter Modulbezug, Quellen nicht
> geprüft. Ergänzt/ersetzt durch `karte_rang_repraesentationen.md` (Neural Collapse,
> Low-Rank Simplicity Bias, intrinsische Dimension) und `karte_peft_adapter.md` (LoRA).

Core idea: Factorize dense mappings to reduce parameter and operation cost.

Transfer to QIRE: already implemented and tested as `LowRankMLP`
(`models_baseline.py`, factorizes the first layer through a smaller `rank_dim`
bottleneck) and as the `state_mapper="low_rank"` option in QIRE's own state
mapper (`quantum_state_mapper.py` picks `state_dim = input_dim // 2` for that
mode). Not yet in the seed_stability comparisons run so far (those used
`state_mapper="linear"`) -- worth adding `low_rank` as an explicit ablation arm
to see whether it changes QIRE's ops/accuracy trade-off versus `linear`.

Candidate sources (verify and read before citing): Denil, Shakibi, Dinh,
Ranzato, de Freitas (2013), "Predicting Parameters in Deep Learning", NeurIPS
2013; Denton, Zaremba, Bruna, LeCun, Fergus (2014), "Exploiting Linear
Structure Within Convolutional Networks for Efficient Evaluation", NeurIPS
2014. (No arXiv ID confirmed for either during this search -- look these up by
title/venue rather than trusting a guessed arXiv number.)

Scientific note: candidate sources found via web search 2026-08-01, not yet
read by the student. Do not cite until read.
