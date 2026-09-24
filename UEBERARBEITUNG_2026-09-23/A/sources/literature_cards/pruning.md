# Pruning

Core idea: Remove low-importance weights or structures; strong classical baseline for resource efficiency.

Transfer to QIRE: this is already implemented and tested as `PrunedMLP`
(magnitude pruning, `models_efficient.py`) -- the project's own strongest
classical baseline so far (see ROADMAP.md: 50% ops reduction, no significant
accuracy loss in the 10-seed comparison). Any QIRE efficiency claim needs to
beat this, not just a plain dense MLP -- PrunedMLP is the harder, more
relevant baseline for the "resource efficient at equal performance" thesis.

Candidate source (verify and read before citing): Han, Mao, Dally (2015/2016),
"Deep Compression: Compressing Deep Neural Networks with Pruning, Trained
Quantization and Huffman Coding", arXiv:1510.00149, ICLR 2016.

Scientific note: candidate source found via web search 2026-08-01, not yet
read by the student. Do not cite until read.
