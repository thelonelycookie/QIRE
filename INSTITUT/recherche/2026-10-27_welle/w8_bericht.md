# w8: Literatur- und Ideenscan "quanten-inspirierte effiziente KI" 2023-2026 (27.10.2026)

Erkundung, nichts gerechnet, nichts in agenda/Laborbuch eingetragen. 13 WebSearch (nur Abstracts/Snippets, keine Volltexte). Zahlen: [WS] = aus Suchergebnis (Abstract/Zusammenfassung), [N] = aus LITERATUR/notizen.md, [G] = Gedächtnis, ungeprüft.

## 1. Kurzfassung
- Ampel gesamt: **gelb**. Fast alle Bausteine sind bekannt (siehe Abschnitt 2); Neuheit ist laut Malte (07.10.) nicht nötig. Es zählt, was sich bei gleichen FLOPs UND Parametern gegen dicht/Niedrigrang hält.
- Qiu 2025 (2410.02117) warnt: bestehende strukturierte Matrizen schlagen dicht im rechenoptimalen Fall nicht deutlich [N]. Jede FFN-Idee (Rang 1, 6) braucht darum genug Training und fairen Vergleich.
- Realistischste "grüne" Kandidaten: (R1) Monarch/BTT-FFN bei **gleichen FLOPs, breiter** statt gleich breit; (R2) Hadamard-Rotation vor Ternär-/Low-Bit-Quantisierung im Training (BitNet v2, QuaRot); (R3) gelernte Cayley-/Givens-Rotation statt fester Hadamard (SpinQuant, ButterflyQuant). R2 und R3 sparen Bytes/Bits, nicht FLOPs: als Pareto "bpc gegen Bytes" fahren, nicht als FLOP-Vorteil.
- Einzige Idee mit echtem FLOP-Hebel: Monarch-Mixer-Tokenmischung (R4), erst bei langem Kontext (>= 1-4k) sichtbar, auf 4 Kernen schwer.
- Zitierpflicht: CompactifAI, QuaRot, SpinQuant, QuIP#, BOFT, Monarch, Monarch Mixer, Compute Better Spent, TT-LoRA/LoRETTA, FlatQuant, BitNet v2, TPA. In notizen.md fehlen noch: QuIP#, FlatQuant, BitNet v2, TPA, TT-LoRA/LoRETTA/MetaTT, Monarch-Mixer-Zahlen (nur "dao_monarch_mixer2023" vorhanden), AMPS/Born-Machine-Sequenzen, Muon.

## 2. Schon im Institut erfasst (nicht doppelt zählen) [N]
CompactifAI (2401.14109), QuaRot (2404.00456), SpinQuant (2405.16406), ButterflyQuant (2509.09679), BOFT (2311.06243), OFT, Monarch (2204.00595), Monarch Mixer, Compute Better Spent (2406.06248), Qiu 2025 (2410.02117), Kron-LoRA, KronA, QuanTA, Quantum-PEFT, Cayley Unitary Adapters (2605.05914), TensorGPT, TensorSLM, LM Tensor Trains, Needle-2 Hadamard-MLP, Fastfood, Deep Fried, ACDC, Primer/ReLU², Pixelated Butterfly, Gonon (Kronecker-sparse, Median 1,4x schneller), HadaNets, Born-Text (Born's Rule), Quantum-Inspired Self-Attention (2603.03318, nur Titel). Neuheitsprüfungen N-STRUKTUR, N-VOLLTEXT1/2, N-QUADRAT-ADAPTER, N-BREIT07/11 decken Struktur+Quadrat, Adapter und Hadamard-Quantisierung ab.

## 3. Rangliste der 10 Ideen
Kriterien: (a) 4 Kerne + Raven, (b) bei gleichen FLOPs gegen dicht bestehbar, (c) Jugend-forscht-tauglich (erklärbar, Quanten-Bezug ehrlich).

### R1. Monarch/BTT-FFN bei gleichem FLOP-Budget, dafür breiter (Qiu-Protokoll im Byte-LM)
- Idee: FFN-Matrizen als Block-Tensor-Train/Monarch; bei festen FLOPs wird die Breite größer. Quanten-Bezug: Tensor-Train = Matrix-Produkt-Operator (MPO), Schaltkreis-Sprache (Butterfly = Gatter-Lagen).
- Zahlen: BTT erreicht dichte ViT-S/32-Qualität auf ImageNet mit 3,8x weniger Rechnung; Breite wächst bei Blockdiagonal ~C^(2/3) statt dicht ~C^(1/2) [WS Qiu 2024]. Monarch 1,7-2,0x schnelleres Training (ViT/GPT-2) [N]. Gegenbefund: Qiu 2025, kein klarer Vorteil im rechenoptimalen Fall [N]; Gonon 2025: Kronecker-sparse nur Median 1,4x schneller, nicht immer [N].
- Warum Rang 1: unser I-045-v4 sagt "bei gleichen FLOPs gewinnt dicht"; das ist genau die Frage, ob Struktur+Breite das kippt. Sauberster Test, kleinster Aufwand (struktur_sprache.py).
- Risiko: Vorsprung nur bei knappem Training (wie I-044-v3); CPU-Laufzeit von Blockmatmul langsamer als FLOPs vermuten lassen (FLOPs zählen, Wandzeit separat berichten).
- Ampel: gelb-grün.

### R2. Hadamard vor Low-Bit im Training (H-BitLinear-Prinzip, QuaRot-Logik) im kleinen Ternär-Sprachmodell
- Idee: Online-Walsh-Hadamard vor Aktivations-/Gewichtsquantisierung glättet Ausreißer; 4-Bit-Aktivationen im 1,58-Bit-Netz. Quanten-Bezug: Hadamard = Clifford-Gatter-Lage; Anschluss an unser ternaer_butterfly.py.
- Zahlen: BitNet v2 mit 8-Bit-Aktivationen = BitNet b1.58; 4-Bit "minimaler" Verlust [WS, Arbeitsstand, kein Venue]. QuaRot: 4-Bit LLaMA2-70B, 0,47 WikiText-ppl Verlust, 99 % Zero-Shot (v1 nannte 0,29) [WS].
- Vorteil: Bytes und Bit-Operationen, nicht FLOPs. Gleiche-Bytes-Vergleich gegen dichte Low-Bit-Baseline (z. B. 2-Bit dicht). Passt zu Maltes Ziel "ressourceneffizient".
- Ampel: grün (billig, klarer Kontrollarm: ohne Hadamard, mit zufälliger Orthogonalen, mit Hadamard).

### R3. Gelernte unitäre Rotation (Cayley/Givens-Butterfly) statt fester Hadamard als Quantisierungs-Vorverarbeitung
- Idee: Rotation parametrisiert als Cayley-Transformation oder Givens-Butterfly (n/2 log2 n Winkel); Post-Training auf kleinem Byte-LM, 2-4 Bit. Quanten-Bezug: unitäre Gatter, Winkel = Rotationsgatter; kompatibel mit unserer Magie/Clifford-Linie (Hadamard = Spezialfall, Winkel auf k*pi/2 runden).
- Zahlen: SpinQuant LLaMA-2 7B W4A4KV4: 2,9 Punkte Zero-Shot-Lücke zu voller Genauigkeit; zufällige Rotationen schwanken bis 13 Punkte; bis 45,1 % kleinere Lücke als QuaRot (LLaMA-3 8B, ICLR-Fassung) [WS]. ButterflyQuant: lernbare Givens-Butterfly ersetzt feste Hadamard bei 2-Bit LLaMA-2 [N].
- Ehrlichkeit: nahe an ButterflyQuant, Neuheit nur über Clifford-Rundung (N-B teilweise bekannt). Neuheit unnötig.
- Ampel: grün-gelb. Billiger Pareto-Lauf (PTQ ohne Training).

### R4. Monarch-Mixer-Tokenmischung (einzig echter FLOP-Hebel)
- Idee: statt Attention (n^2) Monarch-gesteuerte Faltung/Gating, n^1,5. Quanten-Bezug: schwach (FFT/Butterfly-Struktur), aber Butterfly = Quanten-Fourier-Layout.
- Zahlen: M2 = BERT-base/large (GLUE) mit bis zu 27 % weniger Parametern, bis 9,1x Durchsatz bei Länge 4K; ViT-b +1 % mit halben Parametern; GPT-Stil gleich bei 360M auf The Pile [WS].
- Problem: 4K-Kontext und 360M Parameter nicht auf 4 Kernen; im Container nur Kontext 256-512 testbar, dort kaum FLOP-Differenz. Raven nötig.
- Ampel: gelb (Aufwand hoch, Quanten-Bezug dünn).

### R5. Tensor-Produkt-Attention (TPA) im Byte-LM: KV-Cache als Tensorprodukt niedrigen Rangs
- Idee: Q,K,V als Summen von Tensorprodukten (Kontext-Faktoren x Kopf-Faktoren); KV-Cache schrumpft. Quanten-Bezug: Tensorprodukt-Zustände, Schmidt-Rang. Passt zu Schmidt-Mass der Notizen.
- Zahlen: arXiv 2501.06425 (Zhang et al., Tsinghua/UCLA/Princeton), 5 Fassungen; Vergleich mit MQA/GQA/MLA; Abstract-Zahl (kleinerer Cache, T6-Transformer) nicht verifiziert, "10x" nur aus Drittquelle [WS].
- Bei kurzem Kontext spart es Speicher, nicht FLOPs; Test nur "bpc gegen KV-Bytes".
- Ampel: gelb.

### R6. MPO/TT-Kompression eines trainierten Mini-LMs (CompactifAI-Protokoll) gegen SVD-Niedrigrang bei gleichen Parametern und FLOPs
- Idee: FFN/Attention-Matrizen per sequentieller SVD in MPO (Bonddimension chi), kurzes Nachtraining. Quanten-Bezug: Verschränkungsspektrum der Gewichte; chi-Wahl über Singulärwerte = "Entanglement-Budget".
- Zahlen: CompactifAI LLaMA-2 7B auf ~30 % Größe, > 90 % Genauigkeit nach kurzem Nachtraining; kombiniert mit Quantisierung: 93 % weniger Speicher, 70 % weniger Parameter, 50 %/25 % schneller Training/Inferenz, 2-3 % Genauigkeitsverlust (Firmenpapier, ohne Replikation) [WS]. MPO-Feintuning (ACL 2021): ~91 % weniger trainierbare Parameter [WS]. ESANN 2025 LLaMA-2-MPO ähnlich [WS].
- Kritisch: Kontrolle "gleich viele Parameter, normales Niedrigrang (SVD)" fehlt in Firmenzahlen. Das ist die eigentliche Frage und für Jugend forscht stark (ehrliche Gegenprobe).
- Ampel: gelb-grün.

### R7. TT-LoRA / MetaTT-Adapter gegen LoRA bei gleichen Parametern und FLOPs
- Zahlen: LoRETTA bis 100x weniger trainierbare Parameter als LoRA/Adapter (LLaMA-2) [WS]; TT-LoRA (2408.01008) gleich/höher bei "deutlich" kleinerem Modell, ohne Adapter-Latenz [WS]; MetaTT (2506.09105): ein gemeinsamer TT über alle Module, Parameter ~ Summe der Moden statt Produkt [WS]. Intern: Quantum-PEFT saturiert ab L=3 [N].
- Nähe zu T-006/QIRE-Adapter hoch, Platz bekannt (QuanTA, Quantum-PEFT). Wenig Neues, aber billiger Pareto "Parameter gegen Verlust" mit ≥3 Seeds. Mit I-045-v4 verträglich: Adapter-FLOPs sind klein.
- Ampel: gelb.

### R8. Orthogonalisierte Updates (Muon) und orthogonale Gewichte im kleinen LM
- Idee: unitär/orthogonal als Optimierer-Eigenschaft statt Schicht. Quanten-Bezug: unitäre Dynamik; schwach.
- Zahlen: Moonlight ~2x Recheneffizienz gegenüber AdamW (rechenoptimal), 3B/16B-MoE mit 5,7T Tokens [WS]; Essential AI: Muon erweitert Pareto-Front, muP-Transfer bis 4B [WS]. Das ist ein Gleiche-FLOPs-Vorteil in der Praxis (Optimierer-Schritt ~einige % extra).
- Auf kleinem Byte-LM auf CPU billig prüfbar (Newton-Schulz auf d x d). Antwort könnte "gilt nicht bei winzigen Modellen" sein, auch ehrliche Aussage.
- Ampel: gelb. Vorsicht: Quanten-Bezug dünn, nur "unitär/orthogonal" als Leitmotiv.

### R9. Born-Maschine / Tensor-Netz-Sprachmodell (AMPS) als Eigenstück gegen RNN gleicher Parameterzahl
- Idee: Wahrscheinlichkeit = |MPS|^2, exakte Likelihood, exaktes Sampling. Quanten-Bezug: direkt (Born-Regel, Bonddimension = Verschränkung).
- Zahlen: Hou/Li/You (2311.05050, Fassung Dez. 2025): trainierbare Token-Einbettung per POVM verbessert gegenüber festen Indizes; ältere MPS-Modelle deutlich schlechter als Standardmodelle [WS]. Keine Byte-LM-bpc-Zahl gefunden.
- Erwartung: verliert bei gleichen FLOPs gegen kleines GRU/Transformer. Dafür sehr gut als Erklär- und Negativbefund-Baustein ("Dequantisierung/Klassik" gemäß I-038).
- Ampel: rot für Leistung, grün für Anschaulichkeit.

### R10. FlatQuant-artige Kronecker-Transformation (zwei kleine Matrizen statt eine große Rotation)
- Idee: gelerntes P = P1 (x) P2, Parameter 2*sqrt(d)^2, FLOPs d*sqrt(d) statt d^2 (oder d log d bei Hadamard). Quanten-Bezug: Tensorprodukt-Gatter auf zwei Teilregistern.
- Zahlen: <1 % Genauigkeitsverlust W4A4 LLaMA-3-70B, +7,5 % über SpinQuant; Zusatzverlangsamung 0,07x gegenüber 0,26x bei QuaRot; bis 2,3x Prefill-, 1,7x Decode-Speedup gegenüber FP16 (ICML 2025) [WS].
- Als Variante von R3 sinnvoll (Arm 3 "Kronecker-gelernt"), nicht als eigene Linie. QuIP# (RHT + E8-Gitter, 2-Bit, > 3x schneller als FP16, ICML 2024) [WS] bleibt als Zitat; Gitter-Codebuch im Container zu aufwendig.
- Ampel: gelb (als Zusatzarm).

## 4. Deutung und langweiligste Erklärung
- Fast alle Vorteile der Papiere sind Speicher/Bits oder GPU-Wandzeit; echte FLOP-Gewinne gibt es kaum (R4 ausgenommen). Das deckt sich mit I-045-v4.
- Langweiligste Erklärung für jeden Vorsprung bei kleinen Modellen: Unterschiede in effektiver Lernrate/Initialisierung (Qiu: optimale LR skaliert d^-1 dicht vs d~^-1/2 blockdiagonal [WS]); darum LR je Arm tunen (Validierung, Seeds 900-999).
- Herstellerzahlen (CompactifAI, BitNet v2 Arbeitsstand, FlatQuant-Geschwindigkeit) sind nicht unabhängig repliziert.
- "Quanten-inspiriert" bleibt Etikett (I-038); Tensor-Netze/Hadamard sind klassisch, Quantenvorteil nie behaupten.

## 5. Vorschlag (kein Lauf gestartet; Zustimmung Maltes nötig)
- **Lauf A (R1+R6 kombiniert, Byte-LM UD_German-GSD):** Arme: dicht, SVD-Niedrigrang (gleiche Parameter), Monarch/BTT (gleiche FLOPs, breiter), MPO-Nachkompression eines dichten Modells. Kontrolle: LR-Tuning je Arm, 3 Trainingslängen. Hypothese: Monarch-breit schlägt dicht erst ab genug Training/Breite, sonst nicht. Seeds >= 5 (Mess-Seeds laut Regel), Effekt: bpc-Differenz >= 0,02. Laufzeit Container ~1-2 h je Arm-Satz, Raven ~15 min.
- **Lauf B (R2+R3+R10, Quantisierung):** Arme: ohne Rotation, zufällig-orthogonal, Hadamard fest, Cayley/Givens gelernt, Kronecker gelernt; 2/3/4 Bit; Maß bpc gegen Bytes. Container ~1 h (PTQ), Raven nicht nötig.

## 6. BibTeX (Block; nicht in literatur.bib geschrieben)
Autoren/Seiten teils [Gedächtnis]; vor Aufnahme gegen arXiv prüfen. Eintraege mit Stern (*) existieren schon als Key in notizen.md.

```bibtex
@inproceedings{tseng2024quipsharp,
  title={QuIP\#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks},
  author={Tseng, Albert and Chee, Jerry and Sun, Qingyao and Kuleshov, Volodymyr and De Sa, Christopher},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={235}, year={2024}, eprint={2402.04396}}

@inproceedings{sun2025flatquant,
  title={FlatQuant: Flatness Matters for LLM Quantization},
  author={Sun, Yuxuan and Liu, Ruikang and others},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={267}, year={2025}, eprint={2410.09426}}

@article{wang2025bitnetv2,
  title={BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs},
  author={Wang, Hongyu and Ma, Shuming and Wei, Furu},
  journal={arXiv preprint arXiv:2504.18415}, year={2025}}

@article{zhang2025tpa,
  title={Tensor Product Attention Is All You Need},
  author={Zhang, Yifan and Liu, Yifeng and others},
  journal={arXiv preprint arXiv:2501.06425}, year={2025}}

@article{anjum2024ttlora,
  title={Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs},
  author={Anjum, Afia and others},
  journal={arXiv preprint arXiv:2408.01008}, year={2024}}

@article{yang2024loretta,
  title={LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models},
  author={Yang, Yifan and others},
  journal={arXiv preprint arXiv:2402.11417}, year={2024}}

@article{metatt2025,
  title={MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning},
  author={unvollstaendig},
  journal={arXiv preprint arXiv:2506.09105}, year={2025},
  note={Titel unverifiziert; nur arXiv-ID aus Suche}}

@inproceedings{fu2023monarchmixer,
  title={Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture},
  author={Fu, Daniel Y. and Arora, Simran and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023}, eprint={2310.12109}}

@inproceedings{qiu2024computebetterspent, % *
  title={Compute Better Spent: Replacing Dense Layers with Structured Matrices},
  author={Qiu, Shikai and Potapczynski, Andres and Finzi, Marc and Goldblum, Micah and Wilson, Andrew Gordon},
  booktitle={ICML}, series={PMLR}, volume={235}, year={2024}, eprint={2406.06248}}

@article{tomut2024compactifai, % *
  title={CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks},
  author={Tomut, Andrei and Jahromi, Saeed S. and others and Orus, Roman},
  journal={arXiv preprint arXiv:2401.14109}, year={2024}}

@inproceedings{ashkboos2024quarot, % *
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L. and others},
  booktitle={NeurIPS}, year={2024}, eprint={2404.00456}}

@inproceedings{liu2025spinquant, % *
  title={SpinQuant: LLM Quantization with Learned Rotations},
  author={Liu, Zechun and Zhao, Changsheng and Fedorov, Igor and others},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2025}, eprint={2405.16406}}

@article{hou2023tnbornmachine,
  title={Sequential learning on a Tensor Network Born machine with Trainable Token Embedding},
  author={Hou, Wanda and Li, Li and You, Yi-Zhuang},
  journal={arXiv preprint arXiv:2311.05050}, year={2023}, note={Fassung Dez. 2025; Autorenreihenfolge [Gedaechtnis]}}

@article{liu2025moonlight,
  title={Muon is Scalable for LLM Training},
  author={Liu, Jingyuan and others},
  journal={arXiv preprint arXiv:2502.16982}, year={2025},
  note={ID aus Suchtreffer papers.cool/2502.16982 geraten, nicht geprueft}}

@article{essentialai2025muon,
  title={Practical Efficiency of Muon for Pretraining},
  author={Essential AI},
  journal={arXiv preprint arXiv:2505.02222}, year={2025}, note={Autoren unvollstaendig}}

@inproceedings{gao2021mpo, 
  title={Enabling Lightweight Fine-tuning for Pre-trained Language Model Compression based on Matrix Product Operators},
  author={Gao, Peiyu and others},
  booktitle={ACL}, year={2021}, eprint={2106.02205}, note={Autoren [Gedaechtnis]}}
```

## 7. Skriptpfade
Keine Skripte (nichts gerechnet). Bericht: `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w8_bericht.md`.
Lücken: nur Abstracts gesehen; TPA-, SpinQuant- und M2-Zahlen aus Zusammenfassungen; Born-Machine-bpc-Zahlen nicht gefunden; Muon-Moonlight-ID unsicher.
