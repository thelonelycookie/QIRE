# STROM (Späher, 29.10.2026)
Quelle: 7 WebSearch, nur Snippets/Aggregatorseiten, alle Zahlen UNGEPRÜFT. Nicht aufgenommen, da schon in Z3_ideen: HARP (2605.29843), ButterflyQuant (2509.09679); in ARCHIV: Qift (2606.02823). Keine neuen 2026-Funde zu Seed-Basis und MPO (nur PicoGPT 2603.28534, schon in Z1).

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-05 | QAM-W: Joint 2D Codebook Quantization via Hadamard Rotation and Activation-Aware Scaling | arXiv 2605.26339 | ~5,5 Bit/Gewicht, PPL innerhalb 0,4 % von BF16 (WikiText-2) | neuer Gegner/Lloyd: Hadamard + 2D-Lloyd-Max-Buch; zeigt Lloyd-Gitter nach Rotation als Standard; hohe Bitzahl, schwacher Vergleich zu 2 Bit |
| 2026 | PolarQuant (Vicentino): Walsh-Hadamard + skalares Lloyd-Max-Buch | laut QAM-W-Snippet, ID fehlt | keine | neuer Gegner für Hadamard+Lloyd-Arm; Namensverwechslung mit PolarQuant/TurboQuant hüten |
| 2025-02 (EMNLP-F 2025) | Rotate, Clip, and Partition (RCP): W2A4KV4 mit Rotation + lernbarem nicht-uniformem Quantisierer | arXiv 2502.15779 | LLaMA-2-7B W2A4KV4: PPL-Verlust 2,84, 5,29x kleiner | stützt Aktivierungsachse (Rotation trägt A4); QAT-Gegner zu Z2c-FFN-Befund; W2A2 selbst: kein Treffer gefunden |
| 2026-02 | 1-Bit Wonder: QAT durch K-Means-Quantisierung | arXiv 2602.15563 | Skalierungsgitter Größe x Token x Präzision, Nemotron-CC | neue Idee/Gegner: k-Means-Gitter statt uniform im QAT, passt zur Gitterachse (Z3) und Trainingslängen-Achse |
| 2026 (ICML 2026, PMLR 306) | Scaling Law for Quantization-Aware Training (Chen et al.) | proceedings.mlr.press/v306/chen26gl.html | W4A4, 268 Läufe: Fehler sinkt mit Modellgröße, steigt mit Tokens und gröberer Gruppe | stützt (Schaden wächst mit T, wie Z1b); neuer Gegner/Messlatte: Gesetz mit Gruppengröße; QIRE-Rotation als Steigungs-/Niveau-Term prüfen |
| 2026-02 | Scaling laws for precision in high-dimensional linear regression (Zhang et al.) | arXiv 2602.19241 | multiplikatives vs additives Quantisierungsrauschen skalieren verschieden | neue Idee: Theorierahmen für "Rotation senkt Niveau a, nicht Exponent b"; Rotation macht Rauschen eher additiv? (nur Hypothese) |
| 2024-11 / 2025-02 | Low-Bit Quantization Favors Undertrained LLMs (2411.17691); ParetoQ (2502.02631) | arxiv.org/abs/2411.17691, 2502.02631 | >1500 Checkpoints; ParetoQ: 2-Bit/ternär liegt auf Pareto-Front | stützt Befund "PTQ-Schaden wächst mit Trainingslänge"; falls nicht in ARCHIV zitiert: nachtragen (nicht 2026) |
| 2026 (ICLR) | Accelerating Inference for Multilayer NNs with Quantum Computers (Rattew) | iclr.cc 2026 poster 10009571 | quadratischer/quartischer Speedup nur mit Quantenhardware | irrelevant für klassische Wirkung (I-038), nur Abgrenzung |

## Lücken
- W2A2 mit Rotation: keine 2026-Arbeit gefunden (nur W2A4). Unser A-Bit-Befund (FFN-Mitte 4 Stufen) liegt im selben Regime wie Qift/RCP.
- Seed-Basis, MPO, quanten-inspiriert-effizient: 2026 nichts Neues im Snippet.

## 29.10. Lauf 2 (Späher, 6 WebSearch, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Hardware-Energie, MPS-LM, Hadamard im Training, Dequantisierung, Zufallsmerkmale, Byte-Skalengesetze. Nicht aufgenommen (schon da): 2602.15563, 2603.28534.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-07 | LightRot: leichtgewichtige Rotation + Architektur für Low-Bit-LLM-Inferenz | arXiv 2607.27704 | keine im Snippet (Tabelle zählt 1 MAC = 2 Ops) | Hardwarekosten der Rotation (BK/Aktivierung): Gegner/Messlatte für unseren FWHT-Zeitbefund (w12) |
| 2025-03 | FIGLUT: energieeffizienter FP-INT-GEMM-Beschleuniger mit Lookup-Tabellen | arXiv 2503.06862 | 3 Bit: +59 % TOPS/W; bei gleicher PPL (2,4 Bit-Ops) +98 % TOPS/W | Energiemodell für Niedrigbit; LUT statt Multiplikation, Idee für Ops-Zählung (Spalte FL) |
| 2025-08 | Pushing the Envelope of LLM Inference with Ultra-Low-Bit Quantized Models | arXiv 2508.06753 | INT2xINT8 nativ auf Intel Xe2, bis 7x schneller als 16 Bit; Tokens/J 2,75x besser als A100 | stützt: 2-Bit-Kerne existieren real; zeigt Rotationsfreiheit nötig? Bezug Z2 (2-Bit-Großmodell vs fp16 gleicher Bytes) |
| (ohne ID) | 28-nm-LLM-Beschleuniger | im Snippet nur Tabelle | INT4 27,4 TOPS/W (250 MHz, 0,9 V), INT8 22,9 TOPS/W (andere Bedingungen) | grobe pJ/Op: 1 TOPS/W ≈ 1 pJ/Op; nur Größenordnung, nicht vergleichbar |
| 2026-01 | Quartet II: unverzerrte NVFP4-Quantisierung (MS-EDEN) fürs Vortraining | arXiv 2601.22813 | >2x weniger Fehler als stochastisches Runden; bis 1,9B Param, 38B Tokens; 4,2x Kernel vs BF16 | neue Idee/Gegner für TZ x Unitarität (Rotation im Training, ungetestete Pflichtzelle) |
| 2025-01 (v3) | HALO: Hadamard-Assisted Lower-Precision Optimization | arXiv 2501.02625 | bis 1,41x End-to-End (Feintuning, RTX 4090) | Hadamard in Vorwärts+Rückwärts: Referenz für Rotation im Training (TZ) |
| 2026-05 | Winning Lottery Tickets via quanten-inspirierten klassischen Algorithmus | arXiv 2605.13979 | poly(D) statt exponentiell; nur Simulation | echtes quanten-inspiriertes ML (Dequantisierung) mit Netzbezug; Zitat für "quanten-inspiriert" |
| 2023 | Tensor Networks for classical simulation of VQML (Di Marcantonio et al.) | arXiv 2307.06937 | VQML-Modelle = Unterklasse von MPS | Abgrenzung/Zitat (I-038: Wirkung klassisch) |
| 2025-10 (ICLR 2026) | Not All Bits Are Equal: Scale-Dependent Memory Optimization for Reasoning Models | arXiv 2510.10964 | Kippunkt ca. 8-Bit-4B (~4,2 GB): kleiner -> mehr Gewichts-Bits, größer -> mehr Tokens/KV | Byte-Budget-Gesetz mit BG vs BK getrennt; Messlatte für Spalte BK |
| 2025-02 | QuEST: stabiles Training mit 1-Bit Gewichten und Aktivierungen | arXiv 2502.05003 | 4-Bit-Training optimal ab Compute-Schwelle, die mit Größe fällt | Gegner/Gesetz für Trainingsbits (TZ); Hadamard dort Bestandteil |
| 2026-05 | Compute Optimal Tokenization | arXiv 2605.01188 | ~60 Bytes/Parameter konstant bei festem Compute | Datenachse (DA, leer): Byte-Rechnung auf Datenseite |
| 2016 | Orthogonal Random Features (SORF) / Fastfood-Verallgemeinerung (1605.09049) | arXiv 1610.09072, 1605.09049 | SORF nahezu linear in d, kaum Fehler vs ORF | 2025–26: kein neuer Treffer; Hadamard ≡ Zufalls-Orthogonale (I-048 H2) ist alt bekannt, zitieren |

Lücken Lauf 2: keine direkte pJ/Op-Zahl INT2 vs INT4 (nur TOPS/W); kein 2026-MPS-Sprachmodell (nur MPO 2603.28534, Entanglement-Feintuning 2601.06788); keine 2025–26-Fastfood-Arbeit.
