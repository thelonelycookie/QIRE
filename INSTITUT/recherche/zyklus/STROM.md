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

## 29.10. Lauf 3 (Späher, 6 WebSearch, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: SSM/Hybrid + Quantisierung, Sparsity + Quantisierung, Distillation Niedrigbit, Tensornetz-Attention. Nicht aufgenommen (schon da oder ohne 2026): HARP, ButterflyQuant, RCP. Snippets nennen teils widersprüchliche Jahresangaben (SSDi8: "ICLR 2026" vs. ID 2608), vor Zitat prüfen.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-03 | Sparse-BitNet: 1,58-Bit-LLMs sind natürlich freundlich zu semistrukturierter Sparsity | arXiv 2603.05168 | ~42 % Nullgewichte im Ternärmodell; weniger Verlust als FP bei gleicher N:M-Sparsity; bis 1,30x schneller; Qwen-2.5 0,5-3B | neue Idee/stützt: Ternär-Nullen = Gratis-Sparsity; Sparsity x Bits als Achse (Byte-Budget); Gegner zu 2-Bit-ohne-Sparsity |
| 2024-10 | SLiM: One-shot Quantisierung + Sparsity + Low-Rank | arXiv 2410.09615 | LLaMA-2-7B, ca. 8x Kompression: 4 Bit + 2:4 PPL 7,2 vs. 2 Bit allein 12,5 | Messlatte bei gleichem Byte-Budget: 4 Bit + Sparsity schlägt 2 Bit (nicht 2026) |
| 2025-09 | Optimal Brain Restoration (gemeinsame Quantisierung + Sparsifizierung) | arXiv 2509.11177 | bis 4,72x vs. FP16-dense (INT4 + 2:4) | Gegner zur Sparsity-Achse (nicht 2026) |
| 2026 (ICLR?) | SSDi8: 8-Bit-Quantisierung für Mamba-2 (State Space Duality) | arXiv 2608.21952 (ID unsicher) | W4A8/W8A8 nahe FP16, bis 1,4x; W4A4 ausgeschlossen | SSM-Quantisierung: Rotation/Hadamard für Zustandspfad ungetestet; Abgrenzung Transformer-only |
| 2025 | Quamba2 (ICML 2025); Q-Mamba (ACL-F 2025) | arXiv 2503.22879; Q-Mamba ID fehlt | Quamba2-8B: 4x Speicher, -1,6 % Genauigkeit; Q-Mamba: 4-Bit-Zustand, -50 % Speicher, -2,13 % | Zustandsquantisierung = KV-Cache-Analogon (Spalte BK) bei SSM |
| 2025-10 | BitNet Distillation (Logit- + Attention-Destillation in 1,58 Bit) | arXiv 2510.13998 | keine im Snippet | Distillation-Gegner/Weg zu Niedrigbit-Schüler; 2026-Arbeit dazu: kein Treffer |
| 2025-01 (v7 2026-01) | Tensor Product Attention (TPA) | arXiv 2501.06425 | KV-Cache-Verkleinerung durch Tensorzerlegung von Q,K,V (Faktor ungeprüft) | quanten-inspirierte Tensorzerlegung im Attention-KV; Low-Rank, kein echtes MPS |
| 2024-08 | Mamba in the Llama (Distillation Transformer -> Hybrid) | arXiv 2408.15237 | Hybrid mit 1/4 Attention-Schichten | Distillation + Hybrid-Architektur; nicht 2026 |

Lücken Lauf 3: keine 2026-Arbeit zu GatedDeltaNet/Hybrid-Quantisierung (nur Community-Hinweis: SSM-Schichten in BF16 belassen); keine 2026-MPS-/Tensor-Train-Attention; keine 2026-Distillation in 2-Bit/ternär-Schüler (jüngster Treffer 2510.13998).

## 29.10. Lauf 4 (Späher, 6 WebSearch, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: (a) Rotation vs. Gruppenskalen (Overhead/Hardware), (b) Bitflip-Robustheit, (c) Jugend-forscht-Beispiele. Nicht aufgenommen (schon da): LightRot 2607.27704, HARP, ButterflyQuant.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-07 | GyRot: Synergie zwischen Rotation und feinen Gruppenskalen (KAIST, INT4-Beschleuniger) | arXiv 2607.27694 | bis 3,4x Speed, 3,6x Energie ggü. Basis-Beschleunigern; Autoren: globale Rotation widerspricht lokalen Gruppenskalen | stützt unseren Befund (Rotation ≈ Gruppenskala, kein Gewinn); Hardware-Vorteil nur mit Co-Design (Grobe Rotation + feine Gruppen, CoRFiG); Messlatte für Kombi-Arm |
| 2026-03 | Rotated Robustness (RoR): trainingsfreie Abwehr gegen Bitflip-Angriffe via Householder-Rotation der Aktivierungen | arXiv 2603.16382 | Kollaps bei Zufalls-Bitflips Qwen2.5-7B 3,15 % -> 0,00 %; +0,31 % Speicher, +9,1 % Latenz (Llama-2-7B) | WICHTIGSTER Vorteil von Rotation außerhalb PPL: Robustheit gegen Bitflips; neue Idee: Bitflip-Test als Zusatzachse (Rotation vs g16/g32/Lloyd) |
| 2026-03 | Logic/Lookup-Netze unter Parameter-Bitflips (MLPerf Tiny) | arXiv 2603.22770 | geringere Präzision = höhere Resilienz (Herleitung des erwarteten Fehlers) | Gegner/Theorie: 2-Bit-Gitter könnte von sich aus robuster sein; Bitflip-Erwartungsfehler analytisch ableitbar |
| 2026-02 | TFL: gezielter Bitflip-Angriff auf LLMs | arXiv 2602.17837 | Format-Unterschiede (BF16 Exponent vs INT8) | Abgrenzung: INT-Gewichte ohne Exponentenfeld; Flip = Skala x Zweierpotenz |
| 2025 (EMNLP) | Bitflip-Fehler in LLMs, Analyse + Abwehr (Chen et al.) | Link nur über EMNLP 2025 Snippet, ID fehlt | BF16-Flips bis 98 % Leistungsverlust (zitiert) | Motivation für Robustheitsachse |
| 2026-05 | Quantizing with Randomized Hadamard Transforms: Heuristik jetzt bewiesen | arXiv 2605.06014 | Beweis (keine Zahl im Snippet) | Theorie-Zitat: warum Rotation Ausreißer glättet (I-048 H2) |
| 2026-01 | Any Orthogonal Transform Will Do: Range Reduction erklärt Rotation (Super-Resolution) | arXiv 2601.21069 | Hadamard-Schritt ca. 3x Wallclock; Padding auf 2er-Potenz +0,13-0,27 Bit/Param | stützt "jede Orthogonale reicht" (I-048 H2); Padding-Overhead relevant für Nicht-2er-Potenz-Dimensionen |
| 2025-12 / 2026-01 | ConvRot (2512.03673); MixQuant (2601.22347): Block-Hadamard | arXiv 2512.03673, 2601.22347 | ConvRot 2,26x Speed, 4,05x Speicher (FLUX); MixQuant: kleinere Blöcke -> mehr Worst-Case-Ausreißer | Blockgröße b als Achse (Kosten linear statt quadratisch), passt zu g16/g32-Vergleich |

(c) Jugend forscht: keine konkrete Quantisierungs-/Effizienz-KI-Arbeit gefunden (nur Chemie-KI-Projekt Berlin, Lenovo-Leitfaden). Archive jugend-forscht.de manuell durchsuchen ("Modellkompression", "Edge-KI"); kein Treffer = Nische, kein Duplikatrisiko.

Lücken Lauf 4: keine Bitflip-Studie für INT4/INT2-Gewichte mit/ohne Rotation (Hadamard speziell, nur Householder in RoR); keine direkte Skalen-Overhead-gegen-Rotation-Messung bei gleichem Durchsatz.

## 29.10. Lauf 5 (Cloud-Späher, 4 WebSearch, 16:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: KV-Cache-Quantisierung, Codebücher/Trellis-VQ, gelernte Rotation, TurboQuant-Kritik. Übernommen aus der Cloud-Sitzung, deren Push scheiterte.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-08 | OptR: INT2-KV mit Rotation, Key-Reparametrisierung und gelernten orthogonalen Korrekturen | arXiv 2608.02691 | INT2-KV bis 8x kleiner als BF16; Kritik: QuaRot/OSCAR optimieren nur den Rekonstruktionsfehler als Proxy | BK: Rotation + 2 Bit im KV; Proxy- gegen Endverlust-Kritik |
| 2025-01 | RotateKV: 2-Bit-KV mit ausreißer-adaptiven Rotationen | arXiv 2501.16383 | LLaMA-2-13B: PPL-Verlust <0,3 bei 2 Bit; 3,97x weniger Spitzenspeicher | Messlatte 2-Bit-KV |
| 2025-10 | KVLinC: KV mit Hadamard und linearer Korrektur | arXiv 2510.05373 | Keys kanalweise, rotierte Values tokenweise quantisieren | Achsenwahl der Skalen im KV |
| 2026-05 | BCJR-QAT: differenzierbare Trellis-Quantisierung | arXiv 2605.10655 | keine Zahl | Gitterachse: QAT mit Trellis |
| 2026 | Proteus: Trellis-Quantisierung ohne Tabelle für 2-Bit-LLMs (ICML 2026) | icml.cc/virtual/2026/poster/63948 | Llama 2 7B–70B: besser als TCQ | berechneter Code statt gespeichertes Codebuch, verwandt mit Seed-Basis |
| 2026-06 | LC-QAT: dateneffizientes 2-Bit-QAT mit linear beschränkter VQ | arXiv 2606.10531 | 0,1–10 % der Daten | Gegner für 2-Bit-QAT |
| 2026-09 | Leech-Gitter-VQ mit Multi-Shell-Decoder | arXiv 2609.02652 | nur Selbstangabe | dichte Gitter, Dekodierkosten |
| 2026-05 | EVA: VQ-Beschleuniger fürs Decoding | arXiv 2605.24144 | keine | Hardware |
| 2026-05 | Statistik der KV-Quantisierung nach TurboQuant | arXiv 2605.08114 | feste Rotation macht nicht-isotrope Verteilungen nicht isotrop | Theorie-Gegner zu „Rotation macht Gauß“, passt zu Z3 |

## 29.10. Lauf 6 (Späher, 4 WebSearch, 18:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Seed-/Zufallsbasis-Gewichte, FP4-Vortraining, Hypernetz-Gewichte, W2A2. Nicht aufgenommen (schon in Z1/k3/k5/ARCHIV): SeedLM 2410.10714, NVFP4-Pretraining 2509.25149, Quartet 2505.14669, Qift, QAM-W.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-05 | Influence-Inspired Spectral Rotations (BBT-spectral), Gewichts-only W2A16 | arXiv 2605.25203 | wikitext-2-PPL 15-58 % unter Vanilla-AutoRound bei W2A16 | gelernte/spektrale Rotation bei 2 Bit: Gegner zu fester Hadamard im Z3-Rotationsarm |
| 2026 (ICML 2026) | HyperCompress: Hypernetz-Autoencoder regeneriert Gewichte aus Latentcode | icml.cc/virtual/2026/78330 | 1D-CNN (Lager-Fehler): 512 KB -> 128 B, <0,5 % Genauigkeitsverlust; schmaler Benchmark | erzeugte statt gespeicherte Gewichte, verwandt mit Seed-Basis; Skalierung auf LLMs offen |

Lücken Lauf 6: keine 2026-Folgearbeit zu SeedLM/Zufallsbasis-Gewichten; keine 2026-FP4-Skalengesetz-Arbeit (nur Quartet, NVFP4 v2 03/2026); weiterhin kein W2A2-Treffer.

## 29.10. Lauf 7 (Cloud-Späher, 4 WebSearch, 19:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Tensornetze/MPO, Aktivierungs-Ausreißer W4A4, quanten-inspirierte Trainingsalgorithmen, Niedrigbit-Skalengesetze. Nicht aufgenommen (schon in Z1/Welle/ARCHIV): CompactifAI 2401.14109, Scaling Laws for Precision 2411.04330, Spectra 2407.12327, ParetoQ 2502.02631.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-02 | S2D: Selective Spectral Decay für quantisierungsfreundliche Aktivierungen | arXiv 2602.14432 | ImageNet W4A4: bis +7 % PTQ-Genauigkeit, +4 % mit QAT; Experimente v. a. Vision (SigLIP-2), LLM unklar | neue Idee: Ausreißer an der Quelle (dominante Singulärwerte der Gewichte) dämpfen statt rotieren; Gegner/Ergänzung zum Rotationsarm (Aktivierungsachse) |
| 2024-12 | DFRot: ausreißerfreie, Massive-Activation-freie rotierte LLMs mit verfeinerter Rotation | arXiv 2412.00648 | keine Zahl im Snippet; Befund: zufällige Orthogonale erhöhen bei Massive-Activation-Tokens den Fehler | Gegner zu „jede Orthogonale reicht“ (I-048 H2): Hadamard vs. zufällig bei Spezial-Tokens; nicht 2026 |

Lücken Lauf 7: kein 2026-Treffer zu MPO/Tensornetz-Kompression (nur CompactifAI 2024); kein 2026-Paper zu quanten-inspiriertem NN-Training (nur Fan et al., AAAI 2024: Quanten-inspirierte Netze trainieren deutlich langsamer als klassische; MicroAlgo-Pressemitteilung 04/2026 ohne Beleg); kein neues Niedrigbit-Skalengesetz 2026 über die bekannten (Kumar, Spectra, ParetoQ) hinaus.

## 29.10. Lauf 8 (Cloud-Späher, 4 WebSearch, 20:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: strukturierte Matrizen (Monarch/Butterfly) im Training, Ternär-Vortraining, Energie INT2 vs INT4, Kronecker/Hadamard-Parametrisierung. Nicht aufgenommen (schon da): LightRot 2607.27704, GyRot 2607.27694, FIGLUT 2503.06862.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026 (NeurIPS 2026) | Monarch-Muon: strukturbewusstes Muon für schnelles, speichereffizientes Vortraining | neurips.cc/virtual/2026/161502 | -10 % trainierbare Parameter bei +2,3 % Val-Loss ggü. dichtem Muon; Gewichte bleiben vollrangig | strukturierte (Butterfly-artige) Gewichte direkt trainiert: Gegner/Messlatte zur Hadamard-MLP-Linie |
| 2022-04 | Monarch: Expressive Structured Matrices (Dao et al.) | arXiv 2204.00595 | GPT-2-Vortraining auf OpenWebText ca. 2x schneller ohne Qualitätsverlust | Grundlage, zitieren (nicht 2026) |
| 2026-02 | TernaryLM: nativ 1,5-Bit mit schichtweiser Skalierung | arXiv 2602.07374 | 132M Parameter, PPL 58,42 (TinyStories), 2,4x weniger Speicher als FP32 | kleines Ternär-Training von Grund auf; Größenordnung nahe unseren Läufen, Vergleich nur grob |
| 2026 | BitMamba-2: 1,58-Bit-SSM, 255M und 1B von Grund auf | zenodo.org/records/18394665 (kein arXiv) | 1B: +7,8 % ARC-Easy (Selbstangabe) | Ternär x SSM; schwache Quelle (Zenodo) |
| 2026-06 | TWLA: ternäre Gewichte + Niedrigbit-Aktivierungen per PTQ, Kronecker-Hilfstransformation | arXiv 2606.13054 | Kronecker-Transform ca. 2048x kleiner als dichte 4096x4096 | Kronecker statt volle Rotation: Speicherkosten der Rotation; Gegner zu A-Bit-Arm |
| 2026-08 | ScaleQ-1.58: PTQ nach 1,58 Bit | ID fehlt (Snippet) | Qwen3-1.7B >90 % der BitNet-Basis mit 4M Kalibrier-Tokens | Gegenstück zu Training von Grund auf (TZ) |
| 2025-01 | BlockDialect: blockweise Mischformate | arXiv 2501.01144 | 4-Bit-MAC ca. 2,45x leistungseffizienter als INT8 (Selbstangabe) | grober Energiefaktor 4 vs 8 Bit; nicht 2026 |

Lücken Lauf 8: weiterhin keine pJ/MAC-Zahl für INT2 vs INT4 (LightRot: 27,4 TOPS/W bei 4 Bit, ca. 36 pJ/Op laut Snippet-Umrechnung, Systemwert); FIGLUT: Festbreiten-Hardware füllt Sub-4-Bit auf 4 Bit auf (2 Bit spart dort keine Energie, nur Bit-seriell/LUT); keine 2026-Arbeit mit Walsh-Hadamard/Butterfly als trainierte Gewichtsparametrisierung kleiner LMs gefunden.

## 29.10. Lauf 9 (Cloud-Späher, 4 WebSearch, 21:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Aktivierungssparsity + Quantisierung, Mixed-Precision/Bitzuteilung, Optimizer-Zustände niedrigbittig, Weight Sharing/Low-Rank. Nicht aufgenommen (schon da): HARP 2605.29843, QuIP# 2402.04396, ParetoQ.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-06 | SharQ: Aktivierungssparsity + FP4-Quantisierung (Sparse-Dense-Zerlegung, trainingsfrei) | arXiv 2606.26587 | holt 43-63 % der Lücke NVFP4 -> FP16 zurück; 2,2-2,4x Latenz (Zweitquelle Emergent Mind) | Sparsity x Aktivierungsbits (BK/A-Achse); Gegner zum Rotationsarm bei A4 |
| 2025-02 (ICLR 2025) | TEAL: trainingsfreie Aktivierungssparsity | mlanthology.org/iclr/2025/liu2025iclr-trainingfree | mit Gewichtsquantisierung kombinierbar (Angabe im Snippet) | Basis für Sparsity-Achse, zitieren |
| 2026-07 | Prox: trainingsfreie FFN-Aktivierungssparsity (SwiGLU) | arXiv 2607.27591 | keine Zahl im Snippet | FFN-Sparsity, Bezug zu Z2c-FFN-Befund |
| 2026-02 | ScaleBITS: skalierbare Bitbreitensuche, hardwarenah, unterhalb Schichtebene | arXiv 2602.17698 | glatterer Genauigkeit-Kompression-Verlauf als Vorgänger (keine Zahl) | Kritik: schichtweise Bitzuteilung zu grob; Messlatte für gemischte Bits im Byte-Budget |
| 2026 (DATE 2026) | DAIQUIRI: meist 2 Bit, empfindlichste Schichten 3 Bit | past.date-conference.com/.../2026/DATA/3036.pdf | Llama-3.2-1B, ca. 2,6 Bit/Gewicht, nahe uniform 3 Bit | Gegner: gemischt 2/3 Bit statt uniform 2 Bit |
| 2026-05 | GAMMA: trainingsfreie modulweise Präzisionswahl für beliebige Bitbudgets | arXiv 2605.18475 | keine Zahl | Bitbudget-Zuteilung ohne Neurechnung je Budget |
| 2026-03 | Beyond Outliers: datenfreie schichtweise Sensitivität (numerisch + strukturell) | arXiv 2603.17354 | keine Zahl | datenfreie Empfindlichkeit, passt zu Rotationsarmen ohne Kalibrierung |
| 2026-03 | Topollai & Choromanska: niedrigpräzise EMA-Optimizer-Zustände werden "stale" | arXiv 2603.16731 | Reset-Pläne holen Großteil des Verlusts zurück | TZ: Niedrigbit-Training, Optimizer-Speicher als eigene Byte-Position |
| 2026-05 | MuonQ: 4-Bit-Muon-Zustände | arXiv 2605.11396 (Zweitquelle chatpaper) | bis 7,3x weniger Optimizer-Speicher, Loss ≈ Voll-Muon | Trainingsspeicher; Muon-Bezug zu Monarch-Muon (Lauf 8) |
| 2026-04 | STQuant: raumzeitlich adaptive Optimizer-Quantisierung | arXiv 2604.06836 | GPT2-XL/ViT-Base vs. bnb-8-Bit-AdamW (keine Zahl) | Vergleich 8-Bit-AdamW als Standard-Gegner |
| 2026-08 | Adaptive Log-Space-Quantisierung (AL8) für Optimizer-Zustände | arXiv 2608.22322 (ID aus Snippet, unsicher) | TinyLlama-1.1B, 20K Schritte: PPL 72,90 vs. 72,48 (FP32-AdamW) | kleinster Maßstab, nahe unseren Läufen |
| 2026-08 | LegoLM: strukturiertes Weight Sharing für LLMs | arXiv 2608.08652 | keine Zahl | geteilte statt zufällig erzeugte Gewichte; Nachbar der Seed-Basis |
| 2025-06 | LittleBit: Low-Rank-Latent-Faktoren, danach binarisiert | arXiv 2506.13771 | bis 0,1 Bit/Gewicht (Selbstangabe) | Extremkompression unter 1 Bit; Gegner im Byte-Budget (nicht 2026) |

Lücken Lauf 9: keine Gradientenquantisierung für Kleinmodell-Vortraining gefunden (nur Optimizer-Zustände); keine Arbeit, die Weight Sharing + Low-Rank + Hadamard zusammen testet; SharQ-Zahlen nur aus Aggregator.

## 29.10. Lauf 10 (Cloud-Späher, 4 WebSearch, 22:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Gradienten-/Rückwärtsquantisierung im FP4-Training, Distillation in Niedrigbit, Hybrid/GatedDeltaNet-Quantisierung, QAT-Skalengesetze. Nicht aufgenommen (schon da): Quartet 2505.14669, NVFP4-Pretraining 2509.25149, ParetoQ, 2411.17691, Chen et al. (ICML 2026), BitDistiller-Familie als bekannt.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-02 | Dissecting Outlier Dynamics in LLM NVFP4 Pretraining | arXiv 2602.02047 | Rezept: Hadamard nur im Rückwärtspfad, Vorwärts ohne Rotation (16er-Blockskala reicht); RHT stabilisiert stochastisches Runden der Gradienten | TZ x Unitarität: Rotation nur backward als billigere Zelle; stützt "Rotation ≈ feine Gruppenskala" (vgl. GyRot) |
| 2025-09 | Elucidating the Design Space of FP4 training (Graphcore) | arXiv 2509.17791 | Simulation vieler Stabilisierer; nur wenige lohnen (Hadamard, Tensorskala, stochastisches Runden auf Rückwärts-Aktivierungen) laut Zweitquelle | Messlatte für TZ-Rezept; nicht 2026 |
| 2025-09 | Optimal fraction of training compute for QAT (Cooldown+QAT-Fusion) | arXiv 2509.22935 | optimaler QAT-Anteil wächst mit Compute, vorhergesagt aus Tokens pro Parameter-Byte; Verlustgesetz über N, FP-Tokens, QAT-Tokens, Bits | Gegner/Messlatte für Trainingslängen-Achse und Byte-Budget (Tokens/Byte); nicht 2026 |
| 2026-09 | Train Where the Quantized Model Goes: On-Policy Distillation for Low-Bit Reasoning | alphaxiv.org/abs/2609.26708 (ID ungewöhnlich, vor Zitat prüfen) | Qwen3-0.6B bei ca. 2,79 Bit; 53 % der MATH-500-Antworten laufen ins Budget | Distillation in Sub-3-Bit: Gegner; Schaden zeigt sich bei Reasoning stärker als bei PPL |
| 2026 (AISTATS 2026) | SQuaT: Self-Supervised KD mit quantisierten Lehrer-Features | virtual.aistats.org/virtual/2026/poster/13757 | größte Gewinne bei 1–2 Bit (Selbstangabe) | Distillation-Gegner bei 2 Bit, vermutlich Vision; Übertragbarkeit offen |
| 2026-09 | Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 für die rekurrente Hälfte eines 27B-Hybrids | arXiv 2609.04098 | alle 496 Linearschichten W4A4, ≈BF16 im Seed-Rauschen, 17,5 GiB, Prefill +14–19 %; globale Skalen-Fehlanpassung durch fusionierte GDN-Projektionen (1,8–2,75x) | schließt Lücke aus Lauf 3 (Hybrid-Quantisierung); widerspricht Community-Praxis (Gates in BF16); Rotation dort nicht getestet |

Lücken Lauf 10: keine Gradientenquantisierung für Kleinmodell-Vortraining unter 4 Bit; keine Hadamard/QuaRot-Arbeit für GatedDeltaNet-Schichten; kein 2026-Skalengesetz speziell für kleine Modelle (<1B) außer Chen et al.

## 29.10. Lauf 11 (Cloud-Späher, 4 WebSearch, 23:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Training ohne Master-Gewichte, Gradientenquantisierung <4 Bit, Skalengesetze kleiner Modelle, LUT/Bit-seriell-Energie. Nicht aufgenommen (schon da): ParetoQ, LC-QAT 2606.10531, FIGLUT, Kumar 2411.04330, 2411.17691.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-01 | ECO: Quantisiertes Training ohne Vollpräzisions-Master-Gewichte (Nikdan, Zandieh, Alistarh, Mirrokni) | arXiv 2601.22101 | Fehlerrückführung in den Momentum-Term, kein Zusatzspeicher; nahe verlustfrei ggü. Master-Gewicht-Basis (Gemma-3 1B, 2,1B-MoE FP8, INT4-Feintuning DeepSeek-MoE-16B) | TZ: Master-Gewichte als eigene Byte-Position im Training; neue Idee/Gegner für Niedrigbit-Training von Grund auf |
| 2026-02 (ACL-F 2026) | pQuant: QAT von Grund auf für Sub-2-Bit, 1-Bit-Zweig + kleiner Hochpräzisionszweig | arXiv 2602.22592 | Befund: Parameter-Sensitivität wird homogen, begrenzt Ausdrucksstärke; keine Zahl im Snippet | Gegner im Byte-Budget bei <2 Bit; Hinweis passt zu Z2c-FFN-Befund (wenige empfindliche Anteile) |
| 2026 (EMNLP 2026) | Scaling Laws for Mixed Quantization (Cao et al.) | aclanthology.org/2026.emnlp-main.142 (arXiv-ID fehlt) | Gesetz über Anteil quantisierter Parameter, Blockgröße, Modellgröße; größere Modelle vertragen höheren Anteil | Messlatte für Gruppengröße/gemischte Bits; ergänzt Chen et al. (ICML 2026) |
| 2024-08 | LUT Tensor Core (Microsoft): Bit-seriell + Tabellensymmetrie für Mixed-Precision-GEMM | arXiv 2408.06003 | >10x Dichte/Energieeffizienz laut Zweitquelle, bis 18,13x Leistungssenkung (Aggregator) | Energiemodell für 2 Bit: LUT statt MAC (vgl. FIGLUT); nicht 2026 |

Lücken Lauf 11: weiterhin keine pJ/MAC-Zahl für INT2 (nur relative TOPS/W); keine Gradientenquantisierung <4 Bit im Kleinmodell-Vortraining; kein Skalengesetz speziell für Modelle <1B.

## 30.10. Lauf 12 (Cloud-Späher, 4 WebSearch, 00:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: trainierte Walsh-Hadamard-/Butterfly-Gewichte, pJ/MAC INT2 vs INT4, Skalengesetze <1B, Tensor-Train + Quantisierung. Nicht aufgenommen (schon da): Chen et al. ICML 2026 (nur Nachtrag: arXiv-ID 2505.14302, 74M-595M Param., 10-100B Tokens, FC2-Aktivierungsfehler als W4A4-Engpass), CompactifAI, 2411.17691, Monarch.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-09 | Needle 2 (Cactus Compute): Hadamard-MLP statt dichtem FFN, feste Walsh-Hadamard-Transformation + 3 gelernte Diagonalen | github.com/cactus-compute/needle (nur Drittseiten instagit.com, Repo selbst nicht geprüft) | FFN-Parameter O(d²) -> O(n); Beispiel d=768, Padding auf 1024, ca. 590K -> ca. 3K | direkter Nachbar zur Hadamard-MLP-Linie (Fastfood-artig): Gegner/Zitat; Padding auf 2er-Potenz wie bei 2601.21069; Qualität nicht belegt |
| 2025-07 | Jack Unit: flächen-/energieeffiziente MAC-Einheit für mehrere Formate (INT4/INT8 u. a.) | arXiv 2507.04772 | 1,05-1,84x weniger Leistung; 1,32-5,41x Energieeffizienz auf Systemebene (relativ, nicht pJ) | Energiemodell: nur relative Faktoren; weiterhin keine INT2-pJ/MAC-Zahl |
| 2026-07 | Surviving Resource-constraint Compression: Capability Retention under Tensor-train Decomposition for Sub-billion Parameter LMs | Springer 10.1007/s11633-026-1653-x | keine im Snippet; trainingsfrei, TT vs. Pruning | einziger 2026-Treffer zu TT bei <1B (Größe wie unsere Läufe); Quantisierung dort unklar |
| 2021 / 2023 | MPO-Feintuning (Liu et al., ACL 2021); TensorGPT (TT für Embeddings) | aclanthology 2021.acl-long.418; arXiv 2307.00526 | MPO: -91 % Feintuning-Parameter (Selbstangabe) | Grundlagen zum Zitieren (nicht 2026) |

Lücken Lauf 12: kein arXiv-Paper zu trainierten Hadamard-/Butterfly-Gewichten in LMs 2026 (nur Needle 2, Drittquelle); weiterhin keine saubere INT2/INT4/INT8-pJ/MAC-Reihe auf einem Knoten (nur 45 nm: Int8 0,23 pJ vs Int32 3,2 pJ laut Survey-Snippet); kein 2026-Paper TT/MPO + Quantisierung; kein QAT-Skalengesetz <1B über Chen et al. hinaus.

## 30.10. Lauf 13 (Cloud-Späher, 4 WebSearch, 01:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Seed-/PRNG-Gewichte, Sparsity + Quantisierung, quanten-inspirierte Tensornetze, gelernte Rotation W4A4. Nicht aufgenommen (schon da): SeedLM 2410.10714, CompactifAI 2401.14109 (Nachtrag: ESANN 2025 kombiniert TN mit FP16-Quantisierung, 88 %/93 % Speicherersparnis laut Snippet).

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-04 (ICML 2026) | ReSpinQuant: schichtweise Quantisierung mit Subraum-Residual-Rotation | arXiv 2604.11080 | W4A4/W3A3 SOTA ggü. globalen Rotationen; gelernte Rotationen bleiben nahe der Hadamard-Initialisierung | stützt Hadamard als gute Startrotation (Gegner zu "gelernt vs. fest"); Rotationskosten pro Schicht |
| 2026-04 | DuQuant++: MXFP4 W4A4 für LLaMA-3 | arXiv 2604.17789 | keine im Snippet; Argument: datenunabhängige Rotationen (auch lernbare) treffen Ausreißerkanäle nicht | Gegenargument zu fester Hadamard-Rotation; MXFP4-Gitter als Vergleich |
| 2026-06 | OffQ: Ausreißer als gruppenweise Offsets | arXiv 2606.07116 | PPL-Gewinne auf Llama/Qwen (Zahl fehlt); orthogonal zu Rotation kombinierbar | neue Idee: Offset statt Rotation, kombinierbar mit QIRE-Rotation |
| 2025-12 | OptRot: datenfreie Rotation gegen Gewichtsausreißer | arXiv 2512.24124 | schlechter bei W4A4 als bei Nur-Gewichten (Selbstangabe laut Snippet) | stützt Aktivierungsachse: Gewichtsrotation allein reicht für A4 nicht |
| 2026-05 | DiRotQ: PCA-/Rotations-bewusste W4A4 für Diffusion Transformer | arXiv 2605.16732 | hochpräziser Hochvarianz-Unterraum + Rest quantisiert; keine Zahl | nur Bildmodelle; Idee "kleiner Hochpräzisionsunterraum" ähnlich pQuant |
| 2025-01 | Proximal-Operator für 2:4-Sparsity | arXiv 2501.18015 | besser als Vorgänger bis 13B (nur Nachtraining) | Sparsity-Achse: nur PTQ, nicht 2026 |
| 2025 (PyTorch-Blog) | 4 Bit + 50 % Sparsity schlägt reines 2 Bit (LLaMA-2-7B) | pytorch.org/blog/when-quantization-isnt-enough-why-24-sparsity-matters | Genauigkeit höher bei gleichem Budget (Blogangabe) | Gegner im Byte-Budget bei ca. 2 Bit/Gewicht; Hardware-Kernel 2:4 existiert (cuSPARSELt) |

Lücken Lauf 13: kein 2026-Folgepaper zu SeedLM/PRNG-Gewichten; kein 2026-Paper zu 2:4-Sparsity + 4 Bit im Kleinmodell-Vortraining; kein 2026-Paper zu quanten-inspirierten Tensornetzen + Quantisierung (nur CompactifAI-Linie).

## 30.10. Lauf 14 (Cloud-Späher, 4 WebSearch, 02:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: nichtuniforme Gitter/Lloyd-Max, KV-Quantisierung ohne Rotation, Tensorring/quanten-inspiriert, W2A4/W2A2 von Grund auf. Nicht aufgenommen (schon da): Qift 2606.02823, RCP 2502.15779, OptR 2608.02691, RotateKV 2501.16383, CompactifAI 2401.14109, LeanQuant 2407.10032 (nur Hintergrund).

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-03 | PolarQuant (Lloyd-Max-Zentroiden auf Hadamard-rotierten Gewichtsblöcken, Abgrenzung zu NF4) | arXiv 2603.29078 | keine im Snippet | liefert die fehlende ID zum PolarQuant-Eintrag (Lauf 1, "ID fehlt"); Gegner für Hadamard+Lloyd-Arm |
| 2026-02 | BPDQ: Bit-Plane-Zerlegung auf variablem Gitter | arXiv 2602.04163 | 2 Bit: Qwen2.5-72B auf einer RTX 3090, GSM8K 83,85 % vs. 90,83 % (16 Bit) | Gitterachse (Z3): gruppenweises variables Gitter statt festem; Messlatte bei 2 Bit |
| 2026-08 | CubicQuant: parametrische nichtuniforme Codebücher, 1-8 Bit | arXiv 2608.06763 | Snippet ohne Inhalt | Gitterachse, Codebuch als Formel statt Tabelle (verwandt mit Proteus); vor Zitat Inhalt prüfen |
| 2026-04 | RUQuant: Verfeinerung gleichmäßiger Quantisierung | alphaxiv.org/abs/2604.04013 | keine im Snippet | Gegner "uniform verbessern statt Gitter ändern"; Inhalt unklar |
| 2026-09 | D-Quant: Entropiekodierung des KV-Caches (driftfähig) | arXiv 2609.19880 | keine im Snippet; Hadamard davor, Rotation entfernt kanalweisen Key-Mittelwert nicht | neue Idee: Bits/Wert unter Bitbreite durch Entropiekodierung; Byte-Budget-Rechnung BK; stützt Mittelwertzentrierung der Keys (wie OptR) |
| 2026 (ACL 2026) | VecInfer: Glättung + Hadamard + Vektorquantisierung für Low-Bit-Keys | aclanthology.org/2026.acl-long.1454 (aus Snippet, ID/Titel unsicher) | keine | KV mit VQ statt Skalar; Gegner zu Skalar+Rotation |
| 2025 (NeurIPS 2025) | PolarQuant (Key-Cache, Polarkoordinaten statt Rotation) | neurips.cc/virtual/2025/poster/118745 | keine | Achtung Namensgleichheit mit PolarQuant 2603.29078; KV-Quantisierung ohne Rotation |
| 2024-07 | Exploring Quantization for Efficient Pre-Training of Transformer LMs | arXiv 2407.11722 | GPT-2 small (124M) von Grund auf auf OpenWebText | einzige gefundene Von-Grund-auf-Studie in unserer Größenordnung, nicht 2026, nur 8 Bit und höher laut Snippet |

Lücken Lauf 14: weiterhin kein W2A2- oder W2A4-Vortraining von Grund auf (nur PTQ/QAT auf fertigen Modellen); kein 2026-Paper zu Tensorring oder quanten-inspirierten Netzen mit Quantisierung (nur CompactifAI); Qift-Zahlen (LLaMA-2-7B W2A4/KV4: Lloyd-Max 9,265, NF2 9,404, symmetrisch 12,118) nur aus Zweitquelle emergentmind.

## 30.10. Lauf 15 (Cloud-Späher, 4 WebSearch, 03:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: QUBO/Ising-Quantisierung (quanten-inspirierte Optimierung), Tensornetz-Kompression, Energie pro Token, 4-Bit-Attention/Softmax. Nicht aufgenommen (schon da): CompactifAI 2401.14109, PicoGPT 2603.28534, 2508.06753.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-03 | Layer-wise QUBO-Based Training of CNN Classifiers for Quantum Annealing | arXiv 2603.02958 | 5-Bit-Updates zu grob (33 % Test-Genauigkeit), ab 10 Bit konkurrenzfähig; nur simulierte Annealer | quanten-inspirierte Optimierung diskreter Gewichte; Simulation, kein Hardwarebeleg; Abgrenzung (I-038) |
| 2026-01 | QUBO-Training und Regularisierung binärer neuronaler Netze | arXiv 2601.00449 | Pfad für Ising-Maschinen, kein bewiesenes Ergebnis | Zitat für "quanten-inspiriert" bei binären Gewichten; Variablenzahl wächst mit Netzgröße |
| 2025-10 | KARIPAP: iPEPS/TRG-Kompression von LLMs (2D-Tensornetze) | arXiv 2510.21844 | LLaMA-2 7B, keine Zahl im Snippet | 2D- statt MPO-Tensornetz; Gegner/Zitat zur MPO-Linie (nicht 2026) |
| 2026-08 | FluxBin: LUT-basierte Ultra-Niedrigbit-Inferenz (Algorithmus-Kernel-Synergie) | arXiv 2608.15602 | LLaMA-2-13B bei Batch 1: 579 J vs. 5908 J (FP16), ca. 10x | Energiemessung in Joule für Niedrigbit; schließt Teil der pJ/Op-Lücke (Systemwert, Hardware unklar) |
| 2025-04 | LLMPi: Llama 1B auf Raspberry Pi | arXiv 2504.02118 | 2 Bit: 5,18 Tokens/J, ca. 4,3x ggü. FP16; BitNet 1,58 Bit 4,66 Tokens/J | Edge-Energie bei 2 Bit direkt gemessen; nicht 2026 |
| 2025-04 | Sustainable LLM Inference for Edge AI (Quantisierung, Energie, Genauigkeit) | arXiv 2504.03360 | Qwen2.5 0,5B: 2,61 J/Token | Größenordnung für Kleinmodell-Energie; nicht 2026 |
| 2026-05 | Position: Energie pro Token berichten (Joules/Token neben Genauigkeit) | arXiv 2605.11733 (Zweitquelle emergentmind); verwandt 2603.20224 | Forderung, keine Messung | Rechtfertigt Energiespalte neben Byte-Budget |
| 2026-03 (ICML 2026) | Attn-QAT: 4-Bit-Attention mit QAT | arXiv 2603.00040 | naives FP4-QAT instabil; bis 1,5x Speed (RTX 5090); Diffusion + LM | Aktivierungsachse für Attention (QK, PV); Rotation nicht Teil |
| 2026-09 | EFQ-Softmax: Niedrigbit-Wahrscheinlichkeiten in einem Schritt | arXiv 2609.09721 | Qwen3-8B, MXFP4: ca. 40 % weniger Latenz Vektorstufe | Softmax als eigene Quantisierungsstelle; nur 8B |
| 2026-09 | Hardware-Aware FP4 FlashAttention-4 | arXiv 2609.04105 | bis 2,13x Vorwärts-Durchsatz; alle MXFP4-P/V-Trainingsläufe divergierten | stützt: Attention-Wahrscheinlichkeiten sind der harte Teil des Niedrigbit-Trainings; Einzelautor-Bericht |
| 2026-07 | HiFA4: trainingsfreie 4-Bit-FlashAttention (Ascend NPU) | arXiv 2607.04302 | Softmax-Zustand bleibt FP16 | nur Inferenz, andere Hardware; Abgrenzung |

Lücken Lauf 15: kein 2026-Paper, das Bitbreite je Schicht als QUBO formuliert (nur Gewichtsrundung und Training); keine MPS-Attention (nur Gewichtsmatrizen komprimiert); weiterhin kein Vergleich 2 vs. 4 Bit in Joule/Token auf gleicher Hardware; kein 4-Bit-Attention-Training für Kleinmodelle.

## 30.10. Lauf 16 (Cloud-Späher, 4 WebSearch, 04:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Sparsity+Quantisierung, SSM/Mamba niedrigbit, Distillation in Niedrigbit-Schüler, Skalengesetze. Nicht aufgenommen (schon da): Sparse-BitNet 2603.05168, SLiM 2410.09615, ParetoQ 2502.02631, 2505.14302. Kein 2026-Treffer in diesen Suchen; alle Funde älter.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2024-12 | GQSA: Gruppen-Quantisierung + Gruppen-Sparsity | arXiv 2412.17560 | besser als 2:4-Pruning und W2 in Genauigkeit und Tempo | Alternative zu festem 50-%-Muster; gleiche Byte-Budget-Frage wie QIRE (Sparsity statt 2. Bit) |
| 2024-06 | SDQ: strukturierte Sparsity + Quantisierung mit Ausreißerbehandlung | arXiv 2406.13868 | unter ca. 1 % Qualitätsverlust | Ausreißer als Teil des Budgets; Gegner für Sparsity-Arm |
| 2024-11 | Bi-Mamba: 1-Bit-Zustandsraummodelle | arXiv 2411.11843 | 1 Bit, PPL ähnlich GPTQ-8bit (laut Snippet) | SSM-Niedrigbit-Referenz; QIRE-Rotation dort nicht getestet |
| 2025 (COLING) | Slender-Mamba: Mamba-2 (170M) komplett ternär, von Grund auf | aclanthology 2025.coling-main.316 | keine Zahl im Snippet | Kleinmodell-Größe wie unser Regime; ternäres Training von Grund auf |
| 2025-03 | ViM-VQ: Vektorquantisierung für Visual Mamba | arXiv 2503.09509 | Ausreißer in Mamba-Blockgewichten als Hauptfehler | stützt Rotation gegen Ausreißer auch bei SSM (nur Vision) |
| 2025-03 | Q-MambaIR: quantisiertes Mamba, QAT bis 2 Bit | arXiv 2503.21970 | SSM bei 2–4 Bit stark verlustbehaftet wegen Ausreißern | wie oben; Bildrestauration, nicht Sprache |
| 2024-06 | TernaryLLM: ternäre Gewichte + Distillation | arXiv 2406.07177 | schlägt PTQ- und QAT-Vorgänger | Distillation als Standardzutat im Niedrigbit-QAT; Vergleichsbasis |
| 2023-08 | Token-Scaled Logit Distillation (QAT für ternär) | arXiv 2308.06744 | unter 1,0 PPL-Verlust | Distillationsverlust für Niedrigbit-QAT, billig nachbaubar |
| 2025-10 | TernaryCLIP: ternär + Distillation für Bild-Text | arXiv 2510.21879 | keine Zahl im Snippet | nur Zitat; andere Domäne |

Lücken Lauf 16: weiterhin keine 2026-Arbeit zu Rotation+SSM bei 2 Bit; keine 2026-Skalengesetze mit Aktivierungsbits; Sparsity+2 Bit bei Kleinmodellen unter 100M nicht gefunden.

## 30.10. Lauf 17 (Cloud-Späher, 4 WebSearch, 05:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Skalengesetze mit Gewichts- und Aktivierungsbits, quanten-inspirierte Tensornetz-/MPS-Sprachmodelle, 2-Bit-Training von Grund auf, gelernte Rotationen (Cayley/Givens). Nicht aufgenommen (schon da): ReSpinQuant 2604.11080, pQuant 2602.22592, Chen et al. 2505.14302, PicoGPT 2603.28534, 2601.06788, BitNet v2 (w8/w11), ButterflyQuant, RUQuant.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| (Jahr unklar, wohl 2025) | Compression Scaling Laws: Unifying Sparsity and Quantization (Frantar et al.) | ID fehlt; nur themoonlight.io-Review | Voll-Quantisierung (Gewicht+Aktivierung) mit abnehmendem Ertrag unter 4 Bit; Nur-Gewicht bleibt parametereffizienter; 50 % Sparsity ≈ Parameter-Multiplikator wie 8 Bit | EIN Gesetz für Sparsity und Bits im Vortraining: Messlatte für Byte-Budget und Sparsity-Achse; schließt Teil der Lücke "Skalengesetz mit Aktivierungsbits"; vor Zitat Original suchen |
| 2026-08 (ID-Datum ungeprüft) | Transforms for LLM Quantization: The Great Inversion and Format Co-Design | arXiv 2608.25188 | keine im Snippet; Überblick über gelernte Transformationen | Überblick zu Rotation x Zahlenformat; Quelle für ReSpinQuant/ParoQuant; Inhalt lesen, bevor zitiert |
| 2026 | ParoQuant: gelernte Rotation als Produkt paarweiser Givens-Rotationen plus Kanalskalen, für Reasoning-Modelle | ID fehlt (nur über 2608.25188 beschrieben) | keine | Butterfly-/Givens-Struktur wie unsere FWHT-Linie, aber gelernt; Gegner zu fester Hadamard |
| 2025-11 (AAAI 2026) | SingleQuant: Quantisierung in einem Durchgang mit festen Givens-Rotationen | arXiv 2511.22316 | Kritik: Cayley-SGD mit STE konvergiert unter Quantisierung schlecht | stützt "feste statt gelernte Rotation" (billig); Argument gegen SpinQuant-artiges Training |
| 2026-01 | LiQSS: quanten-inspiriertes Zustandsraum-Tensornetz statt Selbstaufmerksamkeit (6G-Netze) | arXiv-ID unsicher (2601.12375 laut Suchtreffer) | 10,8-15,8x kleiner, ca. 1,4x schneller ggü. strukturierten SSM-Basismodellen | einziger 2026-Treffer "quanten-inspiriert" + Architektur; Domäne Funknetze, nicht Sprache; Zitat für Abgrenzung |

Lücken Lauf 17: kein 2026-MPS-Sprachmodell (nur van der Poel/Zhao 2403.12969, 2024); kein 2-Bit-Vortraining von Grund auf mit Aktivierungsbits in Kleinmodellen (nur pQuant, BitNet v2 mit ternären Gewichten, 1,58-Bit-Studie 100K-48M Parameter laut Snippet); kein Skalengesetz mit getrennten Bits für Gewicht und Aktivierung außer Frantar (Original nicht gefunden).

## 30.10. Lauf 18 (Cloud-Späher, 4 WebSearch, 06:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Original zu Compression Scaling Laws, KV-Eviction + Quantisierung, Microscaling-Formate im Training, feste Hadamard-Transformation als Ersatz dichter Schichten. Nicht aufgenommen (schon da): Needle 2 (Lauf 12), D-Quant, OptR.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2025-02 | Compression Scaling Laws: Unifying Sparsity and Quantization (Frantar, Evci, Park, Houlsby, Alistarh) | arXiv 2502.16440 | Llama-Stil auf C4; Gewicht+Aktivierung unter 4 Bit abnehmender Ertrag; 50 % Sparsity ≈ 8-Bit-Multiplikator (Zweitquelle themoonlight) | liefert die fehlende ID zu Lauf 17 (Original gefunden, Zahlen noch gegen das Paper prüfen); Messlatte Byte-Budget/Sparsity |
| 2026-03 | Fester Walsh-Hadamard statt dichter Attention-Ausgabeprojektion + leichte lernbare Affin-Skalierung | arXiv 2603.08343 | ca. 25 % weniger Attention-Parameter pro Block; „kein nennenswerter Genauigkeitsverlust“ (Zweitquelle papers.lunadong.com) | direkter Nachbar der Hadamard-MLP-Linie, jetzt mit arXiv-Paper (Lauf 12 hatte nur Needle 2); Gegner/Zitat; Lücke aus Lauf 12 teilweise geschlossen |
| 2026-05 | RDKV: Rate-Distortion-Bitzuteilung für gemeinsame Eviction + Quantisierung des KV-Caches | arXiv 2605.08317 | keine im Snippet; Token-Verwerfen = niedrigste Präzisionsstufe (Reverse Water-Filling) | BK: Tokens und Bits als ein Byte-Budget; passt zu „Not All Bits Are Equal“ (Lauf 2) |
| 2026-05 | ReST-KV: Eviction mit schichtweiser Ausgaberekonstruktion | arXiv 2605.08840 | hält Genauigkeit bei 6,25 % Gesamtkompression mit 4-Bit (Selbstangabe); Eviction schlägt Nur-Quantisierung | Gegenposition: Eviction stärker als Bits; Gegner zur reinen Bitbreiten-Achse im KV |
| 2026-03 | ARKV: Token je Original/quantisiert/verworfen unter hartem Speicherbudget | arXiv 2603.08727 | ca. 97 % Genauigkeit bei 4x Speicherersparnis (Selbstangabe) | dreistufige Budgetzuteilung im KV, Messlatte |
| 2026 (ACL-F 2026) | HqeKV: Hybrid aus Quantisierung und Eviction | aclanthology 2026.findings-acl.201 | keine im Snippet | Zitat für Hybrid-KV |
| 2025-06 | An Empirical Study of Microscaling Formats for Low-Precision LLM Training (Meta/UMass) | arXiv 2506.08027 | OCP-MX: Gruppe 32, Skala E8M0; maßgeschneiderte MX-Konfiguration erreicht FP8/BF16-Verlust (7B-Llama, Zweitquelle) | Gruppengrößen-/Skalenformat-Achse im Training (TZ); Skalen-Overhead 8/32 = 0,25 Bit pro Wert für Byte-Budget; nicht 2026 |

Lücken Lauf 18: kein 2026-Paper zu Skalenformat (E8M0 vs. E4M3) und Blockgröße im Kleinmodell-Training; Hadamard-Attention-Paper 2603.08343 noch nicht gegen Original geprüft (Qualität unbelegt); keine Studie, die Eviction + 2-Bit-KV + Rotation zusammen misst.

## 30.10. Lauf 19 (Cloud-Späher, 4 WebSearch, 07:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: unitäre/orthogonale Gewichte (quanten-inspiriert), Embedding/LM-Head-Quantisierung, Skalengesetze <1B, quanten-inspirierte Attention/Tensornetze. Nicht aufgenommen (schon da): Chen et al. 2505.14302, ParetoQ, 2411.17691, Kumar 2411.04330, KARIPAP 2510.21844. Nachtrag: LiQSS-ID (Lauf 17) laut Suchtreffer arXiv 2601.12375 bestätigt.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-07 | VBQ: Variable Bit-width Quantization, Bitbreite {1,2,4,8} je 64er-Gruppe beim Training von Grund auf | chatpaper.com/chatpaper/paper/307030 (keine arXiv-ID, nur Aggregator) | LM-Head im Mittel ca. 1,09 Bit, erster MLP-Block ca. 2,5 Bit; 131M-Modell bei 1,82 Bit Mittel schlägt 55M-FP16 auf TinyStories-PPL; Gewinn nur bei Training von Grund auf | Größe wie unser Regime; Gegner im Byte-Budget (gemischte Bits je Gruppe statt uniform 2 Bit); Embedding/Head braucht offenbar wenig Bits; Einzelquelle |
| 2026 (ACL-F 2026) | LieQ: schichtweise Informationswirksamkeit für PTQ kleiner Modelle (<8B) | preview.aclanthology.org/ingest-acl/2026.findings-acl.771/ | keine im Snippet | gemischte Bits je Schicht bei kleinen Modellen; nur PTQ, Zitat |
| 2025-11 | SLMQuant: Benchmark der Quantisierung kleiner Sprachmodelle für Edge | alphaxiv.org/abs/2511.13023 | keine im Snippet | Vergleichsbasis für kleine Modelle; Inhalt vor Zitat prüfen |
| 2026-08 | Q-Interference: speichereffiziente phasenbewusste quanten-inspirierte Attention | arXiv 2608.17288 | Snippet nur Literaturliste, Methode/Zahlen unbekannt | echter quanten-inspirierter Attention-Kandidat 2026; Inhalt lesen, bevor zitiert (I-038: Wirkung klassisch) |
| 2025-03 / 2025-02 | Quantum-PEFT (2503.05431, ICLR 2025); QuIC-Adapter (2502.06916) | arXiv 2503.05431, 2502.06916 | QuIC: 98 % relative Leistung ggü. OFT/BOFT mit 25x weniger Parametern (Drittliste) | orthogonale/unitäre Matrizen über Quantenschaltkreis-Einbettung parametrisiert: quanten-inspiriert, aber nur Feintuning; Nachbar zu Hadamard-/Butterfly-Linie; nicht 2026 |

Lücken Lauf 19: keine 2026-Arbeit zu orthogonalen/unitären Gewichten im Vortraining von Sprachmodellen (Suchvorschlag: "Stiefel manifold LLM pretraining", "unitary RNN"); kein Skalengesetz, das <1B isoliert; Embedding-Quantisierung nur über VBQ (Aggregator).

## 30.10. Lauf 20 (Cloud-Späher, 4 WebSearch, 08:30 UTC, nur Snippets, alle Zahlen UNGEPRÜFT)
Themen: Stiefel/orthogonale Gewichte im LM-Vortraining (Lücke Lauf 19), Hadamard + Mamba/SSM niedrigbit, Aktivierungssparsity + Quantisierung, Skalenformat/Blockgröße im Kleinmodell-FP4-Training. Nicht aufgenommen (schon da): SharQ 2606.26587, Prox 2607.27591, NVFP4-Pretraining 2509.25149, 2601.21069.

| Datum | Titel | ID/Link | Kernzahl (ungeprüft) | Bezug zu QIRE |
|---|---|---|---|---|
| 2026-06 | Manifold Muon für GPT-2-Vortraining: Stiefel für Attention, DGram für MLP statt einheitlicher Nebenbedingung | arXiv 2606.13276 (via emergentmind) | DGram auf Attention: explosives Wachstum, instabil; Maßstab/Basislinien unbekannt | schließt Lücke Lauf 19 (orthogonale Gewichte im LM-Vortraining); Unitaritäts-Achse, Modulwahl wichtig |
| 2026-07 | ManifoldFlow: gelernte Singulärwertspektren statt fester Stiefel-Schichten | arXiv 2607.04535 (via emergentmind) | größte Gewinne bei rekurrenten LMs (Selbstangabe) | gelockerte Orthogonalität; Gegner/Ergänzung zu strikt unitären Schichten |
| 2025-08 (EMNLP-F 2025) | Riemannian Optimization for LoRA on the Stiefel Manifold | arXiv 2508.17901 | schlägt AdamW über Benchmarks (Selbstangabe) | nur Feintuning; Zitat für Stiefel-Optimierung |
| 2025-01 (ICLR 2025) | MambaQuant: varianzangeglichene Rotation (KLT-Rotation, Smooth-Fused) für Mamba | arXiv 2501.13484 | W8A8 unter 1 % Verlust; W4A8 (Vim): Smooth-Fused +11,7 % ggü. Online-Hadamard | Befund: reine Hadamard bei Mamba ungleiche Kanalvarianz, bei 4 Bit schwach; Gegner für Rotation+SSM (Lücke Lauf 16) |
| 2024-10 | Quamba: 8-Bit-PTQ für selektive SSMs, Hadamard auf Ausgabeaktivierungen | arXiv 2410.13229 | 1,72x Latenz (Mamba 2.8B, 8 Bit) | SSM + Hadamard als Praxisfall; nur 8 Bit |
| 2025-07 (v2 2026-01) | LaRoSA: trainingsfreie Aktivierungssparsity mit schichtweiser Rotation + konsistentem Top-K | arXiv 2507.01299 | keine im Snippet | Rotation x Sparsity der Aktivierungen; Nachbar zu SharQ |
| 2025-05 | FP4 All the Way: vollständig quantisiertes Training (350M Llama-Stil) | arXiv 2505.19115 | Skalenformat E4M3/E3M4 beste, E1M6 divergiert; Blockgröße kleinerer Effekt als Skalenformat | liefert Teil der Lücke Lauf 18 (Skalenformat im Kleinmodell); Achse Skala vor Blockgröße; nicht 2026 |

Lücken Lauf 20: kein 2026-Paper zu Hadamard/Rotation für Mamba/SSM bei ≤4 Bit (nur MambaQuant 2025); Skalenformat-Vergleich im Kleinmodell nur 350M (2025), nichts unter 100M; Manifold-Muon-Arbeit nur über Aggregator, Maßstab ungeprüft.
