# Z1 Literatur-Frontier 2025–2026 (z1l, 28.10.2026)
Methode: LOOP.md, notizen.md (nur §1 und Stichproben: bitnet158, compactifai2024, quarot2024, spinquant2024, tensorgpt2023 schon erfasst), f2_bericht.md gelesen. 12 WebSearch. Nicht wiederholt: QuIP#, AQLM, SeedLM-Grundidee, KIVI, TurboQuant-Grundidee, 8-Bit-Adam, Cheung-PSP (siehe f2).
Zahlen: aus Suchtreffern (Abstract/Sekundärseiten), kein Volltext. "(G)" = Gedächtnis, ungeprüft. Rekordangaben sind daher "Stand der Treffer", keine geprüfte Bestenliste.

## Querbefund (kurz)
- Gemeinsamer Mechanismus der Spitze in 1, 2, 5: **Hadamard/zufällige Rotation + Skalar- oder Gitter-Quantisierer**. Die Rotation ist bei allen schon drin. QIRE-Hadamard ist kein Unterscheidungsmerkmal.
- Das, was in der Spitze wirklich Qualität bringt, ist fast immer **Fehlerkompensation/Feintuning mit Daten** (GPTQ-artig, PV-Tuning, Healing). Datenfreie Arme (RTN-Klasse) liegen systematisch dahinter.
- Ehrlichkeit: nichts hier ist Quantenvorteil; alles klassisch (I-038).

## (1) Bits je Gewicht
- **Bestes Verfahren (2 Bit, PTQ):** QTIP (Tseng et al., NeurIPS 2024, v4 Jun 2025): Trellis-codierte Quantisierung (TCQ) + Inkohärenz-Rotation, sehr hohe Quantisierungsdimension (Treffer: 2-Bit-Llama-2-7B Wiki-PPL 6,89 ohne Feintuning mit 1D-Codebuch; Behauptung "2-Bit-Modelle skalieren besser als theoretisch optimale 4-Bit"). Mehr als 2x schneller als AQLM beim Dekodieren. Unabhängige Nachbildung 2026 (Bielik-Q2-Sharp, 2603.04162, 11B): QTIP-Variante geht, aber Müll-Token am Satzanfang, 12 h A100.
- **1–2 Bit mit Feintuning:** PV-Tuning (Malinovskii et al., NeurIPS 2024): ersetzt Straight-Through, beste Llama-2-Zahlen bei 1–2 Bit, "erstes Pareto-optimales 2 Bit" (Codebuch 14–16 Bit je 8–16 Gewichte). VPTQ (Liu et al., 2024): Vektor-PTQ mit Hessian-Gewichtung, Llama-2/3, Mistral. Zahlen nicht aus Treffern.
- **~1 Bit PTQ:** BiLLM (~1,08–1,11 Bit; RTN/GPTQ brechen bei 1 Bit auf PPL in Tausend), OneBit (Vorzeichenmatrix + zwei fp16-Vektoren, braucht QAT). Dezember 2025: datenbewusste 1-Bit-PTQ (2512.21651, Treffer ohne Zahlen).
- **Gegenposition:** ICQuant (2505.00850): Index-Codierung der Ausreißer, simpler Skalar-Quantisierer ohne Feintuning soll VQ-Baselines (AQLM, QuIP#, QTIP) erreichen. Wichtige Warnung: das VQ-Buch ist vielleicht nicht der Hebel.
- **Mechanismus:** (a) Rotation macht Gewichte gaußähnlich (inkohärent); (b) bei Gauß ist Gitter/Trellis-Codierung dem Skalar bis ~1,53 dB (0,25 Bit) überlegen, TCQ erreicht das in hoher Dimension mit linearem Dekodieraufwand; (c) Fehlerkompensation entlang der Hessian.
- **QIRE auf CPU/Raven nachbaubar:** Rotation (Hadamard, billig) + Trellis-Dekodierer ist reine Tabelle/Rechenschieber, CPU-tauglich (QTIP "Bitshift"-Code braucht nur Shifts/Lookups). Machbar im Kleinen: TCQ (Viterbi, 2^L Zustände, L~8–12) auf rotierten MNIST-/Kleinmodell-Gewichten gegen skalar4_had, E8, k-Means-Buch (schließt K03-Lücken aus f2 und ergänzt TCQ als neuen Arm). GPTQ-Kompensation ist 50 Zeilen numpy und gehört in jeden Arm.
- **Fehlende quanteninspirierte Zutat:** Hadamard ist da. Fehlen: (i) **Walsh-Spektrum-Bitzuteilung** (Reverse-Water-Filling in Walsh-Basis, K06 testet das), (ii) **Phasen/Unitarität als Codebuch:** Trellis ist klassisch; ein Codebuch aus Einheitsvektoren auf der Sphäre (Winkelquantisierung wie PolarQuant) wäre die Born/Unitär-Brücke, nirgends gefunden, aber nicht geprüft.

## (2) KV-Cache
- **Bestes Verfahren (Stand Treffer):** TurboQuant (Google, ICLR 2026): PolarQuant (Polarkoordinaten, keine Normierungskonstanten) + QJL (1-Bit-Restkorrektur für unverzerrte Skalarprodukte). Berichtet: 3 Bit je Wert ohne messbaren Verlust (Needle, LongBench; Gemma, Mistral), >=6x Speicher, 4-Bit-Variante bis 8x schneller bei Attention-Logits (H100). Alles Sekundärquellen/Google-Blog, vor Zitat Volltext prüfen.
- **Weitere:** AQUA-KV (Adaptive KV-Quantisierung, 2501.19392): nahezu verlustfrei bei 2–2,5 Bit, <1 % relativer PPL-Fehler (Llama 3.2). CommVQ (2025): 2 Bit, 87,5 % weniger Speicher, 1-Bit "mit kleinem Verlust" (Commutative VQ, Codebuch kommutiert mit RoPE). CQ Coupled Quantization (2405.03917): ~1 Bit je Kanal durch gemeinsame Quantisierung gekoppelter Kanäle. RotateKV (Rotation, 2 Bit). LogQuant (2 Bit, log-verteilt). Neu 02.10.2026: "Tailoring the Quantization Space for 1-Bit KV Cache" (nur Aggregator, unverifiziert).
- **Mechanismus:** Rotation beseitigt Kanal-Ausreißer der Keys; Kopplung/VQ nutzt Korrelation zwischen Kanälen; unverzerrter 1-Bit-Rest hält die Attention-Scores erwartungstreu.
- **Nachbaubar (CPU):** alles; Rotation + Skalar ist in numpy trivial, QJL-Rest = Vorzeichen einer zufälligen Projektion. Passt zu K05. Vorschlag: Arme TurboQuant-artig (had + 1-Bit-QJL-Rest) und CQ (Paare/Vierer-Kopplung mit k-Means) neben KIVI.
- **Fehlt quanteninspiriert:** QJL ist schon eine "Messung" (Vorzeichen zufälliger Projektion = Einbit-Schuss), die Verbindung Schussrauschen/Born-Auslese zu Attention-Scores ist die natürliche QIRE-Brücke: **Attention-Score als geschätzte Überlappung <q|k> aus Schüssen**, Varianz je Bit gegen QJL messen. Neuheit dort ungeprüft.

## (3) Quanteninspirierte Tensornetz-Kompression
- **Bestes (und fast einziges großes):** CompactifAI (Tomut et al., Multiverse, 2401.14109): MPO-Zerlegung (sequentielle SVD, Bonddimension chi) von Attention- und MLP-Gewichten, danach ein Epoch "Healing"-Feintuning. LLaMA-2 7B: ~93 % weniger Speicher (zusammen mit Quantisierung), ~70 % weniger Parameter, Genauigkeit -2 bis -3 %. Tiefere Schichten besser komprimierbar. Eigenangaben der Firma, keine unabhängige Replikation gefunden. Wichtig: die 93 % sind Quantisierung + MPO zusammen; der reine MPO-Anteil ist ~70 % Parameter.
- **Weitere:** 2603.28534 (Mar 2026): jede Linear-Schicht eines 1M-Zeichenmodells (PicoGPT) als MPO, Init per TT-SVD oder zufällig, Autograd: Kleinstbeleg für **MPO-Training von Anfang an**. 2410.17397 (Tensor-Netz-Disentangler, Quanten-LLM). MPO-Feintuning mit festen Zentraltensoren (Liu et al., ACL 2021, G).
- **Mechanismus:** Niedrigrang-Struktur über mehrere Moden (Kronecker/Tensortrain) statt eines Rangs; chi regelt Verschränkung.
- **Nachbaubar:** TT-SVD + Healing ist reines numpy/torch, auf CPU für MNIST-MLP bis Kleintransformer. Pflicht-Gegenspieler (aus Querbefund f2): gleiche Parameter mit **Niedrigrang-SVD**, KronA/BOFT (N-BREIT10). Ohne SVD-Rang-Arm ist ein MPO-Gewinn nicht interpretierbar.
- **Fehlt:** Unitäre/isometrische Kerne (Kanonische Form des MPS, "orthogonales MPO" mit Stiefel-Kernen) werden in LLM-Kompression kaum genutzt (nicht gefunden, Suche dünn). Hadamard-Faktor in den Kernen (HadaCore-artige Struktur als fester Kern + gelernte Diagonale = Fastfood/Monarch, ist bekannt, siehe N-STRUKTUR).

## (4) Seed-/Zufallsbasis
- **SeedLM** (Shafipour et al., 2410.10714, ICLR 2025): Block aus PRNG-Seed (LFSR) + wenige Koeffizienten; datenfrei; 3–4 Bit ~ fp16 (Llama-3 70B); FPGA bis ~4x schneller (Speicherbandbreite gegen Rechnen getauscht). Bei 2 Bit nach f2-Lage kein Spitzenwert (QTIP/AQLM schlagen es; (G) SeedLM-Tabelle Llama-2-70B 3-Bit nahe fp16).
- **PRANC** (Nooralinejad et al., 2022/23): ganzes Netz = Linearkombination vieler zufälliger Netze (Seed + Koeffizienten), Kompression um Größenordnungen für Speicherung. **NOLA** (Koohpayegani et al., ICLR 2024): LoRA-Matrizen A,B als Linearkombination fester zufälliger Basen; Parameterzahl von Rang und Form entkoppelt; Llama-2-70B ~20x kompakter als das kleinste LoRA ohne Verlust. **VeRA** (Kopiczko et al., ICLR 2024): ein gemeinsames zufälliges Paar A,B, nur Skalenvektoren trainiert. **RandLoRA** (2502.00987, ICLR 2025): Vollrang-Update aus Zufallsbasen, behebt den Rangdeckel von VeRA/NOLA (Treffer: LoRA-Niveau bei weniger Parametern, Details nicht gelesen).
- **Mechanismus:** Zufallsmatrizen kosten 0 Byte (Seed), nur Koeffizienten werden gespeichert; Qualität hängt an der Dimension des Zufallsraums (Johnson-Lindenstrauss).
- **Nachbaubar:** NOLA/VeRA/RandLoRA auf CPU im Kleinen sofort (Adapter-Zweig I-045). Vorsicht: sie sparen Trainingsparameter/Speicher, nicht FLOPs (K04-Befund: FLOPs verdoppeln sich).
- **Fehlt quanteninspiriert:** Die Zufallsbasis ist durchweg gaußsch/Rademacher. **Strukturierte Basis (Walsh/Hadamard-Zeilen mit Diagonale) als Seed-Basis** (Dekodierkosten O(n log n), kein Speicher für die Basis, orthogonal -> keine Redundanz) wird in NOLA/VeRA nicht verwendet; in SeedLM ersetzt LFSR die Hadamard-Struktur. Das ist eine saubere QIRE-Lücke (Fastfood-Verwandtschaft, N-QUADRAT-ADAPTER nennt das Nachbar Needle 2 / Fastfood: also "neu kombiniert", nicht "neu").

## (5) Niedrigbit-Training
- **FP4:** NVFP4-Pretraining (NVIDIA, 2509.25149): 12B-Modell, 10 T Tokens, Verlust/Downstream nahe FP8-Baseline (längster öffentlich beschriebener 4-Bit-Lauf). Rezept: Mikroblock 16 mit E4M3-Skala, **zufällige Hadamard-Transformation d=16 nur auf Eingängen der Gradienten-GEMM (Wgrad)**, nicht auf Gewichten; 2D-Blockskalierung der Gewichte (16x16) für Vorwärts/Rückwärts-Konsistenz; stochastisches Runden nur für Gradienten; einige letzte/erste Schichten in hoher Präzision. **Quartet** (Castro et al., NeurIPS 2025, v4 Jan 2026, 2505.14669): vollständig FP4 für die Linear-GEMMs, Hadamard + QuEST-artiger Vorwärtsquantisierer, neues Präzisions-Skalengesetz; behauptet konkurrenzfähig gegen FP8/FP16 (Blackwell-Kernel).
- **Ternär:** BitNet b1.58 2B4T (Microsoft, 2504.12285): natives Ternär, 4 T Tokens; 0,4 GB Nicht-Embedding-Speicher gegen 1,4–4,8 GB; ebenbürtig mit Vollpräzisions-Modellen gleicher Größe (MMLU/HumanEval+ eher schwächer, GSM8K/IFEval stärker); bitnet.cpp CPU-Kernel (Jan 2026 +1,15–2,1x). Spectra (2407.12327): Ternär-Pretraining bei Größe, Binär klar schlechter.
- **Mechanismus:** Latente fp-Gewichte + STE; Hadamard bändigt Ausreißer im Gradienten; stochastisches Runden macht Gradienten erwartungstreu; Skalengesetz (Quartet) sagt voraus, wann sich tiefere Bitzahl rechnet.
- **Nachbaubar:** Simulation (Fake-Quant) von NVFP4/ternär auf CPU geht, aber **keine Zeitgewinne messbar** (kein FP4-Hardwarepfad). Messbar ist nur Verlust gegen Bits. Kleiner Verlust-gegen-Token-Test mit zwei Trainingslängen passt zu LOOP-Regel. Ternäre Inferenz auf Ryzen: bitnet.cpp existiert real (Tabellen-Lookup-Kernel), das ist die einzige Zeitmessung, die auf Maltes PC ehrlich möglich ist.
- **Fehlt quanteninspiriert:** Hadamard ist Stand der Technik (d=16, Wgrad). Offen: (i) **Phasen-/Unitär-Parametrisierung der latenten Gewichte** (Gewicht = orthogonale Matrix + Skala, Cayley/Givens): Quantisierung der Winkel statt der Einträge, Verbindung zu Clifford-Winkelgitter (N-B teilweise bekannt); (ii) Hadamard auch auf Gewichten würde laut NVIDIA die Konsistenz verschlechtern, also gerade dort ein ehrlicher Gegenbefund zur QIRE-Intuition.

## Kurzliste für Z1 (Top-Hebel, keine Läufe)
1. K03 erweitern: TCQ/Trellis-Arm + k-Means-Buch + GPTQ-Kompensation (schließt Hauptlücke in 1).
2. K05 erweitern: TurboQuant-artig (had + QJL-1-Bit-Rest) + CQ-Kopplung; Born/Schuss-Lesart des Attention-Scores als QIRE-Arm.
3. Walsh/Hadamard als **Seed-Basis** für NOLA/VeRA-Adapter (Zufallsbasis ohne Speicher, orthogonal).
4. Winkel-/Unitär-Parametrisierung mit Quantisierung der Winkel (PolarQuant-Verwandtschaft prüfen).
5. MPO/TT als Trainingsstruktur mit SVD-Rang-Gegenspieler.

## Grenzen
- 12 von 20 Suchen verbraucht; keine Primärtabellen gelesen. Sub-2-Bit-Rekordzahlen (VPTQ, PV-Tuning Llama-3) und Quartet/NVFP4-Zahlen nur qualitativ. 2026-Arbeiten zu 1-Bit-KV und 1-Bit-PTQ nur als Titelhinweis. Vor jedem Zitat Volltext oder Tabelle prüfen.

## BibTeX
@inproceedings{tseng2024qtip, title={QTIP: Quantization with Trellises and Incoherence Processing}, author={Tseng, Albert and Sun, Qingyao and Hou, David and De Sa, Christopher}, booktitle={NeurIPS}, year={2024}, note={arXiv:2406.11235 (v4 2025)}}
@inproceedings{malinovskii2024pvtuning, title={PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression}, author={Malinovskii, Vladimir and others}, booktitle={NeurIPS}, year={2024}, note={arXiv:2405.14852}}
@article{liu2024vptq, title={VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models}, author={Liu, Yifei and others}, year={2024}, note={arXiv:2409.17066}}
@article{huang2024billm, title={BiLLM: Pushing the Limit of Post-Training Quantization for LLMs}, author={Huang, Wei and others}, year={2024}, note={ICML 2024; Autoren/ID vor Zitat pruefen}}
@article{xu2024onebit, title={OneBit: Towards Extremely Low-bit Large Language Models}, author={Xu, Yuzhuang and others}, year={2024}, note={arXiv:2402.11295}}
@article{icquant2025, title={ICQuant: Index Coding enables Low-bit LLM Quantization}, author={unvollstaendig}, year={2025}, note={arXiv:2505.00850}}
@article{bielik2026q2sharp, title={Bielik-Q2-Sharp: A Comparative Study of Extreme 2-bit Quantization Methods for a Polish 11B Language Model}, author={unvollstaendig}, year={2026}, note={arXiv:2603.04162}}
@article{nagel2026ptq1bit, title={Data-aware PTQ fuer 1-Bit-LLMs}, author={unvollstaendig}, year={2025}, note={arXiv:2512.21651, Titel unverifiziert}}
@article{kvaqua2025, title={Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models (AQUA-KV)}, author={Shutova, Alina and others}, year={2025}, note={arXiv:2501.19392}}
@article{zhang2024cq, title={KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization}, author={Zhang, Tianyi and others}, year={2024}, note={arXiv:2405.03917}}
@article{commvq2025, title={Commutative Vector Quantization for KV Cache Compression (CommVQ)}, author={unvollstaendig}, year={2025}, note={Titel/ID unverifiziert}}
@article{polarquant2025, title={PolarQuant}, author={Han, Insu and others}, year={2025}, note={zu TurboQuant; ID unverifiziert}}
@inproceedings{zandieh2025qjl, title={QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead}, author={Zandieh, Amir and Daliri, Majid and Han, Insu}, year={2024}, note={AAAI 2025 (G)}}
@article{tomut2024compactifai, title={CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks}, author={Tomut, Andrei and others}, year={2024}, note={arXiv:2401.14109}}
@article{picogpt2026mpo, title={MPO-parametrisiertes PicoGPT}, author={unvollstaendig}, year={2026}, note={arXiv:2603.28534, Titel unverifiziert}}
@article{tensornetz2024disentangler, title={Quantum Large Language Models via Tensor Network Disentanglers}, author={unvollstaendig}, year={2024}, note={arXiv:2410.17397}}
@article{nooralinejad2022pranc, title={PRANC: Pseudo RAndom Networks for Compacting deep models}, author={Nooralinejad, Parsa and others}, year={2022}, note={arXiv:2206.08464 (G)}}
@inproceedings{koohpayegani2024nola, title={NOLA: Compressing LoRA using Linear Combination of Low-Rank Random Basis}, author={Koohpayegani, Soroush Abbasi and others}, booktitle={ICLR}, year={2024}, note={arXiv:2310.02556}}
@inproceedings{kopiczko2024vera, title={VeRA: Vector-based Random Matrix Adaptation}, author={Kopiczko, Dawid J. and Blankevoort, Tijmen and Asano, Yuki M.}, booktitle={ICLR}, year={2024}, note={arXiv:2310.11454 (G)}}
@inproceedings{albert2025randlora, title={RandLoRA: Full-rank parameter-efficient fine-tuning of large models}, author={Albert, Paul and others}, booktitle={ICLR}, year={2025}, note={arXiv:2502.00987}}
@article{nvidia2025nvfp4, title={Pretraining Large Language Models with NVFP4}, author={NVIDIA}, year={2025}, note={arXiv:2509.25149}}
@inproceedings{castro2025quartet, title={Quartet: Native FP4 Training Can Be Optimal for Large Language Models}, author={Castro, Roberto L. and others}, booktitle={NeurIPS}, year={2025}, note={arXiv:2505.14669}}
@article{wang2025bitnet2b4t, title={BitNet b1.58 2B4T Technical Report}, author={Ma, Shuming and others}, year={2025}, note={arXiv:2504.12285}}
@article{kaushal2024spectra, title={Spectra: Surprising Effectiveness of Pretraining Ternary Language Models at Scale}, author={Kaushal, Ayush and others}, year={2024}, note={arXiv:2407.12327}}
