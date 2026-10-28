# f2 — Killer-Recherche Top-10 (Welle 3, 28.10.2026)
Methode: 14 WebSearch, 10 Skripte K01–K12 (Kopfkommentar + Arme) gelesen. "(G)" = aus Gedächtnis, nicht per Suche belegt. Zahlen sonst aus Suchtreffern (Abstract/README), nicht im Volltext geprüft.
Grundregel für alle: QIRE-Rotation/Hadamard ist kein Neuland (QuIP#, QuaRot, TurboQuant). Beitrag kann nur sein: sauber gemessen, kleine Skala, gleiche Bytes UND gleiche FLOPs.

## Ampel-Übersicht
| ID | Ampel | Kern |
|---|---|---|
| K03 | umbauen | E8-Arm ist QuIP# ohne Feintuning; trainiertes Codebuch (AQLM-Stil) fehlt |
| K01 | umbauen | ALBERT = geteilt_o ist drin; gelernte Lagen-Skala/LoRA-Arm fehlt |
| K05 | umbauen | had-Arm = TurboQuant/QuaRot-Idee; KIVI-Arm fehlt |
| K18 | umbauen | Zwilling vs. n-Gramm-/Kopf-Entwurf fehlt; Zeitgewinn bei d=128 nicht übertragbar |
| K02 | streichen (als Haupt) | ist Cheung-2019-PSP fast 1:1; nur Reproduktion |
| K06 | umbauen | Bitzuteilung nach Varianz ist klassische Transformcodierung; orth_alloc-Kontrolle gut |
| K04 | umbauen | Zusatzarme (dicht1024) müssen Pflicht werden |
| K10 | streichen | Born-a² ist nicht die optimale Stichprobe; top-k schlägt sie wohl |
| K07 | umbauen | b8 = Dettmers drin, 4-Bit-Li fehlt; Rotation nur für m sinnvoll |
| K12 | streichen | 256 Klassen, Kopf-FLOPs klein; Hierarchie bekannt weniger genau |

## K03 Seed-Codebuch (Platz 1)
**Bekannt.** QuIP# (Tseng et al. 2024): Hadamard-Inkohärenz + E8-Gitter-Codebuch (8er-Blöcke, 2 Bit/Gewicht) = exakt der Plan von K03; E8P ist formelbasiert, speichert also kaum ein Buch. Llama-2-70B 2 Bit: Wiki-PPL ca. 3,91 (neue README) bzw. 4,16 (alte), fp16 3,12, OPTQ-2-Bit 109,8, QuIP 5,57. AQLM (Egiazarian 2024): gelernte Mehrfach-Codebücher, 2 Bit: Llama-2 7B/13B/70B PPL 6,93/5,70/3,94. SeedLM (Shafipour 2024, ICLR 2025): pro Block ein PRNG-Seed + wenige Koeffizienten, datenfrei, 3–4 Bit ≈ fp16 auf Llama-3-70B, FPGA bis ~4x. Der Gedanke "Gewichte aus Zufallsseed" ist also vergeben.
**Muss in den Test.** (1) Skalar 2 Bit echt 4-stufig mit Rotation (skalar4_had); (2) E8 (= QuIP#-ohne-Feintuning); (3) ein *gelerntes* Codebuch gleicher Größe (k-Means auf den rotierten Blöcken, 65 536 Wörter; Bytes des Buchs mitzählen, bei 512x512-Netz nicht vernachlässigbar!); (4) SeedLM-Arm (Seed + 3–4 Koeffizienten je Block). Zusatz: Fehlerkompensation (GPTQ-artig) – ohne sie sind alle Arme "RTN-Klasse".
**Im Skript?** Ja: skalar4_had, skalar2_had (ternär!), roh, zufall, rm, e8, hex (alle Gleiche Bytes, Zeilenskala/Bias mitgezählt). NEIN: k-Means-Buch, SeedLM, GPTQ. Vorbehalt: skalar2 ist ternär (1,58 Bit, gezählt 2).
**Ehrlicher Beitrag.** Bytes-genaue Messung kleiner MNIST-MLPs: ob zufällige/Gitter-Bücher ein gelerntes Buch bei 2 Bit einholen (0 Byte Buch vs. 128 KB Buch). Kein Neuheitsanspruch; "Vektorquantisierung mit Formel-Buch" = QuIP#.
**Ampel: umbauen** (k-Means-Arm + SeedLM-Arm ergänzen; Rahmung "Reproduktion + Byte-Buchhaltung").

## K01 Geteilte FFN mit Schlüsseln (Platz 2)
**Bekannt.** ALBERT (Lan 2019): Teilen über alle Lagen, ca. 18x weniger Parameter bei BERT-large-Konfiguration, ~1,7x schnelleres Training; Teilen der FFN kostet am meisten Genauigkeit, Teilen der Attention kaum (Blog-Zusammenfassung, nicht Paper). Universal Transformer (Dehghani 2018): gleiche Schicht rekursiv (G). Relaxed Recursive Transformers (Bae 2024, G): Teilen + lagenweise Niedrigrang-LoRA gleicht den Verlust aus. Cheung 2019 (PSP): Vorzeichen-Kontextschlüssel auf Gewichten.
**Muss in den Test.** geteilt_o (ALBERT), klein (gleiche Bytes), dazu: (a) geteilt + gelernte Lagen-Skala/Bias (Diagonale gelernt, 2d Zahlen je Lage – die billigste "Schlüssel"-Alternative); (b) geteilt + Niedrigrang-LoRA je Lage mit gleichen Zusatzbytes (Relaxed-Recursive). Ohne (a) bleibt offen, ob feste Zufallsvorzeichen irgendetwas bringen, was gelernte Diagonalen nicht auch bringen.
**Im Skript?** normal, klein, geteilt_o, geteilt_k, breit_o/k: ja. (a) und (b): nein.
**Beitrag.** Ob feste Zufalls-Schlüssel (0 Byte) einen Teil des Teilungsverlusts zurückholen (geteilt_k − geteilt_o). Realistisch klein; 4 Lagen sind flach, ALBERT-Effekte treten bei 12+ Lagen auf.
**Ampel: umbauen** (Arm a, b ergänzen; Lagenzahl 4 → 8 als Zusatz).

## K05 KV-Cache mit sehr wenigen Bit (Platz 2)
**Bekannt.** KIVI (Liu, ICML 2024): Keys pro Kanal, Values pro Token, 2 Bit, ohne Tuning; ca. 2,6x weniger Spitzenspeicher, bis 4x größere Batches, 2,35–3,47x Durchsatz. TurboQuant (Google, ICLR 2026): zufällige Rotation + Skalar, QJL-1-Bit-Rest für unverzerrte Skalarprodukte; Blogzahlen 3,5 Bit "neutral", 2,5 Bit leicht schlechter (Sekundärquellen, ungeprüft); llama.cpp-Fork mit Walsh-Hadamard: +4,6 % PPL bei 4,6x K-Cache. QuaRot/KVQuant (G): Hadamard bzw. Kanal-Ausreißer-Behandlung. Rotation fürs KV-Cache ist damit Stand der Technik.
**Muss in den Test.** KIVI-Arm (K pro Kanal in Gruppen über Tokens, V pro Token, fp16-Restfenster der letzten R Token) bei gleichen Bit/Wert; had+Restfenster; Rotation ohne Skala-Overhead-Vergleich (Skalen 2 B je Vektor bei Kopfdim 64 = +0,5 Bit, bei KIVI ähnlich – fair zählen).
**Im Skript?** Teilweise: had, roh, roh_kanal (feste Kanalskala aus Eichdaten – KIVI nur angenähert, keine Gruppen über Tokens, kein fp16-Fenster), pca, fenster, zufallfall, alter_*. NEIN: echtes KIVI, Kombination Fenster+Rotation, QJL-Rest.
**Beitrag.** Kleine Skala (d=128, Kopfdim 64): bei kleiner Kopfdim ist Rotation weniger nötig/mehr Overhead; Alterungs-Arme (alter_haar) sind der nicht-triviale Teil (ähnlich Token-Eviction/H2O (G), aber mit Haar-Grobwert).
**Ampel: umbauen** (KIVI-Arm + fp16-Fenster; Haar-Alterung als eigentliche Neuheitsprobe).

## K18 2-Bit-Zwilling als Entwurf (Platz 4)
**Bekannt.** Speculative Decoding (Leviathan 2023, Chen 2023, G). EAGLE (Li 2024): Entwurf auf Feature-Ebene (vorletzte Schicht), ca. 3x auf MT-bench, 2,7–3,5x bei LLaMA2-70B, Verteilung unverändert; ~1,6x über Medusa, ~2x über Lookahead. EAGLE-3: Kopf 2–5 % des Ziels, 2–3x. Quantisiertes Modell als Entwurf (QSpec u. ä., G) existiert.
**Muss in den Test.** (a) Entwurf mit *gleichen Bytes*: kleines dichtes Modell (drin); (b) n-Gramm-/Prompt-Lookup-Entwurf mit 0 Parametern – bei Bytes ist er die ehrliche Untergrenze und wird bei Text oft schwer geschlagen; (c) EAGLE-/Medusa-artiger Kopf auf Zielmerkmalen (kleiner MLP-Kopf, wenige KB); (d) Schicht-Überspringen (Self-Speculation, G).
**Im Skript?** zwilling_had, zwilling_roh, klein_dicht, unigramm: ja. NEIN: n-Gramm (2–4), Feature-Kopf, Layer-Skip. Zeit-Teil ist laut Skript "nur indikativ auf Raven", d ≤ 8192 Matvec – bei d=128 dominiert Overhead, nicht Bandbreite.
**Beitrag.** Akzeptanzrate eines *rotierten* 2-Bit-Zwillings vs. Roh-2-Bit (Rotation wirkt hier real, wenn der Zwilling auch dequantisiert wird). Zeitgewinn: nur Modell/Modellierung, nicht Messung auf Maltes Ryzen behaupten.
**Ampel: umbauen** (n-Gramm- und Kopf-Arm; Aussage auf "Akzeptanz pro Byte" beschränken).

## K02 Aufgaben-Superposition (Platz 5)
**Bekannt.** Cheung et al. 2019 (NeurIPS): mehrere Modelle in einem Gewichtssatz, Kontext = zufällige binäre Schlüssel, Abruf per Entschlüsselung – K02 ist dieselbe Konstruktion (W_t = diag(s)·W·diag(s')). SupSup (Wortsman 2020): feste Zufallsgewichte + Maske je Aufgabe, Aufgabenerkennung per 1 Gradientenschritt, bis 2500 Aufgaben. Außerdem Adapter/FiLM, Multi-Task mit geteiltem Rumpf (G).
**Muss in den Test.** (a) geteilter Rumpf + aufgabeneigener letzter Layer (Standard-Multi-Task, sehr stark bei MNIST/Fashion); (b) aufgabeneigene Diagonalskalen (gelernt, FiLM); (c) PSP selbst (identisch zu geteilt_k). geteilt_o mit nur Biases ist eine zu schwache Kontrolle.
**Im Skript?** geteilt_k, geteilt_o, getrennt_B, getrennt_3x: ja. (a), (b): nein. Die permutierte MNIST-Aufgabe ist künstlich (Permutation = Eingangsschlüssel, K02 kann sie trivial lösen).
**Beitrag.** Allenfalls Reproduktion bei gleichen Bytes. Kein QIRE-spezifischer Vorteil erkennbar.
**Ampel: streichen** als Hauptlinie (Rechenzeit sparen); höchstens als Abschnitt "Stand der Technik reproduziert".

## K06 Haar/Walsh-Zerlegung, Bitzuteilung (Platz 6)
**Bekannt.** Optimale Bitzuteilung nach Varianz (Reverse-Water-Filling, Transformcodierung wie JPEG, G); gemischte Präzision je Zeile/Kanal (SqueezeLLM, AWQ, G); Inkohärenz-Rotation (QuIP#: Hadamard glättet Ausreißer, Zufalls-Orthogonale ähnlich gut). Top-m-Koeffizienten (spärliche Transformcodierung, G). GPTQ/OPTQ-Fehlerkompensation (G).
**Muss in den Test.** Kontrolle "Bitzuteilung auf zufälliger Orthogonaler" (orth_alloc, orth_profil – drin, gut); RTN mit Rotation (had); svd; datenabhängig: GPTQ-artig ist Pflicht, falls man Anschluss an QuIP#-Zahlen will.
**Im Skript?** roh, had, walsh, orth, haar_unif, haar_alloc, orth_alloc, orth_profil, topm*, svd: ja – ungewöhnlich vollständig. NEIN: GPTQ, Zeilen-/Kanal-Bitzuteilung (SqueezeLLM-artig, ohne Haar).
**Beitrag.** Ob Haar-Struktur (Stufen = Frequenzbänder) gegen unstrukturierte Rotation echten Gewinn bringt; MNIST-Gewichte haben räumliche Struktur im ersten Layer (Pixelnachbarschaft), das ist die plausible Stelle (W1-Umfang).
**Ampel: umbauen** (nur W1 + Bilddaten behalten; GPTQ-Kontrolle ergänzen; Rest streichen).

## K04 Eingefrorene Seed-Aufwärtsschicht (Platz 7)
**Bekannt.** Zufallsmerkmale (Rahimi–Recht 2007, G), Reservoir-Computing, "nur BatchNorm trainieren" auf eingefrorenen Zufallsgewichten (Frankle 2020: nichttriviale Leistung). LoRA-FA (Zhang 2023): A einfrieren, nur B trainieren; bis 1,4x weniger Speicher als LoRA bei ähnlicher Genauigkeit – spart aber Aktivierungsspeicher beim Training, keine Modellbytes. Frozen-Transformer-als-Zufallsmerkmal (2025, Zeitreihen) existiert.
**Muss in den Test.** Dichte Modelle bei (i) gleichen Bytes (dicht512 – drin), (ii) gleichen FLOPs UND doppelten Bytes (dicht1024), (iii) gleichen Bytes und gleicher Breite via Niedrigrang W1=UV mit f=1024 (Rang so, dass Bytes = zufall_s1024). (iii) ist die härteste Kontrolle für "Bytes sparen".
**Im Skript?** dicht512, zufall_s1024, zufall_o1024, had_s1024: ja. dicht1024, zufall_s512 nur mit K04_ZUSATZ=1 (muss Pflicht werden). Niedrigrang-Arm: nein.
**Beitrag.** Die FLOPs verdoppeln sich; es ist ein Speicher-gegen-Rechnen-Tausch (Seed regenerieren), kein Ressourcengewinn insgesamt. Ehrlich so berichten.
**Ampel: umbauen** (ZUSATZ fest an, Niedrigrang-Arm; Aussage "Bytes ja, FLOPs nein").

## K10 Born-Stichprobe der FFN-Spalten (Platz 8)
**Bekannt.** Deja Vu (Liu, ICML 2023): kontextabhängige Spärlichkeit, Vorhersage der wichtigen MLP-Neuronen, >2x Latenz gegenüber FasterTransformer, >6x gegenüber HF bei OPT-175B; Zufallsauswahl bricht ein. CRS-Matrixprodukt (Drineas–Kannan–Mahoney 2006, G): optimale Stichprobe p_j ∝ |a_j|·‖W2[j]‖ (nicht a²!). Top-k-Aktivierungen/ReLU-Spärlichkeit (G).
**Muss in den Test.** top-k (deterministisch, FLOP-gleich) und CRS-optimal p ∝ |a_j|·‖w_j‖; Deja-Vu-Prädiktor ist bei d=128 nicht lohnend.
**Im Skript?** dicht_voll, schmal (FLOP-gleich), born/top/unif: ja. NEIN: CRS-optimal. Wichtig: Training als dichte Simulation mit Nullen, FLOPs nur analytisch – keine Messung.
**Beitrag.** "Born" = Varianzreduktion-Variante; Vorteil gegenüber top-k ist theoretisch nur Erwartungstreue. Wahrscheinlich H0.
**Ampel: streichen** (top_S zeigt es nebenbei; sonst nur ein Nebenarm).

## K07 Optimiererzustand rotiert, wenige Bit (Platz 9)
**Bekannt.** 8-Bit-Adam (Dettmers 2021/22): blockweise dynamische Quantisierung + Stable-Embedding, Optimizer-Zustand auf ca. 25 % des Speichers (Sekundärzusammenfassung), 32-Bit-Qualität auf großen Aufgaben. 4-Bit-Zustände (Li, Chen, Zhu, NeurIPS 2023): Zero-Point-Problem beim zweiten Moment, linearer Quantisierer ohne Null, kleinere Blöcke, Zeilen-/Spalten-Normierung, faktorisiertes v. Adafactor (G).
**Muss in den Test.** Li-4-Bit-Variante (Rang-1-Normierung + Quantisierer ohne Null); Adafactor/8-Bit-Block (drin). Die Trennung b4roh/b4rot ist die QIRE-spezifische Kontrolle.
**Im Skript?** adam32, b8, b4roh, b4rot, b2roh, b2rot, adafactor: ja (b4roh wendet schon die Mittelwert-Abzug-Idee an, nahe bei Li). NEIN: Li exakt, Dettmers' dynamische Baumcodierung. Hinweis: v ≥ 0 – Rotation von sqrt(v) nach Mittelabzug ist heuristisch; Rotation für m (symmetrisch) plausibel.
**Beitrag.** Ob Rotation bei 2–4 Bit dem Block-Roh-Quantisierer hilft. T=2500 bei 1 M Parametern: kein Beweis für Milliardenmodelle (dort Ausreißer anders).
**Ampel: umbauen** (Li-Arm ergänzen; auf m-Rotation fokussieren).

## K12 Bit-Baum / Hadamard-Code als Ausgabekopf (Platz 9)
**Bekannt.** Hierarchischer Softmax (Morin–Bengio 2005): O(log V) statt V, ca. zwei Größenordnungen schneller, aber spürbar weniger genau als flach (laut Mnih–Hinton 2008, die datengetriebene Bäume vorschlagen). Huffman-Baum in word2vec (Mikolov 2013). Klassenbasiert (Goodman 2001, √V), Adaptive Softmax (Grave 2017, G). Fehlerkorrigierende Ausgabecodes (Dietterich–Bakiri 1995, G) = had32/rand32.
**Muss in den Test.** Huffman-Baum nach Byte-Häufigkeit (im Mittel ~5 statt 8 Schritte), Klassen-Softmax 16x16, voll.
**Im Skript?** bittree (balanciert MSB-first), had32, rand32, rang32, voll: ja. NEIN: Huffman, 16x16-Klassen, Adaptive Softmax.
**Beitrag.** Bei V=256 und d=128 sind Kopf-FLOPs ein kleiner Teil der Gesamt-FLOPs (Skript weist flops_gesamt aus) – der Gewinn ist gedeckelt. Hadamard-Codewörter vs. Zufallscode ist die einzige QIRE-spezifische Frage und bleibt bei ECOC-Wissen.
**Ampel: streichen.**

## Querbefund
- Mehrfach fehlt der "trainierte/gelernte Gegenspieler" (K03 k-Means, K01 gelernte Diagonale, K02 geteilter Rumpf, K04 Niedrigrang). Das ist die häufigste Lücke.
- Rotation + Skalar ist durch QuIP#/QuaRot/TurboQuant belegt: Für K03/K05/K07 nur als Grundlage zitieren, nicht als Idee verkaufen.
- Empfohlene Reihenfolge nach dieser Recherche: K05 (KIVI-Arm), K03 (k-Means/SeedLM), K01 (gelernte Diagonale), K18 (n-Gramm) – danach erst mehr. K02, K10, K12 nicht rechnen.

## BibTeX
@article{tseng2024quip, title={QuIP\#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks}, author={Tseng, Albert and Chee, Jerry and Sun, Qingyao and Kuleshov, Volodymyr and De Sa, Christopher}, journal={ICML}, year={2024}}
@inproceedings{egiazarian2024aqlm, title={Extreme Compression of Large Language Models via Additive Quantization}, author={Egiazarian, Vage and others}, booktitle={ICML}, year={2024}, note={arXiv:2401.06118}}
@inproceedings{shafipour2025seedlm, title={SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators}, author={Shafipour, Rasoul and others}, booktitle={ICLR}, year={2025}, note={arXiv:2410.10714}}
@inproceedings{liu2024kivi, title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache}, author={Liu, Zirui and others}, booktitle={ICML}, year={2024}, note={arXiv:2402.02750}}
@inproceedings{zandieh2026turboquant, title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate}, author={Zandieh, Amir and others}, booktitle={ICLR}, year={2026}, note={Autoren/Zahlen nur aus Sekundärquellen, vor Zitat prüfen}}
@inproceedings{li2024eagle, title={EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty}, author={Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang}, booktitle={ICML}, year={2024}, note={arXiv:2401.15077}}
@inproceedings{lan2020albert, title={ALBERT: A Lite BERT for Self-supervised Learning of Language Representations}, author={Lan, Zhenzhong and others}, booktitle={ICLR}, year={2020}, note={arXiv:1909.11942}}
@inproceedings{dehghani2019universal, title={Universal Transformers}, author={Dehghani, Mostafa and others}, booktitle={ICLR}, year={2019}}
@inproceedings{cheung2019superposition, title={Superposition of many models into one}, author={Cheung, Brian and Terekhov, Alexander and Chen, Yubei and Agrawal, Pulkit and Olshausen, Bruno}, booktitle={NeurIPS}, year={2019}, note={arXiv:1902.05522}}
@inproceedings{wortsman2020supsup, title={Supermasks in Superposition}, author={Wortsman, Mitchell and others}, booktitle={NeurIPS}, year={2020}, note={arXiv:2006.14769}}
@article{zhang2023lorafa, title={LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning}, author={Zhang, Longteng and others}, year={2023}, note={arXiv:2308.03303}}
@inproceedings{dettmers2022bit8, title={8-bit Optimizers via Block-wise Quantization}, author={Dettmers, Tim and Lewis, Mike and Shleifer, Sam and Zettlemoyer, Luke}, booktitle={ICLR}, year={2022}, note={arXiv:2110.02861}}
@inproceedings{li2023bit4, title={Memory Efficient Optimizers with 4-bit States}, author={Li, Bingrui and Chen, Jianfei and Zhu, Jun}, booktitle={NeurIPS}, year={2023}, note={arXiv:2309.01507}}
@inproceedings{morin2005hierarchical, title={Hierarchical Probabilistic Neural Network Language Model}, author={Morin, Frederic and Bengio, Yoshua}, booktitle={AISTATS}, year={2005}}
@inproceedings{mnih2008scalable, title={A Scalable Hierarchical Distributed Language Model}, author={Mnih, Andriy and Hinton, Geoffrey}, booktitle={NeurIPS}, year={2008}}
@inproceedings{mikolov2013distributed, title={Distributed Representations of Words and Phrases and their Compositionality}, author={Mikolov, Tomas and others}, booktitle={NeurIPS}, year={2013}, note={arXiv:1310.4546}}
@inproceedings{liu2023dejavu, title={Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time}, author={Liu, Zichang and others}, booktitle={ICML}, year={2023}, note={arXiv:2310.17157}}
@article{drineas2006fast, title={Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication}, author={Drineas, Petros and Kannan, Ravi and Mahoney, Michael W.}, journal={SIAM J. Comput.}, year={2006}}
@article{frankle2020bn, title={Training BatchNorm and Only BatchNorm}, author={Frankle, Jonathan and Schwab, David J. and Morcos, Ari S.}, year={2020}, note={arXiv:2003.00152}}
