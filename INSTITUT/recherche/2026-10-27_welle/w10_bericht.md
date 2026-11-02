# w10: Kreativ-Recherche, Konzepte aus Quanten-/Vielteilchenphysik -> effiziente KI (27.10.2026)

Nichts gerechnet, kein git. Nicht abgedeckt (anderer Agent): Tensor-Netze, QuaRot/Hadamard, Monarch, Born-Machines.
Quellenlage: 4 WebSearches (Hamilton/symplektisch, Tang, Lindblad/SSM, Quantenwalk-GNN) lieferten fast nur Abstracts, **keine Tabellenzahlen**. Zahlen unten sind daher meist [Gedächtnis, ungeprüft] und vor jeder Zitation in der Arbeit am Original zu prüfen. Preprint-IDs 26xx stammen nur aus Suchtreffern (nicht gelesen).

## 1. Kurzfassung und Ampel
- Maßstab (I-045-v4): Nur Ideen, die bei **gleichen FLOPs und Parametern** dicht/Niedrigrang schlagen können, zählen. Ideen, die nur Speicher sparen, sind gelb.
- Grün (lohnt Rauchtest, dann Vorregistrierung): **R1 Hourglass/RG-Multiskala (Byte-LM)**, **R2 diagonale dissipative Rekurrenz (LRU/Lindblad)**.
- Gelb: R3 ℓ2-Sampling-Matmul (Tang/DKM) im Training, R4 Annealing/Homotopie für ternär, R5 Hamilton-RNN, R6 reversible Hamilton-Blöcke.
- Rot/schwach: R7 QEC/ECOC, R8 Symmetrie-Gewichtsteilung, R9 Dichtematrix-Embeddings, R10 Quantenwalk-GNN. Zeno/Dekohärenz-Regularisierung = Dropout/Rauschen, keine eigene Idee.
- Ehrlich: Alle Mechanismen sind klassisch. Formulierung "quanten-inspiriert", Bezug oft nur Motivation (Analogie), nicht Beweis.

## 2. Ideen (je: Mechanismus | belegte Zahl | warum gegen dicht bei gleichen FLOPs | kleinster CPU-Test)

### R1 Renormierungsgruppe / Coarse-Graining -> Hourglass-Byte-LM  [GRÜN]
- Mechanismus: RG/MERA fasst k Freiheitsgrade zu einem Block-Freiheitsgrad zusammen, rechnet auf grober Skala, verfeinert zurück. KI: Bytes zu Patches poolen (k=4), Kern-Netz arbeitet auf T/k Positionen, Upsampling + Skip. Mehta/Schwab: variationale RG = Deep-Learning-Schicht (Analogie).
- Zahlen: Hourglass (Nawrot 2022): bei gleicher Rechenzeit besser/gleich gut wie flaches Transformer-LM auf enwik8, "Hierarchische Transformer sind effizienter" [Gedächtnis, genaue bpc nachlesen]. Funnel-Transformer: weniger FLOPs bei ähnlicher GLUE-Güte [Gedächtnis]. MegaByte (Yu 2023): Byte-Patches, billigere Dekodierung bei längerem Kontext [Gedächtnis].
- Warum bei gleichen FLOPs: FLOPs sinken real (Faktor ~k im tiefen Teil), die gesparten FLOPs fließen in breiteren Kern. Das ist der einzige Kandidat, der FLOPs, nicht nur Parameter spart. Struktur passt zu Sprache (Bytes -> Silben/Wörter).
- Risiko: Auf GSD-Skala (kleiner Korpus) könnte Patching Info verlieren; langweilige Erklärung: bessere Wirkung nur durch größere Modellbreite.
- Test (CPU, <5 min/Modell): `struktur_sprache.py` als Basis; Arm A flach (d=128, L=4), Arm B Hourglass (pool 4, Kern d=192, L=4 auf T/4 + 1 lokale Schicht je Seite), FLOPs/Byte gleich gezählt, 3 Tuning-Seeds, bpc. Primärmaß: bpc bei fixem FLOP-Budget.

### R2 Lindblad/offene Systeme -> diagonale dissipative Rekurrenz  [GRÜN]
- Mechanismus: Lindblad-Dynamik = Rotation (Hamilton-Teil, Imaginärteil) + Dämpfung (Dissipator, Realteil); stabil, wenn Realteil < 0. In KI: komplexe diagonale lineare Rekurrenz h_t = λ⊙h_{t-1} + B x_t, |λ|<1 über λ=exp(-exp(ν)+iθ) (LRU). Rotation und Zerfall getrennt parametrisiert (so auch die 2026er "Lindblad-inspired reservoir"-Preprint, Beuria/Shukla, nur Abstract gesehen, Benchmarks NARMA-20/Lorenz).
- Zahlen: LRU (Orvieto 2023) erreicht auf Long Range Arena Mittel nahe S4/S4D (~85-86 % vs ~86 %) [Gedächtnis, nachlesen]. S4D (Gu 2022) diagonal fast so gut wie volles S4 [Gedächtnis].
- Warum bei gleichen FLOPs: dichte Rekurrenz kostet O(d²) je Schritt, diagonal O(d); bei gleichem FLOP-Budget kann d ~10x größer sein, und der Zustand ist parallel (Scan/FFT-Faltung) trainierbar. Stabilität garantiert per Konstruktion.
- Risiko: Das ist bekannte Technik (zitieren, nicht neu). Gegen Transformer auf Byte-LM bei kleinem Budget offen; langweilig: Gewinn nur gegen schlecht eingestelltes GRU.
- Test: Byte-LM mit [Embedding -> LRU-Block (komplex diag, d=256) + MLP] vs GRU (gleiche FLOPs/Byte) vs dichter Niedrigrang-MLP-Block. Faltung per FFT, kein Python-Zeitloop. 3 Seeds, bpc. Optional: |λ|-Init-Ring (0.9..0.999) als einzige HP.

### R3 Grover/Tang: ℓ2-Norm-Sampling statt volles Matmul im Training  [GELB]
- Mechanismus: Tangs Dequantisierung (Quantum-Empfehlung, STOC 2019) ersetzt Amplitudenzugriff durch ℓ2-Norm-Sampling; Chia et al. (JACM) verallgemeinern (SVT, PCA, SVM, Regression); Kernbaustein: Sampling nähert Matrixprodukte dimensionsunabhängig. Klassisch bekannt als Column-Row-Sampling (Drineas/Kannan/Mahoney): Fehler ~ ‖A‖_F‖B‖_F/√s. Grover selbst (√N-Suche) hat kein klassisches NN-Analogon, nur "wichtige Einträge finden" ≈ Top-k/Importance-Sampling.
- Zahl: Adelman et al. (NeurIPS 2021) approximative Tensoroperationen im Backprop, bis ~66 % weniger Rechenaufwand bei kleinem Genauigkeitsverlust [Gedächtnis, nachlesen].
- Warum gegen dicht: spart echt FLOPs (Training, ggf. Inferenz mit Top-k), unabhängig von der Parametrisierung. Nachteil: Sampling-Varianz, bei langem Training holt dicht mit weniger Schritten auf (Lehre I-044/I-045).
- Test: MLP 784-256-10 auf MNIST, Backward-Gewichtsgradient mit s=25/50 % der Zeilen nach ℓ2-Norm gesampelt, FLOP-Zähler; Vergleich: dichtes Netz mit gleich vielen FLOPs (weniger Epochen). 3 Seeds.

### R4 Adiabatisches Annealing / Homotopie fürs Training ternärer Netze  [GELB]
- Mechanismus: Adiabatik = langsam von leicht lösbarem H0 zu Problem-Hamiltonian H1 wechseln. KI: Kontinuierungs-/Homotopie-Training, z. B. Quantisierer Q_τ(w)=tanh(w/τ)-artig, τ -> 0 (Soft-to-Hard, BinaryRelax); Rauschen/Glättung (Mobahi/Fisher: Gaussian-Homotopie).
- Zahlen: Soft-to-Hard VQ (Agustsson 2017), BinaryRelax (Yin 2018) verbessern Binär/Ternär-Genauigkeit gegenüber nacktem STE um ~Prozentpunkte [Gedächtnis, ungeprüft].
- Warum: Kostet null Inferenz-FLOPs; verbessert Qualität des festen Zielnetzes (ternär/Butterfly) bei gleichen Parametern/FLOPs. Hilft aber nur ternär vs ternär, nicht gegen float-dicht.
- Test: `ternaer_butterfly.py`: STE hart vs lineares τ-Schedule vs Zick-Zack; Epochen gleich. 3 Seeds, Tuning-Val.

### R5 Hamilton-/symplektische RNN (coRNN-Linie)  [GELB]
- Mechanismus: Hidden-State als (q,p), Update per symplektischem Integrator (Leapfrog); Phasenraumvolumen erhalten, Gradient beschränkt, Schrittweite stellt Gradientengröße ein.
- Zahlen: Hamilton-RNN (arXiv 1911.05035) schlägt AntisymmetricRNN/LSTM mit weniger Parametern auf Standard-Benchmarks (Suchtreffer, ohne Zahlen); Symplectic RNN (Chen 2020) nur Physiksysteme. Keine Belege für große Sprachmodelle.
- Warum: Gradientenstabilität erlaubt lange Sequenzen mit kleinem d; FLOPs aber ähnlich GRU (nicht sparsamer). Ist R2 im Grunde unterlegen (R2 parallelisierbar).
- Test: psMNIST-Zeilenweise oder Byte-LM klein; Hamilton-RNN vs GRU gleiche Parameter. Erst nach R2.

### R6 Reversible/Hamilton-Residualblöcke  [GELB, nur Speicher]
- Mechanismus: Haber/Ruthotto (stabile Architekturen als Hamilton-System), RevNet (Gomez 2017) rekonstruiert Aktivierungen im Backward.
- Zahl: RevNet ~ Aktivierungsspeicher unabhängig von Tiefe bei ~gleicher Genauigkeit auf CIFAR/ImageNet [Gedächtnis]; kostet ~+33 % FLOPs (Neuberechnung).
- Bewertung: spart Speicher, nicht FLOPs -> für Ziel "FLOPs" ungeeignet. Nur für Raven-Speicherengpass relevant.

### R7 Quantenfehlerkorrektur -> ECOC/Redundanzcodes für Niedrigbit  [ROT/gelb]
- Mechanismus: Logische Information über redundante Bits verteilen; KI: Error-Correcting Output Codes (Dietterich/Bakiri 1995), Verma/Swami (NeurIPS 2019: robuster gegen Angriffe) [Gedächtnis]. Auch Bit-Flip-Robustheit von Gewichten durch Codes.
- Bewertung: verbessert Robustheit, nicht FLOPs; Mehraufwand im Ausgang. Kein Effizienzgewinn, nur Randthema der Arbeit.
- Test (billig): MNIST-MLP, 10 Klassen vs 32-Bit-Hadamard-/BCH-Code-Ausgang, ternäre Gewichte, zufällige Bit-Flips der Gewichte; Robustheitskurve.

### R8 Symmetrie/Erhaltungsgrößen -> Gewichtsteilung  [ROT für diese Daten]
- Mechanismus: Noether: Symmetrie => Gewichtsteilung (Faltung = Translation; DeepSets = Permutation; Cohen/Welling G-CNN).
- Bewertung: spart Parameter und FLOPs, aber nur wenn Daten die Symmetrie haben; MNIST: Conv gewinnt trivial (bekannt, kein quanten-Bezug). Byte-Sprache: keine klare Symmetrie. Als Kontrollarm nützlich ("klassisch gleich teuer").

### R9 Dichtematrix-Embeddings  [ROT]
- Mechanismus: Eingabe als PSD-Matrix ρ=Σ p_i |φ_i><φ_i|; Dichtematrix-KDE mit Random Features (Gonzalez et al., ICLR 2021 [Gedächtnis]); Sordoni 2013 für Sprache.
- Bewertung: Zustandsgröße wächst quadratisch (d²), nicht FLOPs-sparsam; schlägt Niedrigrang kaum. Nur Interpretationswert.

### R10 Quantenwalk -> Graphnetze  [ROT, themenfern]
- Mechanismus: CTQW-Propagator exp(-iHt) unitär -> keine Überglättung; Schrödinger-GNN (arXiv 2605.13383, nur Abstract), CTQW-GNN (2608.20738, nur Abstract), QDC (TMLR 2024).
- Bewertung: Graphdaten fehlen im QIRE-Datensatz; Gewinn = Signalweitergabe, nicht FLOPs (Matrixexponential teuer). Nicht verfolgen.

### Zeno/Dekohärenz -> Regularisierung  [ROT, ohne Nummer]
Dekohärenz = Rauschen, Zeno = häufiges Messen/Projizieren. Entspricht Dropout/Gauß-Rauschen/Zustandsprojektion; Regularisierung erhöht keine Recheneffizienz. Kein eigener Test.

## 3. Rangliste Top 10
| Rang | Idee | Spart | Ampel | Aufwand Test |
|---|---|---|---|---|
| 1 | R1 Hourglass/RG-Multiskala (Byte-LM) | FLOPs real | grün | mittel (Modul bauen) |
| 2 | R2 LRU/Lindblad-diagonal-komplex | FLOPs je Schritt O(d) | grün | mittel |
| 3 | R3 ℓ2-Sampling-Matmul (Tang/DKM) | Trainings-FLOPs | gelb | klein |
| 4 | R4 Annealing/Homotopie ternär | Güte bei fixem Netz | gelb | klein |
| 5 | R5 Hamilton-RNN | Stabilität | gelb | mittel |
| 6 | R6 reversible Hamilton-Blöcke | nur Speicher | gelb | klein |
| 7 | R7 QEC/ECOC | Robustheit | rot/gelb | klein |
| 8 | R8 Symmetrie-Teilung | Parameter+FLOPs (datenabh.) | rot | klein |
| 9 | R9 Dichtematrix-Embedding | nichts | rot | klein |
| 10 | R10 Quantenwalk-GNN | nichts (themenfern) | rot | groß |

## 4. Empfehlung
- Zuerst R1 und R2 rauchtesten (Byte-Sprachmodell, Tuning-Seeds 900-999, 3 Seeds, bpc bei gleichen gezählten FLOPs/Byte, gleichen Parametern, Kontrolle: dichter Niedrigrang). R1 und R2 sind kombinierbar (Hourglass mit LRU-Kern).
- R4 als billiger Zusatz für das ternäre Butterfly-Netz; R3 als Trainings-FLOPs-Experiment.
- Mögliche Vorregistrierung (nur nach Rauchtest + Maltes Zustimmung): Arme flach-dicht / Niedrigrang / Hourglass / LRU / Hourglass+LRU, gleiche FLOPs je Byte (±3 %), Effektgröße ≥ 0,03 bpc, ≥ 10 Seeds, volles Training (nicht knapp trainiert, I-044/I-045). Laufzeit Container ~mehrere Stunden pro Arm, besser Raven/Maltes PC.

## 5. BibTeX (alle [Gedächtnis], Felder vor Verwendung prüfen)
```
@inproceedings{nawrot2022hourglass, title={Hierarchical Transformers Are More Efficient Language Models}, author={Nawrot, Piotr and Tworkowski, Szymon and others}, booktitle={Findings of NAACL}, year={2022}}
@inproceedings{yu2023megabyte, title={MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers}, author={Yu, Lili and others}, booktitle={NeurIPS}, year={2023}}
@inproceedings{dai2020funnel, title={Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing}, author={Dai, Zihang and others}, booktitle={NeurIPS}, year={2020}}
@article{mehta2014rg, title={An exact mapping between the Variational Renormalization Group and Deep Learning}, author={Mehta, Pankaj and Schwab, David J.}, journal={arXiv:1410.3831}, year={2014}}
@article{vidal2007mera, title={Entanglement Renormalization}, author={Vidal, Guifre}, journal={Phys. Rev. Lett.}, volume={99}, pages={220405}, year={2007}}
@inproceedings{orvieto2023lru, title={Resurrecting Recurrent Neural Networks for Long Sequences}, author={Orvieto, Antonio and others}, booktitle={ICML}, year={2023}}
@inproceedings{gu2022s4d, title={On the Parameterization and Initialization of Diagonal State Space Models}, author={Gu, Albert and others}, booktitle={NeurIPS}, year={2022}}
@inproceedings{tang2019recsys, title={A quantum-inspired classical algorithm for recommendation systems}, author={Tang, Ewin}, booktitle={STOC}, year={2019}, note={arXiv:1807.04271}}
@article{chia2022jacm, title={Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning}, author={Chia, Nai-Hui and Gily{\'e}n, Andr{\'a}s and Li, Tongyang and Lin, Han-Hsuan and Tang, Ewin and Wang, Chunhao}, journal={J. ACM}, year={2022}, note={arXiv:1910.06151}}
@article{drineas2006matmul, title={Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication}, author={Drineas, Petros and Kannan, Ravi and Mahoney, Michael W.}, journal={SIAM J. Comput.}, volume={36}, year={2006}}
@inproceedings{adelman2021approx, title={Faster Neural Network Training with Approximate Tensor Operations}, author={Adelman, Menachem and Levy, Kfir and Hakimi, Ido and Silberstein, Mark}, booktitle={NeurIPS}, year={2021}}
@inproceedings{yin2018binaryrelax, title={BinaryRelax: A Relaxation Approach for Training Deep Neural Networks with Quantized Weights}, author={Yin, Penghang and others}, journal={SIAM J. Imaging Sci.}, year={2018}}
@inproceedings{agustsson2017softtohard, title={Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations}, author={Agustsson, Eirikur and others}, booktitle={NeurIPS}, year={2017}}
@inproceedings{mobahi2015continuation, title={A Theoretical Analysis of Optimization by Gaussian Continuation}, author={Mobahi, Hossein and Fisher III, John W.}, booktitle={AAAI}, year={2015}}
@article{hamiltonrnn2019, title={Constructing Gradient Controllable Recurrent Neural Networks Using Hamiltonian Dynamics}, journal={arXiv:1911.05035}, year={2019}, note={Autoren nachtragen}}
@inproceedings{chen2020symplectic, title={Symplectic Recurrent Neural Networks}, author={Chen, Zhengdao and others}, booktitle={ICLR}, year={2020}, note={arXiv:1909.13334}}
@inproceedings{greydanus2019hnn, title={Hamiltonian Neural Networks}, author={Greydanus, Samuel and Dzamba, Misko and Yosinski, Jason}, booktitle={NeurIPS}, year={2019}}
@inproceedings{gomez2017revnet, title={The Reversible Residual Network: Backpropagation Without Storing Activations}, author={Gomez, Aidan N. and others}, booktitle={NeurIPS}, year={2017}}
@article{haber2017stable, title={Stable architectures for deep neural networks}, author={Haber, Eldad and Ruthotto, Lars}, journal={Inverse Problems}, year={2017}}
@article{dietterich1995ecoc, title={Solving Multiclass Learning Problems via Error-Correcting Output Codes}, author={Dietterich, Thomas G. and Bakiri, Ghulum}, journal={JAIR}, year={1995}}
@inproceedings{cohen2016gcnn, title={Group Equivariant Convolutional Networks}, author={Cohen, Taco and Welling, Max}, booktitle={ICML}, year={2016}}
@inproceedings{zaheer2017deepsets, title={Deep Sets}, author={Zaheer, Manzil and others}, booktitle={NeurIPS}, year={2017}}
@inproceedings{gonzalez2021dmkde, title={Learning with Density Matrices and Random Features}, author={Gonz{\'a}lez, Fabio A. and others}, year={2021}, note={arXiv:2102.04394}}
```
Nur aus Suchtreffern (nicht gelesen, Aug. 2026): arXiv 2608.20738 (CTQW-GNN), 2605.13383 (Schrödinger-GNN), 2608.04028 (Lindblad-Reservoir); QDC (TMLR 2024).

Quellen der Suche: arxiv.org/pdf/1911.05035, iclr.cc (Symplectic RNN), arxiv.org/abs/1910.06151, arxiv.org/pdf/2608.20738, alphaxiv.org/abs/2605.13383, alphaxiv.org/abs/2608.04028.

## 6. Skriptpfade
Keine (reine Recherche). Bei Test: `INSTITUT/recherche/2026-10-27_welle/w10/`.
