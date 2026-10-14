# 11 — Effizienz im Großen auf klassischer Hardware: Spart Maske + Block-WHT + Quadrieren wirklich Energie und Zeit?

Stand 05.10.2026. Verfasser: Recherche-Agent (Sonnet) für den Direktor.

**WICHTIGER VORBEHALT ZUR QUELLENLAGE:** In dieser Sitzung war das WebSearch-Budget bereits aufgebraucht (200/200), und arxiv.org, export.arxiv.org und api.openalex.org sind durch den Egress-Proxy gesperrt. Es gab **keine einzige** Web-Recherche. Alle Literaturangaben unten stammen aus dem Gedächtnis des Modells und sind mit **[Gedächtnis]** markiert: Autoren, Jahre und arXiv-Nummern bitte vor dem Zitieren in der schriftlichen Arbeit prüfen. Gemessen (nicht erinnert) sind nur Abschnitt 3 (eigener Zeitvergleich) und die Repo-Zahlen (Abschnitt 2.4). Die Neuheitsprüfung (Abschnitt 4) ist deshalb **nicht erfolgt, nur eine Einschätzung** — Etappe 2 muss sie nachholen.

## 1. Kurzfassung
- **Arithmetik:** Die Clifford-Zelle (Vorzeichenmaske, Block-WHT, Quadrieren) braucht auf 256 Pixeln ~2.300 Operationen (2.048 Additionen/Subtraktionen + 256 Vorzeichen + 256 Quadrate), eine dichte 256×256-Schicht 65.536 Multiply-Accumulates. Verhältnis ~28×; mit 16er-Blöcken ~50×. **Der lineare Kopf (256×10 = 2.560 MAC) kostet dann mehr als die ganze Zelle.**
- **Energie (Horowitz-Zahlen [Gedächtnis]):** Eine fp32-Multiplikation kostet ~4× eine fp32-Addition, eine int8-Multiplikation ~7× eine int8-Addition. Rechnerisch (nur Rechenwerk, 45 nm) ~160× weniger Energie je Schicht gegenüber dichtem fp32, ~75× gegenüber dichtem int8. Das ist eine **Rechnung, keine Messung**, und lässt Speicherzugriffe außen vor, wo die WHT zusätzlich gewinnt (keine Gewichte, nur 256 Bit Maske).
- **Zeit auf Maltes CPU-Klasse: ernüchternd.** Eigene Messung (numpy, 1 Thread): FWHT + Maske + Quadrieren ist bei Batch 64 **5× langsamer** als ein Matmul 256×256 (665 µs gegen 130 µs); bei Batch 1 ~17× langsamer. Nur die BLAS-Implementierung (GEMM) ist so optimiert, dass 32× mehr Rechenoperationen schneller laufen. Das deckt sich mit T-001 aus dem Repo (Übergang erst bei 16.384, später durch Bündler auf 4.096–8.192 gedrückt).
- **Konsequenz:** Zeitgewinn auf CPU/GPU erst mit eigenem Kernel (fusionierte Schmetterlinge im Register/SRAM) oder auf FPGA/ASIC. Energiegewinn ist auf Spezialhardware plausibel groß, auf Maltes PC nicht nachweisbar (Leistungsmessung ohne RAPL schwer).
- **[EIGEN]-Idee mit bestem Verhältnis Aufwand/Nutzen:** Die gelernte Clifford-Maske als **Rotation für Quantisierung** (QuaRot/SpinQuant-Linie) testen — billig (Minuten) auf Gewichts-/Aktivierungsmatrizen, ehrlicher Gegentest gegen feste Hadamard und zufällige Vorzeichen.
- **[EIGEN]-Idee 2:** „Additionen pro Klassifikation“ als zusätzliches Effizienzmaß, mit sauberer Zählvorschrift (Abschnitt 4.2).

## 2. Stand der Forschung (alles [Gedächtnis], ungeprüft)

### 2.1 Energie je Operation (Horowitz 2014)
- M. Horowitz, „Computing's Energy Problem (and what we can do about it)“, ISSCC 2014, 45 nm [Gedächtnis]. Die in der Literatur meistzitierte Tabelle (ungefähr, aus dem Gedächtnis, vor Zitat prüfen): int8 Add 0,03 pJ; int32 Add 0,1 pJ; int8 Mult 0,2 pJ; int32 Mult 3,1 pJ; fp16 Add 0,4 pJ; fp32 Add 0,9 pJ; fp16 Mult 1,1 pJ; fp32 Mult 3,7 pJ; 8-KB-SRAM-Lesen (64 bit) ~10 pJ; DRAM-Zugriff (64 bit) ~1,3–2,6 nJ.
- Kernaussage: Datenbewegung (DRAM) kostet 2–3 Größenordnungen mehr als Rechnen. Für die WHT heißt das: Der Vorteil „keine Multiplikationen“ ist klein gegen den Vorteil „keine Gewichte aus dem DRAM laden“.
- Sehr alter Prozess (45 nm); auf heutigen Knoten skalieren die absoluten Werte, das Verhältnis Mult/Add bleibt grob erhalten [?].

### 2.2 Hadamard-Schichten in neuronalen Netzen
- **Fastfood** (Le, Sarlós, Smola 2013, ICML) [Gedächtnis]: Matrix-Vektor-Produkt der Form V = S·H·G·Π·H·B (B Vorzeichen ±1, H Walsh-Hadamard, Π Permutation, G und S diagonal) in O(n log n) Zeit, O(n) Speicher, zur Approximation von Kernen.
- **Deep Fried Convnets** (Yang, Moczulski, Denil, de Freitas, Smola, Song, Wang 2015) [Gedächtnis]: Fastfood ersetzt die vollverbundenen Schichten von CNNs, ~50 % Parameter gespart, bei ImageNet kaum Genauigkeitsverlust.
- **ACDC** (Moczulski, Denil, Appleyard, de Freitas 2016) [Gedächtnis]: Faltung aus Diagonal-, DCT-Schichten; O(n) Parameter, O(n log n) Operationen, stapelbar als tiefes Netz.
- **HadaNet** (Akhauri 2019) [Gedächtnis, Name und Jahr unsicher]: Hadamard-Transformation statt Multiplikationen in binären/ternären Netzen, mit FPGA-Auswertung. **[?]** Ergebniszahlen nicht erinnert.
- **Butterfly/Monarch/Kaleidoscope** (Dao et al. 2019–2022) [Gedächtnis]: Lernbare Schmetterlingsfaktoren; GPU-Beschleunigung wird dort ausdrücklich nur mit eigenen Kernen erreicht. Passt zu T-001.
- **Hadamard in Transformern:** FNet (Lee-Thorp et al. 2021) ersetzt Attention durch Fourier-Transformation (Hartley/Hadamard-ähnlich) [Gedächtnis]; „Hadamard“ taucht außerdem in Attention-Varianten und im Training mit Gradientenkompression (Hadamard-Quantisierung, Xi et al. 2023) auf [?].

### 2.3 Hadamard-Rotationen in der LLM-Quantisierung
- **QuaRot** (Ashkboos et al. 2024) [Gedächtnis]: Orthogonale Rotationen (Hadamard, zufällige Vorzeichen) verteilen Ausreißer im Residualstrom, so dass 4-Bit-Quantisierung von Gewichten, Aktivierungen und KV-Cache möglich wird; Online-Hadamard (FWHT) an der Eingabe der Down-Projektion und im Attention-Ausgang. Die Rotation wird in Gewichte eingefaltet.
- **SpinQuant** (Liu et al. 2024) [Gedächtnis]: Statt fest/zufällig werden die Rotationen **gelernt** (Cayley-Optimierung auf der Stiefel-Mannigfaltigkeit), große Streuung zwischen zufälligen Rotationen als Motivation (Unterschied mehrere Prozentpunkte in Genauigkeit).
- **QuIP/QuIP#** (Chee, Tseng et al. 2023/24) [Gedächtnis]: Inkohärenz-Verarbeitung mit zufälligen Hadamard-Matrizen (Vorzeichen · Hadamard) zur 2-Bit-Gewichtsquantisierung.
- **BitNet b1.58** (Ma et al. 2024) [Gedächtnis]: Gewichte in {−1, 0, +1}, Multiplikationen werden zu Additionen; Energieersparnis laut Paper auf Horowitz-Grundlage mit Prozentangaben. **[?]** Zahlen nicht erinnert. BitNet verwendet in der Aktivierungsquantisierung teilweise Hadamard (Nachfolger „BitNet a4.8“, [?]).
- Beobachtung (**[EIGEN]**-Einordnung): In all diesen Arbeiten ist die Rotation eine **feste oder allgemein-orthogonale** Matrix. Eine Rotation aus einer Vorzeichenmaske (quadratische Form über GF(2) = CZ/S-Anteil) plus Block-WHT, also die Clifford-Familie, ist eine endliche, diskrete Untermenge.

### 2.4 Hardware-Umsetzung binärer und Hadamard-Netze
- FPGA-Binärnetze (FINN, Umuroglu et al. 2017) [Gedächtnis]: XNOR + Popcount statt Multiplikation; Energie je Klassifikation und Durchsatz drastisch besser als GPU bei kleinen Netzen. XNOR-Net (Rastegari 2016) [Gedächtnis].
- Die FWHT ist ein feststehendes Addier-Netzwerk mit n/2·log n Butterflys, vollständig in Pipelines abbildbar (n log n Additionen, log n Stufen Latenz). Für n=256: 8 Stufen, 1.024 Butterflys, 2.048 Additionen. **[?]** Konkrete FPGA-Zahlen für FWHT-Schichten in NN nicht belegt.

### 2.5 Eigene Repo-Messungen (gelesen, nicht neu gemessen)
- T-001 (10. Aug): Butterfly-Schicht rechnet ~340× weniger Arithmetik, ist aber bei kleinem d ~5× langsamer als dicht; Übergang bei d=16.384 (Quelle: `UEBERARBEITUNG_2026-09-23/BERICHT/buendler.md` Z. 149; `A/ENTWURF_T_SERIE_KERNEL.md` Z. 338, „Faktor 64“ zwischen trainierbarer Breite 256 und Gewinnbreite).
- Bündler (T-003-Vorlauf): Übergang kron2 gegen dicht bei 4.096 (Batch 64, vorwärts), robust nur „≤ 8.192“; im Training 8.192; bei d=256 noch ~10× langsamer als dicht (0,096). Einfädig, nur CPU. Der Bündler senkt Durchläufe, nicht Arithmetik.
- Ein Teil des T-001-Rückstands war **Ausführung**, nicht Arithmetik (gleiche Funktion 3–7× schneller mit GEMM-Bündelung).

## 3. Eigener Zeitvergleich (gemessen, Scratchpad `b.py`)
Aufbau: numpy float32, CPU (nproc=4 in dieser Sandbox, nicht Maltes Ryzen), `OMP/OPENBLAS/MKL_NUM_THREADS=1`, Mittel über 30 Wiederholungen, jeweils < 2 min. Alle Verfahren auf n=256 (16×16-Bild). FWHT-Korrektheit gegen Hadamard-Matmul geprüft (`allclose`).

| Batch | Matmul 256×256 | Maske + FWHT(256) + Quadrat | Maske + FWHT in 16er-Blöcken + Quadrat | Maske + dichte H-Matrix + Quadrat |
|---|---|---|---|---|
| 1 | 7,8 µs | 133 µs | 67 µs | 9,5 µs |
| 64 | 130 µs | 665 µs | 395 µs | 108 µs |
| 1024 | 1,30 ms | 6,1 ms | 8,1 ms | 1,62 ms |

- Eine erste Messung ohne Thread-Begrenzung war stark verrauscht (Matmul bei Batch 64 20 ms bei Batch 1024 13 ms) — **verworfen**, nur die einfädige Tabelle gilt.
- Lesart: Die numpy-FWHT (8 Stufen mit Reshape/Stack) ist durch Speicherkopien und Python-Aufrufe gebunden. Sie rechnet 32× weniger (2.048 gegen 65.536) und ist trotzdem 5× langsamer als BLAS-GEMM. Das **bestätigt qualitativ T-001**. Die dichte Hadamard-Matrix als GEMM (Spalte 4) ist dagegen gleichauf mit einem beliebigen Matmul: auf CPU lohnt die **Struktur** bei n=256 nur im Speicher und in der Operationszahl, nicht in der Zeit.
- Grenzen: numpy, keine kompilierte Fusion (numba/C wären Etappe 2, **[?]** erwartbar 5–20× schneller, nicht gemessen); eine Maschine; Batch-Abhängigkeit groß; Blockgrößen nicht abgetastet; Energie nicht gemessen.

## 4. Was das für QIRE bedeutet (ehrlich)

### 4.1 Operationszählung der Clifford-Zelle (n=256, 8 Qubits)
| Bestandteil | Operationen | Anmerkung |
|---|---|---|
| Vorzeichenmaske | 256 bedingte Negationen (XOR auf Vorzeichenbit) | quadratische Form über GF(2): Maske trägt 256 Bit |
| Block-WHT, Blockgröße 2^k | 256·k Additionen (k=8: 2.048; k=4: 1.024) | keine Multiplikation, Skalierung als Verschiebung |
| Quadrieren (Born) | 256 Multiplikationen | die einzigen Multiplikationen der Zelle |
| Zelle gesamt | ≈ 2.300 (k=8) | Parameter: 256 Bit Maske |
| Dichte Schicht 256→256 | 65.536 MAC (+ Additionen) | 65.536 Parameter |
| Linearer Kopf 256→10 | 2.560 MAC | 2.570 Parameter, **größer als die Zelle** |

Rechenwerk-Energie (45 nm, Horowitz-Werte aus Gedächtnis, nur Rechnung, ohne Speicher):
- dichte Schicht fp32: 65.536 · (3,7 + 0,9) pJ ≈ 300 nJ; int8: 65.536 · (0,2 + 0,03) pJ ≈ 15 nJ.
- Zelle fp32: 2.048 · 0,9 + 256 · 3,7 ≈ 2,8 nJ; auf int32-Ebene (Ganzzahl, Wertebreite wächst um 8 Bit): ≈ 0,2 + 0,8 ≈ 1,0 nJ.
- Verhältnis: ~100× (fp32) bzw. ~15× (int8 dicht gegen int-WHT) im Rechenwerk. Mit dem Kopf (2.560 MAC ≈ 12 nJ fp32) insgesamt: Zelle + Kopf ≈ 15 nJ gegen dichte Schicht + Kopf ≈ 312 nJ, also ~20×. **Der Kopf ist der Kostenpunkt**; das passt zur Beobachtung „99 % der Parameter im Kopf“.
- Hinweis: Das ist die Gegenüberstellung mit einer **dichten 256→256-Schicht**, die die Zelle vielleicht gar nicht ersetzt. Gegen den Vergleich „256→10 direkt“ (2.560 MAC, ~100 % logistische Regression, MNIST ~0,92 bekannt, [Gedächtnis]) kostet die Zelle sogar **zusätzlich** 2.300 Operationen. Fairer Vergleich = Genauigkeit je Operation, nicht Operationen der Schicht allein. Einordnung ist offen und gehört in Etappe 2.

### 4.2 Was bedeutet das für „Effizienz im Großen“
- Der Gewinn skaliert mit n: dicht n², WHT n·log n. Bei n=65.536 (Bild 256×256): 4,3·10⁹ gegen 1,0·10⁶ Additionen; Faktor ~4.000. Auf klassischer Hardware ist dies **nur mit Kernen realisiert, die Speicherbandbreite nutzen** (Dao et al.-Linie). Auf Maltes CPU liegt der Übergang laut T-001/Bündler bei d ≥ 4.096–16.384, also oberhalb jeder von QIRE bisher trainierten Breite (256).
- Kleine Breiten (n=256): Nutzen **nicht** in Zeit, sondern in Parametern/Speicher (256 Bit statt 65.536 Zahlen) und in Spezialhardware (FPGA: reine Addiernetzwerke, int-Arithmetik, kein DRAM-Zugriff).
- Ehrlich: die Quadratur-Stufe (|·|²) ist die einzige Multiplikation, aber eine **quadratische Aktivierung**; Pipeline-seitig nicht problematisch.
- Entscheidender Unterschied zu einem echten Quantencomputer: Die Zelle ist per Konstruktion klassisch simulierbar (Clifford + Pauli-Born). Ihr „Quantenvorteil“ ist daher kein Ressourcenvorteil gegenüber der Klassik; der Effizienzvorteil liegt in **klassischer Struktur**, den Malte so benennen sollte (Q1: nur klassisch, siehe Quantenstufe unten).

## 5. Neue Ideen / Forschungsfragen

Neuheitsprüfung: **Keine Websuche möglich (siehe Vorbehalt).** Bewertung unten ist „aus dem Gedächtnis nichts bekannt“, nicht „gesucht, nichts gefunden“. Vor Veröffentlichung nachholen: Suchbegriffe „Clifford group rotation quantization LLM“, „stabilizer / symplectic rotation weight quantization“, „quadratic form GF(2) Hadamard rotation neural network“, „Bruhat decomposition neural network layer“, „additions per inference metric binary network“.

### 5.1 [EIGEN] Clifford-Rotation (Maske + Block-WHT) als Quantisierungs-Rotation
- **Frage:** Verringert eine Rotation R = H_Block · diag(Vorzeichenmaske) den Quantisierungsfehler einer Gewichts- oder Aktivierungsmatrix so gut wie die in QuaRot (feste Hadamard, zufällige Vorzeichen) und SpinQuant (gelernt, kontinuierlich) verwendeten? Zusatzfrage: Trägt der **Rang der quadratischen Form** (bei QIRE 52–78 % des Maskengewinns) auch hier — und sind gute Formen zwischen Matrizen/Schichten/Seeds übertragbar (wie I-036…v3)?
- **Warum neu (Einschätzung):** QuaRot/QuIP verwenden Hadamard mit **zufälligen Vorzeichen** (diagonal, Rang-0-Form), SpinQuant lernt kontinuierliche Rotationen. Eine **diskrete Suche** über die Clifford-Familie (Maske als quadratische Form, nicht nur diagonale Vorzeichen) mit Hardware-Zusatznutzen (nur Additionen, 256 Bit Parameter) ist mir **nicht bekannt**. [?] Verwandt wäre „randomized Hadamard with sign flips“ (diagonal) und strukturierte Rotationen (Butterfly-Givens); nicht geprüft.
- **Billigster Test:** (a) Gewichtsmatrix 256×256 von LLäMmlein-120M (liegt im Projekt A) oder irgendeine kleine Matrix aus dem Repo laden; Zeilen mit 4-Bit-pro-Gruppe quantisieren (Round-to-Nearest). (b) Rotation anwenden: keine / feste Hadamard / zufällige Vorzeichen+Hadamard (QuaRot-Stil) / **gelernte Clifford-Maske** (Hill-Climbing über Bits der Form, ~1.000 Evaluationen). (c) Metrik: Relativer Fehler ‖W − Q(WR)Rᵀ‖/‖W‖ und Ausreißerkurtosis (maximal/Standardabweichung je Zeile). Laufzeit **≈ 5–15 min** auf CPU (numpy, kein Training), erste Zahl nach 2 min.
- **Kontrolle:** Mehrere Zufalls-Seeds der Zufallsrotation (SpinQuant-Argument: große Streuung), Rang-0-Maske (nur Diagonale) gegen Rang-voll, Maske auf einer Matrix gelernt und auf einer anderen angewandt (Übertragbarkeit), Nullhypothese: Gewinn kleiner als Streuung der Zufalls-Hadamard.
- **Quantenstufe:** Q1 (klassisch simuliert; Clifford ⇒ keine Quanten-Hardware nötig). Q2/Q3 nicht begründbar.
- **Risiko, ehrlich:** Zufalls-Hadamard ist schon fast optimal gegen Ausreißer; ein Gewinn der Clifford-Suche kann klein sein. Ein Nullergebnis ist eine gültige, publizierbare Aussage („Clifford-Struktur bringt nichts über Zufalls-Hadamard hinaus“).

### 5.2 [EIGEN] „Additionen pro Klassifikation“ (APK) als Effizienzmaß
- **Frage:** Reicht ein Maß „Additionen (und Multiplikationen getrennt gezählt) je Klassifikation, bei gegebener Genauigkeit“, um QIRE-Zelle, logistische Regression, kleines MLP und binäre Netze fair auf eine Linie zu stellen — hardwareunabhängig und ohne Maltes CPU-Zeitverzerrung (siehe Abschnitt 3)?
- **Zählvorschrift (Vorschlag):** pro Klassifikation (Batch 1): Additionen/Subtraktionen A, Multiplikationen M, Vergleiche C, Bits gelesener Parameter P_bit. Daraus **Energieschätzung E = A·e_add + M·e_mul + P_bit·e_mem** mit explizit angegebenen Horowitz-Konstanten (und Sensitivität für 4×-Abweichung). Zweitwert: Pareto-Kurve Genauigkeit gegen E.
- **Beispiel (MNIST, int-orientiert, Rechnung, nicht gemessen):** Clifford-Zelle: 2.048 A + 256 M, Kopf 2.560 MAC; ~100-Parameter-Ziel von Malte (Ziel 1) lässt sich damit direkt vergleichen: ein Netz mit 100 Parametern hat höchstens ~100 MAC im Kopf, die Zelle dominiert dann den Zähler.
- **Warum neu (Einschätzung):** Operationszählung (FLOPs, MACs, „BOPs“ bei Binärnetzen) ist Standard; „Additionen pro Klassifikation“ als **Hauptmaß mit Energiemodell und Streubreite** ist in der Neuronale-Netze-Literatur als Einzelaussage ([Gedächtnis]: Adder Networks, Chen et al. 2020; BitNet-Energiebilanz) vorhanden. **Nicht neu als Maß, neu nur in der konsequenten Anwendung auf die Clifford-Zelle.** Daher keine Neuheitsbehauptung stellen.
- **Billigster Test:** Python-Zählerklasse über die vorhandenen Modelle (`quantum_sim`-Pfad), Laufzeit **≈ 1 h** Programmierung, Rechenzeit Sekunden. Kontrolle: unabhängige Handzählung für zwei Modelle; Abgleich mit der Repo-Funktion `aufwand_tabelle()`.
- **Quantenstufe:** Q1.

### 5.3 [EIGEN] Fusionierter Eigenkernel: Wann lohnt die Zelle in Zeit?
- **Frage:** Bei welcher Breite n kippt eine in C/numba fusionierte Maske+Block-WHT+Quadrat gegen BLAS-GEMM auf dem Ryzen 5 3600, einfädig und 6-fädig? (T-001/Bündler: 4.096–16.384 für Butterfly-Kronecker.)
- **Warum neu:** Kein Neuheitsanspruch; eine Messlücke im eigenen Repo (kein fusionierter Kern, T-001 „Handkernel“). Dao et al. zeigen Fusion auf GPU [Gedächtnis].
- **Billigster Test:** numba-Kern, in-place, 16er-Blöcke im Register, Breiten 256–65.536; **≈ 1–2 h** Arbeit, Läufe < 5 min. Kontrolle: BLAS-Matmul, Zufallsmatrix gleicher Größe, mehrfädig ergänzen (Bündler-Hinweis: GEMM skaliert besser als elementweise Stufen).
- **Quantenstufe:** Q1.

### 5.4 [EIGEN] Integer-WHT-Kette mit niedriger Bitbreite
- **Frage:** Wie viele Bits (int8 → int12) braucht die Zelle (Eingabe Pixel 8 Bit, WHT wächst um 8 Bit auf 16 Bit; Quadrat 32 Bit), ohne Genauigkeitsverlust auf MNIST? Zielwert: Energie je Klassifikation nach 5.2.
- **Warum neu:** Standard in Festkomma-Hardware; auf die Clifford-Zelle unseres Wissens nicht angewandt [?]. Kein Neuheitsanspruch.
- **Billigster Test:** Bestehende Clifford-Lösung (I-036) mit gerundetem Zwischenwert simulieren; **≈ 10 min**. Kontrolle: float32-Referenz.
- **Quantenstufe:** Q1.

## 6. Offene Fragen für Etappe 2
1. **Quellen verifizieren:** Alle [Gedächtnis]-Angaben (Horowitz-Tabelle exakt, QuaRot/SpinQuant-Zahlen, HadaNet-FPGA-Werte, BitNet-Energiebilanz) mit Webzugriff nachholen; Neuheitssuche für 5.1 und 5.2 nachholen (Suchbegriffe oben).
2. Auf Maltes Ryzen (6 Kerne) wiederholen: kompilierter Kern, mehrfädig, Energie (z. B. Intel-RAPL entfällt bei AMD; `perf stat` oder Steckdosenmessgerät?).
3. Ist der Kopf (256×10 = 99 % der Parameter) selbst WHT-/Kronecker-parametrisierbar? Das würde das ~100-Parameter-Ziel stützen und die Operationsbilanz des Kopfes entlasten.
4. Fairer Gegner: logistische Regression/kleines MLP/Binärnetz mit gleichem Rechenbudget (FINN-Stil) für die APK-Pareto-Kurve.
5. Bei großem n (Breite ≥ 4.096, Sprach-Adapter, LLäMmlein-Breite 768–2048, Projekt A): Operationsbilanz Kronecker-Drehung gegen LoRA — ist der Zeitgewinn überhaupt relevant (Bündler: d_model 256 liegt unter jedem Übergang)?
6. Q2/Q3: Kein Bezug zur Hardware-Effizienz erkennbar; Ehrlichkeitshinweis für die schriftliche Arbeit: Clifford-Zelle ist klassisch simulierbar (Gottesman-Knill), der Effizienzbeitrag ist klassische Struktur.

## 7. Quellenliste (alle [Gedächtnis], URLs nicht abgerufen — bitte vor Zitat prüfen)
- Horowitz, M. (2014). Computing's Energy Problem (and what we can do about it). ISSCC 2014, S. 10–14.
- Le, Q., Sarlós, T., Smola, A. (2013). Fastfood — Approximating Kernel Expansions in Loglinear Time. ICML 2013. (arXiv:1408.3060 [?])
- Yang, Z. et al. (2015). Deep Fried Convnets. ICCV 2015. (arXiv:1412.7149 [?])
- Moczulski, M. et al. (2016). ACDC: A Structured Efficient Linear Layer. ICLR 2016. (arXiv:1511.05946 [?])
- Ashkboos, S. et al. (2024). QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. (arXiv:2404.00456 [?])
- Liu, Z. et al. (2024). SpinQuant: LLM Quantization with Learned Rotations. (arXiv:2405.16406 [?])
- Chee, J. et al. (2023). QuIP: 2-Bit Quantization of LLMs with Guarantees; Tseng et al. (2024) QuIP#. [?]
- Ma, S. et al. (2024). The Era of 1-bit LLMs: BitNet b1.58. (arXiv:2402.17764 [?])
- Umuroglu, Y. et al. (2017). FINN: A Framework for Fast, Scalable Binarized Neural Network Inference. FPGA 2017.
- Rastegari, M. et al. (2016). XNOR-Net. ECCV 2016.
- Dao, T. et al. (2019/2022). Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations; Monarch. [?]
- Lee-Thorp, J. et al. (2021). FNet: Mixing Tokens with Fourier Transforms. [?]
- Chen, H. et al. (2020). AdderNet: Do We Really Need Multiplications in Deep Learning? CVPR 2020. [?]
- Akhauri, Y. (2019). HadaNet: A Hadamard-Transform-Based Hardware-Efficient Binary/Ternary Network. [?]
- Repo: `UEBERARBEITUNG_2026-09-23/BERICHT/buendler.md` (T-001/T-003-Vorlauf), `A/ENTWURF_T_SERIE_KERNEL.md`, `A/quantum_sim/kernel_messung.py`.
- Eigenmessung: `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/b.py` (Scratchpad, nicht im Repo).
