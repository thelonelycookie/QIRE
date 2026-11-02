# k1 — Morphologischer Kasten (Welle 2, 28.10.2026)

Nur Denken und Lesen, nichts gerechnet. Gelesen: KREATIVITAET, SYNTHESE, KERNAUSSAGE, LABORBUCH I-044 bis I-045-v5 (I-047, I-048, I-049). Keine Websuche; Literatur aus dem Gedächtnis, vor Verwendung prüfen. Wirkung überall klassisch („quanten-inspiriert“).
**Ausgeschlossen (schon vergeben):** Hadamard/Rotation + Niedrigbit-PTQ, Hourglass, LRU, Phasengewichte, Entropie-Rang, Monarch, Tensor-Train, Born-Attention (q·k)², Early Exit, Fastfood-FFN.
**Lehre aus I-045-v4/v5:** Gewinn nur bei gleichen FLOPs oder Bytes, Gegner „schmale dichte FFN“, 10 Epochen auf 1,6 MB = Überanpassungsregime. Deshalb mehrere Ideen gegen Überanpassung und für Bytes statt FLOPs.

## A. Die 12–20 vielversprechendsten Ideen (16)

### 1. Schlüssel-Superposition: Schichten teilen ein Gewicht mit Vorzeichen-Schlüsseln
- **Superposition** · Morphologischer Kasten (Superposition × FFN × Bytes) · Mehrere Operatoren liegen in einem Raum übereinander und werden durch orthogonale Schlüssel getrennt gelesen.
- Mechanismus: Alle L Lagen teilen eine FFN-Matrix W; Lage l nutzt W_l = D1_l · W · D2_l (feste oder 1-Bit-gelernte Vorzeichenvektoren, Zusatzkosten n Bit je Lage), optional mit je Lage eigener Permutation. Bytes sinken um fast L-fach; FLOPs bleiben gleich. Bei gleichen Bytes darf das Modell breiter werden (d oder h größer), und das Überanpassungsregime (10 Epochen) wird durch weniger freie Parameter gedämpft. Verwandt: ALBERT, Cheung et al. 2019 („Superposition of many models into one“).
- Langweiligste Erklärung: Es ist nur Gewichtsteilung (ALBERT); die Schlüssel tun nichts (Kontrolle: Teilung ohne Schlüssel).
- Billigster Test: Byte-LM d=128, 4 Lagen, 3000 Schritte, Tuning-Seeds. Arme: (a) normal, (b) geteilt ohne Schlüssel, (c) geteilt mit Schlüsseln, jeweils auch mit breiterer FFN auf gleiche Bytes (a mit h/4 gegen c). Kontrolle: (a) kleines dichtes Modell mit gleichen Bytes. ≈ 8 min bei 1 Thread.
- Vorab: P(grün) 40 %, Jury 4 (anschaulich: „ein Gewicht, viele Lagen, Schlüssel trennen sie“).

### 2. Reed-Muller/Hamming-Codebuch als Vektorquantisierer ohne Codebuchspeicher
- **Fehlerkorrektur** · Morphologischer Kasten (Fehlerkorrektur × Quantisierung × Bytes) · Klassische Codes aus der Quantenfehlerkorrektur (CSS-Konstruktion) sind dichte, per Syndrom schnell dekodierbare Gitter in {±1}ⁿ.
- Mechanismus: Nach der Rotation (Gaussian-artig) werden Gewichte in Blöcke zu 8 (RM(1,3)/Hamming [8,4]) oder 16 gelegt und auf das nächste Codewort ±-skaliert abgebildet. Das ergibt 0,5–1 Bit/Gewicht mit 0 Byte Codebuch und Syndrom-Dekodierung statt Suche. Bei <1 Bit schlägt Skalarrundung (Vorzeichen) in Verzerrung pro Bit, weil die Codeworte den Raum gleichmäßiger überdecken. Verwandt: QuIP# (E8-Gitter), AQLM; der Code statt Gitter ist die Variante.
- Langweiligste Erklärung: Vektorquantisierung ist eben besser als Skalarquantisierung; der Code ist nur ein schlechteres Gitter als E8 (dann ist E8 zu nehmen, kein Befund).
- Billigster Test: (i) Gaussmatrix 512×512: MSE pro Bit für Vorzeichen, Hamming [8,4], RM, zufälliges Codebuch gleicher Größe. (ii) MNIST-MLP 784-512-512-10, Hadamard-Rotation, PTQ bei gleichen Bytes. Kontrolle: Skalar 1 Bit, Zufalls-Codebuch gleicher Größe. ≈ 3 min.
- Vorab: P(grün) 40 %, Jury 4 (Fehlerkorrektur trifft Kompression).

### 3. Skalenabhängige Bitbreite in der Haar-Wavelet-Basis (RG-Grobkörnung)
- **Renormierung** · Morphologischer Kasten (Renormierung × Quantisierung × Bytes) · Wie in der Renormierungsgruppe tragen grobe Skalen die Relevanz, feine Skalen sind irrelevant und werden gröber aufgelöst.
- Mechanismus: Jede Zeile einer Gewichtsmatrix wird per Haar-Transformation (O(n)) zerlegt; die Koeffizienten der groben Skalen bekommen 4–8 Bit, die feinen 1–2 Bit, im Mittel z. B. 2 Bit. Anders als Hadamard (flaches Spektrum) ist Haar skalenordnend und lässt eine begründete Bitallokation zu. Bei gleichen Bytes sollte der Fehler gegenüber gleichmäßiger Bitbreite kleiner sein, falls trainierte Gewichte spektral glatt sind (sind sie nach Training oft nicht).
- Langweiligste Erklärung: Gemischte Bitbreiten helfen immer ein wenig (Bitallokation), die Wavelet-Basis ist nebensächlich (Kontrolle: Zufallsordnung der Koeffizienten bei gleicher Bitverteilung).
- Billigster Test: MNIST-MLP und kleines Byte-LM (1500 Schritte), Haar vs. identische Bitverteilung auf zufälliger Basis vs. gleichmäßig 2 Bit; gleiche mittlere Bits (+Skalenbytes). ≈ 5 min.
- Vorab: P(grün) 25 %, Jury 3.

### 4. Born-Stichproben der FFN-Aktivierungen (Monte-Carlo-Matmul)
- **Messung/Born** · Morphologischer Kasten (Messung × FFN × FLOPs) · Statt alle Amplituden auszurechnen, werden S Neuronen mit Wahrscheinlichkeit ∝ a_j² „gemessen“; der Schätzer ist erwartungstreu.
- Mechanismus: Die zweite FFN-Matrix y = Σ_j a_j W_j wird auf S ≪ h Spalten gerechnet, gezogen mit p_j ∝ a_j² (Gewicht a_j/(S·p_j)). Kosten: volle erste Matmul + S/h der zweiten (≈ 56 % bei S=h/8). Training mit dem Rauschen wirkt zusätzlich regularisierend; zur Inferenz wahlweise exakt oder top-S. Verwandt: randomisierte Matrixmultiplikation (Adelman et al. 2021), CRS.
- Langweiligste Erklärung: Es ist aktivierungs-Sparsität (top-k) mit Rauschen als Dropout; ein dichtes h'≈0,56·h-Modell gleicher FLOPs ist genauso gut.
- Billigster Test: Byte-LM d=128, h=512, 2500 Schritte; Arme: dicht voll, dicht schmal (FLOP-gleich, ≈ 0,56 h), Born-Sampling S=h/8, top-S deterministisch. Kontrolle: Zufallsauswahl mit Gleichverteilung. ≈ 7 min.
- Vorab: P(grün) 25 %, Jury 4.

### 5. KV-Cache als 1-Bit-Hadamard-Skizze der Schlüssel
- **Messung/Born** · Morphologischer Kasten (Messung × Attention × Bytes/Latenz) · Eine Messung in rotierter Basis liefert pro Komponente nur ein Vorzeichen; der Winkel zweier Vektoren lässt sich aus der Hamming-Distanz schätzen.
- Mechanismus: Schlüssel werden nach D·H gespeichert als Vorzeichenbits (+ eine Norm), Scores aus Popcount, Werte mit 4 Bit. KV-Bytes sinken 4–8-fach, bei gleichen Bytes passt ein längerer Kontext oder ein größeres Modell. Bekannt: SimHash/Reformer, KIVI, KVQuant; hier als Byte-Aussage mit Hadamard-Kontrolle.
- Langweiligste Erklärung: Schlüssel-Quantisierung (KIVI) klappt auch ohne Rotation, wenn pro Kanal skaliert wird; die Rotation fügt nichts.
- Billigster Test: kleines Byte-LM (d=128, Kontext 256, 3000 Schritte) normal trainiert, bei Test KV mit 1-Bit-Schlüssel (mit und ohne Rotation, kanalweise Skala) und 4-Bit-Werten; bpc gegen KV-Bytes. Kontrolle: Zufalls-Schlüssel-Projektion statt Hadamard; Tokenfenster verkürzen auf gleiche Bytes. ≈ 8 min.
- Vorab: P(grün) 45 %, Jury 3.

### 6. Adam in rotierter Basis (billige Vorkonditionierung)
- **Unitarität** · Morphologischer Kasten (Unitarität × Optimierer × FLOPs) · Basiswechsel vor der Messung: die richtige Basis macht die Dynamik diagonal.
- Mechanismus: Der Gradient jeder Gewichtsmatrix wird mit festen Zufallsvorzeichen-Hadamard-Matrizen links und rechts rotiert (n log n), Adam arbeitet dort, dann Rückrotation. Die Diagonale von Adam passt besser, wenn Kopplungen verteilt sind (ähnlich SOAP/Shampoo-Eigenbasis, aber ohne Eigenzerlegung). Schnellere Konvergenz je Schritt heißt weniger Trainings-FLOPs bei gleicher Qualität.
- Langweiligste Erklärung: Nur effektiv geänderte Lernrate/Rauschen; Adam in der Standardbasis mit anderer Lernrate holt auf.
- Billigster Test: Byte-LM d=128, 2000 Schritte, Lernrate je Arm auf Tuning-Seeds gesucht; Arme: Adam, Adam-rotiert, Adam mit nur Zufallsvorzeichen (keine Mischung, Kontrolle), Adam mit Zufalls-Orthogonaler. Gemessen: bpc je Schritt und je Trainings-FLOPs inkl. Rotation. ≈ 9 min.
- Vorab: P(grün) 25 %, Jury 3.

### 7. Optimizer-Zustände (m, v) in rotierter Basis mit 4 Bit
- **Dekohärenz** (Rauschen verteilt sich gleichmäßig) · Morphologischer Kasten (Optimierer × Quantisierung × Bytes) · Eine Rotation macht Ausreißer flach, so dass grobe Rundung gleichmäßig wenig schadet.
- Mechanismus: Adam-Zustände m und v werden blockweise rotiert und mit 4 Bit gespeichert (8-Bit-Adam ist Stand der Technik). Trainingsspeicher sinkt 2–4-fach, bei gleichen Bytes passt ein breiteres Modell. v hat große Dynamik: wahrscheinlich Wurzel oder Log-Skala nötig.
- Langweiligste Erklärung: 8-Bit-Adam mit Blockskalierung reicht; die Rotation verbessert nichts (Kontrolle genau dies).
- Billigster Test: Byte-LM 2000 Schritte, Adam voll, 8 Bit blockweise, 4 Bit blockweise, 4 Bit rotiert, jeweils Lernrate gleich. Maß: bpc gegen Zustand-Bytes. ≈ 8 min.
- Vorab: P(grün) 30 %, Jury 3.

### 8. Dropout in der rotierten (Zeiger-)Basis
- **Dekohärenz** · Morphologischer Kasten (Dekohärenz × FFN × Daten) · Dekohärenz zerstört Superpositionen in der bevorzugten (Zeiger-)Basis; Rauschen dort wirkt auf alle Merkmale gleichmäßig.
- Mechanismus: Dropout (oder Gauß) wird nicht auf Einzelneuronen, sondern nach einer festen Hadamard-Rotation der FFN-Aktivierung/des Residuums angewandt und zurückrotiert. Dadurch wird jedes Merkmal gleichmäßig gestört statt einzelne zu löschen. Im Überanpassungsregime (10 Epochen, schmale Netze besser als breite) könnte das die Lücke Test−Train bei gleichen FLOPs und Bytes schließen; Kosten n log n je Token.
- Langweiligste Erklärung: Es ist normales Dropout mit anderer Stärke (Kontrolle: Dropout in Standardbasis mit auf die Lücke abgestimmter Rate).
- Billigster Test: Byte-LM d=128 auf der Hälfte der Trainingsbytes (zum Überanpassen), 4000 Schritte; Arme: kein Dropout, Standard-Dropout (Rate getunt), rotiertes Dropout (Rate getunt), Gauß-Rauschen rotiert. Protokoll: Train-bpc und Test-bpc. ≈ 9 min.
- Vorab: P(grün) 30 %, Jury 3.

### 9. Adiabatische Quantisierung (Homotopie von fp zu 2 Bit)
- **Adiabatik** · Morphologischer Kasten (Adiabatik × Quantisierung × Bytes) · Langsame Änderung des Hamiltonians hält das System im Grundzustand, hier den Gewichten im guten Minimum.
- Mechanismus: Während des Trainings wird w_eff = (1−s)·w + s·Q(w) mit s: 0→1 (Zeitplan nach einer „Lückenregel“: s steigt nur, solange Verlustanstieg unter einer Schwelle bleibt, sonst Pause). Am Ende ist das Modell exakt 2-Bit, der Weg dorthin war gleitend. Bei gleichen Bytes sollte das besser sein als sofortiges Rundungs-QAT oder PTQ. Verwandt: Soft-to-Hard-Quantisierung, BinaryRelax; die Regelung nach „Lücke“ ist die kleine Neuerung.
- Langweiligste Erklärung: Jedes längere QAT mit weichem Übergang ist gut; die Lückenregel gleicht nur ein festes Zeitplanverhältnis (Kontrolle: linearer Zeitplan gleicher Länge).
- Billigster Test: MNIST-MLP (20 Epochen auf Tuning-Seeds) und Byte-LM 2000 Schritte mit 2 Bit: PTQ, QAT hart (STE), linear s, adaptiv s. Gleiche Schritte und Bytes. ≈ 8 min.
- Vorab: P(grün) 45 %, Jury 3.

### 10. Schuss-Augmentierung: Eingabe als Born-Stichprobe
- **Messung/Born** · Morphologischer Kasten (Messung × Daten/Augmentierung × Daten) · Ein Bild ist eine Wahrscheinlichkeitsverteilung über Pixel; jede Messung (Schuss) zieht eine andere Stichprobe.
- Mechanismus: Pro Epoche wird jedes Bild als Bernoulli(p=Pixelwert) oder als S-Schuss-Histogramm neu gezogen. Das ist Datenvermehrung ohne neue Daten, mit Rauschstruktur, die zur Pixelintensität passt (Poisson-artig). Gewinn in der knappen-Daten-Regime (1000–5000 Beispiele). Verknüpft mit I-040 (Information pro Schuss). Bytes und FLOPs der Inferenz unverändert, auch mit Schüssen zur Testzeit prüfbar.
- Langweiligste Erklärung: Beliebiges Eingaberauschen hilft bei wenig Daten; die Bernoulli-Struktur ist egal (Kontrolle: Gauß-Rauschen gleicher Varianz, Pixel-Dropout).
- Billigster Test: MNIST, 2000 Trainingsbeispiele, MLP 784-256-10, 30 Epochen, 3 Tuning-Seeds: ohne, Gauß, Dropout, Bernoulli, S=4/16-Schuss. ≈ 4 min.
- Vorab: P(grün) 30 %, Jury 3.

### 11. Kopf als Messkette: Byte = 8 Qubit-Messungen (bitweise bedingt)
- **Messung/Born** · Morphologischer Kasten (Messung × Kopf × FLOPs) · Ein Zustand auf n Qubits wird durch n aufeinanderfolgende bedingte Einzelmessungen gezogen, statt die ganze 2ⁿ-Verteilung auszurechnen.
- Mechanismus: Der 256-Wege-Softmax wird durch einen Baum aus 8 binären Entscheidungen ersetzt (jede mit eigenen Gewichten je Knotenebene oder Knoten-Einbettung, Eingabe: Zustand + bisherige Bits), Kopf-FLOPs d·8 statt d·256 (32-fach), Kopf-Bytes ähnlich. Bei Byte-Vokabular nur ~5 % der Gesamt-FLOPs (kleiner Gewinn), bei Wortvokabularen 32 k riesig; die Idee zeigt zugleich die Erwartung der Born-Regel bei Mehrzahl-Wahl. Vergleich: hierarchische Softmax (Morin & Bengio), ECOC. Kontrolle Hadamard-Code-Kopf (16–32 Bit, parallel, Fehlerkorrektur bei der Dekodierung).
- Langweiligste Erklärung: Hierarchische Softmax, bekannt, bei kleinem V schlechter als vollständige.
- Billigster Test: Byte-LM d=128, 3000 Schritte; Arme: voller Softmax, Bit-Baum, Hadamard-Code-Kopf (Bits=32, Kreuzentropie über Bits). Maß: bpc und Kopf-FLOPs. Gleiche Gesamt-FLOPs durch Breitenkorrektur. ≈ 7 min.
- Vorab: P(grün) 50 % (bpc gleichauf), P(Gewinn bei gleichen FLOPs, klein) 15 %; Jury 3.

### 12. RG-Curriculum: erst grob, dann fein trainieren
- **Renormierung** · Morphologischer Kasten (Renormierung × Daten × FLOPs) · In der Renormierungsgruppe wird erst bei grober Auflösung die Struktur festgelegt, die feine kommt als Korrektur dazu.
- Mechanismus: Die ersten ~30 % der Schritte laufen mit Kontext 64 (statt 256) oder auf ×2-gepoolten Bytes (mit gleichem Modell, Positionen skaliert), danach voll. Trainings-FLOPs und Zeit pro Schritt sinken am Anfang um ≥ 2–4×; bei gleichen Gesamt-FLOPs sind mehr Schritte möglich. Verwandt: Sequence-Length-Warmup (Li et al. 2022), progressive Sequenzlänge; hier Bezug zur RG und zum Hourglass.
- Langweiligste Erklärung: Kurze Sequenzen sind schlicht billiger (mehr Schritte je FLOP); der RG-Bezug trägt nichts.
- Billigster Test: Byte-LM d=128, Budget in Trainings-FLOPs festgelegt (nicht in Schritten): Arme: konstant 256, Warmup 64→256, Pool-Curriculum. Maß: bpc bei gleichem FLOP-Budget (≈ 7 min).
- Vorab: P(grün) 50 %, Jury 3 (bekannt, aber ehrlich in FLOP-Währung).

### 13. Born-Entropie-Datenauswahl (Datenregime-Idee)
- **Messung/Born** · Morphologischer Kasten (Messung × Daten × Daten) · Wo die gemessene Verteilung maximal unscharf ist, bringt eine weitere Messung am meisten Information.
- Mechanismus: Alle ~10 % der Schritte wird die Vorhersageentropie je Trainingsfenster bestimmt; gesampelt wird mit Wahrscheinlichkeit ∝ Entropie^γ. In den 10 Epochen mit 1,6 MB wären sonst bereits gelernte Fenster verschwendete Schritte. Spart Schritte (FLOPs) bei gleichem Ergebnis; kein Eingriff in das Modell.
- Langweiligste Erklärung: Verlustbasierte Gewichtung (Importance Sampling, Hard-Example-Mining) ist bekannt; es hilft nur im Frühregime oder gar nicht.
- Billigster Test: Byte-LM, 3000 Schritte, Kontrolle zufällig gleichverteilt, Verlust-proportional, Entropie-proportional; Zusatzkosten der Auswertung in FLOPs mitgezählt. ≈ 9 min.
- Vorab: P(grün) 20 %, Jury 2.

### 14. Komplexes Momentum: zwei Zeitskalen in einem Puffer
- **Interferenz** · Morphologischer Kasten (Interferenz × Optimierer × Bytes) · Eine komplexe Amplitude trägt Betrag und Phase; ihre Schwebung bildet zwei Zeitskalen in einem Register ab.
- Mechanismus: Momentum mit komplexem β = r·e^{iφ} (Lorraine et al. 2022: „Complex Momentum“) gibt oszillierendes Gedächtnis, nützlich für Spiele und schmale Täler; hier als Ersatz für Adams m, v (zwei Puffer) durch einen komplexen Puffer = gleiche Bytes. Bytes −25 bis −50 % im Optimizer, bei gleicher Qualität. Gelernt wird, ob es als Adam-Ersatz taugt (Zweifel groß).
- Langweiligste Erklärung: Es ist heavy-ball mit anderer Lernrate; ohne Zweitmoment (v) verliert es gegen Adam.
- Billigster Test: Byte-LM, 2000 Schritte: Adam, SGD+Momentum, komplexes Momentum + RMS-Normierung (ein Puffer), Lernrate je Arm. ≈ 6 min.
- Vorab: P(grün) 10 %, Jury 2.

### 15. Dekohärenz-Annealing der Attention: weich → hart (Top-k)
- **Dekohärenz** · Morphologischer Kasten (Dekohärenz × Attention × FLOPs/Latenz) · Dekohärenz projiziert eine Superposition auf wenige klassische Zeigerzustände.
- Mechanismus: Im Training wird die Softmax-Temperatur gesenkt, bis Attention praktisch top-k (k=8) ist; zur Inferenz nur k Werte lesen. Wirkt erst bei langem Kontext (N ≥ 1024), bei Kontext 256 klein. Verwandt: Sparsemax/Entmax, Top-k-Attention (Gupta et al. 2021).
- Langweiligste Erklärung: Lokale Aufmerksamkeit/Top-k hilft, weil Byte-Texte vor allem lokal sind (Kontrolle: Fenster-Attention gleicher Zahl k).
- Billigster Test: Byte-LM Kontext 512, 2500 Schritte: dicht, Fenster k=16, Top-k (annealed) k=16; Maß: bpc vs. Attention-FLOPs. ≈ 10 min.
- Vorab: P(grün) 20 %, Jury 2.

### 16. Pauli-/XOR-Summen: dünne Summe signierter Permutationen als FFN
- **Unitarität** (Pauli-Basis ist eine unitäre Operatorbasis) · Morphologischer Kasten (Unitarität × FFN × FLOPs) · Jede Matrix lässt sich als Σ c_{a,b}·X^a Z^b zerlegen; jeder Term ist eine signierte Permutation und kostet nur n Operationen.
- Mechanismus: Die FFN-Matrix besteht aus K gelernten Koeffizienten c_k mit festen zufälligen (a_k, b_k): y = Σ_k c_k·(x[i⊕a_k]·(−1)^{⟨i,b_k⟩}). Kosten K·n statt n² (bei K=8: 64× weniger), Parameter K (plus Index). Die Funktionsklasse ist dünn und fest verdrahtet; gewinnt nur, wenn Training in dieser Basis zielgerichtet Eingabe-Ausgabe-Beziehungen findet. Honest: wahrscheinlich nur zufällige dünne Matrix.
- Langweiligste Erklärung: Zufällige dünne Matrix (Sparse Random) mit gleicher Nichtnullzahl ist gleich gut; die Pauli-Struktur trägt nichts.
- Billigster Test: MNIST-MLP (und Byte-LM FFN 2000 Schritte): Pauli-Summe K=16 gegen Zufalls-Sparse gleicher Nichtnullzahl (n·K) gegen dichtes h' mit gleichen FLOPs. ≈ 5 min.
- Vorab: P(grün) 10 %, Jury 3.

## B. Verworfen während der Auswahl (Kurzgrund)
- QAT mit Rotation (Verlängerung von Rang 1): zu nah an I-048, bei Bedarf als eigene Idee nach dem Ergebnis.
- Layer-Norm-frei durch unitäre Residual-Updates (Cayley/Exp): FLOP-Gewinn <1 %, bekannt (NF-Nets, ReZero).
- Kompositions-Einbettung (Tensorprodukt über Nibbles) / Hash-Embedding mit Vorzeichen: Bei V=256 nichts zu holen; nur bei Wortvokabular, im Container nicht testbar.
- Tunneln/Landschaft (Simulated Quantum Annealing, Lookahead): läuft auf SGLD/Warmstarts hinaus, kein Gleiche-FLOP-Test erkennbar.
- Stochastic Depth/Mixture-of-Depths als Tunneln: bekannt, FLOP-Gewinn aber nur mit Router-Kosten.
- Mixup als Superposition von Samples: bekannt, Gewinn klein.

## C. Anhang: ausgefüllte Tabelle (je Zelle ein Stichwort je Ressource: F = FLOPs / B = Bytes / D = Daten / L = Latenz)

Zeilen = Quantenprinzip, Spalten = KI-Baustein. Pro Zelle „F · B · D · L“. Fettmarkierte Zellen sind in Teil A aufgegriffen (Nummer).

| Prinzip | Einbettung | Attention | FFN | Kopf | Optimierer | Daten/Augm. | Quantisierung | Normalisierung |
|---|---|---|---|---|---|---|---|---|
| **Superposition** | F: Mehrfachbyte-Mischung · B: **Schlüssel-Hash (1)** · D: Wort-Mischung · L: keine | F: Multi-Query-Überlagerung · B: geteilte K/V · D: Mix mehrerer Kontexte · L: Cache-Teilung | F: Neuronen-Superposition · B: **Schlüsselteilung (1)** · D: Merkmale > Neuronen · L: gleich | F: Mehrfachkopf-Überlagerung · B: geteilte Ausgabe · D: Label-Mix · L: Mehrfachziel | F: Mehrzeit-Puffer · B: **komplexer Puffer (14)** · D: Gradienten-Mix · L: – | F: Mixup · B: Mix-Speicher · D: Sample-Überlagerung · L: – | F: Mehrbit-Überlagerung · B: geteilte Skalen · D: Kalibrier-Mix · L: – | F: gemeinsame Skala · B: Gruppen-Skala · D: Batch-Mix · L: – |
| **Interferenz** | F: Zufallsmerkmale cos · B: Phasentabelle · D: – · L: – | F: kohärente Summe der Köpfe · B: Phasen-K/V · D: – · L: RoPE-artig | F: Gate aus Differenzen · B: Interferenzglied (a+b)² · D: – · L: – | F: Konstruktive Logit-Summe · B: Ensemble in einem Kopf · D: – · L: – | F: **Schwebung (14)** · B: Phasenpuffer · D: Rauschauslöschung · L: – | F: Gegenphasen-Paare · B: – · D: Kontrastpaare · L: – | F: Fehlerauslöschung · B: Dither · D: – · L: – | F: Phasen-Normierung · B: – · D: – · L: – |
| **Messung/Born** | F: Schuss-Einbettung · B: Stichproben-Tabelle · D: **Schuss-Aug. (10)** · L: – | F: Top-k nach |a|² · B: **1-Bit-Skizze (5)** · D: Stichprobe der Tokens · L: Popcount | F: **Born-Sampling (4)** · B: aktive Neuronen · D: – · L: weniger Zugriffe | F: **Messkette (11)** · B: Bit-Baum · D: Schuss-Training · L: Frühabbruch der Kette | F: Stichproben-Gradient · B: – · D: **Entropie-Auswahl (13)** · L: – | F: Binarisierung · B: – · D: **Bernoulli (10, 13)** · L: – | F: stochastische Rundung · B: Bit je Gewicht · D: Kalibrier-Stichprobe · L: – | F: Messung der Norm · B: – · D: – · L: – |
| **Verschränkung** | F: Produkt-Einbettung · B: Tensor-Faktor · D: Bit-Paare · L: – | F: Rang-Messung · B: Kopfrang adaptiv · D: Schmidt-Rang · L: dyn. Rang | F: MPS/TT (I-047, vergeben) · B: Bonddimension · D: – · L: – | F: Kopplung Kopf-Tiefe · B: gemeinsame Nachbarn · D: – · L: – | F: Kronecker-Vorkond. · B: Faktorzustand · D: – · L: – | F: gekoppelte Paare · B: – · D: Paar-Beispiele · L: – | F: gemeinsame Gruppenquantisierung · B: Vektor-Q · D: – · L: – | F: Korrelations-Whitening · B: Kovarianz-Faktor · D: – · L: – |
| **Unitarität** | F: orthogonale Init · B: – · D: – · L: – | F: orthogonales V/O · B: faltbar · D: – · L: – | F: **Pauli-Summe (16)** · B: Signaturen · D: – · L: schnell (Perm.) | F: orthogonaler Kopf · B: – · D: – · L: – | F: **rotiertes Adam (6)** · B: Matrixlose Basis · D: schnellere Konv. · L: – | F: Rotationsaug. · B: – · D: Symmetrie-Orbits · L: – | F: Rotation vor Q (I-048, vergeben) · B: – · D: – · L: – | F: Norm-erhaltend (kein LN) · B: – · D: – · L: spart LN-Kern |
| **Renormierung** | F: **Grobkörnung (12)** · B: Skalen-Einbettung · D: **Curriculum (12)** · L: – | F: Hourglass (I-049, vergeben) · B: Pooling-KV · D: – · L: kürzere Folge | F: Skalenlagen · B: **Haar-Bits (3)** · D: – · L: – | F: Mehrskalen-Kopf · B: – · D: – · L: – | F: Lernraten pro Skala · B: – · D: Skalenschritte · L: – | F: Pooling-Daten · B: – · D: **Grob-zu-fein (12)** · L: – | F: **Skalenbits (3)** · B: Wavelet-Skala · D: – · L: – | F: Fixpunkt-Skalierung · B: – · D: – · L: – |
| **Fehlerkorrektur** | F: Code-Einbettung · B: Hamming-Wort · D: – · L: – | F: redundante Köpfe · B: Parität · D: – · L: Ausfall tol. | F: Parität in FFN · B: Ersatz-Neuronen · D: – · L: – | F: **ECOC-Kopf (11)** · B: Hadamard-Code · D: robuste Labels · L: parallel | F: Fehlerrückführung · B: Rest speichern · D: – · L: – | F: Rausch-Robustheit · B: – · D: Label-Codes · L: – | F: **Code-Codebuch (2)** · B: Syndrom · D: – · L: schnelle Dekodierung | F: Prüfsumme der Skala · B: – · D: – · L: – |
| **Dekohärenz** | F: Rauschen auf Einbettung · B: – · D: Rausch-Aug. · L: – | F: **Weich-zu-hart (15)** · B: Top-k-KV · D: – · L: weniger Lesen | F: **Dropout rotiert (8)** · B: – · D: Regularisierung · L: – | F: Temperatur → 0 · B: – · D: – · L: argmax | F: Rauschen im Zustand · B: **Zustand 4 Bit (7)** · D: – · L: – | F: Dephasing-Aug. · B: – · D: Rauschfamilie · L: – | F: Rundungsrauschen · B: Gruppenbits · D: – · L: – | F: Rausch-Floor · B: – · D: – · L: – |
| **Tunneln** | F: Sprung im Vokabular · B: – · D: seltene Tokens · L: – | F: Layer-Skip · B: – · D: – · L: Token-Sprung | F: Neuron-Auslassen · B: – · D: – · L: bedingte Rechnung | F: Frühausgang (I-044-Linie, vergeben) · B: – · D: – · L: kürzer | F: Warmstarts · B: – · D: – · L: – | F: seltene Beispiele · B: – · D: Rand der Verteilung · L: – | F: Rundung über Barrieren · B: – · D: – · L: – | F: – · B: – · D: – · L: – |
| **Adiabatik** | F: langsames Einbetten · B: – · D: – · L: – | F: weich → hart (Hamiltonian) · B: – · D: – · L: – | F: Struktur gelernt → fest (Fastfood-Init → dicht) · B: – · D: – · L: – | F: Kopf-Anneal · B: – · D: – · L: – | F: Lückenregel LR · B: – · D: – · L: – | F: Curriculum-Interpolation · B: – · D: Schwierigkeit · L: – | F: **s: fp→2 Bit (9)** · B: Endbits · D: – · L: – | F: Norm-Anneal · B: – · D: – · L: – |

## D. Beobachtungen für die Konvergenz
- Gruppierung nach Währung: **Bytes** (1, 2, 3, 5, 7, 9), **FLOPs** (4, 11, 12, 15, 16), **Daten/Regime** (8, 10, 13), **Optimierer** (6, 7, 14). FLOP-Siege sind nach I-045-v4 am schwersten; Bytes-Siege am wahrscheinlichsten.
- Sinnvolle Bündelung für einen Agenten: {1, 9, 2, 3} als „Byte-Paket 2“ (alles Gewichte-Bytes, gleiche Gegner wie I-048), {6, 7} als Optimierer-Paket, {8, 10, 13} als Daten/Regime-Paket.
- Meine Top 3 nach P × Jury: **1 (Schlüssel-Superposition)**, **9 (adiabatische Quantisierung)**, **2 (Code-Codebuch)**; knapp dahinter 5 (KV-1-Bit) und 12 (RG-Curriculum).
