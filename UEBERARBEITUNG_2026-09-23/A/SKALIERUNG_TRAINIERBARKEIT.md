# Skalierung und Trainierbarkeit der QIRE-Schicht (Barren Plateaus, Lie-Algebra)

Stand 7. Oktober 2026. Skript: `/tmp/claude-0/-home-user-QIRE/80103e8f-2820-5235-83e7-290e12149136/scratchpad/dla.py` (DLA, Gradientenvarianz Qubit 0) und `dla2.py` (Qubit 1, Eingabe |0..0>). Beide einfaedig, zusammen ca. 340 s CPU. Code der Schicht: `quantum_sim/butterfly_layer.py`.

## 1. Literatur in eigenen Worten

- **McClean u. a. 2018** (Ausgangspunkt, Nature Comm. 9, 4812): Bei zufaellig initialisierten, tiefen Schaltungen faellt die Varianz des Gradienten exponentiell mit n.
- **Grant u. a. 2019** ([arXiv:1903.05076](https://arxiv.org/abs/1903.05076), Quantum 3, 214): Identitaets-Init. Man waehlt einen Teil der Winkel zufaellig und die uebrigen so, dass die Schaltung als Folge flacher Bloecke zusammen die Identitaet ergibt. Beim ersten Schritt ist die wirksame Tiefe dann klein, man startet nicht im Plateau. Das Training kann spaeter trotzdem hineinlaufen.
- **Cerezo u. a. 2021** ([Nature Comm. 12, 1791; PMC7979934](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979934/), [arXiv:2001.00550](https://arxiv.org/abs/2001.00550)): Es kommt auf die Kostenfunktion an. Globale Observable geben schon bei flachen Schaltungen exponentiell verschwindende Gradienten. Lokale Observable geben bei Tiefe O(log n) hoechstens polynomiell kleine Gradienten.
- **Larocca u. a. 2022** ([Diagnosing barren plateaus, arXiv:2105.14377](https://arxiv.org/abs/2105.14377); [Theory of overparametrization, arXiv:2109.11676](https://arxiv.org/abs/2109.11676)): Trainierbarkeit haengt von der Steuerbarkeit ab, diagnostizierbar ueber die dynamische Lie-Algebra (DLA) der Erzeuger. Ab einer kritischen Parameterzahl, die an dim(DLA) haengt, verschwinden falsche lokale Minima (Ueberparametrisierung).
- **Fontana u. a. 2024** ([Adjoint is all you need, PMC11341719](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11341719/), Nature Comm.): Gradientenvarianz skaliert (fuer Observable in der DLA) invers mit der DLA-Dimension.
- **Ragone u. a. 2024** ([arXiv:2309.09342](https://arxiv.org/abs/2309.09342), Nature Comm. 2024, [DOI 10.1038/s41467-024-49909-3](https://doi.org/10.1038/s41467-024-49909-3)): Exakte Formel fuer die Verlustvarianz hinreichend tiefer Schaltungen: Var ~ (Reinheit von Zustand in der DLA) x (Reinheit der Observablen in der DLA) / dim(DLA). Das vereint alle Plateau-Quellen (Ausdrucksstaerke, Observable, Eingangszustand, Rauschen). Voraussetzung ist "tief genug", naemlich nahe an einem Design der DLA-Gruppe.
- **Cerezo u. a. 2025** ([arXiv:2312.09121](https://arxiv.org/abs/2312.09121), Nature Comm. 16, 7907): Wo die Landschaft nachweislich plateaufrei ist, liegt die Dynamik meist in einem klein-dimensionalen, klassisch beschreibbaren Unterraum. Mit klassischen Daten aus einer Vorphase laesst sich der Verlust dann oft klassisch simulieren. Das ist Evidenz, kein allgemeiner Beweis.

**Vorhersage fuer unsere Schaltung** (Ein-Qubit-Drehungen, fester CNOT-Ring, Z-Auslese): Eine einzelne Lage ohne Ring hat DLA = n Kopien von su(2), dim 3n. Das waere trainierbar, wenn die Auslese lokal bliebe. Der Ring macht aber zweierlei. (a) Er mischt die Erzeuger der Lagen, die DLA waechst auf alles (Abschnitt 2). (b) Er verwandelt die Auslese: Rueckwaerts durch den Ring ist <Z_0> ein Z-String aus n-1 Qubits (Z_1...Z_{n-1}, Qubit 0 faellt heraus), also eine fast globale Observable. Nach Cerezo 2021 ist dann Var ~ 2^(-n) zu erwarten, schon bei einer Lage. Die Formel "Var ~ 1/dim(DLA)" gilt nur fuer sehr tiefe Schaltungen und waere hier 4^(-n). Mit L = 1..5 und 2nL Parametern (viel weniger als 4^n) ist die Schaltung weit von einem Design entfernt. 1/dim(DLA) ist hier nur eine Untergrenze, keine Vorhersage.

## 2. Rechnung A: dim(DLA)

Erzeugermenge: Mit U = C R_L C R_(L-1) ... C R_1 gilt U = (C R_L C^-1)(C^2 R_(L-1) C^-2)...(C^L R_1 C^-L) C^L. Die Erzeuger der Lage l sind also C^l {Y_q, Z_q} C^-l fuer l = 1..L (C = CNOT-Ring, Clifford, bildet Pauli-Strings auf Pauli-Strings ab). Die DLA ist der Kommutator-Abschluss, exakt ueber Pauli-Strings (Bitmasken, kein Rangverfahren noetig, weil der Abschluss von Pauli-Mengen von Pauli-Strings aufgespannt wird). Grenzen siehe Abschnitt 5.

| n | Ordnung des Rings | L=1 | L=2 | L=3..8 (alle Potenzen) | su(2^n) = 4^n-1 |
|---|---|---|---|---|---|
| 2 | 3 | 6 | 15 | 15 | 15 |
| 3 | 7 | 9 | 63 | 63 | 63 |
| 4 | 15 | 12 | 255 | 255 | 255 |
| 5 | 21 | 15 | 1023 | 1023 | 1023 |
| 6 | 21 | 18 | 4095 | 4095 | 4095 |

Befund: L=1 gibt 3n, ab L=2 ist die DLA die volle su(2^n) (Dimension 4^n-1). Die Schaltung ist ab zwei Lagen im Lie-Algebra-Sinn voll steuerbar. Das ist Ausdrucksstaerke, nicht Trainierbarkeit: die Parameter sind nur 2nL, die Familie ist daher trotzdem winzig gegenueber der Gruppe.

## 3. Rechnung B: Gradientenvarianz

Var von d<Z_0>/d(theta) per exakter Parameter-Shift-Regel (Ry), 200 Stichproben je Punkt, Winkel gleichverteilt, Eingabe reeller Gauss-Vektor normiert (bzw. |0..0>). Varianz in der Tabelle, Fit log2 Var = c - a n.

| n | L=1, Ry q1, 1. Lage | L=3, Ry q0, 1. Lage | L=3, Ry q1, letzte Lage | L=5, Ry q0, 1. Lage | L=5, Ry q1, letzte Lage |
|---|---|---|---|---|---|
| 2 | 3,25e-01 | 1,15e-01 | 2,11e-01 | 1,38e-01 | 2,03e-01 |
| 4 | 1,27e-01 | 3,23e-02 | 4,75e-02 | 3,09e-02 | 5,74e-02 |
| 6 | 3,14e-02 | 9,34e-03 | 1,32e-02 | 6,57e-03 | 1,61e-02 |
| 8 | 8,61e-03 | 1,90e-03 | 4,06e-03 | 1,71e-03 | 4,82e-03 |
| 10 | 2,24e-03 | 4,90e-04 | 9,83e-04 | 5,03e-04 | 9,80e-04 |
| **a (n=2..10)** | 0,94 | 1,01 | 0,96 | 1,02 | 0,96 |
| a (n>=5) | 0,99 | 1,04 | 0,97 | 0,98 | 0,97 |

Eingabe |0..0> (dla2.py): a = 0,99 (L=1), 1,06/1,02 (L=3, erste/letzte Lage), 1,01/0,95 (L=5). Also derselbe Abfall ohne zufaellige Eingabe.

- **Abfall: Var ~ 0,45 * 2^(-a n) mit a = 1,0 +- 0,05**, unabhaengig von L (1, 3, 5) und von der Lage des Parameters. Streuung der Steigungen ueber Varianten: 0,94 bis 1,06. Mit 200 Stichproben ist die Varianz je Punkt auf ca. +-10 % genau (Schaetzung 1/sqrt(R/2)), die Einzelsteigung entsprechend auf wenige 0,01 bis 0,05. Eine Aussage "a = 1,00" ist nicht gedeckt, "a nahe 1, nicht 2" schon.
- Bemerkenswert: a = 1, nicht 2. Die Vorhersage 1/dim(DLA) = 4^(-n) (a=2) trifft nicht zu, weil die Schaltung nicht tief genug ist (Abschnitt 1).
- **Nullgradienten durch den Ring**: Der Parameter Ry auf Qubit 0 in der letzten Lage (und bei L=1 ueberhaupt) hat exakt Gradient 0 (Varianz ca. 1e-15, Rechenrauschen von float32). Grund: Rueckwaerts durch den Ring wird Z_0 zu Z_1...Z_(n-1), Qubit 0 taucht nicht mehr auf. Das sind pro Schaltung nur wenige Parameter, aber die Auslese <Z_0> haengt von der letzten Lage an Qubit 0 nicht ab. (Fuer die anderen Auslesen <Z_q> gilt es analog fuer ein anderes Qubit.)
- Frueheres Ergebnis zur Kernkonzentration, 1,24 * 2^(-0,96 n), liegt auf derselben Linie: Zustaende konzentrieren mit Rate nahe 1 pro Qubit, Gradienten ebenso.
- Q-005 (`quantum_sim/barren_plateaus.py`) wurde hier **nicht** ausgefuehrt. Meine Messung ist eine eigene, kleinere Rechnung, kein Ersatz fuer den vorregistrierten Lauf (Arme ohne CNOT, einlagig).

## 4. Deutung fuer QIRE

- **Ab wann wird Gradiententraining schwierig?** Die Standardabweichung des Gradienten ist ca. 0,67 * 2^(-n/2). Dann: n=10 ca. 2e-2, n=16 ca. 3e-3, n=20 ca. 6e-4, n=30 ca. 2e-5, n=40 ca. 6e-7 (Extrapolation des Fits, nicht gemessen ueber n=10). In klassischer float32-Rechnung gibt es kein Shot-Rauschen, der Gradient ist exakt berechenbar. Die Grenze ist also Mini-Batch-Rauschen und Optimiererempfindlichkeit, nicht Messstatistik. Grobe Faustregel, nicht gemessen: ab n ca. 16 bis 20 (Gradient unter ca. 1e-3) wird es spuerbar schwerer, um n=40 reicht float32-Genauigkeit (6e-8) nicht mehr. Das ist Schaetzung, kein Messwert.
- **Praxis im Projekt**: Trainierbar ist auf dieser Hardware n=8 (d=256). Dort ist Std ca. 0,04, kein Problem. Das Problem "Faktor 64 zwischen Struktur-Vorteil (d=16384, n=14) und Trainierbarkeit" ist eine Speicher-/Rechengrenze, kein Plateau: bei n=14 ist Std ca. 4e-3, noch trainierbar. Das Plateau kommt erst deutlich darueber.
- **Hilft Identitaets-Init (Grant)?** Mit Winkeln 0 ist die Schaltung bis auf den festen CNOT-Ring die Identitaet; der Ring selbst wird mit einer Gegen-Lage nicht aufgehoben, weil er Teil jeder Lage ist (C^L, Ordnung 3 bis 21). Eine Identitaets-Init muesste die Ringe paaren (C-Lage gefolgt von der Gegenlage mit C^-1, die QIRE nicht hat) oder den Ring ausschalten. Ich habe sie **nicht gemessen**. Theoretisch hilft sie beim Start (Block flach), gegen die globale Observable Z_1...Z_(n-1) hilft sie nur, soweit der Anfangszustand nahe bei eigenen Eigenzustaenden liegt.
- **"Einfach mehr Qubits"**: bringt kein Training-Signal, sondern nimmt ab, mit 2^(-n/2) in der Std. Die Rechenkosten steigen mit n 2^n; die Ausdrucksstaerke (volle su(2^n)) ist da, aber mit 2nL Parametern nutzt man kaum etwas davon. Naheliegende Abhilfe laut Cerezo 2021: lokale Auslese ohne Ring-Rueckwaertskegel, oder flachere Kopplung (nur benachbarte, nicht ring-geschlossene Ketten), die Kegel begrenzt. Das ist hier nicht getestet.
- **Cerezo 2025**: Wo die Landschaft plateaufrei ist, ist sie meist klassisch simulierbar. Fuer QIRE passt das: QIRE wird ohnehin klassisch als Butterfly-Netz mit O(d log d) gerechnet und laeuft nicht auf Quantenhardware. "Trainierbar impliziert klassisch simulierbar" ist fuer ein quanteninspiriertes Verfahren kein Nachteil, es ist die Betriebsart. Es bleibt der Vorbehalt: Die Aussage betrifft die Simulierbarkeit des Verlusts; sie sagt nicht, dass die Schicht besser lernt als eine klassische Vergleichsarchitektur. Das bleibt eine Frage der Kontrollen (Q-020).

## 5. Ehrliche Grenzen

- Die DLA ist Pauli-String-exakt, aber nur n=2..6 (4^n waechst; Abschluss O(M^2) je Runde). Aussagen fuer n>6 sind Extrapolation (Muster 3n, dann 4^n-1 ab L=2 durchgehend).
- DLA-Dimension heisst Steuerbarkeit der Gruppe, nicht Erreichbarkeit mit 2nL Winkeln. Die Ragone-Formel setzt "tief genug" voraus; unsere L=1..5 sind es nicht.
- Eingabe: reelle Vektoren (die Schicht nimmt reelle Merkmale). Die Verteilung "Gauss, normiert" ist Haar-nah nur fuer reelle Zustaende, nicht fuer komplexe. Ragones Reinheitsfaktor des Zustands in der DLA wurde nicht ausgerechnet.
- Nur Gradient eines Parameters je Variante, Parameter-Shift auf Ry, nicht auf Rz. Qubit-0-Varianten der letzten Lage sind Nullgradienten und gehen nicht in die Fits ein (Tabelle zeigt q1). 200 Stichproben, eine Zufallsfolge je Punkt, kein Bootstrap-Intervall.
- n nur bis 10 gemessen. Alle Zahlen jenseits (Std fuer n=14..40) sind Extrapolation mit a = 1.
- Die Schwellen "ab n ca. 16 bis 20" sind Schaetzung ohne Trainingsversuch; ein echter Test waere ein Trainingslauf mit n = 10, 12, 14 auf einer festen Aufgabe.
- Nicht ausgefuehrt: Q-005, Identitaets-Init, lokale Auslese. Nichts committet.

---
**Bewertung Opus (07.10.):** Wichtigster Befund ist Punkt „Rückwärts-String“: Der CNOT-Ring vor der
Auslese macht aus dem lokalen ⟨Z_0⟩ fast eine *globale* Observable (Z-String). Das erklärt a ≈ 1 schon
bei L = 1 und den exakt verschwindenden Gradienten von Qubit 0 in der letzten Lage — ein Designfehler,
der billig zu beheben ist. Vorgeschlagener Folgetest **S1b** (billig, kein Training): dieselbe Varianz-
messung mit (i) Ring nur *zwischen* den Lagen, nicht vor der Auslese, (ii) Kette statt Ring, (iii)
echte Bildeingaben statt Zufallsvektoren (Zufallsvektoren sind selbst schon „konzentriert“). Erwartung:
(i)/(ii) verschieben die trainierbare Breite um mehrere Qubits. Skripte: `skalierung/dla*.py`.

---

## S1b: CNOT-Varianten x Eingabe (Folgetest, 07.10.)

Skript `skalierung/s1b.py` (Rohdaten `skalierung/s1b_ergebnis.json`, Log `s1b_log.txt`). Methode wie `dla.py`: Var von d<Z_0>/dtheta per Parameter-Shift an Ry auf Qubit 0, n = 2..10, L = 1, 3, 5, 200 Stichproben, Winkel gleichverteilt, Fit log2 Var = c - a n. Neu: Bootstrap-95 %-Intervall fuer a (1000 Ziehungen mit Zuruecklegen der 200 Gradienten je n). Gemessen werden zwei Parameter: **letzt** = Ry q0 der letzten Lage (der Parameter, der in V0 exakt null ist), **erst** = Ry q0 der ersten Lage. Bei L=1 sind beide dasselbe. Theta und Eingaben sind fuer alle Varianten gleich (gepaart). V0 reproduziert `ButterflyQuantumLayer.forward` bis zur letzten Stelle (geprueft). Eine Kontrolle: Die Kette V2/V3 liefert fuer <Z_0> exakt dieselben Werte (Qubit 0 ist in der Kette nur Kontrolle, Z_0 kommutiert mit jedem CNOT, der q0 als Kontrolle hat), V3 ist hier also keine eigene Messung, nur die Bestaetigung.

Varianten: V0 Ring nach jeder Lage; V1 Ring nur zwischen den Lagen; V2 Kette (q -> q+1, q = 0..n-2) nach jeder Lage; V3 V1 + V2. Eingaben: (a) Zufallsvektor, normiert; (b) MNIST (200 zufaellige Bilder je n), 2D-Flaeche per Flaechenmittelung (n <= 9, z. B. 16x16 fuer n=8) bzw. bilinear (n=10: 32x32) auf 2^ceil(n/2) x 2^floor(n/2) gebracht, flach, normiert.

### Ergebnis (a mit 95-%-Bootstrap-KI; Var bei n=8 und n=10; Nullanteil = Anteil der Stichproben mit |g| < 1e-6, gemittelt ueber n=2..10)

| Eingabe | Variante | Parameter | L | a [KI] | Var(n=8) | Var(n=10) | Nullanteil |
|---|---|---|---|---|---|---|---|
| Zufall | V0 | erst | 3 | 1,01 [0,97; 1,04] | 1,8e-3 | 4,4e-4 | 0 |
| Zufall | V0 | erst | 5 | 1,06 [1,02; 1,09] | 1,9e-3 | 4,5e-4 | 0 |
| Zufall | V0 | letzt | 3, 5 | (Rauschen, ca. 1e-15) | 7e-16 | 6e-16 | **1,00** |
| Zufall | V0 | L=1 | 1 | (Rauschen) | 7e-16 | 6e-16 | **1,00** |
| Zufall | V1 | L=1 | 1 | 0,95 [0,92; 0,99] | 8,8e-3 | 1,9e-3 | 0 |
| Zufall | V1 | letzt | 3 | 0,96 [0,92; 1,00] | 3,9e-3 | 1,1e-3 | 0 |
| Zufall | V1 | letzt | 5 | 0,99 [0,96; 1,02] | 3,5e-3 | 9,1e-4 | 0 |
| Zufall | V2 = V3 | letzt | 3 | 0,94 [0,91; 0,98] | 4,2e-3 | 1,4e-3 | 0 |
| Zufall | V2 = V3 | letzt | 5 | 0,97 [0,94; 1,01] | 4,6e-3 | 1,1e-3 | 0 |
| MNIST | V0 | erst | 3 | 0,98 [0,94; 1,01] | 1,9e-3 | 5,3e-4 | 0 |
| MNIST | V0 | erst | 5 | 1,01 [0,97; 1,05] | 1,6e-3 | 4,4e-4 | 0 |
| MNIST | V0 | letzt / L=1 | 1, 3, 5 | (Rauschen, ca. 1e-15) | 6e-16 | 6e-16 | **1,00** |
| MNIST | V1 | L=1 | 1 | 0,36 [0,33; 0,39] | 5,5e-2 | 4,7e-2 | 0 |
| MNIST | V1 | letzt | 3 | 0,67 [0,62; 0,72] | 8,8e-3 | 5,3e-3 | 0 |
| MNIST | V1 | letzt | 5 | 0,92 [0,89; 0,96] | 4,0e-3 | 1,4e-3 | 0 |
| MNIST | V2 = V3 | L=1 | 1 | 0,36 [0,33; 0,38] | 5,5e-2 | 4,7e-2 | 0 |
| MNIST | V2 = V3 | letzt | 3 | 0,27 [0,23; 0,30] | 5,8e-2 | 5,3e-2 | 0 |
| MNIST | V2 = V3 | letzt | 5 | 0,35 [0,32; 0,39] | 3,5e-2 | 2,6e-2 | 0 |

Weitere Zeilen (erster Parameter in V1..V3, L=3, 5) liegen alle in `s1b_log.txt`; sie weichen nicht qualitativ ab (Zufall: a = 0,96 bis 1,06; MNIST V1 erst: 0,77 (L=3), 0,96 (L=5); MNIST V2/V3 erst: 0,30 (L=3), 0,36 (L=5)). Die a-Werte der V0-"letzt"-Zeilen (0,2 bis 0,3) sind Fits an Rauschen von 1e-15 und bedeuten nichts.

### Lesart

1. **Nullgradient ist reiner Ring-Effekt, und weg.** V0: Ry q0 der letzten Lage hat in 100 % der Stichproben Gradient null (beide Eingaben, alle n, L). V1, V2, V3: 0 %. Die Kosten der Korrektur sind null Parameter und null Gatter (eher eines weniger).
2. **Hypothese 1 (V1/V3 haben kleineres a als V0): fuer Zufallsvektoren nein.** a bleibt 0,94 bis 1,06 in allen Varianten (KIs ueberlappen mit V0, V0-erst = 1,01/1,06). Die Varianz bei n=10 steigt bei V1/V2 gegenueber V0-erst allenfalls um den Faktor 2 bis 3 (9e-4 bis 1,4e-3 gegen 4,5e-4), die Steigung nicht. Grund (Deutung, nicht separat geprueft, aber mit V1 L=1 vertraeglich): V1 mit L=1 hat gar kein CNOT und liefert trotzdem a = 0,95. Bei Zufallsvektor sitzt die Konzentration in der **Eingabe**: das Randmoment von Qubit 0 eines Zufallszustands fluktuiert mit 2^(-n/2), so schon in der Varianz 2^(-n). Der Ring ist hier nicht die Ursache von a = 1. Das revidiert die Lesart in Abschnitt 3/Bewertung Opus teilweise: Der Ring erzeugt die exakten Nullen, aber nicht den Abfall. Der Befund "a ca. 1" gilt fuer Zufallseingaben und ist durch die Architekturaenderung nicht verschiebbar.
3. **Hypothese 1 fuer echte Bilder: ja, deutlich.** MNIST V0: a = 0,98 bis 1,01. MNIST V2/V3 (Kette): a = 0,27 bis 0,36 [KI ca. +-0,04], Varianz bei n=10 etwa 2,6e-2 bis 5,3e-2 gegen 4,4e-4 bei V0: Faktor 60 bis 120. Extrapoliert (nur Fit-Fortsetzung, nicht gemessen): bei a = 0,3 Std ca. 0,2 bei n=10, ca. 0,09 bei n=20 gegen 6e-4 (a=1). V1 (Ring zwischen den Lagen) ist dazwischen und von L abhaengig: a = 0,36 (L=1, ohne CNOT), 0,67 (L=3), 0,92 (L=5): Der Ring zwischen den Lagen mischt die Bildstruktur mit wachsender Tiefe weg, die Kette nicht. Mit V1 bei L=5 ist der Gewinn gegenueber V0 nur noch ein Faktor 3 in der Varianz (Var10 1,4e-3), die Steigung bleibt 0,92 [0,89; 0,96].
4. **Hypothese 2 (echte Bilder haben groessere Varianz als Zufallsvektoren): nur fuer V1 bei kleinem L und fuer V2/V3.** Bei V0 sind Bilder und Zufall gleich (Var10 4,4e-4 gegen 4,5e-4, a = 1,0). Bei V2/V3 ist die Varianz mit Bildern 25- bis 50-mal groesser bei n=10 (2,6e-2 gegen 1,1e-3). Warum, ist nicht ausgemessen; eine naheliegende Deutung ist, dass Bildvektoren (alle Pixel >= 0, grosse gemeinsame Komponente) nicht Haar-artig sind und eine Randverteilung von Qubit 0 behalten, die nicht gegen 1/2 konzentriert. Bei V0 zerstoert der Ring-Rueckweg diese Struktur, weil <Z_0> dort ein globaler Z-String ist.
5. **Zusammen:** Kette (V2/V3) + Bilder ist die einzige Kombination mit a deutlich unter 1 (ca. 0,3). Alle anderen liegen bei ca. 1. Ob das beim Training hilft, ist offen: Eine grosse Gradientenvarianz ist nicht gleich ein lernbares Signal (siehe Grenzen).

### Deutung fuer Q-020 (kurz)

- **V1 ist eine andere Funktion als V0.** Zu klaeren war, ob das bei Q-020 zaehlt. Bei L=1 ist der CNOT-Ring eine Permutation der Basiszustaende nach den Rotationen. Die Born-Wahrscheinlichkeiten sind dieselben, nur umsortiert; die Auslese <Z_0> ist mit Ring ein Z-String in den umsortierten Wahrscheinlichkeiten, ohne Ring die Randverteilung von Bit 0. Fuer eine **Auslese, die alle Born-Werte nutzt** (volle Wahrscheinlichkeitsverteilung, lineare Auslese ueber alle 2^n Werte), aendert der Ring bei L=1 nichts, ausser der Reihenfolge, die ein lineares Gewicht ohnehin lernt: die Armfunktion ist dann dieselbe Funktionsklasse. Bei der <Z_q>-Auslese ist es nicht dieselbe Funktion (Z_q gegen Z-String), und fuer L>=2 steht der Ring zwischen den Rotationen und ist dann nicht mehr bloss Umsortierung: dort aendern V0 und V1 die Funktionenfamilie wirklich (nur die letzte Permutation faellt weg; bei volllinearer Auslese ist V1 = V0 mit umsortiertem Ausleseträger, also aequivalent, bei <Z_q> nicht).
- **Folge fuer die Arme:** Die Nullgradient-Eigenschaft von V0 betrifft die <Z_q>-Arme (q=0 in der letzten Lage), nicht die lineare Auslese. Die Q-020-Arme mit <Z>-Auslese sollten, wenn sie V0 bleiben, mit der Einschraenkung gelesen werden, dass 1 von 2 Parametern/Qubit in der letzten Lage fuer q=0 (und analog fuer andere q) tot sein kann. Ein V1- oder V2-Arm als zusaetzliche Kontrolle ist billig (kein Parameter, keine Mehrkosten) und trennt "Ring-Artefakt" von "Architektur". Ich habe diese Arme nicht trainiert, und `q020_*` nicht angefasst.

### Ehrliche Grenzen

- Nur die Varianz **eines** Gradienten (Ry q0) wurde gemessen, nicht die Gesamtgradientennorm, und nur fuer die Auslese <Z_0>. Fuer <Z_q> mit q > 0 sind in V2/V3 die Kegel anders (Kette: q ist Ziel von q-1 und Kontrolle von q+1); das ist nicht gemessen, V2 = V3 gilt nur fuer q = 0.
- 200 Stichproben je Punkt, Bootstrap beruecksichtigt nur die Stichprobenstreuung bei gegebenen Bildern und Winkeln, nicht Unterschiede zwischen Bildmengen. Die KIs sind also eher zu eng. Fit ueber n=2..10 (inkl. kleiner n, bei den Bildern stehen n=2..4 mit sehr grober Pixelzahl 4 bis 16): Die Steigungen fuer MNIST sind teils durch die Wahl der Verkleinerung bestimmt. Ein anderes Downsampling (z. B. Zentralausschnitt, Zeilenmittelung) wurde nicht ausprobiert und kann a verschieben.
- Gemeinsamer Teil MNIST: Fuer n=10 wird das 28x28-Bild per bilinearer Interpolation auf 32x32 hochgerechnet (keine neue Information).
- Gradientenvarianz ist eine Trainierbarkeits-Notwendigkeit, kein Beleg fuer Lernerfolg. Kein Training, keine Genauigkeit gemessen. Die Extrapolationen jenseits n=10 sind Fit-Fortsetzungen.
- Die Deutung der Ursachen (Eingabekonzentration, nichtzentrierte Bildrandverteilung) ist eine Erklaerung, die zu den Zahlen passt (V1 L=1 ohne CNOT hat a=0,95), aber nicht durch eine eigene Ablationsmessung abgesichert ist.
- Aufwand: 713 s CPU, 1 Thread (OMP_NUM_THREADS=1). Nichts committet.

---
**Bewertung Opus zu S1b (07.10.):** Meine Hypothese „Ring nur zwischen den Lagen hilft“ ist für
Zufallseingaben **widerlegt** (a bleibt ≈ 1) und für Bilder nur schwach (Faktor ~3). Der eigentliche
Befund ist die **Kette statt Ring auf echten Bildern**: a = 0,27–0,35 statt 1,0, bei n = 10 eine
60–120-fach größere Gradientenvarianz. Der Abfall bei Zufallseingaben liegt in der Eingabe selbst
(Konzentration), nicht in der Schaltung. Grenzen: nur ein Gradient, kein Training. Gradientenvarianz ≠
Genauigkeit. **Nächster Schritt S1c** (wenn Q-020 die Kerne freigibt): Training Kette vs. Ring auf
MNIST bei n = 8 und n = 10–12 (Bilder hochskaliert), gleiche Lernratensuche, vorregistriert:
„Kette ist bei n ≥ 10 nicht schlechter und trainiert schneller“.
