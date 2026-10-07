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
