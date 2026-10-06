# v3 — Kontrollen K1–K3 als Erkundung: „Struktur“ oder „Quanten“?

*Stand 06.10.2026. **Erkundung, kein Befund, nichts vorregistriert.** Seeds 940–944 (5 Seeds; Tuning-Seed 959). Daten: MNIST und Fashion 16×16 (`abteilungen/clifford.daten`: je Seed 6.000 Trainings-/5.000 Testbilder aus dem offiziellen Testsatz). Skript und Rohdaten: Abschnitt 6.*

## 1. Kurzfassung

1. **Das Quadrieren plus irgendeine dichte orthogonale Mischung trägt fast alles.** Eine feste Haar-zufällige Orthogonale (0 trainierte Parameter) erreicht MNIST 0,9384 ± 0,0030 und Fashion 0,8340 ± 0,0068. Die trainierte QIRE-Zelle (Ry+CNOT, L=4, 32 Winkel) liegt bei 0,9379 bzw. 0,8366. Differenz QIRE − Haar: −0,04 ± 0,33 pp (MNIST), +0,27 ± 0,32 pp (Fashion). Praktisch gleich.
2. **Das Training der 32 Winkel bringt kaum etwas:** QIRE mit *ungelernten* Zufallswinkeln (mit CNOT) kommt auf 0,9371 / 0,8336, trainiert auf 0,9379 / 0,8366.
3. **Fastfood/SORF (H·D, nur 256 Vorzeichenbits) ist gleichwertig:** 1 Stufe 0,9315 / 0,8328; 2 Stufen 0,9376 / 0,8350 (≈ Haar).
4. **Klassische Referenz ohne Quadrieren ist stärker als die „Identität“ der Projekt-Tabelle:** Softmax direkt auf Pixeln (2.570 Parameter) 0,9096 / 0,8184; ReLU-MLP 256→9→10 (2.413 Parameter) 0,9014 / 0,8172. „Identität = Pixelquadrate“ (0,900 / 0,796) liegt *unter* der Pixel-Softmax. Der „+3,8 pp“ gegen die Identität sind deshalb nur +2,9 pp (MNIST) bzw. +1,6 pp (Fashion) gegen die ehrliche Referenz.
5. **Die Clifford-Suche (I-036-v3: 0,9359 / 0,8233, 20 andere Seeds) liegt auf MNIST auf dem Niveau einer zufälligen dichten Orthogonalen, auf Fashion darunter.** Sie schlägt zufällige Cliffords deutlich (Median: 0,9128 / 0,8152), aber nicht die zufällige Haar-Matrix. Ein Teil des Clifford-Gerüsts ist also eher *schlecht gemischt*, die Suche holt das nach.
6. **Gelernte volle Orthogonale (Cayley, 32.640 Parameter) und Butterfly (1.024 Winkel) sind besser** (MNIST 0,953 / 0,951; Fashion 0,847 / 0,846; +1,0 bis +1,6 pp gegen QIRE), aber mit 32× bis 1000× mehr Parametern.
7. **CNOT:** Ohne CNOT (reine Kronecker-Drehung) −1,3 ± 1,0 pp (MNIST) und −2,0 ± 0,4 pp (Fashion), 5/5 Seeds. Aber: Ohne CNOT kollabieren die 4 Lagen zu 8 effektiven Winkeln, und die Matrix mischt nur lokal. Das ist ein *Mischungs*-, kein Verschränkungsbefund: Fastfood H·D (ohne jede Quantenstruktur) ist 0,7 bzw. 1,6 pp besser als QIRE-ohne-CNOT.
8. **Ehrlichste Kernaussage** am Ende (Abschnitt 3.4).

## 2. Was gemessen wurde

### 2.1 Aufbau
Alle U-Arme: x → x/‖x‖ → z = U·x → Merkmale f(z) (Standard: z², je 256 Werte) → derselbe Kopf `kopf_logistisch` (Softmax-Regression, standardisierte Merkmale, L2 = 3·10⁻³, L-BFGS), trainiert auf den 6.000 Trainingsbildern, bewertet auf den 5.000 Testbildern. Der Kopf hat in allen U-Armen 2.570 Parameter. Pipeline-Check: Identität 0,9004 (Projekt: 0,9017 / 0,9014).

Gelernte U (Cayley, Butterfly, QIRE ± CNOT): Vollbatch-Adam, 300 Schritte, mit eingebautem Linearkopf trainiert; danach U eingefroren und der Standardkopf neu angepasst (Spalte „eigener Kopf“ im Rohdatensatz ändert die Aussage nicht). **Hyperparameter (Lernrate, Init) nur auf Tuning-Seed 959 mit 1.000 Validierungsbildern gewählt**, nie auf Test:
Cayley lr 0,005 (MNIST) / 0,02 (Fashion), Butterfly lr 0,03 Init 0,3 (MNIST) / lr 0,1 Init π (Fashion; Gleichstand mit 0,03), QIRE lr 0,08 (beide), ReLU-MLP lr 0,003 (60 Epochen, Adam, wd 1e-4), Pixel-Softmax L2 = 3·10⁻³. Das Tuning ist **dünn** (2 Einstellungen je Arm, ein Tuning-Seed, 300 Schritte): die gelernten Arme könnten mit mehr Mühe noch zulegen, das MLP ebenso.

### 2.2 Parameter (ehrlich getrennt)
| Arm | trainierte reelle Zahlen (inkl. Kopf 2.570) | diskrete/feste Wahl |
|---|---|---|
| Pixel-Softmax | 2.570 | — |
| ReLU-MLP 256→9→10 | 2.413 (h=10: 2.680) | — |
| Identität, Haar, Fastfood, Clifford-zufall | 2.570 | Haar: 65.536 zufällige feste Zahlen (nicht trainiert); Fastfood: 256 (1 Stufe) bzw. 512 (2 Stufen) Vorzeichenbits; Clifford: 32 Gatterbits (kanonisch) |
| QIRE Ry+CNOT, L=4 | 2.570 + 32 = 2.602 | CNOT-Ring fest |
| QIRE ohne CNOT | 2.570 + 32 (effektiv nur 8) | — |
| Cayley | 2.570 + 32.640 = 35.210 | — |
| Butterfly 8 Stufen | 2.570 + 1.024 = 3.594 | — |

### 2.3 Ergebnisse: Testgenauigkeit, Mittel ± sd über 5 Seeds

| Arm | MNIST | Fashion |
|---|---|---|
| **(Referenz) Softmax auf Pixeln** | 0,9096 ± 0,0035 | 0,8184 ± 0,0074 |
| (Referenz) Softmax auf x/‖x‖ (ohne Quadrat) | 0,9145 ± 0,0025 | 0,8224 ± 0,0082 |
| (Referenz) ReLU-MLP h=9 (2.413 P.) | 0,9014 ± 0,0043 | 0,8172 ± 0,0041 |
| (Referenz) ReLU-MLP h=10 (2.680 P.) | 0,9036 ± 0,0043 | 0,8208 ± 0,0053 |
| (1) Identität, Quadrat | 0,9004 ± 0,0026 | 0,7959 ± 0,0088 |
| (2) Haar-orthogonal, Quadrat | 0,9384 ± 0,0030 | 0,8340 ± 0,0068 |
| (2′) Haar, \|z\| statt z² | 0,9336 ± 0,0020 | 0,8307 ± 0,0074 |
| (2″) Haar, linear (kein Quadrat) | 0,9136 ± 0,0031 | 0,8222 ± 0,0080 |
| (3) Cayley gelernt (35.210 P.) | **0,9534 ± 0,0027** | **0,8470 ± 0,0048** |
| (4a) Fastfood H·D (1 Stufe) | 0,9315 ± 0,0017 | 0,8328 ± 0,0062 |
| (4b) Fastfood H·D₂·H·D₁ (2 Stufen) | 0,9376 ± 0,0032 | 0,8350 ± 0,0084 |
| (4c) 3 Stufen (SORF) | 0,9353 ± 0,0030 | 0,8338 ± 0,0080 |
| (4d) Walsh ohne Vorzeichen | 0,8828 ± 0,0032 | 0,7933 ± 0,0050 |
| (5) Butterfly 8 Stufen gelernt (3.594 P.) | 0,9512 ± 0,0035 | 0,8461 ± 0,0061 |
| (6) QIRE Ry+CNOT L=4 gelernt (2.602 P.) | 0,9379 ± 0,0041 | 0,8366 ± 0,0076 |
| (6′) QIRE Zufallswinkel, ungelernt | 0,9371 ± 0,0025 | 0,8336 ± 0,0070 |
| (7) QIRE **ohne CNOT**, gelernt | 0,9248 ± 0,0082 | 0,8166 ± 0,0053 |
| (7′) ohne CNOT, Zufallswinkel | 0,9160 ± 0,0084 | 0,8166 ± 0,0110 |
| (8) Zufällige Clifford (Median von 20 je Seed) | 0,9128 ± 0,0019 | 0,8152 ± 0,0076 |
| *Clifford-Suche I-036-v3 (20 andere Seeds, 760 ff.; nicht neu gerechnet)* | *0,9359* | *0,8233* |

Gepaarte Differenzen (pp, Mittel ± sd über 5 Seeds, Anzahl Seeds mit Vorzeichen +):
| Differenz | MNIST | Fashion |
|---|---|---|
| Haar-Quadrat − Identität | +3,79 ± 0,15 (5/5) | +3,80 ± 0,26 (5/5) |
| Haar-Quadrat − Pixel-Softmax | +2,88 ± 0,25 (5/5) | +1,55 ± 0,41 (5/5) |
| Haar-Quadrat − Haar-linear (Wirkung des Quadrats) | +2,48 ± 0,23 (5/5) | +1,18 ± 0,33 (5/5) |
| Haar-Quadrat − Haar-\|z\| | +0,48 ± 0,16 (5/5) | +0,32 ± 0,45 (4/5) |
| QIRE − Haar-Quadrat | −0,04 ± 0,33 (2/5) | +0,27 ± 0,32 (4/5) |
| QIRE − Pixel-Softmax | +2,83 ± 0,42 (5/5) | +1,82 ± 0,20 (5/5) |
| QIRE − MLP h=9 | +3,66 ± 0,61 (5/5) | +1,95 ± 0,50 (5/5) |
| QIRE − QIRE ohne CNOT | +1,32 ± 1,00 (5/5) | +2,00 ± 0,42 (5/5) |
| QIRE − zufällige Clifford (Median) | +2,51 ± 0,49 (5/5) | +2,15 ± 0,11 (5/5) |
| Fastfood 2 Stufen − Haar | −0,07 ± 0,23 (1/5) | +0,10 ± 0,47 (3/5) |
| QIRE ohne CNOT − Fastfood 1 Stufe | −0,68 ± 0,91 (1/5) | −1,62 ± 0,25 (0/5) |
| Cayley − QIRE | +1,55 ± 0,50 (5/5) | +1,04 ± 0,37 (5/5) |
| Butterfly − QIRE | +1,33 ± 0,39 (5/5) | +0,95 ± 0,24 (5/5) |

Die gelernte QIRE-Zelle liegt im Mittel um 0,38 rad (MNIST) vom Clifford-Gitter k·π/2 entfernt (Zufall wäre π/8 ≈ 0,39): die Winkel sind nicht „fast Clifford“, aber auch nicht weit vom Zufall weggelernt.

## 3. Deutung

### 3.1 Wo steht die Clifford-Suche in der Reihe?
Rangordnung (MNIST): Cayley 0,953 > Butterfly 0,951 > Haar ≈ QIRE gelernt ≈ QIRE zufällig ≈ Fastfood-2 ≈ **Clifford-Suche 0,936** > Fastfood-1 0,932 > QIRE ohne CNOT 0,925 > zufällige Clifford 0,913 ≈ Pixel-Softmax 0,910 > Identität/MLP 0,900–0,904.
Fashion: Cayley 0,847 ≈ Butterfly 0,846 > QIRE 0,837 ≈ Haar ≈ Fastfood ≈ 0,833–0,835 > **Clifford-Suche 0,823** ≈ Pixel-Softmax 0,818 > zufällige Clifford 0,815.
Die Clifford-Suche gewinnt gegenüber dem *Clifford-Raum* viel (+2,3 pp über dem Median zufälliger Cliffords), aber nicht gegenüber einer beliebigen zufälligen dichten Orthogonalen. Vorsicht: Der Vergleich Suche (Seeds 760 ff., 20 Seeds) gegen die Reihe (Seeds 940–944) ist nicht gepaart; Abstand auf MNIST −0,25 pp (innerhalb der Streuung), auf Fashion ≈ −1,1 pp (außerhalb). Die Pipeline stimmt (Identität 0,900 vs 0,9017).
Die langweiligste Erklärung: Zufällige Cliffords sind *dünn* (blockweise Walsh über Vorzeichen, mit 1 Bit je Gatter nur wenige Mischmuster). Haar mischt alle 256 Pixel dicht. Die Suche findet einen Clifford, der so gut mischt, wie es der Clifford-Raum erlaubt.

### 3.2 Was trägt das Quadrieren (Born) gegenüber der Matrix?
- Mit Haar-U: Quadrat − linear = +2,5 pp (MNIST), +1,2 pp (Fashion), 5/5. Das Quadrat ist der **Nichtlinearitätsträger**: Haar ohne Quadrat entspricht der Pixel-Softmax (0,9136 vs 0,9096), die Orthogonale allein bringt *nichts*, wie erwartet (lineare Köpfe lernen U mit).
- Quadrat gegen |z|: nur +0,5 / +0,3 pp. Die Nichtlinearität „irgendeine gerade Funktion“ trägt den Löwenanteil, das Quadrat selbst (Born) ist nur eine mögliche Wahl, leicht besser als |z|. (ReLU/andere nicht getestet.)
- Mit Identität schadet das Quadrat: Identität-Quadrat 0,900 < Pixel linear 0,910 (MNIST), 0,796 < 0,818 (Fashion). Erst die Mischung macht das Quadrat nützlich (Kreuzterme (x_i±x_j)²-artig).
- Also: **Mischung × Quadrat** zusammen = ca. +2,9 / +1,6 pp gegen ehrliche Referenz; keiner allein.

### 3.3 Trägt der CNOT, also die Verschränkung?
- Gemessen: Ohne CNOT −1,3 / −2,0 pp (5/5 Seeds). Aber der CNOT-lose Arm hat nur 8 effektive Winkel (4 Lagen Ry je Qubit addieren sich) und mischt nur 2×2-lokal. Fastfood H·D ohne CNOT, ohne Qubits, mit 256 Vorzeichenbits ist 0,7–1,6 pp **besser** als QIRE-ohne-CNOT und gleich QIRE-mit-CNOT.
- Lesart: Der CNOT-Ring liefert die *dichte Durchmischung*; jede klassische dichte Mischung leistet dasselbe. Eine Aussage „Verschränkung als Quantenressource trägt“ lässt sich aus diesen Daten **nicht** ableiten; der Test (QIRE ohne CNOT) ist kein sauberer Verschränkungs-Test, weil er Mischung und Verschränkung vermengt. Sauberer wäre: Ry-Schaltung mit *gleicher dichter Mischung ohne Verschränkung* (z. B. Haar-U) gegen Ry+CNOT: das ist hier QIRE − Haar ≈ 0.
- Bowles et al. (2024, Etappe-1-Bericht 10) stimmen in der Tendenz: Verschränkung zu entfernen ist oft nichts Besonderes. Hier ist es durch die Mischung dominiert.

### 3.4 Ehrlichste Kernaussage (Entwurf)
> „Auf MNIST und Fashion (16×16) erreicht die QIRE-Zelle mit 32 Winkeln dieselbe Genauigkeit wie eine feste zufällige orthogonale Matrix oder ein klassisches Fastfood-Merkmal (H·D) mit 256 Vorzeichenbits; sogar ungelernte Zufallswinkel genügen. Der Gewinn von +2,9 pp (MNIST) bzw. +1,6 pp (Fashion) gegenüber Softmax auf Pixeln mit gleichem Kopf kommt aus **dichter Durchmischung der Pixel vor einem Quadrat**; eine Verschränkung oder Quantenkohärenz wird dafür nicht gebraucht, und gelernte Winkel tragen kaum etwas. Die Struktur ist klassisch, die Parametersparsamkeit liegt im festen, nicht trainierten Merkmalsabbild. Gelernte volle orthogonale oder Butterfly-Matrizen sind um ~1–1,5 pp besser, kosten aber 1.000–32.000 Parameter mehr.“

### 3.5 Einschränkungen
- 5 Seeds, ein Tuning-Seed, 300 Schritte, nur 2 Einstellungen je gelerntem Arm: Reihenfolge der Spitzengruppe (Cayley/Butterfly) ist belastbarer als Abstände unter ~0,5 pp.
- MLP nur mit Lernrate getunt (h=9, 60 Epochen); ein stärker getuntes oder breiteres MLP/CNN wäre besser. „Gleiche Parameterzahl“ gilt nur für den Kopf-plus-U-Zähler wie in Abschnitt 2.2.
- Der Kopf (L2 = 3·10⁻³) ist für alle U-Arme gleich, für Pixel-Softmax per Validierung bestätigt (3·10⁻³ gewählt).
- Haar: eine Ziehung je Seed (3 Ziehungen je Seed streuen ≤ 0,2 pp, in den Rohdaten); Fastfood: je Seed ein Vorzeichensatz.
- Clifford-Suche nicht neu gerechnet (Aufwand ≈ 6 min je Seed und Datensatz); Vergleich ungepaart.
- Kein Zufallslabel- und kein Skalierungstest; nur 16×16 mit n = 8.

## 4. Vorschlag für einen vorregistrierten Lauf
- **Hypothesen:** (H-A) QIRE gelernt ≡ Haar-U + Quadrat (TOST ±0,5 pp). (H-B) QIRE gelernt ≡ QIRE mit Zufallswinkeln (TOST ±0,5 pp). (H-C) Fastfood-2 ≡ Haar (TOST ±0,5 pp). (H-D) Clifford-Suche < Haar auf Fashion (einseitig, Δ ≈ −1 pp). (H-E) Haar-Quadrat − Pixel-Softmax > 0 (Superiorität, erwartet +2,9 / +1,6 pp). (H-F) Cayley − QIRE > 0, mit Parameterzählung.
- **Arme:** Pixel-Softmax, MLP (h=9, mit Lernraten- und Epochenraster auf Validierung), Identität, Haar (3 Ziehungen), Fastfood 1/2 Stufen, QIRE gelernt/zufällig/ohne CNOT, Cayley, Butterfly, zufällige Cliffords, Clifford-Suche (aus I-036-v3-Code, gepaarte Seeds).
- **Kontrollen/Positivkontrolle:** Identität als Negativkontrolle (Haar − Identität = +3,8 pp reproduziert sich in beiden Datensätzen mit sd ≈ 0,2 pp, damit Positivkontrolle der Messkette). Zusätzlicher sauberer Verschränkungstest: Ry-Schaltung + feste dichte klassische Permutation/Hadamard-Mischung statt CNOT, und ein Datensatz mit nichtlokaler Struktur (Parität/Schachbrett).
- **Seeds:** 20 (reservierte Block, z. B. nicht 900–999). Gepaarte sd der Differenzen ≈ 0,3 pp (QIRE − Haar) → SE bei n = 20 ≈ 0,07 pp: TOST ±0,5 pp ist erreichbar (auch ±0,25 pp wäre es). Für Fastfood-Differenz (sd ≈ 0,2–0,5) ebenso. Für QIRE − ohne-CNOT (sd 0,4–1,0) bräuchte ±0,5 pp kein TOST, die Differenz ist 1,3–2,0 pp.
- **Laufzeit:** hier 5 Min je Seed und Datensatz bei 1 Thread unter Last 15 (davon gelernte Arme > 70 %). 20 Seeds × 2 Datensätze ≈ 200 Min Einzelkern = ≈ 45 Min auf 5 Kernen, ohne Clifford-Suche (die verdoppelt ungefähr).

## 5. Offene Fragen
1. Ist QIRE-Training wirklich nutzlos oder nur durch die kleine Lernraten-/Schrittzahl begrenzt? (Zufallswinkel = gelernt.)
2. Warum liegt die Clifford-Suche unter Haar auf Fashion? Dichte der Mischung oder Suchkriterium (LOO-Ridge)?
3. Gibt es eine Aufgabe, bei der Verschränkung nachweislich nötig ist (nichtlokale Struktur)?
4. Skalierung: Gilt „Zufall-Orthogonal ≈ QIRE“ auch bei n = 10–12 und größeren Bildern?
5. Gelernte dünne Orthogonale (Butterfly mit weniger Winkeln, z. B. 8 gemeinsame je Stufe) als Brücke zwischen 32 und 1.024 Parametern.

## 6. Pfad der Skripte
- Skript: `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/v3/kontrollen.py` (Aufruf `STEPS=300 NT=1 python3 kontrollen.py <mnist|fashion_mnist> 940,941,942,943,944`)
- Rohdaten: `.../scratchpad/v3/roh_mnist.jsonl`, `roh_fashion_mnist.jsonl`; Tuning: `hp_mnist.json`, `hp_fashion_mnist.json`; Logs `log_*.txt`.
