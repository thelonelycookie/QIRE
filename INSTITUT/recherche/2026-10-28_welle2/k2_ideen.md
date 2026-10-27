# k2 — Fehlerbuch-Bergbau (Welle 2, 28.10.2026): 16 Ideen aus Anomalien

Nur gelesen und gedacht, nichts gerechnet, kein git, keine Suche. Gelesen: FEHLERBUCH (ganz), LABORBUCH (alle „beschreibend“-/„Nebenbefund“-Stellen, HU/H0-Auswertungen, Ende ab I-045-v5), SYNTHESE, w1–w12, r1–r3. Alle Zahlen stammen aus diesen Dateien. Wirkung überall klassisch (I-038): „quanten-inspirierter Vorteil“. Tuning-Seeds bei „Test“: 985–999 (frei laut seeds.json prüfen).

Nicht wiederholt (steht in SYNTHESE, w9 oder läuft in I-047/48/49/45-v5): Hadamard-Quantisierung allgemein, Hourglass/LRU-Vergleich, Datenregime/Lernrate/FLOP-Zählung von I-045, Phasen-Gewichte, Entropie-Rang, Monarch, TT-FFN, Born-Attention, frühes Aussteigen, Adapter-Pareto, Bit-Pareto mit Kopf, Dekodier-Mikrobenchmark, CIFAR-100-Kopf, Fashion-Kleinst-MLP.

**Leitmuster aus dem Fehlerbuch**, aus dem die Ideen stammen:
1. Jeder Vorsprung einer Struktur hängt an Budget, Lernrate oder Epochen. Gesucht: Stellen, an denen die Erklärung noch fehlt.
2. Der Gradient findet nicht, was diskrete Suche findet (I-003, I-005, k32_alle). Das Becken zählt.
3. Wenige Dinge tragen den Nutzen (M50 = 4 Winkel; Rang der Form 52–78 %). Gesucht: das klassische Gegenstück.
4. Paare wirken nur zusammen (Walsh-Raum × Vorzeichenmaske, +2,1 / +2,7 pp Wechselwirkung).
5. Der Strukturvorsprung wächst, wo keine Information dazukommt (I-041, Hochskalieren).

---

## 1. Born-Normierung der FFN-Mitte (z² / mittleres z²)
· Denkwerkzeug: Fehlerbuch-Bergbau, Widersprüche W2/W3 aus w9 (Sz schlechter bei d = 512 als 256; z² hilft bei d = 512, T = 16000 nicht mehr). Frage: Was müsste wahr sein, damit das kein Zufall war? Die Skala von z² wächst mit der Breite quadratisch, die Lernrate ist aber nur für ein d getunt. · Quantenbezug: Die Born-Regel normiert Amplituden-Quadrate auf Summe 1.
- **Mechanismus:** Statt `z²` rechnet die FFN `z² / (mean(z²)+ε)` (skaleninvariant, 4d Zusatzoperationen). Das entfernt die Breiten- und Lernraten-Abhängigkeit des Quadrats, vermutlich die Ursache von W2/W3. Gleiche FLOPs (+≈1 %), gleiche Bytes. Gewinn nur, wo bisher Instabilität den z²-Vorteil verdeckt.
- **Langweiligste Erklärung:** Es ist LayerNorm in der Mitte (bekannt, hilft immer ein bisschen). Kontrolle LayerNorm vor dem Quadrat trennt das.
- **Billigster Test:** Byte-LM d = 64 und d = 128 (je 2 Lagen), T = 600, 3 Tuning-Seeds, Lernrate je Arm 3 Werte. Arme: D_gelu, D_z2, D_z2 + Born-Norm, D_z2 mit LayerNorm davor. Maß: Abstand D_z2 − D_gelu bei d = 64 gegen 128 (verschwindet der Vorzeichenwechsel?). ≈ 9 min.
- **Vorab-Tipp:** P(grün) = 30 %, Jury 3.

## 2. Trotter-Stapel: Fastfood-Schichten mit Skip statt ohne
· Denkwerkzeug: Fehlerbuch-Bergbau. I-044 H2: drei Fastfood-Schichten sind *sicher schlechter* als eine (−1,50 pp, 90-%-KI [−1,88; −1,12]); I-044-v3 H2 nur HU. w6: Lagen 2–4 eines Netzes ohne Skip sind gleich genau, tiefe Lagen tragen nichts. Frage: Was müsste wahr sein? Tiefe schadet, weil der Stapel *keine Skip-Verbindung* hat, nicht weil die Struktur nichts kann. · Quantenbezug: Trotter-Zerlegung, e^{−iHt} ≈ (1 − iH·dt)^n, also x + dt·f(x) je Schritt.
- **Mechanismus:** Residualstapel x ← x + α·σ(H D₂ H D₁ x), α anfangs 0 oder klein. Jede Lage kostet 2N·log₂N und 2N Parameter; tiefe Stapel sind dann FLOP-billig gegen eine dichte Schicht (ideal gezählt ≈ 50× bei N = 4096). Gewinn nur, falls Tiefe jetzt hilft, was ohne Skip nicht geht.
- **Langweiligste Erklärung:** Jedes Netz ohne Skip trainiert tief schlecht; mit Skip hilft Tiefe immer (ResNet). Nur die Zahl „Tiefe hilft bei ideal 2N-Parameter-Lagen“ wäre neu.
- **Billigster Test:** MNIST 16×16 Rohpixel (N = 256), Köpfe linear; Arme S1, S3, S3 + Skip, S3 + Skip (α-Init 0), dichte 1-Schicht gleicher Operationen (h passend), dichte 3-Schicht-Skip. 20 Epochen, 3 Tuning-Seeds ≈ 6 min. Kontrolle: Skip-Stapel mit *zufällig festen* Diagonalen (trägt das Lernen der Diagonalen?).
- **Vorab-Tipp:** P(grün: S3+Skip > S1 und ≥ dichte gleicher Ops) = 35 %, Jury 3.

## 3. Clifford-Fastfood: nur ±1-Diagonalen (1 Bit), gleiche ideale Operationen gegen ternäre Dichte
· Denkwerkzeug: Fehlerbuch-Bergbau, I-005 (diskrete Clifford-Suche 0,9352 = Gradient 0,9344), I-036 (Vorzeichenmaske nötig), I-045-v4/Bytes (Sz hat 43 k fp32-Parameter = 172 kB; w12: ideal gezählt h ≈ 42–46 hat der Gegner 4× Parameter). Frage: Was müsste wahr sein, damit das Ergebnis nicht am Gleitkomma der Diagonalen hängt? · Quantenbezug: Clifford-Gruppe = Hadamard, Pauli-Vorzeichen, Permutationen; Vorzeichen-Diagonalen sind Pauli-Z-Produkte.
- **Mechanismus:** Fastfood-FFN mit D₁, D₂ ∈ {±1} (Straight-Through) plus *eine* 2-Bit-Skala je Block. Bytes sinken von 172 kB auf ≈ 5,5 kB (+ Skalen). Gleiche ideale Operationen wie Sz. Der faire Gegner bei gleichen Bytes *und* gleichen idealen Ops ist ein ternärer dichter FFN mit h ≈ 53 (≈ 1,58 Bit, ≈ 5,4 kB). Gewinn, falls Mischung (n log n Ops, keine Gewichte) mehr Ausdruck pro Bit liefert als ein winziger dichter Kern.
- **Langweiligste Erklärung:** Binäre Diagonalen sind schlechter als fp32-Diagonalen, und der ternäre Dichte-Gegner ist so schwach (h = 53), dass jedes Mischen gewinnt. Dann zählt nur der Vorsprung gegen *zusätzliche* feste Zufalls-Vorzeichen (Kontrolle).
- **Billigster Test:** Byte-LM d = 128, 2 Lagen, T = 1500, 3 Tuning-Seeds. Arme: Sz fp32, Sz ±1-STE, Sz ±1 fest zufällig (0 gelernte Bits), ternäre Dichte gleicher Bytes, Dk_z2 gleicher Parameterzahl. Maß: bpc gegen FFN-Bytes und ideale Ops. ≈ 9 min.
- **Vorab-Tipp:** P(grün: ±1-Sz ≤ Sz fp32 + 0,03 bpc UND < ternäre Dichte) = 20 %, Jury 4 (Pauli/Clifford-Bezug stark, Bytes-Aussage klar).

## 4. Reed-Muller-Grad-2-Merkmale: quadratische Phasen-Diagonale statt Zufalls-D
· Denkwerkzeug: Fehlerbuch-Bergbau, I-036 (Walsh-Raum und Maske wirken nur als Paar, Wechselwirkung +2,12 / +2,70 pp; Rang der Form über GF(2) erklärt 52 % / 78 % des Maskengewinns; zufällige Form gleichen Rangs holt die Hälfte; Spearman Rang ↔ Gewinn +0,79). Übersehen: Es wurde nie gegen *i.i.d. zufällige ±1-Vorzeichen* getestet, nur gegen zufällige *quadratische* Formen. Frage: Was müsste wahr sein? Der Gewinn kommt von der quadratischen Struktur (flaches Walsh-Spektrum, Bent), nicht von „irgendwelchen Vorzeichen“. · Quantenbezug: CZ/S-Schichten erzeugen Graphzustände; der GF(2)-Rang der Form ist ihre Verschränkung.
- **Mechanismus:** Fastfood-Merkmale H·D·H·D′·x mit D = (−1)^{q(u)}, q quadratisch vom Rang r (0 gelernte Parameter, Form aus ≈ r·n Bit Seed). Bent-Formen (max. Rang) haben perfekt flaches Walsh-Spektrum, also maximal „inkohärente“ Mischung. Wenn das gegen i.i.d.-Vorzeichen gewinnt, ist es ein Merkmalsgewinn bei gleichen FLOPs und 0 Bytes.
- **Langweiligste Erklärung:** i.i.d. Vorzeichen sind gleich gut, Struktur ist egal (I-038: Fastfood ≡ Haar). Dann wäre die ganze Rang-Linie eine Eigenschaft des 4×4-Blocks, nicht der Quadratik.
- **Billigster Test:** I-041-Rahmen (MNIST/Fashion 16×16, Fastfood²-Merkmale + Softmax-Regression), Tuning-Seeds, Arme D ∈ {Gauss, i.i.d. ±1, quadratisch Rang 2/4/8 (Bent)}, 10 Seeds, ≈ 4 min. Maß gepaart; TOST ±0,3 pp als Gleichwertigkeitsmarke.
- **Vorab-Tipp:** P(grün: Bent > i.i.d. ±1 um ≥ 0,3 pp) = 12 %, Jury 3 (ein klares Nein wäre trotzdem ein sauberer Gegenbefund zu I-036).

## 5. Training ohne Gradient: diskrete Strukturssuche mit geschlossenem Kopf
· Denkwerkzeug: Fehlerbuch-Bergbau. I-005 Q-3: die Suche mit **0 Magiewinkeln** erreicht 0,9352 = volles Gradiententraining 0,9344; Q-4: Suche schlägt Zufallssuche gleicher Bewertungszahl um +1,09 pp (9/10); I-003: der Gradient findet die diskrete Lösung nicht; k32_alle (alle 32 Winkel ab der Suche nachtrainiert) fällt auf G = 0,42, *unter* k = 8 (0,69) und 0,80 pp unter voll — „anderes Becken“. Frage: Was müsste wahr sein? Diskrete Becken sind für Vorwärts-Suche mit exaktem Kopf billiger erreichbar als für den Gradienten. · Quantenbezug: Stabilizer-/Clifford-Suche, Hill-Climbing über Pauli-Strukturen.
- **Mechanismus:** Die Merkmalsstruktur (Vorzeichen in D₁, D₂, wenige Permutations-Tausche) wird per Bit-Flip-Hill-Climbing gewählt; der lineare Kopf wird je Kandidat *geschlossen* (Ridge) gelöst. Kein Rückwärtslauf, nur Vorwärtsläufe; Inferenz identisch zu fest-Fastfood. Vorteil: Trainingskosten (kein Backprop) bei gleicher Inferenz; Gewinn über festes Zufalls-Fastfood nur, wenn die Struktur auf die Aufgabe passt.
- **Langweiligste Erklärung:** Fastfood-Merkmale sind so redundant, dass jede Wahl gleich gut ist (I-038: fest ≡ Haar ≡ gelernt); die Suche optimiert Rauschen und überschätzt auf Validierung (Siegerfluch +0,55 pp in I-036).
- **Billigster Test:** MNIST 16×16, 1.200 Validierungsbilder zum Suchen, 400 Bit-Flips, Ridge auf 256–512 Merkmalen (je Kandidat < 0,3 s). Arme: Suche, beste von 400 Zufallsdraws (gleiche Bewertungszahl), fest zufällig, Gradient-Diagonalen. Maß: Testgenauigkeit, Trainings-FLOPs. 3 Tuning-Seeds ≈ 8 min.
- **Vorab-Tipp:** P(grün: Suche ≥ Zufall + 0,5 pp UND Gradient-Diagonalen höchstens gleich) = 25 %, Jury 4.

## 6. Hilbertraum-Breite: Fastfood-Zufallsmerkmale mit geschlossenem Kopf, Breite statt Training
· Denkwerkzeug: Fehlerbuch-Bergbau, I-041: Haar² − Pixel wächst beim reinen *Hochskalieren* von 16×16 auf 32×32 (MNIST: +2,84 → +4,33 pp, H2 +1,64 pp „wächst“), obwohl bilineares Resampling *keine neue Information* hinzufügt (steht so im Laborbuch). Frage: Was müsste wahr sein? Der Vorsprung ist eine Funktion der **Merkmalszahl N** (Näherung an den Polynomkern Grad 2), nicht der Information. Dann ist Breite die Ressource, und sie kostet nur N log N Ops und *kein* Training der Merkmale. · Quantenbezug: Quantenkernel / Merkmalsraum der Dimension 2ⁿ, hier klassisch simulierte Zufallsmerkmale.
- **Mechanismus:** Merkmale z² aus Fastfood-Blöcken, D = N, 4N, 16N; Kopf per Ridge in einem Durchgang (kein SGD). Inferenzkosten: N log N + 2·10·D. Trainingskosten ≈ ein Durchlauf. Gewinn gegen einen *trainierten* MLP gleicher Inferenz-FLOPs nur, falls Breite die Qualität schneller kauft als gelernte Merkmale; sicher gewinnt es bei Trainings-FLOPs.
- **Langweiligste Erklärung:** Zufallsmerkmale mit D = 16N sind einfach ein breiter Kernel; der gleich teure trainierte MLP ist schlicht besser (bekannt: Random Features unterlegen bei kleinen Budgets). Bytes verlieren (Kopf 10·D Zahlen).
- **Billigster Test:** MNIST 16×16 *und* 16×16 auf 32×32 hochgerechnet (Resampling-Kontrolle), D ∈ {256, 1024, 4096}; Arme Fastfood² + Ridge, Pixel + Zufalls-ReLU-Merkmale gleicher Zahl (Kontrolle: Zufall ohne Struktur), trainierter 1-Schicht-MLP mit gleichen Inferenz-FLOPs. Maß: Genauigkeit gegen FLOPs, gegen Bytes, gegen Trainings-FLOPs. 3 Tuning-Seeds ≈ 7 min.
- **Vorab-Tipp:** P(grün: „bei gleichen Inferenz-FLOPs ≥ MLP ODER bei klar kleineren Trainings-FLOPs gleich“) = 30 %, Jury 4.

## 7. Z-Kurven-Blockschicht mit Permutationskontrolle (Flächengesetz)
· Denkwerkzeug: Fehlerbuch-Bergbau, B-010 (Projekt A, „offenes Rätsel“): Blockdiagonal gewinnt bei n = 200 um +1,49 pp (MNIST) / +1,39 (Fashion), *nicht* durch Parameterzahl erklärbar; w9 #7 nannte die n-Reihe, aber nicht die entscheidende Kontrolle. V-kompass (FEHLERBUCH): „Bits ohne Bedeutung“, Ordnung wirkt nur, wo die Indizes Struktur haben; Morton-Ordnung war die harte Konkurrenz. Frage: Was müsste wahr sein? Der Gewinn ist 2D-**Lokalität**; *zufällig permutierte Blöcke* müssen ihn verlieren, Z-Kurven-Quadrate müssen Zeilenstreifen schlagen. · Quantenbezug: Flächengesetz der Verschränkung (lokale Tensornetze/PEPS, niedrige Verschränkung über lokale Schnitte).
- **Mechanismus:** Erste Schicht als B Blöcke (z. B. B = 4: Quadranten in Z-Ordnung, Zeilenstreifen, zufällige Permutation). FLOPs und Parameter sinken *echt* auf 1/B. Gegner: dichte Schicht mit gleicher Operationszahl (h/B) und gleicher Parameterzahl. Gewinn, wenn Lokalität bei kleinen n zuverlässig Schicht-Ersparnis bringt.
- **Langweiligste Erklärung:** Jedes Netz mit weniger Parametern gewinnt bei n = 200 (Regularisierung); Permutationskontrolle zeigt es sofort. Dann ist B-010 gelöst und die „Struktur“ ist Kapazitätsdrosselung.
- **Billigster Test:** MNIST/Fashion 16×16, n ∈ {200, 1.000, 6.000}, 1 versteckte Schicht h = 128, B = 4: Z-Quadranten, Streifen, zufällige Permutation, dicht h = 32 (FLOP-gleich), dicht klein (Parameter-gleich). 5 Tuning-Seeds ≈ 6 min.
- **Vorab-Tipp:** P(grün: Z-Quadranten > Permutation ≥ 0,7 pp bei n = 200) = 45 %, Jury 3.

## 8. Separable Eingangsfilter a ⊗ b (Bonddimension χ = 1…3)
· Denkwerkzeug: Fehlerbuch-Bergbau, V-kompass-Folgefrage (FEHLERBUCH 24.09., „nicht vorregistriert“): „nur die Pixelseite als Quanten-Tensor-Train, die Rangseite als SVD“ — im Laborbuch nirgends getestet; Ergebnis dort: gewöhnliche Niedrigrang-SVD schlägt alle TT-Ordnungen *auf der versteckten Schicht*. Frage: Was müsste wahr sein? Auf der Pixelseite haben die Indizes Struktur (Zeilen, Spalten), auf der versteckten nicht. · Quantenbezug: Produktzustand (χ = 1) und Matrix-Produkt-Zustand kleiner Bonddimension als Filter-Form.
- **Mechanismus:** Jede verdeckte Einheit hat Filter w_r(i,j) = Σ_{c≤χ} a_{rc}(i)·b_{rc}(j): 2·16·χ statt 256 Parameter (Bytes 8/χ-fach weniger); danach dichte versteckte Schicht. FLOPs ≈ dicht (kein Ersparnis, daher reine Bytes-Idee). Gegner: Niedrigrang U·V gleicher Parameter, dichte Schicht kleiner Breite gleicher Bytes.
- **Langweiligste Erklärung:** Separable Filter sind Niedrigrang-Filter (Rang χ); gleiche Parameter in U·V sind gleich gut (V-kompass: SVD gewinnt).
- **Billigster Test:** MNIST/Fashion 16×16, h = 64, χ ∈ {1, 2, 3}; Arme: Kronecker-Filter, U·V gleicher Parameter, schmal dicht gleicher Bytes, voll dicht. 5 Tuning-Seeds ≈ 5 min. Maß: Genauigkeit gegen Bytes.
- **Vorab-Tipp:** P(grün: Kronecker ≥ U·V + 0,5 pp bei gleichen Bytes) = 25 %, Jury 3.

## 9. Skalen-only-Nachtraining nach Rotation + 1–2 Bit
· Denkwerkzeug: Fehlerbuch-Bergbau, Muster „wenige tragen den Nutzen“: M50_S = 4 Winkel (C-fashion/I-005-v3), und k32_alle (alles nachtrainieren) ist *schlechter* als k = 8. Übersehen in der Quantisierungslinie (w4, r1, I-048): Nachtraining wird nur global erwogen. Frage: Was müsste wahr sein? Nach der Rotation sitzt der Restfehler vor allem in Zeilenskalen und Bias; wenige Parameter genügen. · Quantenbezug: Magie-Konzentration, 4 von 32 Winkeln.
- **Mechanismus:** Nach Hadamard-Rotation + b-Bit-Rundung nur die ohnehin gespeicherten Zeilenskalen und Bias nachtrainieren (2·Zeilen Parameter, **0 zusätzliche Bytes**). Bei 1 Bit (Lücke groß genug für Auflösung) deutlich. Gewinn: Genauigkeit bei exakt gleichen Bytes und FLOPs.
- **Langweiligste Erklärung:** Es ist BatchNorm-artige Neukalibrierung (bekannte Bias-Korrektur); Rotation ist unnötig, Roh + Skalen-Nachtraining gleich gut (Kontrolle).
- **Billigster Test:** r1-Code (MNIST 784-512-512-10, groß fp32), 1 und 2 Bit, Arme: roh, rotiert, jeweils ± Skalen/Bias-Nachtraining (1 Epoche, STE), + volles STE-Nachtraining (Kontrolle gegen k32_alle-Muster), 3 Tuning-Seeds ≈ 6 min.
- **Vorab-Tipp:** P(grün: Skalen-only holt ≥ 50 % der Lücke zu fp bei 1 Bit, ≥ volles STE − 0,3 pp) = 45 %, Jury 3.

## 10. Trainings-FLOPs-Gleichheit für Rotation + Niedrigbit (Ergänzung zu I-048)
· Denkwerkzeug: Fehlerbuch-Bergbau, Lehre C-referenz/„Referenz fair“ und Fehlerbuch I-044-v3 (Vorsprung = knappes Budget). Übersehen: r1/I-048 vergleichen bei gleichen *Bytes*, aber das große Modell hat 2,5–6× mehr MACs und mehr **Trainings**-FLOPs; das kleine dichte fp16-Modell läuft nur gleich viele Epochen (10 bzw. 30). w5 zeigt außerdem: von Anfang an schmal trainieren war schlechter als groß trainieren und nachträglich komprimieren (scratch 93,78 gegen 95,4 % bei f = 0,2; Lernrate dort ungetunt). Frage: Was müsste wahr sein? Der Vorsprung (+0,40 pp) bleibt, wenn das kleine Modell dieselben Trainings-FLOPs bekommt. · Quantenbezug: Hadamard-Basiswechsel.
- **Mechanismus:** Kein neuer Baustein, sondern die schärfste Kontrolle: kleines dichtes Modell mit 2,5× bis 6× Epochen und Lernratensuche (gleiche Gesamt-Trainings-FLOPs wie groß + Rotation). Ergebnis zählt für die Arbeit in beide Richtungen.
- **Langweiligste Erklärung:** Das kleine Modell holt auf (Trainingsdauer, nicht Rotation, war der Unterschied).
- **Billigster Test:** r1-Aufbau, 2 Bit, 3 Seeds; klein dicht mit {10, 25, 60} Epochen, groß + Rotation 10 Epochen. Maß: Genauigkeit gegen Trainings-FLOPs *und* Bytes. ≈ 8 min.
- **Vorab-Tipp:** P(grün: Rotation vorn auch bei gleichen Trainings-FLOPs, ≥ 0,2 pp) = 40 %, Jury 4 (Juror-Frage Nummer eins).

## 11. Basiskonflikt: Rotation gegen Aktivierungs-Sparsität (Rotation nur auf der dichten Seite)
· Denkwerkzeug: Fehlerbuch-Bergbau, I-040-v2 H1/H4: die **Identität** liefert die meiste Information je Messung (+0,201 Bit gegen Haar), Mischung erhöht die Genauigkeit (+3,5 pp), kostet aber 0,20 Bit; Gleichung der beiden Linien SYNTHESE Rang 1 (Rotation vor Quantisierung) und w11 K2 (ReLU²-Sparsität als FLOP-Hebel) — beide Linien widersprechen sich unbemerkt: Rotation macht Aktivierungen dicht und zerstört das, was Top-K braucht. Frage: Was müsste wahr sein? Die rotierte Eingangsseite einer Matrix ist nur dort nützlich, wo der Eingang dicht ist (Residualstrom, Eingangs-FFN-Matrix W₁), nicht für W₂ (Eingang = dünnes ReLU²). · Quantenbezug: Basiswahl (Rechenbasis gegen Hadamard-Basis) je nach Zustandsdichte.
- **Mechanismus:** Rotation für W₁ (dichter Eingang), W₂ in der Identitätsbasis mit Top-K-Überspringen der Aktivierungen (spart echte MACs, auch bei 2 Bit). Bytes gleich, MACs von W₂ ≈ K/n. Gewinn gegen "überall rotieren" (verliert Top-K) und gegen "gar nicht rotieren" (verliert Bits).
- **Langweiligste Erklärung:** W₂-Quantisierungsfehler wird mit dünnen Aktivierungen ohnehin kleiner gewichtet (nur aktive Spalten zählen); Rotation ist dort schlicht überflüssig.
- **Billigster Test:** MNIST 784-512-512-10 mit ReLU², 2 Bit; Arme: rotiert überall, rotiert nur W₁, nicht rotiert; je ohne und mit Top-K 25 % der versteckten Aktivierung. Maß: Genauigkeit, effektive MACs, Bytes; zuerst Messung der Energie in den obersten 25 % Koordinaten vor/nach Rotation. 3 Tuning-Seeds ≈ 6 min.
- **Vorab-Tipp:** P(grün: gemischt ≥ „überall rotiert“ − 0,1 pp bei ≤ 70 % der MACs) = 40 %, Jury 3.

## 12. Quantisierung als Regularisierer im überangepassten Regime
· Denkwerkzeug: Fehlerbuch-Bergbau, I-045-v4 Nebenbefund (bei T = 16000 schlägt h = 580 die volle Breite, ≈ 10 Epochen über 1,6 MB) und w4: 8/4/3 Bit „verlustfrei (≤ +0,003 bpc)“ — der Verlust ist *nie negativ*, weil die Modelle dort (1200 Schritte, ≈ 0,75 Epochen) *untertrainiert* sind. Übersehen: Das überangepasste Regime wurde nie quantisiert. Frage: Was müsste wahr sein? Rundungsrauschen ≈ Gewichtsrauschen wirkt als Regularisierung; nach Rotation (Gauß-förmig, ohne Ausreißer) ist es *isotrop*, also das ideale Rauschen. · Quantenbezug: Dekohärenz/Dephasierung als Rauschkanal.
- **Mechanismus:** Überangepasstes Byte-LM (25 % der Daten, 3.000 Schritte) nachträglich auf 3/4 Bit rotiert runden. Wenn Test-bpc nach der Rundung *unter* fp32 fällt, ist Quantisierung gratis *besser*, nicht nur kleiner. Bytes −75 %, FLOPs gleich.
- **Langweiligste Erklärung:** Der fp32-Checkpoint ist einfach am letzten Schritt schlecht gewählt (frühes Stoppen gewinnt noch mehr); Vergleich mit Early-Stopping-Checkpoint trennt das.
- **Billigster Test:** d = 128, 4 Lagen, 25 % Daten, T = 3000; Arme: fp32 Ende, fp32 bester Val-Stand, 4/3 Bit rotiert, fp32 + gleich starkes Gauß-Gewichtsrauschen (Kontrolle). 3 Tuning-Seeds ≈ 9 min.
- **Vorab-Tipp:** P(grün: Quantisiert < fp32 Ende um ≥ 0,01 bpc) = 20 %, Jury 3.

## 13. Feste kausale Zerfallsbank statt Attention in Lage 0 (Null Parameter)
· Denkwerkzeug: Fehlerbuch-Bergbau, r2: LRU in Lage 0 −0,29 bpc bei 190 Schritten, von der Vorab-Erklärung als „guter Prior für lokale Kontexte“ gedeutet — aber *welcher Teil ist gelernt nötig?* Nie getrennt. I-049 prüft nur LRU gelernt gegen flach. Frage: Was müsste wahr sein? Der Gewinn kommt aus *irgendeiner* kausalen lokalen Mischung mit Zeitkonstanten; gelernte komplexe Eigenwerte (Phase) sind dann überflüssig. · Quantenbezug: Lindblad-Dämpfung mit *festen* Dekohärenzzeiten T₁ (log-verteilt), Phasen fest.
- **Mechanismus:** Lage 0 ersetzt Attention durch eine feste Bank von d exponentiellen Gedächtnissen λ_k (log-verteilt, optional mit festen Phasen), nur noch die Mischmatrizen davor/danach werden gelernt. Spart die 4d²-Attentionsparameter dieser Lage und deren Score-FLOPs; O(d)-Zustand. Gewinn gegen Attention, falls Frühgewinn bleibt, gegen volles LRU, falls gleich.
- **Langweiligste Erklärung:** Frühregime-Effekt; ab 1.500 Schritten gleicht Attention aus (I-049 H2 prüft das für LRU gelernt, hier gratis mit).
- **Billigster Test:** r2-Skript, Tuning-Seeds 985–987, Arme F4, LRU Lage 0, feste Zerfallsbank Lage 0 (Phasen fest / ohne Phasen / nur ein Zerfallsspektrum), T = 190 und 1.500. Maß: bpc, Parameter, FLOPs. ≈ 10 min.
- **Vorab-Tipp:** P(grün: fest ≥ 80 % des LRU-Gewinns bei 190 Schritten UND ≤ F4 + 0,02 bei 1.500) = 30 %, Jury 3.

## 14. Strukturierter Warmstart: Sz zuerst, dann dicht erweitert
· Denkwerkzeug: Fehlerbuch-Bergbau, drei Fundstellen: I-045-v2 (Sz 3,403 schlägt D_z2 bei T = 1000, d. h. Struktur gewinnt *früh*), C-fashion/C-referenz (zweite Trainingsphase +6 pp „allein durch die zweite Phase“), I-005 (k32_alle landet in anderem Becken — die Wahl des Startbeckens entscheidet). Frage: Was müsste wahr sein? Strukturierte Initialisierung gibt dem dichten Netz einen besseren Startpunkt als Gauß-Zufall. · Quantenbezug: adiabatische Interpolation: strukturierter Anfangs-Hamiltonian, dann volles Problem.
- **Mechanismus:** Phase 1: Sz (ideal ≈ 9–12× weniger Ops je Lage, w12) bis T₁; dann wird W = H D₂ H D₁ exakt als dichte Matrix ausgeschrieben (kein Sprung im Verlust), Phase 2 dicht. Endmodell dicht (gleiche Inferenz-FLOPs wie dicht); Gewinn nur in **Trainings-FLOPs** bis zum Ziel-bpc.
- **Langweiligste Erklärung:** Orthogonale Initialisierung ist besser als Gauß (Saxe 2014); jede orthogonale Matrix tut es. Kontrolle: Haar-orthogonaler Start.
- **Billigster Test:** Byte-LM d = 128, 2 Lagen, T = 1.200, Phase 1 = 600. Arme: dicht Gauß, dicht orthogonal, Sz → dicht (Wechsel bei 300/600), nur Sz. Maß: bpc gegen *ideal gezählte* Trainings-FLOPs. 3 Tuning-Seeds ≈ 10 min.
- **Vorab-Tipp:** P(grün: Sz→dicht < dicht Gauß bei gleichen Gesamtschritten UND < orthogonal) = 20 %, Jury 3.

## 15. Breitenhalbierung durch das Quadrat (Breite-bis-Ziel-Kurve)
· Denkwerkzeug: Fehlerbuch-Bergbau, Muster aus w2/r3/w7: der z²-Gewinn ist größer, je schmaler das Netz — MNIST 5.083 Parameter: relu 93,74 → z² 94,57 (+0,83 pp), 17.899 Parameter: 96,13 → 96,38 (+0,25 pp); Fashion +0,64 / +0,13; w2: z² bei schmal dicht −0,025 bpc (3/3 Seeds). In v3/v4 verschwindet er bei großem d, T. Frage: Was müsste wahr sein? Quadratische Einheiten tragen Paar-Wechselwirkungen pro Einheit; der Gewinn ist ein Kapazitätsgewinn bei knapper **Breite**, kein Trainings-Frühregime-Effekt. · Quantenbezug: Born-Regel |ψ|².
- **Mechanismus:** Statt eines Endpunkts eine Kurve: kleinste Breite h*, die ein Ziel (z. B. 95 % MNIST, 85 % Fashion) erreicht, für ReLU, GELU, z². Wenn h*(z²) ≈ ½ h*(ReLU), halbiert sich die Operationszahl in TinyML-Netzen (gleiche Genauigkeit). Wichtig: bei **vollem** Training (50+ Epochen), nicht knapp.
- **Langweiligste Erklärung:** Primer/ReLU²-Effekt, bekannt; bei langem Training verschwindet er und die Kurven fallen zusammen (r3 bei 18k: alles innerhalb 0,5 pp).
- **Billigster Test:** MNIST/Fashion 16×16, 1 Schicht, h ∈ {8, 12, 19, 32, 67}, 40 Epochen, Lernrate je Arm auf Val; 3 Tuning-Seeds ≈ 8 min. Maß: h* und Operationen bis Ziel.
- **Vorab-Tipp:** P(grün: h*(z²) ≤ 0,7·h*(ReLU) bei vollem Training) = 25 %, Jury 3.

## 16. Messverstärkung: NLL und gleiche Zufallszahlen statt Genauigkeit
· Denkwerkzeug: Fehlerbuch-Bergbau über das *ganze* Buch: I-001/I-004/I-004-mnist/I-020/C-1 enden wiederholt in HU, weil „0,15 G nicht aufgelöst“, „2,1 pp = ~11 Testbilder“, KI streift die Grenze. Mehr als ein Drittel der Einträge sind HU wegen Auflösung, nicht wegen Wahrheit. Frage: Was müsste wahr sein? Die Metrik, nicht der Effekt, ist zu grob: Genauigkeit ist ein 0/1-Zähler, die NLL (oder die Logit-Marge) ist kontinuierlich. · Quantenbezug: schwache Messung / Weak-Value-Verstärkung (statt ein Bit viele schwache Messungen mitteln).
- **Mechanismus:** Primäres Maß wird die gepaarte Differenz der Test-NLL (und Marge), gleiche Initialisierung und Batchreihenfolge (Common Random Numbers) zwischen Armen; Genauigkeit nur beschreibend. Ressource: weniger Seeds und Rechenminuten je Urteil, weniger HU. Kein Modellgewinn, aber Gewinn je CPU-Minute für jede der anderen Ideen.
- **Langweiligste Erklärung:** Alte Statistik-Praxis (Varianzreduktion, CRN); NLL verfehlt die Eigenschaft „Genauigkeit“, die die Jury versteht. Dann Genauigkeit beibehalten, NLL als Frühwarner.
- **Billigster Test:** r3-Aufbau (MNIST 5k, Arme relu/sq/sq_F, Seeds 985–994): bestimme für bekannte Paare (z. B. sq − relu +0,83 pp; sq_F − phase) das Verhältnis Effekt/Standardfehler für Genauigkeit gegen NLL, gepaart mit und ohne gemeinsame Init. ≈ 5 min. Maß: nötige Seedzahl für 80 % Teststärke.
- **Vorab-Tipp:** P(grün: NLL-Teststärke ≥ 2× Genauigkeit) = 75 %, Jury 2 (Methode, kein Modell).

---

## Auswertung für die Schlussnachricht
- Aus Fehlerbuch-Mustern stammen alle 16; keine Ideen aus Literatur-Zufall.
- Erwartete Top 3 nach Nutzen für die Arbeit (Wert × P × Ehrlichkeit): **#10** (Trainings-FLOPs-Gleichheit, schärfste Kontrolle der Rang-1-Linie), **#6** (Breite statt Training, schließt die I-041-Anomalie), **#5** (Training ohne Gradient, stärkster Quantenbezug mit Fehlerbuch-Beleg).
- Billige Gegenbefund-Kandidaten, die wegen der Fehlerbuch-Lehren trotzdem lohnen: #4, #12.
