# Konvergenz Welle 2 (28.10.2026): 20 Kandidaten + Ergänzungen I-048

Quelle: k1–k5 (78 Ideen: k1 16, k2 16, k3 16, k4 12, k5 18). Nichts gerechnet. Herkunft = Kürzel#Nr. Vorab-P = Mittel der Berichte (grob gerundet; wo die Berichte zwei P nennen, steht die vorsichtigere Zahl gegen die starke Kontrolle). Jury = Mittel 1–5. Wirkung überall klassisch (I-038), „quanten-inspiriert“. Literaturnamen stammen aus dem Gedächtnis der Berichte, vor Verwendung prüfen.
Währung: B = Bytes, F = FLOPs, D = Daten/Regime, TF = Trainings-FLOPs, J = Joule/Latenz.

## A. Die 20 Kandidaten

| ID | Titel | Herkunft | Mechanismus (1 Satz) | Währung | Billigster Test mit Kontrolle | P | Jury |
|---|---|---|---|---|---|---|---|
| K01 | Schlüssel-Superposition über Lagen | k1#1, k3#3, k5#14 | Alle Lagen teilen eine FFN-Matrix, jede Lage liest sie durch eigene feste Vorzeichen-Diagonalen (und Drehung) anders. | B (FLOPs gleich) | Byte-LM d=128, 4 Lagen, 600–3000 Schritte: normal / geteilt ohne Schlüssel / geteilt mit Schlüsseln, je auch auf gleiche Bytes verbreitert. Kontrolle: Teilung ohne Schlüssel, kleines dichtes Modell gleicher Bytes. ≈ 8 min | 38 % | 4 |
| K02 | Aufgaben-Superposition (Adapter) | k3#9 | Mehrere Aufgaben liegen in einem Gewichtssatz, ein Vorzeichen-Schlüssel wählt die Aufgabe. | B | MNIST, Fashion, MNIST mit Pixelpermutation: ein geteiltes Modell mit Schlüsseln gegen 3 getrennte Modelle gleicher Bytes. Kontrolle: geteilt ohne Schlüssel (nur Etikett). ≈ 6 min | 35 % (vs. ohne Schlüssel 20 %) | 4 |
| K03 | Vektor-Codebuch ohne Speicher | k1#2, k3#2, k5#13 | Gewichtsblöcke (8 Stück, nach Rotation) werden durch das nächste Wort eines Seed-Zufalls-, Reed-Muller- oder Hexagon-Codebuchs ersetzt, das nie gespeichert wird. | B | Trainiertes MNIST-MLP 784-512-512-10 ohne Nachtraining: Fehler und Genauigkeit bei gleichen Bytes. Kontrolle: Skalar 2 Bit + Hadamard (I-048-Arm), ternär, E8-Gitter; ohne Rotation. ≈ 4 min | 45 % (gegen I-048-Arm 30 %) | 3,7 |
| K04 | Eingefrorene Seed-Aufwärtsschicht | k3#1, k5#8 | Die erste FFN-Matrix (oder A·R) kommt aus einem Seed (0 Bytes), nur Zeilenskalen und die Auslese werden gelernt, dafür wird breiter gebaut. | B | Byte-LM d=128, 4 Lagen, 600 Schritte: Zufalls-W1 + Skalen (breiter) gegen dicht gelernt (gleiche Bytes). Kontrolle: ohne Zeilenskalen; R = Hadamard-Zeilen; R gelernt. ≈ 6 min | 33 % | 4 |
| K05 | Cache-Bytes: Rotation, 1-Bit-Skizze, Haar-Alterung | k1#5, k3#12, k4#5, k4#6, k5#2 | Schlüssel/Werte werden vor dem Ablegen gedreht und mit 1–2 Bit (oder als Vorzeichenskizze/Haar-Grobwerte für alte Token) gespeichert. | B (Cache) | Kleines Byte-LM normal trainiert, nur bei der Auswertung KV quantisieren, bpc gegen Cache-Bytes. Kontrolle: Rundung ohne Rotation mit Kanalskala, Zufallskanäle, PCA-Skizze, Gleitfenster, Zufalls-Fallenlassen. ≈ 8 min | 40 % (vs. PCA 20 %) | 3 |
| K06 | Haar/Walsh-Basis für Gewichte | k1#3, k4#4 | Gewichte werden in der Haar-/Walsh-Basis abgelegt; grobe Skalen bekommen mehr Bits oder nur die größten Koeffizienten bleiben. | B | Trainiertes MNIST-MLP, gleiche Bytes: Haar-Top-m / Walsh-Top-m / Betrags-Top-m / SVD-Rang, und Haar-Bitallokation gegen gleichmäßig 2 Bit. Kontrolle: Zufalls-Orthogonale (gleiche Bitverteilung/Top-m). ≈ 5 min | 27 % (Eingangsschicht 45 %) | 3 |
| K07 | Optimiererzustand rotiert, 2–4 Bit | k1#7, k3#11, k4#9 | Adams m und v werden blockweise gedreht und mit wenigen Bit (v in Log-Skala bzw. Sketch) gespeichert. | B (Training) | Byte-LM d=128, 600–2000 Schritte: Adam fp32 / 8 Bit / 4 Bit roh / 4 Bit rotiert / 2 Bit rotiert, bpc gegen Zustand-Bytes. Kontrolle: 8-Bit-Blockadam, Adafactor. ≈ 8 min | 27 % | 3 |
| K08 | Adam in gedrehter Basis | k1#6 | Der Gradient wird mit festen Vorzeichen-Hadamard-Matrizen gedreht, Adam rechnet dort, dann Rückdrehung. | TF | Byte-LM d=128, 2000 Schritte, Lernrate je Arm getunt: Adam / rotiert / nur Vorzeichen (ohne Mischung) / Zufalls-Orthogonale; bpc je Trainings-FLOP inkl. Rotation. ≈ 9 min | 25 % | 3 |
| K09 | Born-Normierung der FFN-Mitte | k2#1, k4#2 | Statt z² rechnet die FFN z²/(Mittel z² + ε) (divisive Normierung, Summe 1 wie die Born-Regel), was die Breiten-/Lernraten-Abhängigkeit des Quadrats entfernen könnte. | F (gleich) | Byte-LM d=64 und 128, T=600–1500, Lernrate je Arm getunt: GELU / z² / z²+Born-Norm (Gruppen 8/32) / z²+LayerNorm davor. Kontrolle: LayerNorm-Arm. ≈ 9 min | 30 % | 3,5 |
| K10 | Born-Stichprobe der FFN-Spalten | k1#4, k5#18 | Die zweite FFN-Matrix wird nur auf S Spalten gerechnet, gezogen mit Wahrscheinlichkeit ∝ Aktivierung² (erwartungstreu). | F (und B beim Laden) | Byte-LM d=128 (relu²), 2500 Schritte, S=h/8…25 %: dicht voll / dicht schmal FLOP-gleich / Born-Sampling / Top-S. Kontrolle: Gleichverteilung, deterministisches Top-S. ≈ 7 min | 20 % (vs. Gleichverteilung 30 %, vs. Top-S 10 %) | 4 |
| K11 | Fliegen-Hash-Kopf | k4#3 | Feste Hadamard-Expansion auf 8-fache Breite, nur die k größten Werte bleiben, nur diese gehen in die gelernte Auslese. | F | MNIST/Fashion-Kopf auf festen Merkmalen: FlyHash (m=8192, k=64) gegen dichten Kopf gleicher FLOPs. Kontrolle: Top-k mit Zufalls-Spaltenwahl. ≈ 6 min | 40 % | 4 |
| K12 | Bit-Baum / Hadamard-Code als Ausgabekopf | k1#11, k3#5 | Der 256-Wege-Softmax wird durch 8 binäre Entscheidungen oder ein festes 32-Bit-Hadamard-Codewort-Buch ersetzt. | F, B (Kopf) | Byte-LM d=128, 600–3000 Schritte: voller Softmax / Bit-Baum / Hadamard-Code 32 / Zufalls-Code 32 / Niedrigrang-Kopf 32, bpc bei gleichen Gesamt-FLOPs bzw. Bytes. Kontrolle: Zufalls-Code, gelernter Rang-32-Kopf. ≈ 7 min | 25 % (bpc gleichauf 50 %) | 3 |
| K13 | Grob-zu-fein-Curriculum (Kontextlänge) | k1#12, k5#15 | Das Training startet mit kurzem Kontext (32/64) und wächst auf 128/256, bei festem Trainings-FLOP-Budget. | TF | Byte-LM d=128: fester Kontext / Warmup 64→256 / Pool-Curriculum, Budget in gezählten Trainings-FLOPs (Zählung wie I-045-v5). Kontrolle: fest 32 und fest 64 bei gleichen FLOPs. ≈ 8 min | 45 % kurz, 15 % bei langem Training | 3 |
| K14 | Wachsendes Modell (isometrisch verbreitern) | k3#15 | Training beginnt mit halber Breite und wird funktionserhaltend auf volle Breite erweitert. | TF | Byte-LM, 600 Schritte (und 2400 für Langzeit): d=64→128 / d=128 durchgehend bei gleichen Gesamt-FLOPs / d=128 volle Schritte (obere Schranke). Kontrolle: Wachsen ohne Funktionserhaltung. ≈ 9 min | 35 % kurz, 20 % lang | 3 |
| K15 | Rotiertes Dropout und Schuss-Rauschen | k1#8, k1#10 | Dropout (oder Bernoulli-Schussrauschen auf Pixeln) wirkt in gedrehter Basis und stört alle Merkmale gleichmäßig. | D | Byte-LM d=128 auf halben Trainingsbytes (Überanpassung) bzw. MNIST mit 2000 Bildern: ohne / Dropout getunt / rotiert getunt / Gauß-Rauschen. Kontrolle: normales Dropout mit abgestimmter Rate; Train- und Test-bpc. ≈ 9 min | 30 % | 3 |
| K16 | Trotter-Stapel: Fastfood-Schichten mit Skip | k2#2 | Mehrere Fastfood-Lagen als Residualstapel x + α·σ(H D₂ H D₁ x) mit α anfangs 0, damit Tiefe nicht mehr schadet. | F (ideal gezählt) | MNIST 16×16: S1 / S3 / S3+Skip / dicht 1-Schicht gleicher Operationen / dicht 3-Schicht mit Skip, 20 Epochen. Kontrolle: Skip-Stapel mit fest zufälligen Diagonalen. ≈ 6 min | 35 % | 3 |
| K17 | Zufallsmerkmale + geschlossener Kopf, Suche statt Gradient | k2#5, k2#6 | Breite Fastfood-z²-Merkmale mit Ridge-Kopf in einem Durchgang; optional die Vorzeichen per Bit-Flip-Suche statt Backprop wählen. | TF, F | MNIST 16×16 (auch hochgerechnet auf 32×32): Fastfood²+Ridge bei D=N/4N/16N, Suche gegen beste von 400 Zufallsdraws. Kontrolle: Zufalls-ReLU-Merkmale gleicher Zahl, trainierter 1-Schicht-MLP gleicher Inferenz-FLOPs. ≈ 8 min | 28 % | 4 |
| K18 | 2-Bit-Zwilling als Entwurf + Bandbreiten-Messung | k5#16, k5#17, k5#6 | Die 2-Bit-Kopie rät Token billig, das große Modell prüft sie; eine Zeit-gegen-Bytes-Messung (auch Integer-Walsh) zeigt, ab wann Bytes zu Joule werden. | J (Bytes → Zeit) | Byte-LM: Übereinstimmung Zwilling/Ziel (Akzeptanzrate) und Matvec-Zeit für n=512…8192 (fp32 / int8 / 2 Bit gepackt + Walsh). Kontrolle: 2 Bit roh ohne Rotation, kleines dichtes Modell als Entwurf, Leerlauf. ≈ 8 min | 55 % | 3,5 |
| K19 | Lokale Blöcke auf der Pixelseite (Z-Kurve, getrennte Filter) | k2#7, k2#8 | Erste Schicht aus Blöcken in Z-Ordnung bzw. Filter als Produkt a⊗b, weil die Pixelindizes Struktur haben. | F, B | MNIST/Fashion 16×16, n=200/1000/6000: Z-Quadranten / Streifen / zufällige Permutation / dicht FLOP-gleich / dicht Parameter-gleich; Kronecker-Filter gegen U·V. Kontrolle: Permutation, gleiche Parameter. ≈ 6 min | 40 % | 3 |
| K20 | Beschränkte Schwellen-Aktivierung (Hill, n=2) | k4#11 | a = z²/(z²+θ²) liegt in [0,1] und lässt sich ohne Ausreißer in 3–4 Bit speichern. | B (Aktivierung) | Byte-LM d=128, 1500 Schritte, FFN-Mitte 3/4 Bit (STE): Hill / GELU / z² / ReLU6, bpc je Aktivierungs-Bytes. Kontrolle: ReLU6 (beschränkt). ≈ 9 min | 35 % (vs. GELU 55 %, vs. ReLU6 20 %) | 4 |

## B. Einzeiler für Malte (je ≤ 15 Wörter)

- **K01:** Ein Gewicht für alle Schichten, kleine Schlüssel unterscheiden sie: viel weniger Speicher.
- **K02:** Ein Netz für mehrere Aufgaben, ein Schlüssel stellt die richtige Aufgabe ein.
- **K03:** Gewichte werden durch Zufallsmuster aus einer Zahl ersetzt, die nie gespeichert werden müssen.
- **K04:** Die Hälfte des Netzes würfelt der Computer aus einer Zahl, nur der Rest wird gelernt.
- **K05:** Der Gedächtnisspeicher des Sprachmodells wird mit sehr wenigen Bit abgelegt, ohne viel zu verlieren.
- **K06:** Gewichte erst in grobe und feine Anteile zerlegen, dann die groben genauer speichern.
- **K07:** Auch die Lernhilfen beim Training brauchen weniger Speicher, wenn man sie vorher dreht.
- **K08:** Lernschritte in einer gedrehten Sicht rechnen, damit das Training schneller ans Ziel kommt.
- **K09:** Neuronen-Werte so umrechnen, dass sie sich wie Wahrscheinlichkeiten zu eins aufsummieren.
- **K10:** Nur die auffälligsten Neuronen werden gerechnet, ausgewählt per Würfel mit passenden Chancen.
- **K11:** Wie eine Fliege riechen: erst grob auffächern, dann nur die stärksten Signale nutzen.
- **K12:** Ein Byte wird nicht auf einmal geraten, sondern in acht Ja-Nein-Schritten.
- **K13:** Erst mit kurzen Texten üben, dann mit langen: gleiche Rechenarbeit, besseres Ergebnis?
- **K14:** Klein anfangen und das Netz später breiter machen, statt gleich groß zu trainieren.
- **K15:** Absichtliches Rauschen in der gedrehten Sicht soll das Auswendiglernen verhindern.
- **K16:** Viele billige Mischschichten hintereinander, mit Abkürzung, damit Tiefe endlich nützt.
- **K17:** Feste Zufallsmerkmale und ein einmal berechneter Ausgang ersetzen langes Training.
- **K18:** Ein winziges Schnellmodell rät vor, das große prüft nur: weniger Strom pro Wort.
- **K19:** Nahe Bildpunkte gehören zusammen: die erste Schicht schaut nur auf Nachbarn.
- **K20:** Eine Aktivierung, die nie ausreißt, lässt sich mit sehr wenigen Bit speichern.

## C. Ergänzungen I-048 (gleiche Aufgaben, MNIST-MLP/Byte-LM, gleiche Bytes; Kontrollen und Auswertung, keine neuen Bausteine)

| ID | Titel | Herkunft | Idee und Kontrolle | P | Jury |
|---|---|---|---|---|---|
| E1 | Gruppenskalen statt Rotation | k5#10 | Skala je 16/32/64/128 Gewichte ohne Rotation (Skalen-Bytes bezahlt) gegen Rotation + Zeilenskala; Kontrolle Rotation + Gruppenskala. Entscheidungsversuch für die Linie. ≈ 4 min | 40 % (Rotation vorn) | 4 |
| E2 | Trainings-FLOP-Gleichheit | k2#10 | Kleines dichtes Modell mit 10/25/60 Epochen und Lernratensuche (gleiche Trainings-FLOPs wie groß + Rotation); Maß: Genauigkeit gegen Trainings-FLOPs und Bytes. Wichtigste Juror-Frage. ≈ 8 min | 40 % | 4 |
| E3 | QAT-Paket | k5#11, k5#3, k1#9 (+ k1 Teil B) | 2 Bit mit Gitter trainiert, roh gegen rotiert (STE); Varianten: Projektion alle k Schritte (Zeno), weicher Übergang fp→2 Bit mit Lückenregel gegen linearen Zeitplan. Zeigt, ob der Vorteil nur Nachher-Quantisierung ist. ≈ 8 min | 30 % (Rotation bleibt vorn); adaptiver Zeitplan 45 % | 4 |
| E4 | Block-Hadamard (Blockgröße) | k5#4 | Drehung nur in Blöcken 8/32/128/n, Skala je Block, Bytes exakt; Kontrolle: Blockskala ohne Rotation. ≈ 4 min | 55 % | 3 |
| E5 | Breitenskalierung | k5#12 | Gewinn Rotation vs. roh vs. klein gleicher Bytes für h = 64…1024 mit Streuband (wächst oder schrumpft der Vorteil?). ≈ 9 min | 50 % | 4 |
| E6 | Nur Skalen/Bias nachtrainieren | k2#9 | Nach Rotation + 1–2 Bit nur die ohnehin gespeicherten Zeilenskalen und Bias nachtrainieren (0 Zusatzbytes); Kontrolle: ohne Rotation, volles STE. ≈ 6 min | 45 % | 3 |
| E7 | Kurtosis-Gesetz, selektive Rotation | k5#9 | Je Schicht Kurtosis messen, Rotation nur bei hoher Kurtosis; Kontrolle: Zufallsauswahl der Schichten. ≈ 6 min | 45 % | 4 |
| E8 | Rotation nur auf der dichten Seite | k2#11 | W₁ rotieren, W₂ in Identitätsbasis mit Top-K-Überspringen (ReLU²), 2 Bit; Kontrolle: überall / nirgends rotiert. ≈ 6 min | 40 % | 3 |
| E9 | Quantisierung als Regularisierer | k2#12 | Überangepasstes Byte-LM nachträglich 3/4 Bit rotiert; Kontrolle: bester Val-Stand fp32, gleich starkes Gauß-Gewichtsrauschen. ≈ 9 min | 20 % | 3 |
| E10 | Entropiecodierung nach Rotation | k3#8, k5#5 | Feineres Gitter (5–7 Stufen) mit Huffman/ANS bei gleicher Gesamtentropie wie 2 Bit; Kontrolle: ohne Rotation (schwere Ränder). ≈ 5 min | 60 % | 2,5 |
| E11 | Subtraktiver Dither | k5#1 | Seed-Rauschen vor dem Runden addieren, beim Dekodieren abziehen (0 Bytes); Kontrolle: Rauschen nicht abgezogen, feste Zufallsrundung. ≈ 3 min | 35 % | 3 |
| E12 | 1-Bit-Netz mit Skalen-Nachtraining | k5#7 | Nur Vorzeichen der rotierten Gewichte + Zeilenskala gegen kleines fp16 gleicher Bytes; Kontrolle: roh-Vorzeichen, kleines dichtes Modell. ≈ 5 min | 50 % | 5 |

## D. Gestrichen oder Reserve (alle 78 sind einsortiert)

Gestrichen, weil Linie läuft oder Gegenbefund absehbar:
- Nahe an I-049 (Hourglass/LRU): k2#13 feste Zerfallsbank. Besser als Zusatzarm im Nachfolgelauf von I-049.
- Nahe an I-045-v5 (Sz/Fastfood-Linie): k2#14 Sz→dicht.
- Nahe an I-047 (Tensornetz, nur Parameter) und von den Berichten selbst als „zufällig dünn“ erwartet: k1#16, k3#4 (Pauli-Summe), k3#16 (Walsh-Koeffizienten).
- Langweiligste Erklärung ist laut I-045-v2/v4 schon belegt (Aktivierung, nicht Struktur): k4#1 dendritische Äste (= Block + z²).

Reserve (billig, aber Dopplung oder schwach; nicht in den 20):
- k1#13 Entropie-Datenauswahl, k1#14 komplexes Momentum, k1#15 Weich-zu-hart-Attention.
- k2#3 Clifford-Fastfood ±1 (P 20 %), k2#4 Bent/Reed-Muller (Gegenbefund zu I-036), k2#15 Breite bis Ziel für z².
- k2#16 NLL und gemeinsame Zufallszahlen: Methode, empfohlen als Querschnitt für alle Tests (P 75 %).
- k3#6 Walsh-Eingabe beschneiden (PCA schlägt es), k3#7 eingabeabhängige Fastfood-Diagonalen (GLU-Verdacht), k3#13 Dither-Mittel (ehrliche Negativ-Erkenntnis), k3#14 Rotations-Ensemble.
- k3#10, k4#12 umkehrbare Schichten (RevNet = Checkpointing), k4#7 holografischer Speicher (nur als Kapazitätstest), k4#8 Vorzeichen-Hash, k4#10 Gray-Code.

## E. Hinweise zur Reihenfolge
- Zuerst I-048 abwarten, dann E1, E2, E3, E5 (Entscheidungsversuche); K03, K05, K06, K07 teilen dieselben Gegner und Skripte (Byte-Paket).
- Ohne Konflikt zu I-048: K01, K04, K12, K13, K18 (Byte-LM-Rauchtests auf Tuning-Seeds).
- Währung der 20: Bytes 8 (K01–K07, K20), FLOPs 6 (K09–K12, K16, K19), Trainings-FLOPs 4 (K08, K13, K14, K17), Daten 1 (K15), Joule 1 (K18).
