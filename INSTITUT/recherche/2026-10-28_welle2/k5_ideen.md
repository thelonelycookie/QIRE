# k5 — Perspektiven (Welle 2, 28.10.2026): 6 Rollen × 3 Ideen

Gelesen: KREATIVITAET_2026-10-27, SYNTHESE, KERNAUSSAGE, LABORBUCH ab I-044, Vorregistrierungen I-047/48/49/45-v5. Nichts gerechnet, keine Websuche. Nicht wiederholt: Hadamard/Orthogonal + 2/3 Bit gegen klein/Niedrigrang/Entropie-Rang (I-048), TT (I-047), Hourglass/LRU (I-049), Fastfood/z² (I-045 ff.), Phasen, Born-Attention, frühes Aussteigen. Wirkung überall klassisch (I-038), Wortwahl „quanten-inspiriert“. „Gleiche Bytes“ = Bytes der Linearschichten wie in I-048 gezählt (inkl. Skalen/Bias).
Vorab-Tipp-Kalibrierung: P = Chance, dass der Rauchtest die Hypothese stützt UND die Kontrolle schlägt. Jury 1–5 = verblüffend + erklärbar.

---
## Rolle 1: Experimentalphysikerin (Quantenoptik)

### 1. Subtraktiver Dither für Gewichtsquantisierung
- **Subtraktiver Dither beim Runden** · Perspektive Experimentalphysikerin · Quantenbezug: In der Optik/Zählstatistik macht ein bekanntes Pseudo-Rauschen (Dither, wie Lock-in-Referenz) den Messfehler signalunabhängig und weiß.
- Mechanismus: Beim Quantisieren wird vor dem Runden ein aus dem Seed erzeugtes Pseudozufallsmuster addiert und beim Dekodieren wieder abgezogen (0 gespeicherte Bytes). Der Rundungsfehler wird dadurch von den Gewichten entkoppelt und über die Summe im Matmul unkorreliert, wächst also ∝ √n statt systematisch. Bei gleichen Bytes sinkt der Fehler des *Produkts* W·x, nicht der Gewichte, was bei 2 Bit mit grober Stufung am meisten ausmacht.
- Langweiligste Erklärung: Dither ist klassische Signalverarbeitung; der Gewinn ist reiner Bias-Abbau bei kleinen Gewichten nahe 0 (Totzone des 2-Bit-Gitters).
- Billigster Test: MNIST-MLP 784-512-512-10 (kurz, ≈ 5 Epochen) einmal trainieren, 2 Bit + Hadamard (Zeilenskala) mit/ohne subtraktivem Dither, 3 Tuning-Seeds. Kontrolle: Dither mit *nicht* abgezogenem Rauschen (Fehler) und feste Zufallsrundung; Metrik Test-Acc und relativer Fehler ‖Wx−Ŵx‖/‖Wx‖. ≈ 3 min.
- Vorab-Tipp: P = 35 %, Jury 3.

### 2. Photonenzählung: stochastisches Runden der Aktivierungen
- **Aktivierungen als Zählereignisse (unverzerrtes stochastisches Runden)** · Perspektive Experimentalphysikerin · Quantenbezug: Ein Detektor liefert ganzzahlige Klicks mit Poisson-Statistik, der Mittelwert ist trotzdem exakt.
- Mechanismus: Zwischenaktivierungen (und KV-Cache) werden auf 2–3 Bit stochastisch statt deterministisch gerundet; der Erwartungswert bleibt exakt, der Fehler mittelt sich in der nächsten Summe über viele Einheiten weg. Bei gleichen Bytes (Aktivierungs- bzw. Cache-Bytes) wäre ein tieferes Bitbudget nutzbar. Gilt für Inferenz mit langem Kontext, wo der Cache die Bytes dominiert.
- Langweiligste Erklärung: Stochastisches Runden ist bekannt (Gupta 2015); der Gewinn könnte nur Rundungs-Bias der deterministischen Variante sein.
- Billigster Test: MNIST-MLP fertig trainiert, Hidden-Aktivierungen 2/3 Bit deterministisch gegen stochastisch, 5 Wiederholungen des Rundungs-Zufalls, Rauschen als Mittel ± Spreizung. Kontrolle: deterministisches Runden mit Lloyd-Max-Gitter (gleiche Bits). ≈ 2 min.
- Vorab-Tipp: P = 40 %, Jury 3.

### 3. Zeno-Training: Projektion auf das Gitter alle k Schritte
- **Quanten-Zeno-Projektion statt Straight-Through** · Perspektive Experimentalphysikerin · Quantenbezug: Häufiges „Messen“ (Projizieren) hält einen Zustand fest; hier hält das Projizieren die Gewichte auf dem 2-Bit-Gitter.
- Mechanismus: Gewichte laufen in fp32 frei, werden aber alle k Schritte hart auf das Rotations-Gitter projiziert (nur projizierter Wert wird gespeichert/gemessen); k ∈ {1, 10, 100} steuert die Stärke. Gegenüber STE entstehen weniger gestörte Gradienten, bei gleichen Endbytes (2 Bit + Rotation). Erwartung: Optimum bei mittlerem k.
- Langweiligste Erklärung: Projected-Gradient/BinaryConnect-Varianten; der Gewinn ist bloß ein Lernraten-Effekt des Gitters.
- Billigster Test: MNIST-MLP, 2 Bit in Hadamard-Basis, k ∈ {1, 10, 100, ∞ (nur am Ende)} gegen STE, 5 Epochen, 3 Tuning-Seeds. Kontrolle: STE mit gleich gesuchter Lernrate; „∞“ = Nachher-Quantisierung (I-048-Arm had). ≈ 6 min.
- Vorab-Tipp: P = 30 %, Jury 4 (Zeno-Geschichte trägt).

---
## Rolle 2: Chip-Ingenieur (Speicherbandbreite, Integer-Einheiten, Cache)

### 4. Block-Hadamard in Cache-Zeilengröße
- **Blockdiagonale Hadamard (Blöcke 16/32/64 statt n)** · Perspektive Chip-Ingenieur · Quantenbezug: Mehrere kleine unabhängige „Qubit-Register“ statt eines großen; jeder Block ist ein kleines Hadamard-Gatterfeld.
- Mechanismus: Rotation nur innerhalb von Blöcken b (z. B. 32 Werte = 128 B = 2 Cache-Zeilen): FLOPs n·log b statt n·log n, Daten bleiben im L1, und die Skala kann je Block geteilt werden (feinere Skalen, ohne Extra-Bytes, wenn Block = Skalengruppe). Hypothese: Ausreißer lokal genug, dass b = 32 fast so gut glättet wie volles n.
- Langweiligste Erklärung: Die feineren Blockskalen sind der Hauptgewinn; die Rotation ist fast egal (siehe Idee 13).
- Billigster Test: MNIST-MLP, 2 Bit, Rotationsblock b ∈ {1 (keine), 8, 32, 128, n}, Skala je Block, Bytes exakt gleich gerechnet (Skalen je Block bezahlt). Kontrolle: dieselbe Blockskala ohne Rotation (b = 1). ≈ 4 min.
- Vorab-Tipp: P = 55 %, Jury 3.

### 5. Entropiecodierte Rotations-Gewichte (ECSQ)
- **Entropie-codiertes Gitter nach Rotation** · Perspektive Chip-Ingenieur · Quantenbezug: Nach der Rotation sind Gewichte nahezu gaußsch (Konzentration des Maßes); Shannon-Entropie des Born-ähnlichen Verteilungsmaßes bestimmt die Bytes.
- Mechanismus: Statt festen 2 Bit je Gewicht ein feineres Gitter (5–7 Stufen), dessen Symbole per Huffman/ANS im Mittel < 2 Bit kosten. Bei Gaußverteilung gewinnt ECSQ gegenüber festbreitem Quantisierer ≈ 0,25 Bit je Gewicht, bei gleichen Bytes also ein feineres Gitter. Dekodierung in einem Block-Stream ist auf der CPU vertretbar (Tabelle im L1).
- Langweiligste Erklärung: Klassische Ratenverzerrungstheorie (Entropie < Wortbreite); ohne Rotation ginge es mit schweren Rändern schlechter, weil der Huffman-Code dort verschwendet.
- Billigster Test: MNIST-MLP, Rotation + gleichmäßiges Gitter mit 3/5/7 Stufen, Entropie *exakt* ausgerechnet (Bits/Gewicht aus Häufigkeiten + Tabellenkosten), Vergleich bei gleicher Gesamtentropie mit festen 2 Bit. Kontrolle: dasselbe ohne Rotation (schwere Ränder). ≈ 5 min.
- Vorab-Tipp: P = 65 %, Jury 3.

### 6. Integer-Walsh-Hadamard (nur Addieren/Schieben)
- **Rotation vollständig in int8/int16 ohne Multiplizierer** · Perspektive Chip-Ingenieur · Quantenbezug: Das Hadamard-Gatter ist ±1-Addition/Subtraktion, also wörtlich Integer-Butterfly.
- Mechanismus: WHT auf int8-Aktivierungen mit wachsender Breite (int16-Akkumulator), danach Verschiebung statt 1/√n (Skala wandert in die Zeilenskala). Der Rotationspfad kostet nur Integer-Adds, die auf Zen 2 mit 256-Bit-AVX2 doppelt so breit wie fp32 laufen; Bytes der Aktivierungen bleiben 1 B. Hypothese: keine messbare Genauigkeitseinbuße gegenüber fp32-WHT.
- Langweiligste Erklärung: Integer-Rechnung ist exakt; der Gewinn ist reiner Durchsatz, nicht Qualität (also nur Energiebehauptung).
- Billigster Test: numpy/torch: Layer mit int8-Aktivierungen + int-WHT gegen fp32-WHT, Fehler ‖Δ‖/‖y‖ über 1000 Eingaben und Zeit je Aufruf bei n = 1024/4096 (einzelner Thread). Kontrolle: int8-Matmul ohne Rotation (Ausreißer dominiert Skala). ≈ 4 min.
- Vorab-Tipp: P = 75 % (Genauigkeit), Jury 2.

---
## Rolle 3: Jugend-forscht-Juror (verblüffend und erklärbar)

### 7. Ein Netz im Tweet: 1 Bit pro Gewicht dank Rotation
- **1-Bit-Netz, das 97 % schafft** · Perspektive Juror · Quantenbezug: Ein Gewicht ist nur noch ein Spin ±1; die Hadamard-Drehung macht aus „Spin-Glas mit Ausreißern“ ein gleichmäßiges Feld.
- Mechanismus: Nur Vorzeichen der rotierten Gewichte (+ eine Skala je Zeile) speichern; 1 Bit statt 16 Bit je Gewicht. Bei gleichen Bytes steht ein 16× breiteres Netz gegen ein winziges fp16-Netz. Verblüffend ist: ein Bild-Netz in ≈ 100 kB, das fast gleich gut ist wie fp16.
- Langweiligste Erklärung: BinaryConnect/XNOR bekannt; der Gewinn kommt aus der größeren Breite, nicht aus der Rotation (Kontrolle zeigt das).
- Billigster Test: MNIST-MLP groß trainiert, Vorzeichen-Gewichte mit Rotation (+ kurzes Nachtraining der Skalen) gegen klein fp16 mit gleichen Bytes; Arme: roh-Vorzeichen, rotiert, Orthogonale. 3 Seeds. Kontrolle: kleines dichtes fp16-Modell von Grund auf (wie I-048 „klein“). ≈ 5 min.
- Vorab-Tipp: P = 50 %, Jury 5.

### 8. Gewichte aus einer Zahl: die Hälfte würfelt der Computer
- **Gewichte = trainierte Koeffizienten × Seed-Basis (SeedLM-artig)** · Perspektive Juror · Quantenbezug: Haar-zufällige Zustände füllen den Raum fast gleichmäßig — eine Zufallsbasis ist „fast so gut wie jede andere“.
- Mechanismus: W = A·R mit R aus einem Seed (kostet 0 Bytes, wird beim Laden regeneriert), nur A gespeichert (Zeilen × K). Bei gleichen Bytes passt K größer als der Rang eines gelernten Niedrigrang-Paars (m+n)·r, wenn n ≫ m. Erklärbar: „Der Computer würfelt die Hälfte der Gewichte jedes Mal gleich neu aus einer Zahl.“
- Langweiligste Erklärung: Eingefrorener erster Faktor (LoRA-FA / Random Features); gewinnt nur, wenn n ≫ m, sonst verliert es klar gegen gelernten Niedrigrang.
- Billigster Test: MNIST-MLP, erste Schicht 784→512 als A·R (R Zufall, fest, K = 64…256) gegen gelernten Rang r mit gleichen Bytes und gegen klein dicht; 3 Seeds. Kontrolle: R = Hadamard-Zeilen (strukturiert) und R gelernt. ≈ 5 min.
- Vorab-Tipp: P = 35 %, Jury 5.

### 9. Wo die Rotation hilft, sagt die Kurtosis voraus
- **Rotation nur dort, wo die Gewichte „spitz“ sind** · Perspektive Juror · Quantenbezug: Rotation macht schwerrandige Verteilungen gaußsch (zentraler Grenzwert in der neuen Basis).
- Mechanismus: Je Schicht die Kurtosis messen; Rotation nur bei Kurtosis über Schwelle einbauen. Spart FLOPs und Rotationsbytes und liefert ein erklärbares Gesetz („Gewinn = Kurtosis − 3“), das in der Arbeit als Vorhersage und nicht als Beobachtung steht.
- Langweiligste Erklärung: Bekannte Ausreißer-Theorie (Outlier-Kanäle bei QuaRot); evtl. nur zwei Schichten tragen den ganzen Effekt.
- Billigster Test: MNIST-/Byte-LM-Modell (aus I-048-Smoke bzw. frisch kurz trainiert), je Schicht Kurtosis und Gewinn durch Rotation bei 2 Bit messen, Rangkorrelation; dann „selektiv rotieren“ (nur obere Hälfte) gegen alle/keine. Kontrolle: Zufallsauswahl der Schichten. ≈ 6 min.
- Vorab-Tipp: P = 45 %, Jury 4.

---
## Rolle 4: Harter Skeptiker (der eine Versuch)

### 10. Härtester klassischer Gegner: Gruppenskalen statt Rotation
- **Gruppenweise Skalen ohne Rotation bei exakt gleichen Bytes** · Perspektive Skeptiker · Quantenbezug: Prüft, ob die „Drehung“ überhaupt gebraucht wird oder ob feinere Skalen reichen.
- Mechanismus: Jede Gruppe von g = 16/32/64 Gewichten bekommt eine eigene fp16-Skala (Bytes bezahlt); große Ausreißer werden so lokal gefangen, ohne n·log n FLOPs. I-048 vergleicht nur mit Zeilenskala + Clip, nicht mit diesem üblichen Gegner. Bleibt die Rotation bei gleichen Bytes vorn, lebt die Kernaussage; verliert sie, ist sie begraben.
- Langweiligste Erklärung (falls Rotation doch gewinnt): Rotation kostet keine Skalen-Bytes, Gruppenskalen schon; der Vorteil ist dann reine Skalenbuchhaltung.
- Billigster Test: MNIST-MLP, 2 und 3 Bit: Gruppenskala g ∈ {16, 32, 64, 128} ohne Rotation gegen Rotation + Zeilenskala, Bytes exakt gleich, 3 Seeds. Kontrolle: Rotation + Gruppenskala (beides). ≈ 4 min.
- Vorab-Tipp: P = 40 % (dass Rotation bei gleichen Bytes vorn bleibt), Jury 4. Entscheidungsversuch für die Linie.

### 11. Die Rotation ist eine Krücke für nachträgliches Quantisieren
- **QAT gegen Rotation: verschwindet der Vorteil, wenn man mit Gitter trainiert?** · Perspektive Skeptiker · Quantenbezug: Fragt, ob der Basiswechsel Physik ist oder nur ein Pflaster für den Messfehler *danach*.
- Mechanismus: Ein 2-Bit-Netz, von Anfang an mit Gitter trainiert (STE, ohne Rotation), lernt Ausreißer selbst zu vermeiden und holt den Rückstand. Wenn ja: der QIRE-Gewinn gilt nur für Post-Training-Quantisierung. Wenn nein: Rotation hilft auch dem Training (stärkere Aussage).
- Langweiligste Erklärung (falls Rotation bleibt): Rotation macht das Gitter einheitlicher, STE-Gradient wird besser; nichts Quantenhaftes.
- Billigster Test: MNIST-MLP, vier Arme bei 2 Bit und gleichen Bytes: PTQ roh, PTQ rotiert, QAT roh, QAT rotiert; 5 Epochen, 3 Seeds. Kontrolle: fp32 ohne Quantisierung. ≈ 8 min.
- Vorab-Tipp: P = 30 % (Rotation behält Abstand unter QAT), Jury 4.

### 12. Wächst oder schrumpft der Vorteil mit der Breite?
- **Breitenskalierung des Rotationsgewinns** · Perspektive Skeptiker · Quantenbezug: Konzentration des Maßes wird mit n schärfer — der Gewinn sollte mit n wachsen.
- Mechanismus: Der Gewinn (Rotation vs. roh, Rotation vs. klein bei gleichen Bytes) wird für d = 64…1024 gemessen. Wächst er, trägt er auf große Modelle; schrumpft er, ist er ein Kleinmodell-Artefakt. Das ist die Frage, die jeder Juror stellt („gilt das bei echter Größe?“) und die Raven nur teuer beantwortet.
- Langweiligste Erklärung: Breitere Netze sind robuster gegen Rundung; der Vorsprung gegen *rohes* Quantisieren schrumpft, weil grobes Runden bei großem n von selbst mittelt.
- Billigster Test: MNIST-MLP mit Breite h ∈ {64, 128, 256, 512, 1024}, 2 Bit roh vs rotiert vs kleines dichtes fp16 gleicher Bytes, 3 Seeds; Kurve Gewinn(h) mit Streuband. Kontrolle: Skalierung der Streuung (keine Trendbehauptung bei überlappenden Bändern). ≈ 9 min.
- Vorab-Tipp: P = 50 % (Gewinn wächst oder bleibt), Jury 4.

---
## Rolle 5: Zehnjährige („warum macht man das nicht einfach so …“)

### 13. Warum nicht Kugeln statt Würfel packen?
- **Mehrdimensionale Gitter (Paare/Vierer) statt Einzelzahlen runden** · Perspektive Zehnjährige · Quantenbezug: Ein Kristall packt Kugeln dichter als ein Würfelgitter; die Gitterpackung heißt in der Physik Kristallstruktur.
- Mechanismus: Zwei benachbarte rotierte Gewichte werden als Punkt der Ebene gerundet; ein hexagonales (oder k-Means-)Codebuch mit 16 Einträgen kostet ebenfalls 4 Bit je Paar = 2 Bit je Gewicht, hat aber kleineren mittleren Fehler (bis +0,17 dB bei 2D, mehr in 4D/8D wie E8). Nach der Rotation sind Paare rund-gaußsch, ideal für Kugelpackung.
- Langweiligste Erklärung: Vektorquantisierung; der Gewinn ist die bekannte „Granular-Gain“ von höchstens 1,53 dB, im Alltag ein paar Zehntel dB.
- Billigster Test: Gaußsche Zufallsgewichte und das MNIST-MLP: 2 Bit skalar gegen hexagonales/k-Means-Paar-Codebuch (16 Zentren) bei gleichen Bytes; Fehler in dB und Test-Acc. Kontrolle: derselbe Test ohne Rotation (Verteilung nicht rund). ≈ 4 min.
- Vorab-Tipp: P = 60 % (messbarer dB-Gewinn), 25 % (messbarer Acc-Gewinn), Jury 4.

### 14. Warum nicht dasselbe Stück mehrmals benutzen?
- **Gewichtsteilung in der Tiefe mit wechselnder Drehung (Trotter-Schritte)** · Perspektive Zehnjährige · Quantenbezug: Eine Quantenzeitentwicklung wird in viele gleiche kleine Schritte zerlegt (Trotter); hier ist jede Iteration derselbe Operator in anderer Basis.
- Mechanismus: Eine einzige Transformer-Lage wird L-mal angewandt; vor jeder Anwendung wechselt eine feste, billige Basis (zufällige Vorzeichen D_t und Hadamard, 0 Bytes), damit die Wiederholungen nicht identisch sind. Bytes: 1 Lage statt L, deshalb kann d breiter werden. Mehr FLOPs je Byte — eine Umkehrung der FLOP-Schiene, ehrlich als Bytes-Aussage.
- Langweiligste Erklärung: ALBERT/Universal-Transformer-Gewinn (breiter bei gleichen Bytes); die Basiswechsel sind nebensächlich — die Kontrolle ohne sie zeigt das.
- Billigster Test: Byte-LM (d = 128, Kontext 128, ≈ 300 Schritte, Tuning-Seeds), Arme gleiche Bytes: (a) 4 getrennte Lagen d = 64, (b) 1 geteilte Lage d = 128 × 4 Durchläufe, (c) wie (b) mit festen wechselnden Drehungen. Kontrolle: (b) ohne Drehung; FLOPs ehrlich gezählt (b, c haben mehr). ≈ 9 min. Vorsicht Frühregime: Wiederholung bei längerem Training vorsehen.
- Vorab-Tipp: P = 40 % (b/c schlagen a bei gleichen Bytes), 15 % (c schlägt b), Jury 3.

### 15. Warum nicht erst kurz üben, dann lang? (adiabatisches Curriculum)
- **Kontextlänge langsam hochfahren (Curriculum) bei gleichen Trainings-FLOPs** · Perspektive Zehnjährige · Quantenbezug: Adiabatisches Theorem — langsam verändern, damit das System im Grundzustand bleibt.
- Mechanismus: Training startet mit Kontext 32, wächst auf 128; kurze Fenster kosten weniger Attention-FLOPs und liefern in der frühen Phase genauso gute Gradienten. Bei gleichen Gesamt-FLOPs gibt es mehr Schritte. Gewinn wäre eine Trainings-FLOP-Aussage, nicht Inferenz.
- Langweiligste Erklärung: Sequenzlängen-Warmup ist bekannt (Li et al. 2022); der Gewinn ist ein Frühregime-Effekt, der bei langem Training verschwindet (I-045-Lehre).
- Billigster Test: Byte-LM d = 128, 2 Lagen: fest Kontext 128 für 300 Schritte gegen Curriculum 32→128 mit *gleichen gezählten Trainings-FLOPs* (mehr Schritte); Test-bpc auf Kontext 128. Kontrolle: fest 32 und fest 64 bei gleichen FLOPs; Zählung wie I-045-v5. ≈ 9 min.
- Vorab-Tipp: P = 45 % (kurzfristig grün), Jury 3. Langzeitgültigkeit niedrig (15 %).

---
## Rolle 6: Energieberater (Joule pro Token auf Maltes Ryzen 5 3600)

### 16. Selbst-Spekulation mit dem 2-Bit-Zwilling
- **2-Bit-Rotationskopie als Entwurfsmodell (verlustfrei)** · Perspektive Energieberater · Quantenbezug: Erst grob „messen“, dann einmal genau verifizieren — Näherung plus Verifikation, wie bei vielen Quanten-Hybridverfahren.
- Mechanismus: Dasselbe Modell existiert doppelt: fp16/fp32 (Ziel) und ein 2-Bit-Hadamard-Zwilling (+ 1/8 der Bytes). Der Zwilling entwirft k Token billig (Batch 1 ist bandbreitengebunden, Zeit ∝ Bytes), das große Modell prüft alle k auf einmal; Ausgabe bleibt exakt gleich. Erwartete Joule je Token sinkt um den Faktor 1/(Akzeptanzrate · Byteverhältnis). Das ist die einzige Idee in dieser Liste, die Joule *ohne Qualitätsverlust* senkt.
- Langweiligste Erklärung: Selbst-spekulatives Dekodieren existiert (Zhang et al. 2023, Draft & Verify); der Gewinn ist reine Bandbreitenlogik.
- Billigster Test: Byte-LM kurz trainiert (oder MNIST-Klassenübereinstimmung als Ersatz): Top-1-Übereinstimmung Zwilling gegen Ziel (Akzeptanzrate α), daraus Speedup = (1 − α^(k+1)) / ((1 − α)(k·c + 1)) mit c = 1/8 (Byteverhältnis). Kontrolle: Zwilling roh-2-Bit ohne Rotation und kleines dichtes Modell mit gleichen Bytes als Entwurf. ≈ 6 min.
- Vorab-Tipp: P = 60 % (α(Rotation) > α(roh)), Jury 4.

### 17. Bandbreiten-Roofline: ab welcher Matrixgröße lohnt 2 Bit in Sekunden?
- **Zeit/Token-Kurve gegen Bytes (Roofline) auf Zen 2** · Perspektive Energieberater · Quantenbezug: keiner im Kern; die Messung entscheidet, ob die Bytes-Aussage überhaupt in Joule ankommt.
- Mechanismus: Auf einem Ryzen 5 3600 (L3 2 × 16 MB, DRAM-Bandbreite ≈ 40 GB/s) liegt Batch-1-Matvec unter L3-Größe im Cache und über L3 an der Bandbreite. 2-Bit-Gewichte (+ Entpacken + WHT) gewinnen erst ab einer Matrixgröße, die die Kurve liefert; das ist der Grenzwert „ab hier wirken Bytes als Energie“. Fehlt dieser Beleg, bleibt die Bytes-Aussage Papier.
- Langweiligste Erklärung: Standardmäßige Speicherwand; das Ergebnis ist bekanntes Systemwissen, aber für die Arbeit unverzichtbar.
- Billigster Test: torch/numpy: Matvec-Zeit je Aufruf für n = 512…8192 (einzelner Thread) in fp32 gegen int8 gegen gepackt 2 Bit (Entpacken per Tabelle/Shift) + WHT; Messung als Zeit-pro-Byte. Container ist nicht der Ryzen, deshalb zusätzlich Verhältnisse notieren und auf Maltes PC wiederholen (RAPL/`powercap` falls verfügbar, sonst Zeit × Leistungsannahme). Kontrolle: Leerlauf-Basislinie und fp32-Matvec bei gleichem n. ≈ 8 min.
- Vorab-Tipp: P = 55 % (2 Bit schneller ab ≈ 4096), Jury 3.

### 18. Born-Stichprobe: Matmul durch k „Messungen“ ersetzen
- **Born-Auswahl der Spalten (Importance Sampling ∝ z²)** · Perspektive Energieberater · Quantenbezug: Born-Regel — man misst einen Ausgang mit Wahrscheinlichkeit ∝ Amplitude².
- Mechanismus: Der zweite FFN-Matmul y = W2·z wird nur auf k aus z mit Wahrscheinlichkeit ∝ z_i² gezogenen Spalten berechnet (unverzerrter Schätzer, Drineas–Kannan–Mahoney), dazu die Spalten nur einmal gelesen. Bei squared-ReLU sind viele z_i nahe 0, das Verfahren spart FLOPs *und* Bytes von W2 (nur gezogene Spalten laden). Joule je Token sinkt mit k/h.
- Langweiligste Erklärung: Deterministisches Top-k (Deja Vu, Sparsity) schlägt jedes Sampling bei gleichem k; Born-Sampling ist nur die zufällige, schlechtere Version. Die Kontrolle entscheidet.
- Billigster Test: Kleines Byte-LM bzw. MNIST-MLP mit relu²-Aktivierung (D_relu2-Stil) kurz trainiert; zweiter Matmul mit k ∈ {5, 10, 25 %} der Spalten: Born-Sampling gegen Top-k-Betrag gegen Gleichverteilung gegen kleineres dichtes h gleicher FLOPs; Metrik Fehler und Acc/bpc. Kontrolle: Top-k deterministisch. ≈ 7 min.
- Vorab-Tipp: P = 30 % (Born schlägt Gleichverteilung), 10 % (Born schlägt Top-k), Jury 4.

---
## Hinweise zur Konvergenz
- Entscheidungsversuche (Skeptiker): Nr. 10 (Gruppenskalen), Nr. 11 (QAT), Nr. 12 (Breite). Wenn alle drei bestehen, ist die Bytes-Linie belastbar; fällt einer, ändert sich die Kernaussage.
- Stärkste Joule-Linie: Nr. 16 und 17 (Bytes → Energie), unabhängig von Qualitätsfragen.
- Überschneidungsrisiko mit anderen Kürzeln: 4/9/12 (Rotationsvarianten), 14 (Umkehrung), 15 (Curriculum). Bei Dopplung zusammenführen.
- Alle Tests: Tuning-Seeds (≥ 900), keine Mess-Seeds.
