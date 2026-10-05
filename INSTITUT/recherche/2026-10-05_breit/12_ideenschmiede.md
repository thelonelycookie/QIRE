# 12 — Freie Ideenschmiede: Quantenphysik x effiziente KI (05.10.2026)

## 0. Wichtiger Vorbehalt zur Neuheitsprüfung

**Das WebSearch-Budget der Sitzung war schon aufgebraucht** (200/200, Fehlermeldung bei den
ersten acht Versuchen). Es gibt daher in diesem Bericht **keine einzige belegte Websuche**.
Alle Neuheitsangaben sind `[Gedächtnis]` (Trainingswissen, Stand ca. Mitte 2026) oder `[?]`.
Ich habe das Budget nicht über WebFetch-Umwege "nachgebaut", weil die Sperre ausdrücklich
sagt, der Nutzer solle sie anheben (`CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`). Die in
jeder Idee genannte **Suchanfrage** ist fertig formuliert und muss in Etappe 2 nachgeholt
werden, bevor irgendetwas als neu behauptet wird. Genannte Literatur ohne URL = aus dem
Gedächtnis, ungeprüft.

Gelesen: 00_AUFTRAG, UEBERGABE §0, IDEEN.md, ideen.json (35 Karten; Status `verworfen`
betrifft u. a. Trotter, Pauli-Propagation, Quantum Walk (I-011), klassische Schatten, MERA-
Grobkörnung (I-016/021), ZNE, Adiabatik (I-025/030), Born-Nichtlinearität (I-032),
Imaginärzeit, Lieb-Robinson). Ich habe diese **nicht** als Ideen wiederholt; wo eine
Idee daran angrenzt, steht es dabei.

## 1. Kurzfassung

- 17 Ideen (A1–A17), je mit Frage, Stufe, Suchanfrage, billigstem PC-Test, Kontrolle, Potenzial.
- **Der Hebel für „MNIST mit 100 Parametern" ist der Kopf (2.570 von ~2.600 Parametern).**
  Alle aussichtsreichen Ideen greifen dort an: Kopf durch Messung eines Klassenregisters
  ersetzen (A2), Kopf auf wenige Bits/Merkmale zwingen (A7, A11), Kopf durch Kodierung
  einsparen (A1). Zählt man diskrete Clifford-Wahlen als 2-Bit-Einträge, ist „100
  Parameter" für eine kopffreie Clifford-Zelle realistisch definierbar (Definitionsfrage an Malte).
- **Skalierung („unendlich skalierbar?")**: Free-Fermion/Matchgate-Zellen (A15) und
  Produktzustands-Kodierung (A4) sind die einzigen Ideen, deren Simulationskosten
  polynomiell bleiben, ohne dass die Zelle trivial wird.
- Kleine Vortests wurden gerechnet (Abschnitt 4a, sklearn-Digits 8x8 als **Ersatz** für MNIST,
  keine MNIST-Daten im Container). Sie sind Machbarkeitsandeutungen, keine Befunde.
- Mehrere Wunschthemen sind **ehrlich Q1 oder bekannt** (A6, A12, A14, A16); sie stehen
  unten in der Rangliste, damit nichts doppelt vorgeschlagen wird.

## 2. Stand der Forschung (nur Gedächtnis, ungeprüft)

- ECOC: Dietterich & Bakiri 1995. Tensornetz-Klassifikatoren auf MNIST: Stoudenmire &
  Schwab 2016 (MPS), danach viele MPS/TTN/MERA-Arbeiten. QNN-Klassifikation mit
  Auslese weniger Qubits: Farhi & Neven 2018. Expressibilität/Frame-Potential: Sim u. a. 2019.
  Gruppeninvariantes/geometrisches QML: Larocca u. a. 2022, Meyer u. a. 2023 (u. a.).
  Quantum-Reservoir-Computing: Fujii & Nakajima 2017. Holographische Codes: Pastawski,
  Yoshida, Harlow, Preskill 2015 (HaPPY). Boson Sampling: Aaronson & Arkhipov 2011.
  Matchgate-Klassizität: Valiant 2002, Jozsa & Miyake 2008. Landauer 1961, Bennett 1973.
  Mermin–Peres-Quadrat; Kontextualität im Lernen (Gili u. a. 2023, [?] genaue Angabe).
- Alles Weitere in IDEEN.md/ideen.json ist intern und nicht zu wiederholen.

## 3. Was das für QIRE bedeutet

1. Der Engpass ist der Kopf, nicht die Zelle. Jede Idee unten wird danach bewertet, ob sie
   Kopfparameter spart (a) und ob sie bei 8 → 14+ Qubits trägt (b).
2. Vortest (Digits, 6 Qubit): voller Born-Kopf 0,941 (650 Param); Kopf nur auf 10 Merkmalen
   (110 Param) 0,817; Binärcode-Klassenregister mit logistischen Bit-Köpfen 0,789. Die Lücke
   zwischen 110 Parametern und „voll" ist bei ungeschickter Auswahl ~12 pp. Das ist die Größe,
   die eine kluge Idee schließen muss.
3. Die Clifford-Suche trainiert nur den Vorder-Teil; ein kopffreier Entwurf müsste die
   Klassenentscheidung **in die Suche** verlegen (Zielfunktion = Treffer des Klassenregisters).
   Das ist der natürliche nächste Schritt, aber ein Eingriff in die vorhandene Suche.

## 4a. Kleine Vortests (Scratchpad, je < 1 min, sklearn-Digits, 70/30-Split, 1 Seed)

| Test | Ergebnis | Lesart |
|---|---|---|
| Bild auf Schmidt-Rang r gekappt (8x8-Matrix, logist. Regr. auf allen Pixeln) | r=1: 0,919; r=2: 0,956; r=3: 0,961; r=8: 0,967 | Bildzustand ist fast ein Produktzustand (A4) |
| Schmidt-Entropie je Klasse | Klasse 1: 0,56 bit; sonst 0,9–1,1 bit | Entropie klassenabhängig, aber klein |
| Born-Merkmale (Zufallsmaske + WHT, quadriert), voller Kopf | 0,941 | Referenz |
| Kopf auf Top-k Merkmale nach Gewicht, k=10/20/40 | 0,817 / 0,876 / 0,924 | Auswahl nach trainiertem Kopf |
| Top-k nach Fisher-Kriterium (ohne Kopf), k=10/20 | 0,787 / 0,876 | gleich gut wie Gewichtsauswahl bei k=20 (A7) |
| 20 zufällige Merkmale | 0,798 | Kontrolle: Fisher-Auswahl +7,8 pp |
| 4-Bit-Binärcode-Klassenregister (4 logist. Köpfe) | 0,789 | schlechter als Zufalls-ECOC |
| 15-Bit-Zufalls-ECOC | 0,880 | Redundanz hilft, kostet aber Parameter |

Einschränkung: ein Seed, Digits statt MNIST, logistische Regression statt Clifford-Zelle.

## 4. Ideen / Forschungsfragen

Konvention: Q = Quantenstufe; Suche = vorbereitete, **nicht ausgeführte** WebSearch-Anfrage;
Potenzial (a) 100 Param. MNIST / (b) Skalierung / (c) Jufo-Wirkung, je niedrig/mittel/hoch.

### A1. Syndrom-Merkmale: Stabilizer-/Toric-Code als parameterfreie Merkmalsabbildung
- **Frage:** Binarisierte Pixel (oder Zellen-Ausgabe) als „Fehlermuster"; Syndrome
  s = H·e über GF(2) eines festen Codes (Hamming, Reed–Muller, Toric-Code auf dem 16x16-Torus)
  als Merkmale. Tragen Syndrome von Plaketten-/Sternoperatoren mehr Klasseninformation je Bit
  als gleich viele Zufallsparitäten?
- **Stufe:** Q2 (Stabilizer-Formalismus exakt; Syndrom = Messung der Stabilizer-Generatoren).
- **Suche:** „stabilizer code syndrome features machine learning image classification"
  / „toric code syndrome neural network features" — **[nicht ausgeführt]**; Gedächtnis: Syndrom-
  Dekodierung mit NN ist Standard (umgekehrte Richtung), Syndrome als Lernmerkmale nicht bekannt, [?].
  Angrenzend intern: B-paritaet (IDEEN §5), Paritäten bis Ordnung 2 — A1 ist die
  Code-strukturierte Variante davon.
- **Billigster Test:** Binarisieren (Schwelle), Syndrom für Toric-Code auf 16x16 (512 Bit...
  bei 256 Pixeln: 256 Plaketten-Paritäten), logistischer Kopf. **~5 min.**
- **Kontrolle:** gleich viele zufällige Paritäten niedriger Ordnung; Lokalitätsbruch durch
  Pixel-Permutation. H0: Syndrom = Zufallsparitäten.
- **Potenzial:** (a) mittel-niedrig (spart Zelle, nicht Kopf), (b) mittel (lokal, linear in n),
  (c) mittel (anschauliche Brücke QEC–ML). Ehrlich: Syndrome sind lokale Paritäten = fester
  Gitter-Filter über GF(2); Gefahr „nur Kantendetektor".

### A2. Klassenregister: Messung ist der Kopf (kopffreie Zelle, Quanten-ECOC)
- **Frage:** Misst man nur 4 der 8 Qubits (16 Ausgänge ≥ 10 Klassen) und nimmt das Ergebnis
  direkt als Klasse (bzw. argmax über 10 gruppierte Born-Bins, Gewichte = feste 0/1-Gruppen),
  kann die Clifford-Suche dann die Zelle so bauen, dass **kein trainierter Kopf** nötig ist?
  Holevo-Seite: 4 Qubit liefern ≤ 4 bit ≥ log2(10) = 3,32 bit, es reicht also nur knapp.
- **Stufe:** Q2/Q3 (Teilmessung; auf Hardware direkt umsetzbar).
- **Suche:** „quantum neural network classification measure subset of qubits as class label
  MNIST" — **[nicht ausgeführt]**; Gedächtnis: Farhi–Neven 2018 messen ein Qubit für 2 Klassen;
  Mehrklassen via Qubit-Gruppen bekannt (Quanten-ECOC wurde m. W. vereinzelt vorgeschlagen, [?]).
  **Neu wäre nur:** kopffreie *Clifford-Suche* + Magie-Budget. Ehrlich: Kernidee bekannt.
- **Billigster Test:** In der bestehenden Suche die Zielfunktion von „Kopf(256→10) trainiert"
  auf „Treffer der festen 16→10-Zuordnung" umstellen; Digits-Vortest auf 6 Qubit (3 Klassen-Qubits ≥ 8 Klassen
  nicht ausreichend — für Digits 4 von 6 Qubits). **~10–30 min** je Suchlauf.
- **Kontrolle:** (i) zufällige Clifford-Zelle + feste Zuordnung (Untergrenze), (ii) volle Suche mit
  trainiertem Kopf (Obergrenze), (iii) Zufalls-ECOC aus Vortest 0,88 als Referenz für „Redundanz".
- **Potenzial:** (a) **hoch** — einzige Idee, die die 2.570 Kopfparameter auf 0 setzt; Rest wären
  ~2 Bit x Gatterzahl (≈ 100 diskrete Einträge, Definitionsfrage). (b) mittel (Klassenzahl wächst
  mit log, Qubits des Registers ebenso). (c) hoch, falls Treffer > 0,8 bei 0 Kopf; Risiko: Suche im
  Clifford-Raum hat nur 2 Bit je Gatter, 4-Bit-Entscheidung aus Born-Verteilung ist hart.

### A3. Holographischer Code als Bildmodell (Bulk–Rand-Isometrie)
- **Frage:** Wenn das 16x16-Bild der „Rand" ist und ein kleiner „Bulk" (z. B. 4–6 Qubit) die
  Klasse trägt: Lässt sich die Zelle als **Isometriebaum** (Perfect-Tensor-Kacheln, HaPPY) mit fest
  gewählter Hyperbolik und nur wenigen Winkeln parametrieren — und trägt die Fehlerkorrektur-
  Eigenschaft („Löschung von Randteilen verändert Bulk nicht") als **Robustheit gegen verdeckte Pixel**?
- **Stufe:** Q2 (exakte Tensornetz-Simulation; Perfect Tensors sind Stabilizer-Zustände → Clifford!).
- **Suche:** „holographic code tensor network image classification" — **[nicht ausgeführt]**;
  Gedächtnis: MERA-/TTN-Klassifikatoren bekannt (Stoudenmire; Reyes & Stoudenmire), hyperbolische
  Tensornetze für Daten selten, [?]; I-016/I-021 (MERA-Grobkörnung) intern verworfen — A3 unterscheidet
  sich durch **Fehlerkorrektur/Robustheit** statt Grobkörnung.
- **Billigster Test:** Perfect-Tensor-Baum (6-Leg-Stabilizer-Tensor) über 16 Randqubits, Klassen aus Bulk-
  Bits; Test = Genauigkeit mit 25 % maskierten Pixeln vs. Standardzelle. **~1 h Bauzeit, Lauf Minuten.**
- **Kontrolle:** gleicher Baum mit zufälligem (nicht perfektem) Tensor; Standard-Clifford-Zelle.
- **Potenzial:** (a) niedrig-mittel, (b) mittel (log-Tiefe), (c) hoch (Bildwirkung „Holographie"),
  aber Bauaufwand und reale Rolle der Perfektheit unklar. Fernziel.

### A4. Produktzustands-Kodierung und Schmidt-Rang als Aufgaben-Schwierigkeit
- **Frage:** Ein 16x16-Bild als 4+4-Qubit-Zustand (Zeile ⊗ Spalte) hat Schmidt-Rang ≤ 16. Reicht Rang 1–3
  (Produktzustand, Aufbau mit 2 x 4 Qubit statt 8 Qubit-Amplitudenkodierung)? Und sagt die
  **Schmidt-Entropie der Klasse** voraus, welche Klassen Magie/Tiefe brauchen?
- **Stufe:** Q2.
- **Suche:** „entanglement entropy MNIST images Schmidt rank low-rank image matrix product state"
  — **[nicht ausgeführt]**; Gedächtnis: Low-rank-Bildapproximation (SVD) ist uralt; Schmidt-Rang von Bildern als Zustand
  in Quantenbildverarbeitung diskutiert, [?]. **Neu wäre die Kopplung an das Magie-/Tiefenbudget** der Zelle (I-035, I-020 intern).
- **Billigster Test:** (läuft schon) Digits: Rang 1 → 0,919, Rang 2 → 0,956, Rang 8 → 0,967. Auf MNIST 16x16:
  Rang 1–4 abschneiden, dann bestehende Zelle. **~5 min.**
- **Kontrolle:** gleiche Parameteranzahl Hauptkomponenten-Reduktion (PCA auf 32 Zahlen); Pixelpermutation
  (zerstört die Rangstruktur).
- **Potenzial:** (a) mittel (Kodierung billig, Kopf bleibt), (b) **hoch** (Produktzustands-Aufbau skaliert
  linear; passt zu I-020), (c) mittel. Ehrlich: Rang-1-Näherung ≈ „Zeilen-Spalten-Faktorisierung",
  klassisch trivial; neu ist nur der Quantenaufwand-Vergleich.

### A5. Quantenlauf mit einem Parameter (CTQW-Zeit t als einziger Winkel)
- **Frage:** Kontinuierlicher Quantenlauf exp(−iAt) auf dem Pixelgitter-Graphen, Born-Auslese, 1 Parameter
  t (+ linearer Kopf). Wie viel Genauigkeit bringt **ein** Parameter?
- **Stufe:** Q2. (Angrenzend: I-011 verworfen — dort Engpass-Argument, nicht Merkmalsextraktor.)
- **Suche:** „continuous-time quantum walk feature extraction image classification" —
  **[nicht ausgeführt]**; Gedächtnis: Quantum-Walk-Kernel auf Graphen (Bai, Hancock u. a.), QW-Neural-Netze existieren → wahrscheinlich bekannt.
- **Billigster Test:** Digits-Gitter-Laplace, t ∈ {0,5; 1; 2; 4}, Born-Merkmale, logistischer Kopf. **~3 min.**
- **Kontrolle:** klassischer Diffusionskern exp(−Lt) (reelle Wärmeleitung), Zufallsmaske.
- **Potenzial:** (a) niedrig (Kopf bleibt), (b) mittel, (c) niedrig-mittel. Niedrig priorisieren, vermutlich bekannt.

### A6. Annealing/Adiabatik: Magie „einschalten" statt Clifford-Raum fest
- **Frage:** Homotopietraining: Winkel θ = (π/2)·k + s·δ; s wird adiabatisch von 0 (reines Clifford) auf
  kleine Werte hochgefahren, der Gradient folgt dem Clifford-Becken. Führt das zu den „4 Magiewinkeln"
  (I-005-v3) besser als Suche → Gradient (Becken-Problem, „Suche sitzt in anderem Becken als Gradient")?
- **Stufe:** Q1 (Analogie zu Adiabatik) / Q2 (reale Simulation).
- **Suche:** „annealed discretization continuation training quantized network straight-through" —
  **[nicht ausgeführt]**; Gedächtnis: Quantisierungs-Annealing/Homotopie ist Standard; intern I-025/I-030 verworfen.
- **Billigster Test:** Zelle mit Winkelrundungs-Strafterm λ(t)·dist(θ, π/2·ℤ), λ wächst. **~10 min.**
- **Kontrolle:** festes λ, sofortige Rundung.
- **Potenzial:** (a) niedrig, (b) niedrig, (c) niedrig. Bekanntes Genre → nur Werkzeug, keine Neuheit.

### A7. Fisher-Information als Merkmalsqualität (Bit-/Bin-Auswahl für einen kleinen Kopf)
- **Frage:** Die klassische Fisher-Information bzw. Quanten-Fisher-Information der Born-Verteilung bezüglich
  einer Klassen-Verschiebung bewertet jeden Bin/jede Parität. Wählt man **die 10–20 Bins mit höchster
  Fisher-Zahl ohne Kopftraining**, erreicht ein Kopf mit ~100 Parametern das Niveau des vollen?
- **Stufe:** Q2 (QFI aus Zuständen exakt berechenbar).
- **Suche:** „quantum Fisher information feature selection quantum machine learning encoding" —
  **[nicht ausgeführt]**; Gedächtnis: QFI für Expressibilität/Trainierbarkeit (Haug, Bharti, Kim) bekannt;
  als Merkmalsauswahl für klassische Daten [?]. Einfache Fisher-Score-Auswahl ist klassisch Standard.
- **Billigster Test:** (Vortest läuft) Digits: Top-10/20 nach Fisher 0,787/0,876 vs. zufällig-20 0,798 →
  nur ab k=20 ein klarer Gewinn, bei k=10 kaum. Auf MNIST-Zelle wiederholen. **~5 min.**
- **Kontrolle:** zufällige Bins; Auswahl nach trainiertem Kopf (Obergrenze 0,817/0,876).
- **Potenzial:** (a) mittel (110–210 Param., aber Lücke 6–12 pp), (b) mittel, (c) niedrig-mittel;
  Quantum-Fisher als Brücke ist hübsch, Wirkung begrenzt.

### A8. t-Design-Defizit der gelernten Zelle
- **Frage:** Clifford-Gruppe ist ein exaktes 3-Design. Ist die **Entfernung der gelernten Zelle von einem
  2-/3-Design** (Frame-Potential der Zellenfamilie, nicht der Einzelschaltung) ein Prädiktor für
  Trainierbarkeit/Genauigkeit — und gilt „strukturiert = weit vom Design"?
- **Stufe:** Q2.
- **Suche:** „frame potential trained quantum neural network distance to unitary design accuracy" —
  **[nicht ausgeführt]**; Gedächtnis: Expressibilität (Sim 2019) vs. Barren Plateaus bekannt; **trainierte** Familie vs. Genauigkeit [?].
- **Billigster Test:** Zufalls-Cliffords vs. gesuchte Cliffords: Spur-Distanz der Zweitmomentoperatoren, 8 Qubit
  exakt in 20–60 s je Stichprobenreihe (nur 3-Design-Eigenschaft ausnutzen, siehe Kontrolle). **~15 min.**
- **Kontrolle:** Haar-zufällige 8-Qubit-Zustände.
- **Potenzial:** (a) niedrig, (b) mittel (Barren-Plateau-Argument für Skalierung), (c) mittel (saubere
  Theorie-Aussage). Ehrlich: Für Cliffords gilt es per Definition (3-Design) → Aussagekraft nur für die
  **eingeschränkte** gelernte Familie; Aufwand höher als Nutzen.

### A9. Gruppen-Clifford-Äquivarianz: D4 und XOR-Translation als Parameter-Sharing
- **Frage:** Die Bildsymmetrien Spiegelung/Drehung (D4) wirken auf die Pixelindex-Bits als Qubit-
  Permutation + X-Gatter (Clifford!). Eine Zelle, die mit D4 kommutiert (oder deren Ausgabe invariant ist), spart
  Parameter und Suchraum. Dazu: Der Betrag |WHT| ist invariant unter **bitweiser XOR-Translation** des Index —
  Pixel-Blocktausch. Lohnt exakte Invarianz auf MNIST (wo Ziffern 6/9 nicht drehinvariant sind)?
- **Stufe:** Q2.
- **Suche:** „equivariant quantum neural network image classification Clifford group Z2 symmetry" —
  **[nicht ausgeführt]**; Gedächtnis: Larocca u. a. 2022 (Gruppeninvarianz, S_n), Meyer u. a. (EQNN), Nguyen (Equivariant QNN auf SU(2)) → Prinzip
  bekannt; **Bild-D4 als Clifford auf Indexbits mit diskreter Clifford-Suche** [?], vermutlich nicht gemacht.
- **Billigster Test:** Suche eingeschränkt auf D4-invariante Masken (Spiegelungs-Orbits) – Spiegelungsinvarianz
  nur horizontal (Ziffern!). Parameter/Suchraum halbiert. **~20 min** pro Suchlauf.
- **Kontrolle:** gleiche Parameterzahl in zufälligem Sharing; Standard-Suche.
- **Potenzial:** (a) mittel (halber Suchraum, kein Kopf-Effekt), (b) mittel-hoch (Sharing wächst mit Gruppe),
  (c) mittel-hoch (geometrisches QML = aktuelles Thema). MNIST ist ein schlechter Testfall für volle D4; Fashion-MNIST
  (Spiegel) besser.

### A10. Kontextualität: Mermin–Peres-Auslese
- **Frage:** Mehr Information je Kopie durch Messung **nicht kommutierender** Pauli-Sätze (Mermin-Peres-Quadrat
  auf 2 Qubit, Kontextualitätsmaß) statt nur Z-Basis? Erhöht höhere Kontextualität die Linear-Separierbarkeit?
- **Stufe:** Q2 (Simulation), Q3 auf IBM.
- **Suche:** „contextuality resource quantum machine learning classification advantage" — **[nicht ausgeführt]**;
  Gedächtnis: Kontextualität und Lernvorteil (Gili u. a. 2023, [?]) und Kernel-Papiere existieren.
- **Billigster Test:** Auslese in Z-, X-, Y-Basis (3 Läufe, mehr Merkmale) vs. Z allein; Kontextualitätswert der
  Auslesesätze gegen Genauigkeit. **~10 min.** Überschneidung mit Paritäts-/B-paritaet-Linie (Basiswechsel = WHT).
- **Kontrolle:** gleiche Merkmalszahl in kommutierenden Basen (nur Z-Parität höherer Ordnung).
- **Potenzial:** (a) niedrig, (b) niedrig, (c) mittel. Wahrscheinlich nur ein Basiswechsel unter anderem Namen.

### A11. Clifford-Reservoir mit minimalem Kopf
- **Frage:** Zufällige feste Clifford-Zelle (0 trainierte Parameter) als Quanten-Reservoir, Kopf per
  L1/Hard-Threshold auf ≤ 100 aktive Gewichte (je Klasse ≤ 10). Wo liegt die **kleinste Kopfgröße**,
  die die Genauigkeit der Clifford-Suche (~0,94) erreicht — und ist Suche überhaupt besser als Reservoir + sparsamer Kopf?
- **Stufe:** Q2.
- **Suche:** „quantum reservoir computing MNIST random Clifford sparse readout number of parameters" —
  **[nicht ausgeführt]**; Gedächtnis: QRC auf MNIST gezeigt (z. B. Ghosh, Fujii…), Reservoir-Ausleseparameter werden selten
  minimiert [?]. Intern verwandt: I-004 (zufälliges Clifford-Gerüst).
- **Billigster Test:** (Vortest) Top-k-Kopf 0,817/0,876/0,924 für k=10/20/40 auf Digits (110/210/410 Param.).
  MNIST-Zelle: fester Zufalls-Clifford + L1-Pfad. **~10 min.**
- **Kontrolle:** Pixel-Roh-Merkmale mit gleicher Kopfgröße (Zelle muss etwas beitragen); I-004-Gerüst.
- **Potenzial:** (a) mittel-hoch (der direkteste Weg zur 100er-Marke und gleichzeitig Ehrlichkeitstest für die Suche),
  (b) mittel, (c) mittel. Verbindet sich mit A2/A7.

### A12. Boson-Sampling-Merkmal
- **Frage:** Pixelvektor als Photonen-Eingabemuster/Interferometer-Winkel; Zählstatistik (Permanenten) als Merkmal.
  Schwer klassisch simulierbar → ist das ein **Merkmal mit nachweislich quantischem Vorteil** (oder nur zufälliger Hash)?
- **Stufe:** Q3-Emulation (klein, klassisch berechenbar: Permanenten bis 6 Photonen) — echter Vorteil erst bei
  Skala, die der Laptop nicht zeigt.
- **Suche:** „boson sampling feature map machine learning image classification" — **[nicht ausgeführt]**;
  Gedächtnis: Photonische Quantenkerne und Boson-Sampling-Kernel (Schuld u. a.; Sakurai u. a. 2023 [?]) existieren.
- **Billigster Test:** 4 Moden, 2 Photonen, Permanenten-Verteilung der Pixel-Blocks; Kopf. **~30 min** Bau.
- **Kontrolle:** klassischer Zufallsmerkmals-Hash gleicher Dimension.
- **Potenzial:** (a) niedrig, (b) **fraglich** (Vorteilsargument gilt nur skalenweise), (c) mittel (Optik-Wirkung).
  Vermutlich bekannt, keine Priorität.

### A13. Postselektion als parameterlose Nichtlinearität (mit Kostenbuchhaltung)
- **Frage:** Die Zelle ist bis auf Born linear; Postselektion auf ein Ancilla-Ergebnis erzeugt die rationale
  Nichtlinearität ψ ↦ Pψ/‖Pψ‖. Wie viel Genauigkeit gewinnt eine nachselektierte Zelle gegenüber der Born-Zelle, und wie
  hoch ist der **Erfolgswahrscheinlichkeits-Preis** (Schüsse je Bild)? Abgrenzung zu I-032 (Born als Nichtlinearität, intern verworfen).
- **Stufe:** Q2 (Simulation exakt), Q3 auf Hardware (Verwerfen von Schüssen).
- **Suche:** „postselection nonlinear activation quantum neural network classification cost success probability" —
  **[nicht ausgeführt]**; Gedächtnis: Postselektions-Nichtlinearität (Rebentrost; Cao u. a.) und dissipative QNN bekannt → nur die
  Kostenbilanz zu Genauigkeit [?].
- **Billigster Test:** Projektor auf Qubit-0 = |0>, Renormierung, Born-Bins; Genauigkeit vs. mittlere Erfolgsrate. **~10 min.**
- **Kontrolle:** gleiche Zelle ohne Postselektion; Zelle + klassische Quadrat-Aktivierung.
- **Potenzial:** (a) mittel (Nichtlinearität könnte Kopf kleiner machen), (b) niedrig (Schusskosten wachsen),
  (c) mittel. Kosten-Genauigkeits-Kurve ist die ehrliche Aussage.

### A14. Landauer-Bilanz einer Klassifikation (gelöschte Bits)
- **Frage:** Die unitäre Zelle ist reversibel (0 Landauer-Kosten); die Messung/der Kopf löscht Information. Wieviel
  **bit pro Bild** werden gelöscht, und ist die Untergrenze H(Eingabe | Klasse)-artig? Vergleich: klassisches MLP
  vs. QIRE-Zelle in „Bit gelöscht pro korrekter Klassifikation".
- **Stufe:** Q1 (Thermodynamische Buchführung, keine Quantenressource im Test) → Prio 3.
- **Suche:** „Landauer limit neural network inference reversible computing classification bits erased" —
  **[nicht ausgeführt]**; Gedächtnis: reversibles Rechnen für NN existiert (Frank u. a.), reale Chips liegen 10^6–10^8 x über Landauer — Quelle [?].
- **Billigster Test:** Entropie der Zwischenwerte (diskretisiert) vor/nach jeder Schicht zählen. **~10 min.**
- **Kontrolle:** MLP gleicher Größe.
- **Potenzial:** (a) niedrig, (b) niedrig, (c) mittel-hoch als Erzählung („Rechnen ohne Löschen"), aber
  physikalisch irreführend, solange Chips weit über Landauer arbeiten — ehrlich so benennen.

### A15. Free-Fermion/Matchgate-Zelle: polynomiell skalierende Quantenzelle
- **Frage:** Eine Zelle aus nächste-Nachbar-Matchgates (Givens-Rotationen, orthogonale 2n×2n-Matrix) ist klassisch
  in poly(n) simulierbar (Valiant/Jozsa–Miyake). Wie weit trägt sie als Bildmodell, und skaliert sie bei n = 16, 32, 64 Qubit bei
  Parameterzahl O(n·L) ohne den Zustandsvektor 2ⁿ?
- **Stufe:** Q2 (exakt, nicht-trivial: fermionische Gaußzustände).
- **Suche:** „matchgate circuit classification image free fermion quantum neural network" —
  **[nicht ausgeführt]**; Gedächtnis: Matchgate-QNNs/Fermionic-QNN (z. B. Monbroussou u. a. 2023, Barren-Plateau-Freiheit) bekannt;
  **als Skalierungstest für Bilder** [?].
- **Billigster Test:** Gaußzustand-Kovarianzmatrix 2n×2n, Auslese: Dichte-Dichte-Korrelatoren n_i n_j (n² Merkmale
  statt 2ⁿ), Training per Gradient auf Givens-Winkeln. **~1 h Bauzeit, Läufe Minuten.**
- **Kontrolle:** reine Linearität der Eingabe (die Zelle ist linear in der Einbettung?) — **Warnung:** Gaußzelle ist
  evtl. nur eine orthogonale Matrix + quadratische Merkmale; gegen „orthogonale Matrix + Quadrat-Aktivierung" prüfen, sonst
  kein Quantenvorteil und kein Quantengewinn.
- **Potenzial:** (a) mittel, (b) **hoch** (Antwort auf „unendlich skalierbar?": ja, genau weil klassisch simulierbar —
  ehrlicher Kernsatz), (c) hoch bei sauberer Kontrolle.

### A16. Stabilizer-Prototypen: Klasse = Stabilizer-Zustand, Entscheidung durch Fidelity
- **Frage:** Jede Klasse c bekommt einen Stabilizer-Zustand |S_c> (2 Bit je Generator-Eintrag); Klassifikation = höchste Fidelity |<S_c|ψ(x)>|². Stabilizer-
  Fidelities sind effizient berechenbar. Wie gut ist ein „Stabilizer-Nearest-Centroid"?
- **Stufe:** Q2.
- **Suche:** „stabilizer state prototype nearest centroid quantum classifier fidelity" — **[nicht ausgeführt]**; Gedächtnis:
  Quantum-Nearest-Centroid (Lloyd, Mohseni, Rebentrost 2013), Fidelity-Klassifikatoren bekannt; Stabilizer-Prototypen [?].
- **Billigster Test:** 10 Zufalls-Stabilizerzustände, Auswahl per Greedy auf Trainingsfidelität; 8 Qubit mit Stabilizer-Tableau. **~30 min.**
- **Kontrolle:** zufällige Prototypen; Nearest-Centroid mit realen Klassenmittelwerten.
- **Potenzial:** (a) mittel (kopffrei, ähnlich A2; je Klasse n² Bit = ~640 Bit gesamt), (b) mittel, (c) mittel.
  Fidelity zu einem einzigen Stabilizer-Zustand ist ein hartes Kriterium; erwartbar schwach.

### A17. Messung vs. Information: Holevo-Budget als Vorhersage der Mindest-Qubitzahl
- **Frage:** Holevo-Schranke ⇒ n Qubits liefern ≤ n bit; für 10 Klassen reichen 3,32 bit. Messbare Vorhersage:
  Die Genauigkeit einer Zelle mit **k ausgelesenen Qubits** bricht unter k ≈ 4 ein (A2-Nebenprodukt). Wird die empirische
  Informationsmenge I(Klasse; Messergebnis) nahe 3,3 bit erreicht, dann ist k die „Holevo-Kante".
- **Stufe:** Q2. Berührt Maltes Ziel 3 (Holevo umgehen) — das ist **nicht umgehbar**, nur auf Auslesebreite prüfbar.
- **Suche:** „Holevo bound classification accuracy readout qubits mutual information MNIST" — **[nicht ausgeführt]**;
  Gedächtnis: Informationsgrenzen für QML bekannt, Messung für MNIST-Zellen [?].
- **Billigster Test:** Zelle mit k = 1…8 ausgelesenen Qubits, empirische MI aus Konfusionsmatrix. **~10 min**
  (Teil von A2).
- **Kontrolle:** klassische Zelle mit k-Bit-Flaschenhals.
- **Potenzial:** (a) niedrig-mittel (Kopfgröße-Argument), (b) niedrig, (c) mittel (saubere Zahl). Idee ist ein Anhang zu A2.

## 5. Rangliste Top 5 (Potenzial, Machbarkeit, Neuheit vorläufig!)

Alle Neuheitsurteile **[?] — Websuche fehlt**. Rangfolge nach (a), dann (b), dann (c).

1. **A2 Klassenregister / kopffreie Zelle (+A17 Holevo-Kante).** Einzige Idee, die die 99 % Kopfparameter auf 0
   setzt. Billig (Suche umstellen), ehrlich testbar. Risiko: Kernidee bekannt, Neuheit nur in Clifford-Suche + Magiebudget.
2. **A11 Clifford-Reservoir + minimaler Kopf (mit A7 Fisher-Auswahl).** Direkter Weg zur 100er-Marke und zugleich
   die ehrlichste Kontrolle gegen die Suche: schlägt Suche das Reservoir bei gleicher Kopfgröße?
3. **A15 Matchgate-Zelle.** Beste Antwort auf „unendlich skalierbar?": poly(n), n=64 ohne 2ⁿ. Muss gegen
   „orthogonale Matrix + Quadrat" verteidigt werden.
4. **A4 Produktzustands-Kodierung / Schmidt-Rang.** Spart Kodierungskosten, skaliert linear, schon ein erster
   Zahlenhinweis (Rang 1: 0,919 auf Digits). Neuheit nur in Kopplung an Magiebudget.
5. **A9 Gruppen-Clifford-Äquivarianz (D4/XOR).** Parameter-Sharing im Clifford-Suchraum; geometrisches QML ist
   aktuell und für Jury gut erzählbar; Test auf Fashion sinnvoller als MNIST.

Danach: A13 (Postselektion mit Kostenbilanz), A1 (Syndrome), A3 (Holographie, Fernziel), A16, A8, A10, A12, A5, A14, A6.

## 6. Offene Fragen für Etappe 2

- **Websuchen nachholen** (Budget anheben): je Idee die oben vorbereitete Anfrage; zuerst A2, A11, A15, A9.
- **Definitionsfrage Malte:** Zählt „100 Parameter" auch diskrete Clifford-Wahlen (2 Bit je Gatter)? Ohne
  Antwort ist A2 unklar bewertbar.
- Tragen die Digits-Vortests auf echtes MNIST 16x16 (Container hat keine MNIST-Daten, Download nötig)?
- Reale Kosten der Suche bei kopffreier Zielfunktion (diskrete Zielfunktion „Treffer" ist stückweise konstant → Suche braucht
  weiche Ersatzzielfunktion?).
- Seeds: alle Vortests 1 Seed → vor jeder Aussage ≥ 10 Seeds wie im Institut üblich.

## 7. Quellenliste

Keine geprüften URLs (Websuche-Budget erschöpft). Alle Literaturangaben in Abschnitt 2 und in den
Ideenkarten sind **Gedächtnis, ungeprüft**. Interne Quellen: `INSTITUT/recherche/2026-10-05_breit/00_AUFTRAG.md`,
`INSTITUT/UEBERGABE.md` §0, `INSTITUT/IDEEN.md`, `INSTITUT/ideen.json`. Vortest-Skript: Scratchpad
`/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/e.py` (sklearn-Digits).
