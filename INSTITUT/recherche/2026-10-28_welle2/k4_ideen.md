# k4 — Zufallsimpuls: 6 Paper aus fremden Gebieten, übertragen auf QIRE (Welle 2, 28.10.2026)

Nur gedacht und gelesen, nichts gerechnet. Quellen stammen aus **Suchauszügen (Abstracts), kein Volltext gelesen**; Jahreszahlen stehen dabei, die Suche fand nicht in jedem Gebiet ein Paper von 2026 (ehrlich vermerkt). Wirkung überall klassisch (I-038), „quanten-inspiriert“. Nicht wiederholt: alles aus SYNTHESE.md und I-047/48/49/45-v5. Ideen 4 und 5 sind zu I-048 verwandt, aber andere Achse (Transformdomäne, Inferenzspeicher).

## Gezogene Paper
| Gebiet | Paper (Auszug gelesen) | Idee Nr. |
|---|---|---|
| Neurobiologie | Chavlis & Poirazi 2024, „Dendrites endow ANNs with accurate, robust and parameter-efficient learning“ (arXiv 2404.03708); NeurIPS-2026-Eintrag „Dendritic Structure Enables Parameter-Efficient Neural Representations“ (DendriNet) | 1, 2, 3 |
| Signalverarbeitung | „Neural network compression via learnable wavelet transforms“ (arXiv 2004.09569, 2020; kein 2026-Paper gefunden) | 4, 6 |
| Optik/Holografie | UCLA, „Interleaved diffractive networks“ (arXiv 2603.07975, 03/2026); TRON (arXiv 2604.16228, 04/2026) | 5, 7 |
| Kryptografie/Hashing | HashFormers (EMNLP 2022), Reformer (2020); kein 2026-Paper gefunden | 8, 9 |
| Genetik/DNA | Preuss et al., Sci. Rep. 02/2026, kombinatorische DNA-Speicherung; HEDGES (PNAS 2020) | 10 |
| Ökologie/Schwarm | Krishnan & Mahadevan, „Stigmergic optimal transport“ (arXiv 2601.04111, 01/2026); Antwort-Schwellen-Modell (Bonabeau u. a. 1996, **aus dem Gedächtnis**, prüfen) | 11 |
| (Reversibilität, aus Optik-Idee der Zeitumkehr) | RevNet-Linie (Gomez 2017, aus dem Gedächtnis) | 12 |

---

### 1. Dendritische Born-Äste · Neurobiologie (Dendriten-Teilinheiten) · Interferenz: Eine Dendritenhülle summiert Synapsen kohärent und quadriert (Born) die Astsumme, bevor sie ins Soma geht.
- **Mechanismus:** FFN-Neuron = Soma über K Äste; jeder Ast sieht nur k feste Eingänge (k = 4–8, Gruppierung per Hadamard-Block), bildet erst die Summe und dann z². Nur die Astgewichte sind gelernt. Bei gleichen FLOPs/Bytes gewinnt es, wenn Grad-2-Merkmale pro Parameter mehr leisten als eine breite lineare Schicht (Chavlis & Poirazi: weniger Parameter bei gleicher Güte).
- **Langweiligste Erklärung:** Gruppiertes Conv/Block-Diagonale plus Quadrat; der Vorteil ist nur die Quadrat-Aktivierung (I-045-v2/v4 zeigten schon: Aktivierung, nicht Struktur).
- **Billigster Test:** Synthetisch k-Parität (k = 6, 8) und MNIST-MLP, gleiche Parameterzahl: dendritisch (Äste mit z²) gegen dicht + GELU gegen dicht + z². Kontrolle: Äste ohne Quadrat. 1 Thread, ≈ 5 min.
- **Vorab-Tipp:** P(grün) 35 % (gegen dicht+z² vermutlich gleichauf), Jury 3.

### 2. Born-Normierung statt LayerNorm + GELU · Neurobiologie (divisive Normierung, Sparse Coding) · Born: a_i = z_i² / (ε + Σ_j z_j²) ist genau die Born-Wahrscheinlichkeit der Auslese.
- **Mechanismus:** Aktivierung der FFN-Zwischenschicht wird die Born-Wahrscheinlichkeit über Gruppen von g Einheiten (mal fester Skala). Das ist divisive Normierung (kanonische kortikale Rechnung), erzeugt Sparsität/Winner-take-most und ersetzt Aktivierung und einen Teil der Normierung bei null Parametern und ~2 FLOPs/Einheit. Gewinn möglich über stabilere Skalen (höhere Lernrate, weniger Normschichten).
- **Langweiligste Erklärung:** Es ist eine Softmax-ähnliche Normierung; der Gewinn kommt aus der Normierung, nicht aus dem „Born“.
- **Billigster Test:** Byte-LM d = 128, 4 Lagen, 1500 Schritte (Tuning-Seeds), FFN-Aktivierung: Born-Gruppen g = 8/32 gegen GELU gegen z² gegen z²-ohne-Normierung (Kontrolle) bei gleichen FLOPs; Lernrate je Arm getunt. ≈ 8 min.
- **Vorab-Tipp:** P(grün) 30 %, Jury 4 (schöne Geschichte: Born = Gehirn-Normierung).

### 3. Fliegen-Hash-Kopf: Hadamard-Expansion + Top-k (Sparse Coding) · Neurobiologie (Riechsystem der Fliege, FlyHash, Dasgupta 2017, Gedächtnis; Sparse Coding) · Hadamard: überkomplette, feste Expansion; Messung wählt wenige Ausgänge.
- **Mechanismus:** Feste Expansion x → H·D·x auf Breite m = 8d (zwei bis vier Vorzeichen-Diagonalen, 0 Parameter), nur die k größten |z| bleiben, die gelernte Auslese hat nur k aktive Spalten. FLOPs der zweiten Schicht fallen von m auf k je Token; Bytes: nur k·Klassen gelernte Gewichte aktiv, Speicher aber m·Klassen, also gewinnt es bei **gleichen FLOPs** (breiterer Merkmalsraum bei gleichem Auslesen). Unterschied zu I-045: Gewinn liegt in der Sparsität, nicht in Fastfood-Struktur.
- **Langweiligste Erklärung:** Aktivierungs-Sparsität (Top-k/ReLU²), bekannt; ein Gewinn wäre ein gewöhnlicher Breiteneffekt.
- **Billigster Test:** MNIST/Fashion-Kopf auf festen CNN-Merkmalen (wie I-044): gleiche FLOPs für den Kopf: FlyHash (m = 8192, k = 64) gegen dicht klein (h = Auslese-FLOPs-gleich) gegen Top-k mit Zufalls-Spaltenwahl (Kontrolle). ≈ 6 min.
- **Vorab-Tipp:** P(grün) 40 %, Jury 4.

### 4. Haar/Walsh-Domänen-Beschneidung: Gewichte als wenige Transformkoeffizienten · Signalverarbeitung (Wavelets, Compressed Sensing; Quelle 2004.09569) · Hadamard/Haar: Gewichtsmatrix in der Walsh-Haar-Basis ist (näherungsweise) dünn.
- **Mechanismus:** Trainierte Matrix W → 2D-Haar- oder 2D-Walsh-Transform → nur die m größten Koeffizienten (+ Indizes) speichern; Bytes = m·(Wert + Index). Gewinn gegenüber Niedrigrang/Größenbeschneidung bei gleichen Bytes, falls die Matrizen räumlich glatt/multiskalig sind (Bild, Positionen) – bei Zufallsmatrizen nicht. FLOPs bleiben dicht (reine Bytes-Aussage).
- **Langweiligste Erklärung:** Trainierte Gewichte sind einfach glatt (MNIST-Eingangsschicht); dann ist jede Glättung gut, Haar nichts Besonderes.
- **Billigster Test:** Ohne Training neu: trainiertes MNIST-MLP 784-512-512-10; gleiche Bytes: Betrags-Top-m im Gewichtsraum gegen Haar-Top-m gegen Walsh-Top-m gegen SVD-Rang gegen Zufalls-Orthogonal-Top-m (Kontrolle). Genauigkeit vs. Bytes-Kurve. ≈ 4 min.
- **Vorab-Tipp:** P(grün) 45 % für die Eingangsschicht, bei Mittelschichten 15 %; Jury 3.

### 5. Hadamard-Skizze des KV-Caches (Johnson–Lindenstrauss, Compressed Sensing) · Signalverarbeitung/Optik (zufällige Messungen, „Messmatrix“) · Hadamard: eine Zufalls-Walsh-Projektion bewahrt Skalarprodukte q·k, wie ein Detektor mit wenigen Pixeln.
- **Mechanismus:** Speichere statt k_t ∈ R^d nur S·k_t mit S = Zeilenauswahl der Hadamard·D-Matrix (m < d Zeilen), und rechne Scores im Skizzenraum, (Sq)·(Sk) ≈ q·k. Cache-Bytes und Score-FLOPs sinken um m/d. Die Rotation verteilt Ausreißerkanäle, also bessere Näherung als das bloße Weglassen von Kanälen (gleiche Bytes). Verwandt mit QuaRot/SRHT (bekannt, wird zitiert); nicht in I-048 enthalten.
- **Langweiligste Erklärung:** Attention-Köpfe sind niedrigdimensional; jede Projektion (PCA) täte es ebenso.
- **Billigster Test:** Byte-LM kurz trainieren (Tuning-Seeds), dann nur zur Auswertung q·k durch Skizze ersetzen, m ∈ {d/2, d/4, d/8}: bpc gegen Bytes. Arme: Hadamard-Skizze, Zufallskanäle (Kontrolle gleiche Bytes), PCA-Skizze (starker Gegner). ≈ 7 min.
- **Vorab-Tipp:** P(grün gegen Zufallskanäle) 70 %, gegen PCA 20 %; Jury 3.

### 6. Haar-Pyramide für alte Token (Wavelet-Gedächtnis) · Signalverarbeitung (Mehrfachauflösung) · Unitarität: Haar-Schritte sind orthogonal; weniger Detail für Ältere kostet kontrolliert Energie.
- **Mechanismus:** KV-Cache nach Alter geteilt: die letzten 32 Token voll, ältere nur als Haar-Grobkoeffizienten (Paarmittel, 2×, 4×, 8×), die Detailkoeffizienten werden verworfen. Bytes der Cache-Länge N wachsen so ≈ log N bzw. N/8 statt N bei kleinem bpc-Verlust. Abgegrenzt von Hourglass (I-049): dort trainiert und FLOP-sparend, hier reine Inferenz-Speicher-Aussage.
- **Langweiligste Erklärung:** Nur lokale Nähe zählt in kleinen Byte-Modellen (alte Token kaum nützlich), jede Alterung wäre gleich gut.
- **Billigster Test:** Wie 5, Kontext 256; bei gleichen Bytes: Haar-Pyramide gegen zufälliges Fallenlassen alter Token gegen Gleitfenster (Kontrolle). ≈ 7 min.
- **Vorab-Tipp:** P(grün) 40 %, Jury 3.

### 7. Holografischer Speicher statt KV-Cache (HRR mit Einheitsphasen) · Optik/Holografie (Interferenz von Objekt- und Referenzstrahl; Quellen 2603.07975/2604.16228 als Anlass, HRR/Hrrformer aus dem Gedächtnis) · Interferenz: Schlüssel und Wert werden mit Einheitsphasen verknüpft und überlagert, das Auslesen interferiert den richtigen Eintrag konstruktiv, den Rest als Rauschen weg.
- **Mechanismus:** Fester Speicher M ∈ C^d (2d reelle Zahlen): M ← M + e^{iφ(k_t)} ⊙ v_t; Abruf: Re(e^{−iφ(q)} ⊙ M). Bytes konstant in der Länge, FLOPs O(d) je Token statt O(N d). Gewinnt bei gleichen Bytes gegen Linear-Attention-Matrizen (d×d) für kleine d: Kapazität ≈ d/ln(N) Einträge.
- **Langweiligste Erklärung:** Kapazität ist das Rauschbudget d, gleich dem bekannten Hopfield-/Fast-Weight-Ergebnis; kein QIRE-spezifischer Gewinn.
- **Billigster Test:** Ohne Training: synthetische Assoziativ-Abrufaufgabe, N ∈ {8…256} Paare, gleiche Bytes: HRR-Phasen (d = 128 komplex) gegen Linear-Attention-Matrix (d×d mit gleicher Bytezahl, d′ ≈ 90) gegen reell-HRR (Kontrolle ohne Phasen). Genauigkeit vs. N. ≈ 3 min.
- **Vorab-Tipp:** P(grün) 65 % (nur Kapazität), im trainierten LM 15 %; Jury 4 (Hologramm-Geschichte ist anschaulich).

### 8. Vorzeichen-Hash-Gewichte (Count-Sketch): Kollisionen löschen sich aus · Kryptografie/Hashing (HashedNets-Linie; HashFormers 2022, Reformer) · Interferenz: Zufällige Vorzeichen s(i,j) = ±1 machen Kollisionen im Mittel destruktiv.
- **Mechanismus:** Matrix W_ij = s(i,j)·θ[h(i,j)] aus Tabelle θ der Größe B ≪ n·m, h, s aus Seed-Hash (0 gespeichert). Bytes = B. FLOPs unverändert dicht; gegen Niedrigrang und kleines dichtes Modell bei gleichen Bytes. Unterschied zu Fastfood: keine Struktur, volle Dichte, Kollisionsrauschen statt Strukturgrenze.
- **Langweiligste Erklärung:** Es ist Zufallsprojektion eines Parametervektors, effektiv wie ein zufälliger Unterraum (intrinsic dimension, Li 2018); dichte kleinere Modelle sind ebenso gut.
- **Billigster Test:** MNIST-MLP 784-512-10; gleiche Bytes B: Vorzeichen-Hash gegen Hash ohne Vorzeichen (Kontrolle) gegen kleines dichtes gegen Niedrigrang. ≈ 5 min.
- **Vorab-Tipp:** P(grün gegen Niedrigrang) 25 %, Jury 2 (HashedNets ist alt, kaum neu).

### 9. Count-Sketch für Adam-Zustände (Trainings-Bytes) · Kryptografie/Hashing (Sketching) · Interferenz: Vorzeichen-Hash verteilt Kollisionen; die zweiten Momente sind positiv, brauchen Min-Sketch statt Signed.
- **Mechanismus:** Adam hält 2 Zustände je Parameter (Doppelte Bytes). Speichere v (2. Moment) als Count-Min-Sketch der Breite n/8 (Vorzeichen nur beim 1. Moment), Training mit denselben Bytes erlaubt ein 1,5–2× größeres Modell. Bekannt im Prinzip (Count-Sketch-Optimierer, Adafactor, 8-Bit-Adam); Neues ist nur der Quantenbezug, daher eher Randidee.
- **Langweiligste Erklärung:** Zweiter Moment ist ohnehin fast konstant je Zeile; Adafactor-Faktorisierung täte dasselbe.
- **Billigster Test:** Byte-LM klein: Adam voll gegen Sketch-v (n/8) gegen Adafactor-Faktorisierung (Kontrolle, gleiche Bytes) gegen SGD-Momentum (Bytes einfach); gleiche Schritte. ≈ 9 min.
- **Vorab-Tipp:** P(grün gegen Adafactor) 15 %, Jury 2.

### 10. Codon-Code für Niedrigbit-Gewichte: fehlerrobuste Zuordnung · Genetik/DNA (genetischer Code ist so angeordnet, dass Ein-Basen-Fehler meist ähnliche Aminosäuren ergeben; Preuss 2026, HEDGES) · Hadamard/Fehlerkorrektur: Hamming-Abstand der Bitfolge ↔ Abstand der Gewichtswerte, wie Gray-/Reed-Muller-Code.
- **Mechanismus:** 2/3-Bit-Gewichtsindizes werden so den Werten zugeordnet, dass ein umgekippter Bit nur zu einem Nachbarwert führt (Gray) oder die Zuordnung per Optimierung den erwarteten Fehler bei Bit-Flip minimiert (wie Freeland–Hurst für den genetischen Code). Bei gleichen Bytes robuster gegen Speicherfehler, was energiesparenden SRAM (niedrige Spannung) erlaubt. Reine Robustheits-/Energie-Aussage, nicht Genauigkeit.
- **Langweiligste Erklärung:** Gray-Code ist Lehrbuchwissen; der Gewinn ist trivial und hängt nur von der Fehlerrate ab.
- **Billigster Test:** Ohne Training: trainiertes MNIST-MLP, 3-Bit-Quantisierung, Bit-Flip-Rate p ∈ {1e-3, 1e-2, 5e-2}: natürlich-binär gegen Gray gegen optimierte Zuordnung gegen zufällige Zuordnung (Kontrolle). Genauigkeit vs. p. ≈ 3 min.
- **Vorab-Tipp:** P(grün, Gray besser als binär) 80 %, aber Jury 2 (Neuheit gering; als Anwendungsgeschichte „DNA-Code“ 3).

### 11. Antwortschwellen-Aktivierung (Hill, n = 2): beschränkt und 4-bit-tauglich · Ökologie/Schwarm (Arbeitsteilung bei Ameisen: Reiz s löst Aufgabe mit Wahrscheinlichkeit s²/(s²+θ²); Anlass Stigmergie-Paper 2601.04111, die Zuordnung ist **meine Übertragung**, nicht im Paper) · Born: s²/(s²+θ²) ist die Zwei-Ergebnis-Born-Wahrscheinlichkeit mit Schwelle θ.
- **Mechanismus:** FFN-Aktivierung a = z²/(z²+θ_i²), θ_i je Einheit gelernt oder geometrisch gespreizt (verschiedene Arbeitsteilungsschwellen). Ausgabe liegt in [0,1]: keine Ausreißer, also Aktivierungen in 4 Bit speicherbar (Trainings-Aktivierungsspeicher, KV/Zwischenwerte) mit kleinem Verlust. Gewinn bei **gleichen Bytes der Aktivierung**; FLOPs ≈ gleich.
- **Langweiligste Erklärung:** Beschränkte Aktivierungen sind einfach leichter zu quantisieren (wie ReLU6/Sigmoid); das Hill-Quadrat ist nicht nötig.
- **Billigster Test:** Byte-LM d = 128, 1500 Schritte; FFN-Zwischenschicht auf 4 und 3 Bit quantisiert (STE): Hill-n=2 gegen GELU gegen z² gegen ReLU6 (Kontrolle beschränkt); bpc je Bytes der Aktivierung. ≈ 9 min.
- **Vorab-Tipp:** P(grün gegen GELU) 55 %, gegen ReLU6 20 %; Jury 4.

### 12. Reversible Hadamard-Schichten: Aktivierungen nicht speichern · Optik (Zeitumkehr/Phasenkonjugation: das Feld lässt sich zurückrechnen) · Unitarität: orthogonale/Hadamard-Kopplungen sind exakt umkehrbar, daher kann die Vorwärtsrechnung beim Rückwärtslauf rekonstruiert werden.
- **Mechanismus:** Residual-Blöcke in Kopplungsform (y1 = x1 + F(x2), y2 = x2 + G(y1)) mit Hadamard-Mischung dazwischen; beim Backprop werden Aktivierungen rückgerechnet, Speicher unabhängig von der Tiefe. Bei gleichen Trainings-Bytes tiefere/breitere Modelle; Preis: ≈ +33 % FLOPs, also **kein FLOP-Gewinn**, nur Bytes. Bekannt (RevNet, Reformer); Kernbeitrag wäre Hadamard als freie Invertierung.
- **Langweiligste Erklärung:** Es ist normales Gradient-Checkpointing mit anderem Namen (gleiche Bytes-Ersparnis, ähnlich viele FLOPs).
- **Billigster Test:** Zwei Messungen: (a) Spitzenspeicher und Zeit pro Schritt für MLP/Byte-LM mit 8/16/32 Lagen: reversibel gegen normal gegen Checkpointing (Kontrolle); (b) bpc bei gleichem Speicherbudget. ≈ 8 min.
- **Vorab-Tipp:** P(grün gegen Checkpointing) 15 %, gegen normal 85 % (trivial); Jury 2.

---

## Gesamteinschätzung (Vorab)
- Beste Aussichten bei harter Regel (gleiche FLOPs oder Bytes): **5, 3, 11, 4**; **7** als billigster, anschaulichster Test.
- Wahrscheinlich nur Randbefunde: 8, 9, 10, 12 (bekannt, geringer Neuheitswert, aber als Kontrolle oder Anwendungsgeschichte brauchbar).
- Wegen Vorgabe (Neuheit nicht nötig) sind bekannte Bausteine erlaubt, müssen aber zitiert werden (QuaRot/SRHT, FlyHash, HashedNets, RevNet, HRR/Hrrformer; alle aus dem Gedächtnis, vor Verwendung prüfen).
- Ehrlich: Mehrere Paper (Wavelet 2020, HashFormers 2022, HEDGES 2020) sind älter als 2026; die Suche lieferte dort nichts Aktuelleres.
