# Z1 Grenzen des Möglichen (z1g, 28.10.2026)

Erkundung, keine Befunde. Rechnung: `z1g/rd.py` (Gauß-Quantisierer, 1 Thread, < 1 min). Quellen: 7 WebSearches; Zahlen aus Sekundärquellen sind markiert und vor Zitat in der Arbeit am Original zu prüfen.

## 1. Untergrenzen: Bits je Gewicht

**a) Rate-Distortion (Gauß, MSE).** D(R) = σ²·2^(−2R): jedes Bit senkt den Fehler um den Faktor 4 (6,02 dB). Das ist die harte Grenze für i.i.d.-Gauß-Gewichte bei Fehlerkriterium MSE. Gerechnet (Lloyd-Max-Skalarquantisierer, 2 Mio. Gauß-Werte):

| Bit | R-D-Grenze D | Lloyd-Max D | Lücke | „effektive“ Rate bei diesem D |
|---|---|---|---|---|
| 1 | 0,250 | 0,364 | 1,6 dB | 0,73 Bit |
| 2 | 0,0625 | 0,118 | 2,8 dB | 1,54 Bit |
| 3 | 0,0156 | 0,0345 | 3,4 dB | 2,43 Bit |
| 4 | 0,0039 | 0,0096 | 3,9 dB | 3,35 Bit |

- Ein 2-Bit-Skalarquantisierer verschenkt also rund 0,45 Bit gegenüber der Grenze (bei hoher Rate strebt die Lücke gegen 0,25 Bit = 1,53 dB für Entropiecodierung). Rrest-Spielraum durch bessere Quantisierer: höchstens ca. 2,8 dB bei 2 Bit.
- Literatur (Sekundärquelle, prüfen): bei 2 Bit, Einheitsgauß: Lloyd-Max 0,118, QuIP# (E8, 8-dim) 0,089, QTIP (Trellis) 0,069, Grenze 0,063. QTIP liegt damit ≈ 0,4 dB an der Grenze. Zwei-Bit-Llama-2-70B dann nahe fp16 (Sekundärquelle: +0,7 Perplexität). → **Für Gauß-Gewichte ist bei ≥ 2 Bit praktisch alles erreicht; Neues ist nur noch in der Verteilung (nicht-Gauß) und im Fehlerkriterium zu holen.**
- **Rotation ist der Hebel, nicht der Quantisierer.** Mein Mini-Test (256×256, schwere Ränder t₃ + 3 Ausreißerspalten ×10, 2 Bit, Zeilenskala): relativer Fehler² 0,575 ohne, 0,145 mit Hadamard. Gauß-Fall wäre 0,118. Rotation macht die Gewichte „gaußisch“ und bringt sie damit an die Grenze der Gauß-Quelle; das passt zu R1 (Zufalls-Orthogonale = Hadamard).
- **Aber:** MSE der Gewichte ist nicht das Ziel. Das Ziel ist der Ausgabefehler E‖(W−Ŵ)x‖² = tr((W−Ŵ)Σₓ(W−Ŵ)ᵀ). Mit Eingangskovarianz Σₓ (GPTQ/Hessian) gilt Reverse-Water-Filling: Richtungen mit kleiner Eingangsvarianz brauchen weniger Bits. Die echte Grenze liegt deshalb **unter** der Gauß-Grenze, und zwar um so mehr, je ungleicher die Eigenwerte von Σₓ sind. (Das ist die Rechtfertigung für I-048/r1-artige Bitzuteilung und für Entropie-Rang w5.)

**b) Skalengesetze der Präzision (Kumar u. a. 2024, ICLR 2025).** Aussagen (aus Abstract/Sekundärquellen bestätigt): niedrige Trainingspräzision senkt die „effektive Parameterzahl“; Verlust durch Nachtrainings-Quantisierung (PTQ) wächst mit der Tokenzahl, bis zusätzliche Daten dem Inferenz-Modell schaden; größere Modelle in niedrigerer Präzision können rechenoptimal sein; 465 Läufe bis 1,7 B Parameter/26 B Token. Funktionsform N_eff = N·(1−e^(−P/γ)) steht nur im Gedächtnis, vor Zitat am Original prüfen. Folgen für QIRE:
 - **Frühregime-Warnung bestätigt:** PTQ-Schaden steigt mit Trainingsdauer. Unsere Kurzläufe (190 Schritte, 1 Epoche) unterschätzen den Quantisierungsschaden. Jede Bit-Aussage braucht zwei Trainingslängen, und PTQ am Ende, nicht nur QAT am Anfang.
 - Das Gesetz ist für Gewichte/Aktivierungen/KV-Cache getrennt; wir testen bisher nur Gewichte (MNIST-MLP). Aktivierungen/KV sind die offene Flanke.
 - Rotation wirkt in diesem Bild als „höheres γ“: weniger Verlust von N_eff bei gleichem P. Das ist als Hypothese testbar (siehe Idee 5).

**c) Wissenskapazität (Allen-Zhu & Li, ICLR 2025).** ≈ 2 Bit Faktenwissen je Parameter, auch mit int8 (Sekundärquelle bestätigt); triviale Obergrenze 8 Bit bei int8, also 2 ≪ 8. Das Gesetz gilt für gut trainierte (≥ 1000 Expositionen je Fakt) synthetische Wissensdaten. Zwei Lesarten:
 - Parameter-Bits sind nicht der Engpass, **Training/Exposition** ist es: 2 Bit/Param bei 16 Bit Speicher = 12,5 % Auslastung. Daraus folgt eine Kompressionsgrenze für Wissen von grob 2 Bit/Param, also ein 2-Bit-Netz hat keinen Platz mehr, wenn es dieselbe Kapazität behalten soll, außer das Netz ist vorher stark überparametrisiert (was Praxis-LLMs nicht sind, 2 Bit/Param erreichen sie nur bei langem Training). **Ich nehme an, dass unter ≈ 2 Bit/Param Speicher die Wissenskapazität pro Gewicht schneller fällt als linear; für int4 oder niedriger habe ich keine bestätigte Zahl gesehen. Nachlesen.**
 - Das Gesetz sagt nichts über Fähigkeiten (Rechnen, Syntax), die komprimierbarer sind; Mikro-Aufgaben wie MNIST liegen weit weg von der Grenze.

**d) Wie nah ist QIRE?** Aus SYNTHESE: MNIST-MLP, 2 Bit + Rotation + Clip 97,3 % gegen fp 97,8 %; R1: großes Modell + Rotation + 2 Bit (98,30) schlägt klein fp16 bei gleichen Bytes (97,90). Bewertung:
 - Skalar-2-Bit liegt ≈ 2,8 dB (≈ 0,45 Bit) über der Gauß-Grenze; in Bytes entspräche die Grenze ≈ 1,55 Bit für denselben Fehler. **Maximaler weiterer Bytegewinn durch bessere Quantisierer bei MNIST ≈ 20–25 %**, nicht Faktoren. Das ist der ehrliche Deckel für „Hadamard + 2 Bit“.
 - Die Faktor-7 (1,57 → 0,22 MB) kommt fast ganz von fp32 → 2 Bit (16×) minus Skalen/Overhead, nicht von QIRE-Neuem. Der QIRE-Anteil ist die Rotation (94,9 → 97,3), also die Rückholung der Lücke zur Grenze.
 - MNIST-MLP ist weit über-parametrisiert (Kapazität ≪ Parameterzahl); deshalb sind 2 Bit dort „billig“. Die Aufgabe für die Arbeit: dieselbe Messung am Byte-LM, wo Kapazität knapper ist (Allen-Zhu-Regime).

## 2. Was ist bei gleichen FLOPs gewinnbar?

- Dichte Matmul: y = Wx braucht d_out·d_in MACs und ist für **beliebiges** W optimal im algebraischen Bilinear-Komplexitätsmodell bis auf Strassen-artige Exponenten (ω < 2,372), die in der Praxis nichts bringen (Konstanten, Hardware). Eine Struktur ist nur dann „gratis“, wenn die Zielabbildung in einer kleineren Klasse liegt.
- Strukturierte Klassen und ihre Kosten: Niedrigrang r: r(d_in+d_out); Monarch/Butterfly: d·√d bzw. d·log d; Hadamard/FFT: d·log d; Toeplitz/zirkulant: d·log d; Kronecker: d·(a+b). Alle haben **weniger Parameter und weniger FLOPs, können aber nicht jedes W darstellen** (Dimension der Klasse ≪ d²). Gewinn nur, wenn die gelernte Abbildung nahe an der Klasse liegt (Kriterium: Residuum der besten Projektion von trainiertem W in die Klasse, mit Σₓ gewichtet).
- Konsequenz für unsere Befunde: Monarch „schlägt Niedrigrang, gleichauf mit schmal dicht“ (w2) und Tensor-Train „rot“ (w1) sind genau das erwartete Muster: bei gleichen FLOPs ist das schmale dichte Netz nicht zu schlagen, solange die Aufgabe keine Struktur hat. Ein Vorteil muss aus der **Aufgabe** kommen (Bildfaltungsstruktur, Periodizität, Hierarchie) oder aus dem **Inhaltsabhängigen** (Sparsity/MoE: FLOPs sinken, weil nur ein Teil der Abbildung gebraucht wird; Multiskala/Hourglass: weniger Positionen).
- FLOPs sind nicht Zeit: Auf CPU/GPU ist Speicherbandbreite bei Inferenz mit Batch 1 oft der Engpass; dort zählen Bytes (2-Bit-Gewichte = 8× weniger Verkehr als fp16) mehr als FLOPs. Das stützt die Leitlinie „Speicher-/Bit-Aussage“. Auf dem Papier ist die Entpackung (2 Bit → Float) zusätzliche Arbeit; echte Zeitgewinne muss ein Kernel zeigen.
- Heuristik für die Arbeit: Gewinnbar bei gleichen FLOPs sind (i) Bytes (Quantisierung), (ii) Positions-FLOPs (Multiskala, frühes Aussteigen, spekulatives Dekodieren), (iii) bedingte FLOPs (MoE/Sparsity). Nicht gewinnbar ohne Aufgabenstruktur: dichte Schicht durch strukturierte Schicht ersetzen.

## 3. Echte Quantengrenze

- **Holevo:** n Qubits tragen höchstens n klassische Bit abrufbar (χ ≤ S(ρ̄) − ΣpᵢS(ρᵢ) ≤ log₂ dim). Ein Zustand der Dimension 2ⁿ „speichert“ 2ⁿ Amplituden, liefert aber ≤ n Bit je Messung. Daher ist Amplitudencodierung (Einbettung von d Merkmalen in log₂ d Qubits) kein Speichergewinn für Ausgaben; der Gewinn liegt nur im Zwischenzustand. Für K Klassen genügt ⌈log₂ K⌉ Bit, mehr kann ein Auslesen nicht liefern (I-040: Wirkungsgrad gegen diese Schranke; Ergebnis war klassisch erklärbar).
- **Born-Regel/Schussrauschen:** Eine Wahrscheinlichkeit p wird mit S Schüssen nur auf ±√(p(1−p)/S) geschätzt (Standard-Quantengrenze, 1/√S). Heisenberg-Skalierung 1/S braucht Amplitudenschätzung mit kohärenter Tiefe, also fehlertolerante Hardware. Folge: Jeder Born-Auslese-Layer kostet S Wiederholungen je Ausgabe; ein Layer „(q·k)²“ ist auf Hardware nie billiger als ein klassisches Skalarprodukt, außer das S-fache wird durch Quantenparallelität zurückgeholt. I-046 (Pareto Genauigkeit × Information je Schuss) misst genau diesen Tausch.
- **Datenzugang (QRAM):** Quantenvorteile für lineare Algebra/Kernel (HHL, qSVM) setzen QRAM voraus. Tang u. a. („dequantization“, Chia–Gilyén–Li–Lin–Tang–Wang): für niedrigrangige, per Sampling zugängliche Daten gibt es klassische Algorithmen mit höchstens polynomiell größerem Aufwand; exponentielle Vorteile bleiben nur bei dünnen Eingaben (BQP-vollständig). **Für neuronale Netze ist kein belegter Ressourcenvorteil bekannt; ich habe dazu in dieser Runde nichts Neues gefunden.**
- **Was ein Quantenrechner leisten müsste** (Anforderungsliste, Tendenz, keine Rechnung):
 1. Fehlertolerant, damit Tiefe/Amplitudenschätzung geht (Größenordnung 10³–10⁶ physikalische je logisches Qubit).
 2. Daten laden in polylog Zeit (QRAM) oder die Daten sind selbst Quantenzustände (Sensor, Simulation). Laden von N klassischen Zahlen kostet sonst Ω(N), das frisst den Gewinn bei Einmal-Auswertung.
 3. Ausgabe kurz (wenige Bit, Holevo), nicht ein Vektor mit d Einträgen (Auslesen kostet d·S Schüsse).
 4. Aufgabe mit Quantenstruktur (Simulation, Faktorisierung, Gitter-/Grover-artige Suche mit quadratischem Gewinn √N). Grover gibt nur quadratisch, und bei fehlertoleranter Taktrate (kHz–MHz gegen GHz) kippt der Vorteil erst bei sehr großen N.
 - Für **Training** großer klassischer Daten sehe ich keinen Weg; für Lernen **über Quantendaten** gibt es nachweisbare Vorteile (Huang u. a. 2022, aus dem Gedächtnis, nicht in dieser Runde geprüft).
- Ehrliche Formel für die Arbeit: „Quanten-inspirierter Vorteil“ = klassische Techniken (Rotation, Basiswechsel, Tensornetze), die in der Quantenphysik entstanden. Eine echte Quantengrenze (Holevo, 1/√S) wird nur als Obergrenze zitiert: *selbst ein perfekter Quantenrechner würde unsere Bit-Aussagen nicht unterschreiten*.

## 4. Fünf Ideen, die eine Grenze umgehen

Prinzip: jede Grenze gilt für eine Modellklasse (Gewichte als i.i.d. Gauß gespeichert, MSE-Kriterium, dichte Abbildung). Umgehen = Klasse wechseln.

| # | Idee | Umgeht | Kleinster Test (≤ 10 min CPU, Tuning-Seeds 950–989, 2 Trainingslängen) |
|---|---|---|---|
| 1 | **Gewichte erzeugen statt speichern**: W = Σ aₖ·Gₖ(Seed), Gₖ pseudozufällig (Hadamard-Zeilen mit Vorzeichen-Seed); gespeichert nur aₖ und Seed (Intrinsic-Dimension Li 2018, NOLA, SeedLM) | Rate-Distortion für i.i.d.-Gewichte; Bits/Param < 1 | MNIST-MLP: je Schicht W = Zufalls-Basis × gelernte k Koeffizienten, k so, dass Bytes = 0,25 · 2-Bit-Basismodell. Vergleich bei gleichen Bytes gegen Hadamard+2 Bit und Niedrigrang. Kenngröße: Bytes bei 97 %. Vorbehalt: wird FLOP-teurer, wenn W jedes Mal neu erzeugt wird |
| 2 | **Rechnen im komprimierten Raum**: Eingang und Gewichte im Hadamard-/Zufallsbasis-Raum halten, nur die k wichtigsten Koordinaten (Σₓ-gewichtet) behalten; Matmul k×k statt d×d | Dichte Matmul bei gleicher Genauigkeit (nutzt Aufgabenstruktur: Σₓ hat schnellen Eigenwertabfall) | Eingang mit Hadamard rotieren, vorn k Koordinaten behalten (Reverse-Water-Filling nach Varianz), Rest verwerfen. Gleiche FLOPs: gegen schmal dichtes Netz. Verdacht: gleichauf, siehe w5; also nur sinnvoll, wenn Σₓ extrem steil ist (Messen: Eigenwertspektrum von Σₓ je Schicht vorab) |
| 3 | **Bits < 1 durch Teilen/Hashen**: Hashing-Trick (HashedNets) auf 2-Bit-Werte, mehrere Gewichte teilen einen Codebuch-Eintrag; plus Rotation vorher, damit geteilte Einträge unabhängig-gaußisch wirken | Eine Zahl je Gewicht | W (rotiert, 2 Bit) → 4er-Gruppen per Vektorquantisierung (Codebuch 2^k < 4·2^2): k = 2 Bit je 4 Gewichte = 0,5 Bit/Gewicht. Kenngröße: Fehler gegen Zufalls-Gruppierung und gegen die R-D-Kurve (bei Gauß 0,5 Bit → D ≥ 0,5). Realistisch nur mit Nachtraining (QAT) |
| 4 | **Fehlerkriterium wechseln**: Σₓ-gewichtete Bitzuteilung (Reverse-Water-Filling) statt MSE-gleich; je Zeilenblock 1/2/3 Bit nach Eingangsvarianz bei festem Mittel 2 Bit | MSE-Grenze für Ausgabefehler; Bit-Mittel bei gleicher Güte | MNIST-MLP: gemessene Σₓ je Schicht, Bits nach Varianz zuteilen (Mittel 2,0), Vergleich mit gleichmäßig 2 Bit. Erwartung: kleiner, aber echter Gewinn (≈ log₂ der Eigenwert-Heterogenität); einfachster Test der Liste |
| 5 | **Rotation als γ-Erhöher im Präzisions-Skalengesetz**: Wenn Rotation N_eff erhöht, lässt sich der Abstand verschieben (kleineres Modell + weniger Bit ≙ gleiche Güte). Test, ob Kumars PTQ-Schaden mit Tokenzahl bei rotierten Gewichten flacher wächst | Das Präzisions-Gesetz selbst (nicht umgehen, sondern Konstante verändern) | Byte-LM 2 Größen × 2 Trainingslängen (190 und 4× Schritte): Anstieg des PTQ-Schadens (2-Bit-Verlust − fp-Verlust) mit/ohne Rotation. Wenn Anstieg ohne Rotation > mit, ist das Kern-Argument für die Arbeit und zugleich die Frühregime-Kontrolle |

Rangfolge nach Aufwand/Erwartung: **4 (billig, fast sicher) → 5 (Pflicht zur Absicherung von Rang 1) → 3 → 2 → 1** (1 ist riskant, FLOP-Kosten unklar; kann bei Bytes gewinnen).

## 5. Folgen für den Zyklus
1. In die Arbeit: Obere Schranke „Gauß-R-D“ als Tabelle (oben); QIRE-Rotation + Skalar-2-Bit liegt ≈ 0,45 Bit darüber. Das ist stärker als jede Vergleichstabelle: es zeigt, was **nicht** mehr zu holen ist.
2. R1-Folgetest mit Trainingslängen-Kontrolle (Idee 5) vor jeder vorregistrierten Bytes-Studie L1.
3. Aufgabenstruktur messen (Σₓ-Spektrum je Schicht) vor jedem Strukturversuch; erklärt im Voraus, ob Monarch/TT/Niedrigrang überhaupt Chancen haben.
4. Offen / ungeprüft: Allen-Zhu-Zahl für int4 und tiefer; Kumars genaue Formel und γ-Werte; Huang-2022-Aussage; ob QTIP-Zahlen (0,118/0,089/0,069/0,063) im Original so stehen.

Quellen: [Kumar u. a., Scaling Laws for Precision](https://arxiv.org/abs/2411.04330v1) · [Allen-Zhu & Li, Physics of LMs 3.3](https://arxiv.org/abs/2404.05405v1) · [QTIP](https://arxiv.org/html/2406.11235v4) · [Tang, Dequantization](https://ar5iv.arxiv.org/html/1811.00414) · [SeedLM](https://arxiv.org/pdf/2410.10714) · [NOLA](https://mlanthology.org/iclr/2024/koohpayegani2024iclr-nola) · [NanoQuant (Sub-1-Bit)](https://arxiv.org/html/2602.06694v1) · [LittleBit](https://nips.cc/virtual/2025/poster/115061)
