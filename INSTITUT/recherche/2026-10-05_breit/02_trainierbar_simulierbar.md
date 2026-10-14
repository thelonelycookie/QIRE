# 02 — Trainierbarkeit gegen klassische Simulierbarkeit (Barren Plateaus, Dequantisierung)

Stand: 05.10.2026. Recherche per WebSearch (arxiv.org, PMC und Nature-Volltext waren für WebFetch gesperrt, WebSearch-Budget 200 aufgebraucht). **Fast alle Angaben beruhen nur auf Suchauszügen/Abstracts, nicht auf Volltexten.** Das ist unten je Quelle markiert („nur Auszug“). Eigenes = [EIGEN], Unbelegtes = [?].

## 1. Kurzfassung

- Konsens 2018–2026: Die Beweisstrategien, die Barren Plateaus (BP) ausschließen, legen meist einen kleinen (polynomgroßen) Unterraum offen, in dem die Rechnung klassisch nachvollzogen werden kann. Cerezo et al. (Nat. Commun. 2025) belegen das „Fall für Fall“, ausdrücklich **kein allgemeiner Satz**; Einschränkung: oft braucht der klassische Weg Daten aus einer Quantenvorlaufphase („Yes and No“).
- Gegenstimmen/Gegenbeispiele sind dünn: ein HEA-Ansatz in der Vielteilchen-Lokalisierungsphase (arXiv 2403.04844, nur Auszug), Anschuetz/Kiani (Fallen statt BP), Quantendaten-Aufgaben (Huang et al. 2022), und 2026 Wang/Du/Tao „learnable yet not simulable“ (nur mit quantendaten-gestütztem Surrogat). **Mir ist kein anerkanntes Beispiel bekannt: trainierbar + beweisbar nicht simulierbar + nützlich auf klassischen Daten.** [Stand meiner Suche, keine Garantie]
- Dequantisierung (Tang 2018 ff., Schreiber/Eisert/Meyer 2023, Shin et al. 2024, Gil-Fuster et al. ICLR 2025) und Benchmarks (Bowles et al. 2024: Klassik schlägt Quantenmodelle, Entfernen der Verschränkung schadet oft nicht) zeigen dasselbe Muster aus Anwendungsseite.
- **QIRE:** Bei 8 Qubits ist alles trivial simulierbar; die Clifford-Zelle ist per Konstruktion ein schneller klassischer Transform (signiertes Walsh-Hadamard). QIRE ist ehrlich als **parameter- und strukturreiches klassisches/„quanteninspiriertes“ Modell** zu formulieren, nicht als Quantenvorteil. Der belastbare Beitrag: Befund, **wie wenig Magie (k≈0–4 Winkel) nötig ist** und welche Struktur (Rang der quadratischen Form) trägt.
- Magie ist **notwendig, aber nicht hinreichend** für Nicht-Simulierbarkeit; k Magiewinkel in einer Clifford-Zelle geben einen exakt klassisch darstellbaren Raum mit ≤ 3^k Frequenzen (siehe 4, I-1). Das macht „Magie-Budget als Regler“ zu einer messbaren, billigen Forschungsfrage.
- Top-Ideen: (I-1) exaktes 3^k-Surrogat der QIRE-Zelle + Kippmessung Trainierbarkeit/Simulierbarkeit gegen k; (I-2) Bowles-Ablation: identische Struktur ohne Quantenformalismus; (I-3) Gradientenvarianz gegen k und n im Clifford-Gerüst.

## 2. Stand der Forschung

### 2.1 Barren Plateaus: Theorie
- **McClean et al. 2018**, Nat. Commun. 9, 4812: Gradientenvarianz zufälliger tiefer PQCs fällt exponentiell mit Qubitzahl. https://research.google/pubs/barren-plateaus-in-quantum-neural-network-training-landscapes/ (nur Auszug)
- **Cerezo, Sone, Volkoff, Cincio, Coles 2021**, Nat. Commun. 12, 1791: kostenfunktionsabhängig. Globale Observablen → BP schon bei flachen Schaltungen; lokale Observablen → höchstens polynomiell kleiner Gradient bei Tiefe O(log n). https://www.nature.com/articles/s41467-021-21728-w (nur Auszug)
- **Ragone et al. 2024** (Lie-algebraische Theorie, Nat. Commun. 15) und **Fontana et al. 2024** (adjungierte Darstellung, Nat. Commun. 15, 7171): exakte Varianz tiefer Schaltungen aus der Dimension der dynamischen Lie-Algebra (DLA); löst die Vermutung „Konzentration ↔ DLA-Dimension“. https://www.nature.com/articles/s41467-024-49909-3 (nur Auszug)
- **Larocca et al. 2025**, Nat. Rev. Phys. 7, 174 (Übersicht). https://www.nature.com/articles/s42254-025-00813-9 (nur Auszug)
- **Anschuetz & Kiani 2022**, Nat. Commun. 13, 7760: auch ohne BP sind flache Modelle voller lokaler Minima („Fallen“); Optimierung mit verrauschten Abfragen im SQ-Modell oft unmöglich. https://www.nature.com/articles/s41467-022-35364-5 (nur Auszug)
- Relevant für Clifford: **Heyraud et al. 2023** (PRX Quantum 4, 040335): Gradientenvarianz zufällig initialisierter Schaltungen exakt über eine Abbildung auf Clifford-Schaltungen berechenbar. https://doi.org/10.1103/PRXQuantum.4.040335 (nur Auszug). **Zhang/Zhu/Wang 2024**: Lernbarkeit über Landschaftsfluktuation, per Clifford-Sampling klassisch schätzbar. https://arxiv.org/abs/2406.11805 (nur Auszug)

### 2.2 Die Debatte „BP-frei ⇒ simulierbar?“
- **Cerezo et al. 2025**, Nat. Commun. 16, 7907 (arXiv 2312.09121). Autoren u. a. Larocca, Fontana, Rudolph, Holmes, Anschuetz. Aussage (Auszug): bei vielen gängigen BP-freien Modellen lässt sich der Verlust klassisch simulieren, **sofern man in einer Datenerfassungsphase klassische Daten vom Quantengerät sammelt**; die Beweise der BP-Freiheit identifizieren polynomgroße Unterräume (z. B. Pauli-Operatoren mit Gewicht O(log n)). Fazit „Yes and No“: klassisch **oder mit effizientem Quantenalgorithmus** simulierbar; eine Beweislücke („Fall für Fall“) bleibt. https://www.nature.com/articles/s41467-025-63099-6 (nur Auszug)
- **Gegenbeispiel-Kandidat:** „Hardware-efficient ansatz without barren plateaus in any depth“ (arXiv 2403.04844): nach Sekundärauszug beruht es auf Vielteilchen-Lokalisierung (MBL), für die kein klassischer Algorithmus für exponentiell lange Zeiten bekannt ist. Nur Auszug, **Aussage selbst nicht geprüft** [?]. Verwandt: Exploiting MBL for scalable variational simulation, https://arxiv.org/pdf/2404.17560
- **Bermejo et al. 2026**, PRX Quantum 7, 020304: QCNNs sind „effektiv klassisch simulierbar“, sogar das erfolgreiche Training lässt sich auf simulierter Landschaft nachbilden; typische Benchmarks sind „lokal einfache“ Datensätze. https://journals.aps.org/prxquantum/abstract/10.1103/8qt9-72ts (nur Auszug)
- **Gil-Fuster, Gyurik, Pérez-Salinas, Dunjko, ICLR 2025:** definiert Trainierbarkeit und Dequantisierung formal und fragt, wann ersteres letzteres impliziert; „recent works raised into question“, ob Modelle trainierbar und nicht dequantisierbar sein können. https://arxiv.org/pdf/2406.07072 (nur Auszug; ihr genaues Ergebnis/ihre Auswege nicht gelesen [?])
- **Wang, Du, Tao 2026** (arXiv 2608.02325, nur Auszug): neues Maß „dynamische Stabilizer-Entropie“ (DSE); Phasendiagramm klassischer Simulator vs. quantendaten-gestütztes Surrogat; Beweis (unter Komplexitätsannahmen), dass es Schaltungsfamilien gibt, die ein Surrogat **lernt**, aber nicht aus der Schaltungsbeschreibung allein emulierbar sind. Das ist die präziseste neue Form von „lernbar, aber nicht simulierbar“ — aber das Lernen braucht Daten vom Quantengerät. https://arxiv.org/abs/2608.02325
- Weitere Hinweise (nur Suchtreffer, nicht geprüft [?]): Sekundärquelle nennt „generative sewing“ (Huang/Google) als BP-freien Weg mit generativem Vorteil; Huang et al., „Quantum advantage in learning from experiments“, Science 2022 (Quantendaten, nicht klassische Daten). https://www.science.org/doi/10.1126/science.abn7293

### 2.3 Werkzeuge der klassischen Simulation
- **g-sim** (Goh, Larocca, Cincio, Cerezo, Sauvage; PRResearch 7, 033266, 2025; arXiv 2308.01432): Simulation über die Lie-Algebra; effizient, wenn DLA-Dimension höchstens polynomiell wächst; Training/Initialisierung möglich; kann auch Magiezustände als Anfangszustand behandeln. https://pennylane.ai/demos/tutorial_liesim (nur Auszug)
- **LOWESA** (Rudolph, Fontana, Holmes, Cincio 2023, arXiv 2308.09109): Pauli-Propagations-Surrogat, rekonstruiert ganze Erwartungswert-Landschaften (127-Qubit-Ising, Minuten bis Stunden auf Laptop). https://arxiv.org/abs/2308.09109
- **Pauli-Propagation** allgemein: Angrisani et al. (PRL 2025, arXiv 2409.01706): klassisch schätzbar für die meisten zufälligen unstrukturierten Schaltungen mit einzelqubit-invarianter Verteilung. https://arxiv.org/abs/2409.01706 ; Rahmenwerk https://journals.aps.org/prxquantum/abstract/10.1103/6vd7-l9bn
- **Fast-Clifford:** Bravyi/Gosset (stabilizer rank, Kosten exponentiell in der T-Zahl, Exponent ≈ 0,23 pro T nach Sekundärauszug), Bravyi–Smith–Smolin 2016; Clifford-Störungstheorie (Begušić/Chan, arXiv 2306.04797); **Shallow Magic Depth** (arXiv 2409.13809, nur Titel). CAFQA (arXiv 2202.12924): Clifford-Suche (diskret) als klassisch simulierbarer Ansatz-Start für VQE, 99 % der Korrelationsenergie bis 20 Qubits, mit wenigen T-Gattern weiter verbessert (nur Auszug). **Das ist strukturell Maltes Clifford-Suche.**
- Magie und Lernen: Clifford-Verteilungen sind lernbar, **ein einziges T-Gatter** macht Dichtemodellierung schwer („One T Gate Makes Distribution Learning Hard“, nur Titel/Auszug); t-dotierte Stabilizerzustände sind für kleines t effizient lernbar (Leone, Oliviero, Hamma, Quantum 2024); t=Θ̃(nk) T-Gatter nötig für relatives k-Design (Auszug).

### 2.4 Dequantisierung und Benchmarking
- **Tang 2018 ff.**: quanteninspirierte klassische Algorithmen (Empfehlungssysteme), Rahmen über Sample-and-Query-Zugang (Gilyén/Lloyd/Tang, Chia et al.), polynomielle Verlangsamung. https://arxiv.org/abs/1910.06151 ; Gegenposition **Cotler/Huang/McClean 2021**: SQ-Zugang ist möglicherweise ein zu mächtiger klassischer Rohstoff, Vergleich evtl. unfair. https://arxiv.org/abs/2112.00811
- **Schreiber/Eisert/Meyer 2023** (PRL; arXiv 2206.11740): Re-uploading-Modelle sind abgeschnittene Fourier-Reihen und haben effizient konstruierbare klassische Surrogate; keine Vorteile bei Leistung/Trainierbarkeit in den Tests. https://arxiv.org/abs/2206.11740 ; Skalierbarkeit der Surrogate: https://arxiv.org/pdf/2508.06131
- **Shin/Teo/Jeong 2024** (PRResearch 6, 023218): VQML-Modelle sind Teilklasse von MPS-Modellen mit eingeschränkten Koeffizienten (Tensornetz-Dequantisierung). https://arxiv.org/abs/2307.06937
- **Schuld & Killoran 2022** (PRX Quantum 3, 030101): Aufruf, „Quantenvorteil“ als dominantes Ziel zu hinterfragen; alternative Perspektiven (Modellklassen, Struktur, Theorie). https://arxiv.org/abs/2203.01340
- **Bowles/Ahmed/Schuld 2024** (arXiv 2403.07059): 12 QML-Modelle, 6 Aufgaben, 160 Datensätze; klassische Standardmodelle schlagen die Quantenklassifikatoren, Entfernen der Verschränkung gibt oft gleiche oder bessere Ergebnisse. https://arxiv.org/pdf/2403.07059

## 3. Was das für QIRE bedeutet (ehrlich)

1. **Skala.** 8 Qubits = 256 Amplituden. Jede Zelle (auch mit k=32 Winkeln) ist auf Maltes PC exakt simulierbar; die Debatte betrifft nur die **Skalierung** (n→∞), nicht die heutigen Messungen. Keine Aussage „QIRE ist trainierbar und nicht simulierbar“ ist belegbar.
2. **Die Clifford-Zelle ist klassisch, per Konstruktion.** Maske (CZ/S-Anteil, quadratische Form über GF(2)) × blockweiser Walsh-Hadamard = ein fester, sehr schneller klassischer Transform (O(N log N)). Gottesman-Knill stimmt genau mit Maltes Bruhat-Befund überein. Ein klassisches Netz mit „signierte Maske + WHT“ hat dieselbe Funktion ohne Quantenformalismus (→ Bowles-Ablation, I-2).
3. **Trainierbarkeit ist hier kein Problem, sondern ein Vorteil der Klasse.** Diskrete Clifford-Suche umgeht Gradienten (wie CAFQA); der Befund „Suche sitzt in anderem Becken als Gradient“ ist ein Hinweis auf eine rauhe Landschaft, aber ohne BP-Mechanismus bei n=8.
4. **Magie als Brücke:** Magie ist **notwendig** für Nicht-Simulierbarkeit (Gottesman-Knill), aber **nicht hinreichend**: wenig Magie (k=O(log n) T-Gatter), kleine Winkel (Pauli-Propagation, Clifford-Störungstheorie) oder strukturierte Magie bleiben klassisch. Maltes Magiebedarf (MNIST 0, Fashion ≈ 4 Winkel, M50 ≈ 62 T bei ε=10⁻³ laut UEBERGABE §4) liegt **tief im klassisch simulierbaren Bereich** (Stabilizer-Rang ~2^(0,23·T) ist für T≈62 groß, aber 4 Winkel mit exakter 3^4-Darstellung sind trivial; zwei verschiedene Maße, nicht verwechseln).
5. **Dequantisierungs-Brille:** Die QIRE-Ausgabe mit k trainierten Winkeln liegt in einem Raum trigonometrischer Polynome mit ≤ 3^k Frequenzen (I-1). Für k=4: ≤ 81 Terme je Ausgabe = klassisches Surrogat exakt. Für das Argument „Quantenvorteil“ wäre genau das die Widerlegung.
6. **Ehrliche Formulierung des Beitrags** (Vorschlag für die schriftliche Arbeit):
   - *Nicht:* „Quantenmodell schlägt klassisches“ oder „unendlich skalierbar dank Quantenphysik“.
   - *Sondern:* „Wir zeigen empirisch, dass ein quanteninspiriertes, extrem parameterarmes Modell (Kodierung + feste Clifford-Struktur + k ≤ 4 Magiewinkel) den Großteil der Leistung eines vollen Trainings erreicht; der Gewinn steckt in einer klassisch beschreibbaren Struktur (Rang der quadratischen Form). Das ist ein Effizienzbefund (Parameter, Struktur), kein Rechenvorteil. Die Quantenbrücke ist Magie; ob sie bei größerem n überhaupt jenseits klassischer Simulation führt, ist offen.“
   - Quantenrelevanz bleibt: (a) Magie-Messung als Strukturmaß, (b) mögliche Hardware-Ausführung (Rz virtuell, 0 T), (c) die offene Frage, ob **bei großem n** das Magiebudget k=O(1) reicht und die Auslese (Born, 2^n Werte, mit linearem Kopf) ein Vorteil gegenüber klassischer Abtastung ist (Holevo-Aspekt, Thema anderer Berichte).
7. **Datenlade-Flaschenhals:** Amplitudenkodierung 2^n Pixel braucht entweder klassische Vorbereitung O(2^n) oder QRAM. Das ist der Punkt, an dem die Tang-Dequantisierung greift (Sample-and-Query-Zugang auf Eingabe). Ohne effizientes Laden ist Vorteil unplausibel [?, Argumentationsskizze, keine geprüfte Quelle].
8. **Zu Maltes Frage „unendlich skalierbar?“:** Trainierbarkeit bei großem n ist für QIRE-Lagen (Ry+CNOT-Ring, globale Born-Auslese mit linearem Kopf) theoretisch **ungeklärt**: globale Observablen sind nach Cerezo 2021 BP-gefährdet; die Clifford-Anteile entkommen, weil sie diskret gesucht werden. Sagen: „bei n=8 kein Problem, Skalierung offen, BP-Risiko bei Gradiententraining der Winkel plausibel“. [EIGEN-Einschätzung]

## 4. Neue Ideen / Forschungsfragen

Neuheitsprüfung: je 3–6 Suchen (Stand 05.10.2026, WebSearch). Treffer sind nur Titel/Auszüge. „Nichts gefunden“ heißt nur: in der Suchmaschine nicht gefunden.

### I-1 [EIGEN] Exaktes 3^k-Surrogat und „Simulierbarkeitskosten gegen k“
- **Frage:** Fixiert man in einer QIRE-Zelle ein Clifford-Gerüst und lässt k Ry-Winkel frei, ist jede Born-Wahrscheinlichkeit ein trigonometrisches Polynom mit höchstens 3^k Frequenzen (jedes Ry(θ) mit Generator σ/2 trägt Frequenzen 0,±1 im Quadrat der Amplitude). Wie viele davon sind **praktisch** nötig (Energie in Moden)? Wo liegt Maltes Fashion-Optimum (k≈4) relativ zu 3^k, und ab welchem k wird das Surrogat (Pauli-Propagation mit Abschneiden, oder Moden-Auswahl) unbrauchbar?
- **Neuheit:** Fourier-Surrogate für Daten-Einbettung (Schreiber et al.) und Pauli-Propagation für Parameterlandschaften (LOWESA, Rudolph et al.) sind bekannt; Wang/Du/Tao 2026 messen „Frequenzverteilung“ (DSE). Gesucht: „Magie-Budget k als Regler für Surrogat-Genauigkeit in QML-Modell mit Clifford-Gerüst“ — nichts Spezifisches gefunden. Die Anwendung auf **gesuchte Clifford-Gerüste mit k Winkeln** und **echter Leistungsmetrik (Accuracy)** ist vermutlich neu. Der mathematische Kern (3^k) ist Folklore [?].
- **Billigster Test:** n=8 und n=12 (4096 Amplituden), Gerüst aus Maltes besten Clifford-Suchen; k=1…8 Winkel; Moden per FFT über Winkelgitter 3^k (k≤8: 6561 Punkte × Zelle, Sekunden bis Minuten). Messen: Energieanteil der besten m Moden; Accuracy eines Surrogats mit m Moden gegen Zelle. Laufzeit ≈ 10–30 min Rechnung, ½ Tag Programmierung.
- **Kontrolle:** zufälliges Clifford-Gerüst (nicht gesucht), und Nicht-Clifford-Gerüst (volle Ry-Lagen, k = alle Winkel).
- **Quantenstufe:** Q1 (klassisch simuliert). Aussage über Q3-Linie („Magiebudget“).

### I-2 [EIGEN, Methode nach Bowles 2024] Ablation „Quantenstruktur entfernen“
- **Frage:** Erreicht ein klassisches Netz mit gleicher Struktur (feste Vorzeichenmaske + blockweiser Hadamard-Transform + k trainierte Drehungen, gleicher linearer Kopf) dasselbe? Und: reproduziert ein klassisches Netz gleicher Parameterzahl (~2.570 bzw. ~100) die Leistung? Wo bleibt ein Rest, der „Quanten-Zutat“ heißen darf?
- **Neuheit:** Bowles et al. machten „Verschränkung entfernen“ für 12 Standardmodelle; für Clifford-/Magie-Strukturen und Parametereffizienz nichts gefunden. Verwandte klassische Strukturen (Butterfly-/Hadamard-/Fastfood-Schichten) kenne ich aus dem Gedächtnis, **nicht gesucht** [?].
- **Test:** Ein Tag; je 10 Seeds, MNIST/Fashion, wenige Minuten je Lauf auf CPU.
- **Kontrolle:** Gleiche Parameterzahl, zufällige feste Orthogonalmatrix statt WHT; gleiche Parameterzahl MLP/logistische Regression auf PCA-Merkmalen (Bowles-Stil „out-of-the-box“).
- **Quantenstufe:** Q1; entscheidend für die ehrliche Formulierung (Abschnitt 3.6).

### I-3 [EIGEN] Gradientenvarianz gegen Magiewinkelzahl k und Qubitzahl n
- **Frage:** Gerüst = zufälliges Clifford (+ Maske), k generische Ry-Winkel, Verlust = Kreuzentropie des linearen Kopfes oder einfacher Kosten über Born-Auslese. Wie skaliert Var(∂θ) mit n bei festem k? Gibt es einen Übergang bei k ~ log n (Gottesman-Knill-Grenze bei „poly“ T)? Zusammenhang: Heyraud et al. berechnen Gradientenvarianz über Clifford-Abbildung; Zhang et al. nutzen Clifford-Sampling.
- **Neuheit:** gesucht „gradient variance vs number of non-Clifford rotations / magic budget / trainability transition“: BP- und Magie-Literatur getrennt, T-dotierte Zirkel nur mit Lern-/Design-Bezug (Leone, Oliviero, Hamma; Spektral-/Magie-Generierung). Eine direkte Messung „Var(∂L) vs (n, k)“ nicht gefunden — nur Suchmaschinenbefund, könnte in einer Arbeit stehen, die ich nicht erreicht habe.
- **Test:** n=4…16 (Zustandsvektor 65 536; Gradient per Parameter-Shift), k=0,1,2,4,8,16,32; 200 Zufalls-Instanzen je Zelle. Mit NumPy/Qiskit-Aer-ähnlichem Eigenbau: grob 20–60 min. Zeitbedarf wächst mit 2^n; bis n=14 ok.
- **Kontrolle:** (a) Gerüst nicht Clifford (alle 2nL Winkel frei) = McClean-Fall; (b) k Winkel klein (Pauli-Propagation-freundlich); (c) lokale gegen globale Kostenfunktion (Cerezo 2021).
- **Erwartung** [EIGEN-Hypothese, ungeprüft]: bei festem k fällt Var mit n höchstens polynomiell (nur k Richtungen, 3^k-Raum), erst bei k ∝ n exponentiell; **der Kipppunkt trennt Trainierbarkeit von Simulierbarkeit nicht, weil beide im 3^k-Raum leben** — das wäre selbst die Pointe („Trainierbarkeit und Simulierbarkeit kippen gemeinsam bei k ≈ O(log n)“).
- **Quantenstufe:** Q1.

### I-4 [EIGEN] Kopplung Rang der quadratischen Form ↔ Verschränkung ↔ Simulierbarkeit
- **Frage:** Rang der GF(2)-Form (trägt 52–78 % des Maskengewinns) ist bei Graphzuständen/CZ-Masken zugleich ein Verschränkungsmaß (Cut-Rang/Schmidt-Rang). Korreliert der Leistungsgewinn mit Verschränkung der Maske? Ein Treffer würde „Verschränkung ist hier nützliche Quantenressource“ stützen, ein Nicht-Treffer das Gegenteil (Bowles: oft egal).
- **Neuheit:** Verbindung „Rang ↔ Verschränkung“ für Stabilizer/Graphzustände ist Standardwissen [?]; Verbindung zu **Trainingsgewinn eines QML-Modells** nicht gefunden. Zufällig gefunden: Titel „Entanglement geometry separates circuit cutting, classical hardness, and trainability“ (arXiv 2607.17872, nur Titel, nicht gelesen) — vor Etappe 2 prüfen.
- **Test:** vorhandene I-036-Daten umrechnen (Rang vs Entropie je Schnitt), < 1 h. **Kontrolle:** zufällige Formen gleichen Rangs mit unterschiedlicher Schnittstruktur. Q1.

### I-5 [EIGEN] Holevo/Skalierung-Verbindung: Magie-Budget gegen Daten-Ladekosten (Hinweis)
- Das gesamte „Magie-Budget statt Qubitzahl“-Bild fordert, dass bei großem n der Aufwand der **Eingabe** nicht dominiert. Frage: bei welchem n wird klassische Amplituden-Vorbereitung teurer als der gesamte Rest? Kein Rechentest nötig, Kostenmodell (Papier). Neuheit: nicht gesucht [?]. Nur als Argumentationsteil für Etappe 2.

## 5. Offene Fragen für Etappe 2

1. Volltexte lesen (Zugang zu arXiv fehlt): Cerezo 2025 (genaue Aussage zu „Quantenvorlaufphase“, Warm-Starts, Zulässigkeit von Magie), Gil-Fuster 2025 (gibt es einen Ausweg, „variable partial dequantization“? [?]), Wang/Du/Tao 2026 (ist DSE auf Clifford-Gerüst + k Winkel anwendbar?), arXiv 2403.04844 (MBL-HEA wirklich nicht simulierbar? Wurde es widerlegt?).
2. Gibt es Arbeiten, die zeigen, dass Clifford+k Rotationen mit linearer Auslese ein **lernbares** klassisches Surrogat haben (Leone/Oliviero/Hamma erweitert)? Das würde I-1 theoretisch absichern.
3. Antwort auf „Gegenbeispiele: trainierbar, nicht simulierbar, nützlich?“ systematisch (Suche nach 2025/26 Preprints; Stichworte „provable quantum advantage in learning with trainable circuits“, „generative quantum advantage“, „quantum reservoir“). Bis dahin: kein anerkannter Fall.
4. Existiert ein sauberes Argument, dass Born-Auslese 2^n mit linearem Kopf bei großem n **nicht** classical-surrogate-bar ist (Tang-/SQ-Zugang gegen Cotler et al.)?
5. Bei n>8: skaliert k_needed (M50) mit n? Verknüpfen mit I-3; frühere C-skala-Befunde waren Ratenartefakte (UEBERGABE §4).
6. Wie die wesentlichen Aussagen für eine Jugend-forscht-Arbeit knapp und fachlich vorsichtig formulieren (eigene Textvorschläge Abschnitt 3.6).

## 6. Quellenliste

- McClean et al. 2018: https://research.google/pubs/barren-plateaus-in-quantum-neural-network-training-landscapes/
- Cerezo et al. 2021 (kostenabhängig): https://www.nature.com/articles/s41467-021-21728-w
- Ragone et al. 2024: https://www.nature.com/articles/s41467-024-49909-3 ; Fontana et al. 2024: https://www.nature.com/articles/s41467-024-49910-w
- Larocca et al. 2025 Übersicht: https://www.nature.com/articles/s42254-025-00813-9 (arXiv 2405.00781)
- Cerezo et al. 2025 (Simulierbarkeit): https://www.nature.com/articles/s41467-025-63099-6 ; arXiv https://arxiv.org/pdf/2312.09121
- Anschuetz & Kiani 2022: https://www.nature.com/articles/s41467-022-35364-5
- Hardware-efficient ansatz without BP in any depth: https://arxiv.org/pdf/2403.04844
- Bermejo et al. 2026 (QCNN): https://journals.aps.org/prxquantum/abstract/10.1103/8qt9-72ts
- Gil-Fuster et al. 2025: https://arxiv.org/pdf/2406.07072
- Wang/Du/Tao 2026: https://arxiv.org/abs/2608.02325
- g-sim: https://pennylane.ai/demos/tutorial_liesim ; arXiv 2308.01432
- LOWESA: https://arxiv.org/abs/2308.09109 ; Pauli-Propagation: https://arxiv.org/abs/2409.01706 , https://journals.aps.org/prxquantum/abstract/10.1103/6vd7-l9bn
- Clifford-Störungstheorie: https://arxiv.org/abs/2306.04797 ; Bravyi/Gosset u. a.: https://arxiv.org/pdf/1808.00128 ; Shallow Magic Depth https://arxiv.org/pdf/2409.13809
- CAFQA: https://arxiv.org/abs/2202.12924
- Heyraud et al. 2023: https://doi.org/10.1103/PRXQuantum.4.040335 ; Zhang/Zhu/Wang 2024: https://arxiv.org/abs/2406.11805
- Leone/Oliviero/Hamma, t-doped learning: https://quantum-journal.org/papers/q-2024-05-27-1361/
- Tang-Familie: https://arxiv.org/abs/1910.06151 ; Cotler/Huang/McClean: https://arxiv.org/abs/2112.00811
- Schreiber/Eisert/Meyer: https://arxiv.org/abs/2206.11740 ; Shin/Teo/Jeong: https://arxiv.org/abs/2307.06937
- Schuld & Killoran: https://arxiv.org/abs/2203.01340 ; Bowles et al.: https://arxiv.org/pdf/2403.07059
- Huang et al. 2022 (Science): https://www.science.org/doi/10.1126/science.abn7293
- Entanglement geometry (nur Titel, ungeprüft): https://arxiv.org/pdf/2607.17872
