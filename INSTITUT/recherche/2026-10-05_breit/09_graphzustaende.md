# 09 — Graphzustände, quadratische Formen über GF(2), MBQC und der QIRE-Rangbefund

Stand 05.10.2026. Autor: Recherche-Agent (Sonnet). Alles Unbelegte ist [EIGEN] oder [?].
**Einschränkung vorweg:** Das WebSearch-Budget der Sitzung war nach nur 6 eigenen Anfragen erschöpft
(Zähler 200/200, vermutlich von parallelen Agenten verbraucht). Das Ziel „≥ 15 Anfragen" wurde
verfehlt. Alle Literaturangaben unten stammen aus Such-Snippets (nur Auszug, nichts per WebFetch
gelesen); Standardwissen (Bent-Funktionen, Reed-Muller) ist als [Lehrbuchwissen] markiert, nicht
frisch belegt. Die Neuheitsprüfung ist daher **unvollständig** (siehe §4).

## 1. Kurzfassung
- Der Rang der alternierenden Form B = U+Uᵀ einer quadratischen Form f(x)=xᵀUx über GF(2) ist
  genau der Cut-Rank-/Schmidt-Rang-Typ-Ausdruck der Graphzustands-Theorie (Hein/Eisert/Briegel):
  Die Maske der QIRE-Zelle ist ein Graphzustand (CZ-Anteil) in Phasenform, ihr Rang ist die
  Verschränkung über den Schnitt (Block | Rest).
- **Walsh-Flachheit ist durch den Rang exakt bestimmt** (Rechnung, §3): Rang 2h ⇒ |W| = 2^(n−h)
  auf 2^(2h) Stellen, sonst 0. Rang n = Bent = flaches Spektrum. Das ist keine Vermutung, sondern
  eine Identität [Lehrbuchwissen; im Test bestätigt].
- „Höherer Rang = bessere Merkmale" ist also in der Sprache der Spektren als **„breiteres,
  flacheres Spektrum = jedes Ausgangs-Merkmal mischt mehr Pixel"** erklärbar. Die
  Dekorrelations-Hypothese (flaches Spektrum ⇒ unkorrelierte Merkmale) wurde im Schnelltest
  **nicht gestützt**: die mittlere Merkmals-Korrelation steigt mit dem Rang leicht.
- Schnelltest (sklearn-Digits, 6 Bit, zufällige Formen, linearer Kopf): Genauigkeit steigt
  monoton mit dem Rang (0,764 / 0,837 / 0,856 für Rang 2/4/6). Ein Beleg für die Richtung, kein
  Beleg für den Mechanismus und nicht MNIST.
- Verwandte QML-Literatur (Graphzustand-Merkmalsabbildungen, MBQC-QML) gefunden; **keine** Arbeit
  gefunden, die Rang/Bent einer GF(2)-Phasenmaske als Merkmals-Qualitätsmaß für ein
  Walsh/Clifford-Modell benutzt (Suche dünn, siehe Vorbehalt).
- Vielversprechendste eigene Idee: **Bent-Masken als feste, parameterfreie Maximalrang-Formen**
  (nicht „Rang 6 von k"-Zufall, sondern gezielt volle Rang-Blöcke) als Ersatz für die Suche.

## 2. Stand der Forschung (nur Auszüge, Quelle jeweils angegeben)
**Graphzustände und Verschränkung.** Hein, Eisert, Briegel, „Multi-party entanglement in graph
states", arXiv:quant-ph/0307130 — Schmidt-Rang bzw. Schmidt-Maß werden für jede Bipartition A|B
rein graphentheoretisch über die Adjazenzmatrix Γ ausgedrückt: Schmidt-Rang = Rang über GF(2) des
Teilblocks Γ_AB. (Aus dem Snippet: „For any partition A, the Schmidt rank is … formulated in purely
graph theoretic terms"; die Formel rank_GF(2)(Γ_AB) ist [Lehrbuchwissen], nicht aus dem Snippet.)
https://www.researchgate.net/publication/2190789_Multi-party_entanglement_in_graph_states ;
Schmidt-Maß: https://arxiv.org/pdf/quant-ph/0007081 ; 8-Qubit-Klassifikation:
https://arxiv.org/pdf/0812.4625

**Clifford als quadratische Formen.** Dehaene & De Moor, PRA 68, 042318 (2003),
arXiv:quant-ph/0304125: Die n-Qubit-Clifford-Gruppe ist isomorph zu einer Gruppe aus
(2n+1)×(2n+1)-Binärmatrizen plus binären quadratischen Formen; Koeffizienten von Stabilizerzuständen
und Clifford-Operationen in der Standardbasis sind durch quadratische Formen beschreibbar.
https://arxiv.org/pdf/quant-ph/0304125 (nur Auszug). Das ist die theoretische Grundlage der
Bruhat-Form aus I-036.

**Lokale Komplementierung / LC-Äquivalenz.** Van den Nest, Dehaene, De Moor (PRA 69, 022316, 2004):
LU-Äquivalenz von Graphzuständen = LC-Äquivalenz (Folge lokaler Komplementierungen), im Stabilizer-
Fall bewiesen; Gegenbeispiel LU≠LC erst bei 27 Qubits (Ji et al.; Tsimakuridze). [Alles aus dem
Gedächtnis, Anfrage wurde vom Budget abgelehnt → **[?] nicht frisch belegt**.] Für ≤8 Qubit:
Klassifikation der Graphzustände in LC-Klassen (Hein et al.; 8 Qubit: arXiv:0812.4625, Snippet).

**Bent-Funktionen / Walsh / Reed-Muller.** [Lehrbuchwissen, nicht frisch belegt] Eine Boolesche
Funktion in n (gerade) Variablen ist bent ⇔ ihr Walsh-Spektrum ist flach (|W|=2^(n/2) überall).
Quadratische Funktion mit alternierender Form vom Rang 2h hat Walsh-Spektrum {0, ±2^(n−h)} auf
2^(2h) Stellen (Dickson-Normalform). Bent ⇔ B nicht entartet. Kosets von RM(1,n) in RM(2,n)
werden durch den Rang der Form klassifiziert (MacWilliams/Sloane); der Anteil nicht
entarteter alternierender n×n-Matrizen für n=8 ist ≈ 0,42 (stimmt mit eigenem Zählexperiment 0,43
überein). Stabilizerzustände auf n Qubits haben die Form 2^(-r/2) Σ_{x∈A} i^{l(x)}(−1)^{q(x)}|Rx+t⟩
(Dehaene/De Moor; Van den Nest) — Rang von q|_A bestimmt die Verschränkung.

**MBQC.** Raussendorf & Briegel, „A one-way quantum computer", PRL 86, 5188 (2001);
Raussendorf, Browne, Briegel, PRA 68, 022312 (2003): https://link.aps.org/doi/10.1103/PhysRevA.68.022312
(Snippet). Kernpunkt: Ressourcenzustand = Clusterzustand (Graphzustand), Berechnung durch adaptive
Einzel-Qubit-Messungen; Clifford-Anteil (Pauli-Messungen) klassisch simulierbar, Universalität
erst durch Nicht-Pauli-Messwinkel — dasselbe „0 T / wenige Magiewinkel"-Bild wie in QIRE.

**QML mit Graphzuständen / MBQC (Snippets):**
- „Measurement-based quantum machine learning", arXiv:2405.08319 (auch PRR/PRA-Version
  https://link.aps.org/doi/10.1103/2snk-m8c6): MBQC-Neuronen („MuTA"), „bias engineering,
  monotonic expressivity, tunable entanglement", Quantenzustandsklassifikator, Quantenkernel.
  https://arxiv.org/abs/2405.08319 (nur Auszug; „Majumder" als Autor **[?] nicht bestätigt**).
- Measurement-based quantum CNN, arXiv:2412.08207: Deep Learning durch Messung eines
  Clusterzustands statt gestapelter Lagen. https://arxiv.org/pdf/2412.08207 (nur Auszug).
- Graphzustand-Klassifikatoren: „classical large-scale samples and their features can be mapped
  to entangled graph states" (Snippet, Quelle: Boosted Binary Quantum Classifier via Graphical
  Kernel, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296799/).
- Graph-Daten-Kernel (andere Bedeutung von „Graph"): Pasqal Quantum Evolution Kernel;
  https://link.aps.org/doi/10.1103/PhysRevA.107.042615 (nicht das Gleiche wie Graphzustand als
  *Modellstruktur*).
- „A Resource Efficient Quantum Kernel", arXiv:2507.03689 (Snippet, nicht gelesen) — Titel passt
  zu Effizienz, Inhalt **[?]**.
- Clifford-Kernel / Stabilizer-Merkmalsabbildungen: Die Suche lieferte **nichts Einschlägiges**.

**Klassische Verwandte [?, nicht gesucht]:** Walsh-Hadamard-Schichten und Butterfly-/Fastfood-
Transformationen mit diagonalen Vorzeichenmatrizen (Fastfood, SORF, „structured orthogonal random
features") nutzen H·D mit zufälligen ±1-Diagonalen — das ist exakt „Vorzeichenmaske × WHT", aber
mit *unabhängigen* Vorzeichen, nicht quadratisch-strukturierten. Quellenangabe fehlt (Budget).

## 3. Was das für QIRE bedeutet (konkret, ehrlich)
### 3.1 Dictionary
| QIRE-Objekt | Graph-/Codierungs-Sprache |
|---|---|
| Vorzeichenmaske s(x)=(−1)^{q(x)}, q quadratisch | Graphzustand-Phase (CZ-Kanten = Einträge U_ij, S = Diagonale) |
| Rang der Form B=U+Uᵀ | Cut-Rank / Schmidt-Rang über den Schnitt (Block-Qubits | Rest) bzw. Rang des Graphen in Phasenform |
| Walsh-Transform nach der Maske | Hadamard-Schicht = Messung in X-Basis, „Rest" der MBQC-Kette |
| Rang 2h ⇒ Walsh-Träger 2^(2h) | Faltung des Pixelspektrums mit einem Maskenspektrum der Breite 2^(2h) |
| Bent (Rang = k) | maximal verschränkter Block, flaches Spektrum |
| Rang-Dosis (I-036-v3) | Anzahl Verschränkungs-Bits der Phase |

Wichtig: In der **Blockform** (Bruhat, Blockdimension k = 4…8) kann der Rang höchstens k (gerade)
sein; „Rang 6" in I-036-v3 ist daher bei k=6 bereits **bent** auf dem Block. Ränge MNIST
{2:4, 4:6, 6:10}, Fashion {2:1, 4:5, 6:14} — die Mehrheit der gefundenen Formen ist also
schon maximalrangig bzw. nahe daran. Das ist konsistent damit, dass die Suche in Richtung
„flaches Spektrum" gedrängt wird (oder dass 43 % der Zufallsformen sowieso Maximalrang haben:
Basisrate beachten! [EIGEN]).

### 3.2 Warum Rang ⇒ bessere Merkmale? Drei Erklärungskandidaten
1. **Spreizung (gestützt):** Ausgabe-Merkmal j = |Σ_x (−1)^{j·x+q(x)} x_x|². Bei Rang 0 (q=0,
   reiner Walsh) ist jedes Merkmal ein Parity-Summenmerkmal — I-036 zeigt, dass das global
   schlechtest ist (walsh_fest < Identität). Mit Maske vom Rang 2h werden die Parity-Funktionen
   durch Chirp-artige Funktionen ersetzt (Träger 2^(2h)); bent ⇒ jede Basisfunktion hat
   überall gleichen Betrag der Korrelation mit allen Parities (maximale Entfernung von jeder
   affinen Funktion, Nichtlinearität 2^(n−1)−2^(n/2−1)). Merkmale sind „weit weg von
   Parity" = [EIGEN, als Deutung] weniger redundant zu linearen Pixel-Merkmalen.
2. **Dekorrelation (NICHT gestützt im Test):** mittlere |Korrelation| der 64 Ausgabemerkmale
   (Digits, 6 Bit): Rang 2: 0,102; Rang 4: 0,124; Rang 6: 0,128 — leicht *steigend*.
   Die Pixelkorrelation der Daten dominiert; flaches Spektrum der *Maske* ≠ flaches Spektrum
   der *Daten*. Hypothese „flach = dekorreliert" ist damit auf diesem Datensatz zu verwerfen
   (schwache Evidenz: ein Datensatz, 6 Bit).
3. **Zufalls-Merkmale-Effekt [EIGEN, ungeprüft]:** Hohe Rang ≈ näher an Fastfood-artigen
   pseudozufälligen orthogonalen Projektionen, die für lineare Köpfe bekanntlich gut sind
   (Quelle fehlt, [?]).

### 3.3 Schnelltests (Scratchpad, ≈ 2 s und ≈ 2 min)
Dateien: `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/bent.py`,
`.../dig.py` (Skripte, nicht im Repo).
**Test A (8 Bit, 4000 zufällige Formen, f=xᵀUx):**
| Rang | Anzahl | max|W|/16 | Spektrum-Entropie | Träger |
|---|---|---|---|---|
| 4 | 90 | 4 | 4 bit | 16 |
| 6 | 2188 | 2 | 6 bit | 64 |
| 8 | 1722 | 1 | 8 bit | 256 |
(Rang 0/2 trat nicht auf: unter 4000 Zufallsformen sehr selten.) Befund: Walsh-Flachheit ist
**deterministisch** der Rang (Spektralentropie = Rang bit; max = 2^(h)... exakt wie Theorie),
zufällige Formen haben ≈ 43 % Bent-Anteil (1722/4000).
**Test B (sklearn-Digits 8×8=6 Bit, normierte Eingaben, Merkmale |WHT(s·x)|², LogReg C=10, 150
Zufallsformen, Test 40 %):**
| Rang | n | Genauigkeit (Mittel ± SD) | mittl. |Korr| der Merkmale |
|---|---|---|---|
| 2 | 6 | 0,764 ± 0,021 | 0,102 |
| 4 | 88 | 0,837 ± 0,043 | 0,124 |
| 6 | 56 | 0,856 ± 0,023 | 0,128 |
Rang 0 nicht dabei (Referenz fehlt; aus I-036 weiß man: global schlechter). **Grenzen:** 1 Datensatz,
kleines Digits-Bild, 1 Zuteilung, Block = ganzes Bild (kein Block-Aufbau wie QIRE), keine
Mehrfach-Splits/Konfidenzintervalle, Rang konfundiert mit konkreter Form. Es ist ein *Hinweis*
auf Monotonie, kein Ergebnis für MNIST. Rang 6 vs. 4: Differenz 1,9 pp bei SD 2,3–4,3 pp — nicht
belastbar.

### 3.4 Was das für die Gesamtstory bedeutet
- Die Deutung in I-036-v3 („mehr Verschränkungsphase → bessere Merkmale") ist theoretisch
  sauber anschlussfähig (Cut-Rank), aber **der Rang ist bei Blockdimension k nach oben bei k
  gedeckelt**; die Seeds mit Rang 6 bei k=6 sind schlicht Bent-Blöcke. Rang-Gewinn = Gewinn
  *Bent vs. nicht-Bent*. Das ist eine **schärfere Formulierung**, die in der Auswertung
  noch nicht steht.
- Gottesman-Knill bleibt: kein Quantenvorteil, die Verschränkung ist hier nur ein
  Strukturmaß der Modellklasse. Ehrlich so aufschreiben.
- „Übertragbarkeit der Form zwischen Seeds" (v3) ist mit LC-Äquivalenz natürlich zu erklären
  [EIGEN]: Formen sind wahrscheinlich nur bis auf Basiswechsel P in GL(k,2) und LC-Klasse
  verschieden. Wenn so, ist die „Aufgabenform" ein Element einer *Klasse* — mehr dazu in 4.3.

## 4. Neue Ideen / Forschungsfragen
**Neuheitsprüfung generell:** Gesucht (6 Anfragen): Graphzustand-Kernel, MBQC-QML, Clifford-
Kernel/Stabilizer-Merkmalsabbildung. Gefunden: Graphzustand-Klassifikator (Datenzustände →
Graphzustände), MBQC-Neuronen/CNN (Messbasierte Netzwerke), aber nichts zu
*Rang/Bent einer GF(2)-Phasenmaske als Merkmalsqualität*. Bent/Reed-Muller+QML-Anfragen
wurden durch das Budget abgewiesen → **Neuheit nicht bestätigt, nur „nicht gefunden bei
dünner Suche"**. Vor Veröffentlichung/„neu"-Behauptung in Etappe 2 nachholen.

### 4.1 [EIGEN] Bent-Masken als feste, parameterfreie Maximalrang-Form
- **Frage:** Ersetzt eine *feste* Bent-Maske (z. B. Maiorana-McFarland q(x,y)=x·y bzw. innere
  Produktform xᵀ y auf den k/2+k/2 Blockbits) die gesuchte Maske ohne Genauigkeitsverlust? 0
  Suchparameter für die Maske, nur die Blockwahl V bleibt.
- **Warum neu:** Bent-Funktionen sind bekannt (Kryptographie), Fastfood nutzt H·D, aber
  deterministische Bent-Phase vor WHT als „Merkmalsschicht" nicht gefunden (Suche dünn).
  Verwandt: Quadratische Bent = inneres Produkt = CZ-Paarung benachbarter Qubits = **Linien-/
  Paargraph-Zustand**; trivialer Fall.
- **Billigster Test:** In bestehender Pipeline (`maske_eigenschaft.py`): Arm `bent_fest`
  (V_s fest aus Suche, B = Standard-Bent-Paarung, k gerade) gegen `suche`, `zufall_gleicher_rang`,
  `ohne_maske` mit demselben L-BFGS-Kopf. Falls Rang bereits max: Erwartung `bent_fest ≈
  zufall_gleicher_rang` (+1,2 pp unter Suche in v3 MNIST). Laufzeit ≈ 1 Arm × 20 Seeds ×
  2 Datensätze ≈ 10–15 min auf Maltes PC (kein Suchlauf mehr nötig, Kopf-Anpassung wie in v3).
- **Kontrolle:** Zufall mit *gleichem* Rang (v3-Arm), Zufalls-Rang-<k-Form (Dosis), Identität,
  Fastfood-artige unabhängige ±1-Vorzeichen (Test, ob die *quadratische Struktur* oder nur
  „gemischte Vorzeichen" trägt).
- **Quantenstufe:** Q1 (Clifford, klassisch simulierbar). Verallgemeinerung auf T-Gatter Q2.

### 4.2 [EIGEN] Cut-Rank als Vorhersagegröße je Block (statt nur Gesamt-Rang)
- **Frage:** Die Form wirkt auf k Block-Qubits; der Graph der Maske hat Teil-Schnitte. Sagt der
  *minimale/mittlere Cut-Rank über Block-Bipartitionen* (nicht nur der Gesamtrang) den
  Maskengewinn besser vor als der Rang? Vorgabe: Spearman im v3-Datensatz (Rang ↔ Gewinn
  0,79 / 0,49).
- **Neuheit:** Nicht gefunden (nur Hein et al. als Theorie; keine Anwendung als ML-Prädiktor).
- **Test:** Nachrechnen mit den gespeicherten v3-Formen (JSON, ca. 40 Seeds), nur Numpy-GF(2)-
  Rang je Bipartition, keine neue Suche: < 1 min. Kontrolle: Gesamtrang, Anzahl Kanten,
  Spektralentropie, Nichtlinearität — alles partielle Spearman gegen Gewinn gleichzeitig,
  damit man sieht, ob Cut-Rank *zusätzliche* Information hat (Konfundierung Rang/Kanten!).
- **Quantenstufe:** Q1.

### 4.3 [EIGEN] LC-Äquivalenzklassen als Suchraum-Reduktion
- **Frage:** Wenn (V,q) und (V,q') zu derselben LC-Klasse gehören, haben sie dann gleiche
  Genauigkeit? Wenn nein, ist LC keine Invarianz des Merkmalsraums (weil die Lokalität = Pixel-
  Basis wichtig ist!). Wenn ja, schrumpft der Suchraum um die Klassengröße (bei k=6: 2^15 Graphen
  → einige zig LC-Klassen, [Zahl aus Gedächtnis, [?]]).
- **Warnung [EIGEN]:** LC wirkt als lokale Clifford-Operation *je Qubit* — ändert die
  Qubit-Basis, aber nicht die Pixel-zu-Qubit-Zuordnung; Born-Messung in Z-Basis ist *nicht*
  LC-invariant (Hadamard-Anteil vertauscht Messbasen). Daher ist Invarianz unwahrscheinlich;
  der *Rang* ist LC-invariant, die Genauigkeit i. A. nicht. Das ist gerade ein sauberer Test,
  ob der Rang *oder die Form* trägt (passt zu RANG_UND_FORM).
- **Test:** Für 30 Formen auf k=4 (Rang 4, eine Hand voll Graphen): alle LC-Orbits per
  Hand/Code (lokale Komplementierung = Update B ← B + (Nachbarn von v)(Nachbarn von v)ᵀ),
  Genauigkeit je Orbit-Mitglied messen, Varianz in/zwischen Orbits. ≈ 10 min mit
  vorhandenem Merkmals-Code.
- **Kontrolle:** Zufall mit gleichem Rang (kein LC-Bezug).
- **Quantenstufe:** Q1.

### 4.4 [EIGEN] „Aufgabenform" als Klasse: Übertragbarkeit über Basiswechsel
- **Frage:** Fremde Maske (v3: +2,19 pp ggü. ohne_maske, nur 0,29 pp unter eigener) — ist die
  Übertragung auch dann gut, wenn V_t nur nach GL(k,2)-Konjugation passt? (Normalform nach
  Dickson: B ≅ Σ e_{2i-1}∧e_{2i}.) Dann kann jede Maske der Aufgabe durch (Rang, Normalform,
  einmal gesuchtes P) beschrieben werden.
- **Neuheit:** nicht gesucht (Budget).
- **Test:** Auf gespeicherten v3-Formen: Dickson-Normalform + Basiswechsel P berechnen und
  prüfen, ob die Formen verschiedener Seeds bis auf P gleich sind. Reine Rechnung, Minuten.
- **Quantenstufe:** Q1.

### 4.5 Zusammenhang mit „~100 Parameter insgesamt" (Ziel 1)
Eine feste Bent-Maske (4.1) senkt Such-/Formparameter auf 0, übrig bleiben Blockwahl und Kopf. Der
Kopf (2.570 Parameter) bleibt das Problem, nicht die Zelle — das ändert nichts am Ziel 1,
aber Maske-frei verkleinert die *Suche* (Zeit), nicht den Speicher. [EIGEN] Ehrliche Einschätzung.

## 5. Offene Fragen für Etappe 2
1. Literatur (Budget fehlte): Fastfood/SORF/„Hadamard-Schicht"-Arbeiten; Bent in QML; Clifford-
   Kernel; Graphzustand-basierte Modellfamilien; Quanten-Reservoir mit Clifford. Vor Neuheitsbehauptung
   prüfen.
2. Original-Texte (nicht nur Snippets) von Hein et al. (Cut-Rank-Formel), Dehaene/De Moor
   (Normalform der Rang-Form), Van den Nest (LC/Stabilizer-Struktur) lesen — für die schriftliche
   Arbeit exakte Aussagen und Gleichungsnummern.
3. Basisrate: Wie viel des „Rang 6 = Bent bei k=6" ist durch die Suche erzeugt vs. durch
   Zufall (43 % für k=8, für k=6 selbst zu rechnen: Anteil nicht entarteter alternierender
   6×6-Matrizen = 28/…?) [?] — Kontrolle gegen Suchbias.
4. Dekorrelations-Hypothese sauber testen: Gleichgewicht Rang/Datenspektrum; evtl.
   „Whitening-ähnlicher" Effekt durch Datenkovarianz, nicht durch Maske.
5. MBQC-Ansicht: QIRE-Zelle als Teil eines Clusterzustand-Ablaufs? Ry-Winkel = Messwinkel,
   adaptive Korrekturen = Pauli-Frame — würde „wenige Magiewinkel" mit MBQC-T-Tiefe
   verbinden. Nur Idee [EIGEN, ungeprüft].
6. Skalierung („unendlich skalierbar?"): Cut-Rank-Verschränkung wächst bei Maximalrang linear in k
   (Flächengesetz-Maximum = volle Page-artige Verschränkung) — das ist *schlecht* für MPS-
   Simulierbarkeit, aber Stabilizer-Simulation bleibt poly. Beziehung zu Barren Plateaus
   nur Vermutung [?].

## 6. Quellenliste (alle nur als Snippet/Titel gesehen, nichts volltextlich gelesen)
- Hein, Eisert, Briegel, quant-ph/0307130 — https://www.researchgate.net/publication/2190789_Multi-party_entanglement_in_graph_states
- Hein, Eisert, Briegel Schmidt-Maß — https://arxiv.org/pdf/quant-ph/0007081
- Cabello et al., 8-Qubit-Graphzustände — https://arxiv.org/pdf/0812.4625
- Dehaene, De Moor 2003 — https://arxiv.org/pdf/quant-ph/0304125 ; https://www.alphaxiv.org/abs/quant-ph/0304125
- Raussendorf, Browne, Briegel 2003 — https://link.aps.org/doi/10.1103/PhysRevA.68.022312
- Measurement-based quantum machine learning — https://arxiv.org/abs/2405.08319 ; https://link.aps.org/doi/10.1103/2snk-m8c6
- Measurement-based QCNN — https://arxiv.org/pdf/2412.08207
- Boosted Binary Quantum Classifier via Graphical Kernel — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296799/
- Quantum feature maps for graph ML (Neutralatom) — https://link.aps.org/doi/10.1103/PhysRevA.107.042615
- A Resource Efficient Quantum Kernel — https://arxiv.org/pdf/2507.03689
- Pasqal Quantum Evolution Kernel — https://www.pasqal.com/blog/pasqal-released-quantum-evolution-kernel-open-source-library-for-quantum-graph-machine-learning/
- Eigene Tests: `bent.py`, `dig.py` im Scratchpad (siehe 3.3).
- Aus dem Gedächtnis, **nicht belegt** [?]: Van den Nest/Dehaene/De Moor 2004 (LC), MacWilliams/Sloane
  (RM), Fastfood (Le/Sarlós/Smola), Raussendorf/Briegel 2001.
