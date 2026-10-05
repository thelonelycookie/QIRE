# 04 — Magie (Nicht-Stabilizer) und Clifford+T, speziell in Quanten-ML

*Stand 05.10.2026. Recherche per WebSearch (ca. 22 Anfragen). arxiv.org, alphaxiv und
arxiv-HTML sind per Proxy gesperrt: **alle arXiv-Angaben unten sind „nur Auszug“ (Suchsnippet),
keine Volltexte.** Zahlen aus Snippets sind als solche gekennzeichnet. Eigenes steht als [EIGEN],
Unsicheres als [?].*

## 1. Kurzfassung
- Die Ressourcentheorie (SRE, Robustheit, Mana, Stabilizer-Rang) ist reif. Die **Kosten pro
  T-Gatter** sind durch Gidneys Magic-State-Cultivation (2024) stark gefallen. Magie wird
  dadurch billiger, bleibt aber die teuerste Gatterart. Das Ziel „wenig Magie“ bleibt sinnvoll.
- **Rz-Synthese kostet ~3·log2(1/ε) T** (Ross–Selinger). Maltes Messwert „T ≈ 31·k bei ε=10⁻³“
  passt dazu (3·log2(1000) ≈ 30). Die Synthesekosten lassen sich nach Auszug noch etwa um
  den Faktor 2 senken (Kliuchnikov et al. 2023, gemischte Kanäle). Das sind Kosten **je Winkel,
  nicht je Aufgabe**.
- **Neu gegenüber den bisherigen Prüfungen:** In der Literatur wächst die Magie als Ressource
  für QML (Reservoir-Computing: „Magie ist die unverzichtbare Ressource“, 2026). Das ist ein
  *Hinweis in Gegenrichtung* zu Maltes Befund, dass MNIST mit 0 T auskommt.
- Der **Begriff „Magie-Konzentration einer Aufgabe (M50/M90)“ wurde nicht gefunden.** Es gibt
  Nachbarbegriffe (t-doped-Zustände, T-Budget in VQE, Magietiefe, Magie-Leistung im QRC), aber
  keine Kennzahl „Anteil des Nutzens, der in k nicht-Clifford-Winkeln steckt“.
- **Ehrliche Warnung:** Mit k ≈ 4–30 Magiewinkeln auf 8 Qubits ist die Zelle nach
  Bravyi–Gosset (2^(0,23·t)) **klassisch trivial simulierbar**. „Magie-Budget“ ist hier ein
  Hardware-/Ressourcenargument (T-Zahl), **kein** Beleg für Quantenvorteil.
- Drei eigene Ideen mit billigem PC-Test (Abschnitt 4): (E1) Skalierungsgesetz **M50 gegen
  Klassenzahl**, (E2) **Pareto-Front aus Genauigkeit gegen T-Gatter inklusive
  Winkelpräzision**, (E3) **Tausch Rang gegen Magie**.

## 2. Stand der Forschung

### 2.1 Ressourcentheorie
- **Stabilizer-Rényi-Entropie (SRE)**: Leone, Oliviero, Hamma, PRL 128, 050402 (2022). Nach
  Auszug ein „gut berechenbares Maß“ ohne Minimierung, ein Monoton der Ressourcentheorie
  (spätere Arbeit: „Stabilizer entropies are monotones for magic-state resource theory“).
  https://link.aps.org/doi/10.1103/PhysRevLett.128.050402
  · https://www.researchgate.net/publication/385054423_Stabilizer_entropies_are_monotones_for_magic-state_resource_theory
- **Robustheit der Magie / Mana**: Howard & Campbell 2017 (Robustheit; Kosten einer Gottesman-Knill-
  artigen Simulation mit Magiezuständen); Veitch et al. (Mana, Wigner-Negativität);
  Heinrich & Gross 2019 (Symmetrien des Stabilizer-Polytops). Nach Auszug sind
  „stabilizer rank, min-relative entropy of magic, log-free robustness“ die üblichen Monotone;
  der Stabilizer-Rang-Simulator läuft in O(2^(0,48·t)) für reine Zustände.
  https://eprints.whiterose.ac.uk/id/eprint/113586/1/Robustness_main_appendix.pdf
  · https://ar5iv.labs.arxiv.org/html/1807.10296
- **Stabilizer-Nullität** ist ein ganzzahliges, additives Monoton und eine untere Schranke
  für die T-Zahl (Auszug aus arXiv 2608.14798, „Stabilizer Statistical Mechanics“).
  https://arxiv.org/pdf/2608.14798
- Messung/Schätzung: SRE per GNN/Random Forest schätzbar (arXiv 2511.23224, 2509.16799);
  schneller exakter Algorithmus XOR-FWHT (arXiv 2512.24685); SRE über Reinheit (PRA 2025).
  **Die Walsh-Hadamard-Struktur der Clifford-Zelle ist genau die Struktur, die diese
  Algorithmen ausnutzen** [EIGEN: Zusammenhang, nicht geprüft].
- **Sättigung der SRE** als Komplexitätsmaß: Quantum 2025, arXiv 2406.04190.
  https://quantum-journal.org/papers/q-2025-07-21-1801/

### 2.2 Simulation Clifford-naher Schaltungen
- **Bravyi–Gosset, PRL 116, 250501 (2016)**: Laufzeit poly(n,m) + 2^(0,23·t)·t³·w⁴ (Näherung
  mit Fehler) bzw. 2^(0,5·t) (exakt) mit t = T-Zahl; Demo: 40 Qubits, ~50 T.
  https://link.aps.org/accepted/10.1103/PhysRevLett.116.250501
  (Auszug; Aaronson–Gottesman 2004 für den reinen Clifford-Fall: bekannt, hier nicht neu gesucht.)
- Clifford-Rekompilierung beschleunigt die Simulation: Quantum 3, 170 (2019).
  https://quantum-journal.org/papers/q-2019-08-05-170/
- **Clifford-Störungstheorie** (Winkel nahe Clifford, Pauli-Pfade abschneiden), arXiv 2306.04797;
  genau die Winkelnähe, die QIRE nach dem Runden hat. https://arxiv.org/abs/2306.04797
- **Magietiefe**: „Classical Simulability of Quantum Circuits with Shallow Magic Depth“,
  PRX Quantum 6, 010337 (2025): *wo* die Magie sitzt (Verteilung über Tiefe), nicht nur wie viel,
  bestimmt die Simulierbarkeit. https://link.aps.org/doi/10.1103/PRXQuantum.6.010337
- Klassische Simulation mit verrauschten Magiezuständen (arXiv 2601.10111).

### 2.3 T-Zählung und Synthese
- **Ross–Selinger (arXiv 1403.2975, „gridsynth“)**: ancillafrei, T-Zahl 3·log2(1/ε) + O(log log(1/ε)),
  mit Faktorisierungsorakel optimal, erwartete Laufzeit polylog(1/ε).
  https://arxiv.org/abs/1403.2975v1
- **Kliuchnikov, Lauter, Minko, Paetznick, Petit**, Quantum 7, 1208 (2023), arXiv 2203.10064:
  Sequenzlänge Faktor 7/9, gemischte (probabilistische) Kanäle sparen nochmals Faktor ~2
  (Auszug). Die genauen Konstanten (z. B. „~1,15·log2(1/ε)“) habe ich **nicht** bestätigt [?].
  https://quantum-journal.org/papers/q-2023-12-18-1208/
- **Katalytische Rz-Rotationen in konstanter T-Tiefe**, Quantum 2026 (nur Titel gesehen).
  https://quantum-journal.org/papers/q-2026-08-13-2191/
- ML-gestützte Synthese: arXiv 2504.15990 (Neuronale Netze für näherungsweise Clifford+T-Synthese).

### 2.4 Was kostet ein T-Gatter heute?
- **Gidney, Shutty, Jones, „Magic state cultivation: growing T states as cheap as CNOT gates“**
  (arXiv 2409.17595, 26.09.2024): Aufzucht eines T-Zustands innerhalb eines
  Oberflächencode-Patches; nach Auszug etwa gleich viele physikalische Gatter wie eine
  Lattice-Surgery-CNOT gleicher Güte, Fehlerraten 10⁻⁴…10⁻⁹, **Raumzeit-Kosten etwa Faktor 10
  unter früheren Verfahren**. https://arxiv.org/html/2409.17595v1 · Folgearbeit zur
  Oberflächencode-Integration: https://arxiv.org/html/2502.01743v1
- **Gidney 2025 (arXiv 2505.15917)**: RSA-2048 mit <1 Mio. verrauschter Qubits in <1 Woche,
  u. a. weil Cultivation weniger Platz für Destillation braucht.
  https://arxiv.org/abs/2505.15917
- Eigenes Fazit [EIGEN, aus den Auszügen]: „T-Gatter ≈ CNOT-Kosten“ gilt **nur für
  fehlerkorrigierte Rechner mit Cultivation**. Für Maltes Ziel (Effizienz heute) ist das
  Argument „wenig Magie“ eher ein Argument für **zukünftige** FT-Rechner. Auf NISQ-Hardware ist
  Rz virtuell und kostenlos (UEBERGABE §0, 29.09.). Dort zählt Magie nur über die Simulierbarkeit.
- Gegenstück „Almost fault-tolerant QML“ (arXiv 2507.18954): nur CNOTs korrigieren, die
  parametrisierten Einqubit-Gatter unkorrigiert lassen; Auszug: von Millionen auf wenige
  Tausend Qubits. Das ist ein **konkurrierender Weg**, das Magie-Problem im QML zu umgehen.
  https://arxiv.org/pdf/2507.18954

### 2.5 Magie in QML (2023–2026)
- **Lernen von t-dotierten Stabilizerzuständen**: Grewal, Iyer, Kretschmer, Liang, Quantum 9,
  1907 (2025) und Einzelkopie-Variante Quantum 8, 1250 (2024): Lernzeit poly(n, 2^t, 1/ε) für
  Zustände aus Clifford + t Nicht-Clifford-Gattern; t = O(log n) ist effizient lernbar.
  → **t ist dort die „Ressource, die die Lernkosten bestimmt“.** Ein Aufgaben-Magiebedarf
  im Sinne von M50 ist **nicht** dasselbe, aber der nächste Verwandte.
  https://quantum-journal.org/papers/q-2025-11-06-1907/ · https://quantum-journal.org/papers/q-2024-02-12-1250/
- **„Learnable yet not simulable: a quantum resource theory of learning models“**
  (arXiv 2608.02325, nur Titel/Snippet): Modelle, die lernbar, aber nicht simulierbar sind,
  trotz hoher Magie und Verschränkung. Gegenstück zur Frage „Magie nötig für Vorteil?“. [?]
- **Optimal quantum (reservoir) learning in proximity to universality** (arXiv 2510.18623,
  PRA 2025/26): Lernleistung ist **am besten nahe Universalität**, wo Verschränkung und
  Nicht-Stabilizer-Anteil „extensiv, aber submaximal“ sind. Das ist ein **Optimum-Befund,
  keine Konzentration.** https://arxiv.org/pdf/2510.18623
- **Quantum magic and non-commutativity as computational resources in QRC** (arXiv 2607.12035):
  bedingte T-Gatter regelbar; mehr Abstand vom Clifford-Punkt gibt bessere Leistung, korreliert
  mit aufgebauter Magie; Magie sei „die unverzichtbare Ressource“.
  https://arxiv.org/html/2607.12035
  → **Dem steht Maltes Befund gegenüber** (MNIST: Clifford-Suche = volles Training). Erklärung
  denkbar [EIGEN]: QRC braucht Magie als *Merkmalserzeuger* (Dynamik), die QIRE-Zelle hat nur
  *eine* Auslese und einen Kopf mit 99 % der Parameter, also kann der Kopf die Magie ersetzen.
- **Role of Nonstabilizerness in Quantum Optimization** (arXiv 2505.17185); **Magic-Informed
  Quantum Architecture Search** (arXiv 2605.03932: GNN schätzt SRE von PQCs zur Architektursuche).
  Beide nur Auszug. Magie als Suchkriterium, **nicht** als Kosten/Nutzen-Größe je Aufgabe.
  https://arxiv.org/pdf/2505.17185 · https://arxiv.org/html/2605.03932v1
- **Gottesman–Knill-Grenze der Ein-Weg-Kommunikation**: „Quantenvorteil bis auf Magie
  zurückverfolgt“ (arXiv 2506.19369) und **„Unconditional quantum magic advantage in shallow
  circuit computation“** (Nat. Commun. 2024, arXiv 2402.12246): Magie nötig für Vorteil bei
  bestimmten Aufgaben. https://arxiv.org/html/2506.19369v2 · https://www.nature.com/articles/s41467-024-54864-0
- Auszug-Aussage (arXiv 2510.18623-Umfeld): „there is no known explicit correspondence between
  nonstabilizer resources and performance of QML algorithms“, also bestätigt die Lücke, in der
  QIRE misst. [Aussage aus Suchzusammenfassung, nicht verifiziert.]
- **Pseudomagie**: Gu, Leone, Ghosh, Eisert, Yelin, Quek, PRL 132, 210602 (2024): Zustände mit
  wenig Magie, die für beschränkte Beobachter von Zuständen mit maximaler Magie ununterscheidbar
  sind. Folge: **Magie ist nicht effizient messbar**. https://arxiv.org/pdf/2308.16228
  Ein Hinweis für Maltes SRE-Fehlschlag (I-002, I-035)? [EIGEN, spekulativ]: nein, Pseudomagie
  betrifft rechenbeschränkte Unterscheidung, nicht die exakt berechnete SRE einer kleinen Zelle.
- **Variationsschaltungen / Training**: „Trainability of Quantum Models Beyond Known Classical
  Simulability“ (arXiv 2507.06344, Linear Clifford Encoder: Expansion um Clifford-Schaltungen,
  gesunde Gradienten); CAFQA (arXiv 2202.12924), Clifford Circuit Initialisation
  (arXiv 2207.01539), Warm Starting mit Near-Clifford (Electronics 12, 347),
  „A fault-tolerant variational quantum algorithm with limited T-depth“ (QST 2023). Dort gibt
  es für VQE einen dokumentierten **Kompromiss Clifford ↔ wenige T-Gatter** (CAFQA: ~99 %
  Genauigkeit rein Clifford, bis 99,9 % der Korrelationsenergie mit wenigen T).
  https://arxiv.org/pdf/2507.06344 · https://arxiv.org/abs/2202.12924
- Vorarbeiten im Repo: `recherche/2026-09-28_neuheit_magie.md` (196 Zeilen) deckt
  CAFQA, NCC-WS, Clifford-Perturbation, Neural-Quantum-States-Magie bereits ab. Dieser Bericht
  ergänzt: t-dotiertes Lernen, QRC-Magie 2026, „Learnable yet not simulable“,
  Magietiefe, Almost-FT-QML.

### 2.6 Neuheit von „Magie-Konzentration einer Aufgabe (M50)“
Gesucht: „magic concentration“, „magic budget“, „magic cost of a learning task“, „minimal
T-count to solve classification“, „nonstabilizerness required for learning task“.
**Ergebnis: Der Begriff und die Kennzahl (kleinstes k, sodass k nicht-Clifford-Winkel von 32
den Anteil G(k) ≥ 50 % bzw. 90 % des Gewinns über Clifford erreichen) kommen in den gefundenen
Quellen nicht vor.** Ähnliche Größen:

| Größe | Quelle | Unterschied zu M50 |
|---|---|---|
| t = T-Zahl als Lernkosten | Grewal et al. 2024/25 | Zustandslernen, kein Klassifikator-Nutzen |
| T-Budget gegen Genauigkeit | CAFQA / limited-T-depth VQE | Energie, nicht Nutzenanteil; keine Konzentrationskurve |
| Stabilizer-Nullität, Robustheit | Ressourcentheorie | Eigenschaft des Zustands, nicht der Aufgabe |
| Magie im QRC gegen Leistung | arXiv 2607.12035 | Korrelation SRE ↔ Leistung, nicht minimales k |
| Magietiefe | PRX Quantum 2025 | Simulierbarkeit, nicht Nutzen |
| Gate-Pruning in PQCs | siehe Vorbericht | Winkel nahe 0 entfernen, nicht Clifford-Gitter |

Einschränkung: nur Titel/Snippets, ich kann nicht ausschließen, dass „Nutzenanteil pro
Nicht-Clifford-Gatter“ unter anderem Namen in Compiler-/VQE-Literatur vorkommt. Besser formulieren
als **„aufgabenbezogene Magie-Konzentrationskurve G(k)“**.

## 3. Was das für QIRE bedeutet (ehrlich)
1. **Konsistent mit der Literatur:** Magie lässt sich in Variationsschaltungen budgetieren, und
   wenige Nicht-Clifford-Gatter reichen oft (VQE: CAFQA). Maltes M50/M90 ist die Übertragung
   auf überwachtes Lernen mit einer Kennzahl. Das ist der **neue** Teil, aber **klein**.
2. **Gegenbefund zur Literatur:** MNIST braucht 0 T (I-005-v2), Fashion ~4 Winkel (≈123 T).
   „Magie unverzichtbar“ (QRC 2026) gilt für Merkmalserzeugung in Dynamik, nicht für diese
   Ein-Schicht-Auslese mit dominantem linearen Kopf. Das **Papier-Argument** lautet: „Magie
   nötig“ ist aufgabe- *und* kopfabhängig. Ein linearer Kopf mit 2.570 Parametern kann Magie
   teilweise ersetzen (Rang-/Kopf-Ersatz) [EIGEN].
3. **Simulierbarkeit:** t ≈ 4–30 Winkel, 8 Qubits: Bravyi–Gosset-Kosten 2^(0,23·30) ≈ 120,
   dazu ist 2^8 = 256 ohnehin klein (State-Vector). Daher: **kein Quantenvorteil-Argument**.
   Das Ressourcenargument gilt nur, wenn man später **n ≫ 8** und fehlerkorrigierte Hardware
   annimmt, und dort gilt das Datenladen-Problem (7.636 T, 7,7× Schaltung, C-kodierung) stärker.
4. **T-Kosten sind 2024/25 gefallen** (Cultivation). Die Rechnung „T ≈ 31·k“ bleibt gültig, aber der
   Wert von T gegen CNOT ist laut Gidney nahe 1. **Konsequenz:** Der relevante Gewinn ist nicht
   „T sparen“, sondern „Lagen sparen“ und „Parameter sparen“. Die Magie-Linie sollte im Text als
   *Struktur-/Ressourcenanalyse* verkauft werden, nicht als Kostenhebel.
5. Synthesekosten ∝ log(1/ε): Der Faktor 31 pro Winkel hängt nur an ε = 10⁻³. Wenn die
   Winkel mit ε ≈ 10⁻¹ ausreichen, kostet ein Winkel nur ~10 T (3·log2(10)). Das hat die Magie-
   Linie bisher **nicht** ausgenutzt (siehe E2).

## 4. Neue Ideen / Forschungsfragen

### E1 [EIGEN] Skalierungsgesetz des Magiebedarfs: M50 gegen Klassenzahl und Qubit-Zahl
- **Frage:** Wächst M50 (bzw. M50 als Anteil an den Winkeln) mit der Klassenzahl C
  (2, 3, 5, 10) und der Qubitzahl n (6, 7, 8) linear, logarithmisch oder gar nicht? Passt
  M50 ≈ a·log2(C) (Entropie-Bild: Kopf muss log2 C Bit unterscheiden)?
- **Neuheitsprüfung:** Suche „magic requirement scaling number of classes“, „T-count of
  classifier vs number of classes“: **nichts gefunden.** Repo: C-skala-v2 (beide HU) hat M50
  gegen Winkelzahl getestet, **nicht** gegen Klassenzahl; I-035 hat die Lücke als Prädiktor
  geprüft (H0). Offen, [?] ob es in Compilerliteratur existiert.
- **Billigster PC-Test:** I-005-Pipeline (diskrete Clifford-Suche) + k Winkel auf
  MNIST-Teilmengen mit C ∈ {2, 3, 5, 10}; M50 aus G(k) für k ∈ {0,1,2,4,8,16,32}; je 8–10 Seeds.
  Pro (C,k) ~3–6 min mit Teildaten → ~1,5–3 h auf 5 Kernen. Vorab **stabile Referenzrate**
  (`gittersuche_stabil`), weil Ratenartefakte C-skala zerstört haben.
- **Kontrolle:** gemischte Labels (nur Rauschen → M50 sollte 0 sein); C = 2 ohne Magie als
  Positivkontrolle der Pipeline; Konfidenzintervalle über Seeds, M50 per Bootstrap.
- **Quantenstufe:** Q1 (Simulation); Q2 erst, wenn ein M50-Gesetz steht (dann 8 Qubits auf IBM).

### E2 [EIGEN] Magie-Budget-Pareto-Front: Genauigkeit gegen T-Gatter, inklusive Winkelpräzision
- **Frage:** Wie sieht die Front (T-Gatter, Genauigkeit) aus, wenn man zwei Hebel setzt:
  Zahl k der Nicht-Clifford-Winkel **und** deren Präzision b (Gitter π/2^b, T-Zahl je Winkel
  ≈ 3·log2(1/ε) bzw. aus gridsynth)? Beobachtung: Das Gitter π/4 (b=2, genau 1 T) ist die
  Zwischenstufe zwischen Clifford (0 T) und Feingitter (~30 T). Gesucht: der „Knick“ der Front.
- **Neuheitsprüfung:** Suche „accuracy versus T-count Pareto quantum classifier“: gefunden nur
  VQE-Kompromisse (CAFQA, limited-T-depth). Als **Benchmark-Idee** („Genauigkeit gegen T-Zahl,
  vergleichbar zwischen QML-Modellen“) **nichts gefunden.** Teilbefund im Repo: I-034 (174 T =
  2,25 %), C-hardware (T≈31k), aber ohne Präzisionsachse.
- **Billigster PC-Test:** vorhandene Winkelsätze (aus I-005-v3 + Top-k-Auswahl) auf Gitter
  π/2, π/4, π/8, π/16 runden (b = 1…4, T-Zahl je Winkel aus Qiskit/gridsynth bei ε passend),
  Genauigkeit nur über Neu-Evaluation (kein Training) → < 20 min; mit Nachtraining des Kopfes
  ~1 h. Ergebnis: Tabelle (k, b, T, Genauigkeit), Front per nicht-dominierter Menge.
- **Kontrolle:** zufällig gewählte Winkel (statt Top-k); Rundung statt Synthese (Fehler ε
  vorab als Rauschen); Vergleichspunkt „Identität“ und „volles Training“.
- **Quantenstufe:** Q1. Q3 (FT-Rechner, Cultivation-Kosten) nur als Rechenbeispiel im Text.

### E3 [EIGEN] Sind Rang der quadratischen Form und Magie austauschbare Ressourcen?
- **Frage:** Gibt es Iso-Genauigkeitskurven in der Ebene (Rang r der GF(2)-Form, Magiewinkel k)?
  Also: gleiche Genauigkeit mit höherem r und kleinerem k, und umgekehrt? Wenn ja, ist die
  Austauschrate r↔k konstant? (Beobachtung: r trägt 52–78 % des Maskengewinns, I-036.)
- **Warum ungewöhnlich:** CZ/S sind **kostenlos in T** (Clifford). „Austausch“ ist daher
  *nicht* T-gegen-T, sondern Clifford-Struktur (CZ-Zahl, Tiefe, Suchaufwand) gegen T. Das
  Kostenmodell braucht Gidneys Befund „T ≈ CNOT“ als Wechselkurs [EIGEN: offen, wie man das
  sinnvoll normiert].
- **Neuheitsprüfung:** Suche „trade-off Clifford rank non-Clifford count learning“, „stabilizer
  rank vs T-count tradeoff classifier“: Stabilizer-Rang↔T-Zahl-Austausch wird in der Simulations-
  literatur behandelt (Bravyi–Gosset: Rang ~ 2^(0,48 t)), aber **nicht als lernbare
  Austauschkurve in einem Klassifikator.** [?] Ob „Rang der Maske“ der gleiche Rang ist wie
  Stabilizer-Rang: **nein**, GF(2)-Rang einer quadratischen Form ≠ Stabilizer-Rang eines
  Zustands. Nicht verwechseln.
- **Billigster PC-Test:** Gitter r ∈ {0,2,4,6,8} × k ∈ {0,2,4,8} auf MNIST + Fashion, je 10 Seeds:
  Form mit festem Rang aus I-036-Übertragungsbefund (gute Form übernehmen, Rang abschneiden),
  k Magiewinkel nach Top-k. 20 Zellen × 5 min ≈ 1,5–2 h. Auswertung: Anpassung
  Genauigkeit ≈ f(a·r + b·k) gegen Modell mit Wechselwirkung (F-Test).
- **Kontrolle:** zufällige Form gleichen Rangs; Seed-Übertragung (I-036-v3) als Gegenprobe.
- **Quantenstufe:** Q1.

### E4 [EIGEN] „Magie-Kopf-Ersatz“: Wie viel Magie ersetzt ein größerer Kopf?
- **Frage:** M50 als Funktion der Kopfgröße (linear 2.570 → versus MLP-Kopf → versus
  100-Parameter-Kopf, Maltes Ziel 1). Hypothese: je schwächer der Kopf, desto höher M50.
- **Neuheit:** nichts gefunden (QRC-Papiere variieren Magie, nicht den Auslesekopf).
- **Test:** I-005-Suche mit Kopf-Varianten, ~2 h; Kontrolle: gleiche Seeds, gleicher Datensatz.
- **Quantenstufe:** Q1. Wichtig für Ziel (1) (~100 Parameter): dort ist der Kopf klein, also
  dürfte M50 **steigen** — zuerst prüfen, ob die 100-Parameter-Variante überhaupt Clifford-fähig ist.

### E5 [EIGEN] Klassische Dequantisierung als Obergrenze: „Magie-Kosten“ von 2^(0,23·k)
- **Frage:** Ab welchem k kostet die klassische Simulation der trainierten Zelle (Stabilizer-
  Zerlegung) mehr als das direkte Matrixprodukt? Das wäre die ehrliche Antwort auf Maltes Ziel (2)
  („unendlich skalierbar?“). Nach Bravyi–Gosset: 2^(0,23·k) bei k = 30 ≈ 120, wirkungslos.
- **Neuheit:** Methode bekannt; Neu ist nur die Anwendung auf eine trainierte QML-Zelle mit
  M50-Budget (gesucht, nichts gefunden).
- **Test:** reine Rechnung auf Papier + ein Python-Skript mit PennyLane/Stim-ähnlicher
  Zählung, < 1 h. Kontrolle: Vergleich mit State-Vector für n = 8.
- **Quantenstufe:** Q1 (Theorie).

**Priorität (mein Vorschlag):** E2 (billig, sofort, liefert den Benchmark) > E3 (Alleinstellung,
passt zu I-036) > E1 (braucht saubere Pipeline) > E4 > E5.

## 5. Offene Fragen für Etappe 2
- **Volltexte** von arXiv 2607.12035 (QRC-Magie), 2510.18623, 2608.02325 und 2507.18954 lesen
  (arXiv gesperrt: über Semantic Scholar/Verlagsseiten/PRA-Zugang versuchen). Prüfen, ob
  „Magie-Konzentration“ dort als Größe vorkommt.
- Genaue Synthesekonstanten (Kliuchnikov 2023, gridsynth-Zahlen, „Mixed-fallback“) und
  **echte Cultivation-Kosten** (Raumzeit in Qubit-Runden für T-Gatter ~10⁻⁷ Fehler) aus dem
  Gidney-Paper ziehen, um „1 T = x CNOT“ belegt anzugeben.
- Ist M50 invariant gegen die Wahl der Clifford-Basis? (Magie ist es, M50 vermutlich nicht:
  Suche nach einem invarianten Maß, z. B. über SRE pro Nutzen.)
- Verhältnis zu Maltes Ziel (3), Holevo: Magie begrenzt nicht die Informationsmenge, nur die
  Simulierbarkeit; **Wechselwirkung klären** (nicht in meinem Thema, Anschluss an die anderen Berichte).
- Pseudomagie: kann man die Aufgabe so wählen, dass Magie *nicht effizient messbar* ist und
  trotzdem nützt? Reine Theorie, für Jugend forscht wohl zu weit [?].

## 6. Quellenliste
1. Leone, Oliviero, Hamma, Stabilizer Rényi Entropy, PRL 128, 050402 (2022): https://link.aps.org/doi/10.1103/PhysRevLett.128.050402
2. Bravyi, Gosset, PRL 116, 250501 (2016): https://link.aps.org/accepted/10.1103/PhysRevLett.116.250501
3. Ross, Selinger, arXiv 1403.2975: https://arxiv.org/abs/1403.2975v1
4. Kliuchnikov et al., Quantum 7, 1208 (2023): https://quantum-journal.org/papers/q-2023-12-18-1208/
5. Gidney, Shutty, Jones, Magic state cultivation, arXiv 2409.17595: https://arxiv.org/html/2409.17595v1
6. Gidney, Cultivation on the surface code, arXiv 2502.01743: https://arxiv.org/html/2502.01743v1
7. Gidney, RSA-2048 mit <1 Mio. Qubits, arXiv 2505.15917: https://arxiv.org/abs/2505.15917
8. Howard, Campbell, Robustheit der Magie: https://eprints.whiterose.ac.uk/id/eprint/113586/1/Robustness_main_appendix.pdf
9. Heinrich, Gross, Robustness of Magic and Symmetries of the Stabiliser Polytope, arXiv 1807.10296: https://ar5iv.labs.arxiv.org/html/1807.10296
10. Grewal et al., Efficient learning of quantum states prepared with few non-Clifford gates, Quantum 9, 1907: https://quantum-journal.org/papers/q-2025-11-06-1907/
11. Grewal et al., t-doped, Einzelkopie, Quantum 8, 1250: https://quantum-journal.org/papers/q-2024-02-12-1250/
12. Gu et al., Pseudomagic Quantum States, PRL 132, 210602: https://arxiv.org/pdf/2308.16228
13. Optimal quantum reservoir learning in proximity to universality, arXiv 2510.18623: https://arxiv.org/pdf/2510.18623
14. Quantum magic and non-commutativity in QRC, arXiv 2607.12035: https://arxiv.org/html/2607.12035
15. Role of nonstabilizerness in quantum optimization, arXiv 2505.17185: https://arxiv.org/pdf/2505.17185
16. Magic-Informed Quantum Architecture Search, arXiv 2605.03932: https://arxiv.org/html/2605.03932v1
17. Trainability of quantum models beyond known classical simulability, arXiv 2507.06344: https://arxiv.org/pdf/2507.06344
18. CAFQA, arXiv 2202.12924: https://arxiv.org/abs/2202.12924
19. Classical simulability with shallow magic depth, PRX Quantum 6, 010337: https://link.aps.org/doi/10.1103/PRXQuantum.6.010337
20. Clifford Perturbation Theory, arXiv 2306.04797: https://arxiv.org/abs/2306.04797
21. Almost fault-tolerant QML, arXiv 2507.18954: https://arxiv.org/pdf/2507.18954
22. Unconditional quantum magic advantage in shallow circuits, Nat. Commun. 2024: https://www.nature.com/articles/s41467-024-54864-0
23. Gottesman–Knill limit on one-way communication, arXiv 2506.19369: https://arxiv.org/html/2506.19369v2
24. Stabilizer Statistical Mechanics (Nullität als T-Schranke), arXiv 2608.14798: https://arxiv.org/pdf/2608.14798
25. Probing quantum complexity via saturation of stabilizer entropies, Quantum 2025: https://quantum-journal.org/papers/q-2025-07-21-1801/
26. Interne Quellen: `INSTITUT/UEBERGABE.md` §0/§4; `INSTITUT/recherche/2026-09-28_neuheit_magie.md`.
