# 01 — Holevo-Grenze: was sie sagt, was sie nicht verbietet (Recherche 05.10.2026)

Quellenlage: Alle Webquellen nur als **Suchauszug (Snippet)** gelesen, nicht im Volltext (arxiv.org ist per WebFetch gesperrt; das WebSearch-Budget der Sitzung ist nach ~25 eigenen Anfragen erschöpft, "200 of 200"). Mit [EIGEN] = eigene Überlegung/eigene Rechnung, [?] = unbelegt.

## 1. Kurzfassung

- Holevo: Aus **einem** Zustand aus einem Ensemble {p_i, ρ_i} auf n Qubits lassen sich **höchstens χ = S(ρ̄) − Σ p_i S(ρ_i) ≤ n Bit** klassische Information gewinnen, egal welche Messung. Es begrenzt die Information **pro Zustandskopie**, nicht, was man berechnen kann.
- Die 2ⁿ Amplituden sind trotzdem nicht auslesbar: volle Tomographie braucht ~d·r/δ Kopien (Haah et al.), also exponentiell in n. Viele Kopien ändern das nur linear: M Kopien liefern höchstens M·n Bit.
- "Umgehen" im strengen Sinn geht nicht, aber die Grenze ist bei fast allem, was Quantenalgorithmen nützlich macht, nicht der Engpass: Classical Shadows, Shadow Tomography, QRAC, Fingerprinting, Superdense Coding und Amplitudenschätzung sind alle mit Holevo vereinbar und schlagen trotzdem naive Erwartungen (jeweils anderer Ressourcenbegriff, siehe Tabelle).
- **Für Klassifikation ist Holevo nicht bindend:** ein 10-Klassen-Label braucht nur log₂10 ≈ 3,32 Bit, QIRE hat n = 8 Qubits (8 Bit pro Schuss). Bindend sind (a) die **Unterscheidbarkeit der Klassen-Born-Verteilungen** (Chernoff-Information, Marge) und (b) die Schusszahl.
- Eigene Rechnung [EIGEN] (sklearn-Digits, 6 Qubits, naive Bayes auf Schuss-Histogramm): ein Schuss trägt nur **0,29 Bit** über die Klasse (0,74 Bit über den Zustand), Fano-Untergrenze für 89 % Genauigkeit ≈ 3 Schuss, real nötig ≈ 100. Das ist ein messbarer "Holevo-/Fano-Wirkungsgrad" von nur ~3 %, also viel Luft nach oben.
- "QLM unendlich skalierbar?" Nein, aber nicht wegen Holevo allein: Datenladen, Auslese ~1/ε², Trainierbarkeit und klassische Simulierbarkeit sind die härteren Grenzen. Ehrliche Formulierung in Abschnitt 3.

## 2. Stand der Forschung

### 2.1 Der Satz exakt

- Holevo 1973: für ein Ensemble {p_i, ρ_i} und beliebige Messung gilt I(X;Y) ≤ χ := S(Σp_iρ_i) − Σp_iS(ρ_i); bei dim d = 2ⁿ folgt χ ≤ log₂ d = n. Für reine Zustände ist χ = S(ρ̄) ≤ n. Quellen: Holevo-Originalverweis https://www.mathnet.ru/eng/ppi903 ; Preskill-Skript Kap. 5 https://www.preskill.caltech.edu/ph229/notes/chap5.pdf ; https://en.wikipedia.org/wiki/Holevo%27s_theorem (nur Auszug).
- Wichtig: Schranke gilt für **ein** Mitglied des Ensembles, mit beliebigem POVM. n Bit wird nur erreicht, wenn die Zustände orthogonal sind (Rechenbasis). Nicht-orthogonale Zustände (alle Amplitudenkodierungen mit überlappenden Bildern) liefern < n Bit.
- Folgerung für Amplitudenkodierung (Snippet): Die Eingabe vollständig zurückzulesen würde Holevo widersprechen; exponentiell viele Kopien wären nötig. https://arxiv.org/pdf/2503.01545 (Limitations of Amplitude Encoding on Quantum Classification, nur Auszug).

### 2.2 Was Holevo NICHT verbietet (je mit Ressourcenbegriff)

| Verfahren | Aussage | Warum kein Widerspruch | Quelle |
|---|---|---|---|
| **Viele Kopien/Schüsse** | M Kopien: ≤ M·n Bit Information über ein klassisches Label. Volle Zustands-Tomographie eines Rang-r-Zustands: ~(d·r/δ)·ln(d/δ) Kopien, mit passender Untergrenze | Linear in M; die 2ⁿ Amplituden brauchen exponentiell viele Schüsse | Haah, Harrow, Ji, Wu, Yu (STOC 2016) https://dl.acm.org/doi/abs/10.1109/tit.2017.2719044 ; O'Donnell/Wright https://www.researchgate.net/publication/303901570_Efficient_quantum_tomography |
| **Classical Shadows** (Huang/Kueng/Preskill) | M Eigenschaften (Erwartungswerte) mit O(log M) Zufallsmessungen, unabhängig von Systemgröße, Genauigkeit ε, Aufwand zusätzlich ~Beobachtablen-Norm/ε² | Man schätzt nur **M vorgegebene Zahlen**, nie den ganzen Zustand | Nature Phys. 16, 1050 (2020); https://arxiv.org/pdf/2412.03381 und https://arxiv.org/pdf/2106.10190 (nur Auszug) |
| **Shadow Tomography** (Aaronson) | Õ(ε⁻⁴·log⁴M·log D) Kopien genügen, um M Zwei-Ausgang-Messungen auf ε genau vorherzusagen; Beweis über "postselected learning" | Wieder nur M Zahlen; Messung kollektiv über Kopien (Quantenspeicher nötig) | https://arxiv.org/pdf/1711.01053 ; https://www.scottaaronson.com/papers/batch.pdf |
| Quantenspeicher-Trennung | Ohne ≥ n Qubit Speicher braucht man Ω(2^((n−k)/3)) Kopien für |Pauli|-Erwartungswerte, mit Speicher O(n) | Zeigt, dass kollektive Messung echte Ressource ist | Chen/Cotler/Huang/Li (FOCS 2022) https://catalyzex.com/paper/arxiv:2111.05881 (nur Auszug) |
| **QRAC** | n Bit in m Qubits, jedes einzelne Bit mit Erfolgswkt. p > ½ abrufbar; Nayak: m ≥ (1−H(p))·n; klassisch geht m = (1−H(p))n + O(log n) | Man kann **einzelne** Bits lesen, nie alle; Beweis nutzt gerade Holevo | Nayak, FOCS 1999 https://www.semanticscholar.org/paper/Optimal-lower-bounds-for-quantum-automata-and-codes-Nayak/ca86a3524b766f97850e04c78a3a4cda00a75a55 ; https://en.wikipedia.org/wiki/Quantum_random_access_code |
| **Superdense Coding** | 2 Bit je gesendetem Qubit | Nur mit vorab geteiltem Verschränkungs-Paar; Bob misst ein 4-dim. System (2 Qubits), Holevo gilt für das Gesamtsystem. Entanglement-assistierte Kapazität = 2 log d | Bennett/Wiesner 1992; https://arxiv.org/pdf/0708.3062 (nur Auszug) |
| **Fingerprinting** (Buhrman/Cleve/Watrous/de Wolf) | Gleichheitstest in simultaner Kommunikation mit O(log n) Qubits statt Θ(√n) Bit klassisch (ohne gemeinsame Zufälligkeit) | Es wird nur ein **Ein-Bit-Prädikat** (gleich/ungleich) ausgewertet, nicht der String übertragen | PRL 87, 167902 (2001) https://arxiv.org/pdf/quant-ph/0102001 |
| **Amplitudenschätzung** | Fehler ε mit O(1/ε) Orakelaufrufen statt O(1/ε²) Schüssen | Braucht kohärente Wiederholung von U und U† (Zustandsvorbereitung samt Umkehrung); kein "eine Kopie mit mehr Information" | Brassard/Høyer/Mosca/Tapp https://arxiv.org/pdf/quant-ph/0005055 |
| **Hilbertraum als Rechenraum** | Unitäre wirken auf 2ⁿ Amplituden "gleichzeitig" | Nutzen entsteht nur, wenn die gesuchte **Ausgabe** klein ist; Eingabe-Laden kostet ~Datengröße | Aaronson, "Read the fine print", Nat. Phys. 11, 291 (2015) https://www.nature.com/articles/nphys3272 |

Gegengewicht, wichtig für Malte: Amplitudenkodierungs-Vorteile lassen sich oft klassisch "dequantisieren" (Tang 2019, https://www.researchgate.net/publication/333914977_A_quantum-inspired_classical_algorithm_for_recommendation_systems; Sampling-Rahmen https://arxiv.org/pdf/1910.06151), und schadensfreie, trainierbare Modelle sind oft klassisch simulierbar (Cerezo et al., Nat. Commun. 16, 2025, https://arxiv.org/pdf/2312.09121). Huang et al. (PRL 126, 190505): für Mittelung über Eingabeverteilung kann ein klassisches Modell mit ähnlich vielen Zugriffen mithalten; exponentieller Vorteil nur für "alle Eingaben" https://preskill.caltech.edu/pubs/preskill-2021-information-theoretic.pdf. "Power of data" (Nat. Commun. 2021) https://www.nature.com/articles/s41467-021-22539-9: Mit Daten kann klassisches ML Aufgaben lösen, die nach Rechenhärte aussehen.

### 2.3 Lernen und Klassifikation

- **Ist Holevo bindend?** Hier hilft Fano: Für K Klassen und Fehlerwahrscheinlichkeit P_e gilt I(C; Ĉ) ≥ log₂K − h₂(P_e) − P_e·log₂(K−1) (Fano, Standard; https://arxiv.org/pdf/1901.00555). Für K = 10 und P_e = 0,11 sind das 3,32·0,89 − 0,50 ≈ **2,46 Bit** nötig. Holevo erlaubt bis n = 8 Bit je Schuss. Also **nein, Holevo bindet nicht, solange die Klassenzustände genug voneinander verschieden sind**; es verlangt nur n ≥ log₂K / (Bit je Schuss) im Rahmen einer ausreichend gut unterscheidbaren Kodierung (ohne Zusatzrechnung gilt für perfekte Unterscheidung schon n ≥ 4 Qubits: 10 orthogonale Zustände in 16-dim Raum) [EIGEN, Standard-Linearalgebra].
- **Schusszahl vs Marge:** Bei Entscheidung per Schuss-Mittelwert (Hoeffding + Union) wächst die nötige Schusszahl wie ln(K/δ)/γ² mit Marge γ (Differenz der Klassen-Erwartungswerte); Belege: "You Only Measure Once" (Schusszahl ∝ 1/Marge², nur Auszug) https://arxiv.org/html/2509.20090v1 ; "Shot-frugal and robust quantum kernel classifiers" (Fehlerschranken über Margenfehler) https://arxiv.org/pdf/2210.06971 ; "Single-shot quantum machine learning" (Majority Voting über Wiederholungen) https://arxiv.org/pdf/2406.13812 ; Generalisierung mit Termen O(√(d/n)) + O(√(log n / S)) https://arxiv.org/html/2608.24891 ; empirisch ~100 Schuss, danach stabil https://quantum-journal.org/papers/q-2024-06-25-1385/pdf/ . Alle nur Auszug.
- **Optimale Unterscheidung vieler Kopien:** Der Fehler zweier Klassenzustände fällt wie exp(−M·ξ) mit der Quanten-Chernoff-Information ξ = −ln min_s Tr[ρ₊ˢρ₋^(1−s)] (Audenaert et al., PRL 98, 160501) https://arxiv.org/pdf/quant-ph/0610027 ; für feste Basismessung gilt die klassische Chernoff-Information der Born-Verteilungen (ist ≤ ξ). Deshalb ist **M ≈ ln(1/δ)/ξ** die eigentliche Schuss-Grenze, und Holevo steht nur als obere Kappe darüber.
- Verlustbarriere: Wenn die mittleren kodierten Zustände zweier Klassen trace-distance < ε haben, gibt es eine untere Schranke für den Verlust jeder Optimierung (Amplitudenkodierung konzentriert sich typischerweise); https://arxiv.org/pdf/2503.01545. Das ist die für MNIST relevante *Eingabe*-Seite der Grenze (Pixelvektoren überlappen stark, positive Pixelwerte) und erklärt, warum Vorzeichen-Masken helfen [EIGEN-Deutung].
- "Measure-first"-Grenze: feste Messstrategie für die ganze Klasse hat in konstruierten Aufgaben exponentielle Nachteile; für unsere Born-Auslese (feste Basismessung am Ende, aber lernbare Unitäre davor) gilt das nicht direkt, da die Unitäre im Training angepasst wird. https://arxiv.org/pdf/2311.12618 (nur Auszug).
- Information pro Schuss in Quantum-Decision-Trees wurde als Messgröße untersucht (https://arxiv.org/pdf/2502.11412, nur Auszug); ein Klassifikator-"Wirkungsgrad" wurde in Suchanfrage nicht gefunden.

## 3. Was das für QIRE bedeutet (ehrlich)

1. **Holevo ist für QIRE kein Engpass bei der Klassifikation.** 8 Qubits geben ≤ 8 Bit je Schuss, benötigt werden 3,3 Bit. Aber: 256 Pixelwerte (≈ 256·8 Bit Rohinformation) sind aus 8 Qubits nicht rücklesbar. Das ist für Klassifikation unschädlich, weil nur die Klasseninformation erhalten bleiben muss.
2. **Der 256-Werte-Born-Vektor** wird in den Experimenten (vermutlich) exakt ausgewertet = Grenzfall M → ∞. Eine reale Quantenmaschine liefert Schüsse. Die Genauigkeit fällt dann mit M ab (eigene Rechnung unten); das ist die ehrliche Kostenseite, die in Maltes Parameterzahl bisher nicht auftaucht.
3. **Der lineare Kopf (2.570 von ~2.600 Parametern) ist klassischer Nachverarbeitungsaufwand**, kein Quantenvorteil und von Holevo nicht berührt. Ein Kopf mit ~0 Parametern (Lookup/Argmax über Born-Ausgänge) wäre mit Ziel (1) "~100 Parameter" kompatibel, siehe Idee 3.
4. **"QLM unendlich skalierbar?"** Ehrliche Antwort: *nein*. (a) Holevo: ≤ n Bit je Schuss und je Zustand, also für Ausgabe-Vokabulare (Sprachmodell, ~log₂V Bit je Token) reichen wenige Dutzend Qubits, Holevo begrenzt hier nicht viel; (b) Datenladen für beliebige klassische Daten kostet ~Datengröße (Aaronson 2015); (c) Auslese kostet ~1/ε² Schüsse je Wert bzw. ~ln(1/δ)/ξ für Klassenentscheidung; (d) Trainierbarkeit/Barren Plateaus und klassische Simulierbarkeit (Cerezo 2025); (e) im Simulator auf Maltes CPU wächst der Speicher mit 2ⁿ, bei 8 Qubits trivial, bei ~30 Qubits 16 GB Grenze [EIGEN, 2ⁿ·16 Byte ≈ 17 GB]. "Skalierbar" lässt sich sauber nur als "die *Parameterzahl* wächst nur polylog (log₂-Winkel je Matrix, T-004/T-005) bei wachsender Breite" verteidigen, nicht als unbegrenzter Informationsdurchsatz.
5. Die Clifford-Befunde passen ins Bild: Eine Clifford-Zelle ist klassisch effizient simulierbar (Gottesman-Knill, Standardwissen), der Vorteil liegt also in der Struktur, nicht in Quanteninformation. Das sollte in der Arbeit offen stehen.

### Eigene Mini-Rechnung [EIGEN] (Scratchpad `h.py`, <1 min)
sklearn-Digits 8×8 (64 Amplituden = 6 Qubits, 60 % Training), Born-Verteilung, Klassen-Referenz = mittlere Born-Verteilung je Klasse, Klassifikator = Log-Likelihood des M-Schuss-Histogramms (naive Bayes, keine Parameter außer den 10×64 Mittelwerten). Kein MNIST (offline), keine Quantenschaltung, nur Rechenbasis ("Z") bzw. zufällig vorzeichenmaskierte Walsh-Hadamard ("H"); eine einzelne Messreihe, Zufallsfehler ca. ±1,5 %.

| Schüsse M | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 1000 | exakt |
|---|---|---|---|---|---|---|---|---|---|
| Genauigkeit Z | .185 | .243 | .446 | .626 | .744 | .839 | .890 | .893 | .903 |
| Genauigkeit H | .202 | .260 | .374 | .497 | .609 | .755 | .834 | .889 | .898 |

- Information je Schuss: I(Klasse; Schuss) = **0,29 Bit** (Z), 0,20 Bit (H); I(Zustand; Schuss) = 0,74 Bit (Z), 0,53 Bit (H); Maximum wäre log₂10 = 3,32 (Klasse) bzw. n = 6 (Holevo).
- Gültige Schranke (Standardargument, eigene Herleitung): Da Schüsse bedingt auf den Zustand S unabhängig sind, gilt I(C; X^M) ≤ I(S; X^M) ≤ M·I(S; X¹). Mit Fano: M ≥ [(1−P_e)·log₂10 − h₂(P_e)] / I(S;X¹). Für P_e = 0,11 (≈ 89 %): M ≥ 2,46/0,74 ≈ **3,3 Schuss** (Z). Praktisch gebraucht ≈ 100. **Wirkungsgrad ≈ 3 %.** (Die Schranke ist locker, weil I(S;·) viel Information über irrelevante Pixel-Details zählt; sie ist keine scharfe Untergrenze.)
- Beobachtung: H-Maske war hier *schlechter* als Z. Diese Basis wurde nicht optimiert; sagt nichts über die QIRE-Zelle (Lagen, trainierte Winkel).

## 4. Neue Ideen / Forschungsfragen

Neuheitsprüfung: ~25 Websuchen (Stichworte: Holevo efficiency, bits per shot, Fano classifier, shot complexity margin, measurement design mutual information). Suchbudget danach erschöpft. Treffer sind Nachbarn, aber keine dieser Ideen exakt gefunden. Das ist **keine Garantie**; in Etappe 2 mit Volltext-Zugriff gegenprüfen.

### Idee A — Holevo-/Fano-Wirkungsgrad η eines Klassifikators [EIGEN]
- **Frage:** Definiere η_Fano(M) = M_Fano,min / M_nötig und η_bit = I(C;Schuss)/log₂K. Sind η-Werte vergleichbar zwischen Zellen (Clifford vs. trainiert, L-Lagen) und sagen sie Genauigkeit bei wenigen Schüssen voraus?
- **Warum neu:** Gefunden: Information pro Schuss bei Quantum Decision Trees (2502.11412); Margen-Schuss-Schranken (2509.20090, 2210.06971); Fano/Holevo einzeln Standard. Eine Kennzahl "Informationsausbeute je Schuss relativ zu log₂K" als Gütemaß für Born-Klassifikatoren: nicht gefunden.
- **Billigster Test:** QIRE-Zelle bekannt aus Repo; Born-Vektoren auf MNIST-Trainingsmenge berechnen, Klassen-Mittel p_c(x) schätzen, I(C;X¹) per Formel (Sekunden), Genauigkeit-gegen-M-Kurve per Multinomial-Sampling (Minuten, 10k Bilder × 256 Werte). Gesamt < 5 min auf Ryzen 5.
- **Kontrolle:** Zufällig initialisierte Zelle (gleiche Lagenzahl); klassische Pixel-Histogramm-Baseline; Permutations-Labels (η muss → 0 gehen).
- **Stufe:** Q1 (Simulator exakt), Q2 (Simulator mit Schüssen), Q3 später (IBM, Schusszahl realistisch ≤ 10⁴).

### Idee B — Auslese-Design als Kanalkapazitäts-Problem [EIGEN]
- **Frage:** Wähle Cliffordteil/Maske nicht nach Genauigkeit, sondern so, dass I(C;X¹) maximal wird (Born-Ausgänge je Klasse möglichst disjunkt, Idealfall 3,32 Bit/Schuss = Ein-Schuss-Klassifikation). Reduziert das bei gleicher Exakt-Genauigkeit die nötige Schusszahl?
- **Warum neu:** "Single-shot QML" (2406.13812) bauen Modelle mit Einzel-Schuss-Eigenschaft über Boosting/Majority, nicht über Informationsmaximierung der Auslese-Verteilung (nur Auszug). Gezielte Optimierung der Clifford-Maske auf I(C;X) / Chernoff-Information: nicht gefunden.
- **Billigster Test:** Hill-Climbing über Vorzeichenmasken (Walsh-Hadamard-Form, Repo-Code I-036) mit Zielfunktion I(C;X¹) statt Genauigkeit: je Bewertung Millisekunden–Sekunden, ~10³ Schritte, ≈ 10–20 min (zwischen 1 und 2 Seeds). Vergleich: gleiche Suche mit Genauigkeit als Ziel.
- **Kontrolle:** Gleiche Rechenzeit, Zielfunktion Genauigkeit; Zufallsmaske; Test auf Held-out-Bildern.
- **Stufe:** Q1/Q2.

### Idee C — Parameterfreier Kopf: Lookup "Born-Ausgang → Klasse" [EIGEN, Variante bekannter Ideen]
- **Frage:** Ersetze den 2.570-Parameter-Kopf durch eine feste Zuordnung 256 Ausgänge → 10 Klassen (Mehrheitsvotum über Schüsse oder argmax der Klassen-Likelihood). Kostet nur 256·log₂10 ≈ 850 Bit Tabelle, 0 reelle Parameter. Wie viel Genauigkeit gegenüber linearem Kopf, besonders bei Idee-B-Maske?
- **Warum neu:** Probabilistische Born-Klassifikatoren mit direkter Ausgangszuordnung sind bekannt (Single-shot QML), eine Kombination mit "Clifford/GF(2)-Maske optimiert auf Informationsgehalt" und Zählung als Parameterbudget auf MNIST: nicht gefunden. Direkt relevant für Ziel (1) "~100 Parameter".
- **Billigster Test:** Tabelle aus Training berechnen (Sekunden); Genauigkeit exakt und bei M ∈ {1,10,100} Schüssen (Minuten).
- **Kontrolle:** linearer Kopf auf gleichen Merkmalen; zufällige Zuordnung.
- **Stufe:** Q1 → Q2 → Q3 (hier ist echte Hardware gut: nur 8 Qubits, 256 Zählwerte).

### Idee D — Sequentielle Schusszahl (SPRT) [EIGEN, Standardmethode neu angewandt, [?] Novelty]
- **Frage:** Schüsse einzeln ziehen, stoppen sobald Log-Likelihood-Abstand der besten zwei Klassen einen Schwellwert erreicht. Mittlere Schusszahl bei gleicher Genauigkeit gegenüber fester M?
- **Warum neu:** Wald-Test klassisch; Anwendung auf QIRE-Auslese mit Marge als Schwierigkeitsmaß (leichte Bilder = 1–3 Schüsse): nicht gefunden, aber nicht gezielt gesucht ([?]).
- **Billigster Test:** Erweiterung der Mini-Rechnung (`h.py`), <2 min; Vergleich Mittel-M bei 89 %.
- **Kontrolle:** feste M mit gleicher Durchschnittszahl.
- **Stufe:** Q2.

### Idee E — Skalierungs-Gesetz Schüsse vs Marge für QIRE prüfen [EIGEN]
- **Frage:** Gilt M ≈ c·ln(K/δ)/γ² (Hoeffding) bzw. M ≈ ln(1/δ)/Chernoff-Information empirisch auf MNIST? Wie liegt η (Idee A) dazu?
- **Warum neu:** Allgemeine Form belegt (s. 2.3); Prüfung am konkreten Clifford-Born-Klassifikator: nicht gefunden.
- **Billigster Test:** Pro Testbild Marge γ (Differenz der obersten Klassen-Likelihoods) berechnen, Schusszahl für 95 % Treffsicherheit je Marge-Bin; Log-Log-Steigung ≈ −2 erwartbar. < 5 min.
- **Kontrolle:** Zufalls-Labels, Kleinstmarge-Tests.
- **Stufe:** Q1/Q2.

## 5. Offene Fragen für Etappe 2

1. Volltexte prüfen (arXiv blockiert: andere Zugangswege, z. B. semanticscholar/openalex): HKP-Shadows-Satz exakt (Varianzterm), Aaronson-Schranke und verbesserte Nachfolger, "You Only Measure Once" Marge-Satz exakt, Nayak-Beweis.
2. Messen auf echtem MNIST und der echten QIRE-Zelle (Ideen A, B, C, D, E); Fashion-MNIST mit Magiewinkeln.
3. Zusammenhang zu Maltes Rang der quadratischen Form: Erhöht höherer Rang die Disjunktheit der Klassen-Born-Verteilungen (I(C;X¹))? [EIGEN, ungetestet]
4. Quanten-Chernoff-Information der Klassen-Mittelzustände (nicht nur Rechenbasis) als theoretische Obergrenze berechnen: Wie weit liegt die feste Basismessung darunter?
5. Sprach-Adapter: Informationsbilanz pro Token (~log₂ Vokabular) vs. Qubit-Zahl; gibt es dort eine Holevo-ähnliche Grenze für Adapter-Kapazität? [?]
6. Für die schriftliche Arbeit: eine klare, belegte Seite "Was Holevo nicht sagt" mit Tabelle aus 2.2 plus eigener Schuss-Kurve.
7. Neuheitsprüfung für A–E mit mehr Suchen (Budget wurde ausgeschöpft).

## 6. Quellenliste (alle nur Suchauszug, außer Standardwissen)

- Holevo 1973: https://www.mathnet.ru/eng/ppi903 ; https://en.wikipedia.org/wiki/Holevo%27s_theorem
- Preskill Skript Kap. 5: https://www.preskill.caltech.edu/ph229/notes/chap5.pdf
- Huang/Kueng/Preskill, Classical Shadows (Nat. Phys. 2020): Verweise https://arxiv.org/pdf/2412.03381 , https://arxiv.org/pdf/2106.10190
- Aaronson, Shadow Tomography: https://arxiv.org/pdf/1711.01053 ; https://www.scottaaronson.com/papers/batch.pdf
- Chen/Cotler/Huang/Li: https://catalyzex.com/paper/arxiv:2111.05881
- Haah et al.: https://dl.acm.org/doi/abs/10.1109/tit.2017.2719044 ; O'Donnell/Wright: https://www.researchgate.net/publication/303901570_Efficient_quantum_tomography
- Nayak (QRAC): https://www.semanticscholar.org/paper/Optimal-lower-bounds-for-quantum-automata-and-codes-Nayak/ca86a3524b766f97850e04c78a3a4cda00a75a55 ; https://en.wikipedia.org/wiki/Quantum_random_access_code
- Superdense Coding: https://arxiv.org/pdf/0708.3062
- Quantum Fingerprinting: https://arxiv.org/pdf/quant-ph/0102001
- Amplitudenschätzung: https://arxiv.org/pdf/quant-ph/0005055
- Aaronson "Read the fine print": https://www.nature.com/articles/nphys3272
- Tang: https://www.researchgate.net/publication/333914977_A_quantum-inspired_classical_algorithm_for_recommendation_systems ; https://arxiv.org/pdf/1910.06151
- Cerezo et al.: https://arxiv.org/pdf/2312.09121
- Huang et al., Power of data: https://www.nature.com/articles/s41467-021-22539-9
- Huang/Kueng/Preskill, Info-theoretic bounds: https://preskill.caltech.edu/pubs/preskill-2021-information-theoretic.pdf
- Limitations of Amplitude Encoding: https://arxiv.org/pdf/2503.01545
- Measure-first limitations: https://arxiv.org/pdf/2311.12618
- Quantum Chernoff Bound: https://arxiv.org/pdf/quant-ph/0610027
- Fano: https://arxiv.org/pdf/1901.00555
- Schuss/Marge: https://arxiv.org/html/2509.20090v1 ; https://arxiv.org/pdf/2210.06971 ; https://arxiv.org/pdf/2406.13812 ; https://arxiv.org/html/2608.24891 ; https://quantum-journal.org/papers/q-2024-06-25-1385/pdf/
- Information pro Schuss (Decision Trees): https://arxiv.org/pdf/2502.11412
- Eigene Rechnung: Scratchpad `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/h.py` (nicht im Repo)
