# Clifford-Schaltungen als Klassifikatoren: Einordnung und Folgeexperimente

Stand: 2026-09-29. Recherche per Websuche/Abstract-Abruf. Vollstexte wurden NICHT gelesen; Aussagen stützen sich auf Abstracts bzw. Suchzusammenfassungen. Kennzeichnung:
- [belegt] = aus abgerufener Quelle (Abstract) entnommen
- [Suchtreffer] = nur aus Suchzusammenfassung, Seite nicht selbst geöffnet
- [EIGEN] = eigene Herleitung/Hypothese, nicht aus einer Quelle, muss geprüft werden
- [?] = unsicher

Ausgangsbefund (aus dem Auftrag): Rein-Clifford-Schaltung (8 Qubits, Amplitudenkodierung 16x16, Ry-Lagen + CNOT-Ring, Born-Auslese, linearer Kopf), gefunden per Simulated Annealing mit Ridge-LOO-Kriterium. MNIST: gleichwertig zum voll trainierten variationellen Modell (±0,5 pp, 20 Seeds). Fashion-MNIST: 1,31 pp schlechter.

---

## 0. Kurzfassung

1. Die Methode (SA/Bayes über diskrete Clifford-Winkel, klassisch simuliert) ist als CAFQA-Familie bekannt, aber dort zur INITIALISIERUNG von VQE, nicht als Endmodell für Klassifikation. Ich habe keine Arbeit gefunden, die eine reine Clifford-Schaltung als fertigen Bildklassifikator per diskreter Suche vorführt [?, Negativbefund einer begrenzten Suche, keine Garantie].
2. Ehrliche Formulierung: "0 T-Gatter" bedeutet billig auf fehlertoleranter Hardware, aber gleichzeitig nach Gottesman-Knill klassisch effizient simulierbar. Es ist also KEIN Hinweis auf Quantenvorteil, sondern das Gegenteil. Zusätzlich: Die Amplitudenkodierung ist der teure, nicht-Clifford-Teil.
3. [EIGEN] Das Modell ist sehr wahrscheinlich äquivalent zu: feste orthogonale Matrix (Clifford) auf den Pixelvektor, quadrieren, linearer Kopf. Das ist ein klassisches Modell mit quadratischen Merkmalen. Das erklärt vieles und legt Kontrollexperimente nahe (Abschnitt 4).
4. Die Literatur zu "Magie als Lernressource" ist dünn und indirekt (Reservoir-Computing, Expressibilität). Eine direkte Aussage "MNIST braucht keine Magie, Fashion schon" gibt es meines Wissens nicht; das ist eine offene, testbare Hypothese.
5. Wichtigstes Folgeexperiment: Dosis-Wirkung über die Zahl nicht-Clifford-Gatter (0, 1, 2, 4, 8 ...) plus Kontrollen (Zufalls-Clifford, Haar-orthogonal, klassische Walsh-Hadamard-Quadrate).

---

## 1. Literatur zu Clifford/Stabilizer, diskreter Suche, CAFQA

### 1.1 CAFQA und Verwandte (Clifford-Suche als Initialisierung)
- **CAFQA** (Ravi, Smith, Chong et al.; Autorenliste hier nicht verifiziert [?]): Ansatz nur aus Clifford-Gattern, Parameter per Bayes-Optimierung im diskreten Clifford-Raum, klassisch simuliert. Für VQE-Chemie liefert es gute Startpunkte. [belegt, Abstract/Suchtreffer] https://arxiv.org/abs/2202.12924
- **Cheng, Khosla, Self, Lin, Li, Medina, Kim: "Clifford Circuit Initialisation for Variational Quantum Algorithms"** (arXiv 2207.01539, 2022): Simulated Annealing über Clifford-Parameterpunkte als Vorab-Optimierung. Das ist methodisch am nächsten an Deiner SA-Suche. Die Autoren betonen auch Skalierungsgrenzen im allgemeinen Fall. [belegt] https://arxiv.org/abs/2207.01539
- CAFQA auf Ionenfalle demonstriert (2024). [Suchtreffer] https://arxiv.org/html/2408.06482v1
- **Meyer, Scala, Tacchino, Lucchi: "Trainability of Quantum Models Beyond Known Classical Simulability"** (arXiv 2507.06344): Taylor-Surrogat nahe Clifford-Schaltungen und "Linear Clifford Encoder", der einige Clifford-Gatter in den Ansatz einbaut, um Gradienten konstant zu halten. Relevant als Brücke "Clifford plus wenige Nicht-Clifford". [belegt, Abstract] https://arxiv.org/abs/2507.06344
- Simulation nahe Clifford: **Bravyi et al., "Simulation of quantum circuits by low-rank stabilizer decompositions"** (arXiv 1808.00128; Kosten wachsen exponentiell nur in der Zahl der Nicht-Clifford-Gatter) [belegt] https://arxiv.org/abs/1808.00128 ; **Clifford Perturbation Theory** (arXiv 2306.04797) [Suchtreffer] https://arxiv.org/abs/2306.04797

### 1.2 Diskrete Architektursuche / QAS für QML
- **Lu et al., QAS-Bench** (ICML 2023): vergleicht Brute Force, Simulated Annealing, genetische Algorithmen, RL, differenzierbare Suche; SA skaliert schwach. [Suchtreffer] https://proceedings.mlr.press/v202/lu23f/lu23f.pdf
- **Martyniuk et al., "Quantum Architecture Search: A Survey"** (arXiv 2406.06210). [Suchtreffer] https://arxiv.org/pdf/2406.06210
- **Élivágar** (arXiv 2401.09393): effiziente Schaltungssuche für Klassifikation. [Suchtreffer] https://arxiv.org/pdf/2401.09393
- **Lipardi et al., "Magic-Informed Quantum Architecture Search"** (arXiv 2605.03932): MCTS + GNN, steuert Magie der Kandidatenschaltungen; getestet auf Grundzustandsenergien und Zustandsapproximation, NICHT auf Klassifikation. Als methodischer Baustein nützlich (Magie als Suchsignal). [belegt, Abstract] https://arxiv.org/abs/2605.03932
- Unterschied zu Deiner Arbeit: QAS variiert meist Gatter-TYP und Position (Topologie); Du variierst diskrete WINKEL bei fester Topologie. Das ist der CAFQA-Fall. Für Fairness sollte man das so benennen.

### 1.3 Klassisch simulierbare / entanglement-arme QML-Modelle mit guter Bildklassifikation
Diese stützen Deinen Befund "MNIST braucht wenig Quantenressourcen":
- **De Lorenzis et al., "Entanglement and Classical Simulability in Quantum Extreme Learning Machines"** (arXiv 2509.06873): MNIST, Fashion-MNIST, CIFAR-10; moderate Verschränkung, mit klassischer Simulation vereinbar, reicht für nützliche Merkmale. [belegt, Abstract] https://arxiv.org/abs/2509.06873
- **Bowles, Ahmed, Schuld, "Better than classical?"** (arXiv 2403.07059): 12 QML-Modelle, klassische Standardmodelle schlagen sie; Entfernen der Verschränkung schadet oft nicht. [belegt, Abstract] https://arxiv.org/abs/2403.07059
- **Bermejo et al., "QCNNs are (effectively) classically simulable"** (PRX Quantum; arXiv 2408.12739): klassische Surrogate erreichen oder übertreffen QCNNs; "locally easy" Datensätze. Warnung zur Benchmark-Wahl. [belegt, Abstract] https://arxiv.org/abs/2408.12739
- **Cerezo et al., "Does provable absence of barren plateaus imply classical simulability?"** (Nat. Commun. 2025; arXiv 2312.09121): Trainierbarkeit und klassische Simulierbarkeit hängen oft zusammen. [belegt, Abstract] https://arxiv.org/abs/2312.09121
- **Landman et al., Random Fourier Features als klassische Näherung variationeller QML** (arXiv 2210.13200) [Suchtreffer] und Kritik/Grenzen: https://quantum-journal.org/papers/q-2025-02-20-1640/
- **Schuld, Sweke, Meyer, "Effect of data encoding on the expressive power..."** (PRA 103, 032430, 2021; arXiv 2008.08605): Modelle als Fourier-Reihen; Kodierung bestimmt das Spektrum. Bei Amplitudenkodierung ist das Modell dagegen polynomial in den Eingaben (quadratisch bei einmaliger Kodierung) [EIGEN-Einordnung]. [belegt für das Papier] https://arxiv.org/abs/2008.08605

### 1.4 Lücke
Ich fand keine Arbeit, die (a) eine reine Clifford-Schaltung (b) per diskreter Suche (c) als fertiges Endmodell (d) für Bildklassifikation (e) mit Vergleich zu voll trainiertem VQC durchführt [?]. Die Einordnung "CAFQA-artiges Verfahren, aber als Endmodell statt Initialisierung" ist der ehrliche Anknüpfungspunkt. Vorsicht: Der Begriff "Clifford" tritt in der QML-Literatur auch als Clifford-ALGEBREN (Geometrische Algebra) auf, das ist etwas völlig anderes (z. B. Springer, Adv. Appl. Clifford Algebras 2023).

---

## 2. Gottesman-Knill und ehrliche Formulierung

### 2.1 Zitierbare Quellen
- Gottesman, "The Heisenberg representation of quantum computers" (1998), arXiv quant-ph/9807006. https://arxiv.org/abs/quant-ph/9807006 [belegt]
- Aaronson, Gottesman, "Improved simulation of stabilizer circuits", Phys. Rev. A 70, 052328 (2004), arXiv quant-ph/0406196. Zeigt Simulierbarkeit (CHP-Programm, tausende Qubits) und dass Stabilizer-Simulation ⊕L-vollständig ist, Stabilizer-Schaltungen also vermutlich nicht einmal für klassische Berechnung universell sind. [belegt] https://arxiv.org/abs/quant-ph/0406196
- Bravyi et al. 2019 (siehe 1.1) für "Kosten skalieren mit Zahl der Nicht-Clifford-Gatter".
- Leone, Oliviero, Hamma, "Stabilizer Rényi entropy", PRL 128, 050402 (2022), arXiv 2106.12587: berechenbares Magiemaß. [belegt] https://arxiv.org/abs/2106.12587

### 2.2 Formulierungsvorschlag (ehrlich, für Text/Poster)
Aussage-Baustein, den ich für vertretbar halte:

> "Die gefundene Schaltung besteht ausschließlich aus Clifford-Gattern. Nach dem Gottesman-Knill-Theorem ist sie klassisch effizient simulierbar. Wir behaupten daher keinen Quantenvorteil. Das Ergebnis zeigt, dass für MNIST eine Klasse von Schaltungen genügt, die auf fehlertoleranter Hardware ohne teure T-Gatter auskäme und die gleichzeitig klassisch berechenbar ist. Der nicht-Clifford-Aufwand steckt in der Zustandspräparation (Amplitudenkodierung), nicht in der trainierten Zelle."

Zu beachtende Punkte:
- "0 T-Gatter" heißt: geringe Kosten bei fehlertoleranter Hardware (dort sind Clifford-Gatter vergleichsweise billig, T/Nicht-Clifford teuer). Auf heutiger NISQ-Hardware kosten Clifford-Gatter aber ähnlich viel wie andere Gatter; der Vorteil gilt also dort nicht pauschal. [Standardwissen, hier ohne eigene Quelle; ggf. im Hardware-Recherche-Dokument belegen]
- Gottesman-Knill im engen Sinn bezieht sich auf Stabilizer-Eingangszustände und Pauli-Messungen. Hier ist der Eingang ein beliebiger Amplitudenvektor (hohe Magie). Klassisch ist das Ganze dennoch trivial, weil der Vektor ohnehin klassisch vorliegt: Anwendung einer 256x256-Matrix. Der Gottesman-Knill-Bezug ist daher formal wahr, aber nicht der eigentliche Grund der Simulierbarkeit. [EIGEN, sollte vor Verwendung nochmal gegengeprüft werden]
- Amplitudenkodierung allgemeiner Bilder braucht im Allgemeinen exponentiell viele Gatter in der Qubit-Zahl [Standardwissen, Quelle ergänzen]; "Speed-up" kann daran hängen. Born-Auslese braucht viele Shots.
- Ein Modell, das klassisch simulierbar UND gut ist, ist genau die Situation, die Cerezo et al. und Bermejo et al. als Warnsignal für Quantenvorteil beschreiben. Man kann das offen als Einordnung nutzen: Die Clifford-Zelle ist ein Beleg, dass MNIST/Fashion in diesem Aufbau kein Quantenvorteil-Benchmark ist (vgl. Bowles et al.).

---

## 3. Warum MNIST ohne Magie, Fashion nicht?

### 3.1 Was die Literatur hergibt
- Magie (Nicht-Stabilizerness) ist die Ressource, die Simulierbarkeit bricht (Leone et al.; Bravyi et al.). Als Lernressource ist die Evidenz indirekt:
  - **Ivaki, Karjula, Ala-Nissila, "Optimal quantum reservoir learning in proximity to universality"** (arXiv 2510.18623): Zufallsschaltung, in der ein Anteil p der Clifford-Gatter durch nicht-stabilisierende Gatter ersetzt wird; Leistung korreliert mit Verschränkungsspektrum und langreichweitiger Magie. Reservoir-Aufgaben (zeitliche Verarbeitung), NICHT Bildklassifikation. [belegt, Abstract; Titel laut Abruf "Optimal quantum reservoir learning in proximity to universality"] https://arxiv.org/abs/2510.18623
  - **Maronese, Ferrari, Vandelli, Dragoni, "High-expressibility QNNs using only classical resources"** (Quantum Mach. Intell. 2026; arXiv 2506.13605): Clifford-erweiterte MPS nähern die Haar-Verteilung schneller an (Verschränkung und Magie). Expressibilität, keine Klassifikationsergebnisse. [belegt] https://arxiv.org/abs/2506.13605
  - Lipardi et al. (Magic-informed QAS): Magie als Steuergröße, nicht auf Klassifikation getestet.
- Direkte Belege "Datensatz X braucht Magie Y" habe ich nicht gefunden [?].

### 3.2 Eigene Hypothesen (alle [EIGEN], ungeprüft)
Herleitung: Ein Clifford-Operator bildet Basiszustände auf Stabilizer-Zustände ab. Reelle Stabilizer-Zustände haben Amplituden gleichen Betrags (±2^(-k/2)) auf einem affinen Unterraum, sonst 0. Bei Ry-Winkeln aus {0, π/2, π, 3π/2} und CNOT ist U reell orthogonal. Jede Zeile von U ist dann ein solcher Vektor. Die Born-Wahrscheinlichkeit p_j = (Ux)_j^2 ist also das Quadrat einer vorzeichenbehafteten Mittelung der Pixel über einen affinen Pixel-Unterraum (Walsh-/Hadamard-artige Summen). Das Modell wäre: quadratische Merkmale aus Walsh-artigen Pixelsummen, dann linearer Kopf.

Daraus:
- H1: MNIST ist im Wesentlichen mit groben, globalen Summen-Kontrasten trennbar (ähnlich der beobachteten Güte von 64 Hadamard-Mustern in klassischen Arbeiten, siehe unten). Feste "flache" Gewichte (alle ±1 gleich groß) genügen. Fashion braucht feinere, unterschiedlich gewichtete Muster (Textur, Umrissformen), wofür kontinuierliche Winkel (ungleiche Amplitudenbeträge) nötig sind. Das ist genau das, was Nicht-Clifford-Gatter liefern (Amplituden ungleich flach). Magie wäre hier ein Proxy für "ungleich gewichtete Filter".
- H2: Der Fashion-Rückstand ist kein Magie-, sondern ein Suchproblem (diskreter Raum 4^(#Winkel) mit SA nicht ausgeschöpft). Vertretbar zu prüfen über Suchbudget-Skalierung.
- H3: Der Fashion-Rückstand ist ein Kapazitätsproblem der Merkmalsanzahl (256 quadratische Merkmale bei 8 Qubits) und würde mit mehr Qubits oder Wiederholungen verschwinden, unabhängig von Magie.
Klassische Vergleichsquelle: Klassifikation von MNIST/Fashion aus Hadamard-Basis-Messungen (64 Muster) mit klassischem Netz ca. 96 % / 84 %, QNN 95 % / 81 % (arXiv 2407.12506) [Suchtreffer, Zahlen aus Zusammenfassung, nicht geprüft] https://arxiv.org/pdf/2407.12506. Fashion liegt dort klar unter MNIST, ähnliche Richtung wie Dein Befund, kein Beweis für Magie.

---

## 4. Folgeexperimente (nach Aussagekraft geordnet)

Alle sind rechenarm bis mittel; kein Lauf wurde gestartet (CPU belegt). Vor jedem Lauf gegen die sieben Regeln prüfen (Protokoll-Check).

1. **Kontrollen für "leistet die Clifford-Suche überhaupt etwas?"** (höchste Priorität, billig)
   - Identität (U = 1): quadrierte Pixel + linearer Kopf.
   - Zufalls-Clifford (ohne Suche), Verteilung über z. B. 200 Draws: Wie viel bringt SA über den Zufallsdurchschnitt?
   - Haar-zufällig orthogonal U (nicht Clifford).
   - Klassisch: quadrierte Walsh-Hadamard-Koeffizienten (256-dim.) + Ridge, und quadratischer Kernel/SVM auf Pixeln.
   - Begründung: Prüft die [EIGEN]-Äquivalenz (Abschnitt 3.2) und die Kernaussage "Gleichwertig zum VQC" gegen triviale klassische Modelle (Bowles et al., Bermejo et al.).
2. **Magie-Dosis-Wirkungs-Kurve** (Kernexperiment für Frage 3)
   - Erlaube m Nicht-Clifford-Gatter (z. B. Ry(π/4) an per SA gewählten Positionen), m = 0, 1, 2, 4, 8, alle. SA über Position und Winkel.
   - Ausgabe: Genauigkeit gegen m für MNIST und Fashion. Erwartung H1: Fashion-Lücke schließt sich schon bei kleinem m, MNIST bleibt flach. Falls nicht, ist H1 widerlegt und H2/H3 wahrscheinlicher.
   - Zusätzlich Magiemaß der Schaltung/Ausgangszustände (Stabilizer-Rényi-Entropie, bei 8 Qubits exakt berechenbar, 4^8 = 65536 Pauli-Strings) gegen Genauigkeit. Bemerkung: SRE des Eingabezustands (Bild) ist eine eigene Größe; prüfen, ob sie zwischen MNIST und Fashion systematisch abweicht.
   - Verwandt: Ivaki et al. (p-Substitution), Linear Clifford Encoder (Meyer et al.).
3. **Suchbudget-Skalierung** (trennt H2 von H1/H3)
   - Genauigkeit gegen SA-Schritte, mit Konvergenzkurve und mehreren Neustarts, auch alternatives Suchverfahren (Bayes wie CAFQA, genetisch).
   - Falls Fashion-Lücke mit Budget schrumpft: Suchproblem.
4. **Kapazitäts-Skalierung**: 6/8/10 Qubits bzw. mehrere Schaltungen konkateniert (Merkmale aus mehreren Cliffords). Trennt H3.
5. **Datensatz-Leiter**: KMNIST, EMNIST-Letters, CIFAR-10 in Graustufen 16x16. Frage: Korreliert der Clifford-Rückstand mit der klassischen Nichtlinearitätslücke (z. B. Genauigkeit Kernel-SVM minus Ridge auf Pixeln)? Das wäre ein sparsamer, klassischer Erklärungsansatz statt Magie.
6. **Statistik sauber machen**: Äquivalenz für MNIST mit vorab festgelegter Marge (±0,5 pp) per TOST/gepaartem Test über Seeds statt "innerhalb ±0,5 pp"; Konfidenzintervall für die 1,31 pp bei Fashion. Test-Daten strikt getrennt von der LOO-Suche (Selektionsverzerrung durch Auswahl auf Trainings-LOO ist gering, aber ausweisen). Fairer Vergleich: gleiche Parameter-/Merkmalszahl und Shot-Budget für VQC und Clifford. Beides in Bowles et al. als typische Fehlerquelle diskutiert.
7. **Rauschen/Shots** (falls Hardware-Aussage gewünscht): endliche Shots bei Born-Auslese; die diskrete Schaltung braucht nur bekannte Verteilungen p_j. Aussage "Hardwarekosten" nur mit dieser Rechnung.
8. **Quantum-Extreme-Learning-Analogie**: Das Modell ist ein QELM-artiges System (feste Schaltung + linearer Kopf, hier mit gesuchter Schaltung). Direkter Vergleich mit De Lorenzis et al. auf denselben Datensätzen als Literaturanker.

---

## 5. Quellenliste (Kurz)

| Thema | Quelle | Link | Status |
|---|---|---|---|
| CAFQA | arXiv 2202.12924 | https://arxiv.org/abs/2202.12924 | belegt |
| Clifford-Init per SA | Cheng et al., 2207.01539 | https://arxiv.org/abs/2207.01539 | belegt |
| LCE / Taylor-Surrogat | Meyer et al., 2507.06344 | https://arxiv.org/abs/2507.06344 | belegt |
| Stabilizer-Rang | Bravyi et al., 1808.00128 | https://arxiv.org/abs/1808.00128 | belegt |
| Clifford-Perturbation | 2306.04797 | https://arxiv.org/abs/2306.04797 | Suchtreffer |
| QAS-Bench | Lu et al. ICML 2023 | https://proceedings.mlr.press/v202/lu23f/lu23f.pdf | Suchtreffer |
| QAS-Survey | Martyniuk et al. | https://arxiv.org/pdf/2406.06210 | Suchtreffer |
| Magic-informed QAS | Lipardi et al. | https://arxiv.org/abs/2605.03932 | belegt |
| Gottesman 1998 | quant-ph/9807006 | https://arxiv.org/abs/quant-ph/9807006 | belegt |
| Aaronson-Gottesman 2004 | quant-ph/0406196 | https://arxiv.org/abs/quant-ph/0406196 | belegt |
| Stabilizer-Rényi-Entropie | Leone et al. | https://arxiv.org/abs/2106.12587 | belegt |
| QELM Simulierbarkeit | De Lorenzis et al. | https://arxiv.org/abs/2509.06873 | belegt |
| Benchmarking QML | Bowles et al. | https://arxiv.org/abs/2403.07059 | belegt |
| QCNN simulierbar | Bermejo et al. | https://arxiv.org/abs/2408.12739 | belegt |
| Barren Plateaus/Simulierbarkeit | Cerezo et al. | https://arxiv.org/abs/2312.09121 | belegt |
| Datenkodierung/Fourier | Schuld et al. | https://arxiv.org/abs/2008.08605 | belegt |
| RFF-Dequantisierung | Landman et al. | https://quantum-journal.org/papers/q-2025-02-20-1640/ (Kritik) ; arXiv 2210.13200 | Suchtreffer |
| Reservoir + Magie | Ivaki et al. | https://arxiv.org/abs/2510.18623 | belegt |
| Expressibilität klassisch | Maronese et al. | https://arxiv.org/abs/2506.13605 | belegt |
| Hadamard-Muster MNIST/Fashion | arXiv 2407.12506 | https://arxiv.org/pdf/2407.12506 | Suchtreffer, Zahlen ungeprüft |

## 6. Grenzen dieser Recherche
- Nur Abstracts/Suchzusammenfassungen; Autorenlisten von CAFQA und einigen Suchtreffern nicht verifiziert.
- Negativbefund (keine Clifford-Endmodell-Klassifikationsarbeit) stammt aus wenigen Suchanfragen; eine Lücke kann scheinbar sein.
- Alle Abschnitt-3.2-Aussagen sind Hypothesen und noch nicht gerechnet.
