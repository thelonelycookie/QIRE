# Neuheitsprüfung: Magie-Budget-Linie (Vorgabe 7)

*Stand 28.09.2026. Literatursuche per WebSearch/WebFetch, Zeitraum 2019–2026, gegen die
Kernaussagen aus `UEBERGABE.md` (Abschnitt 4/5) und `ideen.json` (Feld `neuheit`).
Alles unten mit Quelle; Unsicheres ist als "nur Abstract/Snippet gelesen" markiert —
keine Vollprüfung der Paper-Body-Texte.*

---

## 1. Was gefunden wurde (nach Suchthema)

### 1.1 Magie/Nicht-Stabilizerness in trainierten QML-Modellen / variationellen Klassifikatoren
- **Quantum resources of quantum and classical variational methods**, arXiv:2409.13008
  (auch IOPscience, `iopscience.iop.org/article/10.1088/2632-2153/adaca2`) — verbindet
  Nicht-Stabilizerness mit Aufgabenleistung bei Quanten-Reservoir-Computing (QELM). War
  dem Institut schon bekannt (I-035), bestätigt: verwandte, aber andere Aufgabenklasse
  (Reservoir, nicht überwachte Bildklassifikation mit Amplitudenkodierung).
- **Non-stabilizerness of Neural Quantum States**, arXiv:2502.09725 (auch Phys. Rev.
  Research, `v5tw-yn1f`) — Monte-Carlo-Verfahren zur Magie-Messung in beliebigen
  variationellen Wellenfunktionen (Neural Quantum States, Vielteilchenphysik). Kein
  Bild-Klassifikator, kein Ry+CNOT-Born-Modell.
- **A hybrid variational quantum circuit approach for stabilizer states classifiers**,
  arXiv:2511.09430 — Klassifikation *von* Zuständen nach Magie-Gehalt (CNN erkennt
  Nicht-Stabilizer-Zustände), nicht Magie *in* einem trainierten Klassifikator-Modell.
  Andere Fragestellung.
- Fazit: Magie wird an Neural-Quantum-States und Reservoir-Systemen gemessen, aber
  **nicht** an einem trainierten, überwacht gelernten Amplituden-Encoding-Klassifikator
  mit Ry/CNOT-Ring wie der QIRE-Zelle. **Nur Snippets/Abstracts gelesen, keine Volltexte.**

### 1.2 T-count-Reduktion durch Runden variationaler Winkel auf Clifford
- **Optimal compilation of parametrised quantum circuits**, arXiv:2401.12877 — ZX-Kalkül-
  Vereinfachung generalisiert auf parametrisierte Schaltungen; dort ist das Analogon zur
  T-Count-Reduktion die Elimination redundanter freier Parameter (nicht: nachträgliches
  Runden gelernter Winkel auf π/2 in einem ML-Klassifikator).
- **Clifford Perturbation Approximation for Quantum Error Mitigation**, arXiv:2412.09518
  — Winkel werden nahe an Clifford gehalten (kleiner Bereich um Clifford-Punkte), aber
  Zweck ist Fehlerminderung, nicht Ressourcen-/T-Budget eines gelernten Klassifikators.
- **"Optimal number of parametrized rotations and Hadamard gates..."**, arXiv:2407.07846
  — Parameterreduktion in PQCs, anderer Fokus (Gatterzahl-Minimierung, nicht Magie-
  Konzentration nach Trainingsverlust).
- Zu "Ranking von Rotationswinkeln nach Verlustanstieg beim Runden, Top-k bleiben
  magisch, Rest auf Clifford" (I-001): **nichts Vergleichbares gefunden.** Gefundene
  Pruning-Arbeiten (z. B. Adaptive-Pruning mit Quantum-Fisher-Information, Springer
  2024) entfernen Winkel nahe 0° (Gate-Elimination), nicht Winkel nach Distanz zu einem
  Clifford-Gitter mit Wirkungs-Ranking.

### 1.3 "Clifford pre-training" / "Clifford-based circuit initialization" (CAFQA u. ä.)
- **CAFQA: Clifford Ansatz For Quantum Accuracy**, arXiv:2202.12924 (+ Folgearbeit
  arXiv:2308.11616, laut `ideen.json` bereits vom Institut notiert) — sucht eine
  reine Clifford-Schaltung als *Startpunkt* für VQE (Molekül-Grundzustände), per
  Bayes'scher Optimierung im diskreten Clifford-Raum; danach volle Optimierung.
  **Kein** Klassifikator, **kein** Runden nach dem Training, **kein** T-Budget-Fokus.
- **Clifford Circuit Initialisation for Variational Quantum Algorithms**, arXiv:2207.01539
  — Simulated Annealing über Clifford-Punkte als Vor-Optimierung für VQAs, Domäne
  Quantenchemie (laut Volltext-Zusammenfassung, s. u.), nicht ML/Bildklassifikation,
  kein Runden trainierter Winkel, keine Magie-Konzentrations-Kennzahl, kein T-Budget.
  Die Methode ist zudem laut Autoren "nicht beliebig skalierbar". *(WebFetch-Volltext
  gelesen, nicht nur Abstract.)*
- **Warm Starting VQAs with Near Clifford Circuits (NCC-WS)**, MDPI Electronics
  12(2):347 — Pretraining-Strategie mit Nahe-Clifford-Schaltungen zur Initialisierung
  von PQCs; Fokus auf Expressivität/Beschleunigung der Konvergenz, nicht auf T-Budget
  oder Magie-Konzentration als Kennzahl einer Aufgabe. **WebFetch auf die Originalseite
  schlug mit HTTP 403 fehl — nur aus Suchergebnis-Snippet zusammengefasst, unsicher.**
- Fazit: "Clifford als *Startpunkt* vor dem Training" ist ein etabliertes, mehrfach
  publiziertes Genre (CAFQA und Nachfolger, NCC-WS, Clifford-Circuit-Initialisation).
  Das Institut macht das **Gegenteil**: volles Training, danach Runden auf Clifford
  (Magie-*Extraktion* nach dem Training, nicht Clifford-*Start* vor dem Training) — das
  ist der entscheidende Unterschied und bisher nicht in dieser Form gefunden, aber die
  Verwandtschaft zu CAFQA muss in der schriftlichen Arbeit explizit abgegrenzt werden
  (Gefahr, dass ein Gutachter es verwechselt).

### 1.4 "Near-Clifford" VQAs allgemein
- Siehe 1.3 (CAFQA, NCC-WS, 2207.01539). Zusätzlich: **Variational Quantum Algorithms
  in the era of Early Fault Tolerance**, ACM/ISCA 2024 (`10.1145/3695053.3731112`) —
  "partial QEC" korrigiert Clifford-Teile exakt, injiziert Rz(θ) via Magic-State-
  Injection statt T-Distillation; verwandtes Konzept (Clifford billig, Magie teuer,
  gezielt einsetzen), aber Fehlerkorrektur-Kontext, nicht Trainings-/Rundungs-Kontext.
  **Nur Snippet gelesen.**

### 1.5 Pruning von Rotationswinkeln in PQCs (allgemein, nicht Clifford-spezifisch)
- **Adaptive pruning algorithm using a quantum Fisher information matrix for PQCs**,
  Springer/Quantum Machine Intelligence 2024 — entfernt Winkel nahe 0°, alternative
  Zielgröße (Quantum-Fisher-Information) statt Verlustanstieg-Ranking oder
  Clifford-Distanz. Verwandtes Werkzeug-Genre, andere Auswahlregel.
- Bereits vom Institut selbst notiert und eingeordnet: approximative Zustands-
  präparation (Araujo 2021) und das kombinierte approximative Laden (West et al. 2023,
  MNIST/Fashion, IBM-Hardware) — **das ist korrekt als "im Kern schon publiziert"
  eingestuft**, hier nur bestätigt, keine neue Quelle gefunden, die das widerlegt.

### 1.6 Magie als Ressource für Lernbarkeit
- arXiv:2409.13008 (s. o., Reservoir-Computing) ist die einschlägigste Quelle: Magie
  korreliert dort mit Aufgabenleistung. Für überwachte Bildklassifikation mit
  Amplitudenkodierung + Ry/CNOT (QIRE-Zelle) **nichts Vergleichbares gefunden.**

### 1.7 SRE von QNN-Zuständen
- **A Study on Stabilizer Rényi Entropy Estimation using Machine Learning**,
  arXiv:2509.16799 — SRE-*Schätzung* per ML-Regression (Random Forest, SVR) an
  zufälligen/strukturierten Schaltungen (Transverse-Ising-Modell), **nicht** SRE der
  gelernten Gewichtsschicht eines Klassifikators als *Prädiktor* für den Magiebedarf
  der Aufgabe (das ist genau I-002, dort H0). Andere Fragerichtung (Magie schätzen vs.
  Magie vorhersagen aus Aufgabenmerkmalen).
- **Nonstabilizerness Estimation using Graph Neural Networks**, arXiv:2511.23224 —
  gleiches Muster, GNN statt Random Forest, Schaltungs-Graph als Eingabe.
- **No Reference-Free Generalization in Quantum Machine Learning**, arXiv:2606.22331
  *(WebFetch-Volltext-Zusammenfassung gelesen)* — behandelt einen ganz anderen
  Generalisierungsbegriff (fehlender Referenzrahmen im Hilbertraum bei zu wenig
  Trainingszuständen); **keine** Verbindung zu Magie/Nicht-Clifford-Ressourcen. Für
  I-035 (Generalisierungslücke als Magiebedarfs-Prädiktor) **nicht** einschlägig —
  bestätigt eher, dass diese Verbindung in der Literatur fehlt.

---

## 2. Einstufung der Kernaussagen des Instituts

| Kernaussage | Einstufung | Beleg / Begründung |
|---|---|---|
| **Magie-Konzentration** (wenige Magiewinkel tragen den Nutzen einer trainierten Klassifikator-Schaltung, gemessen als G(k)/M50/M90) | **Nichts Vergleichbares gefunden** für Bildklassifikation mit Amplitudenkodierung + Ry/CNOT. Verwandt: Magie-Konzentrations-*Verteilungen* in Vielteilchen-/Zufallsschaltungen (arXiv:2502.20455, arXiv:2502.09725), aber andere Systemklasse, kein "Nutzenanteil einer ML-Aufgabe". Aussage bleibt eine plausible Neuheitskandidatin — Unsicherheit: nur Abstracts/Snippets, keine Volltextsuche in Konferenzbänden (ICML/NeurIPS-QML-Workshops), die evtl. näher liegen. |
| **Hybrid-Kaskade** (Clifford-Stufe klassisch simuliert, magische Stufe nur bei unsicheren Bildern zugeschaltet, T-Zahl je Bild statt fest) | **Teilweise publiziert** als *Mechanismus-Genre*: Early-Exit/Kaskaden sind bekannt (BranchyNet 2016, vom Institut selbst schon so eingeordnet); Magie-getriebene Fehlerkorrektur-Kaskaden (partial QEC, ACM/ISCA 2024) sind ein verwandter, aber anderer (Fehlerkorrektur- statt Trainings-)Kontext. Die *Kombination* "Kaskadenstufe nach T-Zahl definiert, in einem gelernten Bild-Klassifikator" wurde in der Suche **nicht** gefunden — bleibt am ehesten neu, aber als Kombination bekannter Bausteine zu kennzeichnen, nicht als radikal neuer Mechanismus. |
| **Magie vs. Datenmenge** (M50 als Funktion von n_train) | **Nichts Vergleichbares gefunden.** Suche zu "Magie-Ressource + Datensatzgröße" liefert nur allgemeine QML-Datensatz-Arbeiten (MNISQ, QDataSet) ohne Magie-Bezug. Institut selbst befund H0 in Gegenrichtung (Konzentration schärft sich mit *mehr* Daten). Keine Fremdarbeit dazu gefunden. |
| **Magiebedarf-Prädiktor** (SRE oder Generalisierungslücke sagt M50 billig voraus) | **Nichts Vergleichbares gefunden**, mit einer Einschränkung: arXiv:2409.13008 verbindet Magie mit Aufgabenleistung bei Reservoir-Computing (verwandt, andere Aufgabenklasse — vom Institut selbst schon so markiert). Die neu gefundenen SRE-Schätzungsarbeiten (2509.16799, 2511.23224) lösen ein anderes Problem (SRE aus Messdaten schätzen, nicht Magiebedarf aus Aufgabenmerkmalen vorhersagen). "No Reference-Free Generalization..." (2606.22331) bestätigt indirekt die Lücke: dort wird Generalisierung diskutiert, aber ohne jeden Magie-Bezug. |

---

## 3. Vorschläge: Fragen ohne gefundenes Vergleichbares

Reihenfolge nach Institutsvorgabe: neue Dinge finden > neue Zusammenhänge erschließen > Bekanntes nachrechnen.

1. **Magie-Konzentration als Aufgabenkennzahl, aufgabenübergreifend kalibriert**
   (Erweiterung von I-003/C-mnist/C-fashion auf ein drittes, strukturell anderes
   Klassifikationsproblem, z. B. Parität/synthetische Aufgabe mit kontrolliertem
   "Schwierigkeitsknopf"). Q-Stufe: **Q3** (echte Ressourcenkennzahl G(k)/M50/M90 an
   einer echten Amplituden-Encoding-Schaltung). CPU-Machbarkeit: grob, ähnlich C-mnist/
   C-fashion (~1h je Datensatz, 5–8 Kerne) — machbar, aber Rechenlast, nicht während des
   laufenden Prozesses starten.

2. **Kaskadenstufen, definiert über T-Zahl statt Schichttiefe, mit ehrlicher
   Hardware-Buchhaltung inkl. Ladekosten je Stufe** (I-015/I-034 als eigenständige,
   sauber abgegrenzte Linie, explizit gegen partial-QEC-Kaskaden (ACM/ISCA 2024)
   abgegrenzt: dort Fehlerkorrektur-Zweck, hier Trainings-/Inferenz-Ressourcenzweck).
   Q-Stufe: **Q3**. CPU-Machbarkeit: mittel (~1h, wie im Institut bereits budgetiert für
   I-015/I-034) — bereits größtenteils gerechnet, hier nur die *Abgrenzungsfrage neu*.

3. **Magiebedarf als Funktion der Datenmenge, mit Gegenrichtung als eigenständigem
   Befund publizierbar** (I-024: M50 wächst mit mehr Daten statt zu schrumpfen — das
   Gegenteil der naiven Erwartung "wenig Daten → wenig Magie nötig"). Da nichts
   Vergleichbares gefunden wurde, ist der bereits gemessene H0-in-Gegenrichtung-Befund
   selbst die Neuheit, nicht nur eine offene Frage. Q-Stufe: **Q3**. CPU-Machbarkeit:
   bereits gerechnet (I-024 laut `agenda.json` erledigt) — hier nur die
   Publikations-/Framing-Frage, kein neuer Lauf nötig.

4. **Dritter Magiebedarf-Prädiktor: Verschränkungs-/Korrelationsstruktur der Rohdaten**
   (Brücke zu V-kompass, SVD-Verschränkungs-Kompass) statt SRE (I-002, H0) oder
   Generalisierungslücke (I-035, H0) — z. B. Schmidt-Rang/Entanglement-Entropie der
   Bilddaten selbst (vor dem Training) als billiger Kennwert für M50. In der Literatur
   nicht gefunden (SRE-Schätzungsarbeiten schätzen Magie aus dem Zustand, nicht aus
   Dateneigenschaften vor dem Training). Q-Stufe: **Q3** (Schmidt-Zerlegung einer
   echten Amplitudenkodierung). CPU-Machbarkeit: günstig — SVD auf vorhandenen
   Trainingsdaten, kein neues Modelltraining nötig, < 30 min laut V-kompass-Vorarbeit.

5. **Clifford-Rundung nach Trainingsverlust-Ranking statt Gitterabstand (I-001), explizit
   gegen Adaptive-Quantum-Fisher-Pruning (Springer 2024) und Clifford-Perturbation-
   Fehlerminderung (arXiv:2412.09518) abgegrenzt** — beide gefundenen Pruning-Verfahren
   nutzen andere Auswahlkriterien (Nähe zu 0° bzw. Fehlerminderungszweck), keines rankt
   nach Verlustanstieg beim Runden auf ein Clifford-Gitter in einem ML-Klassifikator.
   Q-Stufe: **Q3**. CPU-Machbarkeit: günstig, laut `ideen.json` bereits mit ~10 min
   veranschlagt (Ziffern, vorhandene C-clifford-v2-Rohdaten, kein Neutraining).

**Unsicherheiten, die vor dem Skalieren noch zu prüfen wären:** Punkt 1 und 5 stützen
sich nur auf Abstracts/Snippets aus der automatisierten Suche, nicht auf gelesene
Volltexte aus QML-spezifischen Workshops (z. B. NeurIPS/ICML QML-Workshop-Proceedings,
die WebSearch schlechter indexiert als arXiv). Vor einem größeren Rechenlauf lohnt eine
gezielte zweite Runde auf `dblp.org` oder Google Scholar mit den Begriffen "magic
concentration" bzw. "loss-ranked Clifford rounding" — beide Wortfolgen lieferten in
dieser Suche keinen einzigen direkten Treffer, was für Neuheit spricht, aber kein Beweis
für Abwesenheit ist.

---

## Quellenliste (alle im Text zitierten Links)

- Quantum resources of quantum and classical variational methods — https://arxiv.org/pdf/2409.13008
- Non-stabilizerness of Neural Quantum States — https://arxiv.org/html/2502.09725
- A hybrid variational quantum circuit approach for stabilizer states classifiers — https://arxiv.org/pdf/2511.09430
- Optimal compilation of parametrised quantum circuits — https://arxiv.org/pdf/2401.12877
- Clifford Perturbation Approximation for Quantum Error Mitigation — https://arxiv.org/html/2412.09518
- Optimal number of parametrized rotations and Hadamard gates — https://arxiv.org/pdf/2407.07846
- Adaptive pruning algorithm using a quantum Fisher information matrix for PQCs — https://link.springer.com/article/10.1007/s42484-024-00221-9
- CAFQA: A classical simulation bootstrap for variational quantum algorithms — https://arxiv.org/abs/2202.12924
- Clifford Circuit Initialisation for Variational Quantum Algorithms — https://arxiv.org/pdf/2207.01539
- Warm Starting Variational Quantum Algorithms with Near Clifford Circuits (NCC-WS) — https://www.mdpi.com/2079-9292/12/2/347 (WebFetch 403, nur Snippet)
- Variational Quantum Algorithms in the era of Early Fault Tolerance — https://dl.acm.org/doi/10.1145/3695053.3731112
- Anticoncentration in Clifford Circuits and Beyond: Pseudo-Magic States — https://arxiv.org/pdf/2502.20455
- A Study on Stabilizer Rényi Entropy Estimation using Machine Learning — https://arxiv.org/html/2509.16799
- Nonstabilizerness Estimation using Graph Neural Networks — https://arxiv.org/html/2511.23224
- No Reference-Free Generalization in Quantum Machine Learning — https://arxiv.org/html/2606.22331
- MNISQ: A Large-Scale Quantum Circuit Dataset — https://www.nature.com/articles/s41597-026-07493-9
