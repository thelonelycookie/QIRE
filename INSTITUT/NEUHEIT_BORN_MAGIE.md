# Neuheitsprüfung QIRE: Born-Auslese (A) und Magie als Kompressionsmaß (B)

Stand: 07.10.2026. Nicht committet.

## Wichtige Einschränkung der Methodik

- WebFetch war in dieser Umgebung für arxiv.org, export.arxiv.org, ar5iv, api.semanticscholar.org, scholar.google.com, alphaxiv und huggingface per Egress-Proxy gesperrt.
- Die Abstracts der nächsten Treffer stammen deshalb aus Zusammenfassungen der WebSearch-Ergebnisse, nicht aus eigenem Volltextlesen.
- Die Vorwärtszitate (Semantic-Scholar-citations-Endpoint) konnten NICHT geprüft werden. Das ist eine offene Lücke.
- Vor der Abgabe sollten die genannten Treffer manuell auf arxiv.org gelesen und "Cited by" auf Semantic Scholar / Google Scholar kontrolliert werden.
- Alle URLs unten kamen tatsächlich als Suchergebnis zurück. Die URLs der im Auftrag genannten Nachbarn (Quantum-PEFT etc.) sind nur zitiert, soweit sie in Ergebnissen auftauchten.

---

## Aussage A: Born-Auslese (|psi|^2) vs. lineare Auslese, gleiche Schaltung, gleiche Parameterzahl

**Urteil: TEILWEISE bekannt.** Das Prinzip "quadratische Born-Auslese vs. lineare Auslese" ist in der Literatur 2026 bereits als Ablation vorhanden, aber in Klassifikations-Köpfen und Sequenzmodellen, nicht für Butterfly-Schichten mit Ry/Rz + CNOT-Ring und <Z>-Auslese. Neu bleibt nur die Kombination (Butterfly-Struktur, Winkelteilung, gleiche Parameterzahl, Ablation).

### Nächste Verwandte

1. **Complex-Valued Unitary Representations as Classification Heads for Improved Uncertainty Quantification in Deep Neural Networks** (2026). https://arxiv.org/pdf/2602.15283
   - Merkmale werden in einen komplexen Hilbertraum projiziert, normiert, mit einer gelernten Cayley-Unitären transformiert und entweder per Magnitude-Softmax oder per Born-Regel gelesen. Die Magnitude-Softmax-Auslese ist deutlich besser kalibriert als die Born-Auslese (2,4-fach geringerer ECE auf CIFAR-10 gegenüber Standard-Softmax).
   - Abgrenzung: Das ist ein Vergleich Born vs. Magnitude-Linear im Klassifikationskopf (Fokus Kalibrierung), keine Butterfly-/Ry-Rz-CNOT-Schicht. Wichtig: Dort schneidet Born NICHT besser ab, bei Kalibrierung sogar schlechter. QIRE muss das als Gegenbefund diskutieren.

2. **Deep Sequence Modeling with Quantum Dynamics: Language as a Wave Function** (2026). https://arxiv.org/pdf/2602.22255
   - Streng unitäre Dynamik (Cayley-Diskretisierung), Token-Wahrscheinlichkeit per Born-Regel mit gelernten Messvektoren. Argumentiert algebraisch: Born-Ausgabe hängt von N² Merkmalen des Zustands ab, linear nur von N, und liefert quadratischen Zugriff auf Phaseninformation.
   - Abgrenzung: Sprachmodell mit Wellenfunktion, Ablation Born vs. linear laut Snippet vorhanden; keine Butterfly-Struktur, keine Parametergleichheit über Qubit-Winkelteilung.

3. **The Quantum Shortcut: Complex Phase-State Dynamics Reduce the Optimization Steps of Sequence Models** (2026). https://arxiv.org/pdf/2608.14691
   - Komplexe Phasenzustands-Dynamik für Sequenzmodelle; aus dem Snippet nur Titel und Kontext (Born-Rule-Ausgabe) ersichtlich, Inhalt nicht verifiziert.
   - Abgrenzung: gleiche Richtung wie Treffer 2, keine Butterfly-Schicht (nicht im Volltext geprüft).

Weitere Randtreffer:
- Quantum Neuron Born Machine, Nichtlinearität im Born-Maschinen-Rahmen (3x kleinerer Fehler bei ähnlicher Parameterzahl; Vergleich ist aber QNBM vs. QCBM, nicht Born vs. linear): https://paperswithcode.com/paper/introducing-non-linearity-into-quantum
- Unitary Evolution RNN (lineare Auslese, Re/Im konkateniert): https://arxiv.org/pdf/1511.06464

### Fazit A
Die Aussage "Born-Auslese trägt messbar bei" ist als allgemeine Idee nicht neu, und die Evidenz in der Nachbarliteratur ist gemischt (Kalibrierung: linear besser). Neu ist die kontrollierte Ablation auf einer Butterfly-Schicht mit gleicher Parameterzahl. Formulierung deshalb vorsichtig, keine Priorität für das Born-Prinzip beanspruchen.

---

## Aussage B: Magie als Kompressionsmaß (wenige Nicht-Clifford-Winkel tragen den Nutzen, Rest auf k·pi/2 runden)

**Urteil: TEILWEISE bekannt (Einzelbausteine bekannt, die Kernaussage als Kompressionsmaß für klassisch gerechnete quanteninspirierte Schichten nicht gefunden).** Clifford-Winkel für Initialisierung und Pruning nicht-Clifford-Winkel sind bekannt; "Magie/T-Zahl als Kompressionsmaß einer trainierten Schicht, Rest multiplikationsfrei gerundet" fand ich nicht.

### Nächste Verwandte

1. **CAFQA: A classical simulation bootstrap for variational quantum algorithms / Clifford Ansatz For Quantum Accuracy** (Ravi, Gokhale et al., 2022). https://arxiv.org/abs/2202.12924v2 (Variante: https://par.nsf.gov/biblio/10433426-cafqa-classical-simulation-bootstrap-variational-quantum-algorithms)
   - Hardwareeffizienter Ansatz nur mit Clifford-Gattern, Parameter per Bayes-Optimierung im diskreten Clifford-Raum gefunden, als Startpunkt für VQE (ca. 99 % Genauigkeit über Hartree-Fock, 2,5x schnellere Konvergenz).
   - Abgrenzung: Clifford als Startpunkt (Initialisierung) vor dem kontinuierlichen Training, Ziel Energie/VQE. QIRE B: Clifford-Rundung NACH dem Training als Kompressionsmaß, Magie-Winkelzahl als Messgröße, Ziel Ressourceneffizienz. Umgekehrte Richtung.

2. **Clifford Circuit Initialisation for Variational Quantum Algorithms** (Cheng, Khosla, Self, Lin, Li, Medina, Kim, 2022). https://arxiv.org/pdf/2207.01539
   - Simulated Annealing über Clifford-Parameterpunkte als Vor-Optimierung für niedrige Anfangsenergie; Wirkung hängt von Hamiltonian-Struktur, Qubitzahl, Tiefe ab, nicht beliebig skalierbar.
   - Abgrenzung: wie CAFQA, Initialisierung für VQE. Keine Magiezählung, keine Kompression.

3. **The Stabilizer Bootstrap of Quantum Machine Learning with up to 10000 qubits** (2024). https://arxiv.org/pdf/2412.11356
   - Stabilizer-Bootstrap für Quantum Neural Networks: Optimierung im Clifford-Raum vor dem Quantenlauf; "starke/schwache Stabilizer-Verbesserung" je nach Observablen und Datenmenge, Simulation bis 10000 Qubits.
   - Abgrenzung: QML-Klassifikation mit Clifford-Pre-Training (am nächsten an Aussage B inhaltlich), aber wieder Bootstrap/Initialisierung, nicht Rundung einer trainierten Schicht und keine Magie-als-Kompressionsmaß-Aussage.

Weitere relevante Treffer:
- **QAdaPrune: Adaptive Parameter Pruning For Training Variational Quantum Circuits** (Kulshrestha et al., 2024). https://arxiv.org/html/2408.13352v1 — adaptives Parameter-Pruning (Parameter nahe 0 entfernen) für VQCs, vergleichbare Leistung. Das ist Pruning, nicht Clifford-Rundung; Abgrenzung: QIRE rundet auf k·pi/2, statt nur auf 0.
- **Reducing the number of non-Clifford gates in quantum circuits** (ZX-Kalkül, Kissinger/van de Wetering, PRA 2020). https://isidore.co/misc/Physics%20papers%20and%20books/Zotero/storage/RXBN2THR/PhysRevA.102.html — T-Zahl-Reduktion durch Umschreiben, inkl. Parameter-Elimination für variationale Schaltungen; äquivalenzerhaltend, nicht lernend/approximativ.
- **T Count as a Numerically Solvable Minimization Problem** (2026). https://arxiv.org/pdf/2603.25101 — T-Zahl-Minimierung; Rundung von Rz-Winkeln auf Clifford/T mit Binärsuche unter Syntheseschranke (Compiler-Perspektive, nicht ML-Leistung).
- **Retrieving non-stabilizerness with Neural Networks** (2024). https://arxiv.org/pdf/2403.00919 und **Non-stabilizerness of Neural Quantum States** (2025). https://arxiv.org/html/2502.09725v1 — Magie von NQS; die Aussage "Nicht-Stabilizerness bestimmt Kompression/Darstellungseffizienz von RBMs" findet sich in https://arxiv.org/html/2603.28646v1 (Neural Quantum States in Non-Stabilizer Regimes). Das ist das nächste "Magie ↔ Kompression"-Motiv, aber für Vielteilchen-Wellenfunktionen, nicht für Netzschichten.
- **High-expressibility Quantum Neural Networks using only classical resources** (2025). https://arxiv.org/html/2506.13605v2 — Clifford-erweiterte MPS, Mittelung über vier Clifford-Winkel; Expressibilität statt Kompression.
- **Advantage for Discrete Variational Quantum Algorithms in Circuit Recompilation** (Kyriienko, Umeano, Holmes, 2025). https://arxiv.org/pdf/2510.01154 — diskrete Winkel in VQAs, hohe Magie; Kompilierung statt ML-Kompression.
- ButterflyQuant (Quantisierung über lernbare Butterfly-Givens-Winkel): https://arxiv.org/pdf/2509.09679 — Winkel kontinuierlich, keine Clifford-/Magie-Betrachtung gefunden.

### Unterschied zu CAFQA & Co (die Frage des Auftrags)
- CAFQA, Cheng et al. und Stabilizer Bootstrap suchen die Clifford-Winkel VOR dem kontinuierlichen Training als Startpunkt (Initialisierung).
- QIRE-B trainiert zuerst kontinuierlich, zählt dann die Nicht-Clifford-Winkel (Magie) und rundet den Rest auf k·pi/2 als Kompression, mit der Behauptung, dass wenige Magie-Winkel den Nutzen tragen.
- Das Ergebnis bei CAFQA (Clifford-Punkte liefern oft schon gute Lösungen) stützt B indirekt, ist aber keine Vorwegnahme. Nicht gefunden: Studie, die Post-hoc-Clifford-Rundung einer trainierten Schicht als Kompression bewertet. Das spricht für Neuheit, ist aber ein Negativbefund unter eingeschränkter Suche.
- Risiko: Die Aussage hängt an der Beobachtung, dass die Schicht klassisch (Butterfly) gerechnet wird; dort ist "Clifford" nur ein Winkelraster, kein Gatter-Vorteil. "Magie" ist dann eine Metapher für die Zahl nicht-rasterbarer Winkel. Das sollte offen gesagt werden, sonst wirkt der Magie-Begriff überzogen.

---

## Suchprotokoll (WebSearch; WebFetch gesperrt)

Aussage A:
1. quantum-inspired neural layer Born rule readout |psi|^2 ablation linear readout
2. quadratic readout measurement nonlinearity variational quantum circuit machine learning
3. Born rule activation function unitary neural network probability readout
4. "Complex-Valued Unitary Representations as Classification Heads" Born rule magnitude-softmax abstract
5. "Language as a Wave Function" Born rule readout versus linear readout quantum dynamics sequence model
6. quantum-inspired tensor network Born machine squared amplitude output versus linear output ablation classical neural network
7. linear vs quadratic observable expectation value feature map ... quadratic readout quantum neural network ablation
8. squared-magnitude output unitary RNN modReLU complex-valued network |z|^2 readout versus real part readout ablation
9. Quantum-PEFT butterfly quantum-inspired adapter Pauli parameterization Born readout
10. Cayley Unitary Adapters 2605.05914 abstract (bestätigt nur: Cayley-Adapter in gefrorenen LLM-Projektionen, Zwei-Qubit-Blöcke; Details nicht geprüft)

Aussage B:
1. CAFQA Clifford ansatz initialization variational quantum eigensolver
2. non-stabilizerness magic neural network compression Clifford quantization
3. T-count pruning variational quantum circuit rounding angles to Clifford
4. Clifford pre-training parameterized quantum circuit angles rounded to multiples of pi/2 stabilizer initialization
5. magic budget quantum machine learning non-Clifford gates trainability expressivity few T gates
6. "Clifford Circuit Initialisation for Variational Quantum Algorithms" abstract
7. "Stabilizer Bootstrap of Quantum Machine Learning" abstract Clifford
8. stabilizer rounding trained parameterized quantum circuit post-training Clifford projection accuracy loss
9. Clifford+T budget variational circuit "number of non-Clifford" parameters pruning classification performance
10. Clifford quantization neural network weights multiplication-free angles k*pi/2 butterfly Givens rotations
11. Stabilizer Rényi entropy as compression measure trained variational circuit few magic parameters
12. arXiv 2408.13352 parameter elimination non-Clifford pruning ...
13. Clifford-restricted parameterized circuit machine learning discrete angles ...

Nicht durchführbar (Egress gesperrt): Semantic-Scholar-API (Suche und citations-Endpoint), Google Scholar, arXiv-Abstract-Seiten per WebFetch. Vorwärtszitate daher ungeprüft.

---

## Formulierungsvorschläge

**A:** „Nach unserer Literatursuche (Stand 10/2026) ist die Gegenüberstellung von Born-Auslese und linearer Auslese bei unitären Modellen bereits in Klassifikationsköpfen (Cayley-Unitäre, arXiv:2602.15283) und Sequenzmodellen (arXiv:2602.22255) untersucht; die Ergebnisse sind gemischt. Wir haben keine Arbeit gefunden, die diese Ablation an einer Butterfly-Schicht aus Ry/Rz-Lagen mit CNOT-Ring bei identischer Parameterzahl durchführt. Unser Beitrag beschränkt sich auf diese kontrollierte Messung an unserer Architektur.“

**B:** „Nach unserer Literatursuche (Stand 10/2026) werden Clifford-Winkel in variationalen Schaltungen bisher als Initialisierung (CAFQA, arXiv:2202.12924; Cheng et al., arXiv:2207.01539; Stabilizer Bootstrap, arXiv:2412.11356) und die Zahl der Nicht-Clifford-Gatter als Compiler-Kostenmaß (T-Zahl-Reduktion) verwendet. Wir haben keine Arbeit gefunden, die die Zahl der Nicht-Clifford-Winkel einer trainierten, klassisch gerechneten quanteninspirierten Schicht als Kompressionsmaß vorschlägt und die übrigen Winkel nachträglich auf Vielfache von π/2 rundet. Eine vollständige Prüfung der Zitierungen der genannten Arbeiten stand aus.“

---
**Anmerkung Opus (07.10.):** Punkt „Magie ist eher Metapher“ nur halb richtig. Die Zelle *ist* eine
Quantenschaltung (Q2); die T-Zahl ist mit Qiskit synthetisiert (C-hardware: T ≈ 31·k). Für den Lauf auf
echter Hardware ist Magie die reale Kostengröße. Metaphorisch ist sie nur für die *klassische* Butterfly-
Ausführung (dort heißt Clifford = multiplikationsfrei). In der Arbeit beide Lesarten getrennt nennen.
Gegenbefund arXiv:2602.15283 (Born schlechter kalibriert) in der Diskussion von Q-020 P4 aufgreifen.
