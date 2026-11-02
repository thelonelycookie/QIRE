# F6: Physik-Simulation (Monte-Carlo, Tensornetz-Simulation)

Stand 02.11.2026, Welle 2 (b3). Nur Suchtreffer und Abstracts, kein Volltext.

## Landkarte
- **Klassische Simulation von Quantenschaltkreisen** (bereits im Bestand reichlich): Stabilisator-Simulation [arXiv:quant-ph/0406196], Clifford+T mit niedrigem Stabilisator-Rang [arXiv:1808.00128], Pauli-Propagation/Heisenberg-Bild [arXiv:2409.01706], MPS-Simulation, Dequantisierung von QML [arXiv:2307.06937]. Kosten wachsen mit Magie/Verschränkung; Abschneiden kleiner Terme hält es handhabbar.
- **Übertragung auf ML-Effizienz (gefunden, dünn):**
  - DMRG-artiges Training von MPO-Netzen [arXiv:2211.14657] (Zwei-Orts-Sweep, SVD-Rangwahl).
  - Metropolis-Monte-Carlo als Trainingsverfahren [arXiv:2205.07408]: Genauigkeit wie Gradientenabstieg, aber langsamer. Kein Effizienzgewinn.
  - HMC für Bayes-Netze: beschleunigte Hybride (VI+HMC, 2025, nur Suchtreffer nicht bestätigt), für ReLU-Netze ineffizient (NeurIPS 2024).
  - Stochastische Spurschätzung (Hutchinson, Hutch++) für Hessian-Spektren: Randomisierte Numerik, keine Physik-Spezifik.
  - Kontraktionspfad-Optimierung (cotengra/opt_einsum) kommt aus der Tensornetz-Simulation; Anwendung auf Deep Learning nicht belegt.
- **Keine Arbeit gefunden**, die Pauli-Propagation oder Stabilisator-Rang direkt für Netzschichten nutzt (eine gezielte Suche).

## Stärkste Baselines/Gegner für QIRE
- Gegen „Simulationsmethode → ML-Effizienz“ steht als Gegner immer die klassische Standard-Effizienz: SGD/Adam, BTT/Monarch [arXiv:2406.06248], PTQ (K1). Metropolis-MC unterliegt SGD in der Zeit [arXiv:2205.07408].
- Für die Behauptung „QIRE-Zelle ist simulierbar“ ist die Dequantisierung (v6, [arXiv:2307.06937]) schon eigener Befund.

## Wichtige Zahlen (mit geprüft/ungeprüft)
Keine verlässlichen Zahlen erhoben (nur Abstracts, keine Effizienzzahlen). Keine Eintragung als geprüft.

## Quantenbezug
- **Echt Quanten:** klassische Simulierbarkeit (Clifford, niedrige Magie, kleine Verschränkung) ist ein echter Satzbestand der Quanteninformation, und stützt I-038: ein Modell mit wenig Magie/Verschränkung ist klassisch billig, also kein Quantenvorteil.
- **Analogie:** Verschränkung ≙ Rang/Bonddimension; Magie ≙ Nichtlinearität/Nicht-Strukturiertheit; Abschneiden von Pauli-Termen ≙ Pruning.
- **Quanten-inspirierter Hebel:** „Kosten skalieren mit Magie/Verschränkung“ als Budgetprinzip: Struktur (Clifford/Hadamard, billig) plus wenige teure Nichtlinearitäten. Das ist genau die QIRE-Linie (Fastfood plus Quadrat), liegt aber ohne ML-Beleg vor.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- I-038 (gelernte QIRE-Zelle ≡ feste Mischung): konsistent mit Dequantisierung.
- I-045/I-047: keine Simulationsmethode hat bisher gleiche-FLOPs-Vorteil geliefert; stützt die ehrliche Aussage „quanten-inspiriert, kein Quantenvorteil“.

## Lücken/offene Fragen
- Schwache Datenlage: nur 3 Treffer zu „Simulation → ML-Effizienz“; Cluster-Algorithmen, Replica-Methoden, Renormierung für Training nicht gefunden.
- Ob DMRG-Sweeps bei gleichen FLOPs gegen SGD auf Sprachmodellen etwas bringen, ist unbekannt.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2, b3 | 2 neue für F6 (6 Suchen) | hoch bei wenig Substanz |

## Schlüsse und neue Wege (Agent)
1. **Negativbefund ist Befund:** Physik-Simulationsmethoden wurden kaum erfolgreich als ML-Effizienz übertragen (MC langsamer als SGD, HMC für ReLU ineffizient). Belege: [arXiv:2205.07408], NeurIPS 2024 HMC-Treffer. Langweiligste Erklärung: SGD ist für hochdimensionale Netze schlicht optimal; kein Wunder. Quantenbezug: keiner.
2. **Budgetprinzip „Kosten ∝ Magie/Verschränkung“ als Messgröße in QIRE** (z. B. Rang oder T-Zahl pro Schicht gegen Genauigkeit). Belege: Stabilisator-Simulation [arXiv:1808.00128], Pauli-Propagation [arXiv:2409.01706]. Langweiligste Erklärung: Parameter-/Rang-Budgetkurve, in K7 schon bekannt. Quantenbezug: direkt für die Kostenbegründung, Wirkung klassisch.
3. **DMRG-Sweep für adaptive Raten** (siehe K5, [arXiv:2211.14657]) als einzige konkrete Übertragung mit Substanz; kleine Vortestprobe auf Tuning-Seeds möglich. Langweiligste Erklärung: generische Rangwahl. Quantenbezug: Analogie.
