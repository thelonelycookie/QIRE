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

## Runde 2 (Welle 4, d3)
Suchen: RG-Training, Annealing vs SGD, gradientenfreie LM-Verfahren (nur Abstracts).
- **Annealing als Training:** Ältere Monte-Carlo-Studie (Hamm/Brorsen/Hagan, Snippet, nicht eingetragen): globale Verfahren kosten mehrfach Rechenzeit, kaum Vorteil gegen lokale Suche. SA nur als SGD-Zusatz [arXiv:1906.01504], [arXiv:2107.07558]; Baselines schwach (feste LR, kein Momentum). Faires Protokoll fehlt; Mindestanforderung: gleiche Rechenzeit, getunter SGD/Adam, mehrere Seeds.
- **Gradientenfrei im LM:** MeZO [arXiv:2305.17333]: nur Vorwärtsläufe, Speichergewinn bis 12x (Selbstangabe), Gewinn ist Speicher, nicht FLOPs.
- **Renormierung/Multiskala:** MLMC-artiger Gradientenschätzer für CNNs [arXiv:2501.12739]; RG-Rahmen (AAAI 2025 Studentenabstract, nur Snippet, nicht eingetragen). Kein belegter Effizienzgewinn bei LMs.
- **Befund:** bestätigt Negativbefund aus Runde 1: Simulationsverfahren (SA, MC) schlagen SGD nicht bei Rechenzeit; Ausnahme nur Speicher (MeZO).

Sättigung Runde 2: 4 Suchen F6, 4 neue Quellen von ca. 8 Treffern (ca. 50 % neu); Feld bleibt dünn aber ohne Gegenbeleg.
Neuer Schluss: SA-Test nur mit fairer Gleiche-Zeit-Regel; als QIRE-Hebel wenig Hoffnung.

## Runde 3 (Welle 5, e2)
1 Suche (Ising/RG/Cluster-Update als Trainingsverfahren), Abstracts per arXiv-API bestätigt, ungeprüfte Zahlen.
- Neu: Ising-Maschinen-Hardware für tiefe CNNs [arXiv:2607.00170] (94,9 % CIFAR-10, Selbstangabe; Hardware, nicht CPU); Pfadabhängigkeit autoregressiver Modelle bei Ising-Kritikalität [arXiv:2408.15715]; inverse Realraum-RG mit wenigen Parametern [arXiv:2506.04016]. Alles Physikanwendung oder Spezialhardware.
- Kein Cluster-Algorithmus (Swendsen-Wang) oder Replica-Verfahren als Trainingsbeschleunigung für LMs gefunden.

Sättigung Runde 3: 1 Suche, 3 neue Quellen von ca. 8 Treffern (ca. 35 % neu), aber ohne Effizienzbezug; Negativbefund bleibt, Feld faktisch gesättigt.
