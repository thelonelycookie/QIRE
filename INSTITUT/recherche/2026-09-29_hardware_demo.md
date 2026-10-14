# Hardware-Demo für QIRE: Recherche (Stand 29.09.2026)

Kennzeichnung: **[belegt]** = aus der genannten Quelle abgerufen; **[Schätzung]** = eigene Rechnung/Einschätzung, nicht belegt; **[unsicher]** = nicht verifiziert oder nur aus Drittquellen. Es wurde nichts auf Hardware ausgeführt, kein Konto angelegt.

## 1. Kostenlose Zugänge 2026

### IBM Quantum Open Plan
- **10 Minuten QPU-Zeit pro 28-Tage-Fenster (rollierend)**, ohne Kreditkarte. [belegt] https://quantum.cloud.ibm.com/docs/en/guides/plans-overview
- **Aktion 2026:** Wer innerhalb von 12 Monaten 20 Minuten Laufzeit genutzt hat, kann einmalig 180 Minuten für die folgenden 12 Monate freischalten (frei einteilbar, auch alles an einem Tag). Voraussetzung: Nicht-Trial-Konto. Nach Ablauf wieder 10 min/Monat. Docs nennen "ab 16. März 2026". [belegt] https://www.ibm.com/quantum/blog/open-plan-updates
- **Geräte:** `ibm_kingston` (Heron r2, 156 Qubit) für alle Open-Plan-Nutzer freigegeben; mittlerer 2-Qubit-Fehler ca. 2,03e-3 laut IBM-Blog. [belegt, gleiche Quelle]. Die Flotte umfasste Q1 2026 16 QPUs, u. a. Heron r3 und Nighthawk (Quadratgitter); welche davon im Open Plan sichtbar sind, ändert sich und ist pro Konto zu prüfen. [belegt: https://www.ibm.com/quantum/blog/whats-new-q1-2026 ; Open-Plan-Zuordnung der Geräte: unsicher, Drittseite https://www.freequantumcomputing.com/blog/ibm-quantum-free-tier-2026 nennt kingston, Eagle 127q, Heron r1 133q]
- **Ausführungsmodi:** Job- und Batch-Modus ja, **keine Sessions** im Open Plan. [belegt] https://quantum.cloud.ibm.com/docs/en/guides/execution-modes
- **Region:** Instanzen nur us-east. [belegt, plans-overview]
- **Primitives:** Qiskit Runtime `SamplerV2` (Bitstring-Zählungen) und `EstimatorV2` (Erwartungswerte). Qiskit v2.4 ist erschienen. [belegt: whats-new-q1-2026]. Detailgrenzen (Circuit-Größe, Job-Limits) im Open Plan: [unsicher, in den abgerufenen Seiten nicht spezifiziert].
- **ibm_fez:** Ob es noch verfügbar ist, konnte ich nicht belegen. [unsicher] Im Konto prüfen. Für Vergleichbarkeit mit alten QIRE-Läufen (ibm_fez, Heron r2) ist `ibm_kingston` (ebenfalls Heron r2) der naheliegende Ersatz. [Schätzung]
- Datenschutz/Regeln für Schüler (Altersgrenze, Nutzungsbedingungen): nicht recherchiert. Minderjährig (unter 18): vor Kontoanlage AGB/Elternzustimmung klären. [unsicher, offen]

### Weitere Zugänge
- **IQM Resonance:** Starter-Tarif kostenlos, bis 30 Credits/Monat auf ausgewählten IQM-Geräten; 100 Bonus-Credits über Anmeldung zur Quantum Summer School (dieses Jahr 25.-27.08., vorbei); IQM Research Award 2026: 500 Credits ab erster Runde. [belegt über Suchergebnisse; Details zu Geräten/Credit-Umrechnung nicht geprüft] https://resonance.iqm.tech/sign-up/quantumschool , https://iqm.tech/research-award-2026/ , https://iqm.tech/products/iqm-resonance/. Wie viele Schuss/Minuten 30 Credits sind: [unsicher].
- **Quantum Inspire (QuTech Delft):** kostenloser Cloud-Zugang zu Starmon-7, Spin-2+ und Tuna-5 (5 Transmons, seit Mai 2025); laut Suchtreffer kamen Tuna-9 (Dez. 2025) und Tuna-17 (Apr. 2026) hinzu. [belegt bzw. Tuna-9/17: unsicher, nur ein Treffer] https://www.quantum-inspire.com/backends/ , https://qutech.nl/2025/05/15/delft-quantum-ecosystem-launches-open-architecture-quantum-computer/ . Kleine, rauschige Geräte, aber gut für 4-5-Qubit-Cliffordtests; Kontingent-Regeln nicht geprüft.
- **AQT (Ionenfalle, Innsbruck):** kein dediziertes Gratis-Schülerportal gefunden. Zugang über Partner (Scaleway) bzw. Open Quantum; Open Quantum gibt laut Suchtreffer 50 USD Startguthaben alle 90 Tage. [unsicher, Drittangabe] https://www.aqt.eu/products/cloud-partner/ , https://www.openquantum.com/providers/aqt
- **Amazon Braket:** kein nennenswerter Gratis-QPU-Anteil; nur 1 USD/Monat Simulator-Guthaben im ersten Jahr. Für Studierende/Fakultät gibt es das Programm "AWS Cloud Credit for Research" (Antrag, i. d. R. institutionell). Für Schüler realistisch nicht. [belegt] https://aws.amazon.com/braket/quantum-computing-research/ , https://aws.amazon.com/braket/faqs/

**Fazit:** Primär IBM Open Plan (beste Kombination aus Größe, Fehlerrate, Qiskit). Quantum Inspire als kostenloser Zweitbeleg auf anderer Technologie/Topologie, sofern die Schaltung auf 5 Qubit passt (die 8-Qubit-Schaltung passt auf Tuna-9/17 nur, falls diese wirklich offen sind [unsicher]).

## 2. Was zeigt man sinnvoll in ~10 Minuten QPU-Zeit?

Wichtig: "10 Minuten" ist abgerechnete Ausführungszeit, nicht Wanduhrzeit. Kleine Schaltungen mit 4096 Schuss brauchen typischerweise Sekunden. [Schätzung; konkrete Abrechnung pro Job im Konto prüfen: `job.usage()`/Dashboard, Quelle: plans-overview verweist auf Dashboard/Workloads]. 10 Minuten reichen daher für Hunderte kleine Jobs, nicht für Zehntausende. Mit der 180-min-Aktion (erst nach 20 min Nutzung) deutlich mehr.

### Experiment A: Clifford (0 T) vs. k Nicht-Clifford-Winkel
- Eine QIRE-Schaltung mit 0 T ist reine Clifford-Schaltung. Ihre ideale Ausgabeverteilung ist klassisch effizient simulierbar (Gottesman-Knill). Auf Hardware misst man also **Abweichung von einer exakt bekannten Referenz**, nicht "Quantenvorteil". Das ist ehrlich und gut prüfbar.
- Vorschlag: gleiche 8-Qubit-Struktur (gleiche CNOT-Topologie), Variante 0: Clifford-Winkel (Vielfache von pi/2). Varianten k=1,2,4,8: k Rz-Winkel durch pi/4 (T) oder beliebige Winkel ersetzt, Rest identisch. Für k>0 ist die Simulation weiter klassisch billig (Bravyi-Gosset: Aufwand wächst etwa wie 2^(0,5 t) für Wahrscheinlichkeiten bzw. 2^(0,23 t) fürs Sampling, t = T-Zahl; bei t<=8 trivial, Statevector bei 8 Qubit ohnehin exakt). [belegt] https://arxiv.org/abs/1601.07601
- Messgrößen: (i) Total-Variation-Distanz bzw. Hellinger-Fidelity zwischen Hardware- und Idealverteilung, (ii) **Klassifikationsgenauigkeit auf einer festen Testmenge** (z. B. 200-500 Bilder) Hardware vs. Simulator, (iii) Trend über k. Erwartung [Schätzung]: Da Rz virtuell ist (Abschnitt 3), ändert sich der Hardwarefehler zwischen k=0 und k=8 kaum, weil er von den 2-Qubit-Gattern dominiert wird. Das wäre selbst ein ehrliches, lehrreiches Ergebnis, kein Misserfolg.
- Schusszahl: Für Verteilungen über 2^8=256 Bitstrings sind 4000-8000 Schuss pro Schaltung üblich; für Klassifikation reicht meist nur das Argmax über wenige Klassen-Bits (z. B. 1-4 Ausleseregister), dann genügen ca. 1000-2000 Schuss. Statistischer Fehler eines Anteils p bei N Schuss ~ sqrt(p(1-p)/N), bei N=1000 etwa 1,6 pp. [Schätzung/Standardstatistik] Die Fashion-Magiedifferenz von ~1,3 pp liegt darunter, also auf Hardware **nicht auflösbar** ohne sehr viele Bilder und Schuss. Das ehrlich sagen.

### Fehlerminderung (Runtime-Optionen)
Laut IBM-Doku: Dynamical Decoupling, Pauli-Twirling (Gates/Messung), TREX (Ausleseminderung, kleiner Overhead), ZNE (ca. 3x Overhead mit drei Rauschfaktoren), PEA, PEC (Overhead wächst exponentiell mit der Tiefe). Die Doku-Tabelle behandelt vor allem den Estimator; ob die Optionen beim Sampler gleich greifen, wurde dort nicht ausgewiesen. [belegt/unsicher] https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques
- Empfehlung [Schätzung]: DD + Gate-Twirling + Messfehler-Kalibrierung (TREX bzw. eigene Read-out-Kalibrierung). Kein ZNE/PEC im 10-Minuten-Budget. Jede Option an- und ausschalten und **beides berichten**, damit der Effekt der Minderung sichtbar bleibt.
- Für Cliffords ist ein starker Trick möglich: Clifford-Schaltungen lassen sich exakt klassisch simulieren, also eignen sie sich für Kalibrierung/Fehlerabschätzung (Clifford-Data-Regression o. ä. [nicht recherchiert, unsicher]).

### Amplitudenkodierung von 256 Pixeln
- 256 Pixel = 8 Qubit, aber ein beliebiger 8-Qubit-Zustand braucht Theta(2^n) CNOTs. Untergrenze ohne Ancillas: mindestens floor((2^n - n - 1)/2) CNOTs; Möttönen-artige Verfahren erreichen O(2^n) mit O(2^n) Tiefe; Beispiel aus einer Übersicht: 10 Qubit exakt = 2047 CNOTs. [belegt via Suchtreffer, Primärquelle prüfen] https://arxiv.org/pdf/2111.03132 , https://pennylane.ai/blog/2025/10/state-preparation/
- Für n=8 also größenordnungsmäßig **250-500 CNOTs** [Schätzung aus obigen Formeln; genaue Zahl mit `qiskit` transpilieren]. Bei 2-Qubit-Fehler ~2e-3 (IBM-Angabe für kingston) und schon vor Routing-SWAPs auf Heavy-Hex grob Erfolgswahrscheinlichkeit exp(-500*0,002) ~ 0,37, real wegen Routing, Dekohärenz und Auslese deutlich schlechter. [Schätzung] Die Amplitudenkodierung würde also das Signal verwischen, **bevor** die eigentliche Klassifikationsschaltung anfängt. Nicht für die Demo empfohlen.
- Billigere Kodierungen [Schätzung/eigene Empfehlung]: (a) **Winkelkodierung** (ein Ry(x_i) pro Qubit, Tiefe 1) auf vorab verdichteten Merkmalen (PCA/Pooling 256 -> 8 Werte); (b) Basiszustands-/Binärkodierung nach Schwellwert (X-Gatter); (c) klassische Vorverarbeitung im Kaskaden-Schritt, so dass nur die ~8 % schwierigen Bilder auf die QPU gehen (passt zu eurer Hybrid-Kaskade). Wenn QIRE bisher amplitudenkodiert und die Klassifikator-Schaltung fest ist, ist die saubere Aussage: Kodierung wird auf der Hardware zum Flaschenhals; dokumentieren, nicht verstecken.

## 3. Ehrlich: Was heißt "T-Gatter sparen" auf NISQ-Hardware?

- **Rz ist auf IBM/Transmon-Hardware virtuell:** Z-Drehungen werden als Phasenverschiebung des Ansteuerungsträgers realisiert, Dauer null, praktisch fehlerfrei; zusammen mit zwei X_pi/2-Pulsen ergibt sich jede SU(2)-Gatter. [belegt] McKay et al., "Efficient Z-Gates for Quantum Computing", https://arxiv.org/abs/1612.00858
- Folge: Ein T-Gatter (Rz(pi/4)) kostet auf so einer Maschine **nichts extra**; die Clifford-Eigenschaft bringt auf NISQ-Hardware **keinen Hardwarevorteil** bei Gattern. Kosten und Fehler stecken in den 2-Qubit-Gattern und der Auslese. "0 T" ist dort ein **Struktur-/Simulierbarkeitsmerkmal**, keine Ersparnis. [Schlussfolgerung; Einzelaussage zu allen Gerätetypen nicht belegt: bei Ionenfallen/Neutralatomen ist die Lage anders, [unsicher]]
- Gleichung kommt erst bei **fehlertoleranten** Rechnern: Cliffords sind transversal/billig, ein T-Gatter verbraucht einen destillierten Magic State; T-Gatter dominieren dort Zeit und Platz. [belegt, zusammengefasst] https://www.emergentmind.com/topics/fault-tolerant-t-gate-costs (Sekundärquelle; Primärquellen: Bravyi & Kitaev 2005, quant-ph/0403025 [nicht abgerufen], Litinski 2019 arXiv:1905.06903 [nicht abgerufen])
- Aktuell relativiert: Gidney/Shutty/Jones 2024 ("Magic state cultivation") berechnen T-Zustände mit logischen Fehlerraten um 2e-9 und eine Größenordnung weniger Qubit-Runden als frühere Verfahren; sie vermuten, weitere Destillation werde praktisch womöglich nicht nötig. [belegt] https://arxiv.org/abs/2409.17595 . Das heißt: der T-Preis sinkt, verschwindet aber nicht.
- Relevanz für QIRE: Die Aussage "0 T" ist (a) ein Simulierbarkeits-/Ressourcenbefund (Magie als Ressource; Clifford + wenige T ist klassisch simulierbar, Bravyi-Gosset), (b) eine Vorhersage für spätere fehlertolerante Hardware, (c) **keine** Laufzeit- oder Genauigkeitsersparnis auf ibm_kingston. So formulieren. Ein ehrliches Fazit lautet auch: eine reine Clifford-Klassifikatorschaltung ist klassisch billig simulierbar; der Hardware-Lauf beweist nur, dass die Schaltung auf realem Gerät reproduzierbar läuft, nicht dass sie dort nötig wäre.

## 4. Minimalplan Hardware-Demo (Ziel: bis Januar 2027)

Alle Aufwände sind [Schätzung] in Personenstunden.

1. **Konto/Regeln klären (1-2 h, Malte selbst):** IBM-Quantum-Konto (Open Plan), Alters-/AGB-Frage als Minderjähriger klären; Token nur lokal, nicht ins Repo. Vorher nichts anlegen.
2. **Schaltungen einfrieren (3-5 h):** die 8-Qubit-Clifford-Klassifikatorschaltung (MNIST, 0 T) und Fashion-Variante als Qiskit-`QuantumCircuit` exportieren; Merkmal-Kodierung wählen (Winkelkodierung empfohlen, s. o.). Varianten k = 0, 1, 2, 4, 8 durch Ersetzen einzelner Clifford-Winkel.
3. **Simulator-Referenz (3-4 h):** ideale Ergebnisse (Statevector, exakt), dazu Rauschsimulation mit dem Gerätemodell (`AerSimulator.from_backend`), Testmenge fixieren (z. B. 200 Bilder pro Datensatz), **Erwartungen und Auswertungsskript vor dem Lauf festschreiben** (Protokoll-Check laut Projektregeln).
4. **Transpilation prüfen (2-3 h):** auf kingston transpilieren, 2-Qubit-Zahl und Tiefe je Variante berichten, gute 8-Qubit-Teilkette per Fehlerkarte wählen (`initial_layout`).
5. **Budgetplan (1 h):** Jobs ~ (Bilder x Varianten). Beispiel: 200 Bilder x 5 Varianten = 1000 Schaltungen à 1000-2000 Schuss. Wenn die Abrechnung nach kleinen Testjobs zu teuer ist, reduzieren (z. B. 100 Bilder, nur k = 0, 2, 8) oder die 180-min-Aktion nutzen (setzt vorher 20 min Nutzung voraus). Erst 1-2 Kalibrierjobs (Sekunden), Verbrauch ablesen, dann hochrechnen.
6. **Lauf (2-4 h aktiv, Batch-Modus):** SamplerV2 im Batch/Job-Modus (keine Sessions im Open Plan); je Variante einmal mit und einmal ohne DD/Twirling/Auslesekorrektur; Job-IDs, Backend-Name, Kalibrierdatum und Qiskit-Version protokollieren.
7. **Auswertung + Ehrlichkeitsteil (4-6 h):** Genauigkeit HW vs. Ideal vs. Rauschsimulation, TV-Distanz, Konfidenzintervalle; Aussage zu 1,3 pp (Fashion) nur, wenn Statistik es trägt; sonst als "nicht auflösbar" benennen. Optionaler Zusatz: dieselbe 4-5-Qubit-Teilschaltung auf Quantum Inspire.
8. **Hybrid-Kaskade (optional, +4-6 h):** nur die ~8 % schwierigen Bilder an die QPU; Kosten-Nutzen: QPU-Minuten pro Bild.

**Summe:** etwa 20-30 h Arbeit, davon QPU-Zeit voraussichtlich nur wenige Minuten [Schätzung]. Zeitplan-Vorschlag: Okt. Schritte 1-4; Nov. Schritt 5-6 (Fristen des Wettbewerbs beachten, 30.11.); Dez./Jan. Nachlauf und Schriftteil.

**Risiken:** (1) Gerätewechsel/Retirement zwischen Simulation und Lauf; (2) Hardwarefehler > Effektgröße, dann kein Unterschied sichtbar (ehrliches Ergebnis); (3) Queue-Wartezeit im Open Plan [Länge unsicher]; (4) Konto-/AGB-Frage bei Minderjährigkeit.

## Quellenliste
- IBM Plans overview: https://quantum.cloud.ibm.com/docs/en/guides/plans-overview
- IBM Open Plan updates: https://www.ibm.com/quantum/blog/open-plan-updates
- IBM What's new Q1 2026: https://www.ibm.com/quantum/blog/whats-new-q1-2026
- IBM Execution modes: https://quantum.cloud.ibm.com/docs/en/guides/execution-modes
- IBM Error mitigation: https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques
- IQM Resonance: https://iqm.tech/products/iqm-resonance/ ; https://iqm.tech/research-award-2026/
- Quantum Inspire: https://www.quantum-inspire.com/backends/
- AQT / Open Quantum: https://www.aqt.eu/products/cloud-partner/ ; https://www.openquantum.com/providers/aqt
- Amazon Braket: https://aws.amazon.com/braket/quantum-computing-research/ ; https://aws.amazon.com/braket/faqs/
- McKay et al. 2017 (virtuelle Z-Gatter): https://arxiv.org/abs/1612.00858
- Bravyi & Gosset, PRL 116, 250501 (2016): https://arxiv.org/abs/1601.07601
- Gidney, Shutty, Jones 2024: https://arxiv.org/abs/2409.17595
- Zustandspräparation/CNOT-Zahlen: https://arxiv.org/pdf/2111.03132 ; https://pennylane.ai/blog/2025/10/state-preparation/
