# 08 Echte Quantenhardware für QIRE, realistisch (Stand 05.10.2026)

Kennzeichnung: [belegt] = aus Suchauszug/Quelle; [nur Auszug] = nur Snippet gelesen, Seite nicht abrufbar (arxiv.org, ibm.com und quantumcomputingcost.com sind für WebFetch gesperrt); [EIGEN] = eigene Idee/Rechnung; [?] = unbelegt/aus dem Gedächtnis. Das WebSearch-Budget der Sitzung war nach ~16 Anfragen erschöpft; die Neuheitsprüfungen sind deshalb dünn und als vorläufig zu lesen. Die Vorarbeit `INSTITUT/recherche/2026-09-29_hardware_demo.md` wird nicht wiederholt, nur ergänzt.

## 1. Kurzfassung
- Zugang: IBM Open Plan = 10 min QPU/Monat gratis; nach 20 min Nutzung binnen 12 Monaten einmalig +180 min (frei einteilbar). `ibm_kingston` (Heron r2) offen für alle. Alles andere (IonQ, Quantinuum) kostet echtes Geld oder verlangt institutionelle Anträge: für einen Schüler realistisch nur IBM, ergänzend IQM Starter (30 Credits/Monat) und Quantum Inspire.
- Die **Amplitudenkodierung ist der Hardware-Flaschenhals, nicht die QIRE-Zelle.** Eigene Messung (qiskit 2.5.2, `StatePreparation`, 256 Zufallsamplituden, Basis cz/rz/sx/x): **377 CZ auf Heavy-Hex (best of 5 Seeds), 247 CZ bei Vollvernetzung.** Bei ~2e-3 Fehler je CZ ist die Zustandsqualität nach dem Laden bereits grob e^(-0,75)=0,5 (Heavy-Hex) bzw. 0,6 (Optimum), real schlechter.
- Clifford/T bringt auf IBM **keinen Gatterkosten-Vorteil** (Rz virtuell). Der Vorteil von „0 T" auf NISQ ist ein anderer: die Schaltung ist exakt klassisch simulierbar, damit ist **Fehlermitigation per Clifford Data Regression (CDR) fast geschenkt** ([EIGEN]-Idee 1).
- Schüsse: Ein Anteil p aus N Schüssen hat Fehler sqrt(p(1-p)/N). 256-Bin-Verteilung bei N=1000: mittlerer L1-Fehler grob 0,4 (gleichverteilt, [EIGEN]-Rechnung). Die Klassifikation hängt aber am linearen Kopf (Mittelung über viele Bins), das lässt sich am PC billig vorab quantifizieren (Idee 3).
- Kosten: IBM Pay-as-you-go ab 96 USD/min; IonQ Forte auf Braket 0,30 USD/Task + 0,08 USD/Schuss (1000 Schuss = 80,30 USD). 200 Bilder x 1000 Schuss auf IonQ = rund 16.000 USD, also ausgeschlossen.
- Ehrlich zu sagen: Hardwarelauf zeigt „läuft reproduzierbar", nicht „Vorteil". Der frühere A-Lauf (ibm_fez, 20 Testbeispiele, Winkelkodierung 4/6 Qubits, Übereinstimmung 17/20 bzw. 15/20) rechnete ein **anderes** Modell als QIRE und hat nur Aussagekraft als Prinzipbeweis.

## 2. Stand der Forschung

### 2.1 Zugänge und Preise
| Anbieter | Gratis / Preis | Quelle | Anmerkung |
|---|---|---|---|
| IBM Open Plan | 10 min/Monat; +180 min/12 Monate nach 20 min Nutzung (ab 16.03.2026); ibm_kingston (Heron r2) für alle | https://www.ibm.com/quantum/blog/open-plan-updates ; https://quantum.cloud.ibm.com/docs/en/guides/plans-overview | [belegt]. Keine Sessions, nur Job/Batch. |
| IBM bezahlt | Pay-as-you-go ab 96 USD/min (sekundengenau), Flex 72 USD/min (ab 400 min/Jahr), Premium 48 USD/min (ab 5200 min/Jahr) | https://quantumcomputingcost.com/ibm-quantum-pricing/ | [nur Auszug], Drittseite |
| IBM Geräte 2026 | Heron r3 (ibm_boston/pittsburgh): 2Q-Fehler median ~1,5e-3 (Dashboard), CLOPS 330K; Nighthawk r1/r2 (120 Qubit, Quadratgitter, 5000 bzw. 7500 2Q-Gatter) | https://thequantuminsider.com/2026/01/13/ibm-announces-nighthawk-and-latest-heron-are-now-available/ ; https://postquantum.com/industry-news/ibm-heron-r3-pittsburgh/ | [nur Auszug]. Welche davon im Open Plan sichtbar sind, im Konto prüfen. |
| IQM Resonance | Starter gratis: bis 30 Credits/Monat; Garnet (20 Qubit), Emerald (54 Qubit) zeitweise für Starter | https://iqm.tech/products/iqm-resonance/ ; https://quantumzeitgeist.com/iqm-boosts-cloud-platform-with-54-qubit-system-free-tier/ | [nur Auszug]; Credit-zu-Schuss-Umrechnung unbekannt [?] |
| IonQ (Braket) | Forte 0,30 USD/Task + 0,08 USD/Schuss | https://aws.amazon.com/braket/pricing/ | [belegt über Auszug]. Ionenfalle: Alle-zu-alle-Kopplung (günstig für 256-Amplituden-Laden, aber teuer) |
| Quantinuum Helios | 98 Qubits, 1Q-Treue 99,9975 %, 2Q-Treue 99,921 %; Abrechnung in HQC; Zugang Abo/Azure; Gratis-Kontingent für US-Forschende über QCUP | https://quantumcomputingreport.com/quantinuum-launches-helios-quantum-computer-with-industry-leading-fidelity-and-singapore-partnership/ ; https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/introduction.html | [nur Auszug]. Für deutschen Schüler: kein Schüler-Programm gefunden; kostenlos nur Syntaxprüfer/Emulator (H2-1LE-Emulator laut Drittseite). |
| Jugend forscht / Schule | Kein eigenes Quantenhardware-Programm gefunden; Wege: HPI-Onlinekurs Quantencomputing (kostenlos), Juelicher „Quantum Computing @ School"/JUNIQ-Workshops mit Zugang zu IBM und D-Wave, Betreuer an Uni/Forschungszentrum fragen | https://open.hpi.de/courses/qc-schule2024 ; https://www.fz-juelich.de/de/jsc/systeme/quantencomputing/juniq-infrastruktur/training/lehrer-fortbildung-quantum-computing-school | [nur Auszug]. Konkretes Förderprogramm für Schüler-Rechenzeit: **nicht gefunden**. Offene Frage für Etappe 2. |
| Braket Forschungscredits | Antrag, i. d. R. institutionell | 2026-09-29-Bericht | nicht für Schüler |

Zeitbudget [?]: Abgerechnet wird Ausführungszeit. Pro Schuss auf Heron grob 0,25–0,5 ms (Wiederholrate inkl. Rückstellzeit) [?, aus Gedächtnis]; 10 min ≈ 1–2 Mio. Schuss als Obergrenze minus Job-Overhead. 200 Bilder x 1000 Schuss = 200 000 Schuss ≈ 1–2 min. Kalibrierjob zuerst, `job.usage()` ablesen.

### 2.2 Fehlermitigation: was greift wofür
- **SamplerV2** unterstützt Dynamical Decoupling und Gate-Twirling (`options.dynamical_decoupling.enable`, `options.twirling.enable_gates`, `enable_measure`). ZNE/PEC/PEA sind Estimator-Optionen (resilience level), nicht Sampler [nur Auszug] https://quantum.cloud.ibm.com/docs/en/guides/configure-error-mitigation ; https://quantum.cloud.ibm.com/docs/en/guides/specify-runtime-options . Folge für QIRE: Die Born-Auslese (256 Bitstring-Zählungen) ist ein Sampler-Problem; ZNE/PEC verlangt Erwartungswerte (Estimator). **Wenn der lineare Kopf auf Observablen umformuliert wird** (jede Klassenausgabe = Erwartungswert einer diagonalen Observable sum_i w_ki |i><i|), geht der Estimator, dort aber mit Zeitkosten. [EIGEN]
- **TREX/Messtwirling**: kleiner Overhead, sinnvoll bei 8 Qubit Auslese (Ausleseverfälschung ist bei 8 Qubit nicht klein: 8 x ~1–2 %) [?].
- **ZNE ~3x, PEC exponentiell in der Tiefe** (IBM-Doku, siehe 2026-09-29-Bericht).
- **CDR** (Czarnik, Arrasmith, Coles, Cincio, Quantum 2021): trainiere ein Regressionsmodell auf Paaren (verrauschter Messwert, exakter Wert) für klassisch simulierbare Clifford-nahe Schaltungen, wende es auf die Zielschaltung an. https://arxiv.org/abs/2005.10189 [nur Auszug]. Erweiterung mit Non-Clifford-Extrapolation und Energy Sampling: https://arxiv.org/abs/2511.03556 [nur Auszug]. Endlich-Schuss-Analyse CDR vs PEC (Scavino, Juni 2026): lineares CDR hat kleinere Schussvarianz, behält aber einen Kalibrierungsboden, wenn die Rauschantwort der Trainings- und Zielschaltungen nicht übereinstimmt; PEC kann unverzerrt sein, zahlt aber Varianz. https://arxiv.org/abs/2606.21686 [nur Auszug]. **Kernbefund für QIRE: Wenn die Zielschaltung selbst Clifford ist, ist das Rauschantwort-Mismatch-Problem von CDR kleiner (gleiche Gatterstruktur), weil Trainings- und Zielschaltungen derselben Familie angehören** [EIGEN-Schluss, nicht belegt].
- Symmetrische Clifford-Twirling und Pauli-Fehlerpropagation: https://www.nature.com/articles/s41534-025-01050-9 ; https://arxiv.org/pdf/2412.01311 [nur Auszug]. Verwandt: Clapton (Clifford-unterstützte Problemtransformation für VQAs) https://arxiv.org/pdf/2406.15721 und CAFQA (Clifford-Initialisierung für VQE) [nur Auszug]. Beide nutzen Clifford **als Initialisierung/Rauschmodell für VQE**, nicht als fertiges Modell.

### 2.3 Amplitudenkodierung auf Hardware
- Allgemeine 8-Qubit-Zustandspräparation: Möttönen-artig rund 980 CNOT genannt, untere Schranke ~ (2^n - n - 1)/2 = 123; „~510 CNOT bei D=256; Fidelity nach 510 CNOT ≈ 0,21 bei 3e-3" [nur Auszug, Umfrage] https://arxiv.org/html/2606.05387 .
- **Eigene Messung** (Scratchpad `t.py`, qiskit 2.5.2, Optimierungsstufe 3): 256 Zufallsamplituden, Basis cz/rz/sx/x: **377 CZ (Heavy-Hex, 33-Qubit-Ausschnitt, bestes von 5 Transpiler-Seeds), 247 CZ (vollvernetzt)**; 2Q-Tiefe 319 bzw. 247. Echte Bilder (nichtnegativ, glatt) könnten günstiger sein; nicht gemessen. Zum Vergleich: die Clifford-Zelle (H-Schichten + CZ-Maske) braucht höchstens 28 CZ (Kanten einer Quadratischen Form auf 8 Knoten), auf Heavy-Hex plus Routing [EIGEN-Schätzung, nicht transpiliert; der Lauf mit Rang-Variation überschritt mein Zeitlimit].
- Alternative: Approximative Amplitudenkodierung in flachen parametrisierten Schaltungen https://arxiv.org/pdf/2103.13211 ; Shot-basierte Kodierung https://arxiv.org/html/2604.06135 [nur Auszug, beide nicht geprüft].
- Fehlertolerante Hardware: Allgemeine Zustandsladung ist dort **magiebehaftet** (Rotationen, nicht Clifford). Die Clifford-Zelle ist dort billig, das Laden der 256 Amplituden nicht. Siehe Idee 2.

## 3. Was das für QIRE bedeutet (ehrlich)
1. **Hardware-Demo ja, aber richtig benennen.** Sauberer Aufbau: Eingabe klassisch vorverarbeitet, **Laden entweder (a) ausgelagert (Zustand wird auf der QPU nicht amplitudenkodiert, sondern die Zelle wird auf einem Basiszustand/niedrigdimensionalen Zustand getestet), oder (b) 4–5 Qubit-Bilder (4x4 = 16 Amplituden: dort 2^4 = 16 -> ~10–20 CZ).** Volle 256-Pixel-Kodierung = 250–380 CZ = Signalverwischung; die Zelle allein ist dagegen billig.
2. **Gleichung „0 T = billiger" gilt auf IBM nicht** (virtuelles Rz, McKay 2016/17, https://arxiv.org/abs/1612.00858). Sie gilt für fehlertolerante Hardware und für klassische Simulierbarkeit. Auf NISQ-Hardware spart ein Clifford-Winkel höchstens Einzelqubit-Pulse: Ry(kπ/2) = 0–1 sx statt 2 sx (Fehler je sx ~2–3e-4 [?]), also vernachlässigbar gegen CZ (~1,5–2e-3) und Auslese.
3. Die 0-T-Aussage und Hardware sind **komplementär**: Hardware zeigt Rauschwirkung, Clifford liefert die exakte Referenz, mit der man die Hardware **kalibriert** (CDR). Das ist ein echter, nicht trivialer Nutzen.
4. Statistik: Die Fashion-Magiedifferenz (~1,3 pp) ist mit 200 Bildern auf Hardware nicht auflösbar (Standardfehler eines Anteils bei n=200: ~2,6 pp). Weder behaupten noch verstecken.
5. Token-Hygiene (aus A-Tagebuch): neuen Token erzeugen, Klartextdatei weg; **Counts jetzt speichern** (damals nur Zusammenfassung). Minderjährigen-AGB klären (offen).
6. Schusszahl-Genauigkeit (Faustformel): Fehler je Bin ≈ sqrt(p/N). Logit k = sum_i w_ki p_i: Std = sqrt(Var_p(w_k)/N). Für ein Verhältnis „Hardware-Argmax = Ideal-Argmax" braucht man N so, dass Std < Abstand der zwei besten Logits. Das ist am PC mit den echten Modellgewichten genau zu bestimmen (Idee 3).

## 4. Neue Ideen / Forschungsfragen

### Idee 1 [EIGEN]: „CDR by construction" für Clifford-Klassifikatoren
- **Frage:** Der trainierte 0-T-Klassifikator ist reine Clifford-Schaltung. Kann man daraus Fehlermitigation **ohne Extrapolation** gewinnen, weil Trainingsschaltungen (andere Clifford-Zellen/Eingaben derselben Gatterstruktur und Layout) exakt simulierbar sind **und der Zielschaltung gleichen**? Konkret: Rauschantwort-Matrix R (256x256, verrauschte Ausgabeverteilung gegen ideale) auf **den gleichen Qubit-Layout und CZ-Struktur** schätzen aus ~20–50 Clifford-Eingaben (Basiszuständen/Stabilisatorzuständen, die man ohne 377-CZ-Kodierung laden kann), dann auf echte Eingaben anwenden.
- **Neuheitsprüfung:** CDR (2005.10189) nutzt Clifford-**nahe** Trainingsschaltungen für ein **nicht-Clifford** Ziel; CAFQA/Clapton nutzen Clifford für VQE-Initialisierung. Als Klassifikator-Modell + CDR gesucht: nichts gefunden. **Aber nur ~16 Suchanfragen, Budget erschöpft; Suche zu „Clifford classifier + noise learning/Pauli-Lindblad" wiederholen.** Vorläufig: neuer Winkel, aber nahe an bekannten Methoden; ehrlich als Anwendung, nicht Methode, rahmen.
- **Billigster PC-Test (~30–60 min, ohne Hardware):** Rauschmodell (Depolarisierung je CZ 2e-3, Auslese 1–2 %, optional Qiskit-Aer-Rauschmodell; Aer ist nicht installiert, `pip install qiskit-aer`) auf die Clifford-Zelle + einfache Kodierung (4x4, 4 Qubit) anwenden; Genauigkeitsverlust gegen Modelle mit 4 Magiewinkeln und voller Winkel-Zelle messen; Mitigation: (a) keine, (b) Lineare CDR-Regression pro Klasse, (c) Pauli-Twirl+Rauschinversion. Metrik: Test-Genauigkeit, Argmax-Übereinstimmung mit Ideal.
- **Kontrolle:** gleiche Mitigation auf nicht-Clifford (4 Magie, volle Winkel); Zufalls-Clifford-Zelle statt gelernter; Shot-Rauschen-Boden getrennt ausweisen.
- **Quantenstufe:** Q2 (echte/ simulierte Quantenschaltung mit Zusatznutzen, kein Vorteilsanspruch).

### Idee 2 [EIGEN]: Fehlertoleranz-Kostenmodell des 100-Parameter-QIRE
- **Frage:** Wie verteilt sich der Ressourcenaufwand eines QIRE-Modells (Laden + Zelle + Auslese) in logischen Gattern/T-Zahl? These: **Das Laden dominiert die T-Kosten um Größenordnungen**, die Clifford-Zelle ist T-frei, die k Magiewinkel kosten k Rotationen.
- Rechenmodell [?, zu belegen]: Rotation Rz(θ) auf Genauigkeit ε kostet in Clifford+T ca. 1,15·log2(1/ε)+~9 T-Gatter mit Ancilla (Kliuchnikov-Maslov-Mosca), ~3·log2(1/ε) ohne (Ross-Selinger) [aus Gedächtnis]. Ladung von 256 Amplituden: ~255 steuerbare Rotationen (Möttönen) mit ~2^n Rotationen => T ~ 255·(1,15·log2(1/ε)+9); bei ε=1e-3 ≈ 255·21 ≈ 5400 T. Zelle mit k=4 Magiewinkeln: ~4·21 ≈ 84 T. Verhältnis ~64:1. Mit „Lade-Ersparnis" (z. B. Zustand nur von 8 Bild-Merkmalen, Winkelkodierung) sinkt das stark.
- **Konsequenz:** Die Behauptung „0 T-Gatter" ist für die **Gesamtanwendung** nur dann bedeutsam, wenn auch das Laden magiearm ist. Das bestimmt, ob die Quanteneffizienz-These trägt: der Gewinn „Rang der quadratischen Form" liegt in der Zelle, nicht im Laden.
- **Neuheitsprüfung:** Datenladekosten sind in der Literatur bekannt (Dequantisierung/QRAM-Debatte; „input problem" Aaronson 2015 [?]); ein **konkretes Kostenmodell für ein trainiertes Clifford+k-T-Modell mit Parameterzahl ~100** gesucht: nichts gefunden (Budget erschöpft, vorläufig).
- **Billigster PC-Test (~20 min):** Qiskit-Transpilation (gemessen) + Ross-Selinger-Synthese für die k Winkel (`gridsynth` per pip; sonst Formel) => Tabelle (Lade-CZ/T, Zelle-CZ/T, k, ε). Erweiterung: MNIST mit Teilladungen (Bild auf 4x4 oder 8 Haar-Wavelet-Koeffizienten verdichten) und Genauigkeit gegen T-Kosten auftragen: Pareto-Kurve „Genauigkeit gegen Gesamt-T".
- **Kontrolle:** Vergleich mit klassischem 100-Parameter-Modell (Flops); zufällig verdichtete Eingabe.
- **Quantenstufe:** Q1 (Theorie/Ressourcenzählung) bis Q2.

### Idee 3 [EIGEN]: Schuss-Genauigkeits-Kurve des Born-Kopfs, vorab am PC
- **Frage:** Wie viele Schüsse braucht ein 256-Bin-Born-Auslese + linearer Kopf (2.570 Parameter bzw. das 100-Parameter-Modell), damit Argmax-Genauigkeit gleich dem Idealwert minus 0,5 pp? Gilt die Kopfstruktur als „Schussfilter" (breite Mittelung)?
- **Neuheitsprüfung:** Schussrauschen in QML ist Standard (shot-noise Studien); für **diese** Modellfamilie gesucht: nichts gefunden (Budget). Als Methodik, nicht Entdeckung, rahmen.
- **Billigster PC-Test (~5–10 min):** gespeicherte p-Vektoren der Testmenge (ideal) -> Multinomial mit N ∈ {100, 300, 1000, 3000, 10000, 30000} -> Logits via Kopf -> Genauigkeit; 20 Wiederholungen. Danach QPU-Minuten = N·Bilder·0,25–0,5 ms [?] und IonQ-Dollar = N·0,08·Bilder.
- **Kontrolle:** nur Rauschen der Stichprobe vs. Bernoulli-Modell; Vergleich 0-T vs. 4-T vs. volles Modell (hat sich Verteilung p unterschiedlich „konzentriert"?).
- **Quantenstufe:** Q2.

### Idee 4 [EIGEN]: Rang-gegen-CZ-Kosten (Hardware-Preis des Maskenrangs)
- **Frage:** Wenn der Rang der quadratischen Form 52–78 % des Gewinns trägt, was kostet Rang r in CZ auf Heavy-Hex/Quadratgitter, und gibt es einen Rang, ab dem Hardwarefehler den Gewinn auffrisst? Ergebnis: eine „Nutzen pro CZ"-Kurve; mögliche Aussage: gute Formen niedrigen Ranges sind hardwarefreundlich.
- **Neuheitsprüfung:** Graph-Zustände/Quadratische Formen und CZ-Zahl sind bekannt; die Verbindung zu „trainierter Rang gegen Hardwarekosten" nicht gefunden (vorläufig).
- **PC-Test (~1 h):** Zufalls-/gelernte Formen mit Rang r in qiskit transpilieren (Heavy-Hex, Quadratgitter), CZ-Zahl und Tiefe; mit Simulationsrauschen kombinieren. Mein Versuch mit 5 Transpiler-Seeds je Rang war zu langsam: weniger Seeds, `optimization_level=1`, Hintergrundlauf.
- **Quantenstufe:** Q2.

### Idee 5 [EIGEN, schwach]: Hardware-robuster 0-T-Klassifikator
- Hypothese aus dem Auftrag: ein 0-T-Klassifikator sei auf echtem Gerät robuster als ein voller Winkelklassifikator. **Meine Einschätzung: auf IBM nicht zu erwarten** (CZ-Zahl identisch, Rz virtuell, nur Einzelqubit-Pulse sparen). Erwartbar ist höchstens ein Unterschied durch **Struktur** (Clifford-Rauschen ist nach Twirling Pauli-Rauschen, das exakt durch die Clifford-Struktur propagiert; in Nicht-Clifford-Zellen nicht). Als nullhypothesen-fähiger Test formulieren: gleiche CZ-Zahl, Differenz in TV-Distanz Hardware–Ideal, k = 0 gegen k = 4 gegen k = 8; erwartetes Ergebnis „kein Unterschied" ist ein ehrliches, publizierbares Teilergebnis.
- **PC-Test:** wie Idee 1 (Aer-Rauschmodell), 20 Min.

## 5. Offene Fragen für Etappe 2
1. Förderung/Zugang für Schüler: IBM Quantum Educators/Schulprogramme, Qiskit-Fellowship, Jülicher JUNIQ, HPI, Jugend-forscht-Patenunternehmen (IBM, IQM, planqc). Nichts Konkretes belegt; gezielt suchen und E-Mail an JUNIQ/Betreuer.
2. Was kostet ein Open-Plan-Job wirklich (Abrechnungsgranularität, Mindestdauer pro Job)? Erst am Konto prüfen.
3. Welche Geräte sieht Maltes Konto (Heron r3, Nighthawk, kingston)? Quadratgitter (Nighthawk) wäre für die CZ-Struktur günstiger.
4. Sampler-Optionen vs Estimator: lohnt es, den Kopf als Observablen umzuformulieren, um ZNE/PEC (Estimator) zu nutzen?
5. Hardware-Wert der Kodierung: transpilierte CZ-Zahl **echter** MNIST-Bilder (nicht Zufallsamplituden), plus approximative Kodierung mit Fehlerbudget.
6. Neuheit Idee 1/2: Literatursuche zu „Clifford classifier noise learning", „Pauli-Lindblad/Sparse Pauli Dynamics + Clifford" und „data loading T-count" wiederholen (Budget war aus).
7. Altersgrenze/AGB IBM und Elternzustimmung; neuer Token.
8. Rz-Kosten auf Ionenfallen/Neutralatomen (dort ist Rz nicht gratis): lohnt „0 T" dort als Hardwaremetrik?

## 6. Quellenliste
- IBM Open Plan updates: https://www.ibm.com/quantum/blog/open-plan-updates
- IBM Plans overview: https://quantum.cloud.ibm.com/docs/en/guides/plans-overview
- IBM What's new Q1 2026: https://www.ibm.com/quantum/blog/whats-new-q1-2026 ; Q2: https://www.ibm.com/quantum/blog/whats-new-q2-2026 (nur im Suchtreffer)
- IBM Fehlermitigation konfigurieren: https://quantum.cloud.ibm.com/docs/en/guides/configure-error-mitigation ; Optionen: https://quantum.cloud.ibm.com/docs/en/guides/specify-runtime-options
- IBM Preise (Drittseite): https://quantumcomputingcost.com/ibm-quantum-pricing/ ; Moor Insights Flex: https://moorinsightsstrategy.com/research-notes/ibms-new-flex-plan-fills-a-big-gap-in-quantum-access-pricing/
- Nighthawk/Heron r3: https://thequantuminsider.com/2026/01/13/ibm-announces-nighthawk-and-latest-heron-are-now-available/ ; https://postquantum.com/industry-news/ibm-heron-r3-pittsburgh/
- CDR: https://arxiv.org/abs/2005.10189 ; ES+NCE-CDR: https://arxiv.org/abs/2511.03556 ; CDR/PEC Schussfenster: https://arxiv.org/abs/2606.21686 ; Clapton: https://arxiv.org/pdf/2406.15721 ; symmetrisches Clifford-Twirling: https://www.nature.com/articles/s41534-025-01050-9 ; PEC Pauli-Propagation: https://arxiv.org/pdf/2412.01311
- Merkmalskodierung Übersicht: https://arxiv.org/html/2606.05387 ; Shot-basierte Kodierung: https://arxiv.org/html/2604.06135 ; Approx. Amplitudenkodierung: https://arxiv.org/pdf/2103.13211 ; Born-Maschinen und Nichtstabilität (Treffer, nicht gelesen): https://arxiv.org/html/2609.32025
- McKay et al. virtuelle Z-Gatter: https://arxiv.org/abs/1612.00858 ; Bravyi-Gosset: https://arxiv.org/abs/1601.07601 ; Magic state cultivation: https://arxiv.org/abs/2409.17595
- IQM: https://iqm.tech/products/iqm-resonance/ ; Quantinuum Helios: https://quantumcomputingreport.com/quantinuum-launches-helios-quantum-computer-with-industry-leading-fidelity-and-singapore-partnership/ ; Emulatoren: https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/introduction.html ; Braket: https://aws.amazon.com/braket/pricing/
- Schule/JuFo: https://open.hpi.de/courses/qc-schule2024 ; https://www.fz-juelich.de/de/jsc/systeme/quantencomputing/juniq-infrastruktur/training/lehrer-fortbildung-quantum-computing-school
- Repo: `UEBERARBEITUNG_2026-09-23/A/PROJEKTTAGEBUCH.md` (Z. 162–180 Bell/GHZ auf ibm_marrakesh; Z. 1103 Q-004b ibm_fez; Z. 4568–4583 Stichprobe 20/Counts nicht gespeichert/Token), `INSTITUT/recherche/2026-09-29_hardware_demo.md`
- Eigenmessung: Scratchpad `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/t.py` (nur Amplituden-Teil lief durch; flüchtig)
