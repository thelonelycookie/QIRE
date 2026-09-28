# Kurzfassung — Entwurf v2 (Stand 23. September 2026)

**Das ist ein Gerüst, kein fertiger Text.** Die Argumentationslinie folgt dem
Gesamtreview vom 23. September (`../BERICHT/00_REVIEW_BEFUNDE.md`). Die
Formulierungen musst du dir zu eigen machen — die Jury fragt genau hier nach.

Der Vorgänger `KURZFASSUNG_ENTWURF.md` (6. August) bleibt unverändert stehen.
Er ist in drei Punkten überholt: (1) Die Hauptzahl +0,0932 wurde gegen eine
Kontrolle gemessen, die schlechter ist als ein lineares Modell. (2) Verschränkung
und Tiefe standen dort als „nachgewiesen gleichwertig“, obwohl die
Positivkontrolle desselben Laufs durchgefallen ist. (3) Mehrere Zahlen waren
veraltet (B-014 vor der Korrektur, „+0,0044“ aus zwei vermischten Läufen,
„1,99×“ statt der T-001-Messung, Hardware ohne Stichprobengröße).

**Platzhalter** stehen als **[OFFEN: …]**. Sie werden erst gefüllt, wenn der
jeweilige Lauf vorregistriert gelaufen und in `q_audit` eingetragen ist.

**Zur Anmeldung:** Die Nutzung von KI-Werkzeugen im Formular angeben (wie im
Tagebuch-Kopf vermerkt).

**Fristen:** Anmeldung **30. November 2026**. Schriftliche Arbeit **Mitte Januar
2027** (eine Landesseite nennt den 17.01.2027 — beim eigenen Landeswettbewerb
prüfen). Die GPU-Ergebnisse (Q-020, T-002) müssen also erst im Januar in der
Arbeit stehen, nicht in der Anmeldung.

---

## Titel (Vorschläge)

- *Was bleibt von „quanteninspiriert“? Sieben Abgrenzungstests und eine
  Schaltung mit 16 Winkeln*
- *Quantenschaltungen als Bauteil klassischer KI — was trägt, wenn man fair
  vergleicht*

---

## Kurzfassung (Zielumfang: eine Seite)

**Fragestellung.** „Quanteninspiriert“ ist in der KI-Forschung ein verbreitetes
Etikett, das selten gegen eine faire klassische Kontrolle geprüft wird. Ich
wollte wissen, ob Prinzipien der Quantenmechanik KI tatsächlich
ressourceneffizienter machen — und wie man das so misst, dass die Antwort hält.

**Sieben Abgrenzungstests.** Ich habe sieben Mechanismen geprüft
(Born-Regel-Gating, Verschränkungsentropie, Tensor-Train, Unitarität,
DMRG-Rangzuweisung, parametrisierte Quantenschaltung, Quantenkerne), jeweils
gegen eine klassische Kontrolle, die dieselbe Wirkung ohne den Quantenanteil
erzielt. Fünf bestehen ihre Abgrenzung nicht, einer ist unentschieden
(Unitarität), einer — die Quantenschaltung — trägt den Rest dieser Arbeit. Als
gemeinsame Erklärung für mehrere Negativbefunde habe ich das
**Konzentrationsprinzip** hergeleitet: Zwei zufällige Zustände in 2ⁿ Dimensionen
überlappen im Mittel mit 1/2ⁿ. Gemessen fällt die mittlere Kernähnlichkeit von
0,349 (2 Qubits) auf 0,024 (6 Qubits); die Kernmatrix wird zur Einheitsmatrix,
und das Lernverfahren hat nichts mehr zu verallgemeinern.

**Was trägt: sehr wenige Parameter.** Eine 8-Qubit-Schaltung sitzt als
Mittelschicht zwischen einem klassischen Encoder und einem Klassifikationskopf.
Auf MNIST (15 Seeds, Lernrate je Arm) erreicht sie mit **16 trainierbaren
Winkeln** (eine Lage) 0,9337 balancierte Genauigkeit (fünf Lagen mit 80
Winkeln: 0,9339) und liegt damit vor einem klassischen Quadratmodell mit
**2.048** Parametern (0,9213;
dieser Vergleich war nicht vorregistriert). Die Kontrolle, gegen die ich
bisher den größten Abstand berichtet habe (67.848 Parameter, 0,8407), war zu
schwach: Eine Stichprobe zeigte, dass sie sogar hinter einem rein linearen
Modell liegt. Der faire Vergleich gegen ein gewöhnliches ReLU-Netz mit
derselben Engstelle läuft deshalb als eigener, vorregistrierter Lauf Q-020.
Das Ergebnis (Q-020, vorregistriert, 15 Seeds, offizieller MNIST-Testsatz mit
10.000 Bildern): Die Schaltung mit 16 Winkeln **schlägt** das ReLU-Netz mit
67.848 Mittelschicht-Parametern. Sie erreicht 0,9347 gegen 0,9234, also
**+0,0113**, und liegt in allen 15 Seeds vorn (p = 0,0001, nach Holm 0,0002).
Auf Fashion-MNIST zeigt sie in dieselbe Richtung, aber kleiner: +0,0047,
p = 0,03, nur beschreibend. Die Einsparung betrifft die Mittelschicht, nicht
das Gesamtmodell, denn der Encoder 784→256 steckt in jedem Arm.
**Woher der Vorteil kommt, ist ebenfalls gemessen:** Von den 16 Winkeln wirken
nur 8. Eine rein klassische Kronecker-Drehung aus acht 2×2-Drehungen
erreicht dieselbe Genauigkeit (0,9350; gleichwertig, TOST). Die Architektur
stammt also aus der Quantenschaltung, der Mechanismus ist klassisch.

**Mechanismus.** Die Auslese ⟨Z⟩ summiert |Amplitude|² — sie **ist die
Born-Regel** und wirkt hier als Nichtlinearität. Dieselbe Operation als Gate im
ersten Abgrenzungstest brachte nichts, weil ihr Ergebnis danach wieder auf
Summe 1 normiert wurde. Die **Position im Netz** entscheidet, nicht die
Operation. Ohne Verschränkungsgatter ist die Schaltung ein Kronecker-Produkt
U₁⊗…⊗U₈ aus Ein-Qubit-Drehungen; zusätzliche Lagen bringen dann nichts Neues.
Mit Verschränkung ändert sich das, aber nur, wenn kein trainierbarer Encoder
die Arbeit übernimmt. Ohne Encoder, auf einer Aufgabe mit 128 Haufen (Q-020 b),
gilt:
- Der CNOT-Ring bringt **+0,34**: 0,687 gegen 0,345.
- 5 Lagen mit CNOT bringen gegenüber einer Lage **+0,33**.

Beides gilt in 15 von 15 Seeds. Mit Encoder sind beide Effekte nachweislich
null (TOST), und daher kam der frühere Nullbefund. Ehrlich dazu gehört: Auf
dieser Aufgabe liegen klassische Netze bei 1,000, die beste Schaltung bei
0,687. Dass die Winkel selbst tragen, sobald kein Encoder aushilft, war schon
belegt: trainiert gegen eingefroren +0,4077 (15/15 Seeds, Q-018).

**Effizienz — getrennt von der Qualität.** Klassisch umgesetzt ist die
Schaltung ein Butterfly-Netz mit O(d log d) Operationen und über jede
Qubit-Ebene geteilten Winkeln: 140 Zahlen bei 16.384 Dimensionen statt 268 Mio.
Ab **16.384** Dimensionen ist es auf der CPU in der Inferenz schneller als eine
dichte Schicht gleicher Breite (1,59-fach in der Referenzumsetzung, 2,31-fach
in der optimierten). Der Übergang entsteht an der **Cache-Kante** der dichten
Schicht (Zeitsprung ×9,1 von 8.192 auf 16.384), nicht durch die
Butterfly-Struktur. Trainierbar ist auf meiner Hardware nur Breite 256 —
Qualität und Effizienz sind deshalb an verschiedenen Breiten gemessen und
werden getrennt berichtet.

**Zweite Anwendung: ein Sprachmodell.** In einem kleinen Transformer (d_model
256, 5,3 Mio. Parameter, deutsche Wikipedia) ersetze ich die FFN- bzw.
Q/K/V-Schichten durch die Butterfly-Struktur (Strukturteil im FFN-Arm 324 statt
2,1 Mio. Parameter). **[OFFEN: T-002 — Validierungsverlust dense gegen
butterfly_ffn / butterfly_qkv, 8 Seeds, gleiche Gradientenschritte; Auflösung
dense gegen narrow; Chatbot-Demo.]**

**Methodik als eigener Beitrag.** Aus eigenen Fehlern sind sieben
Protokollregeln entstanden (u. a. Lernrate je Arm mit gleichem Suchbudget,
Äquivalenztest statt „nicht signifikant“, Positivkontrolle). Ein Vortest auf
synthetischen Daten hat vor den Läufen neun Fehler gefunden; Auflösungsprüfungen
haben zwei Läufe als nicht verwertbar erkannt. Die jüngste Lehre: **Auch die
Kontrolle muss gegen eine triviale (lineare) Referenz geprüft werden.**
Zurückgenommene Aussagen bleiben mit Begründung im Tagebuch stehen.

**Einordnung.** Die Schaltung selbst — geteilte Winkel, klassisch als
Butterfly umgesetzt — ist nicht neu; eine praktisch gleiche Schaltung ist
bereits als Adapter für große Sprachmodelle veröffentlicht (Quantum-PEFT,
ICLR 2025, laut `LITERATUR_UND_ALLEINSTELLUNG.md`). Mein Beitrag ist nicht die
Schaltung, sondern die vorregistrierte Abgrenzung und die Zerlegung, *woher*
ihr Vorteil kommt. **[OFFEN: Formulierung nach Lesen der Quellen durch Malte —
bisher nur Abstracts ausgewertet.]**

**Grenzen.** Alle Messungen auf CPU und kleinen Datensätzen (MNIST und
Fashion-MNIST, je 2.800 Trainings- und 1.200 Testbeispiele aus einer
Teilstichprobe, nicht der offizielle Testsatz). Gleichwertigkeit gilt nur
innerhalb ±0,005. Auf Fashion-MNIST ist die Streuung etwa viermal so hoch wie auf
MNIST. Der Lauf auf einem echten IBM-Quantenprozessor (`ibm_fez`) umfasste nur
20 Testbeispiele je Schaltung und ein anderes Modell (Winkelkodierung, 4 bzw.
6 Qubits) als die Kernaussage.

---

## Kurzbeschreibung für das Anmeldeformular (Entwurf, Zeichengrenze im Formular prüfen)

Ich prüfe, ob Prinzipien der Quantenmechanik KI ressourceneffizienter machen.
Sieben „quanteninspirierte“ Mechanismen habe ich gegen faire klassische
Kontrollen getestet; die meisten halten nicht, und ein hergeleitetes
Konzentrationsprinzip erklärt, warum. Was trägt: Eine Quantenschaltung mit nur
16 Winkeln erreicht als Schicht eines Bildklassifikators die Genauigkeit
klassischer Schichten mit tausendfach mehr Parametern; ein vorregistrierter
Lauf prüft das gegen eine starke Referenz. Als klassisches Butterfly-Netz
umgesetzt ist die Struktur ab 16.384 Dimensionen schneller als eine dichte
Schicht. Zweite Anwendung ist ein kleines deutsches Sprachmodell. Ein
Schwerpunkt ist die Messmethodik selbst.

*(„tausendfach“ steht dort nur, wenn Q-020 die Gleichwertigkeit gegen das
ReLU-Netz bestätigt. Sonst: „mit 128-fach weniger Parametern als ein
klassisches Quadratmodell“.)*

---

## Belegtabelle — woher jede Zahl der Kurzfassung kommt

| Zahl | Quelle | Status |
|---|---|---|
| 0,9337 (16 Winkel) / 0,9339 (80) / 0,9213 (2.048) / 0,8407 (67.848), MNIST, 15 Seeds | `results/experiments/Q-003-replication__20260805_020957.json`; Tagebuch 5. Aug. nachts | Rohdaten, `q_audit` |
| 16 Winkel vor dem Quadratmodell (+0,0123, p=0,0003, MNIST) | Tagebuch „Gesamtbilanz“, Nachtrag 5./6. Aug. | **nicht vorregistriert**; Fashion nicht signifikant |
| Kontrolle schlechter als linear (norm_deep 0,8544, linear 0,8873, ReLU 0,9246) | `../BERICHT/00_REVIEW_BEFUNDE.md` §1 | **Stichprobe, KEIN Befund** (3 Seeds, nicht vorregistriert). Nur als Anlass für Q-020 nennen |
| 128-fach, rund 4.200-fach | 2.048/16 und 67.848/16 | gilt nur, wenn Q-020 bestätigt |
| Encoder 784→256, 8 Kanäle, 2.800/1.200 Beispiele | Review §1 (`max_samples=4000`, `test_size=0.30`) | aus dem Code |
| Kernähnlichkeit 0,349 / 0,024 | Tagebuch 3. Aug. (K-Serie), Herleitung 6. Aug. | **nur im Tagebuch, keine Rohdaten** — K-Serie wiederholen oder so kennzeichnen |
| Born-Regel als Nichtlinearität, Position entscheidet | Tagebuch 6. Aug. | Deutung; der Anteil „87 %/92 %“ ist gegen die schwache Kontrolle gerechnet und steht deshalb **nicht** in v2 |
| Kronecker-Struktur ohne CNOT | Review §2 (Gruppenargument) | Theorie |
| +0,4077, 15/15 (Q-018) | `results/experiments/Q-018__20260810_163543.json`; Tagebuch 10. Aug. | Rohdaten, `q_audit`, Auflösungsprüfung bestanden |
| 140 Zahlen bei 16.384; dichte 268 Mio. | Tagebuch 5. Aug. nachts | Rechnung |
| 1,59× / 2,31× bei 16.384; Sprung ×9,1 | `results/experiments/T-001__*.json`; Tagebuch 10. Aug. | Rohdaten, sieben Messreihen |
| d_model 256, 5,3 Mio. Parameter | T-000, Tagebuch 10. Aug. | Rohdaten |
| Strukturteil 324 statt 2,1 Mio. | Tagebuch 10. Aug. „Phase 3 aufgesetzt“ | Zählung |
| neun Fehler im Vortest | `HANDOFF_AKTUELL.md` §5, `CLAUDE.md` §5 | Stand 30. Aug. |
| ibm_fez, 20 Testbeispiele, Winkelkodierung 4/6 Qubits | `ROADMAP.md` Q-004b; `demo/artifacts/hardware_evidence.json` | **nur Zusammenfassung gespeichert, keine Counts** |
| Fashion-Streuung ~4× | Tagebuch 27. Aug. „Machtanalyse“ (SD 0,0123 gegen 0,0030) | aus B-007-Rohdaten |
| Quantum-PEFT (ICLR 2025) nutzt praktisch dieselbe Schaltung | `LITERATUR_UND_ALLEINSTELLUNG.md` (Agent `literatur`, 23./24. Sep.) | nur Abstract/HTML ausgewertet, von Malte nicht gelesen |

---

## Langfassung — Gliederung der schriftlichen Arbeit (Vorschlag)

1. **Einleitung und Fragestellung** — „quanteninspiriert“ als Etikett;
   Ressourceneffizienz (Parameter, Zeit, Speicher) getrennt definiert.
2. **Grundlagen** — Qubit, Amplitude, Born-Regel, Verschränkung;
   parametrisierte Schaltungen; Butterfly-Struktur und FFT-Verbindungsmuster
   (nicht: „ist die FFT“, siehe Tagebuch 27. Aug.). Stand der Forschung:
   `LITERATUR_UND_ALLEINSTELLUNG.md` (wird parallel erstellt).
3. **Methodik** — die sieben Protokollregeln mit dem Fehler, aus dem jede
   entstand; Vortest; Auflösungsprüfung; Rohdaten und `q_audit`; die neue Regel
   zur Kontrolle der Kontrolle.
4. **Sieben Abgrenzungstests** — Tabelle; Konzentrationsprinzip mit Herleitung
   und Vorhersagen (5/8/10 Qubits).
5. **Die Quantenschaltung als Schicht**
   - 5.1 Q-003-replication — und warum die Kontrolle zu schwach war
   - 5.2 Q-020 (a): faire klassische Referenz: 16 Winkel schlagen das ReLU-MLP (+0,0113), Mechanismus = Kronecker-Drehung
   - 5.3 Mechanismus: Rayleigh-Quotient, Born-Regel als Nichtlinearität,
     überlappende Auslese
   - 5.4 Q-015 (MNIST): Schaltung gegen volle Drehung, Trainierbarkeit statt
     Ausdruckskraft; Fashion-Zahl zurückgezogen
   - 5.5 Q-018: das Training trägt, sobald kein Encoder aushilft; Q-014 als
     Lehrstück
   - 5.6 Q-020 (b): Verschränkung × Tiefe ohne Encoder: beide tragen (+0,34 / +0,33), klassisch aber besser
6. **Effizienz auf klassischer Hardware** — Übergangspunkt, Cache-Kante,
   T-001, Inferenz gegen Training, Machbarkeitsgrenze.
7. **Zweite KI-Art: Sprachmodell (T-002)** **[OFFEN]** — optional Zeitreihen
   (Q-019), falls gelaufen.
8. **Echte Quantenhardware** — ibm_fez, mit Stichprobengröße und dem Hinweis auf
   das andere Modell; Bell/GHZ-Tiefeneffekt.
9. **Zurückgenommene Aussagen** — Tabelle mit Grund (siehe unten).
10. **Grenzen und Ausblick**
11. **Projekt B in einem Absatz** (falls nicht als eigenes Projekt angemeldet)

---

## Zurückgenommene oder abgeschwächte Aussagen (für Kapitel 9)

| Aussage | Stand | Grund |
|---|---|---|
| „16–80 Parameter schlagen 67.848 um +0,0932“ als Hauptaussage | abgeschwächt 23. Sep. | formal richtig, aber die Kontrolle ist schwächer als ein lineares Modell (Stichprobe); Ersatz durch Q-020 |
| „Verschränkung trägt nichts bei“ / „Tiefe trägt nichts bei“ (TOST) | zurückgenommen als Befund 23. Sep. | gemessen mit trainierbarem Encoder; Tiefe war die Positivkontrolle und ist durchgefallen (`q_summary`); Nachtest Q-020 (b) |
| „Neun Zehntel des Vorteils sind die Born-Regel“ (87 %/92 %) | zurückgenommen 23. Sep. | Anteil an einem Abstand zur schwachen Kontrolle |
| „Der klassische Nachbau schlägt die Schaltung“ (−0,0147) | zurückgenommen 5./6. Aug. | mit 15 Seeds und Lernrate je Arm +0,0126 zugunsten der Schaltung (nicht vorregistriert). Die früher genannte Zahl „+0,0044 (p=0,49)“ war aus zwei Läufen gemischt |
| Q-015 auf Fashion-MNIST (+0,0201) | zurückgezogen 27. Aug. | Kontrollarm am unteren Gitterrand, nur 1,5× des Lernraten-Artefakts |
| „Die Winkel muss man nicht trainieren“ (Q-014) | zurückgenommen 10. Aug. | trainierbarer Encoder übernahm die Arbeit; Q-018 zeigt +0,4077 |
| „Die Butterfly-Struktur ist die FFT“ | korrigiert 27. Aug. | nur dasselbe Verbindungsmuster; geteilte Winkel können die FFT nicht darstellen |
| „ab 16.384 rund doppelt so schnell (1,99×)“ | präzisiert 10. Aug. | T-001: 1,59× (Referenz) / 2,31× (optimiert) |
| „819× weniger Parameter, besser als dicht“ (Blockschicht) | zurückgezogen 3. Aug. | blinder Aufbau (Positivkontrolle p=0,23) |
| B-014-p-Werte (p=3·10⁻¹⁸⁷) | korrigiert 27. Aug. | Pseudoreplikation; Zellmittel gelten |
