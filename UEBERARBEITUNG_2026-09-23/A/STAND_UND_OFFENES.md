# Stand und Offenes — beide Teilprojekte

Stichpunktfassung, Stand **23. September 2026** (Überarbeitung nach dem
Gesamtreview; vorherige Fassung: 12. August). Anmeldung ist am
**30. November**, das sind noch **knapp zehn Wochen**. Die schriftliche Arbeit
ist **Mitte Januar 2027** fällig (eine Landesseite nennt den 17.01.2027 — beim
eigenen Landeswettbewerb prüfen).

Alle mit ✅ markierten Zahlen sind durch `qire_ai.audit` bzw.
`quantum_sim.q_audit` gegen die Rohdaten nachgerechnet. **Einschränkung
(23. Sep.):** `qire_ai.audit` prüft nur 15 B-Zahlen — B-014, B-011, B-010,
B-007, B-008 und B-009 sind **nicht** darunter (Review §6). Die ✅ bei diesen
Befunden heißen „aus den Rohdaten nachgerechnet, im Tagebuch dokumentiert“,
nicht „automatisch geprüft“.

Grundlage der Änderungen vom 23. September: `../BERICHT/00_REVIEW_BEFUNDE.md`.
Tagebucheintrag dazu: „23. September — Gesamtreview: was hält, was nicht“.

---

## 0 — Was sich am 23. September geändert hat (Kurzform)

- **Die Kontrolle der Kernaussage war zu schwach.** `classical_norm_deep`
  (67.848 Parameter) liegt in einer Stichprobe *hinter einem linearen Modell*.
  Die Stichprobe ist **kein Befund** (3 Seeds, nicht vorregistriert). Neuer,
  fairer Lauf: **Q-020** (Entwurf `ENTWURF_Q020.md`, wird parallel erstellt).
- **Verschränkung und Tiefe sind nicht mehr „belegt“.** Beide mit trainierbarem
  Encoder gemessen; die Tiefe war die Positivkontrolle und ist durchgefallen.
  Nachtest Q-020 (b).
- **B-014:** Die Korrektur der Pseudoreplikation steht nur im Tagebuch, nicht im
  Code; das automatische Urteil widerspricht den Zellmittel-Werten.
- **T-002:** vier Fehler vor jedem GPU-Lauf zu beheben.
- **Q-019:** die `untied`-Entscheidung ist gefallen (27. Aug.), Laufzeit
  4,6–5,5 h; Entwurfsschwächen benannt.
- **Infrastruktur:** kein Git; IBM-Token im Klartext im Archiv; Hardware-Counts
  nicht gespeichert.

---

## 1 — Projekt B (klassisch): abgeschlossen

**Die Frage war:** Kann eine strukturierte Schicht — blockdiagonal, Monarch,
Low-Rank, Tensor-Train — eine dichte Schicht unter fairen Bedingungen ersetzen?

**Die Antwort ist nein.** Und der Wert des Teilprojekts liegt darin, dass es
*erklärt*, warum.

### Der tragende Befund

- **B-014 — der Effektivrang wird von der Aufgabe gesetzt** ✅ *(Überschrift
  23. Sep. eingeschränkt: bisher „…nicht von der Architektur“)*
  - Kontrolliertes 6×4-Gitter: intrinsische Datendimension d gekreuzt mit Klassenzahl C
  - **Untrainiert** folgt der Rang exakt der Datendimension (ρ=+0,987, p=6·10⁻¹⁹ über die 24 Gitterzellen) und ist von der Klassenzahl **völlig unabhängig** (ρ=+0,03, p=0,90)
  - **Trainiert** hängt er von *beiden* ab (d: ρ=+0,61, p=0,002; Klassen: ρ=+0,67, p=0,0003) und liegt drastisch niedriger: **1,4–4,7 statt 3,7–165**
  - Die Klassenabhängigkeit gibt es am untrainierten Netz nicht — sie **entsteht beim Lernen**
  - Das Muster *sieht aus wie* ein **Minimum**: die Schicht nutzt so viele Richtungen, wie der *kleinere* der beiden Zwänge zulässt. *(23. Sep.: bisher nur qualitativ. Prüfbar per Modellvergleich Rang ≈ min(a·d, b·C) gegen additive/multiplikative Modelle auf Zellmitteln, mit Kreuzvalidierung.)*
  - „Rang = Klassenzahl − 1" trifft **nicht** (bei 10 Klassen gemessen 2,7 statt 9) — also keine Neural-Collapse-Aussage
  - **Warum das alles erklärt:** Die Aufgabe braucht nur rund **sieben Richtungen**. Der Rang war nie die bindende Beschränkung — deshalb konnte keine strukturierte Familie durch mehr Kapazität gewinnen
  - ⚠ **Offene Mängel (Review 23. Sep., §6):**
    - `qire_ai/intrinsic_dimension.py` rechnet (Stand Review) **immer noch** die pseudoreplizierte Korrelation über 240 Einzelmessungen (Zeilen 192–202). Korrigiert war nur das Tagebuch. Wer die Auswertung neu startet, bekommt wieder p=3·10⁻¹⁸⁷. *(Code-Korrektur ist Teil der Überarbeitung vom 23. Sep.; ob erledigt, steht in `../BERICHT/`.)*
    - Das automatische Urteil sagt „d erklärt mehr als die Klassenzahl“; die korrigierten Zellmittel sagen trainiert d ρ=0,61, Klassen ρ=0,67. Widerspruch — es gelten die Zellmittel.
    - ~~„nicht von der Architektur“~~ — Architektur wurde in B-014 **nicht variiert** (eine MLP, Breite 4096). Die Aussage gilt für diese MLP, bis der CNN-/Mehrschicht-Lauf da ist.
    - Nur ein Rangmaß (Effektivrang). Robustheit mit weiteren Maßen zeigen (Teilnahmezahl, Entropie-Rang nach Roy/Vetterli, 90-/99-%-Energierang).
    - Literaturbezug fehlt: Neural Collapse (Papyan/Han/Donoho 2020), Low-Rank Simplicity Bias (Huh et al.), intrinsische Dimension von Repräsentationen (Ansuini et al. 2019). Ohne Einordnung wirkt B-014 wie eine Neuentdeckung von Bekanntem.

### Belegte Einzelbefunde

- **B-011 — Rang-64-Projektion einer fertig trainierten Schicht ist gleichwertig** ✅
  - TOST p<0,0001, beide Datensätze, 15 Seeds, exakte SVD
  - **Kompressions-, kein Effizienzbefund** — sie setzt das dichte Training voraus
  - Zugleich Korrektur an B-008: dort war das **falsche Spektrum** gemessen (W allein statt W gewichtet mit der Eingabeverteilung). Rang 64 erfasst **99,8 % der genutzten Varianz**, nicht 65 %
- **B-012 — der Rang kollabiert nicht, er ist sofort da** ✅
  - Effektivrang *steigt* über das Training (5,1 → 6,7 auf MNIST, ρ=+0,988)
  - Schon nach **einer** Epoche liegen 99,9 % der Ausgabeenergie in 64 Richtungen
  - Eigene Zahl korrigiert: schwellenfreier Effektivrang ist **5–7**, nicht 64
- **B-013 — ein einzelner Schnitt schlägt festen Rang** ✅
  - MNIST +0,0097, Holm p=0,0215
  - **Das Curriculum scheitert** (512→256→128→64), auf Fashion sogar signifikant schlechter als der Einzelschnitt (−0,0106, Holm p=0,0012) — vermutlich, weil jede Projektion den Adam-Zustand zurücksetzt
- **B-010 — Blockdiagonal gewinnt bei sehr wenig Daten** ✅
  - n=200: **+0,0149 (MNIST)** und **+0,0139 (Fashion)**, beide signifikant
  - Nicht durch Parameterzahl erklärbar: `lowrank_r8` hat die *wenigsten* Parameter und ist dort am schlechtesten (−0,0204)
  - **Offenes Rätsel**, konkret genug für einen eigenen Nachtest
- **B-004 — Monarchs Vorteil hängt an räumlicher Struktur** ✅
  - +0,0115 gegen Blockdiagonal bei gleichem Budget (p=0,0039)
  - Bei **permutierten Pixeln verschwindet er** (p=0,23)
- **B-007 — Unitarität gegen Spektralnorm: unentschieden**
  - MNIST: Orthogonalität schlägt Spektralnorm auf **beiden** Tiefen (+0,0025 / +0,0051, Holm p=0,0020 / 0,0001)
  - Fashion-MNIST repliziert **nicht** (+0,0048 / −0,0045, Richtungen widersprechen sich) — laut Machtanalyse vom 27. Aug. lag der Aufbau dort **unter der Nachweisgrenze** (Streuung 4× höher): „nicht messbar“, nicht „nicht vorhanden“
  - Nach vorher festgelegtem Kriterium: **HU**
  - Auffällig und für Q-011 festgehalten: auf MNIST *wächst* der Effekt mit der Tiefe

### Zurückgezogene Aussagen — gehören in die Arbeit

- ~~„spektral-normalisierte Blockschicht, 819× weniger Parameter, besser als dicht"~~ — galt nur in einem Aufbau, der einen Faktor 32 im Budget nicht auflöst (Positivkontrolle p=0,23)
- ~~„Low-Rank Rang 64, kein messbarer Verlust"~~ — beruhte auf 10 Seeds und genäherter SVD; mit 30 Seeds und exakter SVD liegt Rang 64 bei −0,0066
- ~~„Tensor-Train ist ein bestätigter Erfolg"~~ — überholt seit A-002; eine simple Blockdiagonale schlägt TT bei gleichem Budget (p=0,004)
- ~~B-014 p=3·10⁻¹⁸⁷~~ — Pseudoreplikation (27. Aug.); es gelten die Zellmittel-Werte oben

---

## 2 — Projekt A (Quanten): läuft

### Belegt

- **Q-003-replication — Zahlen** ✅, **Deutung eingeschränkt (23. Sep.)**
  - MNIST, 15 Seeds, Lernrate je Arm: Schaltung 5 Lagen **0,9339** (80 Parameter), 1 Lage **0,9337** (16), `classical_quad_free` 0,9213 (2.048), `classical_quadratic` 0,8881 (2.048), `classical_norm_deep` 0,8407 (67.848)
  - ~~**Kernaussage** — 16–80 Schaltungsparameter schlagen ein klassisches Netz mit 67.848, +0,0932 (MNIST), +0,0599 (Fashion), Holm p=0,0001~~ — **Zahl formal richtig, als Kernaussage zurückgenommen am 23. Sep.:** `classical_norm_deep` ist in einer Stichprobe (3 Seeds, KEIN Befund) schlechter als ein rein lineares Modell (0,8544 gegen 0,8873), ein ReLU-Netz mit derselben Engstelle erreicht 0,9246. Die +0,0932 messen damit vor allem die Schwäche der Kontrolle (Normierung auf Einheitslänge + Tanh-Engstelle).
  - **Was trägt:** 16 Winkel liegen auf MNIST vor dem 2.048-Parameter-Quadratmodell (+0,0123, p=0,0003, nicht vorregistriert; Fashion nicht signifikant). Der faire Vergleich gegen ein ReLU-Netz mit ~68.000 Parametern folgt in **Q-020 (a)**.
  - Gilt für die **mittlere Schicht**, nicht das Gesamtmodell (201.130 gegen 268.898). Testmenge: 1.200 Beispiele aus `max_samples=4000`, nicht der offizielle MNIST-Testsatz.
  - ⚠ Vorbehalt vom 27. August bleibt: Lernratensuche mit halbem Budget; MNIST eher untertrieben (`hybrid` am oberen Rand), Fashion aufgebläht (Kontrollarm am unteren Rand).
- ~~**Verschränkung trägt nichts bei** ✅ — ohne CNOT gleichwertig (−0,0015, TOST ±0,005). Positiver Gleichwertigkeitsnachweis~~
- ~~**Tiefe trägt nichts bei** ✅ — eine Lage mit 16 Winkeln gleichwertig zu fünf mit 80 (+0,0002, TOST)~~
  - **Beide am 23. Sep. von „belegt“ auf „offen“ gesetzt.** Gründe: (1) gemessen mit **trainierbarem Encoder** 784→256 — genau die Konstellation, die Q-018 als Artefakt entlarvt hat; (2) die Tiefe **war die Positivkontrolle** des Laufs und ist durchgefallen (+0,0002, p=0,7615); `q_summary` sagt ausdrücklich, dass jede Gleichwertigkeitsaussage aus diesem Lauf nicht steht; (3) theoretisch: ohne CNOT ist jede Tiefe ein Kronecker-Produkt U₁⊗…⊗U₈ und *kann* nichts bringen, mit CNOT braucht Universalität dagegen Tiefe — „Tiefe trägt nichts bei“ widerspricht allgemein formuliert der eigenen Universalitätsrechnung.
  - **Neuer Status:** mit trainierbarem Encoder und durchgefallener Positivkontrolle gemessen; Nachtest **Q-020 (b)** — Verschränkung × Tiefe im Q-018-Aufbau ohne Encoder.
- **Q-015 — die Schaltung schlägt die volle globale Drehung** ✅ *(mit Einschränkung, 27. Aug)*
  - +0,0674 **auf MNIST**, 15/15 Seeds, Holm p=0,0001, bei **408-fach weniger Freiheitsgraden**
  - Beide universell, beide dieselbe Funktionsklasse → der Unterschied ist **Trainierbarkeit, nicht Ausdruckskraft**
  - ⚠ **Die Fashion-MNIST-Zahl (+0,0201) ist zurückgezogen.** Die Lernratensuche lief mit halbem Budget, und der Kontrollarm `global_orth` landete dort auf dem **unteren Gitterrand** (0,003 bei einem Gitter, das bei 0,003 beginnt) — ein untertunierter Kontrollarm vergrößert den Abstand, und zwar *zugunsten* der Hypothese. Bei einem Effekt vom nur 1,5-fachen des bekannten Lernraten-Artefakts (±0,013) trägt das nicht
  - Die MNIST-Zahl hält: beide Arme im Inneren des Gitters, 5,2× über dem Artefakt. *(Die frühere Begründung „der Mechanismus-Arm hat die größere Rate“ war zu grob — maßgeblich ist die Randlage, Tagebuch 27. Aug.)*
- **Q-018 — das Training der Winkel trägt bei, sobald kein Encoder aushilft** ✅
  - +0,4077, 15/15 Seeds, Holm p=0,0001, Auflösungsprüfung bestanden
  - Repliziert auf der zweiten Schwierigkeitsstufe (+0,3646)
  - Die Verteilungen überlappen in **keinem** Paar
  - **Nimmt Q-014 zurück**: dort galt das Gegenteil, weil ein trainierbarer Encoder die Arbeit des Blocks übernehmen konnte

### Theorie, hergeleitet und numerisch geprüft

- **Konzentrationsprinzip** — E[|⟨ψ|φ⟩|²] = 1/2ⁿ aus reiner Geometrie; Messung trifft den Exponenten auf 4,2 % (−0,958 gegen −1,000), R²=0,9972. *Die vier Messwerte (0,349/0,162/0,082/0,024) stehen nur im Tagebuch, nicht in Rohdaten — siehe „Sollte“.*
- **Die Auslese ist ein Rayleigh-Quotient** — out_q = xᵀA_qx/xᵀx mit A_q einer echten orthogonalen Projektion (Eigenwerte {0,1}), bestätigt auf 1,0·10⁻⁷
- **Alle Schaltungsfamilien sind universell**, und zwar bei **exakt dim SU(d) Parametern** (64/63 bei 3 Qubits, 256/255 bei 4) — die Teilung kostet **Tiefe, nicht Parameter**
- **Ohne Verschränkung Kronecker-Struktur** *(23. Sep.)* — ohne CNOT ist jede Lagenzahl L ein Produkt von Ein-Qubit-Drehungen, also U₁⊗…⊗U₈; die Gruppe ist abgeschlossen, Tiefe kann dann nichts bringen
- **Überlappende Auslese als struktureller Vorteil** — die Gruppenauslese verschenkt eine Dimension (Effektivrang 6,99 statt 7,99)

### Nicht verwertbar

- **Q-009** — mit drei Seeds ist p=0,25 der kleinstmögliche Wilcoxon-Wert; der Primärtest lag exakt auf diesem Boden
- **Q-014** — Auflösungsprüfung durchgefallen; der trainierbare Encoder machte den Block unsichtbar. Familienzahlen (+0,0257 / −0,0243) **nicht zitierfähig**

---

## 3 — T-Serie (Kernel und Sprachmodell)

- **T-000 — Durchsatz kalibriert** ✅
  - Chatbot-Zielgröße gemessen statt geraten: **d_model 256, 5.289.472 Parameter, 282 Mio. Token in 24 h** (CPU)
- **T-001 — Urteil T0** ✅
  - `butterfly_fast.py` bringt **1,6× gratis**: bitgleiche Funktion (4,0·10⁻⁷), identische Gradienten
  - Bei 16.384 steigt der Vorsprung gegen dicht von **1,59× (Referenz) auf 2,31× (schnell)** — diese Werte ersetzen die ältere Angabe „1,99×“ vom 5. August
  - **Verschiebt den Übergangspunkt aber nicht** — er bleibt bei 16.384
  - Grund: ein konstanter Faktor verschiebt den Schnittpunkt nur ~0,8 Oktaven; für 4.096 wären Faktor 3,1 nötig gewesen
  - Zweiter Grund: der Übergang entsteht durch die **Cache-Kante der dichten Schicht** (Sprung 9,1× von 8.192 auf 16.384), nicht durch die Butterfly-Struktur
  - **Ausstiegskriterium der Roadmap greift** — Phase 2 (Zwischendarstellung) beginnt nicht
  - Der Trainingsvorteil von 1,18× (5. Aug.) ist mit der **Referenzumsetzung** gemessen; mit `butterfly_fast.py` nicht nachgemessen
- **T-002 — gebaut, vorregistriert, Lauf verloren; vier Fehler vor dem nächsten Lauf** *(Review 23. Sep., §3)*
  - Korpus fertig: deutsche Wikipedia, **397,4 Mio. Token**, BPE 4.096, **3,27 Zeichen je Token** (Englisch liegt bei ~4 — der bezifferte Preis für Deutsch)
  - Vier Arme: dense, butterfly_ffn, butterfly_qkv, narrow (Auflösungsprüfung)
  - 8 Seeds (5 wären zu wenig: 4/2ⁿ muss unter 0,05 liegen), gleiche Gradientenschritte statt gleicher Zeit
  - **Auf GPU ist die Struktur 36-fach langsamer** statt 5,9-fach auf CPU — der Nachteil verschlimmert sich auf der Hardware, die für KI zählt
  - **Der Pod wurde terminiert, bevor die Ergebnisse geholt waren. Rohdaten verloren.**
  - ⚠ **Vor jedem GPU-Lauf zu beheben** (in Arbeit, Überarbeitung 23. Sep.):
    1. Lernratensuche `LR_SUCHE_SCHRITTE = 1000` gegen Messlauf 13.000/20.000 Schritte → Regel 1(c) verletzt; Gitterverlängerung nur nach oben → Regel 1(b)
    2. Wiederaufnahme liest nur die **neueste** T-002-Datei; `runpod_start.sh` fährt vorher die Probe → der Hauptlauf beginnt still bei null. *(Die Angabe „Wiederaufnahme getestet“ vom 30. Aug. war zu optimistisch.)*
    3. Schrittzahl nicht vorregistriert: 13.000 (`runpod_t002.sh`) gegen 20.000 (`runpod_start.sh`, CLI-Standard)
    4. `trainiere_arm` speichert `"lr": LR` (Rückfallwert 3e-4) statt der benutzten Rate
  - **Architektur-Einwand (Entscheidung Malte, vor dem Lauf ins Tagebuch):** `ButterflyFFN` gibt `|Amplitude|² · 256 · scale` aus — nichtnegativ, feste Summe, ein skalarer Gewinn; der FFN-Block kann nichts vom Residualstrom abziehen. `butterfly_qkv` lag in der Probe fast gleichauf mit dense (−0,0045 nats). Vorschlag: zusätzlicher Arm außerhalb der Familie mit zentrierter Ausgabe + Gewinn je Kanal (256 Parameter)
  - Rechenbedarf: 32 Läufe + Lernratensuche, mit korrekter Suche ~30–35 GPU-h auf einer RTX 4090; parallel auf 8 GPUs ein Nachmittag

---

## 4 — Was noch aussteht

### Muss

- [ ] **Q-020 vorregistrieren und laufen lassen** — (a) faire klassische Referenz für die Kernaussage (linear, ReLU-MLP mit gleicher Engstelle, ohne Engstelle), (b) Verschränkung × Tiefe ohne Encoder. Entwurf: `ENTWURF_Q020.md`
- [ ] **T-002 neu fahren** — nach Behebung der vier Fehler; Netzwerk-Volume, Rücksicherung nach jedem Seed. Ergebnis wird erst für die schriftliche Arbeit (Januar) gebraucht
- [ ] **Zweite KI-Art: Zeitreihen und Signale (Q-019)** — Butterfly hat *dasselbe Verbindungsmuster* wie die FFT (nicht: *ist* die FFT)
  - Modul gebaut: `quantum_sim/zeitreihen.py`. Vortest bestanden, Bauteile verifiziert
  - **Fünfter Arm `untied` entschieden und eingebaut** (27. Aug. nachts, außerhalb der Familie). Laufzeit **4,6–5,5 h**, nicht 2,5–3 h
  - Vorlauf: auf permutierter Zeitachse fallen `butterfly` (0,14) und `fft_fest` (0,15) auf Rateniveau, `dense` (0,92) und `narrow` (0,80) nicht. Auf periodischen Signalen `dense` 0,913 vor `fft_fest` 0,804 und `butterfly` 0,686
  - ⚠ **Entwurfsschwächen (Review §4):** Die Primärhypothese (Wechselwirkung periodisch × permutiert) ist **fast per Konstruktion wahr** — eine feste Permutation ist eine orthogonale lineare Abbildung, die eine dichte Schicht mitlernt, eine Butterfly nicht. Sie beweist Ordnungsabhängigkeit, keinen Vorteil. P2 fällt laut Vorlauf. Die ⟨Z_q⟩-Auslese ist der Engpass (nur 3 von 8 Größen tragen Bandinformation)
  - **Chancen auf echten Vorteil:** (a) wenig Trainingsdaten (Lernkurve N=50…6.000; 40 gegen 65.536 Parameter als Induktionsbias); (b) Walsh-Hadamard statt Fourier — geteilte Butterfly ohne CNOT ist ein Kronecker-Produkt von 2×2-Drehungen und enthält die Walsh-Hadamard-Transformation exakt. Beides wäre eine neue Vorregistrierung
  - Q-019 fehlt in `q_summary.QUESTIONS` (ebenso T-000) — in Arbeit
- [ ] **Q-011 mitnehmen** (Unitarität in der Tiefe) — Tiefen 1–8 auf MNIST gerettet (225 Zeilen), Tiefe 16 und Fashion fehlen. Vor dem Neustart: Wiederaufnahme einbauen, Gitterverlängerung in beide Richtungen
- [ ] **Anmeldung** — 30. November (Kurzfassung v2: `KURZFASSUNG_ENTWURF_v2.md`)
- [ ] **Schriftliche Arbeit** — Mitte Januar 2027

### Sollte

- [ ] **Literatur nachziehen** — die Karten in `sources/literature_cards/` sind englisch, Stand 1. August, beziehen sich auf alte Module und sind pauschal als ungelesen markiert. Die wichtigste Vorarbeit fehlt: Dao et al. (Butterfly 2019), EUNN (2017), BOFT (2024), QuanTA (2024), Bowles/Ahmed/Schuld (2024), Thanasilp (2024), Huang et al. (2022). Neu: `LITERATUR_UND_ALLEINSTELLUNG.md` (wird parallel erstellt)
- [ ] **K-Serie wiederholen** — die vier Konzentrationswerte stammen aus dem Tagebuch, nicht aus Rohdaten. Ein Lauf würde sie belegen *und* die Vorhersagen für 5/8/10 Qubits testen (0,0448 / 0,0061 / 0,0016)
- [ ] **B-014 robuster machen** — CNN-/Mehrschicht-Lauf, mehrere Rangmaße, Modellvergleich für das „Minimum“
- [ ] **B-010s Rätsel** — warum gewinnt Blockdiagonal bei n=200, wenn es nicht an der Parameterzahl liegt?
- [ ] **IBM-Hardware-Counts sichern** — über die Job-IDs herunterladen, solange IBM sie hält; bisher nur die Zusammenfassung in `demo/artifacts/hardware_evidence.json`

### Aufgegeben (bewusst, für die Zeit)

- Q-010, Q-012, Q-016, Q-017 als eigenständige Läufe — gebaut und vorregistriert, bleiben liegen. *(Die Frage von Q-010 — Schaltung gegen faire Nachbildung — geht in Q-020 auf.)*
- Kompilierter Butterfly-Kernel — eigene Hypothese mit bekannter Zielmarke (Faktor 3), braucht eigene Vorregistrierung

---

## 5 — Entscheidungen, die nur du treffen kannst

- [ ] **Q-020 freigeben** — Vorregistrierung lesen, Arme und Schwellen bestätigen, **vor** dem Lauf ins Tagebuch
- [ ] **T-002-Architektur** — `ButterflyFFN` so lassen oder zusätzlicher zentrierter Arm außerhalb der Familie
- [ ] **Q-019** — so laufen lassen (Ergebnis „HU — geteilt“ absehbar) oder auf Datenknappheit / Walsh-Hadamard umbauen (neue Vorregistrierung)
- [ ] **Git einrichten** — `git init` + privates Repo; ein Commit-Zeitstempel *vor* den Rohdaten ist der stärkste Beleg einer Vorregistrierung
- [ ] **IBM-Token bei IBM neu erzeugen** und `ibm_quantum_token.txt` aus dem Archivordner entfernen (liegt im Klartext; nach der PC-Infektion gelten Zugangsdaten als kompromittiert)
- [ ] **Aufteilung A/B für die Anmeldung** — bis zu drei Projekte gleichzeitig erlaubt
- [ ] **`B-003` ist doppelt vergeben** — DMRG-Rangzuweisung (2. Aug) und Blockschicht-Vergleich (3. Aug). Die Rohdaten gehören dem Blockschicht-Lauf; der DMRG-Lauf braucht eine neue Kennung (im B-Handoff bereits als B-015 geführt)
- [ ] **Echter Lesestand der Literatur**
- [ ] **Frist der schriftlichen Arbeit** beim eigenen Landeswettbewerb prüfen

---

## 6 — Bekannte Mängel, die in die Arbeit gehören

- **12 genannte Experimente haben keine Rohdaten**: A-001, A-002, A-003, B-002, C-001, Q-001, Q-002, Q-010, Q-012, Q-013, Q-016, Q-017 → entweder neu rechnen oder als „nicht mehr reproduzierbar" kennzeichnen. *(Q-011 ist seit dem 27. August aus dieser Liste heraus — der Lauf schreibt Rohdaten nach jeder Tiefe.)* Dazu die K-Serie (Konzentrationswerte) und die IBM-Counts.
- **Alle Zeitmessungen gelten nur für CPU.** Monarchs eigentliches Argument ist Hardware-Effizienz auf GPUs — auf diesem Rechner nicht prüfbar
- **Q-009s `train_seconds` ist kontaminiert** (Fremdlast während des Laufs). Genauigkeiten sind nicht betroffen
- **Energieschätzung** beruht auf angenommener CPU-Wattzahl, keine Hardwaremessung
- **Datensätze sind klein**, Übertragbarkeit auf große reale Daten ungeprüft. MNIST-Testmenge: 1.200 Beispiele aus einer Teilstichprobe, nicht der offizielle Testsatz
- **Qualität und Effizienz sind an verschiedenen Breiten gemessen** — trainierbar ist d_model 256, die Struktur gewinnt ab 16.384. Faktor 64. Der Chatbot demonstriert Qualität, **nicht** Effizienz
- **Die Kontrolle der Kernaussage war schwächer als ein lineares Modell** (Stichprobe 23. Sep.) — gehört offen in die Arbeit, zusammen mit dem Q-020-Ergebnis
- **Hardware-Lauf (ibm_fez):** 20 Testbeispiele je Schaltung; Übereinstimmung Hardware/Simulation 17/20 (flach) und 15/20 (tief); anderes Modell (Winkelkodierung, 4/6 Qubits) als die Kernaussage (Amplitudenkodierung, 8 Qubits)
- **Kein Git** — Vorregistrierungen sind nur durch Tagebuch-Reihenfolge und Dateizeitstempel belegt

---

## 7 — Was dieses Projekt methodisch auszeichnet

Nicht als Eigenlob, sondern weil es in die Arbeit gehört:

- **Sieben Protokollregeln**, aus eigenen Fehlern entstanden — Lernrate je Arm, genug Seeds, TOST statt p>0,05, Holm nur über die angemeldete Familie, Rohdaten nach jedem Datensatz, Positivkontrolle, Kriterien vor dem Lauf
- **Zwei Regeln aus dem Review vom 23. September:**
  - *Eine Kontrolle muss selbst gegen eine triviale Referenz (linear) geprüft werden.* Sonst misst der Abstand die Schwäche der Kontrolle (Q-003-replication).
  - *Korrekturen müssen im Code landen, nicht nur im Tagebuch.* Beleg: B-014 — das Tagebuch war korrigiert, `intrinsic_dimension.py` rechnete weiter pseudorepliziert.
- **Auflösungsprüfungen**, die tatsächlich feuern — Q-009 und Q-014 wurden dadurch als nicht verwertbar erkannt statt als Ergebnis berichtet. *(Und die durchgefallene Positivkontrolle von Q-003-replication hätte schon am 5. August die Gleichwertigkeitsaussagen stoppen müssen — `q_summary` hat es gemeldet, die Dokumente haben es nicht übernommen.)*
- **Zwei Prüfwerkzeuge**, die jede Tagebuchzahl aus *einer* Datei nachrechnen (`q_audit`: 106 Prüfungen, Stand 30. Aug.; `qire_ai.audit` für die B-Serie, derzeit nur 15 Zahlen)
- **Vortests auf synthetischen Daten vor jedem langen Lauf** — haben bisher gefunden: fehlender Urteilszweig (Q-014), zu schwere Aufgabe (Q-018), gesättigte Arme (Q-018), Absturz im Lernratenplan (T-002)
- **Zurückgenommene eigene Aussagen bleiben mit Begründung stehen**, statt gelöscht zu werden
