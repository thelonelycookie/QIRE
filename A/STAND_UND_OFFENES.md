# Stand und Offenes — beide Teilprojekte

Stichpunktfassung, Stand **12. August 2026**. Anmeldung ist am **30. November**,
das sind noch **rund 15 Wochen**.

Alle mit ✅ markierten Zahlen sind durch `qire_ai.audit` bzw.
`quantum_sim.q_audit` gegen die Rohdaten nachgerechnet.

---

## 1 — Projekt B (klassisch): abgeschlossen

**Die Frage war:** Kann eine strukturierte Schicht — blockdiagonal, Monarch,
Low-Rank, Tensor-Train — eine dichte Schicht unter fairen Bedingungen ersetzen?

**Die Antwort ist nein.** Und der Wert des Teilprojekts liegt darin, dass es
*erklärt*, warum.

### Der tragende Befund

- **B-014 — der Effektivrang wird von der Aufgabe gesetzt, nicht von der Architektur** ✅
  - Kontrolliertes 6×4-Gitter: intrinsische Datendimension d gekreuzt mit Klassenzahl C
  - **Untrainiert** folgt der Rang exakt der Datendimension (ρ=+0,987, p=6·10⁻¹⁹ über die 24 Gitterzellen) und ist von der Klassenzahl **völlig unabhängig** (ρ=+0,03, p=0,90)
  - **Trainiert** hängt er von *beiden* ab (d: ρ=+0,61, p=0,002; Klassen: ρ=+0,67, p=0,0003) und liegt drastisch niedriger: **1,4–4,7 statt 3,7–165**
  - Die Klassenabhängigkeit gibt es am untrainierten Netz nicht — sie **entsteht beim Lernen**
  - Das Muster ist ein **Minimum**: die Schicht nutzt so viele Richtungen, wie der *kleinere* der beiden Zwänge zulässt
  - „Rang = Klassenzahl − 1" trifft **nicht** (bei 10 Klassen gemessen 2,7 statt 9) — also keine Neural-Collapse-Aussage
  - **Warum das alles erklärt:** Die Aufgabe braucht nur rund **sieben Richtungen**. Der Rang war nie die bindende Beschränkung — deshalb konnte keine strukturierte Familie durch mehr Kapazität gewinnen

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
  - Fashion-MNIST repliziert **nicht** (+0,0048 / −0,0045, Richtungen widersprechen sich)
  - Nach vorher festgelegtem Kriterium: **HU**
  - Auffällig und für Q-011 festgehalten: auf MNIST *wächst* der Effekt mit der Tiefe

### Zurückgezogene Aussagen — gehören in die Arbeit

- ~~„spektral-normalisierte Blockschicht, 819× weniger Parameter, besser als dicht"~~ — galt nur in einem Aufbau, der einen Faktor 32 im Budget nicht auflöst (Positivkontrolle p=0,23)
- ~~„Low-Rank Rang 64, kein messbarer Verlust"~~ — beruhte auf 10 Seeds und genäherter SVD; mit 30 Seeds und exakter SVD liegt Rang 64 bei −0,0066
- ~~„Tensor-Train ist ein bestätigter Erfolg"~~ — überholt seit A-002; eine simple Blockdiagonale schlägt TT bei gleichem Budget (p=0,004)

---

## 2 — Projekt A (Quanten): läuft

### Belegt

- **Kernaussage** ✅ — 16–80 Schaltungsparameter schlagen ein klassisches Netz mit 67.848
  - +0,0932 (MNIST), +0,0599 (Fashion), Holm p=0,0001, 15 Seeds, Lernrate je Arm
  - **Gilt für die mittlere Schicht**, nicht das Gesamtmodell (201.130 gegen 268.898)
  - ⚠ **Vorbehalt vom 27. August:** Auch dieser Lauf (Q-003-replication) hat die Lernrate mit halbem Budget gesucht. **MNIST hält und ist eher untertrieben** — `hybrid` sitzt am oberen Gitterrand, ist also untertuniert, und das verzerrt gegen den eigenen Befund (7,2× über dem Artefakt von ±0,013). **Fashion (+0,0599) ist dagegen aufgebläht**: der Kontrollarm `classical_norm_deep` sitzt am *unteren* Rand, ein untertunierter Kontrollarm vergrößert den Abstand. Hält bei 4,6×, gehört aber nur mit diesem Vorbehalt zitiert
- **Verschränkung trägt nichts bei** ✅ — ohne CNOT gleichwertig (−0,0015, TOST ±0,005). Positiver Gleichwertigkeitsnachweis
- **Tiefe trägt nichts bei** ✅ — eine Lage mit 16 Winkeln gleichwertig zu fünf mit 80 (+0,0002, TOST)
- **Q-015 — die Schaltung schlägt die volle globale Drehung** ✅ *(mit Einschränkung, 27. Aug)*
  - +0,0674 **auf MNIST**, 15/15 Seeds, Holm p=0,0001, bei **408-fach weniger Freiheitsgraden**
  - Beide universell, beide dieselbe Funktionsklasse → der Unterschied ist **Trainierbarkeit, nicht Ausdruckskraft**
  - ⚠ **Die Fashion-MNIST-Zahl (+0,0201) ist zurückgezogen.** Die Lernratensuche lief mit halbem Budget, und der Kontrollarm `global_orth` landete dort auf dem **unteren Gitterrand** (0,003 bei einem Gitter, das bei 0,003 beginnt) — ein untertunierter Kontrollarm vergrößert den Abstand, und zwar *zugunsten* der Hypothese. Bei einem Effekt vom nur 1,5-fachen des bekannten Lernraten-Artefakts (±0,013) trägt das nicht
  - Die MNIST-Zahl hält: dort hat der Mechanismus-Arm die **größere** Rate, die Verzerrung läuft also gegen die eigene Hypothese
- **Q-018 — das Training der Winkel trägt bei, sobald kein Encoder aushilft** ✅
  - +0,4077, 15/15 Seeds, Holm p=0,0001, Auflösungsprüfung bestanden
  - Repliziert auf der zweiten Schwierigkeitsstufe (+0,3646)
  - Die Verteilungen überlappen in **keinem** Paar
  - **Nimmt Q-014 zurück**: dort galt das Gegenteil, weil ein trainierbarer Encoder die Arbeit des Blocks übernehmen konnte

### Theorie, hergeleitet und numerisch geprüft

- **Konzentrationsprinzip** — E[|⟨ψ|φ⟩|²] = 1/2ⁿ aus reiner Geometrie; Messung trifft den Exponenten auf 4,2 % (−0,958 gegen −1,000), R²=0,9972
- **Die Auslese ist ein Rayleigh-Quotient** — out_q = xᵀA_qx/xᵀx mit A_q einer echten orthogonalen Projektion (Eigenwerte {0,1}), bestätigt auf 1,0·10⁻⁷
- **Alle Schaltungsfamilien sind universell**, und zwar bei **exakt dim SU(d) Parametern** (64/63 bei 3 Qubits, 256/255 bei 4) — die Teilung kostet **Tiefe, nicht Parameter**
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
  - **Verschiebt den Übergangspunkt aber nicht** — er bleibt bei 16.384
  - Grund: ein konstanter Faktor verschiebt den Schnittpunkt nur ~0,8 Oktaven; für 4.096 wären Faktor 3,1 nötig gewesen
  - Zweiter Grund: der Übergang entsteht durch die **Cache-Kante der dichten Schicht** (Sprung 9,1× von 8.192 auf 16.384), nicht durch die Butterfly-Struktur
  - **Ausstiegskriterium der Roadmap greift** — Phase 2 (Zwischendarstellung) beginnt nicht
- **T-002 — gebaut, vorregistriert, Lauf verloren**
  - Korpus fertig: deutsche Wikipedia, **397,4 Mio. Token**, BPE 4.096, **3,27 Zeichen je Token** (Englisch liegt bei ~4 — der bezifferte Preis für Deutsch)
  - Vier Arme: dense, butterfly_ffn, butterfly_qkv, narrow (Auflösungsprüfung)
  - 8 Seeds (5 wären zu wenig: 4/2ⁿ muss unter 0,05 liegen), gleiche Gradientenschritte statt gleicher Zeit
  - **Auf GPU ist die Struktur 36-fach langsamer** statt 5,9-fach auf CPU — der Nachteil verschlimmert sich auf der Hardware, die für KI zählt
  - **Der Pod wurde terminiert, bevor die Ergebnisse geholt waren. Rohdaten verloren.**

---

## 4 — Was noch aussteht

### Muss

- [ ] **T-002 neu fahren** — ~22 h GPU, mit **Rücksicherung nach jedem Seed** und Netzwerk-Volume statt pod-gebundenem Datenträger
- [ ] **Zweite KI-Art: Zeitreihen und Signale (Q-019)** — die Butterfly-Struktur hat *dasselbe Verbindungsmuster* wie die schnelle Fouriertransformation; die einzige Anwendung mit erwartetem *inhaltlichem* Vorteil statt nur Effizienzvorteil
  - Modul gebaut: `quantum_sim/zeitreihen.py`. Vortest bestanden (8 Szenarien × 5 Ziehungen, Marke **und** Zweig geprüft), Bauteile verifiziert
  - Sättigungsprüfung gefahren: nichts gesättigt, Auflösungsabstand `dense` − `narrow` = +0,128
  - **Die Kontrolle trägt:** auf permutierter Zeitachse fallen `butterfly` (0,14) und `fft_fest` (0,15) auf Rateniveau, `dense` (0,92) und `narrow` (0,80) bleiben unberührt. Der Beitrag der Struktur hängt also nachweisbar an der Periodizität
  - **Offene Weichenstellung:** Auf periodischen Signalen liegt `dense` (0,91) vor `fft_fest` (0,80) und `butterfly` (0,66, Lernrate noch nicht ausgereizt). Selbst die *echte* Fouriertransformation schlägt die dichte Schicht also nicht — die `<Z_q>`-Auslese trägt Bandinformation nur in drei ihrer acht Größen. P2 („butterfly schlägt dense auf periodisch") ist damit vermutlich unerreichbar; der Primärtest (Wechselwirkung) dagegen überwältigend
- [ ] **Q-011 mitnehmen** (Unitarität in der Tiefe, Tiefen 1/3/8/16) — bester Kandidat für einen bestandenen Abgrenzungstest, Lauf ist gebaut
- [ ] **Schriftliche Arbeit und Anmeldung** — 30. November

### Sollte

- [ ] **Literaturkarten nachziehen** — in `sources/literature_cards/` pauschal als ungelesen markiert, tatsächlich ~drei Viertel gelesen. Ohne das keine sauberen Zitate
- [ ] **K-Serie wiederholen** — die vier Konzentrationswerte stammen aus dem Tagebuch, nicht aus Rohdaten. Ein Lauf würde sie belegen *und* die Vorhersagen für 5/8/10 Qubits testen (0,0448 / 0,0061 / 0,0016)
- [ ] **B-010s Rätsel** — warum gewinnt Blockdiagonal bei n=200, wenn es nicht an der Parameterzahl liegt?

### Aufgegeben (bewusst, für die Zeit)

- Q-010, Q-012, Q-016, Q-017 als eigenständige Läufe — gebaut und vorregistriert, bleiben liegen
- Kompilierter Butterfly-Kernel — eigene Hypothese mit bekannter Zielmarke (Faktor 3), braucht eigene Vorregistrierung

---

## 5 — Entscheidungen, die nur du treffen kannst

- [ ] **Aufteilung A/B für die Anmeldung** — bis zu drei Projekte gleichzeitig erlaubt
- [ ] **`B-003` ist doppelt vergeben** — DMRG-Rangzuweisung (2. Aug) und Blockschicht-Vergleich (3. Aug). Die Rohdaten gehören dem Blockschicht-Lauf; der DMRG-Lauf braucht eine neue Kennung
- [ ] **Echter Lesestand der Literatur**

---

## 6 — Bekannte Mängel, die in die Arbeit gehören

- **12 genannte Experimente haben keine Rohdaten**: A-001, A-002, A-003, B-002, C-001, Q-001, Q-002, Q-010, Q-012, Q-013, Q-016, Q-017 → entweder neu rechnen oder als „nicht mehr reproduzierbar" kennzeichnen. *(Q-011 ist seit dem 27. August aus dieser Liste heraus — der Lauf schreibt Rohdaten nach jeder Tiefe.)*
- **Alle Zeitmessungen gelten nur für CPU.** Monarchs eigentliches Argument ist Hardware-Effizienz auf GPUs — auf diesem Rechner nicht prüfbar
- **Q-009s `train_seconds` ist kontaminiert** (Fremdlast während des Laufs). Genauigkeiten sind nicht betroffen
- **Energieschätzung** beruht auf angenommener CPU-Wattzahl, keine Hardwaremessung
- **Datensätze sind klein**, Übertragbarkeit auf große reale Daten ungeprüft
- **Qualität und Effizienz sind an verschiedenen Breiten gemessen** — trainierbar ist d_model 256, die Struktur gewinnt ab 16.384. Faktor 64. Der Chatbot demonstriert Qualität, **nicht** Effizienz

---

## 7 — Was dieses Projekt methodisch auszeichnet

Nicht als Eigenlob, sondern weil es in die Arbeit gehört:

- **Sieben Protokollregeln**, aus eigenen Fehlern entstanden — Lernrate je Arm, genug Seeds, TOST statt p>0,05, Holm nur über die angemeldete Familie, Rohdaten nach jedem Datensatz, Positivkontrolle, Kriterien vor dem Lauf
- **Auflösungsprüfungen**, die tatsächlich feuern — Q-009 und Q-014 wurden dadurch als nicht verwertbar erkannt statt als Ergebnis berichtet
- **Zwei Prüfwerkzeuge**, die jede Tagebuchzahl aus *einer* Datei nachrechnen (`q_audit`: 106 Prüfungen; `qire_ai.audit` für die B-Serie)
- **Vortests auf synthetischen Daten vor jedem langen Lauf** — haben bisher gefunden: fehlender Urteilszweig (Q-014), zu schwere Aufgabe (Q-018), gesättigte Arme (Q-018), Absturz im Lernratenplan (T-002)
- **Zurückgenommene eigene Aussagen bleiben mit Begründung stehen**, statt gelöscht zu werden
