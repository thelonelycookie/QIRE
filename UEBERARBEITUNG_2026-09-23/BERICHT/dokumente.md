# Bericht Agent `dokumente` — Überarbeitung vom 23./24. September 2026

Arbeitsordner: `UEBERARBEITUNG_2026-09-23\A\`. Originale in `projects\QIRE AI\A`
und `\B` sowie das Archiv wurden nicht angefasst. Keine Datei gelöscht.
Zurückgenommene Aussagen sind durchgestrichen oder mit datiertem Nachtrag
stehen geblieben. Einziger Rechenaufruf: `python -m quantum_sim.q_summary`
(nur Nachschlagen).

Grundlage: `00_REVIEW_BEFUNDE.md` §2, §4, §6, §7, §8, §9.

---

## Änderungen je Datei

### `A\KURZFASSUNG_ENTWURF_v2.md` (neu)
- Neu aufgebaut um die ehrliche Kernaussage: Frage → sieben Abgrenzungstests +
  Konzentrationsprinzip → 16 Winkel gegen 2.048 (mit Hinweis, dass die bisherige
  67.848-Kontrolle zu schwach war) → Mechanismus (Born-Regel als
  Nichtlinearität, Position, Kronecker-Struktur ohne CNOT) → Effizienz (16.384,
  1,59×/2,31×, Cache-Kante, getrennt von Qualität) → Sprachmodell → Methodik →
  Einordnung (Quantum-PEFT laut `LITERATUR_UND_ALLEINSTELLUNG.md`) → Grenzen.
- Platzhalter **[OFFEN: …]** für Q-020 (a), Q-020 (b), T-002 und die
  Literatur-Formulierung.
- Kurzbeschreibung fürs Anmeldeformular (Zeichengrenze nicht bekannt, im
  Formular prüfen), Belegtabelle mit Quelle und Status jeder Zahl,
  Gliederung der Langfassung, Tabelle zurückgenommener Aussagen.
- Bewusst **nicht** übernommen: +0,0932 als Hauptzahl, der 87-%/92-%-Anteil der
  Born-Regel (gegen die schwache Kontrolle gerechnet), „Verschränkung/Tiefe
  gleichwertig“, „1,99×“, „+0,0044“.
- Die alte `KURZFASSUNG_ENTWURF.md` ist unverändert.

### `A\STAND_UND_OFFENES.md`
- Stand 23. Sep., Fristen (Anmeldung 30.11., Arbeit Mitte Januar 2027), neuer
  Abschnitt 0 „Was sich geändert hat“.
- Hinweis, dass `qire_ai.audit` B-014/B-011/B-010/B-007/B-008/B-009 nicht prüft.
- B-014: Überschrift eingeschränkt („nicht von der Architektur“ durchgestrichen),
  offene Mängel (Code noch pseudorepliziert, Urteil widerspricht Zellmitteln,
  nur ein Rangmaß, „Minimum“ qualitativ, Literaturbezug fehlt); B-007 mit
  Machtanalyse-Hinweis.
- Q-003-replication: Zahlen bleiben, Kernaussage „+0,0932 gegen 67.848“
  durchgestrichen mit Begründung (Stichprobe ausdrücklich als KEIN Befund).
- Verschränkung/Tiefe: von ✅ auf „offen, Nachtest Q-020 (b)“ mit drei Gründen.
- Theorie: Kronecker-Struktur ergänzt; Konzentrationswerte als „nur Tagebuch“.
- T-001: 1,59×/2,31× statt 1,99×; 1,18× nur Referenzumsetzung.
- T-002: vier Fehler, Architektur-Einwand `ButterflyFFN`, Rechenbedarf.
- Q-019: `untied` entschieden, 4,6–5,5 h, Entwurfsschwächen und Chancen.
- Muss/Sollte/Entscheidungen/Mängel um Q-020, Git, IBM-Token, Hardware-Counts,
  Literatur (Dao, EUNN, BOFT, QuanTA, Bowles/Ahmed/Schuld, Thanasilp, Huang)
  ergänzt; zwei neue Regeln in §7.

### `A\HANDOFF_AKTUELL.md`
- Neuer Abschnitt 00 (Stand 23. Sep.): fünf Hauptbefunde, parallel laufende
  Arbeiten, Fristen, Verweis auf Kurzfassung v2. Alter Abschnitt 0 als „Stand
  30. August“ markiert.
- §1: Aufrufe über `.\py.cmd` (alte `.venv/...`-Angabe stimmte für diesen
  Ordner nicht, als Nachtrag vermerkt), neue Lesereihenfolge.
- §3: Belegstand mit durchgestrichener alter Fassung und neuem Stand.
- §5: die zwei neuen Regeln (Kontrolle gegen linear; Korrektur im Code, Beleg
  B-014) plus „durchgefallene Positivkontrolle entwertet Gleichwertigkeit“.
- §6: neue Nächste-Schritte-Tabelle; alte als überholt markiert. Q-019-
  Weichenstellung durchgestrichen, Nachtrag „gefallen am 27. Aug.“, 4,6–5,5 h,
  „nur untied kann FFT“ als offen, Entwurfsschwächen.
- §7: „Wiederaufnahme getestet“ durchgestrichen, vier T-002-Fehler, Architektur-
  Einwand, Rechenbedarf; Korpus-Pfad präzisiert; Git, Token, Counts, 141 GB,
  Demo unter `A\demo`.
- §8: neun statt drei Entscheidungen.

### `A\CLAUDE.md`
- Fristen ergänzt; Lesereihenfolge (Review, Kurzfassung v2, Q-020); Liste der
  historischen Dateien; „Offener Stand“ auf 23. Sep.; Platte 141 GB frei;
  T-002-Warnung; kein Git; die zwei neuen Regeln + Positivkontrollen-Regel in §4.

### `A\README.md`
- Neu gegliedert: Lesereihenfolge, Tabelle „aktuell“ und Tabelle „historisch —
  nicht als Stand lesen“ (mit jeweils aktuellem Ersatz), `demo/` aufgenommen,
  `kernaussage.py`, `ENTWURF_Q020.md`, `LITERATUR_UND_ALLEINSTELLUNG.md`
  verlinkt; `ENTWURF_ZEITREIHEN.md` nicht mehr „nicht gebaut“.

### `A\DATEN_EXTERN.md`
- „99 % voll, 29 GB frei“ durchgestrichen → 141 GB frei.
- Sicherheitshinweis `ibm_quantum_token.txt` (nicht kopieren, Malte erneuert).
- Demo jetzt unter `A\demo`.
- Neuer §4 „Was es nirgends gibt“: IBM-Counts, K-Serie-Rohdaten, Läufe ohne
  Rohdaten, kein Git.

### `A\BEFUNDE_UND_RICHTUNGEN.md`
- Kopfvermerk „nachgeführt 23. Sep.“, neue Legendenmarken.
- Parametereffizienz: Nachtrag zur schwachen Kontrolle.
- Verschränkung, Tiefe, „Verschränkung auf MNIST“: [BELEGT] → [OFFEN] mit
  ausführlichem Nachtrag.
- Status §3 korrigiert: Q-009 nicht verwertbar, Q-015 belegt (MNIST), Q-010 →
  Q-020, Q-011 abgebrochen, Q-012/16/17 aufgegeben, Q-013 → T-002.
- Übergangspunkt: T-001-Werte; Inferenz≠Training: **Widerspruch 14.400
  (BEFUNDE, Q-009-Rohdaten) gegen 14.896 (Tagebuch-Kalibrierung)** vermerkt.
- Konzentration: nur Tagebuch; „alle negativ“ präzisiert.
- Butterfly-Encoder, Gewichtsteilung über Lagen: Tiefen-Begründung gestrichen.
- Q-Audit: 106 Prüfungen, Lücken von `qire_ai.audit`, fehlende QUESTIONS.
- Sprachmodelle: [LÄUFT] → T-002-Stand. §13: Vorrang Q-020.

### `A\ROADMAP.md`, `A\HANDOFF_PROJEKT_A.md`, `A\FORSCHUNGSIDEEN.md`
- Je genau ein Hinweisabsatz oben: „Historisch, Stand …, aktuell ist …“. Text
  darunter unverändert.

### `A\PROJEKTTAGEBUCH.md`
- Neuer Eintrag „23. September — Gesamtreview: was hält, was nicht“ **vor**
  „Zwischenstand“ (dort standen bisher alle neuen Einträge). Erste Person,
  gekennzeichnet als Entwurf zur Prüfung durch Malte. Die Stichprobe ist
  ausdrücklich als „keine Messung im Sinne des Protokolls“ markiert; die
  Gegenüberstellung 0,9337 (Q-003) gegen 0,9246 (Stichprobe) ist ausdrücklich
  als unzulässige Vermischung zweier Läufe benannt. Enthält die zwei neuen
  Regeln und acht Entscheidungen für Malte.

---

## Offene Punkte

1. **Leere Streudateien.** Beim Bearbeiten sind leere Dateien ohne Endung
   entstanden, vermutlich weil ein Hook Dateiinhalte mit `>` an die Shell
   weitergibt (bekannte Falle): in `A\` die Dateien `0`, `0)`, `1`, `2`, `4}`,
   `96`, `anteil`, `bzw`, `die`, `int`, `Prüfwerkzeug` (23.09., 19:03–19:10), in
   `C:\Users\Malte\Downloads\claude\` die Dateien `0`, `1`, `20`, `24}`, `Diese`,
   `Tuple[np.ndarray`, `UT,`, `vorregistriert)`, `⚠` (24.09.). Alle 0 Byte. Ein
   Teil stammt aus meinen Markdown-Edits (Zeilen mit `>`), ein Teil offenbar aus
   Code-Edits anderer Agenten. **Nicht gelöscht** (Regel). Malte oder `main`
   sollte sie entfernen.
2. **Stichprobe gegen Q-003 nicht vergleichen.** Der Review selbst stellt
   „16 Winkel ≈ ReLU-Netz (Stichprobe)“ nebeneinander — das mischt zwei Läufe.
   Q-020 muss beide Arme im selben Lauf haben.
3. **14.896 gegen 14.400** (isochrone Breite): Quelle der Abweichung klären
   (vermutlich Rundung für Q-009).
4. **Konzentrationswerte und IBM-Counts** haben keine Rohdaten; die Kurzfassung
   v2 kennzeichnet das, die Arbeit muss es auch.
5. **Code-Stand der anderen Agenten** (B-014-Korrelation, T-002-Fehler,
   `q_summary.QUESTIONS`, `qire_ai.audit`) habe ich in den Dokumenten als „in
   Arbeit, siehe `BERICHT\`“ vermerkt, nicht als erledigt. Nach deren Abschluss
   STAND §1/§3 und HANDOFF §00/§7 nachziehen.
6. **Kurzfassung v2:** Zeichengrenze des Anmeldeformulars unbekannt; die
   „tausendfach“-Formulierung steht nur unter Vorbehalt von Q-020.
7. **Tagebucheintrag** ist ein Entwurf in Maltes Stimme — Malte muss ihn lesen
   und übernehmen.
8. Nicht angefasst, aber veraltet: `RESULTS.md`, `SCIENTIFIC_LIMITS.md`,
   `ENTWURF_ZEITREIHEN.md` (Kopf), `ENTWURF_T_SERIE_KERNEL.md`,
   `demo/README.md` (Standgespräch nennt „sieben Mechanismen, keiner hat
   gehalten“ und die Hardware ohne Stichprobengröße).
