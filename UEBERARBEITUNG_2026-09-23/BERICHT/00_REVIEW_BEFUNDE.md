# Review vom 23. September 2026 — Befunde (Grundlage für alle Agenten)

Erstellt aus: Code, Rohdaten, `q_audit`, `q_summary`, `qire_ai.audit` (im Archiv),
Tagebuch, Handoffs, Kurzfassung. Jede Aussage hier ist nachgeprüft; wo etwas nur
eine Stichprobe ist, steht das dabei.

---

## 1. Kernaussage von Projekt A — die klassische Kontrolle ist zu schwach

Q-003-replication (Rohdaten `results/experiments/Q-003-replication__20260805_020957.json`):

| MNIST, 15 Seeds | bal. Genauigkeit | Mittelschicht-Parameter |
|---|---|---|
| hybrid (Schaltung, 5 Lagen) | 0,9339 | 80 |
| hybrid_1layer | 0,9337 | 16 |
| classical_quad_free | 0,9213 | 2.048 |
| classical_quadratic (blockorthogonal) | 0,8881 | 2.048 |
| classical_norm_deep | 0,8407 | 67.848 |

Architektur aller Arme: Encoder 784→256 (trainierbar) → Mittelschicht → **8 Kanäle** → Kopf 8→10.
`max_samples=4000` heißt: 2.800 Training, 1.200 Test (`test_size=0.30`), nicht der offizielle MNIST-Testsatz.

**Stichprobe vom 23.09. (3 Seeds 2301–2303, gleicher Lader, gleiche Trainingsschleife,
Lernrate je Arm auf denselben Seeds gewählt — KEIN Befund, nur Plausibilitätscheck):**

| MNIST | bal. Genauigkeit |
|---|---|
| norm_deep (die Kontrolle aus Q-003) | 0,8544 |
| **rein linear** 784→256→8→10 | **0,8873** |
| ReLU-MLP 784→256→256→8→10, ohne Normierung | **0,9246** |
| GELU-MLP mit Normierung (×16 skaliert) | 0,9101 |
| ohne 8-Kanal-Engstelle (784→256→ReLU→10) | 0,9371 |

Dieselbe Stichprobe auf Fashion-MNIST (Schaltung dort 0,8190, quad_free 0,8139 aus Q-003):

| Fashion-MNIST | bal. Genauigkeit |
|---|---|
| norm_deep | 0,7541 (beste lr 0,001 = unterer Gitterrand) |
| rein linear | 0,8037 |
| ReLU-MLP, gleiche Engstelle | 0,8214 |
| GELU-MLP mit Normierung | 0,8183 |
| ohne Engstelle | 0,8189 |

Folgerung: Die „starke klassische Referenz“ mit 67.848 Parametern ist **schlechter als ein
lineares Modell** (Normierung auf Einheitslänge + Tanh-Engstelle lähmt sie). Der Abstand
+0,0932 misst deshalb vor allem die Schwäche der Kontrolle. Gegen ein gewöhnliches
ReLU-Netz mit derselben Engstelle bleibt ungefähr +0,01 — dieselbe Größenordnung wie
gegen `classical_quad_free` (+0,0126). Die Aussage „16–80 Parameter schlagen 67.848“ ist
formal richtig, aber gegen einen Strohmann. Das muss vor der Anmeldung neu, sauber und
vorregistriert gemessen werden (siehe Agent `experiment`).

Was trotzdem trägt und stark ist: **16 Winkel erreichen dieselbe Genauigkeit wie ein
2.048-Parameter-Quadratmodell und wie ein ~68.000-Parameter-ReLU-Netz (Stichprobe) —
Faktor 128 bzw. ~4.000 weniger Parameter in der Mittelschicht.** Das ist die ehrliche,
verteidigbare Form der Aussage, wenn der neue Lauf sie bestätigt.

## 2. Verschränkung und Tiefe — „trägt nichts bei“ ist so nicht haltbar

- Beide Nullbefunde (−0,0015 / +0,0002, TOST) stammen aus Q-003-replication, also **mit
  trainierbarem Encoder** 784→256. Genau diese Konstellation hat Q-018 als Artefakt
  entlarvt („Ein Nullbefund an einem Bauteil ist wertlos, solange ein anderes Bauteil
  dessen Aufgabe übernehmen kann“). Die Lehre wurde auf Q-014 angewandt, nicht aber auf
  Verschränkung und Tiefe.
- Die Tiefe (5 gegen 1 Lage) **war die Positivkontrolle** des Laufs — und die ist laut
  `q_summary` durchgefallen. `q_summary` sagt ausdrücklich: jede Gleichwertigkeitsaussage
  aus diesem Lauf steht nicht. Die Dokumente (BEFUNDE, STAND, KURZFASSUNG) führen beide
  trotzdem als **[BELEGT] / ✅ positiver Gleichwertigkeitsnachweis**. Widerspruch.
- Theorie dazu: Ohne CNOT ist jede Tiefe L wieder ein Produkt von Ein-Qubit-Drehungen,
  also eine **Kronecker-Produkt-Drehung** U₁⊗…⊗U₈. Tiefe *kann* ohne Verschränkung nichts
  bringen (Gruppe abgeschlossen) — mit Verschränkung dagegen schon (Universalität braucht
  Tiefe, siehe BEFUNDE §2 „Teilung kostet Tiefe“). „Tiefe trägt nichts bei“ widerspricht
  also der eigenen Universalitätsrechnung, wenn es allgemein formuliert wird.
- Nachtest: Verschränkung × Tiefe im **Q-018-Aufbau ohne Encoder** (Aufgabe mit
  kalibriertem Kapazitätsbedarf). Billig, das Gerüst steht.

## 3. T-002 (Sprachmodell) — vier Fehler vor jedem GPU-Lauf beheben

1. `LR_SUCHE_SCHRITTE = 1000`, Messlauf 13.000 (Skript) bzw. 20.000 (Standard) → Regel 1(c)
   verletzt, schlimmer als das halbe Budget vom 27.08.; fehlt in der Liste der elf Module.
   Außerdem Gitterverlängerung nur nach oben → Regel 1(b).
2. Wiederaufnahme liest nur die **neueste** T-002-Datei (`result_store.latest`). `runpod_start.sh`
   fährt vor jedem Lauf die Probe (60 Schritte) → danach ist die Probe die neueste Datei,
   und der Hauptlauf fängt still bei null an. Bei parallelen Cluster-Jobs bricht das
   Prinzip ganz.
3. Schrittzahl nicht vorregistriert: 13.000 (`runpod_t002.sh`) gegen 20.000 (`runpod_start.sh`, CLI-Standard).
4. `trainiere_arm` speichert `"lr": LR` (Rückfallwert 3e-4) statt der benutzten Rate.

Architektur-Einwand (Entscheidung Malte, vor dem Lauf ins Tagebuch): `ButterflyFFN` gibt
`|Amplitude|² · 256 · scale` aus — nichtnegativ, feste Summe, **ein** skalarer Gewinn. Der
FFN-Block kann damit nichts vom Residualstrom abziehen. T0 ist sehr wahrscheinlich
(Probe: −0,32 nats nach 60 Schritten). `butterfly_qkv` lag in der Probe fast gleichauf
(−0,0045 nats) und ist der realistischere „Quanten-Chatbot“. Vorschlag: zusätzlicher,
außerhalb der Familie geführter Arm mit zentrierter Ausgabe + Gewinn je Kanal (256 Parameter).

Rechenbedarf T-002: 32 Läufe (4 Arme × 8 Seeds) + Lernratensuche. Butterfly-Lauf ≈ 0,334 s/Schritt
auf RTX 4090 bei Batch 64 → ~72 min bei 13.000 Schritten. Mit korrekter Suche ~30–35 GPU-h
auf einer Karte; parallel auf 8 GPUs ein Nachmittag. Butterfly ist auf der GPU
startaufwandbegrenzt → mehr parallele GPUs helfen, eine schnellere GPU kaum.

## 4. Q-019 (Zeitreihen) — Entwurfsschwächen

- Die Primärhypothese (Wechselwirkung periodisch × permutiert) ist **fast per Konstruktion
  wahr**: Eine feste Permutation ist eine orthogonale lineare Abbildung, die eine dichte
  Schicht mitlernen kann, eine Butterfly nicht. Der Vorlauf zeigt genau das (0,14 gegen 0,92).
  Das beweist Ordnungsabhängigkeit, keinen Vorteil.
- P2 (butterfly schlägt dense auf periodisch) fällt laut Vorlauf (0,686 gegen 0,913).
- Engstelle: ⟨Z_q⟩ liefert nur 8 Zahlen aus 256; laut Handoff tragen nur 3 davon Bandinformation.
- Chance auf echten Vorteil: (a) **wenig Trainingsdaten** (Lernkurve N=50…6000) — 40 gegen
  65.536 Parameter ist ein Induktionsbias-Argument; (b) **Walsh-Hadamard statt Fourier**:
  geteilte Butterfly ohne CNOT = Kronecker-Produkt von 2×2-Drehungen, enthält die
  Walsh-Hadamard-Transformation exakt. Eine Aufgabe mit Sequenz-/Walsh-Bändern
  (Rechtecksignale, Schaltsignale) ist der Fall, in dem die Struktur die ideale Basis mit
  8–40 Parametern exakt darstellen kann.
- HANDOFF §6 nennt `untied` als offene Entscheidung — laut Tagebuch (27.08. nachts) ist sie
  gefallen und eingebaut; Laufzeit 4,6–5,5 h statt 2,5–3 h. Handoff veraltet.
- Q-019 fehlt in `q_summary.QUESTIONS` (Regel aus CLAUDE.md §6). T-000 ebenfalls.

## 5. Q-011 (Unitarität in der Tiefe)

- Keine Wiederaufnahme, Gitterverlängerung nur nach oben (bekannt, nicht behoben).
- Tiefen 1–8 gerettet (225 Zeilen). Tuning-Abstände deuten bei Tiefe 16 auf den
  vorhergesagten Effekt — nur 3 Tuning-Seeds, nicht zitierbar.

## 6. Projekt B

- `qire_ai/intrinsic_dimension.py` rechnet **immer noch** die pseudoreplizierte Korrelation
  über 240 Einzelmessungen (Zeilen 192–202). Korrigiert ist nur das Tagebuch. Wer die
  Auswertung neu startet, bekommt wieder p=3·10⁻¹⁸⁷.
- Das automatische Urteil von B-014 sagt „d erklärt mehr als die Klassenzahl“, die
  korrigierten Zellmittel sagen trainiert d ρ=0,61, Klassen ρ=0,67. Widerspruch.
- Überschrift „Rang wird von der Aufgabe gesetzt, **nicht von der Architektur**“: Architektur
  wurde in B-014 nicht variiert (eine MLP, Breite 4096). Überdehnt, bis der CNN-/Mehrschicht-Lauf da ist.
- „Sieht aus wie ein Minimum“ ist qualitativ. Modellvergleich möglich: Rang ≈ min(a·d, b·C)
  gegen additive/multiplikative Modelle, Zellmittel, Kreuzvalidierung.
- `qire_ai.audit` prüft **nur 15 B-Zahlen**; B-014, B-011, B-010, B-007, B-008, B-009 fehlen —
  also gerade der tragende Befund. Läuft nur im Archiv, nicht aus A oder B.
- Effektivrang: nur ein Maß. Robustheit mit mehreren Maßen (Teilnahmezahl, Entropie-Rang
  nach Roy/Vetterli, 90/99-%-Energierang) zeigen.
- Literaturbezug fehlt: Neural Collapse (Papyan/Han/Donoho 2020), Low-Rank Simplicity Bias
  (Huh et al.), intrinsische Dimension von Repräsentationen (Ansuini et al. 2019). B-014 muss
  sich dazu einordnen, sonst wirkt es wie eine Neuentdeckung von Bekanntem.

## 7. Dokumente — veraltet und widersprüchlich

- `KURZFASSUNG_ENTWURF.md` (6. August): B-014-Korrelationen vor der Korrektur (+0,986 / +0,002
  statt +0,987 / +0,03); Zusatztabelle nennt „+0,0044 (p=0,49)“ für den klassischen Nachbau —
  das ist die **vermischte Zahl** aus dem abgebrochenen Lauf, im Text steht +0,0126;
  Geschwindigkeit „1,99×“ (5. Aug) statt T-001-Messung (1,59× Referenz / 2,31× schnell);
  keine Q-015/Q-018/T-Serie; Fashion-Vorbehalt fehlt; Hardware-Tabelle (0,500/0,450) ohne
  Stichprobengröße (20 Testbeispiele, Übereinstimmung 17/20) und ohne Hinweis, dass die
  Hardware ein **anderes** Modell (Winkelkodierung) rechnete als die Kernaussage.
- `BEFUNDE_UND_RICHTUNGEN.md` führt Q-010/12/13/15/16 als [LÄUFT] (Stand 6. Aug).
- `DATEN_EXTERN.md`: „Platte 99 % voll, 29 GB frei“ — heute 141 GB frei.
- Die Standdemo (`demo/`) lag nur im Archiv; in der Überarbeitung nach `A/demo/` kopiert.
- Zu viele parallele Stand-Dokumente (ROADMAP 110 KB, STAND, HANDOFF_AKTUELL, HANDOFF_PROJEKT_A,
  BEFUNDE, FORSCHUNGSIDEEN, RESULTS, KURZFASSUNG) mit unterschiedlichem Stand.
- Literaturkarten: englisch, Stand 1. August, beziehen sich auf alte QIRE-Module; die
  wichtigste Vorarbeit fehlt komplett (Dao et al. Butterfly 2019, EUNN 2017, BOFT 2024,
  QuanTA 2024, Bowles/Ahmed/Schuld 2024, Thanasilp 2024, Huang et al. 2022).

## 8. Infrastruktur und Sicherheit

- **Kein Git.** Für ein Projekt, dessen Wert an Vorregistrierung hängt, ist ein
  Commit-Zeitstempel *vor* den Rohdaten der stärkste Beleg. Empfehlung: git init + privates Repo.
- `ibm_quantum_token.txt` liegt im Klartext im Archivordner. Nach der PC-Infektion (seit
  Januar 2026) gelten Zugangsdaten als kompromittiert → **Token bei IBM neu erzeugen**, Datei entfernen.
- IBM-Hardware-Rohdaten (Counts) sind nirgends gespeichert, nur Zusammenfassung in
  `demo/artifacts/hardware_evidence.json`. Jobs über die IDs herunterladen, solange IBM sie hält.
- `qire_ai.audit` fällt in A durch (B-Rohdaten fehlen dort) — erwartet, aber verwirrend.

## 9. Fristen

- Anmeldung: 30. November 2026. Schriftliche Arbeit: **Mitte Januar 2027** (eine
  Landesseite nennt 17.01.2027 — beim eigenen Landeswettbewerb prüfen). Die GPU-Ergebnisse
  müssen also erst im Januar in der Arbeit stehen, nicht am 30. November.
