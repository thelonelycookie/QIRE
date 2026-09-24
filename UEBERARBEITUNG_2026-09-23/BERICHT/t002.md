# T-002 — Überarbeitung vor dem GPU-Lauf (Agent t002, 23./24. September 2026)

Grundlage: `00_REVIEW_BEFUNDE.md` §3. Alle vier dort genannten Fehler sind im
Code behoben, der Architektur-Einwand ist als **abschaltbarer, nicht
vorregistrierter Zusatzarm** eingebaut (standardmäßig aus), und T-002 lässt
sich jetzt zellenweise auf einem SLURM-Cluster rechnen. **Kein Messlauf wurde
gerechnet; es gibt keine neue T-002-Zahl.**

---

## 1. Geänderte / neue Dateien

| Datei | Was |
|---|---|
| `A/quantum_sim/sprachmodell.py` | überarbeitet (Details in §2) |
| `A/server_setup/cluster/README.md` | **neu** — Anleitung Cluster, Annahmen, Rechenbedarf, Fragen fürs MPI |
| `A/server_setup/cluster/t002_zelle.sbatch` | **neu** — SLURM-Job-Array, eine Zelle je Element |
| `A/server_setup/cluster/t002_steuerung.sh` | **neu** — Steuerung auf dem Login-Knoten (`pruefen`, `lr`, `messung`, `auto`, `status`, `zusammenfuehren`) |
| `A/server_setup/runpod_t002.sh` | Schrittzahl aus der Konstante, Selbsttest vorab, Suche immer aufrufen (wiederaufnehmbar) |
| `A/server_setup/runpod_start.sh` | 20.000-Standard und Schrittzahl-Argument entfernt, Probe unter eigener ID, Lernratensuche ergänzt |
| `A/server_setup/alles_durchlaufen.sh` | Schrittzahl aus der Konstante, Selbsttest, neue Lage von `lr_map.json` |
| `A/qire_ai/result_store.py` | **nicht geändert** (nicht nötig) |

Nicht angefasst: `sequence_platform.py` (`ButterflyFFN`), Tagebuch, Handoff,
`q_audit`, `q_summary`.

## 2. Was in `sprachmodell.py` geändert ist

1. **Schrittzahl einmal festgelegt:** `SCHRITTE = 13000`. CLI-Standard,
   Lernratensuche (`LR_SUCHE_SCHRITTE = SCHRITTE`) und alle drei Shellskripte
   leiten sich davon ab (`python -m quantum_sim.sprachmodell schritte`). Eine
   andere Zahl wird verweigert, außer mit `--nicht-vorregistriert`.
   *Begründung (nachgezählt mit `SprachModell(arm)`):* 13.000 × 64 × 128 =
   106,5 Mio. Token. `dense` hat 5.289.472 Parameter (1.081.344 davon
   Einbettungen) → **20,1 Token je Parameter**, 25,3 ohne Einbettungen —
   das Chinchilla-Verhältnis (~20, Hoffmann et al. 2022). 20.000 Schritte wären
   31,0 Token/Parameter bei 54 % mehr GPU-Zeit. Die kleineren Arme sehen bei
   gleicher Schrittzahl mehr Token je Parameter (`butterfly_ffn` 33,4) —
   gewollt, das Budget ist die Schrittzahl.
2. **Lernratensuche mit vollem Budget (Regel 1c)** und **Verlängerung in beide
   Richtungen (Regel 1b)**: Gitter 3e-4 … 2,4e-3, Faktor 2, bis zu 3
   Verlängerungen in Richtung des Randes (Reichweite 3,75e-5 … 0,0192). Die
   Entscheidung steckt in der reinen Funktion `lr_entscheidung()`, der
   Suchpfad wird bei jedem Aufruf aus den vorhandenen Zellen nachgerechnet
   (`lr_stand()`) — lokal und auf dem Cluster derselbe Code. `lr_map.json`
   enthält Karte, Tabelle, Vermerk je Arm (`innen`/`oben`/`unten`), Liste
   `am_rand`, Schrittzahl, Seeds, Batch. **Neuer Ort:**
   `results/t002_zellen/T-002/lr_map.json` (alles von T-002 in einem Ordner).
   Der Messlauf lehnt eine fehlende Karte, eine Karte mit anderer Schrittzahl
   oder ohne Eintrag für einen Arm jetzt **ab** (vorher: Warnung und 3e-4).
3. **Wiederaufnahme robust:** Jede Zelle (Arm, Seed) bzw. (Arm, Rate, Seed)
   schreibt eine eigene Datei, atomar, **nie überschreibend** (Doppelrechnung
   landet daneben). `sammle_messungen()` liest **alle** Sammeldateien
   `T-002__*.json` **und** alle Zellendateien, filtert nach Schrittzahl, Batch
   und Lernrate, führt nach (Arm, Seed) zusammen, zählt identische Duplikate
   und meldet Widersprüche (dann gilt die **ältere** Messung, nie die bessere).
   Die Probe speichert unter **`T-002-probe`** und wird vom Hauptlauf gar
   nicht gelesen; sie überschreibt auch die Chatbot-Gewichte nicht mehr.
4. **`"lr"` im Datensatz = tatsächlich benutzte Rate**, dazu `"lr_quelle"`
   (`lr_map` / `rueckfall` / `lernratensuche`).
5. **Cluster:** `zelle --arm A --seed S`, `lernratensuche --arm A --lr X --seed S`,
   `plan lr|messung` (offene Zellen als Zeilen), `status`, `zusammenfuehren`
   (einsammeln → `analyse()` → eine Sammeldatei, die `q_summary` liest),
   `rechenbedarf`. Pfade per `QIRE_KORPUS`, `QIRE_MODELLE`, `QIRE_T002_ZELLEN`.
6. **Zusatzarm `butterfly_ffn_zentriert`** (Klasse `ButterflyFFNZentriert`):
   `ButterflyFFN` unverändert als Kern, Ausgabe minus Mittelwert über die
   Kanäle, Gewinn je Kanal (256), `scale` eingefroren (sonst doppelt).
   336 statt 81 trainierbare Parameter je Block (1.344 gesamt). **Nicht in
   `ARME`**, nur mit `--mit-zusatzarm`; in `analyse()` nur im Abschnitt
   „BESCHREIBEND — außerhalb der Familie, ohne Holm“, nie im Urteil.
7. **`analyse()` — drei Lücken behoben** (vom Vortest gefunden bzw. belegt):
   - (a) **Asymmetrie T1+/T0:** T1+ wurde vor T1 geprüft, T0 erst danach. Ein
     `butterfly_ffn`, der signifikant, aber innerhalb ±0,05 schlechter ist,
     hieß „T1 — die Struktur hält“. **Nachgeprüft mit der alten `analyse()`:
     5 von 5 Ziehungen T1.** Neu: T0 mit Zusatz „zugleich innerhalb der
     angemeldeten Grenze … gleichwertig im Sinne der Anmeldung, aber messbar
     schlechter — beides berichten“. Die Asymmetrie lief für die Hypothese.
   - (b) **Bodenprüfung ohne Folgen:** Sie wurde nur gedruckt. Neu: liegt
     `dense` nicht mindestens 0,10 nats unter der Unigramm-Baseline → TU.
   - (c) **Paarung nach Position statt Seed:** Fehlt eine Zelle, **stürzte die
     alte `analyse()` ab** (nachgeprüft, 5/5); fehlen in zwei Armen
     verschiedene Seeds, hätte sie still falsch gepaart. Neu: Paarung über die
     Seed-Nummer, Vermerk `[VORLAEUFIG: nur n von 8 Seeds gepaart]`.
8. Kleinkram: `ButterflyLinear`-Docstring sagte „4 Lagen × 8 = 40 Winkel“, es
   sind 5 Lagen (`QKV_LAGEN`); `struktur_parameter()` zählt nur trainierbare
   Parameter (für die vier Arme unverändert: 2.102.272 / 324 / 480 / 132.352 —
   Faktor 6.488 bestätigt); `torch.load(..., weights_only=True)` in `erzeuge`.

## 3. Tests

Alle aus `…\UEBERARBEITUNG_2026-09-23\A` mit dem Archiv-Interpreter,
`PYTHONIOENCODING=utf-8`.

| Befehl | Ergebnis |
|---|---|
| `python -m quantum_sim.sprachmodell schritte` | `13000`, Modul importierbar |
| `python -m quantum_sim.sprachmodell selbsttest` | **BESTANDEN**, 5,6 s (Einzelheiten unten) |
| `python -m quantum_sim.sprachmodell rechenbedarf [--mit-zusatzarm]` | Tabelle §4 |
| Vergleich alte/neue `analyse()` (Skript im Scratchpad, lädt das Original nur lesend) | Szenario D alt 5/5 „T1“, neu 5/5 „T0 … zugleich innerhalb“; Szenario I alt 5/5 Absturz, neu 5/5 „T1+ … VORLAEUFIG“ |
| `bash -n` auf alle fünf Shellskripte (Git-Bash) | alle Exit 0, LF-Zeilenenden |
| `python -m quantum_sim.sprachmodell rauchtest` | siehe §3.4 |

### 3.1 Vortest (`vortest`) — jeder Ausgang, Zweigtext geprüft

| Szenario | Erwartet (Marke + Zweigtext) | Ergebnis |
|---|---|---|
| A kein Unterschied, kleine Streuung | T1 „die Struktur haelt“ | 20/20 |
| B Butterfly deutlich besser | T1+ „BESSER als der dichte“ | 5/5 |
| C Butterfly deutlich schlechter | T0 „die Struktur kostet“, **ohne** „zugleich innerhalb“ | 5/5 |
| D signifikant schlechter, innerhalb der Grenze | T0 „zugleich innerhalb der angemeldeten Grenze“ | 5/5 |
| E Nulleffekt, große Streuung | TU „unentschieden“ | 19/20 (1× T1) |
| F Auflösungsprüfung fällt (narrow = dense) | TU „Aufloesungspruefung durchgefallen“ | 19/20 (1× T1+) |
| G Bodenprüfung fällt | TU „Bodenpruefung durchgefallen“ | 5/5 |
| H Auflösung in Gegenrichtung (narrow besser) | T1 + Auflösungstext „NARROW BESSER“ | 17/20 (2× T1+, 1× T0) |
| I eine Zelle fehlt | T1+ „VORLAEUFIG“ | 5/5 |
| Z Zusatzarm an/aus auf denselben Daten | Urteil identisch, nur beschreibend | 5/5 |

Nulleffekt-Szenarien (A, E, F, H) laufen über 20 Ziehungen mit geforderten
80 %: ein Test auf Niveau 0,05 spricht bei wahrem Nulleffekt in einem Teil der
Ziehungen an — das ist sein Niveau, kein Auswertungsfehler (wie Szenario G in
`zeitreihen.py`). Synthetische Daten: eigenes Rauschen je (Arm, Seed), großer
gemeinsamer Seed-Anteil (0,05; eine nicht paarende Auswertung würde anders
urteilen), früher Verlaufspunkt nicht konstant versetzt.

**Befund im Datengenerator, nicht in `analyse()`:** Szenario E lief zuerst mit
dem Standard-Abstand `narrow` 0,10 bei Streuung 0,08 — die Marke TU war 20/20
richtig, aber 11 der 20 Ziehungen landeten im Zweig „Auflösungsprüfung“ statt
„unentschieden“. Nur die Zweigprüfung hat das gezeigt. Behoben durch
`narrow_ab=0,5` im Szenario, im Code kommentiert.

### 3.2 `test_gitter` — Regel 1(b), über die echten Dateifunktionen

| Optimum der künstlichen Verlustkurve | gewählt | Vermerk | Zellen |
|---|---|---|---|
| 1,2e-3 (im Gitter) | 0,0012 | innen | 4 |
| 4,8e-3 (eine Verlängerung oben) | 0,0048 | innen | 6 |
| 1,5e-4 (eine Verlängerung unten) | 0,00015 | innen | 6 |
| 7,5e-5 (zwei Verlängerungen unten) | 7,5e-05 | innen | 7 |
| 10 (außer Reichweite oben) | 0,0192 | oben | 7 |
| 1e-9 (außer Reichweite unten) | 3,75e-05 | unten | 7 |

### 3.3 `test_wiederaufnahme`

Aufbau mit alter Sammeldatei, identischem und widersprüchlichem Duplikat,
Zelle mit falscher Rate, **neuester** Datei mit 60 Schritten (der alte
Probe-Fehler) und einer `T-002-probe`-Datei: genau die 4 richtigen Zellen,
1 Duplikat, 1 Widerspruch (ältere Messung 4,20 gilt, nicht die bessere 4,05),
falsche Rate und falsche Schrittzahl ausgeschlossen, Probe nicht gelesen —
7/7 bestanden.

### 3.4 Rauchtest (echte Kette auf CPU)

AUSSTEHEND — wartet auf die Freigabe der CPU durch Agent `experiment`
(Q-020-Pilot). Wird hier nachgetragen.

## 4. Rechenbedarf — SCHÄTZUNG

Basis: 0,334 s/Schritt, gemessen **nur** für `butterfly_ffn` auf RTX 4090,
Batch 64 (11. August), für alle Arme angesetzt → eher Obergrenze.
~74 min je Zelle inkl. ~2,7 % Auswertung (Annahme Vorwärtslauf ≈ ⅓ Schritt).

| Phase | Zellen | GPU-h |
|---|---:|---:|
| LR-Startgitter (4 Arme × 4 Raten × 1 Seed) | 16 | 19,8 |
| LR-Verlängerungen (höchstens) | 0–12 | 0–14,9 |
| Messlauf (4 × 8) | 32 | 39,6 |
| **Summe** | **48–60** | **59,4–74,3** |

Wanduhr: 1 GPU 59–74 h · 4 GPUs 15–19 h · 8 GPUs 7,4–11,1 h · 16 GPUs
3,7–7,4 h · 32 GPUs 2,5–6,2 h. Mit Zusatzarm: 60–75 Zellen, 74–93 GPU-h.

**Abweichung zum Review (§3: „~30–35 GPU-h“):** Das volle Suchbudget macht
die Suche so teuer wie einen halben Messlauf. Die Review-Zahl galt für die alte
1.000-Schritt-Suche.

## 5. Offene Entscheidungen für Malte

1. **Zusatzarm `butterfly_ffn_zentriert` mitrechnen?** Kosten +25 %
   (+12–15 Zellen, +15–19 GPU-h). Nur beschreibend, außerhalb der Familie.
   Entscheidung **vor** dem Lauf ins Tagebuch (Textvorschlag §6).
2. **Änderungen (a)–(c) an der Auswertung bestätigen** — (a) und (b) ändern,
   welcher Ausgang wann gilt, sind also Änderungen an der Vorregistrierung.
   Sie müssen ins Tagebuch, bevor eine T-002-Zahl entsteht. Alternative zu (a):
   Marke T1 behalten und nur den Kostenzusatz anhängen — dann bleibt die
   Asymmetrie zu T1+ bestehen.
3. **Schrittzahl 13.000 bestätigen** (Begründung §2.1).
4. **Ein oder zwei Tuning-Seeds** für die Lernratensuche? Einer ist
   eingestellt (`LR_SUCHE_SEEDS = (2199,)`). Zwei verdoppeln die Suchkosten
   (+16–28 Zellen), machen die Wahl aber weniger vom Zufall eines Seeds abhängig.
   Wie groß die Seed-Streuung bei 13.000 Schritten ist, ist unbekannt.
5. **Git vor dem Lauf** (Review §8): ein Commit-Zeitstempel vor den Rohdaten
   ist der stärkste Beleg der Vorregistrierung.

Bekannte Grenzen (keine Entscheidung nötig, aber zu berichten):
- Lernrate und Endmessung benutzen dieselbe Validierungsdatei (andere Stapel,
  andere Seeds). Gilt für alle Arme gleich.
- Innerhalb einer Zelle gibt es keine Zwischenstände: ein abgebrochener Job
  verliert bis zu ~74 min.
- `result_store.Recorder.save` benennt Dateien sekundengenau; zwei Speicherungen
  derselben ID in derselben Sekunde überschreiben sich. Für T-002 unkritisch
  (Zellen haben eindeutige Namen), aber derselbe Fehlertyp könnte andere Module
  treffen — nicht geändert, weil nicht nötig.
- Die Cluster-Skripte sind nur syntaktisch geprüft, nicht auf SLURM gelaufen.

## 6. Tagebuch-Textvorschlag (NICHT eingetragen — Malte entscheidet)

> ### 24. September — T-002 vor dem GPU-Lauf überarbeitet
>
> Der Review vom 23. September hat vier Fehler in T-002 gefunden, alle vor
> jedem Messlauf. Behoben:
>
> 1. **Die Lernratensuche hatte 1.000 Schritte, der Messlauf 13.000 bzw.
>    20.000** — Regel 1(c) verletzt, und das Gitter wurde nur nach oben
>    verlängert (Regel 1b). Jetzt sucht sie mit derselben Schrittzahl wie der
>    Messlauf und verlängert in beide Richtungen, bis zu dreimal. Das kostet:
>    mit voller Suche 48–60 Läufe statt 32, geschätzt 59–74 GPU-Stunden auf
>    einer RTX 4090 (Obergrenze, nur `butterfly_ffn` ist gemessen).
> 2. **Die Wiederaufnahme las nur die neueste Datei** — nach der Probe war das
>    die Probe, und der Hauptlauf hätte still bei null angefangen. Jetzt hat
>    jeder einzelne Lauf seine eigene Datei, beim Wiederaufnehmen werden alle
>    gelesen, Duplikate erkannt, und bei Widersprüchen gilt die ältere Messung.
>    Die Probe speichert unter eigener ID.
> 3. **Die Schrittzahl war nicht festgelegt** (13.000 in einem Skript, 20.000
>    im anderen). Vorregistriert: **13.000 Schritte** = 106,5 Mio. Token =
>    20,1 Token je Parameter des dichten Modells, das Chinchilla-Verhältnis.
> 4. **Im Datensatz stand immer die Rückfall-Lernrate 3e-4**, nicht die
>    benutzte. Behoben.
>
> **Änderungen an der Auswertung, vor jeder Zahl:** Der Vortest hat gezeigt,
> dass die alte Auswertung einen `butterfly_ffn`, der signifikant, aber
> innerhalb von ±0,05 nats schlechter ist, als „T1 — die Struktur hält“
> meldete (5 von 5 synthetischen Ziehungen), einen ebenso knapp *besseren*
> aber als T1+. Die Asymmetrie lief für meine Hypothese. Ab jetzt wird
> Signifikanz in beiden Richtungen vor der Gleichwertigkeit geprüft; der Fall
> heißt T0 mit dem Zusatz „zugleich innerhalb der Grenze“, und beides wird
> berichtet. Außerdem: Liegt `dense` nicht deutlich unter der
> Unigramm-Baseline, ist der Lauf TU (Bodenprüfung), und die Arme werden über
> die Seed-Nummer gepaart (die alte Auswertung stürzte bei einer fehlenden
> Zelle ab).
>
> **Architektur-Einwand, Entscheidung:** `ButterflyFFN` gibt
> |Amplitude|² · 256 · scale aus — nichtnegativ, mit fester Summe und einem
> einzigen Gewinn. Der FFN-Block kann dem Residualstrom damit nichts abziehen;
> T0 ist wahrscheinlich (Probe: −0,32 nats nach 60 Schritten). Ob das an der
> Struktur liegt oder nur an der Ausgabeform, trennt ein Zusatzarm
> `butterfly_ffn_zentriert` (Ausgabe minus Mittelwert, Gewinn je Kanal,
> 336 statt 81 Parameter je Block).
>
> *[Variante JA:]* Ich nehme ihn mit. Er steht **außerhalb** der
> vorregistrierten Familie, wird nicht Holm-korrigiert und bestätigt nichts —
> er beschreibt nur. PRIMÄR bleibt `butterfly_ffn` gegen `dense`. Kosten:
> +25 % Rechenzeit.
>
> *[Variante NEIN:]* Ich nehme ihn nicht mit. PRIMÄR bleibt, wie angemeldet,
> `butterfly_ffn` gegen `dense`. Fällt T0, wird der Einwand als offene
> Erklärung berichtet, nicht als Befund.
