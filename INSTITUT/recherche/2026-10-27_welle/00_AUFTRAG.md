# Gemeinsamer Auftrag — Ideenwelle 27.10.2026 (Erkundung, keine Befunde)

Direktor plant, du baust und erkundest. Alles auf **Deutsch**.

## Lage (Stand 27.10.)
- Ziel (Malte, 07.10.): eine **quanten-inspirierte, ressourceneffiziente KI**. Neuheit nicht nötig, Bekanntes wird zitiert. Wirkung ist klassisch (I-038): „quanten-inspiriert“, nie „Quantenvorteil“.
- Lies zuerst: `INSTITUT/KERNAUSSAGE_2026-10-07.md`, im `INSTITUT/LABORBUCH.md` die Einträge I-044-v3, I-045-v3, I-045-v4 (Ende der Datei).
- Wichtigste Lehre bisher: Feste Fastfood-Mischung + Born-Quadrat spart **Parameter/Speicher, nicht FLOPs**; Vorsprünge bei knappem Training verschwinden mit mehr Training (I-044-v3, I-045-v3). Bei **gleichen FLOPs** gewinnen dicht und Niedrigrang (I-045-v4). Jede neue Idee muss sich daher **bei gleichen FLOPs UND gleichen Parametern** gegen dicht/Niedrigrang behaupten, mit ausreichend Training.
- Bausteine wiederverwenden: `INSTITUT/labor/gemeinsam.py` (Daten, lädt MNIST/Fashion nach `INSTITUT/daten/`), `INSTITUT/abteilungen/struktur_sprache.py` (Byte-Sprachmodell UD_German-GSD, bpc), `struktur_kopf_fair.py`, `ternaer_butterfly.py`.

## Regeln
- **Erkundung, kein Befund, kein vorregistrierter Lauf.** Nichts in `agenda.json`, `LABORBUCH.md`, `seeds.json` eintragen, kein `direktor.py lauf`.
- Nur **Tuning-Seeds 900–999** (dein Block steht im Einzelauftrag). Nie Mess-Seeds.
- Sauber trotzdem: ≥ 3 Seeds, Mittel ± sd, Validierungsanteil für Hyperparameter, fester Testsatz.
- **Rechenbudget: 1 Thread** (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), insgesamt **≤ 30 min CPU**. 8 Agenten teilen sich 4 Kerne. Kleine Modelle, kurze Läufe, lieber eine gute Kurve als viele Arme.
- Schreiben **nur**: `INSTITUT/recherche/2026-10-27_welle/<kürzel>_bericht.md` und Skripte in `INSTITUT/recherche/2026-10-27_welle/<kürzel>/`. Kein git, keine anderen Repo-Dateien ändern. Keine Daten ins Repo (daten/ ist gitignored).
- **Ehrlich zählen:** Parameter, FLOPs pro Beispiel/Token, Speicher (Bytes). Immer eine gleich teure klassische Baseline.
- Quellen: aus dem Gedächtnis als [Gedächtnis] markieren, WebSearch nur sparsam (≤ 5).

## Berichtsformat (≤ 200 Zeilen)
1. Kurzfassung (≤ 8 Zeilen) mit Ampel: grün (lohnt Vorregistrierung) / gelb / rot.
2. Idee, quanten-inspirierter Bezug, Literatur.
3. Aufbau, Seeds, Tabelle (Mittel ± sd, Parameter, FLOPs, Bytes).
4. Deutung inkl. langweiligster Erklärung.
5. Vorschlag für einen vorregistrierten Lauf (Hypothesen, Arme, Kontrollen, Seeds-Zahl, Effektgröße, Laufzeit Container/Raven).
6. Skriptpfade.

## Schlussnachricht (≤ 10 Zeilen)
Pfad, Ampel, Kernzahlen, Laufvorschlag in 2 Zeilen.
