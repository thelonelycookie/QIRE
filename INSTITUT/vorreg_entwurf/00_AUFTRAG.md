# Auftrag Modulbau + Vorregistrierung (27.10., Freigabe Malte: "mach alle", Raven)
- Lies CLAUDE.md, INSTITUT/UEBERGABE.md §2–3, und als Vorlage die Vorregistrierung + Modul von I-045-v4 (LABORBUCH.md, abteilungen/struktur_sprache.py lauf_v4) bzw. I-047 (abteilungen/struktur_tt.py). Genau dieses Format übernehmen: Hypothesen mit Entscheidungsregel, Arme, Kontrollen, NK, Seeds, Statistik, Laufzeit, Wiederaufnahme je Aufgabe (ergebnisse/_zwischen_<ID>.json), Checkpoints.
- Grundlage: INSTITUT/recherche/2026-10-27_welle/SYNTHESE.md und die genannten Berichte (Branch `welle-2026-10-27-ergebnisse`, lesen mit `git show origin/welle-2026-10-27-ergebnisse:<pfad>`).
- Rechnen auf MPCDF Raven: bis 72 Prozesse, 24-h-Limit, `INSTITUT_PROZESSE=72 OMP_NUM_THREADS=1 python3 direktor.py lauf <ID>`. Aufgaben fein schneiden, damit 72 Prozesse ausgelastet sind.
- Du schreibst NUR: dein Modul `INSTITUT/abteilungen/<modul>.py` und `INSTITUT/vorreg_entwurf/<ID>.md` mit (a) LABORBUCH-Text, (b) agenda.json-Eintrag als JSON, (c) seeds.json-Eintrag als JSON. Nicht LABORBUCH/agenda/seeds/direktor selbst ändern, kein git. Ein anderer Agent arbeitet parallel auf main.
- Vortest: `schnell`-Profil nur auf deinem Tuning-Seed, im Container ≤ 10 min, 1 Thread; er muss grün sein. Kein Mess-Seed rechnen.
- Wirkung klassisch (I-038), Formulierung „quanten-inspiriert“. Alles Deutsch.
- Schlussnachricht ≤ 10 Zeilen: Dateien, Vortest-Ergebnis, Aufgabenzahl, geschätzte Kernstunden und Wandzeit auf 72 Kernen.
