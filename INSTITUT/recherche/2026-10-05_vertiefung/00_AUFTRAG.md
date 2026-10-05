# Gemeinsamer Auftrag — Vertiefung Etappe 2 (05.10.2026)

Direktor plant (Opus), du vertiefst (Sonnet). Alles auf **Deutsch**.

## Lage
- Etappe 1 (12 Berichte) liegt in `INSTITUT/recherche/2026-10-05_breit/` — lies **zuerst** `00_AUFTRAG.md` dort (Projektkontext) und dann die in deinem Auftrag genannten Berichte.
- **WebSearch ist für diese Sitzung aufgebraucht** (200/200). Etappe 2 ist deshalb **rechnend und theoretisch**: kleine, saubere Erkundungsmessungen auf echtem MNIST/Fashion + Herleitungen. Literatur nur aus den Etappe-1-Berichten bzw. als [Gedächtnis] markiert.
- Daten: `INSTITUT/daten/mnist.npz`, `fashion_mnist.npz` (Schlüssel x, y). 16×16-Vorverarbeitung und gemeinsamer Kopf: siehe `INSTITUT/abteilungen/clifford.py` (`daten`) und `clifford_suche.py` (`kopf_logistisch`), Bruhat-Merkmale W(V,m): `abteilungen/maske_block.py`, `maske_eigenschaft.py`. **Wiederverwenden statt neu erfinden.**

## Regeln
- **Erkundung, kein Befund.** Nichts davon ist vorregistriert. Ergebnisse dienen nur dazu, die nächsten vorregistrierten Läufe zu planen. Im Bericht klar als „Erkundung“ kennzeichnen.
- Trotzdem sauber: **mindestens 5 Seeds** (verschiedene Trainings-/Testaufteilungen bzw. Initialisierungen), Mittel ± Streuung, offizieller bzw. fester Testsatz, keine Wahl von Hyperparametern auf dem Testsatz (Validierungsanteil nutzen).
- **Seeds 900–999 sind für Etappe 2 reserviert** (nicht in Mess-Seeds künftiger Läufe verwenden). Jeder Agent: eigener Block, steht in deinem Auftrag.
- Rechenbudget: höchstens **2 Threads** (`OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`) und insgesamt **≤ 40 min** Rechenzeit; 5 andere Agenten rechnen parallel auf 4 Kernen.
- Schreiben: nur **deine Berichtsdatei** + Skripte im Scratchpad `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/<dein_kürzel>/`. Keine Repo-Dateien sonst ändern, kein git.
- **Parameter ehrlich zählen**: alle trainierten reellen Zahlen + (getrennt) diskrete Wahlen in Bit; Vorverarbeitung, die aus Daten gelernt wird (PCA o. ä.), zählt mit.

## Berichtsformat (≤ 250 Zeilen)
1. Kurzfassung (5–10 Zeilen)
2. Was gemessen/hergeleitet wurde (Aufbau, Seeds, Tabellen mit Mittel ± sd)
3. Deutung — ehrlich, inkl. langweiligster Erklärung
4. **Vorschlag für einen vorregistrierten Lauf**: Hypothesen, Arme, Kontrollen, Positivkontrolle, Seeds-Zahl, erwartete Effektgröße/Streuung (aus deiner Erkundung) und ob TOST-Gleichwertigkeit erreichbar wäre, Laufzeit auf 4–5 Kernen
5. Offene Fragen
6. Pfad der Skripte

## Rückmeldung (Schlussnachricht, ≤ 15 Zeilen)
Dateipfad, Kernzahlen, wichtigste Erkenntnis, Laufvorschlag in 2 Zeilen.
