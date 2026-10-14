# START — Einstieg für jede Sitzung (Stand 14.10.2026)

**Gearbeitet wird nur noch auf `main`.** Alle früheren Branches sind zusammengeführt und bleiben als Archiv stehen.
Abgeschlossene Zwischenstände liegen auf dem Branch `archiv/zwischenstaende-2026-10-14`; die Ergebnisse selbst bleiben in `main`.

## 1. Lesereihenfolge
1. `CLAUDE.md`: Regeln (Deutsch, kurz, Opus plant/Sonnet führt aus, Commits nur unter Maltes Namen).
2. **Diese Datei.**
3. `INSTITUT/UEBERGABE.md` §2–3: Vorgaben und Protokollregeln (verbindlich).
4. Bei Bedarf: `INSTITUT/HANDOVER_2026-10-06.md` (letzter Forschungsstand), `INSTITUT/KERNAUSSAGE_2026-10-07.md`, `INSTITUT/PLAN_2026-10-06_gesamt.md`, `IDEEN_KI/PLAN_2026-10-13.md`.

## 2. Wo was liegt
| Ordner | Inhalt |
|---|---|
| `INSTITUT/` | Forschungsinstitut: `direktor.py`, `abteilungen/`, `agenda.json`, `LABORBUCH.md`, `FEHLERBUCH.md`, `ergebnisse/`, `seeds.json`, `labor/` (gemeinsam, statistik, training, abbildungen, laufstatus) |
| `IDEEN_KI/` | lokale Ideen-KI (llama.cpp, Stufe S1 gebaut, M0-Messung am PC steht aus) |
| `JUGEND_FORSCHT/` | Gliederung, Jury-Übung, Zeitleiste, Veröffentlichungsplan (DOI), Befund-Dashboard |
| `LITERATUR/` | `literatur.bib`, `notizen.md`, `neuheit_index.json` (aus `quellen_daten.py` erzeugt) |
| `WERKSTATT/` | lokale Worker, Routine-Worker, GPU-Sperre |
| `MPI/` | Slurm-Vorlagen und Anleitung für den MPI-Zugang |
| `werkzeuge/`, `tests/`, `.github/` | Seed-Prüfung, FEHLERBUCH-Lint, Zahlen-Audit, Pre-commit-Hook, pytest, CI |
| `UEBERARBEITUNG_2026-09-23/`, `A/`, `B/` | ältere Vorarbeit (Projekt A/B, Chatbot-Adapter T-00x, Q-020) |

## 3. Wichtige Befehle
```bash
cd INSTITUT
python direktor.py status | offen | vortest | lauf <ID> | warte <ID> | neu I-0xx
python ../werkzeuge/seeds_pruefen.py --neu 2100 2119   # Seed-Block vor dem Rechnen prüfen
cd .. && python -m pytest -m "not langsam"             # alle schnellen Tests
bash werkzeuge/hooks_installieren.sh                   # einmalig: Vorregistrierungs-Hook (Windows: .cmd)
IDEEN_KI/start.cmd  /  IDEEN_KI/stopp.cmd [/hart]      # Ideen-KI (siehe IDEEN_KI/README.md)
```

## 4. Offen (Stand 14.10.)
- **Malte am PC:** M0-Modellmessung für die Ideen-KI (`IDEEN_KI/README.md`), dann erste Nacht; Hook einmal installieren.
- **T-006** (Adapter L = 20/40): Rohdaten auf `main`, Auswertung steht aus. Ob der Lauf fertig wurde, prüfen.
- Nächste Forschung laut Handover: I-043 (Fastfood-Schicht im Sprachmodell), jeweils erst nach Maltes Freigabe.
- Jugend forscht: Anmeldung bis **30.11.2026** (Zeitleiste in `JUGEND_FORSCHT/ZEITLEISTE.md`).

## 5. Bekannte Mängel (aus der Endkontrolle 14.10., nicht behoben, weil Ergebnisse festgeschrieben sind)
- `abteilungen/struktur_aequivalenz.py` (I-038): Der einseitige Wilcoxon-Test liefert in Gegenrichtung einen zu kleinen p-Wert. In Datenrichtung ist er korrekt.
- `abteilungen/mnist100.py`: Funktionen sind mehrfach definiert, die toten Fassungen sind teils kaputt (die letzte läuft).
- `skalierung_n10.py` (`WT`) und `struktur_gross.py` (`netz`) enthalten je einen undefinierten Namen; die Laufzeitwirkung ist ungeprüft.
- I-015-hybrid und I-024 teilen die MNIST-Seeds 100–109; I-015-hybrid war als „frisch“ vorregistriert.
- Der `ternaer`-Vortest schlägt fehl, weil seine Seeds inzwischen anderweitig belegt sind (xfail).
- Lint: 52 Warnungen in Altmodulen (`python werkzeuge/fehlerbuch_lint.py`).
