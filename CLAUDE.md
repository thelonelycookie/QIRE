# QIRE — Jugend-forscht-Projekt von Malte (Einstieg für jede Sitzung)

**Alles auf Deutsch:** Code, Kommentare, Dokumente, Antworten.
**Ziel:** ressourceneffizientere KI mithilfe von Quantenphysik. Anmeldung 30.11.2026, schriftliche Arbeit Mitte Januar 2027.

## Zuerst lesen
1. Das neueste `INSTITUT/HANDOVER_*.md`: Stand, laufende Läufe, offene Entscheidungen.
2. `INSTITUT/UEBERGABE.md` §0–3: Maltes Vorgaben und Protokollregeln (verbindlich).
3. Das neueste `INSTITUT/PLAN_*_gesamt.md`: der gültige Plan.
4. Nur für den Chatbot-Adapter (Projekt A): `UEBERARBEITUNG_2026-09-23/A/CLAUDE.md`.

## Arbeitsweise
- **Antworten sehr kurz, in Stichpunkten** (Token sparen). Große Ideen erst in 2–3 Sätzen spiegeln. QIRE-Fakten aus den Dateien lesen, nicht aus dem Gedächtnis.
- **Opus plant, Sonnet-Agenten führen aus** (Agent-Tool, model „sonnet“): Recherche, Modulbau, Rechnungen. Opus nur, wo Sonnet nicht reicht.
- **Vor jedem neuen vorregistrierten Lauf auf Maltes Zustimmung warten** (Stand 06.10.: das Nutzungslimit ist knapp; die Regel kann sich wieder ändern). Modulbau, Vortest und Analyse brauchen keine Freigabe.
- **Ziel (Malte, 07.10.):** eine quanten-inspirierte, ressourceneffiziente KI bauen. Neuheit ist nicht nötig: Bekanntes (Fastfood, Hadamard-MLP, Primer usw.) darf als Grundlage dienen und wird zitiert. Das ersetzt Vorgabe 7 (Vorrang für Neuheit) aus UEBERGABE §2.
- Ehrlich bleiben: Die Wirkung von QIRE ist klassisch (I-038). Formuliert wird „quanten-inspirierter Vorteil“, nicht „Quantenvorteil“.
- Ergebnisse: Institutsläufe kommen in `INSTITUT/LABORBUCH.md` und `agenda.json`, Q/T-Läufe von Projekt A in Tagebuch, `q_audit` und `q_summary`.

## Technik
- Container: `pip install torch numpy scipy scikit-learn qiskit`. Daten liegen in `INSTITUT/daten/` (gitignored), `labor/gemeinsam.py` lädt sie.
- Läufe: `INSTITUT_PROZESSE=4 OMP_NUM_THREADS=1 python3 direktor.py lauf <ID>`. Der Container hat nur 4 Kerne, also einen Lauf nach dem anderen. Wiederaufnahme je Seed ist Pflicht, Zwischenstände committen.
- Auf einen Prozess nie mit `pgrep -f <name>` warten, wenn der wartende Befehl selbst `<name>` enthält; sonst wartet er auf sich selbst.
- Rauchtests nur auf Tuning-Seeds, nie auf Mess-Seeds.
- Lange Läufe (Sprachmodell) auf Maltes PC: Ryzen 5 3600, 6 Kerne, 16 GB. PC wach halten, Energiesparen aus.
- **Commits und Pushes nur unter Maltes Namen:** keine Co-Authored-By-, Session- oder KI-Zeilen, KI im Repo-Inhalt nicht erwähnen. Auf dem zugewiesenen Branch arbeiten, vor dem Push `git pull --no-rebase`.
