# Projekt A — QIRE · Ordnerkarte

**Ressourceneffizientere KI mithilfe von Quantenphysik.** Jugend forscht.
Anmeldung **30. November 2026**, schriftliche Arbeit **Mitte Januar 2027**
(eine Landesseite nennt den 17.01.2027 — prüfen). Ziel: ein **Chatbot** und eine
**zweite KI-Art**.

Dieser Ordner ist eigenständig. Einem neuen Chat reicht der Ordner —
`CLAUDE.md` wird automatisch gelesen und enthält die Arbeitsanweisung.

*Stand dieser Karte: 23. September 2026 (Gesamtreview). Frühere Fassung:
31. August.*

---

## Lesereihenfolge

1. **`CLAUDE.md`** — Arbeitsanweisung: Ziel, Fristen, Regeln, Umgebung.
2. **`HANDOFF_AKTUELL.md`** — der Einstieg. Abschnitt 00 = was seit dem
   23. September gilt, Abschnitt 5 = die Protokollregeln mit Begründung.
3. **`../BERICHT/00_REVIEW_BEFUNDE.md`** — der Gesamtreview vom 23. September.
4. **`STAND_UND_OFFENES.md`** — Stichpunktfassung beider Teilprojekte.
5. **`PROJEKTTAGEBUCH.md`** — maßgeblich. Neue Einträge stehen vor dem
   Abschnitt „Zwischenstand“; der jüngste ist vom 23. September.
6. **`KURZFASSUNG_ENTWURF_v2.md`** — die aktuelle Argumentationslinie für
   Anmeldung und Arbeit.
7. **`ENTWURF_Q020.md`** und **`LITERATUR_UND_ALLEINSTELLUNG.md`** — nächster
   Lauf und Einordnung in die Literatur.

Wer nur zehn Minuten hat: `HANDOFF_AKTUELL.md` Abschnitt 00, dann
`KURZFASSUNG_ENTWURF_v2.md`.

---

## Dokumente

### Aktuell

| Datei | Was drinsteht |
|---|---|
| **`CLAUDE.md`** | Arbeitsanweisung. Wird automatisch geladen. |
| **`HANDOFF_AKTUELL.md`** | **Der Einstieg.** Stand 23. September. |
| **`PROJEKTTAGEBUCH.md`** | **Maßgeblich.** Jede Zahl mit Herleitung, jeder Fehler mit Diagnose. Zurückgenommenes bleibt mit Begründung stehen. |
| `STAND_UND_OFFENES.md` | Stichpunktfassung beider Teilprojekte, Stand 23. September |
| `KURZFASSUNG_ENTWURF_v2.md` | Kurzfassung mit Platzhaltern für Q-020 und T-002, Belegtabelle, Gliederung der Langfassung |
| `BEFUNDE_UND_RICHTUNGEN.md` | ~60 Stichworteinträge; Stand 6. August, Status und Kernaussagen am 23. September nachgeführt |
| `ENTWURF_Q020.md` | Q-020: Kernaussage gegen faire klassische Referenz; Verschränkung × Tiefe ohne Encoder |
| `LITERATUR_UND_ALLEINSTELLUNG.md` | Einordnung in die Literatur (u. a. Quantum-PEFT), was eigen ist |
| `ENTWURF_T_SERIE_KERNEL.md` | Entwurf und Roadmap der T-Serie (T-000 bis T-002) |
| `ENTWURF_ZEITREIHEN.md` | Entwurf für Q-019. Das Modul `quantum_sim/zeitreihen.py` ist **gebaut und geprüft, nicht gelaufen** |
| **`UMGEBUNG.md`** | Python, Pakete, Rechnergrenzen, Neuaufbau |
| **`DATEN_EXTERN.md`** | **was nicht hier liegt** — Korpus (2,9 GB), B-Rohdaten, und was es nirgends gibt |

### Historisch — nicht als Stand lesen

| Datei | Stand | Aktuell ist stattdessen |
|---|---|---|
| `KURZFASSUNG_ENTWURF.md` | 6. August | `KURZFASSUNG_ENTWURF_v2.md` |
| `HANDOFF_PROJEKT_A.md` | 6. August | `HANDOFF_AKTUELL.md` |
| `ROADMAP.md` | 4. August (110 KB, Verlauf ab 1. August) | `HANDOFF_AKTUELL.md`, `STAND_UND_OFFENES.md` |
| `FORSCHUNGSIDEEN.md` | 5. August | `BEFUNDE_UND_RICHTUNGEN.md`, `ENTWURF_Q020.md` |
| `RESULTS.md`, `SCIENTIFIC_LIMITS.md` | Anfang August, ältere Zusammenfassungen | `STAND_UND_OFFENES.md` |

Die historischen Dateien bleiben unverändert stehen, weil sie den Verlauf
dokumentieren. `ROADMAP.md`, `HANDOFF_PROJEKT_A.md` und `FORSCHUNGSIDEEN.md`
tragen oben einen Hinweis.

## Code

| Ort | Inhalt |
|---|---|
| `quantum_sim/` | Projekt A. Experimentmodule (`hybrid_replication`, `global_orthogonal`, `frozen_angles`, `unitarity_depth`, `zeitreihen`, `sprachmodell`, neu `kernaussage` für Q-020 …) sowie `q_summary` und `q_audit`. |
| `qire_ai/` | geteilte Infrastruktur: `result_store` (Rohdaten), `equivalence` (TOST, Holm), `data_loader`, `benchmark`, `metrics`. Enthält auch den abgeschlossenen Projekt-B-Code. |
| `demo/` | **Standdemo** (seit 23. September hier, vorher nur im Archiv): `python -m demo.demo_app`. Die drei angezeigten Genauigkeiten sind Demonstrationswerte aus einem Seed, keine Messergebnisse. |
| `server_setup/` | RunPod: `runpod_start.sh`, `runpod_t002.sh`, `alles_durchlaufen.sh`. **Vor einem T-002-Lauf** `HANDOFF_AKTUELL.md` Abschnitt 7 lesen (vier Fehler). |
| `sources/literature_cards/` | Literaturkarten; zur Einordnung zuerst `LITERATUR_UND_ALLEINSTELLUNG.md` |
| `py.cmd` | ruft die virtuelle Umgebung auf (liegt außerhalb, siehe `UMGEBUNG.md`) |

## Rohdaten

```
results/experiments/     Q-003-replication, Q-009, Q-011, Q-014, Q-015,
                         Q-018, T-000, T-001   (JSON + CSV)
results/experiments/vorlaeufe_q011/   verworfene Q-011-Vorläufe, NICHT verwertbar
results/*.log            Lauf-Logs der Q- und T-Serie
results/korpus/          bpe_4096.json (Tokenizer) + val.npy
```

Seit dem 30. August sind keine neuen Messläufe dazugekommen (Stand
23. September). Der Rest des Korpus (2,9 GB) und die Projekt-B-Rohdaten liegen
im Archivordner — **`DATEN_EXTERN.md` lesen, bevor der gelöscht wird.**

## Prüfen, dass alles da ist

```powershell
.\py.cmd -m quantum_sim.q_audit      # Stand 30. August: 106 von 106
.\py.cmd -m quantum_sim.q_summary    # Belegstand aller Läufe
```

`qire_ai.audit` läuft hier nicht durch (B-Rohdaten fehlen) — das ist erwartet.
