# Projekt A — QIRE · Ordnerkarte

**Ressourceneffizientere KI mithilfe von Quantenphysik.** Jugend forscht,
Anmeldung **30. November 2026**. Ziel bis dahin: ein **Chatbot** und eine
**zweite KI-Art**.

Dieser Ordner ist eigenständig. Einem neuen Chat reicht der Ordner —
`CLAUDE.md` wird automatisch gelesen und enthält die Arbeitsanweisung.

**Zuerst lesen:** `CLAUDE.md`, dann `HANDOFF_AKTUELL.md`.

---

## Dokumente

| Datei | Was drinsteht |
|---|---|
| **`CLAUDE.md`** | Arbeitsanweisung: Ziel, Regeln, Umgebung, Arbeitsweise. Wird automatisch geladen. |
| **`HANDOFF_AKTUELL.md`** | **Der Einstieg.** Stand 30. August. Abschnitt 0 = was offen ist, Abschnitt 5 = die Protokollregeln mit Begründung. |
| `HANDOFF_PROJEKT_A.md` | älterer Einstieg (6. August), gilt für den Verlauf bis dahin |
| **`PROJEKTTAGEBUCH.md`** | **maßgeblich.** Jede Zahl mit Herleitung, jeder Fehler mit Diagnose. 226 KB. |
| `STAND_UND_OFFENES.md` | Stichpunktfassung beider Teilprojekte — bester Schnellüberblick |
| `BEFUNDE_UND_RICHTUNGEN.md` | ~60 Stichworteinträge: Befunde, Theorie, Läufe, Fehlerquellen |
| `ENTWURF_T_SERIE_KERNEL.md` | Entwurf und Roadmap der T-Serie bis 30. November |
| `ENTWURF_ZEITREIHEN.md` | Entwurf für Q-019, Modul noch nicht gebaut |
| `FORSCHUNGSIDEEN.md` | zweite Blicke auf verworfene Befunde |
| `KURZFASSUNG_ENTWURF.md` | Anmeldedokument |
| `ROADMAP.md` | Verlauf, historisch |
| `RESULTS.md`, `SCIENTIFIC_LIMITS.md` | ältere Zusammenfassungen |
| **`UMGEBUNG.md`** | Python, Pakete, Rechnergrenzen, Neuaufbau |
| **`DATEN_EXTERN.md`** | **was nicht hier liegt** — Korpus (2,9 GB) und B-Rohdaten |

## Code

| Ort | Inhalt |
|---|---|
| `quantum_sim/` | Projekt A. Die Experimentmodule (`unitarity_depth`, `frozen_angles`, `global_orthogonal`, `sprachmodell`, `zeitreihen` …) sowie `q_summary` und `q_audit`. |
| `qire_ai/` | geteilte Infrastruktur: `result_store` (Rohdaten), `equivalence` (TOST, Holm), `data_loader`, `benchmark`, `metrics`. Enthält auch den abgeschlossenen Projekt-B-Code. |
| `server_setup/` | RunPod: `runpod_start.sh`, `runpod_t002.sh`, `alles_durchlaufen.sh` |
| `sources/literature_cards/` | acht Literaturkarten. **Pauschal als ungelesen markiert, tatsächlich sind ~drei Viertel gelesen** — muss vor der schriftlichen Arbeit nachgezogen werden. |
| `py.cmd` | ruft die virtuelle Umgebung auf (liegt außerhalb, siehe `UMGEBUNG.md`) |

## Rohdaten

```
results/experiments/     Q-003-replication, Q-009, Q-011, Q-014, Q-015,
                         Q-018, T-000, T-001   (JSON + CSV)
results/experiments/vorlaeufe_q011/   verworfene Q-011-Vorläufe, NICHT verwertbar
results/*.log            23 Lauf-Logs der Q- und T-Serie
results/korpus/          bpe_4096.json (Tokenizer) + val.npy
```

Der Rest des Korpus (2,9 GB) und die Projekt-B-Rohdaten liegen im
Archivordner — **`DATEN_EXTERN.md` lesen, bevor der gelöscht wird.**

## Prüfen, dass alles da ist

```powershell
.\py.cmd -m quantum_sim.q_audit      # muss 106 von 106 bestehen
.\py.cmd -m quantum_sim.q_summary    # Belegstand aller Läufe
```
