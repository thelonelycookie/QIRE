# WERKSTATT — lokale Worker für QIRE

Claude in der Cloud plant und wertet aus. Auf Maltes PC arbeiten lokale
Worker mit einem Qwen-Modell über Ollama. Jeder Worker ist eine
Claude-Code-Session mit **Remote Control**, die Cloud-Session schickt ihr
Aufträge und bekommt Ergebnisse zurück.

```
Cloud-Claude (Planer/Auswerter)
   │  send_message (Remote Control)      + Auftragsdatei in WERKSTATT/auftraege/
   ▼
PC: Claude Code ──► Ollama (qwen, lokal, 8 GB VRAM)
   werkstatt-code     : programmieren, rechnen, testen
   werkstatt-review   : Code des Code-Workers gegenprüfen
   werkstatt-daten    : Auswertungen, Tabellen, Plots
   monitor.py         : überwacht lange Läufe (kein LLM, kostet nichts)
```

## Grenzen dieser Hardware (16 GB RAM, 8 GB VRAM)

- **Ein Modell zur Zeit.** Ollama lädt es einmal, alle Worker teilen es.
  Anfragen werden nacheinander bearbeitet (`OLLAMA_NUM_PARALLEL=1`).
- **Nie Simulation und großes Modell gleichzeitig unter Vollast** — daran ist
  T-006 schon einmal an OOM gestorben. Läuft eine lange Rechnung, wartet der
  Worker und `monitor.py` beobachtet sie.
- **Kontext 32k.** Claude Code braucht viel Kontext. Mehr frisst VRAM.
- **Die Platte ist fast voll.** Ein 7–8B-Modell braucht ~5 GB.

## Einrichtung (einmalig)

### 1. Modell prüfen

```powershell
ollama --version        # mindestens 0.14 (Anthropic-kompatible Schnittstelle)
ollama list             # welche qwen-Modelle sind da?
```

Empfohlen, je nach Platz: `qwen3:8b` (kann Werkzeuge gut aufrufen) oder
`qwen2.5-coder:7b`. Wichtig ist **Tool-Calling**, sonst kann der Worker keine
Dateien bearbeiten. Fehlt eins: `ollama pull qwen3:8b`.

### 2. Ollama-Einstellungen (einmalig, dann Ollama neu starten)

```powershell
setx OLLAMA_CONTEXT_LENGTH 32768
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_MAX_LOADED_MODELS 1
setx OLLAMA_KEEP_ALIVE 10m
```

### 3. Test: Läuft Remote Control mit lokalem Modell?

Das ist **der kritische Punkt**: Remote Control braucht die claude.ai-Anmeldung,
das Modell kommt aber von Ollama. Zuerst testen:

```powershell
cd "C:\Users\Malte\Downloads\claude\projects\QIRE AI"
.\WERKSTATT\start_worker.ps1 -Rolle code
```

Dann der Cloud-Session sagen: „Worker läuft". Sie prüft, ob sie ihn sieht und
ob er antwortet.

**Falls Remote Control mit Ollama nicht geht:** Plan B ist eine normale
Claude-Code-Session (Anthropic-Modell) als „Vorarbeiter" mit Remote Control, die
die eigentliche Arbeit per `ollama run` an das lokale Modell gibt. Dann
entscheidet die Cloud-Session, was weiter.

### 4. Ruhezeit (nachts aus)

Der Startskript verweigert den Start zwischen **22:00 und 07:00**.
Zusätzlich beendet `ruhezeit.ps1` um 22:00 Modell und Worker:

```powershell
schtasks /Create /TN "QIRE Ruhezeit" /SC DAILY /ST 22:00 /TR "powershell -ExecutionPolicy Bypass -File \"C:\Users\Malte\Downloads\claude\projects\QIRE AI\WERKSTATT\ruhezeit.ps1\""
```

## Rechte

| Wo | Was |
|---|---|
| Im Projektordner (`QIRE AI`) | Dateien lesen/ändern ohne Rückfrage, Python über `py.cmd`, Tests, `git add/commit/pull/push` auf dem Arbeitszweig |
| Die venv außerhalb | nur **ausführen** (über `py.cmd`), nicht ändern |
| Alles andere außerhalb | **nur mit Zustimmung** — die Rückfrage erscheint in der Claude-App |
| Immer verboten | `rm -rf`, `git push --force`, `git reset --hard`, `pip install` ohne Rückfrage |

Hinweis: Unter Windows gibt es keine echte Sandbox. Die Grenze setzt
`worker-settings.json` über Erlaubnislisten; Unbekanntes wird immer erfragt.

## Täglicher Ablauf

1. PC an, `.\WERKSTATT\start_worker.ps1 -Rolle code` (bei Bedarf weitere Rollen
   in eigenen Fenstern).
2. Der Cloud-Session Bescheid sagen. Sie verteilt Aufträge selbst.
3. Abends schließt `ruhezeit.ps1` alles.
