# GPU-Sperre: Routine-Worker und Ideen-KI (Schnittstellen-Notiz)

*Stand 14.10.2026 · gehört zu Plan R5.1 · Code: `WERKSTATT/routine/gpu_sperre.py` (nur Standardbibliothek, getestet in `test_routine.py`)*

## Problem
Die RTX 3050 hat 8 GB VRAM. Ideen-KI (Dauerbetrieb, `IDEEN_KI/schmiede/betrieb.py`) und Routine-Worker (`routine.py`) dürfen **nie gleichzeitig** ein Modell nutzen oder laden. Die Ideen-KI läuft ohne Ruhezeit, der Routine-Worker hält die Ruhezeit 00:00–06:00 ein (startet dann nicht).

## Dateien (alle in `WERKSTATT/`, per `WERKSTATT/.gitignore` nie im Repo)
| Datei | Inhalt | Wer schreibt |
|---|---|---|
| `gpu.lock` | Besitzer der GPU: `besitzer`, `zweck`, `pid`, `host`, `start`, `herzschlag`, `ablauf`, `token` (JSON). Wird atomar mit `O_CREAT\|O_EXCL` angelegt, Herzschlag per `os.replace` | wer gerade die GPU nutzt |
| `gpu.anfrage` | Vorrangflagge „Ich will die GPU“: `von`, `zweck`, `pid`, `host`, `ablauf` (30 min), `vram_frei` (bool) | Routine-Worker |
| `gpu.vram_frei` | Quittung „Modell entladen, VRAM ist frei“: `zeit`, `pid` | Ideen-KI (nur wenn `vram_frei: true` angefragt wurde) |

**Veraltet (herrenlos)** ist `gpu.lock`, wenn der Besitzer-Prozess auf demselben Rechner nicht mehr lebt, oder der Herzschlag älter als 180 s ist, oder `ablauf` überschritten ist (Standard 600 s). Veraltete Sperren darf jeder übernehmen (das macht `GpuSperre.versuche()` selbst). Eine Anfrage zählt nicht mehr, wenn ihr Prozess tot oder `ablauf` vorbei ist. Abstürze blockieren die GPU also höchstens ~3 min (Ideen-KI-Absturz: sofort, da die PID geprüft wird).

## Protokoll
1. **Kurze Leasen.** Die Ideen-KI hält die Sperre nur **während eines einzelnen LLM-Aufrufs** (Sekunden bis max. 90 s, `llm_timeout_s`), nicht rundenlang. Dazwischen ist die GPU frei.
2. **Worker meldet sich an:** `gpu.anfrage` schreiben, dann `gpu.lock` mit `O_EXCL` versuchen (alle 2 s, bis `warte_max_s` = 900 s). Bei Zeitüberschreitung nimmt der Worker die Anfrage zurück, macht ohne Modell weiter (nur Rohbericht, `STATUS: teilweise`) und endet mit Exit-Code 0/1, nicht mit Gewalt.
3. **Ideen-KI respektiert die Anfrage:** Solange eine fremde `gpu.anfrage` offen ist, **holt sie sich die Sperre nicht neu** und ruft das Modell nicht auf. Sie pausiert (Stopp-Datei wird dabei weiter jede Sekunde geprüft), schreibt `pause: "GPU an routine-worker"` in `status.json` und wartet.
4. **Worker arbeitet:** Sperre halten nur für seine 1–3 LLM-Aufrufe (je ≤ 90 s), Herzschlag alle 30 s im Hintergrund. Datensammeln, Vortest und Berichte schreiben laufen **ohne** Sperre.
5. **Worker gibt frei:** `gpu.lock` löschen (nur wenn der eigene `token` drinsteht), dann `gpu.anfrage` (und `gpu.vram_frei`) löschen. Danach macht die Ideen-KI weiter.

### Zwei Betriebsarten (`konfig.json` → `gpu_sperre.modus`)
- **`geteilter_server` (Standard, empfohlen):** Der Worker spricht denselben llama.cpp-Server an wie die Ideen-KI (`http://host:port` aus `IDEEN_KI/config.yaml` → `server`). Das Modell bleibt geladen, es gibt keinen VRAM-Konflikt; die Sperre sorgt nur dafür, dass die Ideen-KI während der Worker-Aufrufe schweigt (kein Verdrängen von Slots/Prompt-Cache, saubere Zeitmessung). Keine Quittung nötig.
- **`eigener_server`:** Der Worker braucht ein *anderes* Modell/anderen Server und damit freien VRAM. Er fragt mit `vram_frei: true` an. Die Ideen-KI muss dann nach ihrem laufenden Aufruf **den Server beenden** (`server.beenden()`), `gpu.vram_frei` schreiben und warten, bis `gpu.anfrage` verschwindet; danach Server wieder starten (`server_pruefen()` startet bei ungesundem Server neu). Der Worker wartet auf die Quittung, bevor er sein Modell lädt. *Bisher nicht in `routine.py` eingebaut (kein Start eines eigenen Servers); die Sperre unterstützt das Protokoll bereits.*

## Was `betrieb.py` tun soll (Vorschlag, nicht von diesem Auftrag geändert)
Alle LLM-Aufrufe der Ideen-KI laufen über `Betrieb.frage(...)`. Dort einhängen:

```python
# schmiede/betrieb.py (Vorschlag)
import sys
sys.path.insert(0, str(self.repo / "WERKSTATT" / "routine"))      # oder gpu_sperre.py nach schmiede/ kopieren
from gpu_sperre import GpuSperre, anfrage_offen

# in Betrieb.__init__:
self.gpu = GpuSperre("ideen-ki", "Ideenrunde", datei=self.repo / "WERKSTATT" / "gpu.lock", max_halten_s=120)

def warte_auf_gpu(self) -> None:
    """Blockiert, bis die GPU uns gehört. Prüft dabei jede Sekunde die STOP-Datei (schlafe -> pruefe_stopp)."""
    while True:
        self.pruefe_stopp()
        anf = anfrage_offen(self.gpu.datei)
        if anf and anf.get("von") != "ideen-ki":                  # Vorrang für den Worker
            self.schreibe_status(extra={"pause": f"GPU an {anf.get('von')} ({anf.get('zweck', '')})"})
            if anf.get("vram_frei"):
                self.modell_entladen_und_quittieren()              # nur Betriebsart eigener_server
            self.schlafe(5)
            continue
        if self.gpu.versuche():                                    # kurze Leihe (<= 90 s, Herzschlag nicht nötig)
            return
        self.schlafe(2)                                            # Worker hält die GPU

def frage(self, rolle, system, user):
    self.warte_auf_gpu()
    try:
        ...bisheriger Aufruf von self.llm.chat(...)...
    finally:
        self.gpu.freigeben()                                       # immer, auch bei Fehler/Stopp
```
`modell_entladen_und_quittieren()` (nur für `eigener_server`): `self.server.beenden()`; `gpu.vram_frei` mit `{"zeit": time.time(), "pid": os.getpid()}` atomar schreiben; warten (`schlafe`), bis `anfrage_offen(...)` leer ist; Quittung löschen; `self.server_pruefen()`.

Weitere Regeln für die Ideen-KI:
- `abschluss()` (Stopp/Absturz-Handler) ruft `self.gpu.freigeben()`; die Datei ist sonst nach ≤ 3 min ohnehin herrenlos.
- **Stopp < 5 min** bleibt gültig: Pausen laufen über `schlafe()` (prüft `STOP` jede Sekunde); längste Blockade ist ein laufender Worker-Aufruf (≤ 90 s).
- Bei Pause wegen Worker **nicht** den Fehlerzähler (`max_fehler_in_folge`) erhöhen und **nicht** als Ruhezeit verbuchen. Die Pause erscheint nur in `status.json`.
- Die Ideen-KI schreibt nur `gpu.lock` und `gpu.vram_frei`, nie `gpu.anfrage`; sie löscht fremde Dateien nicht (`GpuSperre.freigeben()` entfernt `gpu.lock` nur bei eigenem `token`).
- Tests (Attrappen-LLM, wie in `IDEEN_KI/tests/`): (1) offene fremde Anfrage ⇒ kein `llm.chat`-Aufruf, `STOP` beendet die Pause in < 5 s; (2) Anfrage verschwindet ⇒ Betrieb läuft weiter; (3) herrenlose `gpu.lock` (toter PID) ⇒ wird übernommen; (4) Absturz im LLM-Aufruf ⇒ `gpu.lock` danach weg.

## Was der Routine-Worker garantiert
- Nur eine Sperre, nur für die LLM-Aufrufe; Freigabe im `finally`; bei Strg+C ebenfalls.
- Hängt der Server (Timeout 90 s) oder ist er nicht erreichbar: zwei Versuche, dann Freigabe und Rohbericht — niemals Endlosschleife auf der GPU.
- Keine Gewalt: kein Beenden fremder Prozesse, kein Löschen einer *lebenden* fremden Sperre (`routine.py sperre --erzwinge-frei` räumt nur herrenlose weg).
- Kein Git (`add/commit/push`), Schreiben nur unter `WERKSTATT/ergebnisse/<ID>/` und die drei `gpu.*`-Dateien.

## Grenzen / offene Punkte
- Die Sperre ist eine **Absprache zweier kooperierender Programme**, kein Betriebssystem-Schutz. Ein dritter Prozess (z. B. von Hand gestartetes `ollama`, das M0-Messskript `m0_messung.py`) ignoriert sie. Vor M0 den Worker und die Ideen-KI stoppen.
- Annahme: beide Prozesse laufen auf **demselben Rechner** mit demselben Dateisystem (WERKSTATT-Ordner). Über Netzlaufwerke sind `O_EXCL` und PID-Prüfung nicht verlässlich.
- Mit dem späteren Heimserver (RTX 3090, 24 GB, `SERVER_PLAN.md`) können beide Modelle zugleich passen; dann `gpu_sperre.modus` bleibt, die Sperre dient nur noch der Zeitmessung/Ordnung und kann in der Ideen-KI abgeschaltet werden.
- `eigener_server` ist nur als Protokoll vorbereitet, nicht in `routine.py` verdrahtet.
