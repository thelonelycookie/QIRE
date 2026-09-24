# Projekt B — Bündel

**Zuerst lesen: [HANDOFF_PROJEKT_B.md](HANDOFF_PROJEKT_B.md).**

Jugend forscht, Anmeldung 30. November 2026. Dieses Teilprojekt fragt:

> Wovon hängt ab, wie breit eine Schicht sein muss — und warum scheitert
> deshalb jeder strukturierte Ersatz einer dichten Schicht?

---

## Der wichtigste Hinweis

Dieser Ordner ist ein **Bündel zum Lesen**, kein Arbeitsverzeichnis.
Experimente laufen im eigentlichen Projekt:

```
C:\Users\Malte\Downloads\qire_ai_console_v1_0_remote_research_platform\qire_ai_console_v1_0
```

`code/` ist ein **Spiegel**. Wer dort etwas ändert, ändert nichts am Projekt.
Nach Änderungen im Projekt den Spiegel auffrischen:

```powershell
.\spiegel_aktualisieren.ps1
```

An Projekt A wird parallel in einer eigenen Sitzung gearbeitet. **Immer nur ein
schwerer Rechenlauf gleichzeitig** — Begründung in der Übergabe, Abschnitt 1.

---

## Inhalt

| Datei / Ordner | Was drin ist |
|---|---|
| `HANDOFF_PROJEKT_B.md` | Stand, belegte Befunde, zurückgezogene Aussagen, Protokollregeln, offene Punkte |
| `RESULTS.md` | jede Aussage mit ihrer Rohdatei — erzeugt, nicht geschrieben |
| `TAGEBUCH_B.md` | die sechs Tagebucheinträge zu B (Auszug) |
| `daten/` | alle Rohdaten der B-Serie, JSON und CSV |
| `abbildungen/` | die vier Abbildungen |
| `code/` | Spiegel der B-Module |

---

## In einem Absatz

Die ursprüngliche These — quanteninspirierte Strukturen ersetzen eine dichte
Schicht bei weniger Parametern — ist gescheitert, und zwar mehrfach und
nachweisbar. Übrig geblieben ist etwas Besseres: eine Erklärung, **warum** sie
scheitern musste. Der Effektivrang einer trainierten Schicht wird von der
**Aufgabe** gesetzt, nicht von der Architektur und nicht von der Geometrie der
Eingaben — belegt mit zwei kontrollierten Variablen und einer untrainierten
Kontrolle. Die Aufgabe braucht rund sieben Richtungen; der Rang war deshalb nie
die bindende Beschränkung, und mehr Budget konnte nie helfen.

Dazu kommen drei einzeln gemessene Gründe, warum frühere Messungen — auch die
eigenen — das Gegenteil zu sehen glaubten.
