# Chat-Sweep B (Sitzungen 24.-25.09.2026): was fehlt im Repo?

Stand 06.10.2026. Nicht committet.

## Abdeckung (ehrlich)

Per `list_events` (nur Typ "user") gelesen, jeweils vom Ende rueckwaerts:
- Sitzung 3 (`01482fV5...`, Übergabe.md Überblick): **vollstaendig** (nur 3 echte Nutzernachrichten).
- Sitzung 2 (`01F8Q8nu...`): nur die letzten ca. 6 Stunden (25.09. ab ca. 09:00 bis Ende). Dort 3 echte Nutzernachrichten, sonst Unteragenten-Rauschen. Der Rest (24.09. abends bis 25.09. 09:00) **nicht gelesen**.
- Sitzung 1 (`0142Abqr...`): nur 24.09. ab ca. 19:50 bis Ende (Übergabe-Erstellung, PR-Merge). Gründung/Vorgaben (Anfang) **nicht gelesen**.
Grund: Die API liefert 100 Rohereignisse je Seite, davon fast alles Werkzeug- und Unteragenten-Ausgaben; eine Sitzung hat hunderte Seiten. Nutzernachrichten sind nur an `client_platform: desktop_app` erkennbar. Vollstaendiger Sweep braucht ein Skript ausserhalb der Chat-Werkzeuge oder gezielte Zeitfenster.

## Funde

| Fund | Quelle | steht im Repo? | Wichtigkeit |
|---|---|---|---|
| Malte will auf seinem PC (Ryzen 5 3600 oder 3500, RTX 3050, vermutlich 8 GB VRAM) eine lokale KI ueber Ollama aufsetzen, mit der er "die ganze Zeit forscht"; Claude behaelt "das letzte Wort". Anforderungen: Interface ohne Kontext-/Laufzeitlimit pro Anfrage (Nacht-Laeufe), eher Coder als Chatter, Internetsuche, Anleitung fuer maximale Rechenleistung | S3, 25.09. 16:03 (user) | NEIN. Nur Ryzen 5 3600 / 16 GB in `B/HANDOFF_PROJEKT_B.md` und `labor/gemeinsam.py`; `A/qire_ai/research_agent/local_agent_adapter.py` erwaehnt Ollama nur technisch. Kein Wunsch, keine Auswahl (Websuche der Sitzung nannte u. a. Qwen3.5-9B, Falcon H1R 7B, Gemma 4 E4B, OpenHands, Ollama auf 0.0.0.0:11434), keine Anleitung | hoch (Arbeitsweise ab jetzt, GPU-Hardware fehlt in Hardware-Notizen) |
| Malte: "Repo und alle Branches aufraeumen, Projekt-Trennung (A/B) belassen" | S3, 25.09. 11:25 | TEILWEISE: `UEBERGABE.md` Z. 41/47 nennt Branch-Frage als offen. Remote hat weiter `ccr-d65e1410-munlkt`, `claude/busy-faraday-e57hna`, `claude/ubergabe-repo-overview-4mx71n`; Loeschen per git push scheiterte mit HTTP 403 (Proxy), Wiederaufnahme per GitHub-MCP (`delete branch`) unklar | mittel |
| Fehlermeldung: `institut.cmd vortest`, `institut.cmd lauf I-004-mnist`, `institut.cmd lauf I-005` "funktioniert nicht" (Malte, Windows) | S2, 25.09. 13:54 | NEIN (Ursache/Fix nicht auffindbar; I-004-mnist-Ergebnis steht spaeter im Repo, vermutlich anders gelaufen) | mittel (Windows-Pfad pruefen: `institut.cmd` vor naechstem Lauf testen) |
| Malte: "Übergabe gucken, ob sie auf dem neuesten Stand ist" (wiederkehrende Pflicht vor Chat-Ende) | S2, 25.09. 11:20 | JA (UEBERGABE.md wird gepflegt) | niedrig |
| Vorgabe 7 "Vorrang fuer echte Neuheit, Literatursuche vor Skalierung" (aus S2 erkennbar im Code/Commit cb96fee) | S2 (Agentenkontext) | JA (UEBERGABE.md, ideen.json, STAND, PLAN) | erledigt |
| Termine: Anmeldung 30.11.2026, schriftliche Arbeit Mitte Januar 2027 | S1-Übergabetext (24.09.) | JA (UEBERGABE.md Z. 58, A/HANDOFF) | erledigt |
| Namenswunsch ("etwas soll meinen Namen tragen"), Arbeitstitel "Magie-Konzentration M50/M90" | S1 Übergabetext | JA (UEBERGABE.md Vorgabe 5, Z. 201) | erledigt |
| Offene Folgearbeiten aus 24.09.: C-skala, I-001 auf MNIST, I-005, I-010 (Magie-Quantisierung Butterfly im Chatbot T-002), B-paritaet, V-fashion | S1 Übergabe | zum Teil (C-skala/I-034/I-035 gelaufen; I-005/I-010/B-paritaet siehe agenda.json: mit `grep` pruefen) | niedrig-mittel |
| Dein Auftrag "nicht beobachten" (PR #1 nicht ueberwachen) | S1, 24.09. 20:29 | nicht relevant (Sitzungsverhalten) | niedrig |
| Hinweis Zugang: Git-Push-Delete bekommt 403 durch Agent-Proxy; IBM-Token/RunPod stehen nur als offene Punkte | S3, UEBERGABE Z. 50 | JA (als offene Punkte) | niedrig |

## Kurzfazit

Aus dem gelesenen Teil fehlt im Repo vor allem Maltes Wunsch nach einer lokalen Forschungs-KI (Ollama, RTX 3050) samt Hardware-Eckdaten und Interface-Anforderungen. Daneben ist die Branch-Aufraeum-Aufgabe nur teilweise erledigt und der Windows-Fehler bei `institut.cmd` nicht dokumentiert. Termine und Vorgaben 1-7 sind bereits in `UEBERGABE.md`. Der grosse Rest der beiden laengeren Sitzungen (Gruendung, Nacht 24./25.09. bis 09:00) ist ungelesen; dort stehen vermutlich weitere Nutzerwuensche (Quantenstufen Q1-Q3, Agenten-Erlaubnis, Token-Sparen), die aber bereits in Vorgaben 1-6 der UEBERGABE.md stehen.
