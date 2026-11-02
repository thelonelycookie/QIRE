# PLAN 02.11.2026: Wissensbasis, dann Kreativ-Neustart (Entwurf, noch nicht gestartet)

## Ziel (Malte, 02.11.)
Alles, was auch nur entfernt für QIRE relevant ist, dokumentieren. Dabei schon logische Schlüsse ziehen und neue Wege und Perspektiven finden. Der quanten-inspirierte Aspekt bleibt immer im Blick. Erst wenn die Wissensbasis gesättigt ist, startet der kreative Teil (Ideen-Loop) wieder.

## Rahmen (entschieden 02.11.)
- **Breite:** Kern + Nachbarn + ausgewählte Fernfelder (Mittelding).
- **Ende von Phase 1:** Sättigung, siehe unten.
- **Format:** Markdown je Themenfeld + JSON-Tabelle aller Quellen.
- **Takt:** nur auf Maltes Zuruf, eine Welle je Zuruf, keine geplanten Aufgaben. Späher und Ideen-Loop bleiben deaktiviert (nicht gelöscht).
- **Rollen:** Opus plant, prüft und zieht Schlüsse. Sonnet-Agenten recherchieren und schreiben.
- **Ehrlichkeit:** Die Wirkung ist klassisch (I-038). Formuliert wird „quanten-inspiriert“. Zahlen aus Snippets gelten als „ungeprüft“, bis sie im Volltext stehen.

## Bestand (wird übernommen, nicht neu gesucht)
- `LITERATUR/literatur.bib`, `notizen.md`, `quellen_daten.py` → `neuheit_index.json`
- `INSTITUT/recherche/*` (Breitrecherche 05.10., Vertiefung, Chat-Sweep, Volltexte)
- Branch `welle-2026-10-27-ergebnisse`: `zyklus/STROM.md` (Späher-Funde), `REAKTOR.md`, `ARCHIV.md`, `Z1–Z4_*`
- Eigene Befunde: `LABORBUCH.md`, `FEHLERBUCH.md`, Z3-Ernte (Rotation gegen faire Gegner ≈ gleichwertig)

## Ablage
```
WISSEN/
  INDEX.md            Übersicht: Felder, Stand, Sättigung, offene Fragen
  quellen.json        eine Zeile je Quelle (Schema unten), einzige Quelle der Wahrheit
  felder/<FELD>.md    je Themenfeld: Landkarte, stärkste Baselines, Zahlen, Lücken, Bezug zu QIRE
  schluesse/S-xxx.md  abgeleitete Schlüsse (Opus-geprüft)
  wege.md             neue Wege/Perspektiven, noch keine Ideen-Bewertung
  werkzeuge/          pruefen.py (Schema, Duplikate, Sättigungszahl)
```
Schema `quellen.json`: `id` (arXiv/DOI), `titel`, `jahr`, `feld[]`, `kernaussage`, `zahlen[{wert, kontext, geprueft:bool}]`, `baseline_fuer_qire`, `quantenbezug` (direkt/Analogie/keiner), `bezug_qire_befund` (I-/Z-ID), `stuetzt_widerspricht`, `volltext_gelesen:bool`, `gefunden_am`, `welle`.

## Themenfelder (Vorschlag, ca. 22)
**Kern (K):** K1 Gewichtsquantisierung PTQ/QAT · K2 Aktivierungs- und KV-Quantisierung · K3 Rotationen (Hadamard, Butterfly, gelernt) · K4 Gitter/Codebücher/Vektorquantisierung · K5 Tensornetze (MPO/MPS/TT) · K6 quanten-inspirierte ML (Born-Maschinen, Dequantisierung, Clifford/Magie) · K7 strukturierte Matrizen (Fastfood, Monarch, Hadamard-MLP) · K8 Niedrigbit-/ternäres Training (BitNet, ParetoQ).
**Nachbarn (N):** N1 Skalengesetze (auch für Präzision) · N2 effiziente Architekturen (SSM, Hybrid, MoE, LRU) · N3 Distillation · N4 Sparsity + Pruning · N5 Energie/Hardware (Bit-Op-Kosten, CPU/Edge) · N6 Zufalls-/Seed-Gewichte, Random Features · N7 Aktivierungsfunktionen (ReLU², Primer) · N8 Rate-Distortion- und Informationsgrenzen.
**Fernfelder (F, Analogien):** F1 Quantenfehlerkorrektur/Codierungstheorie · F2 Renormierungsgruppe/Multiskala · F3 Signalverarbeitung/Kompression (Wavelets, Rahmen) · F4 Neurobiologie (Spikes, sparse Codes) · F5 offene Quantensysteme/Dekohärenz · F6 Physik-Simulation (Monte-Carlo, Tensornetz-Simulation).

Frage an Malte vor dem Start: Felder streichen/ergänzen?

## Phase 1: Wissensbasis

### Welle 0: Aufbau (1 Zuruf, billig)
1. Sonnet A: `WISSEN/` anlegen, `pruefen.py` bauen + Test.
2. Sonnet B: Bestand (bib, notizen, STROM, recherche/*) nach `quellen.json` überführen, Duplikate zusammenführen, je Quelle Felder zuordnen.
3. Opus: Stichprobe 10 % auf Richtigkeit, `INDEX.md` mit Lücken je Feld.

### Wellen 1…n: Recherche je Feld (je Zuruf 4–6 Felder parallel)
Je Feld ein Sonnet-Agent mit festem Auftrag:
1. `quellen.json` + Feld-Datei lesen (was ist bekannt?).
2. ≤ 12 Websuchen: Übersichtsarbeiten zuerst, dann neueste (2025–2026), dann stärkste Baseline.
3. Nur neue Quellen eintragen. Zahlen nur mit Kontext; `geprueft=true` nur bei gelesenem Volltext/Abstract.
4. Feld-Datei aktualisieren: Landkarte, stärkster Gegner für QIRE, offene Lücken, **Quantenbezug je Abschnitt**.
5. Schlussnachricht ≤ 8 Zeilen: Zahl neuer Quellen, wichtigster Fund, Widerspruch zu eigenen Befunden?
Danach Opus: Duplikate/Qualität prüfen, Sättigungszahl je Feld eintragen.

### Querschnitt nach jeder 2. Welle (Opus + 1 Sonnet)
- **Schlüsse:** Was folgt logisch aus Literatur + eigenen Befunden? (Beispiel: Z3 sagt, Gruppenskalen ≈ Rotation. Was sagt die Literatur zu den Kosten beider im Betrieb?) Jeder Schluss bekommt eine Datei `S-xxx` mit Belegen, Gegenbelegen und der langweiligsten Erklärung.
- **Wege:** neue Perspektiven in `wege.md`, jeweils mit Quantenbezug. Eine Bewertung findet hier noch nicht statt.
- **Widerspruchs-Check:** Belegt eine Quelle etwas gegen einen LABORBUCH-Befund, kommt das sofort in die Meldung an Malte.

### Sättigung (Ende von Phase 1)
- Ein Feld ist **gesättigt**, wenn zwei aufeinanderfolgende Runden < 10 % neue relevante Quellen bringen (gemessen an den Einträgen des Feldes).
- Phase 1 ist fertig, wenn alle K- und N-Felder gesättigt sind und jedes F-Feld mindestens eine Runde hatte.
- Kurzmeldung an Malte nach jeder Welle: Stichpunkte, Sättigung je Feld.

## Phase 2: Kreativ-Neustart (erst nach Sättigung und Maltes Zuruf)
- Grundlage ist die Methode aus `KREATIVITAET_2026-10-27.md` und `zyklus/LOOP.md`, ergänzt um:
  - Jede Idee muss `quellen.json` zitieren (nächste bekannte Arbeit) und einen Neuheitsabgleich bestehen.
  - Der stärkste Gegner kommt aus der Feld-Datei, nicht aus dem Gedächtnis.
  - Die Lehre aus Z3 gilt: Gegner sind immer Lloyd-Gitter, Gruppenskala (g16/g32) und QAT, bei gleichen Bytes/FLOPs und zwei Trainingslängen.
- Rauchtests nur auf Tuning-Seeds. Vorregistrierte Läufe nur nach Maltes Freigabe.
- Ein neuer Plan für Phase 2 wird geschrieben, wenn Phase 1 steht.

## Kosten-Schutz
- Nur auf Zuruf, ≤ 6 Sonnet-Agenten je Welle, ≤ 12 Suchen je Agent.
- Wenn das Nutzungslimit erreicht wird: sauber abbrechen, Zwischenstand committen. Der Zuruf geht dann einfach weiter.

## Offene Fragen an Malte
1. Themenfelder so in Ordnung? Fehlt etwas oder ist etwas zu viel?
2. Ort: neuer Ordner `WISSEN/` auf `main`, oder unter `LITERATUR/`?
3. Branch `welle-2026-10-27-ergebnisse`: STROM/REAKTOR/ARCHIV nach `main` übernehmen und den Branch archivieren?
4. Sollen Volltexte (PDF) lokal gespeichert werden (gitignored) oder nur Links?
