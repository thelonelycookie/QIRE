# Übergabe an den nächsten Chat — QIRE-Institut

*Stand 25.09.2026, ca. 00:15 (zweiter Chat 24./25.09.). Wer das hier liest, braucht vorerst nichts anderes.
Einzelheiten stehen in den verlinkten Dateien; lies sie nur, wenn du sie brauchst.*

---

## 1. Wer, was, warum (30 Sekunden)

- **Malte** macht allein ein Jugend-forscht-Projekt: **ressourceneffizientere KI
  mithilfe von Quantenphysik**. Anmeldung 30.11.2026, schriftliche Arbeit
  Mitte Januar 2027. Alles auf **Deutsch** (Code, Kommentare, Dokumente).
- Projekt A/B (Ordner `UEBERARBEITUNG_2026-09-23/`, `A/`, `B/`) ist die ältere,
  sehr gründliche Vorarbeit. Einstieg dort: `UEBERARBEITUNG_2026-09-23/A/HANDOFF_AKTUELL.md`.
- **`INSTITUT/`** ist das neue „Forschungsinstitut“, am 24.09. gegründet: ein
  Direktor (`direktor.py`) koordiniert Abteilungen (`abteilungen/*.py`), jede
  mit vorregistrierten Hypothesen, Vortest und automatischem Urteil.

## 2. Maltes Vorgaben (verbindlich)

1. **Quantenphysikalisch arbeiten**, nicht mit getarnten klassischen Tricks.
   Jede Idee hat eine Quantenstufe: **Q3** echte Quantenschaltung/-ressource,
   **Q2** exakte QM im Simulator, **Q1** nur Analogie. **Q1 wird nicht
   aussortiert, bekommt nur niedrigste Priorität** (Prio 3).
2. **Token sparen:** Rechnen läuft auf Maltes PC (`institut.cmd`). Du liest
   `STAND.md` + dieses Dokument, nicht die Logs.
3. **Aus Fehlern lernen, nicht aufgeben:** Ein Fehlschlag wird vertieft
   (`FEHLERBUCH.md`), die Folgefrage steht vorher in `agenda.json`.
4. **Neue, eigene Ideen** sind erwünscht (Ideenschmiede), aber geprüft.
5. Malte möchte, dass etwas seinen Namen tragen kann. Ehrlich bleiben: Namen
   vergeben andere; Weg = Befund belegen + veröffentlichen.
6. **Selbstständig arbeiten**, nur bei Richtung/Geld/vielstündigen Läufen fragen.
   **Agenten dürfen orchestriert werden** (Malte hat es erlaubt und die
   Skalierung ausdrücklich gewünscht).

## 3. Protokollregeln (aus Projekt A, gelten weiter)

Lernrate je Arm auf eigenen Tuning-Seeds, Gitter in **beide** Richtungen
verlängern, volles Budget · genug Seeds (exakter Wilcoxon, kleinster p = 2/2ⁿ)
· „nicht signifikant“ ≠ „gleich“ (TOST) · Holm nur über die angemeldete
Familie · Rohdaten sofort sichern · Positivkontrolle muss feuern *können* ·
**Kriterien vor dem Lauf in den Code, Vorregistrierung vor den Zahlen
committen** · Vortest: jeden Urteilszweig auf synthetischen Daten erzwingen.
Neu gelernt am 24.09. (Details `FEHLERBUCH.md`): Arme brauchen denselben
Startzustand wie ihre Kontrollen (geerbter Kopf!) · Kontrollen vorher
anschauen, ob sie tun, was ihr Name sagt (Hadamard bei gerader Lagenzahl =
Identität!) · Grenzen relativ zum Effekt setzen · BLAS-Threads in
Parallelprozessen auf 1.
Neu gelernt in der Nacht 24./25.09.: **Lernrate je Arm gilt auch für
Referenzarme** — eine an wenigen freien Parametern (nur Kopf) getunte Rate
ist mit allen Winkeln frei instabil (C-referenz Fashion) · eine eingeführte
Referenz selbst auf Einzelseed-Ausreißer prüfen, bevor sie Nenner wird ·
Positivkontrolle muss mit dem *Lernverfahren* feuern, nicht nur mit der
Architektur (I-003: Gradient findet die Clifford-Lehrerlösung nicht) ·
Abbruchgrenzen an einer ausgerechneten Schranke ausrichten, nicht an der
Hoffnung (I-020) · Module liefern Urteile als **Strings**.

## 4. Die Hauptlinie: Magie-Budget (Q3) — was belegt ist

Modell „QIRE-Zelle“: Amplitudenkodierung → L=4 Lagen aus Ry(θ) je Qubit
(Kronecker) + CNOT-Ring → Born-Auslese |ψ|² → linearer Kopf. Winkel auf
Vielfachen von π/2 = **Clifford** → Schicht rechnet ohne Multiplikation und
braucht **0 T-Gatter**. Nicht-Clifford-Winkel = **Magie**.

| Befund | Stand | Zahl |
|---|---|---|
| Schaltung schlägt Identität | **H1**, Ziffern ×2 + MNIST | MNIST +3,8 pp, p = 0,002 |
| Nur Clifford bringt nichts (gerundet ≈ zufällige Clifford) | **H0** für C-2, 2 Datensätze | — |
| **Magie-Konzentration**: Nutzenanteil G(k) | MNIST + **Fashion**, je 2 unabhängige Seed-Sätze (C-mnist, C-fashion, **C-referenz**) | MNIST G(4) = 0,68/0,70, Fashion G(4) = 1,01/0,99 von 32 Winkeln; **M50 = 2–4 (MNIST), M90 = 4 (Fashion)** |
| Nachtrain-Verdacht (Rundungsarme 2. Phase) | C-referenz: **widerlegt** auf MNIST | d = −0,03 pp, KI ±0,24; Störfaktor war die Lernrate |
| Aufgabe oder Architektur? | I-003 **HU** (Positivkontrolle scheitert am Gradienten) | explorativ: Memorieren M50 = 8 vs. Generalisieren M50 = 1 |
| Gesamtbudget Laden + Schaltung | I-020 G-1 **H0** (25 %-Ziel) | dünn besetzt exakt: **4.372 T statt 8.776, ohne Verlust**; mit k=4 ist Laden 96 % |
| Magie-Kaskade | I-015 K-1 **H1** (28 % der Schaltungs-T), mit Laden 114 % | **Hybrid** (Clifford-Stufe klassisch): 14,5 % der Hardware-T, post hoc → I-015-hybrid vorregistriert |
| Magie lässt sich nicht wegtrainieren | C-bewusst **H0** | Strafterm zum Gitter kostet 1,4 pp |
| Wahl der magischen Winkel nach Wirkung > Zufall | I-001 P-2 **H1** | +0,29 G, p = 0,027 |
| Vorhandene ≠ genutzte Magie | I-002 **H0** | gelernte SRE ≈ zufällige (6,26 vs 6,28) |
| Magie in T-Gattern (Qiskit, verifiziert) | C-hardware | T ≈ 31·k (ε=10⁻³); k=0 → 0 T; M50 ≈ 62 T |
| **Datenladen ist der Engpass** | C-kodierung **K-1 H1** | Laden 7.636 T = 7,7× ganze Schaltung; dünn besetzt 2,4–3,9 Tsd. T |
| Schuss-Budget | I-008 | ~1.000 Messungen je Bild für < 1 pp Verlust |
| Verschränkungs-Kompass (Q1) | HU/HU | SVD schlägt alle Tensor-Train-Ordnungen |
| Zero-Noise-Extrapolation über Bitbreiten | auf Papier widerlegt | MSE 7× schlechter |

**Kernsatz bis jetzt:** *Der Nutzen der Quantenschaltung sitzt in der Magie,
und zwar in sehr wenigen Gattern (Magie-Konzentration). Auf echter Hardware
frisst aber das Laden der Daten die Einsparung — das ist die nächste Front.*
Arbeitstitel für den Namen: **Magie-Konzentration M50/M90 einer Aufgabe**
(Fashion trägt sie jetzt, C-referenz auch; I-003 blieb HU → I-003-v2 und C-skala abwarten).

## 5. Was gerade offen ist (Warteschlange im Cloud-Container, Nacht 24./25.09.)

Prüfen: gibt es `ergebnisse/<ID>__*.json`? Wenn nicht → `python direktor.py lauf <ID>`.

| ID | Frage | Laufzeit (4 Kerne) |
|---|---|---|
| **I-004** | Clifford-Gerüst + k Winkel von Anfang an vs. nachträglich runden | ~25–50 min |
| **I-015-hybrid** | Hybrid-Kaskade: Clifford-Stufe klassisch, nur unsichere Bilder auf Hardware (10 frische Seeds) | ~15 min |
| **I-024** | Magie gegen Datenmenge (n_train 500/2.000/10.000, faire Referenz *nachtrainiert*) | ~50–75 min |
| **I-033** | Magie-Konzentration im **Laden** (Top-m / grob-zu-fein der Gray-Code-Winkel) | ~60 min |
| **C-skala** | Wächst M50 langsamer als n·L? (n = 6/8/10 + fest10, eigene Rate für *nachtrainiert*) | ~85–105 min |

Danach (Ideen, nicht gebaut): **I-003-v2** (Positivkontrolle, die das
Verfahren erreicht), **I-005** (diskrete Clifford-Suche, aufgewertet),
**I-001 auf MNIST**, **I-010** (Prio 2), B-paritaet, V-fashion (Prio 3).

## 6. Wie man arbeitet

```bash
cd INSTITUT
python direktor.py status            # STAND.md neu (eine Seite)
python direktor.py vortest           # Pflicht vor Läufen
python direktor.py lauf I-003        # einen Punkt rechnen (--schnell = Rauchtest)
python direktor.py alle              # alles Offene nach Priorität
python direktor.py ideen neu 8       # Ideenschmiede: Rohlinge ziehen
python direktor.py ideen leiten I-0xx  # Karte durchs Tor an die Abteilung
python -m labor.nachtragen <ID> "Lehre ..."   # Lehre im FEHLERBUCH gezielt nachtragen
```
Windows/Malte: `institut.cmd alle` (seine venv, 5 Kerne, Energiesparmodus aus).
Parallelität: `INSTITUT_PROZESSE=N`. Cloud-Container: `pip install torch numpy
scipy scikit-learn qiskit`; MNIST/Fashion lädt `labor/gemeinsam.py` selbst.

**Neues Modul bauen:** Vorlage `abteilungen/magie_pruning.py` oder
`clifford.py`. Pflicht: `HYPOTHESEN`, `lauf(schnell=False, …)` → dict mit
`urteil`/`auswertung`, `vortest()`. Eintrag in `agenda.json` mit `modul`,
`status: offen`, `prioritaet`, `folgen` (vorregistrierte Folgefragen). Erst
committen, dann rechnen.

**Agenten orchestrieren (hat gut funktioniert):** Jeder Agent baut **eine
neue Datei**, fasst keine gemeinsamen Dateien an (agenda.json, direktor.py,
labor/*, *.md), kein Commit, nur Vortest + Rauchtest mit
`INSTITUT_PROZESSE=1`; der Direktor (du) prüft, trägt in die Agenda ein,
committet und startet den Lauf. Ergebnisse stichprobenartig selbst
nachprüfen, bevor sie ins Laborbuch gehen.

## 7. Dateien

| Datei | wozu |
|---|---|
| `STAND.md` | automatisch, eine Seite: alle Punkte, Status, Urteile |
| `LABORBUCH.md` | alle Ergebnisse mit Tabellen und Herleitung (datiert) |
| `FEHLERBUCH.md` | jeder Fehlschlag mit Lehre |
| `agenda.json` | Forschungsagenda mit Prioritäten und vorregistrierten Folgen |
| `ideen.json` / `IDEEN.md` | Ideenschmiede-Karten / erste Ideenrunde ausführlich |
| `README.md` | Gründungsurkunde, Abteilungen, Forschungsprozess, Quantenstufen |
| `ergebnisse/` | Rohdaten (JSON), nie überschreiben |

Branch: `claude/busy-faraday-e57hna` (Repo `thelonelycookie/qire`).
