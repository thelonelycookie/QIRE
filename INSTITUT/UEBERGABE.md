# Übergabe an den nächsten Chat — QIRE-Institut

*Stand 05.10.2026 (Abschnitt 0 neu; §4/§5 sind Stand 25.09. und durch §0 überholt). Wer das hier liest, braucht vorerst nichts anderes.
Einzelheiten stehen in den verlinkten Dateien; lies sie nur, wenn du sie brauchst.*

---

## 0. Stand 05.10.2026 — zuerst lesen

**Seit dem 25.09. gerechnet (alle im `LABORBUCH.md`, Rohdaten in `ergebnisse/`):**

| ID | Kernzahl | Bedeutung |
|---|---|---|
| I-004-mnist | R-1 H1: zufälliges Clifford-Gerüst + 4 Magiewinkel, **ein** Training, G = 0,69 | weder volles Training noch nachträgliches Runden nötig |
| I-005 | diskrete Clifford-Suche (0 Magie) = Niveau volles Training (MNIST, 10 Seeds) | Positivkontrolle für das Verfahren bestanden (G_L 0,96) |
| I-005-v2 | MNIST **GLEICHWERTIG** (TOST, d −0,14 pp); Fashion **VOLL_BESSER** (−1,31 pp) | **MNIST braucht keine Magie, Fashion schon** |
| I-005-v3 | Fashion: M50_S = **4 Magiewinkel** (≈ 123 T) auf der Suche-Schaltung; k32 ab Suche nur 0,42 | Suche sitzt in anderem Becken als der Gradient |
| I-036 | Suche − globaler Walsh +5,6 pp (MNIST) / +3,0 pp (Fashion); Suche − fester 2×2-Block-Walsh **+0,93 pp** (MNIST, 19/20) | Clifford-Zelle ist **kein** Walsh-Klassifikator — aber ein fester 2×2-Block holt ohne Suche schon 72 % des Abstands |
| I-036-v2 | Suche − gleiche Blöcke ohne Maske **+1,9 pp** (MNIST) / **+2,4 pp** (Fashion); Wechselwirkung Maske×Blöcke +2,1/+2,7 pp | Maske und Blöcke tragen **nur gemeinsam** — der 0-T-Befund ist Leistung der Suche im Clifford-Raum, keine feste Transformation |

**Kernsatz neu (05.10.):** *Auf MNIST erreicht eine reine Clifford-Schaltung
(0 T-Gatter), per diskreter Suche gefunden, nachweislich das Niveau des vollen
Trainings. Auf Fashion fehlen ihr 1,31 pp, von denen 4 gezielt gewählte
Magiewinkel die Hälfte schließen. Der Clifford-Nutzen ist zum größten Teil
lokale Hadamard-Mischung (fester 2×2-Block-Walsh), die Suche liefert den
letzten Prozentpunkt — und zwar nur als abgestimmtes Paar aus Walsh-Blöcken und
quadratischer Vorzeichenmaske (I-036-v2); globale Walsh-Merkmale schaden.* Einordnung (Recherche
29.09.): Rz ist auf NISQ-Hardware virtuell — „0 bzw. 4 Magiewinkel“ ist dort
vor allem eine Aussage über klassische Simulierbarkeit, auf fehlertoleranter
Hardware über T-Kosten.

**Projekt A parallel** (`../UEBERARBEITUNG_2026-09-23/A/`, Einstieg
`HANDOFF_AKTUELL.md` §000): Q-020, Q-019W, T-004, **T-005** gerechnet
(Chatbot-Adapter: Verlust fällt streng monoton mit der Lagenzahl, L10 = 55 %
des LoRA-Gewinns mit 1/16 der Parameter, QIRE+LoRA ergänzt nicht).

**Offen im Institut:** I-036-v3 (Idee: welche Maskeneigenschaft trägt? — vor dem
Bauen Literatursuche), I-003-v2, I-010, B-paritaet, V-fashion (Prio 3).

**Arbeitsorte:** Malte rechnet auf `claude/busy-faraday-e57hna`, Cloud-Sitzungen
auf eigenen Branches (z. B. `claude/remote-control-pc-start-gj69o1`) und mergen
von dort. Ein Container-Lauf setzt in `agenda.json` vorher den Status
„läuft (Container)“. **Lange Läufe im Container ohne Wiederaufnahme sind
riskant** (I-036 ging einmal mit einem Container-Neustart verloren, FEHLERBUCH 05.10.).

**Entscheidungen für Malte:**
- Merge der Cloud-Branches nach `main` bzw. welcher Branch maßgeblich ist.
- Ob lange Läufe künftig im Container (kostenlos, aber abbruchgefährdet) oder
  nur auf dem PC laufen sollen.
- T-002/RunPod (Geld), Aufteilung A/B für die Anmeldung, IBM-Token,
  Frist der schriftlichen Arbeit (siehe `../UEBERARBEITUNG_2026-09-23/A/HANDOFF_AKTUELL.md` §8).

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
7. **Vorrang für echte Neuheit** (25.09., nach einer Nacht voller sauber
   gemessener, aber teils schon publizierter Genres): „Nur überprüfen, ob
   Profis richtig rechnen, kann ich mir auch sparen.“ Vor dem Skalieren
   einer Linie über einen ersten Testlauf hinaus: **echte Literatursuche**
   (WebSearch, nicht nur Trainingswissen) auf die Kernaussage. Fragen
   bevorzugen, zu denen nichts Vergleichbares gefunden wird — nicht Fragen,
   bei denen nur eine andere Verpackung eines bekannten Tricks herauskommt.
   Bekannte Genres (z. B. approximative Kodierung, GPU-Batching von
   Schaltungen) offen als Replikation/Werkzeug kennzeichnen, nie als
   Hauptbefund. Vorrang: **neue Dinge finden** vor **neue Zusammenhänge
   erschließen** vor **Bekanntes nachrechnen**. Quantenphysik ist jung genug,
   dass „einfach mal ausprobieren, was noch niemand gefragt hat“ eine
   legitime Strategie ist.

8. **Opus plant, Sonnet führt aus** (07.10.). Planung, Versuchsdesign,
   Vorregistrierung, Auswertung und Urteile macht Opus. Umsetzung (Module
   schreiben, Vortests, Läufe vorbereiten, Routinearbeit) geht an Sonnet-Agenten
   (`Agent` mit `model: "sonnet"`) mit genauem Auftrag. Opus prüft deren Ergebnis.

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
Neu gelernt spät in der Nacht/am Morgen 25.09.: **Ein kombinierter Befund aus
mehreren datensatzspezifisch kalibrierten Bausteinen generalisiert nicht
automatisch** — I-034 (2,25 % T-Gatter, MNIST) repliziert auf Fashion NICHT
(I-034-fashion H0, 10–12 %, messbar schlechter); jeder Baustein (hier: die
Top-m-Ladegröße) braucht eine eigene Kalibrierung je Aufgabe · **Die einfache
2-Tuning-Seed-Ratensuche kippt zuverlässig** (dreimal in einer Nacht: C-referenz,
I-003, I-035) — dafür gibt es jetzt `labor.gemeinsam.gittersuche_stabil`/
`waehle_rate` (Stabilitätsabstand), kanonisch für neue Module · ein Ausreißer
kann bei populationsbasierten Kennzahlen (M50 aus Stufenmitteln) eine ganze
Messstufe unbrauchbar machen, nicht nur einen Datenpunkt verschieben — vor der
Urteilsbildung auf das modul­eigene Instabilitätskriterium prüfen, nicht nur
danach zählen · **echte Literatursuche vor dem Skalieren** (Vorgabe 7): das
kombinierte Laden-Sparen (I-020/I-033) ist im Kern bereits publiziert
(West et al. 2023, MNIST/Fashion, IBM-Hardware) — als Replikation kennzeichnen.

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
| Magie-Kaskade | I-015 K-1 **H1** (28 % der Schaltungs-T), mit Laden 114 % | **Hybrid** (Clifford-Stufe klassisch): I-015-hybrid **H1** 11,7 % (MNIST), 23,9 % (Fashion) |
| Magie-Konzentration im Laden | I-033 **H1** | Top-48 Ladewinkel = 45 % Lade-T, −0,57 pp; Top-16 schlägt Zufall um 31 pp |
| **Alles kombiniert (MNIST)** | I-034 **H1** | **174 T = 2,25 % von 7.698 T, +0,31 pp** (m=32: 1,47 %, −0,04 pp) |
| **Alles kombiniert (Fashion)** | I-034-fashion **H0** — repliziert NICHT | 10–12 % statt ≤5 %, messbar schlechter (−0,34/−0,64 pp); Top-m-Laden ist auf Fashion verlustreicher, kombinierter Befund ist datensatzabhängig |
| Magie gegen Datenmenge | I-024 **H0 (Gegenrichtung)** | Konzentration schärft sich mit mehr Daten (G(2) 0,34 → 0,61) |
| Magie-Skalierung | C-skala S-1 „H1" **war Ratenartefakt** → C-skala-v2 **beide HU** | Mit stabiler Referenz löst sich der Hinweis auf sublineares Wachstum auf; Frage bleibt offen |
| Magiebedarf aus Aufgaben-Schwierigkeit | I-035 P-2 **H0**, P-1 HU | Generalisierungslücke sagt M50 NICHT voraus (2. gescheiterter billiger Kennwert nach SRE/I-002); Punktschätzer deutet (n. s.) auf *weniger* Magie bei mehr Rauschen — Gegensatz zur I-003-Vermutung |
| Magie lässt sich nicht wegtrainieren | C-bewusst **H0** | Strafterm zum Gitter kostet 1,4 pp |
| Wahl der magischen Winkel nach Wirkung > Zufall | I-001 P-2 **H1** | +0,29 G, p = 0,027 |
| Vorhandene ≠ genutzte Magie | I-002 **H0** | gelernte SRE ≈ zufällige (6,26 vs 6,28) |
| Magie in T-Gattern (Qiskit, verifiziert) | C-hardware | T ≈ 31·k (ε=10⁻³); k=0 → 0 T; M50 ≈ 62 T |
| **Datenladen ist der Engpass** | C-kodierung **K-1 H1** | Laden 7.636 T = 7,7× ganze Schaltung; dünn besetzt 2,4–3,9 Tsd. T |
| Schuss-Budget | I-008 | ~1.000 Messungen je Bild für < 1 pp Verlust |
| Verschränkungs-Kompass (Q1) | HU/HU | SVD schlägt alle Tensor-Train-Ordnungen |
| Zero-Noise-Extrapolation über Bitbreiten | auf Papier widerlegt | MSE 7× schlechter |

**NACHTRAG 29.09. — I-005-v3 (Fashion, 20 Seeds):**
- 4 Magiewinkel auf der gesuchten Clifford-Schaltung schließen die halbe
  Lücke zu voll (M50_S = 4, ≈ 123 T). 8 Winkel schließen sie nicht ganz
  (VOLL_BESSER, −0,43 pp).
- Die Auswahl nach Verlust schlägt die Zufallsauswahl (H1).
- Alle 32 Winkel ab der Suche nachtrainiert erreichen nur G = 0,42. Die Suche
  liegt also in einem anderen Becken als der Gradient.
- Details im LABORBUCH, 29.09. 16:20.

**NACHTRAG 28.09. ABENDS — I-005-v2 (20 Seeds, vorregistriert):** MNIST
Suche = voll (**GLEICHWERTIG**, ±0,5 pp, 0 T-Gatter). Fashion **VOLL_BESSER**
um 1,31 pp (20/20): Dort trägt die Magie wirklich. Der Magiebedarf ist also
aufgabenabhängig, und der Absatz direkt hierunter ist teilweise überholt.
Nächste Frage: Suche + k Magiewinkel auf Fashion (LABORBUCH 28.09. 20:34).

**NACHTRAG 28.09. — I-005 (alle H1) relativiert den Kernsatz unten:** Eine
diskrete Clifford-Suche (**0 Magiewinkel, 0 T-Gatter**) erreicht auf MNIST
0,9352, das volle Gradiententraining 0,9344 (beschreibend). Die
Magie-Konzentration misst also vermutlich, wie viel Magie der **Gradient**
braucht, nicht die Aufgabe. Nächste Fragen: I-005-fashion und Suche gegen
voll per TOST (LABORBUCH 28.09.). Außerdem ist I-004-mnist ausgewertet (R-1 H1:
ein Training im zufälligen Gerüst genügt). Neuheitsrecherche:
`recherche/2026-09-28_neuheit_magie.md`.

**Kernsatz (Stand 25.09. Vormittag):** *Der Nutzen der Quantenschaltung sitzt
in wenigen Magiewinkeln (Magie-Konzentration, MNIST + Fashion repliziert,
zwei unabhängige Seed-Sätze). Auf MNIST braucht die Kombination aus
klassischer Clifford-Vorstufe (Hybrid-Kaskade) + sparsamem Laden + wenigen
Magiewinkeln nur **2,25 % der T-Gatter** des naiven Wegs ohne Genauigkeits-
verlust (I-034) — **auf Fashion repliziert genau diese Kombination NICHT**
(I-034-fashion H0): die Bausteine sind einzeln robust (Magie-Konzentration,
Hybrid-Kaskade je für sich auf beiden Datensätzen bestätigt), aber ihre
gemeinsame Kalibrierung (Top-m-Größe, Kaskadenstufen) ist datensatzabhängig.
Das approximative Laden allein (I-020/I-033) ist zudem im Kern schon
publiziert (West et al. 2023) — als Replikation zu behandeln, nicht als
Hauptbefund. Zwei Versuche, den Magiebedarf einer Aufgabe billig vorherzusagen
(SRE in I-002, Generalisierungslücke in I-035), sind beide gescheitert (H0).
Die Skalierungsfrage (M50 vs. Winkelzahl) ist nach C-skala-v2 wieder offen.*
Arbeitstitel für den Namen: **Magie-Konzentration M50/M90 einer Aufgabe**
(auf zwei Datensätzen, zwei unabhängigen Seed-Sätzen belegt — der Rest der
Nachtarbeit hat eher relativiert als bestätigt, das gehört ehrlich dazu).

## 5. Was gerade offen ist (Stand 25.09. Vormittag — überholt, siehe §0)

Alles aus der Nacht 24./25.09. ist gerechnet (C-referenz, I-003, I-020, I-015,
I-015-hybrid[-fashion], I-024, I-033, I-034[-fashion], C-skala[-v2], I-035 —
alle `status: erledigt` in `agenda.json`, alle in `LABORBUCH.md`/`FEHLERBUCH.md`).
Prüfen wie immer: gibt es `ergebnisse/<ID>__*.json`? Wenn nicht →
`python direktor.py lauf <ID>`.

**Zwei Module vorregistriert, gebaut, Vortest bestanden — noch NICHT gerechnet**
(waren an Malte übergeben, `institut.cmd lauf <ID>`; Stand dieses Chats
unbekannt, da noch kein Push von dort — als Erstes prüfen):

| ID | Frage | Laufzeit (4–5 Kerne) |
|---|---|---|
| **I-004-mnist** | Clifford-Gerüst + k Winkel von Anfang an (inkl. n_train=500) auf MNIST — I-004 auf Ziffern war HU wegen zu kleiner Auflösung | ~75–85 min |
| **I-005** | Diskrete Clifford-Suche (simulated annealing über 2^(nL) Winkelwahlen): findet sie die Lehrer-Lösung aus I-003 (Positivkontrolle fürs Verfahren) und schlägt sie zufällige Clifford-Schaltungen auf MNIST? | ~35–50 min |

**Neue offene Ideen** (Vorgabe 7, „echte Neuheit" — vor dem Bauen: Literatur
prüfen, siehe `ideen.json` Feld `neuheit`):
- **I-003-v2**: Positivkontrolle, die das Lernverfahren tatsächlich erreicht
  (I-005 liefert das evtl. mit, dann I-003 selbst wiederholen).
- **I-034-fashion-v2** (Idee, nicht angelegt): Top-m/Kaskadenstufen je
  Datensatz neu kalibrieren statt MNIST-Werte übernehmen — `zwei_0_8_m48`
  lag mit 6,5 % schon nahe am Ziel.
- Nach I-035: ein dritter Kandidat für einen billigen Magiebedarfs-Prädiktor
  (SRE und Generalisierungslücke sind beide widerlegt) — z. B. Verschränkungs-
  /Korrelationsstruktur der Daten (Brücke zu V-kompass), noch nicht formuliert.
- **I-001 auf MNIST**, **I-010** (Prio 2), B-paritaet, V-fashion (Prio 3).

**Werkzeug, kein Befund:** `labor/stapel.py` (GPU-Encoder/Decoder-Batching,
Korrektheit exakt geprüft, CPU-Gewinn gering) — der eigentliche Test braucht
eine CUDA-GPU; offene Frage an Malte, ob sein PC eine hat.

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
committen, dann rechnen. **Lernrate mit vielen freien Winkeln:**
`labor.gemeinsam.gittersuche_stabil`/`waehle_rate` benutzen (Stabilitätsabstand:
Kriterium = schlechtester Tuning-Seed, eine Rate nur wählbar, wenn ihr oberer
Gitternachbar auf keinem Tuning-Seed kollabiert) — nicht die einfache
2-Seed-Mittelwert-Suche neu erfinden, die hat in einer Nacht dreimal denselben
Fehler produziert (C-referenz, I-003, I-035: FEHLERBUCH). Wer aus
Zeitgründen doch die einfache Suche nimmt, muss das begründen UND eine
Ausreißer-Prüfung *vor* der Urteilsbildung einbauen, nicht nur danach zählen.

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
