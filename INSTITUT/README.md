# Das QIRE-Institut — Gründungsurkunde

*Gegründet am 24. September 2026. Eine Forschungseinrichtung im Kleinen:
mehrere Abteilungen, ein Direktor, ein gemeinsames Labor — und alles, was
rechnet, läuft auf Maltes PC, nicht in einem Chatfenster.*

**Leitfrage:** Wo spart die Quantenphysik einer KI **wirklich** Rechenaufwand —
und wo sieht es nur so aus?

---

## 1. Aufbau

| Rolle | Wer / was | Aufgabe |
|---|---|---|
| **Direktor** (CEO) | `direktor.py` + Claude in den Sitzungen | Agenda führen, Abteilungen starten, Urteile lesen, **vorher festgelegte** Folgefragen freischalten, Fehlerbuch führen, `STAND.md` schreiben |
| **Abt. Clifford** | `abteilungen/clifford.py` | Magie-Budget: wie viele „teure“ Nicht-Clifford-Winkel braucht eine Aufgabe? → multiplikationsfreie Schichten |
| **Abt. Theorie** | Claude, auf Papier (`LABORBUCH.md`) | Ideen **vor** dem Rechnen durchdenken; was sich auf Papier widerlegen lässt, wird nicht gerechnet |
| **Abt. Kontrolle** („Red Team“) | Kontrollarme in jedem Modul | für jede Behauptung die langweiligste Erklärung als eigenen Arm mitlaufen lassen (Zufall, Hadamard, Identität, linear) |
| **Abt. Ideenschmiede** | `abteilungen/ideenschmiede.py`, `ideen.json` | erfindet neue Ideen (morphologischer Kasten: 24 Quantenkonzepte × 10 KI-Engpässe × 8 Denkzüge, Gewinner-Abteilungen werden bevorzugt), schickt jede Karte durch ein Tor (Papierprüfung → Neuheit → Abbruchkriterium, billigster Test, Kontrolle) und **leitet bestandene Karten an die zuständige Abteilung in die Agenda weiter** |
| **Abt. Verschränkung** | `abteilungen/kompass.py` | Tensornetz-/Verschränkungsmethoden (erste Linie: DMRG-Ordnung, ruht nach HU/HU) |
| **Abt. Literatur** | Websuche bzw. ein Recherche-Agent, sparsam | Neuheit prüfen, bevor gebaut wird; Quellen auf der Ideenkarte |
| **Labor** | `labor/gemeinsam.py` | Daten (Ziffern, MNIST, Fashion), exakter Wilcoxon, Parallelisierung, Ablage |

Weitere Abteilungen kommen dazu, sobald eine Idee aus `IDEEN.md` die
Papierprüfung besteht.

## 1a. Grundsatz: quantenphysikalisch, nicht klassisch optimiert

*Maltes Vorgabe vom 24.09.2026.* Das Institut sucht Effizienz **mit**
Quantenphysik, nicht mit gut getarnten klassischen Tricks. Jede Idee trägt
eine Quantenstufe:

- **Q3** — läuft als echte Quantenschaltung (Qiskit, IBM-Hardware), der Effekt
  ist eine Quantenressource (Magie, Verschränkung, Messung).
- **Q2** — exakte Quantenmechanik (Zustandsvektor, Unitäre, Born-Regel) im
  Simulator, auf Hardware übertragbar.
- **Q1** — nur mathematische Analogie. Wird **nicht gestrichen**, bekommt
  aber die niedrigste Priorität (Direktor rechnet Q3 vor Q2 vor Q1). Beispiel:
  V-kompass (Tensor-Train auf klassischen Gewichten) ist Q1 → Priorität 3.

Die Hauptlinie *Magie-Budget* ist Q3: Nicht-Clifford-Gatter (T-Gatter) sind
die teuerste Ressource eines fehlertoleranten Quantencomputers. Eine Aufgabe,
die nur wenige magische Winkel braucht, ist **auf echter Quantenhardware
billiger** — und im Simulator zugleich multiplikationsfrei. Dieselbe Zahl
spart in beiden Welten.

## 2. Wie große Forschungszentren forschen — und was wir davon übernehmen

Kein Forschungszentrum hat ein Geheimrezept, aber die guten teilen dieselben
Gewohnheiten. Wir übernehmen sie ausdrücklich:

1. **Heilmeier-Fragen (DARPA)** — vor jedem Projekt: Was genau wollen wir?
   Wie macht man es heute, und wo ist die Grenze? Was ist neu an unserem Weg?
   Woran merken wir nach einer Woche, ob es klappt? → steht in jedem Modul oben.
2. **Starke Schlussfolgerung (Platt, *Science* 1964)** — immer *mehrere*
   Hypothesen gleichzeitig, und das Experiment so bauen, dass es zwischen
   ihnen **entscheidet**. Deshalb hat jedes Modul Kontrollarme für die
   langweiligen Erklärungen.
3. **Falsifizierbarkeit (Popper) + Vorregistrierung** — Kriterien stehen im
   Code, *bevor* eine Zahl entsteht. Das Urteil schreibt der Code, nicht die
   Hoffnung.
4. **Das billigste entscheidende Experiment zuerst** — Minuten auf 8×8-Ziffern
   vor Stunden auf MNIST vor Tagen auf der GPU. Erst wenn das Kleine trägt,
   wird skaliert.
5. **Theorie vor Rechenzeit** — was sich auf Papier widerlegen lässt, wird
   nicht gerechnet (Beispiel: Idee Z-1 im Laborbuch).
6. **Fehler sind Daten (Bell Labs, Hamming)** — ein gescheitertes Experiment
   wird nicht gelöscht, sondern *vertieft*: Warum scheitert es? Welche
   Annahme war falsch? Was folgt daraus? → `FEHLERBUCH.md`. Die Folgefragen
   für jeden Ausgang stehen **vor** dem Lauf in `agenda.json`.
7. **Rote Mannschaft** — jemand versucht aktiv, den Befund kaputtzumachen.
   Bei uns: die Kontrollarme und die Protokollregeln aus Projekt A.
8. **Laborbuch** — datiert, ehrlich, mit zurückgenommenen Aussagen.

Die sieben Protokollregeln aus Projekt A (`A/CLAUDE.md` §4) gelten hier
unverändert: Lernrate je Arm mit Gitterverlängerung in beide Richtungen und
vollem Budget, genug Seeds, TOST statt „nicht signifikant“, Holm nur über die
angemeldete Familie, Rohdaten sofort sichern, Positivkontrolle bzw.
Auflösungsprüfung, Kriterien vor dem Lauf.

## 3. So wird gearbeitet (token-sparend)

```bash
cd INSTITUT
python direktor.py vortest          # Pflicht vor jedem Lauf
python direktor.py naechstes        # rechnet den nächsten offenen Punkt
python direktor.py status           # schreibt STAND.md (eine Seite)
python direktor.py alle             # alles Freigeschaltete nacheinander (über Nacht)
python direktor.py ideen neu 8      # Ideenschmiede: 8 neue Rohlinge
python direktor.py ideen leiten I-006   # Karte durchs Tor schicken
```

Auf Windows einfach `institut.cmd alle` (benutzt Maltes QIRE-Umgebung, 5 Kerne,
Log in `ergebnisse/direktor.log`). Energiesparmodus aus, sonst stirbt der Lauf
mit dem Ruhezustand (Lehre aus Q-011).

**Wie aus einem Rohling eine Idee wird:** Die Schmiede zieht nur *Fragen*
(„Wie senkt Konzept X den Engpass Y, wenn man Z tut?“). Ein Rohling wird in
`ideen.json` ausgefüllt — oder in einer Zeile verworfen (Status `verworfen`,
Grund dazu; verworfene Karten bleiben stehen). Das Ausfüllen ist der einzige
Schritt, der Denken (Malte oder Claude) braucht; alles andere ist Mechanik.

Mit Maltes Umgebung: `..\UEBERARBEITUNG_2026-09-23\A\py.cmd direktor.py ...`
bzw. der dort genannte Python-Pfad. Braucht nur `torch`, `numpy`,
`scikit-learn` (alles schon in der QIRE-Umgebung). MNIST/Fashion werden aus
dem vorhandenen Cache `A/results/datasets/*.npz` gelesen oder einmalig
heruntergeladen (~30 MB). `INSTITUT_PROZESSE=5` nutzt 5 der 6 Kerne.

**Arbeitsteilung PC ↔ Claude:** Der PC rechnet (kostet nichts). Claude
denkt, baut neue Module und liest danach **nur** `STAND.md` und den
neuesten Eintrag im `FEHLERBUCH.md` — nicht die Logs. Das spart den
Großteil der Tokens.

## 4. Dateien

| Datei | Inhalt |
|---|---|
| `STAND.md` | automatisch: alle Punkte, Status, letztes Urteil |
| `agenda.json` | Forschungsagenda mit vorregistrierten Folgefragen |
| `LABORBUCH.md` | datierte Einträge: Überlegungen, Ergebnisse, Zahlen mit Herleitung |
| `FEHLERBUCH.md` | jeder gescheiterte Ausgang und was daraus gelernt wurde |
| `IDEEN.md` | Ideenbörse (erste Runde, ausführlich) |
| `ideen.json` | Ideenschmiede: alle Karten mit Status, Prüfungen, Quellen |
| `institut.cmd` | Startknopf für Windows |
| `ergebnisse/` | Rohdaten (JSON), nie überschreiben |

## 5. Zum Namen

Malte möchte, dass etwas seinen Namen trägt. Ehrlich: In der Wissenschaft
benennt man eine Methode nicht selbst nach sich — das tun *andere*, wenn
sie sie benutzen und zitieren („Liebscher-Schranke“ entsteht in fremden
Arbeiten). Der Weg dahin ist: etwas wirklich Neues finden, sauber belegen,
veröffentlichen (Jugend forscht → arXiv → Workshop-Paper). Was wir tun
können: einer neuen, belegten Größe einen **Arbeitstitel** geben, der sich
eindeutig auf die eigene Arbeit bezieht. Kandidat, sobald er belegt ist:
das **Magie-Budget M\*** einer Aufgabe (siehe `LABORBUCH.md`).
