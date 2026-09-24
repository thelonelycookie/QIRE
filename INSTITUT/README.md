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
| **Abt. Literatur** | Websuche, sparsam | Neuheit prüfen, bevor gebaut wird; Quellen im Laborbuch |
| **Labor** | `labor/gemeinsam.py` | Daten (Ziffern, MNIST, Fashion), exakter Wilcoxon, Parallelisierung, Ablage |

Weitere Abteilungen kommen dazu, sobald eine Idee aus `IDEEN.md` die
Papierprüfung besteht.

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
```

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
| `IDEEN.md` | Ideenbörse: neue Ansätze mit Papierprüfung und Abbruchkriterium |
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
