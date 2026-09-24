# Übergabe: Projekt B — Struktur, Rang und Messmethodik

Kalt lesbar. Stand: 30. August 2026.

---

## 0. Was dieser Ordner ist

Ein **Bündel**, kein Arbeitsverzeichnis. Experimente laufen weiterhin im
eigentlichen Projekt:

```
C:\Users\Malte\Downloads\qire_ai_console_v1_0_remote_research_platform\qire_ai_console_v1_0
```

Hier liegen: die Rohdaten (unveränderlich, deshalb gefahrlos kopiert), die
Abbildungen, die erzeugten Berichte, der Tagebuchauszug — und unter `code/` ein
**Spiegel** der B-Module. Der Spiegel ist zum Lesen. Geändert wird immer im
Projekt, danach `spiegel_aktualisieren.ps1` laufen lassen.

Warum kein zweites lauffähiges Projekt: Die Module importieren quer durch
`qire_ai` (Datenlader, Metriken, Trainingsschleife). Zwei Kopien, die
auseinanderlaufen, wären in einem Projekt, dessen ganzer Wert an
Nachprüfbarkeit hängt, das schlechteste Ergebnis.

---

## 1. Rechenzeit teilen

6 physische Kerne (Ryzen 5 3600), 16 GB RAM. Jeder Lauf setzt
`torch.set_num_threads(6)`. **Immer nur EIN schwerer Lauf gleichzeitig** — auch
sitzungsübergreifend, denn an Projekt A wird parallel gearbeitet.

```powershell
Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Select-Object ProcessId, CommandLine
```

Läuft dort ein `qire_ai`- oder `quantum_sim`-Prozess, nicht starten. Zwei
parallele Läufe bringen wegen SMT etwa 10 % mehr Durchsatz, verdoppeln den
Speicherdruck und machen **jede Zeitmessung unbrauchbar**. Genau das ist
passiert — die `forward_ms`-Spalte von B-003 ist deshalb wertlos.

Gefahrlos parallel: Auswertungen auf gespeicherten Rohdaten
(`qire_ai.equivalence`, `qire_ai.audit`, `qire_ai.results_report`,
`qire_ai.figures`), Schreiben.

---

## 2. Worum es in B geht

Ursprünglich sollten quanteninspirierte Strukturen eine dichte Schicht bei
weniger Parametern ersetzen. Das ist **vollständig gescheitert** — und aus dem
Scheitern ist die eigentliche Arbeit geworden:

> Wovon hängt ab, wie breit eine Schicht sein muss — und warum scheitert
> deshalb jeder strukturierte Ersatz?

B ist damit eine **Mess- und Methodenarbeit**, keine Effizienzarbeit.

---

## 3. Was belegt ist

### B-014 — der tragende Befund

Synthetische Daten mit **zwei kontrollierten Variablen**: intrinsische
Dimension d (6 Stufen) mal Klassenzahl C (4 Stufen), 10 Seeds, plus eine
untrainierte Kontrolle.

> **Untrainiert** folgt der Effektivrang der Datendimension (rho=+0,987,
> p=6e-19 über 24 Gitterzellen) und ist von der Klassenzahl **unabhängig**
> (rho=+0,03, p=0,90). **Trainiert** hängt er von beiden ab (d: rho=+0,61,
> p=0,002; Klassen: rho=+0,67, p=0,0003) und liegt drastisch niedriger
> (1,4–4,7 statt 3,7–165).

Die Klassenabhängigkeit existiert vor dem Training **nicht**. Sie entsteht beim
Lernen. Das Muster sieht aus wie ein Minimum: bei 2 Klassen bleibt der Rang bei
etwa 1,5 unabhängig von d; bei d=2 bei etwa 1,7 unabhängig von C.

**Das erklärt alle Negativbefunde von B auf einmal:** Rang 8, 64 und 256
liefern dieselbe Genauigkeit, weil die Aufgabe nur rund sieben Richtungen
braucht. Der Rang war nie die bindende Beschränkung — deshalb konnte mehr
Budget nie helfen.

**Korrektur vom 27. August:** Die ursprünglich berichteten p-Werte (p=3e-187)
waren **Pseudoreplikation** — korreliert wurde über alle 240 Einzelmessungen,
obwohl das Gitter nur 24 Zellen hat und die zehn Seeds je Zelle dasselbe d und
dasselbe C tragen. **Es gelten die Zellmittel-Werte oben.** Die Befunde selbst
halten (rho verschiebt sich nur von 0,986 auf 0,987).

### B-011 — die einzige bewiesene Gleichwertigkeit

Eine **Rang-64-SVD-Projektion einer fertig trainierten dichten Schicht** ist
der dichten Schicht statistisch gleichwertig — TOST p<0,0001, beide Datensätze,
15 Seeds, bei 32-fach weniger Parametern. Das ist ein *Nachweis*, kein
„p>0,05".

Wichtig: **Kompressions-, kein Effizienzbefund.** Die Projektion setzt das
dichte Training voraus, kostet also mehr Gesamtrechenzeit, nicht weniger.

Dazu die Erklärung, warum B-008 das Gegenteil zu zeigen schien: Dort wurde das
Spektrum der **Gewichtsmatrix** gemessen (Rang 64 erfasst 65 %). Entscheidend
ist das Spektrum der **Ausgaben auf echten Daten** — dort sind es 99,8 %.

### B-008 und B-009 — der belegte Negativbefund

Keine strukturierte Familie (Low-Rank, Blockdiagonale, Monarch, Tensor-Train)
ersetzt unter fairen Bedingungen eine dichte Schicht. Alle liegen 0,005 bis
0,014 darunter, auf zwei Datensätzen, 30 Seeds, mit Lernrate je Arm. Mehr
Budget hilft nicht.

### B-007 — Befund 4 der Gesamtbilanz

„Unitarität gegen Spektralnorm, kein Unterschied (p=0,32)" ist **widerlegt**:
Auf MNIST schlägt Orthogonalität die Spektralnorm bei beiden Tiefen, je 17 von
20 Seeds, Holm p=0,0020 und p=0,0001, d=0,84 und 1,18. Auf Fashion-MNIST
repliziert es **nicht** (+0,0048 und −0,0045, beide nicht signifikant). Nach
dem vorregistrierten Kriterium: **HU, unentschieden.** Belastbar ist nur die
datensatzabhängige Aussage.

### Drei gemessene Messfehlerquellen

1. Eine Architektur, deren Positivkontrolle durchfällt — in B-004 trennt sich
   ein **32-fach größeres Parameterbudget nicht signifikant** (p=0,23). Der
   Aufbau, aus dem A-002 und A-003 stammen, konnte seine Frage nie beantworten.
2. Eine feste Lernrate kostete den dichten Arm rund **0,006** Genauigkeit.
3. Zu kleine Seed-Zahlen — **drei Befunde kippten an einem Tag** zwischen 10
   und 30 Seeds die Richtung.

---

## 4. Was zurückgezogen ist

Nicht überschreiben, nicht verschweigen — das ist Substanz der Arbeit:

| Zurückgezogen | Warum |
|---|---|
| „Blockschicht, 819-fach weniger Parameter, besser als dicht" | galt nur im blinden Aufbau; ohne dichte Enden dreht das Vorzeichen |
| „Bonds tragen nichts, es ist die Lokalität" | Monarch schlägt die Blockdiagonale auf MNIST |
| „Rang 64 ersetzt dicht" (von null trainiert) | bei 10 Seeds +0,0003, bei 30 Seeds −0,0066 |
| „Monarch ist gleichwertig zu dicht" | bei 10 Seeds −0,0008 (p=0,92), bei 30 Seeds −0,0045 (p=0,001) |
| „Der Rang kollabiert während des Trainings" | er **steigt** (5,1 auf 6,7); die Struktur steht schon nach Epoche 1 |
| „Rang-Curriculum als Trainingsverfahren" | H0 auf beiden Datensätzen; schlechter als ein einzelner Schnitt |

Ebenfalls korrigiert: Der Effektivrang liegt bei **5 bis 7**, nicht bei 64.
„Energie bei Rang 64 = 99,8 %" ist eine kumulierte Größe mit langem Schwanz;
die anschauliche Zahl ist die Teilnahmezahl.

---

## 5. Die Protokollregeln

1. **Lernrate je Arm** auf eigenen Tuning-Seeds — und das Gitter muss das
   Optimum **enthalten**; Randlagen melden.
2. **Genug Seeds.** Bei n Seeds ist 2/2^n der kleinstmögliche Wilcoxon-p-Wert
   (n=3 gibt 0,25; n=5 gibt 0,0625; n=8 gibt 0,0078). Vor dem Lauf prüfen, ob
   Signifikanz überhaupt erreichbar ist.
3. **„nicht signifikant" ist nicht „gleich"** — TOST (`equivalence.tost_paired`).
4. **Multiples Testen korrigieren** — Holm, nur über die angemeldete Familie.
5. **Rohdaten nach jedem Datensatz speichern.** B-007 hat an dieser Regel
   200 fertige Zellen verloren.
6. **Positivkontrolle einbauen** und prüfen, dass sie feuern *kann*.
7. **Kriterien vor dem Lauf** ins Skript schreiben.
8. **Wird ein Fehler gefunden, wird nach seiner Klasse gesucht** — nicht nur
   die Fundstelle repariert. So kam die Pseudoreplikation in B-014 ans Licht.
9. **Keine Pseudoreplikation.** Über Zellmittel korrelieren, nicht über
   Einzelmessungen. 17 weitere `stats.spearmanr`-Fundstellen sind noch zu
   prüfen, sobald ihre Zahlen zitiert werden sollen.

**Der Budget-Fehler vom 27. August (halbierte Lernratensuche) betrifft die
B-Module nicht** — dort läuft die Suche mit demselben `max_samples` und
`epochs` wie der Messlauf. Betroffen waren neun `quantum_sim`-Module.

---

## 6. Was offen ist

**Der eine noch nötige Lauf:** B-014 mit einem **Faltungsnetz** und über
**mehrere Schichten** wiederholen. Das ist die Frage, die eine Jury garantiert
stellt — gilt das auch für die Architektur, die man wirklich benutzt? Das
Design überträgt sich unverändert, nur das Modell wechselt.

**Ungeklärt, aber repliziert:** Die Blockdiagonale gewinnt bei **200
Trainingsbeispielen** auf beiden Datensätzen (+0,0149 und +0,0139, beide
signifikant). Die naheliegende Erklärung ist von denselben Daten widerlegt —
`lowrank_r8` hat die *wenigsten* Parameter und ist dort am schlechtesten. Es
liegt an der Blockstruktur, nicht an der Anzahl, und laut B-004 nicht an
räumlicher Bildstruktur. Kandidat: Blockdiagonal wirkt wie ein **Ensemble**
kleiner Netze — prüfbar gegen ein echtes Ensemble gleicher Größe.

**Fehlt noch komplett:** Kurzfassung für B und eine Demo.
`KURZFASSUNG_ENTWURF.md` führte zuletzt noch den zurückgezogenen
819-fach-Befund als Hauptergebnis.

**Abbruchregel, ernst gemeint:** Jedes Experiment öffnet zwei neue Fragen. Nach
dem CNN-Lauf sollte Schluss sein und geschrieben werden. Offene Fragen kommen
in eine Liste im Tagebuch — die ist selbst ein Qualitätsmerkmal.

---

## 7. Werkzeuge

| Modul | Zweck |
|---|---|
| `qire_ai/result_store.py` | Rohdaten speichern und laden (JSON + CSV) samt Umgebung |
| `qire_ai/equivalence.py` | TOST-Äquivalenztest, Holm-Korrektur |
| `qire_ai/audit.py` | rechnet Tagebuchzahlen gegen die Rohdaten nach |
| `qire_ai/results_report.py` | erzeugt `RESULTS.md` |
| `qire_ai/figures.py` | erzeugt die Abbildungen |
| `qire_ai/spectral_probe.py` | Effektivrang, Spektralentropie, klassenweise Unterräume, Training mit Haken je Epoche |

Schnellster Weg zum Belegstand:

```powershell
python -m qire_ai.audit
```

---

## 8. Zuordnung Experiment zu Modul

| ID | Modul | Frage |
|---|---|---|
| B-003 | `ablation_blockdiag.py` | Blockschicht gegen dicht, zwei Tiefen |
| B-004 / B-006 | `ablation_locality.py` | gleiches Budget, drei Architekturen; Konvergenz |
| B-005 | `weight_structure.py` | lernt die dichte Schicht Blockstruktur? |
| B-007 | `ablation_unitarity.py` | Orthogonalität gegen Spektralnorm |
| B-008 | `rank_prediction.py` | Rangreihe und Spektrum |
| B-009 | `ablation_locality.py` | Monarch gegen dicht, 30 Seeds |
| B-010 | `sample_efficiency.py` | Datenmenge gegen Struktur |
| B-011 | `optimization_gap.py` | Darstellungs- oder Optimierungslücke |
| B-012 | `trajectory_dose.py` | Dosis-Wirkungs-Kurve des Trainingswegs |
| B-013 | `rank_curriculum.py` | Rang-Curriculum als Verfahren |
| B-014 | `intrinsic_dimension.py` | **Datendimension gegen Klassenzahl** |
| B-015 | `dmrg_adaptive_rank.py` | DMRG-Rangzuweisung (umbenannt, war B-003) |

Sammelläufe: `run_project_b.py` (Stufen 0 bis 4), `run_project_b2.py`,
`run_project_b4.py`.
