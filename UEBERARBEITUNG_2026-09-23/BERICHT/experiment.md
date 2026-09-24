# Agent `experiment`: Q-020, die Kernaussage von Projekt A fair neu prüfen

**Stand:** 24. September 2026. Q-020 ist gebaut und vorregistriert, der Vortest ist bestanden. Rechenzeit ist gemessen, der Pilot ist gelaufen.
**Der Messlauf ist NICHT gestartet** — das entscheidet Malte.
Die CPU ist frei, es läuft kein Q-020-Prozess.

## Dateien

| Datei | Inhalt |
|---|---|
| `A/quantum_sim/kernaussage.py` | Modul Q-020: Arme, Zelle, Lernratensuche, Lauf mit Wiederaufnahme, Auswertung, Vortest, Zeitmessung, Pilot. Vorregistrierung im Modulkopf. |
| `A/ENTWURF_Q020.md` | Entwurf mit Vorregistrierung, Begründung für den offiziellen Testsatz, Vortest mit Mutationsprüfung, Rechenzeit, Pilot, offenen Entscheidungen |
| `A/results/experiments/Q-020-pilot__20260924_093937.json/.csv` | Pilot, Zwischenstand nach MNIST (30 Zeilen) |
| `A/results/experiments/Q-020-pilot__20260924_095434.json/.csv` | Pilot vollständig (57 Zeilen) mit Auswertung |
| `A/results/experiments/q020_zwischenstand/Q-020-pilot_wiederaufnahme.json` | Wiederaufnahmedatei des Pilots. Sie lag zuerst im Hauptordner und hat `q_audit` zum Absturz gebracht (Hinweis `werkzeuge`). Die Datei ist verschoben, der Code schreibt ab jetzt in den Unterordner. |
| `A/results/Q-020-pilot_run.log` | Konsolenausgabe des Pilots |

Befehle (aus `A`): `python -m quantum_sim.kernaussage vortest | zeit | kalibrierung | pilot | lauf | auswertung [ID]`.

## Aufbau in Kürze

- **(a) mit Encoder** wie Q-003, MNIST (primär) und Fashion (Wiederholung).
  - Trainiert wird auf 2.800 Bildern wie in Q-003, getestet auf dem **offiziellen Testsatz mit 10.000 Bildern**.
  - Begründung: Bei 1.200 Testbildern liegt allein der Stichprobenfehler der Testmenge bei ≈0,007, also über der TOST-Grenze von 0,005. Bei 10.000 Bildern sind es ≈0,0026. Außerdem ist das der Literaturstandard.
  - Die gecachten OpenML-Dateien haben die Originalreihenfolge. Geprüft habe ich das an den Klassenzahlen des Testteils.
- **(b) ohne Encoder** wie Q-018: Haufen-Aufgabe, 6 Klassen, 128 Haufen.
  - Grund für 128: Bei 32 Haufen lag die Schaltung laut Q-018-Rohdaten an der Decke.
- **Arme:**
  - Schaltungen: `q1`, `q1_ohne`, `q5`, `q5_ohne`
  - `kron`: klassische Kronecker-Drehung mit 8 Parametern
  - klassische Vergleiche: `quad_free` (2.048), `mlp` (67.848, gleiche Engstelle), `linear`
  - beschreibend: `ohne_engstelle`, `norm_deep` (nur (a))
- **Familie mit Holm über 3:**
  - PRIMÄR: q1 gegen mlp auf MNIST
  - P2: q5 gegen q5_ohne in (b) — Verschränkung
  - P3: q5 gegen q1 in (b) — Tiefe
- **P3 habe ich ergänzt.** Das Review nennt Verschränkung *und* Tiefe. Wird P3 gestrichen, sinkt Holm auf 2 Tests; das muss vor dem Lauf ins Tagebuch.
- **Rangfolge wie in `zeitreihen.py`:** zuerst die Ausschlüsse (Auflösung, Boden), dann TOST, dann Signifikanz. HU_DECKE gilt nur im Gleichwertigkeitszweig.
- **Kontrollen:**
  - Auflösung: mlp muss sich von linear trennen. Gefordert ist Trennung, nicht Richtung. Diese Prüfung trägt das Urteil.
  - Boden: bester Arm mindestens Rateniveau + 0,10. Trägt ebenfalls das Urteil.
  - Äquivalenzkontrolle kron ≈ q1_ohne: dient der Deutung, trägt kein Urteil. Grund: Unter der Nullhypothese gibt sie in 5 % der Läufe Fehlalarm.
- **Lernrate** je Arm und Abschnitt, mit vollem Budget, 3 Tuning-Seeds. Das Gitter wird in beide Richtungen verlängert.
- **Gespeichert wird** nach jeder Zelle (Wiederaufnahme, eigene Datei je Lauf-ID, Fingerabdruck der Konfiguration) und nach jedem Abschnitt eine vollständige Rohdatendatei.

## Theorie, vor dem Lauf nachgerechnet (deterministisch, kein Training)

**`q1` und `q1_ohne` haben 16 Winkel, aber nur 8 davon wirken.**
- In einer Lage stehen die Rz-Winkel direkt vor der Messung oder vor einer reinen Basis-Vertauschung. Das CNOT ist eine Permutation.
- Gemessen:
  - `q1_ohne` ≡ `kron` (2,7·10⁻⁷)
  - `q1` ≡ `kron` + vertauschte Auslese (2,4·10⁻⁷)
  - neu gewürfelte Rz-Winkel ändern nichts (≤3,3·10⁻⁷)
  - Gegenprobe mit 5 Lagen und CNOT: Die Rz-Winkel wirken (0,33). Die Prüfung ist also nicht blind.

**Folgerung:** Der 16-Winkel-Arm aus Q-003 ist eine klassische 8-Winkel-Kronecker-Drehung + Quadrieren + Paritätsauslese. Das ist die ehrliche Form der Kernaussage: Die Architektur stammt aus der Quantenschaltung, der Mechanismus ist klassisch.

## Vortest-Ergebnis

`vortest` ist **bestanden**, und zwar in allen drei Teilen:
1. **Theorieprüfung:** Die Butterfly-Umsetzung entspricht dem Zustandsvektor-Simulator (n=8, L=1 und L=5, mit und ohne CNOT, ≤3·10⁻⁷). Dazu kommt die Tabelle oben.
2. **Gitterverlängerung in beide Richtungen:** 5 Fälle, alle OK, einschließlich der Randvermerke „oben“ und „unten“.
3. **Alle Ausgänge einzeln:**
   - 21 Szenarien plus 2 Kontrollstatus, alle OK.
   - Je Szenario 5 Ziehungen, geprüft auf Marke **und** Zweigtext.
   - Nullfälle über 40 Ziehungen mit gefordertem Anteil ≥ 80 %. Sie trafen 38–40 von 40.

**Kann der Vortest durchfallen?** Ja. Ich habe drei absichtliche Fehler eingebaut, alle drei wurden erkannt:

| Eingebauter Fehler | Ergebnis |
|---|---|
| falsches Feld (`accuracy`) | 9 Szenarien schlagen fehl |
| Signifikanz vor TOST | Szenario „signifikant UND gleichwertig“ schlägt fehl |
| Auflösungsprüfung verlangt Richtung statt Trennung | Szenario P-Ar schlägt fehl |

## Pilot: KEIN BEFUND

**Reduziert:**
- nur MNIST und 128 Haufen
- 3 eigene Seeds (20291–20293, nicht die Mess-Seeds)
- 1 Tuning-Seed, 2 Gitterpunkte, keine Verlängerung

**Folgen der Reduktion:**
- Alle Lernraten liegen am Rand.
- Bei n=3 ist das kleinste mögliche p = 0,25. Deshalb meldet der Pilot erwartungsgemäß dreimal HU_AUFLOESUNG.

| Pilot | q1 | q1_ohne | q5 | q5_ohne | kron | quad_free | mlp | linear | ohne_eng. | norm_deep |
|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | 0,9330 | 0,9340 | 0,9336 | 0,9323 | 0,9341 | 0,9134 | 0,9159 | 0,8770 | 0,9360 | 0,8436 |
| 128 Haufen | 0,3916 | 0,3490 | 0,6605 | 0,3386 | 0,3495 | 1,0000 | 0,9924 | 0,8220 | 0,9986 | – |

**Größenordnungen, nur für den Entwurf:**
- **q1 − mlp auf MNIST: +0,017, sd ≈ 0,008.** TOST ±0,005 ist damit erreichbar, die Schwelle liegt bei sd < 0,011. Aber das MLP sitzt am unteren lr-Rand, dafür ist die Verlängerung nach unten da.
- **Äquivalenzkontrolle:** kron − q1_ohne beträgt +0,0001 (MNIST) bzw. +0,0005 (Haufen). Im Haufen-Aufbau besteht TOST schon bei n=3.
- **(b):**
  - q5 − q5_ohne +0,32, q5 − q1 +0,27.
  - Alle Kronecker-artigen Arme liegen bei 0,34–0,35, wie vorhergesagt.
  - Die klassischen Arme sitzen in (b) an der Decke. Die Schaltung ist dort klar unterlegen, das gehört so berichtet.
- **norm_deep** liegt wieder unter linear (0,844 gegen 0,877). Der Strohmann bestätigt sich in der Tendenz.

## Rechenzeit für den vollen Lauf

**Zeitmessung** (je Arm eine Zelle, volles Budget):
- 118 s je Seed über alle Arme auf MNIST, 312 s auf 128 Haufen.
- Fashion ist wie MNIST angesetzt.
- Hochgerechnet mit 3 Tuning-Seeds × ~5 Gitterpunkten + 15 Mess-Seeds: **4,6 h**. Spanne 4,1–5,5 h, je nach Zahl der Gitterverlängerungen.

**Gegenprobe aus dem Pilot:**
- 66 s bzw. 180 s je Seed.
- Die Einzelmessung war um den Faktor ~1,7 zu pessimistisch: Anlauf, zeitweise fremde Last, ungetunte lr.
- Mit den Pilotzeiten ergeben sich ≈ 2,6 h.

**Erwartung: 3–5 h**, ein Lauf, Wiederaufnahme vorhanden. Den größten Anteil haben `q5` und `q5_ohne` in (b).

**Mein eigener Rechenverbrauch:** rund 40 min CPU (Zeitmessung ~8 min, Pilot ~30 min, Vortests wenige Minuten). Damit liegt er im Rahmen von 60 min.

## Vor dem Start zu entscheiden (Malte)

1. Soll P3 in der Familie bleiben? Mein Vorschlag: ja.
2. Den Tagebucheintrag unten **vor** dem Start eintragen.
3. Den Lauf starten mit `python -m quantum_sim.kernaussage lauf`. Das ist ein schwerer Lauf, während er läuft darf kein anderer laufen.
4. Nach dem Lauf: Zahlen in `q_audit.py` und in `q_summary.QUESTIONS` eintragen (CLAUDE.md §6). Das habe ich nicht angefasst, weil es noch keine Messzahlen gibt.

## Textvorschlag Tagebucheintrag

> ## 24. September — Vorregistrierung Q-020: die Kernaussage fair neu
>
> Geschrieben **vor** dem Lauf. Es existiert nur ein Pilot (3 Seeds, reduzierte
> Suche, eigene Seeds), und der ist kein Befund.
>
> **Anlass.** Die Kontrolle aus Q-003 (`classical_norm_deep`, 67.848 Parameter)
> ist schwächer als ein lineares Modell. Stichprobe 23.09.: 0,854 gegen 0,887.
> Der Abstand +0,093 misst also vor allem ihre Schwäche. Und die Nullbefunde
> zu Verschränkung und Tiefe stammen aus einem Aufbau mit trainierbarem
> Encoder. Genau diese Konstellation hat Q-018 als Artefakt entlarvt.
>
> **Aufbau.** Es gibt zwei Aufbauten:
> - (a) wie Q-003, mit Encoder, auf dem **offiziellen MNIST-Testsatz**
>   (10.000 statt 1.200 Bilder; allein der Testmengenfehler von 1.200 Bildern,
>   ≈0,007, lag über der Grenze 0,005);
> - (b) wie Q-018 ohne Encoder, 6 Klassen, 128 Haufen.
>
> Die Arme: Schaltung 1 und 5 Lagen, jeweils mit und ohne CNOT; eine klassische
> Kronecker-Drehung (8 Parameter); quad_free (2.048); ein ReLU-MLP mit
> derselben 8-Kanal-Engstelle (67.848); linear; ohne Engstelle; norm_deep.
>
> **Vorher nachgerechnet:** Die 1-Lagen-Schaltung hat 16 Winkel, aber nur 8
> wirken. Die Rz-Winkel stehen vor der Messung bzw. vor einer reinen
> Vertauschung. `q1_ohne` rechnet exakt dieselbe Funktion wie die klassische
> Kronecker-Drehung, `q1` dieselbe mit vertauschter Auslese (Abweichung
> <4·10⁻⁷). Die Gegenprobe mit 5 Lagen zeigt, dass die Prüfung nicht blind
> ist.
>
> **Familie, Holm über 3:**
> - PRIMÄR: q1 gegen mlp auf MNIST
>   (GLEICHWERTIG / SCHALTUNG_BESSER / MLP_BESSER / UNENTSCHIEDEN)
> - P2: Verschränkung in (b), q5 gegen q5_ohne
> - P3: Tiefe in (b), q5 gegen q1
>
> Balancierte Genauigkeit, 15 Seeds, TOST ±0,005. Die Rangfolge ist
> TOST vor Signifikanz. Auflösungsprüfung: mlp muss sich von linear trennen.
> Bodenprüfung: bester Arm mindestens Rateniveau + 0,10. Die
> Äquivalenz-Positivkontrolle kron ≈ q1_ohne dient der Deutung. Lernrate je
> Arm mit vollem Budget, das Gitter wird in beide Richtungen verlängert.
> Gespeichert wird nach jeder Zelle, die Wiederaufnahme hat eine eigene Datei
> je Lauf.
>
> **Erwartung vorher:**
> - PRIMÄR SCHALTUNG_BESSER oder UNENTSCHIEDEN
> - P2 und P3 „trägt bei“: ohne CNOT ist jede Tiefe wieder eine
>   Kronecker-Drehung.
>
> **Vortest** bestanden: 21 Szenarien, Zweigtext geprüft. Drei absichtlich
> eingebaute Fehler hat er alle gefunden.
>
> **Rechenzeit** geschätzt 3–5 h.
>
> Ändert sich etwas an dieser Vorregistrierung, steht es hier, bevor eine
> Q-020-Zahl existiert.
