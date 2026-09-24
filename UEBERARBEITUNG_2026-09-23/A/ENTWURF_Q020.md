# Q-020 — Die Kernaussage von Projekt A, fair neu gemessen

**Stand:** 24. September 2026, **vor** jeder Q-020-Messzahl. Es existiert nur ein
Pilot (`Q-020-pilot`, 3 Seeds, reduzierte Suche) — er ist **kein Befund** und
fließt in kein Urteil ein.
**Code:** `quantum_sim/kernaussage.py` — die Vorregistrierung steht wortgleich im
Modulkopf; bei Widerspruch gilt der Code, und der Widerspruch gehört ins Tagebuch.

---

## 1. Warum

Aus dem Review vom 23. September (`BERICHT/00_REVIEW_BEFUNDE.md` §1–2):

1. **Die Kontrolle aus Q-003 ist ein Strohmann.** `classical_norm_deep`
   (67.848 Parameter) lag in einer Stichprobe (3 Seeds) **unter einem linearen
   Modell** (0,8544 gegen 0,8873 MNIST). Normierung auf Einheitslänge plus
   Tanh-Engstelle lähmen sie. Der berichtete Abstand +0,0932 misst vor allem
   die Schwäche der Kontrolle. Gegen ein gewöhnliches ReLU-MLP mit derselben
   8-Kanal-Engstelle blieb in der Stichprobe rund +0,01.
2. **„Verschränkung und Tiefe tragen nichts bei“ stammt aus einem Aufbau mit
   trainierbarem Encoder.** Q-018 hat genau diese Konstellation als Artefakt
   entlarvt. Außerdem ist die Positivkontrolle jenes Laufs (5 gegen 1 Lage)
   durchgefallen — laut `q_summary` steht damit keine Gleichwertigkeitsaussage
   aus Q-003-replication.

## 2. Zwei Aufbauten

| | (a) mit Encoder (wie Q-003) | (b) ohne Encoder (wie Q-018) |
|---|---|---|
| Eingabe | MNIST / Fashion-MNIST, 784 → Encoder 784→256 (trainierbar) | Haufen-Aufgabe `frozen_angles.make_data`, 256 = 2⁸ direkt |
| Aufgabe | 10 Klassen | 6 Klassen, **128 Haufen** |
| Training | 2.800 je Seed aus dem offiziellen Trainingsteil, 12 Epochen, Batch 128 | 6.000, 12 Epochen, Batch 64 |
| Test | **offizieller Testsatz, 10.000** | 3.000 |
| Rolle | PRIMÄR (MNIST), Wiederholung (Fashion) | P2, P3 |

Stufe (b): In Q-018 stand die 5-Lagen-Schaltung bei 32 Haufen schon bei 0,9588
(8 von 15 Seeds exakt 1,0 — Decke), bei 128 Haufen bei rund 0,65. Deshalb
128. `kalibrierung` prüft Boden, Decke und Auflösung vor dem Lauf.

## 3. Die Arme

Engstelle überall **8 Kanäle**, Kopf 8 → Klassen. Parameter = Mittelschicht.

| Arm | Was | Parameter |
|---|---|---|
| `q1` | Schaltung, 1 Lage, CNOT-Ring | 16 Winkel, **8 wirksam** |
| `q1_ohne` | Schaltung, 1 Lage, ohne CNOT | 16 Winkel, **8 wirksam** |
| `q5` | Schaltung, 5 Lagen, CNOT-Ring | 80 |
| `q5_ohne` | Schaltung, 5 Lagen, ohne CNOT | 80 |
| `kron` | **klassisch**: R₁⊗…⊗R₈ (reelle 2×2-Drehungen) + Quadrieren + Z-Auslese | 8 |
| `quad_free` | normieren + blockweise freie 8×8-Matrix + Quadrieren (Q-003) | 2.048 |
| `mlp` | ReLU-MLP 256→256→8, gleiche Engstelle | 67.848 |
| `linear` | 256→8 (Untergrenze) | 2.056 |
| `ohne_engstelle` | keine 8-Kanal-Engstelle — beschreibend, Obergrenze | (a) 0 / (b) 65.792 |
| `norm_deep` | die Q-003-Kontrolle — nur (a), beschreibend | 67.848 |

Die Schaltungen laufen über `ButterflyQuantumLayer` statt über den
Zustandsvektor-Simulator. `pruefe_theorie()` zeigt für n=8, L=1 und L=5, mit und
ohne CNOT dieselbe Funktion (max. Abweichung 3·10⁻⁷).

### Nachgerechnet, bevor es ein Urteil trägt: „16 Winkel, 8 wirksam“

In **einer** Lage stehen die Rz-Winkel direkt vor der Messung (ohne CNOT) bzw.
vor einer reinen Basis-Vertauschung (CNOT ist eine Permutation). Beides ändert
die Betragsquadrate nicht. `pruefe_theorie()` misst:

| Prüfung | max. Abweichung |
|---|---|
| 1 Lage ohne CNOT = `kron` bei gleichen Winkeln | 2,7·10⁻⁷ |
| 1 Lage mit CNOT = `kron` + vertauschte (Paritäts-)Auslese | 2,4·10⁻⁷ |
| Rz-Winkel neu gewürfelt, 1 Lage ohne / mit CNOT | 3,3·10⁻⁷ / 3,0·10⁻⁷ |
| Gegenprobe: 5 Lagen mit CNOT, Rz neu gewürfelt | **0,33** (wirkt — Prüfung ist nicht blind) |

Folgerung, schon jetzt und ohne Training: **Der 16-Winkel-Arm aus Q-003
(`hybrid_1layer`) ist eine klassische Kronecker-Drehung mit 8 Winkeln,
gefolgt von Quadrieren und einer festen Paritätsauslese.** Wenn `q1` das MLP
erreicht, dann erreicht es eine *klassisch vollständig beschriebene*
8-Parameter-Abbildung. Das ist keine Schwächung, sondern die ehrliche Form
der Aussage: die Architektur stammt aus der Quantenschaltung, der Mechanismus
ist klassisch.

`kron` ist die **Äquivalenz-Positivkontrolle**: dieselbe Funktionsfamilie wie
`q1_ohne` (bei reeller Eingabe fehlt die y-Komponente jeder reduzierten
Dichtematrix; eine komplexe U(2) sieht die Z-Auslese nicht), dieselbe
Winkelparametrisierung, aber eigene Initialisierung und eigene Lernrate. Sie
muss gleichwertig abschneiden.

## 4. Offizieller Testsatz statt 30-%-Split — Begründung

- **Messgenauigkeit.** Q-003 testete auf 1.200 Bildern. Der
  Stichprobenfehler allein der Testmenge liegt bei Genauigkeit 0,93 bei
  √(0,93·0,07/1200) ≈ **0,0074** — größer als die Gleichwertigkeitsgrenze
  ±0,005. Bei 10.000 Bildern sind es ≈ **0,0026**. TOST ±0,005 wird damit
  überhaupt erst realistisch erreichbar.
- **Vergleichbarkeit.** Jede Literaturzahl zu MNIST steht gegen diesen
  Testsatz. Andere Schreiber als im Trainingsteil → echte Generalisierung.
- **Fester Test, variables Training.** Alle Seeds testen auf denselben
  10.000 Bildern; ein Seed variiert Trainingsauswahl, Initialisierung und
  Reihenfolge. Die Streuung misst also Trainingsvariabilität, nicht, welche
  Bilder zufällig im Test landeten.
- **Trainingsmenge unverändert:** 2.800 wie in Q-003, damit nur **eine**
  Sache wechselt. Kosten: Auswertung auf 10.000 statt 1.200 Bildern, pro Zelle
  Bruchteile einer Sekunde.
- Nachteil, offen benannt: Q-020-Zahlen sind **nicht direkt** mit Q-003
  vergleichbar (andere Testmenge). Verglichen wird nur innerhalb von Q-020.

## 5. Vorregistrierung

**Messgröße:** balancierte Genauigkeit. **15 Seeds** (20201–20215),
Tuning-Seeds 20251–20253. **Lernrate je Arm und Abschnitt** mit dem **vollen
Budget** des Messlaufs; Startgitter (a) 0,003/0,01/0,03/0,1, (b)
0,01/0,03/0,1/0,3, Verlängerung **in beide Richtungen** mit Faktor 3, bis zu
3-mal (Regel 1 a–c). „innen“ gilt erst, wenn hinter dem Optimum ein
schlechterer Punkt gemessen ist.

| Test | Aufbau / Abschnitt | Paar | Ausgänge |
|---|---|---|---|
| **PRIMÄR** | (a) MNIST | `q1` gegen `mlp` | GLEICHWERTIG / SCHALTUNG_BESSER / MLP_BESSER / UNENTSCHIEDEN |
| **P2** | (b) 128 Haufen | `q5` gegen `q5_ohne` (Verschränkung) | VERSCHRAENKUNG_TRAEGT / GLEICHWERTIG / VERSCHRAENKUNG_SCHADET / UNENTSCHIEDEN |
| **P3** | (b) 128 Haufen | `q5` gegen `q1` (Tiefe, mit CNOT) | TIEFE_TRAEGT / GLEICHWERTIG / TIEFE_SCHADET / UNENTSCHIEDEN |

Wilcoxon zweiseitig + TOST ±0,005. **Holm über genau diese drei.** P3 ist
gegenüber dem Auftrag (PRIMÄR + P2) von mir ergänzt, weil das Review
Verschränkung *und* Tiefe nennt; wird P3 gestrichen, sinkt die Familie auf 2 —
Entscheidung vor dem Lauf, mit Tagebucheintrag.

**Rangfolge der Zweige** (wie `zeitreihen.py`): HU_UNVOLLSTAENDIG →
HU_AUFLOESUNG → HU_BODEN → *TOST besteht* → (HU_DECKE, wenn beide Arme > 0,95,
sonst GLEICHWERTIG) → Holm < 0,05 → Richtung → sonst UNENTSCHIEDEN. Signifikant
und zugleich nachweislich kleiner als ±0,005 heißt **GLEICHWERTIG** mit
Anmerkung.

**Kontrollen**

- **Auflösungsprüfung** (urteilstragend): `mlp` muss sich von `linear`
  **trennen** (p < 0,05), in (a) auf MNIST, in (b) auf 128 Haufen. Trennung,
  nicht Richtung (Lehre Q-015). Stichprobe 23.09.: +0,037 — sie kann feuern.
- **Bodenprüfung** (urteilstragend): bester Arm ≥ Rateniveau + 0,10.
- **Äquivalenz-Positivkontrolle** (deutend): `kron` gegen `q1_ohne`.
  *bestanden* (TOST) / *nicht erreicht* (dann ist ein UNENTSCHIEDEN der
  Familie Messgrenze, kein Hinweis auf Unterschied) / *VERLETZT* (signifikant:
  Warnung, Implementierung prüfen; unter der Nullhypothese 5 % Fehlalarm —
  deshalb nicht urteilstragend).
- **Wiederholung:** Fashion-MNIST, `q1` gegen `mlp`, beschreibend, ohne Holm.
- **Erreichbarkeit** (Regel 2): kleinstes Wilcoxon-p bei 15 Seeds
  2/2¹⁵ = 6,1·10⁻⁵, nach Holm ×3 = 1,8·10⁻⁴ — Signifikanz erreichbar. TOST
  ±0,005 besteht nur, wenn die Streuung der gepaarten Differenz unter
  0,005·√15/1,761 = **0,011** liegt. Liegt sie darüber, kann GLEICHWERTIG
  nicht feuern; das steht hier, bevor es als „nicht eingetreten“ gelesen
  werden kann.

**Was vorher erwartet wird**

- PRIMÄR: eher SCHALTUNG_BESSER oder UNENTSCHIEDEN (Stichprobe +0,01, aber
  andere Testmenge).
- P2: VERSCHRAENKUNG_TRAEGT. Ohne CNOT ist jede Tiefe wieder eine
  Kronecker-Drehung (Gruppe abgeschlossen); die 8 Kanäle sehen dann nur 8 feste
  quadratische Formen.
- P3: TIEFE_TRAEGT, aus demselben Grund.
- Beschreibend: `q5_ohne` gleichwertig `q1_ohne` (Theorie); `mlp` deutlich über
  `norm_deep` (Strohmann bestätigt).

## 6. Vortest (CLAUDE.md §5) — bestanden am 24. September

`python -m quantum_sim.kernaussage vortest`, drei Teile:

1. **Theorieprüfung** (Tabelle §3): bestanden.
2. **Gitterverlängerung** in beide Richtungen, fünf Fälle, darunter „außerhalb
   oben/unten“ mit Randvermerk: bestanden.
3. **Jeder Ausgang einzeln erzwungen**, auf synthetischen Daten, je Szenario
   5 Ziehungen (Nullfälle 40 Ziehungen, Anteil ≥ 80 %), geprüft auf Marke
   **und Zweigtext**. 21 Szenarien plus 2 Kontrollstatus, alle OK. Die
   Nullfälle trafen 38–40 von 40 (Fehlurteile im Rahmen des Testniveaus).
   Das Feld `accuracy` trägt im Generator die Versätze mit **umgekehrtem**
   Vorzeichen; eigenes Rauschen je Arm plus gemeinsamer Seed-Anteil (Lehre
   Q-019).

**Kann der Vortest durchfallen?** Drei absichtliche Fehler eingebaut:

| Mutation | Ergebnis |
|---|---|
| Auswertung auf `accuracy` statt `balanced_accuracy` | durchgefallen, 9 Szenarien |
| Signifikanz vor TOST gereiht | durchgefallen (u. a. „signifikant UND gleichwertig“) |
| Auflösungsprüfung verlangt Richtung statt Trennung | durchgefallen (Szenario P-Ar) |

## 7. Wiederaufnahme und Rohdaten

- Nach **jeder** Zelle (auch jeder Tuning-Zelle) atomar nach
  `results/experiments/<ID>_wiederaufnahme.json` (ein Unterstrich — `latest()`
  sucht `<ID>__*` und findet sie nie; das ist der T-002-Fehler, hier
  ausgeschlossen). Angenommen wird die Datei nur bei passendem
  Konfigurations-Fingerabdruck.
- Nach **jedem Abschnitt** eine vollständige Rohdatendatei
  `<ID>__<Zeit>.json/.csv` (Regel 5). Reihenfolge: MNIST → 128 Haufen →
  Fashion, also die Familie zuerst.
- Pilot: ID `Q-020-pilot`, eigene Seeds (20291–20293, Tuning 20299), eigene
  Wiederaufnahmedatei.

## 8. Rechenzeit

*(wird aus der Zeitmessung eingetragen)*

## 9. Pilot — KEIN BEFUND

*(wird eingetragen)*

## 10. Vor dem Lauf zu entscheiden (Malte)

1. P3 in der Familie lassen (Holm über 3) oder streichen (Holm über 2)?
2. Lauf starten — Dauer siehe §8, ein schwerer Lauf zur Zeit, nicht über Nacht
   ohne Wiederaufnahme (die ist jetzt eingebaut).
3. Tagebucheintrag „Vorregistrierung Q-020“ **vor** dem Start (Text in
   `BERICHT/experiment.md`).
4. Nach dem Lauf: Zahlen in `q_audit.py` und `q_summary.QUESTIONS` (CLAUDE.md §6).
