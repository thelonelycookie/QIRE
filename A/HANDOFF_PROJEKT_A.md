# Übergabe: Projekt A (Quanten) — Stand 6. August 2026, morgens

Diese Datei ist so geschrieben, dass man sie **kalt lesen** kann. Sie ersetzt
die Fassung vom 4. August vollständig.

**Es läuft gerade nichts.** Alle Läufe und Wächter wurden am 6. August gegen
07:00 gestoppt. Der Rechner ist frei.

---

## 0. In drei Sätzen

Die Kernaussage von Projekt A ist seit dem 5. August **mit Rohdaten belegt**:
eine Quantenschaltung mit 16–80 Parametern schlägt ein klassisches Netz mit
67.848. In der Nacht zum 6. August kam heraus, **warum** — und der Grund ist
nicht Verschränkung und nicht Tiefe, sondern die Struktur der Auslese und die
Konditionierung der Parametrisierung. Sechs fertig gebaute Läufe warten auf
einen Start.

---

## 1. Rechenzeit teilen — die Regel gilt weiter

6 physische Kerne, 16 GB RAM. **Immer nur EIN schwerer Lauf gleichzeitig.**

```powershell
Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Select-Object ProcessId, CommandLine
```

Zwei parallele Läufe bringen wegen SMT nur ~10 % mehr Durchsatz, verdoppeln den
Speicherdruck und machen **jede Zeitmessung unbrauchbar**. Ich habe in dieser
Nacht selbst dagegen verstoßen (siehe Abschnitt 6) — nicht wiederholen.

Speicher: Eine dichte 14.400×14.400-Schicht belegt mit Adam **4,9 GB**. Adam
hält zwei Zusatzkopien aller Parameter, deshalb das Dreifache der Gewichte plus
Gradienten. Ab etwa 8.000 Dimensionen ist eine dichte Schicht auf diesem Rechner
nicht mehr *trainierbar*, ab 32.768 nicht einmal mehr *baubar*.

---

## 2. Was belegt ist

**Kernaussage (Q-003-replication, 180 Zeilen Rohdaten)**
Die Schaltung schlägt `classical_norm_deep` um **+0,0932 (MNIST)** und
**+0,0599 (Fashion-MNIST)**, Holm p=0,0001, 15 Seeds, Lernrate je Arm,
balancierte Genauigkeit. 80 gegen 67.848 Parameter in der mittleren Schicht.

**Verschränkung trägt nichts bei.** Ohne CNOT gleichwertig (−0,0015, TOST bei
±0,005). Positiver Gleichwertigkeitsnachweis, nicht bloß p>0,05.

**Tiefe trägt nichts bei.** Eine Lage mit 16 Winkeln ist gleichwertig zu fünf
mit 80 (+0,0002, TOST).

**Q-015: die Schaltung schlägt die volle globale Drehung** um **+0,0674**,
15/15 Seeds, Holm p=0,0001 — bei **408-fach weniger Freiheitsgraden** (80 gegen
32.640). Beide haben dieselbe Funktionsklasse und sind beide universell, der
Unterschied liegt also an der **Trainierbarkeit**, nicht an der Ausdruckskraft.

**Butterfly-Architektur auf klassischer Hardware.** Verifiziert identisch zum
Simulator (7,15·10⁻⁷ über 30 Konfigurationen). Ab 16.384 Dimensionen **1,99×
schneller** als eine dichte Schicht (Median aus 7 Messreihen, 7/7 schneller).
Im Training nur **1,18×** — es ist ein Inferenzvorteil.

**Konzentrationsprinzip hergeleitet.** E[|⟨ψ|φ⟩|²] = 1/2ⁿ aus Geometrie; die
Messung trifft den Exponenten auf 4,2 % (−0,958 gegen −1,000), R²=0,9972.

**Die Auslese ist ein Rayleigh-Quotient.** out_q = xᵀA_qx/xᵀx mit A_q einer
echten Projektion (Eigenwerte {0,1}). Skaleninvariant, Ausgaben summieren sich
zu 1. Damit erfüllt die Schaltung den eigenen Prüfmaßstab.

**Universalität, zweifach geprüft.** Geteilt, per-Qubit und ungeteilt erreichen
alle Universalität bei **exakt dim SU(d) Parametern** (64/63 bei 3 Qubits,
256/255 bei 4). Die Teilung kostet **Tiefe, nicht Parameter** — sie spart
Speicher, nicht Rechenzeit.

Alle Zahlen prüfbar mit `python -m quantum_sim.q_audit`.

---

## 3. Was NICHT belegt ist

**Q-009 ist zu wiederholen.** Der Lauf lief mit **drei Seeds**, und dabei ist
p=0,2500 der kleinstmögliche Wilcoxon-Wert — der Primärtest liegt exakt auf
diesem Boden. Der gemeldete Spearman-Trend p=0,0000 ist ein asymptotisches
Artefakt; der exakte Permutationstest über 3! = 6 Anordnungen gibt **p=0,3333**.

Die Effekte sind trotzdem auffällig: der Vorsprung wächst mit der
Aufgabenschwierigkeit (+0,0061 → +0,0297 → **+0,1588**), bei 1024 Gebieten sind
das 12,3 gepoolte Standardabweichungen. **Neu aufsetzen mit ≥10 Seeds und einer
schwereren leichtesten Stufe** — die 64 Haufen waren gesättigt.

**Die 848× gelten für die mittlere Schicht**, nicht das Gesamtmodell (201.130
gegen 268.898). Q-012 greift das an.

**Nichts auf Sequenzdaten.** Q-013 wäre der erste Lauf dazu.

---

## 4. Sechs fertige Läufe, nicht gestartet

Alle kompiliert, vorregistriert, mit Auflösungsprüfung und Zwischenspeicherung.

| Lauf | Modul | Frage | ca. |
|---|---|---|---|
| Q-010 | `circuit_vs_replica` | Schlägt die Schaltung ihre faire Nachbildung? 30 Seeds, 3 Datensätze | 1,5 h |
| Q-011 | `unitarity_depth` | Wächst der Vorteil der Orthogonalität mit der Tiefe? | 5 h |
| Q-012 | `butterfly_encoder` | Encoder ersetzbar? 270 statt 201.130 Parameter | 1,5 h |
| Q-014 | `weight_tying` | Kostet die Gewichtsteilung Qualität? | 2,5 h |
| Q-016 | `learning_curves` | Datenbedarf und Überanpassung | 3 h |
| Q-017 | `reuploading` | Rettet Re-Uploading die hardwaretaugliche Kodierung? | 2 h |

Dazu ungestartet aus früheren Sitzungen: `barren_plateaus` (Q-005, billig),
`qubo_pruning` (Q-006), `breadth_battery` (Q-007).

Start einzeln, nacheinander:

```powershell
python -m quantum_sim.circuit_vs_replica
```

**Empfohlene Reihenfolge:** Q-014 zuerst (beantwortet die Alleinstellungsfrage
und hat eine abgeleitete Vorhersage), dann Q-011, dann Q-012.

**Für Q-014 steht eine Vorhersage im Tagebuch:** Da alle Familien bei gleicher
Parameterzahl gleich gut approximieren und die Aufgabe nur ~7 Richtungen
braucht, müssten `tied_L5` und `untied_L5` **gleichwertig** herauskommen.
Trifft das nicht zu, stimmt einer der beiden Vorbefunde nicht.

---

## 5. Die sieben Protokollregeln

1. **Lernrate je Arm** auf eigenen Tuning-Seeds — und das Gitter muss das
   Optimum **enthalten**. Randlagen melden (alle neuen Skripte tun das).
2. **Genug Seeds.** Bei n Seeds ist 2/2ⁿ der kleinstmögliche Wilcoxon-p-Wert:
   n=3 → 0,25, n=5 → 0,0625, n=8 → 0,0078. **Vor dem Lauf prüfen, ob
   Signifikanz überhaupt erreichbar ist.** Daran ist Q-009 gescheitert.
3. **„nicht signifikant" ist nicht „gleich"** — TOST benutzen.
4. **Multiples Testen korrigieren** — Holm, nur über die angemeldete Familie.
5. **Rohdaten nach jedem Datensatz speichern**, nicht am Ende.
6. **Positivkontrolle einbauen** — und prüfen, dass sie feuern *kann*.
7. **Kriterien vor dem Lauf** ins Skript schreiben.

**Neu dazugelernt am 6. August:** Der Auswertungscode darf nicht strenger sein
als die Vorregistrierung, und er braucht einen Zweig für **jeden** möglichen
Ausgang — auch den unerwarteten. Beide Fehler haben in Q-015 ein Ergebnis mit
p=0,0001 als „nicht verwertbar" abgestempelt.

---

## 6. Bekannte Mängel

- **Q-009s `train_seconds` ist kontaminiert.** Ich habe ab etwa 04:30
  Rechenjobs daneben laufen lassen. Genauigkeiten sind nicht betroffen (alle
  Arme laufen sequentiell), die Zeitspalte ist nicht zitierbar.
- **`B-003` ist doppelt vergeben** — DMRG-Rangzuweisung (2. Aug) und
  Blockschicht-Vergleich (3. Aug). Die Rohdaten gehören dem Blockschicht-Lauf;
  der DMRG-Lauf braucht eine neue Kennung. Vor der schriftlichen Arbeit klären.
- **Literaturkarten** sind pauschal als ungelesen markiert, tatsächlich sind
  ~drei Viertel gelesen. Braucht deine Angabe.
- **K-Serie hat keine Rohdaten.** Die vier Konzentrationswerte stammen aus dem
  Tagebuch. Ein Wiederholungslauf würde sie belegen *und* die Vorhersagen für
  5/8/10 Qubits testen (0,0448 / 0,0061 / 0,0016).

---

## 7. Werkzeuge

| Modul | Zweck |
|---|---|
| `quantum_sim/q_summary.py` | Sammelbericht aller Q-Läufe: gilt / mit Vorbehalt / nicht verwertbar / offen |
| `quantum_sim/q_audit.py` | rechnet jede Tagebuchzahl der Q-Serie aus **einer** Datei nach |
| `quantum_sim/theory.py` | Herleitungen gegen Messungen: Konzentration, Parametergeometrie |
| `qire_ai/audit.py` | dasselbe für die B-Serie |
| `qire_ai/equivalence.py` | TOST, Holm |
| `qire_ai/result_store.py` | Rohdaten speichern und laden |

**Wenn ein neuer Lauf entsteht, gehört er sofort in `q_summary.QUESTIONS` und
seine Zahlen in `q_audit`** — sonst taucht ein fertiger Lauf im Bericht gar
nicht auf. Genau das ist mit Q-015 passiert.

---

## 8. Dokumente

1. **`PROJEKTTAGEBUCH.md`** — maßgeblich. Jede Zahl mit Herleitung.
2. **`BEFUNDE_UND_RICHTUNGEN.md`** — ~60 Stichworteinträge: Befunde, Theorie,
   Läufe, neue Richtungen, Fehlerquellen. Der schnellste Überblick.
3. **`FORSCHUNGSIDEEN.md`** — zweite Blicke auf verworfene Befunde.
4. **`KURZFASSUNG_ENTWURF.md`** — Anmeldedokument, am 6. August neu geschrieben.
5. `ROADMAP.md` — Verlauf, am 6. August um die B-Serie ergänzt.

**Anmeldung: 30. November.** Bis zu drei Projekte gleichzeitig erlaubt.

---

## 9. Wenn du in einer neuen Sitzung weitermachst

Sag der neuen Sitzung sinngemäß:

> Lies `HANDOFF_PROJEKT_A.md`, `PROJEKTTAGEBUCH.md` (die Einträge vom
> 5. und 6. August) und `BEFUNDE_UND_RICHTUNGEN.md`. Dann `python -m
> quantum_sim.q_summary` und `python -m quantum_sim.q_audit` ausführen, um den
> Belegstand zu sehen. Danach Q-014 starten.

Die drei offenen Entscheidungen, die niemand außer dir treffen kann:
Aufteilung A/B für die Anmeldung, die `B-003`-Doppelvergabe, und der echte
Lesestand der Literatur.
