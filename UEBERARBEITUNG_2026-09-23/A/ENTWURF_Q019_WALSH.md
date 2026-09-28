# Q-019W — Walsh statt Fourier, und über die Datenmenge gemessen

**Stand:** 28. September 2026, **vor** jeder Zahl. Ersetzt den Lauf aus
`ENTWURF_ZEITREIHEN.md` / `quantum_sim/zeitreihen.py`. Das alte Modul bleibt
unverändert stehen (Vorlauf, Begründung), wird aber nicht gerechnet.
**Entscheidung Malte, 28.09.:** umbauen auf Walsh-Hadamard **plus**
Lernkurve. Recherche: `../BERICHT/RECHERCHE_2026-09-28/02_q019_walsh_datenknappheit.md`.
**Code (zu bauen):** `quantum_sim/walsh.py`, neues Modul. Vorlage ist
`zeitreihen.py`, vorher gegen die Protokollregeln geprüft (CLAUDE.md §4).

---

## 1. Warum umbauen

1. **Die Fourier-Erzählung ist schon im Vorlauf gefallen.** Die geteilte
   Butterfly (40 Winkel) ist nicht die FFT, sie hat nur dasselbe
   Verbindungsmuster. `butterfly` 0,686, `fft_fest` 0,804, `dense` 0,913.
2. **Die alte Primärhypothese ist fast per Konstruktion wahr** (Review §4).
   Eine feste Permutation kann eine dichte Schicht mitlernen, eine Butterfly
   nicht. Dass die Wechselwirkung auftritt, beweist also nur, dass die
   Butterfly auf die Reihenfolge achtet, aber keinen Vorteil.
3. **Was die Struktur wirklich exakt kann:** Ohne CNOT ist die geteilte
   Butterfly ein Kronecker-Produkt R₁⊗…⊗R₈ von 2×2-Drehungen. Bei Ry(π/2) in
   jeder Lage ist das (bis auf Vorzeichen) die **Walsh-Hadamard-Transformation
   H⊗⁸**. Walsh-Funktionen sind Rechteck- und Schaltsignale. Für sie ist die
   Struktur also die richtige Basis, nicht für Sinus.
4. **Wo 40 gegen 65.536 Parameter eigentlich zählen: bei wenig Daten.** Das
   ist auch „ressourceneffizient“, nur beim Datenbedarf statt bei der
   Rechenzeit. Eine Quelle, die genau Butterfly gegen dicht auf einer
   Walsh-Aufgabe über die Datenmenge misst, hat die Recherche nicht gefunden.
   Theorie und Empirie für strukturierte Matrizen (LDR, Monarch) stützen die
   Richtung.

**Die Gefahr, offen benannt:** Wird die Aufgabe genau auf die Struktur
zugeschnitten, ist ein Sieg wieder per Konstruktion wahr. Dagegen stehen drei
Dinge: der **feste WHT-Arm** (die Struktur ohne Lernen), die **Kontrolle mit
zufälliger Orthobasis** (Passung statt bloßer Parameterarmut) und der
**Low-Rank-Arm** (irgendein parameterarmes Modell). Außerdem ist die
Primärfrage, ob die Winkel die Lösung aus **wenigen Daten lernen**. Dass die
Architektur sie *darstellen* kann, ist vorher schon bekannt.

## 2. Die Aufgabe

Signale der Länge d = 256 = 2⁸, x = Σₖ aₖ·wₖ + Rauschen. Die wₖ sind wenige
(ANTEILE, Start 3) Basisfunktionen der gewählten **Basis**. Die Klasse ist das
**Sequency-Band** des stärksten Anteils: 8 Bänder à 32, also 8 Klassen. Die
Sequency ist die Zahl der Nulldurchgänge, das Walsh-Gegenstück zur Frequenz.

**Warum Sequency:** Die Sequency ist das physikalisch sinnvolle Gegenstück zur
Frequenz. **Korrektur vom 28.09., vor jeder Zahl:** Hier stand zuerst, das
Sequency-Band sei als XOR von Hadamard-Bits über ⟨Z⟩ nicht linear ablesbar. Das
ist falsch, gefunden beim Bau von `walsh.py`. Das Band ist eine Bijektion von
drei Hadamard-Bits, und ein Softmax-Kopf über 8 Klassen trennt jede solche
Bijektion. Im Idealfall ohne Rauschen liest `wht_fest` plus Kopf das Band also
perfekt ab. Ob die **echte** Aufgabe (mehrere Anteile, Rauschen) zu leicht ist,
entscheidet allein `saettigung()` vor dem Lauf. Ist ein Familienarm bei
N = 200 über 0,95, wird die Schwierigkeit erhöht (ANTEILE, RAUSCHEN) und ins
Tagebuch eingetragen, bevor Zahlen entstehen. Ein gesättigtes `wht_fest` ist
kein Fehler, denn es steht außerhalb der Familie.

**Basen (Datenbedingung):**

| Basis | Was | Rolle |
|---|---|---|
| `walsh` | Walsh-Funktionen, sequency-geordnet | Hauptbedingung |
| `zufall` | dieselben Koeffizienten, aber in einer festen Haar-zufälligen Orthobasis Q·H (ein Seed für alle Läufe) | **Kontrolle**: dieselbe Sparsamkeit, keine Kronecker-Struktur |
| `permutiert` | Walsh-Signale mit fester Zeitachsen-Vertauschung | beschreibend (Anschluss an Q-019) |
| `fourier` | Sinus/Kosinus wie Q-019, Bänder nach Frequenz | beschreibend, nur N = 6.000 (Replikation des Vorlaufs) |

Für `dense` sind alle Orthobasen gleich schwer, denn es lernt jede Drehung.
Unterschiede zwischen `walsh` und `zufall` bei `dense` zeigen also nur
Rauschen. Das ist zugleich eine Plausibilitätsprüfung.

## 3. Die Arme

Gleicher Rahmen wie Q-019: normieren → Mittelschicht 256→256 → **gleiche
quadratische Auslese ⟨Z_q⟩ (8 Werte)** → linear 8 → 8 Klassen.

| Arm | Mittelschicht | Parameter | Rolle |
|---|---|---|---|
| `butterfly` | geteilte Butterfly ohne CNOT, 5 Lagen (`OrthoButterfly` aus Q-019) | 40 | Kandidat |
| `dense` | freie 256×256-Drehung | 65.536 | Referenz |
| `lowrank` | I + U·Vᵀ, Rang 1 | 512 | **Kontrolle**: parameterarm, aber ohne Struktur (40 sind mit Rang 1 nicht erreichbar, 512 ist das Minimum; die Abweichung wird benannt) |
| `wht_fest` | feste WHT H⊗⁸, untrainiert | 0 | die Struktur ohne Lernen: Obergrenze bzw. Anker |
| `untied` | ungeteilte Butterfly (`UngeteilteButterfly`) | 5.120 | beschreibend: Preis der Gewichtsteilung |
| `narrow` | Engstelle 256→64→256 | 32.768 | Auflösungsprüfung, nur N = 6.000 |

## 4. Die Lernkurve

N ∈ {50, 200, 1.000, 6.000} Trainingsbeispiele insgesamt, klassenbalanciert
(N = 50 → 6–7 je Klasse). Test fest 2.000 je Seed.

**Trainingsbudget:** feste Zahl an Gradientenschritten, gleich für alle N
(SCHRITTE = 470 ≈ 10 Epochen bei N = 6.000, Batch 128; bei kleinem N entspricht
das vielen Epochen). Begründung: Bei festen Epochen wäre N = 50 nach 4 Schritten
untertrainiert, und die Lernkurve würde das Budget statt der Daten messen.
**Kein** Early Stopping auf den Testdaten.

**Lernrate:** je Arm × Basis × N auf eigenen Tuning-Seeds (3), volles Budget,
Gitter 0,001–0,03, in beide Richtungen verlängert, bis zu 3-mal (Regel 1 a–c).
„Innen“ gilt erst, wenn hinter dem Optimum ein schlechterer Punkt gemessen ist.

## 5. Vorregistrierung

**Messgröße:** balancierte Genauigkeit. **15 Seeds.**
**Vorregistrierter Messpunkt der Familie: N = 200.**

| Test | Bedingung | Vergleich | Ausgänge |
|---|---|---|---|
| **PRIMÄR** | `walsh`, N = 200 | `butterfly` − `dense` | STRUKTUR_BESSER / GLEICHWERTIG / DICHT_BESSER / UNENTSCHIEDEN |
| **P2** | N = 200 | Wechselwirkung (`butterfly` − `dense`)_walsh − (`butterfly` − `dense`)_zufall | PASSUNG_TRAEGT / GLEICHWERTIG / PASSUNG_SCHADET / UNENTSCHIEDEN |
| **P3** | `walsh`, N = 200 | `butterfly` − `lowrank` | STRUKTUR_TRAEGT / GLEICHWERTIG / LOWRANK_BESSER / UNENTSCHIEDEN |

- Wilcoxon zweiseitig + TOST ±0,005, **Holm über genau diese drei.**
- Rangfolge der Zweige wie in `kernaussage.py`: HU_UNVOLLSTAENDIG → HU_AUFLOESUNG
  → HU_BODEN → HU_SAETTIGUNG → TOST → Holm → Richtung → UNENTSCHIEDEN.
- **P2 ist die Absicherung gegen „per Konstruktion“:** Gewinnt die Butterfly
  auch in der zufälligen Basis, kommt der Vorteil aus der Parameterarmut und
  nicht aus der Passung. **P3** trennt „irgendwie wenig Parameter“ von
  „diese Struktur“.

**Beschreibend, ohne Holm:**
- volle Lernkurven aller Arme in `walsh` und `zufall`
- **Übergangspunkt N\*** = kleinstes gemessenes N, ab dem `dense` ≥ `butterfly`
  (linear zwischen den Stützstellen interpoliert, als Bereich berichtet)
- `butterfly` gegen `wht_fest`: Erreicht das Lernen die feste Transformation?
- `untied`, `permutiert`, `fourier` (Vorlauf-Replikation)

**Kontrollen (urteilstragend):**
- **Auflösung:** `dense` muss sich bei N = 6.000 in `walsh` von `narrow`
  trennen (Trennung, nicht Richtung).
- **Boden:** bester Arm ≥ 0,125 + 0,10 in jeder Zelle der Familie.
- **Sättigung:** bei N = 200 kein Familienarm > 0,95 (sonst HU_SAETTIGUNG).
  Vor dem Lauf mit `saettigung()` geprüft.

**Erreichbarkeit (Regel 2):** kleinstes p bei 15 Seeds 6,1·10⁻⁵, nach Holm ×3
1,8·10⁻⁴. TOST ±0,005 nur bei gepaarter sd < 0,011. **Bei N = 200 ist die sd
vermutlich größer.** Das misst `macht()` vor der Freigabe. Liegt sie darüber,
kann GLEICHWERTIG nicht feuern, und das steht hier, bevor Zahlen existieren.

**Erwartung vorher:**
- PRIMÄR: STRUKTUR_BESSER
- P2: PASSUNG_TRAEGT
- P3: STRUKTUR_TRAEGT
- N\* irgendwo zwischen 1.000 und 6.000; `dense` bei 6.000 vorn, wie im
  Fourier-Vorlauf.

Scheitert PRIMÄR schon bei N = 200, trägt das Dateneffizienz-Argument nicht.
Das wäre ein klarer Negativbefund.

## 6. Vortest (Pflicht, CLAUDE.md §5)

- Jeder Ausgang der drei Tests einzeln erzwungen, je 5 Ziehungen (Nullfälle
  40, ≥ 80 %), auf Marke **und** Zweigtext geprüft
- `accuracy` und `balanced_accuracy` im Generator nicht konstant verschieden
- eigenes Rauschen je Arm plus gemeinsamer Seed-Anteil (Lehre Q-019)
- Gitterverlängerung in beide Richtungen
- Theorieprüfung: `butterfly` mit Ry(π/2) überall = H⊗⁸ bis auf Vorzeichen
  (Abweichung < 10⁻⁶). `wht_fest` = H⊗⁸ in **natürlicher** Ordnung, also genau
  das, was die Butterfly darstellen kann. Die Sequency ihrer Zeilen folgt aus
  Gray-Code und Bitumkehr (korrigiert am 28.09., vorher stand hier
  „Sequency-Ordnung“, was dem Arm die Bänder direkt auf die Bits gelegt hätte)
- `permutiert` und `fourier`: nur `butterfly` und `dense`, nur bei N = 6.000
  (so im `PLAN` von `walsh.py`)
- N\*: kleinstes N, ab dem `dense` ≥ `butterfly` bei allen größeren N,
  linear in log N interpoliert
- Mutationsprüfung: drei absichtliche Fehler, alle müssen den Vortest
  durchfallen lassen

## 7. Rechenzeit und Ablauf

Zellen: Familie + beschreibend ≈ 5 Arme × 2 Basen × 4 N, dazu `narrow` und
`fourier` bei 6.000 und `permutiert`. Das sind etwa 45 Bedingungen × (15 Mess- +
~15 Tuning-Zellen). Das Budget pro Zelle ist gleich (470 Schritte). Grob aus
Q-019 (~55 s je Zelle bei 470 Schritten für die Butterfly-Arme, dichte Arme
viel schneller) ergibt das **~8–12 h CPU**. `zeit` misst das vor der Freigabe.
Laufen darf es erst **nach Q-020**, immer nur ein schwerer Lauf zur Zeit.
Die Wiederaufnahme hat wie Q-020 eine eigene Datei je Lauf-ID und nie `latest()`.

## 8. Vor dem Lauf (Malte)

1. Diesen Entwurf freigeben, dann gehört der Vorregistrierungstext ins Tagebuch
   **vor** dem Start, mit Commit.
2. Ergebnisse von `saettigung()`, `macht()` und `zeit` abwarten. Sie können
   ANTEILE/RAUSCHEN oder die Seed-Zahl ändern, und zwar nur vor dem Lauf.
