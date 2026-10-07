# Skalierbarkeit und quanteninspirierter Vorteil — Forschungsplan

*Stand 07.10.2026. Grundlage: Projekt A (`UEBERARBEITUNG_2026-09-23/A/`), INSTITUT
(Magie-Linie), Websuche. Methodik: `INSTITUT/FORSCHUNGSMETHODIK.md`.*

---

## Teil 1 — Warum „einfach mehr Qubits“ nicht einfach ist

Der erste Gedanke stimmt halb: n Qubits = 2ⁿ Dimensionen bei nur 2·n·L Winkeln.
Die Schicht wird also exponentiell breiter, die Parameterzahl wächst nur
logarithmisch. Aber es gibt **drei verschiedene Skalierungsfragen**, die man
trennen muss, weil sie verschiedene Antworten haben:

### 1a. Rechenkosten auf klassischer Hardware (quanteninspiriert)
*Stand: gut belegt.* Die Schaltung ist exakt ein Butterfly-Netz (Abweichung
7·10⁻⁷). Kosten O(d·log d) statt O(d²). Gemessen: Übergang bei 14 Qubits
(16.384 Dim.), 1,59× bzw. 2,31× schneller in der Inferenz, nur 1,18× im Training;
ab 32.768 passt die dichte Schicht nicht mehr in 16 GB.
**Offen:** GPU (T-002), und ob der Vorteil bei echten Modellbreiten
(4.096–16.384, FFN-Blöcke) auch *mit Genauigkeit* hält. Vergleich mit
Monarch/BOFT nötig — die sind die eigentliche klassische Konkurrenz in dieser Kategorie.

### 1b. Lernfähigkeit und Ausdruckskraft bei wachsendem n
Hier wird es interessant. Hindernisse, alle mit 2⁻ⁿ:
| Hindernis | QIRE-Messung | Literatur |
|---|---|---|
| Barren Plateaus (Gradientenvarianz ~2⁻ⁿ) | Q-005 gebaut, nicht ausgewertet | McClean 2018; Cerezo u. a. 2025 |
| Kernkonzentration | 1,24·2^(−0,96n) gemessen | Thanasilp u. a. 2024 |
| Ausdruckskraft: 2nL Winkel in SO(2ⁿ) | 0,245 % bei 8 Qubits | — |
| Born-Auslese: 2ⁿ Wahrscheinlichkeiten, jede ~2⁻ⁿ | — | — |

**Die Frage dahinter:** Reichen logarithmisch viele Parameter, wenn die Breite wächst?
Projekt B legt nahe: ja, *wenn* die Aufgabe nur wenige Richtungen braucht
(B-014: ~7). Das wäre ein echter, eigener Zusammenhang zwischen A und B:
**Die Schaltung skaliert, solange der Effektivrang der Aufgabe klein bleibt.**
Das ist prüfbar.

### 1c. Echte Quantenhardware (nur für Q3-Aussagen)
- **Datenladen** frisst alles: 96 % der T-Gatter (I-020, C-kodierung). Amplitudenkodierung braucht O(2ⁿ) Gatter.
- **Schüsse:** ~1.000 je Bild schon bei kleinem n (I-008); das wächst mit 2ⁿ Ausgängen.
- **Cerezo u. a. (Nature Comm. 2025):** Schaltungen, die beweisbar keine Barren Plateaus haben, sind meist klassisch simulierbar.

**Wichtige Umdeutung für QIRE:** Für ein *quanteninspiriertes* Projekt ist das
kein Gegenargument, sondern genau der Punkt: Was trainierbar ist, lässt sich
klassisch schnell rechnen. Und die Magie-Linie sagt dasselbe von der anderen
Seite: wenig Magie = wenige T-Gatter = klassisch billig simulierbar
(~2^(0,23·t)). **Ehrliche Kernaussage:** Wir nehmen Struktur aus der
Quantenphysik und rechnen sie klassisch. Der Gewinn kommt aus der Struktur,
nicht aus einem Quantencomputer.

### Plan Skalierung (S-Serie)
| Nr. | Frage | Kosten | Entscheidet |
|---|---|---|---|
| **S1** | Q-005 auswerten: Gradientenvarianz vs. n (4–14) für QIRE, ohne CNOT, Identitäts-Init | billig (kein Training) | ob ab n≈10 überhaupt trainierbar |
| **S2** | **Genauigkeit vs. Breite** bei fester Aufgabe: n = 6…14, gleiche Daten, QIRE vs. dicht vs. Low-Rank vs. Monarch, gleiche Parameter/gleiche FLOPs | mittel (CPU-Nacht) | ob der Geschwindigkeitsvorteil ohne Genauigkeitsverlust kommt |
| **S3** | **Rang-Hypothese:** Aufgaben mit kontrolliertem Effektivrang (B-014-Generator) × n. Vorhersage: QIRE hält mit, solange Rang ≲ Anzahl Winkel | mittel | die Verbindung A↔B (eigener Befund) |
| **S4** | Born-Auslese bei großem n: welche Auslese (volle Verteilung, ⟨Z⟩ je Qubit, Paritäten) bleibt informativ? | billig | ob die Auslese der Engpass ist |
| S5 | GPU (T-002 nach Fehlerbehebung): Wall-Clock bis 65.536 | GPU | Effizienz auf echter Hardware |

S1 und S4 zuerst — sie sind billig und können S2 sparen.

---

## Teil 2 — Den quanteninspirierten Vorteil vollständig belegen

### Was „vollständig belegt“ heißt (aus Literatur + eigenen Fehlern)
Bowles/Ahmed/Schuld 2024: einfache klassische Modelle schlagen meist die
Quantenmodelle, und Verschränkung weglassen schadet oft nicht. Eine
Vorteilsaussage muss deshalb **fünf Fragen** beantworten:

1. **Gegen wen?** Stärkste klassische Konkurrenz *bei gleichem Budget*, mit
   eigener Lernrate je Arm. Nicht nur Identität oder schwache Kontrolle
   (`norm_deep`-Lehre!). Konkret: linear, ReLU-MLP gleicher Engstelle, LoRA/Low-Rank,
   **Monarch/BOFT (Butterfly ohne Quantenstruktur)**, zufällige feste Orthogonale.
2. **Bei welchem Budget?** Parameter, FLOPs, Wall-Clock, Speicher — getrennt
   berichten. Am besten als **Pareto-Kurve** (Genauigkeit über Parameter), nicht ein Punkt.
3. **Welche Quantenzutat trägt?** Ablationen, einzeln abschalten:
   CNOT-Ring (Verschränkung) · Born-Auslese (|ψ|² → linear) · Unitarität
   (freie statt orthogonale 2×2-Blöcke) · Tensorprodukt-Teilung der Winkel.
   Was übrig bleibt, wenn eine Zutat weg ist, zeigt, *welche* Zutat den Vorteil liefert.
4. **Wie allgemein?** ≥ 3 Datensätze (MNIST, Fashion, eine echte schwerere
   Aufgabe), ≥ 2 Breiten. Institut hat gelernt: MNIST-Befunde replizieren nicht
   automatisch (I-034-fashion).
5. **Wie sicher?** Vorregistriert, genug Seeds, Holm, TOST für Gleichwertigkeit.
   Habt ihr schon — beibehalten.

### Wo QIRE steht
- **Q-020** ist schon fast genau dieses Design (Arme q1/q5 mit/ohne CNOT, `kron`,
  MLP, linear). **Fehlt:** Monarch/BOFT- bzw. Low-Rank-Arm mit gleicher
  Parameterzahl, Born-Auslese-Ablation, freie-statt-unitäre-Ablation.
  → Q-020 um diese Arme ergänzen, *bevor* er läuft. Das macht ihn zum Hauptbeweis.
- **Stärkste Konkurrenz Quantum-PEFT** (ICLR 2025): fast dieselbe Schaltung,
  schon auf Milliarden-Parameter-Modelle skaliert. Einziger Unterschied:
  QIRE nutzt die **Born-Auslese** (quadratisch, nichtlinear). Damit wird die
  Born-Ablation zur wichtigsten einzelnen Messung: *Trägt die Born-Regel?*
  Wenn ja, ist das QIREs Alleinstellung.
- Warnbeispiel aus 2026 (arXiv:2608.22475): „Vorteil“ von +0,19 % mit
  KI [−1,10; +1,48] — das ist kein Vorteil. So nicht.

### Plan Vorteil (V-Serie)
| Nr. | Was | Entscheidet |
|---|---|---|
| **V1** | Q-020 erweitern: + `monarch` / `lowrank` gleicher Parameterzahl, + `q5_linear` (ohne Born), + `q5_frei` (nicht unitär) | Welche Zutat trägt, gegen die richtige Konkurrenz |
| **V2** | Pareto-Kurve: Genauigkeit vs. Parameter für alle Arme (3 Budgets) | Vorteil als Kurve statt als Einzelpunkt |
| **V3** | Wiederholung auf Fashion + einer schwereren Aufgabe (z. B. CIFAR-10-Graustufen oder ein Finetuning-Szenario) | Allgemeinheit |
| V4 | Direkter Vergleich mit Quantum-PEFT im selben Aufbau | Abgrenzung zum stärksten Vorgänger |

### Mögliche Ausgänge — alle verwertbar
- Born trägt, Verschränkung trägt → stärkste Aussage: echte quanteninspirierte Zutaten.
- Nur die Butterfly-Struktur trägt → ehrlich: Vorteil ist strukturell (bekannt), QIRE ist eine quantenphysikalisch motivierte Herleitung davon. Immer noch eine Arbeit.
- Nichts schlägt Monarch → Befund „quanteninspiriert ≈ Butterfly“, mit Erklärung. Passt zur Linie „Was bleibt von quanteninspiriert?“.

---

## Empfohlene Reihenfolge (bis Anmeldung 30.11.)
1. **V1** in `kernaussage.py` einbauen, Vortest, vorregistrieren, laufen lassen (Hauptbeweis).
2. **S1 + S4** parallel billig nebenher (Agenten im Institut).
3. **S3** (Rang-Hypothese) als eigener Befund, der A und B verbindet.
4. S2/S5 und V3 im Dezember/Januar, wenn GPU da ist.

## Quellen
- Cerezo u. a., *Does provable absence of barren plateaus imply classical simulability?*, Nature Comm. 16, 7907 (2025), arXiv:2312.09121
- Bowles, Ahmed, Schuld, *Better than classical?*, arXiv:2403.07059
- Koike-Akino u. a., *Quantum-PEFT*, ICLR 2025, arXiv:2503.05431
- Yaras u. a., *MonarchAttention*, NeurIPS 2025 · Dao u. a., *Monarch*, arXiv:2204.00595
- *Quantum-Inspired Hybrid Neural Networks for Neural Decoding*, arXiv:2608.22475 (Warnbeispiel)
- *QML advantages beyond hardness of evaluation*, arXiv:2504.15964
