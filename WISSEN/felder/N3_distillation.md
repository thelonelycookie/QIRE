# N3: Distillation

## Landkarte
- **Skalierung/Kosten:** Destillation lohnt bei gleichem Rechenbudget nur, wenn ein Lehrer schon existiert oder viele Schüler entstehen [arXiv:2502.08606]. Sonst gewinnt überwachtes Training ab einem Rechenniveau.
- **Pruning+KD:** Minitron, <3 % der Daten für 2–4× kleinere Modelle [arXiv:2407.14679, arXiv:2408.11796].
- **Niedrigbit QAT+KD (bereits im Bestand):** BitDistiller [arXiv:2402.10631], BitNet Distillation [arXiv:2510.13998], TernaryLLM [arXiv:2406.07177], Token-Scaled Logit Distillation [arXiv:2308.06744]. Übersicht: [arXiv:2409.16694]. KD ist das robuste QAT-Ziel auch für Reasoning [arXiv:2601.14888].
- **QAT-Skalierung:** [arXiv:2505.14302] (Fehler ~ Größe, Tokens, Granularität), [arXiv:2509.22935] (optimaler QAT-Anteil, Tokens pro Parameter-Byte).
- **Informationssicht:** [arXiv:2411.07483] (Partial Information Decomposition): nur aufgabenrelevante, redundante Lehrerinformation ist destillierbar; Kapazität des Schülers begrenzt.
- **Quanten-KD:** [arXiv:2508.13054], [arXiv:2505.13205] (Schaltkreis als Schüler/Merkmalsraum).
- Gegner gegeneinander: Pruning+KD (Struktur klein) gegen Quantisierung+KD (Bits klein); beide gegen Neutraining.

## Stärkste Baselines/Gegner für QIRE
- QAT mit Logit-KD (BitDistiller / BitNet Distillation) bei gleichen Bits und gleichem Rechenbudget.
- Minitron-artiges Pruning+KD bei gleichen Bytes.
- Reines Training von null bei gleichen FLOPs (Distillation Scaling Laws als Entscheidungsregel).

## Wichtige Zahlen (mit geprüft/ungeprüft)
- Minitron: <3 % Daten (Abstract bestätigt, nicht Volltext) ungeprüft; bis +16 % MMLU (Sekundärquelle) ungeprüft.
- QAT-Skalengesetz: 268 Läufe (Abstract) ungeprüft; Dremov: 86M–2,2B ungeprüft.
- Kein Volltext gelesen, alle Zahlen ungeprüft.

## Quantenbezug
- **Echt quanten:** QRKD und QD-LLM nutzen Schaltkreise nur als Trainingswerkzeug, Schüler/Gewinn klassisch simuliert; kein bewiesener Vorteil.
- **Analogie:** Wissenstransfer als Kanal mit begrenzter Kapazität (Information-Bottleneck-Sicht [arXiv:2411.07483]); Destillation als Messung/Projektion auf einen Unterraum.
- **Quanten-inspirierter Hebel für QIRE:** Rotation/Hadamard vor der Quantisierung verteilt Ausreißer; KD-Verlust ergänzt das. Nicht gefunden: Arbeit, die Rotation + ternär + Destillation gemeinsam gegen faire Gegner prüft (Suche, nicht erschöpfend).

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3: Rotation gegen faire Gegner ≈ gleichwertig. Passt dazu: KD und Rechenbudget-Fairness [arXiv:2509.22935] können einen scheinbaren Rotationsvorteil erklären. Kein Widerspruch gefunden.
- I-038 (Wirkung klassisch): QRKD/QD-LLM bestätigen, dass Vorteile sich klassisch simulieren lassen.

## Lücken/offene Fragen
- Kein Skalengesetz für Destillation in Niedrigbit-Schüler (nur getrennt: DSL und QAT-Gesetze).
- Gleiche-Bytes-Vergleich Pruning+KD gegen Quantisierung+KD nicht gefunden.
- Rotation im KD-Training nicht untersucht (Suche nach 2026 ergab nur Diffusions-HQ-DM [arXiv:2512.05746], nicht eingetragen, da nicht bestätigt).
- "Target-Centric Survey of QAT" (arXiv:2608.29667) nur als Treffer gesehen, nicht bestätigt, nicht eingetragen.
- Volltexte offen.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1 (a4) | 9 | 9 neu zu 7 Bestand |

## Schlüsse und neue Wege (Agent)
1. **Schluss:** Rotationsvorteile bei QAT sollten mit Destillation und gleichem Rechenbudget neu gemessen werden. Belege: [arXiv:2509.22935], [arXiv:2601.14888], Z3. Langweiligste Erklärung: KD+mehr Tokens gleichen den Rotationsgewinn aus. Quantenbezug: Rotation ist nur klassische Orthogonaltransformation (Analogie).
2. **Schluss:** Es fehlt ein gemeinsames Gesetz Destillation x Bits; QIRE könnte es klein messen (Schülergröße, Bits, Tokens). Belege: [arXiv:2502.08606], [arXiv:2505.14302]. Langweiligste Erklärung: Effekte addieren sich einfach, kein neues Gesetz. Quantenbezug: keiner.
3. **Schluss:** Kanalsicht: Schüler-Kapazität in Bits begrenzt destillierbare Information; Rate-Distortion-Grenze (N8) prüfen. Beleg: [arXiv:2411.07483]. Langweiligste Erklärung: bekannte Kapazitätsgrenze, nur neu formuliert. Quantenbezug: Analogie (Kanal/Messung).
4. **Schluss:** Pruning+KD (Minitron) gegen Quantisierung+KD bei gleichen Bytes als fairer Gegner einplanen. Beleg: [arXiv:2407.14679]. Langweiligste Erklärung: Pruning gewinnt bei hoher Bitbreite, Quantisierung bei niedriger. Quantenbezug: keiner.
