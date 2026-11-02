# N6: Zufalls-/Seed-Gewichte, Random Features

## Landkarte
- Kompression über Zufallsbasis + Koeffizienten: HashedNets [arXiv:1504.04788], PRANC [arXiv:2206.08464], NOLA [arXiv:2310.02556], VeRA [arXiv:2310.11454], RandLoRA [arXiv:2502.00987], SeedLM (Seed je Block, LFSR, data-free) [arXiv:2410.10714]. Stützpfeiler: LoRA mit festem zufälligem A reicht [arXiv:2402.16842], Theorie [arXiv:2506.14530].
- Auswahl statt Training in Zufallsnetzen: Supermasks, Slot Machines [arXiv:2101.06475], Strong-LTH-Theorie [arXiv:2210.16589], neu differenzierbare Extraktion [arXiv:2603.08914], Double-Scoring [arXiv:2607.20555].
- Random Features/Kernel: Fastfood [arXiv:1408.3060], SORF [arXiv:1610.09072], optische Projektionen [arXiv:1510.06664]; Dequantisierung variationaler QML durch RFF [arXiv:2210.13200].
- Reservoir/ELM: feste Zufallsdynamik + linearer Readout [arXiv:2307.15092]; ELM-Universalität [arXiv:1610.09608]; Quanten-Reservoirs [arXiv:2605.01253], [arXiv:2510.18623].
- Gegner untereinander: Zufallsbasis vs. gelernte Basis (SeedLM hat Seedsuche, NOLA nicht); Zufallsnetz+Maske vs. Training (Maske: Sparsity-Bottleneck bei edge-popup).

## Stärkste Baselines/Gegner für QIRE
- SeedLM: 4 Bit, ca. 97,9 % Zero-Shot-Erhalt (Abstract, ungeprüft), FPGA bis ca. 4x Speed-up bei 70B.
- VeRA/NOLA für Adapter-Kompression; Supermasken (Gates) für Zufallsnetz-Auswahl.
- Für Rotationen: Zufalls-Hadamard-Rotation ist selbst Zufallsprojektion; Gegner ist "Zufall + Skalar".

## Wichtige Zahlen (mit geprüft/ungeprüft)
- SeedLM 97,9 % bei 4 Bit: ungeprüft (nur Abstract/Suchtreffer).
- edge-popup ca. 50 % Sparsity vs. Gates über 90 % (ResNet50): ungeprüft, Drittquelle.
- 2-Design-Fehler Theta(d^-l) [arXiv:1502.07514]: ungeprüft (Abstract).
- Kein Volltext gelesen in Welle 3 c2.

## Quantenbezug
- Echt quanten: Haar-Zufallsunitäre und 2-Designs [arXiv:0802.1919], [arXiv:1502.07514]; Quantenreservoirs mit Haar-Gattern [arXiv:2605.01253]; Quanten-ELM.
- Analogie: Seeds/PRNG = klassischer Pseudozufall; Fastfood = Diagonal*Hadamard*Permutation ähnelt dem Aufbau von X-/Z-diagonalen 2-Designs (Nakata). Nur Strukturanalogie, kein Quantenvorteil (I-038).
- Quanten-inspirierter Hebel: Randomisierung statt Optimierung; ein 2-Design reicht für zweite Momente, also reicht billiger strukturierter Zufall. Passt zu QIRE-Rotationen.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3 (Rotation gegen faire Gegner ≈ gleichwertig): stützend, denn feste Zufallsbasis (LoRA-A, VeRA, SeedLM) wirkt ähnlich wie gelernte; Rotation bringt vermutlich Gleichwertigkeit, keinen Sprung.
- I-038 (Wirkung klassisch): RFF-Dequantisierung zeigt dasselbe Muster in QML.
- Kein Widerspruch gefunden.

## Lücken/offene Fragen
- Keine Folgearbeiten zu SeedLM gefunden (nur Suchtreffer, evtl. Lücke der Suche).
- Kein direkter Vergleich 2-Design- vs. Haar-Reservoir auf Benchmarks gefunden.
- "Fester Zufall + gelernte Skalare pro Gewicht in LLM" nicht gefunden.
- Kein Volltext gelesen; alle Zahlen ungeprüft.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| W3 c2 | 10 | ca. 10 von 10 Treffern neu bei 9 Suchen; Feld noch nicht gesättigt (Reservoir/ELM, Supermasken dünn) |

## Schlüsse und neue Wege (Agent)
1. Schluss: QIRE-Rotation als SeedLM-Variante testen (Hadamard/Diagonal-Struktur aus Seed + wenige Koeffizienten). Belege: [arXiv:2410.10714], [arXiv:1408.3060], [arXiv:1502.07514]. Langweiligste Erklärung: gleich gut wie SeedLM, weil beides Zufallsbasis ist. Quantenbezug: 2-Design-Struktur als Begründung.
2. Schluss: Zufallsbasis muss nicht gelernt werden (A fest in LoRA reicht). Belege: [arXiv:2402.16842], [arXiv:2310.11454]. Langweiligste Erklärung: Überparametrisierung macht jede Basis ausreichend. Quantenbezug: Haar-Mittelung analog.
3. Schluss: Supermaske auf Zufallsgewichten (Seed + Bitmaske) als Kompressionsformat mit Gates vergleichen. Belege: [arXiv:2603.08914], [arXiv:2101.06475]. Langweiligste Erklärung: Maskenbits kosten mehr als ein 4-Bit-Gewicht. Quantenbezug: gering (Analogie).
