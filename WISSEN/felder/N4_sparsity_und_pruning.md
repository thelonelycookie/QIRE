# N4: Sparsity und Pruning

## Landkarte
- Übersichten: [arXiv:2308.06767] (Pruning), [arXiv:2512.02550] (sparse Inferenz).
- Linien: (1) Einmal-Pruning nach Training: SparseGPT [arXiv:2301.00774], Wanda [arXiv:2306.11695], 2:4-Proximal [arXiv:2501.18015]; (2) Sparsity von Grund auf: N:M mit SR-STE [arXiv:2102.04010], Lottery Ticket [arXiv:1803.03635], Pixelated Butterfly [arXiv:2112.00029]; (3) Aktivierungssparsity: ReLU Strikes Back [arXiv:2310.04564], Q-Sparse [arXiv:2407.10969], TEAL, Deja Vu [arXiv:2310.17157]; (4) Sparsity + Quantisierung: SLiM [arXiv:2410.09615], Compression Trinity [arXiv:2608.24070], Compression Scaling Laws [arXiv:2502.16440], Sparse-BitNet [arXiv:2603.05168].
- Wer gegen wen: bei gleichen Bytes schlägt 50 % + 4 Bit reines 3 Bit (ab ~2,7B) und reines 2 Bit [arXiv:2301.00774, arXiv:2410.09615].

## Stärkste Baselines/Gegner für QIRE
- Wanda/SparseGPT + 4 Bit als Gegner für jede "gleiche Bytes"-Aussage.
- Zu beachten: Sparsity-Bitmaske kostet Bytes; 2:4 spart nur auf GPU-Hardware, auf CPU kein Speedup [arXiv:2102.04010].

## Wichtige Zahlen (mit geprüft/ungeprüft)
- 175B: 50 % + 4 Bit PPL 8,29 gegen 3 Bit 8,68 (ungeprüft) [arXiv:2301.00774].
- SLiM: +5,66 % über Stand der Technik, +0,6 % gegen dicht bei gleichem Parameterbudget (Abstract, API bestätigt, Volltext nicht gelesen) [arXiv:2608.24070].
- Alle Zahlen geprueft=false.

## Quantenbezug
- Echt quanten: keiner (Quantenschaltkreis-Pruning wie [arXiv:2408.13352] ist anderes Thema).
- Analogie: Auswahl über Gewicht mal Aktivierung ist klassisch. Hebel für QIRE: Rotation vor Quantisierung glättet Ausreißer, was auch Aktivierungssparsity verbessert (LaRoSA [arXiv:2507.01299]); das ist der anschlussfähige Weg.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3 (Rotation ≈ gleichwertig gegen faire Gegner): Sparsity+Quantisierung ist ein fairer Gegner bei gleichen Bytes, der bisher nicht im Vergleich steht.
- I-049 nicht betroffen.

## Lücken/offene Fragen
- Gleiche Bytes, Kleinmodell (unter 100M), CPU: kaum Belege; Bitmasken-Kosten werden oft nicht eingerechnet.
- Kein neuer Beleg 2025–2026 für den direkten Gleich-Bytes-Vergleich außer Dissertation [arXiv:2608.24070].

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2, b6 | 6 | hoch; Kernarbeiten (SparseGPT, Wanda) fehlten bisher |

## Schlüsse und neue Wege (Agent)
1. Schluss: Vor Behauptungen "gleiche Bytes" einen Wanda/SparseGPT+4-Bit-Gegner mit eingerechneter Maske aufnehmen. Belege: [arXiv:2301.00774, arXiv:2410.09615]. Langweiligste Erklärung: Sparsität ist bei Großmodellen billig, bei Kleinmodellen nicht. Quantenbezug: keiner.
2. Schluss: Rotation (QIRE) plus Aktivierungssparsity testen, da Rotation Ausreißer glättet [arXiv:2507.01299]. Langweiligste Erklärung: klassischer Glättungseffekt wie bei Hadamard-Quantisierung. Quantenbezug: Analogie (unitäre Basiswahl).
3. Schluss: 2:4 nur als Bytes-Rechnung, nicht als Geschwindigkeit auf CPU ausweisen [arXiv:2102.04010]. Langweiligste Erklärung: Hardwareabhängigkeit. Quantenbezug: keiner.
