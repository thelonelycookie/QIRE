# K7: Strukturierte Matrizen (Fastfood, Monarch, Hadamard-MLP)

## Landkarte
- Klassiker: Fastfood [arXiv:1408.3060], ACDC [arXiv:1511.05946], Deep Fried Convnets [arXiv:1412.7149], Butterfly-Faktorisierung [arXiv:1903.05895], Kaleidoscope, Monarch [arXiv:2204.00595], Pixelated Butterfly [arXiv:2112.00029], Monarch Mixer [arXiv:2310.12109].
- Vereinheitlichung: Compute Better Spent (BTT, Monarch als Spezialfall) [arXiv:2406.06248] und Einsum-Suche [arXiv:2410.02117]: strukturierte Schichten können bei gleichem Rechenaufwand bessere Skalenexponenten haben, aber nur mit passender Parametrisierung. Weitere Anwendung: Probabilistic Circuits mit Monarch, gleiche Leistung bei weniger FLOPs, nur gegen dicht verglichen [arXiv:2506.12383].
- Feste Transformationen: Hadamard-MLP (Needle 2), feste WHT + Diagonale statt Attention-Ausgabeprojektion (-25 % Attention-Parameter, steilere Verlustkurve) [arXiv:2603.08343], Block-WHT-Binärschichten [arXiv:2201.02711].
- Kronecker: KroneckerBERT mit Distillation [arXiv:2109.06243], KronA/LoKr als Adapter, Kronecker-Sparse [bib:gonon2025kronecker_sparse].
- Low-Rank vs. strukturiert: Auf vortrainierten Modellen gewinnt einfaches Low-Rank gegen Monarch [arXiv:2406.11307]; Low-Rank-FFN von Null bis 1,3B funktioniert (2,6x FFN-Speedup bei 32 % Parametern, steilere Skalenkurven) [arXiv:2407.09835].

## Stärkste Baselines/Gegner für QIRE
- Low-Rank-FFN bei gleichen Parametern/FLOPs [arXiv:2407.09835]; dichte Schicht mit gleichem Compute (BTT-Studie zeigt, dass Gleichheit nicht selbstverständlich ist).
- Gegen Hadamard-MLP: Low-Rank + Monarch/BTT mit lernbaren Blöcken (mehr Parameter als nur 3 Diagonalen).

## Wichtige Zahlen (mit geprüft/ungeprüft)
- Low-Rank-FFN: 2,6x Speedup, 32 % Parameter (ungeprüft, Abstract).
- WHT-Attention: ca. 25 % weniger Attention-Parameter (ungeprüft, Abstract).
- Volltext nichts gelesen.

## Quantenbezug
- Echt quanten: keiner. Hadamard-Matrix = Walsh-Hadamard-Gatter auf n Qubits; Butterfly = Schaltkreis aus lokalen Gattern, Kronecker = Tensorprodukt. Walsh-Zustandspräparation [arXiv:2307.08384] ist dieselbe Mathematik.
- Analogie/Hebel: Matrix aus log d Stufen lokaler Gatter (Butterfly/Monarch) = "flacher Schaltkreis"; QIRE-Hebel ist Parametersparen bei O(d log d) Rechnung.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3-Muster (einfacher Gegner stark) passt: Low-Rank schlägt Monarch bei Kompression [arXiv:2406.11307]. Neutral zu Hadamard-MLP; kein eigener Befund direkt zu K7 in dieser Welle geprüft.

## Lücken/offene Fragen
- Kein Vergleich Monarch/Kronecker/Hadamard-MLP vs. Low-Rank bei gleichen FLOPs im Training von Null mit kleinen Sprachmodellen gefunden (Suche: nur Teilergebnisse).
- Keine unabhängigen Benchmarks zu Hadamard-MLP (Needle 2 nur Repo/Blog).
- Keine Quelle 2025–2026 für Butterfly/Monarch-Pretraining jenseits der Genannten.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2 (b2) | 4 (K7) | ca. 12 % von ~35 K7-Quellen |

## Schlüsse und neue Wege (Agent)
1. **Fairer Test fehlt: Hadamard-MLP vs. Low-Rank-FFN bei gleichen Parametern/FLOPs.** Belege: [arXiv:2407.09835], [arXiv:2406.06248]. Langweiligste Erklärung: Low-Rank ist gleich gut oder besser (siehe [arXiv:2406.11307]). Quantenbezug: Analogie (Walsh-Hadamard-Gatter).
2. **Compute-Gleichheit entscheidet.** Strukturierte Schichten sparen Parameter, aber nur BTT-artige Parametrisierung gewinnt bei Compute. Langweiligste Erklärung: Parameterzahl ≠ FLOPs. Prüfen, bevor QIRE "effizienter" sagt.
3. **WHT-Ausgabeprojektion als billiger Zusatz** [arXiv:2603.08343] zu QIRE testen (Gewicht sparen, danach quantisieren). Langweiligste Erklärung: Gewinn nur bei kleiner Modellgröße, Autoren-Selbstbericht, unabhängig nicht bestätigt. Quantenbezug: Analogie.

## Runde 2 (Welle 4, d2)
Suchen: Monarch/BTT in Kleinmodellen, muP für strukturierte Matrizen, Fastfood/ACDC/Kronecker in Sprachmodellen (3 Suchen).
- Von Null trainierte strukturierte FFN: Wei et al. [arXiv:2406.16450] (Low-Rank/Blockdiagonal bis 1,3B, Trainingsdynamik problematisch, Self-Guided Training); Monarch GPT-2 ca. 2x Speedup bei gleicher Qualität [arXiv:2204.00595]. Fastfood/ACDC in Sprachmodellen: nichts gefunden. Monarch-Pretraining 2025–26 in Kleinmodellen: nichts gefunden.
- muP: nur Compute Better Spent [arXiv:2406.06248] liefert Regeln für strukturierte Schichten (Init/LR weichen vom dichten Fall ab); Transfer bei Matrix-Optimierern nur dicht [arXiv:2512.05620]. Konsequenz: Hadamard-MLP-Test braucht eigenes LR-Tuning je Variante.
- Lücke bleibt: Hadamard-MLP vs. Low-Rank bei gleichen FLOPs, Kleinmodell, fair getunt.
- Sättigung Runde 2: 2 neue Quellen / 3 Suchen, ca. 5 % von ~37; Feld gesättigt.
