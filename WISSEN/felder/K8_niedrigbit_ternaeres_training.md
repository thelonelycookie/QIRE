# K8: Niedrigbit-/ternäres Training (BitNet, ParetoQ)

## Landkarte
- Linien: BitNet b1.58 [arXiv:2402.17764] und Nachfolger a4.8 [arXiv:2411.04965], v2 (Hadamard) [arXiv:2504.18415], 2B4T; ParetoQ [arXiv:2502.02631] (QAT-Finetuning schlägt Training von Grund auf; 2–3 Bit ≥ 4 Bit); Spectra [arXiv:2407.12327]; FP4-Training: Quartet, NVFP4 [arXiv:2509.25149], Four Over Six [arXiv:2512.02010], MXFP4-Hardware [arXiv:2605.09825], UFP4 [arXiv:2606.20381]; STE-Varianten: CAGE [arXiv:2510.18784], StableQAT [arXiv:2601.19320], progressiv [arXiv:2509.00097].
- Streit Bitbreite: ParetoQ 2–3 Bit vs. Kumar 4–6 Bit vs. FP-Gesetz 4–8 Bit [arXiv:2501.02423]; Setups verschieden.

## Stärkste Baselines/Gegner für QIRE
- Für gleiche Bytes: ParetoQ-QAT-Finetuning (2–3 Bit), BitNet b1.58 von Grund auf, 4-Bit-Gruppen-QAT. Für Kleinmodelle fehlt ein byte-fairer Vergleich ternär vs. 4 Bit.

## Wichtige Zahlen (mit geprüft/ungeprüft)
- NVFP4: Hadamard nur auf Wgrad, bringt bei kleinem Maßstab nichts [arXiv:2509.25149] (Sekundärquelle, ungeprüft).
- MXFP4 [arXiv:2605.09825]: Quantisiertes Wgrad löst Divergenz aus; deterministische Hadamard stabilisiert (ungeprüft).
- Keine Zahl im Volltext geprüft.

## Quantenbezug
- Echt quanten: keiner. Analogie: Ternär/Binär ~ Spin-Zustände (QUBO-Training [arXiv:2601.00449]); Rotation als Basiswechsel. QIRE-Hebel: Hadamard-Basis im Gradientenpfad.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Rotation hilft im Gradienten-/Aktivierungspfad (Wgrad), nicht beim Gewichtsgitter gegen faire Gegner: ergänzt Z3, kein Widerspruch, aber Hinweis auf anderen Einsatzort.

## Lücken/offene Fragen
- Kein Training-von-Grund-auf-Vergleich ternär vs. 4-Bit bytegleich bei <100M Parametern; STE-Varianten selten mit Rotation getestet.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| 2 (b1) | 13 neu (K1+K8) | siehe Schätzung |

## Schlüsse und neue Wege (Agent)
1. **Schluss:** Rotation wirkt im Training am Wgrad (Ausreißergradienten), nicht am Gewichtsgitter. Belege: [arXiv:2605.09825], [arXiv:2509.25149]. Langweilig: Gradienten haben schwerere Ränder als Gewichte. Quantenbezug: Analogie (Basiswechsel).
2. **Schluss:** Gitterform (E2M1 vs. uniform) entscheidet mit; Rotationsgewinne können Gitterartefakte sein (wie Z3b). Beleg: [arXiv:2606.20381]. Langweilig: schlechtes Gitter. Quantenbezug: keiner.
3. **Schluss:** Byte-fairer Test ternär-von-Grund-auf gegen ParetoQ-4-Bit im Kleinmodell wäre ein sauberer QIRE-Beitrag. Belege: ParetoQ, Lücke oben. Langweilig: Skalierungsregime. Quantenbezug: keiner.
