# K1: Gewichtsquantisierung PTQ/QAT

## Landkarte
- Linien: (a) Rundung mit Fehlerkorrektur: GPTQ, GPTAQ, Qronos [arXiv:2505.11695]; (b) Skalen/Äquivalenztransformation: AWQ, OmniQuant [arXiv:2308.13137]; (c) Inkohärenz/Rotation + Gitter: QuIP, QuIP#, QTIP, OptRot [arXiv:2512.24124], KronQ [arXiv:2607.07964]; (d) rotationsfrei sub-4-Bit: D2Quant [arXiv:2602.02546]; (e) Skalengesetze: ParetoQ, Kumar et al. (Präzision), Apple-QAT-Compute [arXiv:2509.22935].
- Stand 2025–26: Bei 4 Bit liegen GPTQ/AWQ mit Gruppe 128 nahe beieinander (Blog, ungeprüft); bei 2–3 Bit trennen sich die Verfahren stark, Rotation+Gitter (QuIP#) vorn. Fast alle Vergleiche halten die nominale Bitzahl fest, nicht die Bytes inkl. Skalen.

## Stärkste Baselines/Gegner für QIRE
- Fairer Gegner bei gleichen Bytes: Gruppenskalen (g16/g32) + asymmetrisch/Lloyd-Gitter + GPTQ/Qronos-Rundung; Skalen-Metadaten in die Bitzahl einrechnen.
- Bei Bedarf Rotationsgegner: OptRot (gelernt), QuIP#/QTIP (Gitter), MR-GPTQ [arXiv:2509.23202].
- Einziger Befund zu fester Rotation in Gruppen: Hadamard hilft bei NVFP4 (Gruppe 16) nicht, bei MXFP4 (Gruppe 32)/INT4 schon [arXiv:2509.23202].

## Wichtige Zahlen (mit geprüft/ungeprüft)
- Qronos 2 Bit, Llama3 1B/3B/8B Wiki2: 17,8/11,4/9,3 gegen OPTQ 24,6/13,2/10,4 (ungeprüft, Sekundärquelle).
- KronQ Llama-3-70B 2 Bit: 7,93 (Abstract) gegen 8,43 (Autorenkommentar), ungeklärt (ungeprüft).
- PolarQuant: Hadamard ≈ 98 % des Gewinns (Qwen3.5-9B PPL 6,90 → 6,40), ungeprüft.
- Keine Zahl im Volltext geprüft.

## Quantenbezug
- Echt quanten: keiner. Rotation/Inkohärenz ist klassische lineare Algebra (Johnson-Lindenstrauss, Konzentration); Analogie zu Quanten-Zufallsunitaries/Designs. Hebel für QIRE: Rotation als Struktur-Prior (Butterfly/Kronecker), Wirkung klassisch (I-038).

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Z3: stützt, dass Rotation gegen faire Gruppenskalen/Lloyd wenig bringt [arXiv:2509.23202]; Z3b Gitterartefakt passt zu Shrinkage-Bias-Befund [arXiv:2606.20381]. Rotation hilft real bei grober Skala (g32 MXFP4) und im Gradientenpfad.

## Lücken/offene Fragen
- Keine Studie vergleicht Rotation vs. Gruppenskalen strikt bytegleich in kleinen Modellen (<1B). Genau diese Lücke füllt QIRE.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| 2 (b1) | 13 neu (K1 anteilig) | siehe Schätzung |

## Schlüsse und neue Wege (Agent)
1. **Schluss:** Rotation lohnt nur, wenn die Skalengranularität grob ist (pro Kanal, g≥32 mit FP-Gitter) oder Ausreißer stark sind; bei g16 oder gutem Lloyd-Gitter ist sie gleichwertig. Belege: [arXiv:2509.23202], Z3. Langweiligste Erklärung: Gruppenskalen leisten selbst schon Ausreißer-Isolation. Quantenbezug: Analogie (Zufallsunitär).
2. **Schluss:** Neben Rotation sind Rundung (Qronos/GPTQ) und Bitzuteilung (KronQ) eigenständige Hebel; QIRE sollte Rotation gegen GPTQ-gerundete Gruppenquantisierung testen, nicht gegen RTN. Belege: [arXiv:2505.11695], [arXiv:2607.07964]. Langweilig: GPTQ-Rundung allein erklärt die Gewinne. Quantenbezug: keiner.
3. **Schluss:** Gelernte Rotation (OptRot) vs. feste Hadamard bei gleichen Bytes ist ein offener, billiger Test. Beleg: [arXiv:2512.24124]. Langweilig: mehr Freiheitsgrade, nicht Struktur. Quantenbezug: Analogie.
