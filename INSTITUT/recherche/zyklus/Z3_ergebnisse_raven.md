# Z3a, Z3b, Z3c, Z4a und Z1_z2lr auf Raven (20 bzw. 5 Tuning-Seeds, Erkundung, kein Befund)

## Kernaussage
**Gegen faire Quantisierer ohne Rotation bleibt vom Rotationsvorteil fast nichts übrig, im Sprachmodell gar nichts.**
- Der große FFN-Gewinn aus Z2 war ein Gitterartefakt: Das Gitter ohne Nullstufe kostet ohne Rotation 1,3–1,4 bpc. Mit einem Lloyd-Gitter ist der Fehler ohne Rotation bei 2 Bit sogar kleiner als mit Hadamard.
- Die stärksten Gegner sind Gruppenskalen (g16, g32) und asymmetrische Quantisierung.

## Z3a Z3_bytegleich
| Ampel | Gewinn | plus | grün | Gewinner Rotation / bester Gegner |
|---|---|---|---|---|
| ampel_mlp | +0,163 ± 0,100 pp | 18/20 | 0/20 | gross_rot_qat / mittel4_roh, gross_roh_qat |
| ampel_lm | −0,115 ± 0,039 bpc | 0/20 | 0/20 | gross_rot / roh_g16 |
| ampel_mlp_alt (gegen klein) | +0,496 ± 0,176 pp | 20/20 | 11/20 | |
| ampel_lm_alt | −0,049 ± 0,062 bpc | 4/20 | 4/20 | |

Rauschtyp R1, Steigung k (Mittel): kanal −0,33, rot 0,10, g16 0,71, g32 0,80, g64 0,86, roh 0,91. Rotation macht das Rauschen fast additiv, das bringt aber keine Qualität gegen Gruppenskalen.

## Z3b Z3_gitter_block
Schaden bei T=2400 in bpc, ohne Rotation / mit Hadamard:

| Gitter | gelu 2 Bit | gelu 3 Bit | z² 2 Bit | z² 3 Bit |
|---|---|---|---|---|
| unif0 | 0,202 / 0,068 | 0,108 / 0,016 | 0,289 / 0,065 | 0,096 / 0,015 |
| nozero | 1,441 / 0,054 | 0,511 / 0,015 | 1,304 / 0,050 | 0,435 / 0,014 |
| lloyd | **0,041** / 0,052 | 0,015 / 0,013 | **0,038** / 0,048 | 0,008 / 0,013 |
| µ-Gesetz | 0,146 / 0,058 | 0,067 / 0,014 | 0,275 / 0,053 | 0,042 / 0,014 |

Alle Ampeln rot. Die Blockgröße ist bei T=2400 hinfällig, weil die Rotation dort meist keinen Gewinn mehr hat.

## Z3c Z3_ffn_fair
Alle 12 Ampeln rot. Bester Gegner ist meist gruppe32. Bei z² mit 3–4 Bit bleibt ein Rest von +0,001 bpc, weit unter der Schwelle.

## Z4a Z4_bitkurve
- ampel_mlp: +0,180 ± 0,080 pp, plus 20/20, grün 0/20.
- ampel_lm: −0,039 ± 0,033 bpc, plus 1/20, grün 0/20.

## Z1_z2lr, Seeds 950–954
Differenz arm − GELU in Test-bpc. In Klammern: Seeds, in denen der Arm besser ist als GELU. ⚠ = Randtreffer der Lernrate.

| (d, T) | z² | z²+LN | ReLU² |
|---|---|---|---|
| 256, 4000 ⚠ | +0,031 ± 0,012 (0/5) | +0,023 ± 0,011 (0/5) | +0,008 ± 0,018 (2/5) |
| 256, 16000 | +0,028 ± 0,007 (0/5) | +0,029 ± 0,006 (0/5) | +0,002 ± 0,004 (2/5) |
| 512, 4000 ⚠ | −0,000 ± 0,012 (3/5) | −0,005 ± 0,010 (3/5) | −0,021 ± 0,005 (5/5) |
| 512, 16000 | +0,045 ± 0,009 (0/5) | +0,026 ± 0,009 (0/5) | −0,009 ± 0,016 (4/5) |

z² ist nirgends sicher besser als GELU. Nur ReLU² (Primer) hält mit oder ist leicht besser.

## Folgen
- Grüne Befunde aus Z1/Z2 zur Rotation gelten nur gegen schwache Gegner. Im ARCHIV werden sie auf „rot gegen fair“ herabgestuft.
- „Rotation flacht den PTQ-Schaden ab“ ist ebenfalls gegen eine schwache Rohbasis gemessen und muss gegen g32 bzw. Lloyd wiederholt werden.
- Ehrliche Erzählung für die Arbeit: Rotation (Hadamard, quanten-inspirierte Unitarität) macht das Quantisierungsrauschen additiv (k≈0,1). Gruppenskalen erreichen dieselbe Qualität mit anderen Mitteln. Der quanten-inspirierte Weg ist damit gleichwertig, aber nicht besser.
