# W7-Vortest (Phase 2, Runde 3): Quadrat-Familie unter 4-Bit-Quantisierung

Erkundung, kein Befund. Tuning-Seed 950, Byte-LM d=128, f=512, 4 Lagen (Rezept W2), Val-Bytes: erste 32768 (gepaart). Einzelseed: Unterschiede < ca. 0,02 bpc sind nicht belastbar.
Aktivierungsquantisierung 4 Bit pro Token symmetrisch (roh | Hadamard davor), Gewichte 4 Bit Lloyd/Zeilenskala (alle Linear, ohne Rotation).

## T = 3000

bpc (Val) und Verlust gegenueber fp32 (dbpc = quantisiert - fp32):

| Akt | FFN-Par. | fp32 | A4 roh (beide) | A4 had (beide) | nur Ein roh/had | nur Mitte roh/had | W4 | W4A4 roh | W4A4 had |
|---|---|---|---|---|---|---|---|---|---|
| gelu | 526848 | 2.512 | +0.090 | +0.016 | +0.009/+0.009 | +0.080/+0.006 | +0.022 | +0.114 | +0.038 |
| relu2 | 526848 | 2.662 | +0.022 | +0.017 | +0.009/+0.009 | +0.012/+0.007 | +0.019 | +0.040 | +0.034 |
| z2 | 526848 | 2.682 | +0.062 | +0.013 | +0.010/+0.007 | +0.055/+0.008 | +0.015 | +0.087 | +0.030 |
| abs | 526848 | 2.694 | +0.022 | +0.021 | +0.014/+0.010 | +0.009/+0.013 | +0.017 | +0.043 | +0.037 |
| swiglu | 527016 | 2.518 | +0.075 | +0.017 | +0.011/+0.009 | +0.068/+0.006 | +0.022 | +0.099 | +0.040 |

Kurtosis (Excess) und Ausreisser:

| Akt | Ort | Kurtosis Mittel | Kurtosis Max (Lage) | Kurt. nach Hadamard | max/rms roh | max/rms had | Nullanteil |
|---|---|---|---|---|---|---|---|
| gelu | ein | 0.3 | 0.6 | -0.0 | 3.0 | 2.8 | nan |
| gelu | mitte | 31.9 | 41.2 | 3.0 | 8.8 | 3.1 | 0.03 |
| relu2 | ein | 0.7 | 1.1 | -0.1 | 3.1 | 2.8 | nan |
| relu2 | mitte | 1096.4 | 2347.6 | 9.3 | 13.7 | 2.7 | 0.84 |
| z2 | ein | 0.4 | 1.2 | -0.1 | 3.0 | 2.8 | nan |
| z2 | mitte | 139.5 | 440.2 | 0.9 | 8.4 | 3.1 | 0.00 |
| abs | ein | 1.0 | 1.6 | -0.0 | 3.4 | 2.8 | nan |
| abs | mitte | 1.6 | 2.2 | 0.2 | 3.3 | 3.4 | 0.00 |
| swiglu | ein | 0.6 | 1.6 | -0.1 | 3.1 | 2.8 | nan |
| swiglu | mitte | 192.9 | 301.4 | 3.3 | 9.3 | 3.0 | 0.00 |

## T = 8000

bpc (Val) und Verlust gegenueber fp32 (dbpc = quantisiert - fp32):

| Akt | FFN-Par. | fp32 | A4 roh (beide) | A4 had (beide) | nur Ein roh/had | nur Mitte roh/had | W4 | W4A4 roh | W4A4 had |
|---|---|---|---|---|---|---|---|---|---|
| gelu | 526848 | 2.139 | +0.084 | +0.035 | +0.020/+0.016 | +0.058/+0.016 | +0.034 | +0.121 | +0.072 |
| relu2 | 526848 | 2.227 | +0.060 | +0.036 | +0.021/+0.018 | +0.034/+0.017 | +0.035 | +0.095 | +0.076 |

Kurtosis (Excess) und Ausreisser:

| Akt | Ort | Kurtosis Mittel | Kurtosis Max (Lage) | Kurt. nach Hadamard | max/rms roh | max/rms had | Nullanteil |
|---|---|---|---|---|---|---|---|
| gelu | ein | 0.5 | 0.7 | -0.0 | 3.2 | 2.9 | nan |
| gelu | mitte | 38.5 | 51.4 | 1.8 | 9.3 | 3.1 | 0.13 |
| relu2 | ein | 0.6 | 0.7 | -0.0 | 3.1 | 2.8 | nan |
| relu2 | mitte | 733.0 | 1285.6 | 8.2 | 13.8 | 2.7 | 0.86 |

## Lesart (Erkundung, Einzelseed)
- **Quadrat erzeugt Ausreisser: ja.** Excess-Kurtosis der Zwischenaktivierung (T=3000): GELU 32, SwiGLU 193, z2 140, ReLU2 1096 (84 % exakte Nullen), |z| 1,6. FFN-Eingang (nach LayerNorm) bei allen ca. 0,3-1.
- **ReLU2 gewinnt: nur im Quantisierungsverlust, nicht absolut.** fp32 ist ReLU2 bei beiden Laengen ca. 0,09-0,15 bpc schlechter als GELU. Auch nach W4A4+Hadamard bleibt GELU vorn (T=8000: 2,211 gegen 2,303; T=3000: 2,550 gegen 2,696; SwiGLU 2,558).
- **Kurtosis sagt den Verlust nicht voraus:** roh, nur Mitte (T=3000): |z| +0,009 (Kurt. 1,6), ReLU2 +0,012 (1096), z2 +0,055 (140), SwiGLU +0,068 (193), GELU +0,080 (32). ReLU2 hat die groesste Kurtosis und fast den kleinsten Verlust (Ausreisser tragen das Signal, die vielen Nullen kosten nichts). Der naive Zusammenhang "Quadrat -> Ausreisser -> Verlust" stimmt nur zur Haelfte.
- **Langweiligste Erklaerung bestaetigt:** Mit Hadamard davor sind alle Aktivierungen gleich (A4 beide: +0,013 bis +0,021 bei T=3000, +0,035 bei T=8000 fuer GELU/ReLU2; max/rms nach Rotation 2,7-3,4, Kurtosis 1-9). Rotation glaettet die Quadrat-Ausreisser (S-001); eine Aktivierungswahl wird danach kaum noch gebraucht. Gewichte W4: ca. +0,02 (T=3000) bzw. +0,035 (T=8000) fuer alle gleich.
- **Laenge:** Der Verlust durch Quantisierung waechst mit T (Hadamard-A4: 0,016 -> 0,035 bei GELU). Das ist wichtiger als der Aktivierungsunterschied.
- **Grenzen:** Ein Seed; z2, |z|, SwiGLU bei T=8000 fehlen (Zeitbudget 40 min, Lauf abgebrochen); Unterschiede < 0,02 bpc nicht belastbar; nur Fake-Quant; Eingang der Aktivierungsquantisierung symmetrisch (verschwendet bei ReLU2 das Vorzeichenbit, asymmetrisch/unsigned waere fairer fuer ReLU2); Val-Bytes nur 32768 und nur die ersten; kein Gegner Clipping/Kanalskalen.

