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

Kurtosis (Excess) und Ausreisser:

| Akt | Ort | Kurtosis Mittel | Kurtosis Max (Lage) | Kurt. nach Hadamard | max/rms roh | max/rms had | Nullanteil |
|---|---|---|---|---|---|---|---|
| gelu | ein | 0.5 | 0.7 | -0.0 | 3.2 | 2.9 | nan |
| gelu | mitte | 38.5 | 51.4 | 1.8 | 9.3 | 3.1 | 0.13 |

