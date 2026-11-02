# W5-Vortest: Bitflips auf gespeicherten Bits (Phase 2, Runde 1)

**Status: Erkundung, kein Befund.** Nur Tuning-Seeds 900/901 (je 1 Modell, 20 Flipmuster je Zelle, also n=40 je Zelle). MNIST 16x16 (`gemeinsam.bilder`), MLP 256-256-256-10, 1500 Adam-Schritte, Test 5000 Bilder. Laufzeit 5.8 min.

Fehlermodell: jedes gespeicherte Bit kippt unabhängig mit p: Gewichtscodes (b Bit), fp16-Skalen UND fp16-Bias. fp16-Werte, die durch Flips NaN/Inf werden, werden auf 0 bzw. ±65504 gesetzt (keine NaN-Ausbreitung; das ist günstig für Arme mit vielen Skalen). Modus `alle` = Hauptergebnis; `nur_codes` / `nur_skalen` trennen die Flipfläche (Idee 2).

Arme: d_roh = eine Skala je Schicht; b_clip_zeile = Zeilenskala, MSE-optimales Clip (33 Faktoren); a_had_zeile_opt = Hadamard (Zufallsvorzeichen aus Seed, 0 B) + Zeilenskala mit optimalem Clip; a0 / b0 = wie a / b, aber ohne Clip (absmax); c_g16 = Gruppenskala fp16 je 16 Gewichte (1 Bit/Gewicht Skala, also deutlich mehr Bytes); c_g16_bytegleich = g16 mit b-1 Bit Codes (≈ gleiche Bytes). Mittpunkt-Gitter, binäre Codes. Hinweis: a und b enthalten beide Clipping; die Rotation allein ist a vs. b, nicht a vs. d. Das Gitter für 2 Bit hat 4 Stufen, g16_bytegleich bei 2 Bit nur 2 Stufen (±).

## Modus `alle`, 2 Bit

| Arm | Bytes | acc ohne Flip | p=1e-05 | p=0.0001 | p=0.001 | p=0.01 |
|---|---|---|---|---|---|---|
| d_roh | 34458 | 0.453 | 0.453 ± 0.069 | 0.455 ± 0.071 | 0.390 ± 0.136 | 0.181 ± 0.144 |
| b_clip_zeile | 35496 | 0.916 | 0.902 ± 0.092 | 0.865 ± 0.171 | 0.574 ± 0.381 | 0.107 ± 0.024 |
| a_had_zeile_opt | 35496 | 0.897 | 0.882 ± 0.096 | 0.840 ± 0.178 | 0.557 ± 0.359 | 0.109 ± 0.030 |
| a0_had_zeile_abs | 35496 | 0.788 | 0.772 ± 0.095 | 0.736 ± 0.164 | 0.504 ± 0.306 | 0.112 ± 0.024 |
| b0_zeile_abs | 35496 | 0.657 | 0.649 ± 0.064 | 0.622 ± 0.118 | 0.420 ± 0.257 | 0.107 ± 0.025 |
| c_g16 | 51156 | 0.875 | 0.828 ± 0.168 | 0.577 ± 0.324 | 0.135 ± 0.045 | 0.111 ± 0.032 |
| c_g16_bytegleich | 34452 | 0.590 | 0.563 ± 0.147 | 0.407 ± 0.220 | 0.114 ± 0.029 | 0.101 ± 0.020 |

## Modus `alle`, 4 Bit

| Arm | Bytes | acc ohne Flip | p=1e-05 | p=0.0001 | p=0.001 | p=0.01 |
|---|---|---|---|---|---|---|
| d_roh | 67866 | 0.835 | 0.818 ± 0.103 | 0.786 ± 0.171 | 0.631 ± 0.268 | 0.120 ± 0.065 |
| b_clip_zeile | 68904 | 0.956 | 0.935 ± 0.133 | 0.885 ± 0.216 | 0.446 ± 0.377 | 0.105 ± 0.026 |
| a_had_zeile_opt | 68904 | 0.958 | 0.936 ± 0.133 | 0.886 ± 0.218 | 0.450 ± 0.378 | 0.116 ± 0.032 |
| a0_had_zeile_abs | 68904 | 0.956 | 0.934 ± 0.133 | 0.885 ± 0.216 | 0.450 ± 0.376 | 0.111 ± 0.025 |
| b0_zeile_abs | 68904 | 0.950 | 0.929 ± 0.132 | 0.879 ± 0.213 | 0.443 ± 0.370 | 0.103 ± 0.030 |
| c_g16 | 84564 | 0.957 | 0.935 ± 0.134 | 0.611 ± 0.357 | 0.118 ± 0.035 | 0.113 ± 0.029 |
| c_g16_bytegleich | 67860 | 0.953 | 0.871 ± 0.222 | 0.574 ± 0.350 | 0.128 ± 0.039 | 0.099 ± 0.031 |

## Modus `nur_codes`, 2 Bit

| Arm | Bytes | acc ohne Flip | p=1e-05 | p=0.0001 | p=0.001 | p=0.01 |
|---|---|---|---|---|---|---|
| d_roh | 34458 | 0.453 | 0.453 ± 0.070 | 0.452 ± 0.070 | 0.450 ± 0.069 | 0.442 ± 0.080 |
| b_clip_zeile | 35496 | 0.916 | 0.916 ± 0.016 | 0.916 ± 0.016 | 0.915 ± 0.015 | 0.907 ± 0.019 |
| a_had_zeile_opt | 35496 | 0.897 | 0.897 ± 0.002 | 0.896 ± 0.005 | 0.895 ± 0.007 | 0.869 ± 0.032 |
| a0_had_zeile_abs | 35496 | 0.788 | 0.787 ± 0.024 | 0.788 ± 0.025 | 0.780 ± 0.034 | 0.693 ± 0.078 |
| b0_zeile_abs | 35496 | 0.657 | 0.657 ± 0.028 | 0.654 ± 0.031 | 0.655 ± 0.030 | 0.615 ± 0.057 |
| c_g16 | 51156 | 0.875 | 0.876 ± 0.016 | 0.873 ± 0.019 | 0.875 ± 0.020 | 0.832 ± 0.039 |
| c_g16_bytegleich | 34452 | 0.590 | 0.590 ± 0.124 | 0.589 ± 0.125 | 0.589 ± 0.120 | 0.581 ± 0.112 |

## Modus `nur_codes`, 4 Bit

| Arm | Bytes | acc ohne Flip | p=1e-05 | p=0.0001 | p=0.001 | p=0.01 |
|---|---|---|---|---|---|---|
| d_roh | 67866 | 0.835 | 0.835 ± 0.030 | 0.830 ± 0.032 | 0.792 ± 0.073 | 0.485 ± 0.146 |
| b_clip_zeile | 68904 | 0.956 | 0.956 ± 0.001 | 0.956 ± 0.001 | 0.953 ± 0.005 | 0.895 ± 0.035 |
| a_had_zeile_opt | 68904 | 0.958 | 0.957 ± 0.000 | 0.957 ± 0.001 | 0.956 ± 0.002 | 0.931 ± 0.013 |
| a0_had_zeile_abs | 68904 | 0.956 | 0.956 ± 0.001 | 0.955 ± 0.001 | 0.953 ± 0.002 | 0.917 ± 0.020 |
| b0_zeile_abs | 68904 | 0.950 | 0.950 ± 0.003 | 0.950 ± 0.004 | 0.944 ± 0.007 | 0.846 ± 0.043 |
| c_g16 | 84564 | 0.957 | 0.957 ± 0.001 | 0.957 ± 0.001 | 0.955 ± 0.002 | 0.935 ± 0.013 |
| c_g16_bytegleich | 67860 | 0.953 | 0.953 ± 0.001 | 0.953 ± 0.001 | 0.951 ± 0.002 | 0.929 ± 0.012 |

## Modus `nur_skalen`, 2 Bit

| Arm | Bytes | acc ohne Flip | p=1e-05 | p=0.0001 | p=0.001 | p=0.01 |
|---|---|---|---|---|---|---|
| d_roh | 34458 | 0.453 | 0.453 ± 0.070 | 0.436 ± 0.102 | 0.399 ± 0.131 | 0.149 ± 0.124 |
| b_clip_zeile | 35496 | 0.916 | 0.916 ± 0.016 | 0.772 ± 0.287 | 0.377 ± 0.345 | 0.119 ± 0.027 |
| a_had_zeile_opt | 35496 | 0.897 | 0.896 ± 0.005 | 0.767 ± 0.266 | 0.383 ± 0.347 | 0.123 ± 0.036 |
| a0_had_zeile_abs | 35496 | 0.788 | 0.787 ± 0.024 | 0.672 ± 0.235 | 0.367 ± 0.302 | 0.107 ± 0.026 |
| b0_zeile_abs | 35496 | 0.657 | 0.657 ± 0.028 | 0.557 ± 0.202 | 0.300 ± 0.237 | 0.106 ± 0.024 |
| c_g16 | 51156 | 0.875 | 0.816 ± 0.184 | 0.486 ± 0.330 | 0.117 ± 0.032 | 0.114 ± 0.028 |
| c_g16_bytegleich | 34452 | 0.590 | 0.559 ± 0.167 | 0.362 ± 0.210 | 0.113 ± 0.038 | 0.112 ± 0.025 |

## Modus `nur_skalen`, 4 Bit

| Arm | Bytes | acc ohne Flip | p=1e-05 | p=0.0001 | p=0.001 | p=0.01 |
|---|---|---|---|---|---|---|
| d_roh | 67866 | 0.835 | 0.835 ± 0.029 | 0.817 ± 0.116 | 0.653 ± 0.303 | 0.143 ± 0.139 |
| b_clip_zeile | 68904 | 0.956 | 0.955 ± 0.008 | 0.869 ± 0.244 | 0.564 ± 0.375 | 0.112 ± 0.031 |
| a_had_zeile_opt | 68904 | 0.958 | 0.957 ± 0.003 | 0.868 ± 0.251 | 0.570 ± 0.370 | 0.114 ± 0.030 |
| a0_had_zeile_abs | 68904 | 0.956 | 0.955 ± 0.005 | 0.865 ± 0.254 | 0.575 ± 0.362 | 0.114 ± 0.030 |
| b0_zeile_abs | 68904 | 0.950 | 0.950 ± 0.003 | 0.860 ± 0.252 | 0.564 ± 0.370 | 0.109 ± 0.029 |
| c_g16 | 84564 | 0.957 | 0.936 ± 0.130 | 0.645 ± 0.383 | 0.132 ± 0.055 | 0.104 ± 0.028 |
| c_g16_bytegleich | 67860 | 0.953 | 0.932 ± 0.127 | 0.660 ± 0.373 | 0.128 ± 0.059 | 0.108 ± 0.027 |

fp32-Genauigkeit (Mittel über 2 Seeds): 0.959.

Bytes pro Modell (Seed 900), inkl. Skalen und fp16-Bias; Hadamard-Vorzeichen aus Seed erzeugt = 0 B (als gespeicherter Vektor wären es n/8 = 32 B je Schicht).

Grenzen: 2 Modelle, 1 Datensatz, kleines MLP, kein RandBET-Training, keine Signifikanztests; Std über alle 40 Flipmuster (Modell+Muster gemischt). Nur Vortest; Aussagen über Rotation vs. Clipping bleiben Hypothese.