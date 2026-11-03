# W4-Vortest: Rotation im Gradientenpfad (4-Bit-Training, Byte-LM)

Erkundung, **kein Befund**. Nur Tuning-Seed 950, Val (dev). Skript `w4_vortest.py`, Rohdaten `W4_vortest.json`.

## Aufbau
- Byte-LM wie Runde 2 (d=128, 4 Lagen, FFN 512, Adam 2e-3, Clip 1,0, Batch 8x128). Alle 16 `nn.Linear` der Blöcke in Fake-Quant (int4, Block 32, absmax, STE). Einbettung/Kopf/LayerNorm fp32.
- Vorwärts: Gewicht und Aktivierung 4 Bit (RTN). Rückwärts: dgrad mit unquantisiertem dy; Wgrad = Q(dy)^T Q(x), Kontraktion über 1024 Token in Blöcken von 32; dy mit stochastischem Runden, x RTN.
- Arme: `roh`; `had_fest` (Hadamard-32 über Tokenblöcke vor der Quantisierung); `had_zuf` (Hadamard-32 mal neue Zufallsvorzeichen je Schritt und Schicht); `clip` (ohne Rotation, MSE-optimale Clipping-Skala je Block aus {0,55; 0,75; 1}·absmax, wirkt auch im Vorwärtspass); `fp32`.
- Nicht gelaufen (Zeitbudget 40 min bei 2 Threads): **T=8000** (alle Arme) und der Zusatzarm `nur_vor` (nur Vorwärts-W4A4, Wgrad exakt). Im Skript vorhanden, Aufruf `python3 w4_vortest.py 8000` bzw. `... 3000 nur_vor`.

## Ergebnis T=3000 (Val-bpc, 1 Seed)
| Arm | Val-bpc | Δ zu fp32 | Divergenz | Verlustsprünge | Gradnorm Median / p99 / Max | rel. Wgrad-Fehler | ms/Schritt |
|---|---|---|---|---|---|---|---|
| fp32 | 2,5303 | – | nein | 0 | 0,72 / 2,20 / 3,93 | – | 68 |
| roh | 2,6357 | +0,105 | nein | 0 | 0,68 / 2,20 / 3,84 | 0,168 | 124 |
| had_fest | 2,6277 | +0,097 | nein | 0 | 0,67 / 2,20 / 3,84 | 0,158 | 140 |
| had_zuf | 2,6480 | +0,118 | nein | 0 | 0,67 / 2,20 / 3,84 | 0,136 | 142 |
| clip | 2,5885 | +0,058 | nein | 0 | 0,69 / 2,20 / 3,88 | 0,170 | 201 |

(Laufzeiten mit parallelem Fremdlauf, nur grobe Größenordnung; Rotation kostet im Fake-Quant-Code etwa +13 bis +15 % je Schritt gegen roh, Clip +60 %.)

## Lesart (vorsichtig)
- **Keine Instabilität:** Kein Arm divergiert, keine Verlustsprünge, Gradnormen praktisch identisch. Der Literaturbefund "quantisiertes Wgrad löst Divergenz aus" [arXiv:2605.09825] zeigt sich in diesem kleinen Modell mit int4 und stochastischem Runden nicht. Die Rotation hat hier nichts zu stabilisieren.
- **Rotation vs. roh:** Unterschiede -0,008 (fest) und +0,012 (zufällig) bpc liegen im Bereich des Einzelseed-Rauschens (nicht gemessen). Die Hadamard senkt den Wgrad-Fehler leicht (0,168 → 0,158 / 0,136), das bessert aber das bpc nicht erkennbar. Passt zur Aussage "bei kleinem Maßstab nichts" [arXiv:2509.25149].
- **Clip ist der beste 4-Bit-Arm (-0,047 bpc gegen roh).** Achtung Verfälschung: der Clip-Arm verbessert auch die Vorwärtsquantisierung von Gewicht und Aktivierung, nicht nur den Gradientenpfad. Der Gewinn kann also ganz aus dem Vorwärtspass kommen. Mit `nur_vor` und einem Arm "Clip nur im Rückwärtspass" wäre das zu trennen (nicht gelaufen).
- Der Hauptverlust (+0,10 bpc gegen fp32) stammt vermutlich aus W4A4 vorwärts, nicht aus dem Gradienten (unbelegt, `nur_vor` fehlt).
- Der relative Wgrad-Fehler ist nur an Schritten s mod 250 = 0 gemessen (Mittel über Schichten und diese Schritte).

## Empfehlung
W4 in dieser Form **keine Vorregistrierung**: kein Instabilitätssignal, Rotationseffekt nicht von Rauschen unterscheidbar, der langweilige Gegner (Clipping) liegt vorn. Nur wenn zuerst ein Regime mit echter Wgrad-Divergenz gefunden wird (3 Bit oder FP4/MXFP4-Format, Gradient ohne SR, längeres T, größeres Modell), lohnt ein neuer Vortest. Vorher offen: T=8000 und `nur_vor`.
