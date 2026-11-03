# W6 Vortest 2 (Phase 2, Runde 1): PTQ eines Byte-LM mit Vektorquantisierern

Status: **Erkundung, kein Befund.** Skript `w6_vortest2.py`, Daten `W6_vortest2.json`. **Ein** Tuning-Seed (950), keine Mess-Seeds, keine Wiederholung, daher keine Streuung.
Modell wie Z3 (Byte-LM d=128, 4 Lagen, FFN 512, Kontext 128, UD_German-GSD), 8000 Schritte, lr 2e-3, 7 min. Quantisiert: alle 16 Linear-Gewichte (786432 Gewichte). Einbettung, Positionen, LayerNorm, Biases bleiben fp16/fp32 und sind in allen Armen gleich.
Maß: **Val-bpc** (dev, 65536 Bytes), NICHT Test (Test bleibt für vorregistrierte Läufe). fp32 2.1415; alle Gewichte auf fp16 gerundet 2.1415.

## Bit-Buchhaltung (Bits je Gewicht, Matrizen)
- bpw = (Indizes + Skalen fp16 + Codebuch fp16 + Hadamard-Vorzeichen 1 Bit je Eingang) / Gewichte. Zeilenskala = 16 Bit je Zeile (0,094 bpw bei diesen Formen), g16 = 1,0 bpw.
- Byte-gleich ist nur ungefähr erreicht: Gruppe 2,00 bpw, Lloyd/Hadamard-Arme 2,095-2,109 bpw (Abstand bis 0,11 bpw = 5 %). Bei 3 Bit: g16 3,00 gegen 3,10 der übrigen. Das hilft g16 minimal.
- E8-Buch: Regel, nur Skala D (16 Bit) gespeichert; die nachgebaute Regel liefert 65633 statt 65536 Punkte (Hash-Teilschale ungenau), also 16,0003 Bit/Block, vernachlässigt. Orbit-Buch: 64x8 fp16 = 8192 Bit einmal für das ganze Modell (0,0104 bpw). Lloyd-1D: Buch je Matrix.

## Budget 2 Bit Index

| Arm | Index-Bit | bpw gesamt | Val-bpc | Δ zu fp32 | Gewichts-SNR dB | Fit s | Kodier s | Dekod. µs (512x128) | Dekod.+Matvec µs |
|---|---|---|---|---|---|---|---|---|---|
| Hadamard + Orbit-Lloyd 8D (D8^3, G=1024, m=64) | 2 | 2.109 | 2.5163 | +0.3748 | 10.61 | 86.7 | 19.5 | 133 | 284 |
| Hadamard + E8-Nachbau (Heuristik-Kodierer) | 2 | 2.098 | 2.5285 | +0.3870 | 10.48 | 18.1 | 7.8 | 125 | 264 |
| Hadamard + E8-Nachbau (exakte Suche in Tabelle) | 2 | 2.098 | 2.5298 | +0.3883 | 10.49 | 17.6 | 24.0 | 136 | 272 |
| Hadamard + Orbit-Lloyd 4D (Pauli+H^n, G=64) | 2 | 2.099 | 2.6546 | +0.5131 | 9.78 | 0.4 | 0.1 | 121 | 205 |
| Hadamard + Lloyd-1D | 2 | 2.100 | 2.8169 | +0.6754 | 9.35 | 1.8 | 0.0 | 271 | 379 |
| Lloyd-1D (Zeilenskala, ohne Rotation) | 2 | 2.095 | 2.8472 | +0.7056 | 9.15 | 1.6 | 0.0 | 293 | 351 |
| g16 + Lloyd (Gruppenskala fp16) | 1 | 2.001 | 6.1630 | +4.0215 | 4.54 | 1.4 | 0.0 | 289 | 334 |
| g16 + uniform (Gruppenschritt fp16) | 1 | 2.000 | 6.3263 | +4.1848 | 4.56 | 0.0 | 0.1 | 124 | 144 |

## Budget 3 Bit Index

| Arm | Index-Bit | bpw gesamt | Val-bpc | Δ zu fp32 | Gewichts-SNR dB | Fit s | Kodier s | Dekod. µs (512x128) | Dekod.+Matvec µs |
|---|---|---|---|---|---|---|---|---|---|
| Hadamard + E8-Nachbau (Heuristik-Kodierer) | 3 | 3.098 | 2.2067 | +0.0651 | 15.92 | 17.6 | 7.4 | n/a | n/a |
| Hadamard + Orbit-Lloyd 4D (Pauli+H^n, G=64) | 3 | 3.104 | 2.2137 | +0.0721 | 15.90 | 11.5 | 1.6 | 144 | 243 |
| Hadamard + Lloyd-1D | 3 | 3.101 | 2.2558 | +0.1142 | 14.69 | 2.5 | 0.0 | 496 | 618 |
| Lloyd-1D (Zeilenskala, ohne Rotation) | 3 | 3.096 | 2.2768 | +0.1352 | 14.45 | 2.2 | 0.0 | 574 | 585 |
| g16 + uniform (Gruppenschritt fp16) | 2 | 3.000 | 2.6788 | +0.5373 | 9.91 | 0.0 | 0.1 | 108 | 134 |
| g16 + Lloyd (Gruppenskala fp16) | 2 | 3.001 | 2.7097 | +0.5681 | 9.62 | 1.7 | 0.0 | 300 | 284 |

## Einordnung (vorläufig, ehrlich)
- **Reihenfolge bei 2 Bit (gleiche Bytes ±5 %):** Orbit-8D 2,516 < E8-Nachbau 2,529 < Orbit-4D 2,655 < Hadamard+Lloyd-1D 2,817 < Lloyd-1D 2,847 (fp32 2,142). Orbit gegen E8: -0,012 bpc, +0,13 dB; das passt zu den +0,12 dB aus Vortest 1, ist aber ein Seed ohne Streuung und liegt in der Größenordnung der Kodierer-Unterschiede (Heuristik gegen exakt: 0,001 bpc). Orbit zahlt 0,010 bpw Buch (ca. 0,5 % der Bytes), E8 nicht.
- **Vektorquantisierung (8D) gegen Lloyd-1D mit Rotation:** -0,29 bpc bzw. +1,2 dB bei +0,009 bpw. Das ist der eigentliche Gewinn, er steckt in der 8D-Packung, nicht in der Quantenstruktur: E8 liegt gleichauf. Rotation allein bringt wenig (-0,03 bpc bei 2 Bit, -0,02 bei 3 Bit), wie in Z3.
- **3 Bit:** E8 (nur Heuristik, keine Tabelle) 2,207, Orbit-4D 2,214, Hadamard+Lloyd-1D 2,256, Lloyd-1D 2,277. Orbit-8D bei 3 Bit (2^24 Punkte) nicht gerechnet.
- **g16 ist hier kein starker Gegner bei gleichen Bytes:** fp16-Skala je 16 Gewichte kostet 1 bpw. Bei 2,0 bpw bleibt 1 Index-Bit, bpc 6,2 bis 6,3 (kaputt); bei 3,0 bpw mit 2 Bit 2,68 bis 2,71, schlechter als alle 3-Bit-Arme mit Zeilenskala. Vorbehalt: g16 mit 8-Bit-Skalen (0,5 bpw) oder kleinerem fp-Anteil wurde nicht probiert, das ist der faire stärkere g16-Gegner. Außerdem bei 2,5 Bit (g16 mit 1,5 Index-Bit) nicht möglich.
- **Dekodierzeit (CPU, 1 Thread, torch-eager, Matrix 512x128, bester von 5 Medianen):** fp32-Matvec ohne Dekodieren ca. 6 µs. Dekodieren (Lookup + Skala) 108 bis 300 µs, also 20- bis 50-mal die Rechnung selbst; das ist reiner Python/torch-Aufwand und kein Hardwaremaß. 8D-Tabellenlookup (E8 125, Orbit 133 µs, Tabelle 2 MB fp32, 65536 x 8) ist nicht langsamer als 1D-Lookup mit Entpacken (271 bis 293 µs bei 2 Bit); die 1D-Zeit ist vom Bit-Entpacken dominiert. Der Hadamard-Matvec kostet 65 bis 85 µs (fwht-Schleife in Python statt Fusion), das trifft alle Rotationsarme gleich. Auf echter Hardware wären Cache-Treffer der 2-MB-Tabelle (L2) und fused-Kerne entscheidend: nicht gemessen. 3-Bit-1D und 12-Bit-4D-Indizes liegen ungepackt (uint8/int64) vor, ihre Zeit ist ohne Entpacken; E8 bei 3 Bit hat keine Tabelle (16,7 Mio. Punkte).
- **Kodierzeit (4 Threads, Gesamtmodell):** Orbit-8D 87 s Training des Buchs + 20 s Brute-Force-Kodierung (65536 Punkte je Block); E8 18 s Skalensuche + 8 s Heuristik (24 s exakt); Lloyd-Arme unter 3 s. Orbit-Brute-Force skaliert nicht auf 3 Bit; ein schneller Orbit-Kodierer (nächstes Gruppenelement je Startvektor) wurde nicht gebaut.

## Vorbehalte
- Ein Tuning-Seed, ein trainiertes Modell, eine Rotation, ein Orbit-Lauf (Zufallsinit, 15 Iterationen, 20000 Trainingsblöcke aus den Modellgewichten selbst, also Buch auf denselben Gewichten gelernt, die es kodiert; 512 Zahlen gegen 786k Gewichte, geringes Überanpassungsrisiko, aber nicht null). Unterschiede unter ca. 0,02 bpc (Orbit gegen E8) sind nicht belegt.
- Val-bpc, nicht Test. Kein Feintuning nach der Quantisierung, kein GPTQ/Hessian, kein E8P (QuIP#), keine Kalibrierdaten.
- Bytes: Einbettung und Biases nicht mitgezählt (in allen Armen gleich); Hadamard-Vorzeichen als 1 Bit je Eingang gezählt (aus einem Seed ablesbar, also eher zu hoch).
- Arme nur auf ca. 5 % byte-gleich; der Abstand Orbit 2,109 gegen Lloyd 2,095 bpw begünstigt Orbit um etwa 0,01 bpw.
- Gewichte nach Rotation und Zeilennormierung etwa gaußisch; Orbit/E8 nutzen das aus. Aktivierungs-Ausreißer spielen bei reinem Gewichts-PTQ ohne Aktivierungsquantisierung keine Rolle.
- Clifford-Wirkung bleibt klassisch (I-038): Formulierung „quanten-inspiriert“.
