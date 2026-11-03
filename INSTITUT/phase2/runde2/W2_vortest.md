# W2 Vortest (Phase 2, Runde 2): Rate-Güte-Kurve im Byte-LM

Status: **Erkundung, kein Befund.** Skript `w2_vortest.py`, Daten `W2_vortest.json` (Lauf ohne Kopierbeimischung) und `W2_vortest_mix.json` (Lauf mit Beimischung, T = 3000, unvollständig, nur zur Kopierprüfung). **Ein** Tuning-Seed (950), keine Streuung, Val-bpc (dev), nie Test.
Modell wie Z3 (d = 128, 4 Lagen, FFN 512), T ∈ {1000, 3000}, 786 432 quantisierte Gewichte; Bytes = Indizes + Skalen + Buch + Vorzeichen + fp16-Rest (112 128 B). 8-Bit-Skala: log-gleichmäßig je Matrix, Min/Max fp16 mitgezählt. Lloyd-Buch fp16 je Matrix.

## Ergebnis (Val-bpc; fp32: T=1000 3,540, T=3000 2,530)

| Arm | Indexbit | bpw | Bytes | bpc T=1000 | bpc T=3000 | SNR dB (T=3000) |
|---|---|---|---|---|---|---|
| Lloyd-1D, Zeilenskala | 2 | 2.10 | 318080 | 3.5532 | 2.8326 | 9.3 |
| Lloyd-1D, Zeilenskala | 3 | 3.10 | 416512 | 3.5423 | 2.6088 | 14.5 |
| Lloyd-1D, Zeilenskala | 4 | 4.10 | 515072 | 3.5412 | 2.5512 | 20.1 |
| Hadamard + Lloyd-1D | 2 | 2.10 | 318528 | 3.5461 | 2.7836 | 9.4 |
| Hadamard + Lloyd-1D | 3 | 3.10 | 416960 | 3.5419 | 2.5883 | 14.7 |
| Hadamard + Lloyd-1D | 4 | 4.10 | 515520 | 3.5398 | 2.5474 | 20.3 |
| g16 + 8-Bit-Skala + Lloyd | 1 | 1.50 | 259712 | 3.6444 | 4.1525 | 4.6 |
| g16 + 8-Bit-Skala + Lloyd | 2 | 2.50 | 358080 | 3.5492 | 2.7711 | 9.7 |
| g16 + 8-Bit-Skala + Lloyd | 3 | 3.50 | 456512 | 3.5426 | 2.5880 | 15.2 |
| g32 + 8-Bit-Skala + Lloyd | 2 | 2.25 | 333504 | 3.5508 | 2.8031 | 9.5 |
| g32 + 8-Bit-Skala + Lloyd | 3 | 3.25 | 431936 | 3.5415 | 2.5923 | 14.8 |
| Mittelmodell fp16 (d=48, f=192, eigenes Training) | 16 | – | 263232 | 3.5373 | 2.7887 | – |

## Was der Vortest zeigt
- **Skript läuft**, Wiederaufnahme je Punkt, Ergebnis-JSON je Punkt gespeichert.
- **T = 1000 ist unbrauchbar** (Plateau, bpc 3,54): alle Arme ≥ 2,1 bpw liegen innerhalb 0,015 bpc, nur g16 mit 1 Index-Bit fällt ab. Vorschlag: T ≥ 2000.
- T = 3000: Reihenfolge bei etwa 2,1–2,5 bpw: **g16-8-Bit (2,50 bpw) 2,771 < Hadamard+Lloyd (2,10) 2,784 < g32-8-Bit (2,25) 2,803 < Lloyd-Zeile (2,10) 2,833**. g16s8 liegt trotz +0,4 bpw nur 0,013 vor Hadamard+Lloyd; beim Interpolieren der Kurve (nicht gezeigt, nur ein Seed) wären Hadamard+Lloyd und g32s8 bei gleichen Bytes etwa gleich. Bei 3 Bit: 2,588–2,609, Unterschiede ≤ 0,02. Bei 4 Bit: 2,547–2,551, alles gleich. Das passt zu S-001 (Rotation ≈ billige Skala), ist aber ein Seed.
- **Mittelmodell ist konkurrenzfähig:** d = 48 fp16 (263 KB, kleiner als jeder 2-Bit-Punkt mit 318 KB) erreicht 2,789, besser als Lloyd-Zeile 2,833 und gleichauf mit g32s8/Hadamard bei 2 Bit. Der Regime-Effekt aus S-002 (Frühregime) ist sichtbar; Wiederholung bei größerem T entscheidend. Das Mittelmodell muss zwingend in die Reihe.
- **Aufgabenmetrik (Kopieren/Abruf) funktioniert in diesem Byte-LM nicht:** Kopier-Treffer in der zweiten Hälfte 0,03 = Kontrolle, auch beim T = 8000-Modell (aus w6_vortest2) und bei natürlichem Text (0,54 = 0,54); Kopierbeimischung (20 % der Batches) bis T = 8000 ändert nichts (bpc 4,70 = log2 26). Das Basismodell hat keine Induktion; die Probe kann Funktionsverlust nicht zeigen. Alternativen in W2_entwurf §5.
- g16 mit 1 Index-Bit (1,5 bpw) bricht bei T = 3000 ein (4,15), bei T = 1000 nicht (3,64): die Bruchkante hängt von T ab.
- Dekodierzeiten (µs, 250–660) sind **nicht verwertbar**: während des Laufs liefen zwei Prozesse, torch-eager, hohe Streuung. Für die Reihe auf ruhiger Maschine messen (w6_vortest2 `bench`).

## Laufzeit je Punkt (2 Threads, Container)
- Training d = 128: 56–70 ms/Schritt (1000 Schritte 70 s, 3000 Schritte 187 s, 8000 Schritte 458 s); d = 48: 22 ms/Schritt.
- Quantisieren 1,3–3,5 s, Val-Auswertung 1,0–1,2 s, Kopiertest < 1 s: **ca. 3–6 s je Punkt**. 23 Punkte + 2 Mittelmodelle + 2 Trainings: Lauf 450 s.
- Schätzung volle Reihe: siehe W2_entwurf §8 (≈ 60–70 min je Seed mit 4 Threads, 5 Seeds ≈ 5–6 h plus ≈ 1 h Codebau).

## Vorbehalte
Ein Tuning-Seed; bpc auf Val; Lernrate nicht je T getuned; Mittelmodell-Breite grob (16er-Schritte, Bytes 263 KB statt 417 KB des 3-Bit-Punkts, also nicht byte-gleich); GPTQ/MSB/E8/Orbit/Zusatzraten nicht gerechnet; Dekodierzeit unbrauchbar.
