# f5 -- Auswertung und Kalibrierung vorbereitet (Welle 3, 28.10.2026)

Nichts gerechnet außer dem Format-Test. Skript: `INSTITUT/erkundung/auswertung.py` (main-Checkout, kein git).

## Aufruf
- `python3 auswertung.py [--nur K01,K05,E2] [--out auswertung.md] [--min-n 3]` (aus `INSTITUT/erkundung/`), liest `roh/<KID>_s95?.json`.
- Formattest: `python3 auswertung.py --schnell --min-n 1` (liest die Schnell-JSONs, ausdrücklich nicht aussagekräftig).
- Ausgabe: Markdown. Teil 1 Hauptvergleich je Kandidat, Teil 2 Nebenvergleiche, Teil 3 Brier (gesamt, je Kandidat, je Denkwerkzeug).

## Rechnung
- Je Seed und Zelle: Gewinn = Idee gegen die **beste** der aufgeführten Kontrollen (konservativ), in bpc (kleiner = besser) oder Prozentpunkten. Zellen (Datensatz, Breite, Bitzahl, T) werden je Seed gemittelt, dann Mittel, sd (ddof 1), Vorzeichen-Zählung plus/n und t über die Seeds.
- Hauptvergleichs-Schlüssel stehen in der Tabelle `SPEC` im Skript (Idee, Kontrollen, JSON-Pfad, Richtung, Einheit, Schwelle). Fehlende Zellen oder Kontrollen werden übersprungen.
- **Ampel (fest):** grün = Mittel > 0,010 bpc bzw. 0,5 pp, plus ≥ 80 % der Seeds, t ≥ 2. Gelb = Mittel > 0 und plus ≥ 60 %. Rot sonst. Grau (keine Ampel, nicht im Brier) bei weniger als 3 Seeds.
- **Brier:** p = führende Prozentzahl der P-Spalte aus `konv_kandidaten.md`, Ereignis = grün. Zusätzlich Variante „vorsichtig“ (kleinste Prozentzahl der Zelle, z. B. K03 30 % statt 45 %), Referenzen p = 0,5 (0,25) und p = beobachtete Grün-Rate samt Skill-Wert. Gruppen: Denkwerkzeug k1 Morphologischer Kasten, k2 Fehlerbuch-Bergbau, k3 Umkehrung, k4 Zufallsimpuls, k5 Perspektiven; einmal nach erster Herkunft (jeder Kandidat einmal), einmal nach allen Herkünften (Mehrfachzählung).

## Hauptvergleiche (Kurzfassung der SPEC-Tabelle)
| ID | Idee | härteste Kontrollen | Maß |
|---|---|---|---|
| K01 | geteilt_k | klein, geteilt_o | bpc |
| K02 | geteilt_k | getrennt_B, geteilt_o | Acc |
| K03 | zufall_rot | skalar2_rot, skalar4_rot, e8_rot | Acc |
| K04 | zufall_s1024 | dicht512, zufall_o1024 | bpc |
| K05 | had@b (b ≤ 2,25) | roh, roh_kanal, pca | bpc |
| K06 | haar_alloc (alle, W1; 2 Bit) | haar_unif, orth_alloc, had | Acc |
| K07 | b4rot, b2rot | b4roh/b2roh, Adafactor | bpc |
| K08 | had | adam, vz, orth | bpc |
| K09 | z2bn32, z2bn8 | gelu, z2, z2ln | bpc |
| K10 | born_S4/S8 | unif, top, schmal | bpc |
| K11 | fly_top | fly_zufallsspalten, fly_zufallsprobe, dicht_flopgleich | Acc |
| K12 | bittree | voll, rang32 | bpc |
| K13 | W64_256 | f32, f64, f128, f256 (ctx128) | bpc |
| K14 | G_iso | D128_eqF, G_naiv, D64_eqF | bpc |
| K15 | bestes rot. Dropout (Byte-LM) | bestes Dropout, keins | bpc |
| K16 | S3skip | S1, S3skip_zufall, Dicht1_F, Dicht3skip | Acc |
| K17 | ff_z2 | ff_relu, mlp_flops_gleich | Acc |
| K18 | zwilling_had (Akzeptanz) | zwilling_roh, klein_dicht | Rate |
| K19 | z16 | perm, dicht_flops, dicht_param | Acc |
| K20 | hill | gelu, z2, relu6 | bpc |

E-Ergänzungen (IDs wie in `konv_kandidaten.md`, Dateien E01..E09): E1, E4, E10, E11 aus E01; E2 aus E02; E3 aus E03; E5 aus E05; E6 und E12 aus E06; E7 und E8 aus E07; E9 aus E09. Nebenvergleiche (nicht im Brier): K01 breit, K02 Schlüssel allein, K03 rm/hex, K04 Hadamard-R, K05 Haar-Alterung, K12 had32, K13 S32_256, K15 MLP, K17 Suche, K19 z64/streif/kron, E3 zeno/lin, E4 Blockgrößen.

## Test
- Mit den Schnell-JSONs laufen 28 von 32 Kandidaten durch (n = 1, nur Format). K13 bis K16 haben keine Schnell-JSON: Pfade dort mit synthetischen Dateien nach den Schreibstellen der Skripte geprüft, laufen.
- Mit den vorhandenen K17_s950/K19_s950 (nicht schnell) läuft der Normalpfad; bei `--min-n 3` korrekt „grau“.

## Zu beachten (Entscheidungen im Skript, bitte prüfen)
1. **Wahl der Idee bei Mehrfachvarianten fest vorab:** K03 zufall_rot (rm/hex nur Neben), K09 beide z2bn gemittelt, K10 S4 und S8 gemittelt, K19 z16, K13 W64_256. Anders wählen heißt, `SPEC` ändern, bevor Daten da sind.
2. **P-Spalte:** Die Zelle nennt teils zwei Zahlen. Führende Zahl ist die Hauptvariante; die vorsichtige Variante wird immer mitgerechnet. Zwei Zahlen für kurz/lang (K13, K14) zählen mit der kurzen.
3. **K07/K18/K08 sind Bytes- bzw. Zeitfragen, die Ampel misst nur bpc/Akzeptanz.** Bei K07 schlägt Adafactor (63 KB Zustand) schon im Schnelltest alles; die Ampel wird dort rot, ohne dass die Idee „falsch“ ist. K08 vergleicht ohne Rechenaufwand-Korrektur (Rotation kostet FLOPs), K18 ohne Zeitmessung.
4. **E1-Kontrollen** nur Arme mit `im_budget`; E5 nutzt den `had`-Test (Mittel über Wiederholungen) gegen roh und klein. E11 nimmt dither_rotv_sub; E10 vergleicht ent_rotv gegen ent_roh (Entropie-Bytes nicht geprüft, das Skript prüft Byte-Gleichheit nirgends, es vertraut den Skripten).
5. Schwellen 0,010 bpc und 0,5 pp sind Setzung, nicht aus Rauschen abgeleitet; nach den ersten fünf Seeds gegen die sd prüfen.
6. Brier bei ~30 Kandidaten und fünf Gruppen von 2 bis 13 Einträgen: nur grobe Tendenz.
7. Bekannter Vorbehalt bleibt: `quant_rows(bits=2)` ist ternär (1,58 Bit), gezählt 2 Bit; benachteiligt die Quantisierungsarme.
