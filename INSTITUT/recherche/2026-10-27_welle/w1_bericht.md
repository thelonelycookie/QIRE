# w1 — Tensor-Train-FFN (MPS) im Byte-Sprachmodell (Erkundung, Tuning-Seeds 900–904)

## 1. Kurzfassung — Ampel: ROT (für die Pareto-Frage bpc × FLOPs)
- Kurz trainiert (300 Schritte): alle Arme gleich (3,55–3,58 bpc); der FFN-Typ spielt dort kaum eine Rolle. TT2/TT4 liegen nominell vorn (3,563/3,550 vs. dicht 3,576), Abstand ~1 sd, kein Befund.
- Länger (1500 Schritte, nur 1 Seed): dicht 2,646 < TT16 2,722 < TT4 2,826 ≈ dicht-schmal Dn24 2,834.
- FLOPs: direkte TT-Kontraktion kostet bei diesen kleinen Modi MEHR FLOPs als Parameter (TT4: 4.864 Param., 329k FLOPs; TT16: 40k Param., 3,7M FLOPs > dicht 526k). TT spart Parameter/Speicher, keine FLOPs; zum Rechnen würde man W rekonstruieren (= dichte FLOPs).
- Bei gleichen FLOPs (TT2 116k, TT4 329k) sind Dn113/LR22 bzw. Dn320/LR64 praktisch gleich gut oder besser (kurz: 3,574/3,620 vs. TT2 3,563). Keine bessere Pareto-Front als Niedrigrang/dicht-schmal belegt.
- Einziger Lichtblick: TT hat bei gleichen Parametern (≤ 12k) in der Kurzläufen niedrigeren bpc als Niedrigrang (TT8 3,576 vs. LR4 3,663; ~6 sd). Dicht-schmal (Dn24 3,570) ist aber ebenso gut.

## 2. Idee, Bezug, Literatur
FFN-Matrizen 128→512→128 als Tensor-Train-Matrix (Matrixproduktoperator), Modi 128=4·4·8, 512=8·8·8, Bond-Dimension χ ∈ {2,4,8,16} (χ = „Verschränkung“ zwischen den Faktoren). Klassisch, quanten-inspiriert (I-038). Literatur [Gedächtnis]: Oseledets 2011 (TT-Zerlegung); Novikov et al. 2015 (TT-Schichten); Ma et al. 2019 (Tensorized Transformer); Stoudenmire/Schwab 2016 (MPS als Lernmodell).

## 3. Aufbau
Decoder-Transformer, Pre-LN, d=128, 2 Lagen, 4 Köpfe, Kontext 64, Batch 16, Adam lr 2e-3 (fest, NICHT getunt), Kosinus, UD_German-GSD Bytes. Metrik: Validierungs-bpc (dev, erste 40k Bytes), kein Test. Seeds 900–902 (300 Schritte), Seed 903 (1500 Schritte, je Arm 1 Lauf). Init der Kerne so skaliert, dass Var(W)=1/d_in.
Arme: D dicht 512; Dn<h> dicht schmal; LR<r> Niedrigrang; TT<χ>. Parameter-/FLOPs-Zählung: nur FFN, inkl. Bias, FLOPs = Multiply-Add×2 pro Token (TT: direkte Kontraktion).

| Arm | bpc Val 300 Schr. (n=3) | Param FFN | FLOPs FFN/Token |
|---|---|---|---|
| D | 3,576 ± 0,026 | 263.424 | 525.568 |
| TT2 | 3,563 ± 0,018 | 2.560 | 115.968 |
| TT4 | 3,550 ± 0,019 | 4.864 | 328.960 |
| TT8 | 3,576 ± 0,015 | 12.544 | 1.049.856 |
| TT16 | 3,576 ± 0,015 | 40.192 | 3.671.296 |
| LR4 | 3,663 ± 0,010 | 11.520 | 21.760 |
| LR15 | 3,646 ± 0,013 | 39.680 | 78.080 |
| LR22 | 3,620 ± 0,018 | 57.600 | 113.920 |
| LR64 | 3,591 ± 0,020 | 165.120 | 328.960 |
| Dn24 | 3,570 ± 0,009 | 12.592 | 24.880 |
| Dn77 | 3,575 ± 0,010 | 39.834 | 79.258 |
| Dn113 | 3,574 ± 0,015 | 58.338 | 116.194 |

Parameter gesamt: 176.896 (TT2) bis 437.760 (D); Bytes = 4 × Parameter (fp32). Langlauf (1500 Schritte, Seed 903): D 2,646; TT16 2,722; TT4 2,826; Dn24 2,834.
Rohdaten: w1/ergebnisse.jsonl, w1/lang.jsonl.

## 4. Deutung (inkl. langweiligster Erklärung)
- Langweilig: nach 300 Schritten (0,3 M Token) ist das Modell weit vom Plateau; FFN-Kapazität ist nicht der Engpass, daher gleichen sich alle Arme (nur LR leicht schlechter). Der TT-Vorsprung bei 300 Schritten ist plausibel Optimierungs-/Initialisierungseffekt, nicht Kapazität.
- Im Langlauf (n=1!) führt dicht klar; TT16 > TT4 ≈ Dn24: mehr Kapazität hilft, TT ist bei gleichen Parametern nicht klar besser als dicht-schmal (TT4 hat 2,6× weniger Parameter als Dn24, aber 13× mehr FLOPs, bpc fast gleich — Hinweis auf Parameter-Effizienz, nicht auf FLOPs-Effizienz).
- Einschränkungen: lr nicht getunt, ein Langlauf-Seed, sehr kurzes Training, FFN-Teil klein gegen Einbettung/Attention (Gesamtparameter wenig verschieden).
- Fazit zur Frage: TT bietet keine bessere Pareto-Front bpc×FLOPs als Niedrigrang/dicht-schmal; höchstens bpc×Parameter (Speicher) bei sehr kleinem FFN.

## 5. Vorschlag für einen vorregistrierten Lauf (nur falls Speicher/Parameter das Ziel ist)
- Hypothese H1: TT8 (12,5k FFN-Param.) ≤ Dn24 + 0,02 bpc (Nichtunterlegenheit) bei ≥ 3000 Schritten; H2: TT8 < LR4 (gleiche Parameter).
- Arme: D, TT4, TT8, TT16, Dn(param-gleich), LR(param-gleich), Dn(FLOPs-gleich); lr je Arm auf Tuning-Seed (Gitter 3 Werte); Kontrolle: permutierte Bytes.
- 5 Seeds, Effektgröße 0,02 bpc; Laufzeit Container ca. 7 Arme×5 Seeds×~6 min ≈ 3,5 h; Raven deutlich kürzer. Priorität niedrig.

## 6. Skripte
- /home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w1/tt_ffn.py (Modell, Zählung, Lauf)
- /home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w1/auswerte.py
- Ergebnisse: w1/ergebnisse.jsonl, w1/lang.jsonl
