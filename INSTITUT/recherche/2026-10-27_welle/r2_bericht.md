# r2: Hourglass (Pooling x4) und LRU-Rekurrenz im Byte-LM (Rauchtest, 27.10.2026)

Erkundung, kein Befund. Tuning-Seeds 975-977, 1 Thread, ca. 20 min CPU.

## 1. Kurzfassung
- Ampel: **gelb** (Richtung stimmt, Aussagekraft gering: nur 190 Schritte, bpc 3,4-3,8, Modelle kaum über Unigramm-Niveau).
- Hourglass (HG) schlägt bei ~gleichen FLOPs das flache F3 (3,666 vs 3,703 bpc) und bei ~gleichen Parametern F6 (3,848) deutlich; FLOPs -26 % gegen F4, µs/Token -21 %.
- LRU statt Attention in Lage 0: 3,454 vs F4 3,739 bpc (-0,29), bei gleichen Parametern/FLOPs (-3 %). Größter Effekt, aber im Frühregime vermutlich Optimierungsvorteil (Rekurrenz = guter Prior für lokale Kontexte), nicht Endqualität.
- Kombination HG+LRU: 3,372 (bester Arm), FLOPs -30 % gegen F4, µs/Token -16 %.
- Offen: Ob das bei vollem Training (I-044/I-045-Lehre) bleibt. Lernrate nicht getunt.

## 2. Idee und Bezug
- (a) RG/Coarse-Graining: kausal gepoolte Kurz-Token (Mittel über 4, um k-1 nach rechts verschoben, Nawrot 2022 Hourglass; vgl. Funnel, MegaByte), mittlere Lagen auf T/4, Upsampling (Linear + wiederholen) + Skip. Mehta/Schwab 2014 nur Analogie.
- (b) Diagonale komplexe Rekurrenz nach LRU (Orvieto 2023): lambda = exp(-exp(nu) + i exp(theta)), Normierung gamma = sqrt(1-|lambda|^2), Init-Ring r in [0,9; 0,999], theta in [0, 2pi/10]. Lindblad-Bild: Rotation + Dämpfung. Zustand N = d = 128, per FFT-Faltung parallel.
- Alles klassisch (I-038), alle Quellen [Gedächtnis].

## 3. Aufbau
- d=128, FFN 4d (GELU), 4 Köpfe, Kontext 256, Batch 8, Adam lr 3e-3 (nicht getunt), Cosine, 190 Schritte (= 0,39 Mio. Trainingsbytes, ca. 3 % einer Epoche), 3 Seeds. Test: erste 32768 Bytes des Testsatzes UD_German-GSD.
- Arme: F4 flach 4 Lagen (Referenz); F3 flach 3 Lagen (FLOPs-gleich zu HG, -3 %); F6 flach 6 Lagen (Parameter-gleich zu HG, -1,3 %); HG = 1 Vorlage (voll) + 4 Mittellagen auf T/4 (Kontext 64) + 1 Nachlage; LRU = F4 mit LRU in Lage 0; HGLRU = HG mit LRU in der Vorlage.
- Kausalität geprüft (`kausal_test`: Änderung an Byte t verändert Ausgaben < t um 0 bzw. 5e-7 bei LRU).
- FLOPs/Token: Vorwärts, 2 je MAC, Attention kausal gemittelt (T+1)/2, LRU als ideale Rekurrenz (O(N)); Einbettung, LayerNorm, Aktivierung ungezählt. FFT-Faltung real teurer als gezählt.
- µs/Token: Vorwärts, Batch 8x256, 1 Thread, Minimum aus 5, nur Seed 975, Container mit 8 Agenten geteilt: ±20 % Rauschen.

| Arm | Test-bpc (Mittel ± sd, n=3) | Parameter | MFLOPs/Token | µs/Token |
|---|---|---|---|---|
| F4 flach | 3,739 ± 0,028 | 858 880 | 1,902 | 38,6 |
| F3 flach (FLOPs ~ HG) | 3,703 ± 0,010 | 660 608 | 1,443 | 26,4 |
| F6 flach (Param. ~ HG) | 3,848 ± 0,092 | 1 255 424 | 2,820 | 54,0 |
| HG | 3,666 ± 0,011 | 1 272 192 | 1,402 | 30,5 |
| LRU | 3,454 ± 0,015 | 858 624 | 1,837 | 36,5 |
| HG+LRU | 3,372 ± 0,008 | 1 271 936 | 1,337 | 32,6 |

Bytes: 4 B/Parameter (fp32) = Parameter x 4. Train-bpc (letzter Batch) 3,4-3,9, liegt nahe am Test-bpc.
Paarungen je Seed (975/976/977): HG-F3 = -0,058/-0,016/-0,037 (3 von 3 negativ); HG-F6 = -0,27/-0,19/-0,08; LRU-F4 = -0,30/-0,30/-0,26; HGLRU-HG = -0,29/-0,30/-0,29.

## 4. Deutung
- Langweiligste Erklärung: bei 190 Schritten gewinnt, was am schnellsten anfängt zu lernen. F6 verliert, weil tiefer = langsamer startet; HG hat nur 2 volle Lagen und kürzere Folge; LRU liefert lokale Kontextmischung (exponentielles Gedächtnis) als Prior, Attention muss das erst lernen. Der Vorsprung wird mit mehr Training sehr wahrscheinlich schrumpfen (I-044-v3, I-045-v3).
- HG-Vorsprung gegen F3 (-0,037 ± ~0,02) liegt im Rauschen der Größenordnung des Effekts; nur "3 von 3 Seeds gleiche Richtung". LRU- und HGLRU-Vorsprünge sind groß gegenüber sd (≥ 10 sd).
- Vorsicht: HG hat 48 % mehr Parameter als F4. FLOP-fair ist HG vs F3 (HG hat 3 % weniger FLOPs, also leicht im Vorteil), Parameter-fair ist HG vs F6.
- LRU spart kaum FLOPs (-3 %): Matmuls gleich teuer wie Attention-Projektionen; gespart wird nur der Score-Anteil bei Kontext 256. Bei längerem Kontext wächst der Vorteil (Attention O(T)/Token).
- Nicht untersucht: LRU in anderer Lage / mehreren Lagen, Lernrate je Arm (LRU-Parameter nu/theta nutzen dasselbe lr), Pooling-Faktor, Rekurrenz-Zustand N, reine Rekurrenz ohne Attention, gelernte statt gemittelte Pooling.

## 5. Vorschlag vorregistrierter Lauf (nur mit Maltes Zustimmung)
- Frage: Bleibt der Vorsprung bei vollem Training und gleichen FLOPs UND Parametern?
- Arme: F4, F3', F6' (Breite so angepasst, dass FLOPs bzw. Parameter exakt ±2 % passen), HG, LRU, HGLRU; zusätzlich "LRU in allen Lagen" und "F4 mit gleicher Lernrate je Arm getunt".
- Hypothesen: H1 HG ≤ F(FLOP-gleich) - 0,03 bpc; H2 LRU ≤ F4 - 0,03 bpc; H3 HGLRU ≤ min(HG, LRU) - 0,03; Zeit: HG schneller als F4 (µs/Token, gepaart).
- Kontrollen: Lernrate je Arm auf Validierung (3 Werte), Train-bpc, Negativkontrolle permutierte Bytes, Trainingsdauer ≥ 10 Epochen (>= 8000 Schritte bei Batch 8, besser mehr).
- Seeds: ≥ 10, Effektgröße 0,03 bpc. Laufzeit: 190 Schritte kosten ca. 50 s; 8000 Schritte ca. 35 min je Lauf x 8 Arme x 10 Seeds (+ Tuning) = ca. 50 h CPU, also Raven/Maltes PC (6 Kerne), nicht Container.

## 6. Skripte
- `INSTITUT/recherche/2026-10-27_welle/r2/hourglass_lru.py` (Modi `zaehlen`, `zeit`, `lauf <arm> <seed> <schritte> <lr>`)
- `INSTITUT/recherche/2026-10-27_welle/r2/ergebnisse.jsonl`, `lauf.log` (Rohdaten)
