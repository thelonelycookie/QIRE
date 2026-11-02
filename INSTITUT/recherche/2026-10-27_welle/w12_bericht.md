# w12: Ist „Sz verliert bei gleichen FLOPs“ (I-045-v4) fair gezählt? (27.10.2026)

Erkundung, kein Training, kein git, nichts in Agenda/Laborbuch. Quellen: `abteilungen/struktur_sprache.py` (`sz_flops_lage`, `_ffn_flops_modul`, `form_v4`), `abteilungen/struktur_gross.py` (`Fastfood`), w9 Punkte 1 und 4. Skript: `w12/bench_w12.py`, Log `w12/bench_w12.log`. Torch 2.14 CPU im Container nachinstalliert, 1 Thread.

## 1. Kurzfassung
- **Ampel gelb.** Die Zählung in v4 ist intern konsistent, aber „as implemented“: Sz zahlt dort für die Walsh-Transformation als dichte Matmuls (Ha·x·Hb). Ideal gezählt hat Sz **~9× weniger** Operationen (d=256: 43.264 statt 397.312; d=512: 94.720 statt 1.187.840 → 12,5×).
- Dichte FFN mit **ideal gleichen Operationen**: h = 42 (d=256), h = 46 (d=512) statt 388 / 580. Das ist aber mit ~21.500 / ~47.000 Parametern **4× / 4,4× mehr Parameter** als Sz (5.376 / 10.752). Dk (h=10) ist parametergleich, aber operationsärmer (4·d·10 = 10.240 bzw. 20.480).
- Epochen: T=16000 × 8×128 = 16,38 Mio Bytes auf 1.614.280 Trainingsbytes = **10,15 Epochen** (T=4000: 2,5; T=1000: 0,63). w9 bestätigt.
- Zeit: Die ideale Zählung wird in Software **nicht** Zeit. Batch 1, echte FWHT (vektorisiert in Torch): 965 µs (d=256) und 1192 µs (d=512), also 8× / 6,7× **langsamer** als dichte h=388/580 (33 / 179 µs) und auch langsamer als dichte h=1024/2048 (122 / 445 µs). Python-/Kernel-Aufrufe dominieren (≈ 40 Stufen-Aufrufe).
- Fazit: „Sz verliert bei gleichen FLOPs“ gilt nur für Matmul-Zählung. Bei idealer Zählung ist der Gegner schwächer (h≈45), der Satz wurde dort nie getestet. Aber ein Zeit-/Hardwarevorteil entsteht daraus ohne eigenen FWHT-Kernel nicht.

## 2. (a) Wie v4 zählt
`_ffn_flops_modul`: Linear 2·ein·aus; Fastfood je Block `2·(2·D·(a+b)) + 2·D` mit a·b = D (Faktoren ≈ √D). Das ist die Kostenform der Matmul-Implementierung (zwei kleine Matmuls je Transformation). Bias, Aktivierung z², Index-Auswahl werden **nicht** gezählt (bei Dichten auch nicht; Fehler ≪ 1 %). `form_v4` wählt Df-Breite = die ganze Zahl h mit 4·d·h nächst `sz_flops_lage(d)`: d=256 → h=388, d=512 → h=580 (nachgerechnet, wie im Auftrag). Lk: Rang 78 / 116 bei Breite 4d.

Fastfood-Aufbau im Code: Matrix a: d→4d = m=4 Blöcke der Größe D=d; Matrix b: 4d→d = ein Block D=4d, danach Auswahl von d Ausgängen (Permutation `idx`, 0 Flops, nur Gather).

## 3. Ideale Zählung (1 Addition/Multiplikation/Subtraktion = 1 Flop, wie v4)
Je Block: Diagonale D1 (D), FWHT (D·log₂D), Diagonale D2 (D), FWHT (D·log₂D) = 2D(1+log₂D).

| | d=256 | d=512 |
|---|---|---|
| Matrix a (4 Blöcke, D=d) | 4·2·256·9 = 18.432 | 4·2·512·10 = 40.960 |
| Matrix b (1 Block, D=4d) | 2·1024·11 = 22.528 | 2·2048·12 = 49.152 |
| Biase (4d + d) | 1.280 | 2.560 |
| Aktivierung z² (4d) | 1.024 | 2.048 |
| **ideal gesamt** | **43.264** | **94.720** |
| as implemented (v4) | 397.312 | 1.187.840 |
| Verhältnis | 9,2× | 12,5× |
| dicht mit gleichen idealen Ops (4·d·h) | h≈42 (43.008) | h≈46 (94.208) |
| Parameter dieser Dichten (2dh+h+d) | 21.802 | 47.150 |
| Parameter Sz (21·d) | 5.376 | 10.752 |

Hinweis: w9 schätzte für d=512 ≈ 90 k und h≈45; bestätigt (94.720, h≈46). Für d=256 nannte w9 h≈15–25; richtig ist **h≈42**. Die ideale Zählung ignoriert Speicherzugriffe (Permutation, FWHT-Stufen lesen/schreiben N Werte je Stufe).

## 4. (b) Epochen
Trainingssplit `daten()`: train 1.614.280 Bytes, val 73.794, test 99.487 (UD_German-GSD, Satztexte). Fenster zufällig (`randint`, überlappend), 1.024 Bytes je Schritt.
| T | Bytes gesehen | Epochen |
|---|---|---|
| 1000 | 1,02 Mio | 0,63 |
| 4000 | 4,10 Mio | 2,54 |
| 16000 | 16,38 Mio | **10,15** |
(Bei zufälligen Fenstern ist „Epoche“ nur als Datenmenge gemeint.) Die Parameterzahl des Modells (d=512, dicht: 4 Lagen × 8,4 Mio ≈ 34 Mio FFN-Parameter) gegen 1,6 Mio Bytes macht Überanpassung bei T=16000 plausibel. Die kleinere dichte FFN (h=388/580) profitiert davon als Regularisierung, nicht unbedingt „Rechenvorteil“. Das stützt w9 Punkt 1: Der Befund hängt am Datenregime, die Einzelzahl der Epochen (≈10) ist aber moderat; Überanpassung ist hier eine Hypothese, kein Messwert (Train-bpc wurde nicht verglichen).

## 5. (c) Mikrobenchmark (Batch 1, 1 Thread, fp32, Median 2000 Aufrufe nach 200 Aufwärm, nur eine FFN-Lage, z²-Aktivierung, `torch.no_grad`)
| d | Variante | µs | Gewichte |
|---|---|---|---|
| 256 | Sz as implemented (Repo, Matmul-WHT) | 158 | 21 kB |
| 256 | **Sz echte FWHT** | **965** | 21 kB |
| 256 | dicht h=388 (Matmul-FLOP-gleich) | 33 | 0,79 MB |
| 256 | dicht h=1024 (=4d) | 122 | 2,1 MB |
| 256 | dicht h=2048 | 236 | 4,2 MB |
| 512 | Sz as implemented | 274 | 43 kB |
| 512 | **Sz echte FWHT** | **1192** | 43 kB |
| 512 | dicht h=580 | 179 | 2,4 MB |
| 512 | dicht h=2048 (=4d) | 445–493 (zwei Messungen) | 8,4 MB |

FWHT-Korrektheit gegen Walsh-Matrix geprüft (max. Fehler 2·10⁻⁵ bzw. 4·10⁻⁵). Die FWHT-Stufen (`stack` je Stufe) sind Python-/Dispatch-gebunden: 2 Transformationen × log₂D Stufen × 2 Matrizen ≈ 40–46 Torch-Aufrufe à ~20 µs. Mit C/SIMD-Kernel (oder Fused CUDA) wären ~43–95 k Flops im µs-Bereich; das ist **nicht** gemessen. Bei Batch 1 liegen alle Gewichte (≤ 8,4 MB) noch im L2/L3-Cache, die Speicherbandbreiten-Nische (w9 Punkt 3) zeigt sich erst bei d ≥ 1024–4096; hier unbeantwortet. Matmul-Dichte ist durch BLAS 5–30× effizienter pro Flop als eine Torch-FWHT.

## 6. Deutung (inkl. langweiligster Erklärung)
- Fairness der Zählung: Die v4-Zählung ist **eine** zulässige Wahl (Flops der tatsächlichen Implementierung, die im Lauf auch die Zeit bestimmt), aber nicht die einzig mögliche. Mit idealen Operationen ist Sz um 9–12× billiger, der Gegner h≈42–46 hat dann aber 4× Parameter. Sz müsste bei gleichen Operationen **und** gleichen Parametern gegen Dk (h=10, 10.240 Ops) bestehen: Dk hat weniger Ops als Sz ideal (0,24× / 0,22×), Dk_z2 liegt nach w9/Laborbuch ohnehin vor Sz. Ein „Niedrigrang“ (Lk) mit ideal gleichen Ops: Rang r bei Breite 4d: 4·r·5d = 43.264 → r≈8,5 (d=256), 94.720 = 4·r·2560 → r≈9,2 (d=512). Parameter ≈ 2·r·5d: ~22 k bzw. ~47 k, ebenfalls ~4× Sz.
- Langweiligste Erklärung: Eine feste Mischung (H D H D) mit 2 Diagonalen je Block hat pro Parameter weniger Ausdruckskraft als ein Rang-r-Faktor; dass Sz bei gleicher Operationszahl nicht besser ist als eine kleine dichte FFN, ist wahrscheinlich auch ideal gezählt so. Die ideale Zählung verschiebt den Vergleich nur in ein Regime (h≈45), das bei Parametergleichheit nicht zu Sz passt.
- Ehrliche Schlussformel für die Arbeit: „Bei Matmul-gezählten Flops und gleichen Parametern verliert Sz; bei idealer Operationszahl ist Sz 9–12× billiger als die Matmul-Implementierung, die Gegner mit gleicher idealer Zahl haben aber 4× Parameter; ein Zeitgewinn ist ohne eigenen Kernel nicht vorhanden (Mikrobenchmark).“
- Grenzen: 1 Messlauf je Konfiguration (Median über 2000 Aufrufe, Streuung nicht berichtet); geteilter Container mit 4 Kernen und anderen Agenten; nur die FFN, nicht das ganze Modell; keine Backward-Zeit.

## 7. Vorschlag (kein Lauf nötig, ggf. billiger Nachtest nach Freigabe)
1. Rechnung: `ffn_flops_ideal` als zweite Spalte in der v4-Tabelle (Formel Abschnitt 3), beide Zählungen nebeneinander ausweisen.
2. Optional d=256, T=4000, Tuning-Seeds: Df_z2 mit h=42 (ideal-gleiche Ops) gegen Sz und Dk, 3 Seeds, ≈ 15 min je Lauf bei 0,2–0,3 s/Schritt; erwarteter Ausgang: Df(h=42) ≥ Dk (mehr Parameter) und evtl. besser als Sz. Würde Sz „bei gleichen idealen Ops“ besiegen, wäre Sz nur parameter-, nicht ops-konkurrenzfähig.
3. Zeit: C-/Triton-FWHT-Kernel bei d ≥ 1024 (w9 Punkt 3) statt weiterer Torch-Messung.

## 8. Skripte
`/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w12/bench_w12.py`, `.../w12/bench_w12.log`
