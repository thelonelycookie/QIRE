# w5 — Nachträgliche Kompression nach „Verschränkungsentropie“ (Erkundung, kein Befund)

## 1. Kurzfassung
- **Ampel: GELB.** Entropie-/Wasserfüll-Rangverteilung schlägt gleichmäßigen Rang **klein, aber konsistent** (+0,3 bis +1,7 pp nach Nachtraining), aber die einfache Kontrolle „globale Singulärwert-Schwelle“ ist ebenso gut und bei knappem Budget instabil.
- Gleicher Speicher, MNIST-16×16-MLP (5 Seeds 940–944, dicht 95,88 ± 0,35 %): bei 10 % der Parameter der drei Matrizen: gleich 94,52 ± 0,30, **Entropie 95,09 ± 0,18**, Wasserfüllen 95,09 ± 0,21, TT-SVD 91,93 ± 0,19, Pruning 92,83 ± 0,55 (alle mit 4 Ep. Nachtraining).
- TT-SVD und Pruning **ohne** Nachtraining kollabieren (12–40 %), mit Nachtraining liegen sie 1,7–3,2 pp unter Niedrigrang. TT lohnt hier nicht (FLOPs ~ 90× Niedrigrang, vgl. Tabelle).
- Der Effekt von Entropie über gleichmäßigem Rang ist die „Rangverteilung nach Perplexität“, im Kern klassische Spektralbudgetierung (AdaLoRA/Rang-Allokation-Verwandtschaft), kein spezifisch quantenartiger Gewinn. Sprachmodell (Byte-LM) **nicht** getestet (Budget).
- Vorschlag: nur lohnend als billiger Zusatz („Spektrum-Wasserfüllen“) in einer Kompressionsstudie; für sich allein keine Vorregistrierung.

## 2. Idee, Bezug, Literatur
- Jede Gewichtsmatrix W (m×n) wird als bipartiter Zustand aufgefasst: Singulärwerte = Schmidt-Koeffizienten, p_k = s_k²/Σs², Verschränkungsentropie S = −Σ p ln p, effektiver Rang exp(S). Hohe Entropie = Matrix braucht viel Rang; niedrige = leicht komprimierbar. Rangbudget je Matrix wird nach Entropie verteilt:
  - **entropie:** r_i ∝ exp(S_i), Skalierung so, dass Σ r_i(m_i+n_i) = Budget.
  - **wasser:** gemeinsame Schwelle λ auf den *normierten* Spektren p_k (Wasserfüllen über alle Matrizen).
  - **global_sv:** Schwelle auf absolutem s_k² (Kontrolle; Frobenius-optimal ohne Netzfunktion).
  - **gleich:** gleicher Rang je Matrix (gleiche Parameter).
  - **TT:** Matrix → Tensor (4,4,4,4)/(4,4,4,8)/(4,4,4,2) mit verschachtelten Indizes (m1,n1,m2,n2,…), TT-SVD wie DMRG, einheitliche Maximalrang-Wahl je Matrix bei Parameterbudget anteilig.
  - **Pruning:** Magnitude, lokal (Budget anteilig) und global (eine Schwelle). **Index-Bits mitgezählt** (4 B Wert + ⌈log₂ mn⌉ Bit Index je Eintrag), daher bei gleichem Byte-Budget weniger Einträge.
  - **scratch_gleich:** Kontrolle, Niedrigrang gleicher Parameter von Anfang an trainiert (kein Nachkomprimieren).
- Literatur [Gedächtnis]: Eckart–Young; Oseledets 2011 (TT-Zerlegung); Novikov et al. 2015 (TT-Schichten); Schollwöck (DMRG); Han et al. 2015 (Pruning); Hu et al. 2021 (LoRA), Zhang et al. 2023 (AdaLoRA: Rangverteilung nach Wichtigkeit); Roy & Vetterli 2007 (effektiver Rang). Keine WebSuche.

## 3. Aufbau und Ergebnisse
- Daten: MNIST auf 16×16 (gemeinsam.bilder), je Seed 8.000 Train / 2.000 Val (nur Test-Gesamtgenauigkeit hier berichtet, Val in der Rohdatei) / 5.000 Test. MLP 256-512-256-128-10, Adam 1e-3, 25 Epochen, Batch 128. Komprimiert: die drei ersten Matrizen (131.072 + 131.072 + 32.768 = **294.912** Parameter); Biases und Endschicht (1.290 Par.) bleiben dicht und stehen in allen Armen gleich.
- Budget f = 5, 10, 20 % der 294.912. Nachtraining: 4 Epochen, Adam 3e-4, trainiert Faktoren/Kerne/nicht-entfernte Gewichte + Biases + Endschicht (Maske fest). Hyperparameter **nicht** getuned (eine Einstellung, kein Gitter), daher keine Aussage über Optima.
- Seeds 940–944 (Tuning-Block), 5 Seeds, Mittel ± sd (Test %, n=5). FLOPs = MAC der drei Matrizen pro Beispiel (Pruning theoretisch, unstrukturiert; dicht = 294.912). Bytes = float32 (4 B/Wert) plus Index bei Pruning.

Spektren (mittlere Entropie nach Training): S/S_max = 0,78 (512×256), **0,62** (256×512), 0,77 (128×256); effektive Ränge 74 / 31 / 41; 90 % der Energie brauchen Rang 115 / 78 / 62. Die mittlere Matrix ist am „wenigsten verschränkt“ und bekommt weniger Rang.

| f | Arm | ohne NT | mit NT | Par. | Bytes | FLOPs |
|---|---|---|---|---|---|---|
| 0,05 | gleich | 73,53 ± 5,30 | 90,69 ± 0,58 | 13.440 | 53.760 | 13.440 |
| 0,05 | **entropie** | 65,81 ± 3,92 | **92,36 ± 0,69** | 14.592 | 58.368 | 14.592 |
| 0,05 | wasser | 77,09 ± 4,10 | 90,24 ± 0,81 | 14.438 | 57.753 | 14.438 |
| 0,05 | global_sv | 23,82 ± 10,30 | 44,17 ± 23,36 | 14.361 | 57.446 | 14.361 |
| 0,05 | TT-SVD | 11,80 ± 1,88 | 86,10 ± 1,37 | 13.360 | 53.440 | 1.214.464 |
| 0,05 | Pruning lokal | 25,59 ± 3,50 | 81,67 ± 1,67 | 9.673 | 58.968 | 9.673 |
| 0,05 | Pruning global | 24,60 ± 4,58 | 77,35 ± 2,55 | 9.650 | 58.976 | 9.650 |
| 0,05 | scratch (Rang gleich) | 89,10 ± 0,58 | – | 13.440 | 53.760 | 13.440 |
| 0,10 | gleich | 92,29 ± 1,20 | 94,52 ± 0,30 | 28.800 | 115.200 | 28.800 |
| 0,10 | **entropie** | 94,20 ± 0,45 | **95,09 ± 0,18** | 29.030 | 116.121 | 29.030 |
| 0,10 | wasser | 94,13 ± 0,65 | 95,09 ± 0,21 | 29.184 | 116.736 | 29.184 |
| 0,10 | global_sv | 90,35 ± 4,85 | 95,17 ± 0,21 | 29.030 | 116.121 | 29.030 |
| 0,10 | TT-SVD | 14,22 ± 4,81 | 91,93 ± 0,19 | 27.936 | 111.744 | 2.541.568 |
| 0,10 | Pruning lokal | 41,30 ± 5,03 | 92,85 ± 0,55 | 19.348 | 117.949 | 19.348 |
| 0,10 | Pruning global | 40,12 ± 7,55 | 92,83 ± 0,55 | 19.320 | 117.960 | 19.320 |
| 0,10 | scratch | 92,56 ± 0,41 | – | 28.800 | 115.200 | 28.800 |
| 0,20 | gleich | 95,03 ± 0,39 | 95,42 ± 0,15 | 57.600 | 230.400 | 57.600 |
| 0,20 | **entropie** | 95,37 ± 0,52 | 95,69 ± 0,19 | 58.598 | 234.393 | 58.598 |
| 0,20 | wasser | 95,23 ± 0,49 | 95,63 ± 0,16 | 58.675 | 234.700 | 58.675 |
| 0,20 | global_sv | 95,51 ± 0,35 | 95,74 ± 0,12 | 58.444 | 233.779 | 58.444 |
| 0,20 | TT-SVD | 33,74 ± 4,65 | 94,61 ± 0,18 | 58.176 | 232.704 | 5.312.512 |
| 0,20 | Pruning lokal | 69,66 ± 4,65 | 95,42 ± 0,20 | 38.699 | 235.916 | 38.699 |
| 0,20 | Pruning global | 66,84 ± 3,95 | 95,53 ± 0,13 | 38.669 | 235.926 | 38.669 |
| 0,20 | scratch | 93,78 ± 0,16 | – | 57.600 | 230.400 | 57.600 |
Dicht (Referenz): 95,88 ± 0,35 %, 294.912 Par., 1.179.648 B, 294.912 MAC.

Gepaarte Differenzen (mit NT, pp, Mittel ± sd über 5 Seeds): entropie − gleich: +1,68 ± 0,56 (f=0,05), +0,58 ± 0,24 (0,1), +0,26 ± 0,13 (0,2). wasser − gleich: −0,44 ± 0,65 / +0,58 ± 0,13 / +0,21 ± 0,11. global_sv − gleich: −46,5 ± 23 (Kollaps einer Schicht auf Rang 2) / +0,66 ± 0,11 / +0,32 ± 0,17. Niedrigrang gleich − Pruning global: +13,3 / +1,7 / −0,1.
Hinweis: Entropie/Wasser/global_sv haben bis ~9 % **mehr Parameter** als gleich (Rundung der Ränge, Rangbudget je Matrix ganzzahlig); bei f=0,05 sind das +8,6 % (entropie), +7,4 % (wasser). Ein Teil des Vorsprungs bei f=0,05 ist damit Parameterüberhang, nicht Verteilung. Das Pruning hat 33 % weniger Einträge (Indexkosten).

## 4. Deutung, langweiligste Erklärung
- Die langweiligste Erklärung: Die Schichten sind verschieden breit/verschieden gut komprimierbar; ein Budget proportional zu irgendeinem Komplexitätsmaß der Schicht (Perplexität, Energie-Schwelle) ist besser als „gleich“. Das ist Rang-Allokation (AdaLoRA-artig), „Entropie“ ist nur ein Gewicht. Eine Frobenius-Schwelle (global_sv) leistet dasselbe, bei f≥0,1 gleich gut, bei f=0,05 kollabiert sie (eine Schicht auf Rang 2), weil sie die Funktion der Schicht ignoriert. Die Entropie-Regel ist *robuster* (Rang proportional ≥ 5), aber das ist eine Eigenschaft der Proportionalität, nicht der Quantenanalogie.
- Nachtraining ist entscheidend: Ohne es ist TT/Pruning unbrauchbar; mit 4 Epochen nähern sich alle bei f=0,2 an (Pruning global 95,53 vs. entropie 95,69, Differenz ~0,2 pp ≈ 1 sd). Niedrigrang gleicher Parameter von Anfang an (scratch) ist ohne Nachtraining-Vergleich schlechter als nachkomprimiert+NT, was für Nachkomprimierung spricht, aber scratch hatte nur 25 Ep. und dasselbe lr-Paar (kein Tuning), also kein fairer Gleichstand.
- TT nutzt hier wenig: gelernte Matrizen haben keine Tensor-Hierarchie in der Indexfaktorisierung (4×4×4×4), SVD-Rang in Tensorform ≈ Strukturannahme, die nicht passt; und FLOPs sind bei Standardkontraktion 40–90× höher als Niedrigrang. TT ist hier schlechter bei Parametern *und* FLOPs.
- Pruning: bei gleichen Bytes hat es durch Indexbits 33 % weniger Einträge; das reicht bei f=0,2 trotzdem für Gleichstand (Netz ist überparametrisiert), bei f≤0,1 verliert es 2–11 pp.
- Gesamtbefund: Alles liegt bei f=0,2 innerhalb ~1 pp des dichten Netzes; der Spielraum für Verteilungsstrategien ist klein. Die 294.912 Parameter dominiert eine MNIST-Aufgabe, die ein deutlich kleineres MLP lösen würde (kein Kompressionsdruck-Test; ein von Anfang an schmales dichtes MLP gleicher Parameter fehlt als Kontrolle!).
- Einschränkungen: nur MNIST-MLP, ein Nachtrainings-Setup ohne Tuning, 5 Seeds, Tuning-Seeds. Kein Sprachmodell, keine Fashion-MNIST. FLOPs der TT als obere Schranke der Standardkontraktion. Unstrukturiertes Pruning zählt theoretische, nicht echte Beschleunigung. Wirkung klassisch (I-038).

## 5. Vorschlag für einen vorregistrierten Lauf (nur falls ein Gesamtpaket „Kompression“ sinnvoll ist)
- **Nicht allein**; allenfalls als Arm in einer Kompressionsstudie. Wenn doch:
  - H1: Entropie-Verteilung − gleicher Rang > +0,3 pp (nach NT, f=0,1) bei exakt gleicher Parameterzahl (Ränge so runden, dass Σ gleich) und gegen *schmales dichtes MLP gleicher Parameter*, gegen global_sv.
  - Arme: gleich, entropie, wasser, global_sv, schmal-dicht (Von-Anfang-an), Pruning global, optional TT (nur Fehlerrate, verworfen wegen FLOPs). Datensätze: MNIST, Fashion, Byte-LM (bpc), dort Matrizen d×4d, d×d.
  - Kontrollen: gleiche Parameter (exakt), gleiche NT-Schritte, lr-Gitter je Arm auf Tuning-Seeds, Nachtraining auch für scratch-Arm verlängert.
  - Seeds: ≥10 Mess-Seeds; erwartete Effektgröße 0,3–0,6 pp bei sd(Differenz) ≈ 0,2 → n=10 reicht.
  - Laufzeit: MNIST-Teil ≈ 2 min/Seed im Container (1 Thread, hier gemessen ≈ 10 min für 5 Seeds mit 7 Methoden × 3 Budgets); Byte-LM-Teil auf Raven/Malte-PC.

## 6. Skripte
- `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w5/w5_kompression.py` (Lauf, ≈ 10 min CPU, 1 Thread)
- `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w5/auswertung.py` (liest `roh.json`, schreibt `auswertung.txt`)
- Rohdaten: `…/w5/roh.json`, Log `…/w5/lauf.log`
