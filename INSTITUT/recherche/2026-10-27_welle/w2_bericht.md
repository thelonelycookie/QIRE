# w2 — Gelernte Monarch-Matrizen in der FFN (Erkundung, Seeds 910–914, kein Befund)

## 1. Kurzfassung — Ampel: GELB (eher rot für Vorregistrierung)
- Bei gleichen FLOPs und Gewichten schlägt Monarch **Niedrigrang** (Test-bpc 3,066 vs 3,112; Δ −0,046, 3/3 Seeds), aber **nicht schmal dicht** (3,059; Δ +0,007, Vorzeichen gemischt).
- Dicht breit (4× FLOPs) ist klar besser (2,939): der Preis der Struktur bleibt.
- z² vs GELU: bei schmal dicht −0,025 (3/3 Seeds besser), bei Monarch −0,017 (nicht einheitlich).
- Grenzen: nur 1200 Schritte (noch fallende Kurven), 3 Seeds, d=128, Lernrate am Gitterrand (1e-3). Monarch ist auf CPU ~2× langsamer (einsum).

## 2. Idee, Bezug, Literatur
Monarch = Produkt zweier blockdiagonaler Matrizen mit Permutation dazwischen (lokale Gatterlagen + SWAP/Mischung wie in einer Schaltung), FLOPs O(N√N); Dao et al. 2022 [Gedächtnis]. Gegenstück zum festen Fastfood (I-045): Blöcke werden gelernt. Wirkung klassisch (I-038); „quanten-inspiriert“.

## 3. Aufbau und Tabelle
Byte-LM (UD_German-GSD, Daten/Bpc-Auswertung aus `abteilungen/struktur_sprache.py`), d=128, 2 Lagen, 4 Köpfe, Kontext 64, Batch 16×64, FFN-Breite 512 (nur Referenz), Adam, Kosinus, 1200 Schritte, 1 Thread. Lernrate je Arm aus {1e-3,3e-3,6e-3} auf Val (Tuning-Seed 913, 500 Schritte; Wahl: 1e-3, lr19 3e-3). Test = fester Testsatz, 3 Messseeds 910–912. Monarch 128→512: 16 Blöcke 8→32, dann 32 Blöcke 16→16; 512→128 analog (12.288 Gewichte je Matrix).

| Arm | FFN-Gewichte | FFN-FLOPs/Token (beide Lagen) | Test-bpc Mittel ± sd |
|---|---|---|---|
| dicht512 (Referenz, 5,3× FLOPs) | 262.144 | 524.288 | 2,939 ± 0,023 |
| dicht96 (schmal dicht) | 49.152 | 98.304 | 3,059 ± 0,012 |
| lr19 (Niedrigrang r=19) | 48.640 | 97.280 | 3,112 ± 0,027 |
| mon (Monarch, GELU) | 49.152 | 98.304 | 3,066 ± 0,003 |
| mon_z2 (Monarch, z²) | 49.152 | 98.304 | 3,048 ± 0,022 |
| dicht96_z2 | 49.152 | 98.304 | 3,034 ± 0,014 |

Gesamtparameter ≈ 224k (Ref. 438k); Bytes = 4·Parameter (float32). Kurve (mittl. Val-bpc bei Schritt 150/300/…/1200): dicht96 3,63→3,47→3,29→3,15→3,05→2,98→2,95→2,94; mon 3,62→3,46→3,28→3,14→3,04→2,99→2,96→2,95; lr19 3,60→3,51→3,42→3,27→3,15→3,06→3,01→3,00. Monarch und schmal dicht laufen über die ganze Kurve praktisch deckungsgleich; Niedrigrang liegt ab Schritt 300 stets zurück.
CPU-Zeit je Lauf: dicht96 ~45 s, lr19 ~55 s, mon ~105 s, dicht512 ~70 s.

## 4. Deutung
- Monarch > Niedrigrang bei gleichen FLOPs plausibel (Rang-19-Engpass vs. volle Blockstruktur), Effekt klein (~0,05 bpc, 3 Seeds).
- Langweiligste Erklärung für „Monarch ≈ schmal dicht“: bei d=128, nur 1200 Schritten ist alles untertrainiert und die Kapazität der FFN kaum der Engpass; schmal dicht (96 volle Neuronen) ist ebenso ausdrucksstark. Monarchs Vorteil (breite, strukturierte Abbildung) zeigt sich evtl. erst bei größerem d/längerem Training. Lernrate am Gitterrand, nicht fein getunt.
- z²-Gewinn (Primer-artig) ist klassisch, kein Quanteneffekt.
- Kein Vorteil in Zeit (CPU) oder Parametern gegenüber schmal dicht → die FLOP-gleiche Hürde (I-045-v4) wird nicht genommen.

## 5. Vorschlag vorregistrierter Lauf (nur falls Malte will)
- Hypothesen: H1 Monarch < Niedrigrang (gleiche FLOPs/Gewichte), H2 Monarch nicht unterlegen gegen schmal dicht (Δ = 0,02 bpc), Nebenarm z².
- Arme: dicht schmal, Monarch, Niedrigrang, Monarch mit mehr Blöcken; d=256, 4 Lagen, ≥ 4000 Schritte, Lernrate fein getunt auf Tuning-Seed, Kurve alle 250 Schritte; Kontrolle: dicht breit; Zeit je Token mitmessen.
- 10 Seeds, Effekt zu erkennen ≈ 0,03 bpc. Laufzeit Container ~mehrere Kernstunden (Monarch langsamer), besser Raven.

## 6. Pfade
- `/home/user/QIRE/INSTITUT/recherche/2026-10-27_welle/w2/monarch_sprache.py` (Modi zaehlung/tuning/mess)
- `w2/tuning.json`, `w2/lr.json`, `w2/mess.json`, `w2/mess.log`
