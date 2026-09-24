# Ergebnisübersicht — jede Aussage mit ihrer Rohdatei

**Diese Datei wird erzeugt, nicht geschrieben.** Sie entsteht aus den
gespeicherten Rohdaten unter `results/experiments/` und kann deshalb
nicht von ihnen abweichen. Neu erzeugen mit:

```
python -m qire_ai.results_report
```

Erzeugt am 04.08.2026, 23:50 Uhr aus 17 gespeicherten Läufen.

Zu jedem Lauf liegen neben der JSON-Datei die Messzeilen als CSV im
selben Verzeichnis. Jede Datei enthält außerdem die vollständige
Konfiguration und die Softwareumgebung (Python-, torch-, numpy-,
scipy-Version, CPU, Thread-Zahl).

---

## B-003 — Blockschicht gegen dicht, mit und ohne Beschraenkung, zwei Tiefen

- **Rohdaten:** `results/experiments/B-003__20260803_021746.json`
- **Messzeilen:** 80
- **Erzeugt:** 2026-08-03T02:17:46
- **Konfiguration:** dataset=mnist; seeds=10 Werte; arms=dense, block_plain, block_spectral, block_orthogonal; depths=1, 3; width=4096; epochs=10; lr=0.01; max_samples=6000

## B-004 — Gleiches Budget: Blockdiagonale, Monarch, TT, Low-Rank; drei Architekturen

- **Rohdaten:** `results/experiments/B-004__20260803_035228.json`
- **Messzeilen:** 210
- **Erzeugt:** 2026-08-03T03:52:28
- **Konfiguration:** dataset=mnist; seeds=10 Werte; arms=7 Werte; archs=dense_ends, local_ends, local_ends_perm; width=4096; epochs=10; max_samples=6000
- **primary_comparisons:** ['monarch_64x64 vs block_diag_128 in local_ends', 'monarch_64x64 vs block_diag_128 in local_ends_perm']
- **Urteil nach den vorher festgelegten Kriterien:**
  - local_ends: H1 -- Monarch schlaegt die Blockdiagonale. Der Befund aus A-002/A-003 haelt hier nicht.
  - local_ends_perm: H0 -- die Blockdiagonale haelt gegen Monarch, und der Aufbau ist nachweislich empfindlich.

## B-004-lr — Lernratensuche je Arm und Architektur

- **Rohdaten:** `results/experiments/B-004-lr__20260803_025454.json`
- **Messzeilen:** 126
- **Erzeugt:** 2026-08-03T02:54:54
- **Konfiguration:** arms=7 Werte; archs=dense_ends, local_ends; width=4096; epochs=10; max_samples=6000

## B-005 — Struktur der gelernten dichten Schicht (Blockmasse, Spektrum)

- **Rohdaten:** `results/experiments/B-005__20260803_055305.json`
- **Messzeilen:** 10
- **Erzeugt:** 2026-08-03T05:53:05
- **Konfiguration:** dataset=mnist; seeds=1501, 1502, 1503, 1504, 1505; archs=dense_ends, local_ends; width=4096; epochs=10; lr=0.01; max_samples=6000

## B-006 — Konvergenzkontrolle zu B-004

- **Rohdaten:** `results/experiments/B-006__20260803_054655.json`
- **Messzeilen:** 48
- **Erzeugt:** 2026-08-03T05:46:55
- **Konfiguration:** dataset=mnist; seeds=1401, 1402, 1403, 1404, 1405, 1406; arms=dense, block_diag_128, monarch_64x64, tt_c2_r64; archs=dense_ends, local_ends; width=4096; epochs=30; max_samples=15000
- **primary_comparisons:** ['monarch_64x64 vs block_diag_128 in local_ends', 'monarch_64x64 vs block_diag_128 in local_ends_perm']
- **Urteil nach den vorher festgelegten Kriterien:**
  - local_ends: H2 -- selbst die dichte Mittelschicht trennt sich nicht von der Blockdiagonale. Aufbau nicht empfindlich genug, aus dieser Architektur folgt nichts.

## B-007 — Orthogonalitaet gegen Spektralnorm, vorregistrierte Wiederholung

- **Rohdaten:** `results/experiments/B-007__20260804_234809.json`
- **Messzeilen:** 400
- **Erzeugt:** 2026-08-04T23:48:09
- **Konfiguration:** datasets=mnist, fashion_mnist; seeds=20 Werte; arms=plain, orthogonal, spectral, frobenius, weight_decay; depths=1, 3; width=4096; epochs=10; max_samples=6000
- **primary:** orthogonal vs spectral auf mnist, Tiefe 1 und 3, Holm
- **Urteil nach den vorher festgelegten Kriterien:**
  - HU -- auf mnist signifikant, aber die Wiederholung zeigt nicht dieselbe Richtung. Nach dem vorregistrierten Kriterium reicht das nicht. Befund 4 bleibt unsicher.

## B-008 — Rangreihe 4..256 plus exaktes Gewichtsspektrum, zwei Datensaetze

- **Rohdaten:** `results/experiments/B-008__20260803_192730.json`
- **Messzeilen:** 480
- **Erzeugt:** 2026-08-03T19:27:30
- **Konfiguration:** datasets=mnist, fashion_mnist; seeds=30 Werte; ranks=7 Werte; width=4096; epochs=10; max_samples=6000
- **prediction:** eine Schwelle t fuer beide Datensaetze; r_E(t) darf hoechstens einen Gitterschritt von r_acc abweichen
- **Urteil nach den vorher festgelegten Kriterien:**
  - H0 -- keine einzelne Schwelle passt auf beide Datensaetze. Das Spektrum sagt den noetigen Rang nicht vorher; B-005 bleibt eine Beschreibung ohne Vorhersagekraft.

## B-009-fashion_mnist — Monarch gegen dicht in der lokalen Architektur, 30 Seeds

- **Rohdaten:** `results/experiments/B-009-fashion_mnist__20260803_201312.json`
- **Messzeilen:** 90
- **Erzeugt:** 2026-08-03T20:13:12
- **Konfiguration:** dataset=fashion_mnist; seeds=30 Werte; arms=dense, monarch_64x64, block_diag_128; archs=local_ends; width=4096; epochs=10; max_samples=6000
- **primary_comparisons:** ['monarch_64x64 vs block_diag_128 in local_ends', 'monarch_64x64 vs block_diag_128 in local_ends_perm']

## B-009-mnist — Monarch gegen dicht in der lokalen Architektur, 30 Seeds

- **Rohdaten:** `results/experiments/B-009-mnist__20260803_194949.json`
- **Messzeilen:** 90
- **Erzeugt:** 2026-08-03T19:49:49
- **Konfiguration:** dataset=mnist; seeds=30 Werte; arms=dense, monarch_64x64, block_diag_128; archs=local_ends; width=4096; epochs=10; max_samples=6000
- **primary_comparisons:** ['monarch_64x64 vs block_diag_128 in local_ends', 'monarch_64x64 vs block_diag_128 in local_ends_perm']

## B-010 — Datenmenge gegen Struktur, sechs Trainingsgroessen

- **Rohdaten:** `results/experiments/B-010__20260804_003143.json`
- **Messzeilen:** 720
- **Erzeugt:** 2026-08-04T00:31:43
- **Konfiguration:** datasets=mnist, fashion_mnist; seeds=15 Werte; arms=dense, lowrank_r8, monarch, block_diag_128; train_sizes=200, 400, 800, 1600, 4200, 14000; width=4096
- **primary:** Spearman(log Trainingsgroesse, Abstand zu dicht) je Arm auf mnist, Holm-korrigiert; Vorhersage negativ
- **Urteil nach den vorher festgelegten Kriterien:**
  - H0 -- kein Arm erfuellt alle drei Bedingungen. Auch die Datenachse gibt keinen Vorteil fuer Struktur her.

## B-011-fashion_mnist — Darstellungs- oder Optimierungsluecke; datengewichtetes Spektrum

- **Rohdaten:** `results/experiments/B-011-fashion_mnist__20260804_013158.json`
- **Messzeilen:** 135
- **Erzeugt:** 2026-08-04T01:31:58
- **Konfiguration:** dataset=fashion_mnist; seeds=15 Werte; arms=dense, scratch, svd_frozen, svd_finetune, distill; ranks=8, 64; width=4096; epochs=10; lr=0.003; max_samples=6000
- **fairness_note:** svd_* und distill setzen ein fertig trainiertes dichtes Modell voraus und kosten daher MEHR Gesamtrechenzeit als scratch. Dieses Experiment beantwortet Erreichbarkeit, nicht Effizienz.
- **Urteil nach den vorher festgelegten Kriterien:**
  - kein klares Bild bei diesem Rang.
  - kein klares Bild bei diesem Rang.

## B-011-mnist — Darstellungs- oder Optimierungsluecke; datengewichtetes Spektrum

- **Rohdaten:** `results/experiments/B-011-mnist__20260804_010306.json`
- **Messzeilen:** 135
- **Erzeugt:** 2026-08-04T01:03:06
- **Konfiguration:** dataset=mnist; seeds=15 Werte; arms=dense, scratch, svd_frozen, svd_finetune, distill; ranks=8, 64; width=4096; epochs=10; lr=0.003; max_samples=6000
- **fairness_note:** svd_* und distill setzen ein fertig trainiertes dichtes Modell voraus und kosten daher MEHR Gesamtrechenzeit als scratch. Dieses Experiment beantwortet Erreichbarkeit, nicht Effizienz.
- **Urteil nach den vorher festgelegten Kriterien:**
  - DARSTELLUNGSGRENZE -- auch die Destillation erreicht die dichte Schicht nicht. Rang ist die falsche Struktur.
  - OPTIMIERUNGSLUECKE -- die Familie kann die Funktion darstellen, das Training von null findet sie nicht.

## B-012-fashion_mnist — Dosis-Wirkungs-Kurve des Trainingswegs, konstantes Budget

- **Rohdaten:** `results/experiments/B-012-fashion_mnist__20260804_052407.json`
- **Messzeilen:** 384
- **Erzeugt:** 2026-08-04T05:24:07
- **Konfiguration:** dataset=fashion_mnist; seeds=12 Werte; ranks=8, 32, 64; k_values=7 Werte; width=4096; lr=0.003; max_samples=6000
- **caveat:** beim Wechsel auf die beschraenkte Schicht wird der Adam-Zustand neu aufgesetzt -- gilt fuer jedes k>0 gleichermassen
- **design:** eine dichte Bahn je Seed, an sieben Stellen abgezweigt; Gesamtbudget konstant 10 Epochen in jedem Arm
- **Urteil nach den vorher festgelegten Kriterien:**
  - H1 -- bei Rang 64 steigt die Genauigkeit monoton mit dem Verzweigungszeitpunkt (rho=+0.857, p=0.0137). Der Weg hat einen messbaren Wert, bei identischem Budget.
  - H2 gestuetzt -- das optimale k waechst mit dem Rang ({8: 5, 32: 7, 64: 7}, rho=+0.866). Damit sind der Faktor-6,4-Befund und B-011 Spezialfaelle derselben Regel.

## B-012-mnist — Dosis-Wirkungs-Kurve des Trainingswegs, konstantes Budget

- **Rohdaten:** `results/experiments/B-012-mnist__20260804_043512.json`
- **Messzeilen:** 384
- **Erzeugt:** 2026-08-04T04:35:12
- **Konfiguration:** dataset=mnist; seeds=12 Werte; ranks=8, 32, 64; k_values=7 Werte; width=4096; lr=0.003; max_samples=6000
- **caveat:** beim Wechsel auf die beschraenkte Schicht wird der Adam-Zustand neu aufgesetzt -- gilt fuer jedes k>0 gleichermassen
- **design:** eine dichte Bahn je Seed, an sieben Stellen abgezweigt; Gesamtbudget konstant 10 Epochen in jedem Arm
- **Urteil nach den vorher festgelegten Kriterien:**
  - H0 -- kein monotoner Zusammenhang bei Rang 64. Der Vorteil in B-011 kam nicht vom Weg.
  - H2 nicht gestuetzt -- das optimale k waechst nicht mit dem Rang ({8: 7, 32: 5, 64: 3}).

## B-013-fashion_mnist — Rang-Curriculum gegen festen Rang und einzelnen Schnitt

- **Rohdaten:** `results/experiments/B-013-fashion_mnist__20260804_145621.json`
- **Messzeilen:** 60
- **Erzeugt:** 2026-08-04T14:56:21
- **Konfiguration:** dataset=fashion_mnist; seeds=15 Werte; arms=dense, fixed, single_cut, curriculum; width=4096; lr=0.003; max_samples=6000
- **primary:** curriculum gegen fixed
- **caveat:** gleiche Epochenzahl und gleicher Endrang, aber nicht gleiche Rechenzeit -- das Curriculum rechnet frueh mit mehr Parametern. Kein Effizienzanspruch.
- **Urteil nach den vorher festgelegten Kriterien:**
  - H0 -- kein signifikanter Vorteil gegenueber festem Rang. Der Befund aus B-012 laesst sich nicht in ein Verfahren uebersetzen.

## B-013-mnist — Rang-Curriculum gegen festen Rang und einzelnen Schnitt

- **Rohdaten:** `results/experiments/B-013-mnist__20260804_142928.json`
- **Messzeilen:** 60
- **Erzeugt:** 2026-08-04T14:29:28
- **Konfiguration:** dataset=mnist; seeds=15 Werte; arms=dense, fixed, single_cut, curriculum; width=4096; lr=0.003; max_samples=6000
- **primary:** curriculum gegen fixed
- **caveat:** gleiche Epochenzahl und gleicher Endrang, aber nicht gleiche Rechenzeit -- das Curriculum rechnet frueh mit mehr Parametern. Kein Effizienzanspruch.
- **Urteil nach den vorher festgelegten Kriterien:**
  - H0 -- kein signifikanter Vorteil gegenueber festem Rang. Der Befund aus B-012 laesst sich nicht in ein Verfahren uebersetzen.

## B-014 — Intrinsische Datendimension gegen Klassenzahl, mit untrainierter Kontrolle

- **Rohdaten:** `results/experiments/B-014__20260804_183624.json`
- **Messzeilen:** 240
- **Erzeugt:** 2026-08-04T18:36:24
- **Konfiguration:** seeds=10 Werte; dimensions=2, 5, 10, 20, 50, 100; class_counts=2, 5, 10, 20; width=4096; epochs=10; lr=0.003
- **primary:** Spearman(d, Effektivrang) > 0
- **control:** dieselbe Messung am untrainierten Netz
- **Urteil nach den vorher festgelegten Kriterien:**
  - H1 -- die intrinsische Datendimension setzt den Rang; sie erklaert mehr als die Klassenzahl. Vorbehalt: das UNTRAINIERTE Netz zeigt den d-Zusammenhang bereits; der Anteil, der aus dem Lernen stammt, ist nur die Differenz zur Kontrolle.

---

## Ohne Rohdaten — nur durch einen Neulauf belegbar

Diese im Tagebuch genannten Experimente haben **keine**
gespeicherten Rohdaten. Ihre Zahlen stammen aus Konsolenausgaben,
die nicht mehr existieren:

- `A-001`
- `A-002`
- `A-003`
- `B-002`
- `C-001`
- `Q-001`
- `Q-002`
- `Q-003`

Für die schriftliche Arbeit gilt: entweder neu rechnen oder
ausdrücklich als nicht mehr reproduzierbar kennzeichnen.

---

Insgesamt **3652 einzelne Messungen** in 17 Läufen.
