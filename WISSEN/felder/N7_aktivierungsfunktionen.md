# N7: Aktivierungsfunktionen (ReLU², Primer)

Stand Welle 1 (a3, 02.11.2026). Zahlen aus Abstracts/Suchtreffern, nichts im Volltext gelesen.

## Landkarte
- **Klassische Reihe:** ReLU -> GELU [arXiv:1606.08415] -> Swish/SiLU, per Suche gefunden [arXiv:1710.05941] -> GLU-Familie (GEGLU, SwiGLU) [arXiv:2002.05202].
- **Quadrat-Linie:** Primer fand Squared ReLU per Evolution [arXiv:2109.08668]; ReLU² für Sparsität [arXiv:2402.03804]; 2:4-Aktivierungssparsität nutzt Squared-ReLU ohne Genauigkeitsverlust [arXiv:2503.16672]; Q-Sparse nutzt es ebenfalls [arXiv:2407.10969].
- **Sparsität durch Aktivierung:** ReLU-Strikes-Back [arXiv:2310.04564], Sparsing Law: ReLU erreicht höhere Sparsity als SiLU [arXiv:2411.02335]; 2,4B-Modell 93,52 % Sparsity, 4,1x Speedup (nicht geprüft).
- **Polynomnetze:** Pi-nets tragen auch ohne Aktivierung, besser mit [arXiv:2003.03828, arXiv:2006.13026]; quadratische Netze [arXiv:2110.06081].
- **Skepsis:** Die meisten Änderungen übertragen sich nicht [arXiv:2102.11972].
- **Born-Quadrat in der Literatur:** keine Arbeit gefunden, die |z|² als Born-Regel-Aktivierung in einem Hadamard-MLP untersucht (Suche; nur Blog-Beschreibungen eines "Hadamard MLP" mit SiLU).

## Stärkste Baselines/Gegner für QIRE
- SwiGLU/GEGLU im gleichen Parameterbudget (GLU ist selbst bilinear, also quadratisch).
- GELU, ReLU², |z| (QIRE-eigene Gegner).
- Quadratische/Polynom-Netze ohne Aktivierung.

## Wichtige Zahlen (mit geprüft/ungeprüft)
| Zahl | Quelle | geprüft |
|---|---|---|
| Swish +0,9 pp / +0,6 pp ImageNet | 1710.05941 | nein |
| 2:4-Sparsity Squared-ReLU: FFN bis 1,3x schneller, ohne Genauigkeitsverlust | 2503.16672 | nein (Abstract) |
| 93,52 % Sparsity, 4,1x | 2411.02335 | nein |
| Pi-net FID 40,45 ohne vs 16,79 mit Aktivierung | 2003.03828 (Drittseite) | nein |

## Quantenbezug
- **Echt quanten:** keiner.
- **Analogie:** Born-Regel p = |z|² ist formal ein Quadrat; für reelle Merkmale identisch mit z² und mit der Squared-Linie (Primer).
- **Hebel für QIRE:** Born-Quadrat = Squared-Activation; dadurch zitierfähig ohne Neuheitsanspruch (Maltes Vorgabe 07.10.). Der Hebel liegt in Sparsität/Hardware (ReLU² -> 2:4), nicht in Quanteneffekt.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- Stützt: I-039-v2 (gleichwertig), I-042 (Wechsel je Aufgabe), Narang et al. (kleine Aktivierungsunterschiede übertragen sich selten).
- Möglicher Widerspruch: QIRE fand z² schlechter als ReLU auf CIFAR (-3,18 pp, I-042), die Literatur (Primer, 2:4) sieht Squared-ReLU (z² nur für z>0) als gut. Wahrscheinlich Unterschied zwischen z² (beidseitig) und ReLU² (einseitig, spärlich). Das ist eine Hypothese, nicht geprüft.

## Lücken/offene Fragen
- Kein Vergleich z² (symmetrisch) gegen ReLU² in der Literatur gefunden.
- Keine Volltexte gelesen.
- Pi-net-Zahl nur aus Drittseite.

## Schlüsse und neue Wege (Agent)
1. **ReLU² statt z² in QIRE-Auslese testen:** Schluss: einseitiges Quadrat könnte den CIFAR-Verlust (-3,18 pp) erklären und liefert Sparsität. Belege: [arXiv:2109.08668], [arXiv:2503.16672], I-042. Langweiligste Erklärung: Vorzeichenverlust bei z² erzeugt nur Symmetrie-Rauschen. Quantenbezug: Analogie (Born-Regel nur für |z|²; ReLU² bricht sie).
2. **SwiGLU als fairer Gegner:** Schluss: GLU ist bilinear = quadratisch; Born-Quadrat ist ein Spezialfall dieser Familie. Belege: [arXiv:2002.05202]. Langweiligste Erklärung: gleiche Funktionsklasse, nur andere Parametrisierung. Quantenbezug: Analogie.
3. **Aktivierungsgleichwertigkeit als Ergebnis behandeln:** Schluss: "gleichwertig" zitieren statt Vorteil behaupten. Belege: [arXiv:2102.11972], I-039-v2. Langweiligste Erklärung: kleine Effekte unter Seed-Rauschen. Quantenbezug: keiner.

## Runde 2 (Welle 3, c5, 02.11.2026): Ergänzungen
- **Polynom-Linie:** PolyCom (Kompositionen höherer Ordnung, LLM-Vortraining) [arXiv:2411.03884]; Hermite/Fourier/tropische Aktivierungen sind mit varianzerhaltender Init stabil trainierbar, GPT-2-Test [arXiv:2502.01247]. Hinweis: Init könnte den z²-Verlust (I-042) mit erklären (Hypothese, nicht geprüft).
- **z² gegen ReLU²:** weiterhin kein direkter Vergleich in der Literatur. Schwacher Hinweis: quadratische Aktivierung führt in Modulo-Addition-Netzen oft zu instabilem Training [arXiv:2602.16849] (Suchzusammenfassung, ungeprüft).
- **Aktivierung und Quantisierbarkeit:** Im SwiGLU-W4A4-Test bleibt das bilineare Produkt (w2-Eingang) die Schwachstelle, volle QuaRot-Rotation schließt die Lücke nicht [arXiv:2604.18128]. Gated Rescaling mindert Ausreißer, W4A4-Verlust 1,2 Punkte [arXiv:2601.22966]; dort (Suchzusammenfassung, ungeprüft) begünstigen ReLU²/PolyNorm Ausreißer. Aktivierungswahl hängt von der Normalisierung ab (ReLU gegen GELU -8,2 % PPL ohne LayerNorm) [arXiv:2410.09637].
- **Folge für QIRE:** Quadratische/bilineare Aktivierung im Auslesepfad ist ein Quantisierungsrisiko; ein Test "z² vs ReLU² vs |z| unter 4-Bit-Aktivierungen" fehlt in der Literatur (Lücke, auch für Kurtosis-Messung).
- Querverweis I-039-v2: Gleichwertigkeit bleibt; keine neue Quelle widerspricht.

| Welle 3 c5 (Runde 2) | 6 | 6 neu zu 14 Bestand (ca. 30 %) |


## Runde 3 (Welle 5, e5; 1 Suche, nur Abstracts)
- Direkter Vergleich z² / ReLU² / |z| unter 4-Bit-Aktivierungen weiter nicht gefunden. Neu: Ausreißer entstehen durch Training, nicht nur Aktivierung: OSP senkt Excess Kurtosis von 1818,56 auf 0,04 [arXiv:2506.19697]; Kurtosis-Rotation KurTail [arXiv:2503.01483]. Kurtosis ist damit die Messgröße für den QIRE-Test.
- Sättigung N7: 2 neue von ca. 6 (ca. 30 %), aber randständig; Kernlücke bleibt eigener Versuch.

| Welle 5 e5 (Runde 3) | 2 | randständig neu |
