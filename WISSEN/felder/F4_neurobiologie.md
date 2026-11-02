# F4: Neurobiologie (Spikes, sparse Codes)

Stand Welle 1 (a3, 02.11.2026). Zahlen aus Abstracts/Suchtreffern, nichts im Volltext gelesen, alle `geprueft=false`.

## Landkarte
- **Energie des Gehirns:** ca. 20 W; Neuronen meist inaktiv, also sehr spärlich (Übersicht: [arXiv:2303.10780]; nur Suchtreffer, nicht aufgenommen).
- **Spiking-Netze (SNN):** Versprechen = ereignisgetriebene Additionen statt Multiplikationen. Skalierung auf LLMs: SpikeLLM [arXiv:2407.04752], SpikingMamba durch Distillation [arXiv:2510.04595].
- **Gegenposition (fairer Vergleich):** SNN-Vorteil verschwindet, wenn Speicherzugriff/Datenbewegung mitgezählt und gegen quantisierte ANNs verglichen wird [arXiv:2409.08290].
- **Sparse expansive Codes:** FlyHash (Fliege: ca. 50 -> 2000 Zellen, zufällige Projektion + Winner-take-all) ist LSH; BioHash lernt die Projektion lokal [arXiv:2001.04907]. Attention ≈ Sparse Distributed Memory [arXiv:2111.05498].
- **Synapsenpräzision:** ca. 26 unterscheidbare Stärken ≈ 4,7 Bit je Synapse (CA1, kleine Probe, obere Schranke) [doi:10.7554/eLife.10778].
- **Dendriten:** schon in Basis [arXiv:2404.03708].

## Stärkste Baselines/Gegner für QIRE
- Quantisiertes ANN (4-8 Bit) mit ehrlichem Energiemodell schlägt naive SNN-Behauptungen [arXiv:2409.08290].
- FlyHash/zufällige Projektion + Top-k: Null-Lernparameter-Verwandte der festen Hadamard-Mischung + Auslese; QIRE muss gegen sie bestehen.
- Fastfood [arXiv:1408.3060] bleibt die engste klassische Baseline.

## Wichtige Zahlen (mit geprüft/ungeprüft)
| Zahl | Quelle | geprüft |
|---|---|---|
| ca. 4,7 Bit/Synapse (26 Stärken) | eLife 10778 | nein (Abstract) |
| SNN braucht >93 % Neuronen-Sparsity (VGG16, T=6) | 2409.08290 v1 | nein |
| SNN-Energie 85 %/78 % des ANN (optimiert) | 2409.08290 | nein |
| SpikingMamba ca. 4,76x Energie | 2510.04595 | nein |

## Quantenbezug
- **Echt quanten:** keiner. Gehirn-Quantenthesen nicht aufgegriffen.
- **Analogie:** Spärliche zufällige Expansion (FlyHash) ≈ feste Zufallsmischung + Auslese; Winner-take-all ≈ Top-k-Messung.
- **Hebel für QIRE:** Die quanten-inspirierte Pipeline (feste Mischung -> Quadrat/Messung) hat ein biologisches Gegenstück; das stützt die Motivation "billige feste Mischung", nicht eine Quantenwirkung (I-038).

## Bezug zu eigenen Befunden (I-/Z-IDs)
- I-038: klassische Erklärung; FlyHash-Linie stützt "feste Mischung + Nichtlinearität trägt".
- Kein Widerspruch gefunden. 4,7 Bit passt zu 4-Bit-Gewichten (K1), beweist aber nichts für QIRE.

## Runde 2 (Welle 3, c4; nur Abstracts/Suchauszüge, ungeprüft)
- **Synapsenpräzision:** Baldassi et al.: Lernen mit diskreten Synapsen braucht wenige Bits, Schätzung 1 bis 5 Bit [arXiv:1602.04129]; stimmt mit 4,7 Bit (eLife) überein, die Bandbreite ist aber uneinheitlich.
- **Dendriten:** parametergleiche Dendritenneuronen schlagen Perzeptrons auf MNIST [arXiv:2207.00708]; dendritische SNNs [arXiv:2412.06355]. Nur kleine Benchmarks.
- **Energiebudget Kortex (Suchauszug, keine Quelle aufgenommen):** erregende Synapsenströme ca. 57 %, Aktionspotentiale ca. 23 % der elektrischen Signalenergie; Synapsen ca. 41 % des Kortex-ATP; spärliche Codes sparen Energie. Kein belastbarer Bit-pro-ATP-Wert für den ganzen Kortex. Daher keine Zahl für QIRE ableitbar.

## Lücken/offene Fragen
- Kein Volltext gelesen; SNN-Zahlen unverifiziert.
- Kein direkter Test FlyHash vs. Hadamard+Quadrat in der Literatur gefunden.
- Dendritische Codes nur durch 2404.03708 abgedeckt.

## Sättigung Runde 2
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 3, c4 Runde 2 (F4, 3 Suchen) | 3 neue (F4) | mittel; Energiebudget nur Suchauszug |

## Schlüsse und neue Wege (Agent)
1. **FlyHash als Pflicht-Gegner:** Schluss: feste spärliche Zufallsexpansion + Top-k gegen Hadamard·Diagonal + Quadrat testen. Belege: [arXiv:2001.04907], I-038. Langweiligste Erklärung: beide sind nur zufällige Merkmale. Quantenbezug: Analogie (Messung als Auslese).
2. **Energie nur mit ehrlichem Modell behaupten:** Schluss: QIRE-Effizienz in Parametern/FLOPs und Speicherzugriffen ausweisen, nicht nur Operationen. Belege: [arXiv:2409.08290]. Langweiligste Erklärung: Zählfehler. Quantenbezug: keiner.
3. **Bit-Budget-Argument:** Schluss: ca. 4-5 Bit als biologisch plausible Zielpräzision für QIRE-Gewichte zitieren (Hinweis, kein Beleg). Belege: [doi:10.7554/eLife.10778] (kleine Probe). Langweiligste Erklärung: Messgrenze statt Kapazität. Quantenbezug: keiner.
4. **(R2) Bit-Budget mit Spanne zitieren:** 1 bis 5 Bit statt einer Punktzahl. Belege [arXiv:1602.04129, doi:10.7554/eLife.10778]. Langweiligste Erklärung: unterschiedliche Messmodelle. Quantenbezug: keiner.