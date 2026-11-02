# K6: Quanten-inspirierte ML (Born-Maschinen, Dequantisierung, Clifford/Magie)

Stand Welle 2 (b4, 02.11.2026). Bestand: ~221 Einträge (Welle 0/1), `quellen.json`. Welle 2 b4 ergänzt 9 Quellen in `WISSEN/neu/w2_b4.json`, alle nur Abstract (`volltext_gelesen=false`, Zahlen ungeprüft). Die Websuche lieferte kaum 2025–26-Treffer; gezielter ging es über die arXiv-API.

## Landkarte (Hauptlinien)
1. **Dequantisierung** (Tang 2018 [arXiv:1811.00414] bis QSVT [arXiv:1910.06151], [arXiv:2111.09079], [arXiv:2304.04932]): klassische Sampling-Algorithmen holen QML-Beschleunigungen polynomiell ein. Neu: Lücke Õ(ρ) gegen Õ(ρ^4) ist im Worst Case scharf [arXiv:2610.42069]. Heißt: Dequantisierung ist ein Argument gegen exponentielle Vorteile, aber der Aufwand bleibt polynomiell höher.
2. **Variationale QML, Simulierbarkeit, Surrogate** (RFF [arXiv:2210.13200], Tensornetze [arXiv:2307.06937], Lie-Algebra, QCNN [arXiv:2408.12739]): trainierbar heißt oft klassisch simulierbar. Neu: neuronales Fourier-Surrogat erreicht Data-Reuploading-QNNs [arXiv:2610.00841].
3. **Clifford, Magie, Stabilizer** (Bestand dicht, siehe quellen.json: SRE, CAFQA, Clifford-Perturbation, Magic Cultivation): Magie als Grenze klassischer Simulierbarkeit. Welle 2 fand hier nichts Neues über den Bestand hinaus.
4. **Born-Maschinen / Tensornetze** ([arXiv:1605.05775], [arXiv:2311.05050], Born-Regel-Textklassifikation): Quadrat-Auslese |ψ|² als generative Wahrscheinlichkeit. Neu nur Randbefund [arXiv:2610.21672] (Diffusions-LM als offenes Quantensystem, Quantum-Inspired Remasking).
5. **Unitäre/orthogonale und komplexwertige Netze**: uRNN [arXiv:1511.06464] wiederbelebt als [arXiv:2609.38598] (RL), [arXiv:2605.13370] (Phasor Memory, byte-level LM), [arXiv:2608.03425] (rein phasenbasiertes SSM), SWave-Rückschau [arXiv:2606.18324]; Interferenz-Attention [arXiv:2509.19331], [arXiv:2608.17288] (Bestand).
6. **Ehrliche Kritik**: Benchmark-Studie [arXiv:2403.07059]; fair getunte Baselines vernichten Quanten-Reservoir-Vorteil [arXiv:2607.09905]; SWave-Retrospektive zeigt, wie viel „Wellen“-Design nicht trägt [arXiv:2606.18324].

## Stärkste Baselines/Gegner für QIRE
- Reeller Block gleicher Größe statt komplexer/Phasenblock: SWave ersetzte sein komplexes Gated Unit durch einen reellen Squared-ReLU-Mixer mit weniger Parametern [arXiv:2606.18324].
- Fourier-/RFF-Surrogate gegen Phasen- und Fourier-Modelle [arXiv:2610.00841], [arXiv:2210.13200].
- Fair getunte klassische Netze gleicher Größe [arXiv:2607.09905].
- Für Gedächtnis/Rekurrenz: Mamba (PMNet vergleicht sich mit 3x größerem Mamba) [arXiv:2605.13370].

## Wichtige Zahlen (alle ungeprüft, nur Abstract)
| Zahl | Kontext | geprüft |
|---|---|---|
| Õ(ρ) gegen Õ(ρ^4) | Anfragen Quanten vs. dequantisiert, scharf [arXiv:2610.42069] | nein |
| PPL 22,0 (Schritt 89.861) | SWave 169 M, FineWeb-Edu [arXiv:2606.18324] | nein |
| 119 M Parameter, 18,8 B Token, ~100 % Copy-Retrieval | PMNet [arXiv:2605.13370] | nein |
| bis 11 Qubit | exakte Simulation in [arXiv:2607.09905] | nein |

## Quantenbezug
- **Echt quanten:** Dequantisierungs- und Simulierbarkeitsresultate (sagen, wann Quantenhardware nichts bringt), QRC-Studien.
- **Nur Analogie:** komplexwertige/unitäre/Phasen-Netze, Interferenz-Attention, Phasor-Gedächtnis. Mathematik ist klassisch (norm-erhaltende Rotationen, Gradientenstabilität).
- **Hebel für QIRE:** Norm-erhaltende Mischung (Rotation/Hadamard) plus Quadrat-Auslese; Stabilität durch Unitarität ist der belegte Teil, ein Mehrwert „der Quantenform“ nicht.

## Bezug zu eigenen Befunden
- I-038 (gelernte QIRE-Zelle ≡ feste Mischung) und Z3 (Rotation ≈ gleichwertig gegen faire Gegner) passen zu [arXiv:2607.09905], [arXiv:2606.18324], [arXiv:2610.00841]: fair getunte reelle/klassische Gegner holen den „Quanten-/Wellen“-Vorsprung ein.
- Möglicher Gegenpol: [arXiv:2605.13370] und [arXiv:2608.03425] behaupten Vorteile unitärer Phasendynamik bei Langzeitgedächtnis; Behauptungen aus Abstracts, nicht geprüft. Das ist ein anderes Regime (Rekurrenz/Zustandsverfolgung) als QIRE-Mischung, also kein direkter Widerspruch.

## Lücken/offene Fragen
- Keine 2025–26-Funde zu Clifford/Magie über den Bestand hinaus, keine neuen quanten-inspirierten Sprachmodelle mit sauberer Baseline.
- Keine Volltexte gelesen; alle Zahlen ungeprüft.
- Offen: Gibt es eine kontrollierte Studie „unitär vs. gleich große reelle Rekurrenz“ im LM?
- Websuche wenig ergiebig für Neuestes; arXiv-API-Suche nach Stichwörtern bleibt lückenhaft (Born + Sprache ohne Treffer).

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2 b4 | 9 | ~4 % (9 von ~230) |

## Schlüsse und neue Wege (Agent)
1. **Schluss:** „Quanten-inspiriert“ ohne fair getunte Gegner hält nicht; Z3 steht damit in guter Gesellschaft. Belege: [arXiv:2607.09905], [arXiv:2606.18324], [arXiv:2610.00841]. Langweiligste Erklärung: Kritiker finden, was sie suchen, Nullergebnisse werden eher veröffentlicht, wenn sie spektakulär sind. Quantenbezug: Analogie bis direkt (QRC).
2. **Schluss:** Ein QIRE-Text sollte die SWave-Lehre übernehmen: Ablationen offen zeigen, was nicht trägt (vier Ideen ohne Nutzen). Belege: [arXiv:2606.18324]. Langweiligste Erklärung: ein einzelner Bericht eines Teams. Quantenbezug: keiner.
3. **Vorschlag (Opus prüfen):** Unitäre Phasenrekurrenz (PMNet-Stil) als zusätzlichen Gegner bzw. Erweiterung für Sequenzaufgaben testen, mit gleichgroßem reellen Gegner. Belege: [arXiv:2605.13370] (nur Abstract). Langweiligste Erklärung: Gewinn kommt von Gradientenstabilität, nicht von Phasen; ein orthogonaler reeller Block würde reichen. Quantenbezug: Analogie.
4. **Schluss:** Dequantisierung begrenzt die Ehrlichkeitsformel: „quanten-inspirierter Vorteil“ ist polynomiell und modellabhängig, nicht exponentiell [arXiv:2610.42069]. Langweiligste Erklärung: Worst-Case-Aussage, Praxis kann besser sein. Quantenbezug: direkt.

## Runde 2 (Welle 4, d4, 02.11.2026)
6 Suchen, Abstracts über arXiv-API bestätigt, 5 neue Quellen in `WISSEN/neu/w4_d4.json` (alle nur Abstract, Zahlen ungeprüft). Bereits im Bestand und hier nur eingeordnet: CompactifAI [arXiv:2401.14109], Hadamard statt Attention-Ausgabeprojektion [arXiv:2603.08343], MPO auf PicoGPT [arXiv:2603.28534].

**Neue Befunde**
- **Kritik/Negativ:** Hybride QNN sind bei gleichen klassischen Schichten bestenfalls gleich, meist schlechter [arXiv:2601.04732].
- **Phasen/Interferenz mit Baselines:** PRISM (komplex 5,80 vs. reell 6,15 PPL, Autoren-Baseline, Autoren räumen ein, reell könne dasselbe lernen) [arXiv:2512.01208]; PCT schlägt reelle Softmax-Attention bei Parameter-Gleichheit, aber nur synthetische Tasks [arXiv:2605.10123]; CNet: Born-Score-Attention |<Q,K>|² ≈ Softmax (1,52 nats/char), komplexer Fourier-Mischer ≈ reeller FNet, schneller konvergent [arXiv:2610.08592]. Muster: gleichwertig, Vorteile nur klein, meist gegen Autoren-Baselines.
- **CompactifAI:** 93 % Speicher weniger (mit Quantisierung, LLaMA-2 7B, −2–3 % Genauigkeit) [arXiv:2401.14109]; MPO wird per sequentieller SVD gebaut, also mit Low-Rank verwandt. Ein unabhängiger Vergleich MPO gegen gut getunte Low-Rank-/Hadamard-Gegner bei gleicher Parameterzahl wurde nicht gefunden (Lücke). MPO-Studie ist nur 1 M Parameter [arXiv:2603.28534].
- **2-Designs:** Nur Quantenliteratur (Nakata: abwechselnd zufällig X-/Z-diagonale Unitäre ≈ 2-Design [arXiv:1502.07514], neu [arXiv:2605.03023]); keine ML-Arbeit mit „2-Design als Mischschicht“ gefunden. Das stützt S-008 (2-Design als echte Quantenstruktur), aber ohne ML-Beleg.
- **Stabilisator/Clifford in klassischen Netzen:** Nur Clifford-Algebra-Äquivarianz [arXiv:2305.11141] (andere Mathematik, kein Stabilisator). Kein Treffer für Stabilisator-/Pauli-Struktur als effiziente Schicht. Echte Lücke und damit mögliches Alleinstellungsmerkmal, aber auch Warnzeichen (keine Nachfrage?).
- **Hadamard als Baustein:** WHT-Ausgabeprojektion −25 % Attention-Parameter, Autoren melden gleich gute Leistung [arXiv:2603.08343] (v2, ungeprüft); passt zur Fastfood/Hadamard-MLP-Linie.

**Widerspruch/Nähe zu eigenen Befunden:** Kein Widerspruch. Zu Z3 (korrigiert laut S-008: im MLP gleichwertig, im LM schlechter als g16) passt, dass Phasen/Born-Varianten höchstens gleichwertig sind. Offene Spannung: Hadamard-Projektion [arXiv:2603.08343] meldet Vorteil bei Rotationsmischung im LM, Z3 fand ihn schlechter; Setup unterschiedlich (Ersatz der Ausgabeprojektion vs. Rotation statt g16).

**Lücken:** Kein Volltext; keine unabhängige MPO-gegen-Low-Rank-Studie; keine ML-Nutzung von 2-Designs oder Stabilisatoren.

| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 2 b4 | 9 | ~4 % |
| Welle 4 d4 (Runde 2) | 5 | ~2 % von ~240; Suchen wiederholen überwiegend Bekanntes, Sättigung nahe |

**Schlüsse und neue Wege (Runde 2, Agent)**
1. **Schluss:** Phasen-/Born-/Komplex-Netze liefern gleichwertig, nicht besser, sobald die Baseline stimmt. Belege: [arXiv:2610.08592], [arXiv:2512.01208], [arXiv:2601.04732]. Langweiligste Erklärung: kleine Skala, Autoren-Baselines. Quantenbezug: Analogie.
2. **Vorschlag (Opus prüfen):** Hadamard-/WHT-Mischung mit Diagonalen als QIRE-Baustein gegen MPO/Low-Rank bei gleicher Parameterzahl messen; CompactifAI-Zahlen sind ohne diesen Vergleich nicht einordenbar. Belege: [arXiv:2603.08343], [arXiv:2401.14109], [arXiv:2603.28534]. Langweiligste Erklärung: Gewinn kommt aus Quantisierung/Low-Rank, nicht aus der Tensorstruktur. Quantenbezug: Analogie (Tensornetz) bzw. Clifford/Hadamard als echte Struktur.
3. **Schluss:** Stabilisator-/2-Design-Mischschichten sind in der ML-Literatur unbesetzt. Belege: Suchlücke (3 Begriffssuchen ohne Treffer), [arXiv:1502.07514]. Langweiligste Erklärung: Suchwerkzeug zu grob oder Idee trägt nicht. Quantenbezug: direkt.
