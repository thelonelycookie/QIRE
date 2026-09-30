# Recherche Q-019: Walsh-Hadamard-Aufgabe + Dateneffizienz-Lernkurve

Stand: 2026-09-28. Kontext: geteilte Butterfly-Schicht (40 Param., 256-dim, Kronecker-Produkt von 2×2-Drehungen) vs. dichte Schicht (65.536 Param.). Vorlauf auf Zeitreihen/Fourier-Bändern: butterfly 0,686 vs. dense 0,913 — Fourier-Erzählung trägt nicht. Entschieden: Umbau auf (b) Walsh-Hadamard-Aufgabe + (a) Lernkurve über N = 50…6.000.

---

## 1. Literatur: Dateneffizienz strukturierter Matrizen

**Kernaussage vorweg (mit Unsicherheit markiert):** Es gibt *theoretische* Sample-Complexity-Resultate, die strukturierte (parameterarme) Netze linear in der Parameterzahl skalieren lassen, und *empirische* Hinweise, dass strukturierte Klassen im Small-Data-Regime geringere Generalisierungslücken zeigen. Eine direkte, sauber kontrollierte Studie speziell "Butterfly/Walsh-Hadamard-Layer vs. dichte Schicht, Lernkurve über N" habe ich in der Kürze der Recherche **nicht** gefunden — das ist eher aus Bausteinen zusammengesetzt als eine fertige Aussage aus einer einzelnen Quelle. Unsicher/nicht verifiziert markiert.

- **Butterfly-Faktorisierungen**: lernen schnelle lineare Transformationen (FFT-artig) mit O(n log n) Parametern statt O(n²); können bis zu 56× weniger Parameter als volle FC-Schichten nutzen und approximieren Fourier-Integral-Operatoren mit wenigen Parametern.
  [Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations](https://arxiv.org/pdf/1903.05895)
- **Monarch-Matrizen** (Produkt zweier blockdiagonaler Matrizen, hardware-effizient und ausdrucksstark genug, viele gängige Transformationen — inkl. Fourier/Hadamard-artige — darzustellen): 2× Trainingsbeschleunigung bei ViT/GPT-2 mit vergleichbarer Qualität, 40 % Fehlerreduktion bei PDE-Lösung und MRT-Rekonstruktion. Zeigt, dass strukturierte Matrizen bei richtiger Parametrisierung nicht zwangsläufig Genauigkeit kosten.
  [Dao et al., Monarch: Expressive Structured Matrices for Efficient and Accurate Training, ICML 2022](https://arxiv.org/abs/2204.00595) / [PDF](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf)
- **Low Displacement Rank (LDR) / strukturierte Gewichtsmatrizen — Sample-Complexity-Theorie**: Die theoretische Sample Complexity von Netzen mit strukturierten Gewichtsmatrizen skaliert *fast linear in der Gesamtparameterzahl*, analog zu voll verbundenen Netzen. Da LDR-Matrizen deutlich weniger Parameter haben, folgt eine entsprechend niedrigere VC-Dimensions-Schranke als bei unstrukturierten Netzen. Empirisch zeigen strukturierte Klassen durchgehend eine geringere Generalisierungslücke (Train-Test-Differenz) als unstrukturierte Baselines.
  [Thomas, Dao et al., Learning Compressed Transforms with Low Displacement Rank, NeurIPS 2018](https://arxiv.org/pdf/1810.02309)
- **Rank-sparse Generalisierungsschranken**: neuere Arbeit (2026) zeigt Sample-Complexity-Schranken der Form O(W·r·L²) (W=Breite, L=Tiefe, r=Rang der Gewichtsmatrizen) — je niedriger der effektive Rang, desto günstiger die Schranke. Stützt die Intuition "weniger effektive Freiheitsgrade → weniger Daten nötig", ist aber nicht butterfly-/Walsh-spezifisch.
  [Ledent et al., Generalization Bounds for Rank-sparse Neural Networks](https://arxiv.org/pdf/2510.21945)
- **Fast-Walsh-Hadamard-Transform-(FWHT)-Layer in echten Netzen**: WHT-basierte Layer ersetzen 1×1-/3×3-Konvolutionen; Parameterreduktion in MobileNet-V2 von 2,27M auf 947K bei O(m log₂ m) statt O(m²) Komplexität. Das ist eine Effizienz-, keine explizite Dateneffizienz-Studie, zeigt aber die praktische Machbarkeit lernbarer WHT-Layer.
  [Pan et al., Fast Walsh-Hadamard Transform and Smooth-Thresholding Based Binary Layers, CVPRW 2021](https://arxiv.org/pdf/2104.07085) / [Block WHT-based Binary Layers, TECS 2022](https://dl.acm.org/doi/abs/10.1145/3510026)
- **Induktionsbias generell (Small-Data-Regime)**: Mehrere Übersichts-/Analyse-Papiere bestätigen das allgemeine Muster "Gewinne durch stärkeren Inductive Bias sind am größten bei wenig Daten und schrumpfen mit wachsendem N" — z. B. Ablationsstudien zu strukturellen Constraints (Siamese-Architekturen o. Ä.), bei denen der Unterschied bei niedrigem Datenvolumen am größten ist und mit mehr Supervision abnimmt. Das ist die allgemeine Form der Hypothese, die Q-019 jetzt konkret für Butterfly/Walsh testen will.
  [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/html/2306.13575v1); [Customizing the Inductive Biases of Softmax Attention using Structured Matrices](https://arxiv.org/pdf/2509.07963)

**Einordnung für QIRE:** Es gibt also plausible theoretische Gründe (VC-Dimension/Sample-Complexity skaliert mit Parameterzahl) UND empirische Präzedenzfälle (strukturierte Klassen generalisieren besser bei wenig Daten) für die Hypothese "butterfly/WHT gewinnt bei kleinem N, dense gewinnt oder gleichzieht bei großem N". Aber: keine der gefundenen Quellen beweist das für *genau* die geteilte-40-Parameter-Butterfly-gegen-Walsh-Aufgabe. Die Jury-Erwartung sollte entsprechend kalibriert werden: die Literatur stützt die Richtung der Hypothese, nicht den genauen Effekt.

---

## 2. Reale Datensätze/Aufgaben mit natürlicher Walsh-/Sequency-Struktur

| Bereich | Beschreibung | Für Schüler auf CPU machbar? | Quelle |
|---|---|---|---|
| **Single-Pixel-Kamera / komprimiertes Abtasten mit Hadamard-Masken** | Bildaufnahme durch sequenzielle Beleuchtung mit Hadamard-Musterpaaren + Einzeldetektor; Rekonstruktion per (inverser) WHT. Es existiert ein öffentliches Rohdaten-Dataset. | Ja, mit Einschränkung: Rohmessdaten vorhanden, aber Rekonstruktionspipeline nicht trivial; als Klassifikationsaufgabe (z. B. Ziffer/Symbol aus Hadamard-Koeffizienten klassifizieren, ohne volles Bild zu rekonstruieren) machbar. | [Dataset and reconstruction pipeline for Hadamard-based single-pixel imaging, Zenodo](https://zenodo.org/records/18412991); [Photon-noise: single-pixel camera SNR-Analyse für Hadamard/Cosine](https://arxiv.org/pdf/2210.01915) |
| **Hadamard-Transform-Spektroskopie** | Encoding-Masken (Hadamard-Muster) erhöhen SNR bei dispersiven Spektrometern; sequency-geordnete Koeffizienten. | Eher nein für Schüler: benötigt Spektrometer-Hardware oder Simulationsdaten, die selbst erst erzeugt werden müssten — kein fertiger offener ML-Datensatz gefunden. | [Performance analysis of a Hadamard Transform Spectral Imaging system](https://arxiv.org/pdf/2510.20195) |
| **CDMA/Walsh-Codes (digitale Kommunikation)** | Walsh-Codes = Zeilen der Hadamard-Matrix, orthogonale Spreizcodes; reale Signalverarbeitungsaufgabe (Codeschätzung aus Mischsignal). Es gibt einen ML-Ansatz mit Sparse-Autoencoder zur Schätzung von Spreizsequenzen. | Bedingt: kein direkt herunterladbarer Standard-Datensatz gefunden, Signale aber leicht synthetisch mit realistischem Rauschmodell erzeugbar (reale Kommunikationsnorm, kein "erfundenes" Setup). | [Estimation of spreading sequences in LC-DS-CDMA signals based on sparse auto-encoder](https://link.springer.com/article/10.1007/s12065-019-00298-3) |
| **Boolesche Funktionen / Paritätsfunktion (Parity)** | Klassischer, exakt kontrollierbarer Testfall: k-sparse Parity hat ein einziges Walsh-Fourier-Koeffizienten-Bündel; bekannt als *hartes* Lernproblem für gewöhnliche MLPs/Gradientenabstieg (Phasenübergang beim Lernen, "Fourier-Gap"-Mechanismus). Leicht synthetisch erzeugbar, beliebig skalierbar in N und Eingabedimension. | Ja, sehr gut CPU-geeignet, da synthetisch und klein. | [Neural networks learn k-sparse parities via a Fourier gap mechanism (Referenz aus Boolean-Circuits-Literatur)](https://arxiv.org/pdf/1910.11923); [Learning Boolean functions with concentrated spectra](https://arxiv.org/pdf/1507.04319) |
| **Sparse Walsh-Hadamard-Transform-Algorithmen (SPRIGHT / SparseFHT)** | Algorithmen, die K-sparse WHT mit O(K log N) Samples statt N rekonstruieren — direkter Beleg, dass Signale mit sparsamer Walsh-Struktur mit *viel weniger* Messungen/Daten auskommen als generische Signale. Kein Klassifikations-Datensatz, aber starkes Argument für die Dateneffizienz-Hypothese aus der Signalverarbeitungs-Ecke. | Ja, als Referenz/Analogie, nicht als Datensatz selbst. | [SPRIGHT: A Fast and Robust Framework for Sparse Walsh-Hadamard Transform](https://arxiv.org/abs/1508.06336) |

**Empfehlung für die Jury-Frage "real vs. synthetisch":**
- Ein **realer** Datensatz (z. B. Single-Pixel-Hadamard-Imaging-Rohdaten von Zenodo) hätte Vorteil bei der Jury-Wirkung ("kein Spielzeugproblem"), bringt aber Rekonstruktions-/Vorverarbeitungsaufwand und ein zusätzliches Fehlerrisiko kurz vor Deadline (30. Nov.) mit sich — **unsicher, ob das im verbleibenden Zeitfenster sauber machbar ist**, das solltest du selbst gegen die Zeit abwägen.
- **Synthetische Rechteck-/Schaltsignale in Walsh-Basis** (direkt aus der entschiedenen Aufgabenstellung (b)) sind der sicherere, kontrollierbare Weg und methodisch sauberer für die geplanten Kontrollexperimente (Abschnitt 3) — die Jury-Erzählung "Walsh-Struktur kommt real in digitalen Schaltsignalen/CDMA/Hadamard-Bildgebung vor" lässt sich auch ohne echten Datensatz glaubwürdig referenzieren, wenn du die obigen Quellen zitierst.
- Als **Kompromiss**: synthetische Hauptexperimente + ein kleiner "Realitätscheck" mit dem öffentlichen Zenodo-Hadamard-Imaging-Datensatz als Zusatz/Anhang, falls Zeit bleibt.

---

## 3. Designempfehlung: nicht-triviale, falsifizierbare Aufgabe

**Risiko:** Die geteilte Butterfly OHNE CNOT ist als Kronecker-Produkt von 2×2-Drehungen mathematisch exakt die (verallgemeinerte) Walsh-Hadamard-Transformation. Wenn die "Aufgabe" darin besteht, genau diese Transformation zu approximieren, ist ein Sieg der Butterfly-Schicht **per Konstruktion garantiert** (Architektur-Match), nicht empirisch interessant, und für eine Jury leicht als Zirkelschluss zu entlarven.

### Kontrollen (wie im Auftrag skizziert, hier konkretisiert)

1. **Kontrolle "Basiswechsel"** — dieselbe zugrunde liegende Aufgabe (Klassifikation von Rechteck-/Schaltsignalen), aber das Zielsignal/Feature einmal in
   - (a) **Walsh-/Sequency-Basis** (Butterfly sollte gewinnen — das ist die Haupthypothese),
   - (b) **Fourier-Basis** (aus Q-019-Vorlauf bekannt: dense gewinnt — Replikation als Negativkontrolle),
   - (c) **permutierter Walsh-Basis** (Spaltenpermutation der Walsh-Matrix — bricht die exakte Kronecker-Struktur, behält aber die "Sparsity/Struktur" des Signals bei; wenn Butterfly hier *ebenfalls* gewinnt, spricht das für "strukturierte Signale generell", wenn nicht, für "exaktes Architektur-Matching"),
   - (d) **zufällige Orthobasis** (Haar-zufällige orthogonale Matrix; hier sollte die Butterfly-Schicht KEINEN Vorteil mehr haben — das ist die zentrale Falsifikationskontrolle: wenn Butterfly auch hier gewinnt, ist die Erklärung nicht "Walsh-Struktur", sondern z. B. allgemeine Regularisierungswirkung von weniger Parametern).

2. **Nicht-triviale Primärhypothese** — nicht "Butterfly gewinnt", sondern konkret und falsifizierbar, z. B.:
   - *H1:* Bei kleinem N (z. B. N ≤ 200) liegt die Test-Accuracy der Butterfly-Schicht signifikant über der dichten Schicht in der Walsh-Basis (c/d als Kontrolle sollten diesen Abstand NICHT zeigen).
   - *H2:* Es gibt einen Übergangspunkt N* (Crossover), ab dem dense die Butterfly einholt oder übertrifft (steilere Lernkurve der dichten Schicht) — das eigentliche "Dateneffizienz statt Endgenauigkeit"-Argument. N* selbst ist eine messbare, berichtbare Größe.
   - *H3 (optional, strenger):* Der Abstand Butterfly−Dense bei kleinem N in der Walsh-Basis ist größer als derselbe Abstand in der permutierten/zufälligen Basis (Interaktionseffekt Basis × Modell, nicht nur Haupteffekt Modell) — das ist die stärkste, am schwersten zu erschummelnde Formulierung, weil sie einen Differenzeffekt statt eines Achsenwerts testet.

3. **Modell-Arme** (wie im Auftrag genannt, plus eine Ergänzung):
   - **dense** (65.536 Param.) — Referenz oben.
   - **geteilte Butterfly** (40 Param., Kronecker/Quantenschaltungsstruktur) — Testkandidat.
   - **low-rank mit gleicher Parameterzahl wie Butterfly** (z. B. Rang-r-Faktorisierung W=UV mit ~40 Parametern) — wichtige Kontrolle, um "irgendein parameterarmes Modell" von "spezifisch Walsh-strukturiertes Modell" zu trennen; passt zur Rank-sparse-Generalisierungsliteratur oben.
   - **feste WHT (0 lernbare Parameter)** — Obergrenze/Sanity-Check: reine Transformation ohne Lernen, zeigt, wie viel der Butterfly-Leistung schon aus der festen Struktur kommt vs. aus den 40 gelernten Winkeln.
   - **ungeteilte Butterfly** (volle Parameterzahl der Butterfly-Faktorisierung ohne Parameter-Sharing, deutlich mehr als 40, aber weniger als dense) — zeigt, wo auf der Parameter-Achse der Nutzen des *Sharing* (nicht nur der Butterfly-Topologie) liegt.
   - **optional: Butterfly MIT CNOT/nichtlinearer Kopplung** — Abgrenzung, weil die Aufgabenstellung explizit "ohne CNOT" als Voraussetzung für die exakte WHT-Äquivalenz nennt; als zusätzlicher Arm zeigt es, ob Nichtlinearität die Walsh-Spezifität bricht.

4. **Lernkurve N = 50…6.000**: pro Basis (a–d) und pro Arm (5–6 Modelle) — das ergibt ein vollständiges Faktorendesign (Basis × Modell × N). Für die Jury-Präsentation reicht vermutlich eine Kernmatrix (Walsh + eine Negativkontrolle, alle Modell-Arme, volle N-Reihe) plus die übrigen Basen als Ergänzungstabelle/Anhang, um den Umfang in der verbleibenden Zeit realistisch zu halten.

**Unsicher/offen:** Wie genau "N" zu zählen ist (Trainingsbeispiele gesamt vs. pro Klasse), und ob 6.000 Beispiele bei 256-dim Eingaben schon in den Bereich kommt, wo dense laut Vorlauf-Ergebnis (0,913 bei vermutlich größerem N) klar überlegen ist — das lässt sich nur mit den konkreten Vorlaufdaten aus dem Projektordner klären, nicht aus der externen Literatur.

---

## Kurzfazit der Literaturlage (mit Vorsicht)

- Theorie (VC-Dimension/Sample-Complexity linear in Parameterzahl bei strukturierten Matrizen) UND Empirie (Monarch, LDR, Rank-sparse-Bounds) stützen die Grundidee "weniger Parameter → weniger Daten nötig für gleiche Generalisierung", **aber** keine gefundene Quelle testet exakt Butterfly-vs-Dense auf einer Walsh-Zielaufgabe mit Lernkurve — das ist echte Neuland-Leistung des Projekts, keine Replikation.
- Die Gefahr der Zirkularität (Butterfly = WHT per Konstruktion) ist real und muss über die vier Basis-Kontrollen (Walsh/Fourier/permutiert/zufällig) und die Interaktions-Hypothese H3 sauber adressiert werden, sonst ist das Ergebnis für die Jury angreifbar.
