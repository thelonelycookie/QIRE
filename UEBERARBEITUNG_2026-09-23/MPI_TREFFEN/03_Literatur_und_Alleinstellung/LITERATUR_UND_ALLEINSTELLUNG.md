# Literatur und Alleinstellungsmerkmal — wo QIRE steht

Stand 23./24. September 2026, Agent `literatur`. Grundlage: Websuche und Abstracts
(bei BOFT, Quantum-PEFT, EUNN, Bowles et al. auch die HTML-Fassung). **Kein Volltext
ist von Malte gelesen.** Jede Quelle unten wurde tatsächlich gefunden; die Details
stehen in den Literaturkarten unter `sources/literature_cards/karte_*.md`.

**Die Kurzfassung in drei Sätzen:** Die Schaltung selbst (geteilte Winkel, O(log d)
Parameter, klassisch als Butterfly/Kronecker umgesetzt) ist **bekannt** — am
deutlichsten durch *Quantum-PEFT* (ICLR 2025), das praktisch dieselbe Schaltung schon
als Adapter für Mistral-7B nutzt. Auch die Theorie „Born-Auslese = quadratische Form“
und die Kernkonzentration sind veröffentlicht. Was QIRE **eigenes** hat, ist die
**Messmethodik** (vorregistrierte Abgrenzungstests mit TOST und Positivkontrolle) und
die **empirische Zerlegung** des Vorteils — beides trägt aber nur, wenn Q-020 die
schwache Kontrolle von Q-003 repariert.

---

## 0. Begriffe in einem Satz

| Begriff | Bedeutung |
|---|---|
| **Butterfly** | Matrix als Produkt von log₂d dünnen Faktoren im Muster der schnellen Fouriertransformation; kostet O(d log d) statt d². |
| **Kronecker-Produkt** A⊗B | „Jeder Eintrag von A mal ganz B“; n Ein-Qubit-Gatter nebeneinander = Kronecker-Produkt von n 2×2-Matrizen. |
| **unitär / orthogonal** | Abbildung, die Längen erhält (alle Singulärwerte 1). |
| **PEFT / Adapter** | Ein großes vortrainiertes Modell bleibt eingefroren; trainiert wird nur ein kleiner Zusatzbaustein. |
| **LoRA** | Der Standard-Adapter: additive Niedrigrang-Änderung ΔW = B·A, 2·d·r Parameter. |
| **TOST** | Äquivalenztest: zeigt aktiv „der Unterschied liegt innerhalb ±δ“ — nicht bloß „nicht signifikant“. |
| **Barren Plateau / Konzentration** | Gradienten bzw. Kernwerte werden mit wachsender Qubitzahl exponentiell flach/gleich. |
| **Dequantisierung** | Zeigen, dass ein vermeintlicher Quantenvorteil klassisch genauso erreichbar ist. |
| **Born-Regel** | Messwahrscheinlichkeit = |Amplitude|²; bei QIRE die quadratische Auslese ⟨Z_q⟩ = xᵀA_q x / xᵀx. |
| **Effektivrang** | Wie viele Richtungen eine Schicht tatsächlich nutzt. |

---

## 1. Die Vorarbeit auf einen Blick

Einordnung: **B** = bestätigt QIRE · **V** = Vorarbeit (QIRE baut darauf) · **K** =
Konkurrenz (macht dasselbe oder mehr) · **G** = Gegenargument (schränkt eine
QIRE-Aussage ein).

### 1.1 Butterfly, strukturierte Matrizen, unitäre Netze

| Arbeit | Kern | Für QIRE |
|---|---|---|
| Dao u. a., ICML 2019 — Butterfly-Faktorisierung | Butterfly-Produkte lernen DFT, Hadamard, DCT selbst | **V** Ursprung der Struktur |
| Dao u. a., ICLR 2020 — Kaleidoscope | Butterfly-Produkte erfassen jede strukturierte Matrix | **V** passt zu „alle Familien universell“ |
| Dao u. a., ICML 2022 — Monarch | Zwei Blockdiagonalen, hardwarefreundlich | **V** (Projekt B), **G** zu (e) |
| Chen, Dao u. a., ICLR 2022 — Pixelated Butterfly | „Butterfly ist nicht hardwareeffizient“ | **G** zu (e) |
| Qiu u. a., ICML 2024 — Compute Better Spent | Strukturierte Schichten brauchen eigene Init- und Lernratenskalen; dann schlagen sie dicht | **G** zu Projekt B, Hinweis für T-002 |
| Le/Sarlós/Smola, ICML 2013 — Fastfood | Hadamard + Diagonalen ≈ dichte Zufallsmatrix, O(d log d) | **V** für Q-019 Walsh |
| Arjovsky/Shah/Bengio, ICML 2016 — uRNN | Unitäre Matrix gegen verschwindende Gradienten | **V** für Q-011 |
| Jing u. a., ICML 2017 — EUNN | Unitär aus 2×2-Drehungen; „FFT-style“ mit O(N log N) *ungeteilten* Winkeln | **V** — QIRE = EUNN-FFT mit geteilten Winkeln |
| Jose/Cissé/Fleuret, ICML 2018 — Kronecker Recurrent Units | Kronecker-Produkt aus 2×2-Faktoren, **O(log N)** Parameter, (weich) unitär | **K** zu (a) |
| Landman, Mathur u. a., Quantum 2022; Cherrat u. a., Quantum 2024 | Orthogonale Schichten/Transformer als Quantenschaltung | **V** für T-002 (Q/K/V) |

### 1.2 Finetuning mit wenigen Parametern (das dichteste Konkurrenzfeld)

| Arbeit | Kern | Parameter (Beispiel d = 256) | Für QIRE |
|---|---|---|---|
| LoRA (Hu u. a., ICLR 2022) | ΔW = B·A | 2·d·r = 512 je Rang | Baseline |
| OFT (Qiu u. a., NeurIPS 2023) | W ← R·W mit R orthogonal | groß (blockweise) | **V** |
| BOFT (Liu u. a., ICLR 2024) | R als Butterfly-Produkt, Blöcke *ungeteilt* | ½(b−1)·d·m ≈ **1.024** (b=2, m=8) | **K** Hauptbaseline |
| KronA (Edalati u. a., arXiv 2022) | ΔW = W_A ⊗ W_B | je nach Faktoren | **V** |
| QuanTA (Chen u. a., NeurIPS 2024) | Quanteninspirierte „Gatter auf Achsen“, hochrangig | klein | **K** |
| **Quantum-PEFT** (Koike-Akino u. a., **ICLR 2025**) | **RY + CZ im Wechsel, L Lagen, log₂d Qubits** | (2L+1)·log₂d − 2L = **22** (L=1) / **78** (L=5) | **K — stärkste** |
| QPA (Liu u. a., ICLR 2025) | Quanten-Netz erzeugt LoRA-Gewichte | 17–52 % von LoRA | **K** |
| QuIC (Raj/Coyle, arXiv 2025) | Hamming-Gewicht-erhaltende Compound-Adapter | > 40× kleiner als LoRA | **K** |
| QTHA (Kong u. a., arXiv 2025) | Quanten-Netz + Tensornetz | −76 % ggü. LoRA | **K** |
| *QIRE* | Ry/Rz je Qubit, CNOT-Ring, geteilte Winkel | 2·n·L = **16** (L=1) / **80** (L=5) | — |

**Lesehilfe:** Quantum-PEFT und QIRE haben bei d = 256 fast dieselbe Parameterzahl
(78 gegen 80), weil sie fast dieselbe Schaltung sind. Der einzige strukturelle
Unterschied: Quantum-PEFT benutzt die Schaltung als **lineare** unitäre
Gewichtsänderung, QIRE als Schicht mit **Born-Auslese** (quadratisch, nichtlinear).

### 1.3 Quanten-ML fair messen und dequantisieren

| Arbeit | Kern | Für QIRE |
|---|---|---|
| **Bowles/Ahmed/Schuld, arXiv 2024** — Better than classical? | 12 QML-Modelle, 160 Datensätze, klassisch gewinnt. Ohne Verschränkung: Schaltung zwischen klassischen Schichten („dressed“) **kein Verlust**, Data-Reuploading und Boltzmann-Maschine **mit Verlust**. Nur Ränge, kein TOST, keine Vorregistrierung. | **B** für Verschränkungs-Nullbefund — *aber nur im Encoder-Aufbau*; **K** zu (b) |
| Kesiku/Garcia-Zapirain, arXiv 2026 — How Quantum Circuits Actually Learn | Kausale Mediation: direkter Anteil : Quantenanteil = 13,1 : 1, Quantenanteil ~0,82 % | **B** (Quantenanteil klein), **K** zu (b)/(c) |
| **Zając u. a., arXiv 2026** — Inverse Born Rule Equivalence | Reelle Amplitudenkodierung + unitär + Rechenbasis-Messung = **genau die klassischen quadratischen Formen** | **B** für Rayleigh-Theorie, nimmt ihr aber die Neuheit |
| Schuld, arXiv 2021 — QML-Modelle sind Kernmethoden | Datenkodierung ist der entscheidende Baustein | **V** |
| Schreiber/Eisert/Meyer, PRL 2023 — Classical surrogates | Trainierte Quantenmodelle haben oft klassische Surrogate | **B** für Schaltung → Butterfly |
| Cerezo u. a., Nat. Commun. 2025 — BP-frei ⇒ simulierbar? | Beweisbar trainierbare Schaltungen sind meist klassisch simulierbar | **G** gegen jeden Quanten-Tempoanspruch |
| Tang, STOC 2019 | Quanten-Empfehlungsalgorithmus klassisch nachgebaut | **V** Begriff „quanteninspiriert“ |

### 1.4 Konzentration, Barren Plateaus, Quantenvorteil

| Arbeit | Kern | Für QIRE |
|---|---|---|
| McClean u. a., Nat. Commun. 2018 | Gradienten exponentiell klein in n | **V** Q-005 wäre Replikation |
| **Thanasilp u. a., Nat. Commun. 2024** | Quantenkerne konzentrieren sich exponentiell → Vorhersage unabhängig von der Eingabe | **B** K-Serie; (d) bekannt |
| Larocca u. a., Nat. Rev. Phys. 2025 | Übersicht Barren Plateaus | Einstiegslektüre |
| Kairon/Jäger/Krems, PRA 2026 | Kernkonzentration ⇔ Barren Plateaus | **V** |
| Aghaei Saem u. a., QST 2026 — Pitfalls | Quantum Natural Gradient & Co. überwinden Konzentration bei endlichem Messbudget nicht | **G** zu Ideen in BEFUNDE §4 |
| Huang u. a., Nat. Commun. 2021 — Power of data | Klassisches ML mit Daten oft ebenbürtig; projizierte Kerne | **G**/**V** |
| Chen/Cotler/Huang/Li, FOCS 2021 | Exponentielle Trennung mit/ohne Quantenspeicher | Theorie |
| **Huang u. a., Science 2022** | Zwei Kopien + Bell-Messung → exponentiell weniger Experimente; 40 Qubits | einziger beweisbarer Vorteil |
| Liu u. a., Science 2025 | Photonisch, ~11,8 Größenordnungen weniger Stichproben | Stand der Technik |
| Seif u. a., Nat. Commun. 2026 | Verschränkungsgestütztes Lernen von Pauli-Kanälen, bis 64 Qubits, supraleitend | Machbarkeit auf IBM-artiger Hardware |
| **Kannan u. a., arXiv 2608.13521 (Aug. 2026)** | Ein Qubit am Sensor → exponentiell weniger Messungen für Fourier-Koeffizienten klassischer Signale; 10⁷-fach im Experiment | Grenze für Q-019, Thema fürs MPI |

### 1.5 Rang von Repräsentationen (Projekt B), Nichtlinearität, Sprache

| Arbeit | Kern | Für QIRE |
|---|---|---|
| Papyan/Han/Donoho, PNAS 2020 — Neural Collapse | Endphase: Merkmale → (C−1)-Simplex | **V**; B-014 misst 2,7 statt 9 → kein Collapse, begründen |
| Huh u. a., TMLR 2023 — Low-Rank Simplicity Bias | Tiefere Netze → niedrigerer Rang | **G** zu „nicht von der Architektur“ |
| Ansuini u. a., NeurIPS 2019 — Intrinsische Dimension | ID ≪ Breite, schichtabhängig, sagt Genauigkeit vorher | **B** + **G** (Schichtabhängigkeit) |
| Novikov u. a., NeurIPS 2015; Stoudenmire/Schwab, NeurIPS 2016 | Tensor-Train-Schichten; MPS als Klassifikator | **V** Projekt B |
| Jayakumar u. a., ICLR 2020 — Multiplicative Interactions | Multiplikative Terme erweitern die Funktionsklasse | **B** „Quadratur trägt“ |
| Tsuchida u. a., NeurIPS 2023 — Squared Neural Families | Quadrieren + normieren als klassischer Baustein | **V** |
| Nebli u. a., arXiv 2026 — The Quantum Shortcut | Born-Formen als Auslese in 253M-Sprachmodellen: ⅓ bzw. ½ der Schritte; beim Transformer nur Aufwärmeffekt | **K** zu (c), Hinweis für T-002 |
| Schuld/Sweke/Meyer, PRA 2021 | Winkelkodierung = Fourier-Reihe; Re-Uploading erweitert Frequenzen | **V** Q-017 |
| LLäMmlein (arXiv 2024) | Deutsche Decoder 120M/1B(/7B), inkl. Chat-Variante | **G** zu (f), **Chance** als Basis |

---

## 2. Alleinstellungsmerkmal — Kandidat für Kandidat

Bewertung: **neu** (keine Vorlage gefunden) · **teilweise** (Teile bekannt, eine
bestimmte Kombination oder Messung nicht) · **bekannt** (veröffentlicht).
„Nicht gefunden“ heißt: in dieser Suche nicht gefunden — die Suche war nicht
erschöpfend.

| # | Kandidat | Urteil | Belegquelle |
|---|---|---|---|
| a | Geteilte Winkel je Qubit-Ebene → 16–80 Parameter | **bekannt** | Quantum-PEFT (ICLR 2025): (2L+1)log₂d−2L; KRU (ICML 2018): O(log N) mit 2×2-Kronecker-Faktoren. BOFT/EUNN teilen *nicht* — gegen die ist QIRE sparsamer, gegen Quantum-PEFT nicht. |
| b | Systematische Abgrenzungstests mit Positivkontrolle, TOST, Vorregistrierung | **teilweise neu** | Bowles 2024: systematisch, entfernt Verschränkung, aber ohne Statistik/Vorregistrierung. Kesiku 2026: Zerlegung per Mediation, ohne Äquivalenztest. TOST in ML existiert allgemein. **Die Kombination für quanteninspirierte Bausteine nicht gefunden.** |
| c | „Die Position der Born-Regel entscheidet“ (Gate: wirkungslos, Auslese: trägt ~87–92 %) | **teilweise neu** | Theorie bekannt: Zając 2026 (quadratische Formen), Schuld 2021, Jayakumar 2020. Born-Auslese in LMs: Nebli 2026. **Die empirische Zerlegung und die Positionsaussage nicht gefunden.** |
| d | Konzentration ↔ Scheitern von Quantenkernen, Exponent gemessen | **bekannt** | Thanasilp 2024 (Beweis), McClean 2018, Kairon 2026. E\|⟨ψ\|φ⟩\|² = 2⁻ⁿ für Zufallszustände ist Lehrbuchwissen. QIREs −0,958 ist eine saubere Replikation. |
| e | Butterfly gewinnt erst ab 16.384 Dim. auf CPU, wegen Cache-Kante der dichten Schicht | **bekannt** (Befund lokal) | Pixelated Butterfly 2022, Monarch 2022: Butterfly ist nicht hardwareeffizient. Die konkrete Zahl gilt nur für diesen Ryzen. Gut als ehrliche Grenze, kein USP. |
| f | Deutscher Korpus | **bekannt** | LLäMmlein 2024 (deutsch, von null, bis 7B). 3,27 Zeichen/Token ist eine nützliche Kennzahl, kein Beitrag. |
| g | *Zusatz:* B-014 — Rang ≈ Minimum aus Datendimension und Klassenzahl, Klassenabhängigkeit entsteht erst im Training (kontrolliertes d×C-Gitter + untrainierte Kontrolle) | **teilweise neu** | Einzelteile bekannt (Papyan 2020, Huh 2023, Ansuini 2019). Das kontrollierte Zwei-Faktor-Gitter nicht gefunden. Architektur nicht variiert → Überschrift einschränken. |
| h | *Zusatz:* „Ein Nullbefund an einem Bauteil ist wertlos, solange ein anderes Bauteil dessen Aufgabe übernehmen kann“ (Q-014 → Q-018) | **teilweise** | Bowles 2024 zeigt indirekt dasselbe (dressed circuit ohne Verlust, Re-Uploading mit Verlust). Als ausdrückliche Prüfregel mit Auflösungstest nicht gefunden. |

### 2.1 Was das für die Arbeit heißt — ehrlich

1. **Nicht mehr behaupten:** „Unsere quanteninspirierte Schicht braucht extrem wenige
   Parameter“ als Neuheit. Stattdessen: „Wir verwenden die aus Quantum-PEFT/KRU
   bekannte Parametrisierung und fragen, *welcher Teil davon wirkt*.“
2. **Die Frage wechseln von „Ist QIRE besser?“ zu „Wie viel Quanten steckt drin?“.**
   Dazu passen Bowles 2024, Kesiku 2026 und Zając 2026 als Rahmen — und QIRE liefert
   die schärfste Messung: vorregistriert, mit Äquivalenztest und Kontrollen.
3. **Voraussetzung:** Die Methodik ist nur dann ein USP, wenn sie überall eingehalten
   ist. Der Review hat gezeigt, dass gerade die Kernaussage gegen eine Kontrolle
   gemessen wurde, die schlechter als ein lineares Modell war, und dass „Tiefe trägt
   nichts bei“ aus einem Lauf mit durchgefallener Positivkontrolle stammt. **Ohne
   Q-020 ist Kandidat (b) angreifbar** — eine Jury würde genau dort bohren.
4. **Bowles liefert die Erklärung für den Verschränkungs-Nullbefund frei Haus:** Beim
   „dressed circuit“ (klassische Schichten drumherum) schadet das Entfernen der
   Verschränkung nicht, bei Re-Uploading schon. Das ist dieselbe Lehre wie Q-018 —
   und spricht dafür, Verschränkung × Tiefe **ohne Encoder** zu messen (Q-020 b).

### 2.2 Die drei stärksten USP-Kandidaten

1. **(b) Vorregistrierte Abgrenzungsbatterie für quanteninspirierte Bausteine** —
   TOST statt „nicht signifikant“, Positivkontrolle, Auflösungsprüfung, lineare
   Referenz je Kontrolle, zurückgezogene Aussagen bleiben sichtbar. Nächste Arbeiten
   (Bowles, Kesiku) haben das nicht. *Bedingung: Q-020.*
2. **(c) Mechanismus-Zerlegung „Born-Regel an der Auslese“** — 86,5 % (MNIST) bzw.
   91,5 % (Fashion) des Vorteils aus der Quadratur, dieselbe Operation als Gate
   wirkungslos. Die Theorie dahinter ist durch Zając 2026 abgesichert (das ist gut:
   eigene Herleitung + fremder Beweis); neu ist die gemessene Aufteilung und die
   Positionsaussage.
3. **(g) B-014 — Rang als Minimum zweier Zwänge, entsteht erst beim Lernen** — mit
   untrainierter Kontrolle. Muss gegen Neural Collapse, Low-Rank Simplicity Bias und
   intrinsische Dimension abgegrenzt und mit einem zweiten Architekturtyp geprüft werden.

### 2.3 Die drei wichtigsten Konkurrenzarbeiten

1. **Quantum-PEFT** (Koike-Akino, Tonin, Wu, Wu, Candogan, Cevher; ICLR 2025,
   https://arxiv.org/abs/2503.05431) — praktisch dieselbe Schaltung, logarithmische
   Parameterzahl, schon auf Mistral-7B. Muss in der Arbeit prominent stehen.
2. **Bowles, Ahmed, Schuld** (arXiv 2403.07059, 2024) — die Referenz für „fair gegen
   klassisch messen“ und „Verschränkung entfernen“. QIRE muss erklären, was es über
   sie hinaus macht (Statistik, Vorregistrierung, Kontrollen, Mechanismuszerlegung).
3. **BOFT** (Liu u. a., ICLR 2024, https://arxiv.org/abs/2311.06243) — orthogonales
   Butterfly-Finetuning; die Baseline, gegen die jede QIRE-Adapter-Aussage antreten
   muss (in Hugging Face PEFT fertig vorhanden).

Knapp dahinter: **Zając u. a. 2026** (nimmt der Rayleigh-Theorie die Neuheit) und
**Kronecker Recurrent Units 2018** (log-Parameter mit 2×2-Kronecker-Faktoren).

---

## 3. Zukunftsrichtungen mit echtem USP-Potenzial

Geordnet nach **Erfolgsaussicht ÷ Aufwand**. Die Nummern C1–C7 verweisen auf
`../BERICHT/01_VERBESSERUNGEN_UND_ZUKUNFT.md`.

| Rang | Richtung | Aussicht | Aufwand | USP, falls es klappt |
|---|---|---|---|---|
| 1 | **Q-020 + Methodik-Paket (C7)** | hoch | gering (CPU, Gerüst steht) | (b) wird unangreifbar |
| 2 | **Walsh-Hadamard-Aufgabe für Q-019, mit Lernkurve (C3 + C4)** | mittel–hoch | gering–mittel | „Struktur = ideale Basis → Datenvorteil“ |
| 3 | **QIRE-Adapter auf deutschem LLäMmlein gegen LoRA/BOFT/Quantum-PEFT (C1)** | mittel (Gleichwertigkeit), gering (Überlegenheit) | mittel–hoch (GPU) | Born-Auslese-Adapter + Äquivalenzstatistik + Deutsch |
| 4 | **Physik-Daten des MPI (C5)** | unbekannt — hängt am Treffen | unbekannt | echte Anwendung, einziger Datensatz, den sonst keiner hat |
| 5 | **Quantendaten-Vorteil nach Huang 2022 im Kleinen (C2)** | mittel (Simulation), gering–mittel (Hardware) | mittel | einziger *beweisbarer* Quantenvorteil — aber Replikation, kein neuer Befund |

### 3.1 Q-020 und die Methodik (Rang 1)
Warum zuerst: Alle drei USP-Kandidaten hängen an sauberer Messung. Q-020 (faire
klassische Referenz mit linearem Vergleichsarm; Verschränkung × Tiefe ohne Encoder)
ist billig und macht aus (b) einen Beitrag. Literatur-Anker: Bowles 2024 als „was es
schon gibt“, Kesiku 2026 als alternative Zerlegungsmethode. **Konkrete Aufwertung:**
Für jeden Q-Befund die „Quantenanteil“-Zahl ausweisen (wie viel bleibt nach Abzug der
klassisch erklärbaren Teile?) — direkt vergleichbar mit Kesikus 0,82 %.

### 3.2 Walsh-Hadamard statt Fourier (Rang 2)
Die geteilte Schaltung **ohne CNOT** ist ein Kronecker-Produkt von 2×2-Drehungen und
enthält die Walsh-Hadamard-Transformation exakt (Ry(π/2) auf jedem Qubit ist bis auf
Vorzeichen die Hadamard-Matrix). Bei Signalen aus Rechteck-/Schaltmustern ist das die
ideale Basis mit 8–16 Parametern; eine dichte Schicht muss sie aus 65.536 Parametern
erst lernen. **Was neu wäre:** Fastfood (2013) nutzt Hadamard fest; KRU (2018) lernt
Kronecker-Faktoren für RNNs — dass eine quanteninspirierte Schicht die Walsh-Basis
*lernt* und dadurch **mit wenig Daten** gewinnt, wurde nicht gefunden.
**Ehrlicher Haken:** Das ist wie die Q-019-Primärhypothese fast per Konstruktion wahr.
Der Wert liegt deshalb in der **Lernkurve** (N = 50…6.000; ab welcher Datenmenge holt
dicht auf?) und in der Gegenkontrolle (Fourier-Bänder, permutierte Zeitachse).
Hinweis: Kannan 2026 zeigt, dass ein Quantenvorteil beim Signallernen im **Sensor**
liegt, nicht in der Nachverarbeitung — Q-019 darf nicht als „Quantenvorteil bei
Signalen“ verkauft werden, sondern als „Induktionsbias einer quanteninspirierten Struktur“.

### 3.3 Adapter für ein deutsches Sprachmodell (Rang 3)
**Idee:** LLäMmlein-120M (später 1B-Chat) einfrieren; in Q/V-Projektionen einen
QIRE-Adapter setzen; gegen LoRA (Rang 1), BOFT und Quantum-PEFT **bei gleicher
Parameterzahl** (Größenordnung 10²–10³ je Schicht) auf einer deutschen Aufgabe
(z. B. Perplexität auf einem Fachkorpus oder eine SuperGLEBer-Aufgabe), mit TOST.
**Warum das den Chatbot rettet:** Aus einem Mini-Modell von null wird ein echter
deutscher Chatbot, und „ressourceneffizient“ heißt ehrlich: **weniger trainierbare
Parameter und weniger Trainingsspeicher**, nicht schnellere Inferenz — das löst den
Konflikt „Struktur gewinnt erst ab 16.384“ (CLAUDE.md §1).
**Was QIRE davon unterscheidet:** (i) Born-Auslese-Variante als *nichtlinearer*
Adapter gegen die rein unitäre Quantum-PEFT-Form — genau der Mechanismus aus (c);
(ii) Ablationen CNOT an/aus, geteilt/ungeteilt, Rz an/aus; (iii) Äquivalenzstatistik.
**Risiken:** Quantum-PEFT hat die Hauptaussage schon; Überlegenheit gegen LoRA ist
unwahrscheinlich; „gleichwertig bei 10× weniger Parametern“ ist realistisch, aber nicht
neu. Braucht GPU (RunPod ohne Guthaben). Nebli 2026 warnt: Vorteile können
Aufwärmeffekte sein → ganze Lernkurven berichten. BOFT/LoRA gibt es fertig in
Hugging Face PEFT (billige Baselines). Lizenz LLäMmlein: nur Forschung — für Jugend
forscht passend, aber im Text nennen.

### 3.4 Physik-Daten des MPI (Rang 4)
Das MPI Halle arbeitet an Spintronik, **neuromorphen Bauelementen**, Nanophotonik und
Quantenmaterialien (Institutsseite) und hat 2017 das Symposium „Beyond von-Neumann
Computing“ zu energieeffizienter Datenverarbeitung ausgerichtet. Echte Messreihen
(Spektren, Zeitreihen, Schaltvorgänge) wären der einzige Datensatz, den keine
Konkurrenzarbeit hat. Aussicht und Aufwand hängen vollständig davon ab, was das MPI
bereitstellen kann — deshalb die Fragen unten.

### 3.5 Quantendaten-Vorteil im Kleinen (Rang 5)
Huang 2022 (Science) ist der einzige Ort, an dem ein Lernvorteil **bewiesen** und
gemessen ist; Seif u. a. 2026 zeigen eine Variante auf supraleitender Hardware mit
bis zu 64 Qubits. Eine Mini-Version (2–4 Qubits, Bell-Messung an zwei Kopien gegen
Einzelmessungen) ist klein simulierbar und auf IBM lauffähig. **Aber:** Es ist
Lernen *über Quantensysteme*, nicht effizientere KI auf klassischen Daten, und es ist
eine Replikation. Als **Kontrast-Kapitel** („so sieht ein echter Quantenvorteil aus —
und darum hat QIRE keinen“) stark, als Hauptlinie schwach.
**Vor jedem IBM-Lauf:** Token neu erzeugen (Review §8, PC-Infektion).

### 3.6 Was die Literatur aus den alten Ideenlisten streicht oder abwertet
- **Quantum Natural Gradient / Identitäts-Initialisierung gegen Konzentration**
  (BEFUNDE §4): Aghaei Saem u. a. 2026 — hilft bei endlichem Messbudget nicht.
- **Q-005 Barren Plateaus, K-Serie**: Replikationen bekannter Theorie (McClean 2018,
  Thanasilp 2024). Nützlich als Prüfung der eigenen Werkzeuge, kein Beitrag.
- **„Quanten-Geschwindigkeitsvorteil“** jeder Art: Cerezo u. a. 2025 und die eigene
  Theorie (BEFUNDE §12) — ausgeschlossen.

---

## 4. Fragen fürs MPI-Treffen (Vorschlag)

1. Gibt es Messreihen mit **Schalt- oder Rechtecksignalen** (Walsh-artig) oder
   Spektren mit **wenigen gelabelten Beispielen**? Genau dort hätte die
   Struktur einen begründeten Vorteil (Abschnitt 3.2).
2. Interessiert die NISE-/Neuromorphik-Seite eine Schicht, die mit **16 Parametern
   und festem Verbindungsmuster** auskommt — als Kandidat für Hardware mit wenig
   Speicher? (Frage, kein Befund.)
3. Wie bewertet ein Physiker die Linie „quanteninspiriert, ehrlich dequantisiert“
   gegenüber „echter Quantenvorteil nur mit Quantendaten/Sensoren“ (Huang 2022,
   Kannan 2026)?
4. Gibt es Zugang zu einer GPU für den Adapter-Vergleich (Abschnitt 3.3)?

---

## 5. Was Malte vor dem Zitieren noch lesen muss (Reihenfolge)

1. Quantum-PEFT (Abschnitt Methode + Tabellen) — sonst droht ein Juryeinwand „das gibt es schon“.
2. Bowles/Ahmed/Schuld (Abschnitt Verschränkungs-Ablation).
3. Zając u. a. 2026 (Beweis quadratische Formen) — mit eigener Rayleigh-Herleitung vergleichen.
4. BOFT (Parametrisierung) und EUNN (FFT-style).
5. Papyan/Han/Donoho und Huh u. a. für Projekt B.

**Offene Unsicherheiten dieser Recherche:** Autorenlisten von Jayakumar 2020 und
LLäMmlein nicht vollständig bestätigt (in den Karten markiert); KronA-Venue unklar;
Bowles 2024 als arXiv geführt, Zeitschriftenfassung nicht gefunden; Parameterformeln
von BOFT und Quantum-PEFT aus einer automatischen Zusammenfassung der HTML-Fassung —
im Volltext nachprüfen.
