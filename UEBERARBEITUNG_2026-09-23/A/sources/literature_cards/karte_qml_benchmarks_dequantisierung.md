# Literaturkarte: Quanten-ML fair messen, Dequantisierung, klassische Nachbauten

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts geprüft; bei
Bowles/Ahmed/Schuld zusätzlich die HTML-Fassung (Methodik, Verschränkungs-Ablation).
Volltexte noch nicht von Malte gelesen.

**Dequantisierung** = zeigen, dass ein angeblich quantenmechanischer Vorteil auch
klassisch (fast) genauso erreichbar ist. Genau das macht QIRE mit der eigenen
Schaltung — diese Arbeiten sind deshalb Bestätigung *und* Messlatte.

---

## Bowles, Ahmed, Schuld (2024) — Better than classical?
- **Titel:** Better than classical? The subtle art of benchmarking quantum machine learning models
- **Venue:** arXiv 2403.07059 (Xanadu), mit Open-Source-Paket auf PennyLane-Basis
- **Link:** https://arxiv.org/abs/2403.07059
- **Was sie zeigt:** 12 QML-Modelle auf 160 Datensätzen gegen einfache klassische
  Modelle; die klassischen gewinnen. Hyperparameter per vollständiger Gittersuche mit
  5-facher Kreuzvalidierung. **Verschränkung entfernt:** Beim
  `DressedQuantumCircuitClassifier` (Schaltung zwischen klassischen Schichten) **kein
  Leistungsverlust**; bei Data-Reuploading und Quanten-Boltzmann-Maschine dagegen
  **Verlust**. Vollständig separable Modelle schneiden „überraschend gut“ ab.
  Laut HTML-Fassung: **keine Signifikanz- oder Äquivalenztests** (nur Ränge),
  **keine Vorregistrierung**.
- **Für QIRE:** **Bestätigt** QIREs Verschränkungs-Nullbefund — aber genau im
  Aufbau „Schaltung zwischen trainierbaren klassischen Schichten“, der bei QIRE
  (Q-003 mit Encoder) vorliegt. Und sie zeigt: Mit Data-Reuploading hängt es sehr wohl
  an der Verschränkung. Das stützt den Review (§2): „Verschränkung trägt nichts bei“
  gilt nur für diesen Aufbau. Zugleich die **nächste Vorarbeit zu USP-Kandidat (b)** —
  QIRE geht mit TOST, Positivkontrolle und Vorregistrierung methodisch weiter.

## Kesiku, Garcia-Zapirain (2026) — How Quantum Circuits Actually Learn
- **Titel:** How Quantum Circuits Actually Learn: A Causal Identification of Genuine Quantum Contributions
- **Venue:** arXiv 2603.16321 (März 2026)
- **Link:** https://arxiv.org/abs/2603.16321
- **Was sie zeigt:** Kausale Mediationsanalyse: Leistungsunterschiede zwischen
  Schaltungen werden in „direkte“ (Parametrisierung, Ausdruckskraft) und über
  Quantengrößen (Verschränkungsentropie, Reinheit, …) vermittelte Anteile zerlegt.
  Ergebnis über 43 Konfigurationen: direkter Anteil zu Quantenanteil im Mittel
  **13,1 : 1**, Quantenanteil im Mittel 0,82 %.
- **Für QIRE:** **Konkurrenz zu USP-Kandidat (b) und (c)** — und inhaltlich
  **bestätigend**: QIRE findet 86,5–91,5 % des Vorteils in der quadratischen Auslese,
  also ebenfalls „der Quantenanteil ist klein“. Unterschiedliche Methode (Mediation
  gegen Ablation mit TOST); in der Arbeit direkt nebeneinanderstellen.

## Zając, Cybulski, Dziewit, Kulpa (2026) — The Inverse Born Rule Equivalence
- **Titel:** The Inverse Born Rule Equivalence. On the Informational Limits of Real-Valued Amplitude Encodings and the Measurement of Quantum Advantage in Data Embeddings
- **Venue:** arXiv 2602.21350 (Feb. 2026, überarbeitet Juni 2026)
- **Link:** https://arxiv.org/abs/2602.21350
- **Was sie zeigt:** Reellwertige Amplitudenkodierung + datenunabhängige unitäre
  Schaltung + Messung in der Rechenbasis ergibt **genau die Klasse klassischer
  quadratischer Formen**. Wer das als Quantenvorteil deutet, begeht den „Inverse
  Born Rule Fallacy“.
- **Für QIRE:** **Bestätigt QIREs Theorie** („Auslese ist ein Rayleigh-Quotient
  xᵀA x / xᵀx“) — und nimmt ihr die Neuheit. QIRE muss diese Arbeit zitieren; die
  eigene Herleitung ist dann eine unabhängige Bestätigung plus die empirische
  Zerlegung (wie viel des Vorteils die Quadratur trägt).

## Schuld (2021) — Supervised quantum machine learning models are kernel methods
- **Venue:** arXiv 2101.11020
- **Link:** https://arxiv.org/abs/2101.11020
- **Was sie zeigt:** Überwachte Quantenmodelle sind mathematisch Kernmethoden: Sie
  arbeiten in einem hochdimensionalen Raum, der nur über Skalarprodukte zugänglich ist.
  Entscheidend für einen möglichen Vorteil ist die **Datenkodierung**.
- **Für QIRE:** **Theorie-Rahmen.** Erklärt, warum bei Amplitudenkodierung die
  Ausgabe quadratisch in x ist (Modell linear in ρ = |x⟩⟨x|).

## Schreiber, Eisert, Meyer (2023) — Classical surrogates for quantum learning models
- **Venue:** Physical Review Letters 131, 100803 (2023)
- **Link:** https://arxiv.org/abs/2206.11740
- **Was sie zeigt:** Aus einem trainierten Quantenmodell lässt sich oft effizient ein
  klassisches „Surrogat“ gewinnen, das dieselbe Ein-Ausgabe-Beziehung liefert;
  Inferenz dann ohne Quantencomputer.
- **Für QIRE:** **Bestätigt das QIRE-Vorgehen** (Schaltung → Butterfly auf CPU).

## Cerezo, Larocca, García-Martín, Diaz, Braccia, Fontana u. a. (2025) — Barren-Plateau-Freiheit und klassische Simulierbarkeit
- **Titel:** Does provable absence of barren plateaus imply classical simulability?
- **Venue:** Nature Communications 16, 7907 (2025)
- **Link:** https://www.nature.com/articles/s41467-025-63099-6 · https://arxiv.org/abs/2312.09121
- **Was sie zeigt:** Starke Hinweise, dass viele Verlustlandschaften, die *beweisbar*
  keine Barren Plateaus haben, klassisch (oder „quantenunterstützt klassisch“) in
  Polynomialzeit simulierbar sind — weil gute Trainierbarkeit das Problem in kleine,
  klassisch simulierbare Unterräume zwingt.
- **Für QIRE:** **Gegenargument gegen jeden Quanten-Geschwindigkeitsanspruch** und
  zugleich Erklärung, warum QIRE überhaupt trainierbar *und* klassisch umsetzbar ist.
  Die ehrliche Einordnung lautet: QIRE ist *quanteninspiriert*, kein Quantenvorteil.

## Tang (2019) — Dequantisierung von Empfehlungssystemen
- **Titel:** A Quantum-Inspired Classical Algorithm for Recommendation Systems
- **Venue:** STOC 2019 (51st ACM Symposium on Theory of Computing)
- **Link:** https://dl.acm.org/doi/10.1145/3313276.3316310
- **Was sie zeigt:** Ein klassischer Algorithmus mit ℓ²-Norm-Sampling erreicht den
  Quanten-Empfehlungsalgorithmus von Kerenidis/Prakash bis auf polynomiellen Faktor —
  der vermutete exponentielle Quantenvorteil verschwindet.
- **Für QIRE:** **Vorarbeit und Begriff.** „Quanteninspiriert“ im Sinne von Tang heißt:
  Idee aus dem Quantenalgorithmus, Umsetzung klassisch. Das ist genau QIREs Anspruch.
