# Literaturkarte: Born-Regel als quadratische Nichtlinearität, Fourier-Sicht

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts geprüft,
Volltexte nicht gelesen.

Wozu: QIREs wichtigster Mechanismusbefund ist, dass 86,5–91,5 % des Vorteils aus der
**quadratischen Auslese** (Born-Regel, ⟨Z_q⟩ = xᵀA_q x / xᵀx) kommen. Diese Arbeiten
zeigen, was daran bekannt ist.

---

## Jayakumar, Czarnecki, Menick, Schwarz, Rae, Osindero, Teh, Harley, Pascanu (2020) — Multiplicative Interactions
- **Titel:** Multiplicative Interactions and Where to Find Them
- **Venue:** ICLR 2020
- **Link:** https://iclr.cc/virtual/2020/poster/1710 · https://openreview.net/forum?id=rylnK6VtDH
- **Was sie zeigt:** Multiplikative Wechselwirkungen (Gating, Aufmerksamkeit,
  Hypernetze) **erweitern beweisbar** die darstellbare Funktionsklasse und sind ein
  starker Induktionsbias beim Verschmelzen mehrerer Informationsströme.
- **Für QIRE:** **Bestätigt** QIRE-Befund „die Quadratur trägt“: Eine quadratische
  Form ist eine multiplikative Wechselwirkung der Eingabe mit sich selbst.
  (Autorenliste bis auf Jayakumar aus dem Gedächtnis ergänzt — vor dem Zitieren auf
  OpenReview prüfen.)

## Tsuchida, Ong, Sejdinovic (2023) — Squared Neural Families
- **Titel:** Squared Neural Families: A New Class of Tractable Density Models
- **Venue:** NeurIPS 2023
- **Link:** https://mlanthology.org/neurips/2023/tsuchida2023neurips-squared/ · arXiv 2305.13552
- **Was sie zeigt:** Dichtemodelle, die aus dem **quadrierten** 2-Norm-Wert eines
  Netzes plus Normierung entstehen — genau die Struktur der Born-Regel
  (|Amplitude|², normiert); oft mit geschlossener Normierungskonstante.
- **Für QIRE:** **Vorarbeit.** „Quadrieren und normieren“ ist als klassischer
  Modellbaustein etabliert; die Born-Regel ist kein exklusiv quantenmechanisches Werkzeug.

## Nebli, Saadatdoorabi, Keibel, Yam (2026) — The Quantum Shortcut
- **Titel:** The Quantum Shortcut: Complex Phase-State Dynamics Reduce the Optimization Steps of Sequence Models
- **Venue:** arXiv 2608.14691 (7. Aug. 2026)
- **Link:** https://arxiv.org/abs/2608.14691
- **Was sie zeigt:** Komplexe Zustände mit Phasen und **„quadratischen Born-Formen“**
  als Ausgabe statt reeller Zustände mit Affin-Softmax; bei 253 Mio. Parametern
  erreicht die Zustandsraum-Variante (Mamba) den Zielverlust in ~1/3, die
  Attention-Variante in ~1/2 der Schritte — beim Transformer schrumpft der Vorteil
  aber nach der Aufwärmphase gegen null.
- **Für QIRE:** **Konkurrenz und Bestätigung** zu USP-Kandidat (c): Die Born-Auslese
  als Baustein in Sprachmodellen wird 2026 schon untersucht. Für T-002 wichtig: Die
  Arbeit spricht von *weniger Optimierungsschritten* (Trainingsaufwand) — genau die
  Achse, auf der QIRE „ressourceneffizient“ messen sollte. Und sie warnt: Vorteile
  können reine Aufwärmphaseneffekte sein → Lernkurven, nicht nur Endwerte, berichten.

## Schuld, Sweke, Meyer (2021) — Datenkodierung als Fourier-Reihe
- **Titel:** Effect of data encoding on the expressive power of variational quantum-machine-learning models
- **Venue:** Physical Review A 103, 032430 (2021)
- **Link:** https://arxiv.org/abs/2008.08605 · Code: https://github.com/XanaduAI/expressive_power_of_quantum_models
- **Was sie zeigt:** Ein Quantenmodell mit Winkelkodierung ist eine **partielle
  Fourier-Reihe** der Daten; die zugänglichen Frequenzen bestimmt die Kodierung;
  mehrfaches Einspeisen (Re-Uploading) erweitert das Spektrum bis zur Universalität.
- **Für QIRE:** **Theorie für Q-017 (Re-Uploading)** und für die Vorhersage „Tiefe
  wirkt nur mit Wiedereinspeisung“. Bei Amplitudenkodierung (Kernaussage) gilt das
  nicht — dort ist die Ausgabe eine quadratische Form (siehe Zając et al. 2026 in
  `karte_qml_benchmarks_dequantisierung.md`).
