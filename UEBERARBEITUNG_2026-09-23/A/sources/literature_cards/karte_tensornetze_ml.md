# Literaturkarte: Tensornetze im maschinellen Lernen

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts geprüft,
Volltexte nicht gelesen. Ersetzt die ältere englische Karte `tensor_networks.md`.

**Tensornetz** = eine riesige Matrix/ein riesiger Tensor wird als Netz vieler kleiner
Tensoren gespeichert (z. B. Tensor-Train / Matrix Product State). Kommt aus der
Vielteilchen-Quantenphysik (DMRG) und ist die älteste „quanteninspirierte“ Technik im ML.

---

## Novikov, Podoprikhin, Osokin, Vetrov (2015) — Tensorizing Neural Networks
- **Venue:** NeurIPS (NIPS) 2015
- **Link:** https://papers.nips.cc/paper/5787-tensorizing-neural-networks · Code: https://github.com/Bihaqo/TensorNet
- **Was sie zeigt:** Dichte Gewichtsmatrizen voll verbundener Schichten werden im
  **Tensor-Train-Format** gespeichert (TensorNet) — drastisch weniger Parameter bei
  kaum Genauigkeitsverlust.
- **Für QIRE:** **Vorarbeit für Projekt B** (Tensor-Train-Arm, A-002: Blockdiagonale
  schlägt TT bei gleichem Budget). Muss als Ursprung der TT-Schicht zitiert werden.

## Stoudenmire, Schwab (2016) — Supervised Learning with Tensor Networks
- **Venue:** NeurIPS (NIPS) 2016
- **Link:** https://proceedings.neurips.cc/paper/2016/hash/5314b9674c86e3f9d1ba25ef9bb32895-Abstract.html
- **Was sie zeigt:** Matrix Product States parametrisieren ein nichtlineares
  Kernmodell; die Optimierung stammt aus der Physik (DMRG-artiges Durchlaufen).
  Unter 1 % Testfehler auf MNIST.
- **Für QIRE:** **Vorarbeit.** Gleiche Idee wie QIREs Amplitudenkodierung (Daten als
  Produktzustand bzw. Zustand in einem 2ⁿ-Raum) — und die DMRG-Rangzuweisung (B-015)
  hat hier ihre Wurzel.

---

**Hinweis:** Ein Tensornetz-Überblick (Wang u. a., „Tensor networks for quantum
machine learning“, arXiv 2303.11735) tauchte in der Suche auf, wurde aber nicht
geprüft — erst lesen, dann zitieren.
