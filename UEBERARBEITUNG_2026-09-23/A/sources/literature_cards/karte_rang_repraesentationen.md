# Literaturkarte: Rang und Dimension gelernter Repräsentationen (für Projekt B, B-014)

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts geprüft,
Volltexte nicht gelesen. Ersetzt/ergänzt die ältere englische Karte `low_rank.md`.

Wozu: B-014 („Effektivrang wird von Datendimension und Klassenzahl bestimmt“) muss
sich zu diesen drei Arbeiten einordnen, sonst wirkt es wie eine Neuentdeckung von
Bekanntem (Review §6).

---

## Papyan, Han, Donoho (2020) — Neural Collapse
- **Titel:** Prevalence of neural collapse during the terminal phase of deep learning training
- **Venue:** PNAS 117 (40), 24652–24663 (2020)
- **Link:** https://www.pnas.org/doi/10.1073/pnas.2015509117 · https://arxiv.org/abs/2008.08186
- **Was sie zeigt:** In der „Endphase“ des Trainings (Trainingsfehler schon null,
  Verlust wird weiter gedrückt) ziehen sich die Merkmale der letzten Schicht auf die
  Klassenmittel zusammen, die ein hochsymmetrisches Gebilde bilden (C Klassen →
  (C−1)-dimensionaler Simplex).
- **Für QIRE/B:** **Vorarbeit und Abgrenzung.** B-014 misst bei 10 Klassen Rang 2,7,
  nicht 9 — also *kein* Neural Collapse. Mögliche Gründe, die in der Arbeit stehen
  müssen: gemessene Schicht ist nicht die letzte, Training nicht bis in die Endphase,
  anderes Rangmaß. Die Klassenabhängigkeit des Rangs an sich ist durch Neural Collapse
  qualitativ vorhergesagt.

## Huh, Mobahi, Zhang, Cheung, Agrawal, Isola (2023) — Low-Rank Simplicity Bias
- **Titel:** The Low-Rank Simplicity Bias in Deep Networks
- **Venue:** Transactions on Machine Learning Research (TMLR) 2023; arXiv 2103.10427
- **Link:** https://arxiv.org/abs/2103.10427 · https://openreview.net/forum?id=bCiNWDmlY2
- **Was sie zeigt:** Tiefere Netze finden bevorzugt Lösungen mit **niedrigem
  effektiven Rang**; für lineare Netze bewiesen, dass der Volumenanteil
  niedrigrangiger Lösungen mit der Tiefe monoton wächst.
- **Für QIRE/B:** **Gegenargument** zur Überschrift „Rang wird von der Aufgabe
  gesetzt, *nicht von der Architektur*“. Huh et al. sagen: Tiefe (also Architektur)
  senkt den Rang. B-014 hat nur eine MLP-Breite/Tiefe getestet — die Überschrift muss
  eingeschränkt werden, bis der CNN-/Mehrschicht-Lauf da ist.

## Ansuini, Laio, Macke, Zoccolan (2019) — Intrinsische Dimension
- **Titel:** Intrinsic dimension of data representations in deep neural networks
- **Venue:** NeurIPS 2019
- **Link:** https://proceedings.neurips.cc/paper/2019/hash/cfcce0621b49c983991ead4c3d4d3b6b-Abstract.html · Code: https://github.com/ansuini/IntrinsicDimDeep
- **Was sie zeigt:** Die intrinsische Dimension (ID) von Repräsentationen ist in
  trainierten Netzen um Größenordnungen kleiner als die Zahl der Einheiten; über die
  Schichten steigt sie erst und fällt dann; die ID der letzten verborgenen Schicht
  sagt die Testgenauigkeit vorher.
- **Für QIRE/B:** **Bestätigt** „Rang weit unter der Breite“ (B-012: 5–7 statt 64/256)
  und **Gegenargument** zur Architekturunabhängigkeit: Die ID hängt von der Schicht ab.
  Neu an B-014 bleibt das **kontrollierte d×C-Gitter mit untrainierter Kontrolle**
  (Klassenabhängigkeit entsteht erst beim Training) — in dieser Form nicht gefunden
  (Suche nicht erschöpfend).
