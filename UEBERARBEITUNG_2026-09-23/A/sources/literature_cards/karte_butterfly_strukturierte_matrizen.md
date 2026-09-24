# Literaturkarte: Butterfly und strukturierte Matrizen

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Titel, Autoren, Venue und
Abstract per Websuche geprüft; Volltext von Malte noch nicht gelesen. Vor dem
Zitieren in der Arbeit mindestens Einleitung und Methodenteil lesen.

Wozu diese Karte: Die klassische Umsetzung der QIRE-Schaltung ist eine
Butterfly-Struktur (O(d log d)). Genau diese Struktur ist in der ML-Literatur seit
2019 intensiv untersucht. Ohne diese Quellen wirkt QIRE wie eine Neuerfindung.

---

## Dao, Gu, Eichhorn, Rudra, Ré (2019) — Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations
- **Venue:** ICML 2019, PMLR 97:1517–1527
- **Link:** https://proceedings.mlr.press/v97/dao19a.html · arXiv: https://arxiv.org/abs/1903.05895
- **Was sie zeigt:** Produkte von „Butterfly“-Faktoren (dünn besetzte Matrizen im
  Verbindungsmuster der schnellen Fouriertransformation) können per Gradientenabstieg
  gelernt werden und finden schnelle Algorithmen für DFT, Hadamard, DCT und Faltung
  bis N=1024 selbst wieder.
- **Für QIRE:** **Vorarbeit.** Die QIRE-Butterfly ist ein Spezialfall dieser Familie —
  mit dem Unterschied, dass QIRE die 2×2-Faktoren einer Ebene **teilt** (ein Winkelpaar
  je Qubit-Ebene statt je Paar). Muss in der Arbeit als Ursprung der Struktur genannt werden.

## Dao, Sohoni, Gu, Eichhorn, Blonder, Leszczynski, Rudra, Ré (2020) — Kaleidoscope
- **Titel:** Kaleidoscope: An Efficient, Learnable Representation For All Structured Linear Maps
- **Venue:** ICLR 2020
- **Link:** https://arxiv.org/abs/2012.14966 · https://dblp.org/rec/conf/iclr/DaoSGEBLRR20.html
- **Was sie zeigt:** „K-Matrizen“ (Produkte von Butterfly-Matrizen) erfassen
  beweisbar jede strukturierte Matrix mit nahezu optimalem Platz- und Zeitbedarf,
  O(n log n) je Matrix-Vektor-Produkt; als lernbare Schicht u. a. in ShuffleNet und
  in der Sprachvorverarbeitung eingesetzt.
- **Für QIRE:** **Vorarbeit / Theorie-Rahmen.** Deckt sich mit der eigenen
  Universalitätsrechnung („alle Schaltungsfamilien universell, Teilung kostet Tiefe“):
  Ausdrucksstärke von Butterfly-Produkten ist bekannt.

## Dao, Chen, Sohoni, Desai, Poli, Grogan, Liu, Rao, Rudra, Ré (2022) — Monarch
- **Titel:** Monarch: Expressive Structured Matrices for Efficient and Accurate Training
- **Venue:** ICML 2022 (Outstanding Paper), PMLR 162:4690–4721
- **Link:** https://proceedings.mlr.press/v162/dao22a.html · https://arxiv.org/abs/2204.00595
- **Was sie zeigt:** Monarch = Produkt zweier blockdiagonaler Matrizen (mit
  Permutation). Hardwarefreundlich und ausdrucksstark; die Projektion einer dichten
  Matrix auf Monarch hat eine analytische Lösung.
- **Für QIRE:** **Vorarbeit (Projekt B) und Gegenargument zu (e).** Monarch wurde
  gebaut, *weil* reine Butterfly-Produkte auf GPUs schlecht laufen — das ist die
  Literaturfassung von QIRE-Befund „36× langsamer auf GPU“ und „Übergang erst bei 16.384“.

## Chen, Dao, Liang, Yang, Song, Rudra, Ré (2022) — Pixelated Butterfly
- **Titel:** Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models
- **Venue:** ICLR 2022 (Spotlight)
- **Link:** https://arxiv.org/abs/2112.00029
- **Was sie zeigt:** Sagt ausdrücklich, dass Butterfly-Matrizen **nicht
  hardwareeffizient** sind, und schlägt Block-/Flach-Varianten vor (3× schneller als
  Butterfly; bis 2,3× schnelleres Training als dichte Modelle).
- **Für QIRE:** **Gegenargument zu USP-Kandidat (e).** Die Beobachtung, dass
  Butterfly erst bei sehr großer Breite gewinnt, ist bekannt; QIREs Übergangspunkt ist
  eine hardwarespezifische Messung, keine neue Erkenntnis.

## Qiu, Potapczynski, Finzi, Goldblum, Wilson (2024) — Compute Better Spent
- **Titel:** Compute Better Spent: Replacing Dense Layers with Structured Matrices
- **Venue:** ICML 2024, PMLR 235:41698–41716
- **Link:** https://proceedings.mlr.press/v235/qiu24f.html · Code: https://github.com/shikaiqiu/compute-better-spent
- **Was sie zeigt:** Strukturierte Schichten brauchen **andere Initialisierungs-
  skalen und Lernraten** als dichte; mit μP-artiger Skalierung schlägt ihre
  Block-Tensor-Train-Familie (BTT) dichte Schichten bei gleichem Rechenaufwand.
- **Für QIRE:** **Gegenargument zu Projekt B** („strukturierte Schichten ersetzen
  keine dichte“). B hat die Lernrate je Arm gesucht, aber nicht die
  Initialisierungsskala. Der B-Negativbefund muss auf „bei Standard-Initialisierung
  und dieser Aufgabengröße“ eingeschränkt werden. Für T-002 relevant: Lernratensuche
  allein reicht bei Butterfly womöglich nicht.

## Le, Sarlós, Smola (2013) — Fastfood
- **Titel:** Fastfood – Approximating Kernel Expansions in Loglinear Time
- **Venue:** ICML 2013, PMLR 28
- **Link:** https://proceedings.mlr.press/v28/le13.html
- **Was sie zeigt:** Hadamard-Matrizen mit diagonalen Zufallsmatrizen verhalten sich
  wie dichte Gauß-Matrizen, kosten aber nur O(d log d) (schnelle Walsh-Hadamard-Transformation).
- **Für QIRE:** **Vorarbeit für Q-019 (Walsh-Hadamard).** Walsh-Hadamard als
  billige Strukturmatrix ist seit 2013 Standard. Neu wäre nur, dass QIREs geteilte
  Schaltung ohne CNOT die Walsh-Hadamard-Basis **lernen** kann (Kronecker-Produkt von
  2×2-Drehungen) — das ist bei einer Aufgabe mit Walsh-Bändern zu zeigen.
