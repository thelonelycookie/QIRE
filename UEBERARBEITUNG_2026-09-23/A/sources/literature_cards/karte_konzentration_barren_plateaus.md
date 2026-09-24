# Literaturkarte: Konzentration und Barren Plateaus

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts geprüft,
Volltexte nicht gelesen.

**Barren Plateau** = die Verlustlandschaft einer Quantenschaltung wird mit wachsender
Qubitzahl exponentiell flach; Gradienten verschwinden. **Konzentration** = Werte
(z. B. Kernähnlichkeiten) sammeln sich exponentiell eng um einen festen Wert. Beides
ist dasselbe Phänomen: der Zustandsraum wächst wie 2ⁿ.

---

## McClean, Boixo, Smelyanskiy, Babbush, Neven (2018) — Barren plateaus
- **Titel:** Barren plateaus in quantum neural network training landscapes
- **Venue:** Nature Communications 9 (2018)
- **Link:** https://www.semanticscholar.org/paper/d699e0958fe1d8a4c1d691765f7e11b823fa606f
- **Was sie zeigt:** Für zufällige parametrisierte Schaltungen ist die
  Wahrscheinlichkeit, dass ein Gradient merklich von null abweicht, **exponentiell
  klein in der Qubitzahl**.
- **Für QIRE:** **Theorie-Grundlage für Q-005** (Vorhersage Var ~ 2⁻ⁿ). Q-005 wäre
  eine Replikation, kein neuer Befund.

## Thanasilp, Wang, Cerezo, Holmes (2024) — Exponentielle Konzentration in Quantenkernen
- **Titel:** Exponential concentration in quantum kernel methods
- **Venue:** Nature Communications 15 (2024), DOI 10.1038/s41467-024-49287-w
- **Link:** https://www.nature.com/articles/s41467-024-49287-w
- **Was sie zeigt:** Quantenkernwerte konzentrieren sich unter allgemeinen Bedingungen
  exponentiell in der Qubitzahl um einen festen Wert; mit polynomiell vielen Messungen
  entsteht ein triviales Modell, dessen Vorhersage **nicht mehr von der Eingabe
  abhängt**.
- **Für QIRE:** **Bestätigt** den K-Serien-Befund (Kernähnlichkeit halbiert sich je
  Qubit, Kernmatrix → Einheitsmatrix). USP-Kandidat (d) ist damit **bekannt**; QIREs
  gemessener Exponent (−0,958) ist eine gute Replikation der Theorie.

## Larocca, Thanasilp, Wang, Sharma, Biamonte, Coles, Cincio, McClean, Holmes, Cerezo (2025) — Übersicht
- **Titel:** Barren plateaus in variational quantum computing
- **Venue:** Nature Reviews Physics 7, 174–189 (2025), DOI 10.1038/s42254-025-00813-9
- **Link:** https://doi.org/10.1038/s42254-025-00813-9
- **Was sie zeigt:** Übersichtsartikel über Ursachen, Diagnose und Gegenmittel von
  Barren Plateaus.
- **Für QIRE:** **Einstiegsquelle** für das Theoriekapitel; besser eine Übersicht
  lesen als zehn Einzelarbeiten.

## Kairon, Jäger, Krems (2026) — Konzentration ⇔ Barren Plateaus
- **Titel:** Equivalence between exponential concentration in quantum machine learning kernels and barren plateaus in variational algorithms
- **Venue:** Physical Review A 114, 022417 (2026)
- **Link:** https://arxiv.org/abs/2501.07433
- **Was sie zeigt:** Formale Verbindung zwischen Kernkonzentration und Barren
  Plateaus; Barren-Plateau-freie Schaltungen lassen sich zum Bau brauchbarer
  Quantenkerne nutzen.
- **Für QIRE:** **Vorarbeit** — die Brücke zwischen K-Serie (Kerne) und Q-005
  (Gradienten) ist bereits geschlagen.

## Aghaei Saem, Tafreshi, Holmes, Thanasilp (2026) — Fallstricke gegen Konzentration
- **Titel:** Pitfalls when tackling the exponential concentration of parameterized quantum models
- **Venue:** Quantum Science and Technology 11, 015049 (2026)
- **Link:** https://arxiv.org/abs/2507.22054
- **Was sie zeigt:** Verbreitete Gegenmittel — **Quantum Natural Gradient**,
  stichprobenbasierte Optimierung, NN-inspirierte Initialisierung — überwinden die
  Konzentration bei endlichem Messbudget **nicht**, auch wenn sie das Training
  anderweitig verbessern können.
- **Für QIRE:** **Gegenargument** zu den Ideen „Quantum Natural Gradient“ und
  „Identitätsnahe Initialisierung“ in `BEFUNDE_UND_RICHTUNGEN.md` §4, *soweit* sie als
  Mittel gegen Konzentration gedacht sind. (Auf dem klassischen Simulator mit exakten
  Erwartungswerten gilt das Messbudget-Argument nicht — dort bleibt nur die
  Gradientenvarianz.)
