# Literaturkarte: Beweisbarer Quantenvorteil beim Lernen

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstracts geprüft,
Volltexte nicht gelesen.

Wozu: Alle bisherigen QIRE-Ergebnisse sind klassisch nachbaubar (siehe
Dequantisierungs-Karte). Der einzige Bereich, in dem ein Quantenvorteil **bewiesen**
und experimentell gezeigt ist, ist Lernen aus **Quantendaten** (bzw. mit einem
Quantensensor). Das ist eine andere Aufgabe als „KI auf Bildern/Text“.

---

## Huang, Broughton, Mohseni, Babbush, Boixo, Neven, McClean (2021) — Power of data
- **Titel:** Power of data in quantum machine learning
- **Venue:** Nature Communications 12, 2631 (2021)
- **Link:** https://www.nature.com/articles/s41467-021-22539-9
- **Was sie zeigt:** Klassische ML-Modelle, die aus **Daten** lernen, können auch
  Quantenprobleme oft gut vorhersagen; Methode zur Abschätzung eines möglichen
  Vorteils; **projizierte Quantenkerne** als Vorschlag.
- **Für QIRE:** **Gegenargument** gegen naive Quantenkerne und **Vorarbeit** für die
  Idee „projizierte Kerne“ (BEFUNDE §6).

## Chen, Cotler, Huang, Li (2021/22) — Lernen mit und ohne Quantenspeicher
- **Titel:** Exponential separations between learning with and without quantum memory
- **Venue:** FOCS 2021 (IEEE 62nd Symposium on Foundations of Computer Science), S. 574–585
- **Link:** https://ui.adsabs.harvard.edu/abs/2021arXiv211105881C/abstract
- **Was sie zeigt:** Ohne Quantenspeicher braucht z. B. Schatten-Tomographie
  Ω(min(M, 2ⁿ)) Stichproben; mit Quantenspeicher exponentiell weniger.
- **Für QIRE:** **Theoriegrundlage** für eine Quantendaten-Demo.

## Huang, Broughton, Cotler, Chen, Li, Mohseni, Neven, Babbush, Kueng, Preskill, McClean (2022)
- **Titel:** Quantum advantage in learning from experiments
- **Venue:** Science 376 (2022)
- **Link:** https://www.science.org/doi/10.1126/science.abn7293 · https://arxiv.org/abs/2112.00778
- **Was sie zeigt:** Wer zwei Kopien eines Quantenzustands im Quantenspeicher hält und
  gemeinsam (Bell-Basis) misst, lernt bestimmte Eigenschaften mit **exponentiell
  weniger Experimenten**; demonstriert mit bis zu 40 supraleitenden Qubits und 1.300 Gattern.
- **Für QIRE:** **Einziger Weg zu einem echten, beweisbaren Quantenvorteil** im
  Projekt. Aber: Es ist Lernen *über Quantensysteme*, nicht „ressourceneffizientere
  KI“ auf klassischen Daten. Klein (2–6 Qubits) simulierbar und auf IBM-Hardware
  lauffähig (siehe Seif et al.).

## Liu, Brunel, Østergaard, … , Huang, Oh, Jiang, Preskill, Neergaard-Nielsen, Andersen (2025)
- **Titel:** Quantum learning advantage on a scalable photonic platform
- **Venue:** Science 389, 1332–1335 (2025)
- **Link:** https://arxiv.org/abs/2502.07770 · https://www.science.org/doi/10.1126/science.adv2560
- **Was sie zeigt:** Lernen eines vielmodigen bosonischen Verschiebungsprozesses mit
  etwa **11,8 Größenordnungen weniger Stichproben** als konventionell (100 Moden),
  trotz unvollkommener EPR-Verschränkung (~5 dB Quetschung).
- **Für QIRE:** **Stand der Technik** — zeigt, dass die Huang-Linie 2025 experimentell
  groß geworden ist. Keine eigene Replikationsmöglichkeit (optischer Aufbau).

## Seif, Chen, Majumder, Liao, Wang, Malekakhlagh, Javadi-Abhari, Jiang, Minev (2026)
- **Titel:** Entanglement-enhanced learning of quantum processes at scale
- **Venue:** Nature Communications 17, 9000 (2026); arXiv 2408.03376
- **Link:** https://arxiv.org/abs/2408.03376
- **Was sie zeigt:** Fehlergemindertes, verschränkungsgestütztes Lernen von
  Pauli-Kanälen auf einem supraleitenden Prozessor (bis 64 Qubits für Hypothesentests,
  16 für Rauschlernen); Aufwand 1,33 ± 0,05 je Qubit statt theoretisch mindestens 2
  ohne Verschränkung/Quantenspeicher.
- **Für QIRE:** **Machbarkeitsbeleg** — die Art Experiment läuft auf IBM-artiger
  Hardware. Eine Mini-Version (2–4 Qubits) wäre für Malte ein ehrlicher
  „echter Quantenvorteil“-Baustein.

## Kannan, Prabhu, Khan, Sohoni, Song, Roy, Senanian, Fatemi, McMahon, Cotler (2026)
- **Titel:** Exponential quantum advantage for learning signals with a single qubit
- **Venue:** arXiv 2608.13521 (13. Aug. 2026); Daten: https://zenodo.org/records/21896087
- **Link:** https://arxiv.org/abs/2608.13521
- **Was sie zeigt:** Ein einzelnes steuerbares Qubit, gekoppelt an einen sonst
  klassischen Sensor, senkt die Zahl nötiger Messungen zum Lernen **klassischer
  Signale** exponentiell — für Fourier-Koeffizienten, zeitliche Korrelationen und
  Transformationen von Observablen; experimentell 10⁷-fache Reduktion
  (supraleitender Resonator + Qubit).
- **Für QIRE:** **Wichtig für Q-019 und das MPI-Gespräch.** Ein Quantenvorteil beim
  Lernen von *Signalen* existiert — aber im **Sensor** (Hardware), nicht in einer
  Butterfly-Schicht, die schon digitalisierte Daten verarbeitet. Das ist die klare
  Grenze: QIRE kann diesen Vorteil nicht auf CPU erreichen, kann ihn aber zitieren,
  um zu zeigen, wo Quantenvorteil bei Signalen tatsächlich liegt.
