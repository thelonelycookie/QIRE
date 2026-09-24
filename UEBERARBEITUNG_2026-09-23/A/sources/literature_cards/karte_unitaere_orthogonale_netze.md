# Literaturkarte: Unitäre und orthogonale Netze

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstract/Zusammenfassung per
Websuche geprüft; bei EUNN zusätzlich die HTML-Fassung (ar5iv). Volltext noch nicht
von Malte gelesen.

Wozu: QIREs Schaltung ist eine normerhaltende (unitäre) Abbildung aus 2×2-Drehungen.
Genau das haben unitäre RNNs seit 2016 gemacht — der wichtigste Vorläufer für Q-011
(Unitarität in der Tiefe) und für die Frage „wie viele Parameter braucht eine
unitäre Schicht“.

---

## Arjovsky, Shah, Bengio (2016) — Unitary Evolution Recurrent Neural Networks
- **Venue:** ICML 2016, PMLR 48:1120–1128
- **Link:** https://proceedings.mlr.press/v48/arjovsky16.html
- **Was sie zeigt:** Eine rekurrente Gewichtsmatrix mit Eigenwerten vom Betrag genau 1
  (unitär) verhindert explodierende und verschwindende Gradienten; die Unitarität wird
  über ein Produkt einfacher, billig parametrisierter Faktoren erzwungen statt über
  teure Zerlegungen.
- **Für QIRE:** **Vorarbeit für Q-011.** Die Hypothese „Unitarität wirkt erst in der
  Tiefe“ ist genau die Motivation dieser Arbeit. Q-011 prüft sie für
  vorwärtsgerichtete Tiefe gegen Spektralnormierung — das ist die eigene Ergänzung.

## Jing, Shen, Dubček, Peurifoy, Skirlo, LeCun, Tegmark, Soljačić (2017) — EUNN
- **Titel:** Tunable Efficient Unitary Neural Networks (EUNN) and their application to RNNs
- **Venue:** ICML 2017, PMLR 70
- **Link:** https://proceedings.mlr.press/v70/jing17a.html · https://arxiv.org/abs/1612.05231
- **Was sie zeigt:** Unitäre Matrizen als Produkt von 2×2-Drehungen auf
  zweidimensionalen Unterräumen. Zwei Varianten: „tunable“ (O(N·L) Parameter,
  abstimmbar bis zum vollen unitären Raum) und **„FFT-style“** (log N Lagen mit
  Paarungen in wachsendem Abstand, O(N log N) Parameter). **Keine Parameterteilung** —
  jede Drehung hat eigene Winkel.
- **Für QIRE:** **Wichtigste Vorarbeit für die Butterfly-Umsetzung.** Die
  QIRE-Butterfly ist EUNN-FFT-style **mit geteilten Winkeln je Ebene** (2·n·L statt
  O(N log N) Parameter). Das „quanteninspirierte“ Element ist genau diese Teilung.

## Jose, Cissé, Fleuret (2018) — Kronecker Recurrent Units (KRU)
- **Venue:** ICML 2018, PMLR 80
- **Link:** https://proceedings.mlr.press/v80/jose18a.html
- **Was sie zeigt:** Die rekurrente Matrix als Kronecker-Produkt kleiner Faktoren;
  mit 2×2-Faktoren braucht eine N×N-Matrix nur **O(log N)** Parameter; dazu eine
  weiche Unitaritätsbedingung gegen verschwindende/explodierende Gradienten.
- **Für QIRE:** **Direkte Konkurrenz zu USP-Kandidat (a)** („geteilte Winkel = extrem
  wenige Parameter“). QIREs Schaltung **ohne CNOT** ist exakt ein Kronecker-Produkt
  von 2×2-Drehungen — also eine KRU-artige Matrix. Die logarithmische Parameterzahl
  ist damit seit 2018 bekannt.

## Landman, Mathur u. a. (2022) — Quantum Methods for Neural Networks and Application to Medical Image Classification
- **Venue:** Quantum 6, 881 (2022)
- **Link:** https://quantum-journal.org/papers/q-2022-12-22-881/
- **Was sie zeigt:** Orthogonale Schichten als „pyramidenförmige“ Quantenschaltung
  aus RBS-Gattern (Hamming-Gewicht-erhaltend), mit klassischem Gegenstück;
  n(n−1)/2 Gatter für eine n×n-Orthogonalmatrix.
- **Für QIRE:** **Vorarbeit.** Zeigt, dass „Quantenschaltung als orthogonale Schicht,
  klassisch nachgebaut“ ein etabliertes Programm ist (Kerenidis-Gruppe).

## Cherrat, Kerenidis, Mathur, Landman, Strahm, Li (2024) — Quantum Vision Transformers
- **Venue:** Quantum 8, 1265 (2024)
- **Link:** https://quantum-journal.org/papers/q-2024-02-22-1265/
- **Was sie zeigt:** Transformer-Bausteine (u. a. Aufmerksamkeit) als orthogonale
  Quantenschichten, darunter eine Variante über Compound-Matrizen.
- **Für QIRE:** **Vorarbeit für T-002 (butterfly_qkv).** Quantenschaltungen in
  Q/K/V-Projektionen sind schon untersucht; T-002 muss sich dazu abgrenzen.
