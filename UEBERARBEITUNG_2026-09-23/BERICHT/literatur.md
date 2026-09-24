# Bericht Agent `literatur` — Kurzfassung

Stand 24.09.2026. Ausführlich: `../A/LITERATUR_UND_ALLEINSTELLUNG.md`.
Karten: `../A/sources/literature_cards/karte_*.md` (10 neue, deutsch; `tensor_networks.md`
und `low_rank.md` als veraltet markiert, nicht gelöscht). Rund 50 Quellen, alle per
Websuche gefunden und mit Link; Volltexte hat Malte **noch nicht** gelesen.

## Kernaussage
Die QIRE-Schaltung (geteilte Winkel, O(log d) Parameter, klassisch als
Butterfly/Kronecker) ist **veröffentlicht**: *Quantum-PEFT* (ICLR 2025) nutzt RY+CZ mit
(2L+1)·log₂d − 2L Parametern (bei d=256, L=5: 78; QIRE: 80) als Adapter bis Mistral-7B;
*Kronecker Recurrent Units* (ICML 2018) hatten O(log N) mit 2×2-Faktoren schon 2018.
„Born-Auslese = quadratische Form“ ist bewiesen (Zając u. a., arXiv 2026), die
Kernkonzentration auch (Thanasilp u. a., Nat. Commun. 2024).

## USP-Bewertung
| Kandidat | Urteil |
|---|---|
| (a) geteilte Winkel, 16–80 Parameter | bekannt (Quantum-PEFT, KRU) |
| (b) vorregistrierte Abgrenzungstests mit TOST/Positivkontrolle | **teilweise neu** — Bowles 2024 ohne Statistik/Vorregistrierung, Kesiku 2026 mit Mediation |
| (c) Position der Born-Regel entscheidet | **teilweise neu** — Theorie bekannt, gemessene Zerlegung (86,5/91,5 %) nicht gefunden |
| (d) Konzentration mit Exponent | bekannt (Replikation) |
| (e) Übergang 16.384 / Cache-Kante | bekannt in der Art (Pixelated Butterfly, Monarch) |
| (f) deutscher Korpus | bekannt (LLäMmlein) |
| (g) B-014 Rang = Minimum aus d und C | **teilweise neu** — gegen Neural Collapse, Huh, Ansuini abgrenzen |

**Drei stärkste USP-Kandidaten:** (b) Methodik, (c) Born-Regel-Zerlegung, (g) B-014.
Alle drei gelten nur, wenn **Q-020** die zu schwache Kontrolle repariert.

**Drei wichtigste Konkurrenzarbeiten:** Quantum-PEFT (ICLR 2025); Bowles/Ahmed/Schuld
„Better than classical?“ (2024); BOFT (ICLR 2024).

## Nebenbefunde für andere Agenten
- **experiment:** Bowles 2024 stützt den Review: Ohne Verschränkung kein Verlust nur
  beim „dressed circuit“ (klassische Schichten drumherum), mit Re-Uploading schon →
  Q-020 b ohne Encoder ist richtig.
- **t002:** Qiu u. a. (ICML 2024): strukturierte Schichten brauchen eigene
  Init-Skala, nicht nur eigene Lernrate. Nebli u. a. (2026): Born-Auslese in LMs,
  Vorteil beim Transformer nur in der Aufwärmphase → Lernkurven berichten.
- **Projekt B:** „nicht von der Architektur“ widerspricht Huh u. a. (Tiefe senkt Rang)
  und Ansuini u. a. (schichtabhängig) — Überschrift einschränken.
- Aghaei Saem u. a. (2026): Quantum Natural Gradient hilft nicht gegen Konzentration.

## Zukunftsrichtungen (Aussicht ÷ Aufwand)
1. Q-020 + Methodik-Paket — billig, sichert (b).
2. Walsh-Hadamard-Aufgabe für Q-019 mit Lernkurve — Struktur enthält die Walsh-Basis exakt; Wert liegt im Datenbedarf.
3. QIRE-Adapter (Born-Auslese) auf LLäMmlein-120M/1B-Chat gegen LoRA/BOFT/Quantum-PEFT bei gleicher Parameterzahl, TOST — macht den Chatbot echt; braucht GPU; Überlegenheit unwahrscheinlich, Gleichwertigkeit realistisch.
4. MPI-Daten — hängt am Treffen (Neuromorphik/Spintronik am MPI; Fragen im Hauptdokument §4).
5. Mini-Replikation Huang 2022 (IBM) — einziger beweisbarer Quantenvorteil, aber Replikation; als Kontrastkapitel.
