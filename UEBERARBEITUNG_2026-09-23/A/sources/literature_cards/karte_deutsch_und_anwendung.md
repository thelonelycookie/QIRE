# Literaturkarte: Deutsches Sprachmodell als Grundlage, MPI Halle als Anwendungspartner

Angelegt 23.09.2026 (Agent `literatur`). **Lesestand:** Abstract bzw. Webseite
geprüft.

---

## Pfister u. a. (2024/25) — LLäMmlein
- **Titel:** LLäMmlein: Transparent, Compact and Competitive German-Only Language Models from Scratch
- **Venue:** arXiv 2411.11171 (Universität Würzburg, LSX)
- **Link:** https://arxiv.org/abs/2411.11171 · Modelle: https://huggingface.co/LSX-UniWue/LLaMmlein_120M · https://huggingface.co/LSX-UniWue/LLaMmlein_1B_chat_all · Code: https://github.com/LSX-UniWue/LLaMmlein
- **Was sie zeigt:** Rein deutsche Decoder-Modelle (120 M, 1 B, laut GitHub auch 7 B),
  transparent von null trainiert (deutscher Teil von RedPajama V2, eigener Tokenizer);
  konkurrenzfähig auf dem SuperGLEBer-Benchmark, deutlich besser als german-gpt2.
  Lizenz: nur für Forschung (RAIL-M).
- **Für QIRE:** **Gegenargument** zu USP-Kandidat (f): Ein deutsches Sprachmodell von
  null ist kein Alleinstellungsmerkmal. **Chance:** Als *vortrainierte Basis* für einen
  QIRE-Adapter ist es ideal — 120 M ist klein genug für ein Schülerbudget, und die
  1B-Chat-Variante macht den Chatbot zu einem echten Chatbot.
  (Erstautor aus der Suche nicht sicher bestätigt — vor dem Zitieren auf arXiv prüfen.)

## Max-Planck-Institut für Mikrostrukturphysik, Halle
- **Link:** https://www.mpi-halle.mpg.de/institute · https://www.mpi-halle.mpg.de/leopoldina-symposium
- **Was die Seite zeigt:** Forschung an nanostrukturierten Bauelementen:
  **Spintronik, neuromorphe Bauelemente und Systeme**, Nanophotonik, topologische
  Materialien, 2D-Schichten; Abteilungen NISE, NINT, SMFD sowie mehrere
  Forschungsgruppen (u. a. Quantenmaterialien, chirale Optoelektronik). 2017
  organisierte das Institut das Symposium „Beyond von-Neumann Computing“ (Leopoldina,
  6.–8. Sept. 2017) zu neuromorpher, energieeffizienter Datenverarbeitung.
- **Für QIRE:** **Anknüpfungspunkt fürs Treffen.** „Ressourceneffiziente KI“ ist am MPI
  eine *Hardware*-Frage (neuromorphe Bauelemente). Eine Schicht mit 16 Parametern und
  festem Verbindungsmuster ist ein Kandidat für Hardware mit wenig Speicher — das ist
  eine Frage, kein Befund.
