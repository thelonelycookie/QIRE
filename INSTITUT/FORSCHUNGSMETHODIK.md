# Forschungsmethodik — wie das Institut forscht (Stand 07.10.2026)

Ergänzt `UEBERGABE.md` Abschnitt 2/3. Ergebnis einer Websuche „wie forscht man gut“,
zugeschnitten auf QIRE. Kurz halten, anwenden.

## 1. Neuheitsprüfung (Vorgabe 7) als festes Verfahren
1. **Kernaussage in einem Satz** formulieren (Behauptung, nicht Methode).
2. **6–12 Suchanfragen** daraus: Kernaufgabe + Beitrag + Synonyme
   (z. B. „magic“ / „non-stabilizerness“ / „T-count“ / „stabilizer Rényi entropy“;
   „pruning“ / „compression“ / „sparsity“). Quellen: arXiv, Semantic Scholar, Google Scholar.
3. **Nach Relevanz, nicht nach Zitaten** filtern — neue Arbeiten haben 0 Zitate.
4. **Zitatverfolgung** am 1–3 nächsten Treffer: rückwärts (deren Literatur) und
   vorwärts (wer zitiert sie). Erst danach „nichts gefunden“ sagen.
5. Ergebnis in `IDEEN.md` notieren: *gefunden / nächster Verwandter / was fehlt dort*.
   Urteil: neu · neue Verbindung · Replikation.

## 2. ML-Experimente (Fallen aus der Literatur)
- Mehrere Seeds, Streuung + Konfidenzintervall berichten, nie den besten Lauf.
- Seeds vorher festlegen (kein „Seed-Mining“); Tuning-Seeds ≠ Mess-Seeds.
- Gleiches Tuning-Budget je Arm, auch für Referenzen.
- Seedzahl vorab aus der Teststärke ableiten, nicht nachträglich.

## 3. Quanten-ML: der ehrliche Vergleich
Laut Bowles/Ahmed/Schuld (arXiv:2403.07059) schlagen einfache klassische Modelle
die meisten Quantenklassifikatoren, und Entfernen der Verschränkung schadet oft
nicht. Für jeden QIRE-Befund daher Pflicht:
- **klassische Baseline gleicher Parameterzahl** (nicht nur „Identität“),
- **Ablation ohne Verschränkung** (CNOT-Ring weg),
- Frage „ist es dequantisierbar?“ — wenige T-Gatter heißen auch: klassisch
  simulierbar (Laufzeit ~2^(0,23·t)). Das ist für die Magie-Linie *kein*
  Gegenargument, sondern eine Aussage: wenig Magie = billig, auch klassisch.
  So formulieren.

## 4. Jugend forscht: woran gemessen wird
Eigenleistung · Wissenschaftlichkeit · Darstellung; max. 15 Seiten;
**Ergebnisse klar vom vorhandenen Wissen abgrenzen** → dafür ist Abschnitt 1 da.

## Quellen
- OpenNovelty, arXiv:2601.01576 · Citationchaser (Haddaway u. a.)
- Pitfalls in ML Research, arXiv:2011.02832
- Better than classical?, arXiv:2403.07059 · QML advantages, arXiv:2504.15964
- Jugend forscht: Bewertungskriterien der Jury in Kurzform (jugend-forscht.de)
