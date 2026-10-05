# Gemeinsamer Auftrag — Breitenrecherche Etappe 1 (05.10.2026)

Der Direktor plant (Opus), du recherchierst (Sonnet). Antworte und schreibe auf **Deutsch**.

## Projekt in 10 Zeilen
- Malte (Jugend forscht, Anmeldung 30.11.2026, schriftliche Arbeit Jan. 2027): **ressourceneffizientere KI mit Quantenphysik**.
- Modell „QIRE-Zelle“: 16×16-Bild → Amplitudenkodierung auf 8 Qubits → L Lagen Ry(θ) je Qubit + CNOT-Ring → Born-Auslese |ψ|² (256 Werte) → linearer Kopf (2.570 Parameter, 99 % aller Parameter!).
- Befunde: Diskrete **Clifford-Suche (0 T-Gatter)** erreicht auf MNIST das volle Training (~0,94); Fashion braucht ~4 Magiewinkel. Die Clifford-Zelle = blockweiser Walsh-Hadamard über vorzeichen-maskierten Pixeln (Bruhat-Form); Maske (quadratische Form über GF(2), = CZ/S-Anteil) und Blöcke tragen nur gemeinsam; der **Rang der quadratischen Form** trägt 52–78 % des Maskengewinns, gute Formen sind zwischen Seeds übertragbar (I-036…I-036-v3).
- Sprach-Adapter (Projekt A, T-004/T-005): QIRE-Kronecker-Drehung mit log₂(Breite) Winkeln je Matrix auf LLäMmlein-120M; Verlust fällt streng monoton mit der Lagenzahl bis L=10, 55 % des LoRA-Gewinns mit 1/16 der Parameter.
- Maltes neue Ziele: **(1) MNIST mit ~100 Parametern insgesamt**, **(2) „Ist ein QLM unendlich skalierbar?“**, **(3) Holevo-Grenze — kann man sie umgehen?**, allgemein **Effizienz im Großen**, und **neue, eigene Ideen**.
- Hardware: Ryzen 5 3600 (6 Kerne, 16 GB, CPU), kein GPU-Budget; IBM-Quantum-Zugang existiert (Token neu zu erzeugen).
- Repo `/home/user/QIRE`, Einstieg `INSTITUT/UEBERGABE.md` §0, Plan `INSTITUT/PLAN_2026-10-05_effizienz.md`. Lies nur, was du brauchst.

## Regeln
- **Nur lesen + EINE eigene Berichtsdatei schreiben** (Pfad steht in deinem Auftrag). Keine anderen Dateien ändern, kein git, keine Läufe über ~5 min (kleine Rechenexperimente zur Prüfung einer Idee sind ok, im Scratchpad).
- Recherche gründlich: viele WebSearch-Anfragen (Ziel ≥ 15). arxiv.org ist für WebFetch evtl. gesperrt — probiere Alternativen (export.arxiv.org/abs/…, arxiv.org/pdf/…, semanticscholar.org, openalex.org API, Verlagsseiten, Blogs der Autoren). Wenn nur Snippets: als „nur Auszug“ kennzeichnen.
- **Nichts erfinden.** Unbelegtes als [EIGEN] oder [?] markieren. Zahlen nur mit Quelle.
- Ausdrücklich erwünscht: **eigene neue Ideen** ([EIGEN]) — mit kurzer Neuheitsprüfung (gesucht, nichts gefunden / gefunden: …).

## Format der Berichtsdatei (≤ 300 Zeilen)
1. Kurzfassung (5–10 Zeilen)
2. Stand der Forschung (mit Quellen-URLs)
3. Was das für QIRE bedeutet (konkret, ehrlich)
4. **Neue Ideen / Forschungsfragen** — je: Frage, warum neu (Neuheitsprüfung), billigster Test auf Maltes PC (Laufzeit grob), Kontrolle, Quantenstufe Q1/Q2/Q3
5. Offene Fragen für Etappe 2 (Vertiefung)
6. Quellenliste

## Rückmeldung an den Direktor (deine Schlussnachricht)
Höchstens **15 Zeilen**: Dateipfad, 3–5 wichtigste Erkenntnisse, die 2–3 vielversprechendsten Ideen (je 1 Zeile), Vorschläge für Vertiefung.
