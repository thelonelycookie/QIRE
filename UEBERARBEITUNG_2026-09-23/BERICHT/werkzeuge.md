# Agent `werkzeuge` — Bericht (23./24. September 2026)

Alles im Arbeitsordner `UEBERARBEITUNG_2026-09-23`. Originale und Archiv nur gelesen.
Kein Training gestartet, kein Lauf angefasst. Alle Aufrufe < 1 min CPU.

---

## 1. Geänderte und neue Dateien

| Datei | Was |
|---|---|
| `A/qire_ai/intrinsic_dimension.py` | Korrelation über **24 Zellmittel** (gilt); alte Rechnung über 240 Einzelmessungen daneben als „PSEUDOREPLIZIERT, nicht zitieren“. Urteilstext neu (kein „d erklärt mehr“). Neu: Modellvergleich (7 Modelle, LOO + Stufen-Kreuzvalidierung), Robustheit über Rangmaße, `energierang()` (90/99 %) für künftige Läufe, `vortest`, CLI `auswerten [DATEI]` (ohne Training). **Achtung: Aufruf ohne Argument startet weiterhin den Lauf.** |
| `A/qire_ai/audit.py` | +149 Prüfzeilen für B-014, B-011, B-010, B-009, B-008, B-007 (inkl. Machtanalyse 27.08.), B-004-Positivkontrolle. Optionen `--daten ORDNER` (mehrfach) und `--tagebuch DATEI`, Umgebungsvariable `QIRE_B_DATEN`. Standard unverändert. Überspringt JSON-Dateien ohne `experiment_id` (s. Problem 5). |
| `A/quantum_sim/q_summary.py` | Q-019 und T-000 (eigener Topf KALIBRIERUNG) aufgenommen. `NACHTRAEGE` (was seit 27.08./23.09. gilt, mit Quelle). Randlagen **aus den Rohdaten** gelesen (`lr_map_*`, `lr_rand_*`, Fallback auf `lr` je Zeile), mit Richtung für das Primärpaar und Kollaps-Erkennung. Vollständigkeit für Läufe ohne Urteil. |
| `A/quantum_sim/q_audit.py` | +48 Prüfzeilen: Q-011 dritter Anlauf (gerettete Datei, fest adressiert), Q-011 zweiter Anlauf (verworfen, Tabelle vom 27.08.), Randlagen-Tabelle vom 27.08. inkl. zurückgezogener Q-015-Fashion-Zahl und Effekt/Artefakt-Faktoren. |
| `A/quantum_sim/unitarity_depth.py` | Wiederaufnahme `run(weiter=True)` (liest **alle** Q-011-Dateien, führt nach (Datensatz, Tiefe, Arm, Seed) zusammen, prüft Suchbudget und Konfiguration, übernimmt gespeicherte Lernraten teilweiser Tiefen). Gitterverlängerung in **beide** Richtungen, „flach“ statt Verlängerung bei Kollaps. Sichern alle 5 Seeds. Tests: `python -m quantum_sim.unitarity_depth test`. Neuer Config-Schlüssel `suchbudget`. |
| `B/pruefen_audit.py` (neu) | Startet das Audit aus dem Bündel B (lädt `code/` als Paket `qire_ai`, ohne Spiegeldateien zu ändern). |
| `B/code/audit.py`, `B/code/intrinsic_dimension.py` | Kopien der neuen A-Fassungen (Spiegel). |

---

## 2. Befehle und Ergebnisse

Alle aus `A` mit `$env:PYTHONIOENCODING="utf-8"` und dem venv-Python.

| Befehl | Ergebnis |
|---|---|
| `python -m quantum_sim.q_summary` | 3 verwertbar, 1 mit Vorbehalt, 2 nicht verwertbar, 11 offen, 1 Kalibrierung; **23 Lernratenwahlen am Rand/kollabiert**, 11 Nachträge. Q-019 und T-000 erscheinen. |
| `python -m quantum_sim.q_audit` | **154/154 OK** (vorher 106). |
| `python -m qire_ai.audit` (Standard, in A) | Exit 1, 22 × FEHLT — erwartet, B-Rohdaten liegen nicht in A; Hinweis auf `--daten` wird gedruckt. |
| `python -m qire_ai.audit --daten ..\B\daten` | **164/164 OK**, Exit 0 |
| `$env:QIRE_B_DATEN="..\B\daten"; python -m qire_ai.audit` | 164/164 OK, Exit 0 |
| aus `B`: `python pruefen_audit.py` | 164/164 OK, Exit 0 (liest `B/daten`, `B/TAGEBUCH_B.md`) |
| `python -m qire_ai.intrinsic_dimension vortest` | BESTANDEN (Zellmittel = Hand-Spearman; pseudorepl. p ≪ Zellmittel-p; Modellvergleich erkennt Minimum- bzw. additive Daten je 5/5; Energierang) |
| `python -m qire_ai.intrinsic_dimension auswerten ..\B\daten\B-014__20260804_183624.json` | siehe 3. |
| `python -m quantum_sim.unitarity_depth test` | Gittertest 6/6 (innen, 1× oben, 1× unten, außerhalb oben/unten, flach ohne Verlängerung); Wiederaufnahmetest 9/9; Trockenlauf auf echten Rohdaten: übernimmt genau **225 Zellen** (mnist Tiefen 1, 2, 3, 4, 8), 0 Widersprüche. Keine Datei in `results/experiments` geschrieben (geprüft). |

Der erste Vortest-Entwurf von `intrinsic_dimension` fiel durch — an meiner **Erwartung** (100:1-Energie legt 99,0 % in eine Richtung, 99-%-Rang ist 1, nicht 2), nicht am Code. Korrigiert und im Code kommentiert.

---

## 3. B-014 — nachgerechnet und Modellvergleich

Zellmittel (24 Zellen) — reproduziert exakt die Tagebuchwerte vom 27.08.:

| | d → Rang | Klassen → Rang |
|---|---|---|
| untrainiert | ρ=+0,987, p=6,5·10⁻¹⁹ | ρ=+0,027, p=0,90 |
| trainiert (Teilnahmezahl) | ρ=+0,606, p=1,7·10⁻³ | ρ=+0,673, p=3,1·10⁻⁴ |
| trainiert (Entropie-Rang) | ρ=+0,553, p=5,0·10⁻³ | ρ=+0,732, p=4,8·10⁻⁵ |

Monte-Carlo-Permutationstest (20.000, nur zur Kontrolle): d p≈0,0022, Klassen p≈0,0004 — passt zur scipy-Näherung. Energierang 90/99 %: **nicht in den Rohdaten** (offen, braucht Neulauf; `run()` speichert ihn jetzt mit).

Neues Urteil (automatisch): „H1 — der Rang wächst mit d, aber ebenso mit der Klassenzahl … WELCHE mehr erklärt, entscheidet dieser Lauf nicht — Randkorrelation und Innerhalb-Effekte (d +0,96, Klassen +0,83) ordnen die beiden sogar verschieden … Architektur wurde NICHT variiert.“ Die vorregistrierte H1 (Spearman(d, Rang) > 0) hält; geändert ist nur die Formulierung — **gehört so ins Tagebuch** (Änderung nach Blick auf die Daten).

Modellvergleich auf Zellmitteln (Teilnahmezahl; RMSE / Leave-one-out; nachträglich, beschreibend):

| Modell | k | RMSE | LOO |
|---|---|---|---|
| konstant | 1 | 0,976 | 1,019 |
| additiv c + a·ln d + b·ln C | 3 | 0,447 | 0,531 |
| multiplikativ c·d^a·C^b | 3 | 0,364 | 0,427 |
| min(a·ln d, b·ln C) | 2 | 0,451 | 0,495 |
| c + min(a·ln d, b·ln C) | 3 | 0,286 | 0,345 |
| **Produkt c + e·ln d·ln C** | **2** | 0,248 | **0,275** |
| Wechselwirkung (additiv + ln d·ln C) | 4 | 0,220 | 0,263 |

**Befund:** Das Minimum schlägt additiv und multiplikativ, verliert aber gegen ein schlichtes Produkt mit 2 Parametern. Beim Entropie-Rang dieselbe Reihenfolge. „Sieht aus wie ein Minimum“ ist damit **nicht** gestützt; tragfähig ist „d und C wirken zusammen (Wechselwirkung), nicht additiv“. 24 Zellen tragen keinen Test zwischen sieben Modellen.

---

## 4. Neu gefundene Probleme

1. **Q-009: alle drei Arme auf allen drei Stufen am unteren Gitterrand** (lr 0,0003 = Gitterminimum). Aus den Rohdaten (`lr` je Zeile; Q-009 speichert keine `lr_map`). Ein weiterer Grund, warum Q-009 nicht verwertbar ist — im Tagebuch nicht erwähnt.
2. **B-012 Fashion zitiert einen asymptotischen p-Wert bei n=7:** Tagebuch „ρ=+0,857, p=0,0137“ (`trajectory_dose.py:287`). Exakter Permutationstest: zweiseitig **p=0,0238**, einseitig 0,0119. Aussage hält (< 0,05), die Zahl ist falsch. Dieselbe Fehlerklasse wie Q-009/Q-011.
3. **B-010: gespeicherte Auswertung in der Rohdatei ist noch die kaputte (NaN, „Holm p=0,0000 signifikant“)**. Die korrigierte Auswertung existiert nur im Code; wer `analysis` aus der JSON liest, bekommt Unsinn. Die Tagebuchzahlen stimmen (Audit OK).
4. **Wiederaufnahme-Muster in `sprachmodell.py`** (Vorlage laut Handoff) prüft weder Herkunft noch Budget. Der verworfene zweite Q-011-Anlauf hat **identische** Konfiguration zum dritten (max_samples, epochs, lr_grid) — der Budgetunterschied ist nirgends gespeichert. Deshalb neuer Schlüssel `suchbudget` + namentliche Zulassung der fünf Dateien des dritten Anlaufs. Gilt sinngemäß für T-002.
5. **Fremde Nicht-Rohdatendatei im Rohdatenordner:** `A/results/experiments/Q-020-pilot_wiederaufnahme.json` (24.09., 09:33, anderer Agent) hat kein `experiment_id` und brachte `qire_ai.audit` mit KeyError zum Absturz. Im Audit abgefangen. Empfehlung an den Urheber: Zustandsdateien nicht als `*.json` neben die Rohdaten legen (Unterordner oder andere Endung) — andere Werkzeuge globben dort `*.json`.
6. **`zeitreihen.sweep_lr` (Vorlage für die Gitterverlängerung)** zählt `ERWEITERUNGEN` insgesamt, nicht je Richtung (Kommentar sagt „je Richtung“), und verlängert bei flacher Suche (Kollaps) sinnlos nach unten. In `unitarity_depth` behoben, in `zeitreihen.py` (nicht meine Datei) offen.
7. **`B/spiegel_aktualisieren.ps1` kopiert aus dem ARCHIV**, nicht aus A. Ein Aufruf überschreibt `B/code/audit.py` und `B/code/intrinsic_dimension.py` mit den alten Fassungen. Nicht geändert (Entscheidung, welcher Ordner die Quelle ist).
8. `unitarity_depth.py` meldete bisher nur den **unteren** Rand als `lr_edge` und nie den oberen nach ausgeschöpfter Verlängerung; Kopf sprach von „verdoppelt“, gerechnet wurde ×3. Behoben.
9. Q-011, dritter Anlauf, **Messzellen** (nicht nur Tuning): `plain` bei Tiefe 8 = exakt 0,1000 → auch im Messlauf kollabiert. Für die Auflösungsprüfung irrelevant (nur kleinste Tiefe), aber `q_summary` meldet es jetzt als „Kollaps“, nicht als Randlage.

---

## 5. Die `stats.spearmanr`-Fundstellen außerhalb meiner Dateien

Stand heute 10 Aufrufe in 8 Modulen (Tagebuch 27.08.: „18 Stellen“; B-014 war ein Teil davon, meine drei Dateien rechnen jetzt über Zellmittel bzw. exakt). Nicht geändert.

| Fundstelle | Worüber | Pseudoreplikation? | Zitierte Zahl? | Einstufung |
|---|---|---|---|---|
| `quantum_sim/breadth_battery.py:271` | Vorteil je Datensatz gegen Eingabedim./Klassenzahl (n = Datensätze, ≥4) | nein | nein (Q-007 ohne Rohdaten) | asympt. p bei kleinem n; explorativ markiert |
| `qire_ai/benchmark_tt_streaming.py:97` | Vorhersage vs. Messung je Konfiguration | nein | ja: „ρ=0,42, p=0,10“ (2. Aug) | asympt. p; Aussage „nicht signifikant“ unkritisch; keine Rohdaten |
| `qire_ai/compression_regimes.py:193` | verworfene Spektralenergie vs. Genauigkeitsverlust, 12 Zellmittel (3 Datensätze × 4 Ränge) | nein (Zellmittel), aber über Datensätze gepoolt (Schwierigkeit konfundiert) | unklar — Tagebuch Z. 354 „ρ=−0,549, p=0,003“ passt nicht zu n=12 (dafür bräuchte es ≈27 Punkte) → **Herkunft offen, möglicherweise pseudorepliziert** | prüfen, bevor zitiert |
| `quantum_sim/isotime_accuracy.py:334` | Abstand über 3 Clusterstufen (Q-009) | nein | ja, bereits als Fehler dokumentiert (p=0,0000 statt exakt 0,3333) | geht in Holm + Urteil ein; Q-009 ohnehin nicht verwertbar |
| `quantum_sim/width_accuracy.py:398` | P2-Trend über wenige Breiten | nein | nicht gefunden | asympt. p bei kleinem n; vor Nutzung auf exakt umstellen |
| `qire_ai/regularization_effect.py:122` | Überanpassung vs. Nutzen **je Seed**, über Datensätze/Konfigurationen gepoolt | **ja** (plus gemeinsamer Term −dense_test) | ja: „+0,170, p=0,016“ — im Tagebuch bereits zurückgezogen, Code warnt selbst | erledigt, nicht zitieren |
| `qire_ai/sample_efficiency.py:259` | B-010 PRIMÄR: Abstand über 6 Datenmengen | nein | ρ=−0,600 / −0,257 (Audit OK); p nicht zitiert | asympt. p bei n=6 in Holm; Urteil H0 ändert sich nicht |
| `qire_ai/trajectory_dose.py:260` | B-012 Effektivrang über 10 Epochen (Epochenmittel) | nein | ja: „ρ=+0,988, p<0,0001“ | hält auch exakt |
| `qire_ai/trajectory_dose.py:287` | B-012 Genauigkeit über 7 Verzweigungszeitpunkte | nein | **ja: „ρ=+0,857, p=0,0137“ — exakt 0,0238** (Problem 2) | Zahl korrigieren |
| `qire_ai/trajectory_dose.py:308` | bestes k über Ränge | nein | nur Vorzeichen benutzt | unkritisch |

Befund: **Pseudoreplikation** nur noch in `regularization_effect` (bereits zurückgezogen) und ggf. an der ungeklärten Quelle von Tagebuch Z. 354. Das verbreitetere Problem ist der **asymptotische p-Wert bei n ≤ 7** (isotime, width, sample_efficiency, trajectory_dose:287, breadth).

---

## 6. Q-011 deskriptiv (nur zur Orientierung, NICHT zitierbar)

Aus `Q-011__20260827_230017.json`, 15 Seeds, volles Suchbudget, nicht vorregistriert ausgewertet (Tiefe 16 und fashion fehlen, Primärtest unmöglich):

| Tiefe | plain | orthogonal | spectral | orth − spec | Siege |
|---|---|---|---|---|---|
| 1 | 0,9409 | 0,9489 | 0,9463 | +0,0026 | 8/15 |
| 2 | 0,9376 | 0,9489 | 0,9472 | +0,0017 | 11/15 |
| 3 | 0,9344 | 0,9484 | 0,9458 | +0,0026 | 9/15 |
| 4 | 0,9197 | 0,9464 | 0,9451 | +0,0013 | 11/15 |
| 8 | 0,1000 | 0,9467 | 0,9423 | +0,0044 | 13/15 |

Nicht ins Tagebuch als Befund, nicht in `q_audit` (keine Tagebuchzahl).

---

## 7. Offene Punkte

- Tagebucheintrag nötig: B-014-Urteilsformulierung geändert (nach Datenblick), Modellvergleich (Produkt > Minimum), neue Problempunkte 1–3.
- Q-011 kann jetzt mit `python -m quantum_sim.unitarity_depth` (Wiederaufnahme) nur Tiefe 16 + fashion_mnist nachrechnen — Lauf **nicht** gestartet. Vorher Vortest `analyse()` wie gewohnt; der neue Code hat Test, aber noch keinen echten Lauf gesehen.
- Energierang 90/99 % für B-014: nur per Neulauf (Stunden).
- `zeitreihen.sweep_lr` (Problem 6), `trajectory_dose`-Zahl (Problem 2), Herkunft Tagebuch Z. 354, Spiegelskript (Problem 7) — außerhalb meiner Dateien.
- `CLAUDE.md` / `HANDOFF_AKTUELL.md` nennen noch „106 Zahlen“ für `q_audit` (jetzt 154) und den Aufruf `python -m qire_ai.audit` ohne `--daten`.
