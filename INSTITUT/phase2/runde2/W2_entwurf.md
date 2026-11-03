# W2 Entwurf: Byte-faire Rate-Güte-Messreihe im Byte-LM (Phase 2, Runde 2)

Status: **Entwurf, keine Vorregistrierung.** Ableitung aus WISSEN/wege.md (W2 + Nachtrag), S-002/S-003/S-004/S-009, Z3, `runde1/w6_vortest2.py`. Der Vortest (`w2_vortest.py`, `W2_vortest.md`) ist Erkundung auf Tuning-Seed 950.

## 1. Frage und Zielbild
Wie gut ist jeder Gewichtsquantisierer als **Kurve** bpc gegen Bytes (Indizes + Skalen + Codebuch + Vorzeichen + fp16-Rest), über 1,5–4 bpw und ≥ 4 Trainingslängen, und wo liegt die klassische Obergrenze (Gegnerliste S-002)? Ergebnis „alle gleich“ ist zulässig und publizierbar.
Kernidee: **Kurven statt Punkte.** Nur ganzzahlige Indexbits sind möglich, g16/g32 und Rotation landen dadurch auf verschiedenen bpw. Ein Vergleich „auf ±5 % byte-gleich“ (I-050) verschenkt das. Stattdessen: Kurve je Arm, Vergleich durch Interpolation (bpc linear in log-Bytes) bei festen Zielbytes, Streuung über Seeds.

## 2. Modell und Trainingslängen
- Byte-LM wie Z3 (d = 128, 4 Lagen, FFN 512, Kontext 128, UD_German-GSD), 786 432 quantisierte Gewichte; fp16-Rest (Einbettung, Positionen, LN, Biases) = 112 128 B in allen Armen, wird in „Bytes gesamt“ mitgezählt (nötig, weil das Mittelmodell einen anderen Rest hat).
- **T ∈ {2000, 4000, 8000, 16000}** (Vortest: T = 1000 liegt noch im Plateau, bpc 3,54, Quantisierung egal; T = 3000 bpc 2,53; T = 8000 etwa 2,14). Lernrate 2e-3 wie Z3, keine Nachjustierung je T (offen, siehe Risiken).
- Mess-Seeds frisch (z. B. 3100–3104, nicht 3000–3004 wegen I-050), 5 Seeds; Tuning-Seed 950 nur für Code-Prüfung.

## 3. Arme (PTQ-Gegnerliste), Raten
| Arm | Raten (Indexbit → bpw) | Skalenbytes | Stand |
|---|---|---|---|
| A1 Lloyd-1D, Zeilenskala fp16 | 2,3,4 → 2,1/3,1/4,1 | 0,09 bpw | Vortest läuft |
| A2 Hadamard + Lloyd-1D | 2,3,4 | + 1 Bit je Eingang | Vortest läuft |
| A3 g16, **8-Bit-Skala**, Lloyd-Buch | 1,2,3 → 1,5/2,5/3,5 | 0,5 bpw | Vortest läuft |
| A4 g32, 8-Bit-Skala | 2,3 (auch 1) → 2,25/3,25 | 0,25 bpw | Vortest läuft |
| A5 g16/g32 mit **fp16-Skala** | 1,2,3 → 2,0/3,0/4,0 | 1,0/0,5 bpw | aus w6_vortest2, bei 2 bpw kaputt (6,2 bpc) |
| A6 GPTQ-artige Rundung (OBS-Fehlerrückgabe, Hessian aus 64 Kalibrierfenstern) auf g32-8-Bit-Raster, ohne und mit Hadamard | 2,3,4 | wie A4 | **billig**: 512×128-Matrizen, geschätzt < 1 min je Modell. Neu zu bauen. |
| A7 Hadamard + E8 (Nachbau) | 2 (3 per Heuristik) | 0,09 bpw | Code in w6_vortest2 |
| A8 Hadamard + Orbit-8D (Clifford-artig) | 2 | + Buch 0,01 bpw | Code da (= I-050-Arm, nur Anschluss, nicht doppeln) |
| A9 MSB (Wertgruppierung, Blockgröße 64, kalibrierfrei) | 2,3,4 | Gruppen-Indizes zählen mit | **umsetzbar als Näherung** (Gewichte je Zeile nach Wert sortieren, in Gruppen teilen; Zuordnungsindex kostet Bytes, wird ausgewiesen). Vorher Volltext lesen (Abstract bisher nur bestätigt). Sonst streichen. |
| A10 Mittelmodell: kleineres Modell, fp16 **und** 4-Bit-Lloyd, Breite aus dem Byte-Budget jedes Zielpunkts (d in 8er-Schritten, f = 4d), **eigenes Training je T** | gleiche Bytes wie 2/3/4 bpw | – | Vortest: d = 48 fp16 (263 KB) |
| A11 fp16-Referenz groß | – | – | Obergrenze (1,68 MB) |

Nicht in W2 (zu teuer / eigener Weg): QAT-roh, Sparsity+4 Bit, Low-Rank, Monarch (W3), SSM (Modellwechsel, nur falls Zeit; wäre eigenes Rezept), LUT-Kerne. VBQ ist Training-von-Grund-auf (QAT), nicht PTQ, daher Folgeschritt.

## 4. Messgrößen
1. **Test-bpc** (Mess-Läufe; im Vortest nur Val) gegen **Bytes gesamt**.
2. **Aufgabenmetrik (S-009):** siehe 5.
3. Gewichts-SNR (Nebengröße), Kodierzeit, **Dekodier-Wallclock** (CPU, 1 Thread, Matvec 512×128, wie w6_vortest2; für ein echtes Maß muss es auf dem Ryzen laufen, torch-eager misst vor allem Python-Aufwand).
4. Kosten: Wallclock für Training und Quantisierung je Punkt.

## 5. Aufgabenmetrik (S-009): Befund aus dem Vortest
- Kandidat Kopieraufgabe (zufällige Buchstaben A, ‚ ‘, A; Maß: bpc und Top-1 in der zweiten Hälfte). **Das Grundmodell kann sie nicht.** Auch das T = 8000-Modell (Val 2,14) hat Kopie-Treffer 0,033 gleich Kontrolle (Zufall); mit natürlichem Text (40 Bytes wiederholt) ist Kopie = Kontrolle (0,54 / 0,54): **keine Induktion im Basismodell**. Mit 20 % Kopierbatches im Training ebenfalls nicht: T = 3000 und T = 8000 (Val 2,22) geben Kopie-bpc 4,70 = log2 26, Treffer 0,04 (Zufall). Eine Aufgabenmetrik braucht also ein Modell, das die Funktion gelernt hat.
- Folge, vorgeschlagene Entscheidung (Malte): (a) Kopierbeimischung bis T = 8000 getestet und **gescheitert**; ginge nur mit viel längerem Training oder größerem Modell (nicht Kleinmodell-tauglich), daher (b) **Aufgabe aus den Daten**, die das Modell kann: Next-Byte-Top-1-Genauigkeit und bpc nur auf **seltenen/schwierigen Positionen** (z. B. Wortanfänge, obere 10 % Verlust des fp-Modells) sowie Wort-Vervollständigung aus dem Wörterbuch der Trainingsdaten (Abruf-Ersatz). Beides ist keine echte Funktionsprobe wie in S-009, aber messbar. (c) Hilfsweise S-009 im Kleinmodell als „nicht prüfbar“ ausweisen (ehrlich, S-009 nennt das selbst als langweiligste Erklärung).
- Achtung: Beimischung ändert das Grundmodell (val-bpc höher: 2,71 statt 2,53 bei T = 3000), also gilt sie für die ganze Reihe oder gar nicht.

## 6. Statistik und Entscheidung (Vorschlag für die spätere Vorregistrierung)
- Gepaart über Seeds × T; Kurvenvergleich bei 3 Zielbytes (≈ 2, 3, 4 bpw) per Interpolation; Holm-Korrektur; Äquivalenzgrenze ±0,02 bpc (TOST wie I-050). „Bester Gegner“ je Zielpunkt wird ausgewiesen, nicht vorab gesetzt.
- Hauptaussage der Arbeit: Welche Familie liegt bei welchem Byte-Budget vorn, und ab wann (T) kippt das Mittelmodell?

## 7. Was ist neu gegenüber I-050?
I-050: **zwei** Raten (2, 3 bpw), **zwei** T (4000, 8000), 5 Arme, Frage „Orbit ≈ E8“ (TOST), Bytes nur ±5 %, nur Rotations-/VQ-Arme plus ein g16.
Neu in W2: (1) Kurven mit Zwischenraten (1,5–4 bpw) statt gleich-byte-Punkte; (2) ≥ 4 Trainingslängen; (3) Gegner, die I-050 nicht hat: **Mittelmodell fp16/4 Bit, GPTQ-artig, g32/g16 mit 8-Bit-Skala über mehrere Raten, MSB**; (4) Aufgabenmetrik und Dekodierzeit; (5) Fragestellung ist die Rangfolge aller Familien, nicht eine Äquivalenz. **Nicht doppeln:** Orbit-8D und E8 laufen in I-050; W2 übernimmt deren Ergebnisse als Arme (nur an den Raten und T, die I-050 liefert) und rechnet sie nicht neu.

## 8. Kosten- und Rechenzeitschätzung (Container, 4 Kerne, ein Lauf zur Zeit)
Vortest gemessen: Training 56–70 ms/Schritt (d = 128, 2 Threads), 22 ms/Schritt (d = 48); Quantisieren + Val-Auswertung je Punkt 3–6 s; Orbit-8D-Buch ≈ 90 s + Kodieren 20 s (2 bpw).
Je Seed: Training groß 30 000 Schritte ≈ 30 min (2 Thr.), ≈ 20 min (4 Thr.); Mittelmodelle (3 Breiten × 4 T) ≈ 10 min; ca. 35 Punkte × 4 T × 6 s ≈ 15 min; GPTQ-artig ≈ 5 min; MSB ≈ 5 min; Test-bpc und Aufgabenmetrik ≈ 5 min. **≈ 60–70 min je Seed mit 4 Threads, 5 Seeds ≈ 5–6 h**, plus ca. 1 h Codebau (GPTQ, MSB, Mittelmodell-Schleife, Test-Auswertung). Mit Orbit/E8 bei 3 bpw (Orbit-4D) + 1 h. Auf Maltes PC (6 Kerne) vermutlich gleich oder etwas schneller.

## 9. Risiken
- Plateau bei kurzem T (T = 1000: alles gleich), deshalb T ≥ 2000.
- Aufgabenmetrik (Abschnitt 5) ungelöst.
- Lernrate nicht je T/Modell getuned (Mittelmodell benachteiligt oder begünstigt); mindestens zwei Lernraten pro Breite auf Tuning-Seeds.
- Dekodierzeit in torch-eager ist kein Hardwaremaß.
- Ein Modell, ein Datensatz (deutsches Treebank, ~ 280 kB Text): Aussagen gelten für Kleinmodelle, nicht für LLMs.
