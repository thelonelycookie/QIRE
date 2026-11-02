# f4: Geschichte zuspitzen, drei Kernaussagen für die schriftliche Arbeit (28.10.2026)

Quellen: 00_AUFTRAG (Welle 3), LABORBUCH bis I-045-v5, KERNAUSSAGE_2026-10-07, SYNTHESE (27.10.), konv_kandidaten, JUGEND_FORSCHT/GLIEDERUNG, JURY_UEBUNG, ZEITLEISTE. Nichts gerechnet. Wirkung überall klassisch (I-038), also "quanten-inspiriert", nie "Quantenvorteil".
Nicht im Repo (nur aus 00_AUFTRAG, nicht gegen Rohdaten geprüft): Ergebnisse I-046-v2, I-047, I-049, T-ternaer-butterfly. I-048, I-045-v5 und das Erkundungspaket laufen, ihre Zahlen fehlen.

## 0. Ergebnis in Kürze
- **Empfehlung: Variante A** ("Speicher, nicht Rechenzeit"), mit B als Schutzregel in jedem Vergleich und C als Teilkapitel.
- Grund: A ist als einzige zugleich (1) vom Nein-Teil schon belegt (I-045-v4, I-047, I-049 sind H0), (2) am Ziel "ressourceneffiziente KI" ausgerichtet, (3) bis zum Rechenstopp 12.12. schließbar. Bleibt I-048 aus, fällt A nicht um, es wird zu "nur Speicher, und nur Fastfood".
- **Wichtiger Fund beim Lesen (bitte prüfen):** Die Deutung von I-045-v3 H2 ("Vorsprung von D_z2 wächst") ist im LABORBUCH vorzeichenverkehrt. Rechnung aus der eigenen Tabelle in 1.1.

## 1. Lage seit der KERNAUSSAGE vom 07.10.
- Überholt: "I-045-v2 ist der erste Befund, in dem die Kombination mehr bringt als ihre Teile" (nur Frühregime, d = 256), "Fastfood spart Rechenzeit" (nur Einzelschicht ab N ≈ 1k, I-041), "Strukturkopf schlägt dichten Kopf" (Budgeteffekt, I-044-v3).
- Gilt weiter: I-038 (Zelle ≡ feste Mischung), I-039-v2 (kein Born-Effekt bei gleichen Bits), I-042-r CIFAR H0, I-046 (Zielkonflikt flach).
- Neu und negativ (H0): I-045-v4, I-047, I-049, I-046-v2, T-ternaer-butterfly. Neu und positiv: nur Born-Quadrat in I-047 H3 (+0,040 bpc gegen GELU).

### 1.1 Vorzeichenfund I-045-v3 (aus Tabelle im LABORBUCH 26.10.)
V = D_z2 minus Sz in bpc; V > 0 heißt Sz besser. Aus der Tabelle:
| d | V bei T=1000 | T=4000 | T=16000 | Δ = V(16000) minus V(1000) |
|---|---|---|---|---|
| 256 | +0,067 | -0,051 | -0,106 | -0,173 |
| 512 | -0,342 | -0,376 | -0,024 | +0,318 |
- Mittel über beide d: Δ = +0,0725, exakt die berichtete "+0,073". Das ist ein **Zuwachs für Sz**. Das LABORBUCH liest es als "Vorsprung von D_z2 wächst" und folgert "Sz überholt, Frühregime".
- Richtig: bei d = 256 verliert Sz mit mehr Training (Frühregime-Muster), bei d = 512 holt Sz auf (Rückstand 0,342 auf 0,024 bpc). Gesamtmuster gemischt, Lernrate nur bei T = 4000 gesucht.
- Folge: B verliert eine Säule, A gewinnt eine (Sz nur 0,024 bpc hinter D_z2 mit 0,5 % der FFN-Parameter). Bitte Laborbuch-Eintrag nachtragen (Korrektur, kein Löschen).

---
## Variante A: Speicher, nicht Rechenzeit
**Titel:** "Speicher statt Rechenzeit: Was quanten-inspirierte Basiswechsel in kleinen neuronalen Netzen wirklich sparen"

**Abstract (5 Sätze):**
1. Quanten-inspirierte Bausteine (Hadamard-Basiswechsel, feste Fastfood-Mischung, Born-Quadrat z²) werden in vorregistrierten Läufen gegen gleich große klassische Gegner geprüft, getrennt nach Bytes, Rechenoperationen und Trainingsaufwand.
2. Die Wirkung ist klassisch erklärbar: Die gelernte QIRE-Zelle ist einer festen Mischung gleichwertig (I-038), eine Zufalls-Orthogonale ist so gut wie Hadamard.
3. Bei gleichem Rechenaufwand verlieren die Strukturschichten im Byte-Sprachmodell (Fastfood -0,08 bpc, Tensornetz etwa -0,06, Hourglass mit Rekurrenz -0,055 gegen flach), und gemessene Rechenzeit sparen sie auf Standard-Hardware nicht.
4. Beim Speicher dagegen liegt eine Fastfood-FFN mit 0,5 % der Parameter nur 0,015 bis 0,024 bpc hinter der vollen dichten Schicht, [I-048: Hadamard + 2/3 Bit gegen kleines dichtes Modell gleicher Bytes, Zahl einsetzen].
5. Das Ergebnis ist ein ehrliches Speicher-Diagramm mit Negativbefunden, ein quanten-inspirierter Speichervorteil und kein Quantenvorteil.

**Tragende Befunde**
| Aussage | Beleg | Zahl | Stärke |
|---|---|---|---|
| Zelle ≡ feste Mischung | I-038 H-A/B/C | +0,18/+0,10 pp, TOST ±0,5, 20 Seeds | belegt |
| Gleiche FLOPs: Struktur verliert | I-045-v4 H1/H2 | -0,081 / -0,088 bpc, n = 8, T=16000 | belegt (H0) |
| Sz fast so gut wie volle FFN bei 0,5 % Parametern | I-045-v4, v3 | 2,178 gegen 2,163 (v4); 2,177 gegen 2,153 (v3); 43 k gegen 8,4 M | beschreibend, aus Registrierungen |
| Parametergleich schlägt Sz den dichten Gegner | I-045-v4 | 2,178 gegen 2,270 (Dk, h=10), aber 58× FLOPs; Dk ist schwacher Gegner | beschreibend |
| Tensornetz spart Parameter, nicht Qualität/Zeit | I-047 H0 | etwa -0,06 bpc; Trainingszeit etwa 2,4× | belegt (Raven 28.10.) |
| FLOP-Sparen (Hourglass+LRU) hält nicht | I-049 H0 | -0,055 bpc bei T=16000 | belegt (Raven 28.10.) |
| Struktur auf Rohpixeln schwer: Verlust | I-042-r CIFAR | -6,3 pp, 15 Seeds | belegt |
| Fashion: feste Struktur ≡ dichte 16,8 M | I-042-r H1 | +0,23/+0,26 pp, 15 Seeds | belegt, aber 4-5 Epochen |
| Zeit: nur Einzelschicht, große N | I-041 | schneller ab N=1024 (Batch 1); 89× weniger Ops | belegt (CPU) |
| Ideal 9-12× weniger Ops, Software 6,7-8× langsamer | w12 (Erkundung) | Torch-FWHT Batch 1 | Hinweis |
| Rotation + 2 Bit gleiche Bytes | r1/w4 (Erkundung) | 98,30 gegen 97,90 % (+0,4 pp, 3 Tuning-Seeds); Orth = Had 98,21 | Hinweis, **noch kein Befund** |
| Sehr kleiner Speicher: Clifford-Zelle vorn | T-ternaer | cliff0 (64 Bit) +6,5 pp gegen ternaer8_quad | schwach (n gering) |

**Lücken und schließende Läufe (alle vor Rechenstopp 12.12.)**
1. **I-048 (läuft, etwa 2 h Wand auf Raven):** H1 had gegen klein bei gleichen Bytes, H2 had ≡ orth, H3 Entropie-Rang. Das ist der positive Kern; ohne ihn ist A nur Fastfood-Speicher.
2. **Byte-Pareto der FFN (neu, vorregistrieren, Raven, etwa 150-250 Kernstunden):** bpc gegen FFN-Bytes bei T=16000, d=512, 8 Seeds, Lernrate je Arm und T. Arme: dichte h-Reihe (z. B. 10/20/40/80/160/580), Niedrigrang, Sz, TT, Sz mit 2 Bit. Schließt die Lücke zwischen Dk (h=10) und Df (h=580); heute gibt es auf der Speicherachse nur zwei Punkte.
3. **I-048-Folge aus E1/E2/E3/E5** (Gruppenskalen, Trainings-FLOP-Gleichheit, QAT, Breitenskalierung): erst Rauchtest auf Tuning-Seeds, dann Vorregistrierung. E2 ist die wichtigste Juror-Frage ("kleines Modell länger trainieren?").
4. **I-045-v5** (ideal gezählte Operationen, Lernrate je T): sichert die FLOP-Zählung gegen den Einwand "Fastfood falsch gezählt".
5. **Zeit gegen Bytes** (K18, Mikrobenchmark im Container, etwa 8 min): macht "nicht Rechenzeit" messbar und kann sie umdrehen, falls Niedrigbit bei Batch 1 bandbreitenbegrenzt schneller ist. Beides wäre ein Ergebnis.
6. Energie wurde nie gemessen; im Text ausdrücklich ausnehmen.

**Jury-Fragen**
- "Rotation vor Quantisierung ist QuIP/QuaRot. Was ist neu?" Antwort: nicht neu, zitiert; Beitrag ist die vorregistrierte Gleiche-Bytes-Messung mit Zufalls-Orthogonale und Fastfood als Kontrolle.
- "Warum nicht einfach ein kleineres Modell?" I-048 klein-Arm (Form getunt), E2.
- "Wo ist Quanten, wenn Zufalls-Orthogonale genauso gut ist?" Ehrlich: nur das Bild des Basiswechsels (Hadamard-Gatter); Hadamard ist billiger (n log n gegen n²).
- "Ist 2 Bit wirklich 2 Bit?" quant_rows(bits=2) ist ternär (1,58 Bit), als 2 Bit gezählt, also gegen die Methode gerechnet; offenlegen.
- "Gilt es bei großen Modellen?" Offen (I-043-1B, MPI). "Energie?" Nicht gemessen.

**Risiko (mittel):** (1) I-048-Vorsprung klein (Rand 0,2 pp / 0,02 bpc) oder LM nur HU; (2) geringe Neuheit, per CLAUDE.md erlaubt, aber Jury-Wirkung gedämpft; (3) negative Hälfte des Titels kann wie Aufgeben klingen, Formulierung "spart X, nicht Y" ist aber ein Befund; (4) Quantenbezug dünn.

---
## Variante B: Struktur hilft nur im Frühregime
**Titel:** "Wenn der Vorsprung mit dem Training verschwindet: Wie lange Strukturschichten und Effizienztricks wirklich helfen"

**Abstract (5 Sätze):**
1. Mehrere Effizienz-Bausteine, die in Kurztests klar vorn lagen, verlieren ihren Vorsprung, sobald Training und Lernrate fair gewählt werden.
2. Beim Strukturkopf auf CIFAR-Merkmalen kippt der Vorsprung gegen den linearen Kopf von +0,37 pp (4 Epochen) auf -0,51 pp (24 Epochen) (I-044-v3).
3. Hourglass mit Rekurrenz schlägt im Rauchtest (190 Schritte) den flachen Gegner um etwa 0,33 bpc, liegt bei 16 000 Schritten aber 0,055 bpc zurück (I-049).
4. Auch der Vorteil des Born-Quadrats gegen GELU schrumpft mit der Trainingslänge (d=256: 0,118 auf 0,013 bpc; d=512 dreht er zu -0,047), bei Lernrate nur bei T=4000 gesucht.
5. Daraus folgt eine Prüfregel für Effizienzbehauptungen, [Kreuzungspunkt T*(d) mit Konfidenzband einsetzen, wenn die T-Reihe vorliegt].

**Tragende Befunde**
| Aussage | Beleg | Zahl | Stärke |
|---|---|---|---|
| Kopfvorsprung kippt | I-044-v3 H1/H3 | +0,37, -0,77, -0,51 pp bei 4/12/24 Ep.; H3 p=0,0003 | belegt, Lernrate nicht je Epochenzahl |
| Hourglass+LRU: Frühregime | r2 gegen I-049 | 3,372 gegen 3,703 bei 190 Schritten; H0 bei T=16000 | belegt (Raven), H2-Wert prüfen |
| z² gegen GELU schrumpft | I-045-v3 (Tabelle) | d=256: 0,045/0,118/0,013; d=512: 0,188/0,081/-0,047 | beschreibend, Lernrate bei T=4000 |
| Sz gegen D_z2 | I-045-v3 | **gemischt**, siehe 1.1 | stützt B nur bei d=256 |
| Alte Kap. 7/8-Ergebnisse | I-042/I-044 | alle 4-5 Epochen | pauschal unter Frühregime-Vorbehalt |
| Gegenbeispiel | I-047 H3 | z² schlägt GELU um 0,040 bpc bei T=16000 | Vorsicht: Regel hat Ausnahmen |

**Lücken und schließende Läufe**
1. **I-045-v5 (läuft):** Lernrate je T, Train-bpc, halbe Datenmenge (trennt Optimierung von Datenregime). Ohne v5 bleibt der Einwand "Lernrate war falsch".
2. **Frühregime-Kurve (neu, Raven, etwa 350-450 Kernstunden):** Sz, D_z2, D_gelu, d=256, T in {250, 500, 1000, 2000, 4000, 8000, 16000, 32000}, Lernrate je T, 8 Seeds; daraus Fit Δ(T) mit Bootstrap-Band und T*. Nur so ist "quantitatives Gesetz" mehr als drei Punkte.
3. **I-044-v4** (Lernrate je Epochenzahl, Augmentierung): schließt den Vorbehalt von I-044-v3.
4. Zweiter Datensatz/zweites Regime (bisher nur UD_German-GSD und MNIST/Fashion/CIFAR): Verfügbarkeit **zu prüfen**.

**Jury-Fragen**
- "Gesetz aus 3 bis 8 Punkten?" Ehrlich: Regel mit Band, kein Naturgesetz; Wortwahl im Titel vermeiden.
- "Ist Frühregime nicht einfach schlechte Lernrate?" v5, I-044-v4.
- "Und bei d=512 kehrt sich Sz um?" (Fund 1.1) Antwort braucht die Kurve für d=512, die im Plan oben nicht enthalten ist (zusätzliche Kosten etwa 3×).
- "Wozu hilft das Gesetz praktisch?" Nur als Vorsicht und als Argument für knappe Budgets (Edge-Training).
- "Was hat das mit Quanten zu tun?" Nichts; B ist Methodik.

**Risiko (mittel-hoch):** (1) Säule I-045-v3 trägt nur halb (1.1); (2) "Gesetz" ist Überversprechen, die Kurve kostet viel und kann "kein einheitliches Muster" ergeben; (3) rein negativ, zum Projektziel (baue eine effiziente KI) passt nur die Randbemerkung; (4) Quantenbezug null. Stärke: gut verteidigbar, hohe Selbstkritik-Punkte.

---
## Variante C: Das Born-Quadrat als Aktivierung
**Titel:** "Das Born-Quadrat als Aktivierung: z² gegen GELU, squared ReLU und |z| bei gleichen Operationen"

**Abstract (5 Sätze):**
1. Die Born-Regel liefert Wahrscheinlichkeiten als Betragsquadrat; als Aktivierung z² in neuronalen Netzen ist dieselbe Idee bekannt (squared ReLU, Primer 2021).
2. In kontrollierten Läufen schlägt z² die lineare Auslese deutlich (+2,2 / +1,3 pp, I-038) und den Pixel-Softmax, ist der besten klassischen Nichtlinearität |z| aber nur gleichwertig bis knapp besser (MNIST +0,35 pp, Fashion 0,00; I-039-v2 gleichwertig).
3. Im Byte-Sprachmodell liegt z² bei knappem Training vor GELU (bis 0,19 bpc bei d=512, T=1000), bei T=16000 fällt der Abstand auf 0,013 bpc (d=256) und kehrt sich bei d=512 um (-0,047).
4. In der Tensornetz-FFN ist z² auch bei T=16000 um 0,040 bpc besser als GELU (I-047), [Aktivierungsstudie: Ergebnis einsetzen].
5. z² ist eine billige, quanten-inspirierte Aktivierung auf GELU-Niveau mit Instabilitätsrisiko bei hoher Lernrate, kein allgemeiner Born-Vorteil.

**Tragende Befunde**
| Aussage | Beleg | Zahl | Stärke |
|---|---|---|---|
| Quadrat gegen linear | I-038 H-E, T-ternaer | +2,21/+1,27 pp; +6,2 pp (ternär) | belegt |
| Quadrat gegen |z| | I-038 H-G, I-039-v2, I-041 H3, I-042 H2 | +0,35 / 0,00; -0,39 gleichwertig; wechselt je Aufgabe | belegt: klein bis null |
| z² ≈ relu² | I-045-v2 | 3,478 gegen 3,485 bpc | belegt (Primer-Effekt) |
| z² gegen GELU, LM | I-045-v2 H3, v3, I-047 H3 | +0,043 (v2); schrumpft/kehrt um (v3); +0,040 (I-047) | gemischt |
| Reell z² gegen Nur-Phase | r3 (Erkundung) | 95,83 gegen 95,44 % | Hinweis |
| Born-Attention (q·k)² | w3 (Erkundung) | kein bpc-Vorteil | Hinweis (negativ) |

**Lücken und schließende Läufe**
1. **Aktivierungsstudie (neu, Raven, etwa 400-500 Kernstunden):** GELU, relu², z², Born-normiert (K09), Hill (K20); d in {128, 256, 512}, T in {4000, 16000}, Lernrate je (Arm, d, T), 10 Seeds; dazu MNIST/Fashion/CIFAR-MLP. Auch I-045-v5 isoliert die Aktivierung nicht (Df_z2 gegen D_gelu mischt Breite und Aktivierung).
2. K09/K20 vorab als Erkundung (je etwa 9 min, Tuning-Seeds).
3. Zweiter Text-Datensatz (zu prüfen).
4. Literaturvolltext "squared ReLU", Primer (So et al. 2021), bisher nur Suchauszüge.

**Jury-Fragen**
- "Das ist Primer 2021, wo ist die Neuheit?" Keine; der Beitrag ist die Kontrolle (Betrag, ReLU², GELU, Lernrate je Arm) und die Grenze (Instabilität, Frühregime).
- "Warum heißt z² dann Born-Regel?" Der Quantenbezug ist Interpretation (|Amplitude|²), nicht Mechanismus (I-038).
- "Effekt von 0,04 bpc: praktisch relevant?" Kaum; ehrlich benennen.
- "Warum kehrt es bei d=512 um?" Lernrate nur bei T=4000 gesucht; v5/Aktivierungsstudie nötig.

**Risiko (hoch):** Neuheit null, Effekt klein und je Aufgabe wechselnd, trägt sich nicht als Hauptaussage. Als Teilkapitel (Kap. 6) gut und billig.

---
## Vergleich
| | A Speicher | B Frühregime | C Born-Quadrat |
|---|---|---|---|
| Nein-Teil schon belegt | ja (v4, I-047, I-049) | teils | teils |
| Positiver Teil | I-048 (läuft) | Methodenregel | klein |
| Passt zu Ziel "effiziente KI" | ja | schwach | schwach |
| Quantenbezug | dünn (Basiswechsel) | keiner | Born-Name |
| Neue Läufe bis 12.12. | Byte-Pareto, I-048-Folge | T-Kurve (teuer) | Aktivierungsstudie |
| Risiko | mittel | mittel-hoch | hoch |
| Jury-Wirkung | solide, ehrlich | Selbstkritik-Punkte | schwach |

## Empfehlung und Zeitplan
**A als Kernaussage.** B bleibt als Prüfregel in Kap. 4 und 8 ("jeder Vergleich bei gleichen Bytes/FLOPs und Lernrate je Budget", Lehre aus I-044-v3, I-049). C wird Kap. 6, ohne eigene neue Läufe, nur mit K09/K20-Erkundung.
- Bis 05.11.: I-048 und I-045-v5 auswerten; Vorzeichenfund 1.1 ins LABORBUCH; Rauchtests E1/E2/E3/E5, K18 (Tuning-Seeds, ohne Freigabe).
- Bis 12.11.: Vorregistrierung Byte-Pareto und I-048-Folge, **Maltes Freigabe nötig** (CLAUDE.md), Lauf auf Raven.
- 16.11.: Schreibbeginn (Kap. 1-4 hängen nicht von A ab). Titel/Kurzbeschreibung für die Anmeldung (intern 23.11., hart 30.11.) so formulieren, dass sie bei beiden I-048-Ausgängen stimmen: "Welche Ressourcen sparen quanten-inspirierte Bausteine wirklich?"
- Entscheidungsregel: Fällt I-048 H1 durch (HU/H0), lautet die Aussage "Speicher nur durch feste Struktur (Fastfood), nicht durch Rotation/Niedrigbit"; Rechenzeit und Genauigkeit wie gehabt negativ.
- 12.12.: Rechenstopp. Spätere Läufe nur als Zusatz am Stand.

## Folgen für GLIEDERUNG und JURY_UEBUNG
- Kap. 6 Zeile I-045-v2 und Kap. 10 sind überholt (v3, v4, v5); Kap. 7 Zeile "Fastfood spart Rechenzeit" und Tabelle KERNAUSSAGE §2 einschränken auf "Einzelschicht, N ≥ 1k, CPU".
- Jury-Antworten 15, 25, Z5 anpassen (Frühregime, I-045-v3/v4); neue Frage "Haben Sie Energie/Zeit auf realer Hardware gemessen?" aufnehmen.
- Offene Ehrlichkeitspunkte: Literatur (QuIP, QuaRot, Primer, Oseledets) steht im LABORBUCH "aus dem Gedächtnis"; Volltextprüfung vor Abgabe.
