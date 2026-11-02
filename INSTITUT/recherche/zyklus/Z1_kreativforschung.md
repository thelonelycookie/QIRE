# Z1: Kreativitätsforschung und KI-Ideenfindung, Folgerungen für den LOOP (28.10.2026)

Kennzeichnung: **[S]** = Zahl aus einem Suchtreffer (Sekundärquelle oder Abstract, nicht am Volltext geprüft). **[G]** = aus dem Gedächtnis, nicht belegt, vor Zitat in der schriftlichen Arbeit prüfen. 20 Suchen verbraucht. Nichts gerechnet.

## (a) Befunde

### 1. Kreativitätsforschung Mensch

| Thema | Befund | Kernzahl / Beleg |
|---|---|---|
| Divergenz/Konvergenz (Guilford) | Divergentes Denken wird über Flüssigkeit, Flexibilität, Originalität, Ausarbeitung gemessen (Torrance-Tests, aus Guilfords Arbeit). Faktorstruktur psychometrisch umstritten. | Torrance 4 Skalen [S] |
| Menge erzeugt Qualität (Simonton) | Gleiche-Chancen-Regel: Treffer wachsen linear und stochastisch mit der Gesamtzahl der Werke. Test: mehr Antworten gingen mit höherer Kreativität einher. | Simonton 1997; Beaty u. a. 2015: r = 0,73, N = 246 [S] (ein Experiment, Urteil durch 3 Juroren) |
| Inkubation | Positiver Effekt, größer bei divergenten Aufgaben, größer bei längerer Vorbereitung, kleiner, wenn die Pause kognitiv fordernd ist. | Sio & Ormerod 2009: mittleres d ca. 0,29–0,32, divergent 0,65 [S] (Quellen uneinheitlich, im Original prüfen) |
| Analogie-Transfer (Gentner) | Strukturabbildung: Übertragen werden Relationen, nicht Oberflächenmerkmale. Fernanalogien sind selten, aber wertvoll. | Gentner 1983. Trefferseite lieferte nur den Literatureintrag, Inhalt [G] |
| Fernassoziation (Mednick) | Kreativität als Verknüpfung entfernter Elemente (Remote Associates Test). | Mednick 1962 [G]. Die Suche fand nichts Direktes |
| Adjacent possible (Kauffman, Johnson) | Erreichbar ist nur, was an das Vorhandene anschließt. Jede neue Sache öffnet neue Nachbarschaft. Heißt für uns: Bausteine, die wir selbst gebaut haben, erweitern den Suchraum. | Qualitativ, keine Kernzahl [S] |
| Kuhn / Paradigmenbruch | Anomalien häufen sich, bevor ein Paradigma kippt. Passt zu unserer Erfahrung: Die besten Funde der Welle 27.10. kamen aus eigenen Ungereimtheiten (10 Epochen, FLOP-Faktor 9–12). | Kuhn 1962 [G] |
| TRIZ (Altschuller) | Aus Patenten abgeleitet: 40 Prinzipien, 39 Parameter, Widerspruchsmatrix. Kern: Widerspruch benennen (A besser, B schlechter), dann gezielt auflösen statt Kompromiss. Die meisten Erfindungen sind niedrig (Stufe 1–2). | 400 000 Patente (andere Quelle: 1,5 Mio.); 77 % Stufe 1–2, ca. 19 % Stufe 3 [S], uneinheitlich |
| Uzzi 2013 | Wirkungsstärkste Arbeiten: überwiegend konventionelle Kombinationen plus einige atypische. | 17,9 Mio. Arbeiten; etwa doppelte Wahrscheinlichkeit hoher Zitierung; Teams 37,7 % häufiger mit Neukombination als Einzelne [S]. Kritik (Wang u. a. 2017): sehr neuartige Arbeiten erscheinen seltener in Top-Zeitschriften und werden später zitiert [S] |
| Park 2023 | Arbeiten und Patente werden über die Zeit weniger disruptiv (CD-Index von -1 konsolidierend bis +1 disruptiv). Ursache: engerer Rückgriff auf Vorwissen. | 45 Mio. Arbeiten, 3,9 Mio. Patente; Rückgang 91,9 % (Sozialw.) bis 100 % (Physik) 1945–2010 [S] |
| Teams vs. Einzelne (Wu 2019) | Kleine Teams stören/disruptieren, große entwickeln weiter. Kleine greifen weiter auf ältere Literatur zurück. | über 65 Mio. Arbeiten/Patente/Software, 1954–2014 [S] |

Folgerung Mensch: (i) Menge zählt, aber nur mit Auswahl danach (Simonton). (ii) Beste Wirkung: konventionelle Basis + wenige atypische Verbindungen (Uzzi), also nicht „alles neu“. (iii) Disruption entsteht eher klein und mit Rückgriff auf Fernes (Wu, Park). (iv) Inkubation hilft vor allem bei divergenten Aufgaben.

### 2. KI-gestützte Ideenfindung

| System | Befund | Kernzahl |
|---|---|---|
| Si, Yang, Hashimoto 2024 | Ideen von LLM wurden als neuer, aber etwas weniger machbar bewertet als Ideen von Experten. Selbstbewertung der LLM unzuverlässig, Vielfalt gering. | über 100 NLP-Forschende; Neuheit p < 0,05 [S] |
| AI Scientist (Lu u. a. 2024) | Ganze Pipeline Idee–Code–Paper–Review. | unter 15 US-Dollar je Paper, Hunderte Paper auf einem 8×H100-Knoten in ca. 1 Woche (Selbstangabe) [S] |
| AI Scientist v2 (2025) | Baumsuche; ein Paper bei Workshop angenommen, Rückzug vor Meta-Review. Sakana selbst: keines hätte die Hauptkonferenz erreicht. | Mittelwert 6,33 [S] |
| FunSearch (Romera-Paredes 2024) | LLM + automatischer Prüfer, durchsucht Programme statt Antworten. Prüfer fängt Halluzinationen ab. | Cap set Dimension 8: Größe 512 [S] |
| AlphaEvolve (2025) | Wie FunSearch, mit Archiv (Quelle: MAP-Elites- und Inselmodell, Sekundärquelle [S]). | 4×4 komplex: 48 Multiplikationen statt 49, verbessert 14 Matrixmultiplikationen [S]. Einschränkung: nur komplex, in Hardware-Studien 2026 oft teurer als naiv bei Breite bis 24 Bit [S] |
| ShinkaEvolve (Sakana 2025) | Neuheits-Rückweisung (Einbettung + LLM-Richter), neuheitsgewichtete Elternwahl, Bandit über Modelle. | Kreispackung SOTA mit 150 Proben [S] |
| Co-Scientist (Gottweis 2025) | Generieren, Debattieren, Weiterentwickeln; Rangagent mit Elo-Turnier. Behauptung „höhere Elo = eher richtig“ stammt aus Blog, nicht aus Volltext. | Zahlen nicht geprüft [S] |
| MAP-Elites (Mouret & Clune 2015) | Raster der besten Lösung je Merkmalszelle, statt einer Spitzenlösung. Schwäche: im Einzelvergleich weniger präzise (p < 0,01), weil Auswertungen auf viele Zellen verteilt. | [S] |
| Novelty Search (Lehman & Stanley 2011) | Zielfunktionen täuschen; reine Neuheitssuche schlug zielgetriebene in Labyrinth und Laufen. | [S] |
| Open-Endedness (Hughes 2024) | Definition über Neuheit und Lernbarkeit aus Sicht eines Beobachters. Beides nötig. | [S] |
| Homogenisierung | Doshi & Hauser 2024: KI-Ideen erhöhen individuelle Kreativität (v. a. bei weniger Kreativen), aber die Texte werden einander ähnlicher. Weitere Arbeiten: Modus-Kollaps, „Artificial Hivemind“ (verschiedene Modelle antworten ähnlich); höhere Temperatur reicht nicht. | 293 Geschichten, 600 Bewerter [S]. Gegenmittel in Funden: Personas, Temperatur-Filter, Portfolio-Diversität, Vendi-Score (Friedman & Dieng 2023) [S]; Arbeiten mit unbekannten arXiv-Nummern, vor Zitat prüfen |

Folgerung KI: Der Verifizierer ist der Hebel (FunSearch, AlphaEvolve): Ideen billig und viele, dann hart prüfen. Wo es keinen Verifizierer gibt (Si), sind LLM-Urteile unzuverlässig. Bei uns ist der Verifizierer der Rauchtest mit der fiesesten Kontrolle. Das ist gut. Vielfalt muss erzwungen werden, sie kommt nicht von selbst.

### 3. Grenzen gezielt überschreiten

Für diesen Punkt lieferte die Suche wenig Direktes. Der Teil unten ist Synthese **[G]** aus Punkt 1–2 und unserem Material, keine Belege.
- **Annahmen-Inventur:** Jede Methode in einer Liste „was wird stillschweigend angenommen?“ (z. B. FLOP als Kosten, Gewichte statisch, Training vor Test) und jede Annahme einzeln umkehren. Entspricht TRIZ-Widerspruch und Umkehrung im Kreativ-Entwurf.
- **Unmöglichkeitsbeweise als Wegweiser:** Jede Schranke (Rate-Distortion, untere Schranken) hat Voraussetzungen. Die Frage lautet: Welche Voraussetzung können wir verletzen (Struktur der Daten, gemeinsame Randinformation, Zufälligkeit, mehr Rechnen für weniger Bytes)? Rezept: Schranke -> Voraussetzungsliste -> je Voraussetzung eine Umgehungsidee.
- **Skalengesetze extrapolieren:** Anpassung der Kurve, dann prüfen, wo ein Bauteil die Kurve verschiebt statt nur entlang läuft. Nur gezählt, was bei gleichen FLOPs/Bytes die Kurve verschiebt (passt zu I-045-v4). Zahlen (Chinchilla, Präzisions-Skalengesetze) wurden in dieser Recherche nicht belegt: Suche lieferte sie nicht.
- **Kontrolle über die „langweilige Erklärung“:** Vor jedem Rauchtest eine Zeile: „Wäre der Effekt auch bei Zufallsvariante, gleichen FLOPs, gleichen Bytes, kürzerem Training da?“ Das ist bei uns schon Praxis (R1, R3) und sollte Pflicht-Feld werden.

## (b) Zehn Änderungen an LOOP.md

Aufwand: S = unter 1 h Skript/Text, M = ca. ein halber Tag, L = mehr. Nutzen sind Erwartungen, keine Messungen.

| # | Änderung | Erwarteter Nutzen | Kosten |
|---|---|---|---|
| 1 | **MAP-Elites-Archiv `ideen_archiv.json`** mit Achsen Quantenprinzip (8) × Ressource (FLOPs, Bytes, Daten, Energie, Latenz). Je Zelle nur die beste Idee (Elo/Ampel). Leere Zellen = Auftrag für den nächsten Zyklus. | Breite wird sichtbar und lückenlos; gegen Mittelmaß-Drift (Si, Doshi). | S–M: ein Skript, ein Agent pflegt |
| 2 | **Neuheitsprämie im Turnier** (nach ShinkaEvolve): Einbettungsähnlichkeit zu Archiv und LABORBUCH; zu ähnliche Ideen werden vor dem Turnier abgewiesen. Elo-Wertung + Aufschlag für Abstand. | Verhindert Duplikate und Homogenisierung; spart Rauchtests. | S: Einbettung + Schwelle; Schwelle muss kalibriert werden |
| 3 | **Verifizierer-getriebene Evolution:** Erkundungsskripte (`Z<n>_*.py`) bekommen ein gemeinsames Bewertungsmodul (gleiche FLOPs, gleiche Bytes, zwei Trainingslängen). Grüne Ideen werden vom Agenten mutiert (Hyperparameter, Variante), Auswahl nur nach Verifizierer. | FunSearch-Logik: Ideenmenge billig, Urteil hart; geringer Aufwand je Mutation. | M: Modul einmal bauen; Container-CPU begrenzt (4 Kerne) |
| 4 | **Pflicht-Fernanalogie:** Je Zyklus muss mindestens eine Idee je Top 5 aus einem fremden Fach stammen (Zufallspaper: Biologie, Akustik, Kompression, Thermodynamik), mit Strukturabbildung nach Gentner (Relationen benennen, keine Oberflächenähnlichkeit). | Zweitbester Weg der Welle 27.10. (RG -> Hourglass); gegen Nähe zum Bekannten. | S: ein Agent; Gefahr: Scheinanalogie -> Relationen-Check |
| 5 | **Malte-Kontingent fest:** Aus den Top 10 als Einzeiler markiert Malte ++/+/-; alle ++ gehen ungeprüft durch das Turnier in den Rauchtest (mindestens 1 von 5 Plätzen). Diktat genügt. | Menschlicher Geschmack und Veto; Gegengewicht zur KI-Mitte. | 5 min Maltes Zeit je Zyklus |
| 6 | **Annahmen-Inventur als eigener Agent** (Schritt 4 „Grenzen“ erweitern): Liste der stillschweigenden Annahmen von BitNet, QuIP#, QuaRot, Mamba usw., je Annahme eine Umkehrung. | Systematischer Weg zur Grenze (TRIZ-Widerspruch); billig, weil nur Text. | S |
| 7 | **Schranken-Umgehungs-Rezept:** Für jede Schranke aus Schritt 4 Voraussetzungen auflisten, je Voraussetzung eine Umgehungsidee. Ergebnis als Tabelle `zyklus/Z<n>_schranken.md`. | Unmöglichkeitsbeweis wird Wegweiser; Idee jenseits Bits-je-Gewicht. | S–M; Gefahr: Voraussetzung nur scheinbar verletzt -> Kontrolle 8 |
| 8 | **Pflichtfeld „langweiligste Erklärung“ + Kontrolle** je Idee vor dem Rauchtest (Zufallsvariante, gleiche FLOPs, gleiche Bytes, kurz/lang trainiert). Ein Rauchtest ohne dieses Feld zählt nicht. | Verhindert Scheingewinne (R1, R3). | S |
| 9 | **Inkubationsschritt** am Zyklusende (Schritt 8): Ein Agent liest verworfene Ideen neu mit dem Wissen des Zyklus; Pause von mindestens einem Zyklus zwischen Verwerfen und Wiedervorlage. Menschlicher Befund: Inkubation hilft vor allem divergent, d = 0,65 für divergente Aufgaben [S]. | Verworfenes bekommt eine zweite Chance; nutzt das gesammelte LABORBUCH. | S |
| 10 | **Verhältnis konventionell : atypisch steuern** (Uzzi): Jede Top-5-Liste enthält mindestens 3 Ideen mit konventioneller Basis (bekannter Baustein, stärkste Baseline) und 1–2 atypische Kombinationen, plus 1 Wildcard (Malte oder Fernanalogie). Dazu **Brier-Wert je Denkwerkzeug** in `ideen.json` (schon in KREATIVITAET §4 vorgesehen) und Agentenzahl nach Trefferquote anpassen. | Erwartung: höhere Trefferquote bei gleicher Breite; Lerneffekt je Werkzeug. | S; braucht mindestens 3–4 Zyklen Daten |

### Reihenfolge
Sofort (S): 8, 5, 4, 6, 9. Danach 2 und 1. Dann 3 (mehr Bauarbeit, höchster Hebel auf Dauer). 7 und 10 begleitend.

### Offene Punkte und Vorsichtsregeln
- Die Teamfrage (Wu 2019) spricht für kleine, getrennte Agentengruppen mit unterschiedlichen Denkwerkzeugen und gegen eine große Gruppe, die sich abstimmt. Das passt zum Plan „verschiedene Denkwerkzeuge statt Themen“.
- Der Elo-Rang im Turnier ist als Prüfer unsicher (Si: LLM-Selbstbewertung unzuverlässig; Co-Scientist-Korrelation nur aus Blog). Darum zählt der Rauchtest, nicht das Turnier, und der Brier-Wert der Vorab-Tipps kalibriert das Turnier.
- Nicht belegt in dieser Recherche (nur [G]): Kuhn, Mednick, Gentner-Inhalt, Chinchilla/Präzisions-Skalengesetze, Kauffman im Original, exakte AlphaEvolve-Ablationen.

## (c) BibTeX

Felder aus Treffern und Gedächtnis. Vor der schriftlichen Arbeit mit den Originalen abgleichen.

```bibtex
@article{uzzi2013,
  author  = {Uzzi, Brian and Mukherjee, Satyam and Stringer, Michael and Jones, Ben},
  title   = {Atypical Combinations and Scientific Impact},
  journal = {Science}, year = {2013}, volume = {342}, pages = {468--472},
  doi     = {10.1126/science.1240474}}

@article{park2023,
  author  = {Park, Michael and Leahey, Erin and Funk, Russell J.},
  title   = {Papers and patents are becoming less disruptive over time},
  journal = {Nature}, year = {2023}, volume = {613}, pages = {138--144},
  doi     = {10.1038/s41586-022-05543-x}}

@article{wu2019,
  author  = {Wu, Lingfei and Wang, Dashun and Evans, James A.},
  title   = {Large teams develop and small teams disrupt science and technology},
  journal = {Nature}, year = {2019}, volume = {566}, pages = {378--382},
  doi     = {10.1038/s41586-019-0941-9}}

@article{wang2017,
  author  = {Wang, Jian and Veugelers, Reinhilde and Stephan, Paula},
  title   = {Bias against novelty in science: A cautionary tale for users of bibliometric indicators},
  journal = {Research Policy}, year = {2017}, volume = {46}, pages = {1416--1436}}

@article{sio2009,
  author  = {Sio, Ut Na and Ormerod, Thomas C.},
  title   = {Does incubation enhance problem solving? A meta-analytic review},
  journal = {Psychological Bulletin}, year = {2009}, volume = {135}, number = {1}, pages = {94--120}}

@article{gentner1983,
  author  = {Gentner, Dedre},
  title   = {Structure-mapping: A theoretical framework for analogy},
  journal = {Cognitive Science}, year = {1983}, volume = {7}, pages = {155--170}}

@article{mednick1962,
  author  = {Mednick, Sarnoff A.},
  title   = {The associative basis of the creative process},
  journal = {Psychological Review}, year = {1962}, volume = {69}, pages = {220--232}}

@article{guilford1950,
  author  = {Guilford, J. P.},
  title   = {Creativity},
  journal = {American Psychologist}, year = {1950}, volume = {5}, pages = {444--454}}

@article{simonton1997,
  author  = {Simonton, Dean Keith},
  title   = {Creative productivity: A predictive and explanatory model of career trajectories and landmarks},
  journal = {Psychological Review}, year = {1997}, volume = {104}, pages = {66--89}}

@article{beaty2015,
  author  = {Beaty, Roger E. and Silvia, Paul J.},
  title   = {Quantity yields quality when it comes to creativity: a brain and behavioral test of the equal-odds rule},
  journal = {Frontiers in Psychology}, year = {2015}, volume = {6}, pages = {864},
  note    = {Autorenschaft und Wortlaut des Titels aus Trefferseite, Autoren prüfen}}

@book{kuhn1962,
  author    = {Kuhn, Thomas S.},
  title     = {The Structure of Scientific Revolutions},
  publisher = {University of Chicago Press}, year = {1962}}

@book{kauffman2000,
  author    = {Kauffman, Stuart A.},
  title     = {Investigations},
  publisher = {Oxford University Press}, year = {2000}}

@book{johnson2010,
  author    = {Johnson, Steven},
  title     = {Where Good Ideas Come From},
  publisher = {Riverhead Books}, year = {2010}}

@book{altshuller1984,
  author    = {Altshuller, Genrich S.},
  title     = {Creativity as an Exact Science},
  publisher = {Gordon and Breach}, year = {1984}}

@misc{si2024,
  author = {Si, Chenglei and Yang, Diyi and Hashimoto, Tatsunori},
  title  = {Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers},
  year   = {2024}, eprint = {2409.04109}, archivePrefix = {arXiv}, note = {ICLR 2025}}

@misc{lu2024,
  author = {Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  title  = {The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},
  year   = {2024}, eprint = {2408.06292}, archivePrefix = {arXiv}}

@misc{yamada2025,
  author = {Yamada, Yutaro and others},
  title  = {The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  year   = {2025}, eprint = {2504.08066}, archivePrefix = {arXiv}}

@article{romera2024,
  author  = {Romera-Paredes, Bernardino and others},
  title   = {Mathematical discoveries from program search with large language models},
  journal = {Nature}, year = {2024}, volume = {625}, pages = {468--475}}

@misc{novikov2025,
  author = {Novikov, Alexander and others},
  title  = {AlphaEvolve: A coding agent for scientific and algorithmic discovery},
  year   = {2025}, eprint = {2506.13131}, archivePrefix = {arXiv}}

@misc{lange2025,
  author = {Lange, Robert Tjarko and others},
  title  = {ShinkaEvolve: Evolving New Algorithms with Large Language Models},
  year   = {2025}, eprint = {2509.19349}, archivePrefix = {arXiv}}

@misc{gottweis2025,
  author = {Gottweis, Juraj and others},
  title  = {Towards an AI co-scientist},
  year   = {2025}, eprint = {2502.18864}, archivePrefix = {arXiv}}

@misc{mouret2015,
  author = {Mouret, Jean-Baptiste and Clune, Jeff},
  title  = {Illuminating search spaces by mapping elites},
  year   = {2015}, eprint = {1504.04909}, archivePrefix = {arXiv}}

@article{lehman2011,
  author  = {Lehman, Joel and Stanley, Kenneth O.},
  title   = {Abandoning Objectives: Evolution through the Search for Novelty Alone},
  journal = {Evolutionary Computation}, year = {2011}, volume = {19}, number = {2}, pages = {189--223},
  doi     = {10.1162/EVCO_a_00025}}

@inproceedings{hughes2024,
  author    = {Hughes, Edward and others},
  title     = {Position: Open-Endedness is Essential for Artificial Superhuman Intelligence},
  booktitle = {ICML}, series = {PMLR}, volume = {235}, pages = {20597--20616}, year = {2024}}

@article{doshi2024,
  author  = {Doshi, Anil R. and Hauser, Oliver P.},
  title   = {Generative AI enhances individual creativity but reduces the collective diversity of novel content},
  journal = {Science Advances}, year = {2024}, volume = {10}, number = {28}, pages = {eadn5290},
  doi     = {10.1126/sciadv.adn5290}}

@article{friedman2023,
  author  = {Friedman, Dan and Dieng, Adji Bousso},
  title   = {The Vendi Score: A Diversity Evaluation Metric for Machine Learning},
  journal = {Transactions on Machine Learning Research}, year = {2023}}
```
