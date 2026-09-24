# Quanten-Ideenkatalog — Agent `forschung_quanten`

Stand 24. September 2026. Grundlage: `BERICHT/00_REVIEW_BEFUNDE.md`,
`BERICHT/01_VERBESSERUNGEN_UND_ZUKUNFT.md`, `A/BEFUNDE_UND_RICHTUNGEN.md`,
`A/FORSCHUNGSIDEEN.md`, `A/ENTWURF_T_SERIE_KERNEL.md`, `A/LITERATUR_UND_ALLEINSTELLUNG.md`
sowie eigene Websuche (Titel/Link/Jahr stehen bei jeder Quelle; kein Volltext gelesen).

**Regel für dieses Dokument:** Ideen, die in `FORSCHUNGSIDEEN.md` oder
`LITERATUR_UND_ALLEINSTELLUNG.md` schon stehen, werden hier **nicht** wiederholt,
sondern nur ergänzt, wenn eine neue Quelle oder ein neuer Blickwinkel dazukommt
(so markiert). Alle 15 Karten sind neu. Jede Karte sagt ehrlich, wenn sie
„klingt gut, bringt aber nichts" ist. Nichts hier ist ein neues physikalisches
Phänomen — wo etwas nur eine mathematische Analogie zur Quantenmechanik ist,
steht das ausdrücklich dabei.

**Begriffe kurz:** *Klassisch simulierbar* = ein Quantenschaltungstyp, den ein
normaler Computer ohne Exponentialaufwand nachrechnen kann. *Strukturierte
Matrix* = eine Matrix, die sich mit wenigen Zahlen und einer festen Rechenregel
beschreiben lässt (statt jeden Eintrag einzeln zu speichern), z. B. Butterfly,
Kronecker-Produkt, zirkulant. *Kontrollarm* = ein Vergleichsmodell im selben
Lauf, das denselben Effekt ohne den zu prüfenden Mechanismus zeigen soll.

---

## Feld 1–2: Klassisch simulierbare Schaltungen & Tensornetze

### Idee 1 — Matchgates / freie Fermionen als reichhaltigerer Mischbaustein
**Was ist es.** Matchgate-Schaltungen sind eine Familie von Zweiqubit-Gattern auf
Nachbarqubits, die unter der Jordan-Wigner-Transformation exakt „freien
Fermionen" (Gauß-Zuständen) entsprechen. Sie sind klassisch in polynomieller
Zeit simulierbar — mehr Ausdruckskraft als reine Kronecker-Rotationen (QIREs
heutige CNOT-freie Schaltung), aber immer noch weit von der vollen,
exponentiell teuren Schaltung entfernt.

**Quellen.** „Matchgates and classical simulation of quantum circuits"
(ResearchGate-Zusammenfassung); „Fermionic Machine Learning", arXiv:2404.19032
(2024); „Improved simulation of quantum circuits dominated by free fermionic
operations", Quantum 8, 1549 (2024); MatchCake, PennyLane-Blog Juni 2026
(Benchmark-Werkzeug: „Wenn ein Modell sein freies-Fermionen-Gegenstück nicht
schlägt, treibt das Quantenverhalten nicht die Leistung").

**Einbau in QIRE.** Ein neuer Mischbaustein zwischen Kronecker-Rotation (0
Verschränkung) und CNOT-Ring (volle Universalität): eine Lage aus n−1
Matchgates auf Nachbarpaaren statt des CNOT-Rings. Parameterzahl bleibt klein
(ein Zusatzwinkel je Paar, also ~n zusätzliche Parameter), Simulation bleibt in
Torch als dichte 4×4-Blockoperation auf Paaren billig — kein neuer
Exponentialaufwand.

**Erwartete Wirkung.** Parameter: +n (klein). Zeit: vernachlässigbar auf CPU
und GPU, da n−1 lokale 4×4-Operationen. Genauigkeit: unklar — genau das ist die
offene Frage, die MatchCakes eigener Test aufwirft.

**Risiko und Einschätzung.** Der MatchCake-Befund oben ist eine Warnung, keine
Empfehlung: Wenn irgendein „quanteninspiriertes" Modell nicht sein
freies-Fermionen-Gegenstück schlägt, ist der Quantenanteil vermutlich
Beiwerk. Für QIRE heißt das: Matchgates sind ein **zusätzlicher, strengerer
Abgrenzungstest** für Verschränkung, nicht sicher ein Gewinn. Mittlere
Erfolgsaussicht, weil offen; mittlerer Neuheitswert, weil MatchCake selbst
schon fordert, genau das zu prüfen.

**Aufwand.** 2–3 Tage (Gatterimplementierung + Verifikation gegen den Simulator).

**Faire Kontrolle.** Gegen `hybrid_ry_only` (keine Mischung) und gegen den
vollen CNOT-Ring, mit TOST wie in Q-020 (b).

---

### Idee 2 — Encoder als Matrix-Product-Operator (MPO) statt Butterfly
**Was ist es.** Eine dichte Gewichtsmatrix wird als Kette kleiner Tensoren
(Bond-Dimension χ) faktorisiert — das Verfahren hinter CompactifAI. Anders als
Butterfly ist das Muster nicht FFT-artig fest, sondern lernt seine eigene
niedrigrangige Struktur entlang einer gewählten Indexaufteilung.

**Quellen.** „CompactifAI: Extreme Compression of Large Language Models using
Quantum-Inspired Tensor Networks", Multiverse Computing, arXiv:2401.14109
(2024) — 70–80 % weniger Parameter, 93 % weniger Speicher, 2 % Genauigkeitsverlust
nach kurzem Nachtraining, inzwischen auf Llama-3.1-8B/3.3-70B angewendet
(Multiverse-Computing-Pressemeldung 2025).

**Einbau in QIRE.** Direkte Ergänzung zu `BEFUNDE_UND_RICHTUNGEN.md` B2
(„Encoder als Butterfly"): den Encoder 784→256 als MPO mit Bond-Dimension
χ=2–8 statt als Butterfly bauen. Parameterzahl ~O(d·χ²·log d) statt d²;
bei χ=4 und d≈784·256 grob im Bereich weniger Tausend statt 200.704.

**Erwartete Wirkung.** Parameter: von 200.704 auf ~1.000–5.000 (χ-abhängig,
deutlich mehr als Butterflys ~1.000, aber mit mehr Ausdruckskraft). Zeit: CPU
und GPU beide unkritisch bei dieser Größe — kein Kernel-Problem wie bei
Butterfly. Genauigkeit: laut CompactifAI bei LLMs nur 2 Prozentpunkte Verlust
bei 70–80 % Kompression — optimistisches Signal, aber an einem ganz anderen
Modelltyp (Milliarden Parameter) gemessen.

**Risiko und Einschätzung.** Realistische, gut belegte Alternative zu B2 — aber
**Konkurrenz**, nicht Ergänzung: MPO und Butterfly lösen dasselbe Problem
(Encoder komprimieren) mit unterschiedlicher Struktur. Beide zu bauen und bei
gleicher Parameterzahl gegeneinander zu testen wäre der informativste
Vergleich. Kein Quantenphänomen — MPO ist eine klassische Tensorzerlegung,
„quanteninspiriert" nur historisch benannt.

**Aufwand.** 3–4 Tage (mit `tensorly` oder `quimb`, beide fertige
MPO-Bibliotheken in Python).

**Faire Kontrolle.** Bei exakt gleicher Parameterzahl gegen Butterfly-Encoder
und gegen eine Low-Rank-Faktorisierung (SVD-Kompression) antreten lassen.

---

### Idee 3 — MERA/Baum-Tensornetz als hierarchischer Bild-Encoder
**Was ist es.** MERA (Multi-Scale Entanglement Renormalization Ansatz) baut den
Encoder aus abwechselnden „Entwirrern" und „Isometrien" auf einem Binärbaum —
sie setzt Renormierungsgruppen-Coarse-Graining direkt in eine Netzarchitektur
um und passt inhaltlich zu Bilddaten mit hierarchischer Nachbarschaftsstruktur
(Pixel → Kanten → Formen).

**Quellen.** „Learning Relevant Features of Data with Multi-scale Tensor
Networks", arXiv:1801.00315; „Deep learning and the renormalization group",
arXiv:1301.3124 (Mehta/Schwab); Suchergebnis-Übersicht „Machine Learning for
Quantum Matter", arXiv:2003.11040, bestätigt die MERA-RG-Deep-Learning-Brücke
als etabliertes Forschungsfeld mit trainierbaren MERA-Modellen.

**Einbau in QIRE.** Ergänzt `BEFUNDE_UND_RICHTUNGEN.md` §5 „Tensornetze
jenseits von MPS" (dort nur als unbelegte IDEE): ein Baum aus log₂(784)≈10
Ebenen von 2-zu-1-Isometrien mit **geteilten** Gewichten je Ebene (dieselbe
Idee wie QIREs Winkelteilung, nur auf einen Baum übertragen) als Encoder-
Alternative zu MPO (Idee 2) und Butterfly (B2).

**Erwartete Wirkung.** Parameter: mit Gewichtsteilung O(log d) Isometrien statt
O(d) — potenziell die sparsamste der drei Encoder-Varianten. Zeit: Baumtiefe
log d, ähnlich billig wie Butterfly. Genauigkeit: unbekannt, MERA ist für
Bilder mit lokaler Struktur gebaut, MNIST ist dafür ein plausibler, aber
kleiner Testfall.

**Risiko und Einschätzung.** Dritte Variante desselben Problems (nach B2 und
Idee 2) — Gefahr, drei halbfertige Encoder statt einem fertigen zu haben.
**Ehrlich:** nur bauen, wenn B2 (Butterfly-Encoder) und Idee 2 (MPO) schon
laufen und noch Zeit bleibt; sonst als Fußnote für die Arbeit reichen
("mehrere tensornetzbasierte Encoder wurden erwogen").

**Aufwand.** 4–5 Tage (kein fertiges Baustein-Set wie bei MPO, muss aus
Grundoperationen gebaut werden).

**Faire Kontrolle.** Gegen Butterfly-Encoder (B2) und MPO-Encoder (Idee 2) bei
gleicher Parameterzahl.

---

## Feld 4: Quanteninformationstheorie

### Idee 4 — Klassische Schatten für reichere, billige Auslese
**Was ist es.** Klassische Schatten (Huang, Kueng, Preskill) schätzen aus
wenigen zufälligen Basiswechseln + Messungen viele verschiedene
Erwartungswerte (auch Produkte mehrerer Z/X/Y) mit logarithmischem statt
linearem Messaufwand — statt für jede gewünschte Größe extra zu messen.

**Quellen.** Huang, Kueng, Preskill, „Predicting Many Properties of a Quantum
System from Very Few Measurements", Nature Physics 2020, arXiv:2002.08953.

**Einbau in QIRE.** Ergänzt die eigene, bisher unbelegte Idee „Reichere
Observablen" (`BEFUNDE_UND_RICHTUNGEN.md` §8) um einen konkreten Mechanismus:
statt fester ⟨Z_q⟩-Auslese (n Zahlen) zufällige Clifford-Vorverarbeitung +
Auslese, daraus zusätzlich Paar-Erwartungswerte ⟨Z_iZ_j⟩ schätzen — mehr
Information pro Schaltungsdurchlauf.

**Erwartete Wirkung.** Parameter: 0 zusätzliche trainierbare Parameter (die
Zufallsbasen sind fest). Zeit: **Auf dem klassischen Simulator kein Vorteil** —
dort liegt die volle Amplitude ohnehin vor, „wenige Messungen" ist eine
Hardware-Sparmaßnahme. Genauigkeit: durch mehr Information pro Kanal potenziell
höher (mehr Eingaben für den Kopf 8→10), aber ungeprüft.

**Risiko und Einschätzung.** **Nur auf echter Hardware oder bei sehr großem n
sinnvoll** — der Grund, warum klassische Schatten existieren (Messkosten
sparen), entfällt komplett im klassischen Simulator. Für B4 (Hardware-Lauf)
und eine mögliche spätere Vergrößerung von n aber die richtige Technik für
„reichere Observablen ohne mehr Schaltungsaufrufe". Ehrlich als
Hardware-Baustein einordnen, nicht als CPU/GPU-Baustein.

**Aufwand.** 2 Tage für den Simulator-Fall (zusätzliche Auslesekanäle ohne
Zufallsbasen, einfacher Spezialfall), 5+ Tage für echte klassische Schatten
auf IBM-Hardware.

**Faire Kontrolle.** Gegen die bestehende ⟨Z_q⟩-Auslese mit gleicher Anzahl
Endkanäle (also z. B. 8 ⟨Z⟩ + Auswahl von ⟨ZZ⟩ statt 8+28 alle Paare).

---

### Idee 5 — Quanten-Fisher-Information als Vorkonditionierer (Quantum Natural Gradient)
**Was ist es.** Statt Adam die Winkel als freie reelle Zahlen behandeln zu
lassen, nutzt QNG die bekannte Krümmung der Rotationsgruppe (Fubini-Study-
Metrik) für den Optimierungsschritt — für reine Ein-Qubit-Rotationsprodukte ist
diese Metrik analytisch bekannt (blockdiagonal, keine teure Schätzung nötig).

**Quellen.** Stokes, Izaac, Killoran, Carleo, „Quantum Natural Gradient",
Quantum 4, 269 (2020), arXiv:1909.02108; Xanadu-Blog dazu.
**Komplementiert** die eigene IDEE in `BEFUNDE_UND_RICHTUNGEN.md` §4, die dort
ohne Literatur stand.

**Einbau in QIRE.** Als Optimierer-Alternative zu Adam für die Winkel — bei
QIREs reiner Produktschaltung (ohne CNOT) ist die Metrik sogar exakt und
billig berechenbar (jede Rotationsachse trägt unabhängig bei), kein
Extra-Schaltungsaufruf wie bei echten Quantengeräten nötig.

**Erwartete Wirkung.** Parameter: 0 zusätzliche. Zeit: pro Schritt etwas
teurer (eine kleine Matrixinversion je Layer), potenziell aber **weniger
Schritte bis zur Konvergenz** — das eigentliche Ziel (Trainingszeit).
Genauigkeit: laut Literatur oft schnellere, nicht unbedingt bessere Konvergenz.

**Risiko und Einschätzung.** Wichtiger Vorbehalt aus der eigenen Recherche
(`LITERATUR_UND_ALLEINSTELLUNG.md` 1.4, Aghaei Saem u. a. 2026): QNG überwindet
Konzentration/Barren-Plateaus bei **endlichem Messbudget** nicht. Das betrifft
QIRE aber **nicht**, weil der Simulator exakte Erwartungswerte liefert, kein
endliches Messbudget hat. Für QIRE bleibt daher nur die reine
Konvergenzgeschwindigkeit als Frage — bei so wenigen Parametern (16–80) ist
unklar, ob Adam nicht ohnehin schon nahe am Optimum konvergiert; der Gewinn
könnte zu klein sein, um ihn zu messen.

**Aufwand.** 2 Tage (die Metrik für Produktrotationen ist Schulbuch-Algebra).

**Faire Kontrolle.** Gleiche Schaltung, gleiche Seeds, Adam gegen QNG,
Schritte bis zu festem Genauigkeitsziel gemessen (nicht nur Endgenauigkeit).

---

### Idee 6 — Feste unitäre t-Designs statt CNOT-Ring als Mischschritt
**Was ist es.** Ein unitäres t-Design ist eine Menge/Verteilung von Unitären,
die sich (bis Ordnung t) wie echter Zufall verhält, aber viel billiger zu
erzeugen ist als eine Haar-zufällige Unitäre. Ein festes Clifford-Element ist
schon ein 3-Design und klassisch in Polynomialzeit auswertbar — meist sogar nur
Vorzeichenwechsel und Permutation, ohne jede Matrixmultiplikation.

**Quellen.** „Efficient unitary designs and pseudorandom unitaries from
permutations", arXiv:2404.16751 (2024); „Simple constructions of linear-depth
t-designs and pseudorandom unitaries", eprint 2024/302.

**Einbau in QIRE.** Ein direkter Test der offenen Frage „trägt irgendeine feste
Mischung etwas bei, oder speziell CNOT?" (Q-020 (b)): den CNOT-Ring durch ein
festes, **parameterfreies** Clifford-Mischelement ersetzen (z. B. eine
zufällig gewählte, aber fixierte Permutation der 2ⁿ Basiszustände mit
Vorzeichen). Kostet keine zusätzlichen Parameter, nur eine andere feste
Indexoperation.

**Erwartete Wirkung.** Parameter: +0. Zeit: wie CNOT-Ring (reine
Indexoperation, R1/R2 aus `ENTWURF_T_SERIE_KERNEL.md` gelten genauso).
Genauigkeit: unbekannt — genau das ist der Test.

**Risiko und Einschätzung.** Billigster Test in diesem Katalog. Wenn eine
zufällige feste Mischung genauso wenig bringt wie CNOT, stützt das die
bisherige (mit Vorbehalt versehene) Aussage „Verschränkung trägt auf MNIST
nichts bei" zusätzlich und unabhängig vom CNOT-Ring-Muster. Wenn sie doch
hilft, ist das eine neue, interessante Spur. Kein Risiko außer
Implementierungszeit.

**Aufwand.** 1 Tag.

**Faire Kontrolle.** Gegen `hybrid` (CNOT-Ring) und `hybrid_ry_only` (keine
Mischung), TOST wie in Q-020 (b), gleiche Seeds.

---

## Feld 5: Quantenkryptographie und Post-Quanten-Kryptographie

### Idee 7 — Zirkulante Matrizen (Ring-LWE-Struktur) als weitere Kontrollfamilie
**Was ist es.** Ring-LWE, die Grundlage vieler Post-Quanten-Verfahren (z. B.
NIST-Standards wie Kyber), rechnet mit zirkulanten Matrizen — jede Zeile ist
eine zyklische Verschiebung der vorigen. Eine zirkulante Matrix hat nur d freie
Zahlen (statt d²) und lässt sich per FFT in O(d log d) multiplizieren.

**Quellen.** „The Ring-LWE Problem in Lattice-Based Cryptography: The Case of
Twisted Embeddings", PMC-Artikel (bestätigt die zirkulante Struktur als Kern
der Effizienz von Ring-LWE); allgemein bekannt aus den NIST-PQC-Standards.

**Einbau in QIRE.** **Kein Quantenbaustein**, sondern eine weitere klassische
strukturierte-Matrix-Familie neben Butterfly/Kronecker — historisch zufällig
mit Kryptographie statt mit Quantenschaltungen verbunden. Als zusätzlicher,
sehr billiger Kontrollarm in Q-020: `classical_circulant` (d Parameter, FFT-
Multiplikation) neben `classical_quad_free`, linear, ReLU.

**Erwartete Wirkung.** Parameter: d (bei d=8: 8 Parameter, vergleichbar mit
QIREs 16). Zeit: FFT ist bereits eine hochoptimierte Standardoperation in jeder
Bibliothek (schneller als Butterfly heute, weil kein 60-Stufen-Problem —
FFT-Bibliotheken sind fusioniert). Genauigkeit: unbekannt, aber als
Kontrollarm nicht das Ziel.

**Risiko und Einschätzung.** Ehrlich: bringt QIRE selbst nichts bei, macht aber
die Kontrollbatterie in Q-020 vollständiger und benennt explizit, dass
„strukturierte Matrix mit wenig Parametern" nicht an Quantenphysik hängt — das
stärkt die eigene, selbstkritische Argumentation aus
`LITERATUR_UND_ALLEINSTELLUNG.md` (Kandidat a: „bekannt").

**Aufwand.** 0,5 Tage (`torch.fft` liefert die Multiplikation fertig).

**Faire Kontrolle.** Ist selbst die Kontrolle — gehört direkt in die
Q-020-Armbatterie.

---

### Idee 8 — Stabilizer-Codes als Vorbild für strukturierte Sparsity (Analogie, kein Baustein)
**Was ist es.** Quantenfehlerkorrekturcodes (Stabilizer-Codes) sichern
Information mit wenigen, strukturiert angeordneten Prüfgleichungen statt mit
zufälliger Redundanz. Die gefundene Literatur behandelt aber ausschließlich die
**umgekehrte** Richtung: neuronale Netze als Decoder für solche Codes, nicht
Codes als Vorlage für Netzarchitektur.

**Quellen.** „A scalable and fast artificial neural network syndrome decoder
for surface codes", Quantum 7, 1058 (2023); „An Efficient Algorithmic Way to
Construct Boltzmann Machine Representations for Arbitrary Stabilizer Code",
Entropy 27(6), 627 (2025).

**Einbau in QIRE.** Nur als lose Analogie denkbar: strukturierte,
codeartige (statt zufällige) Sparsity-Muster in Gewichtsmatrizen, inspiriert
von der Idee „wenige, geometrisch angeordnete Prüfbits genügen für Robustheit".
Kein konkreter Algorithmus in der gefundenen Literatur, der das für
KI-Gewichte vorschlägt.

**Erwartete Wirkung.** Nicht abschätzbar ohne eigene Konstruktion.

**Risiko und Einschätzung.** **Ehrlich: klingt gut, bringt aktuell nichts.** Es
gibt keine gefundene Arbeit, die diese Richtung (QEC-Struktur → KI-Architektur)
tatsächlich verfolgt — nur die entgegengesetzte (KI hilft QEC). Eine eigene
Konstruktion wäre Grundlagenarbeit ohne Vorlage und hohes Risiko, bis zur
Anmeldung nichts Zitierbares zu ergeben. Nicht für die 15-Wochen-Roadmap
geeignet, höchstens als eine Zeile im Ausblick-Kapitel.

**Aufwand.** Nicht geschätzt (Grundlagenarbeit, keine Zeitangabe sinnvoll).

**Faire Kontrolle.** Entfällt — Idee wird nicht umgesetzt.

---

### Idee 9 — QKD/BB84 und Quanten-Zufallszahlen: geprüft, kein Beitrag
**Was ist es.** Quantenschlüsselaustausch (BB84) sichert Übertragungskanäle
gegen Abhören; Quanten-Zufallszahlengeneratoren (QRNG) liefern „echten"
Zufall statt Pseudozufall.

**Quellen.** „On the effects of biased quantum random numbers on the
initialization of artificial neural networks", Machine Learning (Springer)
2023 — Kernbefund: **kein statistisch signifikanter Unterschied** zwischen
quanten- und klassischem Zufall bei der Netzinitialisierung, egal ob verzerrt
oder nicht.

**Einbau in QIRE.** Keiner. Explizit geprüft, weil die Aufgabenstellung
Quantenkryptographie nennt.

**Erwartete Wirkung.** Keine.

**Risiko und Einschätzung.** **Ehrlich negativ:** QKD sichert Kanäle, nicht
Rechenaufwand — es hat mit „ressourceneffizienter KI" nichts zu tun. Die
QRNG-Init-Studie zeigt zusätzlich, dass selbst der naheliegendste
Berührungspunkt (Zufallszahlen fürs Training) keinen messbaren Effekt hat. Die
einzige denkbare, aber andersartige Berührung: Falls QIRE später einen Chatbot
mit echten Nutzerdaten betreibt, wäre Post-Quanten-Kryptographie (vgl. Idee 7)
für die *Absicherung* der Anwendung relevant — das ist Infrastruktur, keine
KI-Architektur, und gehört nicht in diese Arbeit.

**Aufwand.** Keiner (bereits geklärt).

**Faire Kontrolle.** Entfällt.

---

## Feld 6/7: Fehlerkorrektur-Inspiration & physik-inspirierte KI jenseits Quanten

### Idee 10 — Holographic Reduced Representations / Hyperdimensional Computing als klassische Amplituden-Analogie
**Was ist es.** HDC/HRR kodiert Information in sehr hochdimensionalen
Zufallsvektoren und „bindet" mehrere Informationen durch (zirkuläre) Faltung —
strukturell verwandt mit Kronecker-Verknüpfung, aber vollständig klassisch,
ohne Normierung oder Amplituden.

**Quellen.** Plate, „Holographic Reduced Representations", IEEE Trans. Neural
Netw. (1995); „Generalized Holographic Reduced Representations",
arXiv:2405.09689 (2024) — führt eine nicht-kommutative Bindungsoperation ein.

**Einbau in QIRE.** Ergänzt die eigene IDEE „Amplituden-Kodierung als
Kompression" (`BEFUNDE_UND_RICHTUNGEN.md` §5): HRR-Bindung als klassischer
Kopf-Baustein neben der Born-Auslese testen — ⟨Z_q⟩ ist bereits eine Art
Bindungs-/Ausleseoperation; ein HRR-Kopf würde dieselbe Rolle mit
Faltung/Korrelation statt Betragsquadrat spielen.

**Erwartete Wirkung.** Parameter: vergleichbar klein (Bindungsoperation ist
parameterfrei, nur die Basisvektoren wären evtl. trainierbar). Zeit:
Faltung ist auf CPU/GPU eine Standardoperation (FFT-basiert), unkritisch.
Genauigkeit: unbekannt, spekulativ.

**Risiko und Einschätzung.** Interessante **klassische** Parallele, aber kein
neuer Befund zur Quantenphysik — im Gegenteil, sie zeigt, dass „Bindung
hochdimensionaler Vektoren" ein allgemeines Prinzip ist, das mit und ohne
Quantenbezug funktioniert. Nützlich für die Einordnung („was ist wirklich
Quanten, was ist allgemeine Mathematik hoher Dimensionen"), aber kein direkter
Effizienzgewinn absehbar.

**Aufwand.** 2–3 Tage für einen ersten Vergleichslauf.

**Faire Kontrolle.** Gegen die bestehende ⟨Z_q⟩-Auslese bei gleicher
Kanalzahl.

---

### Idee 11 — QUBO-Pruning: aktuelle Literatur bestätigt Q-006
**Was ist es.** Pruning-Entscheidungen (welches Neuron bleibt) als
quadratisches binäres Optimierungsproblem (QUBO) formuliert und mit
Quantenannealing gelöst — exakt der Ansatz, den QIRE in Q-006 bereits gebaut
hat (`BEFUNDE_UND_RICHTUNGEN.md` §3).

**Quellen.** „Neural Network Pruning via QUBO Optimization", arXiv:2604.05856
(2026) — das quantenannealing-geprunte Netz schlägt dort Random Forest, SVM,
das ungeprunte Netz und ein identisch simuliert-annealing-geprunte Variante.

**Einbau in QIRE.** Kein neuer Baustein — **stärkt die Zitierbarkeit von
Q-006**: Q-006 ist damit keine Einzelbeobachtung, sondern Teil eines aktiven
2026er-Forschungsfelds, das den Ansatz sogar auf echter Annealing-Hardware
bestätigt.

**Erwartete Wirkung.** Keine Wirkung auf QIRE-Zahlen — reine
Literatureinordnung.

**Risiko und Einschätzung.** Sicherer Gewinn ohne Risiko: einen Satz und
diese Quelle in `LITERATUR_UND_ALLEINSTELLUNG.md` bzw. den Q-006-Abschnitt
einfügen.

**Aufwand.** < 0,5 Tage (Literatureinbau).

**Faire Kontrolle.** Entfällt (kein neues Experiment).

---

### Idee 12 — Gruppenmatrizen als allgemeinere Kontrollfamilie zur eigenen Theorie
**Was ist es.** „Group Matrices" verallgemeinern die reguläre Darstellung einer
beliebigen Gruppe zu einer strukturierten, parameterarmen Matrixfamilie für
näherungsweise äquivariante Netze — Butterfly/Kronecker ist ein Spezialfall für
die Gruppe (Z₂)ⁿ (Bit-Flips), die QIREs Qubit-Struktur zugrunde liegt.

**Quellen.** „Symmetry-Based Structured Matrices for Efficient Approximately
Equivariant Networks", arXiv:2409.11772 (2024).

**Einbau in QIRE.** Direkter Test der eigenen, schon **zweifach geprüften**
Theorie „Parameterzahl zählt, Parametrisierung nicht"
(`BEFUNDE_UND_RICHTUNGEN.md` §2): eine allgemeine Gruppenmatrix für eine
**andere** kleine Gruppe (z. B. zyklisch statt (Z₂)ⁿ) mit gleicher
Parameterzahl wie QIREs Schaltung bauen und vergleichen.

**Erwartete Wirkung.** Parameter: exakt gleich einstellbar wie QIRE (Test
verlangt das). Zeit: strukturierte Matrixmultiplikation, ähnlich billig wie
Butterfly bei passender Gruppe. Genauigkeit: laut eigener Theorie **gleich**
erwartet, wenn „Parameterzahl zählt, Parametrisierung nicht" allgemein gilt —
das wäre eine unabhängige Bestätigung außerhalb der Quantenfamilie.

**Risiko und Einschätzung.** Günstiger, klar interpretierbarer Test: Bestätigt
er die eigene Theorie, ist das ein zusätzlicher, unabhängiger Beleg (nicht nur
„zweifach geprüft" innerhalb der eigenen Schaltungsfamilien, sondern auch
gegen eine externe Konstruktion). Widerlegt er sie, ist auch das ein Ergebnis
und schärft die eigene Aussage („gilt nur innerhalb quanteninspirierter
Familien").

**Aufwand.** 2 Tage.

**Faire Kontrolle.** Gleiche Parameterzahl wie `hybrid`, gleiche Seeds,
gleiche Aufgabe.

---

## Feld 8: Quantenhardware-nah

### Idee 13 — Reck- vs. Clements-Netz: konkrete Zahlen fürs MPI-Gespräch
**Was ist es.** Zwei Standardschaltpläne, um eine beliebige unitäre Matrix als
Netz aus Mach-Zehnder-Interferometern (Zweistrahl-Bauteile mit einstellbarer
Phase) zu bauen: Reck (1994, Dreiecksnetz) und Clements (Rechtecknetz).
Rechtecknetz braucht gleich viele Bauteile wie das Dreiecksnetz, aber nur die
**halbe** optische Tiefe und ist robuster gegen Fertigungsverluste.

**Quellen.** Zusammenfassende Quellen aus Websuche: „Optical Neural Network
Based on Slimmed Mach-Zehnder Interferometer Mesh" (ResearchGate);
„Programming universal unitary transformations on a general-purpose silicon
photonic platform", arXiv:2407.03235 — bestätigen Reck/Clements-Eigenschaften
und aktuelle (2024) Chip-Umsetzungen mit z. B. 56 Mach-Zehnder-Elementen.

**Einbau in QIRE.** Präzisiert die bereits vorhandene IDEE C5
(`01_VERBESSERUNGEN_UND_ZUKUNFT.md`) mit konkreten Zahlen: Ein **volles**
unitäres Netz braucht N(N−1)/2 Interferometer (bei N=256 also 32.640 —
dieselbe Zahl, die in `BEFUNDE_UND_RICHTUNGEN.md` §2 als „Untermannigfaltigkeit"
schon auftaucht). QIREs geteilte Struktur bräuchte dagegen nur 2n
Phasenwerte (16 bei n=8) — nicht 2n unabhängige Bauteile, aber gemeinsame
Steuerspannungen für viele Bauteile, was Kalibrieraufwand und Rauschanfälligkeit
senkt.

**Erwartete Wirkung.** Kein Effekt auf CPU/GPU-Zahlen — reine
Hardware-Zukunftsfrage, kein Simulationsexperiment.

**Risiko und Einschätzung.** Kein Risiko, kein Aufwand im Sinne von Code —
liefert konkrete Frage für das MPI-Treffen (`MPI_TREFFEN/MPI_GESPRAECH.md`):
„Skaliert die Kalibrierzeit eines Rechtecknetzes wirklich mit der Zahl
unabhängiger Steuerspannungen, oder dominieren andere Effekte (Übersprechen,
Drift)?" — das wäre die Messung, die C5 fordert.

**Aufwand.** 0,5 Tage (Frage ausformulieren, keine eigene Messung möglich ohne
MPI-Zugang).

**Faire Kontrolle.** Entfällt (keine eigene Messung, nur Gesprächsvorbereitung).

---

## Feld 3/methodisch: Dequantisierung und Simulationswerkzeuge

### Idee 14 — Dequantisierungs-Checkliste als Methodik-Baustein
**Was ist es.** Ewin Tangs Dequantisierungstechnik zeigt: Viele behauptete
exponentielle Quantenvorteile bei linearer Algebra verschwinden, sobald ein
klassischer Algorithmus denselben Stichproben-Zugriff (statt Amplitudenzugriff)
auf niedrigrangige/niedrigdimensionale Struktur bekommt.

**Quellen.** Ewin Tang, „A quantum-inspired classical algorithm for
recommendation systems", arXiv:1807.04271 (2019) — bereits in
`LITERATUR_UND_ALLEINSTELLUNG.md` als Begriffsquelle gelistet, hier um die
**methodische Konsequenz** ergänzt: „Ist das Problem niedrigrangig/
niedrigdimensional? Dann zuerst eine Dequantisierung versuchen, bevor ein
Vorteil behauptet wird" (Tangs eigene Heuristik laut Suchergebnis).

**Einbau in QIRE.** Keine neue Messung — eine feste Prüffrage für **jede**
künftige „das ist besser, weil Quanten"-Behauptung im Projekt, als Ergänzung
zu C7 (Methodik als eigenständiger Beitrag). Passt zur schon vorhandenen
Erkenntnis „Butterfly-Umsetzung ist exakt dieselbe Funktion" — das *ist* im
Kern eine Dequantisierung, nur nicht so benannt.

**Erwartete Wirkung.** Keine Zahl — Qualität der Argumentation in der Arbeit.

**Risiko und Einschätzung.** Praktisch risikofrei, geringer, aber sicherer
Beitrag zur Glaubwürdigkeit bei der Jury (passt zur schon erkannten Stärke
„vorregistrierte Abgrenzungstests" aus `LITERATUR_UND_ALLEINSTELLUNG.md`
Abschnitt 2.2).

**Aufwand.** < 0,5 Tage (ein Absatz in der Methodik).

**Faire Kontrolle.** Entfällt.

---

### Idee 15 — Pauli-Propagation als Werkzeug für spätere, tiefere Schaltungen
**Was ist es.** Pauli-Propagation schätzt Erwartungswerte flacher Schaltungen,
indem Pauli-Operatoren rückwärts durchs Schaltungsdiagramm geschoben und kleine
Beiträge abgeschnitten werden — die Rechenzeit hängt von Schaltungstiefe und
Verschränkungsmuster ab, nicht direkt von der Qubitzahl.

**Quellen.** Angrisani, Schmidhuber, Rudolph, Cerezo, Holmes, Huang (2024,
laut Suchzusammenfassung); Rudolph, Jones, Teng, Angrisani, Holmes, „Pauli
Propagation: A Computational Framework for Simulating Quantum Systems",
arXiv:2505.21606 (Mai 2025); Begušić & Chan (2025) — 127-Qubit-IBM-Experiment
klassisch nachgerechnet.

**Einbau in QIRE.** **Für die heutige, flache, CNOT-freie Schaltung
irrelevant** — die ist bereits exakt als Butterfly-Netz in O(d log d)
implementiert, schneller als jede Pauli-Propagation es für dieselbe Aufgabe
wäre. Relevant nur, **falls** QIRE künftig tiefere oder dichter verschränkte
Varianten (z. B. Matchgates aus Idee 1, oder ein voller CNOT-Ring bei
größerem n) testen will, wo die volle 2ⁿ-Zustandsvektor-Simulation zu teuer
wird, aber die Schaltung noch flach genug für Pauli-Propagation bleibt.

**Erwartete Wirkung.** Für die aktuelle Roadmap: keine. Als
Zukunftswerkzeug: könnte n über die heutige Simulationsgrenze hinaus
skalieren, ohne Amplituden zu materialisieren.

**Risiko und Einschätzung.** Ehrlich: **jetzt kein Bedarf.** Nur als Randnotiz
für `ENTWURF_T_SERIE_KERNEL.md` aufheben, falls die Roadmap später auf größere
n oder dichtere Verschränkung erweitert wird.

**Aufwand.** Nicht jetzt investieren.

**Faire Kontrolle.** Entfällt (kein aktuelles Experiment).

---

## Rangliste — Top 10 nach (Erfolgsaussicht × Wirkung / Aufwand)

| Rang | Idee | Begründung kurz |
|---|---|---|
| 1 | **6 — feste t-Design-Mischung statt CNOT-Ring** | 1 Tag, direkter, unabhängiger Test der offenen Verschränkungsfrage |
| 2 | **2 — MPO-Encoder (CompactifAI-Prinzip)** | löst das 848×-nur-in-der-Mitte-Problem, literaturgestützt, ~3–4 Tage |
| 3 | **12 — Gruppenmatrix-Kontrollarm** | billiger, unabhängiger Test der eigenen „Parameterzahl zählt"-Theorie |
| 4 | **7 — zirkulante Matrix als Q-020-Kontrolle** | 0,5 Tage, macht die Kontrollbatterie ehrlicher und vollständiger |
| 5 | **14 — Dequantisierungs-Checkliste** | so gut wie kostenlos, stärkt Methodik-USP |
| 6 | **11 — QUBO-Pruning-Literatur für Q-006** | kostenlos, bestätigt bestehenden Befund als Stand der Technik |
| 7 | **4 — klassische Schatten für reichere Auslese** | mittlerer Aufwand, aber nur für Hardware/größeres n wirklich nötig |
| 8 | **5 — Quantum Natural Gradient** | billig, adressiert Zeitziel, Nutzen bei so wenigen Parametern unsicher |
| 9 | **3 — MERA/Baum-Encoder** | dritte Encoder-Variante, nur bei Zeitüberschuss nach Idee 2 |
| 10 | **13 — Reck/Clements-Frage fürs MPI** | kein Rechenaufwand, aber nur Zukunftskapitel, keine Kernaussage |

**Nicht in der Top 10, aber ehrlich dokumentiert:** Idee 1 (Matchgates,
spekulativ), Idee 10 (HDC, spekulativ, kein Quantenbezug), Idee 8
(Stabilizer-Analogie, keine Vorlage gefunden), Idee 9 und Idee 15 (geprüft,
aktuell kein Beitrag).

---

## Zusammenhänge

- **Encoder-Kompression (Ideen 2, 3) und B2** lösen dasselbe Problem mit
  unterschiedlicher Struktur (MPO, MERA, Butterfly) — Wettbewerb, keine
  Ergänzung. Bei gleicher Parameterzahl gegeneinander antreten lassen.
- **Ideen 6 und 12** testen von zwei Seiten dieselbe offene Frage „ist es die
  Quantenschaltung oder nur ihre Parameterzahl/Struktur?" — Idee 6 innerhalb
  der Quantenfamilie (andere feste Mischung), Idee 12 außerhalb (andere
  Gruppe). Zusammen mit Q-020 (b) und Q-015 die direkteste Prüfung der
  Kernaussage.
- **Idee 7 (zirkulant)** gehört in dieselbe Kontrollbatterie wie
  `classical_quad_free`, linear und ReLU aus Q-020 (a) — ergänzt die dort
  schon geplante faire Referenz um eine weitere strukturierte, aber nicht
  quanteninspirierte Familie.
- **Ideen 11 und 14** verstärken zusammen mit C7 (Methodik als Beitrag,
  `01_VERBESSERUNGEN_UND_ZUKUNFT.md`) denselben USP-Kandidaten (b) aus
  `LITERATUR_UND_ALLEINSTELLUNG.md` — gehören in dasselbe Methodik-Kapitel.
- **Idee 4 (klassische Schatten)** hängt an B4 (Hardware-Minimalkreis) und an
  der eigenen IDEE „Adaptive Messtiefe" — wird erst außerhalb des Simulators
  relevant.
- **Idee 5 (QNG)** verwandt mit der eigenen IDEE „Lernrate aus der
  Spektrallücke" (`BEFUNDE_UND_RICHTUNGEN.md` §4) — beide nutzen die bekannte
  Geometrie der Rotationsgruppe für den Optimierer; ließen sich in einem
  gemeinsamen Experiment prüfen.
- **Idee 1 (Matchgates)** ist der natürliche nächste Schritt, **falls**
  Q-020 (b) zeigt, dass Verschränkung doch trägt — ein Zwischenschritt
  zwischen „keine Mischung" und „volle Universalität".
- **Idee 13 (Reck/Clements)** ist die technische Grundlage für die bereits
  vorhandene IDEE C5 und gehört zusammen mit Idee 8 (falls doch verfolgt) und
  den MPI-Fragen aus `LITERATUR_UND_ALLEINSTELLUNG.md` Abschnitt 4 ins
  MPI-Gespräch.
- **Idee 15 (Pauli-Propagation)** ist ein Werkzeug auf Vorrat, gekoppelt an
  Idee 1 — beide werden erst relevant, wenn QIRE über die heutige
  CNOT-freie, flache Schaltung hinausgeht.
- **Ideen 9 und 8** sind die beiden „ehrlich negativ" geprüften Felder
  (Quantenkryptographie, Fehlerkorrektur als KI-Vorlage) — zusammen ein Beleg
  dafür, dass die Recherche wirklich breit war, nicht nur dort gesucht hat,
  wo etwas zu finden war.
