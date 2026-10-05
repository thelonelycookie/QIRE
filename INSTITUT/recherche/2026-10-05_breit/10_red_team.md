# 10 — Red Team: stärkste Einwände gegen QIRE und gegen QML, und ehrliche Formulierung

*Stand 05.10.2026. Recherche: 16 erfolgreiche WebSearch-Anfragen (danach Suchbudget des Laufs erschöpft). Fast alles nur Suchauszug, keine Volltexte gelesen; wo ich aus eigenem Fachwissen ergänze, steht [Wissen] (nicht in dieser Sitzung belegt) bzw. [EIGEN]/[?]. Repo-Zahlen stammen aus `UEBERGABE.md` §0/§4, `PLAN_2026-10-05_effizienz.md` und `LABORBUCH.md`.*

## 1. Kurzfassung

1. **Die stärkste Kritik am Wort „Quanten“:** Die QIRE-Zelle (Ry + CNOT, reell, 8 Qubits, Born-Auslese) ist mathematisch eine feste/gelernte **orthogonale 256×256-Matrix Q, gefolgt von Quadrieren und linearem Kopf**: p = (Q·x/‖x‖)². Das ist auf einem Laptop in Mikrosekunden klassisch berechenbar (und die Clifford-Variante ohnehin nach Gottesman-Knill). „Quantenphysik“ liefert hier die *Schreibweise* und die Suchregeln (Clifford-Gruppe), keine Rechenleistung, die klassisch fehlt. Das ist kein Makel, wenn man es so sagt; es ist einer, wenn man „KI mit Quantenphysik“ als Ressourcengewinn verkauft.
2. **Die wichtigste fehlende Kontrolle:** Der Kopf (257×10 = 2.570 Parameter) *ist* schon eine logistische Regression auf 256 Merkmalen. Die ehrliche Referenz ist daher nicht „Identität = 0,90“, sondern: gleiche 2.570 Parameter, klassisch optimal ausgenutzt (Softmax auf 16×16-Pixeln, kleines MLP mit 2.570 Parametern, fester Walsh/DCT-Block + Quadrat + Kopf). Ob diese Baselines in den Läufen sauber und gleich getunt vorkommen, ist aus §0/§4 nicht ersichtlich [?].
3. **Der Befund „0 T-Gatter reicht auf MNIST“** ist nach Gottesman-Knill (Aaronson/Gottesman 2004) schlicht „die Schaltung ist klassisch effizient simulierbar“. Er ist als Aussage über T-Kosten auf fehlertoleranter Hardware sinnvoll, als Aussage über „Quantennutzen“ leer. Der Direktor hat das schon richtig eingeordnet (§0 letzter Absatz); es muss im Wortlaut der Arbeit durchgehalten werden.
4. **Literatur stützt die Skepsis:** Bowles/Ahmed/Schuld 2024 (12 QML-Modelle, 160 Datensätze): klassische Standardmodelle schlagen die Quantenklassifizierer, und das *Entfernen von Verschränkung* ist oft gleich gut oder besser. Das ist exakt der Test, den QIRE als Kontrolle braucht (CNOT-Ring aus → ?).
5. **Zu starke Formulierungen im Repo (Auswahl):** „Magie-Konzentration“ als Eigenschaft der *Aufgabe* (I-005 zeigt: misst eher den Gradienten), „2,25 % der T-Gatter“ (nur MNIST, Fashion repliziert nicht), „16 Winkel schlagen MLP“ (Q-020: 200.960 ungezählte Parameter), „Quantenphysik“ allgemein.
6. **Billigste, wirksamste neue Kontrollen [EIGEN]:** (a) Parameter-gleiche klassische Baselines, (b) Q durch zufällige/gelernte orthogonale Matrix ersetzen, (c) Normierung/Quadrat isolieren (Ablation), (d) Verschränkung entfernen (Bowles-Test), (e) Zufallslabel-Test (Gil-Fuster), (f) Nicht-PCA-, Nicht-Binär-Kontrolle ist bei QIRE erfüllt (volle 10 Klassen, kein PCA) — das ist ein Pluspunkt, den man nennen sollte.

## 2. Stand der Forschung (Kritik) — mit Quellen

**2.1 „Ist Quantenvorteil das richtige Ziel?“** Schuld & Killoran, PRX Quantum 3, 030101 (2022): Perspektive; stellt infrage, ob das „Schlagen“ klassischen MLs die Literatur dominieren sollte; theoretische Argumente seien für reale Skalen das Hauptwerkzeug, praktische Benchmarks fehlen. Alternative Perspektiven (z. B. Modelle mit eigener Struktur) werden als sinnvoll genannt. https://arxiv.org/pdf/2203.01340 (nur Auszug). *Für QIRE:* Sie stützt ein *bescheidenes* Ziel (Effizienz/Struktur statt „Vorteil“).

**2.2 Benchmarking.** Bowles, Ahmed, Schuld, arXiv:2403.07059 (2024): 12 QML-Modelle × 6 binäre Aufgaben (160 Datensätze); klassische Out-of-the-box-Modelle gewinnen; Entfernen der Verschränkung gibt oft gleiche oder bessere Leistung; „Quantenhaftigkeit“ sei bei den kleinen Aufgaben nicht der entscheidende Zutat. https://arxiv.org/pdf/2403.07059 (nur Auszug über Sekundärseiten). Übersichtsbefund aus der Suche (Sekundärquellen, viele 2026-Preprints): häufige Fallen sind unfair getunte Baselines, unterschiedliche Rechenbudgets, Datenleck, selektives Berichten, Skalierung/Kernbreite als dominanter Faktor (z. B. https://arxiv.org/html/2604.18837, https://arxiv.org/html/2608.15617v1 — nur Snippets, nicht geprüft).

**2.3 Barren Plateaus und Simulierbarkeit.** Cerezo et al., „Does provable absence of barren plateaus imply classical simulability?“, Nat. Commun. 16, 7907 (2025), arXiv:2312.09121: Viele Modellklassen, die beweisbar trainierbar sind, leben in kleinen, klassisch simulierbaren Unterräumen; Simulation ggf. nach einer Datenerhebungsphase auf Quantenhardware. https://www.nature.com/articles/s41467-025-63099-6. *Für QIRE:* Trainierbarkeit der 8-Qubit-Zelle ist kein Wunder, sondern erwartbar, weil der Raum klein (256-dim.) ist und die Struktur reell-orthogonal.

**2.4 Generalisierung.** Gil-Fuster, Eisert, Bravo-Prieto, Nat. Commun. 15, 2277 (2024): QNNs können Zufallslabels anpassen; uniforme Komplexitätsschranken erklären Generalisierung nicht. https://www.nature.com/articles/s41467-024-45882-z. *Für QIRE:* I-003 (Memorieren M50 = 8 vs. Generalisieren M50 = 1) ist genau dieses Thema; die Zufallslabel-Kontrolle fehlt als saubere Zeile.

**2.5 Klassische Surrogate und Daten.** Schreiber/Eisert/Meyer, PRL 131, 100803 (2023): trainierte Quantenmodelle (Fourier-Darstellung) lassen sich klassisch ersetzen. https://www.researchgate.net/publication/361499932 (nur Auszug). Huang et al., Nat. Commun. 12, 2631 (2021): Mit genug Daten halten klassische Modelle mit, selbst wenn die Datenquelle klassisch schwer ist. https://www.nature.com/articles/s41467-021-22539-9.

**2.6 Dequantisierung.** Tang 2019 (Empfehlungssysteme), danach Chia/Gilyén/Li/Lin/Tang/Wang, J. ACM 69(5) (2022), „Sampling-based sublinear low-rank matrix arithmetic framework“: ganze Klasse von QML-Beschleunigungen (PCA, SVM, Regression …) fällt unter der Annahme von Sampling-Zugriff auf Eingaben klassisch nur polynomiell langsamer. https://arxiv.org/pdf/1910.06151. Gharibian/Le Gall, STOC 2022 (QSVT dequantisieren; mit inverse-polynomieller Genauigkeit BQP-vollständig): https://arxiv.org/abs/2111.09079. Robust: https://arxiv.org/pdf/2304.04932. *Lektion:* Exponentielle QML-Vorteile hängen an Eingabemodell-Annahmen (hier: Amplitudenkodierung = „Zustandspräparation gratis“); QIRE nimmt die Amplitudenkodierung sogar *real* (Laden kostet 7,7× die Schaltung, C-kodierung) — der klassische Normquadrat-Zugriff wäre das Dequantisierungs-Modell.

**2.7 Etikett „quantum-inspired“.** Die Literatur nutzt „quantum-inspired/-like/-enhanced“ überlappend und uneinheitlich; manche nennen klassische Tensor-Netz-ML schon „QML“, passender wäre „quantum-inspired“ (https://arxiv.org/html/2604.14287, nur Snippet; Survey https://arxiv.org/pdf/2308.11269). Zwei völlig verschiedene Dinge teilen den Namen: dequantisierte Algorithmen (Tang) und Tensor-Netz-Architekturen. *Für QIRE:* Wenn QIRE auf Hardware läuft, ist es QML; wenn wir es als Matrixoperation in PyTorch rechnen, ist es höchstens „quantum-inspired“. Beides sagen.

**2.8 Klassische Nachbarn der Idee (Neuheits-Gegenprobe für Projekt A).** KronA (Edalati et al. 2022, Kronecker-Adapter, https://arxiv.org/pdf/2212.10650) und BOFT (Liu et al. 2023, orthogonale Butterfly-Faktorisierung mit O(d log d) Parametern, https://arxiv.org/pdf/2311.06243) sind **direkte klassische Konkurrenten** zur „QIRE-Kronecker-Drehung mit log₂(Breite) Winkeln“. Ein reiner Ry-Kronecker mit log₂ d Winkeln ist ein *Spezialfall* von Givens/Butterfly-Rotationen. Sekundärurteil aus der Suche: KronA allein sei sehr parametersparend, aber mit deutlichem Genauigkeitsverlust (Snippet, https://arxiv.org/pdf/2508.00230 — nicht geprüft). Fastfood/strukturierte Hadamard-Transformen (Le/Sarlós/Smola 2013) [Wissen, nicht recherchiert, Budget erschöpft] sind der klassische Vorläufer von „Walsh-Block × Vorzeichenmaske“.

**2.9 Holevo.** Meine Suche zu Holevo + Amplitudenkodierung lieferte nichts Belastbares (nur Kodierungs-Überblicke). [Wissen] Holevo: n Qubits liefern höchstens n klassische Bit pro Auslese; Born-Auslese mit vielen Schüssen schätzt 2ⁿ Wahrscheinlichkeiten, jede aber nur mit 1/√Schüsse Genauigkeit — das ist die Schuss-Rechnung (I-008, ~1.000 Messungen je Bild). Das wird im Bericht zu Maltes Ziel 3 (anderes Thema) vertieft.

## 3. Was das für QIRE bedeutet — Befund für Befund

Format: **Langweiligste Erklärung / fehlende Kontrolle / zu starke Formulierung → ehrliche Formulierung.**

### B1. „Schaltung schlägt Identität, MNIST +3,8 pp“ (H1)
- *Langweilig:* „Identität“ = Pixelquadrate + linearer Kopf (LABORBUCH: 0,9015 MNIST / 0,80 Fashion). Jede feste orthogonale Mischung vor dem Quadrieren erzeugt Kreuzterme (x_i±x_j)² und damit Merkmale zweiter Ordnung; ein **quadratisches Merkmalsmodell** schlägt ein lineares fast immer. Das ist Merkmalsvielfalt, nicht Quanteneffekt.
- *Fehlt:* Vergleich gegen (i) Softmax auf *rohen* 256 Pixeln (gleiche 2.570 Parameter, ohne Quadrat), (ii) Quadrat-Kopf auf zufälliger orthogonaler Matrix, (iii) kleines MLP/CNN mit ≈ 2.600 Parametern, gleich getunt (Lernrate, Epochen), (iv) ein klassisch gelerntes quadratisches Modell (Kern 2. Grades).
- *Formulierung:* „Eine feste orthogonale Vormischung plus Quadrieren verbessert die lineare Auslese um 3,8 pp“ — nicht „die Quantenschaltung schlägt“.

### B2. „Nur Clifford bringt nichts; Magie-Konzentration M50/M90“
- *Langweilig:* Es misst, wie viele der 32 Winkel *der Gradientenlauf* feinjustieren muss. I-005 zeigte selbst: Suche (0 Magie) = voll auf MNIST → M50 ist eine Eigenschaft des **Optimierers und der Winkelparametrisierung**, nicht der Aufgabe. I-035 (Schwierigkeit sagt M50 nicht voraus) und I-003 (HU) stützen das; C-skala war Ratenartefakt. „Magie“ im Sinn von Nicht-Clifford-Winkeln ist außerdem nur eine Zählung nichtgrader Ry-Winkel — keine Messung von Nichtstabilisierbarkeit des *Zustands* (I-002: SRE-Messung H0).
- *Fehlt:* Dieselbe Konzentrationsmessung mit zufälliger Parametrisierung (Winkel ↔ Basiswechsel), mit anderem Optimierer, und mit einem **klassischen Analogon** (z. B. „wie viele Einträge einer orthogonalen Matrix müssen vom Gitter abweichen“).
- *Formulierung:* „M50 beschreibt, wie viele Gelenke des Gradientenverfahrens nicht-Clifford sein müssen; Aufgabeneigenschaft nur auf Fashion/MNIST und nur mit diesem Optimierer belegt.“ Das Wort „Arbeitstitel für den Namen“ (§4 Ende) besser streichen oder als *Arbeitshypothese* kennzeichnen.

### B3. „0 T-Gatter erreichen volles Training“ (I-005-v2, TOST ±0,5 pp)
- *Langweilig:* Gottesman-Knill: Clifford heißt klassisch simulierbar (Aaronson/Gottesman, https://www.scottaaronson.com/papers/chp6.pdf). Die Clifford-Zelle ist **blockweise Walsh-Hadamard über vorzeichenmaskierten Pixeln** — eine Rechnung ohne Multiplikationen, bekannt als Fast-Walsh/Fastfood-Merkmal. Der Befund heißt dann: „Eine diskrete, multiplikationsfreie Merkmalsabbildung erreicht das Niveau der kontinuierlichen.“ Das ist ein *Effizienzbefund*, kein Quantenbefund.
- *Fehlt:* Dieselbe Suche im Raum **klassischer** diskreter Transformen (Signed-Permutation, Hadamard × Diagonal(±1) × Permutation, DCT-Block) ohne Qubit-Sprache. Wenn der Clifford-Raum nicht besser ist als „Vorzeichen + Permutation + Hadamard“, ist das eine Umbenennung. (Bruhat-Form im Repo deutet auf nahe Verwandtschaft hin.)
- *Formulierung:* „Eine Clifford-Schaltung (klassisch effizient simulierbar, auf fehlertoleranter Hardware T-frei) erreicht …“.

### B4. I-036/-v2/-v3: „Maske × Blöcke nur gemeinsam; Rang der quadratischen Form trägt 52–78 %“
- *Langweilig:* Eine quadratische Form über GF(2) = CZ-Muster = ein Phasenorakel der Form (−1)^{xᵀAx}. Rang großer → mehr Kreuzkorrelationsstruktur zwischen Pixelbits. Das ist die Walsh-Spektraltheorie von **Bent-/Quadratikfunktionen** (MacWilliams/Sloane) [Wissen]; „Rang trägt“ kann trivial bedeuten „mehr Bits sind verkoppelt“.
- *Fehlt:* Kontrolle auf *zufälligen gleich-rangigen* Formen (Rang konstant, Struktur zufällig) und Seed-Übertragbarkeit mit zufälligen Seeds auf zufälligen *Datensätzen* (permutierte Pixel): überträgt die Form Bildstruktur oder nur 2-D-Nachbarschaft? Zweitens: Mehrfachvergleiche — wie viele Hypothesen wurden im Institut insgesamt getestet (viele H0 sind dokumentiert, gut), Korrektur?
- *Formulierung:* „Der Rang erklärt 52–78 % des Maskengewinns *in dieser Suchumgebung*“ (nicht „trägt den Nutzen“).

### B5. I-034 „2,25 % der T-Gatter, MNIST“
- *Langweilig:* Zahl hängt vom Referenzpunkt: „7.698 T“ ist der *naive* Weg der exakten Amplitudenpräparation; Dünnbesetzung/approximatives Laden (West et al. 2023) ist publiziert (§4 selbst). Außerdem Fashion repliziert nicht (H0). Auswahl der Hyperparameter (m, Stufen) auf MNIST getunt → Gewinner-Fluch.
- *Formulierung:* nur als „Beispiel auf einem Datensatz, auf zweitem nicht repliziert“; nie als Schlagzeile.

### B6. „Datenladen ist der Engpass“ (K-1)
- *Das ist die bekannte Dequantisierungs-Lektion* (State Preparation / Input Assumption, Tang, Aaronson „Read the fine print“ [Wissen]). Gut und ehrlich, aber **kein neuer Befund**; als Bestätigung formulieren und Tang/Chia zitieren.

### B7. Q-020: „16 Winkel schlagen MLP“
- *Langweilig:* 200.960 Parameter im Encoder (784→256) tragen die Leistung. Die 16 Winkel sind ein Bruchteil von 0,008 % der Parameter.
- *Formulierung (verbindlich):* immer Gesamtparameterzahl (≈ 201.000) nennen; Plan §1a hat das schon. Ergänzen: Vergleich gegen ein klassisches Modell mit **derselben Gesamtzahl** (nicht „Schaltungsparameter“).

### B8. T-005: „55 % des LoRA-Gewinns mit 1/16 der Parameter, Verlust fällt monoton bis L=10“
- *Langweilig:* Kronecker-Struktur/Butterfly ist ein bekanntes PEFT-Prinzip (KronA, BOFT). „Streng monoton mit der Lagenzahl“ gilt auch für jedes Modell, dessen Kapazität in nested Faktoren wächst; Monotonie ist *erwartet*. Unklar, ob **gleiche Parameterzahl** bei LoRA geprüft wurde (Rang so gewählt, dass ≈ gleiche Parameter) [?]. 120M-Modell, ein Datensatz/Adapter-Setting: Streuung über Seeds und Aufgaben?
- *Fehlt:* (i) Klassische Rotationskontrolle: *beliebige Givens-Winkel statt Ry-Kronecker*, (ii) BOFT/KronA/(IA)³ als Konkurrenten bei ≈ gleicher Parameterzahl, (iii) gleiche Lagenzahl mit *Zufallswinkel* (eingefroren), (iv) Extrinsische Aufgabe (nicht nur Validierungsverlust).
- *Formulierung:* „Eine winkel-parametrisierte Kronecker-Rotation ist eine parametersparende PEFT-Familie; 55 % des LoRA-Gewinns bei 1/16 der Parameter auf LLäMmlein-120M“ — **ohne** „Quanten“ im Titel.

### B9. „Quantenphysik“ im Projektziel
- Es gibt keine Hardware-Messung, die Quanteneffekte (Verschränkung, Interferenz) *als nötig* belegt. Hilfsargument: Bowles et al. — Verschränkung weglassen ändert wenig. Kontrolle fehlt: CNOT-Ring entfernen (QIRE ohne Verschränkung = Produkt von 8 Einzelqubit-Rotationen = Kronecker 2×2-Blöcke). Der „feste 2×2-Block-Walsh holt 72 % des Abstands“ (I-036) deutet bereits in diese Richtung: *Das meiste ist lokale Mischung ohne Verschränkung.* Das sollte die Arbeit sagen.
- *Ehrlicher Rahmen:* **„Quanten-inspirierte Struktur für ressourceneffiziente KI; Rechnung klassisch; Quantenhardware nur als Möglichkeit (IBM-Test), nicht als Vorteil.“**

### B10. Hardware/NISQ
- Rz ist auf IBM virtuell (Direktor, 29.09.). Folge: „0/4 Magiewinkel“ hat auf NISQ keine Kostenbedeutung; T-Kosten sind eine Rechnung für fehlertolerante Hardware, die es nicht gibt. Ein Hardwarelauf zeigt höchstens, dass 8-Qubit-Schaltung + Rauschen Genauigkeit verliert.

## 4. Die 15 härtesten Fragen mit ehrlichen Antworten

(J = Juror Jugend forscht Mathe/Info bzw. Physik, F = Fachwissenschaftler)

1. **(J) Ist das überhaupt Quantencomputing? Läuft es auf einem Quantencomputer?** — Nein, bisher klassische Simulation; die Rechnung ist eine 256×256-Matrixmultiplikation. Quantenhardware wäre möglich (8 Qubits), bringt aber heute keinen Gewinn. Wir nutzen die Quantenmathematik als Struktur.
2. **(F) Warum nicht einfach eine orthogonale Matrix lernen?** — Zu Recht gefragt. Eine volle Orthogonale hat 32.640 Parameter; die Zelle nutzt 32. Aber ob 32 Ry-Winkel *besser* sind als z. B. 32 Butterfly-/Givens-Winkel, ist **nicht getestet** → Kontrolle K3.
3. **(F) Was ist das Neue gegenüber Fastfood, Butterfly/BOFT, KronA, Hadamard-Merkmalen?** — Als Struktur: nichts grundsätzlich. Neu können sein: die Clifford-Suche im diskreten Raum, die Maske×Block-Wechselwirkung, die Magie-Kostenrechnung. Neuheit nur dort behaupten, wo geprüft (siehe `recherche/2026-09-28_neuheit_magie.md`).
4. **(J) Warum dann „16×16“ und kein ganzes MNIST?** — 8 Qubits = 256 Amplituden; Bilder werden verkleinert. Die Verkleinerung (Pixeldurchschnitt) selbst ist Vorverarbeitung, die Genauigkeit begrenzt; sie ist keine PCA, aber auch nicht neutral. Klassische Baselines müssen **dieselbe 16×16-Eingabe** bekommen (und zusätzlich volle 28×28 als Obergrenze).
5. **(F) „99 % der Parameter im linearen Kopf“ — was macht dann die Schaltung?** — Sie liefert Merkmale (+3,8 pp gegenüber Pixelquadraten, I-004/H1). 2.570 Kopfparameter allein sind eine Softmax-Regression; wie stark die ist (und wie viel ein klassisches Modell mit 2.600 Parametern erreicht), muss ich zeigen (K1), bevor ich „Effizienz“ sage.
6. **(J) Ist 94 % auf MNIST gut?** — Nein: CNNs erreichen > 99 %, selbst lineare Modelle 92 % [Wissen]. Es geht nicht um Genauigkeit, sondern um Parameter-/T-Kosten bei fester Genauigkeit. Das Argument wird nur dann tragen, wenn bei **gleichem Budget** eine klassische Kurve (Genauigkeit vs. Parameterzahl) unterschritten wird.
7. **(F) Gottesman-Knill: Ist „0 T-Gatter“ nicht trivial?** — Es ist die Aussage „klassisch simulierbar“. Interessant ist nur, dass die *Suche* in diesem Raum ohne Gradienten auf MNIST das Niveau des vollen Trainings erreicht — ein Befund über Lernen im diskreten Raum.
8. **(F) Was heißt „Magie“ — messen Sie Nichtstabilisierbarkeit?** — Nein: ich zähle nicht-Clifford-Winkel. Die gemessene Stabilizer-Rényi-Entropie (I-002) zeigte keinen Zusammenhang zur Nutzung (6,26 vs. 6,28, H0).
9. **(J) Sind Ihre Ergebnisse reproduzierbar? Wie viele Seeds? Wie viele Hypothesen haben Sie getestet?** — 10–20 Seeds, TOST/vorregistriert für I-005-v2, mehrere Replikationen scheiterten und sind dokumentiert (I-034-fashion, C-skala, I-035). Vollständige Liste der Tests für Mehrfachvergleichs-Korrektur fehlt noch → Tabelle „alle Hypothesen, H1/H0/HU“ erstellen.
10. **(F) Dequantisierung: Warum sollte man Ihren Ansatz nicht klassisch (z. B. per Sampling) ersetzen?** — Er *ist* klassisch. Die Dequantisierungs-Literatur (Tang, Chia et al.) zeigt, dass Vorteile bei niedrigem Rang und Sampling-Zugriff verschwinden; QIRE behauptet keinen Rechenzeitvorteil, nur Parameter-/T-Sparsamkeit.
11. **(J) Was bringt es dem Klima/der KI-Effizienz?** — Bisher nichts Gemessenes. Gemessen sind Parameterzahlen, nicht Energie oder FLOPs auf Hardware. Die Gleichung „weniger Parameter = weniger Energie“ ist [?] ungeprüft; Walsh-Block spart Multiplikationen (Zählung), keine Wattmessung.
12. **(F) Wie unterscheidet sich Ihre „Suche“ von Zufallssuche mit gutem Gerüst?** — I-004: zufälliges Clifford-Gerüst + 4 Magiewinkel + ein Training reicht (G = 0,69), I-001: gewählte Winkel > Zufall (p = 0,027). Aber: bei MNIST holt der feste 2×2-Block 72 % des Abstands; die Suche liefert +0,93 pp. Das ist klein; Streuung ±0,5 pp.
13. **(F) Kann man die Holevo-Grenze umgehen?** — Nein; sie begrenzt Bits pro Auslese (n Qubits → ≤ n Bit klassisch). Was Amplituden-/Born-Auslese „mehr“ liefert, kommt aus vielen Wiederholungen (Schüsse). Diese Frage gehört nicht in die Tragzeile der Arbeit, außer als Abgrenzung [Wissen]; Detail im Holevo-Bericht.
14. **(J) „Unendlich skalierbar?“** — Nein. Laden ist exponentiell (7.636 T) bzw. nur für strukturierte Daten billig; Schuss-Rauschen wächst mit 2ⁿ Ausgängen; Barren Plateaus für globale Verluste. Mit klassischer Simulation: 2ⁿ Speicher. Ehrliche Antwort: Skalierung ist offen, bisher bis 8 Qubits geprüft.
15. **(F) Was ist Ihr stärkstes *nicht*-triviales Ergebnis, das auch dann bleibt, wenn alle Kontrollen gegen Sie laufen?** — Kandidaten: (a) dass diskrete 0-T-Suche auf MNIST gleichwertig ist und auf Fashion ≈ 4 Magiewinkel fehlen (TOST, 20 Seeds); (b) dass Maske und Blöcke nur gemeinsam tragen (Wechselwirkung +2,1/+2,7 pp); (c) die vollständige Dokumentation der gescheiterten Vorhersagen. (c) ist wissenschaftlich unterschätzt: Negativbefunde mit Protokoll sind selten.

## 5. Benchmark-Fallen — Checkliste gegen QIRE

| Falle | QIRE-Status | Aktion |
|---|---|---|
| Binäre MNIST-Teilaufgaben (0 vs 1) | nicht betroffen: 10 Klassen; „Ziffern ×2“ war Zusatz | in Arbeit klar trennen, Zahlen nur für 10 Klassen |
| PCA-Vorverarbeitung | nicht verwendet (Amplitudenkodierung auf 16×16) — aber 16×16-Resize ist Informationsverlust | klassische Baselines auf *derselben* 16×16-Eingabe |
| Unfaire Baselines | ungeklärt [?] | K1: Baselines mit gleicher Parameterzahl/gleicher Tuning-Mühe, vorregistriert |
| Gesamtparameter | Q-020 falsch gezählt, Plan §1a korrigiert | immer Gesamtzahl + Kopf/Encoder getrennt |
| Datenleck/Test-Set-Auswahl | ungeklärt; Hyperparameter (Lernrate!, „Störfaktor Lernrate“ bei C-referenz) | Validierungsteil für Wahl, Test einmalig; im LABORBUCH vermerkt? |
| Mehrfachtests / Gewinner-Fluch | viele H0 dokumentiert (gut) | Gesamttabelle, Holm-Korrektur |
| Nur 2 Datensätze, beide Bilder | MNIST, Fashion | + KMNIST oder EMNIST-Letters als dritter (billig) |
| Simulationsrauschen vs. Hardware | nur Simulation | IBM-Lauf (kleiner Test) als Pluspunkt, keine Vorteilsbehauptung |

## 6. Neue Ideen / Kontrollexperimente [EIGEN]

Neuheitsprüfung generell: Suchbudget erschöpft; für jede Idee „kurz in Auszügen gesucht (Bowles, Cerezo, Gil-Fuster, KronA, BOFT) — nichts Gleiches gefunden“ gilt nur für die genannten Titel, **nicht** für eine systematische Suche [?].

### K1. Parameter-gleiche klassische Baselines (Einwand B1, Fragen 5/6)
- *Frage:* Liegt die QIRE-Zelle (≈ 2.600 Parameter) über der Pareto-Kurve „Genauigkeit vs. Gesamtparameter“ klassischer Modelle auf derselben 16×16-Eingabe?
- *Test (PC):* Softmax auf 256 Pixeln (2.570), Softmax auf Pixeln+Quadraten, MLP 256→8→10 (≈ 2.200), 256→10 mit DCT-Vorstufe, kleines CNN (≈ 2.600). 10 Seeds, gleiche Epochen/Lernratensuche vorregistriert. Laufzeit ca. 10–20 min auf 6 Kernen (CPU, Mini-Netze).
- *Kontrolle:* gleiche Tuning-Mühe (z. B. je 8 Lernraten).
- *Ergebnis wenn H0 (klassisch ≥ QIRE):* Effizienzanspruch streichen, Struktur-/Suchbefunde behalten.
- *Stufe:* Q1 (Kontrolle). Priorität **1**.

### K2. Orthogonal-Kontrollen (B1/B3, Fragen 2/3)
- *Frage:* Hängt der Gewinn an „Ry+CNOT“ oder an jeder orthogonalen Mischung + Quadrat?
- *Test:* Q ersetzen durch (a) Haar-zufällige Orthogonale (fest), (b) Cayley-parametrisierte orthogonale 256×256, trainiert, (c) Butterfly-Givens mit 32 Winkeln (BOFT-artig), (d) Hadamard·Diag(±1)·Permutation zufällig (Fastfood-artig), (e) feste DCT. Alle mit gleichem Kopf. Jeweils ~2–5 min × 10 Seeds.
- *Neu?* Gegenprobe klassisch zur Ry-Struktur; Teil von (d) ≈ Fastfood [Wissen].
- *Stufe:* Q1. Priorität **1**.

### K3. Verschränkungs-Ablation (B9, Bowles-Test)
- *Frage:* Zeigt sich der Gewinn auch ohne CNOT-Ring?
- *Test:* L Lagen Ry ohne CNOT = Kronecker 2×2-Rotationen; plus Variante mit Zufalls-CNOT; Vergleich gegen QIRE. Quadrat/Kopf unverändert. ~3 min × 10 Seeds.
- *Erwartung [?]:* Nahe Null-Unterschied (I-036: fester 2×2-Block holt 72 %). Das wäre ein ehrlicher, mutiger Befund: „Verschränkung trägt Δ pp“.
- *Stufe:* Q1 → klärt, ob Q2/Q3 überhaupt gerechtfertigt. Priorität **1**.

### K4. Zerlegung der Nichtlinearität (B1)
- *Frage:* Wie viel trägt das **Quadrieren (Born)** vs. die **Normierung** vs. die **Orthogonale**?
- *Test:* 2×2×2-Design: {Q, Identität} × {Quadrat, |.|, ReLU, linear} × {Normierung ja/nein}. Lineare Variante mit Q ohne Quadrat ist wertlos (Q ändert nichts am Kopf, Kopf lernt Q mit). Das beweist, dass die **Nichtlinearität** der Träger ist.
- *Stufe:* Q1. Dauer < 15 min.

### K5. Zufallslabel- und Permutationskontrolle (Gil-Fuster 2024)
- *Frage:* Memorisiert QIRE mit 2.600 Parametern Zufallslabels (Trainingsgenauigkeit)? Und überträgt die Clifford-Maske auf **pixelpermutierte** MNIST (Bildstruktur zerstört)?
- *Test:* (a) Zufallslabels (10 % Rauschen/50 %/100 %), Trainingskurve; (b) feste Pixelpermutation: Suche neu, Rang/Walsh-Struktur vergleichen. Wenn Rang-Effekt nur bei Bildern → er kodiert 2-D-Nachbarschaft (Neuheitswert erhöht); wenn gleich → Kapazitätseffekt.
- *Stufe:* Q1. ca. 10 min je Seed-Satz.

### K6. Gleich-Rang-Kontrolle der Maske (B4)
- *Frage:* Trägt der **Rang** oder die **Struktur** der quadratischen Form?
- *Test:* Formen mit fest gewähltem Rang r (r = 0…8) *zufällig* gezogen vs. durch Suche gefundene; Gewinn vs. r mit CIs. Zwei Fälle: Suchmaske, Zufallsmaske gleichen Rangs. Bis zu 20 Seeds, ~15 min.
- *Stufe:* Q1/Q3 (Clifford-Struktur). Teil von I-036-v3 [?, ob schon enthalten].

### K7. „Klassischer Clifford-Raum“ (B3)
- *Frage:* Ist der Clifford-Raum besser als klassische diskrete Transformationen gleicher Größe?
- *Test:* Dieselbe diskrete Suche (Evolution/Hill-Climbing mit gleichem Budget), aber Suchraum = {Signed-Permutationen} oder {Diag(±1)·H-Blöcke·Permutation} statt Clifford. ~15 min.
- *Neu?* Suche nach „Walsh/Hadamard-Merkmale + diskrete Suche + MNIST“ nicht durchgeführt (Budget) [?].
- *Stufe:* Q1.

### K8. PEFT-Parität (B8, T-005)
- *Frage:* Ist die QIRE-Kronecker-Rotation besser als KronA/BOFT/LoRA bei **gleicher Parameterzahl** auf LLäMmlein-120M (oder verkleinertem Proxy: 2–4 Schichten, ~10M)?
- *Test:* Proxy-Schleife auf CPU ist knapp; billig: gleiche T-004-Konfiguration, Rang-/Faktorgrößen so wählen, dass ≈ gleiche Parameter; 3 Seeds. Laufzeit [?] Stunden → nur Container/PC über Nacht.
- *Stufe:* Q1.

### K9. Vorregistrierung & Hypothesenliste (Frage 9)
- *Test:* Tabelle aller H1/H0/HU; einheitliche Mehrfachkorrektur (Holm) und eine „Entdeckungsrate“-Zeile. Kein Rechenaufwand, großer Glaubwürdigkeitsgewinn.

### K10. Hardware-Realitätscheck (B10)
- *Frage:* Bleibt die Genauigkeit der Clifford-Zelle (0 T) auf IBM-Hardware/Rauschmodell mit 1.000 Schuss erhalten?
- *Test:* Qiskit-Aer mit Rauschmodell zuerst (PC, min), dann ≤ 100 Test-Bilder auf echtem Gerät (Token neu erzeugen). Klar als „Demonstration, kein Vorteil“ benennen.
- *Stufe:* Q2.

**Reihenfolge:** K3 + K1 + K2 (zusammen < 1 h PC) entscheiden, ob die Kernbotschaft „Struktur ja, Quantum nein“ lautet. Danach K4/K5/K6.

## 7. Empfohlene Formulierungen für die schriftliche Arbeit

- **Statt** „Mit Quantenphysik ressourceneffizientere KI“ → „Eine von Quantenschaltungen inspirierte orthogonale Struktur mit wenigen Winkeln, klassisch simuliert; Quantenhardware als offene Option.“
- **Statt** „Die Quantenschaltung schlägt …“ → „Ein Merkmalsabbild aus 32 Winkeln (Ry+CNOT) verbessert die lineare Auslese um 3,8 pp gegenüber Pixelquadraten.“
- **Statt** „0 T-Gatter genügen“ → „Eine Clifford-Schaltung (klassisch simulierbar nach Gottesman-Knill) erreicht auf MNIST das Niveau des Gradiententrainings; auf Fashion fehlen 1,31 pp.“
- **Statt** „Magie-Konzentration der Aufgabe“ → „Anzahl der Nicht-Clifford-Winkel, die das Gradientenverfahren für 50 % des Nutzens braucht; abhängig vom Optimierer (I-005)“.
- **Statt** „16 Winkel schlagen ein MLP“ → „16 Winkel + Encoder mit 200.960 Parametern (≈ 201.000 gesamt)“.
- **Statt** „2,25 % der T-Gatter“ → „Auf MNIST bei Referenz X; auf Fashion nicht repliziert.“
- Immer: **Gesamtparameter, Baselines mit gleichem Budget, Seeds, Konfidenzintervalle, gescheiterte Replikationen.**
- Terminologie: „quantum-inspired“ nur für klassisch gerechnete Teile; „Quanten…“ nur, wo Hardware oder Qubit-Rechnung wirklich nötig ist.

## 8. Offene Fragen für Etappe 2

1. Systematische Neuheitssuche (Budget war erschöpft): Fastfood/Walsh-Merkmale + diskrete Suche; „Clifford circuit learning classification“; Stabilizer-/Clifford-Lernen im QML (Clifford-QNN?).
2. Gibt es Literatur zu **Ry-only (reelle) Amplitude-Encoding-Klassifizierer = orthogonale Merkmalsabbildung**? Ob jemand diese Identität in QML-Papern explizit macht [?].
3. Wie sähe ein Test aus, der Verschränkung *nötig* macht (Aufgabe mit nichtlokaler Struktur, z. B. Paritäts-/Schachbrett-Muster; B-paritaet im Institut)?
4. Dequantisierung für QIRE: Gibt es eine formale Aussage „Born-Auslese mit Clifford-Q ist klassisch in O(n·2ⁿ) bzw. via Walsh-Transform“ — in Notation beweisen (einfach, nützlich für die Arbeit).
5. Energiemessung statt Parameterzahl (RAPL auf Ryzen?) für den „Ressourcen“-Anspruch.

## 9. Quellenliste

- Schuld, Killoran: Is quantum advantage the right goal for quantum machine learning? PRX Quantum 3, 030101 (2022). https://arxiv.org/pdf/2203.01340
- Bowles, Ahmed, Schuld: Better than classical? The subtle art of benchmarking quantum machine learning models (2024). https://arxiv.org/pdf/2403.07059
- Cerezo et al.: Does provable absence of barren plateaus imply classical simulability? Nat. Commun. 16, 7907 (2025). https://www.nature.com/articles/s41467-025-63099-6 ; https://arxiv.org/abs/2312.09121
- Gil-Fuster, Eisert, Bravo-Prieto: Understanding QML also requires rethinking generalization. Nat. Commun. 15, 2277 (2024). https://www.nature.com/articles/s41467-024-45882-z
- Schreiber, Eisert, Meyer: Classical surrogates for quantum learning models. PRL 131, 100803 (2023). https://www.researchgate.net/publication/361499932
- Huang et al.: Power of data in quantum machine learning. Nat. Commun. 12, 2631 (2021). https://www.nature.com/articles/s41467-021-22539-9
- Tang: A quantum-inspired classical algorithm for recommendation systems (2018/STOC 2019); Überblick https://ewintang.com/assets/2019-01-28-an-overview-of-quantum-inspired-sampling.pdf
- Chia, Gilyén, Li, Lin, Tang, Wang: Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing QML. J. ACM 69(5) (2022). https://arxiv.org/pdf/1910.06151
- Gharibian, Le Gall: Dequantizing the QSVT. STOC 2022 / SIAM J. Comput. https://arxiv.org/abs/2111.09079 ; Robust: https://arxiv.org/pdf/2304.04932
- Aaronson, Gottesman: Improved simulation of stabilizer circuits (2004). https://www.scottaaronson.com/papers/chp6.pdf
- Quantum-inspired Terminologie/Überblick: https://arxiv.org/html/2604.14287 ; https://arxiv.org/pdf/2308.11269 (nur Snippets)
- Benchmark-Fallen-Sammlung (nur Snippets, ungeprüft): https://arxiv.org/html/2604.18837 ; https://arxiv.org/html/2608.15617v1
- KronA: https://arxiv.org/pdf/2212.10650 ; BOFT: https://arxiv.org/pdf/2311.06243 ; Kron-LoRA https://arxiv.org/pdf/2508.01961
- Repo: `INSTITUT/UEBERGABE.md` §0/§4; `INSTITUT/PLAN_2026-10-05_effizienz.md` §1a/§1b; `INSTITUT/LABORBUCH.md` (Identität/Pixelquadrate 0,9015/0,80); `recherche/2026-09-28_neuheit_magie.md`.
- [Wissen, nicht in dieser Sitzung belegt]: Fastfood (Le/Sarlós/Smola 2013), Aaronson „Read the fine print“ (2015), Holevo 1973, Bent-Funktionen/Walsh-Spektrum.
