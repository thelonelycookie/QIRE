# 06 Skalierungsgesetze: klassisch, quanten, und was „unendlich skalierbar“ heißen könnte

Stand 05.10.2026. Recherche-Sonnet. Hinweis zur Quellenlage: arxiv.org und zenodo.org sind per Proxy gesperrt, das WebSearch-Budget (200) ist aufgebraucht. Fast alles unten ist daher **nur Auszug** (Suchsnippets), nicht gelesen im Volltext. Neuere Arbeiten (arXiv-Nummern 26xx) kenne ich nur aus Snippets.

## 1. Kurzfassung

- Klassisch gilt: Verlust ≈ E + A/N^α + B/D^β (Kaplan 2020, Chinchilla 2022). Die Form ist solide, die Konstanten sind empfindlich (Chinchilla-Fit wurde von Besiroglu et al. als fehlerhaft nachgerechnet). Sättigung („E“) ist die am schlechtesten bestimmte Größe.
- „Unendlich skalierbar“ ist als Aussage nur mit Achse und Maß sinnvoll: Parameter, Daten, Rechenzeit, Qubits, Tiefe, Magie. Mein Vorschlag in §3: drei prüfbare Lesarten statt eines Schlagworts.
- Quantenmodelle: Erste empirische Gesetze existieren (QViT: gemeinsames „verschobenes Potenzgesetz“ in Qubits, Tiefe, Datenmenge; RydbergGPT: Potenzgesetz mit Floor nur nahe am kritischen Punkt). Generalisierung wächst nach Caro et al. höchstens wie √(T/N). Überparametrisierung (Larocca) ist an die Dimension der dynamischen Lie-Algebra gebunden, also eine **harte Sättigung**.
- Eigener Fit an T-005 (4 Punkte, L = 1, 2, 5, 10): Der Gewinn wächst wie L^0,55, ohne erkennbare Sättigung. Mit 4 Punkten sind Potenzgesetz, log-linear und „Potenzgesetz + Sättigung“ **nicht unterscheidbar** (RSS ~5–8·10⁻⁵ bei allen). Extrapolation auf L = 100 streut von 2,30 bis 2,40.
- Für den Adapter ist die plausibelste Struktur-Grenze: Der QIRE-Adapter hat nur O(L·log d) Winkel. Seine Ausdrucksstärke ist beschränkt, deshalb **muss** es eine Sättigung geben (Größenordnung unbekannt). Das ist testbar und billig.

## 2. Stand der Forschung

### 2.1 Klassische Skalierungsgesetze
- **Kaplan et al. 2020**, Scaling Laws for Neural Language Models: Test-Verlust als Potenzgesetz in Modellgröße, Datenmenge, Rechenaufwand über mehr als sieben Größenordnungen. Kritik: sagt die Verlustkurve teils zu pessimistisch voraus; beschreibt Verlust, nicht einzelne Aufgaben (Snippet-Quellen: https://aclanthology.org/2025.acl-long.1163.pdf, https://github.com/AkihikoWatanabe/paper_notes/issues/1828). Nur Auszug.
- **Hoffmann et al. 2022 (Chinchilla)**, https://arxiv.org/pdf/2203.15556: über 400 Modelle (70 M bis >16 Mrd. Parameter, 5–500 Mrd. Token); Modellgröße und Tokenzahl gleich skalieren. Nur Auszug.
- **Besiroglu et al. 2024**, Chinchilla Scaling: A replication attempt, https://arxiv.org/pdf/2404.10102: drei Probleme am Fit von Ansatz 3 (unplausibel enge Konfidenzintervalle, Optimierer vor Konvergenz gestoppt, gerundete Parameter). Lehre für uns: **Fits an wenigen Punkten mit Sättigungsterm sind fragil**, auch bei Profis. Nur Auszug.
- **Caballero et al., Broken Neural Scaling Laws**, https://arxiv.org/pdf/2210.14891: glatt „gebrochenes“ Potenzgesetz, modelliert Knicke, Double Descent, nicht-monotone Verläufe. Relevanz: Wenn QIRE-Kurven bei größerem L knicken, ist ein einfaches Potenzgesetz falsch. Nur Auszug.
- **Muennighoff et al. 2023, Data-Constrained LMs**, https://jmlr.org/papers/volume26/24-1000/24-1000.pdf: bis ~4 Epochen Wiederholung fast wie neue Daten, danach verfällt der Wert zusätzlicher Rechenzeit gegen null; Gesetz mit abnehmendem Wert wiederholter Token und überschüssiger Parameter. Relevanz: Maltes Trainingsdaten sind klein (MNIST 60k, Chat-Korpus klein), wir sind **immer im datenbeschränkten Bereich**. Nur Auszug.
- **Fine-Tuning-Gesetze**: Zhang et al. 2024 geben für LoRA L(D) = A/D^λ + E (Snippet-Zitat, https://arxiv.org/pdf/2512.14237 zitiert es). Biderman et al. 2024 (LoRA Learns Less and Forgets Less, https://github.com/danbider/lora-tradeoffs): LoRA mit niedrigem Rang bleibt hinter Voll-Finetuning, Rang 256 gleicht weitgehend aus; Voll-Finetuning lernt Updates mit 10–100-fach höherem Rang. Nur Auszug.
- Überblicksblog (nur Auszug): https://lilianweng.github.io/posts/2026-06-24-scaling-laws/

### 2.2 Strukturierte Matrizen, Skalierung der Parameter
- **Monarch** (Dao et al. ICML 2022), https://proceedings.mlr.press/v162/dao22a.html: Produkt blockdiagonaler Faktoren mit Permutation, enthält Butterfly/Hadamard/Fourier; ViT/GPT-2 etwa 2× schneller bei vergleichbarer Güte. Parameter ~ d^1,5 (bei zwei Faktoren). [Eigene Rechnung aus Definition, nicht aus Quelle.]
- **Monarch Mixer**, https://arxiv.org/pdf/2310.12109: gleicht GPT-artige Transformer bei 360 M Parametern in Perplexität auf The Pile; BERT-Güte mit bis zu 27 % weniger Parametern. Nur Auszug.
- **BOFT** (Liu et al.), https://arxiv.org/pdf/2311.06243: orthogonale Butterfly-Faktorisierung, O(log d) dünne Faktoren, **O(d log d) Parameter**; schlägt LoRA/OFT bei NLU und Mathe. Nur Auszug. (Das ist die nächste klassische Verwandte des QIRE-Adapters, aber mit d log d statt log d Parametern.)
- **KronA / LoKr**, https://arxiv.org/abs/2212.10650, https://arxiv.org/pdf/2508.01961 (Kron-LoRA: bis 4× weniger Parameter als LoRA bei ähnlicher Ausdrucksstärke, Auszug). Kronecker-Produkt: Rang multipliziert sich, Parameter addieren sich.
- **Compute Better Spent** (Qiu et al. ICML 2024), https://proceedings.mlr.press/v235/qiu24f.html: Low-Rank, Monarch, Kronecker, BTT als Ersatz dichter Schichten; **Skalierung von Breite und Lernrate hängt von der Struktur ab** (dicht: Breite ~ C^(1/2), Blockdiagonal ~ C^(2/3)); BTT gleicht dichtes ViT-S/32 auf ImageNet mit 3,8× weniger Rechenaufwand. Nur Auszug. Wichtig für uns: **Lernraten- und Init-Skalierung muss mit der Breite mitwandern** (µP-artig), sonst sind Skalierungsgesetze Artefakte.
- Quantum-PEFT (https://arxiv.org/abs/2503.05431): Pauli-Parametrisierung, Parameterzahl **logarithmisch in der Dimension**, Rang voll. QuanTA (https://arxiv.org/abs/2406.00132, NeurIPS 2024): quantenschaltkreis-inspirierte Tensoren, hochrangig. QuIC (https://arxiv.org/html/2502.06916): < 0,02 % Speicher. Alle nur Auszug. **Diese drei sind die engste Konkurrenz/Vorarbeit zum QIRE-Adapter**; das gehört in die schriftliche Arbeit (Abgrenzung).

### 2.3 Quantenmodelle
- **Empirical Scaling Laws of Quantum Vision Transformers**, https://zenodo.org/records/23005853: Validierungsverlust gut durch **gemeinsames verschobenes Potenzgesetz in Qubitzahl, Transformer-Tiefe und Datenmenge** beschrieben; in Einheiten klassischer Encoder-Parameter kann mehr Qubits weniger klassische Parameter ausgleichen. Nur Auszug; Autoren, Exponenten, Bereiche **nicht** gesehen. Wichtig: Das ist die direkteste Vorarbeit zu Idee E1/E3, und es ist eine Zenodo-Preprint-Quelle (Prüfstatus unklar).
- **Do Quantum Models Scale Like LLMs?** (RydbergGPT), https://arxiv.org/abs/2609.20912: Transformer auf Messdaten von Rydberg-Atomen; Verlust gegen Datenmenge folgt **nahe am kritischen Punkt** einem Potenzgesetz mit Floor, weg davon deutlich schlechter. Deutung: Skalierung ist Eigenschaft des **Modell-Daten-Paars**, hängt an mehrskaliger Korrelationsstruktur. Nur Auszug. Das ist ein Transformer auf Quantendaten, **kein** Quantenmodell; der Titel führt leicht in die Irre.
- **Scaling Laws for Hybrid QNNs**, https://arxiv.org/abs/2604.06007 (Vyskubov et al.): mehr Qubits verbessert meist, mehr Quantenlagen **nicht zuverlässig**. Auszug. (Gegenpunkt zu T-005, wo Lagen monoton helfen; dort aber Sprach-Adapter mit anderem Aufbau.)
- **Scaling Behavior of PQCs, Lie-algebraic**, https://arxiv.org/abs/2608.20603 (Ohno): Daten-, Modell-, Rechenskalierung auf Regressionsaufgabe; **Dimension der dynamischen Lie-Algebra und Observablen-Orbit sättigen schnell mit der Parameterzahl, die Jacobi-Rang-Dimension bleibt mit Parametern korreliert** → effektive Dimension statt nominale Parameter als Skalierungsachse. Auszug.
- **Caro et al. 2022 (Nat. Commun.)**, https://arxiv.org/abs/2111.05292: Generalisierungsfehler ≲ √(T/N), T = trainierbare Gatter, N = Trainingsdaten; ≲ √(K/N), wenn nur K ≪ T Gatter sich wesentlich ändern. Auszug. Für QIRE: T ist winzig (L·8 Winkel), der Quantenteil kann kaum überanpassen; der **lineare Kopf mit 2570 Parametern** dominiert die Schranke.
- **Larocca et al. 2023 (Nat. Comput. Sci.)**, https://pubmed.ncbi.nlm.nih.gov/38177434/: Überparametrisierung beginnt bei kritischer Parameterzahl M_c ≤ dim(Lie-Algebra); darunter falsche lokale Minima, darüber gut trainierbar. Auszug. Für Ry+CNOT-Ring (reell, Orthogonalgruppe auf 2⁸ = 256 Dimensionen, dim so(256) = 32 640) liegt M_c bei L ≤ 10 (80 Winkel) **weit** unter der Sättigung. [Eigene Rechnung, nicht geprüft, ob die dynamische Lie-Algebra des Ansatzes wirklich so(256) ist; für Ry+CNOT ist das plausibel, aber [?].]
- Magie und T-Count (Auszug): Eine Arbeit (https://arxiv.org/pdf/2502.09725 oder Verwandtes; die Zuordnung im Snippet war unklar [?]) berichtet, dass die Hessematrix mit dem T-Count wächst, und eine **Sättigung bei T-Count ≈ 10**. Nicht verifiziert. Stabilizer-Renyi-Entropie ist nur bis etwa 15 Qubits direkt berechenbar (https://arxiv.org/html/2601.07824, Auszug).

### 2.4 „Unendlich skalierbar“: Lesarten
Ich konnte (Budget) die Literatur zu Barren Plateaus/Dequantisierung nicht mehr nachschlagen; aus dem Gedächtnis (ohne Quelle, [?]): Exponentielle Konzentration und Dequantisierung begrenzen die Rolle tiefer, unstrukturierter Quantenmodelle. Die Lesarten:
1. **Parameter-Skalierung**: Verlust fällt mit Breite d bei O(log d) Parametern weiter (nie sättigend?). Antwort aus Struktur: nein, es sättigt, weil Ausdrucksstärke ≤ Gruppe der Kronecker-Drehungen.
2. **Ressourcen-Skalierung**: Qubits n wachsen, Aufwand wächst nur polynomiell in n bei fester Güte pro Breite. Klassische Simulation der Clifford-Zelle ist exakt polynomiell (Gottesman-Knill); der Quantenvorteil wäre dann null. Das ist die ehrliche Spannung im Projekt.
3. **Daten-/Rechen-Skalierung**: Verlust fällt mit Rechenzeit als Potenzgesetz ohne Wand (Chinchilla-artig). Das ist die einzige Lesart, die man heute **messen** kann.

## 3. Was das für QIRE bedeutet

**Eigener Fit an T-005 (Tagebuch `UEBERARBEITUNG_2026-09-23/A/PROJEKTTAGEBUCH.md`, Eintrag 30. Sept.).** Daten: Val-Verlust bei L = 1, 2, 5, 10: 2,5174 / 2,4964 / 2,4442 / 2,4115; Basis (L = 0) 2,5616. Rechnung in `…/scratchpad/fit.py` (nicht Teil des Repos).

**WARNUNG: Vier Punkte sind sehr wenig.** Ein Modell mit 3 Parametern hat 1 Freiheitsgrad. Alles Folgende ist beschreibend, keine Vorhersage. Seeds streuen sd 0,0004–0,0029, die Mittelwerte sind also genau, aber vier Punkte bleiben vier Punkte.

| Modell | Parameter | Ergebnis | RSS |
|---|---|---|---|
| Gewinn = a·L^b (Basis fest) | 2 | b = 0,546, a = 0,045; Residuen bis 0,009 bei L = 5 und 10 | 1,0·10⁻⁴ (Summe Quadrate der Gewinn-Residuen) |
| Verlust = c − d·ln L (log-linear) | 2 | c = 2,522, d = 0,0475 | 7,8·10⁻⁵ |
| Verlust = E + A·L^(−α), α frei | 3 | Fit **degeneriert**: bester Wert E = 2,71 mit **negativem** A, α = −0,20, also kein Sättigungsfit, sondern L^0,2 (Gewinn wächst unbegrenzt). Mit freiem E>0, α>0 und festem E: siehe unten | 4,7·10⁻⁵ |
| Sättigung bei fest gewähltem E | 2 (Profil) | E = 2,0: α = 0,10; E = 2,2: α = 0,18; E = 2,3: α = 0,30; E = 2,35: α = 0,44; E = 2,38: α = 0,64; E = 2,40: α = 0,99 | 1,2·10⁻⁴ … 1,8·10⁻³ |

Lesart:
- Die Daten **bevorzugen keinen harten Boden** nahe 2,41 (E ≥ 2,40 passt schlecht, RSS 20-fach höher); sie sind aber mit jedem Boden bis herunter zu etwa 2,2 vereinbar. Wo der Boden liegt, ist **nicht bestimmt**.
- Vorhersage bei L = 100 reicht je nach Modell von 2,30 (log-linear, reines Potenzgesetz) bis 2,37–2,39 (Sättigung bei E = 2,3 / 2,38). Das ist die ehrliche Unsicherheit. Ein Zahlenwert bei L = 100 wäre **nicht** belegt.
- Plausibilitätsgrenze: Log-linear „−0,0475 pro e-fache Lagenzahl“ kann nicht ewig gelten. Die LoRA-Referenz (2,2875 mit 30.720 Parametern) liegt unter der Extrapolation bei L ≈ 100–1000; ob QIRE dorthin kommt, ist offen. Der Adapter hat bei L = 100 erst 19.200 Parameter.
- Gewinn pro Parameter fällt von 2,3·10⁻⁴ (L1) auf 7,8·10⁻⁵ (L10), also **abnehmender Grenznutzen**; der Gewinn selbst wächst wie ~P^0,55 (sublinear). Das ist wie bei LoRA-Rang-Skalierung erwartbar (Auszug Biderman).
- Vorsicht, die Parameterzahl ist hier proportional zu L (192·L); „Skalierung in L“ und „in Parametern“ sind identisch, und die Breite (768?) wurde nicht variiert. [Breite der LLäMmlein-Matrizen aus meiner Sicht nicht geprüft.]

**Was „unendlich skalierbar?“ für Malte praktisch heißt:** Die ehrlichste Formulierung für die Arbeit: „Der Adapterverlust fällt bis L = 10 streng monoton; die Kurve ist mit einem Potenzgesetz (Exponent ≈ 0,5 im Gewinn) vereinbar, eine Sättigung konnte nicht nachgewiesen werden; vier Punkte erlauben keine Extrapolation.“ „Unendlich“ lässt sich weder messen noch behaupten; messbar sind Exponenten und der Ort eines Knicks.

**Strukturelle Erwartung (eigene Überlegung, [EIGEN]):** Ein Produkt aus L Kronecker-Drehungen mit je log₂ d Winkeln erreicht höchstens O(L·log d) Freiheitsgrade; Ry+CNOT erzeugt nur orthogonale Matrizen. Klassische Entsprechung: BOFT mit O(d log d), Monarch mit O(d^1,5). Der Adapter liegt ganz unten auf der Parameter-Skala; es gibt daher **mehr** Raum, aber dort ist Sättigung früher zu erwarten.

**Konsequenz für Regeln der Skalierung bei QIRE:**
1. Lernrate/Init **mit der Breite skalieren** (Compute Better Spent), sonst ist ein Breiten-Exponent wertlos.
2. Datenbeschränkung beachten (Muennighoff): Mehrfach-Epochen auf kleinem Korpus verfälschen Exponenten; Epochenzahl und Datengröße im Dokument angeben.
3. Magie: Bei MNIST sättigt der Gewinn mit 0 T-Gattern, bei Fashion bei ~4 Winkeln (Auftrag). Magie ist also **kein** knapper Rohstoff bei diesen Aufgaben; ein Skalierungsgesetz des Magiebedarfs ist hier ein Test auf „Aufgabenschwierigkeit“, nicht auf Quantenvorteil.

## 4. Neue Ideen / Forschungsfragen

Neuheitsprüfung: Das Suchbudget war erschöpft, bevor ich Neuheitsprüfungen für jede Idee durchführen konnte. Was unten als „gesucht“ steht, stammt aus den 19 Suchen davor und ist **keine** systematische Prüfung. Alle Neuheitsaussagen als [?] behandeln und in Etappe 2 nachprüfen.

### E1 [EIGEN] Skalierungsgesetz in log-Breite für O(log d)-Adapter
- **Frage:** Wie fällt der Adapter-Gewinn mit der Breite d (Schichtbreite 64 … 4096) bei Winkelzahl ∝ log₂ d je Lage und fester Lagenzahl? Gilt Gewinn ∝ (log d)^γ, oder d^γ, oder ist er konstant (dann „skaliert“ der Adapter nicht mit der Breite)?
- **Warum neu:** Gefunden: Quantum-PEFT behauptet log-Parameterskalierung (Parameterzahl), QuanTA Skalierung der Qualität, BOFT d log d. **Gesucht, nicht gefunden:** ein Gesetz „Verlustgewinn gegen Breite“ mit explizitem Exponenten bei log-d-Parametern auf wachsender Breite. Nur Snippet-Ebene, [?].
- **Billigster Test (PC):** Kleine Transformer/MLPs mit Breite 32/64/128/256/512 (Zeichenmodell oder MNIST-Vortraining), je Breite vortrainiertes Basismodell, dann QIRE-Adapter L fest (z. B. 5) vs LoRA r1; 5 Seeds. Etwa 1–2 h CPU, wenn Basismodelle klein (<1 M Parameter). Alternative ohne Basismodell: lineare Teacher-Student-Aufgabe, Teacher = zufällige Rotation nahe Identität, Student = Adapter. Minuten.
- **Kontrolle:** LoRA r1 gleicher Breite; Adapter mit zufällig festen Winkeln (kein Training); Breiten-skalierte Lernrate vs feste Lernrate.
- **Quantenstufe:** Q1 (Simulation klassisch). Auf echter Hardware nur bei Breite ≤ 2⁵ sinnvoll.

### E2 [EIGEN] Chinchilla für Quantenadapter: Parameter gegen Daten gegen Lagen
- **Frage:** Gibt es für den Adapter ein Optimum L*(D) bei fester Datenmenge D (Anzahl Token), also „wie viele Lagen lohnen sich bei wie viel Daten“? Caro sagt: Generalisierung ~ √(T/N); bei kleinem N sollte ein Überschuss an Lagen schaden. Frage: Zeigt sich das, oder bleibt der Verlust bei L = 10 monoton (wie T-005)?
- **Warum neu:** Gefunden: Caro-Schranke (Theorie), QViT-Gesetz (Qubits/Tiefe/Daten, klassisches Transformer mit Quantenanteil), Muennighoff (klassisch, datenbeschränkt). **Gesucht, nichts gefunden:** ein Datenmengen × Lagenzahl-Raster für einen **LLM-Adapter** mit Test von √(T/N). [?]
- **Billigster Test (PC):** Rasterdesign L ∈ {1, 2, 5, 10, 20, 40} × D ∈ {1/8, 1/4, 1/2, 1} des vorhandenen T-005-Korpus; 3 Seeds. T-005 brauchte 13,5 h für 56 Läufe (≈ 15 min/Lauf) → 72 Läufe ≈ 18 h: **zu groß**. Verkleinern: 100 statt 300 Schritte, nur L ∈ {1, 5, 20}, D ∈ {1/4, 1}: 18 Läufe, ≈ 2 h. Oder Vorprüfung an der MNIST-Zelle (Minuten pro Lauf).
- **Kontrolle:** gleiche Rastermessung für LoRA r1 (Parameter ~30 k) und für den Adapter mit zufällig gemischten Daten; Test, ob der Generalisierungsabstand (Train − Val) mit √(L/N) skaliert.
- **Quantenstufe:** Q1.

### E3 [EIGEN] Skalierungsgesetz des Magiebedarfs
- **Frage:** Wie viele Magiewinkel K (T-Gatter-Äquivalente) braucht eine Aufgabe, um X % des vollen Gewinns zu erreichen, und wie hängt K(X) von Aufgabenschwierigkeit (MNIST 0, Fashion ~4, schwerere Aufgaben) und Qubitzahl n ab? Hypothese: Sättigungskurve, nicht Potenzgesetz, wegen Larocca-Überparametrisierung ab ~M_c.
- **Warum neu:** Gefunden: Magie wächst mit T-Count in einer Arbeit zur Hesse-Struktur (Auszug, Zuordnung unklar [?]), SRE-Messung skaliert exponentiell mit n. **Gesucht, nichts gefunden:** „wie viele Nicht-Clifford-Winkel braucht ein Lernproblem“ als Funktion der Aufgabe. [?] Bezug zu I-036 (Rang der quadratischen Form trägt 52–78 %).
- **Billigster Test (PC):** Bestehende Clifford-Suche, dann K = 0, 1, 2, 4, 8, 16 Magiewinkel (Ry(π/4)-artig oder frei) zusätzlich, Verlust gegen K auf MNIST, Fashion, KMNIST, CIFAR-Grau 16×16; je 5 Seeds, 8-Qubit-Simulation. Etwa 1–3 h (bei Läufen wie den bisherigen, genau [?]).
- **Kontrolle:** K zufällige Winkel statt gesuchter; K Winkel auf zufällig gewählten Qubits; Vergleich mit Hesse-Befund.
- **Quantenstufe:** Q1; die Aussage „Magie ist knapp“ wäre ein Q3-Argument (Hardwarekosten in T-Gattern).

### E4 [EIGEN] Effektive Dimension als Skalierungsachse für QIRE-Zellen
- **Frage:** Ist der Verlust besser durch die Jacobi-Rang-Dimension der Zelle (Rang der Ableitung der 256 Born-Werte nach den Winkeln) vorhersagbar als durch die Parameterzahl, und ordnet das Clifford-Zellen (diskret) und trainierte Zellen auf eine gemeinsame Kurve?
- **Warum neu:** Gefunden: Ohno 2608.20603 nutzt den Jacobi-Rang für Regression mit Zufalls-Pauli-Generatoren (Auszug). **Gesucht, nichts gefunden:** Anwendung auf Bild-/Sprach-Adapter und Vergleich mit Clifford-Zellen. [?]
- **Billigster Test (PC):** Für L = 1…10 Jacobi-Rang der Zelle numerisch (256 × (8L) Matrix, SVD, Sekunden), Plot Verlust gegen Rang mit vorhandenen T-005-/Zellen-Daten. Minuten.
- **Kontrolle:** Rang bei Zufallswinkeln vs trainierten Winkeln; Plot auch gegen nominale Parameter.
- **Quantenstufe:** Q1.

### E5 [EIGEN] Sättigungstest für T-005 mit L = 20, 40 (nicht „Skalierungsgesetz“, sondern Messung)
- **Frage:** Wo knickt die Kurve? Das ist die direkte Lücke aus T-005 („nächste Schritte“). Ein 5. und 6. Punkt (L = 20, 40) entscheidet zwischen Potenzgesetz und Sättigung viel besser als jede Theorie.
- **Warum neu:** nicht anwendbar (reine Messung).
- **Test:** 2 Zellen × 8 Seeds × (≈ 15 min bei L = 10, linear in L, ≈ 30–60 min je Lauf bei L = 20/40, evtl. kürzere Schritte) ≈ 8–16 h; **über der 5-min-Regel**, Etappe-2-Lauf. Mit 4 Seeds und L = 20 allein ≈ 2–3 h.
- **Kontrolle:** Vorregistrierte Entscheidungsregel (z. B. Gewinn L10→L20 < 0,015 = Sättigung).
- **Quantenstufe:** Q1.

## 5. Offene Fragen für Etappe 2

1. Volltexte lesen (Proxy sperrt arxiv/zenodo; Alternativen: Semantic Scholar API, OpenAlex, Unpaywall, GitHub der Autoren): QViT-Gesetz (Exponenten, Bereiche), RydbergGPT, Ohno 2608.20603, Vyskubov 2604.06007, Quantum-PEFT.
2. Neuheitsprüfung für E1–E4 wiederholen (Google Scholar, Semantic Scholar, Papers-with-Code), besonders: Gesetz in log-Breite; Magiebedarf gegen Aufgabe; Chinchilla-artiges Raster für Adapter.
3. Dynamische Lie-Algebra von Ry+CNOT-Ring exakt bestimmen (Larocca-Kriterium M_c) und mit 8L Winkeln vergleichen.
4. Welche Breite haben die LLäMmlein-120M-Matrizen genau, und wie viele Winkel je Matrix bei L = 1 (192 gesamt)?
5. Klären, ob für die schriftliche Arbeit ein „Skalierungsgesetz“ überhaupt der richtige Rahmen ist; vielleicht reicht „Monotonie + Effizienz pro Parameter“ ehrlich.
6. Holevo-/Informationsgrenzen und Gottesman-Knill (Dequantisierung) gehören zu anderen Berichten; hier nur Schnittstelle: die Clifford-Zelle ist klassisch simulierbar, „Quantenvorteil in der Skalierung“ ist dann Ressourcenfrage (Speicher 2ⁿ vs Stabilisator-Tableau n²), nicht Rechenfrage. [?]

## 6. Quellenliste

- Kaplan et al. 2020 (Auszug über https://aclanthology.org/2025.acl-long.1163.pdf, https://github.com/AkihikoWatanabe/paper_notes/issues/1828)
- Hoffmann et al. 2022 https://arxiv.org/pdf/2203.15556
- Besiroglu et al. 2024 https://arxiv.org/pdf/2404.10102
- Caballero et al. https://arxiv.org/pdf/2210.14891
- Muennighoff et al. https://jmlr.org/papers/volume26/24-1000/24-1000.pdf
- Biderman et al. https://github.com/danbider/lora-tradeoffs
- Dao et al. Monarch https://proceedings.mlr.press/v162/dao22a.html; Monarch Mixer https://arxiv.org/pdf/2310.12109
- BOFT https://arxiv.org/pdf/2311.06243
- KronA https://arxiv.org/abs/2212.10650; Kron-LoRA https://arxiv.org/pdf/2508.01961
- Qiu et al. Compute Better Spent https://proceedings.mlr.press/v235/qiu24f.html
- Quantum-PEFT https://arxiv.org/abs/2503.05431; QuanTA https://arxiv.org/abs/2406.00132; QuIC https://arxiv.org/html/2502.06916
- Empirical Scaling Laws of QViT https://zenodo.org/records/23005853
- Do Quantum Models Scale Like LLMs? https://arxiv.org/abs/2609.20912; RydbergGPT https://arxiv.org/html/2405.21052v1
- Vyskubov et al. https://arxiv.org/abs/2604.06007; Ohno https://arxiv.org/abs/2608.20603
- Caro et al. https://arxiv.org/abs/2111.05292
- Larocca et al. https://pubmed.ncbi.nlm.nih.gov/38177434/
- Magie: https://arxiv.org/pdf/2502.09725, https://arxiv.org/html/2601.07824
- Repo: `UEBERARBEITUNG_2026-09-23/A/PROJEKTTAGEBUCH.md` (Eintrag „30. September — T-005“); Fit-Skript `/tmp/claude-0/-home-user-QIRE/567c975d-029e-55c2-910e-2e51ae0c2108/scratchpad/fit.py`
