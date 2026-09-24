# Ein eigenes QIRE-Modell — Konzept für ein von Grund auf gebautes, universelles Netz

Agent `forschung_modell`, Stand 24. September 2026. Grundlage: `BERICHT/00_REVIEW_BEFUNDE.md`,
`BERICHT/01_VERBESSERUNGEN_UND_ZUKUNFT.md`, `A/ENTWURF_T_SERIE_KERNEL.md`, `A/ENTWURF_ZEITREIHEN.md`,
`A/BEFUNDE_UND_RICHTUNGEN.md`, `A/quantum_sim/sequence_platform.py`, `sprachmodell.py`,
`quantum_layer.py`, sowie `A/LITERATUR_UND_ALLEINSTELLUNG.md` und `BERICHT/literatur.md`
(Agent `literatur` — hier weiterverwendet, nicht neu recherchiert, um Doppelarbeit zu vermeiden).

**Kein Wort hier ist gemessen**, außer wo ausdrücklich auf einen bestehenden Befund verwiesen
wird. Dieses Dokument ist ein Bauplan, kein Ergebnis.

---

## 0. Die Ausgangslage in einem Satz

Die bisherigen QIRE-Experimente zeigen: Eine Mittelschicht aus **Normierung → geteilter
Butterfly-Drehung (Kronecker-Produkt aus 2×2-Rotationen, CNOT-Ring als kostenlose Permutation)
→ quadratischer Auslese (Born-Regel)** erreicht mit 16 Parametern ungefähr, was ein
2.048-Parameter-Quadratmodell erreicht — aber das gilt nur für die Mittelschicht, nicht das
Gesamtmodell, weil der Encoder 784→256 die Ersparnis auffrisst, die Struktur erst ab 16.384
Dimensionen schneller als dicht ist (Faktor 64 über der trainierbaren Breite von 256), und
Verschränkung bisher nichts nachweislich beiträgt. Das eigene Modell in diesem Dokument ist der
Versuch, aus genau diesen drei Einschränkungen ein **durchgängiges** Bauprinzip zu machen, statt
sie nur an einer Schicht zu behandeln — und es an mehreren KI-Aufgaben zu prüfen, nicht nur an
Bildern.

---

## 1. Stand der Technik — woran sich ein eigenes Modell messen muss

Tabelle sortiert danach, wie nah die Idee an QIRE ist. **Passt** = Baustein direkt übertragbar.
**Grenzfall** = verwandtes Prinzip, andere Umsetzung. **Konkurrenz** = eigene Familie, die
dasselbe Ziel (Effizienz) anders löst und gegen die QIRE sich behaupten muss.

| Ansatz | Kernidee | Kosten | Einordnung für QIRE |
|---|---|---|---|
| **Butterfly-Faktorisierung** (Dao u. a., *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*, ICML 2019; *Kaleidoscope*, ICLR 2020) | Jede strukturierte lineare Abbildung (FFT, DCT, Permutationen …) als Produkt von O(log d) dünnbesetzten Butterfly-Faktoren | O(d log d) | **Passt direkt** — QIREs Mittelschicht ist mathematisch ein Spezialfall (geteilte statt freie Butterfly-Faktoren). Dao selbst zeigt, dass das *keine* Quantenidee ist, sondern klassische Algorithmentheorie (FFT-Butterfly-Diagramm seit Cooley-Tukey 1965) |
| **Monarch / Monarch Mixer (M2)** (Dao u. a., NeurIPS 2023, arXiv:2310.12109) | Zwei Block-Diagonal-Matrizen mit dazwischenliegender Permutation, sub-quadratisch in Sequenzlänge UND Modellbreite, GEMM-freundlich auf GPU | O(d^1,5) | **Grenzfall** — verwandte Idee (strukturierte Matrix statt dicht), aber Monarch ist auf GPU-Durchsatz optimiert (genau das Problem, an dem QIRE nach T-001 noch scheitert: 5,9× langsamer bei d=256). M2 ist die unmittelbare Konkurrenzarchitektur für „ressourceneffiziente Sequenzverarbeitung ohne Attention" |
| **Hyena** (Poli u. a., arXiv:2302.10866, 2023) | Implizite lange Faltungen per FFT statt Attention, O(L log L) in der Sequenzlänge | O(L log L) | **Grenzfall** — mischt über die Sequenz wie ein FNet/QIRE-Sequenzmischer, aber mit echten (nicht geteilten) Filterkoeffizienten. Konkurrenzarchitektur zur Sequenzmisch-Idee unten |
| **Mamba / S4** (Gu & Dao, arXiv:2312.00752, 2023) | Selektive State-Space-Modelle, linear in der Sequenzlänge, hardwarebewusster Parallel-Scan | O(L) | **Konkurrenz** — anderer Mechanismus (Zustandsraum statt Butterfly), aber gleiches Ziel (Attention ersetzen). Nicht mit QIRE-Bausteinen kombinierbar, ohne die Kernidee zu verlassen |
| **FNet** (Lee-Thorp u. a., Google, arXiv:2105.03824, 2021) | Ersetzt Self-Attention durch eine **feste, ungelernte** Fourier-Transformation über Sequenz- und Kanalachse; 92–97 % von BERT bei 70–80 % weniger Trainingszeit | O(L log L) | **Passt sehr direkt** — genau das Vorbild für einen „QIRE-Sequenzmischer": FNet zeigt, dass Attention durch eine feste Basistransformation ersetzbar ist. QIRE kann dieselbe Rolle mit der Butterfly-Struktur spielen — mit der in §4 genannten Einschränkung, dass geteilte Winkel die Walsh-Hadamard-, nicht die Fourier-Basis liefern |
| **MLP-Mixer** (Tolstikhin u. a., Google, arXiv:2105.01601, NeurIPS 2021) | Trennt Kanal- und Ortsmischung in zwei MLPs, ganz ohne Attention/Faltung | O(d²) je Mixer, aber einfach | **Grenzfall** — Vorbild für die *Aufteilung* in Kanal- und Sequenzmischung, nicht für die Kosten (MLP-Mixer ist dicht, QIRE ersetzt genau diese Dichte) |
| **Lineare Aufmerksamkeit** (Katharopoulos u. a., *Transformers are RNNs*, ICML 2020) | Attention als Kernel-Feature-Map, linear statt quadratisch in der Sequenzlänge | O(L) | **Konkurrenz** zur Sequenzmisch-Frage, orthogonal zu QIREs Kanalmisch-Beitrag — beide Ideen sind kombinierbar (lineare Attention + QIRE-FFN) |
| **KAN** (Liu u. a., arXiv:2404.19756, 2024) | Lernbare Aktivierungsfunktionen auf den Kanten statt feste Aktivierung an den Knoten (Kolmogorov-Arnold-Darstellungssatz) | je nach Basisgrad | **Konkurrenz**, andere Philosophie: KAN ersetzt die Nichtlinearität, QIRE ersetzt die *lineare* Abbildung und gewinnt die Nichtlinearität als Nebeneffekt der Born-Regel. Beide Ideen sind eher komplementär als konkurrierend — ein KAN-Kopf auf QIRE-Merkmalen wäre denkbar, aber ungeprüft |
| **BitNet b1.58** (Ma u. a., Microsoft, Februar 2024) | Gewichte auf {-1, 0, 1} quantisiert (1,58 Bit), erreicht bei gleicher Größe die Qualität von FP16-Modellen | 1,58 Bit/Gewicht | **Konkurrenz auf derselben Achse** (Speicher pro Parameter), aber orthogonal: BitNet spart *pro Parameter*, QIRE spart *an der Parameterzahl*. Kombinierbar (quantisierte Winkel), aber nicht Gegenstand dieses Projekts — als Vergleichspunkt für die Kosten-pro-Bit-Rechnung nennenswert |
| **Hypernetworks** (Ha, Dai, Le, ICLR 2017, arXiv:1609.09106) | Ein kleines Netz erzeugt die Gewichte eines großen Netzes | Kosten des kleinen Netzes | **Grenzfall** — verwandtes Prinzip (wenige Parameter erzeugen viele effektive Freiheitsgrade), anderer Mechanismus. QIREs geteilte Winkel sind eine *extreme*, festverdrahtete Form von Gewichtsteilung, kein gelerntes Hypernetz |
| **Mixture of Experts** (Shazeer u. a. 2017; V-MoE, Riquelme u. a., NeurIPS 2021) | Bedingte Berechnung: nur wenige von vielen „Experten" je Eingabe aktiv | sub-linear in Rechenzeit, linear in Speicher | **Konkurrenz auf einer anderen Achse** (Rechenzeit bei sehr großen Modellen). Für die Zielgröße dieses Projekts (Millionen, nicht Milliarden Parameter) nicht relevant, aber als Einordnung wichtig: MoE löst „viele Parameter, wenig Rechenzeit", QIRE löst „wenig Parameter, viel Rechenzeit" — fast das Gegenteil |
| **Kleine Sprachmodelle** (TinyStories, Eldan & Li 2023; SmolLM/SmolLM2, Hugging Face 2024/2025) | Kleine, aber vollständig dichte Modelle (<50 M bzw. 135 M–1,7 Mrd. Parameter), trainiert auf kuratierten/synthetischen Korpora | dicht, aber klein | **Vorbild für das Testfeld**, nicht für die Architektur — TinyStories liefert die Methodik „kleiner Korpus, einfache Sprache, prüfbare Kohärenz" für den eigenen Chatbot-Test (§5.3). SmolLM zeigt, wie weit dichte Kleinmodelle heute reichen — das ist die Latte, die QIRE nicht reißen muss, aber kennen sollte |
| **Tensornetz-Sprachmodelle** (MPS/Tensor-Train, z. B. Zhang u. a., *Tensor Network Language Model*, arXiv:1710.10248) | Gewichtsmatrizen als Matrixprodukt-Zustand faktorisiert, blieben laut Literaturübersicht meist theoretisch | O(d·r²) je nach Rang r | **Grenzfall/Konkurrenz** — eng verwandt mit Butterfly (beides Tensor-Faktorisierungen), aber MPS ist rang-basiert, nicht winkel-basiert. Q-014/Q-015 des Projekts (globale orthogonale Auslese, Gewichtsteilung) berühren dieselbe Frage aus einer anderen Richtung |
| **Quanten-NLP / DisCoCat / lambeq** (Coecke, Sadrzadeh, Clark 2010; Kartsaklis u. a., *lambeq*, arXiv:2110.04236) | Satzbedeutung aus Grammatikstruktur (Stringdiagramme) direkt als Quantenschaltung kodiert; **andere Kategorie von Quantenbezug** als QIRE | Schaltungstiefe wächst mit Satzstruktur | **Konkurrenz, andere Familie.** DisCoCat nutzt die Quantenschaltung für die *kompositionale Grammatik*, QIRE nutzt sie für die *numerische Schicht* eines gewöhnlichen neuronalen Netzes. Beide sind „quanteninspiriert", aber nicht kombinierbar ohne Neukonstruktion. Muss in der Arbeit klar abgegrenzt werden, damit die Jury QIRE nicht mit DisCoCat verwechselt |
| **Quanteninspirierte Wort-Embeddings** (Li, Uprety, Wang, Song, *Quantum-Inspired Complex Word Embedding*, ACL-Workshop 2018) | Wörter als Überlagerungszustände mit komplexer Phase in einem Hilbertraum | wie Standard-Embedding, nur komplexwertig | **Grenzfall** — verwandte Sprache („Superposition", „Hilbertraum"), aber eigenständige Idee (Wortbedeutung als Interferenz). Nicht Teil des QIRE-Bauplans, aber erwähnenswert als Nachbarfeld mit ähnlicher Terminologie — Verwechslungsgefahr in der Arbeit vermeiden |
| **Quantum Random Access Codes (QRAC)** (Übersicht: Wikipedia „Quantum random access code"; z. B. Wiesner 1983-Vorläufer, moderne Form Ambainis u. a. 1999) | n klassische Bits verlustbehaftet in log(n) Qubits kodieren, mit begrenzter Abrufwahrscheinlichkeit | log(n) Qubits, p < 1 Abrufgüte | **Grenzfall, eher exotisch für dieses Projekt** — QRAC ist für *binäre* Daten mit *einzelnem* Bit-Abruf gedacht (Kommunikationskomplexität), nicht für kontinuierliche Merkmalsvektoren, die vollständig zurückgelesen werden müssen. Für Bild/Text/Zeitreihen kaum direkt nutzbar; höchstens für stark binäre Tabellenspalten (Entscheidungs-KI) eine Randidee |
| **Bereits im Projekt gefundene direkte Konkurrenz/Vorarbeit** (siehe `LITERATUR_UND_ALLEINSTELLUNG.md`) | Quantum-PEFT (Koike-Akino u. a., ICLR 2025): (2L+1)·log₂d − 2L Parameter, RY+CZ-Schaltung als linearer Adapter bis Mistral-7B. Kronecker Recurrent Units (Jose/Cissé/Fleuret, ICML 2018): O(log N) Kronecker-Faktoren. EUNN (Jing u. a., ICML 2017): unitäre FFT-artige Schaltung, aber *ungeteilte* Winkel. BOFT (Liu u. a., ICLR 2024): Butterfly-Adapter, ebenfalls ungeteilt | — | **Bereits identifiziert, hier nur eingeordnet.** Diese vier sind die engste Konkurrenz zur QIRE-Mittelschicht selbst — nicht zum hier neu entworfenen *Gesamtmodell* (Encoder + Sequenzmischer + Kopf), das in dieser Form nicht referenziert gefunden wurde |

**Kurzfassung für Malte:** Die QIRE-*Schicht* (geteilte Butterfly + Born-Auslese) hat mit
Quantum-PEFT/KRU/BOFT/EUNN direkte Vorarbeit — das ist bereits bekannt. Was in der Literatur
**nicht** gefunden wurde, ist ein **Gesamtmodell**, das dieselbe Struktur konsequent für Encoder,
Zwischenschicht *und* Sequenzmischung verwendet, um das ganze Modell klein zu halten (statt nur
eine Mittelschicht), und das an mehreren KI-Arten mit vorregistrierten Kontrollen geprüft wird.
Genau das ist die Lücke, die das folgende Konzept füllt.

---

## 2. Das QIRE-Netz — Bauplan

### 2.1 Die Grundzelle (unverändert aus dem bestehenden Code übernommen)

```
QIRE-Zelle(x, Θ):
  1. Normieren        x' = x / ||x||₂                         (0 Parameter)
  2. Butterfly-Drehung  n = log2(d) Ebenen, je Ebene ein geteiltes
                        Winkelpaar (Ry, Rz) auf alle d/2 Paare,
                        danach CNOT-Ring (feste, kostenlose Permutation)
                        -- L Lagen wiederholt                  (2·n·L Parameter)
  3. Born-Auslese       entweder |Amplitude|² (volle Breite, Wahrscheinlichkeits-
                        verteilung, Summe 1) oder ⟨Z_q⟩ je Qubit (n Zahlen,
                        starke Kompression)                    (0 Parameter)
```

Das ist **keine neue Erfindung** — es ist exakt `ButterflyQuantumLayer` /
`ButterflyFFN` aus dem bestehenden Code, hier nur als eigenständiger, wiederverwendbarer
Baustein benannt. Die Ehrlichkeit an dieser Stelle ist wichtig: Der Beitrag dieses Konzepts
liegt nicht in der Zelle selbst, sondern darin, sie **konsequent an jeder Stelle** eines Netzes
einzusetzen, wo bisher eine dichte Schicht steht — inklusive Encoder und Sequenzmischung, wo sie
im Projekt bisher nicht oder nur versuchsweise (Q-012, aufgegeben) stand.

**Warum Tiefe innerhalb einer Zelle wenig bringt:** `BEFUNDE_UND_RICHTUNGEN.md` §2 zeigt: Ohne
CNOT ist jede Tiefe wieder dieselbe Gruppe (Kronecker-Produkt), Verkettung bringt nichts Neues.
Mit CNOT ist die Familie universell, aber erst ab vielen Lagen (128 für das, wofür ungeteilt 4
reichen). Deshalb ist **eine tiefe QIRE-Zelle keine gute Idee** — besser mehrere **flache**
QIRE-Zellen mit einer Born-Auslese dazwischen, weil erst die Auslese eine echte Nichtlinearität
und Dimensionsänderung einführt (§2.3, „Warum der Encoder ein eigenes Problem ist").

### 2.2 Warum reine Verkettung von Zellen nicht reicht — der Encoder als eigenes Problem

`BEFUNDE_UND_RICHTUNGEN.md` (§2, „Butterfly-Encoder wäre redundant", [THEORIE, bewiesen]):
Zwei Butterfly-Schaltungen hintereinander sind wieder eine Butterfly-Schaltung mit mehr Lagen —
die Familie ist unter Verkettung abgeschlossen. Ein Encoder, der nur aus QIRE-Zellen besteht,
bringt **nichts**, was eine einzelne größere Zelle nicht auch könnte. Ein Encoder braucht also
einen **nicht-unitären Schritt** — etwas, das entweder die Dimension ändert oder Information
verwirft.

Genau das leistet die **Born-Auslese selbst**: `|Amplitude|²` oder `⟨Z_q⟩` ist nicht umkehrbar
(Phaseninformation geht verloren, bei `⟨Z_q⟩` zusätzlich Dimension von 2ⁿ auf n). Eine Folge von
`QIRE-Zelle → Born-Auslese → erneut normieren → nächste QIRE-Zelle` ist deshalb **keine**
redundante Verkettung, sondern ein echtes mehrschichtiges Netz mit echter Nichtlinearität
zwischen den Schichten — dieselbe Rolle, die eine ReLU zwischen zwei dichten Schichten spielt.

**Das ist der zentrale Konstruktionsvorschlag dieses Konzepts:** Statt eines großen dichten
Encoders (784→256, der laut Review die ganze Parameterersparnis auffrisst) eine **Kaskade
kleiner QIRE-Zellen mit Born-Auslesen dazwischen**, jede auf einem kleinen, handhabbaren
n (z. B. n=6…8), verbunden über ein **festes, parameterfreies Umsortieren** (Patch-Aufteilung,
Pooling) statt einer großen trainierbaren Projektion.

### 2.3 Baustein „Patch-Encoder" — konkreter Vorschlag gegen das Faktor-64-Problem

Für Eingaben, die größer als 2ⁿ sind (Bild 784, Zeitreihe 256+, Tabellenzeile mit vielen
Spalten):

```
Patch-Encoder(x):
  1. x in P Patches gleicher Größe d_patch = 2^k zerlegen (Padding falls nötig)
     -- bei Bildern: räumliche Kacheln (wie MLP-Mixer / Vision Transformer);
     -- bei Zeitreihen: Fenster;
     -- bei Tabellen: feste Spaltengruppen.
  2. JEDES Patch durch DIESELBE QIRE-Zelle schicken (Gewichtsteilung über Patches --
     wie ein 1x1-Faltungskern mit geteilten Gewichten, nur mit der Butterfly-Zelle
     statt einer dichten Matrix)                                (2·k·L Parameter, EINMAL)
  3. Die P Ausleseergebnisse (je k Zahlen) aneinanderhängen -> Vektor der Länge P*k
  4. Falls P*k selbst keine Zweierpotenz ist oder noch zu groß: eine zweite,
     kleinere QIRE-Zelle oder ein Pooling-Schritt darüber (rekursiv, "hierarchisches
     Register" -- Idee aus BEFUNDE_UND_RICHTUNGEN.md §5, "Hierarchische Qubit-Register")
```

**Parameterzahl unabhängig von der Eingabegröße** (nur von der Patchgröße und der Zahl der
Ebenen) — das ist der entscheidende Unterschied zum bisherigen Encoder 784→256, dessen 200.960
Parameter direkt mit der Eingabegröße wachsen. Bei MNIST (784, Patch 16×1=16=2⁴, 49 Patches):
eine geteilte QIRE-Zelle mit k=4, L=3 hat 2·4·3 = 24 Parameter — **für alle 49 Patches
zusammen**, nicht je Patch.

**Das ist ausdrücklich ungeprüft** und der wichtigste offene Punkt dieses Konzepts. Q-012
("Encoder als Butterfly", 270 statt 201.130 Parameter, Faktor 745) hat eine verwandte Idee schon
einmal verfolgt und wurde ohne Messung aufgegeben (Zeitmangel, nicht Ablehnung). Der
Patch-Ansatz hier unterscheidet sich von Q-012 dadurch, dass die Gewichtsteilung **über Patches**
läuft (wie eine Faltung), nicht nur innerhalb einer Schaltung — das ist näher an MLP-Mixer/Vision
Transformer als an Q-012 und sollte als eigener, neu vorregistrierter Lauf behandelt werden
(Vorschlag: **Q-021**, siehe Roadmap §6).

### 2.4 Sequenzmischung — QIRE-Sequenzmischer statt Attention

Für Text (Chatbot) und Zeitreihen: FNet (Lee-Thorp u. a. 2021) zeigt, dass eine **feste**
Transformation über die Sequenzachse Attention ersetzen kann. Vorschlag:

```
QIRE-Sequenzmischer(X):   X hat Form (Sequenzlänge L, Kanäle d)
  1. entlang der Sequenzachse: L muss (nach Padding) eine Zweierpotenz sein;
     eine QIRE-Zelle mischt die L Positionen je Kanal (wie FNet entlang der
     Sequenzachse, aber mit geteilter Butterfly-Drehung statt FFT)
  2. entlang der Kanalachse: eine zweite QIRE-Zelle mischt die d Kanäle je Position
     (das ist bereits `ButterflyFFN` aus `sequence_platform.py` / T-002)
```

**Wichtige Einschränkung, aus dem eigenen Q-019-Befund übernommen** (`ENTWURF_ZEITREIHEN.md`,
Nachtrag 27. August): Die geteilte Butterfly-Struktur hat **geteilte** Winkel (2 je Ebene), die
echte schnelle Fouriertransformation braucht **eigene** Drehfaktoren je Paar. Gemeinsam ist
beiden nur das **Verbindungsmuster** (O(d log d)), nicht die Transformation selbst. Was die
geteilte Struktur *exakt* enthält, ist die **Walsh-Hadamard-Transformation** (ein
Kronecker-Produkt von 2×2-Hadamard-artigen Drehungen ohne CNOT). Ein QIRE-Sequenzmischer ist
deshalb kein „FNet mit Quantenschaltung statt FFT", sondern eher ein „FNet mit
Walsh-Hadamard statt Fourier" — geeignet für Signale mit Sequenz-/Schaltstruktur
(Rechtecksignale), nicht zwingend für glatte periodische Signale. Diese Einschränkung gehört
ausdrücklich in jede Beschreibung des Sequenzmischers, sonst wiederholt sich der bereits einmal
korrigierte Fehler aus Q-019.

Für die Aufmerksamkeitsprojektionen (Q, K, V) gilt die bereits im Projekt getroffene
Entscheidung (T-002, `butterfly_qkv`): **ohne** Born-Regel, als reelle orthogonale
Givens-Butterfly-Drehung, weil Aufmerksamkeit vorzeichenbehaftete Richtungen braucht, keine
Wahrscheinlichkeitsverteilung. Diese Entscheidung wird für das QIRE-Netz übernommen, nicht neu
erfunden.

### 2.5 Der Kopf

Unverändert klassisch: kleine dichte Schicht `n oder d → Klassen/Vokabular`. Bei kleinem
Vokabular (Zeichenebene, ALPHABET=32 wie in T-002) ist das vernachlässigbar; bei einem
wortbasierten Vokabular (Zehntausende Einträge) **dominiert der Kopf die Parameterzahl** —
dieselbe Einschränkung wie beim Encoder, nur am anderen Ende des Netzes. Ehrlich zu berichten:
Für ein QIRE-Netz mit großem Vokabular bräuchte auch der Kopf eine strukturierte Lösung (z. B.
adaptive Softmax oder ein QIRE-Auslese-Baum) — das ist **nicht** Teil dieses Konzepts, weil das
Zeichenmodell aus T-002 dieses Problem durch die Wahl eines kleinen Alphabets bereits umgeht.

### 2.6 Gesamtbild

```
Rohdaten -> [Patch-Encoder: geteilte QIRE-Zellen + Pooling]  -> (2.3)
         -> kleiner "Zustandsvektor" der Länge 2^n
         -> [N x (QIRE-Sequenzmischer + QIRE-Zelle)]         -> (2.4, nur bei Sequenzen)
         ODER [N x QIRE-Zelle mit Born-Auslese dazwischen]   -> (2.2, bei Einzelbeispielen)
         -> kleiner klassischer Kopf                          -> (2.5)
```

### 2.7 Was daran neu ist und was übernommen

| Baustein | Neu oder übernommen |
|---|---|
| QIRE-Zelle (Normieren, geteilte Butterfly, Born-Auslese) | **übernommen**, bereits im Code (`quantum_layer.py`, `butterfly_layer.py`, `butterfly_fast.py`) |
| Born-Auslese als nicht-unitärer Schritt zwischen Zellen (Begründung für mehrschichtige Encoder) | **übernommen als Theorie** (BEFUNDE §2), hier erstmals als **Konstruktionsprinzip** für einen ganzen Encoder ausformuliert |
| Patch-Encoder mit über Patches geteilten QIRE-Zellen | **neu für dieses Projekt** (verwandt mit Q-012, aber mit Gewichtsteilung über Patches statt nur innerhalb der Schaltung; verwandt mit MLP-Mixer/ViT-Patchify, dort aber mit dichten statt QIRE-Projektionen) |
| QIRE-Sequenzmischer (FNet-artig) | **neu für dieses Projekt** als expliziter Baustein; die Grundidee „Butterfly statt Attention" steht bereits in `BEFUNDE_UND_RICHTUNGEN.md` §5, aber nicht als FNet-artiger Sequenzmischer ausgearbeitet |
| butterfly_qkv, butterfly_ffn | **übernommen unverändert** aus T-002 |
| Walsh-Hadamard- statt Fourier-Einordnung | **übernommen** aus dem eigenen Q-019-Korrekturnachtrag |

---

## 3. Datenübersetzer — „Quantensprache" ehrlich eingeordnet

| Kodierung | Funktionsweise | Parameter/Genauigkeit-Effekt | Einordnung |
|---|---|---|---|
| **Amplitudenkodierung** (bereits verwendet) | Merkmalsvektor normieren, direkt als Amplitudenvektor eines n-Qubit-Zustands auffassen | 0 zusätzliche Parameter; **keine** Speicherersparnis in der klassischen Simulation — n Qubits tragen 2ⁿ Amplituden nur auf echter Quantenhardware weniger *physische* Register, klassisch simuliert bleibt es ein Vektor der Länge 2ⁿ | Das ist im Kern **eine Normierung, umbenannt**. Der einzige echte Effekt ist die anschließende Struktur (Butterfly + Born-Regel), nicht die Kodierung selbst. Muss so in der Arbeit stehen |
| **Winkelkodierung** | Jedes Merkmal wird der Rotationswinkel eines Qubits; n Merkmale → n Qubits, linear viele Parameter/Gatter | Kein Kompressionsgewinn, dafür laut Literatur **hardwarerobuster**: Umfrageergebnis (`Feature Encoding in Quantum Machine Learning`, arXiv:2606.05387, 2026) — 19 von 23 Hardware-Experimenten nutzen Winkel- oder IQP-Kodierung statt Amplitudenkodierung, weil Amplitudenkodierung auf echter Hardware ab ~8 Merkmalen keine korrekten Klassifikationen mehr liefert | Für dieses Projekt (klassische Simulation, kein Hardwarezugriff im Alltag) zweitrangig — relevant nur für den Hardware-Demo-Teil (IBM), nicht für die CPU-Läufe |
| **Quanten-Random-Access-Codes (QRAC)** | n klassische Bits verlustbehaftet in log(n) Qubits, mit Abrufwahrscheinlichkeit p < 1 je Bit | Kompression nur für **binäre** Daten mit *partiellem* Abruf; für kontinuierliche Merkmale, die vollständig gebraucht werden, nicht passend | **Nur Umbenennung, wenn falsch angewendet** — für Bild/Text/Zeitreihen ungeeignet. Einzige denkbare Nische: stark binäre Tabellenspalten in der Entscheidungs-KI (§5.4), aber auch dort ungeprüft und mit Wahrscheinlichkeit < 1 pro Abfrage ein Nachteil gegenüber direkter Kodierung |
| **Hierarchische Kodierung** (mehrere kleine Register statt eines großen) | Entspricht dem Patch-Encoder aus §2.3 — mehrere kleine QIRE-Zellen mit lokalem Bezug statt einer globalen | Macht die Parameterzahl unabhängig von der Eingabegröße (§2.3) | **Der wichtigste neue Baustein dieses Konzepts** — siehe §2.3. Kein reiner Kodierungstrick, sondern ein Architekturprinzip |
| **Token → Zustand** (für Text) | Trainierbares Embedding (Standard, dicht) auf Länge d=2ⁿ, danach wie jede andere Amplitudenkodierung behandelt | Kein Unterschied zum bestehenden Vorgehen in T-002 | **Übernommen unverändert** — das Embedding selbst bleibt dicht (Standardpraxis, auch in kleinen Sprachmodellen wie TinyStories/SmolLM), nur die Verarbeitung danach ist QIRE-strukturiert |

**Fazit Datenübersetzer:** Der einzige Teil, der über eine Umbenennung hinausgeht, ist die
**hierarchische/patch-basierte** Kodierung — weil sie eine strukturelle Antwort auf das
Faktor-64-Problem ist, nicht nur eine andere Art, Zahlen in einen Vektor zu packen.
Amplituden- und Winkelkodierung bleiben das, was sie im Projekt schon sind: eine Normierung
bzw. eine direkte Parametrisierung, ohne eigenen Beitrag zur Effizienzaussage.

---

## 4. Varianten klein/mittel — Parameter- und Rechenbudget

Alle Zahlen sind **Entwurfsgrößen**, keine Messungen. „CPU-Zeit" ist aus bestehenden
Kalibrierungen des Projekts hochgerechnet (T-000-Durchsatz, Q-003/Q-020-Zellzeiten), nicht neu
gemessen.

### 4.1 QIRE-Netz-Klein (Bilder klein, Tabellen, Zeitreihen)

| Baustein | Aufbau | Parameter |
|---|---|---|
| Patch-Encoder | 8–64 Patches à 2⁴–2⁶, eine geteilte QIRE-Zelle, L=3 | 24–36 |
| Mittelschicht | 1–2 QIRE-Zellen, n=6–8, L=3–5 | 36–80 |
| Kopf | n oder Patch-Ausgabe → Klassen (≤20) | 100–2.000 |
| **Gesamt** | | **~150–2.100** |

Zum Vergleich: ein ReLU-MLP gleicher Engstelle liegt laut Review bei 67.848 (Mittelschicht
allein) — das Ziel ist, das Gesamtmodell, nicht nur die Mittelschicht, in dieser Größenordnung
zu halten.

**CPU-Zeit:** Auf Basis der Q-003/Q-020-Zellzeiten (Sekunden bis niedrige zweistellige
Sekundenzahl je Seed×Arm bei MNIST-Größe) ist ein voller Lauf (15 Seeds × 5–8 Arme × 1–2
Datenbedingungen) auf dem 6-Kern-Ryzen im Bereich **Minuten bis wenige Stunden**, klar
CPU-tauglich. Der Patch-Encoder fügt einen zusätzlichen Faktor P (Zahl der Patches) an
Vorwärtsdurchläufen hinzu — bei 49 Patches (MNIST, Kachelgröße 4×4) etwa 49-mal so viele
Zellaufrufe wie eine einzelne QIRE-Zelle, aber jeder Aufruf ist klein (n=4). Erwartungswert:
gleiche Größenordnung wie die bereits gemessenen Q-003/Q-020-Läufe, keine neue Hardware nötig.

### 4.2 QIRE-Netz-Mittel (Text/Chatbot)

Identisch mit der bestehenden T-002-Architektur (d_model=256, 4 Lagen/Blöcke, 4 Köpfe,
butterfly_ffn + butterfly_qkv), **hier nicht neu entworfen**, nur als Instanz des
QIRE-Netz-Bauplans eingeordnet: Die Blöcke aus §2.4 entsprechen exakt `Block` in
`sprachmodell.py`. Neu wäre nur, den `QIRE-Sequenzmischer` (§2.4, FNet-artig entlang der
Sequenzachse) als **fünften** Arm neben `dense`, `butterfly_ffn`, `butterfly_qkv`, `narrow`
zu ergänzen — Vorschlag **T-003** (siehe Roadmap).

**Rechenbedarf:** unverändert aus `00_REVIEW_BEFUNDE.md` §3 übernommen — rund 30–35 GPU-Stunden
auf einer Karte, ein Nachmittag auf acht parallelen GPUs (MPI). Der zusätzliche
`QIRE-Sequenzmischer`-Arm braucht dieselbe Größenordnung wie `butterfly_qkv` und **keine**
zusätzliche Vorregistrierungsfamilie über Holm hinaus, wenn er wie die anderen Arme nur gegen
`dense` getestet wird (dann Holm über drei statt zwei Vergleiche).

### 4.3 Trainingsverfahren

Unverändert Adam auf allen Parametern (Winkel wie klassische Gewichte behandelt) — das ist der
bestehende Ansatz und funktioniert nachweislich (T-000 bis T-002, Q-003 bis Q-020).
**Quantum Natural Gradient** (BEFUNDE §4, „Winkel leben auf gekrümmter Mannigfaltigkeit") ist
laut `LITERATUR_UND_ALLEINSTELLUNG.md` bereits geprüft (Aghaei Saem u. a. 2026: hilft nicht
gegen Konzentration) und wird hier **nicht** empfohlen — Adam bleibt Standard.

---

## 5. Testfeld — mehrere KI-Arten für eine mehrschichtige Aussage

Prinzip in jeder Zeile: **Datensatz → faire Kontrollen (gleiche Bauform dicht, linear, kleines
Standardmodell gleicher Gesamtparameterzahl) → Messgrößen (Genauigkeit, Parameter gesamt und je
Baustein, Zeit CPU/GPU, Datenbedarf) → CPU-Schätzung → GPU-Bedarf**.

### 5.1 Bilder — MNIST, Fashion-MNIST, CIFAR-10 (klein)

| | |
|---|---|
| Datensatz | MNIST/Fashion-MNIST (bereits im Projekt, jetzt mit **offiziellem Testsatz**, Lehre aus Q-020); CIFAR-10 mit reduzierter Klassenzahl/Beispielzahl (z. B. 4 Klassen à 2.000 Bilder), damit CPU-tauglich |
| Kontrollen | ReLU-MLP gleicher Engstelle (nicht `classical_norm_deep` — Lehre aus dem Review), lineares Modell, dichter Patch-Mixer (MLP-Mixer-artig, gleiche Patchgröße, dichte statt QIRE-Projektion) |
| Messgrößen | balancierte Genauigkeit, Parameter gesamt/Encoder/Mittelschicht/Kopf, Trainingszeit CPU, Datenbedarf (Lernkurve N=200…6.000, Bezug zu C4 aus `01_VERBESSERUNGEN_UND_ZUKUNFT.md`) |
| CPU-Schätzung | MNIST/Fashion: Minuten (bestehende Kalibrierung). CIFAR-10-klein: grob 5–10× MNIST wegen 3 Kanälen und größerer Bilder, weiterhin CPU-tauglich (kein GPU nötig) |
| GPU-Bedarf | keiner nötig; nur falls die volle CIFAR-10 (10 Klassen, 50.000 Bilder) später gewünscht wird |
| Zweck im Testfeld | Validiert §2.3 (Patch-Encoder) direkt gegen das Faktor-64-Problem — **der wichtigste Einzeltest**, weil er die zentrale neue Idee prüft |

### 5.2 Zeitreihen/Signale

| | |
|---|---|
| Datensatz | (a) das bereits gebaute Q-019-Modul (`zeitreihen.py`, Walsh-Hadamard-Aufgabe statt der widerlegten Fourier-Erzählung, siehe C3 in `01_VERBESSERUNGEN_UND_ZUKUNFT.md`); (b) 2–3 kleine UCR-Archiv-Datensätze mit bekannter periodischer/Schaltstruktur (z. B. `TwoPatterns`, `Wafer` — beide klein genug für CPU, öffentlich über die UCR/UEA-Sammlung, Dau u. a. 2019) |
| Kontrollen | dicht (256×256), `narrow` (Auflösungsprüfung, bereits im Modul), echte FFT fest (`fft_fest`, bereits im Modul) |
| Messgrößen | wie Q-019-Vorregistrierung: balancierte Genauigkeit, Lernkurve über N |
| CPU-Schätzung | Q-019 ist laut Stand bereits fertig gebaut, geprüft, **nur noch die Machtanalyse und der Lauf selbst offen** (~2,5–3 h laut `ENTWURF_ZEITREIHEN.md`). UCR-Zusatzdatensätze: ähnliche Größenordnung, da Reihenlängen vergleichbar (128–512) |
| GPU-Bedarf | keiner |
| Zweck im Testfeld | Einzige Anwendung mit einem *inhaltlichen* (nicht nur Parameter-) Vorteilsanspruch — muss aber die Walsh- statt Fourier-Korrektur konsequent tragen, sonst wiederholt sich der bereits einmal falsifizierte Befund |

### 5.3 Text — zeichenbasiertes Sprachmodell

| | |
|---|---|
| Datensatz | Markow-Text (bereits vorhanden, `sequence_platform.py`) als Vortest; danach echter deutscher Kleinkorpus im TinyStories-Stil (einfache, kurze Geschichten) — der Projektordner „T-002" ist dafür bereits vorbereitet (`DATEN_EXTERN.md`) |
| Kontrollen | `dense`, `narrow` (bereits vorregistriert in T-002) |
| Messgrößen | Kreuzentropie (nats/Zeichen), FFN-Parameter, Gesamtparameter, Trainingszeit bei gleicher Schrittzahl (nicht gleicher Zeit — Lehre aus T-002-Vorregistrierung) |
| CPU-Schätzung | laut T-000: 282 Mio. Token in 24 h bei d_model=256 auf diesem Rechner — für den Vortest (Markow-Text, kleines Korpus) reichen Stunden, nicht Tage |
| GPU-Bedarf | für den vollen, vorregistrierten T-002-Lauf: ~30–35 GPU-Stunden (bereits im Review beziffert), am besten über MPI-Kontakt |
| Zweck im Testfeld | Der „Chatbot" der Aufgabenstellung — bereits gebaut, hier nur als ein Zweig des universellen QIRE-Netzes eingeordnet statt als isoliertes Sondermodell |

### 5.4 Entscheidungs-KI (tabellarische Daten) — **neu, noch nicht im Projekt**

| | |
|---|---|
| Datensatz | kleine Auswahl aus **OpenML-CC18** (Bischl u. a., *OpenML Benchmarking Suites*, NeurIPS Datasets & Benchmarks 2021) — Datensätze mit einigen hundert bis wenigen tausend Zeilen, damit CPU-tauglich (z. B. „credit-g", „diabetes", „blood-transfusion") |
| Kontrollen | Gradient Boosting (z. B. scikit-learn `HistGradientBoostingClassifier`, kein neuer Download nötig), kleines dichtes MLP gleicher Parameterzahl, logistische Regression |
| Messgrößen | balancierte Genauigkeit/AUC, Parameter gesamt, Trainingszeit, Datenbedarf (diese Datensätze sind ohnehin klein — die Frage ist eher Generalisierung als Rechenzeit) |
| CPU-Schätzung | Sekunden bis niedrige Minuten je Modell — die mit Abstand billigste Zelle im ganzen Testfeld |
| GPU-Bedarf | keiner |
| Zweck im Testfeld | Deckt die vom Auftrag ausdrücklich genannte „Entscheidungs-KI" (Klassifikation) ab, die bisher **nicht** Teil des Projekts war; billig genug, um sie sicher bis 30.11. zu zeigen |
| Realistische Erwartung | Gradient Boosting ist auf kleinen Tabellendaten sehr stark; ein realistisches Ergebnis ist „gleichwertig bei viel weniger Parametern", nicht „besser" — das muss vor dem Lauf so vorregistriert werden (Lehre aus der Kernaussagen-Überkorrektur, `00_REVIEW_BEFUNDE.md` §1) |

### 5.5 Stretch Goals — nur wenn Zeit bleibt, nicht für 30.11. eingeplant

| | |
|---|---|
| Audio-Schlüsselwörter | Google Speech Commands (Warden 2018), auf wenige Schlüsselwörter reduziert. Aufwand: mittel (Audio-Vorverarbeitung, MFCC/Spektrogramm als Eingabe für den Patch-Encoder). CPU-tauglich in reduzierter Form, aber zusätzlicher Vorverarbeitungscode nötig |
| Einfache Steuerung/RL | CartPole (Gymnasium/Farama, Nachfolger von OpenAI Gym). Aufwand: gering an Rechenzeit, aber eigene Trainingsschleife (Policy-Gradient) nötig, die im Projekt noch nicht existiert — aus BEFUNDE §9 („Die Politik ist eine Wahrscheinlichkeitsverteilung — genau das, was eine Messung liefert") folgt eine naheliegende QIRE-Politiknetz-Idee, aber komplett ungebaut |

**Ehrliche Einordnung des Testfelds insgesamt:** Bei **jeder** dieser Anwendungen gilt weiterhin
die Kernbeschränkung aus dem Projekt: Die Struktur gewinnt an **Rechenzeit** erst ab 16.384
Dimensionen; alle Testfeld-Breiten liegen weit darunter. Jede Anwendung zeigt deshalb bestenfalls
**Qualität bei drastisch weniger Parametern**, nicht Geschwindigkeit — das muss in jedem
Abschnitt der Arbeit genauso konsequent getrennt werden, wie es die Roadmap für den Chatbot
bereits vorschreibt.

---

## 6. Roadmap bis Januar 2027

### Bis zur Anmeldung, 30. November 2026 — was ohne GPU und ohne große neue Bauarbeit zeigbar ist

| Baustein | Aufwand | Abhängigkeit |
|---|---|---|
| Q-020 (Kernaussage fair, bereits vorregistriert) | läuft/steht kurz bevor | keine |
| Entscheidungs-KI-Testbett (§5.4, OpenML-CC18) | gering — neues, aber kleines Modul, nur scikit-learn + bestehender QIRE-Code | keine |
| Q-019-Lauf (Zeitreihen, bereits gebaut, nur Machtanalyse + Lauf offen) | gering | keine |
| Patch-Encoder-Prototyp + Q-021-Vorregistrierung (§2.3, §7) | mittel — neuer Code, aber auf bestehenden Bausteinen | keine |
| Bild-Testfeld mit korrigierten Kontrollen (5.1, ohne CIFAR) | gering, größtenteils Wiederverwendung von Q-020-Code | Q-020 |
| **Ausstiegskriterium:** Wenn der Patch-Encoder bis Mitte November nicht mindestens den Vortest (Regel aus `CLAUDE.md` §5) besteht, wird er als „Konzept, nicht getestet" in die Anmeldung aufgenommen, statt eine unfertige Messung zu zeigen | — | — |

### Bis zur schriftlichen Arbeit, Mitte Januar 2027

| Baustein | Aufwand | Abhängigkeit |
|---|---|---|
| T-002 vollständiger Lauf (Chatbot, GPU) | hoch (GPU-Zeit), aber bereits geplant | MPI-Rechenzeit oder RunPod-Guthaben |
| T-003: QIRE-Sequenzmischer als fünfter Arm in T-002 (§2.4, §4.2) | mittel — neuer Code, gleiche Infrastruktur | T-002-Infrastruktur |
| CIFAR-10-klein (§5.1) | gering, falls Patch-Encoder bis dahin steht | Patch-Encoder |
| UCR-Zusatzdatensätze (§5.2) | gering | Q-019-Modul |
| Stretch: Audio/RL (§5.5) | hoch, nur wenn alles andere steht | alles Obige |

### Priorisierung in einem Satz

**Bis 30.11.:** zeigen, dass das Bauprinzip (QIRE-Zelle + Patch-Encoder + faire Kontrollen)
an **mindestens zwei billigen, CPU-tauglichen Anwendungen** (Entscheidungs-KI, Zeitreihen)
funktioniert oder ehrlich nicht funktioniert — das trägt die „universell einsetzbar"-Aussage
bereits mehrschichtig, ganz ohne GPU. **Bis Mitte Januar:** der Chatbot (GPU-abhängig) und,
wenn der Patch-Encoder sich bewährt hat, die aufwendigeren Bild-/Sequenzmischer-Erweiterungen.

---

## 7. Neue Läufe, die aus diesem Konzept folgen (zur Übernahme durch `experiment`/`werkzeuge`)

- **Q-021 — Patch-Encoder gegen dichten Encoder.** Primär: QIRE-Patch-Encoder + QIRE-Mittelschicht
  gegen dichten Encoder + QIRE-Mittelschicht (isoliert die Encoder-Frage von der bereits
  beantworteten Mittelschicht-Frage), Auflösungsprüfung: beide müssen `linear` schlagen.
  Vorregistrierung nach demselben Schema wie Q-020 (TOST, Holm, Positivkontrolle: geteilte
  Patch-Zelle mit *ungeteilten* Patch-Gewichten als Äquivalenz-Kontrolle).
- **T-003 — QIRE-Sequenzmischer.** Fünfter Arm neben den vier bestehenden in T-002, gegen
  `dense`, gleiche Schrittzahl, gleiche Vorregistrierungslogik.
- **D-001 — Entscheidungs-KI auf OpenML-CC18.** Neue Kennung (D für „Decision"), weil weder Q
  (Bild) noch T (Sequenz) passt. QIRE-Netz-Klein gegen Gradient Boosting, MLP, logistische
  Regression, auf 3–5 kleinen OpenML-CC18-Datensätzen, Holm über die Datensätze.

---

## 8. Zusammenfassung für die Anmeldung (Kernsatz-Entwurf)

„Wir bauen aus den bereits geprüften QIRE-Bausteinen (Normierung, geteilte Butterfly-Drehung,
Born-Regel-Auslese) ein durchgängiges Netzprinzip, das nicht nur eine Mittelschicht, sondern
auch den Encoder und die Sequenzmischung klein hält, und prüfen es an vier KI-Arten (Bilder,
Zeitreihen, Text, Entscheidungs-Klassifikation) mit denselben strengen Kontrollen und
Abgrenzungstests, die das Projekt bereits für die Mittelschicht entwickelt hat." — Dieser Satz
gilt nur, wenn Q-021 (Patch-Encoder) mindestens die Auflösungsprüfung besteht; sonst lautet er:
„… und zeigen ehrlich, wo dieses Prinzip an Grenzen stößt."
