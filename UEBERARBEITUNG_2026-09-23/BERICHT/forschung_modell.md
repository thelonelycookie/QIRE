# Bericht Agent `forschung_modell` — Kurzfassung

Stand 24.09.2026. Ausführlich: `../FORSCHUNG/EIGENES_MODELL_KONZEPT.md`.

## Auftrag

Kein umgerüstetes LLM, sondern ein **eigenes, von Grund auf gebautes, universelles Modell**
auf Basis der QIRE-Bausteine — spezialisierbar auf mehrere KI-Anwendungen (Chatbot, Entscheidungs-KI,
weitere), damit die Aussage mehrschichtig ist.

## Architekturvorschlag „QIRE-Netz"

Die **Grundzelle ist bereits im Code vorhanden** (Normieren → geteilte Butterfly-Drehung
[Kronecker-Produkt, CNOT-Ring als kostenlose Permutation] → Born-Regel-Auslese) — kein neuer
Baustein. Neu ist, sie **konsequent an jeder Stelle** einzusetzen, wo heute eine dichte Schicht
steht, statt nur in der Mittelschicht:

1. **Patch-Encoder** (neuer Vorschlag, Kern des Konzepts): Statt eines großen dichten Encoders
   (784→256, der laut Review die ganze Parameterersparnis auffrisst) eine über Kacheln/Fenster
   **geteilte** QIRE-Zelle — wie eine Faltung mit geteilten Gewichten, nur mit der
   Butterfly-Zelle statt einer dichten Matrix. Macht die Parameterzahl des Encoders unabhängig
   von der Eingabegröße. **Ungeprüft, wichtigster offener Punkt** — Vorschlag für einen neuen,
   vorregistrierten Lauf **Q-021**.
2. **QIRE-Sequenzmischer** für Text/Zeitreihen: FNet-artig (Lee-Thorp u. a. 2021, Google) —
   feste Transformation über die Sequenzachse statt Attention. Wichtige Einschränkung aus dem
   eigenen Q-019-Befund: geteilte Winkel liefern die **Walsh-Hadamard-**, nicht die
   Fourier-Basis — muss so berichtet werden, sonst wiederholt sich ein bereits korrigierter
   Fehler.
3. **butterfly_ffn / butterfly_qkv** aus T-002 werden unverändert übernommen.

Theoretische Begründung, warum reine Verkettung von QIRE-Zellen nicht reicht: Die Familie ist
unter Verkettung abgeschlossen ([THEORIE, bewiesen], `BEFUNDE_UND_RICHTUNGEN.md`) — erst die
Born-Auslese zwischen zwei Zellen ist der nicht-unitäre, dimensionsändernde Schritt, der ein
echtes mehrschichtiges Netz ergibt.

**Datenübersetzer:** Amplituden- und Winkelkodierung sind im Kern eine Normierung bzw. eine
direkte Parametrisierung — **keine** eigene Effizienzquelle, das muss ehrlich so stehen.
QRAC passt nicht zu kontinuierlichen Merkmalen. Der einzige Baustein mit echtem Effekt ist die
**hierarchische/patch-basierte** Kodierung (= der Patch-Encoder).

## Stand der Technik — Einordnung

Direkte Vorarbeit zur QIRE-*Schicht* bereits bekannt (Agent `literatur`): Quantum-PEFT
(Koike-Akino u. a., ICLR 2025), Kronecker Recurrent Units (Jose/Cissé/Fleuret, ICML 2018), BOFT,
EUNN. **Nicht gefunden:** ein Gesamtmodell, das dieselbe Struktur für Encoder *und*
Sequenzmischung *und* Mittelschicht nutzt. Passende neue Bausteine: FNet, Monarch Mixer,
Butterfly-Faktorisierung (Dao 2019/2020), MLP-Mixer. Konkurrenz auf anderen Achsen: Mamba/S4,
Hyena, KAN, BitNet, MoE. Eigene Familie, nicht kombinierbar: DisCoCat/lambeq (Quanten-NLP über
Grammatikstruktur, nicht über numerische Schichten).

## Testfeld (mehrschichtige Aussage)

| Anwendung | Datensatz | Stand | GPU nötig? |
|---|---|---|---|
| Bilder | MNIST/Fashion (offizieller Testsatz), CIFAR-10 klein | Q-020 läuft; CIFAR neu | nein |
| Zeitreihen | Q-019 (Walsh-Hadamard-Aufgabe), 2–3 UCR-Datensätze | Q-019 gebaut, Lauf offen | nein |
| Text (Chatbot) | Markow-Text (Vortest), deutscher Kleinkorpus | T-002 gebaut, GPU-Lauf offen | ja (~30–35 h) |
| Entscheidungs-KI | OpenML-CC18-Auswahl vs. Gradient Boosting/MLP | **neu, ungebaut** | nein |
| Stretch | Speech Commands, CartPole | Idee, ungebaut | nein, aber Zeitaufwand |

Alle CPU-Zellen sind auf dem 6-Kern-Rechner realistisch in Minuten bis Stunden machbar; nur der
volle T-002-Sprachmodell-Lauf braucht GPU-Zeit (bereits im Review beziffert). **Ehrliche
Einschränkung, für alle Anwendungen gleich:** Die Struktur gewinnt an Rechenzeit erst ab 16.384
Dimensionen — jede Testfeld-Anwendung liegt weit darunter und zeigt bestenfalls Qualität bei
wenig Parametern, nicht Geschwindigkeit.

## Roadmap

**Bis 30.11.:** Entscheidungs-KI-Testbett (neu, billig), Q-019-Lauf, Q-020, Patch-Encoder-Prototyp
mit Vortest (Q-021 vorregistriert, Ausstiegskriterium: besteht der Vortest bis Mitte November
nicht, wird er als ungeprüftes Konzept berichtet, nicht als Messung).
**Bis Mitte Januar:** T-002-GPU-Lauf, T-003 (QIRE-Sequenzmischer als fünfter T-002-Arm),
CIFAR-10-klein und UCR-Zusatzdatensätze, falls der Patch-Encoder sich bewährt hat.

## Neue Läufe zur Übernahme

- **Q-021** — Patch-Encoder gegen dichten Encoder (isoliert die Encoder-Frage).
- **T-003** — QIRE-Sequenzmischer als fünfter T-002-Arm.
- **D-001** — Entscheidungs-KI auf OpenML-CC18 gegen Gradient Boosting.
