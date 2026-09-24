# Forschungsideen — zweite Blicke und neue Richtungen

Stand 5. August 2026, nachts. Zwei Teile: **A** nimmt Befunde, die als erledigt
oder verworfen gelten, und fragt, ob sie unter einem anderen Blickwinkel doch
etwas hergeben. **B** und **C** sind neue Richtungen für die QuantenKI und für
die Quantenarchitektur auf klassischer Hardware.

Jeder Punkt nennt den **Anlass aus den eigenen Daten** — keine Idee steht hier,
weil sie interessant klingt, sondern weil eine vorhandene Messung sie nahelegt.

---

# A — Zweite Blicke auf Verworfenes

## A1. Die Born-Regel war nicht falsch, sie stand an der falschen Stelle
### ✅ ERLEDIGT am 6. August — und die Zahl ist besser als erwartet

Ausgerechnet auf den vorhandenen Rohdaten: **86,5 % (MNIST) beziehungsweise
91,5 % (Fashion-MNIST) des Vorteils der Quantenschaltung** entfallen auf die
quadratische Auslese allein, also auf die Born-Regel als Nichtlinearität —
klassisch implementierbar, ohne jeden Quantenbegriff. Beide Male 15/15 Seeds,
p=0,0001. Die verbleibenden 8,5 bis 13,5 % sind der quantenspezifische Rest,
und genau den prüft Q-010 vorregistriert nach.

Ausführlich im `PROJEKTTAGEBUCH.md`, Eintrag „6. August — Befund 1 und Befund 6
sind derselbe Mechanismus an zwei Stellen". Der Text unten bleibt als
Herleitung stehen.


**Anlass.** Befund 1: Born-Gating gegen Softmax, kein Unterschied (p=0,25,
648 gematchte Paare). Begründung damals: beide bilden auf dieselbe Simplex ab,
also dieselbe Funktionsklasse. — Aber Q-003 hat später herausgearbeitet, dass
die Wirkung der Quantenschaltung an der **quadratischen Auslese** hängt
(⟨Z⟩ summiert über |Amplitude|²). Und das *ist* die Born-Regel.

**Der zweite Blick.** Die Born-Regel wurde als **Gate** getestet — an einer
Stelle, an der anschließend ohnehin normiert wird, sodass die Quadratur
herausfällt. Als **Auslese** getestet, trägt exakt dieselbe Operation
+0,09 Genauigkeit gegenüber 67.848 Parametern. Befund 1 und der Q-Erfolg sind
derselbe Mechanismus an zwei Positionen — einmal wirkungslos, einmal tragend.

**Was zu tun ist.** Das ist keine neue Messung, sondern eine **Neuformulierung
eines vorhandenen Ergebnisses**: „Die Born-Regel trägt nichts bei, wo sie
normiert wird, und sie trägt viel bei, wo sie als Nichtlinearität wirkt." Das
verbindet den ersten und den sechsten Abgrenzungstest und macht aus zwei
Einzelbefunden eine Aussage über die *Position* eines Mechanismus im Netz.
Aufwand: Schreibarbeit. Wert: hoch, weil es die Arbeit zusammenbindet.

## A2. Befund 2 wurde am falschen Spektrum gemessen — nachweislich

**Anlass.** Befund 2 (Verschränkungsentropie sagt Kompression vorher) wurde
verworfen, weil trainierte Matrizen kaum weniger Entropie haben als zufällige.
Gemessen wurde am Spektrum der **Gewichtsmatrix**. B-011 hat später bewiesen,
dass genau das die falsche Größe ist: auf den **Ausgaben echter Daten** erfasst
Rang 64 99,8 % statt 65 %.

**Was zu tun ist.** Entropie der Aktivierungsverteilung messen, nicht der
Gewichte — `qire_ai/spectral_probe.py` (`spectral_entropy`) kann das bereits.
Wenn Befund 2 aus demselben Grund fiel wie B-008, wäre einer von sieben
Negativbefunden zurückzuholen. Aufwand: ein mittlerer Lauf.

## A3. Drei Befunde fielen, weil der Rang nie die bindende Beschränkung war

**Anlass.** B-014: die Aufgabe braucht rund **sieben** Richtungen. Deshalb
lieferten Rang 8, 64 und 256 in B-008 dieselbe Genauigkeit. Das erklärt aber
nicht nur B-008 — es erklärt auch, warum **Befund 3** (Tensor-Train-Bonds) und
**Befund 5** (DMRG-Rangkriterium) nichts zeigen konnten: Beide handeln davon,
Rang *klug zu verteilen*. Wenn Rang nicht knapp ist, kann kein
Verteilungskriterium etwas gewinnen.

**Der zweite Blick.** Beide wurden in einem Regime getestet, in dem ihre eigene
Frage bedeutungslos war. Das ist kein Nullbefund, das ist ein **untauglicher
Aufbau** — dieselbe Diagnose wie bei der durchgefallenen Positivkontrolle in
B-004.

**Was zu tun ist.** B-014 hat den Generator für Daten mit **vorgegebener
intrinsischer Dimension d**. Damit lässt sich eine Aufgabe bauen, bei der Rang
nachweislich bindet (hohes d, viele Klassen), und Befund 3 und 5 dort
wiederholen. Vorregistriert mit der Bedingung: der Aufbau muss zuerst zeigen,
dass mehr Rang überhaupt mehr Genauigkeit bringt — sonst ist er wieder blind.
Aufwand: mittel. Wert: hoch, weil zwei der sieben Negativbefunde davon abhängen.

## A4. Unitarität wurde nur bei Tiefe 1 und 3 getestet

**Anlass.** Befund 4: Orthogonalität gegen Spektralnormalisierung, kein
Unterschied (p=0,32; auf frischen Seeds p=0,037; B-007 klärt es gerade).

**Der zweite Blick.** Der Unterschied zwischen „größter Singulärwert = 1" und
„**alle** Singulärwerte = 1" wirkt sich erst aus, wenn eine Abbildung **oft
hintereinander** angewandt wird. Bei Spektralnormalisierung dürfen die übrigen
Singulärwerte schrumpfen; über 20 Schichten hinweg verschwindet das Signal in
diesen Richtungen, bei echter Unitarität nicht. Genau deshalb sind unitäre RNNs
in der Literatur ein Thema. Bei Tiefe 1–3 kann sich das nicht zeigen.

**Was zu tun ist.** Denselben Vergleich bei Tiefe 16, 32, 64 — oder rekurrent
mit geteilten Gewichten. Vorhersage vorher festlegen: die Lücke wächst monoton
mit der Tiefe. Das wäre ein **falsifizierbares Gesetz** statt eines
Einzelvergleichs, und es ist der Punkt, an dem Unitarität klassisch nicht
ersetzbar wäre. Aufwand: mittel. **Das halte ich für den stärksten Kandidaten
in Teil A.**

## A5. Die Konzentration ist ein Fehler für Ähnlichkeit — und ein Vorteil für Gedächtnis

**Anlass.** Befund 7: Die mittlere Kernähnlichkeit halbiert sich je Qubit
(0,349 → 0,162 → 0,082 → 0,024). Bei 6 Qubits ist die Kernmatrix praktisch die
Einheitsmatrix, die SVM kann nichts verallgemeinern.

**Der zweite Blick — der unkonventionellste Punkt dieser Liste.** Fast
orthogonale Zustände sind für einen *Klassifikator* tödlich. Für einen
**assoziativen Speicher** sind sie das Optimum: Die Kapazität eines solchen
Speichers ist genau dann maximal, wenn die gespeicherten Muster einander nicht
überlappen. Was Befund 7 als Scheitern misst, ist die Eigenschaft, die man für
Schlüssel in einem Schlüssel-Wert-Speicher haben *will* — minimale Interferenz.

**Was zu tun ist.** Quanten-Merkmalsabbildung als **Schlüsselerzeugung** testen,
nicht als Kernfunktion: gegen zufällige Projektionen, gemessen an der
Abrufgenauigkeit bei wachsender Zahl gespeicherter Paare. Kontrolle: zufällige
orthogonale Projektion gleicher Dimension. Wenn die Quantenabbildung dort
gewinnt, ist derselbe gemessene Effekt einmal Negativ- und einmal Positivbefund
— und die Arbeit hätte erklärt, *wofür* ein exponentieller Zustandsraum taugt.
Aufwand: mittel. Für einen Chatbot direkt anschlussfähig (Attention-Schlüssel).

## A6. C-001 ist nicht mehr unprüfbar

**Anlass.** Dichtematrix-Routing wurde zurückgestellt, weil der teure Pfad auf
den verfügbaren Datensätzen nicht besser war als der billige — selbst perfektes
Routing hätte höchstens +0,0066 gebracht.

**Der zweite Blick.** Das war eine Aussage über die **Datensätze**, nicht über
den Mechanismus. Seit B-014 gibt es einen Generator für Aufgaben mit
einstellbarer Schwierigkeit. Eine Aufgabe, bei der Kapazität nachweislich zählt,
lässt sich damit konstruieren.

**Was zu tun ist.** Erst die Vorbedingung messen (teurer Pfad muss signifikant
besser sein als der billige), dann C-001 dort testen. Ohne die Vorbedingung
nicht starten. Aufwand: mittel.

## A7. Zwei ungeklärte Auffälligkeiten, die nie verfolgt wurden

- **Der `weight_decay`-Arm kollabiert auf exakt 0,1000 bei Streuung exakt 0** —
  in B-002 und wieder in B-007. Exakt reproduzierbar ist kein Rauschen, sondern
  ein Mechanismus (vermutlich absterbende Aktivierungen in gestapelten
  Blockschichten). Zweimal als „ungeklärte Auffälligkeit" abgelegt. Eine
  Nachmittagsmessung würde daraus einen sauberen kleinen Befund machen: *wann*
  genau kollabiert eine gestapelte strukturierte Schicht unter Weight Decay.
- **Rang 32 ist schlechter als Rang 16** — mehr Parameter *und* weniger
  Genauigkeit (0,9430 gegen 0,9544). Steht seit dem 2. August als offene Frage
  im Tagebuch. Kandidat: dasselbe Optimierungsproblem, das Q-005 gerade misst.
  Nach Q-005 billig zu klären.

---

# B — Neue Richtungen für die QuantenKI

## B1. Der reellwertige Ry-Kreis — der direkteste Weg zu Geschwindigkeit

**Anlass.** Q-003-replication (5. August): Verschränkung trägt nichts bei
(TOST-gleichwertig innerhalb ±0,005), Tiefe trägt nichts bei (ebenfalls
TOST-gleichwertig, 16 Parameter = 80 Parameter).

**Die Idee.** Wenn auch **Rz** nichts beiträgt, bleibt eine Schaltung aus
reinen Ry-Rotationen — und die ist **reellwertig**. Das spart den Faktor 4 an
Arithmetik gegenüber komplexen Zahlen und halbiert die Parameterzahl noch
einmal. Aus 80 würden 8.

**Was zu tun ist.** Arm `hybrid_ry_only` in die vorhandene Batterie aufnehmen,
gegen `hybrid` mit TOST. Aufwand: gering, das Gerüst steht. **Sofort machbar.**

## B2. Den Encoder mitkomprimieren — der größte Hebel überhaupt

**Anlass.** Gemessen am 5. August: die Schaltung hat 80 Parameter, das
Gesamtmodell 201.130. Der klassische Encoder (784 → 256) frisst den ganzen
Vorteil auf. Jede Gesamt-Parameteraussage scheitert daran.

**Die Idee.** Den Encoder **selbst** als Butterfly mit geteilten Gewichten
bauen, statt als dichte Schicht. Dann kostet er O(d log d) Operationen und
2·n·L Parameter statt 784·256. Das Gesamtmodell fiele von 201.130 auf wenige
hundert Parameter.

**Warum das der wichtigste Punkt der Liste ist.** Ohne ihn bleibt die
848×-Aussage auf „in der Mitte" beschränkt und ist angreifbar. Mit ihm wäre es
eine Aussage über das **ganze Modell**. Aufwand: mittel, `butterfly_layer.py`
liefert die Bausteine. **Priorität 1 nach den laufenden Messungen.**

## B3. Gewichtsteilung auch über die Lagen

**Anlass.** Tiefe trägt nichts bei (gemessen). Trotzdem hat jede Lage eigene
Winkel.

**Die Idee.** Dieselben 2n Winkel in allen L Lagen wiederverwenden. Parameterzahl
fällt von 2nL auf **2n** — bei n=8 also von 80 auf 16, unabhängig von der Tiefe.
Da Tiefe nachweislich nichts beiträgt, sollte es nichts kosten. Aufwand: gering.

## B4. Der minimale Kreis auf echter Hardware

**Anlass.** Die Hardware-Läufe mussten auf Winkel-Kodierung ausweichen, weil
Amplituden-Kodierung hunderte Gatter braucht und bei 2 % Fehler je drei Gatter
im Rauschen untergeht. Jetzt ist gemessen: **eine Lage reicht, CNOTs sind
entbehrlich.**

**Die Idee.** Genau diese minimale Schaltung ist das, was heutige Hardware gut
kann — sehr flach, kaum Zweiqubit-Gatter. Ein Hardware-Lauf damit sollte
deutlich näher an der Simulation liegen als die früheren 0,850 / 0,707
Übereinstimmung. Aufwand: gering, Kontingent reicht. **Für den Stand ein
starkes Belegstück**, und es nutzt die 10 Freiminuten sinnvoller als eine
Wiederholung.

## B5. Quanten-Merkmale als Attention-Schlüssel

Siehe A5. Der Weg von „Quantenkerne scheitern" zu „genau deshalb taugen sie als
Schlüssel" ist die interessanteste Wendung, die in den Daten steckt — und die
einzige Idee dieser Liste, die direkt in einem Sprachmodell landet.

---

# C — Quantenarchitektur auf klassischer Hardware

## C1. Den Übergangspunkt messen statt extrapolieren

**Anlass.** Gemessen am 5. August: die Butterfly-Umsetzung ist gegenüber einer
dichten Schicht 0,00× / 0,01× / 0,04× / **0,13×** bei 6 / 8 / 10 / 12 Qubits.
Der Rückstand schrumpft um Faktor 3–4 je verdoppelter Qubitzahl. Fortgeschrieben
läge der Gleichstand bei 15–16 Qubits.

**Was zu tun ist.** Bis 16–18 Qubits messen. Reine Zeitmessung, kein Training,
Minuten. Danach steht entweder ein **Übergangspunkt** — dann ist der
Geschwindigkeitsbeweis erbracht — oder eine harte Grenze, und die gehört ebenso
klar berichtet. **Priorität 1, weil billig und entscheidend.**

## C2. Die Kleinoperationen zusammenfassen

**Anlass.** Die Butterfly-Schicht macht n·L = 60 sequentielle Kleinoperationen
mit Umformungen. Das ist zum fünften Mal im Projekt der Grund, warum weniger
Arithmetik nicht weniger Zeit bedeutet.

**Die Idee.** Mehrere Qubit-Ebenen zu einem Durchlauf zusammenfassen (Radix-4
statt Radix-2, wie bei schnellen Fouriertransformationen üblich), oder alle
Lagen als eine einzige Indexoperation vorberechnen. Das ändert die Funktion
nicht — verifizierbar mit `verify()` — senkt aber die Zahl der Kernelaufrufe um
den Faktor 2 bis 4. Aufwand: mittel. Direkt auf den Übergangspunkt wirksam.

## C3. Butterfly als Ersatz für den Feed-Forward-Block eines Transformers

**Anlass.** Der Wunsch nach einem ressourceneffizienten Chatbot. Die
Feed-Forward-Blöcke machen rund zwei Drittel der Parameter eines Transformers
aus, und sie sind genau das, was die Butterfly-Schicht ersetzen könnte.

**Was zu tun ist.** Kleiner zeichenbasierter Transformer als Messplattform, zwei
bis vier Schichten, Dimension 256–512 — auf sechs Kernen in Stunden trainierbar.
Dann Butterfly gegen dicht bei gleicher Breite, gemessen an Perplexität,
Parameterzahl **und** Zeit. Aufwand: hoch (neue Datenpipeline). Wert: das ist
der einzige Punkt der Liste, der die Zielfrage direkt beantwortet.

## C4. Die ungeteilte Variante als Abgrenzungskontrolle

**Anlass.** `butterfly_layer.py` kann bereits `tie_weights=False`.

**Die Idee.** Ein gewöhnliches Butterfly-Netz hat bei 12 Qubits 245.760 Winkel,
die quantenabgeleitete Variante 120. Der direkte Vergleich beantwortet die
Frage, die eine Jury stellen wird: **Bringt die Gewichtsteilung etwas, oder ist
sie nur Sparsamkeit?** Wenn die ungeteilte Variante bei gleicher Genauigkeit
landet, ist die Teilung reine Kompression; wenn sie schlechter ist, wirkt sie
als Regularisierung. Beides ist ein Ergebnis. Aufwand: gering — **die
billigste echte Erkenntnis dieser Liste.**

## C5. Orthogonalität in der Tiefe — die Verbindung zu A4

Die Butterfly-Schicht ist per Konstruktion normerhaltend. Genau das ist die
Eigenschaft, die tiefe und rekurrente Netze brauchen, damit Gradienten weder
explodieren noch verschwinden. A4 sagt, dass Unitarität sich erst in der Tiefe
von Spektralnormalisierung unterscheiden kann; C5 sagt, dass die Butterfly-
Schicht genau dort ihr natürliches Einsatzgebiet hätte. **Zusammen sind das ein
Experiment, nicht zwei** — und potenziell der Punkt, an dem die
quantenabgeleitete Struktur klassisch nicht ersetzbar ist.

---

# Wenn nur drei Dinge Zeit haben

1. **C1** — Übergangspunkt messen. Minuten, und es entscheidet, ob der
   Geschwindigkeitsbeleg existiert.
2. **B2** — Encoder als Butterfly. Der einzige Weg, aus „848× in der Mitte"
   eine Aussage über das ganze Modell zu machen.
3. **A4 + C5** — Unitarität in der Tiefe. Der einzige Punkt, an dem die
   quantenabgeleitete Eigenschaft klassisch nicht nachbaubar sein könnte, und
   damit der aussichtsreichste Kandidat für einen echten Vorteil.

Alles in Teil A hat gemeinsam, dass die Befunde nicht falsch waren, sondern in
einem Regime gemessen wurden, in dem ihre eigene Frage nicht auflösbar war.
Das ist dieselbe Diagnose, die B-004 für die Blockschicht gestellt hat — und
sie ist nach der Erfahrung dieses Projekts eher die Regel als die Ausnahme.
