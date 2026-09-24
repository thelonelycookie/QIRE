# Kurzfassung — Entwurf (Stand 6. August 2026)

**Das ist ein Gerüst, kein fertiger Text.** Struktur, Zahlen und
Argumentationslinie stammen aus den echten Versuchsergebnissen und stimmen.
Die Formulierungen musst du dir zu eigen machen — eine Jury wird dich zu genau
diesen Punkten befragen, und du solltest jeden Satz verteidigen können.

**Zur Anmeldung:** Gib die Nutzung von KI-Werkzeugen im Formular an. Das ist
unproblematisch und schützt dich; undeklariert und später auffliegend wäre es
ein echtes Problem.

**Belegstand:** Jede Zahl in diesem Entwurf ist durch gespeicherte Rohdaten
unter `results/experiments/` gedeckt und mit `python -m qire_ai.audit`
nachprüfbar. Die vorige Fassung vom 2. August enthielt einen zurückgezogenen
Befund als Hauptergebnis; sie ist vollständig ersetzt.

---

## Titel

Machen quanteninspirierte Prinzipien KI ressourceneffizienter?
Sieben Mechanismen, sieben Abgrenzungstests — und was übrig bleibt

## Fragestellung

Der Begriff „quanteninspiriert" ist in der KI-Forschung verbreitet, wird aber
selten geprüft. Meine Frage war nicht nur, *ob* solche Prinzipien Ressourcen
sparen, sondern **unter welchen Bedingungen** — und wie man das überhaupt
sauber testet.

Untersucht habe ich sieben Mechanismen. Fünf laufen auf normaler Hardware ohne
Quantencomputer, zwei rechnen echte Quantenmechanik (exakt simuliert, einer
zusätzlich auf einem IBM-Prozessor ausgeführt):

1. **Born-Regel-Gating** — Auswahl von Rechenpfaden über |Amplitude|² statt Softmax
2. **Verschränkungsentropie** als Vorhersage für Kompressionserfolg
3. **Tensor-Train-Zerlegung** — Gewichtsmatrizen als Kette von Kern-Tensoren
   (Matrix-Product-States aus der Quanten-Vielteilchenphysik)
4. **Unitarität** — orthogonale Gewichtsblöcke als klassisches Gegenstück
   zu Quantengattern
5. **DMRG-Rangzuweisung** — Rang dort vergrößern, wo ein
   Verschränkungskriterium es verlangt
6. **Echte Quantenschaltung** — parametrisierte Schaltung als trainierbare
   Netzwerkschicht, Gradienten fließen durch Superposition und Verschränkung
7. **Quantenkerne** — Überlapp zweier Quantenzustände als Kernfunktion einer SVM

Entscheidend ist bei jedem die **Abgrenzung**: geprüft wird nicht gegen ein
unkomprimiertes Netz, sondern gegen eine Kontrollbedingung, die dieselbe
Wirkung ohne den quantenspezifischen Anteil erzielt.

## Hauptergebnis: kein Mechanismus übersteht seine Abgrenzung

| Mechanismus | Kontrollbedingung | Ergebnis |
|---|---|---|
| Born-Regel-Gating | klassisches Softmax-Gating | kein Unterschied (p = 0,25, 648 gematchte Paare) |
| Verschränkungsentropie | zufällig initialisierte Matrizen | widerlegt (trainiert ≈ zufällig) |
| Tensor-Train-Bonds | Blockdiagonal bei gleichem Budget | Kontrolle besser (p = 0,004) |
| Unitarität | Spektralnormalisierung | **unentschieden** (siehe unten) |
| DMRG-Rangzuweisung | zufälliges Wachstum | kein Unterschied (p = 1,00) |
| echte Quantenschaltung | klassischer Nachbau | Kontrolle **nicht** besser (siehe unten) |
| Quantenkerne | RBF / polynomial / linear | Kontrolle besser (p = 0,0078) |

Zur vierten Zeile: Ein eigener vorregistrierter Lauf mit 20 Seeds und Lernrate
je Arm zeigt auf MNIST einen Vorteil der Orthogonalität auf **beiden** Tiefen
(+0,0025 und +0,0051, Holm p = 0,0020 und 0,0001), der sich auf Fashion-MNIST
aber nicht wiederholt. Nach dem vorher festgelegten Kriterium reicht das nicht.
Ich führe die Zeile deshalb als offen und nicht als Befund.

## Warum sie scheitern — ein Prüfmaßstab und eine Erklärung

Aus den Ergebnissen lässt sich ein Kriterium ableiten:

> Ein quanteninspirierter Ansatz kann nur dann etwas beitragen, wenn er dem
> Modell eine **strukturelle Einschränkung** auferlegt, die klassisch nicht
> ohnehin verfügbar ist. Wird lediglich eine bestehende Operation in
> Quantenvokabular umbenannt, oder ist die Wirkung durch eine einfachere
> klassische Struktur reproduzierbar, bleibt kein Beitrag übrig.

Dazu kommt eine gemeinsame physikalische Erklärung, quantitativ aus dem
siebten Test hergeleitet. Die mittlere Ähnlichkeit im Quantenkern **halbiert
sich mit jedem Qubit**:

| Qubits | 2 | 3 | 4 | 6 |
|---|---|---|---|---|
| mittlere Ähnlichkeit | 0,349 | 0,162 | 0,082 | 0,024 |

Bei sechs Qubits ist die Kernmatrix praktisch die Einheitsmatrix — jeder Punkt
ist nur sich selbst ähnlich, und ein Lernverfahren hat nichts mehr zum
Verallgemeinern.

> In einem 2ⁿ-dimensionalen Hilbertraum sind zwei zufällige Zustände fast immer
> orthogonal. Was Quantensysteme ausdrucksstark macht — die exponentiell
> wachsende Dimension — zerstört zugleich die geteilte Struktur, die
> Lernverfahren brauchen.

Das ist für mich der wichtigste theoretische Ertrag der Arbeit: statt sieben
unverbundener Nullresultate eine gemeinsame, gemessene Ursache.

## Was dennoch hält: Parametereffizienz einer echten Quantenschaltung

Der sechste Mechanismus liefert einen Befund, der jeder Kontrolle standhält —
allerdings nicht den, den ich erwartet hatte. MNIST und Fashion-MNIST,
15 Seeds, Lernrate je Arm gesucht, balancierte Genauigkeit, Holm-korrigiert:

| MNIST | Genauigkeit | Parameter der mittleren Schicht |
|---|---|---|
| **Quantenschaltung** | **0,9339** | **80** |
| dieselbe Schaltung, 1 Lage | 0,9337 | **16** |
| klassischer Nachbau (normieren + Drehung + quadrieren) | 0,9213 | 2.048 |
| klassisches Netz (Linear/Tanh + Normierung) | 0,8407 | 67.848 |

**Die Schaltung schlägt das klassische Netz um +0,0932 (MNIST) und +0,0599
(Fashion-MNIST), jeweils Holm p = 0,0001** — bei 848-fach weniger Parametern in
der verglichenen Schicht.

Zwei Kontrollen zerlegen den Grund, und beide fallen überraschend aus:

- **Verschränkung trägt nichts bei.** Dieselbe Schaltung ohne CNOT-Gatter ist
  gleichwertig (−0,0015, Äquivalenztest p < 0,05 bei ±0,005). Das ist ein
  positiver Gleichwertigkeitsnachweis, kein bloßes „nicht signifikant".
- **Tiefe trägt nichts bei.** Eine einzige Lage mit **16** Winkeln ist
  gleichwertig zu fünf Lagen mit 80 (+0,0002).

Was wirkt, ist also nicht Superposition über viele Lagen und nicht
Verschränkung, sondern die **Struktur**: normieren → normerhaltende Drehung →
quadratische Auslese.

### Ein Mechanismus, zweimal getestet, zweimal anders — die Position entscheidet

Die quadratische Auslese ⟨Z⟩ = Σ|Amplitude|² **ist die Born-Regel** — also
genau der Mechanismus, den ich im ersten Abgrenzungstest als wirkungslos
verworfen hatte. Dort saß er als *Gate*, dessen Ergebnis anschließend auf Summe
1 normiert wird; die Normierung wirft gerade die Information wieder weg, die
das Quadrieren erzeugt. Hier sitzt er als *Auslese* und ist damit die
Nichtlinearität selbst.

Wie viel das ausmacht, lässt sich aus denselben Rohdaten abzählen. Der rein
klassische Arm (normieren + freie Matrix + quadrieren, ohne jeden
Quantenbegriff) gegen das Linear/Tanh-Netz:

| | MNIST | Fashion-MNIST |
|---|---|---|
| Vorteil der Schaltung insgesamt | +0,0932 | +0,0599 |
| davon die Quadratur allein | **+0,0806 (87 %)** | **+0,0548 (92 %)** |

Rund **neun Zehntel des Vorteils sind die Born-Regel als Nichtlinearität** — und
die ist klassisch in wenigen Zeilen implementierbar. Daraus folgt eine Aussage,
die über diesen Einzelfall hinausgeht:

> Ein Mechanismus trägt nichts bei, wo seine Wirkung anschließend wieder
> herausnormiert wird, und viel, wo er als Nichtlinearität Merkmale erzeugt.
> Die **Position im Netz** entscheidet, nicht die Operation.

Mein erster Abgrenzungstest hat also nicht die Born-Regel widerlegt, sondern
eine Position. Das ist ein methodischer Fehler, den ich selbst gemacht und
selbst gefunden habe — und er dürfte in der Literatur häufig sein.

**Ein nachgerechneter Vergleich, den ich ausdrücklich nicht als Befund führe.**
Der klassische Nachbau genau dieser drei Schritte (2.048 Parameter) stand nicht
in der vorregistrierten Testfamilie. Nachträglich gerechnet unterliegt er auf
MNIST deutlich (+0,0126 zugunsten der Schaltung, 15 von 15 Seeds, p = 0,0001),
auf Fashion-MNIST in derselben Richtung, aber nicht signifikant (+0,0051,
p = 0,23). Bemerkenswert dabei: auch die Ein-Lagen-Variante mit **16**
Parametern liegt auf MNIST vor dem Nachbau (+0,0123, p = 0,0003).

Weil der Vergleich nicht vorher angemeldet war und auf dem zweiten Datensatz
nicht repliziert, ist das **kein Ergebnis, sondern ein Anlass für einen eigenen
vorregistrierten Lauf.** Dieselbe Zurückhaltung habe ich an anderer Stelle
schon einmal geübt, und dort hat sie sich ausgezahlt.

**Was dazugehört:** Die 848 gelten für die verglichene Schicht. Die
Gesamtmodelle liegen bei 201.130 gegen 268.898 Parametern, weil in jedem Arm
ein klassischer Encoder (784 → 256) steckt. Die Aussage ist eine über die
Schicht, nicht über das ganze Netz.

## Die Architektur auf klassischer Hardware — und wo sie gewinnt

Eine Quantenschaltung mit n Qubits verknüpft in jedem Gatter genau die
Amplitudenpaare, die sich in einem Bit unterscheiden. Über alle Qubits ist das
die Schmetterlingsstruktur der schnellen Fouriertransformation, also
**O(d·log d) statt O(d²)**. Ich habe die Architektur deshalb als klassisches
Rechenverfahren umgesetzt — kein Simulator, sondern dieselbe Abbildung
effizient ausgeführt. Verifiziert gegen den Simulator: identische Funktion,
größte Abweichung 7·10⁻⁷ über 30 Konfigurationen.

Der entscheidende Unterschied zu einem gewöhnlichen Butterfly-Netz ist die
**Gewichtsteilung**: ein Winkel wirkt auf eine ganze Qubit-Ebene, weil ein
Qubit-Gatter physikalisch auf ein Qubit wirkt, unabhängig von der Größe des
Zustandsraums.

| Qubits | Dimension | Quantenarchitektur | Butterfly ungeteilt | dichte Schicht |
|---|---|---|---|---|
| 8 | 256 | **80** | 10.240 | 65.536 |
| 12 | 4.096 | **120** | 245.760 | 16.777.216 |

Gemessene Inferenzzeit gegen eine dichte Schicht gleicher Breite, Median aus
sieben Messreihen:

| Dimension | 4.096 | 8.192 | **16.384** | 32.768 |
|---|---|---|---|---|
| Faktor | 0,21× | 0,40× | **1,99×** | dichte Schicht nicht baubar |

**Ab 16.384 Dimensionen ist die quantenabgeleitete Architektur rund doppelt so
schnell** (in allen sieben Messreihen, Spanne 1,75–2,10). Und ab 32.768 lässt
sich die dichte Schicht auf meinem Rechner gar nicht mehr bauen — ihre
Gewichtsmatrix bräuchte 4,3 GB, die Quantenarchitektur kommt mit 150 Zahlen aus.
Das ist kein Geschwindigkeits-, sondern ein Machbarkeitsunterschied.

**Ehrliche Einschränkung:** Das ist ein Vorteil bei der **Inferenz**. Im
Training, mit Rückwärtslauf und Optimierer, schrumpft er auf etwa 1,18× —
gemessen an der Breite, die eine dichte Schicht im selben Zeitbudget erreicht
(14.896 statt 16.384).

## Ausgeführt auf einem echten Quantenprozessor

Das Modell wurde in Simulation trainiert und anschließend auf `ibm_fez`
ausgeführt — einem IBM-Prozessor mit 156 Qubits. Nachprüfbar über die Job-IDs
`d9ns0pc60llc73caf80g` und `d9ns10ssfqic73ardag0`.

| | flach (33 Gatter) | tief (67 Gatter) |
|---|---|---|
| Genauigkeit Simulation | 0,500 | 0,450 |
| **Genauigkeit Hardware** | **0,500** | **0,450** |
| Übereinstimmung der Vorhersagen | 0,850 | 0,750 |
| Dämpfung der Messwerte | 0,852 | 0,707 |

Die Genauigkeit bleibt trotz Rauschen erhalten. Und der Tiefeneffekt aus einer
früheren Messung bestätigt sich an einer laufenden Anwendung: bei doppelter
Gatterzahl bleiben statt 85 % nur noch 71 % der Signalstärke. Dieselbe Richtung
hatte ich zuvor an Bell- und GHZ-Zuständen gemessen (1,57 % gegen 2,06 %
Fehlerrate, p = 0,043 bei zehn Wiederholungen — bei drei Wiederholungen war
derselbe Effekt noch nicht nachweisbar, p = 0,65).

## Methodische Erkenntnisse

Diese Arbeit ist zur Hälfte eine Methodenarbeit geworden, und das ist ihr
belastbarster Teil.

**Die übliche Effizienzkennzahl führt in die Irre.** Fast jede Veröffentlichung
zu effizienter KI berichtet FLOPs. In diesem Projekt hat sich **fünfmal
unabhängig** gezeigt, dass diese Zahl die tatsächliche Rechenzeit nicht
vorhersagt — teils sagt sie das Gegenteil voraus. Beim letzten Mal betraf es
die Butterfly-Architektur selbst: theoretisch 340-fach weniger Arithmetik,
gemessen zunächst siebenfach langsamer, weil viele kleine Operationen gegen
einen einzigen optimierten Matrixaufruf verlieren.

**Kleine Stichproben kippen.** An einem einzigen Tag haben drei Befunde
zwischen 10 und 30 Seeds die Richtung gewechselt. Bei acht Seeds ist p = 0,0078
der rechnerisch kleinste erreichbare Wert eines gepaarten Tests —
„hochsignifikant" heißt dort nur „so gut wie es geht".

**„Nicht signifikant" ist kein Gleichheitsnachweis.** Für Aussagen der Form
„gleich gut bei weniger Ressourcen" braucht es einen Äquivalenztest gegen eine
vorher festgelegte Grenze. Erst damit wurden aus zwei Vermutungen zwei
nachgewiesene Gleichwertigkeiten.

**Ein Aufbau muss beweisen, dass er seine eigene Frage auflösen kann.** In
einer meiner Architekturen trennte sich ein 32-fach größeres Parameterbudget
nicht signifikant — dieser Aufbau war für seine Frage blind, und ein daraus
gewonnener Hauptbefund musste zurückgezogen werden. Seitdem hat jeder Lauf eine
Positivkontrolle.

## Zwei klassische Befunde aus der Quantensuche

Beide entstanden *aus* der Beschäftigung mit Quantenmechanik, sind aber
klassisch erklärbar — und werden deshalb ausdrücklich **nicht** als
quanteninspiriert ausgegeben:

| Verfahren | Ergebnis |
|---|---|
| Rang-64-Projektion einer **fertig trainierten** dichten Schicht | 32× weniger Parameter, **nachgewiesen gleichwertig** (Äquivalenztest p < 0,0001, beide Datensätze, 15 Seeds) |
| Der Effektivrang einer trainierten Schicht wird von der **Aufgabe** gesetzt, nicht von der Architektur | kontrolliertes 6×4-Gitter mit untrainierter Kontrolle: untrainiert folgt der Rang exakt der Datendimension (ρ = +0,986) und ist von der Klassenzahl unabhängig (ρ = +0,002); trainiert hängt er von beiden ab |

Der zweite erklärt rückblickend mehrere Negativbefunde auf einmal: Wenn eine
Aufgabe nur rund sieben Richtungen braucht, kann keine strukturierte
Ersatzschicht durch mehr Kapazität gewinnen — der Rang war nie die bindende
Beschränkung.

## Einordnung

Das Ergebnis ist negativ für die Ausgangsthese, aber es ist ein Ergebnis:
sieben Mechanismen, jeweils mit fairer Kontrollbedingung geprüft, mit einem
Kriterium, das erklärt warum, einer gemessenen gemeinsamen Ursache — und einer
Architektur, die bei großer Breite nachweislich schneller ist als ihre
klassische Entsprechung.

Der Begriff „quanteninspiriert" wird in der Literatur häufig ohne solche
Prüfung verwendet. Der Prüfmaßstab und die sieben durchgeführten
Abgrenzungstests sind auf andere Arbeiten übertragbar.

---

# Zusatzmaterial (nicht Teil der Kurzfassung)

## Verworfene Ergebnisse — bewusst dokumentiert

| Beobachtung | warum verworfen |
|---|---|
| „Spektral-normalisierte Blockschicht, 819× weniger Parameter, besser als dicht" | galt nur in einer Architektur, deren Blindheit gemessen ist (Positivkontrolle p = 0,23); ohne die dichten Mischschichten dreht das Vorzeichen |
| „Low-Rank Rang 64, kein messbarer Verlust" | 10 Seeds und genäherte SVD; mit 30 Seeds und exakter SVD −0,0066, Äquivalenzgrenze klar verfehlt |
| „Monarch ist gleichwertig zu dicht" | bei 10 Seeds −0,0008 (p = 0,92), bei 30 Seeds −0,0045 (p = 0,001) |
| Tensor-Train sei *besser* als dicht (+0,0065) | bei 3 Seeds; mit 10 Seeds p = 0,23 → Rauschen |
| Strukturbeschränkung wirke regularisierend | auf frischen Seeds zusammengebrochen; 4 von 8 Zellen signifikant *schlechter* |
| Speedup von 30,1× bei Batch 4 | reproduzierbar, aber Ursache ist ein BLAS-Kernelwechsel der *Referenz* |
| „13–15× schneller als dicht" | falsche Vergleichsgruppe — Low-Rank erreicht am selben Budget rund 120× |
| „Der klassische Nachbau schlägt die Quantenschaltung" (−0,0147) | repliziert nicht: mit 15 Seeds und Lernrate je Arm +0,0044 (p = 0,49) |

## Selbst gefundene Fehler im eigenen Messaufbau

1. **Gruppierungsfehler** — Modellkennungen ohne kapazitätsbestimmende
   Parameter, dadurch verschiedene Architekturen als „gleiches Modell" gepoolt
2. **Fehlender Signifikanztest** — die Plattform verglich nur Mittelwerte
3. **Ops statt Zeit** — die zentrale Effizienzkennzahl war von der Realität entkoppelt
4. **Falsche Teststatistik** — Korrelation statt Effektstärke gemessen
5. **Scheinkorrelation durch Konstruktion** — zwei verglichene Größen
   enthielten denselben Term; per Simulation nachgewiesen
6. **Skalierungsfehler bei Bilddaten** — merkmalsweise Standardisierung bläht
   fast konstante Randpixel um Faktor 8368 auf
7. **Fehlende Kontrollbedingung** — Tensor-Train wurde lange nur gegen dichte
   Schichten verglichen, nie gegen parametergleiche Alternativen
8. **Keine gespeicherten Rohdaten** — bis zum 3. August existierte jede Zahl nur
   als Konsolenausgabe; seitdem wird jede Messung als Zeile abgelegt und mit
   `qire_ai.audit` gegen das Tagebuch geprüft
9. **Zu enges Lernratengitter** — fünf von sechs Armen wählten den Randwert; das
   Optimum lag außerhalb des Suchbereichs
10. **Ungleiche Vorverarbeitung zwischen den Armen** — der klassische Arm bekam
    die Normierung nicht, die zur Amplituden-Kodierung gehört, und landete auf
    Rateniveau; das hätte einen Vorteil von 78 Prozentpunkten vorgetäuscht

## Nicht abgeschlossen

- **Dichtematrix-gesteuerte adaptive Berechnung:** implementiert und verifiziert,
  aber in dieser Umgebung **nicht prüfbar** — der teure Pfad ist bei den
  verfügbaren Datensätzen nicht besser als der billige. Ausdrücklich nicht
  falsifiziert.
- **Unitarität in der Tiefe:** Der Unterschied zu Spektralnormalisierung sollte
  sich erst bei vielfacher Anwendung aufbauen; gemessen wurde bisher nur bei
  Tiefe 1 und 3. Auf MNIST wächst der Effekt zwischen beiden Tiefen.

## Grenzen (gehören in die schriftliche Arbeit)

- Energieschätzung beruht auf angenommener CPU-Wattzahl, keine Hardwaremessung
- Alle Experimente auf CPU. Die frühen Läufe einfädig zugunsten der
  Reproduzierbarkeit; die MNIST-Ablationen ab A-002 mit sechs Threads, für alle
  Arme eines Laufs identisch
- Zeitmessungen nur einfädig und überwiegend bei Batch-Inferenz. Das eigentliche
  Argument mancher Vergleichsverfahren ist Hardware-Effizienz auf GPUs — das
  habe ich nicht gemessen und kann darüber nichts sagen
- Datensätze und Modellgrößen sind durch die verfügbare Hardware begrenzt
- Die Gleichwertigkeitsaussagen gelten innerhalb der genannten Grenze
  (±0,005 balancierte Genauigkeit), nicht absolut
