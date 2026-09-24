# Entwurf: T-Serie — Zwischendarstellung und fusionierte Kernel

Angelegt 10. August 2026. Dies ist ein **Entwurf**, kein Befund. Nichts hier ist
gemessen. Jede Zahl in diesem Dokument ist entweder aus dem Tagebuch übernommen
(dann mit Quelle) oder eine Erwartung (dann als solche markiert).

**Kennung:** Vorschlag `T-001`, `T-002`, … für die Läufe dieser Serie
(T für Transformation). `B`, `C`, `K` und `Q` sind vergeben. Falls dir eine
andere Kennung lieber ist, jetzt ändern — `B-003` ist einmal doppelt vergeben
worden, und das kostet später Aufräumarbeit.

---

## 1. Das Problem in einer Zahl

| | Wert | Quelle |
|---|---|---|
| Arithmetik der Butterfly-Struktur bei d=4096 | **340× weniger** als dicht | Tagebuch 5. Aug |
| Gemessene Zeit bei d=4096 | **5× langsamer** als dicht (0,21×) | Tagebuch 5. Aug, 7 Messreihen |
| Übergangspunkt heute | 16.384 Dimensionen (14 Qubits) | Tagebuch 5. Aug, 7/7 Reihen |
| Vorteil dort | 1,99× Inferenz, **1,18× Training** | Tagebuch 5. Aug |

Zwischen „müsste rechnen" und „kostet" liegen rund **drei Größenordnungen**. Die
Ursache ist im Tagebuch schon benannt: *60 sequentielle Kleinoperationen mit
Umformungen gegen einen einzigen BLAS-Aufruf, dazu Faktor 4 für komplexe Zahlen.*

**Die entscheidende Einsicht für den Entwurf:** Das ist kein Arithmetikproblem,
sondern ein **Speicherverkehrsproblem**. Jede der 60 Stufen läuft heute einmal
komplett über den Zustandsvektor. Bei d=4096 sind das 32 KB (komplex, einfache
Genauigkeit) — das passt in den L1-Cache. Ein fusionierter Kernel macht **einen**
Durchlauf und hält den Vektor dabei resident, statt sechzig Durchläufe durch den
Hauptspeicher.

Deshalb ist die Lücke plausibel schließbar. Sie besteht nicht aus Rechenarbeit,
sondern aus vermeidbarem Hin- und Herschieben.

---

## 2. Warum das mit wachsenden Architekturen wichtiger wird, nicht unwichtiger

Heute zahlst du den Overhead an einer Schicht. In einem Transformer ist die
Struktur der Baustein, aus dem alles besteht: vier Projektionen je
Aufmerksamkeitsblock, zwei Matrizen je FFN, mal Zahl der Lagen. Bei zwölf Blöcken
sind das rund siebzig Stellen mit demselben Overhead. Q-013 stellt bereits
162 gegen 1.051.136 Parameter — sind diese 162 fünfmal langsamer, ist der Befund
wertlos, unabhängig von der Genauigkeit.

Solange die Lücke steht, ist die Architektur nicht einmal fair **bewertbar**.
Der Kernel ist damit Vorbedingung für das Projektziel, nicht Beiwerk.

---

## 3. Die Bausteine — und warum es so wenige sind

Die ganze Familie besteht aus vier Operationen. Das ist der Grund, warum eine
eigene Darstellung hier überhaupt tragfähig ist: der Ausdrucksraum ist klein und
geschlossen.

Sei d = 2ⁿ die Zustandsraumgröße, x der Amplitudenvektor.

**(1) `Norm`** — L2-Normierung. Skalar über den ganzen Vektor.

**(2) `PaarDrehung(q, winkel, geteilt)`** — ein Ein-Qubit-Gatter auf Qubit q
wirkt genau auf die Amplitudenpaare, die sich in Bit q unterscheiden: die Paare
(i, i XOR 2^q). Das ist eine 2×2-Matrix, angewandt auf d/2 Paare.
`geteilt=True` heißt: **ein** Winkelpaar für alle d/2 Paare (die
quantenabgeleitete Kopplung). `geteilt=False`: eigene Winkel je Paar.

**(3) `Permutation(p)`** — und hier liegt der wichtigste Fund dieses Entwurfs:
**Der CNOT-Ring ist reine Indexvertauschung.** CNOT mit Kontrolle c und Ziel t
bildet den Basiszustand |i⟩ auf |i XOR (bit_c(i)·2^t)⟩ ab. Keine Arithmetik,
keine Multiplikation — nur eine Umsortierung. Ein Ring aus n CNOTs ist die
Verkettung von n Permutationen, also **wieder genau eine Permutation.**

**(4) `ErwartungZ(q)`** — die Auslese. ⟨Z_q⟩ = Σ_{bit_q(i)=0} |a_i|² −
Σ_{bit_q(i)=1} |a_i|². Eine vorzeichenbehaftete Summe der Betragsquadrate.

Damit ist die gesamte Schaltung:

```
Norm ∘ [ Permutation ∘ PaarDrehung(n−1) ∘ … ∘ PaarDrehung(0) ]^L ∘ ErwartungZ
```

Vier Bausteine, eine Verkettungsregel. Alles, was das Projekt an Varianten kennt
oder plant — geteilt/ungeteilt, mit/ohne Verschränkung, Re-Uploading, andere
Auslesen, hierarchische Register — ist eine andere Zusammensetzung derselben
vier, nicht eine neue Operation.

---

## 4. Die Umschreibregeln, nach erwartetem Ertrag geordnet

**R1 — Permutationen verschmelzen.** *(sicher, größter struktureller Gewinn)*
n aufeinanderfolgende CNOT-Permutationen komponieren zu einer. Statt n
Umsortierungsdurchläufen je Lage ein einziger vorberechneter Indexvektor. Bei
8 Qubits: 8 Durchläufe → 1, je Lage.

**R2 — Permutation in die Adressierung ziehen.** *(sicher, spart den Durchlauf ganz)*
Eine Permutation gefolgt von einer Paardrehung muss nicht materialisiert werden.
Statt den Vektor umzusortieren und dann Paare zu drehen, dreht man direkt die
*permutierten* Paare — die Permutation verschwindet in der Indexrechnung der
nächsten Stufe. Danach kostet die Verschränkung **keinen Speicherverkehr mehr.**

**R3 — Vollständige Fusion über alle Stufen.** *(der eigentliche Hebel)*
Kacheln über den Batch; je Kachel den Zustandsvektor in den Cache laden, alle
L·n Stufen darauf ausführen, erst dann zurückschreiben. Sechzig Speicher-
durchläufe werden zu einem. Bei d=4096 sind 32 KB resident zu halten, bei d=16384
sind es 128 KB — beides liegt im Bereich, in dem das aufgeht.

**R4 — Reelle Darstellung.** *(Faktor 4 — aber erst zu prüfen, nicht anzunehmen)*
Die Auslese sieht nur Betragsquadrate. Ein diagonales Phasengatter (Rz) ändert
Betragsquadrate nicht — für die **letzte** Lage ist es damit beweisbar wirkungslos
und streichbar. Ob das für *alle* Lagen gilt, ist **nicht** klar: dazwischen
liegen weitere Drehungen, und Phasen wirken über Interferenz. **Vorgehen:
numerisch prüfen, ob die Ausgabe identisch bleibt. Bleibt sie es, fällt Faktor 4
ohne Näherung weg. Bleibt sie es nicht, gilt die Regel nur für die letzte Lage.**
Nicht raten — messen.

**R5 — Konstantenfaltung bei eingefrorenen Winkeln.** *(hängt an Q-018)*
Sind die Winkel fest, entfällt der Autograd-Graph, entfallen die
Zwischenspeicher für den Rückwärtslauf, und Sinus/Kosinus werden einmal
vorberechnet. Der Befund vom 10. August (eingefroren ist gleich gut auf MNIST,
besser auf Fashion-MNIST) macht das zum wahrscheinlich wichtigsten Fall — und
er ist zugleich der **einfachste**, weil kein Rückwärtslauf existiert.

**R6 — Auslese in einem Durchlauf.** *(sicher, im Code nachgezählt)*
`butterfly_layer.py:180-182` berechnet die n Erwartungswerte in einer Schleife
über q, jeder mit eigenem `reshape` und zwei Summen — also **n vollständige
Durchläufe** über den Wahrscheinlichkeitsvektor. Alle n Werte lassen sich in
**einem** Durchlauf akkumulieren: pro Amplitude einmal das Betragsquadrat
bilden und es auf n Akkumulatoren mit dem Vorzeichen von Bit q addieren.

### Gegen den vorhandenen Code nachgezählt

Der Entwurf stand zuerst; danach habe ich ihn gegen `butterfly_layer.py`
geprüft. Drei Dinge sind dabei bestätigt oder schärfer geworden:

**Der CNOT-Ring ist bereits als Permutation implementiert** —
`cnot_permutation`, mit dem Kommentar *„klassisch kostenlos, kein
Gatterprodukt"*. Die Voraussetzung von R1 und R2 gilt also nicht nur theoretisch,
sie steht schon da. **Aber:** `forward` wendet in Zeile 175-176 die n
Permutationen **einzeln nacheinander** an, jede als vollständiger `gather` über
den Zustandsvektor. R1 ist damit ein realer, bisher ungehobener Gewinn — bei
8 Qubits und 5 Lagen sind das **40 Umsortierungsdurchläufe, wo 5 genügen.**

**Durchläufe über den Zustand, gezählt für n=8, L=5:**

| | Durchläufe |
|---|---|
| Gatteranwendungen (8 je Lage × 5) | 40 |
| Permutations-Gathers (8 je Lage × 5) | 40 |
| Betragsquadrate | 1 |
| Auslese (n einzelne Reshapes + Summen) | 8 |
| **Summe heute** | **≈ 89** |
| **fusioniert, angestrebt** | **1–2** |

Das deckt sich mit der Tagebuchdiagnose *„60 sequentielle Kleinoperationen"* und
liegt sogar darüber.

**R5 ist größer als oben beschrieben.** `_gate` wird **innerhalb** der
Vorwärtsschleife aufgerufen (Zeile 174) und berechnet dort `cos`, `sin` und zwei
komplexe `exp` — bei **jedem einzelnen Vorwärtslauf** neu. Bei eingefrorenen
Winkeln ist das vollständig verschenkt: L·n transzendente Funktionsauswertungen
pro Aufruf für Werte, die sich nie ändern.

**Und R4 ist bestätigt heikel.** `_gate` baut `Rz(b)·Ry(a)` als echte komplexe
Matrix (Zeile 157-159), die Phasen sind also real vorhanden. Die Vermutung „das
lässt sich reell rechnen" bleibt zu prüfen und darf nicht angenommen werden.

Regeln R1–R3 und R6 sind Umsortierung und Fusion, also **verhaltenserhaltend per
Konstruktion**. R4 und R5 sind an Bedingungen geknüpft, die geprüft werden
müssen, bevor sie angewandt werden.

---

## 5. Der erste Kernel — T-001

Bewusst der kleinste Fall, der etwas beweist.

**Was:** Vorwärtsrichtung, geteilte Winkel, eingefroren, 6–15 Qubits, CPU.
Vorwärts, weil dort der 1,99×-Vorteil sitzt. Eingefroren, weil kein
Rückwärtslauf gebraucht wird und Q-014 nahelegt, dass das der relevante Fall ist.

**Wie:** Prototyp in numba (schnelle Iteration), erst bei Bedarf C++ über
`torch.utils.cpp_extension`. Keine neue schwere Abhängigkeit, kein GPU-Pfad
in Stufe eins.

**Verifikation zuerst, Messung danach.** Bitweise gegen
`quantum_sim/butterfly_layer.py`, dieselben 30 Konfigurationen, mit denen dieser
gegen den Simulator geprüft wurde. Schranke 10⁻⁶ (erreicht wurden dort
7,15·10⁻⁷). **Ohne bestandene Verifikation wird keine Zeit gemessen** — beim
ersten Butterfly-Anlauf ist genau diese Prüfung durchgefallen (Ry·Rz statt
Rz·Ry), und ohne sie wäre jede folgende Messung wertlos gewesen.

**Messmethodik wie am 5. August**, unverändert übernommen: sieben unabhängige
Messreihen, innerhalb jeder Reihe abwechselnd gemessen, Faktor je Reihe
**gepaart** gebildet statt als Verhältnis zweier Mediane. Kein anderer schwerer
Lauf gleichzeitig.

**Vorregistrierung T-001:**

> **PRIMÄR:** Der fusionierte Kernel verschiebt den Übergangspunkt gegenüber der
> jetzigen Umsetzung von 16.384 auf **≤ 4.096 Dimensionen**.
> **Kontrolle:** die bestehende `ButterflyQuantumLayer`, gleiche Hardware,
> gleiche Methodik.
> **Auflösungsprüfung:** Bei d=16.384 muss der neue Kernel mindestens so schnell
> sein wie der alte. Ist er dort langsamer, misst der Aufbau etwas anderes als
> Fusion, und kein Ergebnis ist interpretierbar.
> **T1:** Übergang ≤ 4.096 → die Lücke war handwerklich, die Architektur ist im
> Bereich echter FFN-Breiten einsetzbar.
> **T0:** Übergang > 8.192 → die Fusion trägt nicht weit genug; die Lücke ist
> näher an fundamental als an handwerklich.
> **TU:** dazwischen → erreichten Übergangspunkt berichten, kein Urteil.
>
> Alle vier Ausgänge sind vor dem Lauf kodiert, einschließlich „der neue Kernel
> ist langsamer" — die Lehre aus Q-015 und Q-014.

---

## 6. Und erst danach die Darstellung

**Grundsatz: die Zwischendarstellung wird aus einem laufenden Kernel
herausgezogen, nicht vorher entworfen.** Zuerst T-001 von Hand fusionieren.
Dann T-002 (eine zweite Variante) von Hand. Dann das Gemeinsame herausziehen —
das ergibt eine Darstellung, die zu einem verstandenen Problem passt, statt zu
einem vermuteten.

Das ist auch der Grund, warum hier keine Syntax steht. Es gibt keinen Parser,
keine Schlüsselwörter, keine Fehlermeldungen. Es gibt Bausteine, Regeln und
einen Codegenerator. **Eine Oberfläche lässt sich später ohne Neubau darauf
setzen; umgekehrt ginge es nicht.**

**Zukunftstauglichkeit, vorher als Kriterium festgelegt:**

> Die Darstellung trägt, wenn die **zweite und dritte** Architekturvariante
> zusammen **weniger als 20 %** des Aufwands der ersten kosten — gemessen in
> Zeilen und in Zeit bis zum verifizierten Kernel.
> Variante 1: geteilter Butterfly-Block. Variante 2: Re-Uploading (Q-017).
> Variante 3: die Aufmerksamkeitsprojektionen aus Q-013.
> Wird das verfehlt, ist die Darstellung nur ein umständlicher Kernel und wird
> verworfen.

---

## 7. Roadmap

Heute ist der 10. August, Anmeldung am 30. November — **15 Wochen.** Die Phasen
haben Ausstiegskriterien, damit Werkzeugbau die Messungen nicht verdrängt.

**Zielentscheidung vom 10. August:** Am Ende der 15 Wochen sollen ein
**Chatbot** und eine **zweite KI-Art** stehen. Die Roadmap ist darauf
umgestellt; was dafür entfällt, steht unten.

### Der Konflikt, der die Planung bestimmt

Die Struktur **gewinnt** ab 16.384 Dimensionen, nach Fusion vielleicht ab 4.096.
Ein auf 6 Kernen ohne GPU **trainierbares** Sprachmodell liegt bei d_model
256–1.024; Q-013 sitzt bei etwa 512. **Diese Bereiche überlappen auf dieser
Hardware nicht.**

Daraus folgt die Form der Aussage, und sie ist zusammengesetzt:

> Bei Breite W kostet die Struktur X an Qualität — gemessen. Ab Breite Z spart
> sie Y an Zeit und Speicher — gemessen. Der Chatbot zeigt, dass die Qualität
> bei W trägt.

Was **nicht** behauptet werden darf: „der Chatbot läuft dank der Quantenstruktur
schneller". Dafür müsste er in einer Breite laufen, die dieser Rechner nicht
trainiert. Der Chatbot ist **Demonstration der Qualität**, nicht Beleg der
Effizienz. Diese Trennung gehört ausdrücklich in die Anmeldung.

### Phase 0 — Zwei Zahlen, die alles danach bestimmen (Woche 1)
**Q-018:** der Frozen-Befund sauber gemessen, mit **eingefrorenem Encoder**,
Gitter bis 1,0, Aufgabe mit mehr als sieben nötigen Richtungen. Entscheidet, ob
der Kernel zuerst den eingefrorenen Fall (einfach, kein Rückwärtslauf) oder den
trainierbaren (aufwendig) bedienen muss.
**T-000 Durchsatzkalibrierung:** Wie viele Token je Sekunde schafft dieser
Rechner, und wie groß darf das Sprachmodell danach sein? Nicht schätzen —
messen, wie bei der isochronen Kalibrierung für Q-009.

### Phase 1 — Fundament — ✅ ABGESCHLOSSEN 10. August, Ausstieg gegriffen
T-001 gemessen. Verifikation bestanden (4,0·10⁻⁷, Gradienten identisch), dann
sieben Reihen über fünf Breiten.

**Ergebnis: T0.** Der Übergangspunkt bleibt bei 16.384. Der Umbau bringt einen
über alle Breiten stabilen Gewinn von **1,44–1,87×** (Median ~1,6), aber das
verschiebt den Schnittpunkt nur um **~0,8 Oktaven** — weniger als die Auflösung
des Messgitters. Für das Ziel 4.096 wären zwei Oktaven nötig gewesen, also
**Faktor ~3,1 statt 1,6**.

**Das Ausstiegskriterium greift. Phase 2 beginnt nicht.**

Zwei Erkenntnisse, die bleiben:

- **`butterfly_fast.py` ist ein kostenloser Gewinn** und ersetzt
  `ButterflyQuantumLayer` überall: bitgleiche Funktion, identische Gradienten,
  1,6× schneller. Bei d=16.384 steigt der Vorsprung gegen dicht von 1,59× auf
  **2,31×** — der Übergangspunkt bewegt sich nicht, der Vorsprung dahinter
  wächst um die Hälfte.
- **Der Übergang wird von der Cache-Kante der dichten Schicht erzeugt**, nicht
  von der Effizienz der Butterfly-Struktur: Von 8.192 auf 16.384 springt die
  dichte Schicht um Faktor **9,1** (statt 4), weil ihre Gewichtsmatrix mit
  1,07 GB aus dem Cache fällt. Die Butterfly-Kurve ist glatt (2,27× je
  Verdopplung). An einer Kante lässt sich ein Schnittpunkt von der anderen
  Seite kaum verschieben.

**Was NICHT geprüft wurde, sauber getrennt:** Dieser Lauf hat nur den Teil der
Fusion getestet, der in reinem PyTorch ausdrückbar ist — rund 89 Durchläufe auf
rund 46. Die eigentliche Fusion (alle Stufen in einem Durchlauf, Zustand im
Cache, 89 → 1–2) braucht einen kompilierten Kernel und war nie Gegenstand
dieser Messung; das stand vor dem Lauf im Kopf von `butterfly_fast.py`.
`numba`, `triton` und `cython` fehlen in der Umgebung, nur `torch.compile` wäre
da.

**Der kompilierte Kernel ist damit eine eigene, offene Hypothese und braucht
eine eigene Vorregistrierung — keine Umdeutung dieser hier.** Die Zielmarke ist
bekannt: Faktor 3.

### Phase 2 — Darstellung — entfällt
Eine Zwischendarstellung wird aus zwei laufenden Kerneln herausgezogen. Da
Phase 1 am Ausstiegskriterium endet, gibt es keinen zweiten Kernel, aus dem
etwas herauszuziehen wäre. Der Entwurf in Abschnitt 3–4 bleibt gültig als
Beschreibung des Problems; gebaut wird er nicht.

### Phase 3 — Das Sprachmodell (Woche 2–6, vorgezogen)
Die vier Wochen aus Phase 1 und 2 fallen dem Sprachmodell zu.

**T-000 hat die Zielgröße gemessen:** d_model **256**, rund **5,3 Millionen
Parameter**, **282 Millionen Token in 24 Stunden** (53 Token je Parameter, also
deutlich über der Chinchilla-Faustregel von 20). Breite 512 bräuchte 96 Stunden
für 28. Speicher ist bei 256 unkritisch (81 MB mit Adam).

Dichte Referenz und Butterfly-Variante auf dieser Breite, kleiner Wortschatz,
einfacher Text (TinyStories-Klasse). Ergebnis ist **der Chatbot** plus die
Qualitätskurve über die Breite.

**Die Einschränkung ist jetzt beziffert:** trainierbar ist 256, die Struktur
gewinnt ab 16.384 — **Faktor 64**, und T-001 hat daran nichts geändert. Der
Chatbot demonstriert die **Qualität** der Struktur bei trainierbarer Breite; die
**Effizienz** wird getrennt bei ihrer eigenen Breite berichtet. Diese Trennung
ist keine Vorsichtsformulierung, sondern eine Notwendigkeit mit Zahl.

### Phase 4 — Zweite KI-Art: Zeitreihen und Signale (Woche 10–11)
Gewählt aus einem inhaltlichen Grund: Die Butterfly-Struktur **ist** die
schnelle Fouriertransformation. Für periodische Signale ist das die natürliche
Basis — dort sollte die Architektur nicht nur gleich gut, sondern **besser**
sein. Es ist die einzige Anwendung der Ideenliste mit einem erwarteten
*inhaltlichen* Vorteil statt nur einem Effizienzvorteil, und sie ist billig.
**Q-011 wird hier mitgenommen** (Tiefen 1/3/8/16, kleine Modelle, Lauf ist
gebaut) — sonst fiele der beste Kandidat für einen bestandenen Abgrenzungstest
ersatzlos weg.

### Phase 5 — Schreiben (Woche 12–15)
Kurzfassung, Anmeldung, Literaturkarten nachziehen, `B-003`-Doppelvergabe
klären, Demo aufräumen.

### Was dafür entfällt
**Q-010, Q-012, Q-016, Q-017** als eigenständige Läufe. Sie sind gebaut und
vorregistriert und bleiben es — aber 15 Wochen tragen nicht beides. Wenn Phase 1
am Ausstiegskriterium abbricht, rücken sie nach; das ist dessen Sinn.

---

## 8. Wie diese Roadmap erweitert wird

Damit „erweitern, wenn wir etwas entdecken" nicht zu „unterwegs umdefinieren"
wird — dieselbe Disziplin wie bei den Läufen:

1. Eine Änderung wird **hier eingetragen, bevor** die Arbeit beginnt, mit Datum
   und Anlass.
2. Steht schon eine Zahl aus dem betroffenen Schritt, wird die Änderung als
   *nachträglich* markiert und darf keine Vorregistrierung anfassen.
3. Ausstiegskriterien werden **nicht** aufgeweicht, wenn sie greifen. Ein
   greifendes Ausstiegskriterium ist ein Ergebnis, kein Rückschlag.
4. Neue Bausteine und Regeln kommen in Abschnitt 3 und 4 dazu, mit dem Grund
   und mit der Angabe, ob sie verhaltenserhaltend sind oder an eine geprüfte
   Bedingung geknüpft.

---

## 9. Was dieser Entwurf ausdrücklich nicht behauptet

- Er macht die Architektur **nicht besser**, nur bewertbar und einsetzbar.
- Er sagt **nicht**, dass die Lücke schließbar ist. Er sagt, dass sie nach der
  Ursachenanalyse aus Speicherverkehr besteht und dass das prüfbar ist.
- Die drei Größenordnungen sind der *Abstand zur Arithmetikgrenze*, nicht ein
  erwarteter Gewinn. Ein Kernel erreicht die Arithmetikgrenze nie.
- Ob eine normerhaltende, quadratisch auslesende Struktur einer dichten Schicht
  überlegen ist, entscheiden weiter die Q-Läufe. Der Stand vom 10. August ist,
  dass der **trainierbare** Teil davon nichts beiträgt.
