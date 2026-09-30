# Projekttagebuch — Quanteninspirierte ressourceneffiziente KI

Jugend forscht, Anmeldung bis 30. November.

Dieses Tagebuch hält fest, was ich gemacht habe, warum, und was dabei
herauskam — auch und gerade dann, wenn das Ergebnis nicht das war, was ich
erwartet hatte.

Hinweis zur Arbeitsweise: Ich habe bei der Umsetzung KI-Werkzeuge als
Hilfsmittel eingesetzt (Programmierung, Literaturrecherche, Diskussion von
Ideen). Forschungsfrage, Richtungsentscheidungen und Bewertung der Ergebnisse
liegen bei mir. Das gebe ich in der Anmeldung entsprechend an.

> Hauptfassung. Die Google-Docs-Version ist eine hochgeladene Kopie und kann
> hinter diesem Stand zurückliegen.

---

## Ausgangspunkt

Idee: Quantenphysik kennt Prinzipien, die klassische Rechner nicht haben —
Superposition, Born-Regel, Verschränkung. Wenn man diese nachahmt (nicht auf
Quantenhardware, sondern als Rechenverfahren auf normalen Computern), könnte
man vielleicht KI bauen, die bei gleicher Leistung weniger Ressourcen braucht.

Umsetzung: QIRE. Kern ist ein Born-Regel-Gate in einem
Mixture-of-Experts-Netz. Statt Softmax entscheidet |Amplitude|² / Summe,
welche Experten gerechnet werden. Dazu eine eigene Experimentierplattform mit
Multi-Seed-Läufen, automatischen Vergleichsberichten und Gatekeeping gegen
voreilige Schlüsse.

## 1. August — Pipeline prüfen

Umgebungscheck, Quick-Test, Ablation: alles läuft. Erster echter Vergleich
(80 Jobs, 10 Seeds, wine + breast_cancer) fehlerfrei.

Im Seed-Stability-Diagramm fiel eine QIRE-Variante mit Standardabweichung 0,21
auf — alle anderen lagen bei 0,01–0,05.

### Befund 1: Die "Instabilität" war ein Bug in meiner Auswertung

Vor dem Notieren erst die Rohdaten geprüft. Richtig so: Der Methodenname, unter
dem meine Auswertung gruppiert, enthielt nicht alle größenbestimmenden
Parameter. Zwei völlig verschiedene Architekturen (4 Experten/12 Neuronen vs.
2/8) trugen denselben Namen und wurden als "gleiches Modell, andere Seeds"
gepoolt. Ihre unterschiedliche Genauigkeit (0,685 vs. 0,981) erschien als
Streuung. Betroffen waren auch die Vergleichs- und Claim-Berichte.

Nach dem Fix lagen alle Modelle bei 0,014–0,022. Die Instabilität war weg.

**Lehre:** Ein auffälliges Ergebnis ist erst mal ein Anlass, die eigene
Messmethode zu prüfen, keine Entdeckung.

Denselben Fehlertyp fand ich später in allen Baseline-Modellen. Da MLP die
Referenz für alle Vergleiche ist, wäre das potenziell noch kritischer gewesen.

### Befund 2: Mir fehlte ein echter Signifikanztest

Meine Plattform verglich nur Mittelwerte und Standardabweichungen — kein
einziger statistischer Test. Damit ist prinzipiell nicht unterscheidbar, ob
ein Unterschied echt ist oder Rauschen.

Eingebaut: gepaarter Wilcoxon- und t-Test über exakt gematchte Seeds.
Sofortiger Effekt: StandardMoE galt vorher als "teilweise bestätigt" — der
Test zeigt einen statistisch echten Genauigkeitsverlust (p=0,03 / p=0,004).
Meine Heuristik hatte ein Fehlurteil produziert.

### Befund 3: Meine Effizienz-Kennzahl misst nicht, was ich dachte

Ich hatte Effizienz über `estimated_ops` gemessen. Gegen die tatsächlich
gemessene Zeit gehalten: QIRE auf wine spart 19 % Operationen — und ist
15-mal langsamer. Nicht unkorreliert, sondern gegenläufig.

Ursache: Die Sparsity ist echt, aber die Experten werden in einer
Python-Schleife abgearbeitet. Der Verwaltungsaufwand dominiert bei kleinen
Modellen jede eingesparte Operation. Systematisch vermessen: kein einziger
getesteter Fall war schneller, auch nicht bei 6,25 % aktiven Experten und
Batch 200.000.

Konsequenz: Flag `ops_time_conflict` eingebaut. Danach waren *alle* bisherigen
"teilweise bestätigten" Effizienzbefunde korrekt als "kein echter Speedup"
markiert.

**Das halte ich für den wichtigsten methodischen Punkt der Arbeit:** Fast jede
Veröffentlichung zu effizienter KI berichtet FLOPs. Ich habe dreimal
unabhängig gezeigt, dass diese Zahl bei meinen Größenordnungen in die Irre
führt.

### Den Engpass beheben

Python-Schleife durch sortierten Dispatch ersetzt. Verifikation zuerst:
Ausgaben identisch (Abweichung exakt 0,0), Gradienten korrekt.

Selbstkorrektur: Mein erster Benchmark lief mehrfädig, die Pipeline erzwingt
aber einen Thread. Der Vergleich war damit unfair. Nach Angleichung:

- großes Modell, 16 Experten, Batch 10.000: **58 % schneller**
- kleines Modell (meine echten Datensätze): weiterhin langsamer

"Prinzip funktioniert nicht" und "Prinzip funktioniert, aber nicht bei dieser
Datengröße" sind zwei verschiedene Aussagen.

### Breitere Validierung — meine Hauptthese kippt

Ausgeweitet auf 6 Datensätze und 20 Seeds, 480 Jobs. QIRE gegen MLP:

| Datensatz | p-Wert | Ergebnis |
|---|---|---|
| wine | 0,020 | signifikant schlechter |
| breast_cancer | 0,193 | kein Unterschied, keine Ersparnis |
| iris | 0,058 | grenzwertig |
| digits | 0,000002 | deutlich schlechter |
| synthetic_routing_challenge | 0,000002 | deutlich schlechter |
| synthetic_expert_specialization | 0,000002 | deutlich schlechter |

Der wine-Befund, bei 10 Seeds noch vielversprechend (p=0,176), kippt bei
20 Seeds zum signifikanten Nachteil. Hätte ich bei 10 aufgehört, wäre ein
falsches Ergebnis in die Anmeldung gegangen.

Besonders bitter: Auf den beiden synthetischen Datensätzen, die ich extra für
Experten-Routing gebaut hatte, schneidet QIRE am schlechtesten ab.

### Ablation — woran liegt es genau?

Vollständiges Gitter, 1992 Jobs, 0 Fehler.

Haupteffekte: Gate-Typ praktisch kein Unterschied (0,9679 / 0,9665 / 0,9662),
State-Mapper kein Unterschied, Temperatur kein Effekt. Dagegen: top_k (2 vs. 1
Experte: 0,9722 vs. 0,9616) und Anzahl Experten (2/4/8: 0,9708/0,9673/0,9625)
mit klaren Effekten.

Entscheidender Test — Born-Gate gegen Softmax-Gate bei sonst identischer
Architektur, 648 gematchte Paare: Differenz −0,0014, Wilcoxon p=0,247,
t-Test p=0,096. **Kein signifikanter Unterschied.**

Der quanteninspirierte Teil trägt selbst nichts bei. Was zählt, sind gewöhnliche
Kapazitätsparameter, die jede MoE-Architektur hat.

Im Nachhinein ist auch klar *warum*: Born-Regel und Softmax bilden beide einen
Vektor auf Wahrscheinlichkeiten ab, beide glatt, beide trainierbar, im
Wesentlichen dieselbe Funktionsklasse. Ich habe die Freiheitsgrade nicht
eingeschränkt — nur dieselbe Sache anders aufgeschrieben.

## 2. August — Echte Quantenhardware

Ich wollte verstehen, wie ein Quantencomputer wirklich funktioniert, nicht nur
eine Bibliothek benutzen. Also eigenen Simulator gebaut: n Qubits = 2^n
komplexe Amplituden, Gates als unitäre Matrizen, Messung als Zufallsziehung
mit |Amplitude|² — genau die Born-Regel, die ich vorher nur nachgeahmt hatte.

Zustand als Tensor mit einer Achse pro Qubit, Gates per Tensorkontraktion. So
muss nie eine volle 2^n × 2^n Matrix gebaut werden.

Validiert gegen 9 Schaltungen mit bekanntem Ergebnis — alle bestanden.
Grenzen notiert: kann selbst nie einen Quantenvorteil zeigen, praktikabel bis
etwa 20–22 Qubits.

Gegen Qiskit gegengeprüft: identische Wahrscheinlichkeiten. Dabei fiel ein
Konventionsunterschied auf (Qiskit zählt Qubit 0 von rechts, ich von links) —
übersehen hätte das falsche Vergleiche erzeugt.

Dann echte Hardware (IBM Quantum, 10 Freiminuten/Monat). Erster Versuch schlug
fehl: echte Hardware führt nur ihren nativen Gate-Satz aus, meine Gates mussten
erst transpiliert werden.

Bell-Zustand auf ibm_marrakesh (156 Qubits):

| Zustand | ideal | gemessen |
|---|---|---|
| 00 | 50,0 % | 50,1 % |
| 11 | 50,0 % | 48,0 % |
| 01 | 0 % | 1,1 % |
| 10 | 0 % | 0,9 % |

Etwa 2 % landen in theoretisch unmöglichen Zuständen — reales Quantenrauschen,
gemessen gegen die exakte Vorhersage meines Simulators.

Anschlussfrage: Wächst die Fehlerrate mit der Schaltungstiefe? Bei je 3
Wiederholungen: Richtung stimmte, p=0,65, nicht signifikant. Aufgestockt auf
je 10:

- Bell (2 Gates): 1,57 %
- GHZ (3 Gates): 2,06 %
- t-Test p=0,043, Mann-Whitney p=0,014 — **signifikant**

20 Jobs, 40 von 600 Sekunden Kontingent. Dasselbe Muster wie bei QIRE: n=3 sah
plausibel aus, war aber nicht belegbar. Erst n=10 hat es abgesichert.

## Tensor-Train — der erste echte Erfolg

Nach der QIRE-Widerlegung brauchte ich ein Kriterium für neue Ideen: Der Ansatz
muss dem Modell eine **echte strukturelle Einschränkung** auferlegen, nicht nur
eine andere Schreibweise sein.

Tensor-Train / Matrix-Product-States erfüllen das. Sie stammen aus der
Simulation von Quanten-Vielteilchensystemen. Statt m×n freier Werte eine Kette
kleiner Kern-Tensoren.

Implementiert, validiert (6 Prüfungen, u. a. gegen eine unabhängig
geschriebene Einstein-Summations-Referenz). Dann 10 Seeds, 3 Datensätze,
60 Jobs gegen dichte Baseline:

| Datensatz | Parameter | p-Wert | Ergebnis |
|---|---|---|---|
| wine | 63,4 % | 0,73 | kein Verlust |
| breast_cancer | 32,3 % | 0,16 | kein Verlust |
| digits | 27,6 % | 0,002 | signifikanter Verlust |

37–68 % weniger Parameter ohne nachweisbaren Genauigkeitsverlust auf zwei von
drei Datensätzen. Erster echter positiver Befund.

Vorbehalt sofort notiert: Meine Implementierung setzt die volle Matrix bei
jeder Vorhersage neu zusammen. Spart also Parameter, aber keine Rechenzeit —
gemessene Inferenzzeit ist sogar höher. Derselbe Fallstrick wie bei den Sparse
Experts, diesmal von vornherein beachtet.

## 2. August — Verschränkungsentropie als Vorhersage

Offene Frage: Warum funktioniert TT auf wine und breast_cancer, aber nicht auf
digits? Ich hatte den Rang überall auf 4 gesetzt — geraten, nicht begründet.

In der Quantenphysik gibt es dafür ein Kriterium: Ob sich ein Zustand kompakt
als MPS darstellen lässt, hängt von seiner Verschränkungsentropie ab. Das ist
der Grund, warum die Methode dort funktioniert.

Hypothese: Wenn Gewichtsmatrizen sich analog verhalten, hätte die Entropie das
Scheitern auf digits vorhersagen müssen.

Gemessen (Schmidt-Spektrum an jedem Bond, dazu zufällig initialisierte
Matrizen als Kontrolle):

| Datensatz | S (trainiert) | erfasst@4 | S (zufällig) | erfasst (zufällig) |
|---|---|---|---|---|
| wine | 2,143 | 0,662 | 2,165 | 0,643 |
| breast_cancer | 2,473 | 0,533 | 2,588 | 0,457 |
| digits | 2,649 | 0,425 | 2,693 | 0,382 |

Drei Beobachtungen, nur eine davon gut:

1. **Reihenfolge stimmt exakt** — ordnet die Datensätze genau wie der
   TT-Erfolg. Aber drei Punkte in der richtigen Reihenfolge können Zufall sein.
2. **Die Kontrolle widerlegt meine physikalische Erzählung.** Trainierte
   Matrizen haben kaum weniger Entropie als zufällige (2,143 vs. 2,165). Die
   Entropie hängt fast nur von den Dimensionen ab, nicht von gelernter
   Struktur. Ohne die Kontrolle hätte ich das falsch gedeutet.
3. **Die Theorie sagt die falsche Zahl vorher.** Das MPS-Kriterium verlangt
   Rang 8–13, tatsächlich reicht Rang 4 auf zwei von drei Datensätzen.

Punkt 3 führte zur eigentlichen Erkenntnis: Die Schmidt-Trunkierung beschreibt,
wie gut man eine *gegebene* Matrix annähert. Ich trainiere aber von Anfang an
in der eingeschränkten Form — das Netz muss die dichte Matrix gar nicht
reproduzieren, sondern findet innerhalb der Einschränkung eine andere Lösung.
Zwei verschiedene Regime.

## 2. August — Die beiden Regime direkt gegeneinander

Daraus eine echte, falsifizierbare Vorhersage gemacht und getestet.
TT-SVD implementiert und vorab abgesichert (verlustfreie Umformung, bei vollem
Rang exakte Rekonstruktion, monoton fallender Trunkierungsfehler).

3 Datensätze × 4 Ränge × 5 Seeds:
- **Regime A:** dicht trainieren, Matrix durch Rang-r-Näherung ersetzen, nicht
  nachtrainieren
- **Regime B:** TT-Modell mit Rang r von Anfang an trainieren

### Selbstkorrektur zur Teststatistik

Ich hatte als Test die *Korrelation* zwischen verworfenem Spektralgewicht und
Genauigkeitsverlust gewählt. Ergebnis: in beiden Regimen fast gleich
(r=0,688 vs. 0,675) — Hypothese scheinbar widerlegt.

Das war die falsche Statistik. Korrelation misst nur, ob die *Reihenfolge*
mitläuft — und natürlich wird es in beiden Regimen mit stärkerer Trunkierung
schlechter. Die eigentliche Frage ist, *wie stark*.

### Mit der richtigen Statistik klar bestätigt

| Kennzahl | Regime A | Regime B |
|---|---|---|
| mittlerer Genauigkeitsverlust | 0,0896 | 0,0073 |
| Steigung (Verlust je verworfenem Anteil) | +0,414 ± 0,138 | +0,035 ± 0,012 |

- Verhältnis: **11,9-fach**, Unterschied signifikant (z=2,73, p=0,006)
- Regime B besser in **12 von 12** Fällen (gepaarter Wilcoxon p=0,00049)
- Extremfall digits, Rang 2 (76,6 % verworfen): nachträglich 0,970 → **0,450**
  (Modell zerstört), trainiert 0,970 → **0,930** (weiter brauchbar)

**Interpretation:** Dasselbe Modell, derselbe Rang, dieselbe Einschränkung —
aber ob man sie nachträglich aufzwingt oder von Anfang an mittrainiert, macht
Faktor 12 im Genauigkeitsverlust aus. Das Netz umgeht die Einschränkung
tatsächlich. Damit ist auch geklärt, warum das MPS-Kriterium den Rangbedarf so
überschätzt hat: Es beschreibt Regime A, angewandt wurde es auf Regime B.

Nebenbefund: In 5 von 12 Fällen ist das komprimiert trainierte Modell *besser*
als das dichte — die TT-Struktur wirkt regularisierend.

## 2. August — Gilt das allgemein oder nur für Tensor-Train?

Der Faktor-12-Befund wäre deutlich mehr wert, wenn er nicht nur für
Tensor-Train gilt. Also dieselbe Frage für drei verschiedene Arten,
Struktur aufzuzwingen — jeweils auf dieselbe Schicht desselben Modells,
damit nur die *Art* der Beschränkung variiert:

- **tensor_train** — Gewichtsmatrix als Kernkette
- **low_rank** — Faktorisierung durch einen Flaschenhals
- **width** — schlicht weniger Neuronen

Vorab abgesichert: Bei voller Stärke reproduzieren alle drei
Post-hoc-Projektionen das Original exakt. Damit kann die Lücke nicht von einem
Implementierungsfehler kommen. Beim Neuronen-Pruning habe ich bewusst die
Standard-Magnitude-Heuristik genommen — ein besseres Auswahlkriterium würde
die Lücke verkleinern, der Test ist also konservativ.

27 Zellen, 5 Seeds:

| Beschränkung | post-hoc | mittrainiert | Verhältnis | besser in | p |
|---|---|---|---|---|---|
| width | 0,1119 | 0,0064 | **17,4×** | 9/9 | 0,0039 |
| tensor_train | 0,1184 | 0,0101 | **11,8×** | 9/9 | 0,0039 |
| low_rank | 0,0981 | 0,0352 | 2,8× | 7/9 | 0,164 (n.s.) |
| **gesamt** | **0,1095** | **0,0172** | **6,4×** | **25/27** | **0,000005** |

**Der Effekt ist allgemein.** Mein Tensor-Train-Befund ist ein Sonderfall
eines breiteren Prinzips über neuronale Netze.

### Die Ausnahme war der interessanteste Teil

Warum ist die Lücke ausgerechnet bei `low_rank` klein und nicht signifikant?

Die nachträgliche Projektion ist dort eine SVD-Trunkierung — und die ist nach
dem Eckart-Young-Theorem die *beweisbar optimale* Rang-r-Näherung einer
Matrix. Bei den anderen beiden ist sie es nicht: Neuronen-Pruning nutzt eine
reine Heuristik, TT-SVD arbeitet sequentiell-gierig.

Die Reihenfolge passt exakt invers:

| Güte der Projektion | Lücke |
|---|---|
| optimal (Eckart-Young) → low_rank | 2,8× |
| quasi-optimal → tensor_train | 11,8× |
| Heuristik → width | 17,4× |

Das verfeinert mein Prinzip: Die Lücke entsteht **nicht nur** daraus, dass
sich das Training an die Beschränkung anpasst — ein erheblicher Teil ist
schlicht, dass die üblichen nachträglichen Verfahren suboptimal sind.

Ehrlicher Vorbehalt: Bei `low_rank` war auch das mittrainierte Modell
schlechter (0,0352 vs. 0,0064/0,0101) — der harte Flaschenhals ist offenbar
schwerer zu trainieren. Die Lücke wird also von beiden Seiten
zusammengedrückt. Und drei Familien sind zu wenig für einen Beweis. Aber die
Übereinstimmung ist auffällig genug, um sie als Hypothese aufzuschreiben.

Zwei weitere Muster:
- Die Lücke wächst mit der Kompressionsstärke (Spearman −0,549, p=0,003)
- Sie hängt stark von der Aufgabenschwierigkeit ab: digits 0,224, wine 0,041,
  breast_cancer 0,012

### Was das für mein eigentliches Ziel bedeutet

Wer ein Modell ressourceneffizient machen will, sollte die Beschränkung von
Anfang an mittrainieren statt nachträglich zu komprimieren. Bei starker
Kompression ist das der Unterschied zwischen brauchbar und unbrauchbar
(digits, Rang 2: 0,930 gegenüber 0,450). Und wenn nachträglich komprimiert
werden muss, lohnt sich ein beweisbar optimales Verfahren gegenüber einer
Heuristik.

Das ist eine praktisch verwertbare Aussage, die aus meinen eigenen Messungen
folgt — und sie gilt unabhängig davon, ob die Beschränkung quanteninspiriert
ist oder nicht.

## 2. August — Regularisierungseffekt: widerlegt

In mehreren Zellen war das komprimierte Modell *besser* als das dichte. Wenn
das systematisch wäre, hätte ich statt "spart ohne zu schaden" sagen können
"spart und hilft" — deutlich stärker. Also nachgeprüft.

**Wichtig am Aufbau:** Die zu testenden Konfigurationen hatte ich aus den
alten Daten ausgewählt. Sie auf denselben Seeds nachzutesten wäre zirkulär
gewesen — genau das Rauschen, das sie auffällig gemacht hat, hätte den Effekt
wieder erzeugt. Deshalb frische Seeds 101–125, die kein früheres Experiment
berührt hat. Zusätzlich Trainings-Genauigkeit erhoben, um den vermuteten
Mechanismus (Überanpassung) direkt zu prüfen statt ihn anzunehmen.

### Frage 1: Repliziert der Vorteil? Nein.

| Datensatz | Konfiguration | Nutzen | Siege | p |
|---|---|---|---|---|
| wine | tensor_train_r8 | +0,0032 | 7/25 | 0,298 |
| wine | width_16 | **−0,0087** | 1/25 | **0,008** |
| breast_cancer | tensor_train_r8 | +0,0038 | 14/25 | 0,076 |
| breast_cancer | width_16 | +0,0023 | 8/25 | 0,377 |
| iris | tensor_train_r8 | −0,0036 | 4/25 | 0,857 |
| iris | width_16 | **−0,0062** | 2/25 | **0,039** |
| digits | tensor_train_r8 | **−0,0099** | 3/25 | **0,000** |
| digits | width_16 | **−0,0080** | 5/25 | **0,000** |

Von 8 Zellen sind 4 signifikant — und alle vier zeigen das komprimierte
Modell als **schlechter**. Keine einzige zeigt einen signifikanten Vorteil.
`wine`/`width_16` war in den alten Daten noch besser als dicht und ist jetzt
signifikant schlechter.

Die früheren "Siege" waren also Rauschen. Genau dafür war der Aufbau mit
frischen Seeds gedacht — er hat funktioniert.

### Frage 2: Ein Fehler in meiner eigenen Auswertung

Ich hatte als Mechanismus-Test die Korrelation zwischen Überanpassung und
Nutzen auf Seed-Ebene gerechnet: Spearman +0,170, p=0,016 — scheinbar
signifikant, scheinbar Bestätigung.

Beim Nachdenken über die Formeln fiel mir auf, dass das nicht sein kann:

    dense_overfit = dense_train − dense_test
    benefit       = model_test  − dense_test

Beide enthalten **denselben Term** −dense_test. Ein Seed, bei dem dense_test
zufällig niedrig ausfällt, punktet dadurch automatisch auf beiden Größen —
ganz ohne kausalen Zusammenhang.

Mit einer Simulation überprüft (vollständig unabhängige Modelle, wahrer
Zusammenhang = 0):

| Streuung von dense_train | Spearman (Nullverteilung) |
|---|---|
| 0,000 | +0,689 |
| 0,010 | +0,615 |
| 0,020 | +0,480 |

Die Nullverteilung liegt bei +0,5 bis +0,7. Mein gemessener Wert von +0,170
liegt **darunter** — also eher ein Argument *gegen* meine Hypothese als dafür.

Der ehrliche Test ist die Datensatz-Ebene, weil sich der gemeinsame Term
durch die Mittelung herausrechnet:

| Datensatz | Überanpassung | Nutzen |
|---|---|---|
| wine | 0,0209 | −0,0028 |
| breast_cancer | 0,0293 | +0,0031 |
| iris | 0,0251 | −0,0049 |
| digits | **0,0313** | **−0,0090** |

`digits` hat die höchste Überanpassung und den schlechtesten Nutzen — das
genaue Gegenteil dessen, was Regularisierung vorhersagt.

**Fazit:** Der Effekt existiert nicht. Die Strukturbeschränkung kostet
Genauigkeit, sie schenkt keine. Einen Warnhinweis habe ich direkt in den Code
geschrieben, damit ich diese Korrelation später nicht doch als Beleg lese.

Das ist innerhalb weniger Stunden der zweite Fall, in dem ich die falsche
Statistik gewählt hatte (nach der Korrelation statt Steigung beim
Regime-Vergleich). Beide Male ist es aufgefallen, weil ich mich gefragt habe,
*warum* eine Zahl so herauskommt, statt sie zu übernehmen. Das ist offenbar
der wichtigste Arbeitsschritt bei diesem Projekt.

## 2. August — Strukturiertes Pruning: endlich echte Zeitersparnis

Das Projekt ist bisher viermal an derselben Wand gelandet: Ersparnis in der
Theorie, nichts davon in der gemessenen Zeit. Strukturiertes Pruning ist der
erste Ansatz, bei dem es strukturell klappen *muss* — ganze Neuronen zu
entfernen macht die Matrizen wirklich kleiner, ohne Maskieren, ohne
Rekonstruktion, ohne zusätzlichen Verwaltungsaufwand.

Außerdem ein Test gegen mein eigenes Prinzip: Ich hatte gemessen, dass
nachträgliches Aufzwingen von Struktur 17,4× teurer ist. Aber das
Standard-Rezept in der Praxis ist nicht "prunen und fertig", sondern
"prunen und **kurz nachtrainieren**". Genau dieser Schritt fehlte mir.

### Teil A: Nachtrainieren schließt die Lücke

15 frische Seeds, hidden 32 → 16:

| | Verlust ggü. dicht |
|---|---|
| gepruned, ohne Nachtrainieren | 0,0690 |
| gepruned, mit Nachtrainieren (20 Epochen) | **0,0061** |
| schmal von Anfang an trainiert | 0,0048 |

- mit vs. ohne Nachtrainieren: p=0,00001
- mit Nachtrainieren vs. von-Anfang-an: **p=0,34 — kein nachweisbarer Unterschied**

**Ich muss mein Prinzip also einschränken.** Die 17,4-fache Lücke hatte ich
ohne Nachtrainieren gemessen. Ein Nachlauf von 20 Epochen — ein Viertel des
ursprünglichen Trainings — hebt sie praktisch auf. Neue Fassung:

> Struktur nachträglich aufzuzwingen kostet deutlich mehr Genauigkeit als sie
> mitzutrainieren — es sei denn, man trainiert danach kurz nach. Dann
> verschwindet der Unterschied.

Das ist keine Widerlegung, sondern eine Präzisierung — und sie macht das
Ergebnis praktisch brauchbarer, weil sie sagt, was man konkret tun soll.

Nebenbei: Wenn ich die Zielgröße vorher kenne, ist schmal-von-Anfang-an
trotzdem günstiger (80 Epochen klein statt 80 groß plus 20 Nachlauf). Der
Vorteil von prunen+nachtrainieren liegt darin, aus einem einzigen Training
mehrere Zielgrößen ableiten zu können.

Auf `digits` bleibt auch mit Nachtrainieren ein signifikanter Rest
(−0,0148, p=0,0001). Bei der schwersten Aufgabe reicht der kurze Nachlauf nicht.

### Teil B: Zum ersten Mal kommt die Ersparnis in der Zeit an

Neuronenzahl halbiert, einfädig gemessen. Ideal wäre 2,00×:

| Neuronen | Batch 32 | Batch 1000 | Batch 10000 |
|---|---|---|---|
| 32 | 1,07× | 1,66× | 1,14× |
| 128 | 1,29× | 2,81×* | 1,74× |
| 512 | 1,67× | 1,89× | 1,78× |
| 2048 | 1,73× | **2,13×** | **2,06×** |

\*Über dem theoretischen Ideal — das deutet auf einen Cache-Effekt hin, nicht
auf echte Recheneinsparung. Nicht überinterpretieren.

**Das ist der erste Ansatz im ganzen Projekt, bei dem eine Ressourcenersparnis
tatsächlich in gemessener Zeit ankommt.** Bei 2048 Neuronen erreiche ich das
volle Ideal, ab 512 Neuronen 1,8–1,9×.

Bei meiner tatsächlichen Projektgröße (32 Neuronen) sind es nur 1,07–1,66×.
Dasselbe Overhead-Muster wie bei allem anderen — aber diesmal mit einer klar
bezifferten Grenze, ab der es funktioniert.

Damit ist die Frage, an der ich das ganze Projekt über hängengeblieben bin,
quantitativ beantwortet: **Ab etwa 512 Neuronen wird Ressourcenersparnis zu
echter Zeitersparnis.** Darunter frisst der Aufruf-Overhead sie auf.

## 2. August — Hochskaliert: MNIST, 512 und 2048 Neuronen

Mein Befund 3 sagt, dass Ersparnis erst ab etwa 512 Neuronen in echter Zeit
ankommt. Alle meine Experimente liefen aber bei 32 Neuronen — also unterhalb
der Schwelle, ab der meine eigene Frage überhaupt praktisch relevant wird.
Genau das würde eine Jury fragen. Also MNIST eingebunden (784 Merkmale,
70.000 echte Bilder) und die Kernbefunde oberhalb der Schwelle wiederholt.

### Unterwegs ein Skalierungsproblem gefunden

Beim Einbinden fiel mir auf, dass die Werte nach dem Laden bis 221 reichten,
obwohl ich auf 0–1 normiert hatte. Ursache: Die Pipeline standardisiert jedes
Merkmal durch seine eigene Standardabweichung. Bei Bildern sind viele
Randpixel fast konstant — MNISTs ruhigstes Pixel hat std 0,000119 und wird um
Faktor **8368** aufgeblasen.

Gemessen, welche Skalierung wirklich am besten ist (784×512-Netz, 3 Seeds):

| Skalierung | Genauigkeit | max \|x\| |
|---|---|---|
| **nur /255** | **0,9518** | 1,00 |
| global skaliert | 0,9471 | 2,24 |
| Varianz-Floor 1 % | 0,9391 | 83,67 |
| bisheriges Verhalten | 0,9311 | 221 |

Wichtig: Ich wollte das zuerst global reparieren. Vorher geprüft, ob andere
Datensätze betroffen sind — und das war die richtige Entscheidung: Auf
`breast_cancer` **kostet** ein Varianz-Floor 0,0198 Genauigkeit (p=0,002),
weil die Merkmale mit kleiner Varianz dort echtes Signal tragen. Auf `digits`
ändert er nichts.

Ein globaler Fix hätte also bestehende Ergebnisse verschlechtert. Stattdessen
ist die Standardisierung jetzt pro Datensatz abschaltbar. Mit einem
Regressionstest abgesichert, dass wine, breast_cancer, digits, iris und
synthetic_large bit-genau dieselben Daten liefern wie vorher — alle früheren
Ergebnisse bleiben gültig.

### Die Ergebnisse

| Neuronen | Arm | Genauigkeit | Parameter | Zeit |
|---|---|---|---|---|
| 512 | dicht | 0,9639 | 407.050 | 1,00× |
| 512 | **Tensor-Train r16** | 0,9558 | **26.506 (6,5 %)** | 0,85× |
| 512 | gepruned+Nachlauf | 0,9652 | 203.530 (50 %) | **1,89×** |
| 2048 | dicht | 0,9681 | 1.628.170 | 1,00× |
| 2048 | **Tensor-Train r16** | 0,9544 | **56.442 (3,5 %)** | 0,82× |
| 2048 | gepruned+Nachlauf | 0,9648 | 814.090 (50 %) | **1,95×** |

**Tensor-Train wird mit der Größe dramatisch besser.** Bei meinen kleinen
Modellen sparte es 37–68 % der Parameter. Bei 2048 Neuronen sind es **96,5 %**
— eine 28-fache Kompression für 1,4 Prozentpunkte Genauigkeit. Genau das sagt
die Tensor-Netzwerk-Theorie voraus, und ich habe es jetzt an echten Bilddaten
belegt statt nur an Spielzeugdatensätzen.

**Aber die Zeit kommt weiterhin nicht mit** — 0,74 bis 0,85×, und der
Rückstand wächst sogar mit der Modellgröße. Ursache ist dieselbe wie beim
Sparse-Expert-Problem ganz am Anfang: Die volle Matrix wird bei jedem
Vorwärtslauf neu zusammengesetzt, und dieser Aufwand wächst mit.

**Strukturiertes Pruning liefert dagegen genau das:** 1,89× und 1,95× echte
Beschleunigung, bei 512 sogar minimal besser als dicht.

### Was das für meine Kernaussage bedeutet

Ich habe jetzt eine saubere zweiteilige Antwort:

- Der quanteninspirierte Ansatz (Tensor-Train) liefert **herausragende
  Speicherersparnis, die mit der Modellgröße besser wird** — 28-fache
  Kompression bei realistischer Größe.
- Echte **Zeitersparnis** liefert bisher nur ein klassisches Verfahren
  (strukturiertes Pruning).

Die beiden konkurrieren nicht, sie ergänzen sich: Tensor-Train für Geräte mit
wenig Speicher, Pruning für Anwendungen, wo es auf Geschwindigkeit ankommt.
Das ist eine praktisch verwertbare Aussage.

**Offene Frage, die ich nicht erklären kann:** Rang 32 ist auf beiden Achsen
schlechter als Rang 16 — mehr Parameter *und* weniger Genauigkeit (0,9430 vs.
0,9544). Mehr Kapazität sollte nicht schaden. Vermutlich ein
Optimierungsproblem bei höherem Rang, aber das ist eine Vermutung. Kommt so in
die Arbeit, nicht wegerklärt.

## 2. August — Streaming-Kontraktion: Tensor-Train wird endlich schneller

Der Punkt, der seit Wochen auf meiner Liste stand. Tensor-Train spart
massiv Parameter, war aber immer langsamer, weil meine Implementierung die
volle Matrix bei jedem Vorwärtslauf neu zusammensetzt. Die Alternative:
die Kerne direkt gegen die Eingabe kontrahieren, ohne die Matrix je zu bilden.

Zuerst implementiert und verifiziert: numerisch identisch zur Rekonstruktion
(Abweichung 1e-15 über sechs Konfigurationen), Gradienten fließen.

### Meine eigene Vorhersage war falsch

Vorher hatte ich die Kosten überschlagen: Streaming trägt gleichzeitig die noch
offenen Eingangs- und die schon gesammelten Ausgangsindizes, also etwa
d·r²·q^(1−d) im Verhältnis zu dicht. Daraus folgte: viele Kerne und kleiner Rang
gewinnen.

Gemessen bei 784×512:

| Kerne | Rang | meine Vorhersage | gemessen |
|---|---|---|---|
| 3 | 2 | 0,16× | 2,60× |
| 5 | 2 | 0,10× | 8,83× |
| 6 | 2 | 0,09× | 9,15× |

Die Korrelation zwischen Vorhersage und Messung war ρ=0,42 mit p=0,10 — nicht
einmal signifikant.

Der Grund: Die Zahl der sequentiellen Rechenschritte entspricht der Kernanzahl.
Mehr Kerne senken zwar die Rechenoperationen, erzeugen aber mehr kleine,
schlecht ausgelastete Einzeloperationen.

**Das ist exakt der Fehler, den ich in diesem Projekt schon dreimal
nachgewiesen habe** — Rechenoperationen sagen die tatsächliche Zeit nicht
vorher. Diesmal ist er mir im eigenen Kostenmodell unterlaufen. Ich habe die
Lektion offenbar noch nicht so verinnerlicht, wie ich dachte.

### Der Crossover liegt bei der Modellgröße

Quadratische Schichten, 4 Kerne, Rang 4:

| Schicht | dicht | Rekonstruktion | Streaming | Verhältnis |
|---|---|---|---|---|
| 512² | 1,47 ms | 3,28 ms | 11,35 ms | 7,71× langsamer |
| 1024² | 6,55 ms | 13,33 ms | 23,28 ms | 3,55× langsamer |
| 2048² | 27,36 ms | 53,20 ms | 53,57 ms | 1,96× langsamer |
| **4096²** | 111,38 ms | 200,73 ms | **72,35 ms** | **1,54× schneller** |

**Zum ersten Mal im ganzen Projekt ist Tensor-Train schneller als ein dichtes
Netz** — bei 4096×4096 um Faktor 1,54, und das mit etwa **0,04 % der
Parameter** (2.560 statt 16,7 Millionen).

Eine strukturelle Einsicht dabei: Die Rekonstruktions-Variante kann prinzipiell
**nie** gewinnen, weil sie die dichte Multiplikation ausführt *und* vorher noch
die Matrix baut. Die Messwerte bestätigen das (durchweg etwa doppelt so lange
wie dicht). Nur Streaming vermeidet die große Multiplikation überhaupt.

**Was noch fehlt:** Die Genauigkeit bei 4096 Neuronen mit Rang 4 habe ich nicht
getestet — bisher nur die Geschwindigkeit. Ob so wenig Kapazität dort
ausreicht, ist offen. Das muss ich prüfen, bevor ich irgendetwas behaupte.

### Das Gesamtbild hat jetzt bezifferte Schwellen

| Modellgröße | was Zeit spart |
|---|---|
| unter ~512 Neuronen | nichts |
| ~512 bis ~3000 | strukturiertes Pruning (1,9×) |
| ab ~4096 | zusätzlich Tensor-Train-Streaming (1,54×) bei 0,04 % der Parameter |

## 2. August — Genauigkeit an der Gewinn-Konfiguration

Der Speedup war reine Zeitmessung. Ob eine Rang-4-Kette bei Breite 4096
überhaupt genug Kapazität hat, um zu lernen, war offen — ohne diese Antwort
ist ein Speedup wertlos.

### Fast einen Fehler gemacht

Ich wollte den 4096×4096-Speedup mit einer MNIST-Genauigkeit bei 784→4096
kombinieren. Vorher aber die Geschwindigkeit an der MNIST-Form nachgemessen:

| Schicht | Streaming/dicht |
|---|---|
| 784→4096 (MNIST-Eingang) | **1,89× langsamer** |
| 4096→4096 (quadratisch) | **1,39× schneller** |

Der Vorteil hängt also an der *Form*, nicht nur an der Größe. Hätte ich beide
Zahlen kombiniert, wäre daraus eine Behauptung geworden, die für keine
existierende Konfiguration gilt.

Also ein Netz gebaut, dessen komprimierte Schicht genau die vermessene Form
hat: 784 → 4096 → 4096 → 10, Tensor-Train auf der quadratischen
Zwischenschicht. Genauigkeit und Geschwindigkeit stammen jetzt aus derselben
Konfiguration.

### Ergebnis

| Arm | Genauigkeit | Std | Parameter Zwischenschicht |
|---|---|---|---|
| dicht | 0,9202 | 0,0019 | 16.781.312 |
| **TT Rang 4** | 0,9267 | 0,0063 | **6.656 (0,04 %)** |
| TT Rang 16 | 0,9266 | 0,0050 | 38.912 (0,23 %) |

**Rang 4 reicht: 2521-fache Kompression ohne messbaren Genauigkeitsverlust.**

**Was ich ausdrücklich nicht behaupte:** dass Tensor-Train *besser* ist als
dicht. Die Differenz von +0,0065 liegt bei einer Streuung von 0,0063 über nur
3 Seeds — etwa eine Standardabweichung. Genau dieses Muster hatte ich beim
vermeintlichen Regularisierungseffekt, und dort ist es auf frischen Seeds
komplett zusammengebrochen. Ich schreibe deshalb "kein messbarer Verlust", und
wenn ich mehr behaupten will, brauche ich mindestens 15 frische Seeds und
einen gepaarten Test.

**Messstreuung:** Für dieselbe Konfiguration hatte ich vorher 2,08× gemessen,
jetzt 1,39×. Das gehört genannt — der ehrliche Bereich ist 1,4× bis 2,1×,
nicht der schönere Einzelwert.

**Nur Rang 4 gewinnt bei der Zeit.** Rang 16 ist trotz gleicher Genauigkeit
1,85× langsamer. Für diese Konfiguration ist Rang 4 eindeutig die richtige
Wahl.

### Damit gewinnt der quanteninspirierte Ansatz erstmals auf beiden Achsen

Speicher **und** Zeit gleichzeitig — unter klar benannten Bedingungen:
quadratische Schicht ab etwa 4096 Neuronen, Rang 4, Batch-Inferenz.

Weitere Vorbehalte, die in die Arbeit müssen: Das Trainingsbudget war klein
(6000 Beispiele, 10 Epochen), die absolute Genauigkeit von 0,92 liegt deutlich
unter dem, was auf MNIST erreichbar ist. Für den Vergleich der Arme
untereinander spielt das keine Rolle, aber die absolute Zahl darf ich nicht als
MNIST-Ergebnis zitieren.

## 2. August — Konfigurationsraum richtig vermessen, Kernaussage abgesichert

### Zwei Fehler in meinem eigenen Vorgehen

**Erstens:** Meine (Kerne, Rang)-Landkarte hatte ich bei 784×512 erstellt — also
genau dort, wo *keine* Konfiguration gewinnt. Daraus zu schließen, 4 Kerne mit
Rang 4 sei optimal, war nicht haltbar. Neu vermessen bei 4096×4096:

| Kerne | Rang | Parameter | vs. dicht |
|---|---|---|---|
| **2** | **2** | 0,122 % | **5,44× schneller** |
| 3 | 2 | 0,037 % | 3,18× schneller |
| 4 | 4 (bisherige Wahl) | 0,040 % | 1,72× schneller |

14 von 20 Konfigurationen schlagen dicht. Meine bisherige Wahl war dreifach
suboptimal. Und wieder gilt das Gegenteil meines FLOP-Modells: *weniger* Kerne
sind schneller. Zum dritten Mal sagt die Operationszählung das Falsche vorher.

**Zweitens:** Alles bei Batch 256 gemessen. Bei Batch 1 bis 8 — also
Echtzeit-Inferenz auf schwacher Hardware, dem eigentlichen
Ressourcen-Anwendungsfall — sieht es völlig anders aus:

| Batch | Faktor (2 Kerne, Rang 2) |
|---|---|
| 1 | **15,8×** |
| 8 | **17,8×** |
| 256 | 5,0× |

Statt der bisher berichteten 1,4–2,1× sind es bis zu **17,8×**. Die
Rekonstruktionsvariante ist bei Batch 1 dagegen 25,9× *langsamer* als dicht —
sie baut für ein einziges Beispiel die komplette 16,7-Millionen-Matrix auf.

### Genauigkeit mit 10 frischen Seeds abgesichert

Die Konfigurationen hatte ich nach Geschwindigkeit ausgewählt, deshalb frische
Seeds 501–510 und gepaarter Test:

| Arm | Genauigkeit | Std | vs. dicht | Anteil | p |
|---|---|---|---|---|---|
| dicht | 0,9238 | 0,0092 | — | 1,00000 | — |
| **2 Kerne, Rang 2** | 0,9205 | 0,0164 | −0,0033 | 0,00122 | 0,557 |
| 3 Kerne, Rang 2 | 0,9277 | 0,0079 | +0,0039 | 0,00037 | 0,432 |
| 4 Kerne, Rang 4 | 0,9291 | 0,0053 | +0,0053 | 0,00040 | 0,232 |

**Keine Differenz ist signifikant.** Genau das wollte ich zeigen: Kompression
ohne messbaren Genauigkeitsverlust.

**Meine Vorsicht von vorhin war berechtigt.** Der bei 3 Seeds beobachtete
Vorteil von +0,0065 ist bei 10 Seeds auf +0,0053 mit p=0,232 geschrumpft — also
Rauschen, wie schon beim Regularisierungseffekt. Hätte ich damals "besser als
dicht" geschrieben, wäre es jetzt gekippt. Zum zweiten Mal hat sich dieselbe
Zurückhaltung ausgezahlt.

Streuungsprüfung: Die schnellste Variante (2/2) schwankt stärker über Seeds
als dicht (F=3,18, p=0,100) — knapp nicht signifikant, aber ein Hinweis auf
geringere Stabilität. Die Varianten 3/2 und 4/4 sind eher stabiler als dicht.

### Damit steht meine Hauptaussage

> Eine quanteninspirierte Tensor-Train-Schicht ersetzt eine dichte
> 4096×4096-Schicht mit **0,12 % der Parameter** (819-fache Kompression), ist
> bei Batch 8 **bis zu 17,8× schneller** und zeigt **keinen statistisch
> messbaren Genauigkeitsverlust** (p=0,56, n=10).

Praktisch würde ich eher 3 Kerne/Rang 2 nehmen: nur 0,037 % der Parameter
(2731-fach), 3,18× schneller, und über Seeds stabiler als dicht. Die
Kleinbatch-Messung dafür fehlt allerdings noch — die 17,8× gelten bisher nur
für 2/2.

## 2. August — Kleinbatch-Messung, und ein Wert den ich verwerfe

Die fehlende Messung nachgeholt: alle drei Kandidaten über kleine Batches.

| Batch | dicht | 2 Kerne/R2 | 3 Kerne/R2 |
|---|---|---|---|
| 1 | 3,47 ms | 15,3× | 13,9× |
| 2 | 5,75 ms | 18,7× | 14,7× |
| 4 | 14,38 ms | (30,1×) | (21,9×) |
| 8 | 14,27 ms | 18,8× | 12,8× |
| 16 | 17,30 ms | 12,5× | 8,0× |

Bei Batch 4 kam zunächst ein Spitzenwert von 30× heraus. Das sah verdächtig
aus, weil die dichte Referenz dort fast so lange braucht wie bei Batch 8
(14,38 vs. 14,27 ms), obwohl Batch 2 nur 5,75 ms kostet.

Nachgemessen mit Median aus 5 Messreihen à 200 Wiederholungen — der Effekt ist
**reproduzierbar**, also kein Rauschen. Erklärung: BLAS wechselt ab einer
Mindestgröße die Kernel-Variante und rechnet Batch 4 faktisch wie Batch 8.
Das ist eine Ineffizienz der *dichten Referenz*, nicht eine Stärke von
Tensor-Train.

**Ich verwerfe den Wert deshalb**, obwohl er der schönste wäre. Belastbar sind
13–19× bei kleinen Batches und 3–5× bei Batch 256.

### Meine Empfehlung: 3 Kerne, Rang 2

| | Parameter | Kompression | klein-Batch | Stabilität |
|---|---|---|---|---|
| 2 Kerne / Rang 2 | 0,122 % | 819-fach | 15–19× | F=3,18 (schwankt) |
| **3 Kerne / Rang 2** | **0,037 %** | **2731-fach** | 13–15× | F=0,74 (stabiler als dicht) |

Die zusätzliche Geschwindigkeit von 2/2 kostet 3,3× mehr Parameter und mehr
Streuung über Seeds. Für den praktischen Einsatz ist das kein guter Tausch.

## 2. August — Die Baseline, die mir gefehlt hat

Beim systematischen Durchgehen meiner Methoden ist mir aufgefallen, dass eine
entscheidende Vergleichsgruppe fehlt: Ich habe Tensor-Train immer nur gegen
*dichte* Schichten verglichen. Damit lässt sich aber nicht trennen, ob
"Tensor-Netzwerke helfen" oder ob "irgendeine Low-Rank-Kompression hilft".

Falls eine simple Faktorisierung W ≈ U·V bei gleichem Parameterbudget dasselbe
leistet, wäre meine Tensor-Netzwerk-Struktur bloße Dekoration und mein
Hauptergebnis müsste "Low-Rank-Kompression funktioniert" heißen — weder neu
noch quanteninspiriert.

Also den Test gebaut, bewusst konservativ: Wo Budgets nicht exakt angleichbar
waren, bekommt Low-Rank das *größere*.

### Ergebnis: die Struktur trägt

| Verfahren | Genauigkeit | Parameter | b=256 |
|---|---|---|---|
| dicht | 0,9243 | 16.781.312 | 130,3 ms |
| **TT 3 Kerne/R2** | **0,9279** | **6.144** | 43,5 ms |
| TT 2 Kerne/R2 | 0,9241 | 20.480 | 25,9 ms |
| Low-Rank r=1 | 0,6827 | 12.288 | 0,4 ms |
| Low-Rank r=2 | 0,8848 | 20.480 | 1,0 ms |
| Low-Rank r=5 | 0,9205 | 45.056 | 1,2 ms |

Die drei entscheidenden Vergleiche:

| Vergleich | Differenz | p |
|---|---|---|
| TT (6.144) vs. Low-Rank (12.288) | **+0,2452** | **0,002** |
| TT (20.480) vs. Low-Rank (20.480) | **+0,0393** | **0,002** |
| TT (20.480) vs. Low-Rank (45.056) | +0,0036 | 0,557 |

**Bei exakt gleichem Budget ist Tensor-Train signifikant genauer.** Um meine
Genauigkeit zu erreichen, braucht Low-Rank 2,2-fach mehr Parameter. Und bei
meinem kleinsten Budget bricht Low-Rank komplett ein (0,68 statt 0,93).

### Warum mich das inhaltlich überzeugt

Der Vorteil kommt aus genau dem Merkmal, um das es bei Matrix-Product-States
geht: Low-Rank erzwingt **einen** globalen Flaschenhals, Tensor-Train verteilt
die Beschränkung auf eine **Kette von Bonds**. Damit lassen sich
Korrelationsstrukturen darstellen, die ein einzelner Rang-r-Schnitt nicht
erfassen kann.

Das ist der Punkt, an dem ich sagen kann: Es ist nicht "Low-Rank mit
Physik-Vokabular". Der Unterschied zwischen einem Bond und einer Bond-Kette
ist die Eigenschaft, für die MPS in der Quantenphysik entwickelt wurden — und
sie wirkt hier messbar.

### Was ich korrigieren muss

Auf der **Zeitachse** ist mein Verfahren unterlegen. Low-Rank ist bei gleichem
Budget 24-fach schneller (1,0 ms gegenüber 25,9 ms). Meine bisherige Aussage
"13–15× schneller als dicht" hatte die falsche Vergleichsgruppe — Low-Rank
schafft am selben Budget rund 120× gegenüber dicht.

Es gibt also keine gemeinsame Gewinnerachse, sondern eine Aufgabenteilung:

| Ziel | Verfahren |
|---|---|
| minimale Parameter / Speicher | **Tensor-Train** |
| minimale Latenz | **Low-Rank** |

Meine Kernaussage stelle ich deshalb auf **Parametereffizienz** um. Die
Geschwindigkeitsaussage fällt weg — beziehungsweise wird ehrlich als Nachteil
gegenüber Low-Rank ausgewiesen.

Dass mir diese Baseline so lange gefehlt hat, ist der größte methodische Fehler
in diesem Projekt gewesen. Er wäre einer Jury mit ziemlicher Sicherheit
aufgefallen.

## 2. August — Die härtere Kontrolle, und sie widerlegt mich

A-001 hatte gezeigt: Tensor-Train schlägt eine einzelne Low-Rank-Engstelle bei
gleichem Budget. Das sah nach dem Existenzbeweis für meine Struktur aus. Aber
in diesem Vergleich steckten zwei Eigenschaften gleichzeitig:

- **Lokalität** — jeder Kern wirkt auf einen Index, nicht auf den ganzen Vektor
- **Bonds** — Korrelation wird zwischen den lokalen Faktoren weitergetragen

Nur die Bonds sind das MPS-spezifische, also das Quanteninspirierte. Also
Kontrollen gebaut, die jeweils **nur eine** der beiden Eigenschaften haben.

| Arm | Struktur | Genauigkeit | b=256 |
|---|---|---|---|
| dicht | voll (16,8 Mio.) | 0,9223 | 119,6 ms |
| TT 2 Kerne/R2 | lokal + Bonds | 0,9228 | 24,0 ms |
| Low-Rank r2 | global, ein Schnitt | 0,8739 | 1,0 ms |
| Deep Bottleneck | global, verkettet | 0,8370 | 1,0 ms |
| **Block-Diagonal** | **lokal, ohne Bonds** | **0,9397** | 8,5 ms |

| TT gegen | Differenz | p |
|---|---|---|
| Low-Rank | +0,0490 | 0,002 ✓ |
| Deep Bottleneck | +0,0859 | 0,002 ✓ |
| **Block-Diagonal** | **−0,0168** | **0,004 ✗** |

### Das Ergebnis widerlegt meine Erklärung

**Der Vorteil kommt von der Lokalität, nicht von den Bonds.** Block-Diagonal,
bei dem Information die Blockgrenzen überhaupt nicht überquert, ist bei
identischem Parameterbudget **signifikant besser** als mein Verfahren — und
schlägt sogar die dichte Schicht mit 0,12 % der Parameter.

Die Bonds sind also nicht neutral. Sie **kosten** bei gleichem Budget Leistung:
Parameter, die in Bonds fließen, wären in lokaler Struktur besser angelegt.

Und Block-Diagonal ist zusätzlich 2,8-fach schneller. Mein Verfahren ist damit
auf **beiden** Achsen unterlegen.

Immerhin sauber geklärt: Die Erklärung "TT gewinnt durch Verkettung" ist
ausgeschlossen. Deep Bottleneck war mit Abstand am schlechtesten — mehrere
globale Flaschenhälse hintereinander sind schlechter als einer.

### Was das für mein Projekt heißt

Die Eigenschaft, die wirklich hilft, ist ein **klassisches, bekanntes Konzept**
(Blockstruktur, verwandt mit gruppierten Faltungen). Nichts daran ist
quanteninspiriert. Der MPS-spezifische Teil — die Bond-Kette — ist genau der,
der hier nicht trägt.

Nach A-001 war ich überzeugt, den Existenzbeweis zu haben. A-002 zeigt, dass
mir dort nur die aussagekräftigste Kontrollbedingung fehlte. Zwei Kontrollen
haben mich bestätigt, die dritte hat mich widerlegt — und die dritte war die
richtige.

Das ist bitter, aber es ist genau der Grund, warum ich diese Tests mache. Wenn
ich es nicht gefunden hätte, hätte es eine Jury gefunden.

## 2. August — Mein eigener Rettungsversuch scheitert

Nach A-002 hatte ich einen Verdacht gegen mein eigenes Ergebnis: Block-Diagonal
mischt nie über Blockgrenzen, hatte in meinem Aufbau aber dichte Schichten
davor und dahinter. Vielleicht schnitt es nur deshalb gut ab — weil die
Nachbarn die Mischarbeit übernahmen.

Also getestet: dieselben Verfahren, aber mit **drei gestapelten** komprimierten
Schichten, sodass die Mischung intern passieren muss.

| Arm | depth 1 | depth 3 | Änderung |
|---|---|---|---|
| TT 2 Kerne/R2 | 0,9241 | 0,9225 | −0,0016 |
| **Block-Diagonal** | 0,9381 | **0,9309** | −0,0072 |
| Low-Rank r2 | 0,8747 | **0,1517** | **−0,7230** |
| dicht | 0,9239 | 0,9000 | −0,0238 |

**Meine Hypothese hält nicht.** Block-Diagonal verliert beim Stapeln nur
minimal mehr als Tensor-Train (−0,0072 gegen −0,0016), und der Unterschied ist
nicht signifikant (p=0,383). Bei depth 3 ist Block-Diagonal weiterhin
signifikant besser (p=0,008).

### Warum ich dem Ergebnis traue

Die Low-Rank-Bedingung zeigt, dass mein Test den vermuteten Effekt **hätte
finden können**: Sie bricht beim Stapeln von 0,87 auf 0,15 ein — bei 10 Klassen
ist 0,10 Raten. Drei Rang-2-Engstellen hintereinander zerstören das Signal
vollständig.

Der Aufbau war also empfindlich genug. Bei Block-Diagonal ist der
Zusammenbruch schlicht nicht eingetreten. Das ist ein Negativbefund **mit
Positivkontrolle** — deutlich belastbarer, als wenn einfach nichts passiert
wäre.

### Die ehrliche Bilanz

Der quanteninspirierte Kern meines Projekts ist jetzt dreimal unabhängig
getestet und dreimal widerlegt:

| Test | Mechanismus | Ergebnis |
|---|---|---|
| Gate-Ablation | Born-Regel-Gating | kein Effekt (p=0,25, 648 Paare) |
| Entropie-Analyse | Verschränkung sagt Kompression vorher | Kontrolle widerlegt |
| A-002 / A-003 | Bond-Kette trägt globale Korrelation | Block-Diagonal besser |

Was wirklich wirkt, ist **Lokalität durch Blockstruktur** — ein klassisches
Konzept, verwandt mit gruppierten Faltungen. Mit 0,12 % der Parameter besser
als die dichte Schicht, und 2,8× schneller als mein Tensor-Train.

Ich habe drei ernsthafte Versuche gemacht, dem quanteninspirierten Teil einen
belegbaren Beitrag nachzuweisen — der letzte davon war ein gezielter Angriff
auf mein eigenes vorheriges Ergebnis. Keiner hat gehalten.

Das ist nicht das Ergebnis, das ich mir gewünscht habe. Aber es ist ein
Ergebnis, und es ist sauber belegt.

## 2. August — Lichtkegel und Unitarität

Nachdem A-003 die Bond-Hypothese widerlegt hatte, ist mir aufgefallen, dass ich
zu schnell aufgegeben hatte: **Lokalität ist selbst ein quantenmechanisches
Kernprinzip.** In lokalen Quantenschaltkreisen sind die Blöcke von Schicht zu
Schicht versetzt (Brickwork-Architektur), wodurch sich Korrelation kegelförmig
ausbreitet. Das habe ich gebaut und gemessen: bei festen Blöcken erreicht ein
Eingang nach drei Schichten 4 von 16 Einheiten, bei versetzten 12 von 16 — der
Lichtkegel existiert nachweislich.

Ergebnis trotzdem negativ: Der Versatz bringt **nichts** (−0,0033, p=0,084).
Was signifikant hilft, ist **Orthogonalität** (+0,0139, p=0,002) — die
klassische Entsprechung unitärer Quantengatter.

Also nachgetestet, ob es wirklich die Unitarität ist oder bloß Skalenkontrolle.
Drei Kontrollen: Spektralnorm, Frobenius-Energie, Weight Decay. Ergebnis:
**Spektralnormalisierung zieht gleich** (+0,0024, p=0,322). Der Effekt hängt am
Spektralradius, nicht an vollständiger Unitarität. Also wieder eine bekannte
klassische Technik.

Ein Arm war unbrauchbar: Weight Decay kollabierte auf exakt Rateniveau bei
Streuung exakt null. Vermutlich sterben die Aktivierungen ab. Ich führe das
nicht als Befund, sondern als ungeklärte Auffälligkeit.

## 2. August — DMRG-adaptive Rangzuweisung

Echtes Quantenalgorithmus-Prinzip: DMRG passt die Bond-Dimension während der
Optimierung anhand des verworfenen Spektralgewichts an. Zielachse
**Trainingsaufwand** — die einzige Ressource, die ich nie optimiert hatte.

Der Smoke-Test hat zwei eigene Fehler aufgedeckt: Mein Sättigungsmaß griff auf
den kleinsten aller Singulärwerte zu statt auf den kleinsten *behaltenen* —
bei einer rangdefizitären Matrix ist der immer null, das Maß war wertlos. Und
der Start bei Rang 1 hätte die Wachstumsarme in genau das Kollapsregime
gesetzt, das A-003 dokumentiert hat.

Nach Korrektur (Maß gegen sechs Matrizen mit bekanntem Spektrum verifiziert):
Das DMRG-Kriterium ist von **Zufallswachstum ununterscheidbar** (−0,0006,
p=1,00). Es erzeugt nicht einmal eine andere Verteilung.

Nebenbefund, klassisch: Beide Wachstumsvarianten erreichen dieselbe Genauigkeit
wie das feste Modell bei **79,7 % des Trainingsaufwands**. Klein starten und
wachsen spart ein Fünftel — nur unabhängig davon, *wie* man wächst.

## 2./3. August — Echte Quantenmechanik statt Analogie

Nach fünf gescheiterten quanten*inspirierten* Ansätzen der Wechsel: kein
quantenmotiviertes klassisches Verfahren mehr, sondern echte Quantenrechnung.

Dafür meinen Simulator nach PyTorch portiert, mit komplexen Zahlen — damit
fließen Gradienten durch Superposition und Verschränkung hindurch. Gegen die
bereits Qiskit-validierte NumPy-Fassung geprüft, 6 von 6 Tests bestanden.
Verschränkung nachgewiesen statt behauptet: p(00)·p(11)=0,25 gegen
p(01)·p(10)=0,00, also nicht separabel.

**Q-001:** Das Hybridmodell mit 80 Schaltungsparametern schlägt ein dichtes
Netz mit 67.848 Parametern signifikant (+0,0546, p=0,008). Erstes positives
Ergebnis überhaupt. Aber Verschränkung trägt nichts bei (p=0,141).

**Q-002:** Verdacht, dass die Amplituden-Kodierung heimlich normiert und das
den Vorteil erklärt. Kontrolle mit L2-Normierung: Vorteil hält (p=0,008),
Normierung erklärt nichts (p=0,547). **Erster Mechanismus, der eine faire
Kontrolle übersteht.**

**Q-003:** Dann fiel auch der. Die Schaltung macht drei Dinge — normieren,
normerhaltende Drehung, **quadratische** Auslese (⟨Z⟩ summiert über
|Amplitude|²). Den letzten Punkt hatte ich übersehen; meine klassischen
Kontrollen rechneten Linear/Tanh, eine andere Funktionsklasse. Der klassische
Nachbau aller drei Schritte **schlägt die Quantenschaltung** (−0,0147,
p=0,008). Und die orthogonale Variante — das eigentlich Quantenhafte — ist die
schlechteste und instabilste.

**Q-004b — der Kreis geschlossen:** Modell in Simulation trainiert, gelernte
Winkel auf echte IBM-Hardware übertragen (`ibm_fez`, Jobs
`d9ns0pc60llc73caf80g` und `d9ns10ssfqic73ardag0`). Vorher verifiziert, dass
Torch- und Qiskit-Schaltung identisch rechnen (Abweichung 4,6·10⁻⁶).

Hardware-Genauigkeit **identisch zur Simulation** (0,500 bzw. 0,450), 85 % der
Einzelvorhersagen stimmen exakt überein. Und mein früherer Tiefeneffekt
bestätigt sich an einer echten Anwendung: bei doppelter Gatterzahl bleiben
statt 85 % nur 71 % der Signalstärke.

**KI und Quantenraum sind damit vereint — als laufendes System, nicht als
Analogie.**

## 3. August — Demo für den Stand

Interaktive Demo gebaut (`demo/`): Ziffer zeichnen, drei Modelle im Vergleich.
Läuft offline, ohne Internet und ohne Kontingent.

> ⚠ **Diese drei Zahlen sind Demonstrationswerte, keine Messergebnisse.** Sie
> stammen aus einem einzelnen Trainingslauf in `demo/train_demo_model.py` —
> ein Seed, keine Kontrollbedingung, keine Vorregistrierung. Sie zeigen, was am
> Stand zu sehen ist. Für die Arbeit gelten die Zahlen aus Q-003-replication
> (15 Seeds, zwei Datensätze, Lernrate je Arm). Vermerkt am 6. August.

| Modell | Genauigkeit | Parameter |
|---|---|---|
| Quantenschaltung (Simulation) | 0,9580 | 80 |
| Quantenschaltung (hardwaretauglich) | 0,8078 | 80 |
| klassisch | 0,9132 | 440 |

Der sichtbare Unterschied **ist** die Aussage: In Simulation schlägt die
Schaltung sogar das klassische Modell mit 5,5-fach weniger Parametern, die
hardwaretaugliche Variante fällt ab — und genau das erklärt, warum Quanten-KI
heute noch nicht praktisch ist.

## 3. August — Neue Perspektive: Quantenkerne

Mir ist aufgefallen, dass alle sechs Versuche denselben blinden Fleck hatten:
Ich habe immer *das Modell* verändert und Genauigkeit, Parameter oder Zeit
gemessen. Nie angefasst: die Datenrepräsentation, den Optimierer, und den
**Datenbedarf** — obwohl der ausdrücklich auf meiner Ressourcenliste steht.

Also ein völlig anderes Paradigma: keine neuronalen Netze, sondern
Kernmethoden. Der Quantenzustandsraum als Merkmalsraum,
k(x,y) = |⟨φ(x)|φ(y)⟩|².

**Ergebnis: signifikant schlechter bei jeder Qubitzahl und Datenmenge**
(p=0,0078 durchgehend). Bei 3 Qubits −0,140, bei 4 Qubits −0,290, bei 6
Qubits −0,552.

### Die Ursache — und sie erklärt mehr als diesen Versuch

Der Abstand wächst systematisch mit der Qubitzahl. Nachgemessen, warum:

| Qubits | mittlere Ähnlichkeit im Kern |
|---|---|
| 2 | 0,349 |
| 3 | 0,162 |
| 4 | 0,082 |
| 6 | 0,024 |

**Die Ähnlichkeit halbiert sich mit jedem Qubit.** Bei 6 Qubits ist die
Kernmatrix praktisch die Einheitsmatrix — jeder Punkt ist nur sich selbst
ähnlich. Die SVM kann auswendig lernen, aber nichts verallgemeinern. Der Kern
ist mathematisch einwandfrei (positiv semidefinit geprüft); das Problem ist die
Geometrie.

Daraus folgt ein Prinzip:

> In einem 2ⁿ-dimensionalen Hilbertraum sind zwei zufällige Zustände fast immer
> orthogonal. Was Quantensysteme ausdrucksstark macht — die exponentiell
> wachsende Dimension — zerstört zugleich die geteilte Struktur, die
> Lernverfahren brauchen.

Das ist für mich der wichtigste theoretische Ertrag der ganzen Arbeit. Vorher
hatte ich sechs unverbundene Negativbefunde. Jetzt habe ich eine gemeinsame,
quantitativ belegte Erklärung dafür, warum ein großer Zustandsraum kein Vorteil
ist, wenn die Daten darin keine gemeinsame Struktur mehr teilen.

Methodischer Hinweis an mich selbst: Der erste Kernlauf nutzte 5 Seeds — damit
ist im Wilcoxon-Test p=0,0625 das rechnerische Minimum, Signifikanz war gar
nicht erreichbar. Mit 8 Seeds wiederholt.

## 3. August — Anmeldung: ich darf aufteilen

Nachgesehen, wie viele Projekte ich anmelden darf: **bis zu drei gleichzeitig**
pro Wettbewerbsjahr. Ich war bisher davon ausgegangen, dass ich mich auf eines
festlegen muss, und hatte deshalb entschieden, den klassischen Teil auf das
nächste Jahr zu verschieben. Diese Entscheidung beruhte also auf einer falschen
Annahme und ist damit hinfällig.

Das ist relevant, weil meine Arbeit inzwischen aus zwei Hälften besteht, die
inhaltlich kaum noch miteinander zu tun haben:

| | Inhalt | Fachgebiet |
|---|---|---|
| **A** | sieben Mechanismen, sieben Abgrenzungstests, alle negativ; Hybridmodell auf echter IBM-Hardware; Orthogonalität im 2ⁿ-dimensionalen Hilbertraum als gemeinsame Erklärung | Physik |
| **B** | *(Stand 3. August abends neu gefasst)* Keine strukturierte Familie ersetzt unter fairen Bedingungen eine dichte Schicht — belegt über zwei Datensätze, 30 Seeds, Lernrate je Arm; dazu drei einzeln gemessene Ursachen, warum frühere Messungen das Gegenteil zeigten. Plus: normieren + Transformation + quadrieren (33× weniger, +7,4 Prozentpunkte); Faktor 6,4 zwischen nachträglich aufgezwungener und mittrainierter Struktur; 512-Neuronen-Schwelle | Mathematik/Informatik |

Der Punkt, der mich stört: B steht im Moment als Nebenprodukt in einem Projekt,
dessen Überschrift „quanteninspiriert" lautet — obwohl B die einzigen Ergebnisse
enthält, die praktisch etwas taugen. In A ist es außerdem argumentativ
unbequem, weil es meine eigene Hauptaussage verwässert: A sagt „der
Quantenmechanismus trägt nicht", und mittendrin stehen Ergebnisse, die etwas
tragen — nur eben klassisch.

Dagegen steht der Aufwand. Zwei Anmeldungen heißen zwei schriftliche Arbeiten,
zwei Stände, zwei Präsentationen am selben Wettbewerbstag, und ich arbeite
allein. Bis zum 30. November sind es rund vier Monate.

**Noch nicht entschieden.** Falls ich aufteile, dann unter der Bedingung, dass
A die Hauptarbeit bleibt und B im Wesentlichen aus Messungen besteht, die ich
schon habe — neue Experimente nur für B fange ich nicht mehr an. Was für B
tatsächlich noch fehlt, ist keine Messung, sondern die Einordnung: Blockstruktur
ist mit gruppierten Faltungen verwandt, und ich muss belegen können, was daran
über den bekannten Stand hinausgeht. Ohne diese Abgrenzung wäre B als
eigenständiges Projekt angreifbar — und zwar genau an der Stelle, an der ich in
diesem Projekt schon siebenmal selbst zugeschlagen habe.

Nebenbei den Literaturstand korrigiert: Ich habe inzwischen rund drei Viertel
meiner Quellen wirklich gelesen, zwei Bücher und ein PDF stehen noch aus. Im
Tagebuch stand noch „recherchiert, aber noch nicht selbst gelesen" — das war
veraltet. Die Karten in `sources/literature_cards/` sind allerdings weiterhin
pauschal als ungelesen markiert und geben damit einen falschen Stand wieder.

## 3. August — B-003 bis B-006: meine Blockschicht-Aussage fällt

### Vorweg ein Fund, der alles andere betrifft

Keines meiner Experimentskripte hat seine Ergebnisse jemals gespeichert. `run()`
gab ein Dictionary zurück, `analyse()` druckte auf die Konsole, danach war jede
Zahl weg. **Sämtliche Werte in diesem Tagebuch stammen aus Konsolenausgaben, die
es nicht mehr gibt** — nicht nachrechenbar, nicht anders auswertbar, einer Jury
nicht vorlegbar. Behoben: `result_store.py` speichert jetzt pro Lauf jede
einzelne Messung als eigene Zeile, dazu die vollständige Konfiguration und die
Softwareumgebung, als JSON und CSV unter `results/experiments/`.

Was mich das schon gekostet hat, steht weiter unten bei der Orthogonalität.

### Die Aussage, die ich prüfen wollte, war nie gemessen worden

Im Tagebuch stand als klassisches Hauptergebnis: *„spektral-normalisierte
Blockschicht, 819× weniger Parameter, besser als dicht."* Beim Nachsehen im Code
war das aus zwei Läufen zusammengesetzt, die es so nie zusammen gab:

- A-002: unbeschränkte Blockschicht gegen dicht, Seeds 701–710, Tiefe 1
- B-002: spektral gegen unbeschränkt, Seeds 1001–1010, Tiefe 3, **ohne jeden
  dichten Vergleichsarm**

Aus „unbeschränkt schlägt dicht" und „spektral schlägt unbeschränkt" folgt
rechnerisch nichts: verschiedene Seeds, verschiedene Tiefen, und die
Vertrauensbereiche zweier gepaarter Tests setzen sich nicht zusammen.

### B-003 — erstmals in einem Lauf, gleiche Seeds

Alle vier Arme zusammen, 10 frische Seeds ab 1101, beide Tiefen.

| Tiefe 1 | Genauigkeit | Parameter | gegen dicht |
|---|---|---|---|
| dicht | 0,9237 | 16.781.312 | — |
| Block, unbeschränkt | 0,9362 | 20.480 | +0,0125, 10/10, p=0,0020 |
| Block, spektral | 0,9394 | 20.480 | +0,0157, 10/10, p=0,0020 |
| Block, orthogonal | 0,9442 | 20.480 | +0,0204, 10/10, p=0,0020 |

Der Zusatz „spektral-normalisiert" trägt bei Tiefe 1 nichts (+0,0032, p=0,43),
bei Tiefe 3 etwas (+0,0075, p=0,027). Die Aussage war also in ihrer pauschalen
Form schon hier nicht haltbar. Nebenbei: die *dichte* Schicht wird mit mehr
Tiefe schlechter (0,9237 → 0,9035), die Blockschichten nicht.

### Ein Fund, den ich ausdrücklich nicht als Ergebnis führe

`block_orthogonal` war in jeder Zeile der beste Arm, aber nie direkt gegen
`block_spectral` getestet. Genau das ist Befund 4 meiner Gesamtbilanz
(„Unitarität gegen Spektralnorm, p=0,32, kein Unterschied"). Weil die Rohdaten
jetzt gespeichert werden, konnte ich es nachrechnen: **+0,0068 mit p=0,037 bei
Tiefe 3** — und bei Tiefe 1 nicht signifikant (p=0,065).

Das führe ich nicht als Ergebnis, aus drei Gründen: der Vergleich war nicht
vorregistriert, ich habe auf diesem Datensatz sechs Tests gerechnet (unter
Holm-Korrektur überlebt nur der stärkste), und über beide Tiefen repliziert er
nicht.

Bemerkenswert ist er trotzdem. B-002 hat **exakt denselben Vergleich** gerechnet
— gleiche Architektur, Blockgröße, Tiefe, Epochenzahl, ebenfalls 10 Seeds, nur
andere Seeds — und kam auf p=0,322. Dasselbe Experiment, gegenteilige
Schlussfolgerung, nur wegen der Seed-Wahl. Eine schärfere Demonstration dafür,
wie wackelig ein einzelner n=10-Test nahe der Schwelle ist, finde ich in keinem
Lehrbuch. Zusammenlegen kann ich die beiden Läufe nicht, weil B-002 seine
Rohdaten nicht gespeichert hat — das ist der konkrete Preis des Fundes von oben.

### Der eigentliche Einwand: falscher Gegner, blinde Architektur

Zwei Dinge stimmten an meinem bisherigen Vorgehen nicht.

**Erstens der Gegner.** Blockdiagonale Schichten sind seit Nesky & Stout (ICANN
2018) bekannt, gruppierte Faltungen seit AlexNet. Gegen eine dichte Schicht zu
gewinnen ist kein Ergebnis mehr. Der Stand sind Monarch (Dao et al., ICML 2022:
zwei blockdiagonale Matrizen mit einer Permutation dazwischen) und
Block-Tensor-Train (Qiu et al. 2024). Monarch existiert genau deshalb, *weil*
eine reine Blockdiagonale nicht über Blockgrenzen mischt. Die Literatur
behauptet also das Gegenteil meines Befunds.

Also selbst implementiert und vorab verifiziert: schneller Pfad gegen eine
unabhängig gebaute explizite Matrixkonstruktion (Blockdiagonale mal
Permutationsmatrix mal Blockdiagonale), größte Abweichung 3,3·10⁻¹⁶ über sechs
Größen und Ränge. Dazu die Parameterzahl gegen die Formel und der Gradientenfluss.

Damit ließ sich ein Vergleich bauen, bei dem **vier Familien exakt dieselbe
Parameterzahl haben** — nicht ungefähr, sondern auf den Eintrag genau:

| Familie | Rechnung | Parameter |
|---|---|---|
| Monarch (64×64) | 4096·(64+64) | 524.288 |
| Blockdiagonale (128) | 32·128² | 524.288 |
| Tensor-Train (2 Kerne, Rang 64) | 2·64·64·64 | 524.288 |
| Low-Rank (Rang 64) | 2·4096·64 | 524.288 |

**Zweitens die Architektur.** In allen meinen Aufbauten stand vor und hinter der
komprimierten Schicht eine dichte Schicht. A-003 wollte prüfen, ob die
Blockdiagonale nur deshalb gut abschneidet, weil ihre dichten Nachbarn das
Mischen übernehmen — hat dafür aber nur die Mittelschichten gestapelt und die
dichten Enden stehen lassen.

Ich habe das jetzt gemessen statt diskutiert, über die Jacobi-Matrix der
Aktivierungen vor der Auslese:

| Aufbau | größter Jacobi-Eintrag zu einer fremden Gruppe |
|---|---|
| neuer Aufbau, gruppenweise Enden, Blockdiagonale | **exakt 0** |
| neuer Aufbau, gruppenweise Enden, Monarch | 3,3·10⁻² |
| **mein alter Aufbau mit dichten Enden, Blockdiagonale** | **6,9·10⁻²** |

Im alten Aufbau floss auch mit blockdiagonaler Mitte Information zwischen den
Gruppen. A-003 konnte seine eigene Frage nicht beantworten.

### B-004 — und zuerst fällt meine Lernrate

Alle bisherigen Experimente trainieren jeden Arm mit lr=0,01. Verschiedene
Strukturen haben aber verschiedene Konditionierung. Also vorab eine
Lernratensuche je Arm und Architektur auf **eigenen** Tuning-Seeds (1301–1303),
der Hauptlauf dann mit je eigener Rate auf frischen Seeds (1201–1210).

| dichte Schicht, dichte Enden | Genauigkeit |
|---|---|
| lr 0,003 | **0,9359** |
| lr 0,01 — meine bisherige Einstellung | 0,9302 |
| lr 0,03 | 0,8878 |

Die dichte Schicht war in **jedem** meiner bisherigen Vergleiche mit einer
Lernrate trainiert, die sie rund 0,006 Genauigkeit gekostet hat. Und die
Blockschichten bevorzugen tatsächlich eine andere Rate als die übrigen Arme —
ein fester Wert konnte den Vergleich also gar nicht fair machen.

### Die 819×-Aussage, fair gemessen

`block_diag_4` (20.480 Parameter, das ist das Budget der 819×-Aussage) gegen
dicht, jeder Arm bei seiner besten Lernrate, 10 Seeds, Bootstrap-Intervalle:

| Architektur | Differenz | 95 %-KI | Siege | p |
|---|---|---|---|---|
| dichte Enden (mein alter Aufbau) | +0,0054 | [+0,0013, +0,0104] | 8/10 | 0,037 |
| gruppenweise Enden | **−0,0058** | [−0,0102, −0,0018] | 2/10 | 0,027 |
| gruppenweise Enden, Pixel permutiert | **−0,0109** | [−0,0137, −0,0082] | 0/10 | 0,002 |

**Die Aussage gilt nur in der Architektur mit dichten Enden. Nimmt man die
freie globale Mischung weg, dreht das Vorzeichen.**

Dazu kommt: In dem Aufbau, aus dem A-002 und A-003 stammen, trennt sich die
dichte Schicht mit 16,8 Millionen Parametern **nicht** signifikant von der
Blockschicht mit 528.384 (+0,0046, p=0,23). Meine Positivkontrolle fällt dort
durch — ein Budgetunterschied von Faktor 32 ist in dieser Architektur nicht
auflösbar.

Präzisierung, damit ich nicht überziehe: „blind" wäre zu stark. Der
spektral-normalisierte Blockarm trennt sich in derselben Architektur sehr wohl
von dicht (+0,0095, Holm-korrigiert p=0,023). Der Aufbau ist also nicht
unempfindlich für alles — er kann nur den Vergleich nicht auflösen, auf dem
meine zurückgezogene Aussage beruhte.

### Die Pixelpermutation trifft den anderen als gedacht

Ich hatte vermutet, dass die Blockdiagonale von der Bildstruktur lebt — Gruppen
aus benachbarten Pixeln. Gemessen, mit einer festen zufälligen Permutation für
alle Arme und Seeds:

| gruppenweise Enden → dieselbe Architektur mit gemischten Pixeln | Änderung | p |
|---|---|---|
| Blockdiagonale | +0,0013 | 0,92 |
| dicht | +0,0018 | 0,56 |
| **Monarch** | **−0,0133** | **0,0020** |

Genau umgekehrt. Die Blockdiagonale ist gegen Pixelmischung völlig
gleichgültig, Monarch verliert signifikant. Damit lösen sich die beiden Urteile
auf: bei natürlicher Bildstruktur schlägt Monarch die Blockdiagonale (+0,0115,
p=0,0039), bei gemischten Pixeln nicht mehr (−0,0031, p=0,23) — in beiden Fällen
mit bestandener Positivkontrolle.

**Monarchs Vorteil ist nicht globale Mischung an sich, sondern das Ausnutzen
räumlicher Struktur in den Daten.** Ohne diese Kontrolle hätte ich „Monarch ist
besser" geschrieben. Das wäre falsch gewesen, und niemand hätte es gemerkt, weil
die Kontrolle auch in der Literatur fehlt.

### B-005 — was die dichte Schicht wirklich lernt

Statt noch ein Pferderennen: hineinschauen. Die gelernte 4096×4096-Matrix in
32×32 Blöcke à 128 zerlegt und die Betragsmasse auf der Diagonale gemessen,
gegen zufällige Permutation als Nullverteilung.

| | bei Initialisierung | trainiert |
|---|---|---|
| Blockmasse auf der Diagonale (Zufall: 0,0312) | 0,0313 | **0,0312** |
| dieselbe nach zufälliger Permutation | 0,0313 | 0,0313 |
| beste per Co-Clustering gefundene Anordnung | 0,0341 | 0,0330 |
| **Energie im Top-64-Rang** | **0,0542** | **0,7144** |

Die dichte Schicht lernt **exakt nichts Blockartiges** — Faktor 1,00 gegenüber
Zufall, und auch in keiner anderen Anordnung, die das Co-Clustering findet.

Sie lernt etwas anderes: Das Training presst die Spektralenergie von 5,4 % auf
71 % in die 64 größten von 4096 Richtungen. Und dazu passt ein Messwert aus
B-004, der mir vorher nicht aufgefallen war: **Low-Rank mit Rang 64 gegen dicht
ist +0,0003 bei p=0,85** — statistisch ununterscheidbar, bei 32× weniger
Parametern.

Das ist zum ersten Mal in diesem Projekt eine *Erklärung* statt einer Messung:
Die dichte Schicht nutzt effektiv rund 64 ihrer 4096 Richtungen, deshalb reicht
Rang 64 genau aus — und deshalb sind Blöcke die falsche Struktur für das, was
sie tatsächlich tut.

### B-006 — nach meiner eigenen Regel nicht verwertbar

Konvergenzkontrolle mit 15.000 Beispielen und 30 Epochen statt 6.000 und 10. Bei
dichten Enden erreichen dort alle Arme 0,960 ± 0,001 und **kein** Unterschied
bleibt signifikant. Bei gruppenweisen Enden repliziert die Richtung (Monarch
+0,0152, p=0,031), aber die Positivkontrolle verfehlt die Schwelle (p=0,0625 —
bei 6 Seeds ist 0,031 das rechnerische Minimum, also eine Stufe daneben).

Das vorher festgelegte Kriterium sagt damit: verwerfen. Ich deute das nicht um.
Nötig wären mehr Seeds. Unabhängig davon zeigt B-006 aber: **ein Teil der
Unterschiede in B-004 ist ein Untertrainings-Artefakt** — bei längerem Training
verschwinden sie bei dichten Enden vollständig.

### Was ich zurückziehe

1. *„Spektral-normalisierte Blockschicht, 819× weniger Parameter, besser als
   dicht."* Gilt nur in einer Architektur, deren Blindheit inzwischen gemessen
   ist; in beiden faireren Aufbauten dreht das Vorzeichen. Der Zusatz
   „spektral-normalisiert" war ohnehin nie in einem Lauf mit dichtem Arm geprüft.
2. *„Bonds tragen nichts, es ist die Lokalität."* Monarch schlägt die
   Blockdiagonale bei gleichem Budget auf natürlichen Bilddaten.

### Was stattdessen dasteht

- **Rang 64 ersetzt eine dichte 4096×4096-Schicht bei 32× weniger Parametern**
  (+0,0003, p=0,85) — und B-005 erklärt, warum. Wie stark diese Aussage sein
  darf, klärt der Äquivalenztest weiter unten.
- **Monarchs Vorteil hängt an räumlicher Datenstruktur**, nicht an globaler
  Mischung. Ein Befund über ein aktuelles Verfahren, belegt mit einer Kontrolle,
  die in der Literatur fehlt.
- **Drei Messfehlerquellen nachgewiesen statt vermutet**: feste Lernrate, blinde
  Architektur, undichte Kontrolle in A-003.

### Äquivalenztest — und er nimmt meiner neuen Hauptaussage die Schärfe

„Rang 64 ersetzt eine dichte Schicht **ohne messbaren Verlust**" ist eine
Äquivalenzbehauptung. Ein Wilcoxon-Test mit p=0,85 belegt die nicht — er sagt
nur „Unterschied nicht nachgewiesen". Das ist nicht dasselbe, und es steht als
Vorbehalt schon länger in meinem eigenen Tagebuch. Das richtige Werkzeug ist
TOST: zwei einseitige Tests gegen eine vorher festgelegte Grenze.

Grenze vorab festgelegt auf ±0,005 (kleiner als die Seed-Streuung des dichten
Arms). Ergebnis für Rang 64 gegen dicht:

| | |
|---|---|
| Differenz | +0,0003 |
| Wilcoxon p / nach Holm | 0,85 / 1,00 |
| 90-%-Intervall | [−0,0072, +0,0078] |
| **TOST p bei ±0,005** | **0,139 — Äquivalenz NICHT nachgewiesen** |
| erreichte Schranke | **±0,0078** |

**Ich darf also nicht „ohne messbaren Verlust" schreiben, sondern nur
„gleichwertig innerhalb ±0,008".** Bei 10 Seeds gibt die Datenlage nicht mehr
her. Für die strengere Aussage bräuchte ich rund 30 Seeds — die Breite des
Intervalls fällt mit der Wurzel aus der Seed-Zahl, aus ±0,0078 würden dann etwa
±0,0045.

Das ist genau der Fehler, den ich bei anderen Befunden schon zweimal vermieden
habe, und ich wäre hier fast wieder hineingelaufen: „nicht signifikant" als
„gleich" zu lesen.

Zwei weitere Werte aus derselben Auswertung, mit Holm-Korrektur je Testfamilie:

- **Monarch ist in der lokalen Architektur so gut wie die dichte Schicht**
  (−0,0008, p=0,92), erreichte Schranke ±0,0057 — bei 32× weniger Parametern.
  Das ist der stärkste Äquivalenzkandidat, den ich habe.
- **Bei permutierten Pixeln erreicht kein einziger Arm die dichte Schicht**
  (alle Holm p=0,0117). Ohne räumliche Struktur in den Daten hilft keine der
  getesteten Strukturen.

Ehrliche Vorbehalte: Ein Teil dieser Tests war post-hoc auf Daten, die für eine
andere Frage erhoben wurden. Die knappen Werte (p=0,027 bis 0,037) überleben
teilweise die Holm-Korrektur nicht — `block_diag_4` gegen dicht etwa fällt von
p=0,037 auf p=0,186. Und alles steht weiterhin auf MNIST allein.

Das ist der zweite große Befund, den ich mir selbst kaputtgemacht habe. Beim
ersten Mal (Tensor-Train, A-002) hat es weh getan. Diesmal war es der erwartete
Ausgang — und die Ersatzergebnisse sind besser als das, was sie ersetzen.

## 3. August — B-008 und B-009: auch der Ersatzbefund fällt

Am Vormittag hatte ich zwei Aussagen zurückgezogen und zwei neue an ihre Stelle
gesetzt: „Rang 64 ersetzt eine dichte Schicht" und „Monarch ist in der lokalen
Architektur so gut wie dicht". Beide standen auf 10 Seeds. Der Äquivalenztest
sagte mir noch am selben Tag, dass 10 Seeds für eine Gleichwertigkeitsaussage
nicht reichen — die Datenlage trug nur ±0,008 statt der angestrebten ±0,005.
Also beides mit **30 Seeds** und auf **zwei Datensätzen** wiederholt.

### B-008 — die Vorhersage aus B-005 fällt durch

Die Idee war, aus einer Beobachtung eine Regel zu machen: Wenn die dichte
Schicht ihre Energie auf wenige Richtungen konzentriert, muss die Genauigkeit
dort sättigen, wo die Energie sättigt. Vorher festgelegt: **eine einzige**
Energieschwelle muss auf **beiden** Datensätzen passen, sonst gilt die
Vorhersage als widerlegt.

MNIST, 30 Seeds, exakte SVD:

| Rang | Parameter | Genauigkeit | vs. dicht | TOST p | kum. Energie |
|---|---|---|---|---|---|
| dicht | 16.781.312 | 0,9379 | — | — | 1,0000 |
| 8 | 69.632 | 0,9332 | −0,0047 | 0,46 | 0,4598 |
| 64 | 528.384 | 0,9313 | −0,0066 | 0,72 | 0,6455 |
| 256 | 2.101.248 | 0,9269 | −0,0110 | 1,00 | 0,7416 |

**Kein einziger Rang erreicht Gleichwertigkeit innerhalb ±0,005**, auf keinem
der beiden Datensätze. Die Vorhersage ist widerlegt.

### Und dabei ist mir ein eigener Messfehler aufgefallen

Ich hatte aus B-005 berichtet, das Training presse **71 %** der Spektralenergie
in die 64 größten Richtungen. Exakt gemessen sind es **64,6 %** — und selbst bei
Rang 256 erst 74 %. Die Matrix ist gar nicht stark niedrigrangig.

Zwei Ursachen kommen in Frage, und ich kann sie nicht auseinanderhalten: B-005
benutzte `svd_lowrank` (genähert), B-008 die exakte SVD; und B-005 trainierte
mit lr=0,01 statt der gesuchten 0,003. Beides plausibel, keines von beiden
sauber isoliert. Für die schriftliche Arbeit gilt die exakte Zahl.

Das ist das zweite Mal, dass mich eine Näherung an einer Stelle erwischt hat,
wo ich sie für unkritisch hielt. Beim ersten Mal war es die Streaming-Kostenformel.

### Was der Lauf stattdessen zeigt

Die Genauigkeit ist über die Ränge **flach**: 0,9332 bei Rang 8, 0,9313 bei
Rang 64, 0,9269 bei Rang 256. Rang 8 ist mit 69.632 Parametern genauso gut wie
Rang 256 mit 2,1 Millionen — und **alle** liegen etwa 0,005 unter dicht.

Mehr Rang schließt die Lücke nicht. Zusammen mit der Energiemessung ergibt das
ein stimmiges Bild: Der Vorsprung der dichten Schicht ist keine Rangfrage. Sie
ist nicht effektiv niedrigrangig, und deshalb kann keine Low-Rank-Näherung sie
einholen, egal wie viel Budget man ihr gibt.

### B-009 — auch Monarch ist nicht gleichwertig

Monarch gegen dicht in der lokalen Architektur, 30 Seeds:

| Datensatz | Differenz | p (Holm) | Lesart |
|---|---|---|---|
| MNIST | −0,0045 | 0,0010 | Unterschied belegt |
| Fashion-MNIST | −0,0112 | 0,0001 | Unterschied belegt |

Bei 10 Seeds war es −0,0008 mit p=0,92, und ich hatte es als besten
Äquivalenzkandidaten des Projekts bezeichnet. Bei 30 Seeds ist es ein belegter
Nachteil. **Zum dritten Mal an einem Tag** ein knapper Befund bei kleiner
Seed-Zahl, der bei ausreichender Seed-Zahl in die Gegenrichtung kippt.

### Der Befund aus B-004 repliziert nur auf einem Datensatz

Monarch direkt gegen die Blockdiagonale, 30 Seeds:

| Datensatz | Monarch − Block | Siege | p |
|---|---|---|---|
| MNIST | **+0,0094** | 27/30 | <0,0001 |
| Fashion-MNIST | −0,0028 | 11/30 | 0,077 |

Auf MNIST repliziert der Befund aus B-004 sehr deutlich (dort +0,0115). Auf
Fashion-MNIST dreht das Vorzeichen und die beiden sind praktisch gleichwertig
(TOST-Schranke ±0,0052).

Damit ist auch meine Erklärung von heute Vormittag zu einfach. Ich hatte
geschrieben, Monarchs Vorteil komme vom Ausnutzen **räumlicher Bildstruktur**.
Fashion-MNIST hat räumliche Struktur — und dort gibt es den Vorteil nicht. Die
richtige Aussage ist vorsichtiger: der Vorteil hängt am Datensatz, und woran
genau, weiß ich nicht. Bei permutierten Pixeln verschwindet er (B-004), aber
räumliche Struktur allein erklärt ihn nicht.

### Die Bilanz von B nach diesem Tag

Vier Hauptaussagen sind an einem Tag gefallen: die Blockschicht gegen dicht,
„Bonds tragen nichts", Rang 64 als Ersatz für dicht, und Monarch als
gleichwertig. Was bleibt, ist kein einziger positiver Effizienzbefund — aber
etwas, das ich für tragfähiger halte als das, was ich vorher zu haben glaubte:

**Keine der getesteten strukturierten Familien ersetzt unter fairen Bedingungen
eine dichte Schicht.** Low-Rank, Blockdiagonale, Monarch und Tensor-Train liegen
alle zwischen 0,005 und 0,014 darunter, auf zwei Datensätzen, mit je eigener
Lernrate und 30 Seeds. Mehr Budget hilft nicht.

**Und ich kann benennen, warum frühere Messungen — auch meine eigenen — das
Gegenteil zu sehen glaubten.** Drei Ursachen, jede einzeln gemessen:

1. eine Architektur, die einen Faktor 32 im Parameterbudget nicht auflöst
   (Positivkontrolle fällt durch, p=0,23)
2. eine feste Lernrate, die den dichten Arm rund 0,006 Genauigkeit kostete
3. zu kleine Seed-Zahlen — dreimal an einem Tag kippte ein Befund zwischen
   10 und 30 Seeds

Das ist eine Methodenarbeit geworden, keine Effizienzarbeit. Aber es ist eine,
die etwas belegt, statt etwas zu behaupten.

## 4. August — B-010 und B-011: das falsche Spektrum, und der erste bewiesene Befund

### B-010 — die Datenachse gibt nichts her, mit einer Ausnahme

Erste Frage nach dem Perspektivwechsel: Zahlt sich Struktur aus, wenn die Daten
knapp werden? Sechs Trainingsgrößen von 200 bis 14.000, fester großer Testsatz,
angeglichene Zahl der Gradientenschritte, Lernrate je Arm und Datenmenge,
15 Seeds, zwei Datensätze.

**Urteil: H0.** Kein Arm erfüllt die drei vorher festgelegten Bedingungen.

Eine Sache will ich trotzdem festhalten, weil sie repliziert:

| bei n=200 | Abstand zu dicht | |
|---|---|---|
| MNIST | **+0,0149** | signifikant |
| Fashion-MNIST | **+0,0139** | signifikant |

Die blockdiagonale Schicht gewinnt bei der kleinsten Datenmenge auf **beiden**
Datensätzen. Gescheitert ist nur der Trendtest — die Richtung stimmt überall
(ρ=−0,600 und ρ=−0,257), aber sechs Punkte reichen nicht für Signifikanz.

Und die naheliegende Erklärung ist von denselben Daten schon widerlegt:
„weniger Parameter, weniger Überanpassung" kann es nicht sein, denn
`lowrank_r8` hat mit 69.632 die wenigsten Parameter und ist bei n=200 auf MNIST
am schlechtesten (−0,0204). Der Gewinner hat fast das Achtfache. Es liegt an der
Blockstruktur, nicht an der Anzahl — und laut B-004 auch nicht an räumlicher
Bildstruktur. Offenes Rätsel, konkret genug für einen eigenen Nachtest.

Nebenbei ein eigener Programmierfehler: Meine stratifizierte Ziehung erreicht
14.000 nicht exakt, sondern liefert je Seed eine andere Zahl. Damit landete
jeder Seed in einer eigenen Gruppe und die Statistik der größten Datenmenge
wurde NaN. Rohdaten waren vollständig, Auswertung korrigiert.

### B-011 — ich hatte in B-008 das falsche Spektrum gemessen

B-008 hat das Spektrum der Gewichtsmatrix W gemessen und daraus geschlossen,
die dichte Schicht sei gar nicht niedrigrangig. Für die Frage, was eine
Trunkierung kostet, zählt aber nicht W allein, sondern W **gewichtet mit der
Verteilung der echten Eingaben**: eine Richtung mit großem Singulärwert ist
gleichgültig, wenn die Daten sie nie ansprechen.

Beides gemessen, an derselben trainierten Schicht:

| | Rang 8 | Rang 64 |
|---|---|---|
| Spektrum der Gewichte W (das aus B-008) | 0,459 | 0,646 |
| **Spektrum der Ausgaben auf echten Daten** | 0,923 | **0,998** |

Auf Fashion-MNIST dasselbe Bild (0,963 und 0,999). **Rang 64 erfasst 99,8 % der
tatsächlich genutzten Varianz**, nicht 65 %. Der Negativbefund aus B-008 war
kein Befund über die Sache, sondern über meine Messgröße.

> **Nachtrag vom selben Tag (B-012):** Aus dieser Zahl habe ich gelesen, die
> Schicht benutze „64 Richtungen". Das war zu großzügig. „Energie bei Rang 64"
> ist eine kumulierte Größe mit langem Schwanz; der schwellenfreie Effektivrang
> (Teilnahmezahl) liegt bei **5 bis 7**. Beide Zahlen stimmen und meinen
> Verschiedenes — aber die anschauliche Aussage „die Schicht arbeitet in rund
> sieben Richtungen" ist die richtige. Ich lag um eine Größenordnung daneben.

### Und daraus folgt der erste bewiesene Äquivalenzbefund des Projekts

Vier Wege zur selben Rang-64-Schicht, MNIST, 15 Seeds, dichte Referenz 0,9403:

| Arm | Genauigkeit | vs. dicht | TOST |
|---|---|---|---|
| von null trainiert | 0,9290 | −0,0114 | schlechter (p=0,0067) |
| **SVD-Projektion der trainierten dichten Schicht** | **0,9407** | **+0,0003** | **gleichwertig, p<0,0001** |
| Destillation der dichten Funktion | 0,9387 | −0,0016 | gleichwertig, p<0,0001 |

Auf Fashion-MNIST repliziert die entscheidende Zeile: SVD-Projektion +0,0011,
TOST p=0,0001, ebenfalls gleichwertig.

**Das ist die erste Äquivalenz, die ich in diesem Projekt nicht bloß nicht
widerlegt, sondern nachgewiesen habe** — mit einem Test, der dafür gebaut ist,
nicht mit einem p>0,05.

Gestern hatte ich „Rang 64 ersetzt eine dichte Schicht" zurückgezogen. Das war
richtig für die Art, wie ich es damals trainiert hatte. Die korrigierte Aussage
lautet: **Rang 64 ersetzt eine dichte 4096×4096-Schicht bei 32× weniger
Parametern ohne messbaren Verlust — wenn man ihn aus der trainierten dichten
Schicht gewinnt statt von null zu trainieren.**

### Zwei Regime, und sie erklären die flache Rangkurve

Bei Rang 8 dreht sich das Bild um: Projektion (−0,0321) und Destillation
(−0,0226) fallen deutlich ab, das von null trainierte Modell (−0,0059) ist
**besser** als beide. Bei Rang 8 kann die Familie die dichte Funktion nicht
darstellen — das Training findet dann eine andere, eigene Lösung, und die ist
besser als jede Projektion.

Damit ist auch die flache Rangkurve aus B-008 erklärt: Von null trainiert bringt
mehr Rang nichts, weil das Training gar nicht am Rang hängt, sondern am
Optimieren. Erst wenn man die Lösung von der dichten Schicht übernimmt, zahlt
sich der höhere Rang aus.

### Was ich einschränken muss

- Die **Optimierungslücke gilt nur für MNIST**. Auf Fashion-MNIST ist das von
  null trainierte Modell nicht signifikant schlechter (−0,0011, p=0,80) — dort
  gibt es keine Lücke zu schließen. Repliziert haben nur das Spektrum-Ergebnis
  und die Äquivalenz der Projektion.
- Die Projektion setzt eine **fertig trainierte dichte Schicht** voraus, kostet
  also mehr Gesamtrechenzeit, nicht weniger. Das ist ein Kompressions-, kein
  Effizienzbefund. Als Effizienzgewinn darf ich das nicht ausgeben.
- **Es steht im Widerspruch zu meinem eigenen Faktor-6,4-Befund**, der besagt,
  nachträglich aufgezwungene Struktur koste deutlich mehr als mittrainierte.
  Hier ist es umgekehrt. Der frühere Befund stammt von 32 Neuronen und kleinen
  Datensätzen; offenbar hat er eine Grenze, und ich habe sie überschritten. Das
  gehört nebeneinander in die Arbeit, nicht wegerklärt.

## 4. August — B-012 bis B-014: meine Erzählung fällt, ein Gesetz bleibt

### B-012: der Rang kollabiert nicht, er ist sofort da

Meine Hypothese lautete: Das Netz startet mit Rang 4096 und kollabiert während
des Trainings auf wenige Richtungen — der Weg führe also durch hohen Rang, und
deshalb scheitere ein von Anfang an beschränktes Modell.

Gemessen, je Epoche:

| Epoche | Effektivrang (MNIST) | Energie@64 |
|---|---|---|
| 1 | 5,1 | 0,9990 |
| 10 | 6,7 | 0,9977 |

**Der Rang steigt** (ρ=+0,988, p<0,0001; Fashion-MNIST 3,0 → 4,8). Und schon
nach einer einzigen Epoche liegen 99,9 % der Ausgabeenergie in 64 Richtungen.
Die niedrigdimensionale Struktur ist praktisch sofort da. Meine Erzählung war
falsch.

Auch eine eigene Zahl muss ich korrigieren: Ich hatte „Rang 64 erfasst 99,8 %"
berichtet. Der schwellenfreie Effektivrang liegt aber bei **5 bis 7**. Die
Schicht nutzt effektiv etwa sieben Richtungen, nicht 64 — ich war um eine
Größenordnung zu großzügig.

Was bleibt: Beschränken ab Epoche 0 kostet auf beiden Datensätzen rund **0,010**
gegenüber späterem Beschränken. Der Vorteil ist aber nach zwei bis drei Epochen
ausgeschöpft, deshalb ist der Zusammenhang nur auf Fashion-MNIST monoton
signifikant (ρ=+0,857, p=0,0137) und auf MNIST nicht. Nach dem vorher
festgelegten Kriterium: H0 für MNIST.

### B-013: das Curriculum scheitert — und zwar an der richtigen Kontrolle

Aus B-012 hatte ich vorab eine Vorhersage ins Skript geschrieben: Wenn die
Struktur schon nach einer Epoche steht, darf ein über acht Epochen verteiltes
Curriculum nicht besser sein als ein einzelner Schnitt.

| MNIST | Genauigkeit | |
|---|---|---|
| dicht (16,8 Mio. Parameter) | 0,9367 | |
| fester Rang 64 ab Epoche 0 | 0,9303 | |
| **ein Schnitt bei Epoche 5** | **0,9399** | |
| Curriculum (512→256→128→64) | 0,9350 | |

**H0 — das Curriculum schlägt festen Rang nicht.** Schlimmer: Gegen den
einzelnen Schnitt ist es auf Fashion-MNIST signifikant **schlechter**
(−0,0106, Holm p=0,0012). Vermutlich, weil jede Projektion den Adam-Zustand
zurücksetzt — vier Störungen statt einer.

Was dabei herauskommt, ist trotzdem verwertbar: **Ein einzelner Schnitt zur
richtigen Zeit schlägt festen Rang** (MNIST +0,0097, Holm p=0,0215). Gegenüber
der dichten Schicht liegt er auf beiden Datensätzen numerisch vorn (+0,0032 und
+0,0049), aber bei 15 Seeds ist weder Überlegenheit noch Gleichwertigkeit
nachgewiesen — die Datenlage trägt nur ±0,007 bzw. ±0,010. Das gehört so
gesagt, nicht als Erfolg verkauft.

### B-014: der erste kontrollierte Befund des Projekts

Synthetische Daten mit vorgegebener intrinsischer Dimension d, gekreuzt mit der
Klassenzahl C. Zwei echte unabhängige Variablen — das hatte dieses Projekt noch
nie; alle Datensätze vorher waren vorgefunden.

Effektivrang der Mittelschicht **nach** dem Training:

| d \ C | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| 2 | 1,4 | 1,7 | 1,7 | 1,7 |
| 10 | 1,5 | 2,4 | 2,1 | 2,7 |
| 100 | 1,7 | 3,0 | 3,7 | 4,6 |

Und am **untrainierten** Netz:

| d \ C | 2 | 5 | 10 | 20 |
|---|---|---|---|---|
| 2 | 3,7 | 3,7 | 3,7 | 3,7 |
| 10 | 19,1 | 19,2 | 19,1 | 19,1 |
| 100 | 165,4 | 165,6 | 165,5 | 165,6 |

Das ist die sauberste Trennung, die ich bisher gemessen habe:

- **Untrainiert** folgt der Rang exakt der Datendimension (ρ=+0,986,
  p=3·10⁻¹⁸⁷) und ist von der Klassenzahl **vollständig unabhängig**
  (ρ=+0,002, p=0,98).
- **Trainiert** hängt er von **beiden** ab (d-Effekt bei fester Klassenzahl
  ρ=+0,957; Klassen-Effekt bei festem d ρ=+0,833) — und liegt drastisch
  niedriger (1,4–4,7 statt 3,7–165).

Die Klassenabhängigkeit existiert am untrainierten Netz nicht. Sie **entsteht
also beim Lernen**. Ohne diese Kontrolle hätte ich „der Rang folgt der
Datendimension" berichtet — was untrainiert stimmt und trainiert die halbe
Wahrheit ist.

Das Muster sieht aus wie ein **Minimum**: Bei 2 Klassen bleibt der Rang bei
etwa 1,5, egal wie hoch d ist. Bei d=2 bleibt er bei etwa 1,7, egal wie viele
Klassen. Die Schicht benutzt so viele Richtungen, wie der *kleinere* der beiden
Zwänge zulässt.

Die einfache Vorhersage „Rang = Klassenzahl − 1" trifft nicht (bei 10 Klassen
gemessen 2,7 statt 9). Es ist also keine Neural-Collapse-Aussage, sondern etwas
Schwächeres und Eigenes.

### Warum das die wichtigste Zahl des ganzen Teilprojekts ist

Sie erklärt **alle** Negativbefunde von B auf einmal. Rang 8, 64 und 256 haben
in B-008 praktisch dieselbe Genauigkeit geliefert, und mehr Budget hat nie
geholfen. Der Grund steht jetzt da: Die Aufgabe braucht nur rund sieben
Richtungen. **Der Rang war nie die bindende Beschränkung.** Deshalb konnte
keine strukturierte Familie durch mehr Kapazität gewinnen — sie hatten alle
längst genug.

## 4. August, abends — Roadmap nachgetragen, und eine doppelt vergebene ID

`ROADMAP.md` endete bis heute Abend bei der K-Serie (3. August) und hatte die
gesamte B-Serie B-003 bis B-014 nicht — also anderthalb Tage, in denen vier
Hauptaussagen gefallen und die beiden belegten Positivbefunde entstanden sind.
Schlimmer: die 819×-Tensor-Train-Aussage stand dort weiterhin unmarkiert unter
der Überschrift „Damit steht die Hauptaussage des Projekts", obwohl sie am
3. August zurückgezogen wurde. Nachgetragen sind jetzt die B-Serie, die
Gesamtbilanz mit sieben statt sechs Mechanismen, der Stand der laufenden Läufe,
und Zielsetzung, Phasenplan und Checkliste sind auf den echten Stand gebracht.
Die alten Einträge bleiben stehen, mit Marker an den überholten Stellen.

**Dabei eine Doppelvergabe gefunden: `B-003` bezeichnet zwei verschiedene
Experimente** — die DMRG-adaptive Rangzuweisung vom 2. August und den
Blockschicht-Vergleich vom 3. August (Seeds ab 1101). Entschieden ist es
faktisch durch die Rohdaten: unter `results/experiments/B-003__*` liegt der
Blockschicht-Lauf, und `qire_ai.audit` prüft gegen genau diese Datei. Der
DMRG-Lauf hat keine Rohdaten und muss deshalb die neue Kennung bekommen. Muss
vor der schriftlichen Arbeit erledigt sein, sonst ist jede Zitierung von
„B-003" mehrdeutig.

Dass mir das erst beim Nachtragen aufgefallen ist, passt ins Bild dieses Abends:
Die Fehler stecken nicht in den Messungen, sondern in der Buchhaltung drumherum
— unbelegte Zahlen, veraltete Zusammenfassungen, doppelte Kennungen. Die
Experimente selbst sind sauberer als die Dokumente, die sie beschreiben.

## 4. August, abends — Q-003-replication: Vorregistrierung geändert, vor dem Lauf

Dieser Eintrag steht bewusst **vor** den Ergebnissen. Er hält fest, was ich am
fertigen, aber noch nicht gestarteten `quantum_sim/hybrid_replication.py`
geändert habe, damit später niemand — auch ich nicht — die Änderungen für
Nachbessern nach Sichtung der Zahlen halten kann.

Anlass: `python -m qire_ai.audit` listet Q-001 bis Q-003 weiter ohne Rohdaten.
Der Replikationslauf soll das beheben. Beim Durchsehen gegen die sieben
Protokollregeln fielen vier Abweichungen auf. Die ersten beiden hätten den Lauf
kosten können, die anderen beiden seine Aussage.

**1. Rohdaten wurden erst ganz am Ende geschrieben.** Genau der Fehler, der bei
B-007 200 fertige Zellen gekostet hat. Jetzt wird nach jedem Datensatz
gespeichert (`rec.save()` in der Datensatzschleife), wie in
`ablation_unitarity`.

**2. Die Auswertung lief vor dem Speichern.** `recorder.save(analysis=analyse(recorder))`
wertet erst aus und speichert danach — ein Fehler in `analyse()` hätte zwei
Stunden Rechenzeit vernichtet, ohne eine einzige Zahl zu hinterlassen. Die
Auswertung wird jetzt abgefangen; die Rohdaten werden in jedem Fall geschrieben.

**3. Getestet wurde auf plain accuracy, die Grenze ist aber in balancierter
definiert.** Der Docstring nennt „plain accuracy statt balanced_accuracy" selbst
als einen von fünf Mängeln von Q-003 — der Code testete trotzdem auf plain
accuracy, während `MARGIN = 0,005` aus `equivalence.py` ausdrücklich in
balancierter Genauigkeit hergeleitet ist (gegen deren Seed-Streuung). Der
TOST hätte also eine Grenze für eine andere Größe geprüft als die, für die sie
begründet wurde. Alle Tests laufen jetzt auf `balanced_accuracy`, wie im
ganzen Rest des Projekts.

**4. Holm korrigierte über vier Vergleiche statt über die zwei
vorregistrierten.** Die Vorregistrierung sagt „PRIMÄR und P2, Holm über diese
beiden, alles andere ist Einordnung". Der Code korrigierte über alle vier und
machte die Primäraussage damit schwerer, als angemeldet war. Jetzt wird die
Familie eingehalten; die beiden Einordnungsvergleiche werden unkorrigiert und
als deskriptiv gekennzeichnet berichtet.

**Zusätzlich, als einzige inhaltliche Erweiterung: eine Positivkontrolle**
(Regel 6, als einzige der sieben Regeln bisher nicht umgesetzt). Neuer Arm
`hybrid_1layer` — dieselbe `HybridModel`-Klasse, nur eine Lage statt fünf, also
16 statt 80 Schaltungsparameter. Kein neuer Modellcode, nur ein anderer
Hyperparameter. Der Sinn: Kommt bei PRIMÄR ein Nullbefund heraus, muss
unterscheidbar sein, ob es wirklich keinen Unterschied gibt oder ob der Aufbau
für seine eigene Frage blind ist. Genau daran ist in B-004 der ganze frühere
Messaufbau gescheitert (ein Faktor 32 im Parameterbudget trennte sich nicht).
Trennt sich `hybrid` nicht von `hybrid_1layer`, wird jedes „unentschieden" des
Laufs als *nicht auflösbar* berichtet, nicht als Gleichwertigkeit.

Was **nicht** geändert wurde: die Modelldefinitionen. Sie kommen unverändert aus
`hybrid_final_control`; `classical_norm_deep` hat weiterhin exakt 67.848
Parameter in der Mitte. Nur das Protokoll ist neu — sonst wäre bei einem
abweichenden Ergebnis nicht zu trennen, ob es am Protokoll oder am Modell liegt.

Der Lauf startet automatisch, sobald B-007 die Kerne freigibt (Regel: immer nur
ein schwerer Lauf). Ergebnis offen.

### Fehlstart um 23:49, Abbruch nach acht Minuten — das Lernratengitter war zu eng

B-007 war um 23:48 fertig, der Wächter startete den Q-Lauf um 23:49. Nach acht
Minuten stand die Lernratensuche für MNIST auf der Konsole, und daran war etwas
falsch:

| Arm | gewählte Lernrate |
|---|---|
| hybrid | **0,03** |
| hybrid_no_entangle | **0,03** |
| hybrid_1layer | **0,03** |
| classical_quadratic | **0,03** |
| classical_quad_free | **0,03** |
| classical_norm_deep | 0,01 |

**Fünf von sechs Armen wählten den größten Wert des Gitters.** Ein Randwert
heißt: das Optimum kann außerhalb liegen, das Gitter schneidet es ab. Und der
sechste Arm — ausgerechnet `classical_norm_deep`, der Gegner im Primärvergleich
— lag als einziger im Inneren. Damit wäre genau die Kernaussage asymmetrisch
gemessen worden: der Quantenarm womöglich unterhalb seines Optimums, der
klassische an seinem.

Die Richtung der Verzerrung wäre konservativ gewesen (hätte `hybrid` trotzdem
gewonnen, wäre die Aussage sicher). Aber bei einem negativen Ausgang hätte
niemand entscheiden können, ob es am Mechanismus oder am Gitter lag — und das
ist keine Datenlage, auf die ich eine Kernaussage stellen will.

Also abgebrochen, `LR_GRID` auf (0,003 / 0,01 / 0,03 / **0,1**) erweitert und
neu gestartet. Verloren: acht Minuten. Dazu eine dauerhafte Absicherung ins
Skript geschrieben — `run()` meldet ab sofort jeden Arm, der am Gitterrand
landet, und legt die Liste in der Konfiguration der Rohdaten ab.

Das ist dieselbe Fehlerklasse wie die feste Lernrate aus B-004, nur eine Stufe
subtiler: dort war die Rate für alle Arme gleich, hier war der *Suchbereich* zu
klein. Regel 1 sagt „Lernrate je Arm suchen" — sie sagt nicht, dass der
Suchbereich das Optimum enthalten muss. Jetzt schon.

**Ein Nebenbefund aus der abgebrochenen Suche**, der zum Bild aus Q-003 passt:
`classical_quadratic` (orthogonale Matrix) erreichte in der Suche nur 0,67,
`classical_quad_free` (freie Matrix) dagegen 0,89. Die *orthogonale* Variante —
also gerade die normerhaltende, quantenhafte Beschränkung — ist deutlich
schlechter als die freie. Genau das stand schon im ursprünglichen Q-003. Es ist
noch kein Ergebnis, nur eine Tuning-Zahl auf zwei Seeds, aber es repliziert die
Richtung.

### Und noch etwas, während der Rechner belegt war: die Kurzfassung stimmt nicht mehr

Ich habe `KURZFASSUNG_ENTWURF.md` gegen den Belegstand geprüft. Der Entwurf ist
vom 2. August — davor liegen zwei Tage, an denen mehrere Hauptaussagen gefallen
sind. Der Entwurf hat davon nichts mitbekommen. Gefunden:

- **Ein zurückgezogener Befund steht noch als positives Hauptergebnis drin.**
  Die 819×-Blockschicht (61.440 Parameter, 0,9451) füllt den Abschnitt „Zwei
  positive Nebenbefunde / Erstens" — obwohl sie am 3. August unter „Was ich
  zurückziehe" gelandet ist. Das ist der gefährlichste Fund: eine Jury, die das
  Tagebuch neben die Kurzfassung legt, findet den Widerspruch sofort.
- **Drei Zahlen sind unbelegt**: die quadratische Auslese (2.048 gegen 67.848)
  und die 80-Parameter-Aussage stammen aus Q-001 bis Q-003, für die es keine
  Rohdaten gibt. Sie werden gerade nachgerechnet.
- **Zwei Tabellenzeilen sind nicht mehr haltbar** (Tensor-Train aus dem blinden
  Aufbau; Unitarität mit p=0,32).
- **Die Zählung war in sich widersprüchlich**: Titel „vier Mechanismen",
  Überschrift „alle sechs", Tabelle sechs Zeilen, tatsächlich sieben — die
  Quantenkerne fehlten ganz. Korrigiert, Quantenkern-Zeile ergänzt.
- **Ein Sachfehler**: „Alle Experimente auf CPU, einfädig" — ab A-002 laufen sie
  mit sechs Threads. Stand als Korrekturbedarf schon im Tagebuch, war in der
  Kurzfassung aber nie nachgezogen. Korrigiert.
- **Die beiden am besten belegten positiven Befunde fehlen ganz**: die
  Rang-64-Projektion (B-011, erster echter Äquivalenznachweis des Projekts) und
  das Effektivrang-Ergebnis (B-014). Beide sind besser abgesichert als alles,
  was aktuell im Entwurf steht.

Die Zählung und den Thread-Fehler habe ich direkt korrigiert. Die inhaltlichen
Stellen habe ich **nicht** überschrieben, sondern mit einem Belegstand-Block
oben und Warnmarkern an den betroffenen Stellen versehen — drei der Zahlen
entscheiden sich erst in dieser Nacht, und vorher umzuschreiben hieße, das
Ergebnis vorwegzunehmen.

Die Lehre daraus ist dieselbe wie beim Replikationsskript: Dokumente altern
still. Das Tagebuch wird gepflegt, weil ich darin schreibe; die Kurzfassung
nicht, weil ich sie nur gelegentlich anfasse. Vor der Anmeldung muss sie noch
einmal vollständig gegen die Gesamtbilanz gelesen werden, nicht nur ergänzt.

## 4./5. August — B-007: Befund 4 fällt, aber es folgt nicht das Gegenteil

Zwei frühere Läufe hatten denselben Vergleich gerechnet — Orthogonalität gegen
Spektralnormalisierung, gleiche Architektur, gleiche Blockgröße, gleiche Tiefe,
je 10 Seeds — und waren zu gegenteiligen Schlüssen gekommen (p=0,322 gegen
p=0,037). Der Grund war Teststärke: Bei der beobachteten Effektstärke hatten
beide etwa Münzwurf-Wahrscheinlichkeit, den Effekt überhaupt zu finden.

B-007 war der vorregistrierte Entscheidungslauf: 20 Seeds, Lernrate je Arm auf
eigenen Tuning-Seeds, zwei Datensätze, TOST, Holm.

### MNIST — eindeutig

| | Differenz | Siege | p (Holm) | Cohen's d |
|---|---|---|---|---|
| Tiefe 1 | +0,0025 | 17/20 | 0,0020 | +0,84 |
| Tiefe 3 | +0,0051 | 17/20 | 0,0001 | +1,18 |

**„Kein Unterschied" ist nicht haltbar.** Der Effekt existiert, ist konsistent
über beide Tiefen und übersteht die Holm-Korrektur deutlich.

Ein Detail, das nur der Äquivalenztest sichtbar macht: Bei Tiefe 1 ist der
Unterschied gleichzeitig **signifikant von null verschieden** (p=0,0020) **und
signifikant kleiner als ±0,005** (TOST p=0,0007). Beides stimmt und ist kein
Widerspruch — der Effekt ist real, aber kleiner als die Grenze, die ich vorab
als praktisch bedeutsam festgelegt hatte. Erst bei Tiefe 3 übersteigt er sie
(+0,0051, TOST p=0,54).

Passend dazu: `plain` wird mit mehr Tiefe schlechter (0,9409 → 0,9344),
`orthogonal` nicht (0,9497 → 0,9500). Die Normerhaltung zahlt sich also gerade
dann aus, wenn mehr Schichten gestapelt werden — mechanistisch plausibel, und
es erklärt, warum Tiefe 1 den Effekt fast verschluckt.

### Fashion-MNIST — trägt es nicht

| | Differenz | p |
|---|---|---|
| Tiefe 1 | +0,0048 | 0,114 |
| Tiefe 3 | **−0,0045** | 0,133 |

Die Richtungen widersprechen sich zwischen den Tiefen, keine ist signifikant.

### Das Urteil, und warum ich es nicht schönrechne

Das vorher festgelegte Kriterium verlangte Signifikanz auf MNIST **und**
gleiche Richtung auf dem zweiten Datensatz. Der zweite Teil fällt durch. Also:
**HU — unentschieden.**

Das ist unbequem, weil die MNIST-Zahlen für sich genommen die stärksten sind,
die dieses Projekt zu Orthogonalität hat. Genau deshalb war das Kriterium
vorher festgelegt. Der belastbare Satz lautet:

> Auf MNIST wirkt Orthogonalität nachweislich über bloße Skalenkontrolle
> hinaus. Auf Fashion-MNIST nicht. Ein allgemeiner Befund ist es damit nicht.

Für die Gesamtbilanz heißt das: Zeile 4 in ihrer bisherigen Form („kein
Unterschied, p=0,32") ist **widerlegt**, aber sie wird nicht durch „Unitarität
wirkt" ersetzt, sondern durch eine datensatzabhängige Aussage.

### Nebenbefund

`weight_decay` kollabiert bei Tiefe 3 auf **exakt** 0,1000 mit Streuung exakt
null — auf beiden Datensätzen. Rateniveau bei zehn Klassen. Dasselbe Muster wie
in B-002. Die Auswertung markiert solche Arme automatisch über die Prüfung auf
Streuung null; der Arm geht nicht stillschweigend in die Bilanz ein.

## 5. August, nachts — die Nacht, in der die Kernaussage hielt

Drei Ergebnisse in einer Nacht. Der Reihe nach.

### B-007: Befund 4 bleibt unsicher — aber mit einer Spur

Der Lauf gehört zur B-Sitzung, das Ergebnis gehört hierher: Orthogonalität ist
das klassische Gegenstück unitärer Quantengatter, also der quanteninspirierte
Mechanismus selbst.

| MNIST | orthogonal − spectral | Siege | Holm |
|---|---|---|---|
| Tiefe 1 | +0,0025 | 17/20 | 0,0020 |
| Tiefe 3 | +0,0051 | 17/20 | 0,0001 |

Auf MNIST schlägt Orthogonalität die Spektralnormalisierung also **auf beiden
Tiefen signifikant** — das Gegenteil des ursprünglichen p=0,32. Auf
Fashion-MNIST repliziert die Richtung aber nicht (+0,0048 bei Tiefe 1, −0,0045
bei Tiefe 3). Nach dem vorher festgelegten Kriterium: **HU, unentschieden.**
Befund 4 bleibt als unsicher markiert, und ich deute das nicht um.

**Eine Beobachtung, die ich festhalte, ohne sie als Ergebnis zu führen:** Auf
MNIST *wächst* der Effekt mit der Tiefe, +0,0025 → +0,0051. Genau das würde man
erwarten, wenn der Unterschied zwischen „größter Singulärwert = 1" und „alle
Singulärwerte = 1" sich erst durch wiederholte Anwendung aufbaut. Zwei Tiefen
sind kein Trend, aber es ist der Anlass für die Idee, den Vergleich bei Tiefe
16 bis 64 zu wiederholen (`FORSCHUNGSIDEEN.md`, A4).

Der `weight_decay`-Arm kollabierte zum **dritten Mal** auf exakt 0,1000 bei
Streuung exakt 0 (Fashion-MNIST, Tiefe 3). Nach B-002 und B-007 ist das kein
Zufall mehr, sondern ein reproduzierbarer Mechanismus, den ich bisher dreimal
als „ungeklärte Auffälligkeit" abgelegt habe.

### Q-003-replication: die Kernaussage von Projekt A hält

Zwei Anläufe, beide vollständig. Der erste (Gitter bis 0,03) wurde abgebrochen,
weil fünf von sechs Armen am Gitterrand saßen — er war zu diesem Zeitpunkt aber
schon durchgelaufen, und **die Daten überlebten, weil seit gestern nach jedem
Datensatz gespeichert wird.** Die Regel, die B-007 einmal 200 Zellen gekostet
hat, hat hier eine ganze Messung gerettet.

Der zweite Anlauf mit Gitter bis 0,1:

| MNIST | balanced | Mitte | lr |
|---|---|---|---|
| **hybrid** | **0,9339** | **80** | 0,1 |
| hybrid_1layer | 0,9337 | 16 | 0,1 |
| classical_quad_free | 0,9213 | 2.048 | 0,03 |
| classical_norm_deep | 0,8407 | 67.848 | 0,01 |

**PRIMÄR: +0,0932 (MNIST) und +0,0599 (Fashion-MNIST), Holm p=0,0001 auf
beiden.** 15 Seeds, Lernrate je Arm, balancierte Genauigkeit, Rohdaten
gespeichert. Q-001 und Q-002 sind damit zum ersten Mal belegt — und der Effekt
ist größer als die ursprünglichen +0,0707.

Die Warnung, die ich vor dem zweiten Anlauf eingebaut habe, feuerte erneut:
`hybrid` wählte wieder den Randwert 0,1. Der wahre Vorteil könnte also noch
größer sein. Die Verzerrung geht zugunsten der Aussage, deshalb ist sie sicher —
aber sie ist nicht ausgereizt.

**Zwei Dinge, die meine Erklärung ändern:**

- **Verschränkung trägt nichts bei.** hybrid gegen hybrid_no_entangle: −0,0015,
  TOST weist **Gleichwertigkeit innerhalb ±0,005** nach. Das ist kein „nicht
  signifikant" mehr, sondern ein positiver Nachweis.
- **Tiefe trägt nichts bei.** hybrid (80 Parameter, 5 Lagen) gegen
  hybrid_1layer (16 Parameter, 1 Lage): +0,0002, ebenfalls gleichwertig. Die
  Positivkontrolle ist damit durchgefallen — nicht weil der Aufbau blind wäre,
  sondern weil es den Unterschied nicht gibt.

**Also: 16 Parameter schlagen 67.848.** Und weder Superposition über viele
Lagen noch Verschränkung ist der Grund. Was wirkt, ist die Struktur —
normieren, normerhaltend drehen, quadratisch auslesen.

**Q-003 repliziert nicht.** Der damalige Kernvergleich (hybrid gegen
classical_quad_free) stand nicht in der vorregistrierten Liste; nachgerechnet
ergibt er +0,0044 (p=0,49) und +0,0030 (p=0,30). Das Vorzeichen hat sich
umgedreht, aber nicht signifikant: die beiden sind bei 15 Seeds nicht
unterscheidbar. Die Aussage „der klassische Nachbau schlägt die Schaltung" ist
damit hinfällig.

**Was bei jeder Nennung dazugehört:** Die 848 gelten für die **Mitte**. Die
Gesamtmodelle liegen bei 201.130 gegen 203.098 gegen 268.898 Parameter, weil
der klassische Encoder (784 → 256) in jedem Arm steckt.

### Die Architektur auf klassischer Hardware — und ein Negativbefund

`quantum_sim/butterfly_layer.py`: die Schaltung als klassisches Butterfly-Netz.
Kein Simulator, sondern dieselbe Architektur als Rechenverfahren. Ein
Ein-Qubit-Gatter verknüpft die Amplitudenpaare, die sich in Bit q
unterscheiden — über alle n Qubits ist das die Schmetterlingsstruktur der
schnellen Fouriertransformation, also O(d log d) statt O(d²).

**Verifiziert:** identische Funktion zum Simulator, größte Abweichung 7·10⁻⁷
über 30 Konfigurationen. Beim ersten Anlauf fiel die Prüfung durch — ich hatte
Ry·Rz statt Rz·Ry zusammengefasst, und die beiden kommutieren nicht. Ohne diese
Verifikation wäre jede folgende Zeitmessung wertlos gewesen.

Der eigentliche Befund steckt in der Parameterbilanz:

| Qubits | Dim | Quantenschaltung | Butterfly ungeteilt | dicht |
|---|---|---|---|---|
| 8 | 256 | **80** | 10.240 | 65.536 |
| 12 | 4.096 | **120** | 245.760 | 16.777.216 |

Ein gewöhnliches Butterfly-Netz gibt jedem Paar einen eigenen Winkel. Die
Quantenschaltung teilt einen Winkel über die **ganze Qubit-Ebene**, weil ein
Qubit-Gatter physikalisch auf ein Qubit wirkt, unabhängig von der Größe des
Zustandsraums. Das ist der quantenabgeleitete Anteil der Architektur, und auf
diese Konstruktion führt der klassische Weg nicht.

**Die Zeitmessung ist aber ein Negativbefund.** Gegen eine dichte Schicht:
0,00× / 0,01× / 0,04× / 0,13× bei 6 / 8 / 10 / 12 Qubits — die Butterfly-
Umsetzung ist 7- bis 250-fach **langsamer**, obwohl sie bei d=4096 rund
340-fach weniger Arithmetik braucht. Grund: 60 sequentielle Kleinoperationen
mit Umformungen gegen einen einzigen BLAS-Aufruf, dazu Faktor 4 für komplexe
Zahlen. **Das ist zum fünften Mal in diesem Projekt derselbe Fehler: weniger
Operationen heißt nicht weniger Zeit.**

### Und dann kippt es doch — der Übergang liegt bei 14 Qubits

Der Rückstand schrumpfte systematisch, also habe ich weiter gemessen statt zu
extrapolieren. Batch 64, 5 Lagen, einfädig:

| Qubits | Dim | Butterfly | dicht | Faktor | Gewichte dicht | Parameter Butterfly |
|---|---|---|---|---|---|---|
| 10 | 1.024 | 71,8 ms | 2,8 ms | 0,04× | 0,00 GB | 100 |
| 12 | 4.096 | 259,3 ms | 52,3 ms | 0,20× | 0,07 GB | 120 |
| 13 | 8.192 | 487,9 ms | 237,6 ms | 0,49× | 0,27 GB | 130 |
| **14** | **16.384** | **1328,9 ms** | **2085,4 ms** | **1,57×** | 1,07 GB | **140** |
| 15 | 32.768 | 2610,9 ms | nicht baubar | — | 4,29 GB | 150 |
| 16 | 65.536 | 5816,2 ms | nicht baubar | — | 17,18 GB | 160 |

**Bei 14 Qubits ist die quantenabgeleitete Architektur 1,57-fach schneller als
eine dichte Schicht gleicher Breite** — bei 140 statt 268 Millionen Parametern.
Der Übergang liegt zwischen 13 und 14 Qubits, also früher als meine
Extrapolation (15–16) vermuten ließ.

**Der zweite Befund ist womöglich der wichtigere:** Ab 15 Qubits lässt sich die
dichte Schicht auf diesem Rechner gar nicht mehr bauen — ihre Gewichtsmatrix
bräuchte 4,3 beziehungsweise 17,2 GB. Die Butterfly-Variante rechnet dieselbe
Breite mit **150 beziehungsweise 160 Zahlen**. Das ist kein Geschwindigkeits-,
sondern ein Machbarkeitsunterschied: Es gibt Schichtbreiten, die diese
Architektur darstellen kann und eine dichte schlicht nicht.

### Nachgemessen mit sieben Messreihen — und der Wert war zu niedrig

Der erste Durchgang hatte fünf Wiederholungen je Punkt und keine
Streuungsangabe. Bei Batch 4 hat in diesem Projekt schon einmal ein Einzelwert
getrogen (die 30,1×, die an einer Eigenart der *Referenz* lagen), deshalb
dieselbe Vorgehensweise wie damals: sieben unabhängige Messreihen, innerhalb
jeder Reihe **abwechselnd** gemessen, damit Drift oder Fremdlast beide Arme
gleich trifft, und der Faktor je Reihe **gepaart** gebildet statt als Verhältnis
zweier Mediane.

| Qubits | Dim | Butterfly (Median) | dicht (Median) | Faktor | Reihen schneller | Spanne |
|---|---|---|---|---|---|---|
| 10 | 1.024 | 72,8 ms | 3,2 ms | 0,04× | 0/7 | 0,04–0,05 |
| 12 | 4.096 | 185,8 ms | 40,4 ms | 0,21× | 0/7 | 0,20–0,23 |
| 13 | 8.192 | 578,3 ms | 244,5 ms | 0,40× | 0/7 | 0,31–0,50 |
| **14** | **16.384** | **1220,9 ms** | **2460,6 ms** | **1,99×** | **7/7** | **1,75–2,10** |
| 15 | 32.768 | 3135,2 ms | nicht baubar | — | — | — |

**Der Übergang liegt zwischen 13 und 14 Qubits, und bei 14 ist die
Butterfly-Architektur in ALLEN sieben Messreihen schneller** — Median 1,99×,
Spanne 1,75 bis 2,10. Das Intervall kommt der 1,0 nicht nahe.

Bemerkenswert ist, dass der Wert sich verschoben hat: erst 1,57×, jetzt 1,99×.
Die Richtung hielt, der Betrag lag um 27 % daneben. Genau dafür waren die
Wiederholungen da, und genau deshalb steht der Einzelwert von vorhin nicht in
der Arbeit.

Belastbar formuliert: **Unterhalb von rund 8.000 Dimensionen verliert die
Architektur deutlich, ab 16.384 ist sie etwa doppelt so schnell wie eine dichte
Schicht gleicher Breite** — bei 140 statt 268 Millionen Parametern.

Was weiterhin fehlt: die **Genauigkeit** bei diesen Breiten. Ein
Geschwindigkeitsvorteil ohne Genauigkeitsnachweis ist wertlos — genau der
Fehler, den ich beim Tensor-Train-Streaming schon einmal beinahe gemacht hätte
und der dort ausdrücklich als offener Punkt notiert war.

### Und beim Aufbau des Genauigkeitslaufs fällt der Vorteil zur Hälfte weg

Der erste Entwurf verglich bei **gleicher Breite** — 140 Parameter gegen 268
Millionen. Der Vortest zeigte, dass diese Frage ihre Antwort schon kennt, und
dass mein Aufbau zusätzlich unfair war: der dichte Arm bekam die Normierung
nicht, die zur Amplituden-Kodierung gehört, und landete bei 0,1295, also auf
Rateniveau. Mit gleicher Normierung und passender Lernrate erreicht er 1,0000.
**Hätte ich das nicht geprüft, wäre ein „Butterfly schlägt dicht um 78
Prozentpunkte" ins Tagebuch gewandert** — ein reines Artefakt meines Aufbaus.

Der richtige Vergleich ist deshalb **gleiches Zeitbudget statt gleicher
Breite**, wie in B-004 die exakt angeglichenen 524.288 Parameter. Bei der
empirischen Kalibrierung dafür kam heraus:

| | Breite | Zeit je Trainingsschritt |
|---|---|---|
| Butterfly | 16.384 | 1957 ms |
| dicht, isochron | **14.896** | 1908 ms |

**Der Vorteil schrumpft im Training auf rund 1,18× — gemessen wurden vorher
1,99×.** Der Unterschied: die 1,99× galten für **reine Inferenz**. Im Training
kommen Rückwärtslauf und Optimierer dazu, und der Rückwärtslauf durch 70
sequentielle Kleinoperationen kostet überproportional.

Das ist ein eigenständiger Befund, und für die Zielfrage der wichtigere: **Der
Geschwindigkeitsvorteil dieser Architektur ist ein Inferenzvorteil, kein
Trainingsvorteil.** Bei gleichem Zeitbudget darf der dichte Arm deshalb fast
dieselbe Breite benutzen (14.896 statt 16.384, nur 9 % weniger) — und hat dort
weiterhin 222 Millionen Parameter gegen 140.

Der Lauf `quantum_sim/isotime_accuracy.py` (Q-009) ist damit gebaut und
kalibriert: drei Schwierigkeitsstufen, Auflösungsprüfung, vorregistriert. Er
beantwortet die einzige Frage, die nach dieser Kalibrierung noch offen ist —
ob 140 Parameter bei gleichem Rechenbudget mit 222 Millionen mithalten können,
solange die Aufgabe wenig Kapazität verlangt, und wo die Grenze liegt.

Damit ist der Negativbefund von vor zwei Stunden nicht falsch, sondern
**auf einen Bereich begrenzt**: unterhalb von etwa 8.000 Dimensionen verliert
die Architektur, oberhalb von 16.000 gewinnt sie. Genau dieselbe Form von
Aussage wie beim Tensor-Train-Streaming — nur diesmal zugunsten des
quantenabgeleiteten Verfahrens.

### Was diese Nacht unterm Strich geändert hat

Vorher: sieben Abgrenzungstests, alle negativ, und eine Kernaussage ohne
Rohdaten. Jetzt: dieselbe Kernaussage **belegt**, auf zwei Datensätzen, mit
einem Protokoll, das dieses Projekt sich selbst erarbeitet hat — und mit einer
präziseren Erklärung, denn Verschränkung und Tiefe fallen als Ursache aus.

Und ein neuer, ehrlicher Negativbefund auf der Zeitachse, mit klar benannter
Bedingung, unter der er sich drehen könnte.

## 6. August — Befund 1 und Befund 6 sind derselbe Mechanismus an zwei Stellen

Beim Durchsehen der alten Befunde ist mir etwas aufgefallen, das mir ein halbes
Jahr lang entgangen ist, obwohl beide Zahlen die ganze Zeit im Tagebuch standen.

**Befund 1** hat das Born-Regel-Gating verworfen: |Amplitude|² statt Softmax,
kein Unterschied über 648 gematchte Paare (p=0,25). Meine Begründung damals:
beide bilden auf dieselbe Wahrscheinlichkeitssimplex ab, also dieselbe
Funktionsklasse — ich hätte die Freiheitsgrade nicht eingeschränkt, sondern nur
dieselbe Sache anders aufgeschrieben.

**Q-003** hat später herausgearbeitet, dass der Wirkmechanismus der
Quantenschaltung die **quadratische Auslese** ist: ⟨Z⟩ summiert über
|Amplitude|².

Das ist dieselbe Operation. Einmal wirkungslos, einmal tragend.

### Der Unterschied ist die Position im Netz

Als **Gate** wird das Ergebnis anschließend auf Summe 1 normiert. Die
Quadratur konkurriert dort mit Softmax, das dieselbe Abbildung leistet — und
die Normierung wirft gerade die Information weg, die das Quadrieren erzeugt
hätte. Kein Wunder, dass sich nichts zeigt.

Als **Auslese** ist die Quadratur die Nichtlinearität selbst. Sie erzeugt
Merkmale für den nachfolgenden linearen Kopf, und sie steht dort gegen
Linear/Tanh — eine ganz andere Funktionsklasse. Dort trägt sie.

### Und wie viel sie trägt, lässt sich aus den vorhandenen Rohdaten abzählen

`classical_quad_free` ist die rein klassische Umsetzung genau dieser drei
Schritte: normieren → freie Matrix → quadrieren. Kein Quantenbegriff darin.
Gegen `classical_norm_deep` (Linear/Tanh) gehalten misst der Abstand, was die
Quadratur **allein** beiträgt:

| | MNIST | Fashion-MNIST |
|---|---|---|
| Schaltung gegen Linear/Tanh (gesamt) | +0,0932 | +0,0599 |
| davon durch die Quadratur allein | **+0,0806 (86,5 %)** | **+0,0548 (91,5 %)** |
| Rest, der der Schaltung bleibt | +0,0126 (13,5 %) | +0,0051 (8,5 %) |

Beide Male 15 von 15 Seeds, p=0,0001.

**Rund neun Zehntel des Vorteils der Quantenschaltung sind die Born-Regel als
Nichtlinearität — und die ist klassisch in vier Zeilen implementierbar.**

Das ist eine Zerlegung eines bestehenden Ergebnisses, kein neuer Test; der
Vergleich `classical_quad_free` gegen `classical_norm_deep` war nicht
vorregistriert. Bei 15/15 Seeds und einem Effekt dieser Größe halte ich die
Richtung trotzdem für belastbar, und ich führe sie als beschreibend, nicht als
Befund.

### Warum mir das wichtig ist

Erstens verbindet es zwei meiner sieben Abgrenzungstests zu **einer** Aussage:

> Ein Mechanismus trägt nichts bei, wo seine Wirkung anschließend wieder
> herausnormiert wird, und er trägt viel bei, wo er als Nichtlinearität
> Merkmale erzeugt. Die Position im Netz entscheidet, nicht die Operation.

Zweitens erklärt es, warum ich Befund 1 zu früh abgehakt habe. Ich habe die
Born-Regel getestet, an *einer* Stelle, und daraus geschlossen, sie trage
nichts bei. Getestet hatte ich aber nur diese eine Stelle.

Drittens verschiebt es, was der eigentlich quantenspezifische Anteil ist: die
verbleibenden 8,5 bis 13,5 Prozent. Genau die prüft Q-010 vorregistriert nach.

## 6. August — Das Konzentrationsprinzip ist nicht nur beobachtet, es ist ausrechenbar

Mein „wichtigster theoretischer Ertrag" war bisher eine **abgelesene**
Regelmäßigkeit: die mittlere Kernähnlichkeit halbiert sich je Qubit. Eine
abgelesene Regelmäßigkeit ist eine Beobachtung. Eine hergeleitete ist eine
Vorhersage — und nur die kann falsch sein.

### Die Herleitung

Zwei unabhängige, Haar-zufällige Zustände im 2ⁿ-dimensionalen Raum. Sei |ψ⟩
fest und |φ⟩ zufällig. Aus Symmetriegründen ist der Erwartungswert der
Projektion auf **jede** der d orthogonalen Richtungen gleich, und sie summieren
sich wegen der Normierung zu 1. Also:

> **E[ |⟨ψ|φ⟩|² ] = 1/d = 2⁻ⁿ**

Kein Quantenphänomen im engeren Sinn, sondern reine Geometrie: Ein zusätzliches
Qubit verdoppelt die Dimension, also halbiert sich die erwartete Ähnlichkeit.

### Gegen die eigenen Messungen gehalten

| Qubits | gemessen | Haar 1/d | Verhältnis |
|---|---|---|---|
| 2 | 0,349 | 0,250 | 1,40 |
| 3 | 0,162 | 0,125 | 1,30 |
| 4 | 0,082 | 0,0625 | 1,31 |
| 6 | 0,024 | 0,0156 | 1,54 |

Anpassung von log₂(Ähnlichkeit) = a + b·n:

> **Exponent b = −0,958** (Theorie: −1,000) — Abweichung **4,2 %**
> **Vorfaktor = 1,24** (Theorie: 1,00), **R² = 0,9972**

### Was das schärfer sagt als „die Ähnlichkeit fällt"

Die Zustände, die meine Merkmalsabbildung erzeugt, werden **genauso schnell
orthogonal wie rein zufällige Zustände.** Die Abbildung erhält also praktisch
keine Struktur der Daten — sie verteilt sie so, wie es der Zufall täte.

Der Vorfaktor 1,24 ist der einzige Rest von Struktur: echte Daten sind sich um
diesen Faktor ähnlicher als Zufall. Entscheidend ist, dass er **konstant in n**
ist. Die Struktur wächst nicht mit, während der Raum exponentiell wächst. Genau
daran scheitert das Lernen, und jetzt steht eine Zahl dahinter statt einer
Erzählung.

**Falsifizierbare Vorhersagen** (`quantum_sim/theory.py`): bei 5 Qubits 0,0448,
bei 8 Qubits 0,0061, bei 10 Qubits 0,0016. Weicht eine Messung deutlich ab,
ist das Prinzip falsch.

*Einschränkung, die dazugehört: Für die K-Serie existieren keine gespeicherten
Rohdaten. Diese vier Werte stammen aus dem Tagebuch und sind nicht per
`q_audit` nachprüfbar. Ein Wiederholungslauf, der zugleich die Vorhersage für
5, 8 und 10 Qubits testet, würde beides erledigen.*

### Und was die Gewichtsteilung mathematisch ist

| Qubits | geteilt | ungeteilt | SO(d) | Anteil |
|---|---|---|---|---|
| 8 | 80 | 10.240 | 32.640 | 0,245 % |
| 12 | 120 | 245.760 | 8.386.560 | 0,0014 % |

Die geteilte Schaltung ist eine Untermannigfaltigkeit der Dimension 2·n·L in
einer Gruppe der Dimension d(d−1)/2. Bei 8 Qubits: 80 Richtungen in einem Raum
von 32.640.

**Der entscheidende Punkt ist aber nicht, dass sie kleiner ist.** Der
hardware-efficient ansatz mit CNOT-Verschränkung ist für L → ∞ **universell**.
Die Teilung schränkt also nicht ein, *was* erreichbar ist, sondern nur, *wie
viele Parameter der Weg dorthin kostet.*

Daraus folgt eine prüfbare Aussage für Q-014: Die Teilung darf bei genügend
Lagen nichts kosten. Kostet sie bei wenigen Lagen etwas, ist das eine Aussage
über die nötige **Tiefe**, nicht über Fähigkeit.

Und hier schließt sich ein Kreis zu B-014: Dort wurde gemessen, dass die
Aufgabe nur rund **sieben Richtungen** braucht. Eine 80-dimensionale
Untermannigfaltigkeit reicht dafür mit großem Abstand — was erklärt, warum
Q-003-replication keinen Unterschied zwischen 16 und 80 Parametern fand. Nicht
weil die Schaltung schlecht wäre, sondern weil die Aufgabe ihre Kapazität nicht
ausschöpft.

### Zwei Stellen, an denen Theorie und Messung sich reiben

**Die Quadratur sollte nicht so viel bringen.** 86,5 % bzw. 91,5 % des
Vorteils entfallen auf sie — aber eine Quadratur ist nur eine Nichtlinearität
unter vielen. Vermutung: Sie bildet mit der L2-Normierung davor ein Paar.
Normieren + Quadrieren ergibt eine Größe mit fester Summe, also eine
**strukturelle Einschränkung**, keine bloße Nichtlinearität — und genau
Einschränkungen sind laut meinem eigenen Prüfmaßstab das, was beitragen kann.
*Prüfbar:* Tanh **mit** vorheriger Normierung gegen Quadratur mit Normierung.
Bleibt der Abstand, liegt es an der Quadratur; verschwindet er, an der Normierung.

**Die Verschränkung sollte nicht nichts bringen.** Ohne CNOT ist der Zustand
ein Produktzustand, ⟨Z_q⟩ faktorisiert vollständig, und die Schaltung berechnet
für jedes Qubit unabhängig eine Funktion von 8 der 256 Amplituden. Dass das
gleichwertig ist, heißt: **die Aufgabe braucht keine Korrelation zwischen den
Ausgabekanälen.** Das ist kein Argument gegen Verschränkung, sondern eines
dafür, dass MNIST sie nicht verlangt. Eine Aufgabe mit erzwungener
Kanalkorrelation — etwa Parität über Merkmalsgruppen — wäre der faire Test, und
der fehlt.

Beides sind Vorhersagen, keine Ergebnisse, und gehören als offene Fragen in die
Arbeit.

## 6. August — Was die quadratische Auslese wirklich berechnet, und ein Fehler in meiner Kontrolle

Die quadratische Auslese erklärt 87–92 % des Vorteils meiner Quantenschaltung.
Benutzt habe ich sie seit dem 2. August — **charakterisiert habe ich sie nie.**
Das habe ich nachgeholt, und es ändert die Deutung des ganzen Q-Kapitels.

### Sie ist ein Rayleigh-Quotient

Die Kette normieren → orthogonale Drehung U → quadrieren → gruppenweise
summieren berechnet für jeden Kanal q exakt

> **out_q(x) = xᵀ A_q x / (xᵀ x)   mit   A_q = Uᵀ P_q U**

Numerisch bestätigt, Abweichung 1,0·10⁻⁷ über 200 Zufallseingaben. Und A_q ist
nicht irgendeine Matrix: **Eigenwerte exakt {0,1}, Rang d/n, Spur d/n** — eine
echte orthogonale Projektion.

Die Auslese berechnet also für jeden Kanal den **Anteil der Eingabe-Energie in
einem orthogonalen Unterraum**. Die Ausgaben sind nichtnegativ und summieren
sich auf 1,000000 (Streuung 1,3·10⁻⁷). Sie ist skaleninvariant: out(c·x) =
out(x).

Zum Vergleich Linear/Tanh, dieselbe Prüfung: nicht skaleninvariant
(max|f(x)−f(3x)| = 0,45), Summe der Ausgaben −0,15 mit Streuung 1,93.

**Damit erfüllt die Schaltung meinen eigenen Prüfmaßstab.** Er lautet: „Ein
Ansatz kann nur dann etwas beitragen, wenn er eine strukturelle Einschränkung
auferlegt, die klassisch nicht ohnehin verfügbar ist." Genau das tut sie — sie
zwingt die Ausgabe in eine Wahrscheinlichkeitsverteilung über orthogonale
Unterräume. Ich habe die Schaltung siebenmal gegen Kontrollen gestellt, ohne je
auszurechnen, was sie tut.

### Und dabei fällt ein Fehler in meiner Kontrolle auf

`QuadraticReadout` in `hybrid_final_control.py` benutzt **blockdiagonale**
orthogonale Matrizen — 32 Blöcke à 8×8 bei 256 Dimensionen. Information
überquert dort nie eine Blockgrenze. Die Quantenschaltung mischt dagegen
**global**: der CNOT-Ring koppelt alle Qubits.

Die beiden sind also **nicht dieselbe Funktionsklasse.** Und ausgerechnet
B-004 hat gemessen, wie groß der Unterschied zwischen blocklokaler und globaler
Mischung ist.

Damit fügen sich alle drei gemessenen Zahlen zusammen:

| | Mischung | Freiheitsgrade | MNIST |
|---|---|---|---|
| classical_quadratic | **blocklokal**, orthogonal | 2.048 | 0,8881 |
| classical_quad_free | **blocklokal**, frei | 2.048 | 0,9213 |
| hybrid | **global**, orthogonal | **80** | **0,9339** |

Die Schaltung gewinnt, weil sie global mischt. **Keine meiner beiden Kontrollen
tut das.** Die Aussage „der klassische Nachbau schlägt die Schaltung" aus Q-003
verglich also gegen eine strukturell schwächere Familie — derselbe Fehlertyp
wie die blinde Architektur in B-004, nur an einer anderen Stelle.

### Die Kontrolle, die fehlt

Der Vorteil kann zwei Quellen haben, und bisher sind sie vermischt:

1. **die Funktionsklasse** — global statt blocklokal mischen
2. **die Parametrisierung** — 2·n·L Winkel für eine globale Drehung statt
   d(d−1)/2 freier Parameter

**Nur (2) ist quantenspezifisch.** (1) kann man klassisch auch haben, es kostet
dann 32.640 statt 80 Parameter.

`quantum_sim/global_orthogonal.py` (Q-015) trennt das: volle globale orthogonale
Auslese über expm(A − Aᵀ), verifiziert orthogonal auf 2,4·10⁻⁶, Energieanteile
summieren sich auf 1,000000. Vorregistriert: Ist `hybrid` gleichwertig zu
`global_orth`, dann erreicht die Butterfly-Parametrisierung dieselbe
Funktionsklasse mit **408-fach weniger Freiheitsgraden** — und der
quantenspezifische Beitrag wäre damit präzise benannt statt vermutet.

Ist `global_orth` dagegen deutlich besser, schränken die 80 Winkel die
erreichbaren Drehungen spürbar ein, und die Schaltung ist eine Näherung. Auch
das wäre ein klares Ergebnis — nur das andere.

**Das ist der schärfste Test, den dieses Projekt gerade stellen kann, und er
kommt nicht aus einer Idee, sondern aus dem Nachrechnen einer Formel, die seit
vier Tagen im Code steht.**

## 6. August — Warum die orthogonale Auslese der schlechteste Arm ist

Im Tagebuch stand seit dem 2. August unerklärt: *„die orthogonale Variante — also
die Unitaritätseigenschaft, das eigentlich Quantenhafte — schneidet am
schlechtesten ab und ist hochgradig instabil."* Q-003-replication hat das
bestätigt: `classical_quadratic` 0,8881 gegen `classical_quad_free` 0,9213, bei
**identischer Parameterzahl und identischer blocklokaler Mischung.**

Jetzt weiß ich warum, und es ist nachrechenbar.

### Die orthogonale Auslese verschenkt eine ihrer acht Ausgabedimensionen

`QuadraticReadout` zerlegt die 256 Indizes in **n disjunkte Gruppen** und
summiert die Quadrate gruppenweise. Bei einer orthogonalen Drehung bleibt die
Norm erhalten — also summieren sich die acht Ausgaben auf eine **Konstante**.

Gemessen an 1000 Zufallseingaben:

| Auslese | Summe | Streuung | Effektivrang |
|---|---|---|---|
| QuadraticReadout **orthogonal** | −6,000000 | **3,1·10⁻⁷** | **6,99** von 8 |
| QuadraticReadout frei | −7,902 | 8,4·10⁻³ | 7,97 von 8 |
| Quantenschaltung ⟨Z⟩ | −0,0075 | 1,7·10⁻¹ | 7,99 von 8 |

Eine feste Summe heißt: die acht Ausgaben liegen auf einer siebendimensionalen
Hyperebene. **Eine der acht Dimensionen trägt nur eine Konstante.** Der
Flaschenhals des Arms ist sieben, nicht acht.

Das erklärt den Abstand von 0,0332 zwischen den beiden blocklokalen Armen: Sie
unterscheiden sich in nichts außer darin, dass die orthogonale Variante ein
Achtel ihres Informationsflusses verliert.

### Und warum die Quantenschaltung dieses Problem nicht hat

⟨Z_q⟩ = P(Bit q = 0) − P(Bit q = 1) benutzt **n überlappende Zweiteilungen**
statt einer Zerlegung in n Gruppen. Jedes Qubit halbiert den 256-dimensionalen
Raum anders; die Halbierungen sind nicht disjunkt, sondern kreuzen sich. Deshalb
gibt es keine feste Summe, und der Effektivrang bleibt bei 7,99.

**Das ist ein struktureller Vorteil der Quantenauslese, auf den der klassische
Weg nicht führt.** Wer eine orthogonale quadratische Auslese von Hand baut,
nimmt naheliegend eine Gruppenzerlegung — und verliert damit eine Dimension,
ohne es zu merken. Die Quantenmechanik gibt die überlappende Variante gratis,
weil ein Qubit nun einmal den ganzen Raum halbiert und nicht ein Achtel davon
abschneidet.

Klein, aber konkret, nachrechenbar und nicht offensichtlich.

### Ein Konfundierungsfehler, vor dem Lauf gefangen

Derselbe Fehler steckte in Q-015, den ich zwei Stunden vorher gebaut und
vorgezogen hatte: Der Arm `global_orth` benutzte die Gruppenauslese. Damit hätte
der Lauf **zwei Unterschiede in einem Vergleich** gemessen — Butterfly gegen
freie Drehung *und* überlappende gegen disjunkte Auslese. Ein Ergebnis daraus
wäre nicht zuzuordnen gewesen.

Behoben, bevor er startet: `global_orth` bekommt jetzt dieselbe ⟨Z⟩-Auslese wie
die Schaltung. Nachgemessen: Effektivrang **7,995**, keine feste Summe. Damit
unterscheiden sich die beiden Arme **nur noch in der Parametrisierung** — 80
Winkel gegen 32.640 Freiheitsgrade bei exakt derselben Funktionsklasse.

Das ist jetzt der Vergleich, der die Alleinstellungsfrage beantwortet, und er
misst wirklich nur eine Sache.

### Eine Hypothese, die dabei gefallen ist

Ich hatte vermutet, der Flaschenhals des Hybridmodells sei n−1 = 7 Dimensionen,
weil die Ausgaben eine Wahrscheinlichkeitsverteilung bilden — und das hätte
verblüffend gut zu B-014 gepasst, wo die Aufgabe rund sieben Richtungen braucht.

Die Messung sagt nein: Die ⟨Z⟩-Summe ist **nicht** konstant (Streuung 0,175),
der Rang ist 7,99 von 8. Der Simplex-Zwang gilt für die klassische
Gruppenauslese, nicht für die Schaltung. Die schöne Übereinstimmung mit den
sieben Richtungen war Zufall.

Notiert, weil eine verworfene Hypothese genauso zum Verlauf gehört wie eine
bestätigte — und weil sie mich auf die Dimensionsmessung gebracht hat, aus der
dann der eigentliche Befund kam.

## 6. August — Eine Vermutung geprüft, widerlegt, und dabei etwas Besseres gefunden

In `theory.py` hatte ich geschrieben, die Schaltung sei für L→∞ universell und
die Gewichtsteilung schränke deshalb nur die Parameterzahl ein, nicht die
Ausdruckskraft. Das war **Literaturwissen über den ungeteilten Ansatz**, das ich
auf den geteilten übertragen habe, ohne es zu prüfen. Daran hängt die Deutung
von Q-014 und Q-015 vollständig.

Ich hatte sogar ein Argument dagegen: Wendet man in jeder Lage dasselbe Gatter
auf alle Qubits an, vertauschen alle Erzeuger mit der zyklischen
Qubit-Verschiebung, und die erreichbare Gruppe verlässt diese Symmetrie nie.

### Geprüft, und das Argument war falsch

Aufbau: ein zufälliges Ziel-Unitär in Dimension 2ⁿ, dann die Winkel so
optimiert, dass die Schaltung es trifft. Gemessen 1 − |Spur(U†T)|/d.

| Lagen | tied | per_qubit | untied |
|---|---|---|---|
| 4 | 0,6532 | 0,1897 | **0,0000** |
| 8 | 0,4219 | 0,0322 | 0,0000 |
| 16 | 0,0963 | **0,0000** | 0,0000 |
| 32 | **0,0000** | 0,0000 | 0,0000 |

**Alle drei Familien sind universell**, auch die vollständig geteilte. Mein
Symmetrieargument griff nicht, weil der CNOT-Ring *geordnet* angewandt wird
(q = 0, 1, 2, …) — und diese Reihenfolge bricht die zyklische Symmetrie.

Gut, dass ich es gerechnet habe. Es stand schon als Behauptung im Dokument.

### Und die Zahl dahinter ist der eigentliche Fund

Rechnet man die Parameterzahl an dem Punkt nach, an dem jede Familie
Universalität erreicht:

| Familie | Lagen | Parameter | dim SU(8) |
|---|---|---|---|
| tied | 32 | **64** | 63 |
| per_qubit | 16 | 96 | 63 |
| untied | 4 | 96 | 63 |

**Jede Familie erreicht Universalität ungefähr bei der Parameterzahl, die die
Zielgruppe hat** — die geteilte am knappsten, mit 64 gegen 63 praktisch am
informationstheoretischen Minimum.

### Bei 4 Qubits wiederholt — und es ist noch klarer

dim SU(16) = **255**. Alle drei Familien, unabhängig gefittet:

| Familie | Lagen bis Fehler < 0,01 | Parameter dort | Verhältnis zu dim SU(16) |
|---|---|---|---|
| tied | 128 | **256** | **1,00** |
| per_qubit | 32 | **256** | **1,00** |
| untied | 4 | **256** | **1,00** |

Drei völlig verschieden verteilte Parametrisierungen, dieselbe Zahl auf den
Punkt. Und die Zwischenschritte zeigen, dass es kein Schwelleneffekt ist,
sondern über den ganzen Bereich gilt:

| Parameter | tied | per_qubit | untied |
|---|---|---|---|
| 64 | 0,3052 | 0,3367 | — |
| 128 | 0,0910 | 0,0868 | 0,1703 |
| 256 | 0,0000 | 0,0000 | 0,0011 |

**Bei gleicher Parameterzahl ist die Approximationsgüte praktisch gleich,
unabhängig davon, wie die Parameter verteilt sind.** Das ist eine deutlich
stärkere Aussage als Universalität — sie sagt, dass die Parameterzahl zählt
und die Parametrisierung nicht.

Bemerkenswert nebenbei: Bei 128 Parametern ist die *ungeteilte* Variante mit
0,1703 die schlechteste, geteilt und per_qubit liegen bei 0,09. Mehr Freiheit
in der Verteilung hilft dort also nicht, sie schadet leicht.

Zwei unabhängige Qubitzahlen, dasselbe Bild. Damit halte ich den Befund für
belastbar — anders als die Einzelmessung bei n=3, die ich allein nicht
aufgeschrieben hätte.

### Was daraus für Q-014 folgt — eine abgeleitete Vorhersage

Q-014 vergleicht `tied_L5` (80 Parameter) gegen `untied_L5` (10.240) bei
gleicher Tiefe. Nach dem obigen sollte der Unterschied **allein an der
Parameterzahl hängen**. Und da B-014 gemessen hat, dass die Aufgabe rund sieben
Richtungen braucht, sind vermutlich beide weit über dem Bedarf — dann müssten
sie **gleichwertig** herauskommen.

Das ist jetzt eine Vorhersage aus zwei unabhängig gemessenen Befunden, nicht
mehr eine Hoffnung. Trifft sie nicht zu, stimmt einer der beiden nicht.

### Was das für die Alleinstellungsfrage bedeutet — und es ist nüchterner als gehofft

Die Teilung kostet **keine Parameter**, sondern **Tiefe**: 32 Lagen statt 4 für
dieselbe Ausdruckskraft. Auf einem Simulator ist das ein schlechter Tausch, weil
Tiefe unmittelbar Rechenzeit kostet. Auf echter Hardware kostet Tiefe Rauschen.

Die Teilung spart also **Speicher, nicht Rechenzeit** — und das ist eine
schwächere Aussage als „kostenlose Kompression um Faktor 128", die ich vorhin
geschrieben hatte.

**Aber** — und das ist der Punkt, der Q-014 rettet — die Aufgabe braucht gar
keine Universalität. B-014 hat gemessen, dass rund sieben Richtungen genügen,
und Q-003-replication, dass eine Lage mit 16 Winkeln so gut ist wie fünf mit 80.
Die richtige Frage ist deshalb nicht, ob die geteilte Familie asymptotisch
alles erreicht, sondern **ob sie bei praktischer Tiefe das Nötige abdeckt.**

Genau das misst Q-014, und die Frage ist damit präziser gestellt als vorher.

## 6. August — Q-015: die Schaltung schlägt die volle Drehung

Der Lauf, der Funktionsklasse von Parametrisierung trennen sollte, ist durch.
Ergebnis anders als vorhergesagt, und zwar in beide Richtungen.

| MNIST | Mischung | balanced | std | Freiheitsgrade |
|---|---|---|---|---|
| **hybrid** | global | **0,9327** | 0,0079 | **80** |
| global_orth | global | 0,8653 | **0,0346** | 32.640 |
| block_orth | blocklokal | 0,8994 | 0,0132 | 2.048 |
| quad_free | blocklokal | 0,9115 | 0,0108 | 2.048 |

**PRIMÄR: hybrid gegen global_orth = +0,0674, 15 von 15 Seeds, Holm p=0,0001.**

Ich hatte Gleichwertigkeit erwartet — beide haben dieselbe Funktionsklasse, und
der Universalitätstest hat gezeigt, dass beide alles erreichen können. Die
Schaltung schlägt die volle Drehung stattdessen deutlich, **bei 408-fach
weniger Freiheitsgraden.**

### Was das heißt, und was es nicht heißt

Weil beide dieselbe Funktionsklasse haben und beide universell sind, kann der
Unterschied **nicht an der Ausdruckskraft liegen**. Er liegt an der
**Trainierbarkeit**: Die Butterfly-Parametrisierung ist besser konditioniert
als eine freie schiefsymmetrische Matrix durch expm.

Was ehrlich dazugehört:
- Der freie Arm hat 32.640 Parameter auf 4.000 Beispielen — reichlich
  Gelegenheit zur Überanpassung.
- Seine Streuung ist mit 0,0346 **vierfach höher** als die der Schaltung
  (0,0079). Das deutet auf Optimierungsinstabilität, nicht auf eine schlechtere
  Lösung.
- Auf Fashion-MNIST wählte er die kleinste Rate des Gitters (0,003, Randlage) —
  dort ist er möglicherweise untertuniert. Auf MNIST lag er mit 0,03 im Inneren,
  die Primärzahl ist also die sauberere.

Die Aussage lautet damit: **bei gleichem Trainingsbudget und fairer
Lernratensuche ist die quantenabgeleitete Parametrisierung deutlich leichter zu
optimieren als die freie.** Nicht: sie kann mehr.

### Und die zweite Vorhersage war ebenfalls falsch

**P2: global_orth gegen block_orth = −0,0341, 0 von 15 Seeds, Holm p=0,0001.**
Die *blocklokale* Mischung ist besser als die globale — genau umgekehrt zu dem,
was ich aus B-004 abgeleitet hatte.

### Zwei Fehler in meinem Auswertungscode

**Erstens** war die Auflösungsprüfung im Code strenger als in der
Vorregistrierung. Angemeldet war „trennt sich global_orth nicht von block_orth,
dann löst der Aufbau nicht auf" — also **Trennung**. Der Code verlangte
zusätzlich die **Richtung** und hat ein Ergebnis mit p=0,0001 als „nicht
verwertbar" abgestempelt, nur weil es unerwartet ausfiel.

**Zweitens** hatte meine Urteilslogik gar keinen Zweig für „die Schaltung
gewinnt". Ich hatte nur „gleichwertig" oder „die volle Drehung gewinnt"
vorgesehen. Das eigentliche Ergebnis fiel durch das Raster und landete bei
„unentschieden".

Beides korrigiert, die Analyse neu gerechnet und in die Rohdatendatei
zurückgeschrieben; die Messdaten selbst sind unangetastet. Der Merkposten steht
im Code: **wer nur die erwarteten Ausgänge kodiert, verliert den interessanten.**

## 6. August — Q-009: drei Seeds können nicht signifikant werden

| Haufen | butterfly | dense_isotime | Differenz |
|---|---|---|---|
| 64 | 1,0000 | 0,9939 | +0,0061 |
| 256 | 1,0000 | 0,9703 | +0,0297 |
| 1024 | 0,9724 | 0,8137 | **+0,1588** |

Bei gleichem Zeitbudget wächst der Vorsprung der Schaltung mit der
Aufgabenschwierigkeit — entgegen meiner Vorhersage, die das Gegenteil erwartete.
Bei 1024 Gebieten sind 140 Parameter um **0,1588** besser als 222 Millionen.

**Und keine dieser Zahlen ist statistisch belegbar.** Ich habe den Lauf mit
**drei Seeds** entworfen, und das macht Signifikanz unerreichbar:

- Gepaarter Wilcoxon bei 3 Seeds: kleinstmöglicher p-Wert ist **0,2500**.
  Der Primärtest meldet exakt 0,2500 — er liegt auf dem Boden.
- Die Auflösungsprüfung ebenfalls: p=0,2500.
- Spearman über 3 Punkte: scipy meldet **p=0,0000**, der exakte
  Permutationstest über alle 3! = 6 Anordnungen gibt **p=0,3333**. Die
  asymptotische Formel ist bei drei Punkten wertlos.

Die Effektstärke bei 1024 Haufen beträgt **12,3 gepoolte
Standardabweichungen** — der Unterschied ist offensichtlich, nur nicht
zertifizierbar.

Das ist derselbe Fehler, den das Tagebuch für Wilcoxon schon dokumentiert hat
(„bei n=8 ist p=0,0078 der rechnerisch kleinste erreichbare Wert"), nur eine
Stufe schlimmer. Ich habe die Seedzahl aus Rechenzeitgründen gewählt und nicht
geprüft, ob der Test damit überhaupt anschlagen kann.

**Q-009 ist zu wiederholen**, mit mindestens 10 Seeds und einer schwereren
leichtesten Stufe — die 64 Haufen waren gesättigt, weshalb der Primärtest ohnehin
auf der falschen Stufe saß.

## 10. August — Q-014: derselbe Auswertungsfehler noch einmal, diesmal vor dem Lauf

Vor dem Start von Q-014 habe ich das Skript gegen die sieben Regeln gelesen und
die Auswertung auf synthetischen Daten laufen lassen — der billige Test, der
keine Rechenzeit kostet. Er hat etwas gefunden.

### Die Urteilslogik hatte keinen Zweig für „geteilt gewinnt"

`analyse()` kannte drei Ausgänge: gleichwertig (H1), ungeteilt besser (H0),
unentschieden (HU). Der vierte fehlte. Ein konstruierter Fall mit **+0,0510
zugunsten von `tied_L5`, 15 von 15 Seeds, Holm p=0,0001** landete bei:

> BEWERTUNG: HU — unentschieden; die Daten tragen erst ab ±0,0528.

**Das ist wortwörtlich der Fehler aus Q-015**, vier Tage nachdem ich ihn dort
gefunden, behoben und mir als Merkposten in den Code geschrieben habe: *wer nur
die erwarteten Ausgänge kodiert, verliert den interessanten.* Ich habe ihn beim
Schreiben von Q-014 trotzdem wieder gemacht.

Und der Ausgang ist hier nicht exotisch, sondern der von Q-015 nahegelegte:
Dort haben **80 geteilte Winkel 32.640 freie Freiheitsgrade um +0,0674
geschlagen**. Q-014 stellt dieselbe Achse noch einmal — 80 geteilte gegen
10.240 ungeteilte Winkel. Wäre der Lauf so gestartet, hätte er eine Bestätigung
von Q-015 an einer zweiten, unabhängigen Kontrolle als „unentschieden" gemeldet.

### Was geändert wurde, und wann

Ergänzt vor dem ersten Lauf, es existieren keine Rohdaten zu Q-014:

- **H1+** als vierter Ausgang: `tied_L5` signifikant besser als `untied_L5`.
  Deutung dann zwingend über die **Trainierbarkeit**, nicht die Ausdruckskraft —
  denn beide Familien approximieren bei gleicher Parameterzahl gleich gut, und
  das ist bei 3 und bei 4 Qubits unabhängig geprüft.
- **Überschneidungsregel**, ebenfalls vorher festgelegt: Ist ein Ergebnis
  zugleich signifikant und TOST-gleichwertig innerhalb ±0,005, dann gilt die
  Gleichwertigkeitsaussage und die Richtung wird nur beschreibend genannt. Sonst
  könnte ich mir hinterher den günstigeren der beiden Sätze aussuchen.

### Die übrigen sechs Regeln, nachgeprüft

| Regel | Stand |
|---|---|
| Lernrate je Arm, Randlage gemeldet | erfüllt, Gitter 0,001–0,3 |
| Genug Seeds | 15 Seeds, kleinstmöglicher Wilcoxon-p = 0,00006 |
| TOST statt p>0,05 | erfüllt |
| Holm nur über die angemeldete Familie | **nachgemessen: Faktor 2,00** bei zwei Vergleichen |
| Rohdaten nach jedem Datensatz | erfüllt, `rec.save()` je Datensatz |
| Positivkontrolle kann feuern | im synthetischen Durchgang ausgelöst |
| Kriterien vor dem Lauf im Skript | erfüllt |

Alle vier Ausgänge wurden einzeln erzwungen und melden jetzt das Richtige. Die
synthetischen Genauigkeiten waren dabei absichtlich nicht-konstant von den
balancierten verschieden — sonst wäre nicht sichtbar, welches Feld der Test
benutzt. Er benutzt das balancierte, wie es `MARGIN` verlangt.

**Was ich daraus mitnehme:** Die Regel „ein Zweig für jeden Ausgang" stand seit
dem 6. August im Tagebuch und im Code, und hat den nächsten Lauf trotzdem nicht
erreicht. Eine Lehre, die nur aufgeschrieben ist, wirkt nicht — es braucht den
mechanischen Vortest auf synthetischen Daten, und der kostet zwei Minuten gegen
zweieinhalb Stunden Lauf.

## 10. August — Q-014, MNIST-Hälfte: die Winkel muss man gar nicht trainieren

MNIST ist durch und gesichert (`Q-014__20260810_041708`), Fashion-MNIST läuft
noch. Die Auswertung auf dem Zwischenstand — die Vorregistrierung stand seit
Nachts fest, MNIST ist der angemeldete Primärdatensatz:

| Arm | geteilt | Lagen | Block-Par. | balanced | std | lr |
|---|---|---|---|---|---|---|
| tied_L1 | ja | 1 | 16 | 0,9344 | 0,0059 | 0,1 |
| **tied_L5** | ja | 5 | **80** | **0,9363** | 0,0065 | 0,1 |
| tied_L80 | ja | 80 | 1.280 | 0,9098 | 0,0103 | 0,03 |
| untied_L1 | nein | 1 | 2.048 | 0,9341 | 0,0079 | 0,1 |
| untied_L5 | nein | 5 | 10.240 | 0,9107 | 0,0101 | 0,03 |
| **tied_L5_frozen** | ja | 5 | **0 trainierbar** | **0,9346** | 0,0079 | 0,1 |

**AUFLÖSUNGSPRÜFUNG DURCHGEFALLEN: trainiert gegen eingefroren +0,0018,
p=0,1205.** Nach der eigenen Vorregistrierung ist der Lauf damit **nicht
verwertbar** — kein Vergleich darüber ist interpretierbar.

### Was durchgefallen hier heißt, und es ist kein Messversagen

Die Prüfung sagt: `tied_L5` mit trainierten Winkeln ist nicht von `tied_L5`
mit **eingefrorenen Zufallswinkeln** zu unterscheiden. Der Butterfly-Block
lernt auf dieser Aufgabe nichts, was messbar wäre. Encoder und Kopf holen alles.

Das reiht sich exakt in die bisherigen Abgrenzungen ein:

- Verschränkung trägt nichts bei (TOST, 5. August)
- Tiefe trägt nichts bei (TOST, 5. August)
- **und jetzt: die Winkel überhaupt zu trainieren trägt nichts bei**

Damit ist die Deutung der Kernaussage zu schärfen. „80 Parameter schlagen
67.848" stimmt weiter als Zahl, aber die 80 sind offenbar **nicht der Grund** —
eine feste zufällige orthogonale Mischung mit quadratischer Auslese tut es
genauso. Das passt zur Rayleigh-Quotient-Charakterisierung vom 6. August: Was
wirkt, ist die *Struktur* (normieren → normerhaltend mischen → quadratisch
auslesen), nicht die gelernte Drehung. Als Effizienzaussage ist das eher
stärker, als Quantenaussage deutlich schwächer.

**Der Entwurfsfehler ist benennbar:** In jedem Arm steckt ein trainierbarer
Encoder 784 → 256. Der kann sich an jede feste nachgeschaltete Drehung
anpassen — er übernimmt genau die Arbeit, die der Block tun sollte. Solange der
Encoder frei trainiert, ist die Frage „kostet die Gewichtsteilung etwas" gar
nicht stellbar. Ein Nachfolgelauf muss den Encoder einfrieren oder verkleinern.

### Und ein Konfundierungsverdacht, der die beiden Primärzahlen betrifft

| | PRIMÄR tied_L5 − untied_L5 | P2 tied_L80 − untied_L1 |
|---|---|---|
| Differenz | **+0,0257**, 15/15, Holm p=0,0001 | **−0,0243**, 0/15, Holm p=0,0001 |

Beide hochsignifikant, beide gegenläufig — und beide fallen **exakt mit der
Lernrate zusammen**:

- lr 0,1: tied_L1 0,9344 · tied_L5 0,9363 · untied_L1 0,9341 · frozen 0,9346
- lr 0,03: tied_L80 0,9098 · untied_L5 0,9107

Vier Arme bei 0,1 liegen zwischen 0,934 und 0,936, die zwei Arme bei 0,03 bei
0,910. Die Trennung folgt der Lernrate, nicht der Teilung. Die Lernrate wurde
auf **2 Tuning-Seeds bei halbem Datensatz und halber Epochenzahl** gesucht und
dann bei vollem Budget eingesetzt — wenn diese Suche für die schwerer
trainierbaren Arme systematisch zu klein wählt, misst der Vergleich die
Tuning-Bedingung mit.

Gegenrede, die dazugehört: Dass ein Arm eine kleinere Rate *braucht*, ist selbst
ein Symptom schlechterer Konditionierung — Lernratenwahl und Trainierbarkeit
sind nicht unabhängig, und genau Trainierbarkeit ist die Größe, um die es seit
Q-015 geht. Aus diesen Daten ist beides nicht zu trennen. Der saubere Test wäre,
die beiden 0,03-Arme zusätzlich bei 0,1 zu fahren; das wäre post hoc und dürfte
nur beschreibend genannt werden.

### Was der ergänzte Zweig gebracht hat

`tied_L5` schlägt `untied_L5` um +0,0257 bei 15/15 Seeds — genau der Ausgang
H1+, für den heute Nacht der fehlende Zweig ergänzt wurde. Hätte die
Auflösungsprüfung gehalten, hätte die alte Fassung dieses Ergebnis als
„unentschieden" gemeldet. Der Zweig war also nicht hypothetisch; der Lauf ist
tatsächlich dort gelandet.

Zitierbar ist die Zahl trotzdem nicht — die durchgefallene Auflösungsprüfung
steht davor, und diese Reihenfolge gilt auch dann, wenn das Ergebnis gefällt.

## 10. August — Q-014 abgeschlossen: auf Fashion-MNIST ist das Training sogar schädlich

Der Lauf ist durch (`Q-014__20260810_062132`, 180 Messzeilen, Status 0, 4 h 14 min).
Das Urteil steht wie nach der MNIST-Hälfte: **HU, nicht verwertbar**, weil die
Auflösungsprüfung auf dem Primärdatensatz durchgefallen ist. Familie und
Auflösungsprüfung rechnen ausschließlich auf MNIST; Fashion-MNIST ist
beschreibend.

| Fashion-MNIST | geteilt | Lagen | Block-Par. | balanced | std | lr |
|---|---|---|---|---|---|---|
| **tied_L1** | ja | 1 | **16** | **0,8330** | 0,0126 | 0,03 |
| tied_L5 | ja | 5 | 80 | 0,7987 | 0,0213 | **0,3 (Rand)** |
| tied_L80 | ja | 80 | 1.280 | 0,7894 | 0,0292 | 0,003 |
| untied_L1 | nein | 1 | 2.048 | 0,8312 | 0,0182 | 0,1 |
| untied_L5 | nein | 5 | 10.240 | 0,8186 | 0,0109 | 0,01 |
| tied_L5_frozen | ja | 5 | **0 trainierbar** | 0,8198 | 0,0165 | **0,3 (Rand)** |

### Der eingefrorene Arm gewinnt

| trainiert gegen eingefroren | Differenz | Siege | p |
|---|---|---|---|
| MNIST | +0,0018 | 10/15 | 0,1205 |
| **Fashion-MNIST** | **−0,0211** | **1/15** | **0,0009** |

Auf MNIST ist das Training der Winkel wirkungslos, auf Fashion-MNIST ist es
**schädlich**: in 14 von 15 Seeds ist die eingefrorene Zufallsdrehung besser.
Der Befund repliziert also nicht nur, er verschärft sich. Und er war schon in
der Lernratensuche sichtbar — der eingefrorene Arm lag dort bei **allen sechs**
Raten vorn (0,6598/0,6642/0,6662/0,6956/0,7205/0,7407 gegen
0,6615/0,6598/0,6605/0,6793/0,7021/0,7193).

**Vorbehalt, der dazugehört:** Beide Arme wählten 0,3, den oberen Gitterrand,
und beide Kurven stiegen dort noch. Beide sind untertuniert. Der *Kontrast*
zwischen ihnen ist davon weniger betroffen, weil beide dieselbe Rate und
dasselbe Budget bekamen und der Vorsprung über das ganze Gitter hält — aber die
Absolutwerte sind nicht ausgereizt, und die Zahl −0,0211 ist beschreibend, nicht
vorregistriert. Regel 1 ist auf diesem Datensatz verletzt.

### Was Q-014 unterm Strich geändert hat

Die Frage „kostet die Gewichtsteilung Qualität" ist **nicht beantwortet** — sie
war mit diesem Aufbau nicht stellbar. Beantwortet ist eine andere, die nicht
gestellt war:

> Auf diesen Aufgaben trägt es nichts bei, die Winkel der Schaltung überhaupt
> zu trainieren. Auf Fashion-MNIST schadet es.

Zusammen mit „Verschränkung trägt nichts bei" und „Tiefe trägt nichts bei"
ergibt das eine geschlossene Aussage: Von der Quantenschaltung wirkt die feste
Struktur — normieren, normerhaltend mischen, quadratisch auslesen — und weder
die Verschränkung noch die Tiefe noch die gelernten Parameter. Das ist die
schärfste Fassung des Rayleigh-Quotient-Befunds vom 6. August und die
unbequemste: Der trainierbare Teil der Schaltung ist der Teil, der nichts tut.

Zwei Einschränkungen, ohne die der Satz falsch wird. **Erstens** steckt in jedem
Arm ein trainierbarer Encoder 784 → 256, der sich an jede feste nachgeschaltete
Drehung anpassen kann; er übernimmt die Arbeit des Blocks. **Zweitens** brauchen
beide Datensätze laut B-014 nur rund sieben Richtungen — eine Aufgabe, die die
Kapazität nicht ausschöpft, kann den Unterschied gar nicht zeigen.

### Der Nachfolgelauf, vor dem Start festgelegt

Damit die ursprüngliche Frage stellbar wird, muss der Block der Engpass sein:

1. **Encoder einfrieren oder verkleinern** — sonst ist der Vergleich
   strukturell blind, und das ist der Grund für das HU.
2. **Lernratengitter bis mindestens 1,0** — auf Fashion-MNIST saßen zwei Arme
   am oberen Rand und stiegen dort noch.
3. **Aufgabe mit mehr als sieben nötigen Richtungen**, sonst wiederholt sich
   der gesättigte Fall von Q-009.
4. `tied_L5_frozen` bleibt als Auflösungsprüfung — sie hat diesmal genau das
   getan, wofür sie eingebaut wurde, und vier Stunden Rechenzeit vor einer
   falschen Schlussfolgerung bewahrt.

Die beiden Familienzahlen dieses Laufs (+0,0257 und −0,0243, beide Holm
p=0,0001) sind **nicht zitierbar**. Die Auflösungsprüfung steht davor. Dass
+0,0257 in die erhoffte Richtung zeigt, ändert daran nichts — diese Reihenfolge
gilt gerade dann, wenn das Ergebnis gefällt. Hinzu kommt der
Lernraten-Konfundierungsverdacht aus dem Eintrag zur MNIST-Hälfte.

## 10. August — Zielentscheidung, T-Serie, und Q-018 vorregistriert

Geschrieben **vor** dem Lauf, es existieren keine Q-018-Rohdaten.

### Die Zielentscheidung

Am Ende der 15 Wochen bis zur Anmeldung sollen ein **Chatbot** und eine
**zweite KI-Art** stehen. Dafür fallen Q-010, Q-012, Q-016 und Q-017 als
eigenständige Läufe weg; Q-011 wird in die Zeitreihen-Phase mitgenommen, weil es
der beste Kandidat für einen bestandenen Abgrenzungstest ist.

**Der Konflikt, den das mit sich bringt, und der in die Arbeit gehört:** Die
Struktur gewinnt ab 16.384 Dimensionen, nach Fusion vielleicht ab 4.096. Ein auf
diesem Rechner trainierbares Sprachmodell liegt bei d_model 256–1.024 — Q-013
sitzt bei etwa 512. **Diese Bereiche überlappen hier nicht.** Die Aussage ist
deshalb zusammengesetzt: Qualität wird bei trainierbarer Breite gemessen,
Effizienz bei der Breite, wo sie auftritt, und der Chatbot demonstriert die
Qualität. „Der Chatbot läuft dank der Quantenstruktur schneller" wäre falsch und
wird nicht behauptet.

### Die T-Serie

Anlass ist der älteste ungelöste Punkt des Projekts: 80 Parameter statt 67.848,
und trotzdem langsamer. Bei d=4096 braucht die Struktur **340× weniger
Arithmetik und ist 5× langsamer.** Der Entwurf steht in
`ENTWURF_T_SERIE_KERNEL.md`.

Beim Nachzählen im eigenen Code kam heraus, dass die Diagnose „60 sequentielle
Kleinoperationen" noch untertrieben ist. Für 8 Qubits und 5 Lagen:

| | Durchläufe über den Zustandsvektor |
|---|---|
| Gatteranwendungen | 40 |
| Permutations-Gathers | 40 |
| Betragsquadrate | 1 |
| Auslese (n einzelne Reshapes) | 8 |
| **heute** | **≈ 89** |
| **fusioniert, angestrebt** | **1–2** |

Drei konkrete, bisher ungehobene Posten: Die n CNOT-Permutationen werden einzeln
angewandt (`butterfly_layer.py:175`), obwohl sie zu **einer** komponieren —
CNOT ist reine Indexvertauschung, das steht sogar so im eigenen Kommentar.
`_gate` steht in der Vorwärtsschleife und rechnet `cos`, `sin` und zwei komplexe
`exp` bei jedem Vorwärtslauf neu, bei eingefrorenen Winkeln also vollständig
umsonst. Und die Auslese macht n Durchläufe, wo einer reicht.

Das ist ein Speicherverkehrsproblem, kein Arithmetikproblem — bei d=4096 sind
32 KB, das passt in den L1-Cache. Deshalb halte ich die Lücke für handwerklich
und nicht für fundamental. **Geprüft ist das nicht**, T-001 prüft es
vorregistriert.

### Q-018 — und was der Vortest sofort gefunden hat

Q-018 misst den Frozen-Befund unter verschärften Bedingungen: **kein Encoder**
(die Haufen-Aufgabe liefert 256 = 2⁸ Dimensionen direkt), Gitter bis 1,0, und
eine Aufgabe mit kalibriertem Kapazitätsbedarf. Auflösungsprüfung ist
`tied_frozen` gegen `frozen_4ch` — dieselbe feste Drehung, aber nur vier der
acht Kanäle erreichen den Kopf. Trennen die sich nicht, ist der Engpass nicht
bindend und der Aufbau blind.

**Der Vortest hat beim ersten Anlauf sofort etwas gefunden**, und es war das
Gegenteil der Erwartung. Mit den Haufenzahlen aus `isotime_accuracy` (64/256/1024)
kam heraus:

| Haufen | tied_frozen | Rateniveau |
|---|---|---|
| 64 | 0,2892 | 0,100 |
| 256 | 0,1808 | 0,100 |
| 1024 | **0,1268** | 0,100 |

Die Aufgabe war **zu schwer**, nicht zu leicht — bei 1024 Haufen lernt das
Modell praktisch nichts. Die Stufen stammten aus einem Lauf mit Breite 16.384;
für einen 8-Kanal-Block sind sie sinnlos. Hätte ich sie ungeprüft übernommen,
wäre der Lauf an einem Vergleich von Rauschen mit Rauschen gescheitert.

Also über zwei Achsen kalibriert, Klassenzahl mit dazu. Gewählt: **6 Klassen,
32 Haufen** — `tied_frozen` bei 0,4697 gegen Rateniveau 0,167, nicht gesättigt,
Abstand zu `frozen_4ch` +0,1277. Zweitstufe 128 Haufen; sie liegt mit 0,2882
näher am Zufall und ist deshalb **nur beschreibend**, nicht urteilstragend.

Alle sechs Ausgänge wurden vorher auf synthetischen Daten erzwungen und melden
das Richtige, einschließlich „eingefroren gewinnt", „Auflösungsprüfung
umgekehrt" und „unentschieden". Holm-Familiengröße nachgemessen: 2.

**Der Vortest kostet zwei Minuten und hat jetzt zweimal hintereinander einen
Entwurfsfehler gefunden, den ein Lauf in Stunden bezahlt hätte.** Er gehört ab
sofort vor jeden Lauf dieses Projekts.

## 10. August — Q-018: das Training trägt bei, und Q-014 galt nur für Q-014

Der Lauf ist durch, Status 0, 120 Messzeilen. **Auflösungsprüfung bestanden** —
der erste Q-Lauf seit Q-015, der das schafft.

| 32 Haufen (Primär) | trainierbar | Kanäle | balanced | std | lr |
|---|---|---|---|---|---|
| **tied_trained** | 80 | 8 | **0,9588** | 0,0882 | 0,1 |
| **tied_frozen** | **0** | 8 | **0,5511** | 0,0713 | 0,3 |
| untied_trained | 10.240 | 8 | 1,0000 | 0,0000 | 0,003 |
| frozen_4ch | 0 | 4 | 0,3437 | 0,0552 | 1,0 (Rand) |

**PRIMÄR: +0,4077, 15 von 15 Seeds, Holm p=0,0001.**
Auflösungsprüfung: +0,2074, p=0,0001.
Zweitstufe 128 Haufen, beschreibend: **+0,3646**, 15/15, p=0,0001.

### Das Vorzeichen dreht von −0,02 auf +0,41

Q-014 hatte gemessen, dass eingefrorene Zufallswinkel so gut sind wie trainierte
(MNIST +0,0018) oder besser (Fashion-MNIST −0,0211). Ohne Encoder ist der
trainierte Block um **+0,4077** besser. Das ist keine Verschiebung, das ist eine
andere Antwort.

**Und die Trennung hängt an keinem Mittelwert.** Auf beiden Stufen überlappen
sich die Verteilungen in keinem einzigen Paar:

| | schlechtester trainierter Seed | bester eingefrorener Seed |
|---|---|---|
| 32 Haufen | 0,7302 | 0,6578 |
| 128 Haufen | 0,5918 | 0,3252 |

`tied_trained` ist dabei zweigipflig: Median **0,9988**, acht von fünfzehn Seeds
exakt 1,0000, aber zwei Seeds (7101, 7105) bei 0,73–0,76. Die gewählte Rate 0,1
liegt nahe an der Stabilitätsgrenze — bei 0,3 fällt der Arm auf 0,7367, bei 1,0
auf 0,3382. Der Mittelwert **unterschätzt** den typischen Fall. Der
vorregistrierte Test ist Wilcoxon und damit rangbasiert, von den zwei
Ausreißern also ohnehin unberührt.

### Warum beide Läufe recht haben, und was daraus wirklich folgt

Die saubere Fassung ist nicht „Q-014 war falsch", sondern:

> **Ob das Training der Winkel etwas beiträgt, hängt daran, ob die Aufgabe die
> Kapazität der Schaltung ausschöpft.** Auf MNIST braucht die Aufgabe laut
> B-014 rund sieben Richtungen — dort reicht eine feste Zufallsdrehung, und ein
> trainierbarer Encoder holt den Rest. Auf einer Aufgabe, die die 80 Parameter
> wirklich fordert, trägt das Training +0,41.

Dass die Aufgabe hier wirklich fordert, sieht man an `untied_trained`: **auf
beiden Stufen exakt 1,0000 bei Streuung 0,0000.** Mit 10.240 freien Winkeln ist
sie trivial, mit 80 geteilten nicht. Zum ersten Mal in diesem Projekt bindet die
Parameterzahl der Schaltung tatsächlich — auf MNIST tat sie das nie.

Das entwertet allerdings **P2 vollständig**: ein gesättigter Arm gegen einen
ungesättigten gemessen ergibt eine Untergrenze, keinen Messwert. Die +0,4489
sind nicht zitierfähig.

### Was Q-014 rückwirkend heißt

Der Q-014-Eintrag bleibt stehen, aber seine Verallgemeinerung wird
zurückgenommen. In `BEFUNDE_UND_RICHTUNGEN.md` ist der Eintrag ersetzt, mit
Begründung statt Löschung. Die Lehre ist nicht neu, nur teuer bestätigt:

> **Ein Nullbefund an einem Bauteil ist wertlos, solange ein anderes Bauteil
> dessen Aufgabe übernehmen kann.**

Das ist derselbe Fehlertyp wie die blinde Architektur in B-004 und die
blocklokale Kontrolle in Q-003 — nur diesmal war das aushelfende Bauteil der
Encoder, und er stand in *jedem* Arm, war also unsichtbar.

### Zwei Mängel dieses Laufs, die dazugehören

- **`frozen_4ch` sitzt auf dem oberen Gitterrand (1,0) und stieg dort noch.**
  Der Arm ist untertuniert; die Auflösungsprüfung bestand mit +0,2074 trotzdem
  klar, ihr Abstand ist aber eine Obergrenze.
- **Mein Vortest war halb gemacht.** Er kalibrierte die Schwierigkeit an den
  beiden eingefrorenen Armen, weil die die Auflösungsprüfung bilden, und prüfte
  nie, ob die *trainierten* Arme an die Decke stoßen. Genau das ist passiert.
  Behoben: der Vortest ist jetzt zweistufig und verwirft jeden Kandidaten, bei
  dem irgendein Arm über 0,95 kommt.

### Folge für die Roadmap

**R5 (Konstantenfaltung bei eingefrorenen Winkeln) fällt weg**, der Kernel muss
den trainierbaren Fall bedienen, und Phase 3 bleibt bei drei Wochen statt auf
eine zu schrumpfen. Genau dafür stand Phase 0 vor Phase 1: Die Antwort fällt in
die aufwendigere Richtung — aber vor dem Kernelbau und nicht danach.

Und eine gute Nachricht für die Gesamtaussage: Die Parameter der Schaltung sind
nicht bloß Beiwerk einer festen Struktur. Sie tragen, wenn man sie fordert.

## 10. August — T-000: die Zielgröße des Chatbots ist gemessen

Kalibrierung, kein Experiment: keine Hypothese, keine Kontrolle, kein p-Wert.
4 Lagen, 4 Köpfe, Sequenz 128, Batch 16, Vokabular 4.096, 6 Threads.

| Breite | Parameter | s/Schritt | Token/s | Adam-MB |
|---|---|---|---|---|
| 128 | 1.858.304 | 0,332 | 6.168 | 28 |
| **256** | **5.289.472** | **0,627** | **3.267** | **81** |
| 512 | 16.870.400 | 1,482 | 1.382 | 257 |
| 1024 | 58.906.624 | 5,387 | 380 | 899 |

Abgeleitet über die Chinchilla-Faustregel (mindestens 20 Token je Parameter):

| Budget | größte Breite | Token im Budget | Token/Parameter |
|---|---|---|---|
| 12 h | 256 | 141 Mio. | 27 |
| **24 h** | **256** | **282 Mio.** | **53** |
| 48 h | 256 | 565 Mio. | 107 |
| 96 h | 512 | 478 Mio. | 28 |

**Die Zielgröße für Phase 3 ist damit gemessen statt geraten: d_model 256, rund
5,3 Millionen Parameter, 282 Millionen Token in 24 Stunden.** Das ist die
Größenordnung, in der zusammenhängender Text entsteht — der Chatbot ist auf
dieser Hardware erreichbar.

Und es beziffert den Konflikt aus der Roadmap zum ersten Mal: **trainierbar ist
Breite 256, die Struktur gewinnt ab 16.384. Faktor 64.**

Nebenbefund für die Planung: Der Speicher wächst schneller, als der Durchsatz
fällt. Bei Breite 1024 sind 899 MB allein für Gewichte, Gradienten und die zwei
Adam-Kopien nötig; Breite 2048 wäre auf 16 GB nicht mehr sinnvoll trainierbar.

## 10. August — T-001: der Umbau bringt 1,6×, und das reicht nicht

Verifikation zuerst: größte Abweichung **4,0·10⁻⁷** über 30 Konfigurationen,
Gradienten identisch auf 0,0. R1, R5 und R6 sind nachweislich
verhaltenserhaltend. Erst danach gemessen — sieben Reihen, abwechselnd, Faktor
je Reihe gepaart, einfädig, Batch 64, 5 Lagen.

| d | referenz | schnell | dicht | Gewinn | gegen dicht: referenz / schnell |
|---|---|---|---|---|---|
| 1.024 | 82,4 ms | 47,3 ms | 3,1 ms | 1,74× | 0,037 / 0,055 |
| 2.048 | 155,6 ms | 95,1 ms | 13,5 ms | 1,64× | 0,087 / 0,138 |
| 4.096 | 329,5 ms | 194,8 ms | 61,0 ms | 1,69× | 0,177 / 0,331 |
| 8.192 | 673,1 ms | 440,0 ms | 258,3 ms | 1,53× | 0,370 / 0,590 |
| **16.384** | **1433,8 ms** | **998,0 ms** | **2355,7 ms** | **1,44×** | **1,590 / 2,311** |

**BEWERTUNG: T0.** Der Übergangspunkt liegt weiterhin bei 16.384, der Umbau hat
ihn nicht verschoben. Auflösungsprüfung bestanden (1,44×). Das
Ausstiegskriterium der Roadmap greift.

**Der Aufbau reproduziert die Messung vom 5. August.** Die Referenz erreicht
0,037 / 0,177 / 0,370 / 1,590 gegen die damals gemeldeten 0,04 / 0,21 / 0,40 /
1,99. Der Maßstab stimmt, bevor über die neue Zahl geredet wird.

### Warum 1,6× den Übergangspunkt nicht bewegt — und das ist der eigentliche Ertrag

Der Gewinn ist über alle Breiten bemerkenswert stabil: 1,44 bis 1,87, Median
etwa 1,6. Er ist also ein **konstanter Faktor**, keine Änderung der Skalierung.
Und ein konstanter Faktor verschiebt einen Schnittpunkt nur so weit, wie die
beiden Kurven auseinanderlaufen:

- `schnell` wächst je Verdopplung um **2,27×** (d·log d)
- eine dichte Schicht wüchse glatt um **4×** (d²)
- das Verhältnis verbessert sich also je Oktave um 4/2,27 = **1,76×**

Ein Gewinn von 1,6× entspricht damit log(1,6)/log(1,76) ≈ **0,8 Oktaven**. Das
Messgitter *sind* Oktaven — die Verschiebung ist kleiner als die Auflösung und
konnte gar nicht sichtbar werden.

**Um das vorregistrierte Ziel 4.096 zu erreichen, hätte es zwei Oktaven
gebraucht, also Faktor 1,76² ≈ 3,1. Erreicht wurden 1,6.**

Dazu kommt ein zweiter Grund, den ich vorher nicht bedacht hatte: Der Übergang
entsteht gar nicht dadurch, dass Butterfly schnell wird, sondern dadurch, dass
**die dichte Schicht abstürzt.** Von 8.192 auf 16.384 springt sie von 258 ms auf
2356 ms — Faktor **9,1** statt der erwarteten 4, weil ihre Gewichtsmatrix mit
1,07 GB nicht mehr in den Cache passt. Die Butterfly-Kurve ist glatt, die dichte
hat eine Kante. **An einer Kante lässt sich der Schnittpunkt von der anderen
Seite kaum verschieben.**

### Was das für die T-Serie heißt

Das Ausstiegskriterium lautet: Übergang > 8.192 → Serie abbrechen. Es greift,
und ich weiche es nicht auf. Ein greifendes Ausstiegskriterium ist ein Ergebnis.

**Was dieser Lauf aber nicht gezeigt hat**, und das gehört sauber getrennt: Er
hat nur den Teil der Fusion geprüft, der in reinem PyTorch ausdrückbar ist.
R1/R5/R6 senken die Durchläufe über den Zustand von rund 89 auf rund 46. Die
eigentliche Fusion — alle L·n Stufen in einem Durchlauf, Zustand im Cache, also
89 auf 1–2 — braucht einen kompilierten Kernel und war nie Gegenstand dieser
Messung. Das stand vor dem Lauf so im Kopf von `butterfly_fast.py`.

Die Zahl dazu: Ein Kernel müsste **Faktor 3 statt 1,6** bringen, um den
Übergang auf 4.096 zu holen. Ob das erreichbar ist, ist offen — und es ist eine
**eigene Hypothese mit eigener Vorregistrierung**, keine Umdeutung dieser hier.
`numba`, `triton` und `cython` fehlen in der Umgebung; nur `torch.compile` wäre
vorhanden.

### Was trotzdem bleibt

- **1,6× schneller bei bitgleicher Funktion und identischen Gradienten**, über
  fünf Breiten stabil. Das ist ein echter, verifizierter Gewinn und kostet
  nichts — `butterfly_fast.py` kann `ButterflyQuantumLayer` überall ersetzen.
- **Bei 16.384 steigt der Vorsprung gegen dicht von 1,59× auf 2,31×.** Der
  Übergangspunkt hat sich nicht bewegt, der Vorsprung dahinter ist um die
  Hälfte gewachsen.
- Und ein Befund über die *Form* des Problems: Der Übergang wird von der
  Cache-Kante der dichten Schicht bestimmt, nicht von der Effizienz der
  Butterfly-Struktur. Wer ihn verschieben will, muss um Oktaven besser werden,
  nicht um Prozente.

## 10. August — Phase 3 aufgesetzt, und ein Fehler in Q-013 vor dem ersten Lauf

### Ein ungeteilter Arm, der geteilt rechnet

Beim Lesen von `sequence_platform.py` (Q-013), um die Transformer-Integration
nicht neu zu erfinden, stand in Zeile 172:

```python
t = self.theta[layer, q] if self.tie else self.theta[layer, q].mean(0)
```

Der **ungeteilte** Arm legt einen Winkel je Amplitudenpaar an — und mittelt sie
dann zu einem einzigen, der über `apply_1q` auf alle Paare wirkt. Nachgemessen
bei d=64:

| | Parameter | wirksame Freiheitsgrade | Abweichung zum geteilten Arm |
|---|---|---|---|
| `butterfly_ffn` | 61 | 61 | — |
| `butterfly_untied` | **1.921** | **61** | **exakt 0,0** |

Q-013s vorregistriertes P2 — *„butterfly_ffn gegen butterfly_untied: trägt die
Gewichtsteilung?"* — hätte damit eine Sache gegen sich selbst gestellt und
zuverlässig „gleichwertig" gemeldet. **Das wäre als Befund ins Tagebuch
gegangen.**

Behoben und dreifach geprüft: Ein einzelner Paar-Winkel wirkt jetzt (2,1520
statt 0,5034 über den Mittelwert); setzt man alle Paar-Winkel gleich, kommt
exakt der geteilte Arm heraus (Abweichung 0,0); und sobald einer abweicht,
trennen sich die beiden (1,4939).

Derselbe Fehler war in `butterfly_layer.py` schon einmal gefunden und dort
behoben worden — in `sequence_platform.py` nicht. **Ein Fehler, der an einer
Stelle repariert wird, ist nicht an allen repariert.** Q-013 war nie gestartet,
es ist also kein Ergebnis betroffen.

### Der Korpus

Deutsche Wikipedia, Stand 20231101, ein Shard von 745 MB über HuggingFace.
140.218 Artikel behalten, 2.048 unter 200 Zeichen verworfen. BPE mit 4.096
Stücken, trainiert auf 200 MB Ausschnitt.

> **397,4 Mio. Trainingstoken, 2,0 Mio. zur Validierung.**

Die Zahl, auf die es ankommt:

> **Zeichen je Token: 3,27** — englisches BPE liegt typisch bei 4.

Deutsch zerfällt bei gleichem Wortschatz also um rund **22 % stärker**. Bei
gleicher Tokenzahl sieht das Modell entsprechend weniger Inhalt. Das war mein
Einwand gegen Deutsch als Korpussprache, und er ist jetzt beziffert statt
vermutet. Es ist kein Ausschlussgrund, gehört aber in die Arbeit.

### Durchsatz je Arm — und eine Messung, die nicht reproduziert

| Arm | Parameter | Strukturteil | s/Schritt | Token/s |
|---|---|---|---|---|
| dense | 5.289.472 | 2.102.272 | 0,371 | 5.522 |
| butterfly_ffn | 3.187.524 | **324** | 2,141 | 956 |
| butterfly_qkv | 4.500.448 | **480** | 2,170 | 944 |

Der Strukturteil schrumpft von 2,1 Millionen auf **324** — Faktor 6.488. Am
Gesamtmodell gemessen nur von 5,29 auf 3,19 Millionen, weil Einbettung und
Ausgabekopf mit 2,1 Millionen unverändert bleiben. Die Unterscheidung, die seit
dem 5. August bei jeder Nennung mitgeführt wird.

**Die Butterfly-Arme brauchen das 5,9-fache der Wanduhrzeit.** Erwartbar: bei
d_model 256 liegt die Struktur weit unterhalb ihres Übergangspunkts von 16.384
(T-001).

**Und eine Unstimmigkeit, die ich festhalte statt sie zu glätten:** T-000 hat für
dasselbe dichte Modell **3.267 Token/s** gemessen, diese Messung **5.522** —
Faktor 1,7 auseinander, bei exakt übereinstimmender Parameterzahl und
Architektur. Beide Messungen sind schwach (acht bzw. zehn Schritte, einmalig,
ohne die sieben abwechselnden Reihen aus T-001). **Keine der beiden absoluten
Zahlen ist belastbar.** Belastbarer ist das Verhältnis 5,9× zwischen den Armen,
weil es innerhalb eines Laufs unter gleichen Bedingungen entstand. Bevor eine
Budgetzahl in die Arbeit geht, wird der Durchsatz mit der T-001-Methodik neu
gemessen.

### T-002 vorregistriert

Vier Arme: `dense`, `butterfly_ffn`, `butterfly_qkv`, `narrow`. Der letzte ist
die Auflösungsprüfung — dieselbe Bauform wie dense mit einem Viertel der
FFN-Breite. Schlägt dense ihn nicht, sieht der Aufbau Kapazitätsunterschiede im
FFN gar nicht.

Für Q/K/V wird die **reelle orthogonale Butterfly ohne Born-Regel** benutzt.
`BEFUNDE_UND_RICHTUNGEN.md` hält fest, dass diese Projektionen linear sind — eine
Wahrscheinlichkeitsverteilung als Q-Vektor wäre sinnlos. Orthogonalität
nachgemessen: **U·Uᵀ = I auf 6,0·10⁻⁷**, 40 Winkel gegen 65.536 Gewichte.

**8 Seeds, nicht 5.** Bei n Seeds ist 2/2ⁿ der kleinstmögliche Wilcoxon-Wert:
n=5 gibt 0,0625, Signifikanz wäre unerreichbar — genau der Fehler aus Q-009,
und Q-013 war mit 5 Seeds geplant und hätte ihn geerbt. Bei n=8 sind es 0,0078,
nach Holm über zwei Vergleiche 0,0156.

**Gleiche Gradientenschritte, nicht gleiche Zeit.** Sonst misst die
Qualitätsaussage den Umsetzungsnachteil aus T-001 mit. Die Wanduhrzeit wird
getrennt daneben berichtet. Dieselbe Lehre wie aus B-010.

### Die Probe hat sofort einen Absturz gefunden

Erster Durchlauf des kurzen Testlaufs: `pct_start = WARMUP_SCHRITTE / schritte
= 200/60 = 3,33`, und `OneCycleLR` bricht ab. Auf dem gemieteten Pod wäre das
nach bezahlter GPU-Zeit passiert. Behoben.

## 27. August — Der Budget-Fehler steckt in neun Modulen. Die Befunde halten.

Nach der eigenen Regel — *wird ein Fehler gefunden, wird nach seiner Klasse
gesucht* — habe ich alle Module auf die halbierte Lernratensuche durchsucht.

**Neun Module** benutzen `max_samples=2000, epochs=6` in der Suche, während ihr
Messlauf mit 4.000 und 12 rechnet:

`breadth_battery` · `butterfly_encoder` · `circuit_vs_replica` ·
`frozen_angles` · `global_orthogonal` · `hybrid_replication` · `qubo_pruning` ·
`reuploading` · `weight_tying`

Drei davon tragen **zitierte** Zahlen: Q-003-replication (die Kernaussage von
Projekt A), Q-015 und Q-018.

### Die Richtung entscheidet, nicht das Vorhandensein

Der Artefakt begünstigt den Arm mit der **kleineren** Rate — das ist aus dem
Q-011-Lauf mit ±0,013 beziffert. Für jeden zitierten Vergleich lässt sich also
ausrechnen, ob er den Befund stützt oder ihm entgegenläuft:

> **Korrektur an dieser Stelle, noch am selben Abend.** Ich hatte zunächst
> geschrieben, „der Arm mit der kleineren Rate gewinnt" gelte allgemein, und
> daraus geschlossen, Q-003 und Q-015 seien konservativ verzerrt. Das ist
> falsch. Die Regel gilt nur, solange die kleinere Rate *näher am Optimum*
> liegt. Sitzt ein Arm auf dem **unteren Gitterrand**, ist er dagegen
> **untertuniert** — und ein untertunierter *Kontrollarm* vergrößert den
> Abstand, und zwar zugunsten der Hypothese. Entscheidend ist also nicht, wer
> die kleinere Rate hat, sondern **an welchem Rand** er sitzt.

Mit den tatsächlichen Randlagen aus den Rohdaten:

| Lauf | Randlage | Wirkung auf den Befund | Effekt / Artefakt |
|---|---|---|---|
| **Q-003, MNIST** | `hybrid` am **oberen** Rand (0,1) | **gegen** — Vorsprung eher zu klein | **7,2×** |
| Q-003, Fashion | `classical_norm_deep` am **unteren** Rand | **für** — Vorsprung aufgebläht | 4,6× |
| **Q-015, MNIST** | beide **innen** | keine | **5,2×** |
| Q-015, Fashion | `global_orth` am **unteren** Rand | **für** — Vorsprung aufgebläht | **1,5×** |
| **Q-018** | Primärpaar beide **innen** | keine | **31,4×** / 28,0× |

**Q-015 auf Fashion-MNIST wird zurückgezogen.** Ein untertunierter Kontrollarm
und nur das 1,5-fache des Artefakts — das trägt nicht. Die MNIST-Zahl (+0,0674,
beide Arme innen, 5,2×) trägt und bleibt.

**Q-003 auf MNIST ist sauber und eher untertrieben** — `hybrid` sitzt am oberen
Rand, was schon der Eintrag vom 5. August vermerkte („der wahre Vorteil könnte
noch größer sein"). Die Fashion-Zahl (+0,0599) ist dagegen aufgebläht, hält aber
bei 4,6-fachem Abstand zum Artefakt; sie gehört mit diesem Vorbehalt zitiert.

**Q-018 ist unberührt** — im Primärpaar sitzt kein Arm am Rand, und bei
31-fachem Effekt wäre es ohnehin gleichgültig. (`untied_trained` sitzt bei 32
Haufen am unteren Rand, gehört aber zu P2, das wegen Sättigung ohnehin nicht
zitierfähig ist.)

### Und die Lehre über die Lehre

Meine erste Fassung dieses Eintrags war selbst falsch — ich hatte eine Regel
verallgemeinert, die nur unter einer Bedingung gilt, und daraufhin zwei Zahlen
als *sicher* eingestuft, von denen eine zurückgezogen werden muss.

Das ist innerhalb eines Tages der zweite Fall: Auch die erste Korrektur des
fehlenden Q-011-Zweigs prüfte den falschen Rand. Die Regel „eine Korrektur ist
kein Beleg, sondern eine neue Behauptung" gilt offenbar besonders für Regeln,
die man gerade erst aus einem Fehler gewonnen hat — sie fühlen sich sicherer an,
als sie sind.

### Was noch zu tun ist

Die sechs übrigen Module haben noch keine zitierten Zahlen erzeugt.
`weight_tying` gehört zu Q-014, das ohnehin nicht verwertbar ist. Alle sechs
müssen **vor** ihrem jeweiligen Lauf auf volles Budget umgestellt werden;
`unitarity_depth` und `frozen_angles` sind es bereits.

### Was ich daraus mitnehme

Ein Fehler in einer Vorlage vermehrt sich mit ihr. Die neun Module sind
erkennbar voneinander abgeschrieben — dieselbe Zeile, dieselben Zahlen, dieselbe
Struktur. Das ist bequem und richtig, solange die Vorlage stimmt, und ein
Vervielfältiger, sobald sie es nicht tut. Dieselbe Beobachtung wie beim
ungeteilten Butterfly-Pfad, der von `butterfly_layer.py` nach
`sequence_platform.py` gewandert ist, und beim asymptotischen Spearman.

> **Vor dem Kopieren einer Datei als Vorlage: gegen die aktuellen
> Protokollregeln prüfen.** Sie wurde geschrieben, als die Regel noch nicht
> existierte.

Und der beruhigende Teil: Dass die Befunde halten, ist kein Glück, sondern liegt
daran, dass sie **groß** sind. Ein Effekt von +0,0932 überlebt einen Störeinfluss
von ±0,013; ein Effekt von +0,005 nicht. Genau deshalb ist Q-011 an ihm
gescheitert und die Kernaussage nicht.

## 27. August — Q-011, zweiter Abbruch: die Lernrate war der ganze Effekt

Der zweite Anlauf lief fünf Tiefen weit, dann habe ich die echten 15-Seed-Zahlen
angesehen. Sie sehen zunächst nach einem starken Ergebnis aus:

| Tiefe | orth − spec | Siege | p |
|---|---|---|---|
| 1 | **+0,0146** | 15/15 | 0,0001 |
| 2 | **−0,0132** | 0/15 | 0,0001 |
| 3 | **+0,0121** | 14/15 | 0,0003 |
| 4 | −0,0006 | 6/15 | 0,5995 |
| 8 | **+0,0122** | 14/15 | 0,0006 |

Jede Zelle hochsignifikant — und die Vorzeichen wechseln. 15/15 in die eine
Richtung, 0/15 in die andere. Das ist kein Rauschen, das ist ein Mechanismus.
Nur nicht der gesuchte.

### Der Mechanismus ist die Lernrate

| Tiefe | lr orthogonal | lr spectral | Gewinner |
|---|---|---|---|
| 1 | **0,003** | 0,01 | orthogonal |
| 2 | 0,01 | **0,003** | spectral |
| 3 | **0,003** | 0,01 | orthogonal |
| 4 | 0,01 | 0,01 | unentschieden |
| 8 | **0,003** | 0,01 | orthogonal |

**In fünf von fünf Tiefen gewinnt der Arm mit der kleineren Rate.** Sind beide
gleich, ist es unentschieden. Der gemessene „Unterschied zwischen Orthogonalität
und Spektralnormalisierung" ist der Unterschied zwischen 0,003 und 0,01.

### Warum die Lernratensuche das nicht gemerkt hat

Sie lief mit **halbiertem Budget**: 3.000 Beispiele und 5 Epochen, während der
Messlauf 6.000 und 10 benutzt. Bei kleinerem Budget sehen größere Raten besser
aus — es bleibt weniger Zeit zum Konvergieren, also gewinnt, wer schneller
vorankommt. Die Suche wählt deshalb **systematisch zu hoch**, und welchen Arm es
in einer bestimmten Tiefe erwischt, entscheidet das Rauschen zweier Tuning-Seeds.

Das ist kein Zufallsfehler, sondern eine **Verzerrung mit Vorzeichen**. Die
Suche ist als Schätzer für das Optimum bei vollem Budget schlicht ungeeignet.

### Die Größenordnung ist das eigentlich Beunruhigende

Der Lernraten-Artefakt beträgt **±0,013**. Der Effekt, den Q-011 finden soll,
liegt laut B-007 bei **+0,0025 bis +0,0051** — also bei einem Drittel bis einem
Fünftel davon. Ein Störeinfluss, der den gesuchten Effekt um das Dreifache
übertrifft, macht jede Aussage unmöglich.

### Und es stützt einen älteren Verdacht

Am 10. August steht zu Q-014 im Tagebuch: *„Die Trennung folgt der Lernrate,
nicht der Teilung"* — vier Arme bei lr 0,1 lagen zwischen 0,934 und 0,936, die
zwei bei lr 0,03 bei 0,910. Das war damals ein Verdacht. Jetzt ist es belegt:
Eine Lernratendifferenz dieser Art erzeugt in diesem Aufbau reproduzierbar
Effekte von ±0,013. Q-014s +0,0257 lag in derselben Größenordnung.

Q-014 war ohnehin nicht verwertbar. Aber der Grund ist jetzt ein zweiter.

### Behoben

Die Lernratensuche läuft ab sofort mit **demselben Budget wie der Messlauf**
(6.000 Beispiele, 10 Epochen) und mit **drei statt zwei Tuning-Seeds**. Das
verteuert den Lauf spürbar — die Suche kostet jetzt so viel wie echte Zellen —,
aber eine billige Suche, die das falsche Optimum liefert, ist teurer.

### Regel 1, geschärft

> **Die Lernratensuche muss unter denselben Bedingungen laufen wie der Messlauf.**
> Ein reduziertes Budget ist kein Sparmodell, sondern ein anderer Versuch: Es
> verschiebt das Optimum systematisch nach oben. Wer bei halbem Budget sucht und
> bei vollem misst, misst die Suche mit.

Damit hat Regel 1 jetzt drei Teile: Rate je Arm, Gitter muss das Optimum
enthalten, **und die Suche muss dasselbe Budget haben wie der Lauf**.

### Zwei Abbrüche an einem Tag

Der erste (nach 40 Minuten) galt einer Sorge, die sich als unbegründet erwies —
0,01 war bei Tiefe 2 tatsächlich das Optimum. Der zweite (nach 90 Minuten) galt
einem echten, belegten Fehler.

Beide waren richtig. Der erste hat die Gitterverlängerung gebracht, ohne die der
zweite Fehler gar nicht sichtbar geworden wäre: Erst als alle Arme ihr Optimum
im Inneren hatten, ließ sich die Lernrate als Ursache isolieren.

## 27. August — Machtanalyse: Regel 2 war zu schwach formuliert

Während Q-011 lief, habe ich aus den Rohdaten von B-007 ausgerechnet, welche
Effektgrößen der Aufbau überhaupt finden kann. Das Ergebnis ändert die Deutung
eines bestehenden Befunds und die Regel selbst.

| | Effekt (B-007) | SD der Paardifferenz | mit 15 Seeds nachweisbar ab |
|---|---|---|---|
| MNIST, Tiefe 1 | +0,0025 | 0,0030 | **+0,0022** ✓ |
| MNIST, Tiefe 3 | +0,0051 | 0,0043 | **+0,0031** ✓ |
| Fashion, Tiefe 1 | +0,0048 | **0,0123** | **+0,0089** ✗ |
| Fashion, Tiefe 3 | −0,0045 | **0,0126** | **+0,0091** ✗ |

(Nachweisgrenze bei 80 % Macht und α=0,05, gepaart.)

### Was das für B-007 heißt

**Auf Fashion-MNIST ist die Streuung viermal so groß wie auf MNIST.** B-007s
dort gemessene Effekte (+0,0048 und −0,0045) liegen damit *unter* der
Nachweisgrenze — selbst mit den 20 Seeds, die B-007 tatsächlich hatte (Grenze
dort +0,0077).

Die bisherige Lesart lautete: *„auf MNIST wirkt Orthogonalität nachweislich
über bloße Skalenkontrolle hinaus, auf Fashion-MNIST nicht."* Der zweite
Halbsatz ist so nicht haltbar. Richtig ist: **Auf Fashion-MNIST hatte der
Aufbau nie die Macht, einen Effekt dieser Größe zu finden.** Das Urteil
„unentschieden" bleibt korrekt — aber es bedeutet „nicht messbar", nicht
„nicht vorhanden".

Das ist derselbe Fehlertyp wie bei Befund 4 selbst: eine Frage bei Bedingungen
gestellt, unter denen sie nicht auflösbar ist. Nur diesmal nicht bei der Tiefe,
sondern bei der Stichprobengröße.

### Was das für Q-011 heißt, vor den Ergebnissen aufgeschrieben

Q-011 fährt 15 Seeds. Auf **MNIST reicht das** — die Nachweisgrenze (+0,0022 bis
+0,0031) liegt unter den erwarteten Effekten. Auf **Fashion-MNIST reicht es
nicht**: dort wäre erst ab etwa +0,009 etwas zu sehen.

Die Vorregistrierung verlangt für H1 und H0−, dass die Wiederholung auf
Fashion-MNIST *dieselbe Richtung* zeigt — nicht, dass sie signifikant ist. Das
ist die schwächere und hier einzig sinnvolle Forderung. Trotzdem gilt:

> **Zeigt Fashion-MNIST eine andere Richtung, ist das kein Gegenbeweis,
> sondern mit hoher Wahrscheinlichkeit Rauschen.** Bei einer SD von 0,0126 und
> 15 Seeds ist die Richtung eines Effekts von 0,005 praktisch ein Münzwurf.

Das steht hier, bevor Q-011 Zahlen liefert. Hinterher geschrieben wäre es eine
Ausrede.

### Und die Regel selbst wird geschärft

Regel 2 lautete: *„Genug Seeds. Bei n Seeds ist 2/2ⁿ der kleinstmögliche
Wilcoxon-p-Wert. Vor dem Lauf prüfen, ob Signifikanz überhaupt erreichbar ist."*

Das ist notwendig, aber nicht hinreichend. Es prüft, ob der **kleinstmögliche**
p-Wert unter 0,05 liegt — nicht, ob der **erwartete Effekt** gefunden werden
kann. Q-009 ist an der ersten Hürde gescheitert; B-007 auf Fashion-MNIST an der
zweiten, und das ist bis heute unbemerkt geblieben.

> **Regel 2, neu:** Vor dem Lauf zweierlei prüfen. Erstens, ob Signifikanz
> rechnerisch erreichbar ist (2/2ⁿ < α). Zweitens, ob der **erwartete Effekt**
> bei der vorhandenen Streuung nachweisbar ist — die Streuung dafür aus einem
> Vorlauf oder verwandten Datensatz schätzen. Ist sie es nicht, ist der Lauf
> ein Nullbefund mit Ansage.

Für künftige Läufe heißt das konkret: Datensätze mit hoher Streuung brauchen
mehr Seeds als solche mit niedriger. Ein pauschales „15 Seeds für alles" ist
zu grob — auf Fashion-MNIST wären für dieselbe Empfindlichkeit rund
**130 Seeds** nötig — gegen 15 im aktuellen Lauf.

## 27. August — Q-011 vor dem Start geprüft: fünf Fehler, einer davon still

Der Vortest vor Q-011 hat fünf Fehler gefunden. Zwei davon hätten den Lauf
falsch berichtet, einer hätte ihn unbemerkt entwertet.

### 1. Der Primärtest benutzte scipys asymptotischen p-Wert

Der Trend über die Tiefen wurde mit `stats.spearmanr(...).pvalue` geprüft.
Nachgemessen bei vier Punkten mit perfekt monotonem Zuwachs:

| | p-Wert |
|---|---|
| scipy, asymptotisch | **0,0000** |
| exakter Permutationstest über alle 4! = 24 Anordnungen | **0,0417** |

Faktor über 400. **Das ist derselbe Fehler, an dem Q-009 gescheitert ist** — im
Tagebuch steht seit dem 6. August: *„Spearman über 3 Punkte: scipy meldet
p=0,0000, der exakte Permutationstest gibt p=0,3333."* `unitarity_depth.py`
wurde am selben Tag geschrieben und hat ihn geerbt.

Behoben: eigener exakter Permutationstest, und **einseitig**, wie die
Vorregistrierung es verlangt — scipy liefert zweiseitig.

### 2. Vier Tiefen konnten den Primärtest gar nicht bestehen

Bei n Punkten ist 1/n! der kleinstmögliche exakte p-Wert:

| Tiefen | kleinstmöglich | nach Holm über zwei Vergleiche |
|---|---|---|
| 4 | 0,0417 | **0,0833** — unerreichbar |
| 5 | 0,0083 | 0,0167 |
| **6** | **0,0014** | **0,0028** |

Mit vier Tiefen hätte der Primärtest nur bestehen können, wenn P2 einen
*kleineren* p-Wert liefert — er wäre also nicht eigenständig gewesen. Dieselbe
Klasse Entwurfsfehler wie Q-009, nur subtiler.

Behoben: **Tiefen 1, 2, 3, 4, 8, 16.** Eins und drei bleiben für die
Vergleichbarkeit mit B-007; zwei und vier sind die billigsten Ergänzungen
(Tiefensumme 28 → 34, also rund ein Viertel mehr Rechenzeit).

### 3. Kein Zweig für „der Abstand schrumpft mit der Tiefe"

Ein synthetischer Fall mit **ρ = −1,000** — perfekt monotoner Abfall — wurde
gemeldet als:

> „H0 — kein Zuwachs mit der Tiefe; bei Tiefe 16 sind die beiden gleichwertig.
> Befund 4 hält."

Das ist nicht bloß ein fehlender Zweig, sondern eine **Aussage, die den Daten
widerspricht**. Dritter Fall desselben Fehlers nach Q-014 und Q-015.

### 4. Und mein erster Korrekturversuch war selbst falsch

Der ergänzte Zweig prüfte `p < 0,05` auf dem einseitig-**positiven** Test. Der
meldet bei fallendem Trend p = 1,0 und kann nie unter 0,05 kommen — der Zweig
hätte nie gefeuert. Erst der Vortest hat das gezeigt. Behoben: Der
Permutationstest liefert jetzt **beide Ränder**.

### 5. Der stille Fehler: die Auflösungsprüfung wurde ignoriert

Der schwerwiegendste Fund, und ohne Vortest unsichtbar:

```python
res_ok = o1.mean() > pl1.mean() and rp < 0.05   # ergibt np.bool_
...
if res_ok is False:                              # np.False_ is False -> False
```

`o1.mean() > pl1.mean()` liefert einen NumPy-Wahrheitswert, und der
Identitätsvergleich `is False` greift darauf **nicht**. Die Auflösungsprüfung
wurde also berechnet, ausgegeben — und beim Urteil übergangen. Im Vortest
meldete Szenario D „Auflösungsprüfung NICHT BESTANDEN" und direkt darunter
„BEWERTUNG: H1".

Ein Lauf mit nachweislich blindem Aufbau hätte ein Ergebnis berichtet.

### Eine Grenze des Verfahrens, die bleibt

Szenario B war konstruktionsbedingt **flach** — alle Abstände 0,004, nur
Messrauschen — und ergab trotzdem ρ = −0,886. Spearman rechnet auf den sechs
Mittelwerten und ignoriert deren Streuung; Rauschen kann eine Scheinordnung
erzeugen.

Das ist kein Codefehler, sondern eine Eigenschaft des Tests. Konsequenz, vorher
festgelegt: **Sowohl H1 als auch H0− gelten nur als Befund, wenn die
Wiederholung auf Fashion-MNIST dieselbe Richtung zeigt.** Sonst Beobachtung.

### Was ich daraus mitnehme

Der Vortest kostet zehn Minuten und hat jetzt zum fünften Mal in Folge etwas
gefunden. Bemerkenswert ist diesmal Nummer 4: **Auch meine eigene Korrektur war
fehlerhaft und wurde nur gefunden, weil ich sie nochmal gegen synthetische
Daten gehalten habe.** Eine Reparatur ist kein Beleg — sie ist eine neue
Behauptung und gehört genauso geprüft wie die alte.

### Nachtrag zu Q-011: Lauf nach zwei Tiefen abgebrochen und neu gestartet

Der erste Anlauf lief 40 Minuten, dann habe ich ihn gestoppt. Bei Tiefe 2 trat
die ungünstigste denkbare Kombination auf:

```
orthogonal -> lr 0,01   (oberer Gitterrand, stieg noch: 0,9274 -> 0,9311)
spectral   -> lr 0,003  (innen, Optimum gefunden: 0,9184 -> 0,9165 fällt)
```

**Der Mechanismus-Arm war untertuniert, der Kontrollarm nicht.** Der
Primärabstand (orthogonal − spectral) wäre damit systematisch zu klein gemessen
worden — also *gegen* die eigene Hypothese. Das ist zwar die konservative
Richtung, aber Q-011 existiert genau deshalb, weil Befund 4 bei einer Tiefe
gemessen wurde, bei der die Frage nicht auflösbar war. Denselben Fehler mit
einer schiefen Lernrate zu wiederholen wäre absurd.

Behoben mit derselben automatischen Gitterverlängerung, die sich bei T-002
bewährt hat: Landet ein Arm auf dem oberen Rand, wird verdreifacht und
nachgemessen, bis zu dreimal. Dort hat das Verfahren für `dense` die Kette
0,0024 → 0,0048 → 0,0096 durchlaufen und das Optimum ins Innere geholt.

**Kosten des Abbruchs: 40 Minuten.** Kosten, hätte ich es laufen lassen: ein
sechs- bis siebenstündiger Lauf, dessen Primärvergleich zugunsten der
Nullhypothese verzerrt ist — und der Verdacht wäre erst bei der Auswertung
aufgefallen, wenn überhaupt.

## 27. August — B-014s p-Werte sind Pseudoreplikation. Die Befunde halten.

Beim Durchsuchen des Codes nach der Spearman-Fehlerklasse aus Q-011 bin ich auf
`qire_ai/intrinsic_dimension.py` gestoßen — B-014, den tragenden Befund von
Projekt B. Dort wird über **alle 240 Einzelmessungen** korreliert:

```python
for d in dims:
    for c in classes:
        for v_ in cell(d, c, "effective_rank"):
            d_all.append(d); r_all.append(float(v_))
rho_d, p_d = stats.spearmanr(d_all, r_all)
```

Das Gitter hat aber nur **24 Zellen** (6 Datendimensionen × 4 Klassenzahlen),
mit je 10 Seeds. Die zehn Seeds einer Zelle tragen alle **dasselbe d und
dieselbe Klassenzahl** — sie sind zehn Messungen *einer* Beobachtung des
Zusammenhangs, nicht zehn unabhängige Beobachtungen. Der Test bekommt so eine
zehnfach zu große Stichprobe vorgesetzt. Das ist **Pseudoreplikation**.

### Nachgerechnet, beides aus derselben Rohdatendatei

| | über 240 Messungen | über 24 Zellmittel |
|---|---|---|
| untrainiert, d → Rang | ρ=+0,986 · **p=3,1·10⁻¹⁸⁷** | ρ=+0,987 · **p=6,5·10⁻¹⁹** |
| untrainiert, C → Rang | ρ=+0,002 · p=0,98 | ρ=+0,027 · p=0,90 |
| trainiert, d → Rang | ρ=+0,565 · p=1,1·10⁻²¹ | ρ=+0,606 · **p=1,7·10⁻³** |
| trainiert, C → Rang | ρ=+0,575 · p=1,7·10⁻²² | ρ=+0,673 · **p=3,1·10⁻⁴** |

### Was das ändert und was nicht

**Die Befunde halten vollständig.** Die Korrelationsstärken verschieben sich
kaum (0,986 → 0,987; 0,565 → 0,606), und jeder p-Wert bleibt auch bei ehrlicher
Stichprobengröße deutlich signifikant. Die Aussage — untrainiert folgt der Rang
der Datendimension und ist klassenunabhängig, trainiert hängt er von beiden ab —
ist unberührt.

**Die berichteten p-Werte sind es nicht.** 3·10⁻¹⁸⁷ ist um 168 Größenordnungen
zu klein und wäre in einer schriftlichen Arbeit angreifbar; eine solche Zahl
sagt schon auf den ersten Blick, dass etwas an der Stichprobe nicht stimmt. Ich
habe sie monatelang zitiert, ohne stutzig zu werden.

**Ab sofort gelten die Zellmittel-Werte.** Für die Arbeit:

> Untrainiert folgt der Effektivrang der Datendimension (ρ=+0,987, p=6·10⁻¹⁹
> über 24 Gitterzellen) und ist von der Klassenzahl unabhängig (ρ=+0,03,
> p=0,90). Trainiert hängt er von beiden ab (d: ρ=+0,61, p=0,002; Klassen:
> ρ=+0,67, p=0,0003).

Die alten Zahlen bleiben oben stehen, wie alle zurückgenommenen Angaben dieses
Tagebuchs — aber sie sind hiermit ersetzt.

### Und der Anlass ist bemerkenswert

Gefunden habe ich das nicht durch eine Prüfung von B-014, sondern weil ich nach
dem Q-011-Fehler **den ganzen Code nach derselben Fehlerklasse durchsucht**
habe. Die Suche ergab 18 Stellen mit `stats.spearmanr`; B-014 war die einzige,
deren Zahlen im Tagebuch stehen.

Daraus folgt eine Arbeitsregel: **Wenn ein Fehler gefunden wird, wird nicht nur
er behoben, sondern nach seiner Klasse gesucht.** Derselbe Gedanke hat am
10. August schon einmal getragen, als der ungeteilte Butterfly-Pfad in
`sequence_platform.py` gefunden wurde, nachdem er in `butterfly_layer.py`
längst repariert war.

**Zu prüfen bleiben** die übrigen 17 Fundstellen, sobald ihre Läufe Zahlen
liefern, die zitiert werden sollen.

## 27. August, spät abends — Das halbe Suchbudget steckt in zehn Modulen

> **Korrektur, unmittelbar nach dem Schreiben.** Dieser Eintrag ist parallel zu
> dem weiter oben („Der Budget-Fehler steckt in neun Modulen") entstanden — wir
> haben denselben Befund unabhängig aufgeschrieben. Der obere ist an zwei
> Stellen richtiger, und diese hier sind entsprechend falsch:
>
> 1. **Meine Richtungsregel ist zu grob.** Ich habe aus Q-011 abgeleitet: „Der
>    Arm mit der kleineren Rate gewinnt, also läuft die Verzerrung gegen die
>    Hypothese, wenn der Mechanismus-Arm die größere Rate hat." Das gilt nur,
>    solange die kleinere Rate *näher am Optimum* liegt. Maßgeblich ist **an
>    welchem Gitterrand** ein Arm sitzt: am oberen Rand ist er untertuniert
>    (verzerrt gegen seinen eigenen Befund), am unteren ebenso — und ein
>    untertunierter *Kontrollarm* bläht den Abstand auf, also *für* die
>    Hypothese. Die Tabelle unten liest deshalb die falsche Spalte.
> 2. **Q-003-replication auf Fashion-MNIST ist nicht konservativ, sondern
>    aufgebläht.** Dort sitzt der Kontrollarm `classical_norm_deep` auf dem
>    unteren Rand (0,003). Die Zahl (+0,0599) hält bei 4,6-fachem Abstand zum
>    Artefakt, gehört aber mit diesem Vorbehalt zitiert, nicht ohne.
> 3. **Beide Modul-Listen waren unvollständig, in verschiedene Richtungen.**
>    Mir fehlte `qubo_pruning` (sucht mit 3.000 gegen 6.000 im Messlauf); dem
>    oberen Eintrag fehlen `width_accuracy` (`EPOCHS // 2`) und
>    `isotime_accuracy` (`EPOCHS - 1`). Die Vereinigung sind **elf** Module,
>    nicht neun und nicht zehn. Nachgezählt gegen den Quelltext.
>
> Übereinstimmend und unstrittig: **Q-015 auf Fashion-MNIST ist zurückgezogen**,
> die MNIST-Zahlen halten, und die Vorlagen-Regel gilt.
>
> Dass zwei unabhängige Fassungen desselben Befunds an verschiedenen Stellen
> danebenlagen, ist selbst ein Datenpunkt: Eine Regel, die man gerade aus einem
> Fehler gewonnen hat, fühlt sich sicherer an, als sie ist.

Die Regel von heute Nachmittag lautet: *Wird ein Fehler gefunden, wird nach
seiner Klasse gesucht.* Der Fehler war das halbierte Budget der Lernratensuche,
an dem Q-011s zweiter Anlauf gescheitert ist. Aufgefallen ist mir die Klasse
nicht durch eine gezielte Suche, sondern beim Lesen von `global_orthogonal.py`
als Vorlage für Q-019 — dort steht in der Suche `max_samples=2000, epochs=6`
und im Messlauf `4000/12`. Exakt dasselbe Muster.

Danach den ganzen Code durchsucht. Das Muster sitzt in **zehn** Modulen:

| Modul | Suche | Messlauf |
|---|---|---|
| `hybrid_replication.py` | 2000 / 6 | 4000 / 12 |
| `global_orthogonal.py` | 2000 / 6 | 4000 / 12 |
| `weight_tying.py` | 2000 / 6 | 4000 / 12 |
| `circuit_vs_replica.py` | 2000 / 6 | 4000 / 12 |
| `butterfly_encoder.py` | 2000 / 6 | 4000 / 12 |
| `breadth_battery.py` | 2000 / 6 | 4000 / 12 |
| `reuploading.py` | 2000 / 6 | 4000 / 12 |
| `frozen_angles.py` | 6 Epochen **und** halber Datensatz | 12 Epochen, voll |
| `width_accuracy.py` | `EPOCHS // 2` | `EPOCHS` |
| `isotime_accuracy.py` | `EPOCHS - 1` | `EPOCHS` |

`frozen_angles.py` ist der schlimmste Fall: dort ist **zweifach** reduziert,
Epochen und Trainingsmenge. `isotime_accuracy.py` ist der mildeste (eine Epoche
weniger von dreien).

### Welche davon haben Rohdaten und zitierte Zahlen

Fünf. Zwei davon sind ohnehin schon als nicht verwertbar geführt (Q-009 aus
`isotime_accuracy`, Q-014 aus `weight_tying`) — für sie ist das jetzt der
dritte Grund. Bleiben drei, und darunter ist die Kernaussage des Projekts.

| Lauf | Effekt | Rate Mechanismus | Rate Kontrolle | Verhältnis zum Artefakt |
|---|---|---|---|---|
| Q-003-replication, mnist | **+0,0932** | 0,1 | 0,01 | 7-fach |
| Q-003-replication, fashion | **+0,0599** | 0,1 | 0,003 | 4,6-fach |
| Q-015, mnist | **+0,0674** | 0,1 | 0,03 | 5-fach |
| Q-015, fashion | **+0,0201** | 0,1 | 0,003 | **1,5-fach** |
| Q-018 | **+0,4077** | 0,1 | 0,3 | 31-fach |

Das Artefakt aus Q-011 beträgt ±0,013. Die letzte Spalte sagt, um welchen
Faktor der gemessene Effekt darüber liegt.

### Die gute Nachricht steht in der Richtung

Q-011 hat nicht nur gezeigt, **dass** die Suche verzerrt, sondern auch
**wohin**: In fünf von fünf Tiefen gewann der Arm mit der **kleineren** Rate.
Die Suche wählt zu hoch, und wen es trifft, entscheidet das Rauschen — aber wen
es trifft, der verliert.

Bei Q-003-replication und Q-015 hat der Mechanismus-Arm (`hybrid`) die
**größere** Rate bekommen, 0,1 gegen 0,01 bzw. 0,03. Nach dem Q-011-Muster
lief die Verzerrung dort also **gegen** die eigene Hypothese. Beide Effekte sind
demnach eher zu klein als zu gross gemessen. Das ist die konservative Richtung,
und sie rettet beide Befunde.

Bei Q-018 ist es umgekehrt — `tied_trained` (0,1) schlägt `tied_frozen` (0,3),
der Gewinner hat die kleinere Rate, also die begünstigende Richtung. Aber bei
einem Effekt von +0,4077 gegen ein Artefakt von 0,013 ist das 31-fache Reserve,
und die Verteilungen überlappen in keinem Paar. Q-018 hält.

### Die schlechte Nachricht ist eine Zahl

**Q-015 auf fashion_mnist, +0,0201, ist nicht mehr zitierfähig.** Zwei Gründe,
die zusammenkommen:

1. Der Effekt liegt beim **1,5-fachen** des bekannten Artefakts. Das ist keine
   Reserve.
2. Der Kontrollarm `global_orth` sitzt dort auf dem **unteren Gitterrand**
   (0,003 bei einem Gitter, das bei 0,003 anfängt) — im Rohdatensatz als
   `lr_at_grid_edge_fashion_mnist: ['global_orth']` sogar vermerkt. Ein
   untertunierter Kontrollarm vergrößert den Abstand. Diese Verzerrung läuft
   **für** die Hypothese, nicht gegen sie.

Die mnist-Zahl (+0,0674) trägt Q-015 weiterhin. Die fashion-Zahl gehört
künftig mit dem Zusatz „Kontrollarm am Gitterrand, nicht interpretierbar"
geführt oder gar nicht.

### Und ein zweiter Mangel, der dabei sichtbar wurde

Q-003-replication hat auf mnist **drei von sechs** Armen am oberen Gitterrand,
auf fashion_mnist **fünf von sechs** an einem der beiden Ränder. Das ist
Regel 1(b) — das Gitter muss das Optimum enthalten —, und sie war am 5. August
noch nicht formuliert.

Wichtiger ist, was das über die **heutige** Korrektur sagt: Die automatische
Gitterverlängerung in `unitarity_depth.py` verlängert nur **nach oben**.
Landet ein Arm auf dem unteren Rand, druckt sie eine Warnung und macht weiter.
Genau dieser Fall — unterer Rand — ist es, der Q-015 auf fashion_mnist
entwertet. Die Verlängerung deckt also die Hälfte des Problems nicht ab, das
sie lösen soll.

Q-011 läuft gerade mit dieser Lücke. Sie zu schließen heißt, den Lauf ein
drittes Mal abzubrechen; das wäre unverhältnismäßig, solange kein Arm
tatsächlich unten landet. Der Lauf protokolliert `lr_edge_*` je Tiefe — **bei
der Auswertung ist das als Erstes zu prüfen.** Für alle künftigen Skripte
gilt die Verlängerung in beide Richtungen.

> **Regel 1(b), ergänzt:** Das Gitter muss das Optimum enthalten, und die
> automatische Verlängerung gilt in **beide** Richtungen. Ein Arm am unteren
> Rand ist genauso untertuniert wie einer am oberen — nur fällt es weniger auf,
> weil die Verzerrung dort häufig für die eigene Hypothese läuft.

### Was ich daraus mitnehme

Die Klassensuche hat diesmal nicht einen Fehler gefunden, sondern zehn
Fundstellen, eine unbrauchbare Zahl und eine Lücke in der Korrektur von heute
Nachmittag. Bemerkenswert ist die Reihenfolge: Ich habe die Klasse nicht
gesucht, sondern bin über sie gestolpert, weil ich eine alte Datei als
**Vorlage** benutzen wollte. Fehler vererben sich über Vorlagen — `Q-011` hatte
den scipy-Spearman aus demselben Grund, und `sequence_platform.py` den
ungeteilten Butterfly-Pfad.

Daraus folgt eine bescheidenere Arbeitsregel als die von heute Nachmittag:

> **Wer eine Datei als Vorlage benutzt, prüft sie zuerst gegen die aktuellen
> Protokollregeln.** Sie ist zu dem Zeitpunkt geschrieben worden, als die Regel
> noch nicht galt.

**Nicht getan, bewusst:** Q-003-replication und Q-015 werden **nicht** neu
gerechnet. Beide Effekte liegen weit über dem Artefakt, und die Verzerrung
läuft bei beiden in die konservative Richtung. Ein Neulauf kostet Stunden und
würde die Aussage nur bestätigen. Das ist eine Abwägung, keine Gewissheit —
sie steht hier, damit sie angreifbar ist.

## 27. August, nachts — Q-019 gebaut, und der Vorlauf nimmt der Erzählung die Grundlage

`quantum_sim/zeitreihen.py` steht. Vortest bestanden, Bauteile verifiziert,
Sättigungs- und Lernratenprüfung gefahren. **Der Lauf ist noch nicht gestartet**
— Q-011 rechnet, und es wird nicht parallel gerechnet. Alles Folgende ist
Vorlauf mit zwei Seeds und steht hier, *bevor* Messzahlen existieren.

### Was der Vorlauf zeigt

Balancierte Genauigkeit, 6.000 Beispiele, 10 Epochen, Rateniveau 0,125:

| Arm | Parameter | periodisch | permutiert |
|---|---|---|---|
| `dense` | 65.536 | **0,913** | **0,915** |
| `fft_fest` | 0 | 0,804 | 0,151 |
| `narrow` | 32.768 | 0,785 | 0,797 |
| `butterfly` | 40 | 0,686 | 0,143 |

Zwei Dinge stehen da, und sie ziehen in verschiedene Richtungen.

### Erstens: die Kontrolle trägt, und zwar spektakulär

`butterfly` und `fft_fest` fallen auf permutierter Zeitachse auf **Rateniveau**
(0,143 und 0,151 gegen 0,125). `dense` und `narrow` bleiben unberührt — eine
freie Matrix kann jede feste Vertauschung einfach rückgängig machen, eine
strukturierte nicht.

Der Beitrag der Struktur hängt also **vollständig** an der Periodizität. Das ist
genau die Abgrenzung, die B-004 für Monarch geliefert hat, nur viel deutlicher:
dort verschwand ein Vorteil von +0,0115, hier verschwindet der halbe Abstand
zum Rateniveau. Die vorregistrierte Wechselwirkung wird dadurch riesig — grob
geschätzt +0,55 gegen eine Streuung in der Größenordnung 0,01.

### Zweitens: die Fourier-Erzählung stimmt so nicht

In `ENTWURF_ZEITREIHEN.md` und im Kopf von `butterfly_layer.py` steht seit
Wochen derselbe Satz: *„Die Butterfly-Struktur **ist** die schnelle
Fouriertransformation."* Der Vorlauf sagt zweimal nein.

**`fft_fest` schlägt `dense` nicht** (0,804 gegen 0,913). Die *echte*
Fouriertransformation, auf einer Aufgabe, die im Frequenzraum trivial ist,
verliert gegen eine freie Matrix. Der Grund ist ausrechenbar und liegt nicht an
Fourier, sondern an der Auslese: `<Z_q>` liefert acht Größen, aber nur die drei
höchstwertigen Bits tragen Bandinformation. Ein 3-Bit-Code für acht Klassen ist
gerade noch eindeutig — solange die Energie in *einem* Band liegt. Bei drei
Anteilen mit Amplitudenabstand 0,15 ist er es nicht mehr. Eine gelernte dichte
Drehung kann die Bandinformation dagegen über alle acht Größen verteilen. **Die
Auslese ist der Engpass, nicht die Basis.**

**Und `butterfly` erreicht `fft_fest` nicht** (0,686 gegen 0,804). Das ist keine
Tuning-Frage — ich habe sechs Lernraten gefahren:

| lr | 0,001 | 0,003 | 0,01 | 0,03 | 0,1 | 0,3 |
|---|---|---|---|---|---|---|
| balancierte Genauigkeit | 0,6045 | 0,6714 | 0,6858 | **0,6862** | 0,6762 | 0,6711 |

Das Optimum liegt im Inneren bei 0,01–0,03, die Kurve ist flach, der Arm ist
ausgetunt. 0,686 ist, was diese Familie kann.

Der Grund ist strukturell und hätte mir vorher auffallen müssen: Die
Butterfly-Schicht hat **geteilte** Winkel — ein Winkel je Lage und Bit, 40
insgesamt. Die schnelle Fouriertransformation braucht aber je Paar einen
**eigenen** Drehfaktor; das ist die *ungeteilte* Variante mit (d/2)·n·L =
5.120 Parametern. Was die Schicht mit der FFT teilt, ist das
**Verbindungsmuster**, nicht die Transformation.

> **Zu korrigieren, überall wo er steht:** „Die Butterfly-Struktur *ist* die
> schnelle Fouriertransformation" ist falsch. Richtig: Sie hat **dasselbe
> Verbindungsmuster** wie die FFT und dadurch dieselbe Kostenordnung
> O(d log d). Die Gewichtsteilung, die sie quantenabgeleitet macht, nimmt ihr
> genau die Freiheitsgrade, aus denen die FFT besteht.

Das ist keine Kleinigkeit. Es war die Begründung dafür, dass Zeitreihen *die*
Anwendung mit erwartetem inhaltlichem Vorteil sind.

### Was das für die Vorregistrierung heißt — vor dem Lauf aufgeschrieben

Der Primärtest (Wechselwirkung) wird mit großer Sicherheit halten. **P2
(`butterfly` schlägt `dense` auf periodisch) wird mit großer Sicherheit
fallen** — der Abstand liegt bei etwa −0,23. Nach der Vorregistrierung ist H1
damit nicht erreichbar; der Lauf landet im Zweig „HU — geteiltes Ergebnis":

> Der Unterschied zwischen den Bedingungen kommt nicht daher, dass die Struktur
> auf periodischen Signalen *gewinnt*, sondern daher, dass sie auf permutierten
> *verliert*.

Dieser Zweig existiert erst seit heute Nacht. Er ist beim Vortest aufgefallen:
Ohne ihn wäre genau dieser Ausgang als „weder Wechselwirkung noch
Gleichwertigkeit belegt" gemeldet worden — obwohl die Wechselwirkung belegt
*ist*. Dieselbe Fehlerklasse wie der fehlende H0−-Zweig in Q-011, und diesmal
hätte sie den wahrscheinlichsten Ausgang des Laufs getroffen.

**Ich ändere die Vorregistrierung nicht.** Ein Primärtest, der hält, und ein
Sekundärtest, der fällt, sind ein Ergebnis. Wer nach dem Vorlauf die Kriterien
umschreibt, bis sie passen, misst nur noch sich selbst.

### Der Vortest hat wieder etwas gefunden — diesmal an sich selbst

Szenario G fiel zunächst durch. Der Fehler lag aber nicht in der Auswertung,
sondern im **Prüfstand**: `dense` bekam gar kein eigenes Rauschen, und der
gemeinsame Seed-Anteil kürzt sich in der gepaarten Differenz weg. Das
„schwache, verrauschte" Szenario war in Wahrheit ein starkes, sauberes.

> **Ein falsch gebauter Prüfstand ist genauso schädlich wie ein falsch gebauter
> Test** — er meldet Fehler, wo keine sind, und irgendwann glaubt man ihm nicht
> mehr.

Danach zeigte sich bei Szenario F, dass das Urteil in zwei von fünf Ziehungen
kippte. Grenzwertig ist kein Urteil. Jedes Szenario läuft jetzt über **fünf**
Rauschziehungen, und geprüft wird nicht die Marke, sondern der **Zweig** — vier
verschiedene Ausgänge heißen „HU", und ein Vortest, der nur auf die Marke
schaut, würde ihr Verschmelzen nicht bemerken.

Aus dem Nachrechnen fiel eine Zahl ab, die vor den Lauf gehört: **H0 ist nur
erreichbar, wenn die Streuung der Wechselwirkung unter 0,0110 liegt**
(0,005·√15/1,7613). Darüber kann der Gleichwertigkeitsausgang gar nicht feuern.
Steht jetzt in der Vorregistrierung, damit hinterher niemand „nicht
eingetreten" liest, wo „unerreichbar" richtig wäre.

### Und die Gitterverlängerung ist geprüft

Die Verlängerung nach unten ist meine Korrektur von heute Nacht — also eine
neue Behauptung, kein Beleg. `test_gitterverlaengerung()` prüft sie gegen eine
künstliche Genauigkeitskurve mit bekanntem Optimum: innen, eine Stufe oben,
eine Stufe unten, außerhalb der Reichweite oben, außerhalb unten. Fünf von
fünf.

Zwei meiner fünf **Erwartungen** waren dabei falsch, nicht der Code: Ein
Optimum genau am Rand der Reichweite wird als „Rand" gemeldet, auch wenn die
gewählte Rate zufällig stimmt. Das ist richtig so — ohne eine Messung jenseits
des Gipfels lässt sich „Gipfel" nicht von „steigt noch" unterscheiden. Die
Reichweite beträgt bei drei Verlängerungen 3,7·10⁻⁵ bis 0,81.

### Stand

Gebaut, geprüft, nicht gelaufen. Der Lauf kostet geschätzt **2,5 bis 3 Stunden**
(die Butterfly-Schicht ist eine Python-Schleife über 40 Einzelschritte und
braucht rund 2,5 Minuten je Zelle, die drei anderen Arme je etwa 10 Sekunden).
Er gehört hinter Q-011, nicht daneben.

## 27. August, nachts — Q-019: der fünfte Arm, und zwei Fehler in meinem eigenen Urteilsbaum

Entscheidung: Q-019 bekommt einen fünften Arm `untied` — dieselbe
Butterfly-Verbindungsstruktur, aber ein **eigener** Winkel je Amplitudenpaar,
also 5·8·128 = **5.120** statt 40 Parameter.

**Er steht ausdrücklich außerhalb der angemeldeten Familie.** Holm korrigiert
weiterhin nur über PRIMÄR und P2, die Vorregistrierung ist unverändert, und der
Arm wird rein beschreibend berichtet. Ein nach dem Vorlauf ergänzter Arm gehört
nicht in eine Familie, die vor dem Vorlauf angemeldet wurde — wer ihn
hineinzöge, hätte die Kriterien nach dem Blick auf die Daten geändert.

### Wozu er da ist

Der Vorlauf hat gezeigt, dass `butterfly` (0,686) die echte
Fouriertransformation (0,804) nicht erreicht. Zwei Erklärungen kommen in Frage,
und sie sind sehr verschieden wichtig:

- Es liegt an der **Gewichtsteilung** — dann ist es eine Aussage über genau den
  Anteil, den dieses Projekt als quantenabgeleitet führt (ein Qubit-Gatter wirkt
  auf ein Qubit, gleichgültig wie groß der Zustandsraum ist).
- Es liegt am **Verbindungsmuster** selbst — dann trägt die
  Fourier-Verwandtschaft die Erzählung überhaupt nicht.

`untied` trennt die beiden. Er hat dieselbe Struktur, aber keine Teilung.

### Und dann habe ich nachgemessen — und die Positivkontrolle hat mich gestoppt

Die Begründung für den Arm enthielt einen Satz, den ich nicht belegen konnte:
*„nur die ungeteilte Variante kann die FFT darstellen."* Der erste Halbsatz —
geteilt kann es nicht — folgt aus der Konstruktion. Der zweite ist eine
Behauptung, und mein Butterfly hat weder Bit-Umkehr-Permutation noch komplexe
Arithmetik.

Also `fourier_erreichbarkeit()` gebaut: die Winkel direkt an die Fouriermatrix
anpassen und den relativen Restfehler berichten. Ergebnis:

| Ziel | Parameter | relativer Restfehler |
|---|---|---|
| Fouriermatrix, `butterfly` (geteilt) | 40 | 1,386 |
| Fouriermatrix, `untied` (ungeteilt) | 5.120 | 0,944 |
| **Positivkontrolle: Ziel liegt IN der Familie** | 5.120 | **0,944** |

Die letzte Zeile erledigt die ersten beiden. Das Ziel der Positivkontrolle ist
eine zufällig gewürfelte ungeteilte Butterfly — sie ist per Konstruktion exakt
darstellbar, der Rest müsste gegen null gehen. Er geht auf **0,9441**, also auf
denselben Wert wie bei Fourier (0,9436).

> **Der Anpassungsvorgang konvergiert nicht. Beide Fourier-Zahlen sind sein
> Boden, keine Darstellbarkeitsgrenze.** Die Frage „kann die ungeteilte
> Butterfly die Fouriertransformation darstellen?" ist damit **offen**, nicht
> beantwortet. Ich hätte sie ohne die Positivkontrolle beantwortet geglaubt —
> und zwar falsch, mit zwei Nachkommastellen.

Das Anpassen eines Produkts aus 40 Drehstufen an eine vorgegebene Matrix ist
ein hartes nichtkonvexes Problem; 2.500 Adam-Schritte reichen sichtbar nicht.
Ein ernsthafter Versuch bräuchte eine bessere Anfangsbelegung oder ein
stufenweises Verfahren. **Bleibt offen**, das Werkzeug samt Positivkontrolle
steht im Modul.

Für Q-019 ändert das nichts: `untied` misst empirisch, **was die
Gewichtsteilung auf dieser Aufgabe kostet**, und dafür braucht es keine Aussage
über Darstellbarkeit. Nur die stärkere Deutung — „die Struktur reicht, die
Teilung ist der Preis" — ist ohne die offene Frage nicht zu haben.

### Zwei Fehler in meinem eigenen Urteilsbaum

Beide gefunden, weil das Hinzufügen von `untied` die Zufallsziehungen des
Vortests verschoben hat und zwei Szenarien kippten.

**Erstens: der Vortest hing an der Reihenfolge des Zufallsgenerators.** Ein
gemeinsamer Strom für alle Arme heißt, dass ein zusätzlicher Arm jede
nachfolgende Ziehung verschiebt. Ein Prüfstand, dessen Ergebnis davon abhängt,
prüft nicht die Auswertung. Jetzt bekommt jedes Tripel (Arm, Bedingung, Seed)
einen eigenen Strom, abgeleitet über `zlib.crc32` — Pythons String-Hash ist
zwischen Prozessen zufällig und taugt dafür nicht.

**Zweitens, und das ist der ernste:** Die Gleichwertigkeitsprüfung stand
**hinter** H1 und H0−. Damit meldete ein wahrer Nulleffekt in rund zehn Prozent
der Fälle „inhaltlicher Vorteil" oder „Gegenrichtung" — obwohl TOST im selben
Atemzug beweist, dass der Effekt **kleiner als die eigene Erheblichkeitsgrenze
von ±0,005** ist. Ein einseitiger Test bei α=0,05 spricht bei einem Nulleffekt
nun einmal in fünf Prozent der Fälle an, in jede Richtung.

Das ist Regel 3, in der anderen Richtung gelesen:

> **„Nicht signifikant" ist nicht „gleich" — aber „signifikant" ist auch nicht
> „erheblich".** Wo TOST die Erheblichkeitsgrenze positiv ausschließt, schlägt
> das den Signifikanzbefund. Sonst berichtet man einen Effekt als Befund, den
> man selbst vorher für zu klein erklärt hat.

Die Vorregistrierung nennt vier Ausgänge, aber keine Rangfolge. Sie steht jetzt
fest, vor dem Lauf: Auflösung → Boden → **H0 (TOST)** → H1 → geteilt → H0− → HU.
Feuert H1 und ist der Effekt trotzdem innerhalb der Grenze, sagt das Urteil das
ausdrücklich dazu.

### Der Vortest prüft jetzt auch das Niveau

Szenario G soll „unentschieden" ergeben, liegt also konstruktionsbedingt
zwischen den Schwellen — und dort ist ein Fehlurteil in fünf Prozent der
Ziehungen **kein Fehler, sondern das Testniveau**. Von 5 von 5 zu verlangen
hieße, vom Test zu fordern, dass er sein eigenes α unterschreitet.

G läuft deshalb über **40 Ziehungen** und prüft den Anteil statt der
Einzelurteile. Gemessen: **37/40 unentschieden, 3/40 fälschlich H0−**, also
7,5 % gegen ein nominelles Niveau von 5 %. Das ist bei 40 Ziehungen im Rahmen —
und wichtiger: es ist jetzt eine **ausgewiesene Zahl** statt einer stillen
Annahme. Die übrigen sieben Szenarien sind deterministisch und müssen in jeder
Ziehung dasselbe liefern.

### Stand

Vortest 8/8 (davon sieben deterministisch), Gittertest 5/5, Bauteile
verifiziert, `untied` normerhaltend mit exakt 5.120 Parametern.

Der Lauf kostet mit fünf Armen jetzt **rund 4,6 Stunden**, mit Reserve für
Gitterverlängerungen **bis 5,5**. Nachgerechnet, nicht geschätzt: 108 langsame
Zellen (`butterfly` und `untied`, je rund 141 Sekunden — beide sind
Python-Schleifen über 40 Einzelschritte) und 162 schnelle zu je rund 8
Sekunden. Er wartet auf das Ende von Q-011; parallel wird nicht gerechnet.

## 30. August — Q-011 ist gestorben, und Projekt A hat jetzt einen eigenen Ordner

### Der dritte Anlauf ist nicht durchgelaufen

Kein Python-Prozess mehr aktiv. Das Log endet am **27. August, 23:58** bei
mnist Tiefe 16, Seed 1810 — zehn von fünfzehn. Kein Abbruchgrund im Log, keine
Ausnahme, kein Speicherfehler: der Prozess ist einfach weg. Das spricht dafür,
dass er mit der Sitzung gestorben ist und nicht an sich selbst.

Das ist der dritte Abbruch von Q-011. Die ersten beiden habe ich selbst
ausgelöst und beide aus gutem Grund. Dieser hier ist der erste, den ich nicht
wollte, und er ist der ärgerlichste, weil der Lauf diesmal methodisch sauber
war.

Gerettet ist trotzdem einiges, und zwar genau wegen Protokollregel 5 — Rohdaten
nach jedem Abschnitt sichern, nicht am Ende:

    results/experiments/Q-011__20260827_230017.json    225 Messzeilen

Darin stecken die Tiefen **1, 2, 3, 4 und 8 vollständig** auf mnist, mit vollem
Suchbudget. Es fehlen Tiefe 16 und der ganze zweite Datensatz. Die Regel, die
nach dem B-007-Verlust von 200 Zellen entstanden ist, hat hier zum zweiten Mal
gehalten, was sie versprach.

### Was die Tuning-Zahlen andeuten, ohne dass ich es zitieren darf

Abstand `orthogonal` minus `spectral` in der Lernratensuche, über drei
Tuning-Seeds:

| Tiefe | 1 | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| Abstand | +0,0049 | +0,0014 | +0,0044 | +0,0050 | +0,0067 | **+0,0232** |

Bei Tiefe 16 fällt `spectral` erstmals deutlich ab — das Fünffache aller
vorherigen Tiefen, und genau die Richtung, die die Hypothese vorhersagt.

**Ich schreibe das hier auf, damit die Beobachtung nicht verlorengeht, und ich
zitiere es nirgends.** Es sind Tuning-Zahlen aus drei Seeds, kein Messergebnis;
sie sind nicht vorregistriert, nicht korrigiert, und der Wert bei Tiefe 16 ist
zudem aus einem abgebrochenen Lauf. Wenn ich anfinge, Suchnebenprodukte als
Befunde zu lesen, hätte ich die ganze Disziplin dieses Sommers weggeworfen.
Der Wert davon ist, dass ich weiß, **wo** ich hinschauen muss, wenn Q-011
richtig durchläuft — nicht, was dabei herauskommt.

### Die Entscheidung, die daraus folgt

`unitarity_depth.py` hat **keine Wiederaufnahme**. Ein Neustart fängt bei Tiefe
1 an und kostet die vollen 10–14 Stunden noch einmal, obwohl fünf von sechs
Tiefen des ersten Datensatzes fertig auf der Platte liegen.

Der offensichtliche Weg ist, erst die Wiederaufnahme einzubauen und dann nur
Tiefe 16 und fashion_mnist nachzurechnen. Das Muster steht in
`sprachmodell.py` — `weiter=True` lädt die früheren Aufzeichnungen und
überspringt fertige Arm/Seed-Paare — und ist dort erprobt. Etwa eine Stunde
Arbeit gegen etwa zehn Stunden Rechnen.

Ein zweiter Punkt gehört mit erledigt, bevor der Lauf wieder startet: die
Gitterverlängerung in `unitarity_depth.py` verlängert **nur nach oben**. Regel
1(b) verlangt beide Richtungen, seit der untere Rand Q-015 auf Fashion-MNIST
entwertet hat. `zeitreihen.py` macht es richtig und ist die Vorlage — und das
ist genau die Regel, dass man eine Datei vor dem Weiterbenutzen gegen die
aktuellen Regeln prüft, diesmal auf die eigene alte Datei angewandt.

### Projekt A liegt jetzt in einem eigenen Ordner

    C:\Users\Malte\Downloads\claude\projects\QIRE AI\A\

Vollständig und eigenständig: Dokumente, Code beider Pakete, alle Q- und
T-Rohdaten, alle Lauf-Logs, die Serverskripte. 7,7 MB, 288 Dateien. Ein neuer
Chat bekommt nur diesen Ordner und findet dort ein `CLAUDE.md`, das automatisch
gelesen wird.

Nachgeprüft, nicht angenommen: `q_audit` läuft aus dem neuen Ordner durch,
**106 von 106**, und `q_summary` ebenso. Das war nicht selbstverständlich —
es geht nur, weil alle Pfade im Projekt über `Path(__file__).parent.parent`
auflösen und nicht über das Arbeitsverzeichnis. Wäre irgendwo ein relativer
Pfad gegen das Arbeitsverzeichnis gestanden, hätte der Umzug ihn gebrochen.

Zwei Dinge sind bewusst draußen geblieben, beide aus Platzgründen — die
Systemplatte hat noch 29 GB von 1,9 TB frei:

- die virtuelle Umgebung (1,3 GB); der neue Ordner ruft sie über den absoluten
  Pfad auf, `py.cmd` kürzt das ab
- der deutsche Korpus (2,9 GB); mitgekommen sind nur `bpe_4096.json` und
  `val.npy`, also der Tokenizer und die Validierungsmenge — die beiden
  kleinen Dateien, die die Reproduzierbarkeit festlegen

Der alte Ordner bleibt als Archiv stehen und trägt einen Wegweiser
(`UMGEZOGEN_PROJEKT_A.md`). `DATEN_EXTERN.md` im neuen Ordner listet auf, was
bei einem Löschen verlorenginge und was sich zu welchem Preis neu erzeugen
lässt. Das ist kein Ordnungsfimmel: 795 MB fertig tokenisierter Text sind eine
Stunde Rechnen, und der Tokenizer dazu ist unersetzlich, weil dieselben Token-IDs
sonst andere Wörter bedeuten.

### Was ich daraus mitnehme

Der Verlust wäre vollständig gewesen, wenn nicht eine Regel aus einem früheren
Verlust im Skript gestanden hätte. Regel 5 ist aus B-007 entstanden — 200
fertige Zellen weg, weil erst am Ende gespeichert wurde. Sie hat jetzt 225
Messzeilen gerettet.

Die Regel, die diesmal gefehlt hat, ist die dazu passende zweite Hälfte:
**wer Zwischenstände sichert, sollte sie auch wieder einlesen können.**
Speichern ohne Wiederaufnahme rettet die Daten, aber nicht die Rechenzeit.
`sprachmodell.py` kann beides, `unitarity_depth.py` nur das eine — und das ist
kein Zufall, sondern schlicht das ältere Skript.

## 23. September — Gesamtreview: was hält, was nicht

*Entwurf aus der Überarbeitung vom 23./24. September, mit KI-Unterstützung
geschrieben wie alle Einträge (siehe Kopf dieses Tagebuchs). Ich prüfe ihn und
übernehme ihn erst dann in die Hauptfassung.*

Seit dem 30. August habe ich nichts gerechnet. Bevor ich weitermache, habe ich
alles einmal gegen alles gelesen: Code, Rohdaten, `q_audit`, `q_summary`, dieses
Tagebuch, die Übergaben und die Kurzfassung. Die vollständige Liste steht in
`BERICHT/00_REVIEW_BEFUNDE.md` im Überarbeitungsordner. Hier das, was meine
Aussagen ändert.

### 1. Die Kontrolle meiner Kernaussage war zu schwach

Seit dem 5. August führe ich als Kernaussage: 16–80 Schaltungsparameter
schlagen ein klassisches Netz mit 67.848 um +0,0932. Geprüft hatte ich diese
Kontrolle immer nur *gegen die Schaltung* — nie gegen etwas Einfacheres.

Genau das wurde jetzt nachgeholt, allerdings nur als **Stichprobe**: 3 Seeds
(2301–2303), gleicher Lader, gleiche Trainingsschleife, Lernrate je Arm auf
denselben Seeds gewählt. **Das ist keine Messung im Sinne meines Protokolls** —
zu wenige Seeds, nicht vorregistriert, Suche und Messung auf denselben Seeds.
Ich führe die Zahlen nur als Anlass, nicht als Befund:

| MNIST, Stichprobe | balancierte Genauigkeit |
|---|---|
| `classical_norm_deep` (die Kontrolle aus Q-003) | 0,8544 |
| rein linear 784→256→8→10 | 0,8873 |
| ReLU-Netz 784→256→256→8→10, ohne Normierung | 0,9246 |
| ohne 8-Kanal-Engstelle | 0,9371 |

Auf Fashion-MNIST dasselbe Bild (Kontrolle 0,7541, linear 0,8037, ReLU 0,8214).

Meine „starke klassische Referenz“ ist also **schlechter als ein lineares
Modell**. Normierung auf Einheitslänge plus Tanh-Engstelle lähmen sie. Der
Abstand +0,0932 misst deshalb vor allem, wie schwach die Kontrolle ist. Die
Zahl ist nicht falsch — sie ist gegen einen Strohmann gemessen, und das habe
ich zwei Monate lang nicht gesehen.

Davon ist auch meine Zerlegung vom 6. August betroffen („neun Zehntel des
Vorteils sind die Born-Regel“): Die 87 % und 92 % sind Anteile an einem Abstand
zu genau dieser Kontrolle. Ich nehme sie als Zahl zurück. Die Beobachtung, dass
dieselbe Operation als Gate nichts bringt und als Auslese trägt, bleibt.

Was trotzdem trägt: **16 Winkel** (0,9337) liegen vor dem klassischen
Quadratmodell mit **2.048** Parametern (0,9213) — nicht vorregistriert, auf
Fashion-MNIST nicht signifikant. Und das ReLU-Netz der Stichprobe (0,9246) hat
in der Mittelschicht ebenfalls rund 68.000 Parameter. **Diese Gegenüberstellung
darf ich aber nicht als Ergebnis schreiben:** 0,9337 stammt aus Q-003, 0,9246 aus
der Stichprobe — zwei Läufe, zwei Seed-Sätze. Genau diese Sorte Vermischung hat
am 6. August das „+0,0044“ erzeugt. Beides muss in *einem* Lauf gemessen werden.
Das ist Q-020 (a).

Daraus eine neue Regel:

**Eine Kontrolle muss selbst gegen eine triviale Referenz (linear) geprüft
werden.** Sonst misst der Abstand die Schwäche der Kontrolle, nicht die Stärke
des Mechanismus.

### 2. Verschränkung und Tiefe — „belegt“ war falsch

Beide Gleichwertigkeiten (−0,0015 und +0,0002, TOST) stammen aus
Q-003-replication. Drei Gründe, warum ich sie nicht mehr als belegt führe:

- **Trainierbarer Encoder.** Am 10. August habe ich an Q-014/Q-018 gelernt:
  *Ein Nullbefund an einem Bauteil ist wertlos, solange ein anderes Bauteil
  dessen Aufgabe übernehmen kann.* Ich habe die Lehre auf Q-014 angewandt, aber
  nicht auf Verschränkung und Tiefe — obwohl dort derselbe Encoder 784→256 in
  jedem Arm steckte.
- **Die Tiefe war die Positivkontrolle — und ist durchgefallen.** Am 5. August
  habe ich geschrieben, sie sei durchgefallen, „nicht weil der Aufbau blind
  wäre, sondern weil es den Unterschied nicht gibt“. Das konnte ich nicht
  wissen. `q_summary` sagt ausdrücklich, dass aus diesem Lauf keine
  Gleichwertigkeitsaussage steht. Die Dokumente haben es trotzdem als
  „positiven Gleichwertigkeitsnachweis“ geführt.
- **Es widerspricht meiner eigenen Theorie.** Ohne CNOT ist jede Tiefe ein
  Produkt aus Ein-Qubit-Drehungen, also ein Kronecker-Produkt U₁⊗…⊗U₈ — dort
  *kann* Tiefe nichts bringen, die Gruppe ist abgeschlossen. Mit Verschränkung
  braucht Universalität dagegen Tiefe; das habe ich am 6. August selbst
  gerechnet („Teilung kostet Tiefe, nicht Parameter“). „Tiefe trägt nichts bei“
  ist allgemein formuliert also mit meiner eigenen Rechnung unvereinbar.

Neuer Status: **mit trainierbarem Encoder und durchgefallener Positivkontrolle
gemessen, Nachtest Q-020 (b)** — Verschränkung × Tiefe im Q-018-Aufbau ohne
Encoder, wo die Winkel nachweislich tragen (+0,4077).

### 3. B-014: korrigiert im Tagebuch, nicht im Code

Am 27. August habe ich die Pseudoreplikation in B-014 gefunden und hier
korrigiert. `intrinsic_dimension.py` rechnete aber weiter über 240
Einzelmessungen (Zeilen 192–202). Wer die Auswertung neu gestartet hätte, hätte
wieder p=3·10⁻¹⁸⁷ bekommen. Dazu:

- Das automatische Urteil sagt „d erklärt mehr als die Klassenzahl“, die
  korrigierten Zellmittel sagen trainiert d ρ=0,61, Klassen ρ=0,67. Es gelten die
  Zellmittel.
- Die Überschrift „Rang wird von der Aufgabe gesetzt, **nicht von der
  Architektur**“ ist überdehnt: Die Architektur habe ich in B-014 gar nicht
  variiert (eine MLP, Breite 4096).
- `qire_ai.audit` prüft nur 15 B-Zahlen — ausgerechnet B-014 nicht.

Die zweite neue Regel:

**Korrekturen müssen im Code landen, nicht nur im Tagebuch.** Eine Korrektur
ist erst erledigt, wenn der Code das Richtige ausgibt und ein Prüfwerkzeug die
Zahl nachrechnet.

### 4. T-002: vier Fehler vor jedem GPU-Lauf

1. Die Lernratensuche läuft 1.000 Schritte, der Messlauf 13.000 bzw. 20.000 —
   Regel 1(c), und schlimmer als das halbe Budget vom 27. August. Das Modul
   fehlte in meiner Liste der elf.
2. Die Wiederaufnahme liest nur die neueste T-002-Datei. Weil `runpod_start.sh`
   vorher die Probe fährt, ist danach die Probe die neueste Datei, und der
   Hauptlauf fängt still bei null an. Meine Notiz vom 30. August („getestet“)
   war zu optimistisch.
3. Die Schrittzahl ist nicht festgelegt (13.000 gegen 20.000).
4. Gespeichert wird der Rückfallwert der Lernrate, nicht die benutzte.

Dazu ein Einwand gegen meine eigene Architektur: `ButterflyFFN` gibt
|Amplitude|² aus — nichtnegativ, mit fester Summe. Der FFN-Block kann damit
nichts vom Residualstrom abziehen. `butterfly_qkv` lag in der Probe fast
gleichauf mit dense. Ob ich einen zusätzlichen, zentrierten Arm außerhalb der
Familie aufnehme, entscheide ich **vor** dem Lauf und schreibe es hier hinein.

### 5. Q-019 und die Dokumente

- Die `untied`-Entscheidung ist am 27. August nachts gefallen; die Übergabe
  führte sie noch als offen. Laufzeit 4,6–5,5 h.
- Die Primärhypothese von Q-019 ist fast per Konstruktion wahr: Eine feste
  Permutation kann eine dichte Schicht mitlernen, eine Butterfly nicht. Das
  zeigt Ordnungsabhängigkeit, keinen Vorteil. Echte Chancen sehe ich bei wenig
  Trainingsdaten und bei Walsh-Hadamard statt Fourier — die geteilte Butterfly
  ohne CNOT enthält die Walsh-Hadamard-Transformation exakt.
- Die Kurzfassung vom 6. August hatte veraltete Zahlen: B-014 vor der
  Korrektur, das vermischte „+0,0044“ in der Zusatztabelle, „1,99×“ statt der
  T-001-Werte (1,59× Referenz, 2,31× schnell), und die Hardware-Tabelle ohne
  Stichprobengröße. Der IBM-Lauf hatte 20 Testbeispiele je Schaltung
  (Übereinstimmung 17/20 bzw. 15/20) und rechnete ein **anderes** Modell
  (Winkelkodierung, 4/6 Qubits) als meine Kernaussage. Neu:
  `KURZFASSUNG_ENTWURF_v2.md`.
- Die Literatur: Eine praktisch gleiche Schaltung ist als Adapter für große
  Sprachmodelle veröffentlicht (Quantum-PEFT, laut
  `LITERATUR_UND_ALLEINSTELLUNG.md`, bisher nur Abstracts). Die Schaltung ist
  also nicht mein Beitrag — die Abgrenzung und die Zerlegung sind es.

### 6. Infrastruktur

- Kein Git. Für ein Projekt, das von Vorregistrierung lebt, ist ein Commit
  *vor* den Rohdaten der stärkste Beleg.
- `ibm_quantum_token.txt` liegt im Klartext im Archivordner. Nach der
  PC-Infektion gilt der Token als kompromittiert.
- Die Counts der IBM-Läufe habe ich nie gespeichert, nur die Zusammenfassung.

### Was hält

- Q-015 auf MNIST: +0,0674 gegen die volle globale Drehung bei 408-fach weniger
  Freiheitsgraden (Fashion bleibt zurückgezogen).
- Q-018: +0,4077, 15/15, Auflösungsprüfung bestanden.
- T-001: Übergangspunkt 16.384, Cache-Kante der dichten Schicht.
- Die Theorie: Konzentrationsprinzip (die Messwerte aber nur im Tagebuch, nicht
  in Rohdaten), Rayleigh-Quotient, Universalität bei dim SU(d) Parametern.
- Aus Projekt B: B-011, B-012, B-014 in der Zellmittel-Fassung.
- 16 Winkel vor dem 2.048-Parameter-Quadratmodell — als Anlass, nicht als Befund.

### Was ich entscheiden muss

1. Q-020 freigeben — (a) faire Referenz, (b) Verschränkung × Tiefe ohne
   Encoder; Vorregistrierung vor dem Lauf hier eintragen. *(Laut
   `ENTWURF_Q020.md` existiert ein Pilot mit 3 Seeds; er ist kein Befund.)*
2. T-002: `ButterflyFFN` so lassen oder zusätzlicher zentrierter Arm.
3. Q-019 so laufen lassen oder auf Datenknappheit / Walsh-Hadamard umbauen.
4. Q-011 mit Wiederaufnahme zu Ende rechnen.
5. Git einrichten.
6. IBM-Token neu erzeugen und die Klartextdatei entfernen.
7. Aufteilung A/B für die Anmeldung; die doppelte Kennung B-003.
8. Die Frist der schriftlichen Arbeit beim Landeswettbewerb prüfen: Anmeldung
   30. November, Arbeit Mitte Januar 2027 (eine Landesseite nennt den 17.01.).
   Die GPU-Ergebnisse brauche ich also erst im Januar.

### Was ich daraus mitnehme

Ich habe diesen Sommer gelernt, Mechanismen gegen Kontrollen zu prüfen. Die
Kontrollen selbst habe ich nie geprüft. Eine Kontrolle ist auch eine Behauptung
— nämlich die, dass sie das Beste ist, was die klassische Seite kann — und sie
gehört genauso gegen etwas Einfacheres gehalten wie der Mechanismus.

Und: Zwei meiner eigenen Werkzeuge haben das Richtige gesagt, und ich habe es
nicht übernommen. `q_summary` meldet die durchgefallene Positivkontrolle
ausdrücklich, der Code von B-014 hat die alte Zahl weiter ausgegeben. Ein
Werkzeug hilft nur, wenn ich seine Ausgabe auch in die Dokumente trage.

## 28. September — Vorregistrierung Q-020: die Kernaussage fair neu

Geschrieben **vor** dem Lauf. Der Text ist am 24. September entworfen worden
(`../BERICHT/experiment.md`); heute gebe ich ihn frei. Bis jetzt gibt es nur
einen Pilot (`Q-020-pilot`, 24.09.: 3 Seeds, reduzierte Suche, eigene Seeds
20291–20293). Der ist **kein Befund** und fließt in kein Urteil ein.

**Anlass.** Die Kontrolle aus Q-003 (`classical_norm_deep`, 67.848 Parameter)
ist schwächer als ein lineares Modell. Stichprobe vom 23.09.: 0,854 gegen 0,887.
Der Abstand +0,093 misst also vor allem ihre Schwäche. Außerdem stammen die
Nullbefunde zu Verschränkung und Tiefe aus einem Aufbau mit trainierbarem
Encoder. Genau diese Konstellation hat Q-018 als Artefakt entlarvt.

**Aufbau.** Es gibt zwei Aufbauten:
- (a) wie Q-003, mit Encoder, auf dem **offiziellen MNIST-Testsatz** (10.000
  statt 1.200 Bilder). Allein der Testmengenfehler bei 1.200 Bildern (≈0,007)
  lag über der Grenze 0,005.
- (b) wie Q-018 ohne Encoder, 6 Klassen, 128 Haufen.

Die Arme: Schaltung mit 1 und mit 5 Lagen, jeweils mit und ohne CNOT; eine
klassische Kronecker-Drehung (8 Parameter); quad_free (2.048); ein ReLU-MLP mit
derselben 8-Kanal-Engstelle (67.848); linear; ohne Engstelle; norm_deep. Code:
`quantum_sim/kernaussage.py`, bei Widerspruch gilt der Code.

**Vorher nachgerechnet.** Die 1-Lagen-Schaltung hat 16 Winkel, aber nur 8
davon wirken. `q1_ohne` rechnet exakt dieselbe Funktion wie die klassische
Kronecker-Drehung, `q1` dieselbe mit vertauschter Auslese (Abweichung
<4·10⁻⁷). Die Gegenprobe mit 5 Lagen zeigt, dass die Prüfung nicht blind ist.

**Familie, Holm über 3** (Entscheidung heute: P3 bleibt drin):
- PRIMÄR: q1 gegen mlp auf MNIST
  (GLEICHWERTIG / SCHALTUNG_BESSER / MLP_BESSER / UNENTSCHIEDEN)
- P2: Verschränkung in (b), q5 gegen q5_ohne
- P3: Tiefe in (b), q5 gegen q1

Gemessen wird die balancierte Genauigkeit, mit 15 Seeds (20201–20215) und TOST
±0,005. TOST wird vor der Signifikanz geprüft.
- **Auflösungsprüfung:** mlp muss sich von linear trennen.
- **Bodenprüfung:** der beste Arm liegt mindestens 0,10 über dem Rateniveau.
- **Äquivalenz-Positivkontrolle:** kron ≈ q1_ohne, nur zur Deutung.

Die Lernrate wird je Arm mit vollem Budget gesucht, das Gitter wird in beide
Richtungen verlängert. Gespeichert wird nach jeder Zelle, die Wiederaufnahme
hat eine eigene Datei je Lauf. Fashion-MNIST dient als Wiederholung und wird
nur beschreibend ausgewertet.

**Erreichbarkeit.** GLEICHWERTIG kann nur feuern, wenn die Streuung der
gepaarten Differenz unter 0,011 liegt. Liegt sie darüber, wäre „nicht
eingetreten“ kein Befund.

**Erwartung vorher:**
- PRIMÄR: SCHALTUNG_BESSER oder UNENTSCHIEDEN.
- P2 und P3: „trägt bei“, weil ohne CNOT jede Tiefe wieder eine
  Kronecker-Drehung ist.

**Vortest** bestanden: 21 Szenarien, Zweigtext geprüft, drei absichtlich
eingebaute Fehler alle gefunden. Vor dem Start läuft er noch einmal
automatisch mit.

**Rechenzeit:** geschätzt 3–5 h, auf dem Ryzen 5 3600. Während des Laufs läuft
kein anderer schwerer Lauf.

Ändert sich etwas an dieser Vorregistrierung, steht es hier, bevor eine
Q-020-Zahl existiert.

## 28. September — T-002: der zentrierte Zusatzarm läuft mit

Entscheidung heute, vor jedem T-002-Messlauf: `butterfly_ffn_zentriert` läuft
**standardmäßig mit**. Er steht außerhalb der Holm-Familie und wird nur
beschreibend ausgewertet. PRIMÄR und P2 bleiben unverändert. Im Code (in
`sprachmodell.py`, `cluster/t002_steuerung.sh`) heißt der Schalter jetzt
`--ohne-zusatzarm` statt `--mit-zusatzarm`. Selbsttest bestanden.

**Warum:** `ButterflyFFN` gibt |a|²·256·scale aus. Das ist nichtnegativ, hat
eine feste Summe und nur einen skalaren Gewinn. Ohne den Arm wäre ein T0 nicht
zu deuten: Liegt es an der Struktur oder nur an der Ausgabeform? Laut Recherche
(`../BERICHT/RECHERCHE_2026-09-28/03_t002_architektur_gpu.md`) benutzt keine
der strukturierten Transformer-Arbeiten eine nichtnegative Ausgabe.

**Den Gewinn lasse ich bewusst bei 1** und nicht bei 0, wie die Recherche es
nahelegt (LayerScale-Praxis). Mit 1 unterscheidet sich der Arm beim Start
*nur* durch das Abziehen des Mittelwerts. Der Vergleich misst also genau die
Ausgabeform und nicht zusätzlich eine andere Initialisierung.

**Kosten, neu gerechnet** (`rechenbedarf`, 13.000 Schritte, 0,334 s/Schritt
auf der RTX 4090, mit Lernratensuche im vollen Budget):

| Arme | GPU-h auf einer Karte |
|---|---|
| 4 Arme | 59–74 |
| 5 Arme (mit Zusatzarm) | 74–93 |

Die Angabe „30–35 GPU-h“ in `HANDOFF_AKTUELL.md` §7 ist damit überholt.

## 28. September — Q-019 wird umgebaut: Walsh statt Fourier, über die Datenmenge

Entscheidung heute: Q-019 wird **nicht** so gerechnet. Entwurf und
Vorregistrierungstext stehen in `ENTWURF_Q019_WALSH.md`, das neue Modul heißt
`quantum_sim/walsh.py`. `zeitreihen.py` bleibt als Vorlauf stehen.

**Die Kernidee:** Ohne CNOT ist die geteilte Butterfly ein Kronecker-Produkt
von Drehungen und enthält die Walsh-Hadamard-Transformation exakt. Die
Fourier-Transformation enthält sie nicht. Und 40 gegen 65.536 Parameter sollte
vor allem bei **wenig Daten** zählen.

**Familie** (bei N = 200, Holm über 3):
- PRIMÄR: butterfly gegen dense in der Walsh-Basis
- P2: Wechselwirkung mit einer zufälligen Orthobasis (Passung statt bloßer
  Parameterarmut)
- P3: butterfly gegen einen Low-Rank-Arm (Struktur statt „irgendwie wenige
  Parameter“)

Die Klassen liegen auf **Sequency**-Bändern, damit die ⟨Z⟩-Auslese sie nicht
direkt ablesen kann.

**Noch nicht vorregistriert.** Der Entwurf wird erst gültig, wenn
`saettigung`, `macht` und `zeit` gelaufen sind und Malte ihn freigibt. Dann
kommt der endgültige Text hierher, und zwar vor dem Lauf. Gerechnet wird erst
nach Q-020.

## 28. September — Q-020 gerechnet: alle drei Hypothesen halten

Lauf 08:47–11:12, Ryzen 5 3600, mit Wiederaufnahme ohne Unterbrechung.
Rohdaten `results/experiments/Q-020__20260928_111219_02.json` (435 Zeilen,
alle drei Abschnitte). Der MNIST-Zwischenstand liegt zusätzlich in
`…_092518.json`. Alle Zahlen hier prüft `q_audit` (14/14 OK). Vortest vor dem
Start bestanden.

**Familie (Holm über 3), jeweils 15/15 Seeds, p = 0,0001, Holm-p = 0,0002:**

| Test | Urteil | Zahl |
|---|---|---|
| PRIMÄR (a) MNIST, offizieller Testsatz | **SCHALTUNG_BESSER** | q1 0,9347 gegen mlp 0,9234, **+0,0113** |
| P2 (b) 128 Haufen | **VERSCHRAENKUNG_TRAEGT** | q5 0,6867 gegen q5_ohne 0,3453, **+0,3414** |
| P3 (b) 128 Haufen | **TIEFE_TRAEGT** | q5 0,6867 gegen q1 0,3548, **+0,3319** |

**Kontrollen alle bestanden:**
- Auflösung: mlp − linear +0,0416 (MNIST) bzw. +0,0960 (Haufen)
- Boden: bestanden
- Äquivalenz-Positivkontrolle: kron ≈ q1_ohne, +0,0001 bzw. +0,0003, TOST
  p < 10⁻⁴

Alle MNIST-Lernraten liegen im Inneren des Gitters. q1 ist nach oben auf 0,3
verlängert worden, bei 0,9 wird es wieder schlechter.

**Was das heißt, ehrlich gelesen:**

1. **Die Kernaussage steht jetzt gegen eine faire Kontrolle.** 16 Winkel (8
   wirksam) schlagen ein gewöhnliches ReLU-MLP mit 67.848 Parametern und
   derselben 8-Kanal-Engstelle, auf dem offiziellen Testsatz, in jedem Seed.
   Das sind rund 8.500-mal weniger wirksame Parameter in der Mittelschicht.
   Gegen `quad_free` (2.048) sind es +0,0109.
2. **Der Mechanismus ist klassisch, und das ist gemessen, nicht vermutet:**
   `kron`, die klassische Kronecker-Drehung mit 8 Parametern, erreicht 0,9350.
   Das ist dieselbe Zahl wie q1 (0,9347) und q1_ohne (0,9349), bei TOST-
   Gleichwertigkeit. Die ehrliche Formulierung lautet: *eine aus der
   Quantenschaltung abgeleitete Architektur, deren Wirkung eine klassische
   8-Parameter-Kronecker-Drehung mit quadratischer Auslese ist*.
3. **Die Wiederholung auf Fashion-MNIST** geht in dieselbe Richtung, aber
   kleiner: q1 − mlp **+0,0047**, p = 0,029, beschreibend und ohne Holm.
4. **Verschränkung und Tiefe tragen, sobald kein Encoder aushilft.** Mit Encoder
   (MNIST, beschreibend) sind q5 ≈ q1 und q5 ≈ q5_ohne (TOST gleichwertig):
   Genau dort übernimmt der Encoder die Arbeit. Damit ist der Q-003-Nullbefund
   als Encoder-Artefakt erklärt, wie das Review vermutet hatte. Und wie die
   Theorie vorhersagt, bringt Tiefe ohne CNOT nichts (q5_ohne ≈ q1_ohne, TOST).
5. **Ungünstig für die Schaltung, und es gehört so berichtet:** In (b) liegt
   selbst die beste Schaltung (q5, 0,687) weit hinter den klassischen Armen
   (mlp und quad_free 1,000, linear 0,904). Verschränkung trägt also, aber
   klassisch geht es dort viel besser. Auf Fashion ist mit Encoder q5 sogar
   *schlechter* als q1 (−0,0068, p = 0,002).
6. **Der Strohmann ist bestätigt:** mlp − norm_deep +0,0803 (MNIST). Die alte
   Kontrolle aus Q-003 war fast so weit hinter dem MLP wie das MLP hinter
   nichts.

**Für die Kurzfassung:** `[OFFEN: Q-020]` lässt sich jetzt füllen, mit Satz 1,
Satz 2 und dem Fashion-Vorbehalt (Satz 3). Erledigt am 28.09.

## 28. September — Q-019W: `walsh.py` gebaut, mein Entwurf hatte einen Denkfehler

Stand: vor jeder Q-019W-Zahl. Gelaufen sind nur Vortest und Rauchtest, mit
Ergebnissen im Scratchpad und nichts unter `results/`.

Der Vortest ist beim ersten Bau **durchgefallen**, und zwar zu Recht. Im
Entwurf hatte ich behauptet, das Sequency-Band sei als XOR von Hadamard-Bits
über die ⟨Z⟩-Auslese nicht linear ablesbar. Das ist falsch. Das Band ist eine
**Bijektion** von drei Hadamard-Bits, und ein Softmax-Kopf über 8 Klassen
trennt jede solche Bijektion (jede Ecke des Würfels {±1}³ ist als Gewicht
trennbar). Mein XOR-Argument gilt nur für einen Kopf, der einzelne Bits
ausgibt. Die Prüfung war nicht blind: Die Positivkontrolle (Bänder direkt auf
den oberen Bits) kam auf 1,0000.

**Korrekturen, alle vor jeder Zahl:**
1. Die Prüfung im Vortest ist nur noch ein Hinweis. Ob die echte Aufgabe zu
   leicht ist, entscheidet allein `saettigung()` vor dem Lauf, dazu der Zweig
   HU_SAETTIGUNG im Urteil.
2. `wht_fest` ist H⊗⁸ in **natürlicher** Ordnung. Im Entwurf stand
   „Sequency-Ordnung“, was dem Arm die Bänder direkt auf die Bits gelegt hätte.
3. `permutiert` und `fourier` gibt es nur für `butterfly` und `dense` bei
   N = 6.000.
4. N\* wird linear in **log N** interpoliert, weil das Gitter logarithmisch ist.

Vortest danach bestanden:
- Theorie: Butterfly = H⊗⁸, Abweichung 4·10⁻¹⁶
- 23 Szenarien, Gitterverlängerung 5/5
- Mutationsprüfung 3/3

**Lehre:** Diesmal hat der Vortest einen Fehler in der **Begründung** gefunden,
nicht im Code. Eine Behauptung im Entwurf, die der Vortest prüfen kann, gehört
auch in den Vortest.

**Nächster Schritt:** `saettigung`, `macht` und `zeit` laufen, sobald die CPU
frei ist (nach I-005). Danach wird der Entwurf endgültig, und der
Vorregistrierungstext kommt hierher, vor dem Lauf.

### Nachtrag, 28.09. nachmittags — Vorprüfungen: die P3-Kontrolle war ein Strohmann

`saettigung` (2 Seeds, lr 0,01 fest, **kein Befund**), bei N = 200:

| Arm | walsh | zufall |
|---|---|---|
| butterfly | 0,86 | 0,14 |
| dense | 0,53 | 0,52 |
| **lowrank** | **0,20** | 0,21 |
| wht_fest | 0,86 | 0,13 |
| untied | 0,49 | 0,50 |

- Keine Sättigung, der beste Familienarm liegt bei 0,86.
- **`lowrank` liegt unter dem Boden.** P3 (butterfly − lowrank ≈ +0,64) wäre
  damit fast per Konstruktion wahr gewesen. Das ist genau die Regel vom 23.09.:
  eine Kontrolle muss selbst gegen etwas Einfaches bestehen.
- **Ersetzt, vor jeder Zahl, durch `givens_zufall`:** dieselben 40 geteilten,
  orthogonalen Winkel wie die Butterfly (5 × 8 Schritte, je Schritt 128
  disjunkte Paare mit einem Winkel). Nur die Paare sind je Schritt zufällig
  verdrahtet statt nach Bit q, bei fester Verdrahtung mit Seed 19199.
  Geprüft: orthogonal (4,8·10⁻⁷), 40 Parameter, gleiche Winkel ergeben eine
  andere Abbildung (0,62). P3 fragt jetzt: Trägt die **Bitstruktur**, oder
  reicht die Teilung? `lowrank` bleibt als beschreibender Arm.

`macht` (5 Seeds, lr fest, kein Befund): gepaarte sd 0,012 / 0,028 / 0,025.
**TOST ±0,005 ist für keinen der drei Tests erreichbar.** GLEICHWERTIG kann
also nicht feuern, und ein Ausbleiben ist kein Befund. Nachweisbar sind
Effekte ab etwa 0,011 / 0,025 / 0,022. `zeit`: 3,4 h für den ganzen Lauf.

Vortest und Vorprüfungen laufen mit der neuen Kontrolle noch einmal.
Beschreibend fällt schon jetzt auf: Bei N = 200 liegen `butterfly` und
`wht_fest` gleichauf (0,86). Die gelernten Winkel finden also die feste WHT,
und das Lernen bringt darüber hinaus nichts. `untied` (5.120 Parameter)
liegt bei N = 200 dagegen auf dem Niveau von `dense`. Für die Dateneffizienz
entscheidet offenbar die **Teilung** der Winkel.

## 28. September — Vorregistrierung Q-019W (vor dem Lauf)

Freigegeben im Rahmen von Maltes Auftrag vom 28.09. („die Entscheidungen
durchführen … mach selber weiter“). Der Lauf braucht etwa 5,1 h, die CPU ist
sonst frei.

Maßgeblich ist der Modulkopf von `quantum_sim/walsh.py`, Stand dieses Commits.
Kurzfassung:

**Aufgabe:** Walsh-Signale (d = 256, 3 Anteile, Rauschen 0,45). Die Klasse ist
das Sequency-Band des stärksten Anteils, 8 Klassen. Kontrollbasis ist eine
feste Haar-zufällige Orthobasis.

**Lernkurve:** N ∈ {50, 200, 1.000, 6.000}, 470 Schritte fest, Test 2.000.
15 Seeds (19101–19115), Tuning-Seeds 19151–19153. Die Lernrate wird je Arm ×
Basis × N mit vollem Budget gesucht, das Gitter in beide Richtungen verlängert.

**Familie bei N = 200, Holm über 3, Wilcoxon zweiseitig + TOST ±0,005:**
- PRIMÄR: walsh, butterfly (40) gegen dense (65.536)
- P2: Wechselwirkung (butterfly − dense) walsh gegen zufall
- P3: walsh, butterfly gegen `givens_zufall` (dieselben 40 geteilten Winkel,
  zufällige Verdrahtung)

**Kontrollen:**
- Auflösung: dense trennt sich von narrow, walsh, N = 6.000
- Boden: bester Arm ≥ 0,225
- Sättigung: kein Familienarm > 0,95

**Vorher angesagt:**
- TOST ist für keinen Test erreichbar (sd 0,012 / 0,028 / 0,018 > 0,011).
  GLEICHWERTIG kann nicht feuern.
- P3 und P2 messen beide die Passung zur Bitstruktur: `givens_zufall` verhält
  sich in der Vorprüfung wie die Butterfly in der Zufallsbasis. Hält P3, heißt
  das „die Bitstruktur trägt“, nicht „die Butterfly lernt etwas Besonderes“.
- Die eigentlich offenen, beschreibenden Fragen sind:
  - der Übergangspunkt N\* (Vorprüfung: dense bei 6.000 vorn, 0,934 gegen 0,910)
  - butterfly gegen wht_fest (in der Vorprüfung gleichauf)
  - untied gegen butterfly (Teilung als Grund der Dateneffizienz)

**Erwartung:** PRIMÄR STRUKTUR_BESSER, P2 PASSUNG_TRAEGT, P3 STRUKTUR_TRAEGT,
N\* zwischen 1.000 und 6.000.

## 28. September — Q-019W gerechnet: die zweite KI-Art hat einen echten Vorteil, und er ist klassisch

Lauf von 13:0x bis 17:00. Rohdaten `results/experiments/Q-019W__20260928_170030_02.json`
(795 Zeilen). `q_audit` hat 11/11 Zahlen geprüft, alle OK.

**Familie bei N = 200, Holm über 3:** alle drei in 15/15 Seeds, p = 0,0001,
Holm-p = 0,0002.

| Test | Urteil | Zahl |
|---|---|---|
| PRIMÄR | **STRUKTUR_BESSER** | butterfly 0,8472 gegen dense 0,5390, **+0,308** |
| P2 | **PASSUNG_TRAEGT** | Wechselwirkung walsh − zufall **+0,693** |
| P3 | **STRUKTUR_TRAEGT** | butterfly − givens_zufall **+0,678** |

Kontrollen alle bestanden: Auflösung dense − narrow +0,101 (p = 0,0007),
Boden, keine Sättigung.

**Lernkurve, walsh:**

| Arm | N = 50 | 200 | 1.000 | 6.000 |
|---|---|---|---|---|
| butterfly (40) | 0,660 | 0,847 | 0,896 | 0,908 |
| wht_fest (0) | 0,655 | 0,847 | 0,895 | 0,907 |
| dense (65.536) | 0,276 | 0,539 | 0,824 | **0,950** |
| untied (5.120) | 0,274 | 0,495 | 0,763 | 0,877 |
| givens_zufall (40) | 0,151 | 0,169 | 0,187 | 0,208 |

**Was das heißt, ehrlich:**
1. **Dateneffizienz ist belegt.** Bei 50 bis 1.000 Beispielen liegt die
   Butterfly weit vor der dichten Schicht (+0,38 bei N = 50). Ab etwa
   **N\* ≈ 3.100** (zwischen 1.000 und 6.000) holt die dichte Schicht auf, bei
   6.000 liegt sie um 0,043 vorn. Das ist der erste Befund des Projekts, der
   über „gleichwertig, aber billiger“ hinausgeht: **bei wenig Daten klar
   besser**.
2. **Der Mechanismus ist die feste Walsh-Hadamard-Transformation.** Die
   gelernte Butterfly ist ab N = 200 per TOST gleichwertig zu `wht_fest` mit
   **0 Parametern**. Das Lernen der 40 Winkel findet die WHT und bringt darüber
   hinaus nichts.
3. **Die Teilung der Winkel ist der Grund für die Dateneffizienz.** Die
   ungeteilte Variante (5.120) ist bei wenig Daten so schlecht wie `dense`
   (N = 200: −0,35 gegen die Butterfly).
4. **Die Bitstruktur ist nötig:** Dieselben 40 geteilten Winkel mit zufälliger
   Verdrahtung lernen gar nichts (0,17–0,21). In der Zufallsbasis und bei
   permutierter Zeitachse fällt auch die Butterfly aufs Rateniveau.
5. **Vorher angesagt und eingetreten:** P2 und P3 messen beide die Passung. Sie
   sind stark, aber erwartbar. Neu und nicht vorhersehbar waren N\*, die
   Gleichheit zu `wht_fest` und der Befund zur Teilung.
6. **Fourier-Anschluss:** Wie in Q-019 verliert die Butterfly auf
   Sinussignalen (0,677 gegen 0,921 bei N = 6.000). Die richtige Basis ist
   Walsh, nicht Fourier.

**Für die Arbeit:** Die zweite KI-Art ist ein Signalklassifikator, der mit 40
Parametern (bzw. 0, als feste WHT) bei wenig Daten eine 65.536-Parameter-
Schicht deutlich schlägt, solange die Signale Rechteck-/Schaltcharakter haben.
Das ist klassisch erklärbar (Kronecker-Struktur = WHT), aber es ist genau die
Struktur, die die Quantenschaltung ohne CNOT hat.

## 28. September, nachts — T-004 (Chatbot-Adapter) gebaut, noch nicht vorregistriert

`quantum_sim/chatbot_adapter.py` baut auf LLäMmlein-120M auf (research-only
RAIL-M). Der Datensatz ist alpaca-gpt4-deutsch, 54 MB, Apache-2.0,
aufgeteilt in 5.000 Training / 500 Abstimmung / 1.000 Validierung / 50
Vorzeigefragen. Vortest bestanden, darunter Theorie, 19 Ausgänge und 5
Mutationen.

**Arme und trainierbare Parameter:**

| Arm | Parameter |
|---|---|
| `qire_kron` | 192 (8 je Matrix × 24) |
| `qire_5` (mit CNOT-Ring) | 960 |
| `lora_r1` | 30.720 |
| `basis` | 0 |

**Korrekturen am Entwurf**, alle vor jeder Zahl:
- Ohne CNOT sind 5 geteilte Lagen dasselbe wie 1 Lage. P3 misst also nur
  den CNOT-Ring.
- `qire_5` braucht eine feste Rückpermutation, damit es bei R = I startet.
- `v_proj` ist 256×768, weil das Modell nur 4 KV-Köpfe hat.
- Das Verhältnis ist 160-fach, nicht 372-fach.
- Die Schritte kosten 1,8–2,8 s, nicht ~1 s, gemessen bei 3 Threads unter
  Last.

**TOST-Grenze für P2 (Direktor 28.09.):** 20 % des Gewinns von `lora_r1`
gegenüber `basis`, relativ zum Effekt. Ob die Grenze erreichbar ist, ist
ohne Vorlauf unbekannt. Ein Ausbleiben von GLEICHWERTIG ist deshalb kein
Befund.

**Offen, vor dem Lauf:** Die Zeitmessung bei 6 Threads auf freiem Rechner
legt SCHRITTE fest (vorläufig 200). Danach kommt der Vorregistrierungstext
hierher, dann der Commit, dann der Lauf.

## 29. September, morgens — Vorregistrierung T-004 (vor dem Lauf)

**Zuerst, was nachts passiert ist:** I-005-v3 (Institut) ist beim
Herunterfahren des Rechners am 28.09. um 23:04 abgebrochen, in der Messphase,
nach der Lernratensuche. Ein Ergebnis gibt es nicht, und es wurde keine Zahl
gesichert. Das Modul hat keine Wiederaufnahme, deshalb wird es nach T-004 neu
gestartet.

**T-004, festgelegt vor jeder Zahl** (maßgeblich ist der Modulkopf von
`quantum_sim/chatbot_adapter.py`, Stand dieses Commits):

- **Budget:** `zeit` bei 6 Threads auf freiem Rechner, pro Schritt: qire_kron
  1,5 s, qire_5 1,5 s, lora_r1 0,9 s. Die Auswertung kostet etwa 0,13 s je
  Beispiel. Daraus folgt **SCHRITTE = 300**, zusammen etwa 6,6 h.
- **Familie, Holm über 3, 8 Seeds (40001–40008), Tuning-Seeds 40401/40402:**
  - PRIMÄR: `qire_kron` (192 Parameter) gegen `basis`, einseitig
  - P2: `qire_kron` gegen `lora_r1` (30.720), TOST mit einer Grenze von 20 %
    des LoRA-Gewinns gegenüber basis
  - P3: `qire_5` (960, mit CNOT-Ring) gegen `qire_kron`
- **Auflösung:** `lora_r1` muss sich von `basis` trennen.
- **Vorher angesagt:** Die Erreichbarkeit der TOST-Grenze ist unbekannt.
  PRIMÄR ist faktisch ein Ein-Stichproben-Test, weil basis deterministisch
  ist.
- **Erwartung:** PRIMÄR hält (die Drehung senkt den Verlust ein wenig), P2
  eher LORA_BESSER oder HU, P3 offen.

Die Wiederaufnahme läuft nach jeder Zelle. Ein erneutes Herunterfahren kostet
also höchstens eine Zelle.

## 29. September — T-004 gerechnet: der Chatbot-Adapter wirkt, und der CNOT-Ring trägt

Lauf 07:45–13:28, 6 Threads. Rohdaten
`results/experiments/T-004__20260929_132842_02.json` (32 Zeilen). `q_audit`
hat 12/12 Zahlen geprüft, alle OK. Vortest vor dem Start bestanden.

| Arm | trainierbar | Val-Verlust (nats/Token, Antwort-Tokens) | gegen basis |
|---|---|---|---|
| basis | 0 | 2,5616 | – |
| qire_kron | 192 | 2,5172 | −0,044 |
| qire_5 (CNOT-Ring) | 960 | 2,4437 | −0,118 |
| lora_r1 | 30.720 | 2,2893 | −0,272 |

**Familie (Holm über 3), jeweils 8/8 Seeds:**
- PRIMÄR **SENKT_VERLUST**: −0,0444, Holm-p = 0,012
- P2 **LORA_BESSER**: +0,228, Holm-p = 0,016. Die TOST-Grenze lag bei ±0,054,
  weit entfernt.
- P3 **VERSCHRAENKUNG_TRAEGT**: −0,0735, Holm-p = 0,016

Auflösung bestanden: LoRA − basis −0,272.

Die Lernraten liegen **innen**. `q_summary` meldet für qire_kron und qire_5
„unten“ (lr 0,001), das ist aber ein Fehlalarm: Das Gitter wurde nach unten
bis 0,00033 verlängert, und dort war der Verlust schlechter (qire_kron
2,5331 gegen 2,5286). `q_summary` kennt nur das Startgitter. Das Tool gehört
nachgebessert.

**Was das heißt:**
1. **Der Chatbot mit QIRE-Adapter funktioniert messbar.** 192 Winkel senken
   den Verlust in jedem Seed. Absolut ist der Gewinn klein: 16 % des
   LoRA-Gewinns.
2. **Pro Parameter ist die Drehung weit effizienter:**
   - qire_kron: 2,3·10⁻⁴ nats je Parameter
   - qire_5: 1,2·10⁻⁴
   - LoRA: 8,9·10⁻⁶

   Das ist 13- bis 26-mal mehr Wirkung je Parameter. Es ist aber keine
   Gleichwertigkeit, und das gehört ehrlich so gesagt.
3. **Der CNOT-Ring trägt im Sprachmodell:** Die Verschränkung fast
   verdreifacht den Gewinn (43 % statt 16 % des LoRA-Gewinns), bei 5-mal so
   vielen Parametern. Das passt zu Q-020 (b). Ohne CNOT wären 5 Lagen nach
   der Theorie gleich 1 Lage.
4. Die Seed-Streuung ist winzig (sd 0,0005–0,0024). Die Unterschiede sind klar,
   gemessen auf 1.000 festen Validierungsbeispielen.

**Nächste Schritte für den Chatbot:**
- Chat-Demo mit dem gespeicherten qire_5-Adapter (`chat qire_5`)
- Parameterkurve (mehr Lagen oder Winkel je Block)
- QIRE + LoRA kombiniert (Recherche 06)
- blinde Mini-Bewertung durch Menschen

## 29. September, abends — Vorregistrierung T-005 (vor dem Lauf)

`quantum_sim/chatbot_kurve.py`: Parameterkurve des Chatbot-Adapters.
Maßgeblich ist der Modulkopf, Stand dieses Commits. Vortest bestanden,
darunter:
- Theorie für alle L
- Bitgleichheit mit den T-004-Zellen
- exakter Page-Test gegen scipy
- 23 Szenarien und 7 Mutationen

**Arme (trainierbare Parameter):**

| Arm | Parameter | Bemerkung |
|---|---|---|
| qire_L1 | 192 | = qire_kron |
| qire_L2 | 384 | |
| qire_L5 | 960 | = qire_5 |
| qire_L10 | 1.920 | |
| lora_r1 | 30.720 | |
| kombi | 31.680 | L5 + LoRA gemeinsam |
| basis | 0 | |

Seeds 50001–50008, Tuning 50401/50402, SCHRITTE 300.

**Familie, Holm über 3:**
- K1 Trend über L: exakter Page-Test, einseitig. Dazu das Zusatzurteil
  „streng monoton“ über Nachbarvergleiche.
- K2 qire_L10 gegen lora_r1: TOST mit ±20 % des LoRA-Gewinns, wie P2 in
  T-004.
- K3 kombi gegen lora_r1, einseitig: Ergänzt QIRE LoRA?

Auflösung: lora_r1 trennt sich von basis.

**Zwei Kürzungen, vor jeder Zahl (Budget ≤ 9 h):**
- L3 gestrichen, die Kurve 1-2-5-10 ist ungefähr log-gleichmäßig.
- Für qire_L1, qire_L5 und lora_r1 werden die T-004-Lernraten übernommen
  (alle 1e-3, innen, bei unten verlängertem Gitter). Die Zellen sind
  bitgleich, das Budget und die Abstimmmenge sind gleich, und die neuen
  Mess-Seeds sind disjunkt. Regel 1 ist für diese Arme durch T-004 erfüllt.
  Neu gesucht wird für L2, L10 und kombi.

**Vorher angesagt:**
- K2 wird fast sicher LORA_BESSER und ist wenig informativ.
- Bei einer Seed-Streuung um 0,0005 werden auch winzige Stufen signifikant.
  Deshalb werden die Effektgrößen je Stufe mitberichtet.

Laufzeit 8,2–9,0 h. Die Wiederaufnahme läuft nach jeder Zelle, ein
Herunterfahren kostet höchstens eine Zelle.

**Vorbehalt (`q_summary`, Randlage):** `wht_fest` sitzt in einer Zelle am
oberen Gitterrand (lr 0,03). Der Arm hat 0 Mittelschicht-Parameter, die Rate
wirkt nur auf den Kopf. Ist er dort untertuniert, wäre `wht_fest` eher noch
besser. Die Aussage „butterfly ≈ wht_fest“ ist beschreibend und trägt dieses
Fragezeichen.

## 30. September — T-005 gerechnet: mehr Lagen helfen streng monoton, LoRA bleibt vorn, QIRE ergänzt LoRA nicht

Lauf über mehrere Sitzungen mit Wiederaufnahme (29.09. abends bis 30.09.
morgens, 6 Threads). Rohdaten `results/experiments/T-005__20260930_084358_02.json`
(56 Zeilen = 7 Arme × 8 Seeds, Seeds 50001–50008). `q_audit` prüft die Zahlen
dieses Eintrags, alle OK. Vortest vor dem Start bestanden.

| Arm | trainierbar | Val-Verlust (nats/Token) | Gewinn gegen basis | Anteil am LoRA-Gewinn |
|---|---|---|---|---|
| basis | 0 | 2,5616 | – | – |
| qire_L1 | 192 | 2,5174 | 0,044 | 16 % |
| qire_L2 | 384 | 2,4964 | 0,065 | 24 % |
| qire_L5 | 960 | 2,4442 | 0,117 | 43 % |
| qire_L10 | 1.920 | 2,4115 | 0,150 | 55 % |
| lora_r1 | 30.720 | 2,2875 | 0,274 | 100 % |
| kombi (L5 + LoRA) | 31.680 | 2,2910 | 0,271 | 99 % |

**Familie (Holm über 3), jeweils 8 Seeds:**
- K1 **STRENG_MONOTON**: Page L = 240, Holm-p ≈ 3·10⁻¹¹. Jede Stufe senkt den
  Verlust in 8/8 Seeds (p = 0,0039 je Stufe, Schwelle 0,0167):
  L1→L2 −0,021, L2→L5 −0,052, L5→L10 −0,033.
- K2 **LORA_BESSER**: qire_L10 − lora_r1 = +0,124, 0/8 Seeds besser,
  Holm-p = 0,016. Die TOST-Grenze lag bei ±0,055 (20 % des LoRA-Gewinns),
  weit entfernt. Vorher angesagt, so eingetreten.
- K3 **KEIN_NACHWEIS**: kombi − lora_r1 = +0,0035, 0/8 Seeds besser,
  Holm-p = 1,0. QIRE ergänzt LoRA nach diesem Test nicht.

Auflösung bestanden: lora_r1 − basis = −0,274, p = 0,0078.

**Was das heißt:**
1. **Die Kurve ist monoton und ohne Sättigung bis L = 10.** Der Gewinn je
   Lagenstufe wird pro Parameter kleiner (2,3·10⁻⁴ bei L1, 7,8·10⁻⁵ bei L10),
   steigt in der Summe aber weiter. Wo die Kurve abflacht, zeigt dieser Lauf
   nicht; L > 10 wurde nicht gemessen.
2. **Mit 1.920 Parametern erreicht QIRE 55 % des LoRA-Gewinns**, LoRA hat
   16-mal so viele. Pro Parameter bleibt die Drehung weit effizienter
   (7,8·10⁻⁵ gegen 8,9·10⁻⁶), aber es ist keine Gleichwertigkeit, und der
   absolute Abstand ist bei allen Lagenzahlen signifikant.
3. **K3 ist ein Nichtbeleg, kein Beleg der Gleichheit.** Beschreibend liegt
   kombi sogar minimal über lora_r1. Die einseitige Frage („ergänzt QIRE?")
   ist damit beantwortet: nein, nicht messbar. Der Nebenbefund
   kombi − qire_L5 = −0,153 (p = 0,0078, ohne Holm) sagt nur, dass LoRA das
   Meiste trägt.
4. Die Seed-Streuung ist klein (sd 0,0004–0,0029), gemessen auf denselben
   Validierungsbeispielen wie in T-004.

**Grenzen, ehrlich dazu:**
- Lernraten: für qire_L1, qire_L5 und lora_r1 aus T-004 übernommen, für
  qire_L2, qire_L10 und kombi neu gesucht (Gitter 0,0003 / 0,001 / 0,003);
  alle Raten liegen **innen** (lr 0,001). Die vor dem Lauf angesagte Kürzung
  (L3 gestrichen) ist eingehalten.
- `q_summary` meldet für `basis` „lr 0, unten“. Das ist ein Fehlalarm: Der
  Arm hat 0 trainierbare Parameter, es gibt keine Rate.
- Die Summe der Messzeiten in den Rohdaten beträgt 13,5 h, die
  Vorregistrierung nannte 8,2–9,0 h. Ich habe die Ursache nicht geprüft.
- Ein Verlust ist keine Chatqualität. Die blinde Mini-Bewertung durch
  Menschen steht weiter aus.

**Nächste Schritte:** L > 10 oder Winkel je Block, um die Sättigung zu
finden; Chat-Demo mit dem L10-Adapter; blinde Mini-Bewertung (Recherche 06).

## Zwischenstand

> ⚠ **Dieser Abschnitt ist vom 2. August und stimmt nicht mehr mit der
> Gesamtbilanz darunter überein** (bemerkt am 4. August). Die
> Tensor-Train-Punkte unten — der zweite Aufzählungspunkt und der Beitrag
> „Erstens" — führen TT weiterhin als belegten Erfolg. Das ist seit A-002
> überholt: Eine simple Blockdiagonale schlägt TT bei gleichem Budget
> (p=0,004), und nach B-008/B-009 ersetzt **keine** strukturierte Familie eine
> dichte Schicht unter fairen Bedingungen. Die Korrektur weiter unten
> („WICHTIG — Korrektur an früheren Einträgen") deckt die *frühen Einträge* ab,
> nicht diese Zusammenfassung — deshalb steht der Hinweis jetzt auch hier.
> Auch der Beitrag „Zweitens" (Faktor 6,4) braucht die Grenze aus B-011: bei
> 4096 Neuronen ist es umgekehrt, die Projektion einer trainierten Schicht ist
> dort gleichwertig. Maßgeblich ist der Abschnitt „Gesamtbilanz".

Wo ich stehe:

- Meine ursprüngliche Hauptthese (Born-Gating) ist widerlegt — nicht durch
  einen Fehler, sondern durch saubere Methodik. Und ich kann erklären, warum
  sie nicht funktionieren konnte.
- Ein anderer quanteninspirierter Ansatz (Tensor-Train) funktioniert
  nachweislich: 37–68 % Parameterersparnis ohne Genauigkeitsverlust.
- Drei methodische Schwachstellen im eigenen Messaufbau selbst gefunden und
  behoben, bevor sie zu falschen Schlüssen führten. Plus eine falsch gewählte
  Teststatistik selbst korrigiert.
- Eigenen Quantensimulator gebaut, gegen Qiskit verifiziert, auf echter
  IBM-Hardware gemessen.

**Was ich als eigentliche Beiträge sehe — inzwischen sind es vier:**

*Erstens, ein Kriterium für quanteninspirierte Ansätze.* Nicht "Quantenideen
helfen" oder "helfen nicht", sondern *wann*: Ein Ansatz bringt etwas, wenn er
eine echte strukturelle Einschränkung auferlegt, die Freiheitsgrade reduziert
ohne Ausdruckskraft zu zerstören (Tensor-Train). Er bringt nichts, wenn er
eine klassische Operation nur umbenennt (Born-Gate). Und der Nutzen wächst mit
der Modellgröße: 28-fache Parameterkompression auf MNIST bei 2048 Neuronen,
gegenüber knapp 2-facher bei meinen ursprünglichen Spielzeugmodellen.

*Zweitens:* Ob man eine Strukturbeschränkung nachträglich aufzwingt oder von
Anfang an mittrainiert, macht im Mittel Faktor 6,4 im Genauigkeitsverlust aus
(25 von 27 Fällen, p=0,000005) — bei drei ganz verschiedenen
Beschränkungsarten, also unabhängig davon, ob sie quanteninspiriert ist. Zwei
Einschränkungen, die ich selbst gefunden habe: Die Lücke hängt davon ab, wie
gut das nachträgliche Verfahren ist (bei beweisbar optimalem Vorgehen nur
2,8×), und **ein kurzer Nachlauf hebt sie praktisch auf** (p=0,34 gegen
von-Anfang-an-trainiert).

*Drittens, und für mein eigentliches Ziel am wichtigsten:* Eine
Ressourcenersparnis wird erst ab etwa 512 Neuronen zu messbarer
Zeitersparnis. Darunter frisst der Aufruf-Overhead sie auf. Bei 2048 Neuronen
erreicht halbierte Neuronenzahl auch halbierte Laufzeit. Das erklärt
rückblickend alle vier vorherigen Fehlschläge in diesem Punkt — sie lagen
sämtlich unterhalb dieser Grenze.

*Viertens, das eigentliche Ziel erreicht — aber anders als gedacht:* KI und
Quantenraum sind vereint, als laufendes System. Eigener differenzierbarer
Simulator, trainierbare Quantenschaltung, auf echter IBM-Hardware ausgeführt
mit identischer Genauigkeit zur Simulation. Nur eben **ohne** Effizienzvorteil.

**WICHTIG — Korrektur an früheren Einträgen dieses Tagebuchs:** An mehreren
Stellen oben steht, Tensor-Train sei ein bestätigter Erfolg. Das galt bis
A-002. Danach zeigte sich: Eine simple Blockdiagonale schlägt Tensor-Train bei
gleichem Budget (p=0,004), und A-003 hat meinen eigenen Rettungsversuch
widerlegt. Die frühen Einträge sind bewusst stehen geblieben, weil sie den
Verlauf dokumentieren — aber der Stand ist der hier.

## Gesamtbilanz: sieben Mechanismen, sieben Abgrenzungstests

| # | Mechanismus | Kontrolle | Ergebnis |
|---|---|---|---|
| 1 | Born-Regel-Gating | Softmax | kein Unterschied (p=0,25) |
| 2 | Verschränkungsentropie | Zufallsmatrizen | widerlegt |
| 3 | Tensor-Train-Bonds | Blockdiagonal | Kontrolle besser (p=0,004) — **überholt, siehe unten** |
| 4 | Unitarität | Spektralnorm | **unentschieden** — B-007 aufgelöst, siehe unten |
| 5 | DMRG-Kriterium | Zufallswachstum | kein Unterschied (p=1,00) |
| 6 | echte Quantenschaltung | klassischer Nachbau | **nicht unterscheidbar (p=0,49)** — korrigiert, siehe unten |
| 7 | Quantenkerne | RBF/poly/linear | Kontrolle besser (p=0,008) |

**Nachtrag vom 5./6. August — zwei Zeilen sind jetzt aufgelöst:**

- **Nr. 4** ist nicht mehr „p=0,32". B-007 hat den Vergleich vorregistriert mit
  20 Seeds und Lernrate je Arm wiederholt: auf MNIST schlägt Orthogonalität die
  Spektralnorm auf **beiden** Tiefen (+0,0025 und +0,0051, Holm p=0,0020 und
  0,0001), auf Fashion-MNIST repliziert die Richtung nicht. Nach dem vorher
  festgelegten Kriterium: **unentschieden**. Bemerkenswert und für einen
  Nachtest festgehalten: auf MNIST *wächst* der Effekt mit der Tiefe.
- **Nr. 6 stand falsch herum.** Q-003 hatte berichtet, der klassische Nachbau
  schlage die Schaltung (−0,0147, p=0,0078). Nachgerechnet auf den Rohdaten des
  Laufs mit erweitertem Lernratengitter dreht sich das um: **+0,0126 zugunsten
  der Schaltung auf MNIST, 15 von 15 Seeds, p=0,0001**; auf Fashion-MNIST
  dieselbe Richtung, aber nicht signifikant (+0,0051, p=0,23). Auch die
  Ein-Lagen-Variante mit 16 Parametern liegt auf MNIST vor dem Nachbau
  (+0,0123, p=0,0003).

  **Ich führe das nicht als Befund.** Der Vergleich stand nicht in der
  vorregistrierten Familie, und er repliziert nicht auf dem zweiten Datensatz —
  dieselben zwei Gründe, aus denen ich am 3. August den Fund
  `block_orthogonal` gegen `block_spectral` nicht als Ergebnis geführt habe. Er
  ist der Anlass für einen eigenen vorregistrierten Lauf, nicht sein Ersatz.

  **Fehler dabei bemerkt:** In der ersten Fassung dieses Eintrags stand
  „+0,0044 (p=0,49)". Diese Zahl stammt aus dem *abgebrochenen* Lauf mit dem
  engen Gitter, während die Genauigkeiten daneben aus dem zweiten Lauf kamen —
  Zahlen aus zwei verschiedenen Läufen in einer Aussage gemischt. Korrigiert
  am 6. August. Genau diese Sorte Fehler hat die 819×-Aussage erzeugt, die am
  3. August zurückgezogen werden musste: dort waren es A-002 und B-002.

Plus C-001 (Dichtematrix-Routing): in dieser Umgebung nicht prüfbar, nicht
falsifiziert.

**Zwei Zeilen dieser Tabelle stimmen seit dem 3. August nicht mehr so:**

- **Nr. 3** stammt aus einem Aufbau mit dichten Mischschichten, der nachweislich
  blind ist. Bei fairer Lernrate und ohne diese Enden schlägt Monarch — also eine
  Struktur *mit* Bonds — die Blockdiagonale bei gleichem Budget (+0,0115,
  p=0,0039). Die richtige Aussage ist nicht „Bonds tragen nichts", sondern:
  Monarchs Vorteil hängt an räumlicher Struktur in den Daten und verschwindet
  bei permutierten Pixeln (p=0,23).
- **Nr. 4 ist in dieser Form widerlegt.** Der vorregistrierte Lauf B-007
  (20 Seeds, Lernrate je Arm, zwei Datensätze) zeigt auf MNIST einen klaren
  Unterschied: Orthogonalität schlägt Spektralnormalisierung bei **beiden**
  Tiefen, je 17 von 20 Seeds, nach Holm-Korrektur signifikant (p=0,0020 und
  p=0,0001), mit großen Effektstärken (d=0,84 und d=1,18). „Kein Unterschied
  (p=0,32)" ist damit nicht haltbar.
  Die Wiederholung auf Fashion-MNIST trägt es aber nicht: +0,0048 bei Tiefe 1,
  **−0,0045** bei Tiefe 3, beide nicht signifikant — die Richtungen widersprechen
  sich. Nach dem vor dem Lauf festgelegten Kriterium reicht das nicht, um Nr. 4
  durch „Unitarität wirkt" zu ersetzen. Der belegbare Stand lautet: **auf MNIST
  wirkt Orthogonalität nachweislich über bloße Skalenkontrolle hinaus, auf
  Fashion-MNIST nicht.** Details im Eintrag „4./5. August — B-007".

**Und das verbindende Prinzip**, aus Nr. 7 quantitativ hergeleitet: In einem
2ⁿ-dimensionalen Hilbertraum sind zufällige Zustände fast immer orthogonal.
Die exponentielle Dimension macht Quantensysteme ausdrucksstark und zugleich
als Lernmaschine untauglich. Das ist ein plausibler gemeinsamer Grund für
mehrere der sieben Befunde.

## Was klassisch dabei herauskam

Beides entstand *aus* der Quantensuche, ist aber klassisch erklärbar — und
wird deshalb ausdrücklich nicht als quanteninspiriert ausgegeben:

| Verfahren | Ergebnis |
|---|---|
| normieren + Transformation + quadrieren | 33× weniger Parameter, +7,4 Prozentpunkte |
| Rang-64-Projektion einer **fertig trainierten** dichten Schicht | 32× weniger Parameter, **nachgewiesen gleichwertig** (TOST p<0,0001, beide Datensätze, 15 Seeds) |
| Effektivrang einer trainierten Schicht wird von der Aufgabe gesetzt, nicht von der Architektur | kontrolliertes 6×4-Gitter mit untrainierter Kontrolle (B-014) |

Auf das zweite wäre ich klassisch nie gekommen — es entstand aus der Frage,
was eine Quantenmessung eigentlich berechnet.

In dieser Tabelle standen bis zum 3. August nacheinander zwei weitere Zeilen,
beide inzwischen zurückgezogen:

- *„spektral-normalisierte Blockschicht, 819× weniger Parameter, besser als
  dicht"* — galt nur in einem Aufbau, der einen Faktor 32 im Parameterbudget
  nicht auflöst (Positivkontrolle p=0,23). Ohne die dichten Mischschichten dreht
  das Vorzeichen.
- *„Low-Rank Rang 64, kein messbarer Verlust"* — beruhte auf 10 Seeds und einer
  genäherten SVD. Mit 30 Seeds, zwei Datensätzen und exakter SVD liegt Rang 64
  bei −0,0066 und verfehlt die Äquivalenzgrenze klar.

Zum Stand des 3. August abends blieb damit **kein positiver Effizienzbefund**.
Das hat sich am 4. August geändert, allerdings anders als gedacht: Die
Projektion einer fertig trainierten Schicht ist nachweislich gleichwertig
(B-011) — ein Kompressions-, kein Effizienzbefund, weil sie das dichte Training
voraussetzt. Und B-014 liefert den Grund, warum die ganze Suche nach
strukturierten Ersatzschichten scheitern musste: Der Rang war nie die bindende
Beschränkung. Siehe die Einträge „B-003 bis B-006", „B-008 und B-009",
„B-010 und B-011" sowie „B-012 bis B-014".

## Offene Punkte

- Quantenkerne bei Daten mit **passender Symmetrie** (MNIST hat keine) — der
  Befund ist für generische Bilddaten gezeigt, nicht allgemein
- Quanteninspirierte **Optimierung** statt Architektur — nie getestet
- Umgekehrte Richtung: neuronale Netze zur Darstellung von Quantenzuständen,
  wo der Nutzen in der Fachwelt belegt ist
- Dichtematrix-Routing (C-001) auf einer Aufgabe, wo Kapazität wirklich zählt
- Literatur zu etwa drei Vierteln gelesen; zwei Bücher und ein PDF stehen noch
  aus. Die Karten in `sources/literature_cards/` sind aber noch pauschal als
  ungelesen markiert und geben damit einen falschen Stand wieder — muss ich
  nachziehen, sonst kann ich in der schriftlichen Arbeit nicht sauber zitieren

## Für die schriftliche Arbeit unbedingt erwähnen

- Energieschätzung beruht auf angenommener CPU-Wattzahl, keine Hardwaremessung
- Alle Experimente auf CPU. Die frühen Läufe einfädig wegen Reproduzierbarkeit;
  die MNIST-Ablationen ab A-002 laufen mit sechs Threads (`torch.set_num_threads(6)`
  nach `set_seed`). Für alle Arme eines Laufs identisch, aber die frühere pauschale
  Angabe „ein Thread" stimmte für diese Läufe nicht
- Zeitmessungen nur einfädig-CPU und nur bei Batch-Inferenz. Monarchs eigentliches
  Argument ist Hardware-Effizienz auf GPUs — das habe ich nicht gemessen und kann
  darüber nichts sagen
- Datensätze sind klein; Übertragbarkeit auf große reale Daten ungeprüft
- „Nicht signifikant" ist kein Äquivalenznachweis — bei knappen Fällen
  (orthogonal vs. spektral, p=0,32) könnte ein kleiner echter Unterschied
  unentdeckt geblieben sein
