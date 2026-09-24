# Projekttagebuch — die Einträge zu Projekt B

**Auszug, nicht Original.** Die lebende Fassung ist
`PROJEKTTAGEBUCH.md` im Projektordner (siehe README). Dort stehen auch alle
Einträge zu Projekt A, die Gesamtbilanz und die Abschnitte „Offene Punkte" und
„Für die schriftliche Arbeit unbedingt erwähnen" — die sind beiden Projekten
gemeinsam und deshalb hier nicht kopiert.

Erzeugt mit dem Skript `spiegel_aktualisieren.ps1`.

---

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


