# Befunde und Forschungsrichtungen — Projekt A

Stand 6. August 2026. Stichwortliste mit Kurzbeschreibung. Ziel des Projekts:
**Ressourceneffizientere KI mithilfe von Quantenphysik**, möglichst breit über
verschiedene KI-Arten. Der Quantenanteil ist der entscheidende Punkt.

> **Nachgeführt am 23. September 2026** nach dem Gesamtreview
> (`../BERICHT/00_REVIEW_BEFUNDE.md`). Geändert wurden Status-Markierungen, die
> nicht mehr stimmten (fünf Läufe standen noch als [LÄUFT]), und die Einträge zu
> Kernaussage, Verschränkung und Tiefe. Alte Formulierungen bleiben
> durchgestrichen oder mit Nachtrag stehen. Alles andere ist Stand 6. August.

Legende: **[BELEGT]** = mit Rohdaten und Statistik abgesichert · **[THEORIE]** =
hergeleitet und numerisch geprüft · **[LÄUFT]** = gebaut, in der Warteschlange ·
**[GEBAUT]** = fertig, nicht gestartet · **[IDEE]** = begründet, nicht gebaut ·
*neu 23. Sep.:* **[OFFEN]** = Aussage zurückgestellt, Nachtest geplant ·
**[ABGEBROCHEN]** · **[AUFGEGEBEN]** = bewusst nicht gerechnet

---

## 1 — Gesicherte Befunde

**Parametereffizienz der Quantenschaltung** **[BELEGT]**
16–80 Schaltungsparameter schlagen ein klassisches Netz mit 67.848 um +0,0932
(MNIST) und +0,0599 (Fashion-MNIST), Holm p=0,0001, 15 Seeds, Lernrate je Arm.
Die 848-fache Ersparnis gilt für die mittlere Schicht, nicht das Gesamtmodell.
> **Nachtrag 23. September — Zahl richtig, Deutung eingeschränkt.** Die
> Kontrolle `classical_norm_deep` (67.848) liegt in einer Stichprobe (3 Seeds,
> **kein Befund**) hinter einem rein linearen Modell (0,8544 gegen 0,8873); ein
> ReLU-Netz mit derselben Engstelle erreicht 0,9246. Der Abstand +0,0932 misst
> also vor allem die Schwäche der Kontrolle. Verteidigbar ist: 16 Winkel
> (0,9337) liegen vor dem 2.048-Parameter-Quadratmodell (0,9213; nicht
> vorregistriert). Der faire Vergleich gegen ein ReLU-Netz folgt in **Q-020 (a)**.

**Verschränkung trägt nichts bei** ~~**[BELEGT]**~~ **[OFFEN, 23. Sep.]**
Dieselbe Schaltung ohne CNOT-Gatter ist gleichwertig (−0,0015, TOST p<0,05 bei
±0,005). ~~Kein „nicht signifikant", sondern ein positiver Gleichwertigkeitsnachweis.~~
Auf MNIST braucht die Aufgabe keine Korrelation zwischen den Ausgabekanälen.

**Tiefe trägt nichts bei** ~~**[BELEGT]**~~ **[OFFEN, 23. Sep.]**
Eine Lage mit 16 Winkeln ist gleichwertig zu fünf Lagen mit 80 (+0,0002, TOST).
Der Grund ist strukturell: ohne Wiedereinspeisung der Daten ist die Verkettung
von Drehungen wieder eine Drehung derselben Gruppe.

> **Nachtrag 23. September zu beiden Einträgen.** Beide Gleichwertigkeiten
> stammen aus Q-003-replication, also **mit trainierbarem Encoder** 784→256 —
> genau die Konstellation, die Q-018 als Artefakt entlarvt hat („ein Nullbefund
> an einem Bauteil ist wertlos, solange ein anderes Bauteil dessen Aufgabe
> übernehmen kann“). Die Lehre wurde auf Q-014 angewandt, nicht auf diese
> beiden. Zweitens **war die Tiefe die Positivkontrolle** des Laufs, und die ist
> durchgefallen (+0,0002, p=0,7615); `q_summary` sagt ausdrücklich, dass keine
> Gleichwertigkeitsaussage aus diesem Lauf steht. Drittens die Theorie: Ohne
> CNOT ist jede Tiefe ein Kronecker-Produkt U₁⊗…⊗U₈ — dort *kann* Tiefe nichts
> bringen. Mit CNOT braucht Universalität dagegen Tiefe (§2 „Teilung kostet
> Tiefe“). „Tiefe trägt nichts bei“ widerspricht allgemein formuliert der
> eigenen Universalitätsrechnung. Die Begründung „ohne Wiedereinspeisung ist die
> Verkettung wieder eine Drehung derselben Gruppe“ stimmt, sagt aber nur, dass
> die *Funktionsklasse* abgeschlossen ist — nicht, wie viel Tiefe man braucht,
> um sie auszuschöpfen.
> **Neuer Status:** mit trainierbarem Encoder und durchgefallener
> Positivkontrolle gemessen; Nachtest **Q-020 (b)** — Verschränkung × Tiefe im
> Q-018-Aufbau ohne Encoder.

**Q-003 repliziert nicht** **[BELEGT]**
Die alte Aussage „der klassische Nachbau schlägt die Schaltung" (−0,0147) dreht
sich bei 15 Seeds mit Lernrate je Arm um: +0,0126 zugunsten der Schaltung,
15/15 Seeds, p=0,0001 auf MNIST. Nicht vorregistriert, deshalb kein Befund.
*(Nachtrag 23. Sep.: Die Frage geht in Q-020 auf. Die in der alten Kurzfassung
genannte Zahl „+0,0044 (p=0,49)“ war aus zwei Läufen gemischt, siehe §11.)*

**Butterfly-Umsetzung ist exakt dieselbe Funktion** **[BELEGT]**
Die Quantenschaltung als klassisches Butterfly-Netz implementiert, verifiziert
gegen den Simulator über 30 Konfigurationen: größte Abweichung 7,15·10⁻⁷.
Damit ist die Architektur nachweislich auf klassische Hardware übertragen.

**Übergangspunkt bei 14 Qubits** **[BELEGT]**
Ab 16.384 Dimensionen ist die Butterfly-Architektur 1,99× schneller als eine
dichte Schicht gleicher Breite — Median aus sieben Messreihen, 7/7 schneller,
Spanne 1,75–2,10. Darunter verliert sie deutlich (0,21× bei 4.096).
*(Nachtrag 23. Sep.: Die maßgeblichen Werte sind seit T-001 (10. Aug.)
**1,59×** in der Referenzumsetzung und **2,31×** mit `butterfly_fast.py`; der
Übergang bleibt bei 16.384 und entsteht an der Cache-Kante der dichten Schicht
— Zeitsprung ×9,1 von 8.192 auf 16.384.)*

**Machbarkeitsgrenze statt Geschwindigkeitsgrenze** **[BELEGT]**
Ab 32.768 Dimensionen lässt sich die dichte Schicht auf 16 GB nicht mehr bauen
(4,3 GB Gewichte, bei 65.536 dann 17,2 GB). Die Butterfly-Variante kommt mit
150 bzw. 160 Zahlen aus. Das ist kein Effizienz-, sondern ein Möglichkeitsunterschied.

**Inferenz ≠ Training** **[BELEGT]**
Der 1,99-fache Vorteil gilt für reine Inferenz. Im Training, mit Rückwärtslauf
und Optimierer, schrumpft er auf **1,18×** — gemessen an der Breite, die eine
dichte Schicht im selben Zeitbudget erreicht (14.400 statt 16.384).
*(Nachtrag 23. Sep.: Die Kalibrierung im Tagebuch (5. Aug.) ergab **14.896**;
Q-009 lief mit 14.400. Beide Zahlen stehen so in den Quellen — für die Arbeit
14.896 als Kalibrierwert nennen. Die 1,18× sind mit der Referenzumsetzung
gemessen, mit `butterfly_fast.py` nicht nachgemessen.)*

**Das Training der Winkel trägt bei — sobald der Encoder weg ist** **[BELEGT, Q-018]**
Ohne Encoder und auf einer Aufgabe mit kalibriertem Kapazitätsbedarf schlägt der
trainierte Block den eingefrorenen um **+0,4077**, 15/15 Seeds, Holm p=0,0001,
Auflösungsprüfung bestanden (+0,2074, p=0,0001). Die Parameter der Schaltung
tragen also sehr wohl — sie sind nicht bloß Beiwerk einer festen Struktur.

> **Zurückgenommen am 10. August:** Hier stand zuvor das Gegenteil („die Winkel
> muss man nicht trainieren", aus Q-014: +0,0018 auf MNIST, −0,0211 auf
> Fashion-MNIST). Q-014 hatte in jedem Arm einen trainierbaren Encoder
> 784 → 256, der sich an jede feste nachgeschaltete Drehung anpassen und dem
> Block die Arbeit abnehmen kann. Der Befund galt für diesen Aufbau, nicht
> allgemein. Q-018 hat den Encoder entfernt, und das Vorzeichen dreht sich von
> −0,02 auf +0,41. **Die Lehre ist nicht neu, nur teuer bestätigt:** Ein
> Nullbefund an einem Bauteil ist wertlos, solange ein anderes Bauteil dessen
> Aufgabe übernehmen kann.

**Konzentration der Quantenkerne** **[BELEGT]**
Mittlere Kernähnlichkeit halbiert sich je Qubit: 0,349 → 0,162 → 0,082 → 0,024.
Bei 6 Qubits ist die Kernmatrix praktisch die Einheitsmatrix, die SVM kann
nichts verallgemeinern. Sieben Abgrenzungstests, alle negativ.
*(Nachtrag 23. Sep.: Die vier Werte stehen nur im Tagebuch, nicht in Rohdaten —
genau genommen [BELEGT] nur im Sinne „dokumentiert“. K-Serie wiederholen oder in
der Arbeit so kennzeichnen. „Alle negativ“ ist zu grob: fünf negativ, Unitarität
unentschieden, die Schaltung trägt den Rest.)*

---

## 2 — Theoretische Ergebnisse

**Konzentrationsprinzip hergeleitet** **[THEORIE]**
E[|⟨ψ|φ⟩|²] = 1/2ⁿ folgt aus Symmetrie und Normierung — reine Geometrie, kein
Quantenphänomen. Die Messung trifft den Exponenten auf 4,2 % (−0,958 gegen
−1,000), R² = 0,9972. Vorfaktor 1,24 und konstant in n.

**Die Zustände sind so orthogonal wie Zufall** **[THEORIE]**
Aus dem obigen folgt die schärfere Aussage: Die Merkmalsabbildung erhält
praktisch keine Datenstruktur — sie verteilt die Daten so, wie es der Zufall
täte. Der Vorfaktor 1,24 ist der einzige Rest und wächst nicht mit n.

**Falsifizierbare Vorhersagen der Konzentration** **[THEORIE]**
Für künftige Läufe: 5 Qubits → 0,0448, 8 Qubits → 0,0061, 10 Qubits → 0,0016.
Weicht eine Messung deutlich ab, ist das Prinzip falsch. Damit ist aus einer
abgelesenen Regelmäßigkeit eine prüfbare Theorie geworden.

**Die Auslese ist ein Rayleigh-Quotient** **[THEORIE]**
out_q(x) = xᵀA_q x / xᵀx mit A_q = UᵀP_qU, numerisch bestätigt auf 1,0·10⁻⁷.
A_q hat Eigenwerte exakt {0,1}: eine echte orthogonale Projektion. Die Schaltung
misst den Energieanteil der Eingabe in orthogonalen Unterräumen.

**Die Auslese ist skaleninvariant und normiert** **[THEORIE]**
Ausgaben nichtnegativ, Summe exakt 1, out(c·x) = out(x). Linear/Tanh hat nichts
davon (Skalenabweichung 0,45, Summe −0,15 ± 1,93). Der Vergleich stellte also
eine strukturell eingeschränkte Abbildung gegen eine freie.

**Der Prüfmaßstab ist erfüllt** **[THEORIE]**
Das eigene Kriterium lautet: beitragen kann nur, was eine strukturelle
Einschränkung auferlegt. Die Schaltung tut genau das — sie zwingt die Ausgabe
in eine Verteilung über orthogonale Unterräume. Vier Tage lang unbemerkt.

**Orthogonale Gruppenauslese verschenkt eine Dimension** **[THEORIE]**
Bei disjunkter Gruppierung und orthogonaler Drehung ist die Ausgabesumme
konstant (−6,000000, Streuung 3,1·10⁻⁷). Effektivrang fällt von 8 auf 6,99.
Das erklärt, warum der orthogonale Arm mit 0,8881 der schlechteste ist.

**Überlappende Zweiteilungen als Vorteil** **[THEORIE]**
⟨Z_q⟩ nutzt n überlappende Halbierungen statt einer Zerlegung in n Gruppen —
deshalb keine feste Summe und voller Rang 7,99. Klassisch nimmt man naheliegend
die Gruppierung und verliert eine Dimension, ohne es zu merken.

**Alle Schaltungsfamilien sind universell** **[THEORIE, zweifach geprüft]**
Geteilt, per-Qubit und ungeteilt erreichen alle Fehler null gegen ein
zufälliges Ziel-Unitär — nur bei verschiedener Tiefe (32 / 16 / 4 Lagen bei
3 Qubits). Meine Vermutung, die volle Teilung sei nicht universell, war falsch.

**Parameterzahl zählt, Parametrisierung nicht** **[THEORIE, zweifach geprüft]**
Alle drei Familien erreichen Universalität bei **exakt** dim SU(d) Parametern:
64 gegen 63 bei 3 Qubits, 256 gegen 255 bei 4 Qubits, Verhältnis 1,00 für alle.
Bei gleicher Parameterzahl ist auch die Approximationsgüte gleich.

**Teilung kostet Tiefe, nicht Parameter** **[THEORIE, zweifach geprüft]**
Die geteilte Familie braucht 128 Lagen für das, wofür die ungeteilte 4 braucht —
bei identischer Parameterzahl. Sie spart also **Speicher, nicht Rechenzeit**.
Das ist schwächer als „kostenlose Kompression" und muss so berichtet werden.

**Gewichtsteilung als Untermannigfaltigkeit** **[THEORIE]**
Die geteilte Schaltung hat 2nL Dimensionen in SO(d) mit d(d−1)/2 — bei 8 Qubits
80 von 32.640, also 0,245 %. Bei 12 Qubits 120 von 8.386.560.

**Universalität macht Teilung zur Tiefenfrage** **[THEORIE, teilweise geprüft]**
Der hardware-efficient ansatz ist für L→∞ universell. Die Teilung schränkt also
nicht ein, *was* erreichbar ist, sondern nur, wie viele Parameter der Weg
kostet. „Kostet Teilung etwas" ist damit eine Frage nach der nötigen Tiefe.

**Präzisierung: zwei verschiedene Teilungen** **[THEORIE]**
Die Schaltung teilt Winkel über die **d/2 Amplitudenpaare** einer Qubit-Ebene;
jedes Qubit behält eigene Winkel (`theta` hat Form (L, n, 2)). Das ist der
Standardansatz, für den Universalität belegt ist — nicht eine Teilung über die
Qubits hinweg. Die wäre stärker und vermutlich **nicht** universell, weil dann
alle Erzeuger mit der zyklischen Qubit-Verschiebung vertauschen und die
erreichbare Gruppe diese Symmetrie nie verlässt. Numerisch geprüft in
`scratchpad/universality.py`; die Unterscheidung war in der ersten Fassung
dieses Dokuments unsauber.

**Butterfly-Encoder wäre redundant** **[THEORIE, bewiesen]**
Eine Lage ist CNOT-Ring ∘ Π_q G_q; L Lagen sind die L-fache Verkettung. Zwei
solche Schaltungen hintereinander ergeben per Konstruktion eine mit L₁+L₂
Lagen — die Familie ist unter Verkettung abgeschlossen. Ein Butterfly-Encoder
vor der Schaltung ist also nur zusätzliche Tiefe, ~~und Tiefe ist
TOST-gleichwertig~~. Deshalb braucht der Encoder einen nicht-unitären Schritt.
*(23. Sep.: Die Abgeschlossenheit trägt das Argument allein. „Tiefe ist
TOST-gleichwertig“ ist zurückgestellt, siehe §1.)*

**Verworfene Hypothese: n−1-Flaschenhals** **[THEORIE]**
Vermutet, die Ausgaben lägen auf einem Simplex mit 7 Freiheitsgraden, passend zu
B-014s sieben Richtungen. Die Messung sagt nein: ⟨Z⟩-Summe nicht konstant
(Streuung 0,175), Rang 7,99. Die Übereinstimmung war Zufall.

---

## 3 — Läufe in der Warteschlange

> **Stand 23. September:** Die Status-Markierungen darunter waren vom
> 6. August. Aktuell: Q-009 **[NICHT VERWERTBAR]** · Q-015 **[BELEGT, MNIST;
> Fashion zurückgezogen]** · Q-010 **[AUFGEGEBEN, 10. Aug.; Frage geht in
> Q-020 auf]** · Q-011 **[ABGEBROCHEN, 27. Aug.; Tiefen 1–8 auf MNIST
> gerettet]** · Q-012, Q-016, Q-017 **[AUFGEGEBEN, 10. Aug.]** · Q-013
> **[ERSETZT durch T-002]** · Q-014 [NICHT VERWERTBAR] · Q-005/006/007 [GEBAUT,
> nicht gestartet]. Neu: **Q-018** [BELEGT], **Q-019** [GEBAUT, nicht
> gelaufen], **Q-020** [ENTWURF, `ENTWURF_Q020.md`].

**Q-009 · Gleiches Zeitbudget** ~~**[LÄUFT]**~~ **[NICHT VERWERTBAR]**
Halten 140 Parameter bei identischem Rechenbudget gegen 222 Millionen? Isochrone
Breiten empirisch kalibriert (14.400 statt 16.384). Erste Schwierigkeitsstufe
bereits gesättigt — Auflösungsprüfung wird dort durchfallen.

**Q-015 · Global gegen blocklokal** ~~**[LÄUFT]**~~ **[BELEGT auf MNIST: +0,0674]**
Volle globale orthogonale Auslese (32.640 Freiheitsgrade) gegen die Schaltung
(80). Trennt Funktionsklasse von Parametrisierung. Vorgezogen, weil es die
Alleinstellungsfrage am direktesten beantwortet.

**Q-010 · Schaltung gegen Nachbildung** ~~**[LÄUFT]**~~ **[AUFGEGEBEN → Q-020]**
Vorregistrierte Fassung des nachgerechneten +0,0126-Befunds. 30 Seeds, drei
Datensätze, Holm über zwei Vergleiche. Prüft, ob der post-hoc gefundene Vorteil
einer angemeldeten Prüfung standhält.

**Q-011 · Unitarität in der Tiefe** ~~**[LÄUFT]**~~ **[ABGEBROCHEN, Tiefen 1–8 gerettet]**
Wächst der Vorteil der Orthogonalität gegenüber Spektralnorm mit der Tiefe?
Tiefen 1, 3, 8, 16. B-007 zeigte +0,0025 → +0,0051 zwischen Tiefe 1 und 3.
Aussichtsreichster Kandidat für einen bestandenen Abgrenzungstest.

**Q-012 · Encoder als Butterfly** ~~**[LÄUFT]**~~ **[AUFGEGEBEN]**
270 statt 201.130 Parameter für das gesamte Modell, Faktor 745. Der Encoder
bekommt Butterfly plus Kürzung; die Kürzung ist der nicht-unitäre Anteil, ohne
den er nur zusätzliche Tiefe wäre.

**Q-014 · Kostet die Gewichtsteilung?** **[NICHT VERWERTBAR — 10. August]**
Gelaufen, 180 Zeilen, Auflösungsprüfung durchgefallen (+0,0018, p=0,1205). Die
Frage war mit diesem Aufbau nicht stellbar: der trainierbare Encoder 784 → 256
übernimmt die Arbeit des Blocks. Die Familienzahlen (+0,0257 und −0,0243) sind
nicht zitierfähig. Neuauflage braucht eingefrorenen Encoder, Gitter bis 1,0 und
eine Aufgabe mit mehr als sieben nötigen Richtungen.

**Q-013 · Transformer** ~~**[LÄUFT]**~~ **[ERSETZT durch T-002]**
Erster Sequenzlauf des Projekts. Butterfly-FFN gegen dichten FFN, 162 gegen
1.051.136 Parameter, auf Markow-Text mit bekannter Entropie als Prüfmaß.
Vorstudie mit 5 Seeds, kein Beleg.

**Q-016 · Lernkurven** ~~**[LÄUFT]**~~ **[AUFGEGEBEN]**
Datenbedarf und Schrittbedarf statt Endgenauigkeit — die Achse, die das
Projektziel nennt. Gradientenschritte angeglichen statt Epochen. Misst auch die
Überanpassung, die aus 80 gegen 67.848 Parametern folgen sollte.

**Q-017 · Data Re-Uploading** ~~**[GEBAUT]**~~ **[AUFGEGEBEN als eigenständiger Lauf]**
Daten in jeder Lage neu einspeisen statt nur am Anfang. Rettet die
hardwaretaugliche Winkel-Kodierung (0,808 gegen 0,958) und prüft die Vorhersage,
dass Tiefe *mit* Wiedereinspeisung wirkt, während sie ohne wirkungslos ist.

**Q-005 · Barren Plateaus** **[GEBAUT]**
Gradientenvarianz gegen Qubitzahl, Vorhersage Var ~ 2⁻ⁿ. Kein Training nötig,
also billig. Sagt, welche Schaltungsfamilie bei wachsender Qubitzahl überhaupt
trainierbar bleibt — Vorbedingung für jede Skalierung.

**Q-006 · Ising-Pruning** **[GEBAUT]**
Die Maskenwahl beim Pruning *ist* eine Ising-Energie: E(u) = uᵀQu mit
Q = Gram(H)⊙Gram(W). Verifiziert exakt gleich dem Rekonstruktionsfehler.
Magnitude-Heuristik nutzt nur die Diagonale, das Annealing die Kopplungen.

**Q-007 · Breitentest** **[GEBAUT]**
Die Schaltung über 13 Aufgabentypen mit je eigener fairer Kontrolle, Holm über
alle Datensätze. Beantwortet nicht „ist es überall besser", sondern die
stärkere Frage: wovon hängt der Vorteil ab.

---

## 4 — Neue Richtungen: Optimierung und Training

**Quantum Natural Gradient** **[IDEE]**
Winkel leben auf einer gekrümmten Mannigfaltigkeit; Adam behandelt sie als
freie Reelle. Die Fubini-Study-Metrik gibt den natürlichen Gradienten und
braucht in der Literatur deutlich weniger Schritte. Direkt die Zielgröße.

**Identitätsnahe Initialisierung** **[IDEE]**
Schaltungen, die als Identität starten (Grant et al. 2019), umgehen Barren
Plateaus. Reine Initialisierungsänderung, kostet nichts, könnte den
trainierbaren Bereich um mehrere Qubits verschieben.

**Parameter-Shift statt Backpropagation** **[IDEE]**
Quantenschaltungen haben exakte analytische Gradienten über zwei Auswertungen
je Parameter. Klassisch teurer als Backprop, aber rauschrobust — und auf echter
Hardware die einzige Option. Relevant für Hardware-Training.

**Imaginäre Zeitentwicklung als Optimierer** **[IDEE]**
Statt Gradientenabstieg die Schrödingergleichung in imaginärer Zeit — führt
zum Grundzustand, also zum Minimum. Ein quantenmechanisch abgeleiteter
Optimierer für klassische Modelle; im Projekt nie betrachtet.

**QUBO für Architektursuche** **[IDEE]**
Nicht nur Pruning-Masken, auch Schichtbreiten, Quantisierungsbits und
Verbindungsmuster sind binäre Optimierungsprobleme. Dieselbe Ising-Maschinerie
wie Q-006, andere Zielfunktion. Spart Suchaufwand statt Modellgröße.

**Progressives Wachstum, quantenmotiviert** **[IDEE]**
Der DMRG-Lauf ergab 79,7 % Trainingsaufwand bei gleicher Genauigkeit — als
„nicht quantenspezifisch" abgehakt. Für das Trainingsaufwand-Ziel ist es der
bisher beste Befund und gehört wieder aufgenommen.

**Lernrate aus der Spektrallücke** **[IDEE]**
Die optimale Schrittweite hängt an der Krümmung; bei unitären Schichten ist das
Spektrum bekannt (alle Singulärwerte 1). Daraus ließe sich die Lernrate
herleiten statt zu suchen — das würde die Lernratensuche einsparen.

---

## 5 — Neue Richtungen: Architektur und Repräsentation

**Tensornetze jenseits von MPS** **[IDEE]**
Das Projekt kennt nur Matrix-Product-States. MERA erfasst Skaleninvarianz,
PEPS zweidimensionale Struktur, Tree-Tensornetze hierarchische. Für Bilddaten
ist PEPS die naheliegende Wahl und wurde nie probiert.

**Unitäre rekurrente Netze** **[IDEE]**
Unitarität verhindert explodierende und verschwindende Gradienten — der
klassische Anwendungsfall in der Literatur. Q-011 prüft die Tiefe; rekurrent
mit geteilten Gewichten wäre der nächste Schritt und trifft Sequenzmodelle.

**Butterfly als Aufmerksamkeitsprojektion** **[IDEE]**
In Transformern sind Q-, K- und V-Projektionen dichte d×d-Matrizen. Butterfly
ersetzt sie mit O(d log d) und wenigen Parametern. Anders als der FFN-Block
sind sie linear, also die natürlichere Einsatzstelle.

**Überlappende Auslese als Bauprinzip** **[IDEE]**
Der heute gefundene Rangverlust bei disjunkter Gruppierung gilt für jede
normerhaltende Schicht mit gruppenweiser Auslese. Als allgemeine Designregel
formuliert: überlappende Partitionen statt Zerlegungen.

**Amplituden-Kodierung als Kompression** **[IDEE]**
n Qubits tragen 2ⁿ Amplituden. Als reine Datenkompression betrachtet — ohne
jede Schaltung — ist das ein Verhältnis, das klassisch nur mit Verlust erreichbar
ist. Nie isoliert gemessen, immer nur zusammen mit der Schaltung.

**Hierarchische Qubit-Register** **[IDEE]**
Statt einer Schaltung über alle n Qubits mehrere kleine Register mit sparsamer
Kopplung — das quantenmechanische Gegenstück zu Gruppenfaltungen. Verbindet
die Lokalitätsbefunde aus B-004 mit der Quantenarchitektur.

**Gewichtsteilung über Lagen** **[IDEE]**
Die Schaltung teilt Winkel über die Qubit-Ebene. Zusätzlich über die Lagen zu
teilen ergäbe 2n Parameter unabhängig von der Tiefe — bei 8 Qubits 16 statt 80.
~~Da Tiefe nichts beiträgt, sollte es nichts kosten.~~ *(23. Sep.: Diese
Begründung ist zurückgestellt, siehe §1 „Tiefe“. Ob es etwas kostet, zeigt erst
ein Lauf ohne Encoder.)*

---

## 6 — Neue Richtungen: Kerne und Merkmalsräume

**Symmetrie-angepasste Merkmalsabbildungen** **[IDEE]**
Die Konzentration trifft *generische* Abbildungen. Ist die Abbildung an eine
Symmetrie der Daten angepasst (Permutation, Translation), bleibt Struktur
erhalten. Das ist die bekannte Gegenmaßnahme und im Projekt ungetestet.

**Projizierte Quantenkerne** **[IDEE]**
Huang et al. schlagen vor, statt des vollen Überlapps reduzierte Dichtematrizen
zu vergleichen — das umgeht die exponentielle Konzentration nachweislich.
Direkte Antwort auf den siebten Negativbefund, mit vorhandenem Code machbar.

**Quantenmerkmale als Speicherschlüssel** **[IDEE]**
Fast orthogonale Zustände sind für Klassifikation tödlich, für assoziative
Speicher optimal — Kapazität ist maximal bei minimaler Überlappung. Dieselbe
Messung wäre einmal Negativ- und einmal Positivbefund.

**Konzentration als Vorhersagewerkzeug** **[IDEE]**
Die gemessene Formel 1,24·2^(−0,96n) sagt vorher, ab welcher Qubitzahl ein
Kern unbrauchbar wird. Damit ließe sich *vor* einem Lauf entscheiden, ob er
sich lohnt — Rechenzeit sparen durch Theorie.

**Kernmatrizen niedrigen Rangs** **[IDEE]**
Wenn die Kernmatrix gegen die Einheitsmatrix geht, ist sie maximal
hochrangig — das Gegenteil dessen, was schnelle Kernverfahren brauchen. Die
Konzentration erklärt also auch, warum Quantenkerne nicht beschleunigbar sind.

---

## 7 — Neue Richtungen: Generative Modelle und Sampling

**Born-Maschinen** **[IDEE]**
Eine Schaltung als generatives Modell: Messungen erzeugen Stichproben aus
|ψ|². Das ist die natürlichste Nutzung eines Quantenzustands überhaupt und
im Projekt nie betrachtet — bisher wird nur klassifiziert.

**Quanten-Boltzmann-Maschinen** **[IDEE]**
Ein Ising-Modell mit transversalem Feld als generatives Modell. Verbindet die
QUBO-Linie mit generativem Lernen und hat eine direkte Hardware-Entsprechung
in Quantenannealern.

**Dequantisierte Sampling-Verfahren** **[IDEE]**
Tang hat gezeigt, dass mehrere behauptete Quantenvorteile klassisch durch
Sampling erreichbar sind. Diese Algorithmen sind selbst quanteninspiriert und
schnell — eine ganze Familie, die das Projekt nicht kennt.

**Verschränkungsentropie als Regularisierer** **[IDEE]**
Bisher wurde Entropie als *Vorhersage* für Kompression getestet und verworfen.
Als Strafterm im Training benutzt, würde sie niedrige Verschränkung erzwingen —
also Komprimierbarkeit erzeugen statt sie vorherzusagen.

---

## 8 — Neue Richtungen: Messung und Auslese

**Reichere Observablen** **[IDEE]**
⟨Z⟩ je Qubit gibt n Ausgaben aus einem Zustand mit 4ⁿ messbaren Observablen.
Mehrere Pauli-Produkte zu messen liefert mehr Information je Schaltungsauswertung
— direkt Effizienz, weil weniger Auswertungen nötig sind.

**Klassische Schatten** **[IDEE]**
Huang/Kueng/Preskill: aus wenigen zufälligen Messungen viele Erwartungswerte
schätzen, mit logarithmischer statt linearer Skalierung. Genau das Werkzeug für
„mehr Information pro Messung" und im Projekt unbekannt.

**Messungsbasierte Sparsität** **[IDEE]**
Eine Messung kollabiert den Zustand auf einen Basisvektor — das ist maximale
Sparsität, physikalisch erzwungen. Als Aktivierungsmechanismus gedacht wäre es
ein quantenabgeleiteter Weg zu sparsamen Netzen.

**Adaptive Messtiefe** **[IDEE]**
Nicht jedes Beispiel braucht dieselbe Schaltungstiefe. Ein Kriterium aus der
Zustandsentropie könnte je Beispiel entscheiden, wann abgebrochen wird — die
C-001-Idee, aber mit einem Kriterium, das messbar ist.

---

## 9 — Neue Richtungen: andere KI-Arten

**Sprachmodelle** ~~**[LÄUFT]**~~ **[T-002: gebaut, vorregistriert, Lauf verloren; vier Fehler vor dem nächsten Lauf, siehe `HANDOFF_AKTUELL.md` §7]**
~~Q-013 ist der Einstieg.~~ *(Q-013 ist durch T-002 ersetzt. Der folgende Satz
gilt für echte große Modelle; das auf dieser Hardware trainierbare Modell hat
d_model 256, also weit unter dem Übergangspunkt — Qualität und Effizienz werden
getrennt berichtet.)* Die FFN-Blöcke echter Modelle liegen bei 4.096–16.384
Dimensionen, also genau im Bereich, wo die Butterfly-Architektur gewinnt. Das
ist kein Zufall, sondern der Bereich, für den O(d log d) gebaut ist.

**Graphneuronale Netze** **[IDEE]**
Permutationssymmetrie ist bei Graphen zentral, und genau dort helfen
symmetrie-angepasste Quantenmerkmale. Zudem sind Graphprobleme oft direkt als
QUBO formulierbar — zwei Anknüpfungspunkte gleichzeitig.

**Reinforcement Learning** **[IDEE]**
Die Politik ist eine Wahrscheinlichkeitsverteilung — genau das, was eine
Messung liefert. Eine Quantenschaltung als Politiknetz ist strukturell passend
und braucht sehr wenige Parameter.

**Zeitreihen und Signale** **[IDEE]**
Die Butterfly-Struktur *ist* die schnelle Fouriertransformation. Für periodische
Signale ist das die natürliche Basis, und dort sollte die Architektur nicht nur
gleich gut, sondern besser sein als eine dichte Schicht.
> **Nachtrag 27. August, nachts — dieser Satz ist falsch.** Die Butterfly-Schicht hat **geteilte** Winkel (40), die schnelle Fouriertransformation braucht je Paar einen **eigenen** Drehfaktor (5.120). Gemeinsam ist beiden das **Verbindungsmuster** und damit die Kostenordnung O(d log d) — nicht die Transformation. Im Q-019-Vorlauf erreicht `butterfly` 0,686, die echte Fouriertransformation 0,804. Der Satz bleibt hier stehen, weil er den Verlauf dokumentiert.

**Kompression trainierter Modelle** **[IDEE]**
B-011 hat bewiesen, dass eine Rang-64-Projektion einer trainierten Schicht
gleichwertig ist. Dieselbe Frage für die Butterfly-Struktur: lässt sich eine
fertige dichte Schicht in die geteilte Form projizieren?

---

## 10 — Werkzeuge und Methodik

**Rohdatenspeicherung** **[BELEGT]**
`result_store.py` legt jede Messung als Zeile ab, mit Konfiguration und
Softwareumgebung. Nach jedem Datensatz sichern, nicht erst am Ende — das hat
in dieser Nacht eine vollständige Messung gerettet.

**Äquivalenztest statt p>0,05** **[BELEGT]**
`equivalence.py` liefert TOST und Holm. Erst damit wurden aus zwei Vermutungen
zwei nachgewiesene Gleichwertigkeiten (Verschränkung, Tiefe). „Nicht
signifikant" ist kein Gleichheitsnachweis.

**Sammelbericht der Q-Serie** **[BELEGT]**
`q_summary.py` liest alle Rohdaten und sortiert nach gilt / gilt mit Vorbehalt /
nicht verwertbar / offen. Ein Aufruf statt sechs Logs, und die
Auflösungsprüfung geht nicht mehr unter.

**Q-Audit** **[BELEGT]**
`q_audit.py` rechnet jede Tagebuchzahl der Q-Serie aus *einer* Datei nach.
Alle 19 geprüften Zahlen stimmen. Ergänzt `qire_ai.audit`, der nur B-Zahlen
prüft und dessen „alles stimmt" sonst irreführend wäre.
*(Nachtrag 23. Sep.: Stand 30. Aug. 106 Prüfungen. `qire_ai.audit` prüft nur 15
B-Zahlen — ausgerechnet B-014, B-011, B-010, B-007, B-008, B-009 fehlen — und
läuft nur im Archiv. Q-019 und T-000 fehlen in `q_summary.QUESTIONS`. Beides
wird in der Überarbeitung vom 23. Sep. angegangen.)*

**Theoriewerkzeug** **[BELEGT]**
`theory.py` hält Herleitungen und Vergleich mit den Messungen zusammen —
Konzentrationsgesetz, Parametergeometrie, offene Spannungen. Reproduzierbar
ohne Rechenzeit.

**Positivkontrolle differenziert** **[BELEGT]**
Eine durchgefallene Kontrolle entwertet *Nullbefunde*, nicht gefundene
Unterschiede. Bei Q-003-replication ist der Primäreffekt das 400-fache dessen,
was die Kontrolle nicht trennen konnte — die Kernaussage bleibt gültig.
*(Nachtrag 23. Sep.: Die Regel ist richtig — und sie heißt umgekehrt, dass
„Verschränkung“ und „Tiefe“ aus genau diesem Lauf nicht als Gleichwertigkeit
stehen. Das wurde hier und in drei weiteren Dokumenten übersehen. Der gefundene
Unterschied steht, misst aber vor allem die Schwäche der Kontrolle, siehe §1.)*

---

## 11 — Fehlerquellen, in dieser Nacht gefunden

**Zu enges Lernratengitter** **[BELEGT]**
Fünf von sechs Armen wählten den Randwert; das Optimum lag außerhalb. Zweimal
passiert (Q-003-replication, Q-009). Alle Skripte melden jetzt Randlagen und
legen sie in der Konfiguration ab.

**Zahlen aus zwei Läufen vermischt** **[BELEGT]**
+0,0044 aus dem abgebrochenen Lauf neben 0,9339 aus dem zweiten gestellt.
Genau so entstand die zurückgezogene 819×-Aussage. `q_audit` rechnet deshalb
jede Behauptung aus einer einzigen Datei nach.

**Ungleiche Vorverarbeitung zwischen Armen** **[BELEGT]**
Der klassische Arm bekam die Normierung nicht, die zur Amplituden-Kodierung
gehört, und landete bei 0,1295 statt 1,0000. Das hätte einen Vorteil von 78
Prozentpunkten vorgetäuscht.

**Gesättigte Aufgabe, blinde Kontrolle** **[BELEGT]**
Bei 64 Haufen erreichen Butterfly und ein Viertel der Breite beide 1,0000 —
die Auflösungsprüfung fällt durch, und der Primärtest sitzt auf dieser Stufe.
Entwurfsfehler, vor der Auswertung dokumentiert.

**Epochen statt Gradientenschritte** **[BELEGT]**
Bei 250 Beispielen sind 12 Epochen nur 24 Schritte — das Modell konvergiert
nicht, und der Datenvergleich misst den Abbruch statt den Datenbedarf.
B-010 hatte das bereits richtig gemacht.

**Ungeteilter Butterfly-Pfad falsch implementiert** **[BELEGT]**
Die Parameter wurden angelegt, aber falsch indiziert — der Kontrollarm hätte
still etwas anderes gerechnet. Behoben und verifiziert; die geteilte Variante
stimmt weiterhin exakt mit dem Simulator überein.

**Konfundierung in Q-015 vor dem Start** **[BELEGT]**
Der Vergleichsarm hätte zwei Unterschiede gleichzeitig gemessen — Butterfly
gegen freie Drehung *und* überlappende gegen disjunkte Auslese. Behoben,
Effektivrang jetzt 7,995 statt 6,99.

**Falsche Funktionsklasse als Kontrolle** **[BELEGT]**
`classical_quadratic` ist blockdiagonal, die Schaltung mischt global. Q-003
verglich also gegen eine strukturell schwächere Familie — derselbe Fehlertyp
wie die blinde Architektur in B-004.

---

## 12 — Was nicht funktioniert, und warum

**Quantenvorteil bei Geschwindigkeit auf Simulator** **[THEORIE]**
Ausgeschlossen durch Konstruktion: 2ⁿ Amplituden fortzuschreiben kostet mehr
als die Matrixmultiplikation, die es ersetzt. Nur die *Architektur* kann
klassisch gewinnen, nie die Simulation.

**Butterfly-Encoder ohne Kürzung** **[THEORIE]**
Wäre nur zusätzliche Tiefe derselben Gruppe, und Tiefe ist gleichwertig.
Musste nicht gemessen werden — die Gruppenstruktur sagt es vorher. Beispiel
dafür, dass Theorie Rechenzeit spart.

**Verschränkung auf MNIST** ~~**[BELEGT]**~~ **[OFFEN, 23. Sep. — siehe §1, Nachtest Q-020 (b)]**
Gleichwertig ohne CNOT (mit trainierbarem Encoder, Positivkontrolle durchgefallen). Kein Argument gegen Verschränkung an sich, sondern
dafür, dass diese Aufgabe keine Korrelation zwischen Ausgabekanälen braucht.
Der faire Test wäre eine Aufgabe mit erzwungener Kanalkorrelation.

**Generische Quantenkerne** **[BELEGT]**
Sieben Abgrenzungstests, alle negativ, mit hergeleiteter Erklärung. Nicht
widerlegt ist, dass symmetrie-angepasste Abbildungen funktionieren — das ist
eine andere Frage und offen.

**Dichtematrix-Routing** **[BELEGT]**
Implementiert und verifiziert, aber in dieser Umgebung nicht prüfbar: der teure
Pfad ist nicht besser als der billige, selbst perfektes Routing brächte
höchstens +0,0066. Ausdrücklich nicht falsifiziert.

---

## 13 — Die drei aussichtsreichsten Richtungen

> **Stand 23. September:** Vorrang hat jetzt **Q-020** — (a) die Kernaussage
> gegen eine faire klassische Referenz (linear, ReLU-MLP mit gleicher
> Engstelle), (b) Verschränkung × Tiefe ohne Encoder. Q-017 ist als
> eigenständiger Lauf aufgegeben (10. Aug.); Q-011 ist abgebrochen und braucht
> Wiederaufnahme. Die drei Richtungen darunter sind Stand 6./10. August.

**Unitarität in der Tiefe (Q-011)**
Spektralnormalisierung ist *die* klassische Ersatztechnik. Versagt sie bei
Tiefe 16, während echte Unitarität hält, gibt es eine Eigenschaft, die
klassisch nicht ersetzbar ist — ein Mechanismusunterschied, kein Implementierungsunterschied.

**Gewichtsteilung gratis (Q-015 belegt, Q-014 neu aufzusetzen)**
Kostet die Kopplung über die Qubit-Ebene nichts, ist sie eine kostenlose
Kompression um Faktor 128 bis 408. Auf diese Kopplung führt der klassische Weg
nicht — sie folgt daraus, dass ein Gatter auf ein Qubit wirkt. Q-015 stützt das
(+0,0674 bei 408× weniger Freiheitsgraden); Q-014 ist am 10. August an der
Auflösungsprüfung gescheitert und muss mit eingefrorenem Encoder wiederholt
werden. **Vorher zu klären**, weil es die ganze Richtung berührt: Wenn die
Winkel eingefroren genauso gut sind, ist „Gewichtsteilung" womöglich die
falsche Frage — dann geht es gar nicht darum, wie viele Parameter der Block
hat, sondern darum, dass er keine braucht.

**Data Re-Uploading (Q-017)**
Erzeugt Ausdruckskraft aus Tiefe statt Breite, bleibt hardwaretauglich, und
das angebundene Modell hat 6.466 statt 201.130 Gesamtparameter. Zugleich Test
einer Vorhersage aus der eigenen Gruppenstruktur-Überlegung.
