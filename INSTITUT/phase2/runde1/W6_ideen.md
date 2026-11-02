# W6 Ideen: Clifford-/Kerdock-/Stabilisator-Codebuch (Phase 2, Runde 1, 02.11.2026)

Status: Ideen, keine Läufe, kein Code ausgeführt. Alle dB-Werte sind Schätzungen aus Literaturzahlen (ungeprüft, nur Abstracts/Suchauszüge) und Theorie; sie müssen im Test gemessen werden. Quellen: `WISSEN/felder/K4_*`, `F1_*`, `wege.md`, S-001 bis S-009, `quellen.json`.

## Gemeinsamer Rahmen (gilt für alle Ideen)
- **Quelle:** i.i.d. Gauß (Stand-in für Hadamard-rotierte Gewichte) und eine rotierte, echte Gewichtsmatrix (z. B. aus dem MLP/Byte-LM der Z3-Skripte, nur Tuning-Seeds).
- **Maß:** MSE in dB relativ zur Varianz (SQNR) bei fester Rate R = 2 Bit/Gewicht (zusätzlich 3 und 4 Bit).
- **Bytes/Gewicht:** Index-Bits + Gewinn-/Skalenbits + Codebuch/Rechenregel. Referenzwerte: Shannon-Grenze 2 Bit = 12,04 dB; Lloyd-Max 1D 2 Bit ≈ 0,1175 (9,3 dB); E8P ≈ 0,089 (10,5 dB); Dim 256 ≈ 0,069 (11,6 dB) [arXiv:2608.25188, Suchauszug]. Lloyd-2D (16 Punkte) liegt grob bei 9,6 bis 9,8 dB (Schätzung, mitmessen).
- **Pflichtgegner (S-002):** Lloyd-1D, Lloyd-2D (k-means, QAM-W [arXiv:2605.26339]), E8P-artiges Gitter (QuIP# [arXiv:2402.04396], GLQ-Werkzeug zum Nachbauen), Gruppenskala g16 (kostet 1 Bit/Gewicht, also bei gleichen Bytes ein 3-Bit-Arm).
- **Quantenbezug in allen Fällen:** Struktur direkt (Stabilisatorzustände, Clifford-Gruppe, Kerdock/MUB [arXiv:1904.07842]), Wirkung klassisch (I-038). Formulierung: „quanten-inspiriert“.
- **Grundskepsis (langweiligste Erklärung für alle):** Der Raumfüllgewinn ist klein (2D: 0,17 dB bei hoher Rate, E8 maximal ≈ 0,65 dB geholt [arXiv:2606.23406]); nach Rotation ist die Quelle fast Gauß, ein beliebig guter Block-Quantisierer gewinnt höchstens die Lücke zur Shannon-Grenze (2,7 dB bei 2 Bit), und schon E8P holt davon nur etwa 1,2 dB. Außerdem schlugen in Z3 Gruppenskalen die Rotation im Sprachmodell.

---

## Idee 1: Stabilisator-Shape-Buch in 8D (Gain-Shape-VQ)
- **Hypothese:** Die realen Stabilisatorzustände von 3 Qubits (Vektoren in R^8 mit Einträgen 0, ±1/√m, Anzahl nachzählen) bilden ein sphärisches Code-Buch mit guter Winkelverteilung. Als Shape-Buch, kombiniert mit einem Lloyd-Skalar für den Betrag (Gain), erreicht es bei 2 Bit/Gewicht MSE zwischen Lloyd-1D und E8P.
- **Nächste Arbeit:** QuIP#/E8P [arXiv:2402.04396]; Gain-Shape-Variante PCDVQ [arXiv:2506.05432] und NSNQuant (festes VQ-Buch nach Hadamard) [arXiv:2505.18231]; Strukturverwandtschaft: Kerdock-2-Design [arXiv:1904.07842]. Ein Stabilisator-Gewichtsbuch wurde zweimal nicht gefunden (K4).
- **Stärkster Gegner:** E8P (bzw. Leech-LLVQ [arXiv:2603.11021]); E8-Wurzeln sind das optimale 8D-Kontaktsystem (240 Punkte), ein Stabilisatorsatz ist dort nach Konstruktion kaum besser. Zusatzgegner: Lloyd-1D mit g16 bei gleichen Bytes.
- **Quantenbezug:** direkt (Stabilisatorzustände, Clifford-Orbit). Die reale 3-Qubit-Cliffordgruppe liegt in der Weyl-Gruppe von E8 (aus der Gittertheorie, unverbürgt, prüfen); das wäre ein sauberer Bezug zum Gegner und eine ehrliche Begründung, falls es nur E8 neu verpackt.
- **Langweiligste Erklärung:** Es entsteht E8 oder eine schlechtere Teilmenge davon. Dann gilt „Stabilisator = Spezialfall des Gitters“, kein Gewinn, aber ein sauberer Zusammenhang.
- **Billigster Test (≈ 2 Min CPU):** n = 3 (8D): alle realen Stabilisatorvektoren erzeugen (Clifford-Orbit von |000>), Anzahl und Winkelspektrum ausgeben; Gauß-Blöcke, Encode = größte Projektion (Shape) + Lloyd-Gain. Rate: log2(|Buch|)/8 + Gain-Bits/8. Bei 240 bis 1080 Punkten sind das nur 1 bis 1,25 Bit/Gewicht für den Shape, also zuerst die 1-Bit-Variante gegen Lloyd-1D bei 1 bis 1,5 Bit messen, danach 2 Bit durch Vereinigung mit Vorzeichen-/Verfeinerungsstufe (Residual-VQ). Bytes: Buch = 0 (aus Regel erzeugt) oder ≤ 1 KiB.
- **Erwarteter Effekt:** gegen Lloyd-1D bei gleicher Rate +0,2 bis +0,8 dB; gegen E8P −0,2 bis −0,7 dB. Reale Gewichte: ≤ ±0,1 dB Unterschied (Skalenprobleme dominieren).

## Idee 2: Orbit-Lloyd: Codebuch als Gruppenorbit mit gelerntem Startvektor
- **Hypothese:** Ein Buch, das als Orbit einer kleinen Untergruppe der Clifford-Gruppe (Vorzeichen-/Paulis × Hadamard-/CNOT-artige Permutationen) eines optimierten Startvektors v entsteht, schlägt 1D-Lloyd bei gleicher Rate und braucht keine Tabelle, nur v und die Gruppenregel (E8P selbst ist ein solches Orbit-Buch einer Vorzeichen-/Permutationsgruppe).
- **Nächste Arbeit:** E8P [arXiv:2402.04396]; Trellis ohne Tabelle (QTIP [arXiv:2406.11235], Proteus, CubicQuant [arXiv:2608.06763]); klassische Gruppencodes für Gauß-Quellen (Slepian 1968, nicht in `quellen.json`, vor Zitat nachtragen und prüfen).
- **Stärkster Gegner:** QTIP/Trellis (billiger zu dekodieren, besser bei 2 Bit laut Selbstangaben) und E8P. Auf CPU zusätzlich Lloyd-Skalar + LUT-Kernel (T-MAC [arXiv:2407.00088]).
- **Quantenbezug:** direkt (Clifford-Gruppe erzeugt das Buch; Gate-Folge = Dekodierregel). Wirkung klassisch.
- **Langweiligste Erklärung:** Es entsteht ein gewöhnlicher Permutations-/Vorzeichencode; die Clifford-Struktur bringt nichts über Slepian hinaus.
- **Billigster Test (≈ 5 Min CPU):** 8D und 4D, Untergruppen der Ordnung 2^8 bis 2^12 (bei 4D bis 2^8 = 2 Bit), Startvektor v per „symmetrisiertem Lloyd“ (Schwerpunkte im Orbit mitteln). Nächster-Nachbar-Suche brute force (≤ 4096 Punkte). Vergleich mit Lloyd-2D/4D-k-means gleicher Größe (Obergrenze ohne Struktur) und E8P. Bytes: nur v und Gruppenparameter, also ≈ 0 plus 8 Gleitkommazahlen.
- **Erwarteter Effekt:** +0,2 bis +0,6 dB gegen Lloyd-1D bei 2 Bit; −0,2 bis −0,8 dB gegen den unstrukturierten k-means gleicher Größe (die Struktur kostet Qualität, spart Bytes und Dekodierzeit).

## Idee 3: Kerdock-/Z4-Winkelbuch für Paare (Polarquantisierung)
- **Hypothese:** Nach Hadamard + Zufallsdiagonale sind Paare (a,b) fast isotrop; Winkel in Z8 (Phasen i^k-Familie, Kerdock-Phasen) plus Lloyd-Betrag (Rayleigh) ergibt ein buchfreies 2D-Buch mit nahezu der Güte von Lloyd-2D.
- **Nächste Arbeit:** TurboAngle (gleichförmige Winkel im WHT-Bereich) [arXiv:2603.27467]; PolarQuant [arXiv:2502.02617]; QAM-W (Lloyd-2D nach Hadamard) [arXiv:2605.26339]; Kerdock-Phasen [arXiv:1904.07842].
- **Stärkster Gegner:** Lloyd-2D (k-means) und QAM-W; der Gegner ist hier praktisch das Optimum in 2D.
- **Quantenbezug:** Phase i^k wie Qubit-Phasen (Z4), Strukturbezug direkt, aber dünn. Ehrlich ist „Analogie bis Strukturverwandtschaft“.
- **Langweiligste Erklärung:** Gewöhnliche Polarquantisierung. Bei Z4 (4 Winkel) ist es QPSK, ein Reißbrettcode ohne Kerdock-Besonderheit.
- **Billigster Test (≈ 1 Min):** Gauß-Paare, Winkel in 4/8/16 Stufen, Betrag Lloyd mit 2^(4−log2 W) Stufen (4 Bit pro Paar = 2 Bit/Gewicht). Alle Winkel/Betrag-Aufteilungen durchprobieren. Bytes: Betragsbuch ≤ 16 Zahlen.
- **Erwarteter Effekt:** −0,3 bis 0 dB gegen Lloyd-2D, ≈ 0 bis +0,3 dB gegen Lloyd-1D. Dient als Kontrollgruppe: Wenn schon die Polarform nicht über Lloyd-2D kommt, ist die Kerdock-Struktur in 2D tot.

## Idee 4: Kerdock-Basenwahl als Nebeninformation (Best-of-(N+1)-Rotationen)
- **Hypothese:** Statt einer festen Hadamard-Rotation wählt jeder Block unter den N+1 gegenseitig unverzerrten Basen eines Kerdock-Systems [arXiv:1904.07842] die mit dem kleinsten Quantisierungsfehler; die Wahl kostet log2(N+1)/N Bit/Gewicht.
- **Nächste Arbeit:** gelernte Rotation ≈ Hadamard (BASE-Q, ReSpinQuant, S-001); „Wahl der Orthogonalen egal“ [arXiv:2601.21069], [arXiv:2604.04440].
- **Stärkster Gegner:** dieselben Bits als zusätzliche Quantisierungsstufe (bei N = 16 sind 0,26 Bit/Gewicht = 1,5 dB Gegenwert) und Gruppenskala g16.
- **Quantenbezug:** direkt (MUB/Kerdock, 2-Design), Wirkung klassisch.
- **Langweiligste Erklärung:** Die Nebeninformation ist teurer als der Gewinn der Auswahl (Auswahl aus K Zufallsrotationen bringt nur den Streuungsvorteil der MSE; bei Blockgröße 16 grob 0,1 bis 0,3 dB).
- **Billigster Test (≈ 2 Min):** Gauß-Blöcke 16/32/64, Lloyd-1D 2 Bit in jeder der N+1 Basen (Hadamard, signierte Varianten, reelle Näherung der Kerdock-Familie), kleinsten MSE nehmen; Rate = 2 + log2(N+1)/N Bit. Vergleich bei gleichem Gesamtbit (2,26 Bit Lloyd-1D bzw. 2-Bit + Skalen).
- **Erwarteter Effekt:** brutto +0,1 bis +0,3 dB, netto gegen gleiche Bytes −1 bis −1,4 dB. Erwartet wird ein sauberes Negativergebnis (nutzbar als ehrlicher Abschnitt „Warum nicht“).

## Idee 5: Bitfehler-robuste Indexzuordnung per Kerdock-Distanz (Brücke zu W5)
- **Hypothese:** Ordnet man den Indizes eines Stabilisator-/Orbit-Buchs (Idee 1/2) so zu, dass Bitflips im Index nahe Codewörter treffen (Pseudo-Gray, Kerdock-/Z4-Distanzstruktur), sinkt der Verlust bei Speicherbitfehlern gegenüber zufälliger Zuordnung und gegenüber Skalar-Lloyd mit Clipping.
- **Nächste Arbeit:** Hamming-Umcodierung DeepNcode [arXiv:2405.13891]; Clipping + RandBET [arXiv:2006.13977]; Rotation gegen Bitflips [arXiv:2603.16382]; Index-Zuordnung (Zeger/Gersho, „Pseudo-Gray Coding“, nicht in `quellen.json`, nachtragen).
- **Stärkster Gegner:** Clipping + RandBET und Gewichtsskalierung [arXiv:2411.18993]; für Skalar-Gitter ist die Gray-Zuordnung schon Stand der Technik.
- **Quantenbezug:** Analogie (Fehlerkorrektur, Kerdock als Z4-lineare Codes). Direkt nur in der Struktur des Codebuchs.
- **Langweiligste Erklärung:** Gray-Zuordnung ist altbekannt und Clipping dominiert; das Buch trägt nichts bei.
- **Billigster Test (≈ 5 Min):** Das Buch aus Idee 1 oder 2; Flip-Rate p ∈ {1e-4, 1e-3, 1e-2} auf den Indexbits; MSE-Verlust in dB gegen Lloyd-1D (Gray) und gegen Lloyd-1D mit Clipping bei gleichen Bytes.
- **Erwarteter Effekt:** bei p = 1e-3 −0,1 bis +0,3 dB gegen Gray-Lloyd-1D; kaum Aussicht, Clipping zu schlagen. Nur sinnvoll, wenn Idee 1 oder 2 funktioniert hat.

---

## Prüfhinweise (für Opus)
- Neuheit: Stabilisator-/Clifford-Orbit als Gewichtsbuch wurde zweimal nicht gefunden (K4); das ist ein Suchergebnis. Vor einer Behauptung noch Slepian-Gruppencodes, Pseudo-Gray und Spherical-Codes-Literatur (Kontaktzahlen, Orbit-Codes) nachschlagen.
- Hauptrisiko der Ergebnisdeutung: E8P ist keine volle E8-Quantisierung [arXiv:2608.25188]; der Gegner muss mit gleichem Aufbau (Hadamard, gleiche Rate, gleicher Gain) nachgebaut werden, nicht aus Papierzahlen.
- Gaußannahme prüfen: auf rotierten echten Gewichten die Kurtosis messen (S-006); bei Blockgröße 8 reicht nach Hadamard über ≥ 64 Eingänge die Gaußnäherung, aber die Randverteilung ist nicht exakt isotrop.
- Bytes und Skalen: g16 = 1 Bit/Gewicht (S-004). Jede Variante weist Index-, Gain-, Skalen- und Buchbytes getrennt aus.
- Dekodierkosten: ein Gewinn von < 0,3 dB ist ohne Dekodierkostenvorteil (buchfrei, Bit-Operationen) nicht verwertbar (K4 Schluss 4).

## Empfehlung
1. **Idee 2 (Orbit-Lloyd) zusammen mit Idee 1 als Spezialfall (Stabilisator-Orbit als Startpunkt):** einzige Variante mit einem möglichen echten Befund (+0,2 bis +0,6 dB gegen Lloyd-1D ohne Tabelle, klarer Quantenbezug, auf CPU billig dekodierbar). Ein Test, ≤ 10 Min, ein Skript: erst Zählung und Winkelspektrum der Stabilisatorvektoren (3 Qubit, real), dann Orbit-Lloyd, dann Vergleich mit E8P-Nachbau und Lloyd-2D.
2. **Idee 3 als Kontrollgruppe (1 Min):** liefert den ehrlichen Minimalwert für Winkelcodes; wenn schon Lloyd-2D nicht erreicht wird, bleibt Idee 3 tot.
- Ideen 4 und 5 sind Negativkandidaten bzw. Zusatz; nur laufen lassen, wenn der Test 1 positiv ausfällt (Idee 5) oder als ein Absatz „Warum nicht“ in der Arbeit (Idee 4).
- Gesamterwartung: Selbst im besten Fall liegt der Zuwachs unter dem von E8P (≈ 1,2 dB gegen Lloyd-1D). Der Beitrag wäre „buchloses, gruppenbasiertes Buch mit Quantenstruktur, nahe an E8P“, nicht „besser als E8P“. Hinweis von S-001: Auf echten Sprachmodellen kann die Skalenwahl (g16) jeden Gewinn von < 0,5 dB verdecken.
