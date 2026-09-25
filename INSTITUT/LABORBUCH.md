# Laborbuch des Instituts

Datiert, ehrlich, zurückgenommene Aussagen bleiben stehen.

---

## 24.09.2026 — Gründung, erste Linie, eine Idee auf Papier getötet

**Ausgangslage (aus Projekt A/B gelesen):** 16 Winkel liegen auf MNIST vor dem
2.048-Parameter-Quadratmodell; Verschränkung und Tiefe sind *nicht* belegt;
Struktur wird erst ab 16.384 Dimensionen schneller; die Kontrolle der alten
Kernaussage war zu schwach (Review 23.09.). Das Institut soll deshalb nicht
„noch einen Quantenbaustein“ bauen, sondern eine **Ressource** finden, die
wirklich an der Quantenstruktur hängt.

**Entscheidung des Direktors:** erste Linie = *Magie-Budget* (IDEEN Nr. 1).
Begründung: (a) es ist die Ressourcengröße, mit der die Quanteninformatik
selbst Kosten misst (T-Zahl); (b) es gibt eine harte, nachzählbare
Effizienzaussage (Multiplikationen → Additionen), die nicht an der
Faktor-64-Breitenfrage hängt; (c) ein Test dauert Minuten; (d) in der
Websuche nicht gefunden.

**Verifikation des Bausteins, vor jeder Zahl:**
- Schaltung als Matrix orthogonal: Abweichung 4,8·10⁻⁷.
- CNOT-Ring gegen Gatterdefinition: |100000⟩ → |011111⟩ (CNOT 0→1 … 4→5
  setzt alle Bits, CNOT 5→0 löscht Bit 0) — stimmt.
- Zufällige Clifford-Winkel: Matrixeinträge ∈ {−1, 0, 1}·c — die
  Schicht ist **tatsächlich multiplikationsfrei**.
- Vortest des Urteilsbaums: 3 Szenarien × 5 Ziehungen + fehlschlagende
  Auflösungsprüfung — bestanden.
- Laufzeit: Gatterform 0,1 s/Schritt → Matrixform 0,023 s/Schritt
  (mathematisch identisch; die Multiplikationszahl der *Inferenz* wird
  analytisch für die Gatterform gezählt).

**Vorregistrierung** (Code `abteilungen/clifford.py`, Folgefragen in
`agenda.json`, beides committet *bevor* das Ergebnis vorlag):
C-0 Schaltung > Identität; C-1 Rundung kostet ≤ 2 pp; C-2 gerundet > zufällige
Clifford **und** > feste Hadamard; C-3 M\* beschreibend. 10 Seeds (kleinster
exakter Wilcoxon-p 0,002), Lernrate je Arm auf Tuning-Seeds 900/901,
Gitter in beide Richtungen, 1.500 Schritte in Suche und Messung.

### Idee Z-1 auf Papier widerlegt (kein Rechenlauf nötig)

*Idee:* Zero-Noise-Extrapolation aus der Quanten-Fehlermitigation auf
Quantisierung übertragen — Logits mit b und b−1 Bit (verschachtelt: der
(b−1)-Code ist der gerundete halbe b-Code) auf „unendlich viele Bit“
extrapolieren. Neuheit laut Suche gegeben.

*Rechnung:* In einer linearen Schicht ist Extrapolation der Ausgaben
dasselbe wie Extrapolation der Gewichte. Sei c = round(W/s). Für gerade c ist
W_{b−1} = W_b, es passiert nichts. Für ungerade c ist W_b − W_{b−1} = ∓s, und
die lineare Extrapolation 2W_b − W_{b−1} = s(c ∓ 1) schiebt den Wert **auf
die andere Seite, weg vom Sollwert**. Mittlerer quadratischer Fehler:
s²/12 (gerade) bzw. s²/12 + s² (ungerade) → gesamt ≈ 7× schlechter als
b Bit. Quadratische Richardson-Variante: ≈ 1,7× schlechter.
*Nachgerechnet (200.000 Normalgewichte, b=4):* 7,21× bzw. 1,74×.

*Warum es scheitert:* ZNE setzt voraus, dass der Fehler eine **glatte
Funktion der Rauschstärke** ist (Erwartungswert über Zufall). Deterministisches
Runden ist das Gegenteil: sägezahnförmig. *Was übrig bleibt:* nur für
echtes, zufälliges Rauschen (analoge/photonische Rechenwerke) — dort ist aber
die Streuung eines Einzeldurchlaufs größer als der Bias, den ZNE entfernt.
Zurückgestellt als MPI-Frage.

*Lehre (→ FEHLERBUCH):* Eine Idee, die „aus der Quantenphysik kommt“, trägt
deren **Voraussetzungen** mit. Zuerst die Voraussetzung prüfen, dann bauen.
Hat 0 Rechenstunden gekostet.

## 24.09.2026, nachmittags — C-clifford v1: ein Artefakt, sauber gefangen

Lauf 15 min, 10 Seeds, Rohdaten `ergebnisse/C-clifford__20260924_160847.json`.

| Arm | Genauigkeit (Mittel ± SD) | Mult./Inferenz |
|---|---|---|
| voll (24 Winkel + Kopf) | 0,9664 ± 0,0067 | 3.776 |
| dicht_linear | 0,9584 ± 0,0092 | 640 |
| ohne_schaltung (Identität) | 0,9432 ± 0,0096 | – |
| hadamard_fest | 0,9429 ± 0,0098 | 704 |
| zufall_clifford | 0,9386 ± 0,0179 | 704 |
| voll_z (nur ⟨Z_q⟩) | 0,8566 ± 0,0209 | – |
| clifford_k0 (v1, geerbter Kopf) | 0,8696 ± 0,0419 | 704 |

- **C-0: H1** — die trainierte Schaltung schlägt die Identität (+2,3 pp,
  p = 0,002, 10/10 Seeds). *Erste belastbare Zahl des Instituts.*
- **C-1/C-2/C-3 (v1): nicht verwertbar** — siehe FEHLERBUCH. Die
  explorative Diagnose mit frischem Kopf gab k=0: 0,950; k=4: 0,962;
  k=8: 0,965 (Rate nach Testergebnis gewählt → nur Hinweis, kein Befund).
- **Winkel liegen nicht am Gitter:** mittlerer Abstand 0,390 rad — exakt
  der Wert für gleichverteilte Winkel (π/8 = 0,393). Das Training zieht die
  Winkel *nicht* von selbst zu Clifford-Punkten. → Begründet `C-bewusst`.
- **Ehrliche Effizienzeinordnung:** Auf 8×8-Ziffern ist ein rein linearer
  Klassifikator (640 Mult., 0,958) billiger als jede QIRE-Zelle. Das
  Magie-Budget misst die Kosten der *Mischschicht* (64×64 dicht: 4.096 Mult.;
  Clifford: 0 Mult.; volle Winkel: 3.072). Relevant wird das erst in der
  Mitte eines größeren Netzes — dieselbe Lage wie in Projekt A.
- **⟨Z_q⟩ allein (6 Zahlen) verliert 11 pp** gegenüber der vollen
  Born-Verteilung (64 Zahlen). Bestätigt den Engpass aus dem Review →
  stützt `B-paritaet`.

**Design v2 (vor jeder neuen Zahl festgelegt, committet):** Rundungsarme mit
frischem Kopf und eigener, auf Tuning-Seeds 902/903 gesuchter Rate
(an k=0); C-1 als Nichtunterlegenheit (einseitiges 95-%-KI ganz über −2 pp →
H1, ganz darunter → H0); frische Mess-Seeds 10–19. `C-mnist` war durch
C-2: H0 vorregistriert freigeschaltet und läuft ebenfalls mit v2 — die
Änderung liegt vor jeder MNIST-Zahl.

## 24.09.2026, 16:30 — C-clifford-v2 (frische Seeds 10–19, faires Design)

Rohdaten `ergebnisse/C-clifford-v2__20260924_162826.json`, 11,7 min.

| Arm | Genauigkeit | | Arm | Genauigkeit |
|---|---|---|---|---|
| voll | 0,9683 ± 0,0101 | | clifford_k0 | 0,9499 ± 0,0087 |
| dicht_linear | 0,9623 ± 0,0097 | | clifford_k1 | 0,9571 ± 0,0114 |
| ohne_schaltung | 0,9516 ± 0,0062 | | clifford_k2 | 0,9601 ± 0,0138 |
| hadamard_fest (entartet) | 0,9516 ± 0,0062 | | clifford_k4 | 0,9642 ± 0,0101 |
| zufall_clifford | 0,9495 ± 0,0218 | | clifford_k8 | 0,9660 ± 0,0075 |
| voll_z | 0,8410 ± 0,0211 | | clifford_k12 | 0,9686 ± 0,0054 |

**Vorregistrierte Urteile:** C-0 **H1** (repliziert: +1,7 pp, p = 0,002,
frische Seeds). C-1 **HU** (−1,84 pp, KI [−2,57; −1,11] überdeckt −2 pp).
C-2 **H0** (gerundet = zufällige Clifford = Identität). C-3 M\* = 0 — aber
nichtssagend, siehe FEHLERBUCH (Grenze > Effekt).

**Was das heißt:** Die Rundung auf das Clifford-Gitter vernichtet den
*ganzen* Gewinn der Schaltung — der Nutzen lebt vollständig in den
Nicht-Clifford-Winkeln, also in der „Magie“. Das ist die quantenphysikalisch
interessante Lesart: *Clifford allein bringt nichts, genau wie Clifford-
Schaltungen allein keinen Quantenvorteil bringen (Gottesman–Knill).*

**Nachträglich, beschreibend (nicht vorregistriert):** Gewinnanteil
G(k) = (acc_k − acc_Identität)/(acc_voll − acc_Identität):

| k (freie Winkel von 24) | 0 | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|---|
| G(k) | −0,10 | 0,33 | **0,51** | 0,76 | 0,87 | 1,02 |

**Die Hälfte des Nutzens steckt in 2 von 24 Winkeln, drei Viertel in 4.**
Die Magie ist *konzentriert*. Das ist die Hypothese, die jetzt zählt; G(k),
M50 und M90 sind für MNIST vorregistriert (Code-Änderung vor der ersten
MNIST-Zahl). Wenn sich das auf MNIST und Fashion hält, ist das die Größe,
die einen Namen verdient: **die Magie-Konzentration einer Aufgabe.**

**Verfolgt:** `C-bewusst` ist durch C-1: HU vorregistriert freigeschaltet
und als Modul gebaut (Vortest bestanden — erst nach Korrektur eines Fehlers
im eigenen synthetischen Grenzfall).

## 24.09.2026, 17:10 — V-kompass (MNIST 16×16, 8 Seeds, 34 min)

Relativer Fehler / Genauigkeit des Netzes mit komprimiertem ΔW der ersten
Schicht (dicht: 0,9562):

| Budget | MI-Fiedler | Zufall (Ø 8) | natürlich | verzahnt | Morton | **SVD** |
|---|---|---|---|---|---|---|
| 512 | 0,978 / 0,49 | 0,988 | 0,983 / 0,54 | 0,988 / 0,50 | 0,984 / 0,47 | **0,930** / 0,45 |
| 2.048 | 0,916 / 0,50 | 0,961 | 0,916 / 0,47 | 0,961 / 0,38 | 0,918 / 0,47 | **0,777** / 0,53 |
| 4.096 | 0,831 / 0,62 | 0,927 | 0,831 / 0,55 | 0,926 / 0,38 | 0,832 / 0,54 | **0,644** / 0,73 |

Urteile: V-1 **HU** (MI/Zufall = 0,954, p = 0,008, aber < 10 %). V-2 **HU**
(MI = natürlich, Δ = 0,0004, p = 0,84). Die MI-Ordnung legt in allen Seeds
die 8 Ausgangsbits und die 8 Eingangsbits je in eine Hälfte — sie
*entdeckt* die natürliche Ordnung, mehr nicht.

**Einordnung:** Negativ für die Tensornetz-Kompression gelernter Filter,
positiv für die Methode des Instituts: Das Rote Team (SVD) hat den Befund
gefunden, den keine der „quanteninspirierten“ Varianten zeigt. Linie
V-kompass ruht; Folgeidee „Rang außen, Tensornetz innen“ in FEHLERBUCH.

## 24.09.2026, 18:00 — Neue Abteilung: Ideenschmiede (auf Maltes Wunsch)

Aufbau siehe README. Erste Runde: fünf Karten aus dem Befund
„Magie-Konzentration“ abgeleitet, Neuheit von einem Recherche-Agenten geprüft
(11 Websuchen, nur Trefferlisten — „nicht gefunden“ heißt nicht „existiert
nicht“). Alle fünf durchs Tor an Abt. Clifford:

| Karte | Idee | Neuheit |
|---|---|---|
| I-001 | Magie-Pruning (Winkel nach Verlustanstieg statt Gitterabstand) | teilweise |
| I-002 | Stabilisator-Rényi-Entropie sagt M50 vorher, ohne Rundungsversuche | nicht gefunden |
| I-003 | Magie-Konzentration als Kennzahl einer *Aufgabe* (Kontrolle: Zufallsetiketten) | teilweise |
| I-004 | Clifford-Gerüst + k Winkel von Anfang an | teilweise (CAFQA, VQE) |
| I-005 | nur diskrete Clifford-Wahl, Magie allein aus den Daten | teilweise (Stabilizer Bootstrap) |

Reihenfolge-Empfehlung des Direktors: **I-002 zuerst** (kein Training,
Minuten, rechnet auf gespeicherten Winkeln), dann I-001, dann I-003 mit
Zufallsetiketten. I-005 hat nach C-2 (feste Clifford = Identität) die
schlechteste Papierprognose — bleibt trotzdem stehen, weil ein klares Nein
dort die C-2-Aussage verallgemeinern würde.

## 24.09.2026, 19:15 — C-mnist: die Magie-Konzentration hält (vorregistriert)

MNIST 16×16 (8 Qubits, 256 Amplituden, 4 Lagen = 32 Winkel), 6.000
Trainings-/5.000 Testbilder je Seed, 10 Seeds, Design v3 (Kontrolle
Hadamard korrigiert, G(k)/M50/M90 vor der ersten MNIST-Zahl im Code).
Rohdaten `ergebnisse/C-mnist__20260924_181325.json`, 71 min.

| Arm | Genauigkeit | Mult./Inferenz |
|---|---|---|
| voll (32 Winkel) | **0,9385 ± 0,0035** | 19.200 |
| dicht_linear | 0,9118 ± 0,0038 | 2.560 |
| ohne_schaltung (Identität) | 0,9006 ± 0,0037 | – |
| hadamard_fest (jetzt echt) | 0,8831 ± 0,0039 | 2.816 |
| zufall_clifford | 0,9054 ± 0,0100 | 2.816 |
| voll_z (nur 8 ⟨Z⟩) | 0,7637 ± 0,0109 | – |
| clifford_k0 | 0,9069 ± 0,0160 | 2.816 |
| clifford_k1 | 0,9192 ± 0,0092 | 3.328 |
| clifford_k2 | 0,9208 ± 0,0079 | 3.840 |
| clifford_k4 | 0,9264 ± 0,0072 | 4.864 |
| clifford_k8 | 0,9342 ± 0,0069 | 6.912 |
| clifford_k12 | 0,9353 ± 0,0054 | 8.960 |

**Urteile:** C-0 **H1** (+3,8 pp über der Identität, p = 0,002, 10/10).
C-1 **H0** (Rundung kostet 3,2 pp, KI [−3,95; −2,36] — Magie wird gebraucht).
C-2 **HU** (gerundet > echte Hadamard-Mischung, p = 0,010; ≈ zufällige
Clifford). C-3: **G(1) = 0,49, G(2) = 0,53, G(4) = 0,68, G(8) = 0,89,
G(12) = 0,92 → M50 = 2 (bereits k = 1 fast 0,5), M90 = 12 von 32.**

**Was das heißt:**
1. **Magie-Konzentration repliziert, jetzt vorregistriert:** Auf den Ziffern
   (explorativ) trugen 2 von 24 Winkeln die Hälfte des Nutzens, auf MNIST
   (vorregistriert) trägt **ein einziger von 32** fast die Hälfte. Die
   Kurve ist steil am Anfang und flach danach — der Nutzen der
   Quantenschaltung sitzt in sehr wenigen magischen Gattern.
2. **Clifford allein trägt nichts** — auch auf MNIST ist die gerundete
   Schaltung nicht besser als eine zufällige Clifford-Schaltung.
   Quantenphysikalisch sauber: Clifford = klassisch simulierbar
   (Gottesman–Knill) = kein Gewinn; der Gewinn kommt aus der Magie.
3. **Auf MNIST schlägt die Schaltung den linearen Klassifikator** (+2,7 pp,
   beschreibend, nicht vorregistriert) — auf den Ziffern war es nur +0,6 pp.
   Mit k = 2 Magiewinkeln: 0,921 bei 3.840 Mult. gegen linear 0,912 bei 2.560.
4. **Die Auslese bleibt der Engpass:** nur 8 ⟨Z⟩-Werte verlieren 17 pp.

**Freigeschaltet (vorregistriert):** C-fashion (C-1: H0). C-bewusst ist
schon offen. Beide mit Priorität 1 für Maltes PC (`institut.cmd alle`).

**Arbeitstitel für die Größe, falls Fashion es bestätigt:** *Magie-
Konzentration* M50/M90 einer Aufgabe — „wie viele magische Gatter braucht
eine Lernaufgabe für die Hälfte / 90 % ihres Quantennutzens?“

## 24.09.2026, 18:40 — Skalierung: 3 Rechenstränge + 3 Forscher-Agenten parallel

Auf Maltes Wunsch vergrößert. Parallel: C-fashion (2 Kerne), C-bewusst
(2 Kerne), I-008 (1 Kern); drei Agenten bauen I-002 (SRE-Vorhersage),
I-001 (Magie-Pruning) und C-hardware (Qiskit, Clifford+T). Regel für die
Agenten: nur neue Dateien, keine gemeinsamen Dateien anfassen, kein Commit —
der Direktor integriert. Ideenschmiede Runde 2: I-008 (Q3) und I-010 (Q2)
weitergeleitet, I-006/007/009/011 mit Begründung verworfen (I-009
nachgerechnet: 10× teurer; I-011: Quantenlauf auf dem Hyperwürfel *ist*
die QIRE-Lage).

### I-008 Schuss-Budget (Q3), Ziffern 8 Seeds + MNIST 3 Seeds beschreibend, 8 min

Rohdaten `ergebnisse/I-008__20260924_183851.json`. Genauigkeit bei S Messungen je Bild:

| Auslese | S=10 | 100 | 1.000 | 10.000 | exakt |
|---|---|---|---|---|---|
| born (64) | 0,444 | 0,877 | 0,959 | 0,967 | 0,967 |
| paritaet (22) | 0,328 | 0,704 | 0,884 | 0,906 | 0,907 |
| z (6) | 0,215 | 0,456 | 0,625 | 0,645 | 0,650 |
| MNIST born (256) | 0,327 | 0,769 | 0,924 | 0,935 | 0,938 |

A-1 **HU** (−0,81 pp bei S=1.000, KI [−1,14; −0,48]). A-2 **H0** (born
schlägt paritaet bei S=100 um 17 pp, p = 0,008) — Hypothese falsch, siehe
FEHLERBUCH. **Hardware-Einordnung:** Die Schaltung braucht auf echter
Hardware rund **1.000 Schüsse je Bild** (MNIST: 1.000 → −1,4 pp,
10.000 → −0,3 pp). Das ist die Zahl, die in jede ehrliche Aussage über
„Quantenhardware als KI-Beschleuniger“ gehört — und eine gute Frage fürs
MPI-Gespräch.

## 24.09.2026, 18:45 — C-hardware (Q3): Magie-Konzentration in T-Gattern

Forscher-Agent, Modul `abteilungen/hardware.py`, Rohdaten `ergebnisse/C-hardware__20260924_184246.json`.
Die QIRE-Zelle als Qiskit-Schaltung (Bitordnung: Modell-Qubit q =
Qiskit-Qubit n−1−q), Operatorfehler 5·10⁻¹⁶ (float64); vom Direktor mit
eigenen Zufallswinkeln nachgeprüft: 2,5·10⁻⁷ (float32-Modell). Negativ-
kontrolle ohne Umnummerierung: 0,4–0,7 (der Test *kann* die Bitordnung
erkennen). Clifford-Winkel exakt als h/x/z, magische Ry-Winkel mit
Ross–Selinger-gridsynth bzw. Solovay–Kitaev nach Clifford+T übersetzt.

| k magische Winkel (MNIST, 32 CNOT) | T-Gatter ε=10⁻² | ε=10⁻³ | ε=10⁻⁴ | Gewinnanteil G(k) |
|---|---|---|---|---|
| 0 | **0** | **0** | **0** | 0,17 |
| 1 | 11 | 30 | 42 | 0,49 |
| 2 | 28 | 62 | 84 | 0,53 |
| 4 | 69 | 123 | 168 | 0,68 |
| 8 | 152 | 247 | 337 | 0,89 |
| 12 | 236 | 371 | 503 | 0,92 |

H-1 **bestätigt** (k=0 → 0 T-Gatter, exakter Operator, alle Seeds).
T ≈ 30,9·k bei ε=10⁻³ (R² = 1,000). Born-Verteilung der Clifford+T-Schaltung
weicht bei ε=10⁻³ höchstens 1,3·10⁻⁵ vom Modell ab.

**Bedeutung:** Damit ist das Magie-Budget in die Kostenwährung eines
fehlertoleranten Quantencomputers übersetzt: *die Hälfte des Quantennutzens
für ~30–60 T-Gatter, 90 % für ~370.* Die volle Schaltung (32 magische
Winkel) bräuchte ~990 T-Gatter. Grenzen: nur rauschfrei simuliert; die Kosten
der Amplitudenkodierung (Zustandspräparation, selbst sehr T-teuer) sind nicht
mitgezählt — das ist die nächste ehrliche Frage.

## 24.09.2026, 19:10 — C-bewusst: Magie lässt sich nicht wegtrainieren

Ziffern, frische Seeds 20–29, λ auf Tuning-Seeds gesucht (0,3, innen),
Rohdaten `ergebnisse/C-bewusst__20260924_185324.json`, 27 min.

| | normal | Clifford-bewusst |
|---|---|---|
| volles Modell | 0,9609 ± 0,0133 | 0,9473 ± 0,0200 |
| Abstand zum Gitter (rad) | 0,396 | 0,229 |
| gerundet k=0 | 0,9477 | 0,9438 |
| k=2 | 0,9536 | 0,9514 |
| k=4 | 0,9623 | 0,9590 |

B-1 **H0** (−0,4 pp, p = 1,0). B-2 **HU**. Deutung im FEHLERBUCH: Der
Strafterm zieht *alle* Winkel in den Clifford-Bereich, der nachweislich
nichts bringt (C-2). Das Institut hat damit drei unabhängige Belege für
denselben Satz: **Der Nutzen der Schaltung sitzt in der Magie, und zwar in
wenigen Winkeln.** (C-2 auf zwei Datensätzen, C-3/G(k), jetzt C-bewusst.)

## 24.09.2026, 19:30 — C-kodierung (Q3): das Laden frisst die Einsparung

Forscher-Agent, Modul `abteilungen/kodierung.py`, Rohdaten `ergebnisse/C-kodierung__20260924_192627.json`,
MNIST 16×16, 1 Seed (Seed 10 exakt reproduziert), 20 Bilder, ε=10⁻³ je Rotation.

| Kodierung | T-Gatter | CNOT | Treue | Gen. direkt | Gen. Kopf neu |
|---|---|---|---|---|---|
| exakt (Möttönen, nur Ry) | 7.636 | 250 | 1,000 | 93,2 % | 93,3 % |
| MPS χ=4 | 6.329 (Untergrenze ~3.250) | 107 | 0,950 | 92,6 % | 93,2 % |
| MPS χ=2 | 1.176 | 38 | 0,736 | 70,8 % | 87,6 % |
| MPS χ=1 (Produkt) | 225 | 0 | 0,321 | 31,8 % | 64,4 % |
| Winkel auf π/2 (Clifford) | 0 | 250 | 0,102 | 13,3 % | 50,7 % |

K-1 **bestätigt**: Laden 7.636 T gegen 989 T für die ganze Schaltung (k=32)
und 62 T für den Halbnutzen-Punkt. K-2 **nein**: keine Kodierung unter 10 %
der T-Kosten ohne großen Verlust; χ=2 kostet 15 % und verliert 5,7 pp.

**Einordnung, ehrlich:** Das ist der aus der Quanten-KI bekannte
Datenlade-Flaschenhals (Aaronson, „Read the fine print“, Nature Physics 2015),
jetzt in T-Gattern für QIREs eigene Schaltung beziffert. Auf echter
fehlertoleranter Hardware dominiert die Kodierung die Kosten; die
Magie-Konzentration der *Schaltung* spart dort nur ~13 % der Gesamtkosten.
Im Simulator (Multiplikationen) gilt die Aussage weiter.

**Nicht aufgeben, vertiefen:** Die Kodierung ist selbst der größte
Magieverbraucher — also muss das Magie-Budget über **Kodierung + Schaltung
gemeinsam** gerechnet und optimiert werden (neue Karte I-020). Noch nicht
gemessen: dünn besetzte Präparation (71 % der Pixel sind null) und eine
Zelle, die direkt auf der billigen Kodierung trainiert wird.

## 24.09.2026, 19:45 — I-001 Magie-Pruning und I-002 Stabilisator-Entropie

**I-001** (Forscher-Agent, Ziffern, Seeds 30–39, eigene Rate je Arm):
G(k) für k = 0 / 1 / 2 / 4 —

| Rangfolge | k=1 | k=2 | k=4 |
|---|---|---|---|
| Abstand zum Gitter (bisher) | −0,04 | 0,60 | 0,69 |
| Verlustanstieg (neu) | 0,15 | **0,67** | 0,79 |
| gierig | 0,20 | 0,54 | **0,88** |
| Zufall | 0,15 | 0,38 | 0,80 |

P-0 H1 (Schaltung > Identität, p = 0,002). P-1 **HU** (+0,07 G gegenüber
Abstand, p = 0,375 — Auflösung auf den Ziffern zu klein). P-2 **H1**
(Wirkung schlägt Zufall bei k=2: +0,29 G, p = 0,027). → *Welche* Winkel
magisch bleiben, ist nicht egal; die Wahl nach Wirkung ist messbar besser als
Zufall. Wiederholung auf MNIST vorgemerkt.

**I-002** (Forscher-Agent, exakte Operator-SRE über alle 4²ⁿ Pauli-Strings,
Vortest gegen Brute Force n=1–3, π/4-Winkel → log₂(4/3)): S-1 **H0**
(ρ = 0,26, p = 0,26; innerhalb der Datensätze p = 0,58), S-2 HU.
**Einsicht:** gelernte Schaltungen tragen *so viel* Magie wie zufällige
(Ziffern 4,38 vs. 4,58; MNIST 6,26 vs. 6,28) — nutzen davon aber nur wenige
Winkel. **Vorhandene Magie ≠ genutzte Magie.** Ein Maß am Zustand oder
Operator allein sieht die Aufgabe nicht; G(k) tut es.

## 24.09.2026, 21:45 — C-fashion: Konzentration repliziert, Designfehler gefunden

Fashion-MNIST 16×16, 8 Qubits, 32 Winkel, Seeds 10–19, Design v3 wie C-mnist
(Rohdaten `ergebnisse/C-fashion__20260924_214236.json`, 68 min, 4 Kerne).

| Arm | Fashion | MNIST (C-mnist) |
|---|---|---|
| Identität (ohne Schaltung) | 0,796 | 0,901 |
| zufällige Clifford | 0,816 | 0,905 |
| k = 0 (gerundet) | 0,817 | 0,907 |
| k = 2 | 0,827 | 0,921 |
| k = 4 | 0,833 | 0,926 |
| k = 8 | **0,838** | 0,934 |
| voll | 0,833 | 0,939 |

- C-0 **H1**: Schaltung schlägt Identität, +3,6 pp, p = 0,002 (MNIST +3,8).
- C-1 HU (knapp: KI der Rundung [−2,05; −1,02] pp streift die 2-pp-Grenze).
- C-2: k0 = zufällige Clifford (+0,17 pp, p = 0,77) — wie auf beiden
  anderen Datensätzen. Anders als auf MNIST trägt auf Fashion aber schon
  *irgendeine* Clifford-Schaltung 2,1 pp (p = 0,002) über die Identität:
  G(0) = 0,58 statt 0,17. Das ist Clifford-Durchmischung, keine Magie.
- **Magie-Anteil** (Basis k0, Nachrechnung Direktor): k=1 0,24 · k=2 0,64 ·
  **k=4 1,01** · k=8 1,34. MNIST: 0,39 · 0,44 · 0,62 · 0,86. Die
  Konzentration ist auf Fashion *stärker*: M50 = 2, M90 = 4 von 32 Winkeln
  (vorregistriertes Maß mit Basis Identität: M50 = 0, M90 = 4).

**Warnsignal:** k=8 und k=12 *schlagen* den vollen Arm (+0,53 pp, p = 0,010;
+0,46 pp, p = 0,025). Ursache gefunden: `_gerundet` startet beim fertig
trainierten Modell und trainiert **noch einmal 1.500 Schritte** (Rate 0,32,
frischer Kopf) — der volle Arm bekommt diese zweite Phase nicht. Ob k8 > voll
Regularisierung durch Runden ist oder bloß mehr Training, ist so nicht
trennbar, und alle G(k) gegen „voll“ können aufgebläht sein (FEHLERBUCH).
**Konsequenz, vor jeder weiteren Zahl:** C-referenz (Arm *nachtrainiert* =
alle Winkel frei, identische zweite Phase; MNIST + Fashion, frische Seeds
30–39) vorregistriert; C-skala und I-024 bekommen denselben Arm als
Referenz. Der Name „Magie-Konzentration“ wartet auf C-referenz.

## 24.09.2026, 22:10 — I-003: Aufgabe oder Architektur? Positivkontrolle scheitert am Lernverfahren

Ziffern 8×8 (6 Qubits, 24 Winkel), Seeds 60–69, Forscher-Agent-Modul
`magie_aufgabe.py` (Rohdaten `ergebnisse/I-003__*.json`, 22 min).

| Aufgabe (Messgröße) | Identität | k0 | k1 | k2 | k4 | k8 | k12 | voll |
|---|---|---|---|---|---|---|---|---|
| echt (Test) | 0,945 | 0,941 | 0,953 | 0,957 | 0,962 | 0,958 | 0,963 | 0,959 |
| zufall (Training, Memorieren) | 0,240 | 0,252 | 0,250 | 0,254 | 0,251 | 0,278 | 0,285 | 0,272 |
| struktur (Test, Clifford-Lehrer) | 0,784 | 0,735 | 0,737 | 0,751 | 0,766 | 0,773 | **0,786** | **0,726** |

- T-1 **HU (Vorbedingung)**: Auf der Strukturaufgabe — per Konstruktion mit
  Magie 0 exakt lösbar, Kopf auf den Lehrer-Merkmalen 0,934 — landet die
  voll trainierte Zelle *unter* der Identität (−5,7 pp, p = 0,010).
  Gradientenabstieg aus Zufallswinkeln findet die diskrete Clifford-Lösung
  nicht. T-2 damit nicht auswertbar (vorregistriert).
- Zweiter Beleg für den Nachtrain-Fehler (C-fashion): k12 schlägt voll auf
  der Strukturaufgabe um **+6 pp**, allein durch die zweite Trainingsphase.
- Explorativ (T-3, beschreibend): echt M50 = 1, M90 = 4; zufall (Memorieren)
  M50 = 8. Die G-Kurve von echt liegt über zufall (+0,51 G, p = 0,002).
  Hinweis, dass Generalisieren weniger Magie braucht als Auswendiglernen —
  **nicht belastbar**, bis Referenz (C-referenz) und Positivkontrolle
  (I-003-v2) stehen.

Lehre: *exakt lösbar* ≠ *lernbar*. Eine Positivkontrolle muss mit dem
Verfahren feuern können, nicht nur mit der Architektur. I-005 (diskrete
Clifford-Suche) bekommt dadurch Gewicht; I-003-v2 als Idee eingetragen.

## 24.09.2026, 23:05 — I-020 Gesamtbudget (Q3): Laden halbiert, nicht geviertelt

MNIST 16×16, 3 Seeds, beschreibend (vorregistriert, kein Signifikanztest);
Rohdaten `ergebnisse/I-020__20260924_230045.json`, 56 min (Modul Forscher-Agent).
T-Zahl gesamt = Laden (Qiskit-Synthese, exakt) + Schaltung (≈ 31·k).

| Kodierung | k | T gesamt | Genauigkeit | Anteil Laden |
|---|---|---|---|---|
| dicht exakt (Referenz) | 32 | 8.776 | 0,940 | — |
| **dünn besetzt exakt** | 32 | **4.372** | **0,940** | 77 % |
| dünn besetzt exakt | 4 | 3.506 | 0,933 | 96 % |
| dünn besetzt exakt | 0 | 3.382 | 0,911 | 100 % |
| MPS χ=2 | 32 | 2.178 | 0,881 | 55 % |
| MPS χ=2 | 4 | 1.312 | 0,857 | 91 % |
| dünn, π/4-gerundet | 32 | 1.001 | 0,731 | 1 % |

- G-1 **H0** (Abbruchkriterium der Karte): keine Kombination ≤ 2.194 T (25 %)
  hält 2 pp an der Referenz. Aber: **dünn besetzte Präparation halbiert die
  T-Zahl ohne Verlust**, mit k=4 auf 40 % bei −0,7 pp.
- Kodierungs-Verluste (MPS, π/4) sind nicht durch Schaltungs-Magie
  aufzuholen: Die Zelle auf billiger Kodierung neu zu trainieren rettet 6–20
  pp nicht.
- Kernsatz wird schärfer: *Mit Magie-Konzentration (k=4) ist die Schaltung
  4 % der Rechnung; 96 % sind Datenladen.* Die nächste Front ist eindeutig
  die Kodierung selbst.
- Vorbehalt: Die Magie-Arme nutzen `_gerundet` (Nachtrain-Fehler, C-referenz).

## 24.09.2026, 23:50 — C-referenz: Verdacht widerlegt, Konzentration repliziert auf frischen Seeds

MNIST + Fashion 16×16, frische Seeds 30–39, Raten aus C-mnist/C-fashion,
Arm *nachtrainiert* = alle 32 Winkel frei mit derselben zweiten Phase wie die
Rundungsarme (Rohdaten `ergebnisse/C-referenz__20260924_234210.json`, 80 min).

| | Identität | k0 | k1 | k2 | k4 | k8 | nachtrainiert | voll |
|---|---|---|---|---|---|---|---|---|
| MNIST | 0,900 | 0,910 | 0,916 | 0,918 | 0,926 | 0,931 | 0,938 | 0,938 |
| Fashion | 0,798 | 0,816 | 0,822 | 0,827 | 0,831 | 0,834 | 0,804* | 0,832 |

\* 4 von 10 Seeds kollabieren (0,725–0,812) bei Rate 0,32.

- R-1 (vorregistriert) **HU** gesamt; **MNIST: keine Verzerrung** (d = −0,03 pp,
  KI [−0,24; +0,18] ganz in ±0,3 pp). Fashion: Referenz instabil, auf den 6
  stabilen Seeds d = +0,03 pp.
- R-2 **HU (Vorbedingung)** wegen der kaputten Fashion-Referenz. MNIST allein:
  Magieanteil Gm*(4) = 0,59, KI [0,30; 0,77], Magiegewinn +2,7 pp (p = 0,002).
- R-3 formal „H1“, aber **Artefakt** der instabilen Referenz — verworfen.
  k8 − voll auf Fashion: +0,24 pp, p = 0,08 (repliziert +0,53 pp nicht).
- **Replikation auf frischen Seeds** (G mit Basis Identität, gegen voll):

| k von 32 | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| MNIST C-mnist (Seeds 10–19) | 0,17 | 0,49 | 0,53 | 0,68 | 0,89 |
| MNIST C-referenz (Seeds 30–39) | 0,27 | 0,43 | 0,49 | 0,70 | 0,83 |
| Fashion C-fashion (10–19) | 0,58 | 0,68 | 0,85 | 1,01 | 1,14 |
| Fashion C-referenz (30–39) | 0,52 | 0,73 | 0,85 | 0,99 | 1,07 |

**Ergebnis:** Die Magie-Konzentration ist auf zwei Datensätzen und je zwei
unabhängigen Seed-Sätzen stabil: **4 von 32 Winkeln tragen 70 % (MNIST) bzw.
~100 % (Fashion) des Schaltungsgewinns**. M50 = 2–4 (MNIST liegt bei k=2
genau an der Grenze), M90 ≈ 8–12 bzw. 4. Der Störfaktor war nicht das
Nachtraining, sondern eine für wenige Parameter getunte Lernrate (FEHLERBUCH).

## 24.09.2026, 23:55 — I-015 Magie-Kaskade (Q3): spart Schaltungs-T, das Laden frisst es — außer die Clifford-Stufe läuft klassisch

MNIST 16×16, Seeds 90–94 (5, beschreibend für K-2), Kaskade 0 → 2 → 8,
Schwelle τ auf Validierung (Rohdaten `ergebnisse/I-015__20260924_234941.json`, 7 min).

- K-1 **H1**: Kaskade erreicht die Testgenauigkeit von fest k=2 (+0,05 pp,
  KI [−0,45; +0,55]) mit **17 T = 28 %** der Schaltungs-T von k=2.
- K-2 B+: Konfidenz schlägt Zufallsweiterleitung in 5/5 Seeds (+1,2 pp;
  p = 0,0625 = kleinstmöglich bei 5 Seeds).
- K-3: **mit Ladekosten 114 %** von k=2 — 10–20 % der Bilder werden neu
  geladen (7.636 T je Lauf). Papierschranke der Karte bestätigt (lohnt erst
  unter 0,8 % Weiterleitung).
- **Nachträgliche Rechnung (Direktor, nicht vorregistriert):** Die Stufe k=0
  ist eine Clifford-Schaltung auf einem *klassischen* Vektor — klassisch
  exakt mit ≤ 8.192 Additionen, ohne Multiplikation und ohne Laden. Läuft sie
  klassisch und gehen nur die unsicheren ~10 % der Bilder auf den
  Quantenrechner, kostet die Kaskade **1.118 T = 14,5 % der Hardware-T von
  k=2** bei gleicher Genauigkeit (92,28 % gegen 92,24 %). 0 → 8 direkt: 9,4 %.
  → Als **I-015-hybrid** mit 10 frischen Seeds vorregistriert (H-1: ≤ 25 %
  und Nichtunterlegenheit 0,5 pp).

## 25.09.2026, 00:00 — I-004 Clifford-Gerüst von Anfang an (Ziffern): Auflösung reicht nicht

Ziffern 8×8, Seeds 80–89 (Rohdaten `ergebnisse/I-004__20260924_235757.json`, 8 min).
R-0 H1 (Gewinn +2,1 pp, p = 0,004); R-1 HU (zufälliges Gerüst + 4 Winkel:
G = 0,71, KI 0,47–0,96); R-2 HU (+0,28 G für nachträgliches Runden, Holm-p =
0,15); R-3 HU (Gerüst mit neu initialisierten Magiewinkeln vs. Nachtraining,
−0,06 G, KI −0,23…+0,10).

| G(k) | k=1 | k=2 | k=4 | Trainings |
|---|---|---|---|---|
| nachträglich runden | 0,41 | 0,71 | 0,77 | 2 |
| Gerüst aus vollem Modell, Winkel neu | 0,26 | 0,65 | 0,83 | 2 |
| **zufälliges Gerüst**, Winkel von Anfang an | 0,23 | 0,43 | 0,71 | **1** |

Beschreibend: Wenig-magische Schaltungen brauchen das volle Modell nicht —
zufälliges Clifford-Gerüst + 4 trainierte Winkel holt ~70 % des Gewinns mit
halbem Trainingsaufwand. Belastbar erst auf MNIST (I-004-mnist, Idee).

## 25.09.2026, 00:10 — I-015-hybrid (Q3, vorregistriert): **Hybrid-Magie-Kaskade spart 88 % der Hardware-T** ✔

MNIST 16×16, 10 frische Seeds 100–109, Kaskade 0 → 2 → 8, Stufe 0 (Clifford,
0 T) **klassisch** (≤ 8.192 Additionen, 0 Multiplikationen, kein Laden), jede
Quantenstufe = Laden 7.636 T + Schaltung; τ auf Validierung nach den
Hybrid-Kosten (Rohdaten `ergebnisse/I-015-hybrid__20260925_000945.json`, 12 min).

- **H-1 H1**: mittlere Hardware-T je Bild **902 = 11,7 % von k=2 auf
  Hardware** (7.698 T); Genauigkeit 91,74 % gegen 91,68 % (+0,06 pp,
  einseitiges KI [−0,20; +0,31], Grenze −0,5 pp). Nur **7,9 %** der Bilder
  erreichen den Quantenrechner.
- **H-2 H1**: Konfidenz schlägt Zufallsweiterleitung gleicher Quote
  (+1,06 pp, p = 0,031).
- H-3: zweistufig 0 → 2: 9,5 %; 0 → 8: 8,0 % (beide ≈ k2-Genauigkeit).
- **Robustheit (Direktor):** Auf 4/10 Seeds wählt die Validierung τ* = 0
  (die klassische Clifford-Stufe allein genügt dort; Test −0,2 … −0,7 pp).
  Ohne diese Seeds: 1.503 T = **19,5 %** — immer noch unter 25 %.
- Replikation des Post-hoc-Werts (I-015, Seeds 90–94: 14,5 %) auf frischen Seeds.

**Einordnung:** Das ist der erste Befund des Instituts, der eine echte
Hardware-Ersparnis *mit* Ladekosten zeigt. Er verbindet die beiden
Hauptbefunde: Magie-Konzentration (wenige Winkel reichen) und
Clifford = klassisch billig (die magiefreie Stufe braucht keinen
Quantenrechner). Ehrlich: Es ist ein **Hybrid**-Befund — die Ersparnis
entsteht, weil ein klassisches multiplikationsfreies Netz die meisten Bilder
selbst entscheidet und nur schwierige an die Quantenschaltung abgibt.

## 25.09.2026, 01:50 — I-024 Magie gegen Datenmenge: Gegenrichtung — Konzentration schärft sich mit Daten

MNIST 16×16, n_train ∈ {500, 2.000, 10.000} (geschachtelt), Seeds 100–109,
faire Referenz *nachtrainiert*, Rate je Arm und n eigen (Rohdaten
`ergebnisse/I-024__20260925_014438.json`, 95 min).

| n_train | Identität | k0 | k1 | k2 | k4 | k8 | nachtrainiert | voll | G(2) | G(8) |
|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 0,835 | 0,831 | 0,842 | 0,847 | 0,855 | 0,867 | 0,873 | 0,872 | 0,34 | 0,86 |
| 2.000 | 0,883 | 0,882 | 0,892 | 0,898 | 0,898 | 0,910 | 0,919 | 0,917 | 0,43 | 0,75 |
| 10.000 | 0,908 | 0,914 | 0,921 | 0,929 | 0,932 | 0,941 | 0,943 | 0,942 | **0,61** | 0,96 |

- D-0 **H1** (+3,5 pp bei 10.000, p = 0,002).
- D-1 **H0 (Gegenrichtung)**: Der Magie-Abstand (nachtrainiert − k0) ist bei
  wenig Daten *größer* (4,18 gegen 2,91 pp, p = 0,019).
- D-2 **H0 (Gegenrichtung)**: 2 Magiewinkel tragen bei 500 Bildern *weniger*
  (G(2) −0,28, p = 0,042). Beide Gegenrichtungs-Tests einseitig ohne Holm:
  deutlicher Hinweis, kein Beweis.
- d_nach ≈ 0 bei allen n → dritter Beleg gegen den Nachtrain-Verdacht.

**Einordnung:** Der Gewinn der Schaltung ist von der Datenmenge fast
unabhängig (~3,5 pp), aber seine *Verteilung* nicht: Mit mehr Daten
konzentriert das Training die genutzte Magie auf weniger Winkel. Die
Magie-Konzentration ist also eine Eigenschaft des **gelernten** Modells, die
mit der Datenmenge wächst — nicht der Architektur allein (passt zum
explorativen I-003-Hinweis). Die Regularisierungsidee war falsch gestellt
(Runden danach ≠ Training im Clifford-Raum) → I-004-mnist mit n = 500.

## 25.09.2026, 02:00 — I-015-hybrid-fashion: Hybrid-Kaskade repliziert auf Fashion ✔

Fashion-MNIST 16×16, 10 frische Seeds 110–119, sonst identisch vorregistriert
(Rohdaten `ergebnisse/I-015-hybrid-fashion__20260925_015647.json`, 12 min).

- **H-1 H1**: 1.837 T = **23,9 % der Hardware-T von k=2**, Genauigkeit
  81,98 % gegen 81,67 % (**+0,31 pp**, KI [+0,04; +0,57] — sogar besser als
  k2 allein, weil unsichere Bilder bis k=8 durchgereicht werden). 14,2 % der
  Bilder gehen auf den Quantenrechner.
- **H-2 H1**: Konfidenz schlägt Zufall (+0,82 pp, p = 0,004).
- H-3: 0 → 8 direkt: 12,1 % bei 81,79 %.
- Robustheit: nur 1/10 Seeds mit τ* = 0; ohne ihn 26,5 % (knapp über der
  25-%-Marke) — die Ersparnis ist auf Fashion kleiner als auf MNIST (11,7 %),
  weil die Clifford-Stufe dort öfter unsicher ist.

**Zwischenfazit Hybrid-Magie-Kaskade (zwei Datensätze, 20 frische Seeds):**
Ein klassisches, multiplikationsfreies Clifford-Netz entscheidet 86–92 % der
Bilder selbst; nur der Rest geht auf die Quantenschaltung mit wenigen
Magiewinkeln. Hardware-Aufwand **12–24 %** von „alles auf den
Quantenrechner, k=2“, bei gleicher oder besserer Genauigkeit.

## 25.09.2026, 03:10 — I-033 (Q3, vorregistriert): **Magie-Konzentration gilt auch für das Laden** ✔

MNIST 16×16, Seeds 140–149, gerundet werden die Gatterwinkel φ der
Gray-Code-Zerlegung der dünn besetzten Präparation (Basis I-020: im Mittel
~110 von 255 φ ≠ 0, 3.260 T); faire Referenz = exakt geladen mit gleicher
zweiter Phase (Rohdaten `ergebnisse/I-033__20260925_030825.json`, 72 min).

| Laden | alle π/2 | Top-4 | Top-16 | Top-32 | **Top-48** | Ebene 6 | exakt |
|---|---|---|---|---|---|---|---|
| Lade-T je Bild | 0 | 123 | 495 | 988 | **1.479** | 1.211 | 3.260 |
| Anteil | 0 % | 4 % | 15 % | 30 % | **45 %** | 37 % | 100 % |
| Genauigkeit | 0,535 | 0,677 | 0,881 | 0,928 | **0,938** | 0,883 | 0,944 |

- L-0 H1 (+3,6 pp über Identität, p = 0,002).
- **L-1 H1**: Top-48 hält die Referenz per TOST innerhalb 1 pp (−0,57 pp,
  Holm-p = 0,002) bei **45 % der Lade-T** — in allen 10 Seeds −0,1 … −0,9 pp.
- **L-2 H1**: *welche* Winkel exakt bleiben, entscheidet: Top-16 schlägt 16
  zufällige um **+31 pp** (p = 0,002).
- L-3: grob-zu-fein (obere Baumebenen) ist schwächer als Top-m. Lade-M50 =
  16 Winkel (15 % der T), Lade-M90 = 32 Winkel (30 %).

**Einordnung:** Dieselbe Struktur wie in der Schaltung: Wenige Winkel tragen
den Nutzen, wenn man sie nach Wirkung (hier: Abstand zum Clifford-Gitter)
wählt. Damit gibt es die Magie-Konzentration jetzt auf **beiden** Seiten der
Q3-Rechnung — Modell (4 von 32 Winkeln) und Daten (32–48 von ~110).
Kombiniert mit der Hybrid-Kaskade (nur 8–14 % der Bilder auf Hardware) ist
eine Größenordnung von ~2 % der heutigen Hardware-T denkbar — **nicht
gemessen**, nur geschätzt → Karte I-034.

## 25.09.2026, 04:50 — C-skala: Hinweis auf sublineares Magie-Wachstum, saubere Kontrolle fällt der Lernrate zum Opfer

MNIST, n = 6/8/10 Qubits (8×8, 16×16, 32×32) + fest10 (8×8 hochskaliert ⊗
|+⟩⁴), L = 4, 2.500 Trainingsbilder, 500 Schritte, Seeds 120–129, faire
Referenz *nachtrainiert* mit eigener Rate (Rohdaten
`ergebnisse/C-skala__20260925_044543.json`, 97 min).

| Konfiguration | Winkel | S-0 (nachtrainiert > Identität) | instabil | M50 | M90 | G(1) |
|---|---|---|---|---|---|---|
| echt6 | 24 | H1 (+3,8 pp) | 0/10 | 2 | 12 | 0,45 |
| echt8 | 32 | H0 (−1,3 pp) | **2/10** | – | – | – |
| echt10 | 40 | H1 (+5,3 pp) | 0/10 | **1** | 10 | 0,63 |
| fest10 | 40 | H0 (−8,8 pp) | **5/10** | – | – | – |

- **S-1 H1**: bei gleichem Winkelanteil (1/8, 1/4) ist der Gewinnanteil für
  n = 10 höher als für n = 6 (+0,14 G, KI +0,05 … +0,23, p = 0,0098, Holm).
  Magiebedarf wächst langsamer als n·L — **aber** mit der Bildauflösung
  konfundiert.
- **S-3 HU (Vorbedingung)**: die saubere Trennung (fest10) fällt aus, weil
  *nachtrainiert* auf 5/10 Seeds kollabiert.
- Ursache: Die Ratensuche (2 Tuning-Seeds) wählt 0,32 direkt an der
  Stabilitätskante (1,28 kollabiert). → Lehre Stabilitätsabstand
  (FEHLERBUCH), Folge C-skala-v2 (Idee).

## 25.09.2026, 05:25 — I-034 (Q3, vorregistriert): **Gesamt-Magie-Budget kombiniert: 2,25 % der Hardware-T, ohne Verlust** ✔✔

MNIST 16×16, 10 frische Seeds 210–219. Kaskade: Stufe 0 = Clifford-Schaltung
**klassisch** (0 T, kein Laden) → k=2 → k=8 auf dem Quantenrechner mit
**Top-48-geladenen** Zuständen (Lade-T bildweise exakt gezählt), τ auf
Validierung nach Hardware-Kosten (Rohdaten `ergebnisse/I-034__20260925_052057.json`, 35 min).
Referenz „heute“: k=2 fest, dicht geladen = 7.698 T je Bild.

| Verfahren | Hardware-T je Bild | Anteil | Genauigkeit |
|---|---|---|---|
| k=2 dicht (heute) | 7.698 | 100 % | 91,63 % |
| k=8 dicht | 7.883 | 102 % | 92,88 % |
| Hybrid-Kaskade, dicht geladen (I-015-hybrid) | ~900 | ~12 % | ≈ k2 |
| k=8, Top-48-Laden, ohne Kaskade | 1.727 | 22 % | 92,19 % |
| **Kombi m=48** | **174** | **2,25 %** | **91,94 %** |
| Kombi m=32 | 113 | 1,47 % | 91,59 % |
| k0 klassisch allein | 0 | 0 % | 90,83 % |

- PK H1 (k2 exakt schlägt Identität, +2,2 pp, p = 0,002).
- **G-1 H1**: m=48: 2,25 % der Hardware-T, **+0,31 pp** (einseitiges KI
  [+0,12; +0,51]); m=32: 1,47 %, −0,04 pp (KI [−0,20; +0,12]). Beide
  nichtunterlegen (Holm-p < 0,001). Teuerster Seed 370 T (4,8 %); ohne den
  einen τ* = 0-Seed im Mittel 2,5 %.
- **G-2 H0 (TOST)**: genähertes Laden kostet *in der Kaskade* nichts
  (−0,03 pp gegen die Kaskade mit exaktem Laden, TOST p = 0,005) — obwohl es
  die feste Stufe k=2 −0,93 pp kostet. **Wechselwirkung +0,90 pp (p = 0,002)**:
  Die Kaskade schickt nur unsichere Bilder weiter und reicht sie bis k=8
  durch; dort gleicht die zusätzliche Magie den Ladefehler aus.
- Nur 7,8 % der Bilder erreichen den Quantenrechner.

**Das ist der Kernbefund des Instituts bisher:** Drei Magie-Einsparungen —
*welche Bilder* (klassische Clifford-Stufe), *wie viel Laden* (Top-m),
*wie viel Schaltung* (wenige Magiewinkel) — multiplizieren sich, statt sich
zu stören: **~44-mal weniger T-Gatter** je Bild als der naive Weg, bei
gleicher Genauigkeit. Einschränkung: MNIST, ein Datensatz, T-Zahl mit
ε = 10⁻³ je Rotation; Fashion-Replikation und echte Hardware stehen aus.

## 25.09.2026, 07:15 — C-skala-v2: Mit stabiler Referenz löst sich der Skalierungs-Hinweis auf

Gleiche Konfigurationen wie C-skala (echt6/echt10/fest10), aber Ratenwahl mit
Stabilitätsabstand (min über 3 Tuning-Seeds 930–932, Gitter Faktor 2, oberer
Nachbar darf nicht kollabieren), frische Mess-Seeds 220–229 (Rohdaten
`ergebnisse/C-skala-v2__20260925_071112.json`, 110 min).

| Konfiguration | Gewinn (nachtrainiert − Id.) | instabil | Kollaps | G(1) | M50 |
|---|---|---|---|---|---|
| echt6 (24 Winkel) | +3,2 pp (p=0,002) | 0/10 | keiner | 0,49 | 2 |
| echt10 (40 Winkel) | +5,1 pp (p=0,002) | 0/10 | keiner | 0,61 | 0 |
| fest10 (40 Winkel) | +4,9 pp (p=0,002) | 1/10 | keiner | 0,31 | 5 |

- S-0 dreimal **H1** — diesmal mit sauberer, stabiler Referenz (d_nach ≈ 0
  in allen drei Konfigurationen).
- **S-1 HU**: Der Hinweis aus C-skala v1 (+0,14 G für n=10 gegenüber n=6)
  **verschwindet** — mit stabiler Rate d = **−0,08 G** (n.s., KI
  [−0,16; 0,00], p_rechts = 0,93). Kein Beleg mehr für sublineares Wachstum.
- **S-3 HU**: Die saubere Trennung Breite/Auflösung (fest10 gegen echt6)
  bleibt unentschieden (+0,13 G, KI [−0,04; 0,29], n.s.) — diesmal aber mit
  **sauberer** Methode (fast keine Instabilität), nicht wegen eines
  Auflösungsproblems.

**Wichtige methodische Lehre (FEHLERBUCH):** Der v1-Befund (S-1 „H1“) war
ein **Artefakt der instabilen Lernrate**, kein echtes, nur ungenau
gemessenes Signal. Nach der Korrektur löst er sich auf, statt sich zu
bestätigen. **Die Magie-Skalierungsfrage bleibt offen** — weder bestätigt
noch widerlegt, mit jetzt solider Methode für einen künftigen, größer
angelegten Versuch (mehr Seeds, mehr Breiten).

## 25.09.2026, 07:55 — I-034-fashion: Kombination repliziert NICHT — wichtiges Gegenbeispiel

Fashion-MNIST, 10 frische Seeds 230–239, sonst identisch zu I-034 (Rohdaten
`ergebnisse/I-034-fashion__20260925_075400.json`, 43 min).

- PK H1 (k2 exakt schlägt Identität, +2,4 pp).
- **G-1 H0** (Abbruchkriterium erfüllt): kombi_m48 kostet **12,4 %** statt
  ≤5 % und ist **nachweislich schlechter** (−0,34 pp, KI [−0,65; −0,03]);
  kombi_m32 kostet 10,3 %, −0,64 pp. Beide über der 10-%-Schranke.
- G-2 **H1**: Genähertes Laden kostet in der Kaskade auf Fashion diesmal
  *doch* nachweisbar Genauigkeit (−0,34 pp, p = 0,008) — anders als auf
  MNIST, wo die Wechselwirkung den Verlust genau ausglich.
- Ursache (Einordnung, nicht Teil der Vorregistrierung): Top-48-Laden ist auf
  Fashion verlustreicher (I-033 war nur MNIST) und mehr Bilder brauchen
  überhaupt die Quantenstufe (31–36 % statt 8 %, weil Fashion mehr Magie
  braucht, siehe C-fashion). Ein einzelner Blick lohnt sich: `zwei_0_8_m48`
  (0→8 ohne k2-Zwischenstufe) erreicht 504 T = **6,5 %** bei 81,61 % (Referenz
  81,88 %) — nahe dran, in der Vorregistrierung aber nicht die Hauptkonfiguration.

**Einordnung: Das ist kein Rückschlag, sondern der Beleg, dass wir ehrlich
vorregistrieren.** Der MNIST-Befund (I-034, 2,25 %) gilt für MNIST, nicht
allgemein — die Kombination ist datensatzabhängig und muss je Aufgabe neu
kalibriert werden (Top-m-Größe, welche Stufen). Genau das relativiert auch
den "44-fach"-Satz im Kernsatz: er gilt nachgewiesen für MNIST, nicht per se.
