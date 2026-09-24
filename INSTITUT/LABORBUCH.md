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
