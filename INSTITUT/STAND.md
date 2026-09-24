# STAND des Instituts (automatisch, `python direktor.py status`)

Erzeugt 24.09.2026 18:14. Eine Seite -- mehr muss niemand lesen, um weiterzumachen.

| Punkt | Abteilung | Prio | Status | Frage | Letztes Urteil |
|---|---|---|---|---|---|
| C-clifford | Clifford | 1 | erledigt | Wie viele Nicht-Clifford-Winkel braucht die QIRE-Zelle auf 8x8-Ziffern (6 Qubits)? | C-1: HU: Aufloesung reicht nicht (SE der Differenz > 1 pp) -- kein Urteil; C-0: H1: Schaltung schlaegt Identitaet; C-2: H0: gelernte Rundung nicht besser als feste Clifford-Mischung; C-3: M* = 8 von 24 Winkeln |
| C-clifford-v2 | Clifford | 1 | erledigt | Wiederholung mit fairem Design (frischer Kopf, eigene Lernrate, Nichtunterlegenheits-KI) auf frischen Seeds 10-19. | C-1: HU: KI ueberdeckt die 2-pp-Grenze -- kein Urteil; C-0: H1: Schaltung schlaegt Identitaet; C-2: H0: gelernte Rundung nicht besser als feste Clifford-Mischung; C-3: M* = 0 (Stufen [0, 1, 2, 4, 8, 12]) |
| C-mnist | Clifford | 1 | erledigt | Dasselbe auf MNIST 16x16 (8 Qubits, 256 Dim.). Traegt die Schaltung dort mehr bei als die Identitaet? | C-1: H0: Rundung kostet nachweislich > 2 pp -- Magie wird gebraucht (siehe C-3); C-0: H1: Schaltung schlaegt Identitaet; C-2: HU: schlaegt nur Hadamard; C-3: M* = 1; M50 = 2, M90 = 12 (Stufen [0, 1, 2, 4, 8, 12], von 4*n Winkeln) |
| C-fashion | Clifford | 1 | offen | Replikation auf Fashion-MNIST 16x16 (schwerer, viermal hoehere Streuung laut Projekt A). | – |
| C-bewusst | Clifford | 1 | offen | Clifford-bewusstes Training (Winkel werden waehrend des Trainings zum Gitter gezogen) senkt M*. | Kriterien B-1..B-3 im Modul; freigeschaltet durch C-clifford-v2 C-1:HU. |
| C-skala | Clifford | 1 | wartet | Magie-Skalierung: waechst M* langsamer als die Winkelzahl n*L (n=6,8,10)? Dann wird die Schicht mit wachsender Breite fast multiplikationsfrei. | Modul noch nicht gebaut. |
| Z-zne | Theorie | 2 | verworfen_papier | Zero-Noise-Extrapolation ueber verschachtelte Bitbreiten (b, b-1) ersetzt ein Bit Genauigkeit. | Auf Papier widerlegt, siehe LABORBUCH 24.09. (Extrapolation verschiebt ungerade Codes weg vom Sollwert, MSE 7x schlechter). |
| V-kompass | Verschraenkung | 3 | erledigt | Qubit-Reihenfolge nach Mutual Information (wie Orbital-Ordnung in der DMRG-Quantenchemie) macht die Tensor-Train-Kompression trainierter Gewichte besser als zufaellige Reihenfolge bei gleicher Parameterzahl. | V-1: HU: besser, aber < 10 %; V-2: HU: kein Unterschied nachweisbar |
| B-paritaet | Auslese | 1 | idee | Paritaetsmerkmale <Z_S> niedriger Ordnung aus EINER Messbasis schlagen <Z_q> bei gleicher Kopfgroesse. | IDEEN.md Nr. 5 |
| V-fashion | Verschraenkung | 3 | wartet | Replikation des Verschraenkungs-Kompasses auf Fashion-MNIST. |  | Quantenstufe Q1 (Tensor-Train auf klassischen Gewichten = nur Analogie) -- niedrige Prioritaet (3), nicht gestrichen. |
| I-001 | Clifford | 1 | idee | Magie-Pruning: Winkel nach Verlustanstieg beim Runden ordnen, nur die Top-k bleiben Nicht-Clifford | Aus der Ideenschmiede. Abbruch: Wenn Verlust-Ranking G(2) nicht um >=0,1 ueber Abstands-Ranking hebt (gepaart, 10 Seeds). Test: Ziffern, Winkel aus C-clifford-v2-Rohdaten, zwei Rankings, k in {1,2,4}: ~10 min CPU. |
| I-002 | Clifford | 1 | idee | Stabilisator-Renyi-Entropie der gelernten Schicht sagt vorher, wie viel Magie die Aufgabe braucht (M50) -- ohne Rundungsversuche | Aus der Ideenschmiede. Abbruch: Rangkorrelation SRE vs. gemessenem Rundungsverlust ueber Seeds und Datensaetze |rho| < 0,3. Test: SRE aus gespeicherten Winkeln der C-Laeufe berechnen, gegen G(k) korrelieren: <5 min, kein Training. |
| I-003 | Clifford | 1 | idee | Magie-Konzentration als Kennzahl einer Aufgabe: Anteil des Nutzens in den Top-k Nicht-Clifford-Winkeln | Aus der Ideenschmiede. Abbruch: Wenn M50 auf allen drei Datensaetzen gleich ist und sich mit zufaelligen Etiketten nicht aendert -> Architektureigenschaft, keine Aufgabeneigenschaft. Test: laeuft bereits (C-mnist), dann C-fashion; zusaetzlich ein Lauf mit zufaelligen Etiketten als Kontrolle. |
| I-004 | Clifford | 1 | idee | Clifford-Geruest + k trainierbare Nicht-Clifford-Winkel von Anfang an (statt nachtraeglich runden) | Aus der Ideenschmiede. Abbruch: Nicht besser als nachtraegliches Runden mit gleichem k (gepaart). Test: Ziffern, k=2 und 4, Positionen aus Magie-Pruning: ~20 min. |
| I-005 | Clifford | 1 | idee | Nur diskrete Clifford-Wahl trainiert (2 Bit je Gatter), Magie kommt allein aus den Daten | Aus der Ideenschmiede. Abbruch: Beste gefundene Clifford nicht besser als Median von 100 zufaelligen Clifford-Schaltungen. Test: 100 zufaellige + simulated annealing ueber 4^24 Clifford-Wahlen, nur Kopf trainieren (konvex): ~30 min. |
| C-hardware | Clifford | 1 | idee | Q3-Bruecke: die gerundete Schaltung mit k Magiewinkeln mit Qiskit nach Clifford+T uebersetzen (T-Zahl zaehlen) und auf IBM-Hardware laufen lassen. Uebersetzt M50 direkt in T-Gatter. | Bausteine in Projekt A: quantum_sim/qiskit_bridge.py, run_on_hardware.py. IBM-Token vorher neu erzeugen (HANDOFF §7). |

**Ideenschmiede:** roh 6 -> geprueft 0 -> neu 0 -> weitergeleitet 5 -> verworfen 0

Naechster offener Punkt: C-fashion
