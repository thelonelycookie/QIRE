# STAND des Instituts (automatisch, `python direktor.py status`)

Erzeugt 24.09.2026 16:28. Eine Seite -- mehr muss niemand lesen, um weiterzumachen.

| Punkt | Abteilung | Status | Frage | Letztes Urteil |
|---|---|---|---|---|
| C-clifford | Clifford | erledigt | Wie viele Nicht-Clifford-Winkel braucht die QIRE-Zelle auf 8x8-Ziffern (6 Qubits)? | C-1: HU: Aufloesung reicht nicht (SE der Differenz > 1 pp) -- kein Urteil; C-0: H1: Schaltung schlaegt Identitaet; C-2: H0: gelernte Rundung nicht besser als feste Clifford-Mischung; C-3: M* = 8 von 24 Winkeln |
| C-clifford-v2 | Clifford | erledigt | Wiederholung mit fairem Design (frischer Kopf, eigene Lernrate, Nichtunterlegenheits-KI) auf frischen Seeds 10-19. | C-1: HU: KI ueberdeckt die 2-pp-Grenze -- kein Urteil; C-0: H1: Schaltung schlaegt Identitaet; C-2: H0: gelernte Rundung nicht besser als feste Clifford-Mischung; C-3: M* = 0 (Stufen [0, 1, 2, 4, 8, 12]) |
| C-mnist | Clifford | offen | Dasselbe auf MNIST 16x16 (8 Qubits, 256 Dim.). Traegt die Schaltung dort mehr bei als die Identitaet? | freigeschaltet durch C-clifford C-2:H0; laeuft mit Design v2 (Aenderung vor jeder MNIST-Zahl, LABORBUCH). |
| C-fashion | Clifford | wartet | Replikation auf Fashion-MNIST 16x16 (schwerer, viermal hoehere Streuung laut Projekt A). | – |
| C-bewusst | Clifford | offen | Clifford-bewusstes Training (Winkel werden waehrend des Trainings zum Gitter gezogen) senkt M*. | Modul noch nicht gebaut; wird gebaut, sobald freigeschaltet. |
| C-skala | Clifford | wartet | Magie-Skalierung: waechst M* langsamer als die Winkelzahl n*L (n=6,8,10)? Dann wird die Schicht mit wachsender Breite fast multiplikationsfrei. | Modul noch nicht gebaut. |
| Z-zne | Theorie | verworfen_papier | Zero-Noise-Extrapolation ueber verschachtelte Bitbreiten (b, b-1) ersetzt ein Bit Genauigkeit. | Auf Papier widerlegt, siehe LABORBUCH 24.09. (Extrapolation verschiebt ungerade Codes weg vom Sollwert, MSE 7x schlechter). |
| V-kompass | Verschraenkung | offen | Qubit-Reihenfolge nach Mutual Information (wie Orbital-Ordnung in der DMRG-Quantenchemie) macht die Tensor-Train-Kompression trainierter Gewichte besser als zufaellige Reihenfolge bei gleicher Parameterzahl. | Kriterien in abteilungen/kompass.py (V-1..V-3). |
| B-paritaet | Auslese | idee | Paritaetsmerkmale <Z_S> niedriger Ordnung aus EINER Messbasis schlagen <Z_q> bei gleicher Kopfgroesse. | IDEEN.md Nr. 5 |
| V-fashion | Verschraenkung | wartet | Replikation des Verschraenkungs-Kompasses auf Fashion-MNIST. | – |

Naechster offener Punkt: C-mnist
