from __future__ import annotations

"""RESEARCH ID: T-001 -- Die Theorie hinter den Messungen, gegen die Messungen geprueft.

WARUM DIESE DATEI EXISTIERT
Das Projekt hat sieben Messungen und ein Prinzip: "In einem 2^n-dimensionalen
Hilbertraum sind zufaellige Zustaende fast immer orthogonal." Dieses Prinzip ist
aus den Daten **abgelesen** -- die mittlere Kernaehnlichkeit halbiert sich je
Qubit -- aber nirgends **hergeleitet**. Eine abgelesene Regelmaessigkeit ist eine
Beobachtung; eine hergeleitete ist eine Vorhersage, und nur die kann falsch sein.

Hier wird nachgerechnet, was die Theorie sagt, und gegen die eigenen Zahlen
gehalten. Drei Teile, alle ohne neue Rechenzeit -- es sind geschlossene Formeln
und die bereits gemessenen Werte.

Run: python -m quantum_sim.theory
"""

import math
from typing import Dict, List, Tuple

import numpy as np

# Gemessene mittlere ausserdiagonale Kernaehnlichkeit (K-Serie, 3. August).
# ACHTUNG: fuer die K-Serie existieren keine gespeicherten Rohdaten -- diese
# Werte stammen aus dem Tagebuch und sind nicht per `q_audit` nachpruefbar.
# Das ist eine Schwaeche, die hier ausdruecklich mitgetragen wird.
MEASURED_SIMILARITY: Dict[int, float] = {2: 0.349, 3: 0.162, 4: 0.082, 6: 0.024}


# ---------------------------------------------------------------------------
# Teil 1 -- Das Konzentrationsprinzip, hergeleitet statt abgelesen
# ---------------------------------------------------------------------------

def haar_expected_overlap(n_qubits: int) -> float:
    """E[|<psi|phi>|^2] fuer zwei unabhaengige Haar-zufaellige Zustaende.

    Herleitung: Sei |psi> fest und |phi> Haar-verteilt auf der Einheitssphaere
    des C^d mit d = 2^n. Dann ist |<psi|phi>|^2 der Betrag der Projektion auf
    eine feste Richtung. Aus Symmetriegruenden ist der Erwartungswert dieser
    Projektion auf JEDE der d orthogonalen Richtungen gleich, und sie summieren
    sich zu 1 (Normierung). Also:

        E[|<psi|phi>|^2] = 1/d = 2^(-n)

    Das ist die Vorhersage: Die Aehnlichkeit halbiert sich je Qubit, weil ein
    zusaetzliches Qubit die Dimension verdoppelt -- nicht wegen irgendeiner
    Eigenschaft der Merkmalsabbildung, sondern aus reiner Geometrie.
    """
    return 1.0 / (2 ** n_qubits)


def concentration_fit() -> Dict[str, float]:
    """Gemessene Konzentration gegen das Haar-Gesetz.

    Angepasst wird  log2(similarity) = a + b*n.  Die Theorie sagt b = -1
    (Halbierung je Qubit) und a = 0 (Vorfaktor 1). Ein Vorfaktor groesser als 1
    heisst: die echten Daten sind sich AEHNLICHER als Zufallszustaende -- es ist
    also noch Struktur uebrig. Ein Exponent nahe -1 heisst dagegen: die Struktur
    verschwindet genauso schnell, wie die Geometrie es erzwingt.
    """
    ns = np.array(sorted(MEASURED_SIMILARITY), dtype=float)
    ys = np.array([math.log2(MEASURED_SIMILARITY[int(n)]) for n in ns])
    b, a = np.polyfit(ns, ys, 1)
    pred = a + b * ns
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {"exponent": float(b), "log2_prefactor": float(a),
            "prefactor": float(2 ** a), "r2": 1.0 - ss_res / ss_tot}


def report_concentration() -> Dict[str, float]:
    fit = concentration_fit()
    print("=" * 90)
    print("TEIL 1 -- Das Konzentrationsprinzip: hergeleitet und geprueft")
    print("=" * 90)
    print()
    print("  Vorhersage aus der Geometrie:  E[|<psi|phi>|^2] = 1/2^n")
    print("  (Herleitung im Docstring von `haar_expected_overlap`)")
    print()
    head = (f'  {"Qubits":>6} {"Dim d":>7} {"gemessen":>10} {"Haar 1/d":>10} '
            f'{"Verhaeltnis":>12}')
    print(head); print("  " + "-" * (len(head) - 2))
    ratios = []
    for n in sorted(MEASURED_SIMILARITY):
        m, h = MEASURED_SIMILARITY[n], haar_expected_overlap(n)
        ratios.append(m / h)
        print(f'  {n:>6} {2 ** n:>7} {m:>10.4f} {h:>10.4f} {m / h:>12.2f}')
    print()
    print(f"  Anpassung  log2(Aehnlichkeit) = a + b*n :")
    print(f"    Exponent b   = {fit['exponent']:+.3f}   (Theorie: -1,000)")
    print(f"    Vorfaktor    = {fit['prefactor']:.2f}    (Theorie: 1,00)")
    print(f"    R^2          = {fit['r2']:.4f}")
    print()
    dev = abs(fit["exponent"] + 1.0) * 100
    print(f"  **Der gemessene Exponent weicht um {dev:.1f} % von der Vorhersage ab.**")
    print()
    print("  Was das bedeutet -- und es ist schaerfer als 'die Aehnlichkeit faellt':")
    print("  Die Zustaende, die die Merkmalsabbildung erzeugt, werden mit")
    print("  wachsender Qubitzahl GENAUSO SCHNELL orthogonal wie rein zufaellige")
    print("  Zustaende. Die Abbildung erhaelt also keine nennenswerte Struktur der")
    print("  Daten -- sie verteilt sie praktisch so, wie der Zufall es taete.")
    print()
    print(f"  Der Vorfaktor {fit['prefactor']:.2f} ist der einzige Rest von Struktur:")
    print("  echte Daten sind sich um diesen Faktor aehnlicher als Zufall. Er ist")
    print("  konstant in n -- die Struktur waechst also nicht mit, waehrend der")
    print("  Raum exponentiell waechst. Genau daran scheitert das Lernen.")
    print()
    print("  FALSIFIZIERBARE VORHERSAGE fuer einen kuenftigen Lauf:")
    for n in (5, 8, 10):
        p = fit["prefactor"] * 2 ** (fit["exponent"] * n)
        print(f"    {n:>2} Qubits -> mittlere Aehnlichkeit {p:.5f}  "
              f"(Haar: {haar_expected_overlap(n):.5f})")
    print("  Weicht eine Messung davon deutlich ab, ist das Prinzip falsch.")
    return fit


# ---------------------------------------------------------------------------
# Teil 2 -- Was die Gewichtsteilung mathematisch bedeutet
# ---------------------------------------------------------------------------

def parameter_geometry(n_qubits: int, layers: int) -> Dict[str, int]:
    """Dimensionen der drei Parametrisierungen einer Drehung im C^(2^n).

    * volle Gruppe: SU(d) hat d^2 - 1 reelle Parameter, SO(d) hat d(d-1)/2.
      Als Vergleichsgroesse fuer eine dichte Schicht dient d^2 (freie Matrix).
    * ungeteiltes Butterfly: je Lage n Ebenen mit je d/2 unabhaengigen
      2x2-Drehungen, jede mit 2 Winkeln  ->  2 * n * d/2 * L = n*d*L
    * geteiltes Butterfly (die Quantenschaltung): je Lage n Ebenen mit je EINEM
      gemeinsamen Gatter, 2 Winkel  ->  2*n*L
    """
    d = 2 ** n_qubits
    return {"dim": d,
            "dense": d * d,
            "so_group": d * (d - 1) // 2,
            "untied": n_qubits * d * layers,
            "tied": 2 * n_qubits * layers}


def report_geometry(layers: int = 5) -> None:
    print()
    print("=" * 90)
    print("TEIL 2 -- Was die Gewichtsteilung mathematisch ist")
    print("=" * 90)
    print()
    print(f"  Bei {layers} Lagen:")
    head = (f'  {"Qubits":>6} {"Dim d":>7} {"geteilt":>9} {"ungeteilt":>11} '
            f'{"SO(d)":>13} {"Anteil an SO(d)":>17}')
    print(head); print("  " + "-" * (len(head) - 2))
    for n in (4, 6, 8, 10, 12):
        g = parameter_geometry(n, layers)
        frac = g["tied"] / g["so_group"] * 100
        print(f'  {n:>6} {g["dim"]:>7,} {g["tied"]:>9,} {g["untied"]:>11,} '
              f'{g["so_group"]:>13,} {frac:>16.4f} %')
    print()
    print("  Die entscheidende Einsicht -- und sie ist NICHT 'kleiner ist besser':")
    print()
    print("  Die geteilte Schaltung ist eine Untermannigfaltigkeit der Dimension")
    print("  2*n*L in einer Gruppe der Dimension d(d-1)/2. Bei 8 Qubits sind das")
    print("  80 Richtungen in einem Raum von 32.640 -- ein Viertelprozent.")
    print()
    print("  ABER: Der 'hardware-efficient ansatz' mit CNOT-Verschraenkung ist")
    print("  fuer L -> unendlich UNIVERSELL, erreicht also jede Transformation.")
    print("  Die Teilung schraenkt damit nicht ein, WAS erreichbar ist, sondern")
    print("  nur, WIE VIELE Parameter der Weg dorthin kostet.")
    print()
    print("  Daraus folgt eine pruefbare Aussage: Die Teilung darf bei genuegend")
    print("  Lagen nichts kosten. Kostet sie bei WENIGEN Lagen etwas, ist das ein")
    print("  Aussage ueber die noetige Tiefe, nicht ueber die Ausdruckskraft.")
    print()
    print("  Und hier schliesst sich ein Kreis zu B-014: dort wurde gemessen, dass")
    print("  die Aufgabe nur rund sieben Richtungen braucht. Eine 80-dimensionale")
    print("  Untermannigfaltigkeit reicht dafuer mit grossem Abstand -- was")
    print("  erklaert, warum Q-003-replication keinen Unterschied zwischen 16 und")
    print("  80 Parametern fand. Nicht weil die Schaltung schlecht waere, sondern")
    print("  weil die Aufgabe die Kapazitaet nicht ausschoepft.")


# ---------------------------------------------------------------------------
# Teil 3 -- Wo die Theorie widerspricht, was gemessen wurde
# ---------------------------------------------------------------------------

def report_tensions() -> None:
    print()
    print("=" * 90)
    print("TEIL 3 -- Zwei Stellen, an denen Theorie und Messung sich reiben")
    print("=" * 90)
    print()
    print("  (1) DIE QUADRATUR SOLLTE NICHT SO VIEL BRINGEN.")
    print("      Gemessen: 86,5 % bzw. 91,5 % des Vorteils entfallen auf die")
    print("      quadratische Auslese. Eine Quadratur ist aber nur eine")
    print("      Nichtlinearitaet unter vielen -- warum schlaegt sie Tanh so")
    print("      deutlich? Vermutung: weil sie mit der L2-Normierung davor ein")
    print("      Paar bildet. Normieren + Quadrieren ergibt eine")
    print("      Wahrscheinlichkeitsverteilung, also eine Groesse mit fester")
    print("      Summe. Das ist eine STRUKTURELLE Einschraenkung, keine blosse")
    print("      Nichtlinearitaet -- und Einschraenkungen sind genau das, was")
    print("      laut dem eigenen Pruefmassstab etwas beitragen kann.")
    print("      PRUEFBAR: Tanh MIT vorheriger Normierung gegen Quadratur mit")
    print("      Normierung. Bleibt der Abstand, liegt es an der Quadratur;")
    print("      verschwindet er, lag es an der Normierung.")
    print()
    print("  (2) DIE VERSCHRAENKUNG SOLLTE NICHT NICHTS BRINGEN.")
    print("      Gemessen: ohne CNOT gleichwertig (TOST, +-0,005). Aber ohne")
    print("      CNOT ist der Zustand ein PRODUKTZUSTAND -- dann faktorisiert")
    print("      <Z_q> vollstaendig, und die Schaltung berechnet fuer jedes Qubit")
    print("      unabhaengig eine Funktion von 8 der 256 Amplituden. Dass das")
    print("      genauso gut ist, heisst: die Aufgabe braucht keine Korrelation")
    print("      zwischen den Ausgabekanaelen.")
    print("      Das ist KEIN Argument gegen Verschraenkung an sich, sondern")
    print("      eines dafuer, dass MNIST sie nicht verlangt. Eine Aufgabe mit")
    print("      erzwungener Kanalkorrelation (z. B. Paritaet ueber Merkmals-")
    print("      gruppen) waere der faire Test -- und der fehlt.")
    print()
    print("  Beide Punkte sind Vorhersagen, keine Ergebnisse. Sie gehoeren als")
    print("  offene Fragen in die Arbeit, nicht als Befunde.")


def main() -> None:
    fit = report_concentration()
    report_geometry()
    report_tensions()
    print()
    print("=" * 90)
    print("ZUSAMMENFASSUNG")
    print("=" * 90)
    print(f"  * Das Konzentrationsprinzip ist nicht nur beobachtet, sondern")
    print(f"    hergeleitet: 1/2^n aus reiner Geometrie. Die Messung trifft den")
    print(f"    Exponenten auf {abs(fit['exponent'] + 1) * 100:.1f} % genau.")
    print("  * Die Gewichtsteilung schraenkt nicht die Ausdruckskraft ein,")
    print("    sondern die Parameterzahl des Weges -- die Schaltung bleibt")
    print("    universell. Damit ist 'kostet die Teilung etwas' eine Frage nach")
    print("    der noetigen TIEFE, nicht nach Faehigkeit.")
    print("  * Zwei Messergebnisse widersprechen der naiven Erwartung und haben")
    print("    je einen konkreten Nachtest, der sie aufloesen wuerde.")
    print()
    print("  Fuer die Arbeit: Das ist der Unterschied zwischen 'ich habe")
    print("  gemessen, dass es faellt' und 'ich kann ausrechnen, wie schnell,")
    print("  und die Messung bestaetigt es'.")


if __name__ == "__main__":
    main()
