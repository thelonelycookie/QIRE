"""EXPLORATIV (kein Befund): Liegt der Rundungsverlust in C-clifford am
geerbten, uebersaettigten Kopf? Gleiche gerundete Winkel (aus den Rohdaten),
aber frischer Kopf und eigene Lernrate aus dem Gitter von hadamard_fest.
Nur die Winkel aus dem Lauf werden wiederverwendet, kein Nachtraining der
Schaltung."""
import glob
import json

import torch

from abteilungen.clifford import LAGEN, N_QUBITS, QireZelle, _magie_reihenfolge, runde_clifford
from labor.gemeinsam import genauigkeit, parallel, speichere, trainiere, ziffern


def _eins(a):
    s, th, k, lr, schritte = a
    th = torch.tensor(th).reshape(LAGEN, N_QUBITS)
    xtr, ytr, xte, yte = ziffern(s)
    torch.manual_seed(s)
    m = QireZelle()                               # frischer Kopf
    frei = torch.zeros(LAGEN * N_QUBITS)
    frei[_magie_reihenfolge(th)[:k]] = 1
    m.fixiere(runde_clifford(th), frei.reshape(LAGEN, N_QUBITS))
    with torch.no_grad():
        m.theta.copy_(torch.where(m.frei.bool(), th, runde_clifford(th)))
    trainiere(m, xtr, ytr, schritte, lr, seed=s)
    return {"seed": s, "k": k, "lr": lr, "acc": genauigkeit(m(xte), yte)}


if __name__ == "__main__":
    e = json.load(open(sorted(glob.glob("ergebnisse/C-clifford__*.json"))[-1]))
    aufg = [(z["seed"], z["winkel_voll"], k, lr, e["schritte"]) for z in e["zeilen"]
            for k in (0, 4, 8) for lr in (0.02, 0.08)]
    r = parallel(_eins, aufg)
    speichere("C-diagnose-kopf", {"explorativ": True, "zeilen": r})
    import numpy as np
    for k in (0, 4, 8):
        for lr in (0.02, 0.08):
            v = [x["acc"] for x in r if x["k"] == k and x["lr"] == lr]
            alt = np.mean([z[f"clifford_k{k}"] for z in e["zeilen"]])
            print(f"k={k} lr={lr}: frischer Kopf {np.mean(v):.4f}  (geerbter Kopf {alt:.4f})")
