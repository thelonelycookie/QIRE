from __future__ import annotations

"""RESEARCH ID: T-001 -- Butterfly-Schicht mit den verhaltenserhaltenden
Umschreibregeln R1, R5 und R6 aus `ENTWURF_T_SERIE_KERNEL.md`.

WAS HIER ANDERS IST ALS IN `butterfly_layer.py`
Dieselbe Funktion, andere Ausfuehrung. Drei Aenderungen, alle rein strukturell:

  R1  DIE CNOT-PERMUTATIONEN WERDEN ZUSAMMENGEFASST.
      `butterfly_layer` wendet die n Ringpermutationen einzeln an (Zeile 175),
      jede als vollstaendiger `gather` ueber den Zustand. CNOT ist aber reine
      Indexvertauschung, und n Vertauschungen komponieren zu genau EINER. Die
      zusammengesetzte Permutation wird einmal beim Bau berechnet.
      Bei 8 Qubits und 5 Lagen: 40 Gathers -> 5.

  R5  DIE GATTER WERDEN GESAMMELT BERECHNET.
      `butterfly_layer._gate` steht in der Vorwaertsschleife und wertet fuer
      jedes (Lage, Qubit) einzeln `cos`, `sin` und zwei komplexe `exp` aus --
      bei jedem Vorwaertslauf neu, also L*n = 40 transzendente Aufrufe. Hier
      werden alle Gatter in einem vektorisierten Aufruf gebaut. Sind die Winkel
      eingefroren, wird das Ergebnis zusaetzlich zwischengespeichert und gar
      nicht mehr neu gerechnet.

  R6  DIE AUSLESE BRAUCHT EINEN DURCHLAUF STATT n.
      `butterfly_layer` berechnet die n Erwartungswerte in einer Schleife, jeder
      mit eigenem `reshape` und zwei Summen -- n vollstaendige Durchlaeufe ueber
      den Wahrscheinlichkeitsvektor. <Z_q> ist aber linear in den
      Betragsquadraten: mit einer festen Vorzeichenmatrix S (n x d), S[q,i] =
      +1 falls Bit q von i null ist, sonst -1, ist die ganze Auslese ein
      einziges Matrixprodukt prob @ S^T. Ein Durchlauf, dazu BLAS statt
      Python-Schleife.

WAS HIER NICHT PASSIERT
Die L*n Gatteranwendungen selbst bleiben einzelne Tensoroperationen. Sie sind
der Hauptposten, und sie zu fusionieren braucht einen echten Kernel, der den
Zustand ueber alle Stufen im Cache haelt -- das geht in reinem PyTorch nicht
auszudruecken. Diese Datei ist deshalb die **untere Schranke** dessen, was ohne
Compiler erreichbar ist, und genau als solche zu messen.

Erwartete Wirkung, vorher notiert: von rund 89 Durchlaeufen ueber den Zustand
auf rund 46. Also etwa Faktor zwei, nicht die drei Groessenordnungen. Wer mehr
erwartet, hat die Gatteranwendungen vergessen.

VERIFIKATION VOR JEDER ZEITMESSUNG.
`python -m quantum_sim.butterfly_fast` prueft gegen `ButterflyQuantumLayer`
ueber dieselben Konfigurationen, mit denen jene gegen den Simulator geprueft
wurde. Ohne bestandene Verifikation ist jede Zeitmessung wertlos -- beim ersten
Butterfly-Anlauf ist genau diese Pruefung durchgefallen (Ry*Rz statt Rz*Ry).
"""

import math
from typing import Optional

import torch
from torch import nn

from .butterfly_layer import ButterflyQuantumLayer, apply_1q, cnot_permutation


def komponiere(perms) -> torch.Tensor:
    """R1: mehrere nacheinander angewandte Indexpermutationen zu einer.

    Ist s1 = s0[p1] und s2 = s1[p2], dann s2[i] = s0[p1[p2[i]]] -- die
    Zusammensetzung ist also p1[p2] und nicht p2[p1]. Die Reihenfolge ist nicht
    beliebig; bei Vertauschung faellt die Verifikation durch.
    """
    acc = perms[0]
    for p in perms[1:]:
        acc = acc[p]
    return acc


def vorzeichenmatrix(n: int) -> torch.Tensor:
    """R6: S[q, i] = +1 falls Bit q von i null ist, sonst -1.

    Qubit 0 ist das hoechstwertige Bit -- dieselbe Konvention wie in
    `simulator.py` und `apply_1q`.
    """
    idx = torch.arange(2 ** n)
    bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)])
    return 1.0 - 2.0 * bits.to(torch.float32)


class ButterflyFast(nn.Module):
    """Funktionsgleich zu `ButterflyQuantumLayer(num_qubits, num_layers,
    entangle, tie_weights=True)`, mit R1/R5/R6 angewandt.

    Nur die geteilte Variante ist umgesetzt -- sie ist die quantenabgeleitete
    und die einzige, die in der T-Serie gebraucht wird. Fuer `tie_weights=False`
    bleibt `ButterflyQuantumLayer` zustaendig.
    """

    def __init__(self, num_qubits: int, num_layers: int, entangle: bool = True):
        super().__init__()
        self.n, self.L, self.entangle = num_qubits, num_layers, entangle
        self.theta = nn.Parameter(torch.rand(num_layers, num_qubits, 2) * 2 * math.pi)
        if entangle and num_qubits > 1:
            perms = [cnot_permutation(num_qubits, q, (q + 1) % num_qubits)
                     for q in range(num_qubits)]
            self.register_buffer("perm", komponiere(perms))       # R1
        else:
            self.perm = None
        self.register_buffer("vz", vorzeichenmatrix(num_qubits))   # R6
        self._gatter_cache: Optional[torch.Tensor] = None

    @property
    def input_dim(self) -> int:
        return 2 ** self.n

    def num_circuit_parameters(self) -> int:
        return self.theta.numel()

    def _gatter(self) -> torch.Tensor:
        """R5: alle L*n Gatter in einem vektorisierten Aufruf.

        Identisch zu `ButterflyQuantumLayer._gate`, nur ueber alle (Lage, Qubit)
        auf einmal: Rz(b) @ Ry(a), nicht Ry @ Rz -- die beiden kommutieren
        nicht.
        """
        if self._gatter_cache is not None:
            return self._gatter_cache
        a, b = self.theta[..., 0], self.theta[..., 1]              # (L, n)
        ca, sa = torch.cos(a / 2), torch.sin(a / 2)
        eb = torch.exp(-0.5j * b.to(torch.complex64))
        ebc = torch.exp(0.5j * b.to(torch.complex64))
        ca, sa = ca.to(torch.complex64), sa.to(torch.complex64)
        g = torch.stack([torch.stack([ca * eb, -sa * eb], dim=-1),
                         torch.stack([sa * ebc, ca * ebc], dim=-1)], dim=-2)
        if not self.theta.requires_grad:
            self._gatter_cache = g          # eingefroren -> nie wieder rechnen
        return g

    def train(self, mode: bool = True):
        self._gatter_cache = None           # Winkel koennten sich geaendert haben
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        state = x.to(torch.complex64)
        state = state / state.norm(dim=1, keepdim=True).clamp_min(1e-12)
        g = self._gatter()
        for layer in range(self.L):
            for q in range(n):
                state = apply_1q(state, g[layer, q], q, n)
            if self.perm is not None:
                state = state[:, self.perm]                        # R1: einer statt n
        prob = state.real ** 2 + state.imag ** 2
        return prob @ self.vz.t()                                  # R6: ein Durchlauf


def verifiziere(tol: float = 1e-6) -> bool:
    """Gegen `ButterflyQuantumLayer` ueber dieselben Konfigurationen, mit denen
    diese gegen `QuantumLayer` geprueft wurde."""
    print("=" * 94)
    print("T-001 VERIFIKATION -- ButterflyFast gegen ButterflyQuantumLayer")
    print(f"Schranke {tol:g}. Ohne bestandene Pruefung keine Zeitmessung.")
    print("=" * 94)
    head = f'  {"Qubits":>7} {"Lagen":>6} {"entangle":>9} {"Batch":>6} {"max. Abw.":>12} {"":>6}'
    print(head); print("  " + "-" * (len(head) - 2))
    schlimmste, alle_ok = 0.0, True
    for n in (3, 4, 6, 8, 10):
        for L in (1, 2, 5):
            for ent in (True, False):
                torch.manual_seed(1000 + n * 100 + L * 10 + int(ent))
                ref = ButterflyQuantumLayer(n, L, entangle=ent, tie_weights=True)
                schnell = ButterflyFast(n, L, entangle=ent)
                with torch.no_grad():
                    schnell.theta.copy_(ref.theta)
                x = torch.randn(7, 2 ** n)
                with torch.no_grad():
                    d = float((ref(x) - schnell(x)).abs().max())
                ok = d <= tol
                alle_ok &= ok
                schlimmste = max(schlimmste, d)
                print(f'  {n:>7} {L:>6} {str(ent):>9} {7:>6} {d:>12.3e} '
                      f'{"OK" if ok else "FEHLER":>6}')
    print()
    print("=" * 94)
    if alle_ok:
        print(f"VERIFIKATION BESTANDEN -- groesste Abweichung {schlimmste:.3e}")
        print("Die Umschreibungen R1, R5 und R6 sind verhaltenserhaltend.")
    else:
        print(f"VERIFIKATION DURCHGEFALLEN -- groesste Abweichung {schlimmste:.3e}")
        print("Keine Zeitmessung durchfuehren. Erst die Abweichung erklaeren.")
    print("=" * 94)
    return alle_ok


def gradientenpruefung(tol: float = 1e-4) -> bool:
    """Zusaetzlich: stimmen auch die Gradienten ueberein? Sonst waere die
    Schicht zwar vorwaerts gleich, aber nicht als Ersatz im Training
    brauchbar."""
    torch.manual_seed(77)
    n, L = 6, 3
    ref = ButterflyQuantumLayer(n, L, entangle=True, tie_weights=True)
    schnell = ButterflyFast(n, L, entangle=True)
    with torch.no_grad():
        schnell.theta.copy_(ref.theta)
    x = torch.randn(5, 2 ** n)
    ref(x).sum().backward()
    schnell(x).sum().backward()
    d = float((ref.theta.grad - schnell.theta.grad).abs().max())
    ok = d <= tol
    print(f"\n  Gradientenpruefung: groesste Abweichung {d:.3e} -> "
          f"{'OK' if ok else 'FEHLER'}")
    return ok


if __name__ == "__main__":
    bestanden = verifiziere()
    bestanden &= gradientenpruefung()
    raise SystemExit(0 if bestanden else 1)
