from __future__ import annotations

"""Die Quantenarchitektur als eigenstaendige klassische Schicht.

WAS DAS HIER IST -- und was es NICHT ist
Dies ist **kein Simulator**. `torch_simulator.py` schreibt einen Quantenzustand
fort, um Quantenmechanik nachzurechnen. Diese Datei nimmt die *Architektur*, die
dabei herauskommt, und setzt sie als klassisches Rechenverfahren um -- mit dem
Ziel, sie auf gewoehnlicher Hardware effizient auszufuehren.

Die Unterscheidung ist der Punkt: Ein Simulator kann per Konstruktion keinen
Vorteil zeigen, er zahlt die exponentiellen Kosten des Zustandsraums. Eine
Architektur dagegen kann man auf jeder Hardware umsetzen, und dann ist die
Frage, was sie kostet.

WARUM DIE SCHALTUNG EIN BUTTERFLY-NETZ IST
Ein Ein-Qubit-Gatter auf Qubit q verknuepft genau die Amplitudenpaare, deren
Indizes sich in Bit q unterscheiden. Ueber alle n Qubits hinweg ist das exakt
die Schmetterlingsstruktur der schnellen Fouriertransformation: n Ebenen mit je
d/2 unabhaengigen 2x2-Operationen, also **O(d log d)** statt der O(d^2) einer
dichten Matrix. Bei d=256 sind das rund 2.000 statt 65.000 Multiplikationen.

WAS DARAN NICHT KLASSISCH IST -- der eigentliche Befund
Ein gewoehnliches Butterfly-Netz hat pro Paar einen eigenen Winkel, also
(d/2)*n*L Parameter. Die Quantenschaltung hat **2*n*L**, weil derselbe Winkel
auf ALLE d/2 Paare einer Ebene wirkt. Bei n=8, L=5: 80 statt 5.120 Parameter,
und gegenueber einer vollen orthogonalen 256x256-Matrix (32.640 freie
Parameter) ein Faktor 408.

Diese Gewichtsteilung ueber die Qubit-Ebene ist die Struktur, auf die der
klassische Weg nicht fuehrt -- sie ergibt sich daraus, dass ein Qubit-Gatter
physikalisch auf *ein Qubit* wirkt, gleichgueltig wie gross der Zustandsraum
ist. Genau das ist der quantenabgeleitete Anteil der Architektur.

VERIFIKATION VOR JEDER BEHAUPTUNG
`verify()` prueft, dass diese Schicht **dieselbe Funktion** berechnet wie
`quantum_layer.QuantumLayer` -- bei identischen Winkeln, bis auf
Maschinengenauigkeit. Ohne diesen Nachweis waere jede Geschwindigkeitsaussage
wertlos, weil sie sich auf eine andere Funktion beziehen koennte.

`benchmark()` misst danach die Zeit gegen den Simulator und gegen eine dichte
Schicht gleicher Breite.

Run: python -m quantum_sim.butterfly_layer
"""

import math
import time
from typing import Dict, List, Tuple

import torch
from torch import nn

from .quantum_layer import QuantumLayer


def apply_1q_untied(state: torch.Tensor, u: torch.Tensor, q: int,
                    n: int) -> torch.Tensor:
    """Wie `apply_1q`, aber mit einem EIGENEN 2x2-Gatter je Amplitudenpaar.

    `u` hat die Form (2**q, 2**(n-1-q), 2, 2) -- also eine Matrix fuer jedes der
    d/2 Paare dieser Ebene. Das ist das gewoehnliche Butterfly-Netz, auf das der
    klassische Weg fuehrt; die Quantenschaltung teilt sich stattdessen EIN
    Gatter ueber die ganze Ebene.
    """
    batch = state.shape[0]
    left, right = 2 ** q, 2 ** (n - 1 - q)
    v = state.reshape(batch, left, 2, right)
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    out = torch.stack((u[..., 0, 0] * a + u[..., 0, 1] * b,
                       u[..., 1, 0] * a + u[..., 1, 1] * b), dim=2)
    return out.reshape(batch, 2 ** n)


def apply_1q(state: torch.Tensor, u: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """2x2-Gatter `u` auf Qubit `q` eines Zustands (batch, 2**n).

    Qubit 0 ist das hoechstwertige Bit -- dieselbe Konvention wie in
    `simulator.py` (Qubit i = Tensorachse i). Die Paare, die sich in Bit q
    unterscheiden, liegen dadurch im Abstand 2**(n-1-q).
    """
    batch = state.shape[0]
    left, right = 2 ** q, 2 ** (n - 1 - q)
    v = state.reshape(batch, left, 2, right)
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    out = torch.stack((u[0, 0] * a + u[0, 1] * b,
                       u[1, 0] * a + u[1, 1] * b), dim=2)
    return out.reshape(batch, 2 ** n)


def cnot_permutation(n: int, control: int, target: int) -> torch.Tensor:
    """CNOT als Indexpermutation -- klassisch kostenlos, kein Gatterprodukt."""
    idx = torch.arange(2 ** n)
    cm, tm = 1 << (n - 1 - control), 1 << (n - 1 - target)
    return torch.where((idx & cm) != 0, idx ^ tm, idx)


class ButterflyQuantumLayer(nn.Module):
    """Die Quantenschaltung als klassisches Butterfly-Netz mit geteilten Winkeln.

    Rechnet dieselbe Funktion wie `QuantumLayer`, aber in O(d log d) statt ueber
    einen fortgeschriebenen Zustandsvektor mit Tensorkontraktionen.

    Eingabe : (batch, 2**num_qubits) reelle Merkmale, als Amplituden gelesen
    Ausgabe : (batch, num_qubits) Erwartungswerte <Z> je Qubit
    """

    def __init__(self, num_qubits: int, num_layers: int = 3,
                 entangle: bool = True, tie_weights: bool = True):
        super().__init__()
        self.n, self.L = num_qubits, num_layers
        self.entangle, self.tie_weights = entangle, tie_weights
        if tie_weights:
            # 2 Winkel je Qubit und Lage -- die quantenabgeleitete Zahl
            self.theta = nn.Parameter(torch.rand(num_layers, num_qubits, 2) * 2 * math.pi)
        else:
            # Kontrollvariante: jedes Paar bekommt einen eigenen Winkel. Das ist
            # das Butterfly-Netz, auf das der klassische Weg fuehren wuerde.
            # Form (L, n, 2**q, 2**(n-1-q), 2) waere je Ebene verschieden gross;
            # gespeichert wird deshalb flach mit d/2 Paaren und beim Anwenden
            # auf die Ebenenform gebracht.
            self.theta = nn.Parameter(
                torch.rand(num_layers, num_qubits, 2 ** (num_qubits - 1), 2) * 2 * math.pi)
        perms = [cnot_permutation(num_qubits, q, (q + 1) % num_qubits)
                 for q in range(num_qubits)] if entangle and num_qubits > 1 else []
        for i, p in enumerate(perms):
            self.register_buffer(f"perm_{i}", p)
        self.num_perms = len(perms)

    @property
    def input_dim(self) -> int:
        return 2 ** self.n

    def num_circuit_parameters(self) -> int:
        return self.theta.numel()

    def dense_equivalent_parameters(self) -> int:
        d = 2 ** self.n
        return d * d

    def _gate(self, layer: int, q: int) -> torch.Tensor:
        """Rz(b) @ Ry(a) als einzelne komplexe 2x2-Matrix.

        `QuantumLayer` wendet erst ry, dann rz an -- die zusammengefasste Matrix
        ist deshalb Rz*Ry, nicht Ry*Rz. Die Reihenfolge ist nicht beliebig, weil
        die beiden Rotationen nicht kommutieren; die Verifikation unten faellt
        sonst sofort durch (das ist beim ersten Anlauf genau so passiert).

        Zusammenfassen statt nacheinander anwenden halbiert die Zahl der
        Butterfly-Durchlaeufe, ohne die Funktion zu aendern:
            Rz(b) Ry(a) = [[ e^{-ib/2} cos(a/2), -e^{-ib/2} sin(a/2) ],
                           [ e^{+ib/2} sin(a/2),  e^{+ib/2} cos(a/2) ]]
        """
        t = self.theta[layer, q]
        a, b = (t[0], t[1]) if self.tie_weights else (t[..., 0], t[..., 1])
        ca, sa = torch.cos(a / 2), torch.sin(a / 2)
        eb = torch.exp(-0.5j * b.to(torch.complex64))
        ebc = torch.exp(0.5j * b.to(torch.complex64))
        ca, sa = ca.to(torch.complex64), sa.to(torch.complex64)
        gate = torch.stack([torch.stack([ca * eb, -sa * eb], dim=-1),
                            torch.stack([sa * ebc, ca * ebc], dim=-1)], dim=-2)
        if self.tie_weights:
            return gate
        # (d/2, 2, 2) -> (2**q, 2**(n-1-q), 2, 2) fuer diese Ebene
        return gate.reshape(2 ** q, 2 ** (self.n - 1 - q), 2, 2)

    def _zustand(self, x: torch.Tensor, normieren: bool = True) -> torch.Tensor:
        """Zustand nach allen Lagen (komplex, batch x 2**n). Herausgezogen aus
        `forward`, damit Auslese-Varianten (Q-020: lineare Auslese) die
        Zustandsentwicklung wiederverwenden. `normieren=False` ist nur fuer
        Linearitaetspruefungen gedacht."""
        n = self.n
        state = x.to(torch.complex64)
        if normieren:
            state = state / state.norm(dim=1, keepdim=True).clamp_min(1e-12)
        step = apply_1q if self.tie_weights else apply_1q_untied
        for layer in range(self.L):
            for q in range(n):
                state = step(state, self._gate(layer, q), q, n)
            for i in range(self.num_perms):
                state = state[:, getattr(self, f"perm_{i}")]
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        state = self._zustand(x)
        prob = state.real ** 2 + state.imag ** 2
        # <Z_q> = Summe der Wahrscheinlichkeiten mit Bit q = 0 minus jene mit 1
        out = []
        for q in range(n):
            p = prob.reshape(-1, 2 ** q, 2, 2 ** (n - 1 - q))
            out.append((p[:, :, 0, :].sum(dim=(1, 2)) - p[:, :, 1, :].sum(dim=(1, 2))))
        return torch.stack(out, dim=1)


# --------------------------------------------------------------------------
# Verifikation -- ohne sie ist jede Zeitmessung wertlos
# --------------------------------------------------------------------------

def verify(qubits: Tuple[int, ...] = (2, 3, 4, 6, 8),
           layers: Tuple[int, ...] = (1, 3, 5)) -> bool:
    print("=" * 78)
    print("VERIFIKATION -- berechnet die Butterfly-Schicht dieselbe Funktion")
    print("wie der Zustandsvektor-Simulator `QuantumLayer`?")
    print("=" * 78)
    ok, worst = True, 0.0
    for n in qubits:
        for L in layers:
            for ent in (True, False):
                torch.manual_seed(n * 100 + L)
                ref = QuantumLayer(n, L, entangle=ent)
                fast = ButterflyQuantumLayer(n, L, entangle=ent)
                with torch.no_grad():
                    fast.theta.copy_(ref.theta)
                x = torch.randn(7, 2 ** n)
                with torch.no_grad():
                    a, b = ref(x), fast(x)
                dev = float((a - b).abs().max())
                worst = max(worst, dev)
                flag = "OK " if dev < 1e-4 else "ABWEICHUNG"
                ok &= dev < 1e-4
                print(f"  n={n} L={L} entangle={str(ent):>5}  "
                      f"max|Differenz| = {dev:.2e}  {flag}")
    print()
    print(f"  groesste Abweichung insgesamt: {worst:.2e}")
    print("  " + ("ALLE PRUEFUNGEN BESTANDEN -- identische Funktion."
                  if ok else "FEHLGESCHLAGEN -- nicht dieselbe Funktion!"))
    return ok


def parameter_table(qubits: Tuple[int, ...] = (4, 6, 8, 10, 12)) -> None:
    print()
    print("=" * 78)
    print("PARAMETERBILANZ -- worin die Architektur sich unterscheidet")
    print("=" * 78)
    head = (f'  {"Qubits":>6} {"Dim d":>7} {"geteilt":>9} {"ungeteilt":>11} '
            f'{"dicht d^2":>12} {"Faktor":>9}')
    print(head); print("  " + "-" * (len(head) - 2))
    L = 5
    for n in qubits:
        d = 2 ** n
        tied, untied, dense = 2 * n * L, (d // 2) * n * L * 2, d * d
        print(f'  {n:>6} {d:>7,} {tied:>9,} {untied:>11,} {dense:>12,} '
              f'{dense / tied:>8.0f}x')
    print()
    print("  'geteilt'   = die Quantenschaltung: ein Winkel je Qubit und Lage")
    print("  'ungeteilt' = gewoehnliches Butterfly-Netz, ein Winkel je Paar")
    print("  Die Gewichtsteilung ueber die Qubit-Ebene ist der quantenabgeleitete")
    print("  Anteil -- klassisch wuerde man 'ungeteilt' bauen.")


def benchmark(qubits: Tuple[int, ...] = (6, 8, 10, 12), layers: int = 5,
              batch: int = 256, repeats: int = 20) -> List[Dict[str, object]]:
    print()
    print("=" * 78)
    print(f"ZEITMESSUNG -- Batch {batch}, {layers} Lagen, einfaedig")
    print("=" * 78)
    torch.set_num_threads(1)
    head = (f'  {"Qubits":>6} {"Dim":>7} {"Simulator":>12} {"Butterfly":>12} '
            f'{"dicht":>12} {"vs Sim":>9} {"vs dicht":>9}')
    print(head); print("  " + "-" * (len(head) - 2))
    rows: List[Dict[str, object]] = []
    for n in qubits:
        d = 2 ** n
        x = torch.randn(batch, d)
        sim = QuantumLayer(n, layers).eval()
        fast = ButterflyQuantumLayer(n, layers).eval()
        dense = nn.Linear(d, d, bias=False).eval()
        times = {}
        for name, fn in (("sim", sim), ("fast", fast), ("dense", dense)):
            with torch.no_grad():
                for _ in range(3):
                    fn(x)
                t0 = time.perf_counter()
                for _ in range(repeats):
                    fn(x)
                times[name] = (time.perf_counter() - t0) / repeats * 1000
        rows.append({"num_qubits": n, "dim": d, **{f"{k}_ms": v for k, v in times.items()},
                     "speedup_vs_sim": times["sim"] / times["fast"],
                     "speedup_vs_dense": times["dense"] / times["fast"]})
        print(f'  {n:>6} {d:>7,} {times["sim"]:>10.3f}ms {times["fast"]:>10.3f}ms '
              f'{times["dense"]:>10.3f}ms {times["sim"] / times["fast"]:>8.2f}x '
              f'{times["dense"] / times["fast"]:>8.2f}x')
    print()
    print("  'vs dicht' ist die entscheidende Spalte: eine dichte d x d Schicht")
    print("  kostet O(d^2), die Butterfly-Umsetzung O(d log d). Der Vorteil muss")
    print("  also mit d wachsen -- tut er das nicht, ist der Overhead groesser")
    print("  als die Ersparnis, und das waere ehrlich zu berichten.")
    return rows


if __name__ == "__main__":
    if verify():
        parameter_table()
        benchmark()
