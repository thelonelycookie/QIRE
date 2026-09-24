from __future__ import annotations

"""RESEARCH ID: T-003 -- Der Rechen-Buendler: n*L kleine Schritte zu wenigen GEMMs.

DIE IDEE (Malte, 23. September)
`butterfly_fast` fuehrt je Lage n Paardrehungen einzeln aus -- n*L kleine
Kernel, jeder ein voller Durchlauf ueber den Zustand. Auf der GPU ist das
startaufwandbegrenzt, auf der CPU speicherverkehrsbegrenzt (T-001). Der
Buendler fasst die kleinen Schritte zu wenigen grossen Matrixprodukten (GEMM)
zusammen -- genau die Operation, fuer die CPU und GPU gebaut sind.

DIE MATHEMATIK, ohne die nichts hiervon stimmt
Eine Lage ohne CNOT ist das Kronecker-Produkt der Ein-Qubit-Gatter
    K = G_0 (x) G_1 (x) ... (x) G_{n-1}       (Qubit 0 = hoechstwertiges Bit).
Teilt man die Qubits in A = {0..k-1} und B = {k..n-1} und liest den Zustand
als Matrix X (2^k x 2^(n-k)), dann ist K x  ==  U_A . X . U_B^T  mit
U_A = G_0 (x) ... (x) G_{k-1}. Zwei GEMMs statt n Paardrehungen. Allgemein mit
m Gruppen: m GEMMs -- m = n ist wieder der Butterfly, m = 1 die volle Matrix.
Das ist dieselbe Wahl wie der Radix einer FFT.

Ohne CNOT kollabieren alle L Lagen auf EINE: (G'(x)H')(G(x)H) = G'G (x) H'H.
Also immer 2 GEMMs, gleich wie tief die Schaltung ist.

Mit CNOT-Ring bleibt je Lage eine Permutation (ein `gather`, R1). Sie laesst
sich aber in die Faktoren falten -- `kron2_gefaltet`, hergeleitet und
numerisch geprueft (s. `_gefaltet_matrizen`): Der Ring zerfaellt an der
Trennstelle in CNOTs innerhalb A (in U_A multipliziert), innerhalb B (in U_B),
einen CNOT A->B (die B-Seite bekommt ZWEI Matrizen, je nach einem Bit der
Zeile) und einen CNOT B->A (die A-Seite der NAECHSTEN Lage bekommt zwei
Matrizen, je nach einem Bit der Spalte; nach der letzten Lage wandert er in
die Vorzeichenmatrix der Auslese). Kein Gather mehr -- dafuer gruppierte GEMMs.
Das ist eine Monarch-artige Faktorisierung (Dao et al. 2022): blockdiagonal
mal Permutation mal blockdiagonal, hier mit geteilten Bloecken.

`vorkompiliert` baut aus den Winkeln einmal die volle Matrix U und rechnet
danach EIN reelles GEMM (x reell => U x = Re(U) x + i Im(U) x). Kostet d^2
Speicher und 2*d^2 Arithmetik -- lohnt nur fuer Inferenz mit eingefrorenen
Winkeln, kleinem d und grossen Batches. Zeigt: 2nL Parameter, Tempo wie dicht.

`reell` ist ein EIGENER ARM, keine Umschreibung: nur Ry-Drehungen, keine
Rz-Phasen, alles float32 statt complex64 (Faktor ~4 weniger Arithmetik).
Es ist eine andere Funktion (n*L statt 2*n*L Parameter) und wird deshalb
gegen `butterfly_fast` mit Rz-Winkel 0 geprueft, nicht gegen das Original.
Dass Rz im allgemeinen NICHT streichbar ist (R4), zeigt die Verifikation mit.

VERIFIKATION VOR JEDER ZEITMESSUNG
`python -m quantum_sim.buendler` prueft alle Strategien gegen
`ButterflyFast` und `ButterflyQuantumLayer` ueber n=4..14, L=1..5, mit und
ohne CNOT, dazu gegen eine unabhaengige float64-Rechnung, und vergleicht die
Gradienten nach Winkeln UND Eingabe. Ohne bestandene Pruefung keine Messung.
"""

import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from .butterfly_fast import ButterflyFast, komponiere, vorzeichenmatrix
from .butterfly_layer import ButterflyQuantumLayer, apply_1q, cnot_permutation

STRATEGIEN = ("kron2", "kron3", "kron2_gefaltet", "vorkompiliert", "reell_kron2")


def gruppen_fuer(n: int, m: int) -> List[int]:
    """Qubitzahlen je Gruppe, moeglichst gleich; die letzte (grosse, zusammen-
    haengende GEMM) bekommt den Rest."""
    m = max(1, min(m, n))
    basis = [n // m] * m
    for i in range(n - sum(basis)):
        basis[m - 1 - i] += 1
    return basis


def kron_kette(g: torch.Tensor) -> torch.Tensor:
    """(..., m, 2, 2) -> (..., 2^m, 2^m): G_0 (x) G_1 (x) ... (x) G_{m-1}.

    kron(A, B)[i1*2+i2, j1*2+j2] = A[i1, j1] * B[i2, j2]. Differenzierbar,
    weil nur Broadcast-Multiplikation und reshape."""
    u = g[..., 0, :, :]
    for j in range(1, g.shape[-3]):
        s = u.shape[-1]
        u = (u[..., :, None, :, None] * g[..., j, None, :, None, :]).reshape(
            *u.shape[:-2], 2 * s, 2 * s)
    return u


def ring_permutation(n: int) -> torch.Tensor:
    return komponiere([cnot_permutation(n, q, (q + 1) % n) for q in range(n)])


class Buendler(nn.Module):
    """Funktionsgleich zu `ButterflyFast(n, L, entangle)` (ausser `reell_kron2`).

    theta hat dieselbe Form (L, n, 2) -- Winkel sind 1:1 austauschbar.
    """

    def __init__(self, num_qubits: int, num_layers: int, entangle: bool = True,
                 strategie: str = "kron2", gruppen: Optional[Sequence[int]] = None):
        super().__init__()
        if strategie not in STRATEGIEN:
            raise ValueError(f"unbekannte Strategie {strategie!r}")
        n = num_qubits
        self.n, self.L, self.d, self.strategie = n, num_layers, 2 ** n, strategie
        self.entangle = entangle and n > 1
        self.reell = strategie.startswith("reell")
        self.dtype = torch.float32 if self.reell else torch.complex64
        breite = 1 if self.reell else 2
        self.theta = nn.Parameter(torch.rand(num_layers, n, breite) * 2 * math.pi)
        if gruppen is None:
            gruppen = gruppen_fuer(n, 3 if strategie == "kron3" else 2)
        if sum(gruppen) != n or min(gruppen) < 1:
            raise ValueError(f"Gruppen {gruppen} passen nicht zu {n} Qubits")
        self.gruppen = list(gruppen)
        self.register_buffer("vz", vorzeichenmatrix(n))
        self.register_buffer("perm", ring_permutation(n) if self.entangle else None)
        if strategie == "kron2_gefaltet":
            if not self.entangle:
                raise ValueError("kron2_gefaltet braucht den CNOT-Ring")
            self._baue_faltung()
        self._cache: Optional[object] = None

    # ------------------------------------------------------------------ Gatter
    def num_circuit_parameters(self) -> int:
        return self.theta.numel()

    def _gatter(self) -> torch.Tensor:
        """(L, n, 2, 2). Rz(b) @ Ry(a) wie `ButterflyFast._gatter`; reell nur Ry(a)."""
        a = self.theta[..., 0]
        ca, sa = torch.cos(a / 2), torch.sin(a / 2)
        if self.reell:
            return torch.stack([torch.stack([ca, -sa], -1),
                                torch.stack([sa, ca], -1)], -2)
        b = self.theta[..., 1]
        eb = torch.exp(-0.5j * b.to(torch.complex64))
        ebc = torch.exp(0.5j * b.to(torch.complex64))
        ca, sa = ca.to(torch.complex64), sa.to(torch.complex64)
        return torch.stack([torch.stack([ca * eb, -sa * eb], -1),
                            torch.stack([sa * ebc, ca * ebc], -1)], -2)

    def _gruppenmatrizen(self) -> List[List[torch.Tensor]]:
        """Je wirksamer Lage die Gruppenfaktoren U_A, U_B(, U_C)."""
        g = self._gatter()                                   # (L, n, 2, 2)
        if not self.entangle:                                # L Lagen -> eine
            acc = g[0]
            for layer in range(1, self.L):
                acc = g[layer] @ acc
            g = acc[None]
        lagen, start = [], 0
        mats = []
        for k in self.gruppen:
            mats.append(kron_kette(g[:, start:start + k]))   # (L', 2^k, 2^k)
            start += k
        for layer in range(g.shape[0]):
            lagen.append([u[layer] for u in mats])
        return lagen

    # ---------------------------------------------------- Faltung des Rings
    def _baue_faltung(self) -> None:
        """Indexvektoren fuer `kron2_gefaltet` (Herleitung im Modulkopf).

        Konvention wie `cnot_permutation`: neu[i] = alt[perm[i]], und fuer eine
        Matrix M auf der Zeilenseite gilt P @ M == M[perm]."""
        n, (k, m) = self.n, self.gruppen
        if len(self.gruppen) != 2:
            raise ValueError("Faltung ist fuer genau zwei Gruppen hergeleitet")
        ident = lambda q: torch.arange(2 ** q)
        self.register_buffer("pA", komponiere([cnot_permutation(k, q, q + 1)
                                               for q in range(k - 1)]) if k > 1 else ident(k))
        self.register_buffer("pG", komponiere([cnot_permutation(m, q, q + 1)
                                               for q in range(m - 1)]) if m > 1 else ident(m))
        self.register_buffer("pXB", ident(m) ^ (1 << (m - 1)))   # CNOT(k-1 -> k): MSB von b
        self.register_buffer("pXA", ident(k) ^ (1 << (k - 1)))   # CNOT(n-1 -> 0): MSB von a
        p4 = cnot_permutation(n, n - 1, 0)                        # Involution
        self.register_buffer("vz_gefaltet", vorzeichenmatrix(n)[:, p4])

    def _gefaltet_matrizen(self):
        out = []
        for ua, ub in self._gruppenmatrizen():
            a0 = ua[self.pA]                   # F_A U_A
            a1 = a0[:, self.pXA]               # F_A U_A X_0   (offener CNOT der Vorlage)
            v0 = ub[self.pG]                   # G_B U_B
            v1 = ub[self.pXB][self.pG]         # G_B X_k U_B
            out.append((a0, a1, v0.transpose(0, 1), v1.transpose(0, 1)))
        return out

    # ------------------------------------------------------- Ausfuehrung
    def _zustand_kron(self, x: torch.Tensor, lagen) -> torch.Tensor:
        batch = x.shape[0]
        for us in lagen:
            vor = 1
            for k, u in zip(self.gruppen, us):
                g = 1 << k
                nach = self.d // (vor * g)
                if nach == 1:                                # grosses, zusammenhaengendes GEMM
                    x = x.reshape(-1, g) @ u.transpose(0, 1)
                else:                                        # gebatchtes GEMM auf der Mittelachse
                    x = torch.matmul(u, x.reshape(batch * vor, g, nach))
                vor *= g
            x = x.reshape(batch, self.d)
            if self.perm is not None:
                x = x[:, self.perm]                          # R1: ein Gather je Lage
        return x

    def _zustand_gefaltet(self, x: torch.Tensor, mats) -> torch.Tensor:
        batch, (k, m) = x.shape[0], self.gruppen
        ka, kb = 1 << k, 1 << m
        for layer, (a0, a1, v0t, v1t) in enumerate(mats):
            xv = x.reshape(batch, ka, kb)
            if layer == 0:
                y = torch.matmul(a0, xv)
            else:                              # Spalten mit gesetztem LSB(b): offener CNOT B->A
                xs = xv.reshape(batch, ka, kb // 2, 2)
                y = torch.stack((torch.matmul(a0, xs[..., 0]),
                                 torch.matmul(a1, xs[..., 1])), dim=-1)
            yv = y.reshape(batch, ka // 2, 2, kb)            # Zeilen mit gesetztem LSB(a): CNOT A->B
            x = torch.stack((yv[:, :, 0, :] @ v0t, yv[:, :, 1, :] @ v1t),
                            dim=2).reshape(batch, self.d)
        return x

    def _vorkompiliert_W(self) -> torch.Tensor:
        """W = [Re U^T | Im U^T] (d x 2d), gebaut durch Anwenden auf die Einheitsbasis."""
        lagen = self._gruppenmatrizen()
        spalten = []
        for s in range(0, self.d, 1024):                     # stueckweise: Speicher
            e = torch.eye(self.d, dtype=self.dtype, device=self.theta.device)[s:s + 1024]
            spalten.append(self._zustand_kron(e, lagen))
        ut = torch.cat(spalten, 0)                           # Zeile j = U e_j  ->  U^T
        return torch.cat([ut.real, ut.imag], dim=1).contiguous()

    def _vorbereitet(self):
        """R5: bei eingefrorenen Winkeln alles Winkelabhaengige einmal bauen."""
        if self._cache is not None:
            return self._cache
        if self.strategie == "vorkompiliert":
            obj = self._vorkompiliert_W()
        elif self.strategie == "kron2_gefaltet":
            obj = self._gefaltet_matrizen()
        else:
            obj = self._gruppenmatrizen()
        if not self.theta.requires_grad:
            self._cache = obj
        return obj

    def leere_cache(self) -> None:
        self._cache = None

    def train(self, mode: bool = True):
        self._cache = None
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vorbereitet = self._vorbereitet()
        if self.strategie == "vorkompiliert":
            if x.is_complex():
                raise ValueError("vorkompiliert setzt reelle Eingaben voraus")
            y = x @ vorbereitet                              # EIN reelles GEMM, (B, 2d)
            prob = y[:, :self.d] ** 2 + y[:, self.d:] ** 2
            norm2 = (x * x).sum(dim=1, keepdim=True).clamp_min(1e-24)
            return (prob / norm2) @ self.vz.t()
        state = x.to(self.dtype)
        state = state / state.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
        if self.strategie == "kron2_gefaltet":
            state = self._zustand_gefaltet(state, vorbereitet)
            vz = self.vz_gefaltet
        else:
            state = self._zustand_kron(state, vorbereitet)
            vz = self.vz
        prob = state * state if self.reell else state.real ** 2 + state.imag ** 2
        return prob @ vz.t()                                 # R6


# ============================================================ Verifikation
def _referenz_float64(theta: torch.Tensor, x: torch.Tensor, n: int, L: int,
                      entangle: bool) -> torch.Tensor:
    """Unabhaengige Rechnung in complex128: dieselben Paardrehungen wie der
    Butterfly, aber doppelt genau. Dient als Schiedsrichter, falls float32-
    Rundung und ein echter Fehler auseinanderzuhalten sind."""
    t = theta.detach().double()
    a = t[..., 0]
    b = t[..., 1] if t.shape[-1] > 1 else torch.zeros_like(a)
    ca, sa = torch.cos(a / 2).to(torch.complex128), torch.sin(a / 2).to(torch.complex128)
    eb, ebc = torch.exp(-0.5j * b.to(torch.complex128)), torch.exp(0.5j * b.to(torch.complex128))
    g = torch.stack([torch.stack([ca * eb, -sa * eb], -1),
                     torch.stack([sa * ebc, ca * ebc], -1)], -2)
    s = x.detach().to(torch.complex128)
    s = s / s.abs().pow(2).sum(1, keepdim=True).sqrt()
    perm = ring_permutation(n) if entangle and n > 1 else None
    for layer in range(L):
        for q in range(n):
            s = apply_1q(s, g[layer, q], q, n)
        if perm is not None:
            s = s[:, perm]
    return (s.abs() ** 2) @ vorzeichenmatrix(n).double().t()


def _paar(n, L, ent, strategie, seed):
    torch.manual_seed(seed)
    ref = ButterflyFast(n, L, entangle=ent)
    b = Buendler(n, L, entangle=ent, strategie=strategie)
    with torch.no_grad():
        if b.reell:
            ref.theta[..., 1] = 0.0                          # Rz(0) = 1
            b.theta.copy_(ref.theta[..., :1])
        else:
            b.theta.copy_(ref.theta)
    return ref, b


def verifiziere(qubits: Sequence[int] = tuple(range(4, 15)),
                lagen: Sequence[int] = (1, 2, 3, 4, 5), tol: float = 1e-6,
                max_n_vorkompiliert: int = 10) -> bool:
    print("=" * 100)
    print("T-003 VERIFIKATION -- Buendler gegen ButterflyFast, ButterflyQuantumLayer, float64")
    print(f"Schranke {tol:g} gegen ButterflyFast. Ohne bestandene Pruefung keine Zeitmessung.")
    print("=" * 100)
    alle_ok, schlimmst, n_konf = True, {}, 0
    for n in qubits:
        for L in lagen:
            for ent in (True, False):
                x = torch.randn(7, 2 ** n, generator=torch.Generator().manual_seed(n * 97 + L))
                for s in STRATEGIEN:
                    if s == "kron2_gefaltet" and not ent:
                        continue
                    if s == "vorkompiliert" and n > max_n_vorkompiliert:
                        continue
                    ref, b = _paar(n, L, ent, s, 3000 + 100 * n + 10 * L + int(ent))
                    with torch.no_grad():
                        o_ref, o_b = ref(x), b(x)
                        altref = ButterflyQuantumLayer(n, L, entangle=ent)
                        altref.theta.copy_(ref.theta)
                        o_alt = altref(x)
                        o_64 = _referenz_float64(ref.theta, x, n, L, ent)
                    d_ref = float((o_b - o_ref).abs().max())
                    d_alt = float((o_b - o_alt).abs().max())
                    d_64b = float((o_b.double() - o_64).abs().max())
                    d_64r = float((o_ref.double() - o_64).abs().max())
                    ok = d_ref <= tol and d_alt <= tol
                    alle_ok &= ok
                    n_konf += 1
                    w = schlimmst.setdefault(s, [0.0, 0.0, 0.0, 0])
                    w[0], w[1] = max(w[0], d_ref), max(w[1], d_64b)
                    w[2], w[3] = max(w[2], d_64r), w[3] + (0 if ok else 1)
                    if not ok:
                        print(f"  FEHLER n={n} L={L} ent={ent} {s}: gegen fast {d_ref:.2e}, "
                              f"gegen Layer {d_alt:.2e}, float64 B/F {d_64b:.2e}/{d_64r:.2e}")
    head = (f'  {"Strategie":<16}{"max |B-fast|":>14}{"max |B-f64|":>14}'
            f'{"max |fast-f64|":>16}{"Fehler":>8}')
    print(head); print("  " + "-" * (len(head) - 2))
    for s, (a, b64, r64, f) in schlimmst.items():
        print(f"  {s:<16}{a:>14.2e}{b64:>14.2e}{r64:>16.2e}{f:>8}")
    print(f"\n  {n_konf} Konfigurationen (n={min(qubits)}..{max(qubits)}, "
          f"L={min(lagen)}..{max(lagen)}, mit/ohne CNOT; vorkompiliert bis n={max_n_vorkompiliert}).")
    print("  reell_kron2 wird gegen ButterflyFast mit Rz-Winkel 0 geprueft (eigener Arm).")
    print("VERIFIKATION " + ("BESTANDEN" if alle_ok else "DURCHGEFALLEN -- keine Zeitmessung."))
    return alle_ok


def gradientenpruefung(tol: float = 1e-4) -> bool:
    """Gradienten nach Winkeln UND nach der Eingabe -- die Schicht sitzt in
    einem Netz, beides muss stimmen."""
    print("\n  Gradientenpruefung (gewichtete Summe der Ausgaben, Batch 5)")
    alle_ok, schlimmst = True, 0.0
    for n in (4, 6, 8, 11):
        for L in (1, 3, 5):
            for ent in (True, False):
                for s in STRATEGIEN:
                    if (s == "kron2_gefaltet" and not ent) or (s == "vorkompiliert" and n > 8):
                        continue
                    ref, b = _paar(n, L, ent, s, 9000 + 100 * n + 10 * L + int(ent))
                    x1 = torch.randn(5, 2 ** n, generator=torch.Generator().manual_seed(n + L))
                    w = torch.randn(5, n, generator=torch.Generator().manual_seed(7 * n + L))
                    x2 = x1.clone().requires_grad_(True)
                    x1.requires_grad_(True)
                    (ref(x1) * w).sum().backward()
                    (b(x2) * w).sum().backward()
                    g_ref = ref.theta.grad[..., :1] if b.reell else ref.theta.grad
                    d = max(float((g_ref - b.theta.grad).abs().max()),
                            float((x1.grad - x2.grad).abs().max()))
                    schlimmst = max(schlimmst, d)
                    if d > tol:
                        alle_ok = False
                        print(f"    FEHLER n={n} L={L} ent={ent} {s}: {d:.2e}")
    print(f"    groesste Abweichung (Winkel und Eingabe): {schlimmst:.2e} -> "
          f"{'OK' if alle_ok else 'FEHLER'}")
    return alle_ok


def rz_streichbar(n: int = 6, L: int = 3) -> float:
    """R4 numerisch: aendert sich die Ausgabe, wenn man alle Rz weglaesst?"""
    torch.manual_seed(5)
    voll = ButterflyFast(n, L, entangle=True)
    ohne = ButterflyFast(n, L, entangle=True)
    with torch.no_grad():
        ohne.theta.copy_(voll.theta)
        ohne.theta[..., 1] = 0.0
        x = torch.randn(9, 2 ** n)
        return float((voll(x) - ohne(x)).abs().max())


# ============================================================ Aufwand
def aufwand(strategie: str, n: int, L: int, batch: int, entangle: bool = True) -> Dict[str, float]:
    """Reelle Multiply-Adds (MAC) und Durchlaeufe ueber den Zustand, Formel.

    Komplexe MAC = 4 reelle. Durchlaeufe zaehlen Operationen, die den ganzen
    Zustand lesen und schreiben (GEMM, Gather, Stack-Kopie, Paardrehung).
    FLOPs sagen die Zeit NICHT vorher (T-001: 340x weniger, 5x langsamer)."""
    d = 2 ** n
    Lw = L if entangle else 1
    if strategie == "dicht":
        return {"mac": batch * d * d, "durchlaeufe": 1, "gewichte_bytes": 4 * d * d}
    if strategie == "referenz":
        return {"mac": 4 * 2 * batch * d * n * L,
                "durchlaeufe": L * n + (L if entangle else 0) + 2, "gewichte_bytes": 0}
    if strategie == "vorkompiliert":
        return {"mac": 2 * batch * d * d, "durchlaeufe": 1, "gewichte_bytes": 8 * d * d}
    m = 3 if strategie == "kron3" else 2
    gr = gruppen_fuer(n, m)
    faktor = 1 if strategie.startswith("reell") else 4
    mac = faktor * batch * d * sum(2 ** k for k in gr) * Lw
    stufen = m * Lw + (Lw if entangle and strategie != "kron2_gefaltet" else 0)
    if strategie == "kron2_gefaltet":
        stufen = 2 * L + (2 * L - 1)                        # GEMMs + Stack-Kopien
    return {"mac": mac, "durchlaeufe": stufen + 2,
            "gewichte_bytes": (4 if faktor == 1 else 8) * sum(4 ** k for k in gr) * Lw}


def aufwand_tabelle(qubits: Sequence[int] = range(8, 17), L: int = 5, batch: int = 64,
                    entangle: bool = True) -> None:
    arme = ("dicht", "referenz", "kron2", "kron3", "kron2_gefaltet", "vorkompiliert",
            "reell_kron2")
    print("\n" + "=" * 100)
    print(f"AUFWAND (Formel) -- reelle MAC relativ zu dicht, Batch {batch}, L={L}, "
          f"CNOT={'ja' if entangle else 'nein'}; in Klammern Durchlaeufe ueber den Zustand")
    print("=" * 100)
    print("  " + f'{"d":>7}' + "".join(f"{a:>15}" for a in arme))
    for n in qubits:
        d = 2 ** n
        dicht = aufwand("dicht", n, L, batch)["mac"]
        zeile = f"  {d:>7}"
        for a in arme:
            if a == "kron2_gefaltet" and not entangle:
                zeile += f"{'-':>15}"
                continue
            r = aufwand(a, n, L, batch, entangle)
            zeile += f"{r['mac'] / dicht:>9.4f} ({r['durchlaeufe']:>2})"
        print(zeile)


if __name__ == "__main__":
    ok = verifiziere()
    ok &= gradientenpruefung()
    print(f"\n  R4-Pruefung: Ausgabe mit gegen ohne Rz, max. Abweichung "
          f"{rz_streichbar():.3e} (0 hiesse: Rz streichbar)")
    aufwand_tabelle()
    aufwand_tabelle(entangle=False)
    raise SystemExit(0 if ok else 1)
