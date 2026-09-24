from __future__ import annotations

"""Monarch- und Block-Tensor-Train-Schichten -- die Gegner aus der Literatur.

WARUM DIESE DATEI EXISTIERT
Meine bisherigen Vergleiche stellten die blockdiagonale Schicht gegen `dense`,
gegen Tensor-Train und gegen Low-Rank. Gegen `dense` zu gewinnen ist aber kein
Ergebnis mehr: blockdiagonale Schichten sind seit Nesky & Stout (ICANN 2018)
bekannt, und gruppierte Faltungen gibt es seit AlexNet.

Der tatsaechliche Stand sind strukturierte Matrixfamilien:

  Monarch (Dao et al., ICML 2022, arXiv:2204.00595)
      M = L * P * R  mit L, R blockdiagonal und P einer Permutation.
      Genau deshalb konstruiert, WEIL eine reine Blockdiagonale nicht ueber
      Blockgrenzen mischt: die Permutation zwischen den beiden Blockmatrizen
      stellt die globale Mischung her.

  Block-Tensor-Train (Qiu et al. 2024, arXiv:2406.06248)
      Monarch mit zusaetzlichem Rangindex zwischen den beiden Faktoren --
      kombiniert also Blockstruktur MIT Bond, d.h. genau die Kombination, von
      der meine A-002/A-003-Ergebnisse behaupten, dass der Bond-Anteil nichts
      beitraegt.

Die Literatur behauptet damit das Gegenteil meines Befunds. Wenn meine Aussage
"Lokalitaet ohne Bonds reicht" tragen soll, muss sie gegen diese beiden
bestehen, nicht gegen eine dichte Schicht.

PARAMETERZAHL
Fuer eine n x n-Schicht mit n = m1 * m2:
    Monarch : n * (m1 + m2)              (Rang 1)
    BTT     : n * (m1 + m2) * rank
Bei n = 4096 und m1 = m2 = 64 sind das 524.288 Parameter. Dieselbe Zahl
erreichen eine Blockdiagonale mit Blockgroesse 128 (32 * 128^2) und ein
TT-Layer mit 2 Kernen und Rang 64 (8192 * 64) -- exakt gleiches Budget fuer
alle drei Familien, ohne dass eine Familie bevorzugt werden muss.

EHRLICHKEITSHINWEIS
Beide Schichten sind aus der Strukturbeschreibung der Arbeiten implementiert,
nicht aus deren Referenzcode uebernommen. Deshalb werden sie in
`validate_structured.py` gegen eine unabhaengig gebaute, explizite
Matrixkonstruktion geprueft (Blockdiagonale mal Permutationsmatrix mal
Blockdiagonale), bevor sie in einem Experiment benutzt werden.
"""

from typing import Optional, Tuple

import torch
from torch import nn


def balanced_split(dim: int) -> Tuple[int, int]:
    """Groesster Teiler <= sqrt(dim) und sein Partner -- die ausgewogenste
    Zerlegung dim = m1 * m2, die ohne Auffuellen auskommt."""
    best = 1
    for cand in range(1, int(dim ** 0.5) + 1):
        if dim % cand == 0:
            best = cand
    return best, dim // best


class MonarchLinear(nn.Module):
    """Monarch-Schicht fuer eine quadratische Abbildung dim -> dim.

    Ablauf (x hat Form (batch, dim), dim = m1 * m2):
        X = x.reshape(batch, m1, m2)
        Z = R X          blockdiagonal ueber den m1-Index, mischt innerhalb m2
        W = Z^T          die Permutation: was vorher "innerhalb" war, ist jetzt
                         "zwischen" den Bloecken
        Y = L W          blockdiagonal ueber den m2-Index, mischt innerhalb m1
        out = Y.reshape(batch, dim)

    Durch die Transposition zwischen den beiden Blockmatrizen erreicht jede
    Ausgabe jede Eingabe -- der Unterschied zur reinen Blockdiagonale.

    `btt_rank` > 1 traegt zwischen den beiden Faktoren einen zusaetzlichen
    Rangindex (Block-Tensor-Train). Bei btt_rank == 1 ist das exakt Monarch.
    """

    def __init__(self, dim: int, m1: Optional[int] = None, m2: Optional[int] = None,
                 btt_rank: int = 1, bias: bool = True):
        super().__init__()
        if m1 is None or m2 is None:
            m1, m2 = balanced_split(dim)
        if m1 * m2 != dim:
            raise ValueError(f"m1*m2 muss dim ergeben: {m1}*{m2} != {dim}")
        self.dim, self.m1, self.m2, self.rank = dim, m1, m2, btt_rank

        # R[i] bildet den m2-Vektor des i-ten Blocks ab: (m1, m2_out, m2_in, rank)
        self.R = nn.Parameter(torch.empty(m1, m2, m2, btt_rank))
        # L[j] bildet den m1-Vektor des j-ten Blocks ab: (m2, m1_out, m1_in, rank)
        self.L = nn.Parameter(torch.empty(m2, m1, m1, btt_rank))
        for tensor in (self.R, self.L):
            nn.init.xavier_uniform_(tensor.reshape(tensor.shape[0], -1))
        # Rang-Mittelung, damit die Ausgangsvarianz nicht mit btt_rank waechst
        self._scale = btt_rank ** -0.5
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        X = x.reshape(batch, self.m1, self.m2)
        # blockdiagonal ueber i, mit Rangindex s
        Z = torch.einsum("ikjs,bij->biks", self.R, X)
        # Permutation: (i, k) -> (k, i)
        W = Z.permute(0, 2, 1, 3)                      # (batch, m2, m1, rank)
        # blockdiagonal ueber j (= der frueheren k-Achse), Rangindex wird kontrahiert
        Y = torch.einsum("jkis,bjis->bjk", self.L, W) * self._scale
        out = Y.reshape(batch, self.dim)
        if self.bias is not None:
            out = out + self.bias
        return out

    def dense_reference(self) -> torch.Tensor:
        """Baut dieselbe Abbildung explizit als volle Matrix -- unabhaengig vom
        schnellen Pfad, ueber echte Blockdiagonal- und Permutationsmatrizen.
        Nur zur Verifikation gedacht, nicht fuer den Betrieb."""
        m1, m2, r = self.m1, self.m2, self.rank
        n = self.dim
        device, dtype = self.R.device, self.R.dtype

        # R als blockdiagonale (n x n)-Matrix je Rangkanal, Index i-major/j-minor
        R_dense = torch.zeros(r, n, n, device=device, dtype=dtype)
        for s in range(r):
            for i in range(m1):
                R_dense[s, i * m2:(i + 1) * m2, i * m2:(i + 1) * m2] = self.R[i, :, :, s]

        # Permutation: flach (i, k) -> (k, i)
        P = torch.zeros(n, n, device=device, dtype=dtype)
        for i in range(m1):
            for k in range(m2):
                P[k * m1 + i, i * m2 + k] = 1.0

        # L als blockdiagonale (n x n)-Matrix je Rangkanal, Index j-major/i-minor
        L_dense = torch.zeros(r, n, n, device=device, dtype=dtype)
        for s in range(r):
            for j in range(m2):
                L_dense[s, j * m1:(j + 1) * m1, j * m1:(j + 1) * m1] = self.L[j, :, :, s]

        M = torch.zeros(n, n, device=device, dtype=dtype)
        for s in range(r):
            M = M + L_dense[s] @ P @ R_dense[s]
        return M * self._scale

    def num_structured_parameters(self) -> int:
        total = self.R.numel() + self.L.numel()
        return total + (self.bias.numel() if self.bias is not None else 0)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, m1={self.m1}, m2={self.m2}, btt_rank={self.rank}"


def monarch_parameter_count(dim: int, m1: int, m2: int, btt_rank: int = 1,
                            bias: bool = True) -> int:
    return dim * (m1 + m2) * btt_rank + (dim if bias else 0)
