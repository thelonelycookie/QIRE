from __future__ import annotations

"""Verifikation der Monarch-/BTT-Schicht, BEVOR sie in einem Experiment benutzt wird.

Dieselbe Regel wie bei Tensor-Train und TT-SVD: eine selbst implementierte
Struktur wird erst gegen eine unabhaengig gebaute Referenz geprueft. Sonst ist
jeder spaetere Vergleich wertlos -- ein Implementierungsfehler saehe genauso aus
wie ein wissenschaftliches Ergebnis.

Geprueft wird:
  1. schneller Pfad == explizite Matrixkonstruktion (Blockdiagonale mal
     Permutation mal Blockdiagonale), fuer mehrere Groessen und Raenge
  2. Parameterzahl == Formel n*(m1+m2)*rank
  3. Gradienten erreichen jeden Parameterblock
  4. STRUKTUR: Monarch mischt global (jede Ausgabe haengt von jeder Eingabe ab),
     die Blockdiagonale nicht -- das ist der inhaltliche Unterschied, um den es
     im Vergleich geht, und er wird hier gemessen statt behauptet
  5. Budgetgleichheit bei 4096: Monarch(64,64), Blockdiagonale(128) und
     TT(2 Kerne, Rang 64) haben exakt dieselbe Parameterzahl

Run: python -m qire_ai.validate_structured
"""

import torch

from .ablation_structure import BlockDiagonal
from .structured_layers import MonarchLinear, monarch_parameter_count
from .tensor_train import TTLinear

TOL = 1e-5


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FEHLER'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def check_against_dense_reference() -> bool:
    print("\n1. Schneller Pfad gegen explizite Matrixkonstruktion")
    ok = True
    for dim, m1, m2, rank in [(16, 4, 4, 1), (36, 6, 6, 1), (64, 8, 8, 1),
                              (24, 4, 6, 1), (16, 4, 4, 3), (36, 6, 6, 2)]:
        torch.manual_seed(0)
        layer = MonarchLinear(dim, m1, m2, btt_rank=rank, bias=False)
        x = torch.randn(7, dim, dtype=torch.float64)
        layer = layer.double()
        fast = layer(x)
        slow = x @ layer.dense_reference().t()
        dev = float((fast - slow).abs().max())
        ok &= _check(f"dim={dim} ({m1}x{m2}), rank={rank}", dev < 1e-10,
                     f"groesste Abweichung {dev:.2e}")
    return ok


def check_parameter_count() -> bool:
    print("\n2. Parameterzahl gegen Formel n*(m1+m2)*rank")
    ok = True
    for dim, m1, m2, rank in [(64, 8, 8, 1), (4096, 64, 64, 1), (36, 6, 6, 3)]:
        layer = MonarchLinear(dim, m1, m2, btt_rank=rank)
        want = monarch_parameter_count(dim, m1, m2, rank)
        got = layer.num_structured_parameters()
        ok &= _check(f"dim={dim}, rank={rank}", want == got, f"{got:,} == {want:,}")
    return ok


def check_gradients() -> bool:
    print("\n3. Gradientenfluss")
    torch.manual_seed(0)
    layer = MonarchLinear(64, 8, 8)
    out = layer(torch.randn(5, 64))
    out.pow(2).sum().backward()
    ok = True
    for name, p in layer.named_parameters():
        has = p.grad is not None and float(p.grad.abs().sum()) > 0
        ok &= _check(f"Gradient erreicht {name}", has)
    return ok


def check_mixing_structure() -> bool:
    """Der inhaltliche Kern: erreicht jede Ausgabe jede Eingabe?

    Gemessen ueber die Jacobi-Matrix. Fuer Monarch muessen alle Eintraege
    ungleich null sein, fuer die Blockdiagonale darf nur der eigene Block
    besetzt sein. Genau diese Eigenschaft ist der Grund, warum Monarch
    ueberhaupt konstruiert wurde.
    """
    print("\n4. Mischungsstruktur (Jacobi-Matrix)")
    dim, block = 16, 4
    torch.manual_seed(0)

    def density(module: torch.nn.Module) -> float:
        x = torch.randn(1, dim, requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda t: module(t).sum(0), x)
        jac = jac.reshape(dim, dim).abs()
        return float((jac > 1e-9).float().mean())

    mono = density(MonarchLinear(dim, 4, 4, bias=False))
    blk = density(BlockDiagonal(dim, block))
    ok = _check("Monarch mischt global", mono > 0.999,
                f"besetzte Jacobi-Eintraege: {mono:.3f} (erwartet 1.000)")
    ok &= _check("Blockdiagonale mischt nur lokal", abs(blk - block / dim) < 1e-6,
                 f"besetzte Jacobi-Eintraege: {blk:.3f} (erwartet {block/dim:.3f})")
    return ok


def check_equal_budget() -> bool:
    """Ohne exakt gleiches Budget waere der geplante Vergleich wertlos."""
    print("\n5. Budgetgleichheit bei 4096x4096 (ohne Bias gerechnet)")
    dim = 4096
    mono = MonarchLinear(dim, 64, 64, bias=False).num_structured_parameters()
    blk = BlockDiagonal(dim, 128)
    blk_n = int(blk.weight.numel())
    tt = TTLinear(dim, dim, num_cores=2, tt_rank=64, bias=False)
    tt_n = sum(c.numel() for c in tt.cores)
    print(f"        Monarch(64x64)          : {mono:>10,}")
    print(f"        Blockdiagonale(128)     : {blk_n:>10,}")
    print(f"        Tensor-Train(2 Kerne,64): {tt_n:>10,}")
    return _check("alle drei Familien exakt gleich", mono == blk_n == tt_n)


def check_local_architecture() -> bool:
    """Haelt `local_ends` wirklich, was es verspricht?

    Geprueft an den Aktivierungen VOR der Auslese (die Summe am Ende wuerde
    ohnehin alles verbinden). Mit blockdiagonaler Mittelschicht darf kein
    verstecktes Merkmal Information aus einer fremden Gruppe enthalten; mit
    Monarch als Mittelschicht muss es das sehr wohl.

    Ohne diese Pruefung waere B-004 wertlos: ein Fehler in der Architektur
    saehe genau aus wie ein wissenschaftliches Ergebnis.
    """
    print("\n6. Architektur local_ends -- fliesst wirklich nichts zwischen den Gruppen?")
    from .ablation_structure import BlockDiagonal as BD
    from .local_net import MiddleNet

    in_dim, width, groups, classes = 32, 64, 4, 3
    per_in, per_out = in_dim // groups, width // groups
    torch.manual_seed(0)

    def cross_group_mass(label: str, factory, arch: str) -> float:
        net = MiddleNet(in_dim, width, classes, factory, label, arch, groups=groups)
        x = torch.randn(1, in_dim)
        jac = torch.autograd.functional.jacobian(lambda t: net.features(t).sum(0), x)
        jac = jac.reshape(width, in_dim).abs()
        mask = torch.ones(width, in_dim, dtype=torch.bool)
        for g in range(groups):                      # eigene Gruppe ausblenden
            mask[g * per_out:(g + 1) * per_out, g * per_in:(g + 1) * per_in] = False
        return float(jac[mask].max())

    # ReLU kann Eintraege zufaellig auf 0 setzen; getestet wird die Obergrenze,
    # deshalb ist "exakt 0 ueber alle fremden Paare" die richtige Bedingung.
    blk = cross_group_mass("block_diag", lambda: BD(width, per_out), "local_ends")
    mono = cross_group_mass("monarch", lambda: MonarchLinear(width, 8, 8), "local_ends")
    dense_arch = cross_group_mass("block_diag", lambda: BD(width, per_out), "dense_ends")

    ok = _check("local_ends + Blockdiagonale: keine Querverbindung", blk == 0.0,
                f"groesster fremder Jacobi-Eintrag {blk:.2e} (muss exakt 0 sein)")
    ok &= _check("local_ends + Monarch: Querverbindung vorhanden", mono > 1e-6,
                 f"groesster fremder Jacobi-Eintrag {mono:.2e}")
    ok &= _check("dense_ends + Blockdiagonale: Querverbindung vorhanden (der alte "
                 "Aufbau mischt global)", dense_arch > 1e-6,
                 f"groesster fremder Jacobi-Eintrag {dense_arch:.2e}")
    return ok


def main() -> bool:
    print("=" * 68)
    print("VERIFIKATION der Monarch-/BTT-Schicht")
    print("=" * 68)
    results = [
        check_against_dense_reference(),
        check_parameter_count(),
        check_gradients(),
        check_mixing_structure(),
        check_equal_budget(),
        check_local_architecture(),
    ]
    print()
    print("=" * 68)
    if all(results):
        print("ALLE PRUEFUNGEN BESTANDEN -- die Schicht darf in Experimenten benutzt werden.")
    else:
        print("MINDESTENS EINE PRUEFUNG IST GESCHEITERT -- nicht in Experimenten benutzen.")
    print("=" * 68)
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
