"""K09 Born-Normierung der FFN-Mitte (Erkundung, kein Befund).
Mechanismus: statt z^2 rechnet die FFN z^2/(Mittel_Gruppe z^2 + eps) (divisive Normierung, Gruppen 8/32 der Mitte), FLOPs gleich.
Arme (je d = 64 und 128, FFN-Breite 4d, 4 Lagen, Byte-LM deutscher Text): gelu / z2 / z2_bn8 / z2_bn32 / z2_ln (LayerNorm ohne Skala vor dem Quadrat = Kontrolle).
Lernrate je Arm gewählt (Validierung, Gitter 1e-3, 2,5e-3, 6e-3, bei Randtreffer verlängert). Parameter gleich (LN ohne Skala), FFN-FLOPs gleich (Elementweises nicht gezählt).
Aufruf: python3 erkundung/K09.py --seed 950 [--schnell]   ->   erkundung/roh/K09_s<seed>.json"""
import functools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bytelm import LM, FFNDicht, cli, lauf_zellen  # noqa: E402

KID = "K09"
ARME = ("gelu", "z2", "z2bn8", "z2bn32", "z2ln")


def bauer(d, akt, seed):
    return LM(d, lambda dd, i: FFNDicht(dd, 4 * dd, akt))


def main(seed: int, schnell: bool = False) -> dict:
    if schnell:
        plan = [(64, 40, 80)]
    else:
        plan = [(128, 250, 4000), (64, 250, 1500)]
    zellen = [(f"d{d}/{a}", functools.partial(bauer, d, a), tt, tf, None) for d, tt, tf in plan for a in ARME]
    return lauf_zellen(KID, seed, schnell, zellen, __doc__)


if __name__ == "__main__":
    cli(KID, main)
