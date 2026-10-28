"""K10 Born-Stichprobe der FFN-Spalten (Erkundung, kein Befund).
Mechanismus: relu^2-FFN (d=128, h=4d=512). Die ZWEITE Matrix wird je Token nur auf S gezogenen Spalten gerechnet; Ziehung mit Wahrscheinlichkeit
p_j ~ a_j^2 (a = relu(z)^2, mit 1 % Gleichverteilung gemischt, damit Gewichte beschränkt bleiben), mit Zurücklegen, erwartungstreue Schätzung sum_s a_js / (S p_js) * W2[js].
(Training als dichte Simulation mit Nullen, FLOPs analytisch für die dünne Rechnung.)
Auswertung mit derselben Stichprobe (feste Zufallszahlen); zusätzlich bpc bei dichter Auswertung (alle Spalten) als Nebenwert 'bpc_dicht_ausgewertet'.
Arme: dicht_voll (h=512) | schmal_S4 / schmal_S8 (dicht, FLOP-gleich: 4*d*h' = 2*d*h + 2*d*S, h'=320 bzw. 288) |
      born_S4/top_S4/unif_S4 (S=h/4=128) | born_S8/top_S8/unif_S8 (S=h/8=64). Kontrollen: unif = Gleichverteilung (Gewicht h/S), top = deterministisches Top-S (ohne Skalierung).
FLOPs je Token und Lage: 2*d*h (erste Matrix) + 2*d*S (zweite); Auswahl/Ziehung (O(h)) nicht gezählt. Bytes beim Laden der zweiten Matrix: S/h.
Lernrate je Arm gewählt (Validierung, 3 Raten, ggf. verlängert). 2000 Schritte (Tabelle: 2500; Rechenbudget 45 min).
Aufruf: python3 erkundung/K10.py --seed 950 [--schnell]   ->   erkundung/roh/K10_s<seed>.json"""
import functools
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bytelm import LM, FFNDicht, bpc, cli, lauf_zellen  # noqa: E402

KID = "K10"


class FFNSpalten(nn.Module):
    def __init__(self, d, h, S, modus):
        super().__init__()
        self.a, self.b_gew, self.b_bias = nn.Linear(d, h), nn.Parameter(torch.empty(h, d)), nn.Parameter(torch.zeros(d))
        nn.init.kaiming_uniform_(self.b_gew, a=5 ** 0.5)
        self.h, self.S, self.modus, self.dicht = h, S, modus, False
        self.flops = 2 * d * h + 2 * d * S

    def forward(self, x):
        sh = x.shape
        a = F.relu(self.a(x.reshape(-1, sh[-1]))) ** 2                    # (N,h)
        if self.dicht:
            return (a @ self.b_gew + self.b_bias).reshape(sh)
        N, h, S = a.shape[0], self.h, self.S
        if self.modus == "top":
            w, idx = a.topk(S, dim=1)
        elif self.modus == "unif":
            idx = torch.randint(0, h, (N, S))
            w = a.gather(1, idx) * (h / S)
        else:
            with torch.no_grad():
                p = a * a + 1e-12
                p = 0.99 * p / p.sum(1, keepdim=True) + 0.01 / h
                c = p.cumsum(1)
                idx = torch.searchsorted(c, torch.rand(N, S) * c[:, -1:]).clamp(max=h - 1)
                pj = p.gather(1, idx)
            w = a.gather(1, idx) / (S * pj)
        # Simulation der dünnen Rechnung: Gewichte der gezogenen Spalten (Mehrfachziehungen addiert) in eine (N,h)-Matrix, dann dicht multipliziert.
        # Ergebnis und Gradienten sind identisch zur Rechnung auf nur S Spalten; gezählt werden die FLOPs der echten dünnen Rechnung (2*d*S).
        voll = torch.zeros_like(a).scatter_add(1, idx, w)
        return (voll @ self.b_gew + self.b_bias).reshape(sh)


def setze_dicht(netz, an):
    for b in netz.bl:
        if isinstance(b.ffn, FFNSpalten):
            b.ffn.dicht = an


def extra_dicht(netz, va, te, mess):
    if not any(isinstance(b.ffn, FFNSpalten) and b.ffn.modus in ("born", "unif") for b in netz.bl):
        return {}
    setze_dicht(netz, True)
    out = {}
    if "val" in mess:
        out["bpc_dicht_ausgewertet_val"] = bpc(netz, va)
    if "test" in mess:
        out["bpc_dicht_ausgewertet"] = bpc(netz, te)
    setze_dicht(netz, False)
    return out


def bauer(d, h, S, modus, seed):
    if modus == "dicht":
        return LM(d, lambda dd, i: FFNDicht(dd, h, "relu2"))
    return LM(d, lambda dd, i: FFNSpalten(dd, h, S, modus))


def main(seed: int, schnell: bool = False) -> dict:
    d, h = 128, 512
    if schnell:
        d, h, tt, tf = 64, 256, 30, 60
    else:
        tt, tf = 200, 2000
    arme = [("dicht_voll", h, 0, "dicht")]
    for q, nm in ((4, "S4"), (8, "S8")):
        S = h // q
        arme += [(f"schmal_{nm}", (h + S) // 2, 0, "dicht")]
    for q, nm in ((4, "S4"), (8, "S8")):
        S = h // q
        arme += [(f"born_{nm}", h, S, "born"), (f"top_{nm}", h, S, "top"), (f"unif_{nm}", h, S, "unif")]
    if schnell:
        arme = [a for a in arme if a[0] in ("dicht_voll", "schmal_S4", "born_S4", "top_S4", "unif_S4")]
    zellen = [(f"d{d}/{n}", functools.partial(bauer, d, hh, S, m), tt, tf, extra_dicht) for n, hh, S, m in arme]
    return lauf_zellen(KID, seed, schnell, zellen, __doc__)


if __name__ == "__main__":
    cli(KID, main)
