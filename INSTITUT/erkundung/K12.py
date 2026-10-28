"""K12 Bit-Baum / Hadamard-Code als Ausgabekopf (Erkundung, kein Befund).
Mechanismus: der 256-Wege-Softmax (2*256*d FLOPs je Token, gebunden an die Einbettung) wird ersetzt durch
  bittree  8 binäre Entscheidungen (MSB zuerst, 255 Knotenvektoren; bpc = Summe der 8 binären Kreuzentropien; Auswertung der Wahrscheinlichkeit des echten Bytes kostet 8*2*d FLOPs)
  had32    d -> 32 gelernt, dann feste Korrelation mit einem festen 32-Bit-Codebuch (256 Wörter, ±1; Softmax über die 256 Korrelationen, gelernte Temperatur)
           Codebuch: die 64 Hadamard-Wörter (±Zeilen von H32 = Reed-Muller RM(1,5)) plus 192 Wörter aus RM(2,5) nach Farthest-First (größter Mindestabstand)
  rand32   dasselbe mit zufälligem ±1-Codebuch (Kontrolle)
  rang32   d -> 32 -> 256 beides gelernt (Niedrigrang-Kopf, Kontrolle)
  voll     256-Wege-Softmax, gebundene Einbettung (Referenz)
FLOPs Kopf je Token: voll 2*256*d; bittree 16*d; had32/rand32 2*32*d + 2*32*256 (Korrelation als Additionen gezählt); rang32 2*32*(d+256). Parameter ausgewiesen (voll: gebunden, 256*d effektiv).
Wichtig: Kopf-FLOPs sind nur ein kleiner Teil der Gesamt-FLOPs (siehe flops_gesamt); die Gesamt-FLOPs-Differenz wird ausgewiesen, nicht zusätzlich in Breite zurückgegeben.
d=128, FFN 4d GELU, 4 Lagen. Lernrate je Arm gewählt. 4000 Schritte.
Aufruf: python3 erkundung/K12.py --seed 950 [--schnell]   ->   erkundung/roh/K12_s<seed>.json"""
import functools
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bytelm import LM, FFNDicht, cli, lauf_zellen  # noqa: E402

KID = "K12"


class KopfBitBaum(nn.Module):
    def __init__(self, d, emb):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.randn(255, d) * 0.02), nn.Parameter(torch.zeros(255))
        self.flops, self.eff_param = 16 * d, 255 * d + 255

    def nll(self, h, y):
        tot = 0.0
        for i in range(8):
            knoten = (1 << i) + (y >> (8 - i)) - 1                  # Präfix der ersten i Bits -> Knoten 0..254
            bit = ((y >> (7 - i)) & 1).float()
            logit = (h * self.w[knoten]).sum(-1) + self.b[knoten]
            tot = tot + F.binary_cross_entropy_with_logits(logit, bit, reduction="none")
        return tot


class KopfCode(nn.Module):
    def __init__(self, d, emb, code, lernbar=False):
        super().__init__()
        self.lin = nn.Linear(d, code.shape[1])
        self.log_s = nn.Parameter(torch.tensor(math.log(0.5)))
        self.register_buffer("code", code)
        self.flops = 2 * d * code.shape[1] + 2 * code.shape[1] * 256
        self.eff_param = d * code.shape[1] + code.shape[1] + 1

    def nll(self, h, y):
        z = self.lin(h)
        return F.cross_entropy(z @ self.code.T * self.log_s.exp(), y, reduction="none")


class KopfRang(nn.Module):
    def __init__(self, d, emb, r=32):
        super().__init__()
        self.a, self.b = nn.Linear(d, r, bias=False), nn.Linear(r, 256)
        self.flops = 2 * r * (d + 256)
        self.eff_param = r * (d + 256) + 256

    def nll(self, h, y):
        return F.cross_entropy(self.b(self.a(h)), y, reduction="none")


def codebuch_hadamard32():
    """64 Hadamard-Wörter + 192 aus RM(2,5) (Farthest-First). Rückgabe (256,32) ±1, Mindestabstand."""
    n = 32
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    x = torch.tensor([[(p >> k) & 1 for k in range(5)] for p in range(32)])             # 32 Punkte in {0,1}^5
    mon = [torch.ones(32, dtype=torch.long)] + [x[:, i] for i in range(5)] + [x[:, i] * x[:, j] for i in range(5) for j in range(i + 1, 5)]
    mon = torch.stack(mon, 1)                                                           # (32,16)
    msg = torch.tensor([[(m >> k) & 1 for k in range(16)] for m in range(1 << 16)])
    alle = 1.0 - 2.0 * ((msg @ mon.T) % 2).float()                                      # (65536,32) ±1
    wahl = [r for r in torch.cat([H, -H], 0)]
    md = torch.full((alle.shape[0],), 99.0)
    for w in wahl:
        md = torch.minimum(md, (32 - alle @ w) / 2)
    while len(wahl) < 256:
        j = int(md.argmax())
        wahl.append(alle[j])
        md = torch.minimum(md, (32 - alle @ alle[j]) / 2)
    C = torch.stack(wahl)
    dist = (32 - C @ C.T) / 2
    dist.fill_diagonal_(99)
    return C, float(dist.min())


def codebuch_zufall(seed):
    g = torch.Generator().manual_seed(seed + 31)
    C = torch.randint(0, 2, (256, 32), generator=g).float() * 2 - 1
    dist = (32 - C @ C.T) / 2
    dist.fill_diagonal_(99)
    return C, float(dist.min())


_HAD = {}


def bauer(d, art, seed):
    if art == "had32" and "c" not in _HAD:
        _HAD["c"] = codebuch_hadamard32()
    kopf = {"voll": None,
            "bittree": lambda d, e: KopfBitBaum(d, e),
            "had32": lambda d, e: KopfCode(d, e, _HAD["c"][0]),
            "rand32": lambda d, e: KopfCode(d, e, codebuch_zufall(seed)[0]),
            "rang32": lambda d, e: KopfRang(d, e)}[art]
    return LM(d, lambda dd, i: FFNDicht(dd, 4 * dd, "gelu"), kopf)


def main(seed: int, schnell: bool = False) -> dict:
    d, tt, tf = (64, 40, 80) if schnell else (128, 300, 4000)
    arten = ("voll", "bittree", "had32", "rand32", "rang32")
    zellen = [(f"d{d}/{a}", functools.partial(bauer, d, a), tt, tf, None) for a in arten]
    out = lauf_zellen(KID, seed, schnell, zellen, __doc__)
    out["codebuch_mindestabstand"] = {"had32": codebuch_hadamard32()[1], "rand32": codebuch_zufall(seed)[1]}
    from _bytelm import speichere
    speichere(KID, seed, schnell, out)
    return out


if __name__ == "__main__":
    cli(KID, main)
