"""K16 Trotter-Stapel: Fastfood-Schichten mit Skip -- ERKUNDUNG, kein Befund.
Frage: Hilft Tiefe bei Fastfood-Schichten (x + alpha*GELU(H D2 H D1 x), alpha anfangs 0), wo sie ohne Skip schadet? Schlägt der Stapel dichte Netze bei gleichen Operationen?
Daten: MNIST und Fashion-MNIST 16x16 (256 Merkmale), 8000 Training, 2000 Val (nur Wahl der Lernrate aus 1e-3/3e-3/1e-2), 5000 Test (offizielle Testbilder). 20 Epochen, Batch 128, Adam.
Fastfood-Lage: y = H D2 H D1 x (H = normierte Walsh-Hadamard-Matrix 256x256, orthogonal, D1/D2 trainierte Diagonalen, Start +-1) + Bias, dann GELU. Ausgang 256 -> linearer Kopf 256 -> 10.
ARME
  S1        1 Fastfood-Lage, Kopf
  S3        3 Lagen hintereinander ohne Skip, Kopf
  S3skip    3 Lagen mit Skip x + alpha*GELU(...), alpha je Lage trainiert, Start 0
  S3skip_zufall  wie S3skip, aber Diagonalen fest zufällig (nicht trainiert) -- Kontrolle: bringt Lernen der Diagonalen etwas?
  Dicht1_F  dichtes Netz 256 -> h -> 10 (GELU), h so gewählt, dass die ideal gezählten FLOPs (Vorwärts) denen von S3skip entsprechen
  Dicht3skip dichtes 3-Lagen-Residualnetz 256-256 mit Skip (alpha Start 0), Kopf  (viel mehr FLOPs: Obergrenze der Architektur)
FLOPs ideal: Walsh-Hadamard 256 = 256*8 Additionen, je Lage 2 WHT + 2 Diagonalen (je 256) + Bias 256 + Aktivierung (4*256) ; dicht Linear n_in*n_out*2 + Bias; Kopf 2*256*10+10. Parameter ausgewiesen.
Aufruf: python3 erkundung/K16.py --seed 950 [--schnell]   (aus INSTITUT/)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lm import *
from scipy.linalg import hadamard
from labor.gemeinsam import bilder

N = 256
LRS = (1e-3, 3e-3, 1e-2)
H_ = torch.tensor(hadamard(N), dtype=torch.float32) / math.sqrt(N)


class FFLage(nn.Module):
    def __init__(self, gen, trainiert=True, skip=False):
        super().__init__()
        d1 = torch.randint(0, 2, (N,), generator=gen).float() * 2 - 1
        d2 = torch.randint(0, 2, (N,), generator=gen).float() * 2 - 1
        if trainiert:
            self.d1, self.d2 = nn.Parameter(d1), nn.Parameter(d2)
        else:
            self.register_buffer("d1", d1); self.register_buffer("d2", d2)
        self.bias = nn.Parameter(torch.zeros(N))
        self.alpha = nn.Parameter(torch.zeros(1)) if skip else None

    def forward(self, x):
        z = (((x * self.d1) @ H_) * self.d2) @ H_ + self.bias
        z = F.gelu(z)
        return x + self.alpha * z if self.alpha is not None else z


class Dichtlage(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(N, N)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * F.gelu(self.l(x))


class Netz(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.arm = arm
        if arm == "S1":
            self.k = nn.ModuleList([FFLage(gen)]); n_out = N
        elif arm == "S3":
            self.k = nn.ModuleList([FFLage(gen) for _ in range(3)]); n_out = N
        elif arm == "S3skip":
            self.k = nn.ModuleList([FFLage(gen, True, True) for _ in range(3)]); n_out = N
        elif arm == "S3skip_zufall":
            self.k = nn.ModuleList([FFLage(gen, False, True) for _ in range(3)]); n_out = N
        elif arm == "Dicht3skip":
            self.k = nn.ModuleList([Dichtlage() for _ in range(3)]); n_out = N
        elif arm == "Dicht1_F":
            h = dicht1_breite()
            self.k = nn.ModuleList([nn.Linear(N, h), nn.GELU()]); n_out = h
        else:
            raise ValueError(arm)
        self.kopf = nn.Linear(n_out, 10)

    def forward(self, x):
        for m in self.k:
            x = m(x)
        return self.kopf(x)


def flops_ideal(arm, h=None):
    wht = N * int(math.log2(N))
    ff_lage = 2 * wht + 2 * N + N + 4 * N + (3 * N if "skip" in arm else 0)       # 2 WHT, 2 Diagonalen, Bias, GELU, Skip (alpha*z + x)
    kopf = lambda n: 2 * n * 10 + 10
    if arm == "S1": return ff_lage + kopf(N)
    if arm == "S3": return 3 * ff_lage + kopf(N)
    if arm in ("S3skip", "S3skip_zufall"): return 3 * ff_lage + kopf(N)
    if arm == "Dicht3skip": return 3 * (2 * N * N + N + 4 * N + 3 * N) + kopf(N)
    if arm == "Dicht1_F":
        h = h or dicht1_breite()
        return 2 * N * h + h + 4 * h + kopf(h)
    raise ValueError(arm)


def dicht1_breite():
    ziel = flops_ideal("S3skip")
    return max(1, min(range(1, 400), key=lambda h: abs((2 * N * h + h + 4 * h + 2 * h * 10 + 10) - ziel)))


def lauf(arm, lr, seed, d, epochen):
    xt, yt, xv, yv, xe, ye = d
    netz = Netz(arm, seed)
    opt = torch.optim.Adam(netz.parameters(), lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochen):
        perm = torch.randperm(len(xt), generator=g)
        netz.train()
        for i in range(0, len(xt), 128):
            j = perm[i:i + 128]
            opt.zero_grad()
            l = F.cross_entropy(netz(xt[j]), yt[j])
            l.backward()
            opt.step()
    netz.eval()
    with torch.no_grad():
        acc = lambda x, y: float((netz(x).argmax(1) == y).float().mean())
        return {"train": acc(xt, yt), "val": acc(xv, yv), "test": acc(xe, ye), "alpha": [float(m.alpha) for m in netz.k if getattr(m, "alpha", None) is not None]}


ARME = ("S1", "S3", "S3skip", "S3skip_zufall", "Dicht1_F", "Dicht3skip")


def main(seed: int, schnell: bool = False) -> dict:
    ep = 2 if schnell else 20
    out = {"id": "K16", "seed": seed, "schnell": schnell, "epochen": ep, "dicht1_breite": dicht1_breite(), "daten": {}}
    for name in (("mnist",) if schnell else ("mnist", "fashion_mnist")):
        x, y, xe, ye = bilder(name, seed, 10000, 1000 if schnell else 5000)
        d_ = (x[:8000], y[:8000], x[8000:], y[8000:], xe, ye)
        res = {}
        for arm in ARME:
            lrs = (3e-3,) if schnell else LRS
            versuche = {lr: lauf(arm, lr, seed, d_, ep) for lr in lrs}
            lr = max(versuche, key=lambda k: versuche[k]["val"])
            r = dict(versuche[lr]); r["lr"] = lr
            n_ = Netz(arm, seed)
            r["param"] = int(sum(p.numel() for p in n_.parameters())); r["flops_ideal_vorwaerts"] = flops_ideal(arm)
            res[arm] = r
            print(name, arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        out["daten"][name] = {"arme": res,
                              "diff_S3skip_minus_S1": res["S3skip"]["test"] - res["S1"]["test"],
                              "diff_S3_minus_S1": res["S3"]["test"] - res["S1"]["test"],
                              "diff_S3skip_minus_Dicht1_F": res["S3skip"]["test"] - res["Dicht1_F"]["test"],
                              "diff_S3skip_minus_zufall": res["S3skip"]["test"] - res["S3skip_zufall"]["test"],
                              "diff_S3skip_minus_Dicht3skip": res["S3skip"]["test"] - res["Dicht3skip"]["test"]}
    out["lesart"] = "diff in Testgenauigkeit (positiv = erster Arm besser). Eine Seed-Runde, Lernrate je Arm auf Val gewählt. Erkundung."
    return out


if __name__ == "__main__":
    cli("K16", main, __doc__)
