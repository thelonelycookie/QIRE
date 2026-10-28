"""K15 Rotiertes Dropout und Schuss-Rauschen -- ERKUNDUNG, kein Befund.
Frage: Wirkt Dropout in gedrehter Basis (R = Walsh-Hadamard mal feste Zufallsvorzeichen, orthogonal) besser gegen Überanpassung als normales Dropout mit abgestimmter Rate?
Teil A Byte-LM (d=128, 4 Lagen, 4000 Schritte, Kontext 128): TRAININGSDATEN NUR 200.000 Bytes (Abweichung vom Auftrag "halbe Trainingsbytes" = 800 kB: dort überanpasst
  das Modell in 4000 Schritten kaum; 200 kB = ca. 20 Epochen), Val = dev (65536 Bytes), Haken hinter der GELU der FFN-Mitte (512 Kanäle) in allen Lagen.
  Arme: keine | Dropout p=0,1 / 0,3 | rotiertes Dropout p=0,1 / 0,3 (z = H D x / sqrt(n), Maske, Rückdrehung, Skala 1/(1-p)) | Gauß-Rauschen (Std = 0,5 x Effektivwert, EIN Punkt, nicht getunt).
  Ausgewiesen: Train-bpc und Val-bpc (beide Eval-Modus), Lücke, Parameter, Zusatz-FLOPs der Rotation (2 x 2 x 512² je Token und Lage nur im Training).
  Kontrolle = normales Dropout mit besserer der beiden Raten (Wahl auf Val; deshalb Val leicht optimistisch für ALLE Arme gleichermaßen).
Teil B MNIST und Fashion 16x16, 2000 Trainingsbilder (500 Val zur Ratenwahl, 5000 Test), MLP 256-512-10, Adam, 150 Epochen, Mittel über 3 Inits.
  Arme: keine | Dropout | rotiertes Dropout (je p in 0,2/0,5/0,7, beste auf Val) | Gauß (Std 0,5/1/2 x Effektivwert, beste) | Bernoulli-Schussrauschen auf Pixeln (x -> Bernoulli(x)).
Aufruf: python3 erkundung/K15.py --seed 950 [--schnell]   (aus INSTITUT/)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lm import *
from scipy.linalg import hadamard
from labor.gemeinsam import bilder

LR, T_LM, CTX, N_TRAIN = 1e-3, 4000, 128, 200_000


class Rausch(nn.Module):
    """art: 'drop' | 'rot' | 'gauss'. Nur im Training aktiv."""

    def __init__(self, n, art, p, seed):
        super().__init__()
        self.art, self.p = art, p
        if art == "rot":
            g = torch.Generator().manual_seed(seed)
            s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
            self.register_buffer("R", torch.tensor(hadamard(n), dtype=torch.float32) * s[:, None] / math.sqrt(n))  # orthogonal

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if self.art == "drop":
            return F.dropout(x, self.p, True)
        if self.art == "rot":
            z = x @ self.R
            return F.dropout(z, self.p, True) @ self.R.T
        return x + self.p * x.pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(x)      # gauss: p = relative Std


def lm_lauf(art, p, seed, T, tr, va, n_hid=512):
    fab = None
    if art != "keine":
        k = [0]
        def fab():
            k[0] += 1
            return Rausch(n_hid, art, p, seed * 10 + k[0])
    netz = LM(128, seed=seed, hid_fabrik=fab)
    opt = torch.optim.Adam(netz.parameters(), LR)
    g = torch.Generator().manual_seed(seed + 5)
    t0 = time.time()
    r = lauf_schritte(netz, opt, tr, g, [CTX] * T, LR, [i / T for i in range(T)])
    if r["instabil"]:
        return {"instabil": True}
    trb, vab = bpc(netz, tr[:65536]), bpc(netz, va)
    return {"instabil": False, "train_bpc": trb, "val_bpc": vab, "luecke": vab - trb, "param": n_param(netz), "param_ohne_emb": n_param(netz, True),
            "flops_train_basis_je_token": flops_train_token(128, 4, CTX),
            "flops_zusatz_je_token": 3 * 4 * 2 * 2 * n_hid * n_hid if art == "rot" else 0, "sek": time.time() - t0}


def teil_a(seed, schnell):
    T = 40 if schnell else T_LM
    tr, va, te = daten(klein=schnell)
    tr = tr[:(20_000 if schnell else N_TRAIN)]
    va = va[:8192] if schnell else va
    arme = [("keine", 0.0), ("drop", 0.1), ("drop", 0.3), ("rot", 0.1), ("rot", 0.3), ("gauss", 0.5)]
    if schnell:
        arme = [("keine", 0.0), ("drop", 0.3), ("rot", 0.3), ("gauss", 0.5)]
    out = {"T": T, "n_train_bytes": len(tr), "lr": LR, "arme": {}}
    for art, p in arme:
        e = lm_lauf(art, p, seed, T, tr, va)
        out["arme"][f"{art}_{p}"] = e
        print("A", art, p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
    ok = {k: v for k, v in out["arme"].items() if not v["instabil"]}
    bd = min((k for k in ok if k.startswith("drop")), key=lambda k: ok[k]["val_bpc"], default=None)
    br = min((k for k in ok if k.startswith("rot")), key=lambda k: ok[k]["val_bpc"], default=None)
    out["bestes_drop"], out["bestes_rot"] = bd, br
    if bd and br:
        out["diff_rot_minus_drop_val_bpc"] = ok[br]["val_bpc"] - ok[bd]["val_bpc"]
    if bd and "keine_0.0" in ok:
        out["diff_drop_minus_keine_val_bpc"] = ok[bd]["val_bpc"] - ok["keine_0.0"]["val_bpc"]
    return out


class MLP(nn.Module):
    def __init__(self, art, p, seed, n_in=256, h=512):
        super().__init__()
        torch.manual_seed(seed)
        self.a, self.b = nn.Linear(n_in, h), nn.Linear(h, 10)
        self.art, self.p = art, p
        self.rs = Rausch(h, art if art in ("drop", "rot", "gauss") else "drop", p if art in ("drop", "rot", "gauss") else 0.0, seed)

    def forward(self, x):
        if self.training and self.art == "shot":
            x = torch.bernoulli(x.clamp(0, 1))
        return self.b(self.rs(F.relu(self.a(x))))


def mlp_lauf(art, p, seed, daten_, epochen):
    xt, yt, xv, yv, xe, ye = daten_
    netz = MLP(art, p, seed)
    opt = torch.optim.Adam(netz.parameters(), 2e-3)
    g = torch.Generator().manual_seed(seed)
    n = len(xt)
    for _ in range(epochen):
        perm = torch.randperm(n, generator=g)
        netz.train()
        for i in range(0, n, 200):
            j = perm[i:i + 200]
            opt.zero_grad()
            F.cross_entropy(netz(xt[j]), yt[j]).backward()
            opt.step()
    netz.eval()
    with torch.no_grad():
        acc = lambda x, y: float((netz(x).argmax(1) == y).float().mean())
        return {"train": acc(xt, yt), "val": acc(xv, yv), "test": acc(xe, ye)}


def teil_b(seed, schnell):
    ep = 3 if schnell else 150
    out = {}
    for name in (("mnist",) if schnell else ("mnist", "fashion_mnist")):
        x, y, xe, ye = bilder(name, seed, 2500, 1000 if schnell else 5000)
        d_ = (x[:2000], y[:2000], x[2000:], y[2000:], xe, ye)
        res = {}
        arme = [("keine", 0.0)] + [(a, p) for a in ("drop", "rot") for p in (0.2, 0.5, 0.7)] + [("gauss", s) for s in (0.5, 1.0, 2.0)] + [("shot", 0.0)]
        if schnell:
            arme = [("keine", 0.0), ("drop", 0.5), ("rot", 0.5), ("gauss", 1.0), ("shot", 0.0)]
        for art, p in arme:
            r = [mlp_lauf(art, p, seed * 3 + k, d_, ep) for k in range(3)]
            res[f"{art}_{p}"] = {k: float(np.mean([q[k] for q in r])) for k in r[0]}
        best = lambda pre: max((k for k in res if k.startswith(pre)), key=lambda k: res[k]["val"])
        sel = {a: best(a) for a in ("keine", "drop", "rot", "gauss", "shot")}
        out[name] = {"alle": res, "gewaehlt_auf_val": sel,
                     "test": {a: res[k]["test"] for a, k in sel.items()},
                     "diff_rot_minus_drop_test": res[sel["rot"]]["test"] - res[sel["drop"]]["test"]}
        print("B", name, out[name]["test"], flush=True)
    return out


def main(seed: int, schnell: bool = False) -> dict:
    out = {"id": "K15", "seed": seed, "schnell": schnell}
    out["B_mlp"] = teil_b(seed, schnell)
    out["A_bytelm"] = teil_a(seed, schnell)
    out["lesart"] = "diff_rot_minus_drop: Val-bpc (A, negativ = rotiert besser) bzw. Test-Genauigkeit (B, positiv = rotiert besser). Ein Tuning-Seed, Erkundung."
    return out


if __name__ == "__main__":
    cli("K15", main, __doc__)
