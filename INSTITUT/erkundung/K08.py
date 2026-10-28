"""K08 (Erkundung, kein Befund): Adam in gedrehter Basis. Byte-LM d=128, 4 Lagen (kb.LM), gleiche Initialisierung und Batches in allen Armen.
Der Gradient jeder 2D-Gewichtsmatrix G (o×i) wird gedreht, G' = L G Rᵀ, Adam rechnet elementweise auf G', der Schritt wird zurueckgedreht, U = Lᵀ u' R. Gewichte selbst bleiben unverdreht.
1D-Parameter (LayerNorm, Bias) laufen normal. Drehung je Seite: Block-Diagonale aus Hadamard-Bloecken (Blockbreite = groesste Zweierpotenz, die die Dimension teilt, <= 512; 384 -> 3×128)
mit festen Zufalls-Vorzeichen (Seed).
Arme:   adam (ohne Drehung) | had (L, R = Block-Hadamard·Vorzeichen) | vz (nur Zufalls-Vorzeichen, OHNE Mischung: rechnerisch identisch mit adam, Kontrolle) | orth (dichte Zufalls-Orthogonale, Haar-verteilt, gleiche Mischung, n² Operationen)
Lernrate JE ARM getunt: Gitter {1e-3, 2e-3, 4e-3} auf kurzem Lauf (T_tune, Val-bpc, Kosinus ueber T_tune); danach voller Lauf mit der besten Rate (vz uebernimmt die Rate von adam).
Mass: Test-bpc je Trainings-FLOP. Trainings-FLOPs je Schritt = 6·MAC·Tokens (Vorwaerts/Rueckwaerts) + Adam (10 je Parameter) + Drehung: Hadamard ideal (schnelle Transformation, 4 Transformationen je Matrix und Schritt:
Drehen+Zurueckdrehen, beide Seiten, Additionen), orth dicht (2·o·i·(o+i) je Drehung·2). Gemessen werden zusaetzlich Sekunden je Schritt (Drehung im Programm dicht gerechnet).
T_tune 600, T 2500 (schnell: 30/60, klein).
Aufruf: python3 erkundung/K08.py --seed 950 [--schnell]"""
import math
import time

import _hilfe_k0508 as H
import torch

kb, ss = H.kb, H.ss
LRS = (1e-3, 2e-3, 4e-3)
T_TUNE_VOLL, T_VOLL = 600, 2500
T_TUNE_SCHNELL, T_SCHNELL = 30, 60


def blockbreite(n):
    b = 1
    while n % (b * 2) == 0 and b * 2 <= 512:
        b *= 2
    return b


def dreh(n, art, seed):
    """Orthogonale n×n-Matrix: art 'had' = blockdiag(Hadamard)·diag(±1), 'orth' = Zufalls-Orthogonale."""
    if art == "orth":
        return kb.orth_matrix(n, seed)
    b = blockbreite(n)
    Hb = kb.hadamard(b)
    s = (torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)) * 2 - 1).float()
    return torch.block_diag(*[Hb] * (n // b)) * s[None, :]


class AdamRot:
    def __init__(self, netz, art, seed):
        self.art = art
        self.p = [p for p in netz.parameters()]
        self.m = [torch.zeros_like(p) for p in self.p]
        self.v = [torch.zeros_like(p) for p in self.p]
        self.t = 0
        self.L, self.R = [], []
        for k, p in enumerate(self.p):
            if art != "adam" and p.ndim == 2:
                if art == "vz":
                    g = lambda n, off: (torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed * 7919 + k * 2 + off)) * 2 - 1).float()
                    self.L.append(g(p.shape[0], 0))
                    self.R.append(g(p.shape[1], 1))
                else:
                    self.L.append(dreh(p.shape[0], art, seed * 7919 + k * 2))
                    self.R.append(dreh(p.shape[1], art, seed * 7919 + k * 2 + 1))
            else:
                self.L.append(None)
                self.R.append(None)

    def step(self, lr):
        self.t += 1
        b1, b2 = 0.9, 0.999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        with torch.no_grad():
            for k, p in enumerate(self.p):
                if p.grad is None:
                    continue
                g = p.grad
                L, R = self.L[k], self.R[k]
                if L is not None:
                    g = (L[:, None] * g * R[None, :]) if self.art == "vz" else (L @ g @ R.T)
                self.m[k] = b1 * self.m[k] + (1 - b1) * g
                self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
                u = (self.m[k] / c1) / ((self.v[k] / c2).sqrt() + 1e-8)
                if L is not None:
                    u = (L[:, None] * u * R[None, :]) if self.art == "vz" else (L.T @ u @ R)
                p.sub_(lr * u)


def flops_zusatz(netz, art):
    """Zusatz-FLOPs je Schritt (ideal): Adam 10 je Parameter; Drehung: had 4 Transformationen · (o·i·(log2 bo + log2 bi)) Additionen, orth dicht, vz 0 (Vorzeichen: 2·o·i)."""
    z = 10 * H.n_param(netz)
    for p in netz.parameters():
        if art != "adam" and p.ndim == 2:
            o, i = p.shape
            if art == "had":
                z += 2 * (o * i * int(math.log2(blockbreite(o))) + o * i * int(math.log2(blockbreite(i))))
            elif art == "orth":
                z += 2 * (2 * o * o * i + 2 * o * i * i)
            else:
                z += 4 * o * i
    return int(z)


def lauf(art, seed, lr, T, klein):
    netz, inf = H.lm_trainiere(seed, T, lr, lambda n: AdamRot(n, art, seed), klein=klein)
    return netz, inf


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    klein = schnell
    T = T_SCHNELL if schnell else T_VOLL
    Tt = T_TUNE_SCHNELL if schnell else T_TUNE_VOLL
    arme = ("adam", "had", "orth")
    tuning, beste = {}, {}
    for a in arme:
        tuning[a] = {}
        for lr in LRS:
            netz, inf = lauf(a, seed, lr, Tt, klein)
            tuning[a][str(lr)] = None if inf["instabil"] else H.bpc(netz, klein, "val")
            print(f"  tune {a} lr {lr}: val {tuning[a][str(lr)]}", flush=True)
        ok = {float(k): v for k, v in tuning[a].items() if v is not None}
        beste[a] = min(ok, key=ok.get)
    beste["vz"] = beste["adam"]
    flops0, mac = H.lm_flops_schritt(kb.LM(128, 4))
    zeilen = {}
    for a in ("adam", "had", "vz", "orth"):
        netz, inf = lauf(a, seed, beste[a], T, klein)
        fl_schritt = flops0 + flops_zusatz(netz, a)
        z = {"lr": beste[a], "info": inf, "sekunden": inf["sekunden"], "sek_je_schritt": inf["sekunden"] / T, "flops_je_schritt": fl_schritt, "flops_gesamt": fl_schritt * T,
             "flops_zusatz_je_schritt": fl_schritt - flops0}
        if not inf["instabil"]:
            z["val_bpc"] = H.bpc(netz, klein, "val")
            z["test_bpc"] = H.bpc(netz, klein, "test")
        zeilen[a] = z
        print(f"  {a}: test {z.get('test_bpc')} lr {beste[a]} {z['sekunden']:.0f} s", flush=True)
    ref = zeilen["adam"].get("test_bpc")
    r = {"kid": "K08", "seed": seed, "schnell": schnell, "T": T, "T_tune": Tt, "klein": klein, "parameter": H.n_param(kb.LM(128, 4)), "mac_je_token": mac,
         "flops_je_schritt_ohne_zusatz": flops0, "tuning_val_bpc": tuning, "beste_lr": beste, "arme": zeilen,
         "abstand_zu_adam_bpc": {a: (v["test_bpc"] - ref) if (ref is not None and "test_bpc" in v) else None for a, v in zeilen.items()},
         "vz_gleich_adam_bpc_diff": (zeilen["vz"].get("test_bpc") - ref) if (ref is not None and "test_bpc" in zeilen["vz"]) else None}
    r["zusammenfassung"] = "\n".join(f"{a}: test {v.get('test_bpc')} bpc, lr {v['lr']}, {v['flops_gesamt']:.3e} FLOPs, {v['sek_je_schritt']:.3f} s/Schritt" for a, v in zeilen.items())
    r["sekunden"] = time.time() - t0
    return r


if __name__ == "__main__":
    H.cli("K08", main)
