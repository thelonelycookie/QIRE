"""E06 = Ergaenzungen I-048  E6 (nur Skalen/Bias nachtrainieren) + E12 (1-Bit-Netz mit Skalen-Nachtraining). Gleiche Maschinerie, daher ein Skript.
MLP 784-512-512-10, MNIST, 30 Epochen Adam lr 1e-3 (kb-Profil) -> dann nachtraeglich quantisiert:
  Bits b in {2 (E6), 1 (E12)}; 2 Bit = 3 Stufen wie kb; 1 Bit = Vorzeichen x Zeilenskala (Skala = mittlerer Betrag der Zeile).
  Codes q (ganzzahlig, in der gedrehten Basis) bleiben FEST; trainiert werden nur die ohnehin gespeicherten Zeilenskalen s (als s0*exp(theta)) und die Bias -> 0 Zusatzbytes.
Arme:
  ptq_roh / ptq_had         nachtraeglich, kein Nachtraining (Bezug)
  skala_roh / skala_had     Skalen + Bias nachtrainiert (5 Epochen; E12 zusaetzlich 15 Epochen), lr aus {1e-3,3e-3,1e-2} nach Val
  bias_had                  nur Bias nachtrainiert (Ablation)
  ste_roh / ste_had         Kontrolle „volles STE“: ALLE Gewichte per Fake-Quant weitertrainiert (nur 2 Bit; 5 Epochen, lr aus {1e-4,3e-4} nach Val; Skala laufend neu gesucht) -> mehr Freiheit, gleiche Bytes
  klein_fp16                kleines dichtes fp16-Modell gleicher Bytes (flach/zwei/trichter, 30 Epochen, lr 1e-3, Wahl nach Val) = haerteste Kontrolle fuer E12
Bytes: roh = Ziel ohne Vorzeichenvektor, had = I-048-Ziel (Bits 1/2). MNIST-Eingang 784->1024 aufgefuellt (nur gedreht).
Kosten Raven (geschaetzt): ~22 min je Seed. Aufruf: python3 erkundung/E06.py --seed 950 [--schnell]"""
import math
import numpy as np

import e_basis as eb
import E03
from e_basis import F, kb, nn, torch


class SkalaLin(nn.Module):
    """W_eff = diag(s) * A mit festem A = (q @ Q)[:, :i]; trainierbar: theta (log-Skala) und Bias."""

    def __init__(self, lin, bits, rot, seed, theta_train=True):
        super().__init__()
        W = lin.weight.data
        o, i = W.shape
        Wr, Q = eb.rotiere(W, "voll" if rot else None, seed)
        if bits == 1:
            s = Wr.abs().mean(1, keepdim=True)
            q = torch.sign(Wr + 1e-12)
        else:
            qmax = 2 ** (bits - 1) - 1
            f = E03.suche_f(Wr, bits)
            s = Wr.abs().amax(1, keepdim=True).clamp_min(1e-12) * f / qmax
            q = (Wr / s).round().clamp(-qmax, qmax)
        self.register_buffer("A", eb.zurueck(q, Q, i))
        self.register_buffer("s0", s)
        self.theta = nn.Parameter(torch.zeros(o, 1), requires_grad=theta_train)
        self.bias = nn.Parameter(lin.bias.data.clone())
        self.weight = None

    def forward(self, x):
        return F.linear(x, self.A * (self.s0 * torch.exp(self.theta)), self.bias)

    def w_eff(self):
        return (self.A * (self.s0 * torch.exp(self.theta))).detach()


def baue_skala(netz, bits, rot, seed, theta_train=True):
    return kb.Netz([SkalaLin(l, bits, rot, seed * 101 + k + 1, theta_train) for k, l in enumerate(netz.ls)])


def nachtrainiere(netz_fn, ds, seed, ep, lrs, klein, modus="skala"):
    """Trainiert fuer jede lr eine frische Kopie (netz_fn() liefert sie), nimmt die beste nach Val. Rueckgabe (netz, lr, val)."""
    best = None
    for lr in lrs:
        n = netz_fn()
        kb.trainiere_bild(n, ds, seed + 11, lr, ep, klein=klein)
        va = eb.acc(n, ds, klein, "val")
        if best is None or va > best[2]:
            best = (n, lr, va)
    return best


def bytes_arm(bits, rot):
    return float(sum(kb.bytes_q_linear(o, i, bits, rot, pad=rot) for o, i in ((512, 784), (512, 512), (10, 512))))


def main(seed, schnell=False):
    P = eb.profil(schnell)
    klein, ds, ep_nt = P["klein"], "mnist", P["ep_nt"]
    out = {"profil": P, "arme": {}}
    fp, info = eb.trainiere_mlp(ds, seed, (512, 512), P["ep"], 1e-3, klein)
    out["fp32"] = {"test": eb.acc(fp, ds, klein), "val": eb.acc(fp, ds, klein, "val")}
    lrs_s = (1e-3, 3e-3, 1e-2)
    lrs_ste = (1e-4, 3e-4)
    for bits in (2, 1):
        B = {"ziel_bytes": kb.ziel_bytes(ds, bits), "arme": {}}
        for rot, nm in ((False, "roh"), (True, "had")):
            sd = seed * 1000 + 7
            n0 = baue_skala(fp, bits, rot, sd)
            B["arme"][f"ptq_{nm}"] = {"test": eb.acc(n0, ds, klein), "val": eb.acc(n0, ds, klein, "val"), "bytes": bytes_arm(bits, rot)}
            n, lr, va = nachtrainiere(lambda: baue_skala(fp, bits, rot, sd), ds, seed, ep_nt, lrs_s, klein)
            B["arme"][f"skala_{nm}"] = {"test": eb.acc(n, ds, klein), "val": va, "lr": lr, "ep": ep_nt, "bytes": bytes_arm(bits, rot)}
            if bits == 1:           # E12: laenger nachtrainieren
                ep_l = 3 if schnell else 15
                n, lr2, va2 = nachtrainiere(lambda: baue_skala(fp, bits, rot, sd), ds, seed, ep_l, (lr,), klein)
                B["arme"][f"skala15_{nm}"] = {"test": eb.acc(n, ds, klein), "val": va2, "lr": lr2, "ep": ep_l, "bytes": bytes_arm(bits, rot)}
            if rot:
                n, lrb, vab = nachtrainiere(lambda: baue_skala(fp, bits, rot, sd, theta_train=False), ds, seed, ep_nt, lrs_s, klein)
                B["arme"][f"bias_{nm}"] = {"test": eb.acc(n, ds, klein), "val": vab, "lr": lrb, "ep": ep_nt, "bytes": bytes_arm(bits, rot)}
            if bits == 2:           # volles STE als Kontrolle mit mehr Freiheit
                best = None
                for lr_ in lrs_ste:
                    q = E03.baue_q(fp, bits, rot, sd)
                    E03.trainiere_q(q, ds, seed + 11, lr_, ep_nt, klein, "qat")
                    r = E03.auswerten_hart(q, ds, klein)
                    if best is None or r["val"] > best[1]["val"]:
                        best = (lr_, r)
                B["arme"][f"ste_{nm}"] = {**best[1], "lr": best[0], "ep": ep_nt}
            eb.lg(f"[E06] {bits}Bit {nm}: " + ", ".join(f"{k.split('_')[0]} {v['test']:.4f}" for k, v in B["arme"].items() if k.endswith(nm)))
        B["klein_fp16"] = eb.klein_bester(ds, B["ziel_bytes"], seed, P["ep"], klein, formen=("flach", "zwei", "trichter") if not schnell else ("flach",), lrs=(1e-3,))
        a = B["arme"]
        B["vergleich_pp"] = {"skala_had_minus_ptq_had": 100 * (a["skala_had"]["test"] - a["ptq_had"]["test"]),
                             "skala_had_minus_skala_roh": 100 * (a["skala_had"]["test"] - a["skala_roh"]["test"]),
                             "skala_had_minus_klein": 100 * (a["skala_had"]["test"] - B["klein_fp16"]["test"]),
                             "skala_had_minus_ste_had": (100 * (a["skala_had"]["test"] - a["ste_had"]["test"])) if "ste_had" in a else None}
        out[f"bits{bits}"] = B
    return out


if __name__ == "__main__":
    eb.cli("E06", main)
