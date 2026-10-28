"""K03B -- Zusatzarme zu K03: GELERNTES k-Means-Codebuch (Erkundung Welle 3, 28.10.2026; kein Befund, nicht in LABORBUCH).
K03.py bleibt unverändert; Bausteine (normiere, nn_eine, quant_skalar4, QE, Netz-Ersatz) werden importiert. Gleiche Daten (MNIST 55 000/5 000), gleicher Seed, gleiches Training
(30 Epochen Adam 1e-3, --schnell 3), gleiche Rotation (kb.rot_matrix, Seeds seed*101+k, Eingang 784->1024), Zeilen-RMS-Normierung und Bytes-Zählung (2 Bit/Gewicht,
Zeilenskala+Bias fp16, Vorzeichendiagonale bei Rotation = kb.bytes_q_linear). NEU: die Bytes des Codebuchs werden mitgezählt (fp16, K·Blockgröße·2 B).
ARME (alle 2 Bit/Gewicht; EIN Buch für alle drei Schichten, per Lloyd-k-Means auf den normierten, rotierten Blöcken der trainierten Gewichte; Verstärkung g = 1):
  km8_rot        Blöcke à 8, 65 536 Wörter (wie K03-Bücher) -- Buch 1 MB fp16 > Modell; bei ~99 000 Blöcken fast Auswendiglernen. Nur zur Einordnung.
  km4_K256_rot    Blöcke à 4, 256 Wörter (Buch 2 KB)  <- fairer 2-Bit-Arm mit Buchbytes
  km2_K16_rot     Blöcke à 2, 16 Wörter (Buch 64 B)
  km4_K256_roh    wie km4_K256_rot ohne Rotation
ANKER (identisch zu K03): skalar4_rot, skalar2_rot (ternär). Formelbücher (zufall/rm/e8/hex) stehen in roh/K03_s<seed>.json.
AUSGABE wie K03 (acc, gewichtsfehler_rel, logit_fehler_rel, bytes, g_je_schicht, sekunden); zusätzlich bytes_buch und bytes_ohne_buch; `bytes` ist inklusive Buch.
Lies zuerst: acc/bytes von km4_K256_rot und km2_K16_rot gegen skalar4_rot; dann km8_rot (K im Feld "K" (Buch mitgezählt) gegen skalar4_rot."""
from __future__ import annotations
from _basis import *
import K03 as K


def kmeans(X, Kw, iters, seed, chunk=256):
    """Lloyd-k-Means (Mini-Matrixprodukte, leere Cluster werden neu aus zufälligen Blöcken besetzt). Gibt (K, d)-Buch."""
    g = torch.Generator().manual_seed(seed)
    Kw = min(Kw, len(X))
    C = X[torch.randperm(len(X), generator=g)[:Kw]].clone()
    for _ in range(iters):
        cn = 0.5 * (C ** 2).sum(1)
        idx = torch.empty(len(X), dtype=torch.long)
        for a in range(0, len(X), chunk):
            idx[a:a + chunk] = (X[a:a + chunk] @ C.T - cn).argmax(1)
        cnt = torch.bincount(idx, minlength=Kw).float()
        S = torch.zeros_like(C).index_add_(0, idx, X)
        leer = cnt == 0
        C = torch.where(leer[:, None], C, S / cnt.clamp_min(1)[:, None])
        if leer.any():
            C[leer] = X[torch.randint(0, len(X), (int(leer.sum()),), generator=g)]
    return C


class QK(nn.Module):
    """Wie K03.QE, aber Gewichte werden von aussen gesetzt (Buch vorher gelernt)."""

    def __init__(self, lin, rot, seed):
        super().__init__()
        W = lin.weight.data.clone()
        self.pad = (1024 - W.shape[1]) if (rot and W.shape[1] == 784) else 0
        if self.pad:
            W = F.pad(W, (0, self.pad))
        self.Q = kb.rot_matrix(W.shape[1], seed) if rot else None
        self.Wr = W @ self.Q.T if rot else W
        self.b, self.shape, self.rot, self.W, self.fehl, self.g = lin.bias.data.clone(), tuple(W.shape), rot, None, None, 1.0

    def setze(self, Wq):
        self.W = Wq
        self.fehl = float(((Wq - self.Wr) ** 2).sum() / (self.Wr ** 2).sum())

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (0, self.pad))
        if self.Q is not None:
            x = x @ self.Q.T
        return F.linear(x, self.W, self.b)

    def bytes(self):
        o, i = self.shape
        return kb.bytes_q_linear(o, i if i != 1024 else 784, 2, self.rot, pad=True) if self.rot else o * i * 2 / 8 + 4 * o


def km_ersetze(netz, blk, Kw, rot, seed, iters):
    ls = [QK(l, rot, seed * 101 + k) for k, l in enumerate(netz.ls)]
    rms = [l.Wr.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12) for l in ls]
    Xs = [(l.Wr / r).reshape(-1, blk) for l, r in zip(ls, rms)]
    C = kmeans(torch.cat(Xs), Kw, iters, seed * 31 + blk)
    for l, r, X in zip(ls, rms, Xs):
        idx = K.nn_eine(X, C, 1.0, chunk=256)
        l.setze(C[idx].reshape(l.shape) * r)
    return kb.Netz(ls), ls, C


def main(seed: int, schnell: bool = False) -> dict:
    ep = 3 if schnell else 30
    K8 = 4096 if schnell else 65536
    it8 = 4 if schnell else 6
    xtr, ytr, xva, yva = kb.bild_daten("mnist", schnell)[:4]
    torch.manual_seed(seed)
    netz = kb.baue_bild([512, 512])
    t0 = time.time()
    mlp_trainiere(netz, "mnist", seed, ep, lr=1e-3, klein=schnell)
    ref = acc(netz, xva, yva)
    with torch.no_grad():
        z0 = netz(xva)
    out = {"ep": ep, "K": K8, "acc_fp32": ref, "bytes_fp16": kb.bytes_fp16_linear(netz), "sek_training": time.time() - t0, "arme": {}}
    print(f"K03B s{seed} fp32 {ref:.4f}", flush=True)

    def eintrag(name, q, ls, t, buch_bytes, gs):
        q.eval()
        with torch.no_grad():
            zq = q(xva)
        w = sum(l.shape[0] * l.shape[1] for l in ls)
        b0 = sum(l.bytes() for l in ls)
        out["arme"][name] = {"acc": acc(q, xva, yva), "gewichtsfehler_rel": sum(l.fehl * l.shape[0] * l.shape[1] for l in ls) / w,
                             "logit_fehler_rel": float((zq - z0).norm() / z0.norm()), "bytes": b0 + buch_bytes, "bytes_ohne_buch": b0, "bytes_buch": buch_bytes,
                             "g_je_schicht": gs, "sekunden": time.time() - t}
        print(f"K03B s{seed} {name}: acc {out['arme'][name]['acc']:.4f} Gew.fehler {out['arme'][name]['gewichtsfehler_rel']:.4f} Bytes {b0 + buch_bytes:.0f} (Buch {buch_bytes}) ({time.time()-t:.0f}s)", flush=True)

    for name, modus in (("skalar4_rot", "skalar4"), ("skalar2_rot (I-048, ternaer)", "skalar2")):
        t = time.time()
        q, ls = K.ersetze(netz, modus, True, seed, None)
        eintrag(name, q, ls, t, 0, [None] * 3)
    plan = [("km8_rot", 8, K8, True, it8), ("km4_K256_rot", 4, 256, True, 12), ("km2_K16_rot", 2, 16, True, 12), ("km4_K256_roh", 4, 256, False, 12)]
    for name, blk, Kw, rot, it in plan:
        t = time.time()
        q, ls, C = km_ersetze(netz, blk, Kw, rot, seed, it)
        eintrag(name, q, ls, t, int(C.shape[0] * blk * 2), [1.0] * 3)
    return out


if __name__ == "__main__":
    rahmen("K03B", main)
