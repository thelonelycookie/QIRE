"""K03 -- Vektor-Codebuch ohne Speicher (Erkundung Welle 2, 28.10.2026; kein Befund, nicht in LABORBUCH).
FRAGE: Ein trainiertes MNIST-MLP 784-512-512-10 wird OHNE Nachtraining gespeichert: nach Hadamard-Drehung (Vorzeichen aus Seed) werden Blöcke von 8 Gewichten
  (nach Zeilen-RMS-Normierung) durch das nächste Wort eines NIE gespeicherten Codebuchs (65 536 Wörter = 16 Bit je 8 Gewichte = 2 Bit/Gewicht) ersetzt.
  Schlägt das die Skalar-Quantisierung bei GLEICHEN Bytes?
CODEBÜCHER (alle aus Seed bzw. Formel, 0 Bytes; je auf RMS 1 pro Koordinate normiert, EIN globaler Verstärkungsfaktor g je Schicht aus {0,8; 0,9; 1,0; 1,1}, gewählt auf einer Probe von 3072 Blöcken per Fehlerminimum):
  zufall   65 536 Gauß-Wörter im R^8 (Seed)
  rm       Reed-Muller-Ebenen: p1 aus dem Gerade-Gewicht-Code RM(2,3) (128 Vorzeichenwörter) + r·p2 (alle 256 Vorzeichenwörter), r in {0,5; 0,75} = 7+8+1 = 16 Bit
  e8       die 65 536 kürzesten Gitterpunkte des E8-Gitters (Ränder der letzten Schale per Seed ausgelost)
  hex      A2-Sechseckgitter je Koordinatenpaar, 16 Punkte (4 Bit je Paar = 2 Bit/Gewicht), 4 Paare je Block unabhängig
KONTROLLEN (gleiche Bytes, bytes_q_linear(2 Bit, Rotation)): skalar4_had (echte 4 Stufen {-1,5;-0,5;0,5;1,5}·s mit Zeilenskala per MSE-Suche), skalar2_had (= I-048-Arm = quant_rows(…,2) = TERNÄR {-1,0,1}·s, 
  kb.quant_rows), jeweils auch ohne Rotation (roh); zufall/e8 ebenfalls ohne Rotation.
DATEN/TRAINING: MNIST 55 000 Train / 5 000 Val, 30 Epochen Adam lr 1e-3 (kb-Muster), fp32-Referenz mit. Gemessen: Val-Genauigkeit, relativer Gewichtsfehler (gewichtet) und relativer Logit-Fehler.
BYTES: 2 Bit/Gewicht + Zeilenskala fp16 + Bias fp16 + Vorzeichendiagonale (n/8 B) bei Rotation, MNIST-Eingang für Rotation auf 1024 aufgefüllt (kb.bytes_q_linear). fp16-Referenz des Modells ausgewiesen. FLOPs: gleiche MACs wie dicht (+ schnelle Hadamard bei Rotation).
Lies zuerst: acc der 2-Bit-Arme mit Rotation (zufall/rm/e8/hex) gegen skalar4_had und skalar2_had; dann Rotation gegen roh."""
from __future__ import annotations
import itertools
from _basis import *

BLOCK = 8
GRID_G = (0.8, 0.9, 1.0, 1.1)


# ------------------------------------------------------------------ Codebücher
def cb_zufall(K, seed):
    return torch.randn(K, BLOCK, generator=torch.Generator().manual_seed(seed * 17 + 1))


def cb_rm(K, seed):
    alle = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=BLOCK)))
    gerade = alle[(alle < 0).sum(1) % 2 == 0]                       # 128 Wörter, RM(2,3)
    ws = []
    for r in (0.5, 0.75):
        ws.append((gerade[:, None, :] + r * alle[None, :, :]).reshape(-1, BLOCK))
    return torch.cat(ws)                                             # 128*256*2 = 65536


def cb_e8(K, seed):
    pts = []
    rng = range(-3, 4)
    ganz = torch.tensor(list(itertools.product(rng, repeat=BLOCK)), dtype=torch.float32)
    ganz = ganz[(ganz.sum(1) % 2 == 0) & ((ganz ** 2).sum(1) <= 12)]
    rh = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    halb = torch.tensor(list(itertools.product(rh, repeat=BLOCK)), dtype=torch.float32)
    halb = halb[(halb.sum(1) % 2 == 0) & ((halb ** 2).sum(1) <= 12)]
    P = torch.cat([ganz, halb])
    n = (P ** 2).sum(1)
    g = torch.Generator().manual_seed(seed * 13 + 5)
    ordnung = torch.argsort(n * 1000 + torch.rand(len(n), generator=g))   # Norm, Rand per Seed
    return P[ordnung[:K]]


def cb_hex_paare():
    b1, b2 = torch.tensor([1.0, 0.0]), torch.tensor([0.5, math.sqrt(3) / 2])
    pts = [(a * b1 + b * b2) for a in range(-3, 4) for b in range(-3, 4)]
    P = torch.stack(pts)
    n = (P ** 2).sum(1)
    n1 = P[n.round() == 4.0 * 1.0] if False else None
    ordn = torch.argsort(n + 1e-3 * torch.atan2(P[:, 1], P[:, 0]))      # Norm, dann Winkel
    P = P[ordn]
    n = (P ** 2).sum(1)
    drei = P[(n - 4.0).abs() < 1e-4]                                      # 6 Punkte der Schale |p|²=4
    behalte = drei[::2]                                                   # alternierend: 3-zählige Symmetrie bleibt
    rest = P[n < 3.5]                                                     # 1+6+6 = 13
    return torch.cat([rest, behalte])                                     # 16


def normiere(C):
    return C / C.pow(2).mean().sqrt()


def nn_blocks(X, C, gs, chunk=1024):
    """X (m,d), C (K,d): je Verstärkung g der Index argmax_c (x·c − ½ g |c|²) und Gesamtfehler. Gibt (idx[len(gs), m], err[len(gs)])."""
    m = X.shape[0]
    cn = 0.5 * (C ** 2).sum(1)
    idx = torch.empty(len(gs), m, dtype=torch.long)
    err = torch.zeros(len(gs), dtype=torch.float64)
    xx = (X ** 2).sum(1)
    for a in range(0, m, chunk):
        xc = X[a:a + chunk] @ C.T
        for k, g in enumerate(gs):
            v, i = (xc - g * cn).max(1)
            idx[k, a:a + chunk] = i
            err[k] += float((xx[a:a + chunk] - 2 * g * v).sum())          # |x−g c|² = |x|² − 2g(x·c − ½g|c|²)
    return idx, err


def nn_eine(X, C, g, chunk=2048):
    """Index des nächsten Worts g·c je Block (ein einziger Durchlauf; Bias −½g|c|² in die Matrixmultiplikation eingebaut)."""
    Ca = torch.cat([C, (0.5 * g * (C ** 2).sum(1, keepdim=True))], 1).T.contiguous()     # (d+1, K)
    idx = torch.empty(len(X), dtype=torch.long)
    for a in range(0, len(X), chunk):
        xa = torch.cat([X[a:a + chunk], -torch.ones(len(X[a:a + chunk]), 1)], 1)
        idx[a:a + chunk] = (xa @ Ca).argmax(1)
    return idx


def quant_codebuch(Wr, C, hex_paar=False, probe=3072):
    """Wr (o,n) rotiert. Zeilen-RMS normieren, Blöcke zum nächsten Wort. Globales g je Schicht wird auf einer Zufallsprobe von `probe` Blöcken gewählt
    (Fehlerminimum über GRID_G), danach ein Durchlauf über alle Blöcke."""
    o, n = Wr.shape
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    d = 2 if hex_paar else BLOCK
    X = (Wr / rms).reshape(-1, d)
    sel = torch.randperm(len(X), generator=torch.Generator().manual_seed(len(X)))[:probe]
    _, err = nn_blocks(X[sel], C, GRID_G)
    g = GRID_G[int(err.argmin())]
    idx = nn_eine(X, C, g)
    return (g * C[idx]).reshape(o, n) * rms, g


def quant_skalar4(W):
    """Echte 4 Stufen {-1,5,-0,5,0,5,1,5}·s je Zeile, s per MSE-Suche."""
    amax = W.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, bs = None, None
    for f in np.linspace(0.3, 1.0, 15):
        s = amax * float(f) / 1.5
        Wq = ((W / s - 0.5).round() + 0.5).clamp(-1.5, 1.5) * s
        e = ((W - Wq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bs = e, Wq
        else:
            mk = e < best
            best, bs = torch.where(mk, e, best), torch.where(mk, Wq, bs)
    return bs


class QE(nn.Module):
    def __init__(self, lin, modus, rot, seed, codebuch=None, hexp=False):
        super().__init__()
        W = lin.weight.data.clone()
        n0 = W.shape[1]
        self.pad = (1024 - n0) if (rot and n0 == 784) else 0
        if self.pad:
            W = F.pad(W, (0, self.pad))
        self.Q = kb.rot_matrix(W.shape[1], seed) if rot else None
        Wr = W @ self.Q.T if rot else W
        g = None
        if modus == "skalar4":
            Wq = quant_skalar4(Wr)
        elif modus == "skalar2":
            Wq = kb.quant_rows(Wr, 2)
        else:
            Wq, g = quant_codebuch(Wr, codebuch, hexp)
        self.W, self.b, self.g, self.shape = Wq, lin.bias.data.clone(), g, tuple(W.shape)
        self.fehl = float(((Wq - Wr) ** 2).sum() / (Wr ** 2).sum())
        self.rot = rot

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (0, self.pad))
        if self.Q is not None:
            x = x @ self.Q.T
        return F.linear(x, self.W, self.b)

    def bytes(self):
        o, i = self.shape
        return kb.bytes_q_linear(o, i if i != 1024 else 784, 2, self.rot, pad=True) if self.rot else o * i * 2 / 8 + 4 * o


def ersetze(netz, modus, rot, seed, cb, hexp=False):
    ls = [QE(l, modus, rot, seed * 101 + k, cb, hexp) for k, l in enumerate(netz.ls)]
    q = kb.Netz(ls)
    return q, ls


def main(seed: int, schnell: bool = False) -> dict:
    ep = 3 if schnell else 30
    K = 4096 if schnell else 65536
    xtr, ytr, xva, yva = kb.bild_daten("mnist", schnell)[:4]
    torch.manual_seed(seed)
    netz = kb.baue_bild([512, 512])
    t0 = time.time()
    mlp_trainiere(netz, "mnist", seed, ep, lr=1e-3, klein=schnell)
    ref = acc(netz, xva, yva)
    with torch.no_grad():
        z0 = netz(xva)
    out = {"ep": ep, "K": K, "acc_fp32": ref, "bytes_fp16": kb.bytes_fp16_linear(netz), "sek_training": time.time() - t0, "arme": {}}
    print(f"K03 s{seed} fp32 {ref:.4f}", flush=True)
    cbs = {}
    def holen(name):
        if name not in cbs:
            if name == "hex":
                cbs[name] = normiere(cb_hex_paare())
            else:
                C = {"zufall": cb_zufall, "rm": cb_rm, "e8": cb_e8}[name](K, seed)
                if len(C) > K:
                    C = C[torch.randperm(len(C), generator=torch.Generator().manual_seed(seed))[:K]]
                cbs[name] = normiere(C)
        return cbs[name]
    plan = [("zufall_rot", "cb", "zufall", True), ("rm_rot", "cb", "rm", True), ("e8_rot", "cb", "e8", True), ("hex_rot", "hex", "hex", True),
            ("skalar4_rot", "skalar4", None, True), ("skalar2_rot (I-048, ternaer)", "skalar2", None, True),
            ("zufall_roh", "cb", "zufall", False), ("e8_roh", "cb", "e8", False), ("skalar4_roh", "skalar4", None, False), ("skalar2_roh", "skalar2", None, False)]
    for name, modus, cbn, rot in plan:
        t = time.time()
        C = holen(cbn) if cbn else None
        q, ls = ersetze(netz, "skalar4" if modus == "skalar4" else ("skalar2" if modus == "skalar2" else "cb"), rot, seed, C, hexp=(modus == "hex"))
        q.eval()
        with torch.no_grad():
            zq = q(xva)
        w = sum(l.shape[0] * l.shape[1] for l in ls)
        out["arme"][name] = {"acc": acc(q, xva, yva), "gewichtsfehler_rel": sum(l.fehl * l.shape[0] * l.shape[1] for l in ls) / w,
                             "logit_fehler_rel": float((zq - z0).norm() / z0.norm()), "bytes": sum(l.bytes() for l in ls),
                             "g_je_schicht": [l.g for l in ls], "sekunden": time.time() - t}
        print(f"K03 s{seed} {name}: acc {out['arme'][name]['acc']:.4f} Gew.fehler {out['arme'][name]['gewichtsfehler_rel']:.4f} Bytes {out['arme'][name]['bytes']:.0f} ({time.time()-t:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    rahmen("K03", main)
