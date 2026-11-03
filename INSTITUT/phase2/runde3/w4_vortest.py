#!/usr/bin/env python3
"""W4-Vortest (Phase 2, Runde 3): Rotation im Gradientenpfad beim 4-Bit-Training eines Byte-LM. Erkundung, kein Befund.
NUR Tuning-Seed 950. Bewertet auf Val (dev), nie Test. Rezept wie runde2/w2_vortest.py (Byte-LM d=128, 4 Lagen, FFN 512, Adam 2e-3, Clip 1,0).
Start: OMP_NUM_THREADS=2 python3 w4_vortest.py [T1,T2] [arm1,arm2,..]   (Standard 3000,8000; schreibt W4_vortest.json, Wiederaufnahme je Punkt)
Alle nn.Linear der Bloecke (qkv, proj, ffn.a, ffn.b) sind Fake-Quant-4-Bit (symmetrisch int4 -8..7, Block 32, absmax-Skala, STE):
  Vorwaerts: Gewicht (Block 32 ueber Eingaenge) und Aktivierung (Block 32 ueber Merkmale) RTN.
  Rueckwaerts: dgrad = dy_fp @ Wq (Ausgabegradient nicht quantisiert);
               Wgrad = Q(dy)^T @ Q(x) mit Kontraktion ueber die N=Batch*Kontext=1024 Token, Bloecke von 32 Token (MX-artig);
               dy per stochastischem Runden (wie in der FP4-Literatur ueblich), x (bereits Q-Aktivierung aus dem Vorwaertspass) wird erneut tokenweise-blockig quantisiert.
Arme: roh (ohne Rotation) | had_fest (Hadamard 32 ueber Tokenbloecke, vor Quantisierung von dy und x, Wgrad exakt invariant) |
      had_zuf (Hadamard * neue Zufallsvorzeichen je Schritt UND Schicht) | clip (ohne Rotation, MSE-optimale Clipping-Skala je Block aus {0.55,0.75,1}*absmax) | fp32 (Referenz).
Zusatz-Arm nur_vor: W4A4-Vorwaerts, Wgrad exakt (trennt Vorwaerts- von Gradientenquantisierung).
Messungen: Val-bpc, nichtendliche Verluste (Abbruch), Gradientennorm vor Clip (Median/p99/Max), Verlustspruenge, ms/Schritt,
  relativer Fehler des quantisierten Wgrad gegen exaktes Wgrad (alle 250 Schritte, Mittel ueber Schichten).
"""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(INST, "erkundung")); sys.path.insert(0, INST)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from Z3_bytegleich import lm_bauen2, lm_val
import abteilungen.struktur_sprache as ss
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
SEED = 950
JS = os.path.join(HERE, "W4_vortest.json")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

BLK = 32
H32 = None
def had_matrix():
    global H32
    if H32 is None:
        H = torch.ones(1, 1)
        while H.shape[0] < BLK: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H32 = H / math.sqrt(BLK)
    return H32

def blk0(x):   # (N, C) -> (N/32, 32, C): Bloecke laengs der Zeilen (Token)
    return x.reshape(x.shape[0] // BLK, BLK, x.shape[1])

CLIPS = (0.55, 0.75, 1.0)
def quant_blk(xb, dim, mode="rtn", clip=False):
    """Fake-Quant int4 (-8..7) mit Blockskala entlang `dim` (Bloecke sind bereits ausgeklappt). mode rtn | sr."""
    amax = xb.abs().amax(dim, keepdim=True).clamp_min(1e-30)
    if clip:
        best, bs = None, None
        for c in CLIPS:
            s = amax * c / 7.0
            q = (xb / s).round().clamp(-8, 7) * s
            e = ((q - xb) ** 2).sum(dim, keepdim=True)
            if best is None: best, bs = e, s
            else:
                m = e < best; best = torch.where(m, e, best); bs = torch.where(m, s, bs)
        s = bs
    else:
        s = amax / 7.0
    z = xb / s
    z = (z + torch.rand_like(z) - 0.5 if mode == "sr" else z)
    z = torch.floor(z + 0.5) if mode == "sr" else z.round()
    return z.clamp(-8, 7) * s

def q_rows(x, clip=False):     # Block 32 entlang der letzten Achse (Merkmale/Eingaenge)
    sh = x.shape; xb = x.reshape(-1, sh[-1] // BLK, BLK)
    return quant_blk(xb, -1, "rtn", clip).reshape(sh)

class QFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, arm, mod):
        sh = x.shape; x2 = x.reshape(-1, sh[-1])
        clip = arm == "clip"
        xq = q_rows(x2, clip); Wq = q_rows(W, clip)
        ctx.save_for_backward(xq, Wq); ctx.arm, ctx.mod = arm, mod
        return (xq @ Wq.T + b).reshape(*sh[:-1], -1)
    @staticmethod
    def backward(ctx, dy):
        xq, Wq = ctx.saved_tensors; arm, mod = ctx.arm, ctx.mod
        dy2 = dy.reshape(-1, dy.shape[-1]); N = dy2.shape[0]
        dx = (dy2 @ Wq).reshape(*dy.shape[:-1], -1)      # dgrad: dy nicht quantisiert
        db = dy2.sum(0)
        if arm == "nur_vor":
            dW = dy2.T @ xq
        else:
            A, B = dy2, xq                                  # Wgrad = A^T B, Kontraktion ueber Token
            if arm in ("had_fest", "had_zuf"):
                if arm == "had_zuf":
                    s = (torch.randint(0, 2, (N, 1)) * 2 - 1).float(); A, B = A * s, B * s
                H = had_matrix()
                A = torch.einsum("ij,njc->nic", H, blk0(A)).reshape(N, -1)   # H symmetrisch & orthogonal
                B = torch.einsum("ij,njc->nic", H, blk0(B)).reshape(N, -1)
            clip = arm == "clip"
            Aq = quant_blk(blk0(A), 1, "sr", clip).reshape(N, -1)           # Ausgabegradient: SR
            Bq = quant_blk(blk0(B), 1, "rtn", clip).reshape(N, -1)
            dW = Aq.T @ Bq
        if mod.messen:
            ex = dy2.T @ xq
            mod.fehler = float(((dW - ex).norm() / ex.norm().clamp_min(1e-30)))
        return dx, dW, db, None, None

class QLin(nn.Linear):
    arm = "roh"; messen = False; fehler = float("nan")
    def forward(self, x): return QFn.apply(x, self.weight, self.bias, self.arm, self)

def umbauen(net, arm):
    for _, m in net.named_modules():
        if isinstance(m, nn.Linear):
            m.__class__ = QLin; m.arm = arm; m.messen = False; m.fehler = float("nan")
    return net
def lins(n): return [m for m in n.modules() if isinstance(m, nn.Linear)]

ARME = ["fp32", "roh", "had_fest", "had_zuf", "clip", "nur_vor"]

def trainiere(arm, T, lr=2e-3):
    tr, va, _ = ss.daten(False)
    torch.manual_seed(SEED); net = lm_bauen2(128, 512)
    if arm != "fp32": umbauen(net, arm)
    g = torch.Generator().manual_seed(SEED + 5)
    opt = torch.optim.Adam(net.parameters(), lr=lr); warm = max(1, T // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, T - warm))))
    net.train(); offs = torch.arange(ss.KONTEXT); t0 = time.time()
    gn, losses, fehl, kurve = [], [], [], []; nonfin = 0; diverg_schritt = None
    for s in range(T):
        mess = arm != "fp32" and arm != "nur_vor" and s % 250 == 0
        if mess:
            for m in lins(net): m.messen = True
        i = torch.randint(0, len(tr) - ss.KONTEXT - 1, (ss.BATCH,), generator=g)
        x, y = tr[i[:, None] + offs].long(), tr[i[:, None] + offs + 1].long()
        l = F.cross_entropy(net(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(l): nonfin += 1; diverg_schritt = s; break
        opt.zero_grad(set_to_none=True); l.backward()
        n = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        if not math.isfinite(n): nonfin += 1; diverg_schritt = s; break
        if mess:
            fehl.append((s, float(np.mean([m.fehler for m in lins(net)]))))
            for m in lins(net): m.messen = False
        gn.append(n); losses.append(float(l)); opt.step(); sch.step()
        if s % 250 == 249: kurve.append((s + 1, float(np.mean(losses[-250:])) / math.log(2)))
    sek = time.time() - t0
    gn_a = np.array(gn) if gn else np.array([float("nan")]); ls = np.array(losses) if losses else np.array([float("nan")])
    # Verlustspruenge: Verlust > 1,5 x gleitendes Median der letzten 50 Schritte (nach Aufwaermen)
    spr = 0
    for k in range(max(50, T // 20), len(ls)):
        if ls[k] > 1.5 * np.median(ls[k - 50:k]): spr += 1
    net.eval()
    bpc = 8.0 if diverg_schritt is not None else float(lm_val(net))
    return dict(arm=arm, T=T, val_bpc=bpc, diverg_schritt=diverg_schritt, nichtendlich=nonfin, verlustspruenge=spr,
        gn_median=float(np.median(gn_a)), gn_p99=float(np.percentile(gn_a, 99)), gn_max=float(gn_a.max()),
        gn_median_letztes_viertel=float(np.median(gn_a[-max(1, len(gn_a) // 4):])),
        wgrad_relfehler=fehl, wgrad_relfehler_mittel=float(np.mean([f for _, f in fehl])) if fehl else None,
        ms_je_schritt=sek / max(1, len(gn)) * 1000, sekunden=sek, kurve_train_bpc=kurve, schritte=len(gn))

def main(Ts, arme):
    res = json.load(open(JS)) if os.path.exists(JS) else dict(seed=SEED, punkte={})
    for T in Ts:
        for arm in arme:
            key = f"{arm}_T{T}"
            if key in res["punkte"]: continue
            r = trainiere(arm, T); res["punkte"][key] = r
            res["laufzeit_s"] = time.time() - T0; json.dump(res, open(JS, "w"), indent=1)
            log(f"{key:16s} val {r['val_bpc']:.4f} diverg {r['diverg_schritt']} spruenge {r['verlustspruenge']} gn med/p99/max {r['gn_median']:.2f}/{r['gn_p99']:.2f}/{r['gn_max']:.2f} wgrad-err {r['wgrad_relfehler_mittel']} {r['ms_je_schritt']:.0f} ms/Schritt")

if __name__ == "__main__":
    Ts = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "3000,8000").split(",")]
    arme = (sys.argv[2].split(",") if len(sys.argv) > 2 else ARME)
    main(Ts, arme)
