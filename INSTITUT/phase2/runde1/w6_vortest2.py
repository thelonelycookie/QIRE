#!/usr/bin/env python3
"""W6-Vortest 2 (Phase 2, Runde 1): PTQ eines kleinen Byte-LM mit Vektorquantisierern. Erkundung, kein Befund.
NUR Tuning-Seed 950 (Z3-Konvention: Tuning-Seeds 950-989). Bewertet wird auf der Validierung (dev), nicht auf dem Test.
Start:  OMP_NUM_THREADS=4 python3 w6_vortest2.py train [T]   (Byte-LM trainieren, Checkpoint im Scratchpad)
        OMP_NUM_THREADS=4 python3 w6_vortest2.py ptq          (alle Arme, schreibt W6_vortest2.json/.md)
Modell wie Z3: Byte-LM d=128, 4 Lagen, FFN 512, Kontext 128 (UD_German-GSD), Adam lr 2e-3, Z3_bytegleich.lm_bauen2/lm_trainiere.
Quantisiert werden ALLE nn.Linear-Gewichte (qkv, proj, ffn.a, ffn.b je Lage = 16 Matrizen, 786432 Gewichte).
Nicht quantisiert (alle Arme gleich, fp16): Einbettung (gebunden), Positionen, LayerNorm, Biases.
"""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(INST, "erkundung")); sys.path.insert(0, INST); sys.path.insert(0, HERE)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import w6_vortest as W6                       # E8Book, orbit_lloyd, gen_group, group_sets, lloyd1d, argmin_dist
from Z3_bytegleich import lm_bauen2, lm_trainiere, lm_val
from _basis import fwht
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
SEED = 950
CK = os.environ.get("W6V2_CK", "/tmp/claude-0/-home-user-QIRE/c85d0af3-9103-522e-a686-8bafbfdddd9a/scratchpad/lm_s950.pt")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
def f16(x): return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)

# ------------------------------------------------------------------ Training
def train(T):
    torch.manual_seed(SEED)
    n = lm_bauen2(128, 512)
    r = lm_trainiere(n, SEED, T)
    torch.save({"sd": n.state_dict(), "T": T, "val_bpc": r["bpc"], "sek": r["sekunden"]}, CK)
    log(f"trainiert T={T}: val-bpc {r['bpc']:.4f}, {r['sekunden']:.0f}s")

def lade():
    c = torch.load(CK)
    n = lm_bauen2(128, 512); n.load_state_dict(c["sd"]); n.eval()
    return n, c

def lins(n): return [(nm, m) for nm, m in n.named_modules() if isinstance(m, nn.Linear)]

# ------------------------------------------------------------------ Rotation
def vorz(i, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (i,), generator=g) * 2 - 1).float().numpy()
def rot(W, s): return fwht(torch.from_numpy(W * s)).numpy()
def unrot(Y, s): return fwht(torch.from_numpy(Y)).numpy() * s

# ------------------------------------------------------------------ Quantisierer
def lloyd_q(Z, bits, rng):
    """Lloyd-1D-Buch (fp16) auf Z (beliebige Form), liefert (Zq, Buch)."""
    z = Z.ravel(); sub = z if len(z) <= 200000 else z[rng.choice(len(z), 200000, replace=False)]
    cb = f16(W6.lloyd1d(sub.astype(np.float64), 2 ** bits, 60)); cb.sort()
    t = (cb[1:] + cb[:-1]) / 2
    return cb[np.searchsorted(t, z)].reshape(Z.shape), cb

def q_lloyd_matrix(W, bits, g, rng):
    """Skala je Gruppe von g Eingaengen (g=None: Zeilenskala), RMS, fp16; Lloyd-Buch je Matrix."""
    o, i = W.shape; g = g or i
    Wg = W.reshape(o * (i // g), g)
    sc = f16(np.sqrt((Wg ** 2).mean(1, keepdims=True)) + 1e-12)
    Zq, cb = lloyd_q(Wg / sc, bits, rng)
    return (Zq * sc).reshape(o, i), dict(scale_bits=16 * sc.size, cb_bits=16 * len(cb), cb=cb, sc=sc)

def q_uniform_group(W, bits, g):
    """Gleichgitter (Mittelpunkte), Schritt je Gruppe fp16, Schritt per MSE-Suche."""
    o, i = W.shape; Wg = W.reshape(o * (i // g), g); L = 2 ** bits; hi = L / 2 - .5
    amax = np.abs(Wg).max(1, keepdims=True) + 1e-12
    best = None
    for f in np.linspace(0.3, 1.0, 15):
        s = f16(amax * f / (L / 2)); Q = np.clip(np.round(Wg / s - .5) + .5, -hi, hi) * s
        e = ((Wg - Q) ** 2).sum(1, keepdims=True)
        if best is None: best, bQ = e, Q
        else:
            m = e < best; best = np.where(m, e, best); bQ = np.where(m, Q, bQ)
    return bQ.reshape(o, i), dict(scale_bits=16 * Wg.shape[0], cb_bits=0)

def blocks(Z, d): return Z.reshape(-1, d)

class Arm:
    def __init__(self, name, bits, desc): self.name, self.bits, self.desc = name, bits, desc

def snr_db(W, Wq): return float(10 * np.log10((W ** 2).sum() / ((W - Wq) ** 2).sum()))

def run_arm(kind, bits, Ws, names, rng, ctx):
    """Liefert (Wq-Liste, Buchhaltung, Zeiten, Dekoder-Info). Ws: Liste float32-Matrizen."""
    nW = sum(w.size for w in Ws); acct = dict(index_bits=0, scale_bits=0, cb_bits=0, sign_bits=0)
    t_fit = t_enc = 0.0; Wq = []; info = {}
    rotated = kind.startswith("had_")
    signs = [vorz(w.shape[1], SEED * 101 + k + 1) for k, w in enumerate(Ws)] if rotated else None
    if kind in ("lloyd1d", "had_lloyd1d", "g16_lloyd"):
        g = 16 if kind == "g16_lloyd" else None
        for k, W in enumerate(Ws):
            t = time.time(); X = rot(W, signs[k]) if rotated else W
            Q, a = q_lloyd_matrix(X, bits, g, rng); t_fit += time.time() - t   # Buch+Skala+Zuordnung gemeinsam gemessen
            Wq.append(unrot(Q, signs[k]) if rotated else Q)
            acct["index_bits"] += bits * W.size; acct["scale_bits"] += a["scale_bits"]; acct["cb_bits"] += a["cb_bits"]
            if rotated: acct["sign_bits"] += W.shape[1]
        info["decoder"] = "1d"
    elif kind == "g16_uniform":
        for k, W in enumerate(Ws):
            t = time.time(); Q, a = q_uniform_group(W, bits, 16); t_enc += time.time() - t
            Wq.append(Q); acct["index_bits"] += bits * W.size; acct["scale_bits"] += a["scale_bits"]
        info["decoder"] = "g16u"
    elif kind in ("had_e8", "had_e8_exakt", "had_orbit8", "had_orbit4"):
        d = 4 if kind == "had_orbit4" else 8
        t = time.time(); Xs = []; scs = []
        for k, W in enumerate(Ws):
            X = rot(W, signs[k]); sc = f16(np.sqrt((X ** 2).mean(1, keepdims=True)) + 1e-12)
            Xs.append(X / sc); scs.append(sc); acct["scale_bits"] += 16 * sc.size; acct["sign_bits"] += W.shape[1]
        B = np.concatenate([blocks(x, d) for x in Xs]).astype(np.float32)
        acct["index_bits"] = bits * W6.np.int64(B.shape[0] * d)
        t_prep = time.time() - t
        if kind in ("had_e8", "had_e8_exakt"):
            bk = W6.E8Book(8 * bits); sub = B[np.random.default_rng(1).choice(len(B), 20000, replace=False)].astype(np.float64)
            t = time.time()
            gr = (np.sqrt(5) - 1) / 2; a_, b_ = 0.15, 1.2; c = b_ - gr * (b_ - a_); dd = a_ + gr * (b_ - a_); fc, fd = bk.mse(sub, c), bk.mse(sub, dd)
            for _ in range(9):
                if fc < fd: b_, dd, fd = dd, c, fc; c = b_ - gr * (b_ - a_); fc = bk.mse(sub, c)
                else: a_, c, fc = c, dd, fd; dd = a_ + gr * (b_ - a_); fd = bk.mse(sub, dd)
            D = float(f16((a_ + b_) / 2)); t_fit = time.time() - t; acct["cb_bits"] = 16; info["D"] = D
            if kind == "had_e8":
                t = time.time(); Bq = np.concatenate([bk.enc(B[s:s + 20000].astype(np.float64), D) for s in range(0, len(B), 20000)]); t_enc = time.time() - t
            else:
                tab = ctx["e8tab"](bk) * D
                t = time.time(); idx = W6.argmin_dist(B, tab.astype(np.float32), 1500); Bq = tab[idx]; t_enc = time.time() - t
                info["n_punkte"] = len(tab)
        else:
            if kind == "had_orbit8": gname, n_q, m_ = "D8^n lokal (H,X,Z je Qubit)", 3, 64 if bits == 2 else None
            else: gname, n_q, m_ = "Pauli+H^n", 2, (4 if bits == 2 else 64)
            Gm = W6.gen_group(W6.group_sets(n_q)[gname]); K = 2 ** (d * bits); m_ = K // len(Gm)
            ntr = 20000 if d == 8 else 60000
            t = time.time(); rg = np.random.default_rng(2)
            V, _ = W6.orbit_lloyd(Gm, m_, B[rg.choice(len(B), ntr, replace=False)].astype(np.float64), rg, 15 if d == 8 else 20)
            V = f16(V); C = np.einsum('gij,mj->mgi', Gm, V).reshape(m_ * len(Gm), d).astype(np.float32); t_fit = time.time() - t
            acct["cb_bits"] = 16 * V.size; info.update(gruppe=gname, G=len(Gm), m=m_, n_punkte=len(C))
            t = time.time(); idx = W6.argmin_dist(B, C, 1500 if d == 8 else 4000); Bq = C[idx]; t_enc = time.time() - t
            tab = C
        info["decoder"] = "vq"; info["tab"] = (tab if kind != "had_e8" else ctx["e8tab_cached"]) if True else None
        info["d"] = d; info["idx_bits"] = bits * d
        # zurueck in Matrizen
        pos = 0
        for k, W in enumerate(Ws):
            n_ = W.size // d; Q = Bq[pos:pos + n_].reshape(W.shape) * scs[k]; pos += n_
            Wq.append(unrot(Q.astype(np.float32), signs[k]))
        t_fit += t_prep
    acct["index_bits"] = int(acct["index_bits"]); tot = sum(acct.values())
    return Wq, dict(**acct, total_bits=int(tot), bpw=tot / nW, bpw_index=acct["index_bits"] / nW,
                    bpw_scale=acct["scale_bits"] / nW, bpw_cb=acct["cb_bits"] / nW), dict(fit_s=t_fit, enc_s=t_enc), info

def e8_table(bk):
    """Alle Punkte des E8-Buchs (Regel von E8Book.inside) als Tabelle (N,8); fuer die Dekodierung per Lookup."""
    K = bk.K; out = []
    for half in (False, True):
        vals = np.array([-2.5, -1.5, -.5, .5, 1.5, 2.5]) if half else np.arange(-3., 4.)
        nv = len(vals); n = nv ** 8
        for s in range(0, n, 2_000_000):
            ii = np.arange(s, min(n, s + 2_000_000)); q = np.empty((len(ii), 8))
            for j in range(8): q[:, j] = vals[(ii // nv ** j) % nv]
            ok = (q.sum(1) % 2 == 0) & ((q * q).sum(1) <= 2 * K)
            q = q[ok]; q = q[bk.inside(q)]
            out.append(q)
    return np.concatenate(out).astype(np.float32)

# ------------------------------------------------------------------ Dekodierzeit
def med_us(fn, rep=300, warm=30):
    for _ in range(warm): fn()
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e6)

def decode_bench(kind, bits, Wq_ref, W, info, rng):
    """Dekodierzeit fuer eine Matrix-Vektor-Multiplikation y = W x, W = ffn.a (512 x 128), CPU, 1 Thread, torch-eager.
    Indizes zufaellig (Zeit haengt nicht von den Werten ab); Speicher: 1D-Indizes gepackt (2 Bit: 4/Byte, 1 Bit: 8/Byte), 3 Bit ungepackt uint8, VQ als uint16-Index.
    Gibt (decode_us, matvec_us, gesamt_us, fp32_matvec_us)."""
    torch.set_num_threads(1)
    o, i = W.shape; x = torch.randn(i); Wt = torch.from_numpy(W)
    ref = med_us(lambda: Wt @ x)
    s = torch.from_numpy(vorz(i, 1)); rotd = kind.startswith("had_")
    sc = torch.rand(o, 1) + .5
    if info["decoder"] in ("1d", "g16u"):
        g = 16 if kind in ("g16_lloyd", "g16_uniform") else i
        per = 8 // bits if bits in (1, 2) else 1
        P = torch.randint(0, 256, (o * i // per,), dtype=torch.uint8); sh = torch.arange(per, dtype=torch.uint8) * bits
        cb = torch.randn(2 ** bits); scale = torch.rand(o * i // g, 1) + .5
        def dec():
            if per > 1: idx = ((P[:, None] >> sh) & (2 ** bits - 1)).reshape(-1).long()
            else: idx = P.long() % (2 ** bits)
            return (cb[idx].reshape(-1, g) * scale).reshape(o, i) if info["decoder"] == "1d" else ((idx.reshape(-1, g).float() - (2 ** bits - 1) / 2) * scale).reshape(o, i)
    else:
        d = info["d"]; tab = torch.from_numpy(info["tab"]) if info.get("tab") is not None else None
        if tab is None: return None
        P = torch.randint(0, len(tab), (o * i // d,), dtype=torch.int32).to(torch.int64)
        def dec(): return (tab[P].reshape(o, i) * sc)
    Wd = dec()
    if rotd:
        mv = lambda: Wd @ fwht(x * s)
        full = lambda: dec() @ fwht(x * s)
    else:
        mv = lambda: Wd @ x
        full = lambda: dec() @ x
    return dict(decode_us=med_us(dec), matvec_us=med_us(mv), gesamt_us=med_us(full), fp32_matvec_us=ref)

# ------------------------------------------------------------------ Haupt
def ptq():
    net, c = lade(); L = lins(net)
    Ws = [m.weight.data.numpy().astype(np.float32).copy() for _, m in L]; names = [n for n, _ in L]
    nW = sum(w.size for w in Ws); log("Matrizen", len(Ws), "Gewichte", nW, "fp32-val-bpc", c["val_bpc"])
    res = dict(seed=SEED, T=c["T"], n_gewichte=nW, matrizen=[(n, list(w.shape)) for n, w in zip(names, Ws)],
               fp32_val_bpc=float(lm_val(net)), arme={})
    orig = [w.copy() for w in Ws]
    # fp16-Gewichte als Zusatzreferenz
    def setze(Wq):
        n2, _ = lade()
        for (nm, m), w in zip(lins(n2), Wq): m.weight.data = torch.from_numpy(np.asarray(w, np.float32))
        return n2
    res["fp16_gewichte_val_bpc"] = float(lm_val(setze([f16(w) for w in orig])))
    log("fp32", res["fp32_val_bpc"], "fp16", res["fp16_gewichte_val_bpc"])
    rng = np.random.default_rng(SEED); ctx = {}
    ARME = [("lloyd1d", 2), ("g16_lloyd", 1), ("g16_uniform", 1), ("had_lloyd1d", 2), ("had_e8", 2), ("had_e8_exakt", 2), ("had_orbit8", 2), ("had_orbit4", 2),
            ("lloyd1d", 3), ("g16_lloyd", 2), ("g16_uniform", 2), ("had_lloyd1d", 3), ("had_e8", 3), ("had_orbit4", 3)]
    only = os.environ.get("ARME")
    for kind, bits in ARME:
        key = f"{kind}_b{bits}"
        if only and key not in only.split(","): continue
        if kind == "had_e8_exakt" or (kind == "had_e8" and bits == 2):
            if "e8tab_val" not in ctx:
                t = time.time(); bk = W6.E8Book(16); ctx["e8tab_val"] = e8_table(bk); log("E8-Tabelle", len(ctx["e8tab_val"]), f"{time.time()-t:.0f}s")
            ctx["e8tab"] = lambda bk: ctx["e8tab_val"]; ctx["e8tab_cached"] = ctx["e8tab_val"]
        elif kind == "had_e8": ctx["e8tab_cached"] = None
        Wq, acct, tm, info = run_arm(kind, bits, orig, names, rng, ctx)
        net2 = setze(Wq); bpc = float(lm_val(net2))
        err = {"snr_db_gepoolt": snr_db(np.concatenate([w.ravel() for w in orig]), np.concatenate([w.ravel() for w in Wq])),
               "snr_db_mittel_matrizen": float(np.mean([snr_db(a, b) for a, b in zip(orig, Wq)]))}
        dec = decode_bench(kind, bits, None, orig[2], info, rng)   # L[2] = Lage 0 ffn.a? siehe unten
        r = dict(bits_index=bits, val_bpc=bpc, delta_bpc_vs_fp32=bpc - res["fp32_val_bpc"], **acct, **err, **tm, decode=dec,
                 info={k: v for k, v in info.items() if k not in ("tab",)})
        res["arme"][key] = r
        log(f"{key:18s} bpw {acct['bpw']:.3f} (idx {acct['bpw_index']:.2f} skala {acct['bpw_scale']:.3f} buch {acct['bpw_cb']:.4f}) snr {err['snr_db_gepoolt']:.2f} dB  bpc {bpc:.4f}  fit {tm['fit_s']:.1f}s enc {tm['enc_s']:.1f}s  dec {dec and round(dec['decode_us'])}us")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    # E8: Heuristik gegen exakte Suche auf einer Stichprobe wird in den Armen had_e8 (heuristisch) und had_e8_exakt sichtbar.
    res["laufzeit_s"] = time.time() - T0
    json.dump(res, open(os.path.join(HERE, "W6_vortest2.json"), "w"), indent=1, default=str)
    log("json geschrieben")

def bench():
    """Dekodierzeit neu messen (ruhig, bester von 5 Medianlaeufen je 300 Wdh.), Ergebnis in die JSON (Feld decode)."""
    p = os.path.join(HERE, "W6_vortest2.json"); res = json.load(open(p)); rg = np.random.default_rng(0)
    W = rg.standard_normal((512, 128)).astype(np.float32)
    e8 = rg.standard_normal((65633, 8)).astype(np.float32)
    for key, r in res["arme"].items():
        kind, bits = key.rsplit("_b", 1); bits = int(bits)
        if kind in ("lloyd1d", "had_lloyd1d", "g16_lloyd"): info = {"decoder": "1d"}
        elif kind == "g16_uniform": info = {"decoder": "g16u"}
        elif kind.startswith("had_e8"): info = {"decoder": "vq", "d": 8, "tab": e8 if bits == 2 else None}
        elif kind == "had_orbit8": info = {"decoder": "vq", "d": 8, "tab": rg.standard_normal((65536, 8)).astype(np.float32)}
        else: info = {"decoder": "vq", "d": 4, "tab": rg.standard_normal((2 ** (4 * bits), 4)).astype(np.float32)}
        ds = [decode_bench(kind, bits, None, W, info, rg) for _ in range(5)]
        if ds[0] is None: r["decode"] = None; continue
        r["decode"] = {k: min(d[k] for d in ds) for k in ds[0]}
        log(key, {k: round(v, 1) for k, v in r["decode"].items()})
    json.dump(res, open(p, "w"), indent=1, default=str)


if __name__ == "__main__":
    if sys.argv[1] == "train": train(int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
    elif sys.argv[1] == "bench": bench()
    else: ptq()
