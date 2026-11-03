#!/usr/bin/env python3
"""W2-Vortest (Phase 2, Runde 2): Rate-Guete-Kurve (bpc gegen Bytes inkl. Skalen) im Byte-LM. Erkundung, kein Befund.
NUR Tuning-Seed 950. Bewertet auf Val (dev), nie Test. Rezept/Quantisierer aus ../runde1/w6_vortest2.py.
Start: OMP_NUM_THREADS=2 python3 w2_vortest.py [T1,T2]    (Standard 1000,3000; schreibt W2_vortest.json, Wiederaufnahme je Punkt)
Arme: lloyd_row (Zeilenskala fp16), had_lloyd (Hadamard+Zeilenskala), g16s8 / g32s8 (Gruppenskala 8 Bit, Lloyd-Buch je Matrix),
      klein_fp16 (Mittelmodell: kleineres Modell fp16, Breite aus Byte-Budget des 3-Bit-Punkts).
Aufgabenmetrik (S-009): Kopier-Test. Zufaellige Kleinbuchstaben-Folge A (L=40) + Leerzeichen + A; gemessen wird bpc und
Top-1-Treffer in der ZWEITEN Haelfte (nur durch Kopieren/Abruf erreichbar; erste Haelfte = Rauschen als Kontrolle).
"""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "runde1"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import w6_vortest2 as V
from w6_vortest2 import f16, rot, unrot, vorz, lloyd_q, lins, snr_db
from Z3_bytegleich import lm_bauen2, lm_trainiere, lm_val
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
SEED = 950
SCR = os.environ.get("W2V_SCR", "/tmp/claude-0/-home-user-QIRE/c85d0af3-9103-522e-a686-8bafbfdddd9a/scratchpad")
JS = os.path.join(HERE, "W2_vortest.json" if os.environ.get("W2V_MIX", "1") == "0" else "W2_vortest_mix.json")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- Aufgabenmetrik: Kopieren/Abruf
def kopier_daten(n=256, L=40, seed=7001):
    g = np.random.default_rng(seed)
    A = g.integers(97, 123, (n, L))
    return torch.from_numpy(np.concatenate([A, np.full((n, 1), 32), A], 1)).long()   # (n, 2L+1)

@torch.no_grad()
def kopier_test(net, X):
    net.eval(); L = (X.shape[1] - 1) // 2
    lg = net(X[:, :-1]).float(); lp = F.log_softmax(lg, -1)
    nll = -lp.gather(-1, X[:, 1:, None])[..., 0]               # (n, 2L) Ziel = Position 1..2L
    acc = (lg.argmax(-1) == X[:, 1:]).float()
    a1, a2 = slice(0, L), slice(L + 1, 2 * L)                    # erste Haelfte / zweite Haelfte (ohne Trenner)
    b = 1 / math.log(2)
    return dict(kopie_bpc=float(nll[:, a2].mean() * b), kopie_acc=float(acc[:, a2].mean()),
                kontrolle_bpc=float(nll[:, a1].mean() * b), kontrolle_acc=float(acc[:, a1].mean()))

# ---------------------------------------------------------------- Quantisierer
def q_s8(sc):
    """Skala auf 8 Bit (log-gleichmaessig zwischen Min und Max der Matrix; Min/Max fp16 = 32 Bit je Matrix)."""
    l = np.log2(sc); lo, hi = float(l.min()), float(l.max())
    q = np.round((l - lo) / max(hi - lo, 1e-9) * 255); return (2 ** (lo + q / 255 * (hi - lo))).astype(np.float32)

def quant_matrix(W, bits, g, skala, rotiert, k, rng):
    """g=None: Zeilenskala; skala 16 oder 8 Bit. Liefert (Wq, buchhaltung_bits)."""
    o, i = W.shape; s = vorz(i, SEED * 101 + k + 1) if rotiert else None
    X = rot(W, s) if rotiert else W
    g = g or i; Xg = X.reshape(o * (i // g), g)
    sc = np.sqrt((Xg ** 2).mean(1, keepdims=True)) + 1e-12
    sc = f16(sc) if skala == 16 else q_s8(sc)
    Zq, cb = lloyd_q(Xg / sc, bits, rng)
    Q = (Zq * sc).reshape(o, i)
    a = dict(index=bits * W.size, scale=skala * sc.size + (32 if skala == 8 else 0), cb=16 * len(cb), sign=i if rotiert else 0)
    return (unrot(Q, s) if rotiert else Q).astype(np.float32), a

ARME = {  # name: (g, skala_bits, rotiert), Indexbits je Rate
    "lloyd_row": (None, 16, False), "had_lloyd": (None, 16, True), "g16s8": (16, 8, False), "g32s8": (32, 8, False)}
RATEN = {"lloyd_row": (2, 3, 4), "had_lloyd": (2, 3, 4), "g16s8": (1, 2, 3), "g32s8": (2, 3)}  # -> bpw ~2.1/3.1/4.1 | 1.5/2.5/3.5 | 2.25/3.25

def rest_bytes(net):
    """fp16-Teil (Einbettung, Positionen, LayerNorm, Biases): in allen quantisierten Armen gleich."""
    lw = {id(m.weight) for _, m in lins(net)}
    return 2 * sum(p.numel() for p in net.parameters() if id(p) not in lw)


# ---------------------------------------------------------------- Training mit Kopier-Beimischung (jeder 5. Schritt = Kopierbatch)
def lm_trainiere_mix(netz, seed, T, lr=2e-3, mix=5, L=40):
    import abteilungen.struktur_sprache as ss
    tr = ss.daten(False)[0]; BATCH = ss.BATCH; ctx = ss.KONTEXT
    g = torch.Generator().manual_seed(seed + 5); ng = np.random.default_rng(seed + 6)
    torch.manual_seed(seed); opt = torch.optim.Adam(netz.parameters(), lr=lr); warm = max(1, T // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, T - warm))))
    netz.train(); t0 = time.time()
    for s in range(T):
        if s % mix == mix - 1:
            A = ng.integers(97, 123, (BATCH, L)); S = torch.from_numpy(np.concatenate([A, np.full((BATCH, 1), 32), A], 1)).long()
            x, y = S[:, :-1], S[:, 1:]
        else:
            i = torch.randint(0, len(tr) - ctx - 1, (BATCH,), generator=g); o = torch.arange(ctx)
            x, y = tr[i[:, None] + o].long(), tr[i[:, None] + o + 1].long()
        l = F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0); opt.step(); sch.step()
    return {"bpc": lm_val(netz), "sekunden": time.time() - t0}

# ---------------------------------------------------------------- Training (mit Checkpoint)
MIX = os.environ.get("W2V_MIX", "1") == "1"     # Standard: Kopier-Beimischung (Grundmodell lernt die Aufgabe ueberhaupt erst)
def modell(T, d=128, f=512):
    ck = os.path.join(SCR, f"w2v_s{SEED}_d{d}_T{T}" + ("_mix" if MIX else "") + ".pt")
    if os.path.exists(ck):
        c = torch.load(ck); n = lm_bauen2(d, f); n.load_state_dict(c["sd"]); n.eval(); return n, c
    torch.manual_seed(SEED); n = lm_bauen2(d, f); r = lm_trainiere_mix(n, SEED, T) if MIX else lm_trainiere(n, SEED, T)
    c = dict(sd=n.state_dict(), val_bpc=r["bpc"], sek=r["sekunden"], T=T); torch.save(c, ck); n.eval()
    log(f"trainiert d={d} T={T}: val {r['bpc']:.4f}, {r['sekunden']:.0f}s ({r['sekunden']/T*1000:.0f} ms/Schritt)"); return n, c

def setze(T, Wq, d=128, f=512):
    n2, _ = modell(T, d, f)
    for (nm, m), w in zip(lins(n2), Wq): m.weight.data = torch.from_numpy(np.asarray(w, np.float32))
    n2.eval(); return n2

def decode_us(g, skala, rotiert, bits, rep=200):
    """Dekodierzeit je Matrix 512x128 (CPU, 1 Thread, torch-eager, Indizes ungepackt uint8): Lookup + Skala (+ 8-Bit-Skala per exp2) (+ FWHT beim Matvec)."""
    torch.set_num_threads(1); o, i = 512, 128; g = g or i
    P = torch.randint(0, 2 ** bits, (o * i,), dtype=torch.uint8); cb = torch.randn(2 ** bits)
    sq = torch.randint(0, 256, (o * i // g, 1), dtype=torch.uint8); lo = torch.tensor(-3.)
    sf = torch.rand(o * i // g, 1) + .5; x = torch.randn(i); s = torch.from_numpy(vorz(i, 1))
    def dec():
        scale = sf if skala == 16 else torch.exp2(lo + sq.float() * (6. / 255))
        return (cb[P.long()].reshape(-1, g) * scale).reshape(o, i)
    def full(): return dec() @ (V.fwht(x * s) if rotiert else x)
    r = V.med_us(full, rep, 20); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2"))); return r

# ---------------------------------------------------------------- Haupt
def main(Ts):
    res = json.load(open(JS)) if os.path.exists(JS) else dict(seed=SEED, punkte={}, ref={})
    X = kopier_daten()
    for T in Ts:
        net, c = modell(T); L = lins(net); orig = [m.weight.data.numpy().astype(np.float32).copy() for _, m in L]
        nW = sum(w.size for w in orig); rb = rest_bytes(net)
        if str(T) not in res["ref"]:
            res["ref"][str(T)] = dict(val_bpc=float(lm_val(net)), bytes_fp16=2 * nW + rb, bytes_rest=rb, n_gewichte=nW, **kopier_test(net, X), train_s=c["sek"])
            log(f"T={T} fp32", res["ref"][str(T)])
        for arm, (g, sb, rotd) in ARME.items():
            for bits in RATEN[arm]:
                key = f"{arm}_b{bits}_T{T}"
                if key in res["punkte"]: continue
                t = time.time(); rng = np.random.default_rng(SEED); Wq = []; acc = dict(index=0, scale=0, cb=0, sign=0)
                for k, W in enumerate(orig):
                    q, a = quant_matrix(W, bits, g, sb, rotd, k, rng); Wq.append(q)
                    for kk in acc: acc[kk] += a[kk]
                tq = time.time() - t; t = time.time(); n2 = setze(T, Wq); bpc = float(lm_val(n2)); te = time.time() - t
                tot = sum(acc.values()); kp = kopier_test(n2, X)
                res["punkte"][key] = dict(arm=arm, bits=bits, T=T, bpw=tot / nW, bpw_skala=acc["scale"] / nW, bytes=tot / 8 + rb, val_bpc=bpc,
                    snr_db=snr_db(np.concatenate([w.ravel() for w in orig]), np.concatenate([w.ravel() for w in Wq])),
                    quant_s=tq, eval_s=te, dec_us=decode_us(g, sb, rotd, bits), **kp)
                json.dump(res, open(JS, "w"), indent=1)
                p = res["punkte"][key]; log(f"{key:20s} bpw {p['bpw']:.3f} bpc {bpc:.4f} kopie {kp['kopie_bpc']:.2f}/{kp['kopie_acc']:.2f} snr {p['snr_db']:.1f} q {tq:.1f}s ev {te:.1f}s dec {p['dec_us']:.0f}us")
        # Mittelmodell: kleineres fp16-Modell mit den Bytes des 3-Bit-Punkts (Zeilenskala Lloyd)
        key = f"klein_fp16_T{T}"
        if key not in res["punkte"]:
            ziel = res["punkte"][f"lloyd_row_b3_T{T}"]["bytes"]; best = None
            for d in range(16, 129, 16):
                f = 4 * d; n_ = lm_bauen2(d, f); b = 2 * sum(p.numel() for p in n_.parameters())
                if b <= ziel: best = (d, f, b)
            d, f, b = best; n3, c3 = modell(T, d, f)
            res["punkte"][key] = dict(arm="klein_fp16", bits=16, T=T, d=d, f=f, bytes=b, bpw=None, val_bpc=float(lm_val(n3)), **kopier_test(n3, X), train_s=c3["sek"])
            json.dump(res, open(JS, "w"), indent=1); log(key, res["punkte"][key])
    res["laufzeit_s"] = time.time() - T0; json.dump(res, open(JS, "w"), indent=1); log("fertig")

if __name__ == "__main__":
    main([int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "1000,3000").split(",")])
