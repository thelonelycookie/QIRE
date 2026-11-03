#!/usr/bin/env python3
"""W7-Vortest (Phase 2, Runde 3): Quadrat-Familie unter Quantisierung im Byte-LM. Erkundung, kein Befund.
NUR Tuning-Seed 950. Bewertet auf Val (dev), nie Test. Rezept wie ../runde2/w2_vortest.py (Byte-LM d=128, f=512, 4 Lagen, lr 2e-3, Adam, Kosinus).
Start: OMP_NUM_THREADS=2 python3 w7_vortest.py [T1,T2]      (Standard 3000,8000; Wiederaufnahme je Modell/Messung; 'md' = nur Bericht neu schreiben)
Aktivierungen der FFN: gelu | relu2 (relu(z)^2) | z2 (z^2) | abs (|z|) | swiglu (silu(g)*v, f'=341, parametergleich auf 0,03 %).
Quantisierung (Simulation, fake-quant): Aktivierung 4 Bit, pro Token, symmetrisch (Skala = max|x|/7, Stufen -7..7), roh oder mit Hadamard davor
  (feste Vorzeichen, orthogonal, bei f'=341 auf 512 mit Nullen aufgefuellt); Orte: FFN-Eingang (nach LayerNorm) und Zwischenaktivierung (Eingang der Down-Projektion).
  Gewichte 4 Bit: Lloyd-Buch je Matrix, Zeilenskala RMS fp16 (alle nn.Linear: Attention + FFN), ohne Rotation.
Kurtosis: Excess-Kurtosis (gepoolt ueber alle Elemente) der Zwischenaktivierung je Lage; Mittel und Max ueber die 4 Lagen; roh und nach Hadamard; dazu max|x|/rms je Token.
Messung auf den ersten VALQ Bytes der Val-Daten (alle Messungen derselben Bytes, gepaart).
"""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "runde1"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import w6_vortest2 as V
from w6_vortest2 import f16, lins, vorz, lloyd_q
from Z3_bytegleich import LMX, lm_trainiere
from _basis import fwht
import abteilungen.struktur_sprache as ss
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
SEED = 950
SCR = os.environ.get("W7V_SCR", "/tmp/claude-0/-home-user-QIRE/c85d0af3-9103-522e-a686-8bafbfdddd9a/scratchpad")
JS = os.path.join(HERE, "W7_vortest.json"); MD = os.path.join(HERE, "W7_vortest.md")
D, FF, LAGEN, FSW, VALQ = 128, 512, 4, 341, 32768
ACTS = ("gelu", "relu2", "z2", "abs", "swiglu")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- Quantisierung der Aktivierung (global schaltbar)
CFG = dict(ein=False, mitte=False, had=False, rec=None)       # rec: dict -> Kurtosis-Aufnahme
_S = {}
def _vz(n):
    if n not in _S: _S[n] = torch.from_numpy(vorz(n, SEED * 7 + n))
    return _S[n]
def had(x, inv=False):
    n = x.shape[-1]; p = 1 << (n - 1).bit_length(); s = _vz(p)
    if p != n: x = F.pad(x, (0, p - n))
    return fwht(x * s) if not inv else fwht(x) * s
def tokq(x, bits=4):
    L = 2 ** (bits - 1) - 1; sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / L
    return (x / sc).round().clamp(-L, L) * sc
def aq(x, an):
    if not an: return x
    if CFG["had"]:
        n = x.shape[-1]; y = tokq(had(x)); return had(y, inv=True)[..., :n]
    return tokq(x)
def kurt(x):
    x = x.reshape(-1).double(); m = x.mean(); v = ((x - m) ** 2).mean()
    return float(((x - m) ** 4).mean() / v ** 2 - 3)
def maxrms(x):
    x = x.reshape(-1, x.shape[-1]); return float((x.abs().amax(-1) / (x.pow(2).mean(-1).sqrt() + 1e-12)).mean())

class FFNW(nn.Module):
    def __init__(self, d, f, akt):
        super().__init__(); self.akt = akt; fz = FSW if akt == "swiglu" else f
        self.a, self.b = nn.Linear(d, 2 * fz if akt == "swiglu" else fz), nn.Linear(fz, d); self.lage = 0
    def act(self, z):
        k = self.akt
        if k == "gelu": return F.gelu(z)
        if k == "relu2": return F.relu(z) ** 2
        if k == "z2": return z * z
        if k == "abs": return z.abs()
        g, v = z.chunk(2, -1); return F.silu(g) * v
    def forward(self, x):
        r = CFG["rec"]
        if r is not None: r.setdefault("ein", {}).setdefault(self.lage, []).append(x.detach())
        z = self.a(aq(x, CFG["ein"])); h = self.act(z)
        if r is not None: r.setdefault("mitte", {}).setdefault(self.lage, []).append(h.detach())
        return self.b(aq(h, CFG["mitte"]))

def bauen(akt):
    torch.manual_seed(SEED); ffns = [FFNW(D, FF, akt) for _ in range(LAGEN)]
    for l, f in enumerate(ffns): f.lage = l
    return LMX(ffns, d=D)

def nparams(n): return sum(p.numel() for p in n.parameters())
def ffn_params(n): return sum(p.numel() for b in n.bl for p in b.ffn.parameters())

# ---------------------------------------------------------------- Gewichte 4 Bit
def wq4(net, bits=4):
    rng = np.random.default_rng(SEED)
    for _, m in lins(net):
        W = m.weight.data.numpy().astype(np.float32)
        sc = f16(np.sqrt((W ** 2).mean(1, keepdims=True)) + 1e-12)
        Zq, _ = lloyd_q(W / sc, bits, rng); m.weight.data = torch.from_numpy((Zq * sc).astype(np.float32))

# ---------------------------------------------------------------- Training (Checkpoint)
def modell(akt, T):
    ck = os.path.join(SCR, f"w7v_s{SEED}_{akt}_T{T}.pt"); n = bauen(akt)
    if os.path.exists(ck):
        c = torch.load(ck); n.load_state_dict(c["sd"]); n.eval(); return n, c
    lr = 2e-3; r = lm_trainiere(n, SEED, T, lr=lr)
    if r.get("instabil"):
        log(f"{akt} T={T} instabil bei lr {lr}; Neustart lr 7e-4"); n = bauen(akt); lr = 7e-4; r = lm_trainiere(n, SEED, T, lr=lr)
    c = dict(sd=n.state_dict(), val_bpc=r["bpc"], sek=r["sekunden"], T=T, lr=lr, instabil=bool(r.get("instabil"))); torch.save(c, ck); n.eval()
    log(f"trainiert {akt} T={T}: val {r['bpc']:.4f}, {r['sekunden']:.0f}s, lr {lr}, instabil={c['instabil']}"); return n, c

def val(n):
    CFG.update(rec=None); return ss.bpc_auswerten(n, ss.daten(False)[1][:VALQ])

def kurtosen(n):
    va = ss.daten(False)[1][:VALQ + 1].long(); x = va[:(VALQ // 128) * 128].reshape(-1, 128)[:96]
    CFG.update(ein=False, mitte=False, rec={}); n.eval()
    with torch.no_grad(): n(x)
    r = CFG["rec"]; CFG["rec"] = None; o = {}
    for ort in ("ein", "mitte"):
        ks, kh, mr, mh = [], [], [], []
        for l in range(LAGEN):
            h = torch.cat(r[ort][l]); hh = had(h)
            ks.append(kurt(h)); kh.append(kurt(hh)); mr.append(maxrms(h)); mh.append(maxrms(hh))
        o[ort] = dict(kurt_je_lage=ks, kurt_mittel=float(np.mean(ks)), kurt_max=float(max(ks)), kurt_had_mittel=float(np.mean(kh)),
                      maxrms_roh=float(np.mean(mr)), maxrms_had=float(np.mean(mh)))
    # Anteil exakter Nullen der Zwischenaktivierung (Sparsitaet)
    o["mitte"]["nullanteil"] = float(np.mean([(torch.cat(r["mitte"][l]) == 0).float().mean() for l in range(LAGEN)]))
    return o

def messe(akt, T, res):
    key = f"{akt}_T{T}"
    if key in res["modelle"] and "fertig" in res["modelle"][key]: return
    n, c = modell(akt, T); m = res["modelle"].setdefault(key, {})
    m.update(akt=akt, T=T, train_s=c["sek"], lr=c["lr"], instabil=c["instabil"], n_params=nparams(n), ffn_params=ffn_params(n))
    m["fp32"] = val(n); m["kurtosis"] = kurtosen(n); log(key, "fp32", round(m["fp32"], 4), "kurt mitte", round(m["kurtosis"]["mitte"]["kurt_mittel"], 1))
    plan = [("A4_beide_roh", dict(ein=True, mitte=True, had=False), False), ("A4_beide_had", dict(ein=True, mitte=True, had=True), False),
            ("A4_nurein_roh", dict(ein=True, mitte=False, had=False), False), ("A4_nurein_had", dict(ein=True, mitte=False, had=True), False),
            ("A4_nurmitte_roh", dict(ein=False, mitte=True, had=False), False), ("A4_nurmitte_had", dict(ein=False, mitte=True, had=True), False),
            ("W4", dict(ein=False, mitte=False, had=False), True), ("W4A4_roh", dict(ein=True, mitte=True, had=False), True),
            ("W4A4_had", dict(ein=True, mitte=True, had=True), True)]
    nw = None
    for nm, cfg, w in plan:
        if nm in m: continue
        if w:
            if nw is None: nw, _ = modell(akt, T); wq4(nw)
            net = nw
        else: net = n
        CFG.update(cfg); CFG["rec"] = None; m[nm] = ss.bpc_auswerten(net, ss.daten(False)[1][:VALQ]); m[nm + "_dbpc"] = m[nm] - m["fp32"]
        log(f"  {key:12s} {nm:16s} bpc {m[nm]:.4f}  dbpc {m[nm+'_dbpc']:+.4f}")
    CFG.update(ein=False, mitte=False, had=False)
    m["fertig"] = True; res["laufzeit_s"] = time.time() - T0; json.dump(res, open(JS, "w"), indent=1)

def bericht(res):
    L = ["# W7-Vortest (Phase 2, Runde 3): Quadrat-Familie unter 4-Bit-Quantisierung", "",
         "Erkundung, kein Befund. Tuning-Seed 950, Byte-LM d=128, f=512, 4 Lagen (Rezept W2), Val-Bytes: erste 32768 (gepaart). Einzelseed: Unterschiede < ca. 0,02 bpc sind nicht belastbar.",
         "Aktivierungsquantisierung 4 Bit pro Token symmetrisch (roh | Hadamard davor), Gewichte 4 Bit Lloyd/Zeilenskala (alle Linear, ohne Rotation).", ""]
    for T in sorted({m["T"] for m in res["modelle"].values()}):
        ms = [res["modelle"].get(f"{a}_T{T}") for a in ACTS]; ms = [m for m in ms if m and "fertig" in m]
        if not ms: continue
        L += [f"## T = {T}", "", "bpc (Val) und Verlust gegenueber fp32 (dbpc = quantisiert - fp32):", "",
              "| Akt | FFN-Par. | fp32 | A4 roh (beide) | A4 had (beide) | nur Ein roh/had | nur Mitte roh/had | W4 | W4A4 roh | W4A4 had |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m in ms:
            d = lambda k: f"{m[k+'_dbpc']:+.3f}"
            L.append(f"| {m['akt']}{' (lr 7e-4)' if m['lr']!=2e-3 else ''} | {m['ffn_params']} | {m['fp32']:.3f} | {d('A4_beide_roh')} | {d('A4_beide_had')} | {d('A4_nurein_roh')}/{d('A4_nurein_had')} | {d('A4_nurmitte_roh')}/{d('A4_nurmitte_had')} | {d('W4')} | {d('W4A4_roh')} | {d('W4A4_had')} |")
        L += ["", "Kurtosis (Excess) und Ausreisser:", "", "| Akt | Ort | Kurtosis Mittel | Kurtosis Max (Lage) | Kurt. nach Hadamard | max/rms roh | max/rms had | Nullanteil |", "|---|---|---|---|---|---|---|---|"]
        for m in ms:
            for ort in ("ein", "mitte"):
                k = m["kurtosis"][ort]
                L.append(f"| {m['akt']} | {ort} | {k['kurt_mittel']:.1f} | {k['kurt_max']:.1f} | {k['kurt_had_mittel']:.1f} | {k['maxrms_roh']:.1f} | {k['maxrms_had']:.1f} | {k.get('nullanteil', float('nan')):.2f} |")
        L.append("")
    json.dump(res, open(JS, "w"), indent=1); open(MD, "w").write("\n".join(L) + "\n")

def main(Ts):
    res = json.load(open(JS)) if os.path.exists(JS) else dict(seed=SEED, modelle={})
    for T in Ts:
        for a in ACTS:
            messe(a, T, res); bericht(res)
    bericht(res); log("fertig")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "md": bericht(json.load(open(JS)))
    else: main([int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "3000,8000").split(",")])
