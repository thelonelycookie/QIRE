"""Z5_komplement -- Ist Rotation ADDITIV zu den fairen Gegnern (Gruppenskala, Lloyd-Gitter, asymmetrisch)? (Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
ANLASS (Z3_ergebnisse_raven): Gegen g16/g32 und Lloyd verschwindet der Rotationsvorteil, wenn man Rotation GEGEN den Gegner stellt. Offen: bringt Rotation ZUSÄTZLICH zum Gegner (rot+X gegen X) noch etwas? Rotation macht das Rauschen additiv (k~0,1),
  Gruppenskalen erreichen das auf anderem Weg; ob beides zusammen mehr bringt, ist nie gemessen.
ARME (jeder mit EIGENER Form bei exakt <= gleichen Bytes, Skalen/Nullpunkte/Lloyd-Pegel/Vorzeichenbits/Auffüllung mitgezählt; PTQ, kein Neutraining je Arm):
  g32, g16      Gleichgitter (no-zero-Mittelpunkte, 2^Bit Stufen), fp16-Skala je Gruppe von 32 bzw. 16 Eingängen, MSE-Suche
  lloyd         fp16-Zeilen-RMS als Skala, Lloyd-Max-Pegel je Matrix (2^Bit fp16-Pegel je Matrix gezählt) auf den RMS-normierten Werten gelernt
  asym          min/max-Gitter je Zeile (fp16 Minimum + fp16 Schritt = 4 Byte je Zeile) mit MSE-Clipping-Suche
  rot_<X>       Vorzeichen*Hadamard (Eingang auf 2er-Potenz aufgefüllt, Auffüllung + 1 Vorzeichenbit je Eingang gezählt), dann Quantisierer X in der rotierten Basis, zurückgedreht
  zeile, rot_zeile (Referenz): Zeilenskala; rot_zeile ist die Budgetdefinition B(Bit) = Bytes des rotierten Großmodells (MLP h=512, LM d=128/f=512; Bytes der Z3-Formeln, per assert geprüft)
FORMEN: MLP 784-h-h-10, h in 1er-Schritten <= 1024; LM d in 16er-, f in 16er-Schritten, d <= 144, f <= 576, f/d möglichst nahe 4 (Regel wie Z3/Z4); größte Form mit Bytes <= B. Modell = Z3-Byte-LM (4 Lagen, Kontext 128, fp16-Rest gezählt).
ACHSEN: (1) Gewichte, MLP (Genauigkeit) und Byte-LM (Val-bpc, alle Linearschichten); (2) FFN-Mitte-Aktivierung im LM (je Token quantisiert, Gewichte fp32, Netz d=128/f=512, Schaden = bpc(q) - bpc(fp32);
  Skalenbytes je Token nicht gezählt, aber Feld "skalenbit_je_wert" nennt den Aufwand: g16 = 1 Bit/Wert, g32 = 0,5, Zeile/asym/lloyd ~0 -- die nicht gezählten Skalenbytes begünstigen hier also die Gruppenskalen, nicht die Rotation);
  Bits in {2, 3}; ZWEI Trainingslängen (MLP Epochen Z5_EPOCHS, Standard 2,6; LM Schritte Z5_T, Standard 100,400 (Verhältnis 1:3 bzw. 1:4; Laufzeit je Seed geschätzt 25-28 min bei 1 Faden aus Rauchtest-Rate: LM ca. 0,09 s je Schritt und Netz, 14 Netze; MLP h=512 ca. 4 s je Epoche); SCHNELL: 1,2 bzw. 20,60). Lloyd-Pegel Aktivierung: auf 8192 Kalibriertokens (64 Fenster), pro Lage.
AMPEL (Einzelseed, vorläufig): "ampel_<mlp|lm>_gew_b<bits>", "ampel_lm_akt_b<bits>": gewinn = Minimum über Längen von [bestes rot_X - bestes X] (X in g32, g16, lloyd, asym; Einheiten pp bzw. bpc; Schwellen 0,5 pp, 0,01 bpc, Aktivierung 0,005 bpc).
  Zusatz "komplement": rot_X minus X je Gegner und Länge (ist Rotation additiv zum EINZELNEN Gegner?).
PARETO (Bytes-Pareto, Bit 2 und 3): roh-g32 gegen rot-Zeilenskala über Breiten (MLP h in 64..512 + Budgetformen; LM d=32/64/128 mit f=4d + Budgetformen); "pareto_<modell>_b<bits>": je Länge Differenz Qualität bei gleichen Bytes
  (Interpolation von g32 in log-Bytes an den Stützstellen von rot_zeile) und Byteersparnis (1 - Bytes_rot / Bytes, die g32 für dieselbe Qualität bräuchte); Ampel = Minimum über Längen der mittleren Differenz (Schwellen wie oben).
VORAB-P (vor dem Lauf): Ampel grün (rot_X schlägt bestes X über Schwelle, beide Längen): MLP Gew 2 Bit 15 %, 3 Bit 10 %; LM Gew 2 Bit 10 %, 3 Bit 5 %; LM Akt 2 Bit 10 %, 3 Bit 5 %. Gelb oder besser (Gewinn > 0): MLP 35 %, LM Gew 30 %, Akt 2 Bit 40 %.
  Pareto: rot_zeile >= g32 in beiden Längen (Differenz > 0) MLP 35 %, LM 25 %; Byteersparnis >= 10 % MLP 25 %, LM 15 %. Beste Gegner meist g16 oder lloyd (65 %).
Nicht untersucht: QAT je Arm, Kernelzeit der Rotation, Aktivierungs-Skalenbytes, Sparsamkeit. fp16-Rundung der Skalen ist nicht simuliert (für alle Arme gleich vernachlässigt).
Lies zuerst: formen, bytes, ampel_*, dann komplement, pareto_*, tabelle."""
from __future__ import annotations
import os
os.environ.setdefault("Z2_VAL", "16384")
from Z3_bytegleich import pow2, mlp_formen, lm_formen, lm_rest, lin_bytes, MODI, LAGEN_, ampel_gegen, lm_ffn_mitte, linears
from Z2_rauschformung import lm_kalib
from _z2 import *

BITS = (2, 3)
X_ARME = ("g32", "g16", "lloyd", "asym")
ROT_X = tuple("rot_" + x for x in X_ARME)
ALLE_ARME = X_ARME + ROT_X + ("zeile", "rot_zeile")
LM_DMAX, LM_FMAX = 144, 576
SCHW = {"mlp": 0.5, "lm": 0.01, "akt": 0.005}
EPOCHEN = tuple(int(x) for x in os.environ.get("Z5_EPOCHS", "2,6").split(","))
SCHRITTE = tuple(int(x) for x in os.environ.get("Z5_T", "100,400").split(","))
MLP_BREITEN = (64, 128, 192, 256, 384, 512)
LM_BREITEN = ((32, 128), (64, 256), (128, 512))


def zerlege(arm):
    return (True, arm[4:]) if arm.startswith("rot_") else (False, arm)


# ------------------------------------------------------------------ Quantisierer (letzte Achse = Zeile)
def lloyd_pegel(v, L, iters=30):
    v = v.flatten()
    if len(v) > 200_000:
        v = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(1))[:200_000]]
    v = v.sort().values
    p = torch.stack([v[int((i + 0.5) / L * (len(v) - 1))] for i in range(L)])
    for _ in range(iters):
        idx = torch.bucketize(v, (p[1:] + p[:-1]) / 2)
        s = torch.zeros(L).index_add_(0, idx, v)
        c = torch.bincount(idx, minlength=L).float()
        p = torch.where(c > 0, s / c.clamp_min(1), p).sort().values
    return p


def q_asym(X, L, nf=12):
    lo, hi = X.amin(-1, keepdim=True), X.amax(-1, keepdim=True)
    mid, half = (lo + hi) / 2, ((hi - lo) / 2).clamp_min(1e-12)
    best, bq = None, None
    for f in np.linspace(0.5, 1.0, nf):
        l, s = mid - half * float(f), 2 * half * float(f) / (L - 1)
        q = ((X - l) / s).round().clamp(0, L - 1) * s + l
        e = ((X - q) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bq = e, q
        else:
            m = e < best
            best, bq = torch.where(m, e, best), torch.where(m, q, bq)
    return bq


def q_gruppe(X, L, g):
    n = X.shape[-1]
    ng = math.ceil(n / g)
    Xp = F.pad(X, (0, ng * g - n)).reshape(*X.shape[:-1], ng, g)
    return stufen_quant(Xp, L, 12).reshape(*X.shape[:-1], ng * g)[..., :n]


def quant_zeilen(X, bits, kind, p=None):
    L = 2 ** bits
    if kind == "zeile":
        return stufen_quant(X, L, 12)
    if kind in ("g16", "g32"):
        return q_gruppe(X, L, int(kind[1:]))
    if kind == "asym":
        return q_asym(X, L)
    rms = X.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)       # lloyd
    return p[torch.bucketize(X / rms, (p[1:] + p[:-1]) / 2)] * rms


def pegel_fuer(X, bits, kind):
    if kind != "lloyd":
        return None
    return lloyd_pegel(X / X.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12), 2 ** bits)


def rot_hin(X, s):
    return fwht(F.pad(X, (0, len(s) - X.shape[-1])) * s)


def rot_rueck(Y, s, n):
    return (fwht(Y) * s)[..., :n]


# ------------------------------------------------------------------ Bytes und Formen
def lin_b(o, i, bits, arm):
    rot, kind = zerlege(arm)
    ip = pow2(i) if rot else i
    g = {"g16": 16, "g32": 32}.get(kind)
    sk = o * math.ceil(ip / g) * 2 if g else (4 * o if kind == "asym" else 2 * o)
    return o * ip * bits / 8 + sk + (2 * 2 ** bits if kind == "lloyd" else 0) + 2 * o + (ip / 8 if rot else 0)


def fb(formen, bits, arm, rest=0):
    return float(sum(lin_b(o, i, bits, arm) for o, i in formen) + rest)


def mlp_form_b(bits, arm, B):
    best = 8
    for h in range(8, 1025):
        if fb(mlp_formen(h), bits, arm) <= B:
            best = h
    return (best,)


def lm_form_b(bits, arm, B):
    kand = []
    for d in range(16, LM_DMAX + 1, 16):
        fm = 0
        for f in range(16, LM_FMAX + 1, 16):
            if fb(lm_formen(d, f), bits, arm, lm_rest(d)) <= B:
                fm = f
        if fm:
            kand.append((abs(math.log(fm / d / 4)), -d, d, fm))
    return tuple(min(kand)[2:]) if kand else (16, 16)


def bytes_form(modell, form, bits, arm):
    return fb(mlp_formen(form[0]), bits, arm) if modell == "mlp" else fb(lm_formen(*form), bits, arm, lm_rest(form[0]))


# ------------------------------------------------------------------ Netz
class FFNA(FFNDichtX):
    def __init__(self, d, f):
        super().__init__(d, f)
        self.qfn = None

    def forward(self, x):
        a = self.akt(self.a(x))
        return self.b(self.qfn(a) if self.qfn is not None else a)


def lm_bauen(d, f):
    return LMX([FFNA(d, f) for _ in range(LAGEN_)], d=d)


def ptq_gew(netz, bits, arm, seed):
    rot, kind = zerlege(arm)
    n = copy.deepcopy(netz)
    for k, m in enumerate(linears(n), 1):
        with torch.no_grad():
            W = m.weight.data
            i = W.shape[1]
            s = vorzeichen(pow2(i), seed * 101 + k) if rot else None
            X = rot_hin(W, s) if rot else W
            Xq = quant_zeilen(X, bits, kind, pegel_fuer(X, bits, kind))
            m.weight.data = rot_rueck(Xq, s, i) if rot else Xq
    return n.eval()


def akt_qfn(Xcal, bits, arm, sgn):
    rot, kind = zerlege(arm)
    Xc = rot_hin(Xcal, sgn) if rot else Xcal
    p = pegel_fuer(Xc, bits, kind)

    def f(x):
        sh = x.shape
        x2 = x.reshape(-1, sh[-1])
        if rot:
            return rot_rueck(quant_zeilen(rot_hin(x2, sgn), bits, kind, p), sgn, sh[-1]).reshape(sh)
        return quant_zeilen(x2, bits, kind, p).reshape(sh)
    return f


# ------------------------------------------------------------------ Pareto
def pareto_auswerten(roh, rot):
    """roh/rot: Liste (Bytes, Qualität, größer = besser). Differenz bei gleichen Bytes (rot - roh, Interpolation in log-Bytes) und Byteersparnis."""
    roh, rot = sorted(roh), sorted(rot)
    xr = np.log([b for b, _ in roh])
    qr = np.array([q for _, q in roh])
    qm = np.maximum.accumulate(qr) + np.arange(len(qr)) * 1e-12
    gaps, spar = [], []
    for b, q in rot:
        x = math.log(b)
        if xr[0] <= x <= xr[-1]:
            gaps.append(float(q - np.interp(x, xr, qr)))
        if qm[0] <= q <= qm[-1]:
            spar.append(float(1 - b / math.exp(np.interp(q, qm, xr))))
    return {"differenz_je_punkt": gaps, "differenz_mittel": float(np.mean(gaps)) if gaps else float("nan"),
            "byteersparnis_je_punkt": spar, "byteersparnis_mittel": float(np.mean(spar)) if spar else float("nan"), "n_punkte": len(gaps)}


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 2) if schnell else EPOCHEN
    Ts = (20, 60) if schnell else SCHRITTE
    xtr, ytr, xva, yva = mlp_daten(schnell)
    # Bytes-Formeln gegen Z3 prüfen
    assert abs(fb(mlp_formen(512), 2, "rot_zeile") - form_bytes_z3(mlp_formen(512))) < 1e-6
    B = {"mlp": {b: fb(mlp_formen(512), b, "rot_zeile") for b in BITS}, "lm": {b: fb(lm_formen(128, 512), b, "rot_zeile", lm_rest(128)) for b in BITS}}
    formen = {"mlp": {b: {a: mlp_form_b(b, a, B["mlp"][b]) for a in ALLE_ARME} for b in BITS}, "lm": {b: {a: lm_form_b(b, a, B["lm"][b]) for a in ALLE_ARME} for b in BITS}}
    formen["mlp"] = {b: {a: (512,) if a == "rot_zeile" else f for a, f in d.items()} for b, d in formen["mlp"].items()}
    formen["lm"] = {b: {a: (128, 512) if a == "rot_zeile" else f for a, f in d.items()} for b, d in formen["lm"].items()}
    out = {"epochen": eps, "schritte": Ts, "val_bytes": VAL_N, "budget": B, "skalenbit_je_wert": {"g16": 1.0, "g32": 0.5, "zeile": 0.0, "lloyd": 0.0, "asym": 0.0},
           "formen": {m: {b: {a: list(f) for a, f in d.items()} for b, d in v.items()} for m, v in formen.items()},
           "bytes": {m: {b: {a: bytes_form(m, f, b, a) for a, f in d.items()} for b, d in v.items()} for m, v in formen.items()},
           "tabelle": {"mlp": {b: {} for b in BITS}, "lm": {b: {} for b in BITS}, "akt": {b: {} for b in BITS}}, "fp32": {"mlp": {}, "lm": {}}, "komplement": {}, "pareto_punkte": {}, "netz_sekunden": {"lm": {}, "mlp": {}}}
    for m in ("mlp", "lm"):
        for b in BITS:
            for a, f in formen[m][b].items():
                assert out["bytes"][m][b][a] <= B[m][b] + 1e-6, (m, b, a)
    print("Formen MLP", out["formen"]["mlp"], "\nFormen LM", out["formen"]["lm"], flush=True)
    # ---------------------------------------------------------------- MLP
    for ep in eps:
        t = time.time()
        cache, memo = {}, {}

        def netz(h):
            if h not in cache:
                torch.manual_seed(seed if h == 512 else seed + 1)
                n = kb.baue_bild([h, h])
                t1 = time.time()
                mlp_train(n, xtr, ytr, seed, ep, 1e-3)
                out["netz_sekunden"]["mlp"][f"ep{ep}_h{h}"] = time.time() - t1
                cache[h] = n
            return cache[h]

        def messe(h, b, a):
            if (h, b, a) not in memo:
                memo[(h, b, a)] = acc(ptq_gew(netz(h), b, a, seed), xva, yva)
            return memo[(h, b, a)]
        out["fp32"]["mlp"][ep] = acc(netz(512), xva, yva)
        for b in BITS:
            out["tabelle"]["mlp"][b][ep] = {a: messe(formen["mlp"][b][a][0], b, a) for a in ALLE_ARME}
            print(f"Z5_komplement s{seed} MLP ep{ep} b{b}: " + " ".join(f"{a} {v:.4f}" for a, v in out["tabelle"]["mlp"][b][ep].items()) + f" ({time.time() - t:.0f}s)", flush=True)
            pts = {"g32": {}, "rot_zeile": {}}
            for a in pts:
                for h in sorted(set(MLP_BREITEN) | {formen["mlp"][b]["g32"][0], formen["mlp"][b]["rot_zeile"][0]}):
                    pts[a][h] = (bytes_form("mlp", (h,), b, a), 100 * messe(h, b, a))
            out["pareto_punkte"][f"mlp_b{b}_ep{ep}"] = {a: {h: list(v) for h, v in d.items()} for a, d in pts.items()}
    # ---------------------------------------------------------------- LM
    schn = schnell
    mess = lambda q: val_bpc(q, schn)
    for T in Ts:
        t = time.time()
        cache, memo = {}, {}

        def netz(df):
            df = tuple(df)
            if df not in cache:
                torch.manual_seed(seed if df == (128, 512) else seed + 1)
                n = lm_bauen(*df)
                r = lm_trainiere(n, seed, T, schnell=schn)
                out["netz_sekunden"]["lm"][f"T{T}_d{df[0]}_f{df[1]}"] = r["sekunden"]
                cache[df] = n.eval()
            return cache[df]

        def messe(df, b, a):
            if (tuple(df), b, a) not in memo:
                memo[(tuple(df), b, a)] = mess(ptq_gew(netz(df), b, a, seed))
            return memo[(tuple(df), b, a)]
        big = netz((128, 512))
        v0 = mess(big)
        out["fp32"]["lm"][T] = v0
        for b in BITS:
            out["tabelle"]["lm"][b][T] = {a: messe(formen["lm"][b][a], b, a) for a in ALLE_ARME}
            print(f"Z5_komplement s{seed} LM T{T} b{b}: fp32 {v0:.4f} | " + " ".join(f"{a} {v:.4f}" for a, v in out["tabelle"]["lm"][b][T].items()) + f" ({time.time() - t:.0f}s)", flush=True)
            pts = {"g32": {}, "rot_zeile": {}}
            for a in pts:
                for df in sorted(set(LM_BREITEN) | {tuple(formen["lm"][b]["g32"]), tuple(formen["lm"][b]["rot_zeile"])}):
                    pts[a][df] = (bytes_form("lm", df, b, a), -messe(df, b, a))
            out["pareto_punkte"][f"lm_b{b}_T{T}"] = {a: {f"{d}x{f}": list(v) for (d, f), v in dd.items()} for a, dd in pts.items()}
        # Aktivierung FFN-Mitte (Netz 128/512, fp-Gewichte)
        akt = lm_ffn_mitte(big, lm_kalib(seed, schn))
        sg = [vorzeichen(512, seed * 7 + 31 * l) for l in range(LAGEN_)]
        for b in BITS:
            tab = {}
            for a in ALLE_ARME:
                for l, bl in enumerate(big.bl):
                    bl.ffn.qfn = akt_qfn(akt[f"ffn_mitte_L{l}"], b, a, sg[l])
                tab[a] = mess(big) - v0
                for bl in big.bl:
                    bl.ffn.qfn = None
            out["tabelle"]["akt"][b][T] = tab
            print(f"Z5_komplement s{seed} AKT T{T} b{b} Schaden: " + " ".join(f"{a} {v:.4f}" for a, v in tab.items()) + f" ({time.time() - t:.0f}s)", flush=True)
    # ---------------------------------------------------------------- Ampeln
    wm, wl, wa = (lambda a: 100 * a), (lambda v: -v), (lambda v: -v)
    for b in BITS:
        out[f"ampel_mlp_gew_b{b}"] = ampel_gegen(out["tabelle"]["mlp"][b], eps, ROT_X, X_ARME, wm, SCHW["mlp"], "pp", f"bestes rot_X - bestes X, MLP-Gewichte {b} Bit, gleiche Bytes, Minimum über Längen")
        out[f"ampel_lm_gew_b{b}"] = ampel_gegen(out["tabelle"]["lm"][b], Ts, ROT_X, X_ARME, wl, SCHW["lm"], "bpc", f"bestes rot_X - bestes X, LM-Gewichte {b} Bit, gleiche Bytes, Minimum über Längen")
        out[f"ampel_lm_akt_b{b}"] = ampel_gegen(out["tabelle"]["akt"][b], Ts, ROT_X, X_ARME, wa, SCHW["akt"], "bpc", f"bestes rot_X - bestes X, LM-FFN-Aktivierung {b} Bit (Schaden), Minimum über Längen")
        for mod, Ls, w in (("mlp", eps, wm), ("lm", Ts, wl), ("akt", Ts, wa)):
            tb = out["tabelle"][mod][b]
            out["komplement"][f"{mod}_b{b}"] = {x: {L: w(tb[L]["rot_" + x]) - w(tb[L][x]) for L in Ls} for x in X_ARME}
        for mod, Ls, sch, w in (("mlp", eps, "mlp", 1), ("lm", Ts, "lm", 1)):
            roh = {L: out["pareto_punkte"][f"{mod}_b{b}_{'ep' if mod == 'mlp' else 'T'}{L}"] for L in Ls}
            erg = {L: pareto_auswerten([tuple(v) for v in roh[L]["g32"].values()], [tuple(v) for v in roh[L]["rot_zeile"].values()]) for L in Ls}
            out[f"pareto_{mod}_b{b}"] = erg
            gew = min((erg[L]["differenz_mittel"] for L in Ls), default=float("nan"))
            out[f"ampel_pareto_{mod}_b{b}"] = ampel(gew if gew == gew else -9.0, SCHW[sch], "pp" if mod == "mlp" else "bpc", f"rot_zeile - roh_g32 bei gleichen Bytes ({mod}, {b} Bit), mittlere Differenz, Minimum über Längen")
    for k in sorted(out):
        if k.startswith("ampel_"):
            print(f"{k}: {out[k]['farbe']} {out[k]['gewinn']:+.4f} {out[k]['einheit']}", flush=True)
    return out


def form_bytes_z3(formen):
    from Z3_bytegleich import form_bytes
    return form_bytes(formen, MODI["gross_rot"])


if __name__ == "__main__":
    rahmen("Z5_komplement", main)
