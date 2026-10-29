"""Z4_bitkurve -- Bei festem Byte-Budget B: Qualität über Bits je Gewicht; verschiebt Rotation das Optimum nach unten? (Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
ANLASS (REAKTOR R7, G6-G9): Z3 prüft nur 2 Bit gegen Gegner. R7 fragt: Gibt es ein Optimum der Bitbreite bei festen Bytes, und liegt es mit Hadamard-Rotation tiefer als ohne? Verallgemeinerung von R5 (Teilmenge {2, 4, 16}).
BYTEBUDGET B = wie Z3_bytegleich (Bytes des rotierten 2-Bit-Großmodells; MLP h=512, LM d=128/f=512; Skalen, Bias, Vorzeichen und fp16-Rest gezählt, Formeln aus Z3 wiederverwendet, Z3 unverändert).
BITBREITEN: 1,58 (ternär, 3 Stufen, mit log2(3) Bit gezählt), 2, 3, 4, 8, 16 (fp16, nicht quantisiert). Je Bitbreite und Art die GRÖSSTE Form mit Bytes <= B (MLP h in 1er-Schritten <= 1024; LM d in 16er-, f in 16er-Schritten, d <= 144, f <= 576, f/d nahe 4 wie Z3; Caps höher als Z3, damit 1,58 Bit mehr Breite bekommt).
ARTEN: roh = Gruppenskala g32 (fp16 je 32 Eingänge, bei 16 Bit keine Skala; beste Skala laut Z3, Skalen zählen); rot = Vorzeichen*Hadamard, Zeilenskala (Eingang auf 2er-Potenz aufgefüllt, Auffüllung und Vorzeichenbits zählen).
VERFAHREN: PTQ (fp trainiert, dann einmal runden; Gitter MSE-Suche) und QAT (STE von Anfang an, Skala alle 16 Läufe neu). 16 Bit: roh = rot = PTQ = QAT = fp16-gerundetes fp-Netz (Rotation hier verlustfrei, kein Unterschied).
MODELLE/LÄNGEN: MLP MNIST 784-h-h-10, Epochen {2, 6} (schnell {1, 2}; gekürzt gegen Z3 {3, 12}, damit ein Seed <= 30 min läuft: gemessen ca. 85 s je Epoche für alle 12 MLP-Netze + QAT), Adam 1e-3; Byte-LM 4 Lagen, Schritte {80, 320} (schnell {40, 160}; Verhältnis 1:4 wie Z3, gekürzt wegen Laufzeit: ca. 1,9 s je Schritt für alle LM-Netze); LM-Bitbreiten reduziert auf {1,58, 2, 4, 8, 16} (3 Bit nur im MLP), Umgebungsvariable Z4_LM_BITS, lr 2e-3 (wie Z3). Maß: Genauigkeit bzw. Val-bpc (Z2_VAL Bytes). Netze gleicher Form teilen das fp-Training (Seed wie Z3 `seed`).
AUSGABE: tabelle[modell][länge][roh_ptq|roh_qat|rot_ptq|rot_qat][bit] ; formen/bytes ; optimum[modell][länge][verfahren] = {roh_bit, rot_bit, roh_best, rot_best, bit_verschiebung, gewinn} ; ampel_mlp / ampel_lm ;
  zaehlmodell[modell][bit][roh|rot]: Gewichtsbytes, Aktivierungsbytes (fp16, je Linear Eingang+Ausgang je Token), MACs je Token (Linear, gepolstert), Rotations-Additionen (ip*log2 ip je Linear), Vorzeichen-Mults, gemessene Wandzeit der Rotation pro Token (FWHT, CPU, 1 Faden) neben Matmul-Zeit, grobes Energieintervall (Konstanten aus dem Gedächtnis, Horowitz 2014, vor Zitat prüfen; nur Modell, keine Messung).
AMPEL (Einzelseed, vorläufig): grün = Rotation verschiebt das Optimum (bestes Bit, strikt argmax, beste von PTQ/QAT je Art) nach unten UND beste Rot-Qualität > beste Roh-Qualität + Schwelle (MLP 0,5 pp, LM 0,01 bpc), in BEIDEN Längen;
  gelb = in beiden Längen Gewinn > 0 oder Verschiebung nach unten (nicht beides, oder nur in einer Länge); rot sonst. Zusatz je (Länge, Verfahren): Einzelzellen in `optimum`, plus Toleranzoptimum (kleinstes Bit innerhalb 0,1 pp / 0,003 bpc vom Besten).
VORAB-P (vor dem Lauf): Optimum roh bei 4 Bit 50 %, bei 3 Bit 30 %; Optimum rot < roh (ganze Stufe) in beiden Längen, beste von PTQ/QAT: MLP 30 %, LM 25 %. Rot-Qualität besser als roh um > Schwelle: MLP 25 %, LM 20 %. Ampel grün (beides) MLP 15 %, LM 10 %. Gelb oder besser MLP 45 %, LM 40 %.
  Kurve glatt genug (Optimum nicht an Rand 1,58 oder 16 Bit): 60 %. Rotationszeit je Token > 20 % der Matmul-Zeit der rotierten Schichten (CPU): 55 %.
Lies zuerst: formen, optimum, ampel_mlp / ampel_lm, dann tabelle und zaehlmodell."""
from __future__ import annotations
from Z3_bytegleich import lin_bytes, mlp_formen, lm_formen, lm_rest, form_bytes, pow2, MODI, LAGEN_
from _z2 import *

BITS = (("1.58", math.log2(3), 3), ("2", 2.0, 4), ("3", 3.0, 8), ("4", 4.0, 16), ("8", 8.0, 256), ("16", 16.0, 65536))
BITS_LM = tuple(b for b in BITS if b[0] in os.environ.get("Z4_LM_BITS", "1.58,2,4,8,16").split(","))     # LM: Bitbreiten reduziert (Laufzeit <= 30 min je Seed)
G_ROH = 32
NUR = os.environ.get("Z4_NUR", "")          # "mlp" oder "lm": nur dieses Modell (Zeitmessung)
ARTEN = ("roh_ptq", "roh_qat", "rot_ptq", "rot_qat")
LM_DMAX, LM_FMAX = 144, 576
# Energie-Konstanten (pJ), Größenordnung aus dem Gedächtnis (Horowitz 2014, 45 nm), Spanne ±3x, vor Zitat prüfen
E_DRAM = (50.0, 450.0)       # je Byte
E_SRAM = (1.0, 5.0)          # je Byte
E_MAC = (0.2, 1.0)           # INT8-MAC (kein 2-Bit-Kern angenommen: entpackt zu INT8)
E_MAC16 = (1.0, 2.0)         # fp16-MAC
E_ADD = (0.03, 0.2)          # 16-Bit-Addition


# ------------------------------------------------------------------ Quantisierer mit Skalenrückgabe
def quant_L(x, L, nf=12, skala=None):
    """Wie _z2.stufen_quant (letzte Achse, L Stufen, MSE-Suche), gibt (q, s). Mit skala: nur runden."""
    gerade = (L % 2 == 0)
    hi = L / 2 - 0.5 if gerade else (L - 1) / 2
    norm = L / 2 if gerade else hi

    def rund(s):
        return ((x / s - 0.5).round() + 0.5).clamp(-hi, hi) * s if gerade else (x / s).round().clamp(-hi, hi) * s
    if skala is not None:
        return rund(skala), skala
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best, bq, bs = None, None, None
    for f in np.linspace(0.3, 1.0, nf):
        s = amax * float(f) / norm
        q = rund(s)
        e = ((x - q) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bq, bs = e, q, s
        else:
            m = e < best
            best, bq, bs = torch.where(m, e, best), torch.where(m, q, bq), torch.where(m, s, bs)
    return bq, bs


def gq(W, L, g, nf=12, skala=None):
    """Gewicht (o x i) mit L Stufen, Skala je Gruppe von g Eingängen (g None: je Zeile). L>=65536: fp16-Rundung."""
    if L >= 65536:
        return W.half().float(), None
    o, i = W.shape
    if not g:
        return quant_L(W, L, nf, skala)
    ng = math.ceil(i / g)
    Wp = F.pad(W, (0, ng * g - i)).reshape(o * ng, g)
    q, s = quant_L(Wp, L, nf, skala)
    return q.reshape(o, ng * g)[:, :i], s


class BLin(QATLin):
    """QATLin mit beliebiger Stufenzahl L, Gruppenskala g und Auffüllung auf 2er-Potenz bei Rotation."""

    def __init__(self, lin, L, rot, g, seed):
        nn.Module.__init__(self)
        W = lin.weight.data.clone()
        self.i0, self.o = W.shape[1], W.shape[0]
        self.pad = (pow2(self.i0) - self.i0) if rot else 0
        if self.pad:
            W = F.pad(W, (0, self.pad))
        self.L, self.g, self.rot, self.bits = L, g, rot, L
        self.register_buffer("s", vorzeichen(W.shape[1], seed) if rot else torch.ones(1))
        self.V = nn.Parameter(rot_vor(W, self.s) if rot else W)
        self.b = nn.Parameter(lin.bias.data.clone())
        self.fest, self._skala, self._zaehl = None, None, 0

    def gewicht(self):
        if self.fest is not None:
            return self.fest
        if self.L >= 65536:
            return self.V
        with torch.no_grad():
            if self._skala is None or self._zaehl % 16 == 0:
                _, self._skala = gq(self.V.detach(), self.L, self.g, nf=8)
            self._zaehl += 1
            q, _ = gq(self.V.detach(), self.L, self.g, skala=self._skala)
        return self.V + (q - self.V).detach()

    def einfrieren(self):
        with torch.no_grad():
            self.fest = gq(self.V.detach(), self.L, self.g)[0]
        return self


def ersetze_b(netz, L, rot, g, seed):
    netz = copy.deepcopy(netz)
    k = [0]

    def rek(m):
        for name, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                k[0] += 1
                setattr(m, name, BLin(c, L, rot, g, seed * 101 + k[0]))
            else:
                rek(c)
    rek(netz)
    return netz


# ------------------------------------------------------------------ Formen
def modus(b, rot):
    """Bytes-Modus (Bits, Rotation, Gruppe, Vorskalierung) für Z3.lin_bytes."""
    return (b, rot, None if (rot or b >= 16) else G_ROH, False)


def mlp_form(md, B):
    best = 8
    for h in range(8, 1025):
        if form_bytes(mlp_formen(h), md) <= B:
            best = h
    return (best,)


def lm_form(md, B):
    kand = []
    for d in range(16, LM_DMAX + 1, 16):
        fm = 0
        for f in range(16, LM_FMAX + 1, 16):
            if form_bytes(lm_formen(d, f), md, lm_rest(d)) <= B:
                fm = f
        if fm:
            kand.append((abs(math.log(fm / d / 4)), -d, d, fm))
    return tuple(min(kand)[2:]) if kand else (16, 16)


def lm_bauen2(d, f):
    return LMX([FFNDichtX(d, f) for _ in range(LAGEN_)], d=d)


# ------------------------------------------------------------------ Zählmodell
_ZEIT = {}


def zeit_rot(w, o, N=2048, wdh=5):
    """Sekunden je Token: Vorzeichen*FWHT der Breite w, und Matmul (N x w) @ (w x o), beides 1 Faden, Minimum aus wdh."""
    if (w, o) not in _ZEIT:
        x, s, W = torch.randn(N, w), vorzeichen(w, 1), torch.randn(o, w)
        tr, tm = [], []
        with torch.no_grad():
            for _ in range(wdh):
                if w & (w - 1) == 0:
                    t = time.perf_counter(); rot_vor(x, s); tr.append(time.perf_counter() - t)
                t = time.perf_counter(); F.linear(x, W); tm.append(time.perf_counter() - t)
        _ZEIT[(w, o)] = (min(tr) / N if tr else 0.0, min(tm) / N)
    return _ZEIT[(w, o)]


def zaehle(formen, b, rot, rest=0, fp_macs=0):
    md = modus(b, rot)
    wb = form_bytes(formen, md, rest)
    ab = mac = add = vz = 0.0
    tr = tm = 0.0
    for o, i in formen:
        ip = pow2(i) if (rot and b < 16) else i
        ab += 2 * (ip + o)
        mac += o * ip
        if rot and b < 16:
            add += ip * math.log2(ip)
            vz += ip
            t1, t2 = zeit_rot(ip, o)
            tr += t1
            tm += t2
        else:
            tm += zeit_rot(ip, o)[1]
    macs_tot = mac + fp_macs
    fp16 = b >= 16
    mac_e = E_MAC16 if fp16 else E_MAC
    e = {}
    for nm, ew in (("dram", E_DRAM), ("sram", E_SRAM)):
        e[nm + "_min"] = wb * ew[0] + ab * E_SRAM[0] + macs_tot * mac_e[0] + (add + vz) * E_ADD[0]
        e[nm + "_max"] = wb * ew[1] + ab * E_SRAM[1] + macs_tot * mac_e[1] + (add + vz) * E_ADD[1]
    return {"gewichtsbytes": wb, "aktivierungsbytes": ab, "macs": macs_tot, "rot_additionen": add, "rot_vorzeichen": vz,
            "rot_zeit_je_token_s": tr, "matmul_zeit_je_token_s": tm, "rot_zeit_anteil": (tr / tm if tm else 0.0), "energie_pJ_je_token": e}


# ------------------------------------------------------------------ Auswertung
def optimum(tab, hoeher_besser, schwelle, tol):
    """tab[art][bitlabel] -> je Verfahren (ptq, qat, best) Optimum roh/rot."""
    sg = 1 if hoeher_besser else -1
    out = {}
    for ver in ("ptq", "qat", "best"):
        def kurve(kind):
            if ver == "best":
                return {bl: max(sg * tab[f"{kind}_ptq"][bl], sg * tab[f"{kind}_qat"][bl]) * sg for bl in tab["roh_ptq"]}
            return dict(tab[f"{kind}_{ver}"])
        r = {}
        for kind in ("roh", "rot"):
            k = kurve(kind)
            bl = max(k, key=lambda x: sg * k[x])
            bestv = k[bl]
            innerhalb = [x for x in k if sg * (bestv - k[x]) <= tol]
            bl_tol = min(innerhalb, key=lambda x: float(x))
            r[kind] = {"bit": bl, "wert": bestv, "bit_tol": bl_tol, "kurve": k}
        gew = sg * (r["rot"]["wert"] - r["roh"]["wert"])
        out[ver] = {"roh_bit": r["roh"]["bit"], "rot_bit": r["rot"]["bit"], "roh_best": r["roh"]["wert"], "rot_best": r["rot"]["wert"],
                    "roh_bit_tol": r["roh"]["bit_tol"], "rot_bit_tol": r["rot"]["bit_tol"], "gewinn": gew,
                    "bit_verschiebung": float(r["rot"]["bit"]) - float(r["roh"]["bit"]), "kurve_roh": r["roh"]["kurve"], "kurve_rot": r["rot"]["kurve"]}
    return out


def ampel_bit(opt, laengen, schwelle, einheit, skala_gewinn=1.0, ver="best"):
    zellen = {T: opt[T][ver] for T in laengen}
    runter = [z["bit_verschiebung"] < 0 for z in zellen.values()]
    gew = [z["gewinn"] * skala_gewinn for z in zellen.values()]
    if all(runter) and all(g > schwelle for g in gew):
        farbe = "gruen"
    elif (all(g > 0 for g in gew) or all(runter)) and not (not any(runter) and not any(g > 0 for g in gew)):
        farbe = "gelb"
    else:
        farbe = "rot"
    return {"farbe": farbe, "verschiebung_nach_unten_je_laenge": {str(T): r for T, r in zip(laengen, runter)},
            "gewinn_je_laenge": {str(T): g for T, g in zip(laengen, gew)}, "schwelle": schwelle, "einheit": einheit,
            "bit_roh": {str(T): z["roh_bit"] for T, z in zellen.items()}, "bit_rot": {str(T): z["rot_bit"] for T, z in zellen.items()},
            "verfahren": ver, "vorlaeufig_ein_seed": True}


def tabelle_fuellen(tab, mess, netz_fp, netz_qat_bau, formen, trainiere_qat, instabil=None, bits=BITS):
    """Füllt tab[art][bitlabel]. netz_fp(form) -> trainiertes fp-Netz (gecacht); netz_qat_bau(form) -> frisches Netz; trainiere_qat(q) -> instabil?"""
    for bl, _b, L in bits:
        for kind, rot in (("roh", False), ("rot", True)):
            form = formen[(bl, kind)]
            g = None if (rot or L >= 65536) else G_ROH
            if L >= 65536:
                if kind == "roh":
                    v = mess(fp16_runden(netz_fp(form)))
                    tab["roh_ptq"][bl] = tab["roh_qat"][bl] = v
                else:
                    tab["rot_ptq"][bl], tab["rot_qat"][bl] = tab["roh_ptq"][bl], tab["roh_qat"][bl]
                continue
            tab[f"{kind}_ptq"][bl] = mess(einfrieren(ersetze_b(netz_fp(form), L, rot, g, SEEDV[0])))
            q = ersetze_b(netz_qat_bau(form), L, rot, g, SEEDV[0])
            inst = trainiere_qat(q)
            if inst:
                instabil[f"{kind}_qat_{bl}"] = True
            tab[f"{kind}_qat"][bl] = None if inst else mess(einfrieren(q))
            if inst:
                tab[f"{kind}_qat"][bl] = INSTABIL_WERT[0]
    return tab


SEEDV, INSTABIL_WERT = [0], [0.0]


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    SEEDV[0] = seed
    eps = (1, 2) if schnell else (2, 6)
    Ts = (40, 160) if schnell else (80, 320)
    xtr, ytr, xva, yva = mlp_daten(schnell)
    t_ges = time.time()
    out = {"mlp_epochen": eps, "lm_schritte": Ts, "val_bytes": VAL_N, "bits_mlp": [b[0] for b in BITS], "bits_lm": [b[0] for b in BITS_LM], "g_roh": G_ROH, "tabelle": {"mlp": {}, "lm": {}},
           "formen": {"mlp": {}, "lm": {}}, "bytes": {"mlp": {}, "lm": {}}, "optimum": {"mlp": {}, "lm": {}}, "zaehlmodell": {"mlp": {}, "lm": {}}, "instabil": {}}
    leer = lambda: {a: {} for a in ARTEN}
    # ---------------------------------------------------------------- Budget und Formen
    B_mlp = form_bytes(mlp_formen(512), MODI["gross_rot"])
    fm = {(bl, k): mlp_form(modus(b, k == "rot"), B_mlp) for bl, b, L in BITS for k in ("roh", "rot")}
    out["formen"]["mlp"] = {f"{bl}_{k}": list(v) for (bl, k), v in fm.items()}
    out["bytes"]["mlp"] = {"budget": B_mlp, **{f"{bl}_{k}": form_bytes(mlp_formen(v[0]), modus(dict((x[0], x[1]) for x in BITS)[bl], k == "rot")) for (bl, k), v in fm.items()}}
    for kk, v in out["bytes"]["mlp"].items():
        assert v <= B_mlp + 1e-6, kk
    B_lm = form_bytes(lm_formen(128, 512), MODI["gross_rot"], lm_rest(128))
    fl = {(bl, k): lm_form(modus(b, k == "rot"), B_lm) for bl, b, L in BITS_LM for k in ("roh", "rot")}
    out["formen"]["lm"] = {f"{bl}_{k}": list(v) for (bl, k), v in fl.items()}
    out["bytes"]["lm"] = {"budget": B_lm, **{f"{bl}_{k}": form_bytes(lm_formen(*v), modus(dict((x[0], x[1]) for x in BITS)[bl], k == "rot"), lm_rest(v[0])) for (bl, k), v in fl.items()}}
    for kk, v in out["bytes"]["lm"].items():
        assert v <= B_lm + 1e-6, kk
    print("Formen MLP", out["formen"]["mlp"], "\nFormen LM", out["formen"]["lm"], flush=True)
    # ---------------------------------------------------------------- Zählmodell
    bd = dict((x[0], x[1]) for x in BITS)
    for bl, b, L in BITS:
        for k in ("roh", "rot"):
            out["zaehlmodell"]["mlp"][f"{bl}_{k}"] = zaehle(mlp_formen(fm[(bl, k)][0]), b, k == "rot")
            if (bl, k) not in fl:
                continue
            d, f = fl[(bl, k)]
            out["zaehlmodell"]["lm"][f"{bl}_{k}"] = zaehle(lm_formen(d, f), b, k == "rot", lm_rest(d), LAGEN_ * d * (KONTEXT + 1) + 256 * d)
    for m_ in ("mlp", "lm"):
        print(f"Zählmodell {m_}: " + " | ".join(f"{k}: W {v['gewichtsbytes']:.0f}B MAC {v['macs']:.2e} rotAdd {v['rot_additionen']:.2e} tRot {v['rot_zeit_je_token_s'] * 1e6:.1f}us ({100 * v['rot_zeit_anteil']:.0f}% Matmul)" for k, v in out["zaehlmodell"][m_].items() if k.endswith("rot") and k[:-4] in ("2", "4")), flush=True)
    # ---------------------------------------------------------------- MLP
    INSTABIL_WERT[0] = 0.0
    for ep in (eps if NUR != "lm" else ()):
        t = time.time()
        cache = {}

        def fp(form):
            if form not in cache:
                torch.manual_seed(seed)
                n = kb.baue_bild([form[0], form[0]])
                mlp_train(n, xtr, ytr, seed, ep, 1e-3)
                cache[form] = n
            return cache[form]

        def bau(form):
            torch.manual_seed(seed)
            return kb.baue_bild([form[0], form[0]])
        mess = lambda q: acc(q, xva, yva)
        tab = leer()
        tabelle_fuellen(tab, mess, fp, bau, fm, lambda q: mlp_train(q, xtr, ytr, seed, ep, 1e-3) is None, {})
        out["tabelle"]["mlp"][ep] = tab
        out["optimum"]["mlp"][ep] = optimum(tab, True, 0.005, 0.001)
        print(f"Z4_bitkurve s{seed} MLP ep{ep} ({time.time() - t:.0f}s):", flush=True)
        for a in ARTEN:
            print(f"   {a:8s} " + " ".join(f"{bl} {tab[a][bl]:.4f}" for bl, _, _ in BITS), flush=True)
    # ---------------------------------------------------------------- LM
    mess = lambda q: val_bpc(q, schnell)
    INSTABIL_WERT[0] = 8.0
    for T in (Ts if NUR != "mlp" else ()):
        t = time.time()
        cache = {}

        def fp(form):
            if form not in cache:
                torch.manual_seed(seed)
                n = lm_bauen2(*form)
                lm_trainiere(n, seed, T, schnell=schnell)
                cache[form] = n.eval()
            return cache[form]

        def bau(form):
            torch.manual_seed(seed)
            return lm_bauen2(*form)
        tab = leer()
        inst = {}
        tabelle_fuellen(tab, mess, fp, bau, fl, lambda q: bool(lm_trainiere(q, seed, T, schnell=schnell)["instabil"]), inst, BITS_LM)
        tab["instabil"] = inst
        out["tabelle"]["lm"][T] = tab
        out["optimum"]["lm"][T] = optimum({a: tab[a] for a in ARTEN}, False, 0.01, 0.003)
        print(f"Z4_bitkurve s{seed} LM T{T} ({time.time() - t:.0f}s):", flush=True)
        for a in ARTEN:
            print(f"   {a:8s} " + " ".join(f"{bl} {tab[a][bl]:.4f}" for bl, _, _ in BITS_LM), flush=True)
    # ---------------------------------------------------------------- Ampeln und Optima
    if NUR:
        return out
    out["ampel_mlp"] = ampel_bit(out["optimum"]["mlp"], eps, 0.5, "pp", 100.0)
    out["ampel_lm"] = ampel_bit(out["optimum"]["lm"], Ts, 0.01, "bpc")
    for nm, o_, ls in (("mlp", out["optimum"]["mlp"], eps), ("lm", out["optimum"]["lm"], Ts)):
        for T in ls:
            for ver in ("ptq", "qat", "best"):
                z = o_[T][ver]
                print(f"Optimum {nm} {T} {ver}: roh {z['roh_bit']} ({z['roh_best']:.4f}) rot {z['rot_bit']} ({z['rot_best']:.4f}) Tol roh {z['roh_bit_tol']} rot {z['rot_bit_tol']}", flush=True)
    print("Ampel MLP", out["ampel_mlp"]["farbe"], out["ampel_mlp"]["gewinn_je_laenge"], out["ampel_mlp"]["verschiebung_nach_unten_je_laenge"], "\nAmpel LM", out["ampel_lm"]["farbe"], out["ampel_lm"]["gewinn_je_laenge"], out["ampel_lm"]["verschiebung_nach_unten_je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z4_bitkurve", main)
