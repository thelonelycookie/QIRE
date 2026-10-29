"""Z3_bytegleich -- Hält "rotiertes 2-Bit-Großmodell schlägt klein fp16 gleicher Bytes (+0,54 pp)" gegen FAIRE Gegner gleicher Bytes? (Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
ANLASS (REAKTOR R5/R4/R1, Pflicht-Gegner G4-G7): Klein-fp16 ist ein schwacher Gegner (ParetoQ, RCP). Stärkere Gegner bei exakt/höchstens gleichen Bytes INKLUSIVE aller Skalen:
FRAGE: Schlägt Groß-2-Bit-rotiert (bester von PTQ und QAT) den besten Arm OHNE Rotation, bei beiden Trainingslängen?
BYTEBUDGET B = Bytes des rotierten 2-Bit-Großmodells (Gewicht 2 Bit, Zeilenskala fp16, Bias fp16, Vorzeichen 1 Bit/Eingang, MNIST-Eingang 784->1024 gepolstert; LM: plus fp16-Rest wie Z1/Z2).
  Jeder Arm bekommt die größte Form (MLP: Breite h-h; LM: d in 16er-Schritten, FFN-Breite f in 16er-Schritten, f/d möglichst nahe 4, f<=512, d<=128) mit Bytes <= B. Bytes je Arm stehen im Feld "formen".
  Die Form wird mit Bytes der Skalen gerechnet, daher sind Arme mit Zusatzskalen (Gruppe, Vorskalierung) KLEINER trainiert (eigenes Netz, gleiche Länge, Seed+1). Der Big-Arm trainiert mit Seed.
ARME:
  gross_rot_ptq   Groß 2 Bit, Vorzeichen*Hadamard, Zeilenskala, PTQ (= Z1/Z2 "rot")           gross_rot_qat   dasselbe, aber QAT von Anfang an (G7 rot, STE, Skala wie QATLin)
  gross_roh_ptq   Groß 2 Bit roh, Zeilenskala (billig, nur Referenz)                          gross_roh_qat   Groß 2 Bit roh QAT (G7)
  roh_g16/32/64   roh 2 Bit, Skala je Gruppe von g Eingängen (fp16 je Gruppe, Zeilenskala entfällt), Bytes zählen -> kleinere Form (G4)
  roh_vor         roh 2 Bit, Kanal-/RMS-Vorskalierung: W'=W*c, c_j = RMS_j der fp-Eingangsaktivierung (Kalibrierung 256 Bilder / 64 LM-Fenster), quantisiert W', zurückgerechnet W'q/c; c fp16 (2 B je Eingang) zählt (G5)
  mittel4_roh/rot Mittelmodell 4 Bit (RTN, Zeilenskala), roh bzw. Vorzeichen*Hadamard, eigene Form gleiche Bytes (G6)
  klein           fp16, gelernt, gleiche Bytes (hier mit derselben Formregel wie die anderen Arme; im LM daher feiner als Z1/Z2, dort nur d in 16er-Schritten mit f=4d)
MODELLE/LÄNGEN: MLP MNIST 784-h-h-10, Epochen {3, 12} (--schnell {1, 2}), Adam 1e-3; Byte-LM d=128, 4 Lagen, Schritte {300, 1200} (--schnell {40, 160}; Verhältnis 1:4 wie Z1, damit ein Seed in <= 30 min läuft: ca. 0,9 s je Schritt für alle ~9 Netze zusammen), lr 2e-3. Maß: Genauigkeit bzw. Val-bpc (Z2_VAL Bytes, Standard 32768).
ZUSATZ R1 (ohne Training, auf den trainierten Groß-fp-Modellen): Steigung k des Fits log Var(Fehler_i) = c + k log Var(Signal_i) je Koordinate i bei 2 Bit (4 Stufen), Fehler im Originalkoordinatensystem.
  Aktivierung: MLP-Eingang der Schichten 2 und 3 (Breite 512), LM FFN-Mitte (Eingang der Down-Projektion, 4 Lagen); je Token quantisiert. Gewicht: Fehler je Eingangsspalte (zweites Moment statt Varianz, Gewichte mittelwertfrei), alle Schichten mit Eingangsbreite 2er-Potenz.
  Varianten: roh, rot (Hadamard, zurückgedreht), g16/g32/g64 (Gruppenskala), kanal (RMS-Vorskalierung). Hypothese: roh k nahe 1 (multiplikativ), rot k nahe 0 (additiv). Langweilig: Gruppe/Kanal senken k genauso. Feld "r1".
VORAB-P (vor dem Lauf): Ampel MLP grün (Groß-rot schlägt besten Nicht-Rotationsarm um > 0,5 pp, Minimum über Längen) 25 %; LM grün (> 0,01 bpc) 20 %; gelb bis grün (Gewinn > 0) MLP 40 %, LM 35 %.
  Bester Gegner im MLP: roh_g16 oder QAT-roh (60 %), Mittel-4-Bit roh 25 %. R1: k_roh - k_rot >= 0,3 bei Aktivierung 65 %; Gruppe g16 senkt k mindestens halb so stark wie Rotation 40 %.
AMPEL (Einzelseed, vorläufig): "ampel_mlp" (Schwelle 0,5 pp), "ampel_lm" (0,01 bpc): gewinn = Minimum über Längen von [bester Rot-Arm (gross_rot_ptq, gross_rot_qat) - bester Nicht-Rot-Arm (gross_roh_ptq, gross_roh_qat, roh_g16/32/64, roh_vor, mittel4_roh, klein)].
  Zusatz: "ampel_*_mit_mittel_rot" (Rot-Arme dürfen auch mittel4_rot sein) und "ampel_*_alt" (nur gross_rot_ptq gegen klein, wie Z1).
Lies zuerst: tabelle[<modell>][<länge>] (Werte), formen, dann ampel_mlp / ampel_lm, dann r1."""
from __future__ import annotations
from _z2 import *
from Z2_rauschformung import kalib_eingaenge, lm_kalib

LM_D, LM_F, LAGEN_ = 128, 512, 4
# Modus = (Bits, Rotation, Gruppe, Vorskalierung)
MODI = {
    "gross_rot": (2, True, None, False), "gross_roh": (2, False, None, False),
    "roh_g16": (2, False, 16, False), "roh_g32": (2, False, 32, False), "roh_g64": (2, False, 64, False),
    "roh_vor": (2, False, None, True), "mittel4_roh": (4, False, None, False), "mittel4_rot": (4, True, None, False),
    "klein": (16, False, None, False),
}
ROT_ARME = ("gross_rot_ptq", "gross_rot_qat")
NICHT_ROT = ("gross_roh_ptq", "gross_roh_qat", "roh_g16", "roh_g32", "roh_g64", "roh_vor", "mittel4_roh", "klein")


# ------------------------------------------------------------------ Bytes
def pow2(i):
    return 1 << (int(i) - 1).bit_length()


class RLin(QATLin):
    """Wie QATLin, aber der Eingang wird bei Rotation auf die nächste 2er-Potenz mit Nullen aufgefüllt (beliebige Breiten); Bytes zählen die gepolsterte Breite."""

    def __init__(self, lin, bits, rot, seed):
        nn.Module.__init__(self)
        W = lin.weight.data.clone()
        self.i0, self.o = W.shape[1], W.shape[0]
        self.pad = (pow2(self.i0) - self.i0) if rot else 0
        if self.pad:
            W = F.pad(W, (0, self.pad))
        self.bits, self.rot = bits, rot
        self.register_buffer("s", vorzeichen(W.shape[1], seed) if rot else torch.ones(1))
        self.V = nn.Parameter(rot_vor(W, self.s) if rot else W)
        self.b = nn.Parameter(lin.bias.data.clone())
        self.fest, self._skala, self._zaehl = None, None, 0


def ersetze_linear(netz, bits, rot, seed):
    """Wie _z1.ersetze_linear (gleiche Seeds seed*101+k), mit RLin."""
    netz = copy.deepcopy(netz)
    k = [0]

    def rek(m):
        for name, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                k[0] += 1
                setattr(m, name, RLin(c, bits, rot, seed * 101 + k[0]))
            else:
                rek(c)
    rek(netz)
    return netz

def lin_bytes(o, i, modus):
    bits, rot, g, vor = modus
    if bits >= 16:
        return 2 * o * i + 2 * o
    ip = pow2(i) if rot else i
    sk = o * math.ceil(ip / (g or ip)) * 2
    return o * ip * bits / 8 + sk + 2 * o + (ip / 8 if rot else 0) + (2 * ip if vor else 0)


def mlp_formen(h):
    return [(h, 784), (h, h), (10, h)]


def lm_formen(d, f):
    return LAGEN_ * [(3 * d, d), (d, d), (f, d), (d, f)]


def lm_rest(d):
    return 2 * (256 * d + 128 * d + LAGEN_ * 4 * d + 2 * d)       # Einbettung, Position, 2 LayerNorm je Lage, Endnorm (fp16, alle Arme gleich)


def form_bytes(formen, modus, rest=0):
    return float(sum(lin_bytes(o, i, modus) for o, i in formen) + rest)


def mlp_form(modus, B):
    best = 8
    for h in range(8, 1025):
        if form_bytes(mlp_formen(h), modus) <= B:
            best = h
    return (best,)


def lm_form(modus, B):
    """(d, f): d in 16er-Schritten <= 128, f in 16er-Schritten <= 512 maximal mit Bytes <= B; unter den d das mit f/d am nächsten an 4 (Gleichstand: größeres d)."""
    kand = []
    for d in range(16, LM_D + 1, 16):
        fm = 0
        for f in range(16, LM_F + 1, 16):
            if form_bytes(lm_formen(d, f), modus, lm_rest(d)) <= B:
                fm = f
        if fm:
            kand.append((abs(math.log(fm / d / 4)), -d, d, fm))
    return tuple(min(kand)[2:]) if kand else (16, 16)


def lm_bauen2(d, f):
    return LMX([FFNDichtX(d, f) for _ in range(LAGEN_)], d=d)


# ------------------------------------------------------------------ PTQ ohne Rotation
def gruppen_quant(W, bits, g):
    """Skala je Gruppe von g Eingängen (letzte Achse; Rest wird mit Nullen aufgefüllt), MSE-Suche wie quant_gitter."""
    o, i = W.shape
    ng = math.ceil(i / g)
    Wp = F.pad(W, (0, ng * g - i)).reshape(o * ng, g)
    return quant_gitter(Wp, bits).reshape(o, ng * g)[:, :i]


def linears(netz):
    return [m for m in netz.modules() if isinstance(m, nn.Linear)]


def ptq_roh(netz, bits, g=None, ein=None):
    """Tiefe Kopie, jede nn.Linear roh quantisiert (Zeilenskala | Gruppenskala | RMS-Vorskalierung mit ein = Eingangsaktivierungen je Linear)."""
    n = copy.deepcopy(netz)
    for k, m in enumerate(linears(n)):
        with torch.no_grad():
            W = m.weight.data
            if ein is not None:
                c = ein[k].pow(2).mean(0).sqrt()
                c = c.clamp_min(1e-2 * float(c.mean()) + 1e-8)
                m.weight.data = quant_gitter(W * c[None, :], bits) / c[None, :]
            elif g:
                m.weight.data = gruppen_quant(W, bits, g)
            else:
                m.weight.data = quant_gitter(W, bits)
    return n.eval()


# ------------------------------------------------------------------ R1: Rauschtyp
def k_fit(sig, err):
    ok = (sig > 1e-8 * float(sig.max())) & (err > 0)
    if int(ok.sum()) < 8:
        return {"k": float("nan"), "r2": float("nan"), "n": int(ok.sum())}
    lx, ly = torch.log(sig[ok]).double().numpy(), torch.log(err[ok]).double().numpy()
    k, c = np.polyfit(lx, ly, 1)
    res = ly - (k * lx + c)
    return {"k": float(k), "r2": float(1 - res.var() / max(ly.var(), 1e-12)), "n": int(ok.sum())}


def r1_akt(X, seed, bits=2):
    L = 2 ** bits
    N, n = X.shape
    V = {"roh": stufen_quant(X, L), "kanal": None}
    Q = rot_q(n, "hadamard", seed)
    V["rot"] = quant_rot(X, L, Q)
    for g in (16, 32, 64):
        V[f"g{g}"] = stufen_quant(X.reshape(N, n // g, g), L).reshape(N, n)
    c = X.pow(2).mean(0).sqrt().clamp_min(1e-6)
    V["kanal"] = stufen_quant(X / c, L) * c
    sig = X.var(0)
    return {k: {**k_fit(sig, (v - X).var(0)), "mse_rel": float(((v - X) ** 2).sum() / (X ** 2).sum())} for k, v in V.items()}


def r1_gew(W, ein_i, seed, bits=2):
    o, i = W.shape
    V = {"roh": quant_gitter(W, bits)}
    if i & (i - 1) == 0:
        Q = rot_q(i, "hadamard", seed)
        V["rot"] = quant_gitter(W @ Q.T, bits) @ Q
    for g in (16, 32, 64):
        V[f"g{g}"] = gruppen_quant(W, bits, g)
    c = ein_i.pow(2).mean(0).sqrt()
    c = c.clamp_min(1e-2 * float(c.mean()) + 1e-8)
    V["kanal"] = quant_gitter(W * c[None, :], bits) / c[None, :]
    sig = (W ** 2).mean(0)
    return {k: {**k_fit(sig, ((v - W) ** 2).mean(0)), "mse_rel": float(((v - W) ** 2).sum() / (W ** 2).sum())} for k, v in V.items()}


def r1_mittel(teile):
    """Mittelwert von k je Variante über die Schichten."""
    out = {}
    for v in next(iter(teile.values())):
        ks = [t[v]["k"] for t in teile.values() if v in t and t[v]["k"] == t[v]["k"]]
        out[v] = float(np.mean(ks)) if ks else float("nan")
    return out


def r1_modell(netz, akt_liste, ein_alle, seed):
    """akt_liste: {Name: (N x n)-Aktivierung}; ein_alle: Eingaben aller nn.Linear (modules()-Reihenfolge) für Vorskalierung und Gewichtsfehler."""
    akt = {nm: r1_akt(X, seed) for nm, X in akt_liste.items()}
    gew = {}
    for k, m in enumerate(linears(netz)):
        if m.weight.shape[1] in (784, 10):
            continue
        gew[f"lin{k}_{m.weight.shape[0]}x{m.weight.shape[1]}"] = r1_gew(m.weight.data, ein_alle[k], seed)
    return {"akt": akt, "gew": gew, "k_mittel_akt": r1_mittel(akt), "k_mittel_gew": r1_mittel(gew)}


def lm_ffn_mitte(netz, fwd):
    buf = [[] for _ in netz.bl]
    hs = [b.ffn.b.register_forward_hook(lambda mod, inp, o, k=k: buf[k].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))) for k, b in enumerate(netz.bl)]
    netz.eval()
    with torch.no_grad():
        fwd(netz)
    for h in hs:
        h.remove()
    return {f"ffn_mitte_L{k}": torch.cat(b) for k, b in enumerate(buf)}


# ------------------------------------------------------------------ Ampel
def ampel_gegen(tab, laengen, rot_arme, gegner, wert, schwelle, einheit, text):
    """Gewinn = Minimum über Längen von [bester Rot-Arm - bester Gegner] in der Einheit; wert(tab[länge][arm]) größer = besser."""
    g, detail = [], {}
    for T in laengen:
        r = {a: wert(tab[T][a]) for a in rot_arme if a in tab[T]}
        n_ = {a: wert(tab[T][a]) for a in gegner if a in tab[T]}
        br, bn = max(r, key=r.get), max(n_, key=n_.get)
        g.append(r[br] - n_[bn])
        detail[T] = {"bester_rot": br, "bester_gegner": bn, "differenz": r[br] - n_[bn]}
    a = ampel(min(g), schwelle, einheit, text)
    a["je_laenge"] = detail
    return a


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 2) if schnell else (3, 12)
    Ts = (40, 160) if schnell else (300, 1200)
    xtr, ytr, xva, yva = mlp_daten(schnell)
    out = {"mlp_epochen": eps, "lm_schritte": Ts, "val_bytes": VAL_N, "modi": {k: list(v) for k, v in MODI.items()}, "tabelle": {"mlp": {}, "lm": {}}, "formen": {"mlp": {}, "lm": {}}, "bytes": {"mlp": {}, "lm": {}}, "r1": {"mlp": {}, "lm": {}}}
    # ---------------------------------------------------------------- MLP
    B_mlp = form_bytes(mlp_formen(512), MODI["gross_rot"])
    assert abs(B_mlp - bytes_q(einfrieren(ersetze_linear(kb.baue_bild([512, 512]), 2, True, 0)))) < 1e-6
    formen = {}
    for nm, md in MODI.items():
        formen[nm] = mlp_form(md, B_mlp) if nm != "gross_rot" else (512,)
    formen["gross_roh"] = (512,)       # billiger als B (Referenz)
    out["formen"]["mlp"] = {k: list(v) for k, v in formen.items()}
    out["bytes"]["mlp"] = {"budget": B_mlp, **{k: form_bytes(mlp_formen(v[0]), MODI[k]) for k, v in formen.items()}}
    for nm in formen:
        assert out["bytes"]["mlp"][nm] <= B_mlp + 1e-6, nm
    for ep in eps:
        t = time.time()
        cache = {}

        def netz(h):
            if h not in cache:
                torch.manual_seed(seed if h == 512 else seed + 1)
                n = kb.baue_bild([h, h])
                mlp_train(n, xtr, ytr, seed, ep, 1e-3)
                cache[h] = n
            return cache[h]
        mess = lambda q: acc(q, xva, yva)
        big = netz(512)
        a0 = mess(big)
        tab = {"fp32": a0}
        tab["gross_rot_ptq"] = mess(einfrieren(ersetze_linear(big, 2, True, seed)))
        tab["gross_roh_ptq"] = mess(ptq_roh(big, 2))
        for g in (16, 32, 64):
            tab[f"roh_g{g}"] = mess(ptq_roh(netz(formen[f"roh_g{g}"][0]), 2, g))
        h = formen["roh_vor"][0]
        nv = netz(h)
        tab["roh_vor"] = mess(ptq_roh(nv, 2, ein=kalib_eingaenge(nv, lambda m: m(xtr[:256]))))
        tab["mittel4_roh"] = mess(ptq_roh(netz(formen["mittel4_roh"][0]), 4))
        tab["mittel4_rot"] = mess(einfrieren(ersetze_linear(netz(formen["mittel4_rot"][0]), 4, True, seed)))
        tab["klein"] = mess(fp16_runden(netz(formen["klein"][0])))
        for rot in (False, True):
            torch.manual_seed(seed)
            q = ersetze_linear(kb.baue_bild([512, 512]), 2, rot, seed)
            mlp_train(q, xtr, ytr, seed, ep, 1e-3)
            tab["gross_rot_qat" if rot else "gross_roh_qat"] = mess(einfrieren(q))
        out["tabelle"]["mlp"][ep] = tab
        ein = kalib_eingaenge(big, lambda m: m(xtr[:2048]))
        out["r1"]["mlp"][ep] = r1_modell(big, {"mlp_h1": ein[1], "mlp_h2": ein[2]}, ein, seed)
        print(f"Z3_bytegleich s{seed} MLP ep{ep}: fp32 {a0:.4f} | " + " ".join(f"{k} {v:.4f}" for k, v in tab.items() if k != "fp32") + f" ({time.time() - t:.0f}s)", flush=True)
        print("   R1 k(akt) " + " ".join(f"{k} {v:.2f}" for k, v in out["r1"]["mlp"][ep]["k_mittel_akt"].items()) + " | k(gew) " + " ".join(f"{k} {v:.2f}" for k, v in out["r1"]["mlp"][ep]["k_mittel_gew"].items()), flush=True)
    # ---------------------------------------------------------------- LM
    B_lm = form_bytes(lm_formen(LM_D, LM_F), MODI["gross_rot"], lm_rest(LM_D))
    n_ = lm_bauen2(LM_D, LM_F)
    assert abs(B_lm - (bytes_q(einfrieren(ersetze_linear(n_, 2, True, 0))) + kb.bytes_rest_fp16(n_))) < 1e-6
    fl = {nm: lm_form(md, B_lm) for nm, md in MODI.items()}
    fl["gross_rot"] = (LM_D, LM_F)
    fl["gross_roh"] = (LM_D, LM_F)
    out["formen"]["lm"] = {k: list(v) for k, v in fl.items()}
    out["bytes"]["lm"] = {"budget": B_lm, **{k: form_bytes(lm_formen(*v), MODI[k], lm_rest(v[0])) for k, v in fl.items()}}
    for nm in fl:
        assert out["bytes"]["lm"][nm] <= B_lm + 1e-6, nm
    print("Formen MLP", out["formen"]["mlp"], "\nFormen LM", out["formen"]["lm"], flush=True)
    mess = lambda q: val_bpc(q, schnell)
    for T in Ts:
        t = time.time()
        cache = {}

        def netz(df):
            if df not in cache:
                torch.manual_seed(seed if df == (LM_D, LM_F) else seed + 1)
                n = lm_bauen2(*df)
                lm_trainiere(n, seed, T, schnell=schnell)
                cache[df] = n.eval()
            return cache[df]
        big = netz((LM_D, LM_F))
        v0 = mess(big)
        tab = {"fp32": v0}
        tab["gross_rot_ptq"] = mess(einfrieren(ersetze_linear(big, 2, True, seed)))
        tab["gross_roh_ptq"] = mess(ptq_roh(big, 2))
        for g in (16, 32, 64):
            tab[f"roh_g{g}"] = mess(ptq_roh(netz(fl[f"roh_g{g}"]), 2, g))
        nv = netz(fl["roh_vor"])
        tab["roh_vor"] = mess(ptq_roh(nv, 2, ein=kalib_eingaenge(nv, lm_kalib(seed, schnell))))
        tab["mittel4_roh"] = mess(ptq_roh(netz(fl["mittel4_roh"]), 4))
        tab["mittel4_rot"] = mess(einfrieren(ersetze_linear(netz(fl["mittel4_rot"]), 4, True, seed)))
        tab["klein"] = mess(fp16_runden(netz(fl["klein"])))
        inst = {}
        for rot in (False, True):
            torch.manual_seed(seed)
            q = ersetze_linear(lm_bauen2(LM_D, LM_F), 2, rot, seed)
            r = lm_trainiere(q, seed, T, schnell=schnell)
            nm = "gross_rot_qat" if rot else "gross_roh_qat"
            inst[nm] = bool(r["instabil"])
            tab[nm] = 8.0 if r["instabil"] else mess(einfrieren(q))
        tab["instabil"] = inst
        out["tabelle"]["lm"][T] = tab
        ein = kalib_eingaenge(big, lm_kalib(seed, schnell))
        akt = lm_ffn_mitte(big, lm_kalib(seed, schnell))
        out["r1"]["lm"][T] = r1_modell(big, akt, ein, seed)
        print(f"Z3_bytegleich s{seed} LM T{T}: fp32 {v0:.4f} | " + " ".join(f"{k} {v:.4f}" for k, v in tab.items() if k not in ("fp32", "instabil")) + f" ({time.time() - t:.0f}s)", flush=True)
        print("   R1 k(akt) " + " ".join(f"{k} {v:.2f}" for k, v in out["r1"]["lm"][T]["k_mittel_akt"].items()) + " | k(gew) " + " ".join(f"{k} {v:.2f}" for k, v in out["r1"]["lm"][T]["k_mittel_gew"].items()), flush=True)
    # ---------------------------------------------------------------- Ampeln
    m, l = out["tabelle"]["mlp"], out["tabelle"]["lm"]
    wm, wl = (lambda a: 100 * a), (lambda b: -b)
    out["ampel_mlp"] = ampel_gegen(m, eps, ROT_ARME, NICHT_ROT, wm, 0.5, "pp", "bester Rot-Arm (PTQ/QAT) - bester Nicht-Rot-Arm gleicher Bytes, MLP 2 Bit, Minimum über Längen")
    out["ampel_lm"] = ampel_gegen(l, Ts, ROT_ARME, NICHT_ROT, wl, 0.01, "bpc", "bester Rot-Arm (PTQ/QAT) - bester Nicht-Rot-Arm gleicher Bytes, LM 2 Bit, Minimum über Längen")
    out["ampel_mlp_mit_mittel_rot"] = ampel_gegen(m, eps, ROT_ARME + ("mittel4_rot",), NICHT_ROT, wm, 0.5, "pp", "wie ampel_mlp, Rot-Arme dürfen auch mittel4_rot sein")
    out["ampel_lm_mit_mittel_rot"] = ampel_gegen(l, Ts, ROT_ARME + ("mittel4_rot",), NICHT_ROT, wl, 0.01, "bpc", "wie ampel_lm, Rot-Arme dürfen auch mittel4_rot sein")
    out["ampel_mlp_alt"] = ampel_gegen(m, eps, ("gross_rot_ptq",), ("klein",), wm, 0.5, "pp", "gross_rot_ptq - klein (Z1-Stand)")
    out["ampel_lm_alt"] = ampel_gegen(l, Ts, ("gross_rot_ptq",), ("klein",), wl, 0.01, "bpc", "gross_rot_ptq - klein (Z1-Stand)")
    print("Ampel MLP", out["ampel_mlp"]["farbe"], f"{out['ampel_mlp']['gewinn']:.2f}pp", out["ampel_mlp"]["je_laenge"], "\nAmpel LM", out["ampel_lm"]["farbe"], f"{out['ampel_lm']['gewinn']:.4f}bpc", out["ampel_lm"]["je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z3_bytegleich", main)
