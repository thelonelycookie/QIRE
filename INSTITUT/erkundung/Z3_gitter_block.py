"""Z3_gitter_block -- Teil B1 (Gitterachse) und Teil A1 (Hadamard-Blockgröße) der FFN-Mitte-Aktivierungsquantisierung (Ideen B1, A1 aus Z3_ideen; Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
AUFGABE (wie Z2_ptq_varianten ffn_gelu / ffn_z2): Byte-LM d=128, 4 Lagen, FFN 4d (Breite 512), Kontext 128; Aktivierung vor der Down-Projektion wird je Token quantisiert (Gewichte fp32).
  Aktivierungen GELU und z² (je ein eigenes Netz), zwei Trainingslängen T ∈ {600, 2400} (SCHNELL=1 / --schnell: {40, 160}), Bits ∈ {2, 3} (4 bzw. 8 Stufen). Schaden = bpc(quantisiert) − bpc(fp32).
TEIL B1, Gitter (Pegel in Einheiten der Skala s = Max_Token · f; die Skala f wird je Token per MSE aus 10 geometrischen Kandidaten 0,06..1 gewählt = MSE-optimales Clipping, für ALLE Arme gleich):
  unif0   : uniform mit Null, {−L/2..L/2−1}/(L/2), 2 Bit = {−2,−1,0,1}·Schritt (asymmetrisch, Standard-Zweierkomplement)
  nozero  : no-zero-Mittelpunktsgitter ±0,5, ±1,5, ... (Qift; = "sym" in Z2/Z3_ffn_fair)
  lloyd   : Lloyd-Max, Pegel aus Kalibrierdaten gelernt (je Lage; auf RMS-normierte Werte, asymmetrisch erlaubt, 40 Iterationen), Pegelwerte fp (Zusatz nur L Zahlen je Lage)
  mu      : µ-Gesetz-Companding: gleichmäßiges no-zero-Gitter im komprimierten Bereich y, Pegel x = sign(y)·((1+µ)^|y| − 1)/µ; µ ∈ {4,16,64,256} je Lage nach Kalibrier-MSE gewählt
  jeweils "roh" (Basis wie sie ist) und "had" (Vorzeichen*Hadamard über volle Breite 512, Quantisierung in rotierter Basis, Rückdrehung in W_down faltbar).
  AMPEL B1 (Einzelseed, vorläufig): je Aktivierung und Bits: gewinn = Minimum über T von (Schaden_bester_roher_Arm − Schaden_bester_rotierter_Arm), bpc, Schwelle 0,005 (wie Z1/Z2/Z3_ffn_fair).
TEIL A1: Hadamard blockdiagonal mit Block b ∈ {1 (nur Vorzeichen, Kontrolle), 8, 16, 32, 64, 128, 256, 512 (voll)}, mit dem besten rotierten Gitter aus B1 DERSELBEN Länge/Aktivierung/Bits (Auswahl auf Val-Schaden, leicht optimistisch).
  Gewinn(b) = Schaden(bester roher Arm) − Schaden(b). Anteil(b) = Gewinn(b)/Gewinn(512). Gilt nur, wenn Gewinn(512) > 0,005 bpc (sonst nan). Kleinstes b mit Anteil ≥ 0,9 (je Zelle; zusätzlich: Minimum über T).
  FLOP-Aufschlag je Token: online-Vorwärtshadamard 512·log2(b) Additionen je Lage (Rückdrehung in W_down gefaltet), relativ zu den MACs je Token des Modells (lm_macs, 1 MAC = 1 Flop wie _basis); zusätzlich analytisch für d = 512..4096 (Breite f = 4d, FFN-MACs 2·d·f).
VORAB-P (vor dem Lauf): Rotation bleibt grün gegen den besten rohen Arm mit fairem Gitter (Minimum über T, GELU 2 Bit) 50 %; z² 2 Bit 40 %; 3 Bit GELU 30 %. Lloyd/µ schließen > 50 % der Lücke roh-uniform zu rot: 35 %.
  A1: b = 32 hält ≥ 90 % des Gewinns (GELU 2 Bit) 50 %; b = 8 hält ≥ 90 % 20 %. Kontrolle b = 1 (nur Vorzeichen) bringt < 10 % des Gewinns: 85 %.
Nicht untersucht: Gewichte/KV, Kernelzeit (FWHT ist bei Batch 1 langsam), Bytes der Skalen (je Token gleich für alle Arme), Lernrate pro Arm. Kalibrierung: 16 Trainings-Fenster (2048 Tokens), nach dem Training, fp-Aktivierung.
Lies zuerst: ampel_<akt>_b<bits>, dann a1[akt][bits], dann schaden[akt][bits][arm][T]."""
from __future__ import annotations
import os
os.environ.setdefault("Z2_VAL", "16384")      # Val-Bytes für bpc (Laufzeit: Quantisierung dominiert; Standard in Z2 wäre 32768)
from _z2 import *

BITS = (2, 3)
N = 512
GITTER = ("unif0", "nozero", "lloyd", "mu")
BLOECKE = (1, 8, 16, 32, 64, 128, 256, 512)
FGRID = [float(f) for f in np.geomspace(0.06, 1.0, 10)]
MUS = (4.0, 16.0, 64.0, 256.0)


# ------------------------------------------------------------------ Gitter
def pegel_fest(gitter, L, mu=None):
    """Einheitsmax-Pegel (max|p| = 1) für feste Gitter."""
    if gitter == "unif0":
        p = torch.arange(-L // 2, L // 2, dtype=torch.float32) / (L / 2)
    elif gitter == "nozero":
        p = (torch.arange(L, dtype=torch.float32) - (L - 1) / 2) / (L / 2)
    else:  # mu
        y = (2 * torch.arange(L, dtype=torch.float32) + 1 - L) / L
        p = torch.sign(y) * ((1 + mu) ** y.abs() - 1) / mu
    return p / p.abs().max()


def lloyd_pegel(v, L, iters=40):
    """Lloyd-Max in 1-D auf Stichprobe v (RMS-normierte Werte); Rückgabe: Einheitsmax-Pegel, sortiert."""
    v = v.flatten()
    if len(v) > 400_000:
        v = v[torch.randperm(len(v))[:400_000]]
    v = v.sort().values
    p = torch.stack([v[int((i + 0.5) / L * (len(v) - 1))] for i in range(L)])
    for _ in range(iters):
        b = (p[1:] + p[:-1]) / 2
        idx = torch.bucketize(v, b)
        s = torch.zeros(L).index_add_(0, idx, v)
        c = torch.bincount(idx, minlength=L).float()
        p = torch.where(c > 0, s / c.clamp_min(1), p)
        p = p.sort().values
    return p / p.abs().max().clamp_min(1e-9)


def quant_pegel(x, p):
    """x (n x 512) -> je Zeile Skala s = Max·f (f aus FGRID per MSE), Rundung auf Pegel p·s (nächster Pegel, Werte außerhalb werden auf den Randpegel gekappt)."""
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    b = (p[1:] + p[:-1]) / 2
    best, bq = None, None
    for f in FGRID:
        s = amax * f
        q = p[torch.bucketize(x / s, b)] * s
        e = ((x - q) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bq = e, q
        else:
            m = e < best
            best, bq = torch.where(m, e, best), torch.where(m, q, bq)
    return bq, best


# ------------------------------------------------------------------ Blockhadamard
_BD = {}


def blockmat(b):
    """Blockdiagonale (512 x 512) aus normierten Hadamard-Blöcken der Größe b (symmetrisch, selbstinvers); dichte Matmul statt FWHT ist numerisch gleich, aber schneller auf der CPU."""
    if b not in _BD:
        H = fwht(torch.eye(b))
        _BD[b] = torch.block_diag(*([H] * (N // b)))
    return _BD[b]


def bl_rot(x, sgn, b, rueck=False):
    """x -> blockdiagonale normierte Hadamard (Blockgröße b) nach Vorzeichen sgn. rueck=True: Umkehrung."""
    if rueck:
        return (x @ blockmat(b)) * sgn
    return (x * sgn) @ blockmat(b)


def qfn_bauen(Xcal, gitter, bits, block, sgn):
    """Quantisierungsfunktion einer Lage; Xcal (n x 512) fp-Aktivierungen der Kalibrierung. block = 0: roh."""
    L = 2 ** bits
    Xc = Xcal if block == 0 else bl_rot(Xcal, sgn, block)
    info = {}
    if gitter == "lloyd":
        rms = Xc.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-9)
        p = lloyd_pegel(Xc / rms, L)
    elif gitter == "mu":
        sub = Xc[:512]
        fehler = {mu: float(quant_pegel(sub, pegel_fest("mu", L, mu))[1].sum()) for mu in MUS}
        info["mu"] = min(fehler, key=fehler.get)
        p = pegel_fest("mu", L, info["mu"])
    else:
        p = pegel_fest(gitter, L)
    info["pegel"] = [float(t) for t in p]

    def f(x):
        if block == 0:
            return quant_pegel(x, p)[0]
        return bl_rot(quant_pegel(bl_rot(x, sgn, block), p)[0], sgn, block, rueck=True)
    return f, info


# ------------------------------------------------------------------ Netz
class FFNG(FFNQ):
    def __init__(self, d, f, akt="gelu"):
        super().__init__(d, f, akt)
        self.rec, self.qfn = None, None

    def forward(self, x):
        a = self.akt(self.a(x))
        if self.rec is not None:
            self.rec.append(a.detach().reshape(-1, a.shape[-1]))
        if self.qfn is not None:
            a = self.qfn(a)
        return self.b(a)


def netz_bauen(d, akt):
    n = LMX([FFNG(d, 4 * d, akt) for _ in range(LAGEN)], d=d)
    for b in n.bl:
        b.att = AttnQ(d, kb.koepfe(d))
    return n


LAGEN = 4


def kalibrieren(n, seed, schnell):
    tr = ss.daten(schnell)[0]
    g = torch.Generator().manual_seed(seed + 99)
    i = torch.randint(0, len(tr) - KONTEXT - 1, (16,), generator=g)
    x = tr[i[:, None] + torch.arange(KONTEXT)].long()
    for b in n.bl:
        b.ffn.rec = []
    n.eval()
    with torch.no_grad():
        n(x)
    cal = [torch.cat(b.ffn.rec) for b in n.bl]
    for b in n.bl:
        b.ffn.rec = None
    return cal


def arm_schaden(n, cal, gitter, bits, block, seeds_sgn, v0, schnell, infos=None):
    for l, b in enumerate(n.bl):
        b.ffn.qfn, inf = qfn_bauen(cal[l], gitter, bits, block, seeds_sgn[l])
        if infos is not None:
            infos.append(inf)
    d = val_bpc(n, schnell) - v0
    for b in n.bl:
        b.ffn.qfn = None
    return float(d)


def flops_tabelle():
    d = 128
    macs = lm_macs(d, LAGEN, 2 * d * 4 * d)
    t = {"macs_je_token_modell_d128": macs, "je_block": {}}
    for b in BLOECKE:
        add = LAGEN * N * int(math.log2(b))
        t["je_block"][b] = {"additionen_je_token": add, "aufschlag_prozent_d128": 100 * add / macs}
    t["voll_dicht_n2_je_token"] = LAGEN * N * N
    t["analytisch_nach_d"] = {}
    for dd in (512, 1024, 2048, 4096):
        f = 4 * dd
        m = LAGEN * (4 * dd * dd + dd * (KONTEXT + 1) + 2 * dd * f) + 256 * dd
        t["analytisch_nach_d"][dd] = {str(b): 100 * LAGEN * f * math.log2(b) / m for b in (8, 32, 128, 512) if b <= f} | {"voll_f": 100 * LAGEN * f * math.log2(f) / m}
    return t


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    Ts = (40, 160) if schnell else (600, 2400)
    out = {"Ts": Ts, "bits": BITS, "gitter": GITTER, "bloecke": BLOECKE, "val_bytes": VAL_N, "bpc_fp32": {"gelu": {}, "z2": {}}, "instabil": {},
           "schaden": {a: {b: {} for b in BITS} for a in ("gelu", "z2")}, "bestes_rot_gitter": {a: {b: {} for b in BITS} for a in ("gelu", "z2")},
           "mu_gewaehlt": {}, "kosten": flops_tabelle()}
    for T in Ts:
        for akt in ("gelu", "z2"):
            t = time.time()
            torch.manual_seed(seed)
            n = netz_bauen(128, akt)
            r = lm_trainiere(n, seed, T, schnell=schnell)
            n.eval()
            v0 = val_bpc(n, schnell)
            out["bpc_fp32"][akt][T], out["instabil"][f"{akt}_{T}"] = v0, r["instabil"]
            if r["instabil"]:
                print(f"Z3_gitter_block s{seed} {akt} T{T}: INSTABIL, übersprungen", flush=True)
                continue
            cal = kalibrieren(n, seed, schnell)
            sg = [(torch.randint(0, 2, (N,), generator=torch.Generator().manual_seed(seed * 7 + 31 * l)).float() * 2 - 1) for l in range(LAGEN)]
            print(f"Z3_gitter_block s{seed} {akt} T{T}: fp32 {v0:.4f} bpc, trainiert ({time.time() - t:.0f}s)", flush=True)
            for bits in BITS:
                S = out["schaden"][akt][bits]
                for g in GITTER:
                    for tr_, blk in (("roh", 0), ("had", N)):
                        infos = []
                        S.setdefault(f"{g}_{tr_}", {})[T] = arm_schaden(n, cal, g, bits, blk, sg, v0, schnell, infos)
                        if g == "mu":
                            out["mu_gewaehlt"][f"{akt}_b{bits}_T{T}_{tr_}"] = [i["mu"] for i in infos]
                rot = {g: S[f"{g}_had"][T] for g in GITTER}
                gb = min(rot, key=rot.get)
                out["bestes_rot_gitter"][akt][bits][T] = gb
                for blk in BLOECKE:
                    S.setdefault(f"blk{blk}_{gb}", {})[T] = S[f"{gb}_had"][T] if blk == N else arm_schaden(n, cal, gb, bits, blk, sg, v0, schnell)
                print("   b{} schaden  ".format(bits) + " ".join(f"{k}:{v[T]:.4f}" for k, v in S.items() if "blk" not in k) + f" | bestes rot {gb}  ({time.time() - t:.0f}s)", flush=True)
    # ---------------------------------------------------------------- Auswertung
    out["b1"], out["a1"] = {}, {}
    for akt in ("gelu", "z2"):
        Tz = [T for T in Ts if not out["instabil"].get(f"{akt}_{T}")]
        out["b1"][akt], out["a1"][akt] = {}, {}
        for bits in BITS:
            S = out["schaden"][akt][bits]
            if not Tz or not all(T in S.get(f"{g}_roh", {}) for g in GITTER for T in Tz):
                continue
            je_T = {}
            for T in Tz:
                roh = {g: S[f"{g}_roh"][T] for g in GITTER}
                rot = {g: S[f"{g}_had"][T] for g in GITTER}
                gr, gh = min(roh, key=roh.get), min(rot, key=rot.get)
                je_T[T] = {"bester_roh": gr, "schaden_roh": roh[gr], "bester_rot": gh, "schaden_rot": rot[gh], "gewinn": roh[gr] - rot[gh],
                           "roh_unif0": roh["unif0"], "roh_nozero": roh["nozero"], "gewinn_gegen_unif0": roh["unif0"] - rot[gh]}
            out["b1"][akt][bits] = je_T
            gew = min(v["gewinn"] for v in je_T.values())
            out[f"ampel_{akt}_b{bits}"] = ampel(gew, 0.005, "bpc", f"B1 {akt} {bits} Bit: min über T (Schaden bester roher Arm − bester Hadamard-Arm, volle Breite)")
            # A1
            a1 = {"je_T": {}}
            for T in Tz:
                gb = out["bestes_rot_gitter"][akt][bits][T]
                d_roh, d_voll = je_T[T]["schaden_roh"], S[f"blk{N}_{gb}"][T]
                gv = d_roh - d_voll
                zeile = {"gitter": gb, "schaden_roh_best": d_roh, "gewinn_voll": gv, "bloecke": {}}
                for blk in BLOECKE:
                    dd = S[f"blk{blk}_{gb}"][T]
                    zeile["bloecke"][blk] = {"schaden": dd, "gewinn": d_roh - dd, "anteil": ((d_roh - dd) / gv if gv > 0.005 else float("nan"))}
                ok = [blk for blk in BLOECKE if zeile["bloecke"][blk]["anteil"] == zeile["bloecke"][blk]["anteil"] and zeile["bloecke"][blk]["anteil"] >= 0.9]
                zeile["kleinstes_b_90"] = min(ok) if ok else None
                a1["je_T"][T] = zeile
            ant_min = {blk: min(a1["je_T"][T]["bloecke"][blk]["anteil"] for T in Tz) for blk in BLOECKE}
            a1["anteil_min_ueber_T"] = ant_min
            ok = [blk for blk in BLOECKE if ant_min[blk] == ant_min[blk] and ant_min[blk] >= 0.9]
            a1["kleinstes_b_90_min_ueber_T"] = min(ok) if ok else None
            out["a1"][akt][bits] = a1
    print("Ampel B1 " + "  ".join(f"{a}_b{b} {out[f'ampel_{a}_b{b}']['farbe']}({out[f'ampel_{a}_b{b}']['gewinn']:.3f})" for a in ("gelu", "z2") for b in BITS if f"ampel_{a}_b{b}" in out), flush=True)
    for a in out["a1"]:
        for b, v in out["a1"][a].items():
            print(f"A1 {a} b{b}: kleinstes Block-b >=90 %: {v['kleinstes_b_90_min_ueber_T']}  Anteile(min über T) " + " ".join(f"{k}:{x:.2f}" for k, x in v["anteil_min_ueber_T"].items()), flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z3_gitter_block", main)
