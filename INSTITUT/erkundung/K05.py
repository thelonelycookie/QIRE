"""K05 (Erkundung, kein Befund): Cache-Bytes. Schluessel/Werte (KV) werden erst bei der AUSWERTUNG des normal trainierten Byte-LM (d=128, 4 Lagen, 2 Koepfe, Kopfdim 64)
quantisiert bzw. verkuerzt. Mass: Test-bpc gegen Cache-Bytes (bits_je_wert = mittlere Cache-Bits je gespeichertem Wert ueber alle Positionen, inkl. Skalen, fp16 = 16).
Arme (je Ziel 1,25 / 2,25 / 3,25 Bit je Wert):
  had     Zufalls-Vorzeichen-Hadamard (64) je Kopf, dann Rundung je Token+Kopf (MSE-Skala, 2 B je Vektor); b=1: Vorzeichenskizze (sign · mittl. Betrag)   <- Hauptarm
  roh     Rundung ohne Rotation, Skala je Token+Kopf
  roh_kanal  wie roh, aber vorher durch feste Kanalstreuung (Eichdaten, einmalig) geteilt (Kontrolle: Kanalskala)
  pca     Projektion auf die m groessten PCA-Richtungen (Eichdaten), m Koeffizienten fp16, m = 4·Ziel     (Kontrolle: PCA-Skizze)
  zufallskanal  m zufaellige Kanaele fp16, Rest = Eichmittel (Kontrolle: Zufallskanaele)
  fenster Gleitfenster: nur die letzten W Token fp16 (W so, dass Cache-Bits gleich)
  zufallfall  Token werden zufaellig (Wahrscheinlichkeit p) fp16 behalten, Rest fallengelassen (aktuelles Token immer da)
  alter_bits   Alterung: Alter < R: 4 Bit rotiert, R..2R: 2 Bit, >= 2R: 1 Bit (nur wo Ziel > 1,25)
  alter_haar   Haar-Alterung: junge Token 4 Bit rotiert, alte Token nur als Haar-Grobwert (Mittel je 4er-Block, 4 Bit rotiert, g=4)
Alle Arme: gleiche Gewichte, gleiche Eichdaten, gleicher Test. Parameter und Attention-FLOPs unveraendert (Rotation 64·6 Additionen je Vektor, ausgewiesen).
Training: kb.LM normal, Adam, Kosinus, Clip 1 (kb.trainiere_lm), lr 2e-3, T=4000 (schnell: klein, 250).
Aufruf: python3 erkundung/K05.py --seed 950 [--schnell]"""
import math
import time

import _hilfe_k0508 as H
import torch
import torch.nn.functional as F

kb, ss = H.kb, H.ss
LR, T_VOLL, T_SCHNELL = 2e-3, 4000, 250
ZIELE = (1.25, 2.25, 3.25)           # Bit je Wert inkl. Skala (b + 16/64)
KONTROLLEN = ("roh", "roh_kanal", "pca", "zufallskanal", "fenster", "zufallfall")


# ------------------------------------------------------------------ Vorwaertsrechnung mit eigener Attention
def lm_fwd(netz, idx, att):
    B, T = idx.shape
    x = netz.emb(idx) + netz.pos(torch.arange(T))
    for l, b in enumerate(netz.bl):
        q, k, v = b.qkv(b.ln1(x)).reshape(B, T, 3, b.h, x.shape[-1] // b.h).permute(2, 0, 3, 1, 4)
        a = att(l, q, k, v).transpose(1, 2).reshape(B, T, -1)
        x = x + b.proj(a)
        x = x + b.b(F.gelu(b.a(b.ln2(x))))
    return netz.ln(x) @ netz.emb.weight.T


def att_klassen(q, klassen):
    """klassen = [(K, V, Maske[T,T])], Masken disjunkt; Softmax ueber die Vereinigung."""
    dh = q.shape[-1]
    tot = None
    for K, V, m in klassen:
        sc = q @ K.transpose(-1, -2) / math.sqrt(dh)
        tot = torch.where(m, sc, torch.full((), float("-inf"))) if tot is None else torch.where(m, sc, tot)
    P = torch.softmax(tot, -1)
    out = 0
    for K, V, m in klassen:
        out = out + torch.where(m, P, torch.zeros(())) @ V
    return out


def quant_vek(x, bits, Q=None, kanal=None):
    sh = x.shape
    y = x
    if kanal is not None:
        y = y / kanal
    if Q is not None:
        y = y @ Q.T
    y2 = y.reshape(-1, sh[-1])
    if bits == 1:
        yq = torch.sign(y2) * y2.abs().mean(1, keepdim=True)
    else:
        yq = kb.quant_rows(y2, bits)
    yq = yq.reshape(sh)
    if Q is not None:
        yq = yq @ Q
    if kanal is not None:
        yq = yq * kanal
    return yq


# ------------------------------------------------------------------ Arme (Masken + Kosten sind datenunabhaengig)
class Arm:
    def __init__(self, name, fn, kosten, info=None):
        self.name, self.fn, self.kosten, self.info = name, fn, kosten, info or {}


def bits_je_wert(kosten, T):
    """Mittlere Cache-Bits je Wert ueber alle Positionen / (Bits eines vollen fp16-Caches)."""
    return float(sum(float(m.sum()) * b for m, b in kosten) / (T * (T + 1) / 2))


def masken(T):
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return i, j, (j <= i)


class Eich:
    """Statistiken aus Trainings-Fenstern: Mittel, Streuung, PCA-Basis je (Lage, k/v, Kopf)."""

    def __init__(self, netz, n_fenster, seed, klein):
        tr, _, _ = ss.daten(klein)
        g = torch.Generator().manual_seed(seed + 123)
        i = torch.randint(0, len(tr) - ss.KONTEXT - 1, (n_fenster,), generator=g)
        x = tr[i[:, None] + torch.arange(ss.KONTEXT)].long()
        self.k, self.v = {}, {}

        def att(l, q, k, v):
            self.k[l] = k.permute(0, 2, 1, 3).reshape(-1, k.shape[1], k.shape[3]).clone()
            self.v[l] = v.permute(0, 2, 1, 3).reshape(-1, v.shape[1], v.shape[3]).clone()
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        with torch.no_grad():
            lm_fwd(netz, x, att)
        self.cache = {}

    def roh(self, l, kv):
        return (self.k if kv == 0 else self.v)[l]            # (N, H, dh)

    def mittel(self, l, kv):
        return self.roh(l, kv).mean(0)[None, :, None, :]      # (1,H,1,dh)

    def streu(self, l, kv):
        return self.roh(l, kv).std(0).clamp_min(1e-6)[None, :, None, :]

    def pca(self, l, kv, m):
        key = (l, kv)
        if key not in self.cache:
            X = self.roh(l, kv)
            P = []
            for h in range(X.shape[1]):
                Xc = X[:, h] - X[:, h].mean(0)
                w, V = torch.linalg.eigh(Xc.T @ Xc / len(Xc))
                P.append(V[:, torch.argsort(w, descending=True)])
            self.cache[key] = torch.stack(P)                  # (H, dh, dh) Spalten absteigend
        return self.cache[key][:, :, :m]


def arm_fp(T):
    _, _, c = masken(T)
    return Arm("fp", lambda l, k, v: [(k, v, c)], [(c, 16.0)])


def arm_quant(name, T, b, rot, kanal, eich, seed, dh=64):
    _, _, c = masken(T)

    def fn(l, k, v):
        out = []
        for kv, x in enumerate((k, v)):
            Q = kb.rot_matrix(dh, seed * 977 + l * 2 + kv) if rot else None
            out.append(quant_vek(x, b, Q, eich.streu(l, kv) if kanal else None))
        return [(out[0], out[1], c)]

    return Arm(name, fn, [(c, b + 16.0 / dh)])


def arm_pca(T, m, eich, dh=64):
    _, _, c = masken(T)

    def proj(l, kv, x):
        P = eich.pca(l, kv, m)
        mu = eich.mittel(l, kv)
        co = torch.einsum("bhtd,hdm->bhtm", x - mu, P).half().float()
        return mu + torch.einsum("bhtm,hdm->bhtd", co, P)

    return Arm("pca", lambda l, k, v: [(proj(l, 0, k), proj(l, 1, v), c)], [(c, 16.0 * m / dh)], {"m": m})


def arm_zufallskanal(T, m, eich, seed, dh=64):
    _, _, c = masken(T)

    def kan(l, kv, x):
        g = torch.Generator().manual_seed(seed * 313 + l * 2 + kv)
        mk = torch.zeros(x.shape[1], dh)
        for h in range(x.shape[1]):
            mk[h, torch.randperm(dh, generator=g)[:m]] = 1.0
        mu = eich.mittel(l, kv)
        return mu + (x - mu).half().float() * mk[None, :, None, :]

    return Arm("zufallskanal", lambda l, k, v: [(kan(l, 0, k), kan(l, 1, v), c)], [(c, 16.0 * m / dh)], {"m": m})


def arm_fenster(T, W):
    i, j, c = masken(T)
    mk = c & ((i - j) < W)
    return Arm("fenster", lambda l, k, v: [(k, v, mk)], [(mk, 16.0)], {"W": W})


def arm_zufallfall(T, p, seed):
    i, j, c = masken(T)
    keep = torch.rand(T, generator=torch.Generator().manual_seed(seed * 17 + 3)) < p      # feste Zufallszahlen u, Schwelle p: monoton in p
    mk = (c & keep[None, :]) | (i == j)
    return Arm("zufallfall", lambda l, k, v: [(k, v, mk)], [(mk, 16.0)], {"p": p})


def _alter_bits_masken(T, R):
    i, j, c = masken(T)
    a = i - j
    return [(c & (a < R), 4), (c & (a >= R) & (a < 2 * R), 2), (c & (a >= 2 * R), 1)]


def arm_alter_bits(T, R, seed, dh=64):
    ms = _alter_bits_masken(T, R)

    def fn(l, k, v):
        Qk, Qv = kb.rot_matrix(dh, seed * 977 + l * 2), kb.rot_matrix(dh, seed * 977 + l * 2 + 1)
        return [(quant_vek(k, b, Qk), quant_vek(v, b, Qv), m) for m, b in ms if bool(m.any())]

    return Arm("alter_bits", fn, [(m, b + 16.0 / dh) for m, b in ms], {"R": R})


def _haar_masken(T, A, g):
    i, j, c = masken(T)
    ende = (j // g) * g + g - 1
    alt = c & ((i - ende) >= A)
    return c & ~alt, alt


def arm_alter_haar(T, A, seed, g=4, dh=64):
    jung, alt = _haar_masken(T, A, g)

    def pool(x):
        B, Hh, Tt, d = x.shape
        return x.reshape(B, Hh, Tt // g, g, d).mean(3)

    def fn(l, k, v):
        Qk, Qv = kb.rot_matrix(dh, seed * 977 + l * 2), kb.rot_matrix(dh, seed * 977 + l * 2 + 1)
        kl = [(quant_vek(k, 4, Qk), quant_vek(v, 4, Qv), jung)]
        if bool(alt.any()):
            kl.append((quant_vek(pool(k), 4, Qk).repeat_interleave(g, 2), quant_vek(pool(v), 4, Qv).repeat_interleave(g, 2), alt))
        return kl

    return Arm("alter_haar", fn, [(jung, 4 + 16.0 / dh), (alt, (4 + 16.0 / dh) / g)], {"A": A, "g": g})


def passe(T, ziel, kosten_fn, grid):
    """Parameter aus grid, dessen Bits je Wert am naechsten an ziel liegen."""
    return min(grid, key=lambda p: abs(bits_je_wert(kosten_fn(p), T) - ziel))


# ------------------------------------------------------------------ Auswertung
def bewerte(netz, text, arm, bs=16):
    ctx = ss.KONTEXT
    n = (len(text) - 1) // ctx
    x = text[:n * ctx].long().reshape(n, ctx)
    y = text[1:n * ctx + 1].long().reshape(n, ctx)
    netz.eval()
    s = 0.0

    def att(l, q, k, v):
        return att_klassen(q, arm.fn(l, k, v))

    with torch.no_grad():
        for i in range(0, n, bs):
            s += float(F.cross_entropy(lm_fwd(netz, x[i:i + bs], att).reshape(-1, 256), y[i:i + bs].reshape(-1), reduction="sum"))
    return s / (n * ctx) / math.log(2)


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    klein = schnell
    T = T_SCHNELL if schnell else T_VOLL
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    netz = kb.LM(128, 4)
    inf = kb.trainiere_lm(netz, seed, LR, T, klein=klein)
    t_train = time.time() - t0
    _, _, te = ss.daten(klein)
    ctx = ss.KONTEXT
    eich = Eich(netz, 16 if schnell else 64, seed, klein)
    bpc_ref = ss.bpc_auswerten(netz, te)
    arme = {"fp": arm_fp(ctx)}
    for z in ZIELE:
        b = int(z)
        arme[f"had@{z}"] = arm_quant("had", ctx, b, True, False, eich, seed)
        arme[f"roh@{z}"] = arm_quant("roh", ctx, b, False, False, eich, seed)
        arme[f"roh_kanal@{z}"] = arm_quant("roh_kanal", ctx, b, False, True, eich, seed)
        m = int(round(z * 4))
        arme[f"pca@{z}"] = arm_pca(ctx, m, eich)
        arme[f"zufallskanal@{z}"] = arm_zufallskanal(ctx, m, eich, seed)
        W = passe(ctx, z, lambda w: arm_fenster(ctx, w).kosten, range(1, ctx + 1))
        arme[f"fenster@{z}"] = arm_fenster(ctx, W)
        p = passe(ctx, z, lambda q_: arm_zufallfall(ctx, q_, seed).kosten, [i / 400 for i in range(401)])
        arme[f"zufallfall@{z}"] = arm_zufallfall(ctx, p, seed)
        A = passe(ctx, z, lambda a: [(m_, b_) for m_, b_ in zip(_haar_masken(ctx, a, 4), (4.25, 4.25 / 4))], range(0, ctx + 1))
        arme[f"alter_haar@{z}"] = arm_alter_haar(ctx, A, seed)
        if z > 1.25:
            R = passe(ctx, z, lambda r: [(m_, b_ + .25) for m_, b_ in _alter_bits_masken(ctx, r)], range(1, ctx))
            arme[f"alter_bits@{z}"] = arm_alter_bits(ctx, R, seed)
    zeilen = {}
    for name, arm in arme.items():
        zeilen[name] = {"bpc": bewerte(netz, te, arm), "bits_je_wert": bits_je_wert(arm.kosten, ctx), "par": arm.info}
    pruef = abs(zeilen["fp"]["bpc"] - bpc_ref)
    fp = zeilen["fp"]["bpc"]
    fazit = {}
    for z in ZIELE:
        had = zeilen[f"had@{z}"]["bpc"]
        kont = {k: zeilen[f"{k}@{z}"]["bpc"] for k in KONTROLLEN}
        best = min(kont, key=kont.get)
        fazit[str(z)] = {"had_minus_fp": had - fp, "beste_kontrolle": best, "beste_kontrolle_minus_fp": kont[best] - fp,
                         "had_minus_beste_kontrolle": had - kont[best], "alter_haar_minus_fp": zeilen[f"alter_haar@{z}"]["bpc"] - fp}
    d_par = H.n_param(netz)
    r = {"kid": "K05", "seed": seed, "schnell": schnell, "T": T, "lr": LR, "klein": klein,
         "modell": {"d": 128, "lagen": 4, "kopfdim": 64, "parameter": d_par, "kv_fp16_bytes_je_token_lage": 2 * 128 * 2},
         "flops_hinweis": "Attention- und Linear-FLOPs unveraendert; Rotation 64·log2(64)=384 Additionen je KV-Vektor; Cache-Bytes siehe bits_je_wert (16 = fp16)",
         "bpc_ref_eigene_auswertung": bpc_ref, "pruefung_fp_gleich": pruef, "train_info": inf, "arme": zeilen, "fazit": fazit,
         "sekunden_training": t_train, "sekunden_auswertung": time.time() - t0 - t_train}
    r["zusammenfassung"] = "\n".join(f"Ziel {z} Bit: had {fazit[str(z)]['had_minus_fp']:+.3f} bpc, beste Kontrolle ({fazit[str(z)]['beste_kontrolle']}) {fazit[str(z)]['beste_kontrolle_minus_fp']:+.3f}" for z in ZIELE)
    return r


if __name__ == "__main__":
    H.cli("K05", main)
