"""Gemeinsame Bausteine der Z2-Erkundungsskripte (29.10.2026). Erkundung, kein Befund.
Ergänzt _z1.py um: Quantisierer mit beliebiger Stufenzahl L (3 = ternär, 4 = 2 Bit, 8, 16 ...), Rotationen (roh / Zufalls-Orthogonal / Vorzeichen*Hadamard),
Byte-LM mit Quantisierungs-Haken (Gewichte, K/V-Cache, FFN-Mitte), Potenzgesetz-Fit, GPTQ-artige Fehlerrückführung."""
from __future__ import annotations
from _z1 import *

VAL_N = int(os.environ.get("Z2_VAL", "32768"))      # Val-Bytes für bpc (Tempo; je Skript in der Ausgabe vermerkt)
ROT_ARTEN = ("roh", "zufall", "hadamard")


# ------------------------------------------------------------------ Quantisierer mit L Stufen
def stufen_quant(x, L, nf=8):
    """Symmetrisches Gleichgitter mit L Stufen entlang der LETZTEN Achse. L gerade: Mittelpunkte ±0,5 s, ±1,5 s ...; L ungerade: 0, ±s, ... (L=3 ternär).
    Skala je Zeile per MSE-Suche (nf Kandidaten 0,3..1,0 x Maximum). L >= 65536: unverändert (fp16-Näherung)."""
    if L >= 65536:
        return x
    gerade = (L % 2 == 0)
    hi = L / 2 - 0.5 if gerade else (L - 1) / 2
    norm = L / 2 if gerade else hi
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best, bq = None, None
    for f in np.linspace(0.3, 1.0, nf):
        s = amax * float(f) / norm
        q = ((x / s - 0.5).round() + 0.5).clamp(-hi, hi) * s if gerade else (x / s).round().clamp(-hi, hi) * s
        e = ((x - q) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bq = e, q
        else:
            m = e < best
            best, bq = torch.where(m, e, best), torch.where(m, q, bq)
    return bq


def rot_q(n, art, seed):
    """Matrix Q (n x n) mit x_rot = x @ Q.T; None für 'roh'. zufall = Haar-Orthogonal; hadamard = H*diag(Vorzeichen) (wie Z1 'rot')."""
    if art == "roh":
        return None
    return kb.orth_matrix(n, seed) if art == "zufall" else kb.rot_matrix(n, seed)


def quant_rot(x, L, Q, nf=8):
    """Aktivierung/Zeile in rotierter Basis quantisieren und zurückdrehen (äquivalent zu: Rotation in das nächste Gewicht falten)."""
    if Q is None:
        return stufen_quant(x, L, nf)
    return stufen_quant(x @ Q.T, L, nf) @ Q


# ------------------------------------------------------------------ Byte-LM mit Haken
class QLin(nn.Module):
    """nn.Linear mit Gewicht in rotierter Basis, auf L Stufen gerundet (PTQ, Zeilenskala per MSE)."""

    def __init__(self, lin, L, art, seed, nf=12):
        super().__init__()
        W = lin.weight.data
        self.Q = rot_q(W.shape[1], art, seed)
        Wr = W if self.Q is None else W @ self.Q.T
        self.register_buffer("Wq", stufen_quant(Wr, L, nf))
        self.register_buffer("b", lin.bias.data.clone())

    def forward(self, x):
        if self.Q is not None:
            x = x @ self.Q.T
        return F.linear(x, self.Wq, self.b)


def gewichte_quant(netz, L, art, seed):
    n = copy.deepcopy(netz)
    k = [0]

    def rek(m):
        for name, c in list(m.named_children()):
            if isinstance(c, nn.Linear):
                k[0] += 1
                setattr(m, name, QLin(c, L, art, seed * 101 + k[0]))
            else:
                rek(c)
    rek(n)
    return n.eval()


class AttnQ(Attn):
    """Attention mit optionalen Haken. self.kv = None | dict(L, Q, kanal=False): K und V (je Kopf, Länge 64) werden vor der Aufmerksamkeit quantisiert.
    kanal=True (nur 'roh', KIVI-artig): K mit Skala je Kanal über die Tokens des Fensters (kleine Zukunftsinformation, nur als Kontrolle), V je Token."""

    def __init__(self, d, h):
        super().__init__(d, h)
        self.kv = None
        self.treppe = None          # Z2_kv_treppe setzt hier sein Verfahren
        self.stat = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        if self.treppe is not None:
            return self.proj(self.treppe(self, q, k, v).transpose(1, 2).reshape(B, T, C))
        if self.kv is not None:
            L, Q = self.kv["L"], self.kv["Q"]
            if self.kv.get("kanal"):
                k = stufen_quant(k.transpose(-1, -2), L).transpose(-1, -2)
            else:
                k = quant_rot(k, L, Q)
            v = quant_rot(v, L, Q)
        return self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))


class FFNQ(FFNDichtX):
    """FFN mit Haken `mitte` = None | dict(L, Q): Aktivierung vor der Down-Projektion (Breite f) quantisiert."""

    def __init__(self, d, f, akt="gelu"):
        super().__init__(d, f, F.gelu if akt == "gelu" else (lambda z: z * z))
        self.mitte = None

    def forward(self, x):
        a = self.akt(self.a(x))
        if self.mitte is not None:
            a = quant_rot(a, self.mitte["L"], self.mitte["Q"])
        return self.b(a)


def lm_q_bauen(d=128, akt="gelu", lagen=4):
    n = LMX([FFNQ(d, 4 * d, akt) for _ in range(lagen)], d=d)
    for b in n.bl:
        b.att = AttnQ(d, kb.koepfe(d))
    return n


def haken_setzen(netz, kv=None, mitte=None):
    for b in netz.bl:
        b.att.kv, b.att.treppe = kv, None
        b.ffn.mitte = mitte
    netz.eval()


def val_bpc(netz, schnell=False):
    va = ss.daten(schnell)[1]
    return ss.bpc_auswerten(netz, va[:VAL_N if not schnell else len(va)])


# ------------------------------------------------------------------ Auswertung
def fit_potenz(Ts, ys):
    """Schaden = a * T^b über positive Punkte (Kleinste Quadrate im Log-Log). Gibt (a, b) oder (nan, nan)."""
    p = [(t, y) for t, y in zip(Ts, ys) if y > 1e-6]
    if len(p) < 3:
        return float("nan"), float("nan")
    b, la = np.polyfit(np.log([t for t, _ in p]), np.log([y for _, y in p]), 1)
    return float(np.exp(la)), float(b)


def zeile_auswerten(Ts, ys):
    a, b = fit_potenz(Ts, ys)
    return {"schaden": {int(t): float(y) for t, y in zip(Ts, ys)}, "steigung_je_verdopplung": steigung(Ts, ys), "fit_a": a, "fit_b": b,
            "vorhersage_2xT": (a * (2 * max(Ts)) ** b if a == a else float("nan")), "T_vorhersage": 2 * max(Ts)}


# ------------------------------------------------------------------ GPTQ-artige Fehlerrückführung
def gptq(W, Hk, bits, skala, damp=0.01):
    """W (o x i) in der Basis, in der gerundet wird; Hk = X^T X / N (i x i) Eingangskovarianz in derselben Basis; skala (o x 1) feste Zeilenskala.
    Spalte für Spalte runden; der Rundungsfehler wird mit der inversen Cholesky-Zeile auf die noch offenen Spalten verteilt (OBQ/GPTQ). Gibt gerundetes W."""
    W = W.clone().double()
    H = Hk.clone().double()
    n = H.shape[0]
    H += damp * H.diagonal().mean().clamp_min(1e-9) * torch.eye(n, dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    s = skala.double()
    for j in range(n):
        w = W[:, j:j + 1]
        q = quant_skala(w, bits, s)
        Q[:, j:j + 1] = q
        err = (w - q) / U[j, j]
        if j + 1 < n:
            W[:, j + 1:] -= err @ U[j:j + 1, j + 1:]
    return Q.float()
