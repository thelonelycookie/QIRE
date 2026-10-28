"""K18 -- 2-Bit-Zwilling als Entwurf + Bandbreiten-Messung (Erkundung Welle 2, KEIN Befund).

MECHANISMUS: Die auf 2 Bit (ternär + Zeilenskala, Hadamard-Rotation) quantisierte Kopie des trainierten Byte-LM („Zwilling“) rät Token billig vor,
  das große Modell prüft sie in einem Durchgang (spekulatives Dekodieren). Spart es Zeit/Joule, hängt davon ab, ob Bytes zu Zeit werden (Bandbreite).
TEIL 1 (Akzeptanz): Byte-LM d=128, 4 Lagen (kompression_bytes.LM), T = 4000 Schritte (schnell: 40). Ziel = fp32-Modell. Entwürfe:
  zwilling_had  2 Bit + Hadamard-Rotation (Bytes: QLinear.bytes)
  zwilling_roh  2 Bit roh ohne Rotation                    [KONTROLLE]
  klein_dicht   kleines dichtes LM (d=64) mit GLEICHEN Linear-Bytes wie der Zwilling, gleiche Schritte, Lernrate aus {2e-3,4e-3} auf Validierung [KONTROLLE]
  unigramm      häufigstes Byte immer                      [UNTERGRENZE / Leerlauf]
  Maße: Greedy-Übereinstimmung (teacher forcing, ganzer Test), Stichproben-Akzeptanz sum min(p,q), spekulative Simulation gamma = 4
  (128 Startpunkte à 64 Byte Kontext, Entwurf greedy autoregressiv, Ziel prüft in einem Durchgang): mittlere akzeptierte Länge, Token je Ziel-Durchgang.
TEIL 2 (Zeit, nur INDIKATIV auf Raven, Rechenknoten ≠ Maltes Ryzen): µs je Matvec (Batch 1, 1 Thread) n = 512..8192 für fp32 / bf16 / int8 (torch dynamic) /
  2 Bit gepackt (LUT-Entpacken in Blöcken) roh und mit Walsh-Vorrotation (float und ganzzahlig); Kontrolle Leerlauf: fp32 zweimal gemessen (Rauschverhältnis).
  Dazu µs/Token des ganzen LM (Batch 8x128, fp32) für Ziel und kleines dichtes Modell; modellierter Speed-up = (1 + akzeptiert) / (1 + gamma·c), c = Kostenverhältnis.
  Hinweis: Das simulierte 2-Bit-Modell (QLinear) rechnet in fp32 und ist NICHT schneller; echte Zeitaussagen nur aus den gepackten Kerneln.
Nur Tuning-Seeds. Quanten-inspiriert, klassisch (I-038).
"""
from __future__ import annotations
import math, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hilfe_k17_20 as H
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import abteilungen.struktur_sprache as ss
import abteilungen.kompression_bytes as kb
from abteilungen.struktur_gross import _stoppuhr

KID = "K18"
GAMMA = 4


class Konstant(nn.Module):
    """Entwurf „immer das häufigste Byte“."""
    def __init__(self, byte):
        super().__init__()
        self.byte = int(byte)

    def forward(self, idx):
        lg = torch.full((*idx.shape, 256), -10.0)
        lg[..., self.byte] = 10.0
        return lg


def train_lm(d, f, seed, T, lr, klein):
    torch.manual_seed(seed)
    netz = kb.LM(d, 4, f)
    st = kb.trainiere_lm(netz, seed, lr, T, klein=klein)
    if st.get("instabil"):
        raise RuntimeError(f"Training instabil d={d} lr={lr}")
    return netz, st


def uebereinstimmung(ziel, entwurf, text):
    n = (len(text) - 1) // ss.KONTEXT
    x = text[:n * ss.KONTEXT].long().reshape(n, ss.KONTEXT)
    gm, sm, c = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, n, 8):
            a = ziel(x[i:i + 8]); b = entwurf(x[i:i + 8])
            gm += float((a.argmax(-1) == b.argmax(-1)).float().sum())
            sm += float(torch.minimum(a.softmax(-1), b.softmax(-1)).sum(-1).sum())
            c += a.shape[0] * a.shape[1]
    return gm / c, sm / c


def spekulativ(ziel, entwurf, text, starts=128, L0=64, gamma=GAMMA, seed=0):
    g = torch.Generator().manual_seed(seed + 99)
    st = torch.randint(0, len(text) - L0 - gamma - 2, (starts,), generator=g)
    ctx = torch.stack([text[s:s + L0] for s in st]).long()
    with torch.no_grad():
        x = ctx.clone()
        for _ in range(gamma):
            x = torch.cat([x, entwurf(x)[:, -1].argmax(-1, keepdim=True)], 1)
        draft = x[:, L0:]                                   # (S, gamma)
        tg = ziel(x)[:, L0 - 1:L0 - 1 + gamma].argmax(-1)    # was das Ziel an denselben Stellen wählt
    ok = (tg == draft).long()
    acc = ok.cumprod(1).sum(1).float()                      # führende Treffer
    return {"akzeptiert_mittel": float(acc.mean()), "token_je_ziel_durchgang": float(acc.mean() + 1),
            "anteil_alle_gamma_richtig": float((acc == gamma).float().mean())}


# ---------------------------------------------------------------- Zeitmessung
def pack_ternaer(Wq, scale):
    """Wq in {-1,0,1} (o,i) -> uint8 (o,i/4)."""
    c = (Wq.round().long() + 1).clamp(0, 2).reshape(Wq.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).to(torch.uint8)


LUT = torch.tensor([[(b >> (2 * j)) & 3 for j in range(4)] for b in range(256)], dtype=torch.float32) - 1.0


def matvec_2bit(P, scale, x, R):
    n = P.shape[0]
    out = torch.empty(n)
    for r0 in range(0, n, R):
        blk = LUT[P[r0:r0 + R].long()].reshape(min(R, n - r0), -1)
        out[r0:r0 + R] = blk @ x
    return out * scale


def faktoren(n):
    a = 1 << (int(math.log2(n)) // 2)
    return a, n // a


def zeit_matvec(n, schnell):
    torch.manual_seed(n)
    W = torch.randn(n, n) / math.sqrt(n)
    x = torch.randn(n)
    ref = W @ x
    r = {"n": n, "mb_fp32": n * n * 4 / 1e6}
    mw = 0.02 if schnell else 0.05
    wdh = 3 if schnell else 7
    u = lambda f: _stoppuhr(f, mindest=mw, wdh=wdh) * 1e6
    r["fp32_us"] = u(lambda: W @ x)
    r["fp32_us_wiederholung"] = u(lambda: W @ x)             # Leerlauf-/Rauschkontrolle
    Wb, xb = W.bfloat16(), x.bfloat16()
    try:
        r["bf16_us"] = u(lambda: Wb @ xb)
    except Exception as e:
        r["bf16_us"] = None
    try:
        q = torch.ao.quantization.quantize_dynamic(nn.Sequential(nn.Linear(n, n, bias=False)), {nn.Linear}, dtype=torch.qint8)
        x1 = x[None]
        with torch.no_grad():
            r["int8_us"] = u(lambda: q(x1))
    except Exception as e:
        r["int8_us"] = None
        r["int8_fehler"] = str(e)[:100]
    # 2 Bit: ternär + Zeilenskala (hier ungerotierte Zufallsgewichte, nur Zeit/Bytes zählen)
    amax = W.abs().amax(1, keepdim=True)
    scale = amax[:, 0] * 0.6
    Wq = (W / scale[:, None]).round().clamp(-1, 1)
    P = pack_ternaer(Wq, scale)
    R = max(8, 262144 // n)
    y2 = matvec_2bit(P, scale, x, R)
    r["fehler_2bit_gegen_dequant"] = float((y2 - (Wq * scale[:, None]) @ x).abs().max())
    r["mb_2bit"] = P.numel() / 1e6
    r["zwei_bit_roh_us"] = u(lambda: matvec_2bit(P, scale, x, R))
    a, b = faktoren(n)
    Ha, Hb = kb.hadamard(a), kb.hadamard(b)
    walsh = lambda v: (Ha @ v.reshape(a, b) @ Hb).reshape(-1)
    r["walsh_float_us"] = u(lambda: walsh(x))
    r["zwei_bit_walsh_us"] = u(lambda: matvec_2bit(P, scale, walsh(x), R))
    # ganzzahlige Walsh (Additionen/Subtraktionen auf int32, Eingang als int8 quantisiert)
    Hai, Hbi = (torch.sign(Ha)).to(torch.int32), (torch.sign(Hb)).to(torch.int32)
    xi = (x / x.abs().max() * 127).round().to(torch.int32)
    try:
        r["walsh_int_us"] = u(lambda: (Hai @ xi.reshape(a, b) @ Hbi).reshape(-1))
    except Exception as e:
        r["walsh_int_us"] = None
    r["verhaeltnis_2bit_walsh_zu_fp32"] = r["zwei_bit_walsh_us"] / r["fp32_us"]
    r["verhaeltnis_2bit_roh_zu_fp32"] = r["zwei_bit_roh_us"] / r["fp32_us"]
    r["verhaeltnis_int8_zu_fp32"] = (r["int8_us"] / r["fp32_us"]) if r.get("int8_us") else None
    r["rauschverhaeltnis_leerlauf"] = r["fp32_us_wiederholung"] / r["fp32_us"]
    r["gb_je_s_fp32"] = n * n * 4 / (r["fp32_us"] * 1e-6) / 1e9
    return r


def us_je_token(netz):
    netz.eval()
    x = torch.randint(0, 256, (8, ss.KONTEXT))
    with torch.no_grad():
        return _stoppuhr(lambda: netz(x), mindest=0.1, wdh=5) / (8 * ss.KONTEXT) * 1e6


def main(seed: int, schnell: bool = False) -> dict:
    H.pruefe_seed(seed)
    torch.set_num_threads(1)
    t0 = time.time()
    T = 40 if schnell else 4000
    r = {"kid": KID, "seed": seed, "schnell": schnell, "T": T, "gamma": GAMMA}
    _, va, te = ss.daten(schnell)
    # --- Ziel
    ziel, st = train_lm(128, 512, seed, T, 2e-3, schnell)
    ziel.eval()
    r["ziel"] = {"bpc_test": kb.bewerte_lm(ziel, schnell, "test"), "parameter": int(sum(p.numel() for p in ziel.parameters())),
                 "bytes_linear_fp16": kb.bytes_fp16_linear(ziel), "bytes_rest_fp16": kb.bytes_rest_fp16(ziel), "sekunden_training": st["sekunden"]}
    # --- Zwillinge
    tw = {"zwilling_had": kb.quantisiere(ziel, 2, "had", seed), "zwilling_roh": kb.quantisiere(ziel, 2, "roh", seed)}
    for k, m in tw.items():
        m.eval()
    twin_bytes = {k: float(sum(q.bytes() for q in m.modules() if isinstance(q, kb.QLinear))) for k, m in tw.items()}
    # --- kleines dichtes Modell gleicher Bytes (Linear-Bytes <= Zwilling-Bytes), Lernrate auf Validierung
    d_kl = 64
    f_kl = kb.lm_f_fit(d_kl, 4, twin_bytes["zwilling_had"])
    kl, bestv = None, None
    for lr in (2e-3, 4e-3):
        m, s2 = train_lm(d_kl, f_kl, seed, T, lr, schnell)
        v = kb.bewerte_lm(m, schnell, "val")
        if bestv is None or v < bestv:
            kl, bestv, kl_lr, kl_st = m, v, lr, s2
    kl.eval()
    uni = Konstant(int(torch.bincount(ss.daten(schnell)[0].long(), minlength=256).argmax()))
    entw = {"zwilling_had": tw["zwilling_had"], "zwilling_roh": tw["zwilling_roh"], "klein_dicht": kl, "unigramm": uni}
    r["entwuerfe"] = {}
    for k, m in entw.items():
        gm, sm = uebereinstimmung(ziel, m, te)
        z = {"greedy_uebereinstimmung": gm, "stichprobe_akzeptanz": sm, "spekulativ": spekulativ(ziel, m, te, starts=32 if schnell else 128, seed=seed)}
        z["bpc_test"] = kb.bewerte_lm(m, schnell, "test") if k != "unigramm" else None
        if k.startswith("zwilling"):
            z["bytes_linear"] = twin_bytes[k]
        elif k == "klein_dicht":
            z.update({"bytes_linear": kb.bytes_fp16_linear(kl), "d": d_kl, "f": f_kl, "lr": kl_lr, "parameter": int(sum(p.numel() for p in kl.parameters())),
                      "bytes_rest_fp16": kb.bytes_rest_fp16(kl)})
        r["entwuerfe"][k] = z
    # --- Zeit je Token (ganzes Modell, fp32)
    r["us_je_token"] = {"ziel_fp32": us_je_token(ziel), "klein_dicht_fp32": us_je_token(kl),
                        "zwilling_simuliert_fp32": us_je_token(tw["zwilling_had"]),
                        "hinweis": "Simulation rechnet in fp32; nur Ziel und kleines dichtes Modell sind echte Zeiten."}
    # --- Matvec
    ns = (512, 2048) if schnell else (512, 1024, 2048, 4096, 8192)
    r["matvec"] = [zeit_matvec(n, schnell) for n in ns]
    # --- modellierter Speed-up
    byte_ratio = twin_bytes["zwilling_had"] / r["ziel"]["bytes_linear_fp16"]
    c_opts = {"bytes_2bit_zu_fp16": byte_ratio, "klein_dicht_zeit": r["us_je_token"]["klein_dicht_fp32"] / r["us_je_token"]["ziel_fp32"]}
    for mv in r["matvec"]:
        c_opts[f"gepackt2bit_walsh_n{mv['n']}"] = mv["verhaeltnis_2bit_walsh_zu_fp32"]
    r["speedup_modell"] = {}
    for dk, dv in r["entwuerfe"].items():
        if dk == "unigramm":
            continue
        tok = dv["spekulativ"]["token_je_ziel_durchgang"]
        r["speedup_modell"][dk] = {ck: tok / (1 + GAMMA * cv) for ck, cv in c_opts.items()}
    r["c_kostenverhaeltnis"] = c_opts
    r["hinweis"] = ("Erkundung, kein Befund. Zeit nur indikativ (Raven-Rechenknoten, 1 Thread, kein SIMD-Kernel für 2 Bit; LUT-Entpacken in torch). "
                    "Speed-up-Modell: (1+akzeptiert)/(1+gamma·c); c bei verbrauchsgebundenem Dekodieren ~ Byteverhältnis, hier nur Messwerte.")
    r["sekunden_gesamt"] = time.time() - t0
    H.speichere(KID, seed, r, schnell)
    return r


if __name__ == "__main__":
    H.cli(KID, main)
