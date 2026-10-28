"""K14 Wachsendes Modell (isometrisch verbreitern) -- ERKUNDUNG, kein Befund.
Frage: Bringt "klein anfangen (d=64), dann funktionserhaltend auf d=128 verbreitern" bei gleichen Trainings-FLOPs bessere Byte-LM-bpc als durchgehend d=128?
Verbreiterung (exakt funktionserhaltend, geprüft): jeder Kanal wird verdoppelt ([x,x]); LayerNorm ändert sich dadurch nicht, Eingangsgewichte werden halbiert,
  Ausgangszeilen dupliziert, q-Zeilen mit 1/sqrt(2) (die Köpfe werden 16 -> 32 breit, Skala 1/sqrt(hd)), Ausgabe mit 0,5 (gebundene Einbettung).
  Danach kleines Gauß-Rauschen (2 % der Gewichts-Streuung), sonst blieben die beiden Hälften ewig gleich (Symmetrie). Neuer Adam, 100 Schritte Anlauf.
ARME (Tokens/Schritt 1024, Kontext 128, Kosinus über alle Schritte):
  G_iso    T/2 Schritte d=64, Verbreiterung funktionserhaltend, T/2 Schritte d=128
  G_naiv   dasselbe, aber Verbreiterung OHNE Funktionserhaltung (alte Gewichte in die Vorderteile kopiert, Rest zufällig initialisiert)   [Kontrolle]
  D128_eqF d=128 durchgehend mit denselben Gesamt-FLOPs wie G_*    [harte Kontrolle]
  D64_eqF  d=64 durchgehend mit denselben Gesamt-FLOPs
  D128_voll d=128 durchgehend, T Schritte (obere Schranke, mehr FLOPs)
Zwei Längen: T=4000 (Haupt) und T=1000 (kurz). FLOPs: flops_train_token (3x Vorwärts, Zählung wie I-045-v4/v5). Mass: Val-bpc (dev, 65536 Bytes, Kontext 128).
Aufruf: python3 erkundung/K14.py --seed 950 [--schnell]   (aus INSTITUT/)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lm import *

LR128, LR64 = 1e-3, 1.5e-3
CTX = 128
RAUSCH = 0.02


def verbreitere(alt: LM, seed: int, erhaltend=True) -> LM:
    d, H, ff = alt.d, alt.nh, alt.ff
    hd = d // H
    neu = LM(2 * d, alt.lagen, H, 2 * ff, seed=seed + 1000)
    S = {n: p for n, p in alt.named_parameters()}
    if not erhaltend:                                  # naiv: alte Gewichte in die Vorderteile, Rest bleibt zufällig
        with torch.no_grad():
            for n, p in neu.named_parameters():
                o = S[n]
                p[tuple(slice(0, s) for s in o.shape)] = o
        return neu
    m_res = torch.arange(2 * d) % d
    m_ff = torch.arange(2 * ff) % ff
    m_qkv = torch.tensor([(t * H + h) * hd + k % hd for t in range(3) for h in range(H) for k in range(2 * hd)])
    m_hd = torch.tensor([h * hd + k % hd for h in range(H) for k in range(2 * hd)])
    qs = torch.ones(3 * 2 * d)
    qs[:2 * d] = 2 ** -0.5
    with torch.no_grad():
        neu.emb.weight.copy_(alt.emb.weight[:, m_res])
        neu.pos.weight.copy_(alt.pos.weight[:, m_res])
        neu.ln.weight.copy_(alt.ln.weight[m_res]); neu.ln.bias.copy_(alt.ln.bias[m_res])
        for bn, ba in zip(neu.bl, alt.bl):
            for l in ("ln1", "ln2"):
                getattr(bn, l).weight.copy_(getattr(ba, l).weight[m_res]); getattr(bn, l).bias.copy_(getattr(ba, l).bias[m_res])
            bn.qkv.weight.copy_(ba.qkv.weight[m_qkv][:, m_res] / 2 * qs[:, None]); bn.qkv.bias.copy_(ba.qkv.bias[m_qkv] * qs)
            bn.proj.weight.copy_(ba.proj.weight[m_res][:, m_hd] / 2); bn.proj.bias.copy_(ba.proj.bias[m_res])
            bn.a.weight.copy_(ba.a.weight[m_ff][:, m_res] / 2); bn.a.bias.copy_(ba.a.bias[m_ff])
            bn.b.weight.copy_(ba.b.weight[m_res][:, m_ff] / 2); bn.b.bias.copy_(ba.b.bias[m_res])
        neu.ausg = 0.5
    return neu


def rausche(netz, seed, rel=RAUSCH):
    g = torch.Generator().manual_seed(seed + 31)
    with torch.no_grad():
        for p in netz.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn(p.shape, generator=g) * rel * p.std())


def funktionspruefung(seed=0):
    torch.manual_seed(seed)
    a = LM(64, seed=seed)
    for p in a.parameters():                           # nichttriviale Gewichte (Standard-Init hat Bias 0, LN = 1)
        p.data += torch.randn_like(p) * 0.05
    x = torch.randint(0, 256, (2, 40))
    a.eval()
    with torch.no_grad():
        ya = a(x)
        n_ = verbreitere(a, seed, True).eval()
        dif = float((n_(x) - ya).abs().max())
        rausche(n_, seed)
        dif_r = float((n_(x) - ya).abs().max())
    return dif, dif_r


def lauf(arm, T, tr, va, seed, schnell):
    f64 = flops_train_token(64, 4, CTX) * 1024
    f128 = flops_train_token(128, 4, CTX) * 1024
    n1 = T // 2
    F_G = n1 * f64 + (T - n1) * f128
    g = torch.Generator().manual_seed(seed + 5)
    t0 = time.time()
    e = {"T": T}
    if arm in ("G_iso", "G_naiv"):
        klein = LM(64, seed=seed)
        opt = torch.optim.Adam(klein.parameters(), LR64)
        r1 = lauf_schritte(klein, opt, tr, g, [CTX] * n1, LR64, [i / T for i in range(n1)])
        if r1["instabil"]:
            return {"instabil": True}
        e["val_bpc_vor_wachsen"] = bpc(klein, va)
        netz = verbreitere(klein, seed, erhaltend=(arm == "G_iso"))
        if arm == "G_iso":
            rausche(netz, seed)
        e["val_bpc_nach_wachsen"] = bpc(netz, va)
        opt = torch.optim.Adam(netz.parameters(), LR128)
        r = lauf_schritte(netz, opt, tr, g, [CTX] * (T - n1), LR128, [(n1 + i) / T for i in range(T - n1)], anlauf=100)
        e.update(schritte_klein=n1, schritte_gross=T - n1, flops=F_G)
    else:
        d, lr, fl = {"D128_eqF": (128, LR128, f128), "D64_eqF": (64, LR64, f64), "D128_voll": (128, LR128, f128)}[arm]
        S = T if arm == "D128_voll" else int(round(F_G / fl))
        netz = LM(d, seed=seed)
        opt = torch.optim.Adam(netz.parameters(), lr)
        r = lauf_schritte(netz, opt, tr, g, [CTX] * S, lr, [i / S for i in range(S)])
        e.update(schritte=S, flops=S * fl)
    e.update(instabil=r["instabil"], param=n_param(netz), param_ohne_emb=n_param(netz, True), train_verlust_bpc=r["train_verlust"],
             val_bpc=None if r["instabil"] else bpc(netz, va), sek=time.time() - t0)
    return e


def main(seed: int, schnell: bool = False) -> dict:
    tr, va, te = daten(klein=schnell)
    va = va[:8192] if schnell else va
    dif, dif_r = funktionspruefung()
    out = {"id": "K14", "seed": seed, "schnell": schnell, "lr128": LR128, "lr64": LR64, "pruefung_max_logit_diff": dif,
           "pruefung_nach_rauschen": dif_r, "laeufe": {}}
    assert dif < 1e-3, f"Verbreiterung nicht funktionserhaltend: {dif}"
    for T in ((60,) if schnell else (1000, 4000)):
        out["laeufe"][f"T{T}"] = {}
        for arm in ("G_iso", "G_naiv", "D128_eqF", "D64_eqF", "D128_voll"):
            e = lauf(arm, T, tr, va, seed, schnell)
            out["laeufe"][f"T{T}"][arm] = e
            print(T, arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
        L = out["laeufe"][f"T{T}"]
        if all(not L[a].get("instabil") for a in ("G_iso", "D128_eqF")):
            L["diff_Giso_minus_D128eqF"] = L["G_iso"]["val_bpc"] - L["D128_eqF"]["val_bpc"]
            L["diff_Giso_minus_Gnaiv"] = L["G_iso"]["val_bpc"] - L["G_naiv"]["val_bpc"]
    out["lesart"] = "diff < 0: Wachsen besser als d=128 durchgehend bei gleichen FLOPs. Ein Tuning-Seed, Lernraten fest (nicht je Arm getunt)."
    return out


if __name__ == "__main__":
    cli("K14", main, __doc__)
