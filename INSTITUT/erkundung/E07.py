"""E07 = Ergaenzungen I-048  E7 (Kurtosis-Gesetz, selektive Rotation) + E8 (Rotation nur auf der dichten Seite, ReLU² mit Top-K-Ueberspringen). Beide brauchen dasselbe Modell -> ein Training.
Modell: Byte-LM d=128, 4 Lagen, FFN 4d mit ReLU² (z -> relu(z)^2), Kontext 128, Batch 8x128, T = 4000 Schritte (--schnell: 150), Adam lr 1e-3, Kosinus, Clip 1,0 (kb.trainiere_lm), UD_German-GSD-Bytes. Test-bpc ueber den ganzen Testsatz.
Quantisiert werden alle 16 Linearschichten der Lagen (qkv, proj, a=W1, b=W2) nachtraeglich mit kb.QLinear, 2 und 3 Bit (Skala je Zeile, Clip-Suche), Einbettung/LayerNorm fp.

E7: je Matrix mittlere Exzess-Kurtosis der Zeilen; Gewinn der Rotation einer EINZELNEN Matrix (alle anderen roh): Delta bpc (Val) und relativer Gewichtsfehler roh/had.
    Spearman(Kurtosis, Gewinn). Auswahlregel: die k in {4,8,12} Matrizen mit der hoechsten Kurtosis rotieren gegen 10 Zufallsauswahlen gleicher Groesse (Zufallskontrolle), gegen alle / keine; Test-bpc und Bytes (Vorzeichenvektor nur fuer gedrehte Schichten).
E8: Top-K-Ueberspringen im ReLU²-FFN (K = f/4 = 128 und f/8 = 64 von f = 512 Aktivierungen je Token; nur die K groessten bleiben, der Rest wird uebersprungen -> W2 braucht nur K Spalten).
    Arme (2/3 Bit): fp32 dicht / fp32 +Top-K; nirgends rotiert (roh) dicht / +Top-K; nur W1 gedreht (alles andere roh) +Top-K; alles ausser W2 gedreht +Top-K (W2 in Identitaetsbasis); ueberall gedreht dicht (W2 gedreht -> Eingang dicht, kein Ueberspringen moeglich);
    ueberall gedreht + Top-K nur als Genauigkeitswert (Ueberspringen dann nicht realisierbar: W2-MACs bleiben dicht ausgewiesen). Ausgewiesen: Bytes, W2-MACs je Token, Gesamt-MACs je Token.
Kosten Raven (geschaetzt): ~15 min je Seed. Aufruf: python3 erkundung/E07.py --seed 950 [--schnell]"""
import numpy as np

import e_basis as eb
from e_basis import kb, nn, torch


def rang(v):
    v = np.asarray(v, float)
    r = np.empty(len(v))
    r[np.argsort(v)] = np.arange(len(v))
    return r


def spearman(a, b):
    ra, rb = rang(a), rang(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def macs_je_token(netz, K=None):
    d, f = netz.d, netz.f
    lin = sum(m.weight.numel() for m in netz.modules() if isinstance(m, nn.Linear))
    w2_dicht = netz.lagen * f * d
    w2 = netz.lagen * (K if K else f) * d
    return kb.lm_macs_gesamt(d, netz.lagen, lin) - (w2_dicht - w2), w2


def main(seed, schnell=False):
    P = eb.profil(schnell)
    klein = P["klein"]
    out = {"profil": P}
    netz = eb.baue_lm(seed, "relu2")
    info = kb.trainiere_lm(netz, seed, 1e-3, P["T"], klein=klein)
    eb.lg(f"[E07] LM trainiert ({info['sekunden']:.0f} s, instabil {info['instabil']})")
    names = eb.lm_linear_namen(netz)
    mods = dict(netz.named_modules())
    f_ = netz.f
    out["fp32"] = {"test": eb.lm_bpc(netz, klein), "val": eb.lm_bpc(netz, klein, "val")}
    kurt = {n: eb.kurtosis_zeilen(mods[n].weight.data) for n in names}
    out["kurtosis"] = kurt
    # ---------------- E7
    e7 = {}
    for bits in (2, 3):
        allroh, b_roh = eb.quantisiere_lm_sel(netz, bits, lambda n: "roh", seed)
        allhad, b_had = eb.quantisiere_lm_sel(netz, bits, lambda n: "had", seed)
        base = {"roh": {"val": eb.lm_bpc(allroh, klein, "val"), "test": eb.lm_bpc(allroh, klein), "bytes": b_roh},
                "had": {"val": eb.lm_bpc(allhad, klein, "val"), "test": eb.lm_bpc(allhad, klein), "bytes": b_had}}
        einzel, relf = {}, {}
        for n in names:
            q, _ = eb.quantisiere_lm_sel(netz, bits, lambda m, n=n: "had" if m == n else "roh", seed)
            einzel[n] = base["roh"]["val"] - eb.lm_bpc(q, klein, "val")           # positiv = Rotation dieser Matrix hilft
            W = mods[n].weight.data
            qr, qh = kb.QLinear(mods[n], bits, "roh", 1), kb.QLinear(mods[n], bits, "had", 1)
            relf[n] = (float(((W - qr.W) ** 2).sum() / (W ** 2).sum()), float(((W @ qh.Q.T - qh.W) ** 2).sum() / (W ** 2).sum()))
        kl = [kurt[n] for n in names]
        g_bpc = [einzel[n] for n in names]
        g_err = [relf[n][0] - relf[n][1] for n in names]
        rng = np.random.default_rng(seed)
        sel = {}
        for k in ((4,) if schnell else (4, 8, 12)):
            top = set(sorted(names, key=lambda n: -kurt[n])[:k])
            q, by = eb.quantisiere_lm_sel(netz, bits, lambda m: "had" if m in top else "roh", seed)
            zuf = []
            for _ in range(2 if schnell else 10):
                z = set(rng.choice(names, k, replace=False).tolist())
                qz, byz = eb.quantisiere_lm_sel(netz, bits, lambda m: "had" if m in z else "roh", seed)
                zuf.append(eb.lm_bpc(qz, klein))
            sel[f"k{k}"] = {"kurtosis_top": {"test": eb.lm_bpc(q, klein), "bytes": by}, "zufall": {"mittel": float(np.mean(zuf)), "sd": float(np.std(zuf)), "werte": zuf, "bytes": byz},
                            "vorteil_gegen_zufall_bpc": float(np.mean(zuf)) - eb.lm_bpc(q, klein)}
        e7[f"bits{bits}"] = {"basis": base, "einzelgewinn_bpc_val": einzel, "relfehler_roh_had": relf,
                             "spearman_kurt_gewinn_bpc": spearman(kl, g_bpc), "spearman_kurt_gewinn_fehler": spearman(kl, g_err), "auswahl": sel}
        eb.lg(f"[E07] E7 {bits}Bit: roh {base['roh']['test']:.4f} had {base['had']['test']:.4f} rho_bpc {e7[f'bits{bits}']['spearman_kurt_gewinn_bpc']} rho_err {e7[f'bits{bits}']['spearman_kurt_gewinn_fehler']}")
    out["E7"] = e7
    # ---------------- E8
    e8 = {}
    for Kd in ((f_ // 4,) if schnell else (f_ // 4, f_ // 8)):
        arme = {}
        eb.setze_topk(netz, Kd)
        arme["fp32_topk"] = {"test": eb.lm_bpc(netz, klein), "macs": macs_je_token(netz, Kd)[0], "w2_macs": macs_je_token(netz, Kd)[1]}
        eb.setze_topk(netz, None)
        for bits in (2, 3):
            def lauf(name, wahl, topk, rot_w2):
                q, by = eb.quantisiere_lm_sel(netz, bits, wahl, seed)
                eb.setze_topk(q, topk)
                m, w2 = macs_je_token(q, topk if not rot_w2 else None)
                arme[f"{name}_b{bits}"] = {"test": eb.lm_bpc(q, klein), "bytes": by, "macs": m, "w2_macs": w2, "topk": topk}
            lauf("nirgends_dicht", lambda n: "roh", None, False)
            lauf("nirgends_topk", lambda n: "roh", Kd, False)
            lauf("nur_W1_topk", lambda n: "had" if n.endswith(".a") else "roh", Kd, False)
            lauf("ohne_W2_topk", lambda n: "roh" if n.endswith(".b") else "had", Kd, False)
            lauf("ueberall_dicht", lambda n: "had", None, True)
            lauf("ueberall_topk_nur_genauigkeit", lambda n: "had", Kd, True)
            eb.lg(f"[E07] E8 K={Kd} {bits}Bit: " + ", ".join(f"{k.rsplit('_b', 1)[0]} {v['test']:.4f}" for k, v in arme.items() if k.endswith(f"_b{bits}")))
        e8[f"K{Kd}"] = arme
    out["E8"] = e8
    return out


if __name__ == "__main__":
    eb.cli("E07", main)
