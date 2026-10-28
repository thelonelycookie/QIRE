"""E05 = Ergaenzung I-048  E5 (Breitenskalierung): Waechst oder schrumpft der Vorteil „gross + Rotation + 2/3 Bit“ gegenueber (a) roh quantisiert und (b) kleinem dichtem fp16-Modell GLEICHER Bytes mit der Breite?
MLP 784-h-h-10, h in {64,128,256,512,1024} (--schnell: {64,128}), MNIST, 30 Epochen Adam lr 1e-3 (kb-Profil; alle Breiten gleiche Einstellungen, keine Breiten-Anpassung der lr -> bewusst, damit die Skalierung nicht von Tuning abhaengt),
Bytes = I-048-Ziel (Rotation + Zeilenskala, kb.bytes_q_linear, Eingang 784->1024 aufgefuellt) bei 2 und 3 Bit, je Breite neu berechnet.
Arme je Breite und Bit: fp32 (Bezug), roh (nachtraeglich), had (Vorzeichen-Hadamard, 5 Ziehungen D; Streuband = Mittel/SD ueber die Ziehungen), klein_fp16 (groesste Breite <= Ziel; Formen flach/zwei, Wahl nach Val, 30 Epochen).
Auswertung: gewinn_had_roh_pp und gewinn_had_klein_pp je Breite, dazu Steigung (pp je Verdopplung von h, lineare Regression ueber log2 h) und die Streuung der Ziehungen als Band.
Hinweis: pro Seed ist das Streuband nur Rotationsrauschen; Seed-Streuung ergibt sich aus den Tuning-Seeds 950-959 (lauf_alle).
Kosten Raven (geschaetzt): ~20 min je Seed. Aufruf: python3 erkundung/E05.py --seed 950 [--schnell]"""
import numpy as np

import e_basis as eb
from e_basis import kb, torch


def main(seed, schnell=False):
    P = eb.profil(schnell)
    klein, ds = P["klein"], "mnist"
    breiten = (64, 128) if schnell else (64, 128, 256, 512, 1024)
    R = 2 if schnell else 5
    out = {"profil": P, "breiten": {}}
    for h in breiten:
        netz, info = eb.trainiere_mlp(ds, seed, (h, h), P["ep"], 1e-3, klein)
        B = {"fp32": {"test": eb.acc(netz, ds, klein), "val": eb.acc(netz, ds, klein, "val")}, "params_linear": kb.macs_linear(netz) + 2 * h + 10}
        for bits in (2, 3):
            ziel = eb.ziel_mlp(h, bits)
            roh = kb.quantisiere(netz, bits, "roh", seed, pad_to={784: 1024})
            hd = [kb.quantisiere(netz, bits, "had", seed * 1000 + r, pad_to={784: 1024}) for r in range(R)]
            a_h = [eb.acc(n, ds, klein) for n in hd]
            kl = eb.klein_bester(ds, ziel, seed, P["ep"], klein, formen=("flach", "zwei"), lrs=(1e-3,))
            a_r = eb.acc(roh, ds, klein)
            B[f"bits{bits}"] = {"ziel_bytes": ziel, "roh": {"test": a_r, "val": eb.acc(roh, ds, klein, "val")},
                                "had": {"test": float(np.mean(a_h)), "sd": float(np.std(a_h)), "min": min(a_h), "max": max(a_h), "werte": a_h},
                                "klein_fp16": kl, "gewinn_had_roh_pp": 100 * (float(np.mean(a_h)) - a_r),
                                "gewinn_had_klein_pp": 100 * (float(np.mean(a_h)) - kl["test"]), "sd_pp": 100 * float(np.std(a_h))}
            eb.lg(f"[E05] h={h} {bits}Bit: fp {B['fp32']['test']:.4f} roh {a_r:.4f} had {np.mean(a_h):.4f}+-{np.std(a_h):.4f} klein {kl['test']:.4f} (h_klein {kl['h']})")
        out["breiten"][h] = B
    out["steigung"] = {}
    for bits in (2, 3):
        x = np.log2(np.array(breiten, float))
        for g in ("gewinn_had_roh_pp", "gewinn_had_klein_pp"):
            y = np.array([out["breiten"][h][f"bits{bits}"][g] for h in breiten])
            out["steigung"][f"bits{bits}_{g}_je_verdopplung"] = float(np.polyfit(x, y, 1)[0])
    return out


if __name__ == "__main__":
    eb.cli("E05", main)
