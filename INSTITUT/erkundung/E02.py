"""E02 = Ergaenzung I-048  E2 (Trainings-FLOP-Gleichheit). Wichtigste Juror-Frage: Gewinnt „gross trainieren + Rotation/2 Bit“ nur, weil es mehr Trainings-FLOPs bekam?

Arme (MNIST und Fashion, MLP; Bytes = I-048-Ziel der Linearschichten bei 2 Bit (Hauptpunkt) und 3 Bit):
  gross_fp32       784-512-512-10, 30 Epochen Adam (kb-Profil), lr auf der VALIDIERUNG aus kb.LR_GITTER gewaehlt (auch die Grossen werden getunt)
  gross_roh / had / orth   nachtraeglich 2 bzw. 3 Bit (kb.quantisiere, Clip), Bytes = Ziel
  klein_ep{10,25,60}     kleines dichtes fp16-Modell (Formen flach/zwei/trichter, groesste Breite <= Ziel), Form + lr aus kb.LR_GITTER nach Val-Genauigkeit (Tuning bei 10 Epochen,
                         Feinsuche der lr (3 Werte) bei 60 Epochen und bei FLOP-Gleichheit; 10/25 benutzen die lr des Tunings)
  klein_flopgleich       Epochen so, dass Trainings-FLOPs (6 x MACs x Beispiele x Epochen) = die des grossen Modells (30 Epochen), Epochen <= 400 gedeckelt; lr-Feinsuche
Massstab: Test-Genauigkeit gegen Trainings-FLOPs UND Bytes. Trainings-FLOPs des Grossen + Rotation = 6 x MACs x N x 30 + Quantisierung (vernachlaessigt, ausgewiesen als 0).
Auswertung `vergleich`: Differenz had - bester Kleiner (Test, pp) bei gleichen Trainings-FLOPs und bei weniger FLOPs.
Nur MNIST (Zeitbudget); 3 Bit mit uebernommenem 2-Bit-Tuning, nur ep25 und FLOP-gleich. Kosten Raven (geschaetzt): ~25 min je Seed. Aufruf: python3 erkundung/E02.py --seed 950 [--schnell]"""
import numpy as np

import e_basis as eb
from e_basis import kb, torch


def trainiere_klein(ds, formd, ziel, seed, ep, lr, klein):
    torch.manual_seed(seed)
    n, d = kb.form_klein(ds, formd, ziel)
    info = kb.trainiere_bild(n, ds, seed, lr, ep, klein=klein)
    n = eb.runde_fp16(n)
    macs = kb.macs_linear(n)
    return {"form": formd, "h": d["h"], "dims": d["dims"], "lr": lr, "ep": ep, "val": eb.acc(n, ds, klein, "val"), "test": eb.acc(n, ds, klein),
            "bytes": kb.bytes_fp16_linear(n), "macs": macs, "train_flops": eb.train_flops(macs, ds, klein, ep), "instabil": info["instabil"]}


def beste(ds, formen, ziel, seed, ep, lrs, klein):
    b = None
    for fm in formen:
        for lr in lrs:
            r = trainiere_klein(ds, fm, ziel, seed, ep, lr, klein)
            if b is None or r["val"] > b["val"]:
                b = r
    return b


def main(seed, schnell=False):
    P = eb.profil(schnell)
    klein = P["klein"]
    out = {"profil": P, "ergebnisse": {}}
    for ds in P["ds"][:1]:          # nur MNIST (Zeitbudget); Fashion ueber P["ds"] bei Bedarf ergaenzen
        res = {}
        # ---- grosses Modell, lr nach Val
        lrs = kb.LR_GITTER if not schnell else (1e-3,)
        gross = None
        for lr in lrs:
            n, info = eb.trainiere_mlp(ds, seed, (512, 512), P["ep"], lr, klein)
            va = eb.acc(n, ds, klein, "val")
            eb.lg(f"[E02] {ds} gross lr {lr}: val {va:.4f}")
            if gross is None or va > gross[0]:
                gross = (va, lr, n, info)
        va, lr_g, netz, info = gross
        macs_g = kb.macs_linear(netz)
        fl_g = eb.train_flops(macs_g, ds, klein, P["ep"])
        res["gross"] = {"lr": lr_g, "val": va, "test": eb.acc(netz, ds, klein), "bytes_fp16": kb.bytes_fp16_linear(netz), "macs": macs_g, "train_flops": fl_g}
        for bits in (2, 3) if not schnell else (2,):
            ziel = kb.ziel_bytes(ds, bits)
            B = {"ziel_bytes": ziel, "gross_quant": {}, "klein": {}}
            for modus in ("roh", "had", "orth"):
                nq = kb.quantisiere(netz, bits, modus, seed, pad_to={784: 1024})
                B["gross_quant"][modus] = {"test": eb.acc(nq, ds, klein), "val": eb.acc(nq, ds, klein, "val"), "bytes": float(sum(m.bytes() for m in nq.modules() if isinstance(m, kb.QLinear))),
                                           "train_flops": fl_g, "macs": macs_g}
            # ---- kleine dichte Modelle: Tuning bei kurzem Lauf (alle Formen x lr), dann Epochen-Reihe
            if bits == 2:
                t = {}
                for fm in ("flach", "zwei", "trichter"):
                    t[fm] = [trainiere_klein(ds, fm, ziel, seed, P["ep_nt"] * 2, lr, klein) for lr in lrs]
                best_t = max((r for L in t.values() for r in L), key=lambda r: r["val"])
                fm, lr0 = best_t["form"], best_t["lr"]
                B["klein_tuning"] = {"form": fm, "lr": lr0, "alle": [(r["form"], r["lr"], r["val"]) for L in t.values() for r in L]}
            else:       # 3 Bit: Form und lr vom 2-Bit-Tuning uebernommen (Zeitbudget), nur ep25 und FLOP-gleich
                B["klein_tuning"] = {"form": fm, "lr": lr0, "uebernommen": True}
            macs_k = kb.macs_linear(kb.form_klein(ds, fm, ziel)[0])
            ep_gleich = int(min(400, max(1, round(P["ep"] * macs_g / macs_k)))) if not schnell else 8
            # Epochenreihe der Aufgabe (10/25/60), FLOP-gleich; bei 60 und FLOP-gleich lr-Feinsuche; Schnell: kleine Reihe
            reihe = ([("ep10", 10, False), ("ep25", 25, False), ("ep60", 60, True), ("flopgleich", ep_gleich, True)] if bits == 2 else [("ep25", 25, False), ("flopgleich", ep_gleich, False)]) \
                if not schnell else [("ep3", 3, False), ("flopgleich", ep_gleich, True)]
            for name, ep, fein in reihe:
                r = beste(ds, (fm,), ziel, seed, ep, lrs if fein else (lr0,), klein)
                r["name"] = name
                B["klein"][name] = r
                eb.lg(f"[E02] {ds} {bits}Bit klein {name} (ep {ep}, lr {r['lr']}, h {r['h']}): test {r['test']:.4f} FLOP {r['train_flops']:.3g} (gross {fl_g:.3g})")
            kl = B["klein"]["flopgleich"]
            B["vergleich"] = {"had_minus_klein_flopgleich_pp": 100 * (B["gross_quant"]["had"]["test"] - kl["test"]),
                              "had_minus_bester_klein_pp": 100 * (B["gross_quant"]["had"]["test"] - max(v["test"] for v in B["klein"].values())),
                              "had_minus_roh_pp": 100 * (B["gross_quant"]["had"]["test"] - B["gross_quant"]["roh"]["test"]),
                              "flop_verhaeltnis_klein_zu_gross": kl["train_flops"] / fl_g, "ep_gleich": ep_gleich}
            res[f"bits{bits}"] = B
        out["ergebnisse"][ds] = res
    return out


if __name__ == "__main__":
    eb.cli("E02", main)
