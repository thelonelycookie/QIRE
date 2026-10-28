"""E09 = Ergaenzung I-048  E9 (Quantisierung als Regularisierer): Hilft Nachher-Quantisierung (rotiert, 3/4 Bit) einem UEBERANGEPASSTEN Byte-LM auf neuen Daten, so wie Rauschen/Early-Stopping es taten?
Modell: Byte-LM d=128, 4 Lagen, GELU, Kontext 128, Batch 8x128. Trainiert wird nur auf den ersten 100 000 Trainingsbytes (--schnell: 20 000 Bytes, 150 Schritte) mit T = 4000 Schritten (~34 Durchlaeufe) -> klare Ueberanpassung.
Adam lr 1e-3, 5 % Aufwaermen + Kosinus, Clip 1,0 (kb._schrittlauf). Alle 250 Schritte: Val-bpc (65 536 Bytes) und Train-bpc (erste 16 384 Subset-Bytes); der Val-beste Zustand wird gehalten.
Arme (Test-, Val-, Train-bpc; nachtraegliche Quantisierung mit kb.QLinear, Skala je Zeile + Clip-Suche; Embedding/LayerNorm fp):
  fp_ende         Zustand nach T Schritten (ueberangepasst)
  fp_bestval      Val-bester Zustand (Early-Stopping-Kontrolle: „bester Val-Stand fp32“)
  had3 / had4 / roh3 / roh4   fp_ende nachtraeglich 3 bzw. 4 Bit, rotiert (Vorzeichen-Hadamard) bzw. roh
  rausch_<arm>    fp_ende + iid Gauss-Rauschen je Linearschicht mit sigma = RMS des Quantisierungsfehlers dieses Arms in dieser Schicht (gleich starkes Gewichtsrauschen; Mittel ueber 3 Ziehungen)
  had3_bestval / had4_bestval   Quantisierung des Val-besten Zustands (additiv?)
Lesart: Regulariserungseffekt, wenn had/roh auf Val/Test BESSER als fp_ende ist UND besser als gleich starkes Rauschen. Bytes (Linearschichten) ausgewiesen; Ergebnis ist Erkundung, kein Befund.
Kosten Raven (geschaetzt): ~12 min je Seed. Aufruf: python3 erkundung/E09.py --seed 950 [--schnell]"""
import numpy as np

import e_basis as eb
from e_basis import F, kb, nn, ss, torch


def main(seed, schnell=False):
    P = eb.profil(schnell)
    klein = P["klein"]
    tr, va, te = ss.daten(klein)
    sub = tr[:20000] if schnell else tr[:100000]
    T, alle = P["T_sub"], (50 if schnell else 250)
    va_ = va[:ss.VAL_BYTES]
    tr_pruef = sub[:16384]
    netz = eb.baue_lm(seed, "gelu")
    g = torch.Generator().manual_seed(seed + 5)
    offs = torch.arange(ss.KONTEXT)
    verlauf, best = [], {"val": 1e9, "zustand": None, "schritt": 0}

    def pruefe(s):
        v = ss.bpc_auswerten(netz, va_)
        t = ss.bpc_auswerten(netz, tr_pruef)
        verlauf.append({"schritt": s, "val": v, "train": t})
        if v < best["val"]:
            best.update(val=v, zustand=eb.copy.deepcopy(netz.state_dict()), schritt=s)
        netz.train()

    def schritt(s):
        if s > 0 and s % alle == 0:
            pruefe(s)
        i = torch.randint(0, len(sub) - ss.KONTEXT - 1, (ss.BATCH,), generator=g)
        x = sub[i[:, None] + offs].long()
        y = sub[i[:, None] + offs + 1].long()
        return F.cross_entropy(netz(x).reshape(-1, 256), y.reshape(-1))

    info = kb._schrittlauf(netz, T, 1e-3, schritt, None, 0, {}, g, clip=1.0)
    pruefe(T)
    eb.lg(f"[E09] trainiert ({info['sekunden']:.0f} s); bester Val-Schritt {best['schritt']} val {best['val']:.4f}; Ende val {verlauf[-1]['val']:.4f} train {verlauf[-1]['train']:.4f}")
    out = {"profil": P, "verlauf": verlauf, "bester_schritt": best["schritt"], "subset_bytes": len(sub), "arme": {}}

    def bewerte(n, name, extra=None):
        r = {"test": ss.bpc_auswerten(n, te), "val": ss.bpc_auswerten(n, va_), "train": ss.bpc_auswerten(n, tr_pruef)}
        if extra:
            r.update(extra)
        out["arme"][name] = r
        return r

    ende = eb.copy.deepcopy(netz)
    bv = eb.copy.deepcopy(netz)
    bv.load_state_dict(best["zustand"])
    bewerte(ende, "fp_ende")
    bewerte(bv, "fp_bestval")
    sigma = {}
    for bits in (3, 4):
        for modus in ("had", "roh"):
            q, by = eb.quantisiere_lm_sel(ende, bits, lambda n, m=modus: m, seed)
            name = f"{modus}{bits}"
            bewerte(q, name, {"bytes": by})
            # sigma je Schicht = RMS-Fehler der Quantisierung (in der gedrehten Basis gleiche Frobenius-Norm wie in der Originalbasis)
            sg = {}
            mods = dict(ende.named_modules())
            for n_ in eb.lm_linear_namen(ende):
                ql = kb.QLinear(mods[n_], bits, modus, 3)
                W = mods[n_].weight.data
                ref = (W @ ql.Q.T) if ql.Q is not None else W
                sg[n_] = float(((ql.W - ref) ** 2).mean().sqrt())
            sigma[name] = sg
        if bits in (3, 4):
            q, by = eb.quantisiere_lm_sel(bv, bits, lambda n: "had", seed)
            bewerte(q, f"had{bits}_bestval", {"bytes": by})
    for name, sg in sigma.items():
        res = []
        for z in range(3):
            n = eb.copy.deepcopy(ende)
            gz = torch.Generator().manual_seed(seed * 17 + z)
            for n_, m in n.named_modules():
                if isinstance(m, nn.Linear):
                    m.weight.data += torch.randn(m.weight.shape, generator=gz) * sg[n_]
            res.append((ss.bpc_auswerten(n, te), ss.bpc_auswerten(n, va_), ss.bpc_auswerten(n, tr_pruef)))
        a = np.mean(res, 0)
        out["arme"][f"rausch_{name}"] = {"test": float(a[0]), "val": float(a[1]), "train": float(a[2]), "test_sd": float(np.std([r[0] for r in res])), "ziehungen": 3}
    A = out["arme"]
    out["differenz_bpc_test"] = {k: A[k]["test"] - A["fp_ende"]["test"] for k in A if k != "fp_ende"}       # negativ = besser als ueberangepasster fp-Stand
    out["had_minus_rauschen_bpc_test"] = {n: A[n]["test"] - A[f"rausch_{n}"]["test"] for n in ("had3", "had4", "roh3", "roh4")}
    eb.lg("[E09] Test-bpc:", {k: round(v["test"], 4) for k, v in A.items()})
    return out


if __name__ == "__main__":
    eb.cli("E09", main)
