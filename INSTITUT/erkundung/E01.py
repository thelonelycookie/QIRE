"""E01 = Ergaenzungen I-048  E1 (Gruppenskalen ohne Rotation), E4 (Block-Hadamard), E10 (Entropiecodierung nach Rotation), E11 (subtraktiver Dither).
Alle vier sind NACHHER-Quantisierungen desselben trainierten MLP 784-512-512-10 (kb-Profil: 30 Epochen Adam, lr 1e-3, 55 000 Bilder) -> ein Training je (Datensatz, Seed), viele Arme.
Erkundung, kein Befund (Tuning-Seed 950-959).

FRAGE E1: Schlaegt Rotation + Zeilenskala (I-048-Arm) die Skala je g in {16,32,64,128} Gewichten OHNE Rotation, wenn die Skalen-Bytes bezahlt werden?
  Exakte Bytes: Budget je Schicht = kb.bytes_q_linear(.., Bits, rot=True, pad) (I-048-Ziel). Jeder Arm bekommt gemischte Zeilen-Bits (b und b+1) bis zu dieser Grenze (Luecke < 1 Zeile je Schicht),
  Skalen (je Zeile bzw. je Gruppe, 2 B), Bias und ggf. Vorzeichenvektor sind bezahlt. Arme, deren Skalen allein das Budget sprengen (g=8: 2 Bit/Gewicht) sind als `im_budget=false` markiert.
  Kontrolle: Rotation + Gruppenskala (gleiche g), Rotation + Zeilenskala, kleines dichtes fp16-Modell gleicher Bytes (Formen flach/zwei, lr 1e-3, gleiche 30 Epochen).
FRAGE E4: Blockrotation (Block k in {8,32,128}, voll=n) mit Skala je Block (g=k) bzw. Zeilenskala; Kontrolle: Gruppenskala k OHNE Rotation. Vorzeichenvektor bleibt n/8 B.
FRAGE E10: Gitter mit Huffman-Codierung (Schrittweite c*rms je Zeile per Bisektion, so dass Huffman-Bytes <= I-048-Ziel; ungeclippt, |q|<=3 (7 Stufen), |q|<=2 (5 Stufen)), mit / ohne Rotation.
  Huffman-Tabelle: 3 B je Symbol. Gegen den Festlaengen-Arm (rotv_zeile).
FRAGE E11: Quantisierung bei festem Ziel (Festlaenge, ohne Fuellung): kein Dither / subtraktiver Dither (u~U(-1/2,1/2) Schritt, aus Seed, abgezogen) / Dither NICHT abgezogen / feste Zufallsrundung, roh und rotiert.
  Dither kostet 0 Bytes (Seed). Skala je Zeile per MSE-Gitter ueber den jeweiligen Quantisierer.
Rotationsziehungen: 3 (Voll) bzw. 1 (--schnell); angegeben mit Mittel und Streuung. Maße: Test-/Val-Genauigkeit, Bytes, relativer Gewichtsfehler.
Kosten Raven (geschaetzt): ~25-30 min je Seed (2 Datensaetze). Aufruf: python3 erkundung/E01.py --seed 950 [--schnell]"""
import heapq
import numpy as np

import e_basis as eb
from e_basis import F, kb, nn, torch


def huffman_laengen(zaehl):
    """Huffman-Codelaengen (Bits) je Symbol aus Zaehlungen (dict Symbol->Anzahl)."""
    if len(zaehl) == 1:
        return {s: 1 for s in zaehl}
    h = [(c, i, [s]) for i, (s, c) in enumerate(zaehl.items())]
    heapq.heapify(h)
    lang = {s: 0 for s in zaehl}
    k = len(h)
    while len(h) > 1:
        c1, _, s1 = heapq.heappop(h)
        c2, _, s2 = heapq.heappop(h)
        for s in s1 + s2:
            lang[s] += 1
        k += 1
        heapq.heappush(h, (c1 + c2, k, s1 + s2))
    return lang


def ent_schicht(Wr, c, qm):
    """Gitter c*rms je Zeile, Rundung (optional Clip |q|<=qm). Rueckgabe (Wq, Huffman-Bytes inkl. Tabelle, ideale Entropie-Bytes)."""
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    d = rms * c
    q = (Wr / d).round()
    if qm:
        q = q.clamp(-qm, qm)
    vals, cnt = torch.unique(q, return_counts=True)
    z = {int(v): int(n) for v, n in zip(vals, cnt)}
    L = huffman_laengen(z)
    hb = sum(z[s] * L[s] for s in z) / 8 + 3 * len(z)
    p = cnt.double() / cnt.sum()
    eb_ = float(-(p * p.log2()).sum()) * Wr.numel() / 8
    return q * d, hb, eb_


def ent_netz(netz, rot, qm, seed, bits):
    """Je Schicht Bisektion auf c, bis Huffman-Bytes + Zeilenskalen + Bias + Vorzeichenvektor <= I-048-Ziel (Rotation+Zeilenskala, bits)."""
    n = eb.copy.deepcopy(netz)
    ges = ziel = ent = 0.0
    k = 0
    for m in n.modules():
        if isinstance(m, nn.Linear):
            k += 1
            W = m.weight.data
            o, i = W.shape
            Wr, Q = eb.rotiere(W, rot, seed * 101 + k)
            ip = Wr.shape[1]
            bud = kb.bytes_q_linear(o, i, bits, True, pad=True)
            uber = o * 2 + o * 2 + (ip / 8 if rot is not None else 0)
            lo, hi = 0.02, 8.0
            for _ in range(28):
                mid = (lo * hi) ** 0.5
                _, hb, _ = ent_schicht(Wr, mid, qm)
                if hb + uber > bud:
                    lo = mid
                else:
                    hi = mid
            Wq, hb, e_ = ent_schicht(Wr, hi, qm)
            m.weight.data = eb.zurueck(Wq, Q, i)
            ges += hb + uber
            ziel += bud
            ent += e_ + uber
    return n, ges, ziel, ent


def dither_schicht(W, bits, modus, rot, seed):
    """Festlaenge, Skala je Zeile per MSE-Gitter (13 Werte) ueber den jeweiligen Quantisierer. modus: det | sub | nosub | stoch."""
    Wr, Q = eb.rotiere(W, rot, seed)
    qmax = 2 ** (bits - 1) - 1
    g = torch.Generator().manual_seed(seed + 31337)
    u = torch.rand(Wr.shape, generator=g) - 0.5          # U(-1/2, 1/2) in Schritten
    r = torch.rand(Wr.shape, generator=g)                # fuer feste Zufallsrundung
    amax = Wr.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, bs = None, None
    for f in np.linspace(0.4, 1.0, 13):
        s = amax * float(f) / qmax
        x = Wr / s
        if modus == "det":
            Wq = x.round().clamp(-qmax, qmax) * s
        elif modus == "sub":
            Wq = ((x + u).round().clamp(-qmax, qmax) - u) * s
        elif modus == "nosub":
            Wq = (x + u).round().clamp(-qmax, qmax) * s
        else:
            Wq = torch.floor(x + r).clamp(-qmax, qmax) * s
        e = ((Wr - Wq) ** 2).sum(1, keepdim=True)
        if best is None:
            best, bs = e, Wq
        else:
            mk = e < best
            best, bs = torch.where(mk, e, best), torch.where(mk, Wq, bs)
    return eb.zurueck(bs, Q, W.shape[1])


ARME = ([("roh_zeile", None, None)] + [(f"roh_g{g}", None, g) for g in (8, 16, 32, 64, 128)]
        + [("rotv_zeile", "voll", None)] + [(f"rotv_g{g}", "voll", g) for g in (16, 32, 64, 128)]
        + [(f"rotb{k}_g{k}", k, k) for k in (8, 32, 128)] + [(f"rotb{k}_zeile", k, None) for k in (8, 32, 128)])


def stat(v):
    v = [float(x) for x in v]
    return {"mittel": float(np.mean(v)), "sd": float(np.std(v)), "werte": v}


def main(seed, schnell=False):
    P = eb.profil(schnell)
    R = 1 if schnell else 3
    out = {"profil": P, "ergebnisse": {}}
    for ds in P["ds"]:
        klein = P["klein"]
        eb.lg(f"[E01] {ds}: trainiere MLP")
        netz, info = eb.trainiere_mlp(ds, seed, (512, 512), P["ep"], 1e-3, klein)
        res = {"fp": {"test": eb.acc(netz, ds, klein), "val": eb.acc(netz, ds, klein, "val"), "bytes_fp16": kb.bytes_fp16_linear(netz), "macs": kb.macs_linear(netz),
                      "train_flops": eb.train_flops(kb.macs_linear(netz), ds, klein, P["ep"]), "instabil": info["instabil"]}}
        for bits in (2, 3):
            ziel = kb.ziel_bytes(ds, bits)
            B = {"ziel_bytes": ziel, "arme": {}}
            # ---- E1 / E4: Gruppenskalen und Blockrotation bei exakten Bytes
            for name, rot, g in ARME:
                te, va, by, fe = [], [], [], []
                for r in range(R if rot is not None else 1):
                    nq, b, z = eb.quantisiere_netz(netz, bits, rot, g, seed * 1000 + r, True, bits)
                    te.append(eb.acc(nq, ds, klein)); va.append(eb.acc(nq, ds, klein, "val")); by.append(b); fe.append(eb.relfehler(netz, nq))
                B["arme"][name] = {"test": stat(te), "val": stat(va), "bytes": float(np.mean(by)), "relfehler": float(np.mean(fe)), "rot": rot, "gruppe": g,
                                   "im_budget": bool(max(by) <= z + 1e-6), "block": "E4" if name.startswith("rotb") else "E1"}
            # ---- E10: Entropiecodierung
            for name, rot, qm in (("ent_roh", None, None), ("ent_roh_q3", None, 3), ("ent_roh_q2", None, 2),
                                  ("ent_rotv", "voll", None), ("ent_rotv_q3", "voll", 3), ("ent_rotv_q2", "voll", 2)):
                te, va, by, en = [], [], [], []
                for r in range(R if rot is not None else 1):
                    nq, b, z, e_ = ent_netz(netz, rot, qm, seed * 1000 + r, bits)
                    te.append(eb.acc(nq, ds, klein)); va.append(eb.acc(nq, ds, klein, "val")); by.append(b); en.append(e_)
                B["arme"][name] = {"test": stat(te), "val": stat(va), "bytes_huffman": float(np.mean(by)), "bytes_entropie_ideal": float(np.mean(en)), "ziel": z, "block": "E10"}
            # ---- E11: Dither (Festlaenge, feste Bits = Ziel, ohne Fuellung)
            for rot in (None, "voll"):
                for modus in ("det", "sub", "nosub", "stoch"):
                    te, va = [], []
                    for r in range(R):
                        s0 = seed * 1000 + r
                        nq = eb.abbilde_netz(netz, lambda W, k: dither_schicht(W, bits, modus, rot, s0 * 101 + k))
                        te.append(eb.acc(nq, ds, klein)); va.append(eb.acc(nq, ds, klein, "val"))
                    B["arme"][f"dither_{'rotv' if rot else 'roh'}_{modus}"] = {"test": stat(te), "val": stat(va), "bytes": float(kb.bytes_q_gross(ds, bits, rot is not None)), "block": "E11"}
            # ---- Kontrolle: kleines dichtes fp16-Modell gleicher Bytes
            B["klein_fp16"] = eb.klein_bester(ds, ziel, seed, P["ep"], klein, formen=("flach", "zwei"), lrs=(1e-3,))
            res[f"bits{bits}"] = B
            tab = sorted(((a["test"]["mittel"], n) for n, a in B["arme"].items() if a.get("im_budget", True)), reverse=True)
            eb.lg(f"[E01] {ds} {bits} Bit fp {res['fp']['test']:.4f} klein {B['klein_fp16']['test']:.4f}; Spitze:", [(n, round(t, 4)) for t, n in tab[:5]])
        out["ergebnisse"][ds] = res
    return out


if __name__ == "__main__":
    eb.cli("E01", main)
