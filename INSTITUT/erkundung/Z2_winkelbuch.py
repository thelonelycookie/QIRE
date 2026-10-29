"""Z2_winkelbuch -- Norm + Richtung: Winkel-/Orbit-Codebuch für rotierte Gewichtsblöcke (Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: Ein Block aus d = 2, 4, 8 Gewichten (nach Hadamard-Rotation und Zeilen-RMS-Normierung) wird in Norm und Richtung zerlegt (Gain-Shape). Die Richtung liegt auf der Einheitssphäre S^(d-1)
  und wird über ein Buch OHNE Speicher quantisiert, die Norm mit wenigen Bit (Lloyd-Stufen je Schicht). Schlägt das bei GLEICHEN Bits/Gewicht (2 Bit, Overhead mitgezählt) die Gegner?
VORAB-P (vor dem Lauf): grün gegen alle Gegner 10 % (E8 und das gelernte k-Means-Buch sind für 2 Bit bei Gaußblöcken schon fast an der Ratenverzerrungsgrenze; ein Winkelgitter ist auf der Sphäre
  ungleichmäßig dicht und verliert dort); Orbit-Buch besser als Winkelgitter bei d=8: 70 %; Orbit-Buch schlägt Skalar 4-stufig: 40 %; schlägt e8_rot: 10 %; schlägt km4_K256_rot: 15 %.
MODELL: MNIST-MLP 784-512-512-10 fp32, Adam 1e-3, Kosinus, ZWEI Trainingslängen EP = 3 und 12 (--schnell: 1 und 4); PTQ ohne Nachtraining; Val 5 000 Bilder; Rotation = kb.rot_matrix wie K03.
ARME (alle genau 2 Bit/Gewicht = 2d Bit je d-Block: B_n Bit Norm + 2d-B_n Bit Richtung; B_n = 1 / 2 / 3 für d = 2 / 4 / 8; Norm-Stufen (2^B_n fp16 je Schicht) werden mitgezählt):
  winkel_d{2,4,8}  Richtung: hypersphärische Winkel gleichmäßig (Polarwinkel Mittelpunktsraster in [0,π], Azimut gleichmäßig in [0,2π)), Bits gleichmäßig auf die d-1 Winkel verteilt, Buch aus Formel (0 Byte)
  orbit_d{2,4,8}   Richtung: Orbits der reellen Pauli-Gruppe (X^a Z^b mit Vorzeichen) zusammen mit H^(⊗q) (q = log2 d, "Zustände auf der Bloch-Kugel"): |G| = 16 / 64 / 256; Seed-Vektoren per Zufall (Seed),
                   bester von 6 Versuchen auf 20 000 Kugelproben (datenfrei, 0 Byte); ist das Buch kleiner als |G| (d=2), wird ein Orbit per Farthest-Point-Auswahl ausgedünnt
  Kodierung: Richtung = Buchwort mit größtem Skalarprodukt, Norm-Wert = Skalarprodukt rho = x·u, auf Lloyd-Stufen von rho gerundet; Block = rho_q * u.
GEGNER (gleiche Bytes bzw. mehr): skalar4_rot (4 echte Stufen je Zeile, = 2 Bit), skalar2_rot (ternär, Zusatz), km4_K256_rot (gelernt, 8 Bit je 4er-Block, Buchbytes mitgezählt),
  e8_rot (65 536 kürzeste E8-Punkte, 0 Byte; --schnell 4 096), klein_fp16 (MLP h-h in fp16 mit Bytes <= skalar4_rot, gleiche Epochen).
ZÄHLUNG: bytes (inkl. Zeilenskala, Bias, Vorzeichendiagonale, Norm-Stufen, Buch), bits_pro_gewicht_effektiv; flops_inferenz = 2*MACs + Hadamard-Additionen der Eingänge (gleiche Formel für alle Rotationsarme);
  flops_rekonstruktion_einmalig = Dekodierung (1 Multiplikation je Gewicht) + Aufbau des Buchs (Winkel: ~d je Wort; Orbit: 2 d² je Orbitpunkt); k-Means/E8: Buch gespeichert bzw. Tabelle (0).
AMPEL (Einzelseed, vorläufig): Gewinn = beste Val-Acc der 6 eigenen Arme − max(skalar4_rot, km4_K256_rot, e8_rot, klein_fp16), Pp, Minimum über beide Längen, Schwelle 0,5 pp.
  (6 eigene Arme = Mehrfachvergleich, daher gewinn_je_arm mitgeliefert; ampel_winkel / ampel_orbit getrennt.)
Lies zuerst: arme[<länge>] winkel_* gegen orbit_* gleiches d, dann gegen skalar4_rot, km4, e8, zuletzt gewinn_pp_je_laenge."""
from __future__ import annotations
from _z1 import *
import K03 as K
import K03B as KB

NB = {2: 1, 4: 2, 8: 3}                       # Norm-Bits je Blocklänge
GEW = 784 * 512 + 512 * 512 + 512 * 10


# ------------------------------------------------------------------ Richtungsbücher (Einheitsvektoren)
def winkel_buch(d, B):
    """2^B Richtungen auf S^(d-1): d-1 Winkel, Bits gleichmäßig verteilt (Rest auf die ersten Winkel), Polarwinkel Mittelpunkte in [0,π], Azimut in [0,2π)."""
    na = d - 1
    bits = [B // na + (1 if k < B % na else 0) for k in range(na)]
    ach = []
    for k, b in enumerate(bits):
        m = 2 ** b
        if k < na - 1:
            ach.append((torch.arange(m) + 0.5) / m * math.pi)
        else:
            ach.append(torch.arange(m) / m * 2 * math.pi)
    gitter = torch.cartesian_prod(*ach) if na > 1 else ach[0][:, None]
    gitter = gitter.reshape(-1, na)
    n = gitter.shape[0]
    u = torch.ones(n, d)
    for k in range(na - 1):
        u[:, k] = u[:, k] * torch.cos(gitter[:, k])
        u[:, k + 1:] = u[:, k + 1:] * torch.sin(gitter[:, k])[:, None]
    u[:, na - 1] = u[:, na - 1] * torch.cos(gitter[:, na - 1])
    u[:, na] = u[:, na] * torch.sin(gitter[:, na - 1])
    return u


def pauli_h_gruppe(d):
    """Reelle Pauli-Gruppe {±X^a Z^b} und H^(⊗q)·Pauli als Matrizen (|G| = 2·2·d·d)."""
    q = int(round(math.log2(d)))
    idx = torch.arange(d)
    Hs = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
    Hq = torch.ones(1, 1)
    for _ in range(q):
        Hq = torch.kron(Hq, Hs)
    mats = []
    for a in range(d):
        Xa = torch.zeros(d, d)
        Xa[idx ^ a, idx] = 1.0
        for b in range(d):
            Zb = torch.diag(torch.tensor([(-1.0) ** bin(int(i) & b).count("1") for i in range(d)]))
            P = Xa @ Zb
            for s in (1.0, -1.0):
                mats.append(s * P)
                mats.append(s * Hq @ P)
    return torch.stack(mats)


def fps(P, n, seed):
    g = torch.Generator().manual_seed(seed)
    ersten = int(torch.randint(0, len(P), (1,), generator=g))
    sel = [ersten]
    dist = ((P - P[ersten]) ** 2).sum(1)
    for _ in range(n - 1):
        j = int(dist.argmax())
        sel.append(j)
        dist = torch.minimum(dist, ((P - P[j]) ** 2).sum(1))
    return P[sel]


def fehler_kugel(C, d, seed, m=20000):
    g = torch.Generator().manual_seed(seed + 99)
    X = torch.randn(m, d, generator=g)
    X = X / X.norm(dim=1, keepdim=True)
    best = torch.cat([(X[a:a + 4096] @ C.T).max(1)[0] for a in range(0, m, 4096)])
    return float((1 - best).mean())


def orbit_buch(d, B, seed):
    G = pauli_h_gruppe(d)
    Kd = 2 ** B
    best, bc = None, None
    for versuch in range(6):
        g = torch.Generator().manual_seed(seed * 211 + d * 7 + versuch)
        if Kd >= len(G):
            ns = math.ceil(Kd / len(G))
            V = torch.randn(ns, d, generator=g)
            V = V / V.norm(dim=1, keepdim=True)
            C = torch.einsum("gij,sj->sgi", G, V).reshape(-1, d)[:Kd]
        else:
            v = torch.randn(d, generator=g)
            v = v / v.norm()
            C = fps(torch.einsum("gij,j->gi", G, v), Kd, seed + versuch)
        e = fehler_kugel(C, d, seed)
        if best is None or e < best:
            best, bc = e, C
    return bc, best


def buchbau_flops(art, d, B):
    return int(2 ** B * d) if art == "winkel" else int(2 ** B * 2 * d * d)


# ------------------------------------------------------------------ Norm-Stufen (Lloyd 1D)
def lloyd1d(v, L, iters=25):
    v = v.flatten()
    if len(v) > 60000:
        v = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(len(v)))[:60000]]
    c = torch.quantile(v, (torch.arange(L) + 0.5) / L)
    for _ in range(iters):
        i = (v[:, None] - c[None]).abs().argmin(1)
        for k in range(L):
            m = i == k
            if m.any():
                c[k] = v[m].mean()
    return c.half().float()


def quant_gain_shape(Wr, U, d, bn):
    """Wr (o,n) rotiert -> Zeilen-RMS-normierte d-Blöcke -> nächste Richtung, rho auf 2^bn Lloyd-Stufen. Gibt (Wq, Stufen)."""
    o, n = Wr.shape
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    X = (Wr / rms).reshape(-1, d)
    idx = torch.empty(len(X), dtype=torch.long)
    for a in range(0, len(X), 4096):
        idx[a:a + 4096] = (X[a:a + 4096] @ U.T).argmax(1)
    u = U[idx]
    rho = (X * u).sum(1)
    st = lloyd1d(rho, 2 ** bn)
    rq = st[(rho[:, None] - st[None]).abs().argmin(1)]
    return (rq[:, None] * u).reshape(o, n) * rms, st


def gs_ersetze(netz, art, d, rot, seed):
    B = 2 * d - NB[d]
    if art == "winkel":
        U, buchfehler = winkel_buch(d, B), None
        buchfehler = fehler_kugel(U, d, seed)
    else:
        U, buchfehler = orbit_buch(d, B, seed)
    ls = [KB.QK(l, rot, seed * 101 + k) for k, l in enumerate(netz.ls)]
    zus = 0
    for l in ls:
        Wq, st = quant_gain_shape(l.Wr, U, d, NB[d])
        l.setze(Wq)
        zus += len(st) * 2
    return kb.Netz(ls), ls, zus, buchfehler


def rot_adds(dims=((1024, 512), (512, 512), (512, 10))):
    return int(sum(n * math.log2(n) for n, _ in dims))


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 4) if schnell else (3, 12)
    K8 = 4096 if schnell else 65536
    xtr, ytr, xva, yva = mlp_daten(schnell)
    out = {"epochen": eps, "K_e8": K8, "norm_bits": NB, "arme": {}}
    macs = mlp_macs([512, 512])
    fl_inf = 2 * macs + rot_adds()
    for ep in eps:
        tl = time.time()
        torch.manual_seed(seed)
        netz = kb.baue_bild([512, 512])
        mlp_train(netz, xtr, ytr, seed, ep, 1e-3)
        a0 = acc(netz, xva, yva)
        with torch.no_grad():
            z0 = netz(xva)
        A = out["arme"][ep] = {"fp32": {"acc": a0, "bytes_fp16": kb.bytes_fp16_linear(netz)}}
        print(f"Z2_winkelbuch s{seed} ep{ep} fp32 {a0:.4f}", flush=True)

        def eintrag(name, q, ls, t, extra_bytes=0, buch=0, rek=None, extra=None):
            q.eval()
            with torch.no_grad():
                zq = q(xva)
            b = sum(l.bytes() for l in ls) + extra_bytes + buch
            A[name] = {"acc": acc(q, xva, yva), "gewichtsfehler_rel": sum(l.fehl * l.shape[0] * l.shape[1] for l in ls) / sum(l.shape[0] * l.shape[1] for l in ls),
                       "logit_fehler_rel": float((zq - z0).norm() / z0.norm()), "bytes": b, "bytes_buch": buch, "bits_pro_gewicht_effektiv": 8 * b / GEW,
                       "flops_inferenz": fl_inf, "flops_rekonstruktion_einmalig": GEW + (rek or 0), "sekunden": time.time() - t}
            if extra:
                A[name].update(extra)
            print(f"   {name:16s} acc {A[name]['acc']:.4f}  Gew.fehler {A[name]['gewichtsfehler_rel']:.4f}  Bytes {b:.0f}  ({time.time() - t:.0f}s)", flush=True)

        for art in ("winkel", "orbit"):
            for d in (2, 4, 8):
                t = time.time()
                qn, ls, zus, bf = gs_ersetze(netz, art, d, True, seed)
                eintrag(f"{art}_d{d}", qn, ls, t, extra_bytes=zus, rek=buchbau_flops(art, d, 2 * d - NB[d]),
                        extra={"d": d, "bits_norm": NB[d], "bits_richtung": 2 * d - NB[d], "kugelfehler_buch": bf})
        t = time.time()
        qn, ls = K.ersetze(netz, "skalar4", True, seed, None)
        eintrag("skalar4_rot", qn, ls, t)
        t = time.time()
        qn, ls = K.ersetze(netz, "skalar2", True, seed, None)
        eintrag("skalar2_rot", qn, ls, t)
        t = time.time()
        qn, ls, C = KB.km_ersetze(netz, 4, 256, True, seed, 12)
        eintrag("km4_K256_rot", qn, ls, t, buch=int(C.shape[0] * 4 * 2))
        t = time.time()
        C = K.cb_e8(K8, seed)
        if len(C) > K8:
            C = C[torch.randperm(len(C), generator=torch.Generator().manual_seed(seed))[:K8]]
        qn, ls = K.ersetze(netz, "cb", True, seed, K.normiere(C))
        eintrag("e8_rot", qn, ls, t, extra={"g_je_schicht": [l.g for l in ls]})
        h = mlp_gleiche_bytes_breite(A["skalar4_rot"]["bytes"])
        torch.manual_seed(seed + 1)
        kl = kb.baue_bild([h, h])
        mlp_train(kl, xtr, ytr, seed, ep, 1e-3)
        kl = fp16_runden(kl)
        A["klein_fp16"] = {"acc": acc(kl, xva, yva), "bytes": kb.bytes_fp16_linear(kl), "breite": h, "flops_inferenz": 2 * mlp_macs([h, h]), "flops_rekonstruktion_einmalig": 0}
        print(f"   klein_fp16 h={h} acc {A['klein_fp16']['acc']:.4f}  Bytes {A['klein_fp16']['bytes']}  (Länge {ep} Ep. gesamt {time.time() - tl:.0f}s)", flush=True)
    eigene = [f"{a}_d{d}" for a in ("winkel", "orbit") for d in (2, 4, 8)]
    gegner = ("skalar4_rot", "km4_K256_rot", "e8_rot", "klein_fp16")
    g_je_arm, g_best, g_w, g_o = {n: [] for n in eigene}, [], [], []
    for ep in eps:
        A = out["arme"][ep]
        gm = max(A[k]["acc"] for k in gegner)
        for n in eigene:
            g_je_arm[n].append(100 * (A[n]["acc"] - gm))
        g_best.append(max(g_je_arm[n][-1] for n in eigene))
        g_w.append(max(g_je_arm[n][-1] for n in eigene if n.startswith("winkel")))
        g_o.append(max(g_je_arm[n][-1] for n in eigene if n.startswith("orbit")))
    out["gewinn_pp_je_laenge"] = {"bester_eigener_arm_gegen_haertesten": g_best, "bester_winkel": g_w, "bester_orbit": g_o}
    out["gewinn_pp_je_arm"] = g_je_arm
    txt = "bester der 6 eigenen Arme gegen max(skalar4_rot, km4_K256_rot, e8_rot, klein_fp16), Minimum über beide Längen"
    out["ampel"] = ampel(min(g_best), 0.5, "pp", txt)
    out["ampel_winkel"] = ampel(min(g_w), 0.5, "pp", "bester Winkelarm gegen dieselben Gegner")
    out["ampel_orbit"] = ampel(min(g_o), 0.5, "pp", "bester Orbitarm gegen dieselben Gegner")
    print("Ampel", out["ampel"]["farbe"], "winkel", out["ampel_winkel"]["farbe"], "orbit", out["ampel_orbit"]["farbe"], out["gewinn_pp_je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z2_winkelbuch", main)
