"""Z1_seedbuch -- Seed-erzeugtes Zufalls-Codebuch auf rotierten Blöcken (SeedLM + Rotation, C01; Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: Schlägt ein SeedLM-artiges Buch (Block aus 8 Gewichten = Seed-Index + wenige Koeffizienten, Buch 0 Byte) NACH Hadamard-Rotation bei gleichen Bits je Gewicht (2 Bit, Overhead mitgezählt)
  (a) die Rotationslosen Varianten (hilft die Rotation dem Seed-Buch?), (b) das gelernte k-Means-Buch mit mitgezählten Buchbytes, (c) Skalar 4-stufig, (d) das E8-Gitter-Buch (QuIP#-Idee ohne Feintuning, 0 Byte),
  (e) ein kleineres fp16-Modell gleicher Bytes?
VORAB-P (vor dem Lauf): 45 % grün (Quelle Z1_kombinator C01). Teil: Rotation hilft dem Seed-Buch (seedlm_rot > seedlm_roh) 75 %; seedlm_rot schlägt e8_rot 20 %; schlägt k-Means-Buch (Buchbytes mitgezählt) 35 %;
  schlägt skalar4_rot 35 %. Das Ampelfeld verlangt den Sieg über ALLE Gegner, also ist grün hier eher 20 %; die 45 % beziehen sich auf die weichere Fassung "Rotation hilft + Reproduktion".
MODELL: MNIST-MLP 784-512-512-10 fp32, Adam 1e-3, Kosinus, ZWEI Trainingslängen EP_KURZ = 3 und 12 Epochen (--schnell: 1 und 4); PTQ ohne Nachtraining; Val 5 000 Bilder.
ARME (alle 2 Bit/Gewicht = 16 Bit je 8er-Block, Zeilen-RMS-Normierung, Zeilenskala + Bias fp16 + Vorzeichendiagonale bei Rotation = kb.bytes_q_linear):
  seedlm_s10p2q3   SeedLM-artig: Block = 10 Bit Seed-Index (1024 Seeds) + 2 Koeffizienten à 3 Bit; Seed -> Zufallsmatrix U (8x2) aus festem Generator (datenfrei, Index+Gesamtseed bestimmen U), Koeffizienten per kleinste Quadrate
                   (Pseudoinverse), auf 3 Bit gerundet, Seed mit kleinstem Fehler gewählt; Schrittweite der Koeffizienten je Schicht aus einer Probe von 1 024 Blöcken (5 Werte)
  seedlm_s12p1q4   12 Bit Seed + 1 Koeffizient à 4 Bit (Form-Verstärkung)
  zufall_rot       K03-Arm: 65 536 Gauß-Wörter (16 Bit Seed, kein Koeffizient), globaler Verstärkungsfaktor je Schicht
  e8_rot           K03-Arm: die 65 536 kürzesten E8-Punkte (0 Byte)
  km4_K256_rot     K03B-Arm: gelerntes k-Means-Buch, Blöcke à 4, 256 Wörter (8 Bit je 4 Gewichte), EIN Buch für alle Schichten, Buchbytes (fp16) im Feld bytes_buch MITGEZÄHLT
  skalar4_rot      K03-Arm: 4 echte Stufen je Zeile
  *_roh            seedlm_s10p2q3, seedlm_s12p1q4, km4_K256, skalar4 ohne Rotation
  klein_fp16       GELERNTER Gegenspieler: MLP h-h in fp16, Breite so, dass Bytes <= skalar4_rot, gleiche Epochen
ZUSATZ: bits_pro_gewicht_effektiv = 8 * Bytes / Gewichte (zeigt den Overhead: Skalen, Diagonale, Bias, Buch).
AMPEL (Einzelseed, vorläufig): Gewinn = Val-Acc(seedlm_s10p2q3_rot) − max(e8_rot, km4_K256_rot, skalar4_rot, klein_fp16), Pp, Minimum über beide Längen, Schwelle 0,5 pp.
  Zusatz: ampel_rotation = seedlm_rot − seedlm_roh; ampel_e8 = seedlm_rot − e8_rot (nur gegen die 0-Byte-Konkurrenz).
Lies zuerst: arme[<länge>] seedlm_*_rot gegen *_roh (Rotation hilft?), dann gegen e8_rot und km4, zuletzt gewinn_pp_je_laenge."""
from __future__ import annotations
from _z1 import *
import K03 as K
import K03B as KB

BLOCK = 8
DELTA_F = (0.3, 0.45, 0.6, 0.8, 1.1)


def seedlm_buch(S, P, seed):
    """2^S Seeds -> Zufallsmatrizen U (8 x P), Pseudoinversen. U[s] hängt nur von (Gesamtseed, s) ab: 0 Byte Buch."""
    g = torch.Generator().manual_seed(seed * 19 + 3)
    U = torch.randn(2 ** S, BLOCK, P, generator=g)
    return U, torch.linalg.pinv(U)


def _rund(T, delta, q):
    L = 2 ** q
    hi = L / 2 - 0.5
    return ((T / delta - 0.5).round() + 0.5).clamp(-hi, hi) * delta


def _kodiere(Xc, U, Up, delta, q):
    T = torch.einsum("bd,spd->bsp", Xc, Up)
    R = torch.einsum("bsp,sdp->bsd", _rund(T, delta, q), U)
    err = ((R - Xc[:, None, :]) ** 2).sum(-1)
    e, i = err.min(1)
    return i, R[torch.arange(len(Xc)), i], e


def seedlm_quant(Wr, U, Up, q, probe=1024, chunk=256):
    o, n = Wr.shape
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    X = (Wr / rms).reshape(-1, BLOCK)
    sel = torch.randperm(len(X), generator=torch.Generator().manual_seed(len(X)))[:probe]
    sig = float(torch.einsum("bd,spd->bsp", X[sel[:128]], Up).std())
    fehler = []
    for f in DELTA_F:
        delta = sig * f * 8 / 2 ** q
        fehler.append(sum(float(_kodiere(X[sel][a:a + chunk], U, Up, delta, q)[2].sum()) for a in range(0, len(sel), chunk)))
    f = DELTA_F[int(np.argmin(fehler))]
    delta = sig * f * 8 / 2 ** q
    Xq = torch.empty_like(X)
    with torch.no_grad():
        for a in range(0, len(X), chunk):
            Xq[a:a + chunk] = _kodiere(X[a:a + chunk], U, Up, delta, q)[1]
    return Xq.reshape(o, n) * rms, f


def seedlm_ersetze(netz, S, P, q, rot, seed):
    U, Up = seedlm_buch(S, P, seed)
    ls = [KB.QK(l, rot, seed * 101 + k) for k, l in enumerate(netz.ls)]
    fs = []
    for l in ls:
        Wq, f = seedlm_quant(l.Wr, U, Up, q)
        l.setze(Wq)
        fs.append(f)
    return kb.Netz(ls), ls, fs


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 4) if schnell else (3, 12)
    K8 = 4096 if schnell else 65536
    xtr, ytr, xva, yva = mlp_daten(schnell)
    out = {"epochen": eps, "K": K8, "arme": {}}
    gew = 784 * 512 + 512 * 512 + 512 * 10
    for ep in eps:
        tl = time.time()
        torch.manual_seed(seed)
        netz = kb.baue_bild([512, 512])
        mlp_train(netz, xtr, ytr, seed, ep, 1e-3)
        a0 = acc(netz, xva, yva)
        with torch.no_grad():
            z0 = netz(xva)
        A = out["arme"][ep] = {"fp32": {"acc": a0, "bytes_fp16": kb.bytes_fp16_linear(netz)}}
        print(f"Z1_seedbuch s{seed} ep{ep} fp32 {a0:.4f}", flush=True)

        def eintrag(name, q, ls, t, buch=0, extra=None):
            q.eval()
            with torch.no_grad():
                zq = q(xva)
            b0 = sum(l.bytes() for l in ls)
            A[name] = {"acc": acc(q, xva, yva), "gewichtsfehler_rel": sum(l.fehl * l.shape[0] * l.shape[1] for l in ls) / sum(l.shape[0] * l.shape[1] for l in ls),
                       "logit_fehler_rel": float((zq - z0).norm() / z0.norm()), "bytes": b0 + buch, "bytes_buch": buch, "bits_pro_gewicht_effektiv": 8 * (b0 + buch) / gew,
                       "sekunden": time.time() - t}
            if extra:
                A[name].update(extra)
            print(f"   {name:18s} acc {A[name]['acc']:.4f}  Gew.fehler {A[name]['gewichtsfehler_rel']:.4f}  Bytes {b0 + buch:.0f}  ({time.time() - t:.0f}s)", flush=True)

        cbs = {}
        for rot in (True, False):
            r = "rot" if rot else "roh"
            for S, P, q in ((10, 2, 3), (12, 1, 4)):
                t = time.time()
                qn, ls, fs = seedlm_ersetze(netz, S, P, q, rot, seed)
                eintrag(f"seedlm_s{S}p{P}q{q}_{r}", qn, ls, t, extra={"delta_faktor_je_schicht": fs})
            t = time.time()
            qn, ls = K.ersetze(netz, "skalar4", rot, seed, None)
            eintrag(f"skalar4_{r}", qn, ls, t)
            t = time.time()
            qn, ls, C = KB.km_ersetze(netz, 4, 256, rot, seed, 12)
            eintrag(f"km4_K256_{r}", qn, ls, t, buch=int(C.shape[0] * 4 * 2))
        for name, fn in (("zufall_rot", K.cb_zufall), ("e8_rot", K.cb_e8)):
            t = time.time()
            C = fn(K8, seed)
            if len(C) > K8:
                C = C[torch.randperm(len(C), generator=torch.Generator().manual_seed(seed))[:K8]]
            qn, ls = K.ersetze(netz, "cb", True, seed, K.normiere(C))
            eintrag(name, qn, ls, t, extra={"g_je_schicht": [l.g for l in ls]})
        # gelernter Gegenspieler gleicher Bytes
        h = mlp_gleiche_bytes_breite(A["skalar4_rot"]["bytes"])
        torch.manual_seed(seed + 1)
        kl = kb.baue_bild([h, h])
        mlp_train(kl, xtr, ytr, seed, ep, 1e-3)
        kl = fp16_runden(kl)
        A["klein_fp16"] = {"acc": acc(kl, xva, yva), "bytes": kb.bytes_fp16_linear(kl), "breite": h, "macs": mlp_macs([h, h])}
        print(f"   klein_fp16 h={h} acc {A['klein_fp16']['acc']:.4f}  Bytes {A['klein_fp16']['bytes']}  (Länge {ep} Ep. gesamt {time.time() - tl:.0f}s)", flush=True)
    g_h, g_r, g_e = [], [], []
    for ep in eps:
        A = out["arme"][ep]
        gegner = max(A[k]["acc"] for k in ("e8_rot", "km4_K256_rot", "skalar4_rot", "klein_fp16"))
        g_h.append(100 * (A["seedlm_s10p2q3_rot"]["acc"] - gegner))
        g_r.append(100 * (A["seedlm_s10p2q3_rot"]["acc"] - A["seedlm_s10p2q3_roh"]["acc"]))
        g_e.append(100 * (A["seedlm_s10p2q3_rot"]["acc"] - A["e8_rot"]["acc"]))
    out["gewinn_pp_je_laenge"] = {"seedlm_rot_gegen_haertesten": g_h, "seedlm_rot_minus_roh": g_r, "seedlm_rot_minus_e8": g_e}
    out["ampel"] = ampel(min(g_h), 0.5, "pp", "seedlm_s10p2q3_rot gegen max(e8_rot, km4_K256_rot, skalar4_rot, klein_fp16), Minimum über beide Längen")
    out["ampel_rotation"] = ampel(min(g_r), 0.5, "pp", "seedlm_s10p2q3_rot gegen seedlm_s10p2q3_roh")
    out["ampel_e8"] = ampel(min(g_e), 0.5, "pp", "seedlm_s10p2q3_rot gegen e8_rot")
    print("Ampel", out["ampel"]["farbe"], "Rotation", out["ampel_rotation"]["farbe"], "e8", out["ampel_e8"]["farbe"], out["gewinn_pp_je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z1_seedbuch", main)
