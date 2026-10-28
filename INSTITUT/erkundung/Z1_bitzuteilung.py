"""Z1_bitzuteilung -- Bits je Zeile/Kanal nach Eingangsvarianz zuteilen, nach Zufalls-Rotation (Idee 4 aus Z1_grenzen; Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: Bringt es bei GLEICHEN Bytes (Mittel 2 Bit/Gewicht, Zusatzbytes für Skalen und Bitkarte abgezogen) etwas, die Bitbreiten (0..4 Bit) nach der Varianz der Eingangskanäle bzw. der Wichtigkeit der Zeilen
  zu verteilen statt überall 2 Bit zu geben? Nach Zufalls-Rotation sind die Eingangsvarianzen fast gleich (Rotation gaußisiert und ebnet), also erwartet die Theorie (reverse water-filling) kaum Gewinn.
VORAB-P (vor dem Lauf): 15 % grün. (Rotation ebnet die Varianzen, Zeilenzuteilung bei 3 Schichten grob; Gewinn eher ohne Rotation, dort aber nicht die Frage.) Teil: spalten_var_rot > gleich_rot um >= 0,5 pp: 15 %;
  zusätzlich über klein-fp16 gleicher Bytes: 10 %. Gelesen wird auch: spalten_var_roh > gleich_roh (tote MNIST-Randpixel erhalten 0 Bit): 70 %.
MODELL: MNIST-MLP 784-512-512-10, fp32, Adam 1e-3, Kosinus, ZWEI Trainingslängen: EP_KURZ = 6 und 24 Epochen (--schnell: 1 und 4). PTQ der fp-Netze, kein Nachtraining, Kalibrierung = 5 000 Trainingsbilder.
ARME (alle PTQ, Zeilenskala per MSE-Suche, echte Gleichgitter-Stufen; "rot" = Vorzeichen*Hadamard auf dem Eingang, MNIST-Eingang 784->1024; "roh" = ohne Drehung):
  gleich        2 Bit überall  (= Basis, kb.bytes_q_linear)
  spalten_var   Bits 0..4 je EINGANGSKANAL (Spalte) nach der Eingangsvarianz σ_j² (zweites Moment der Kalibrierdaten, rotiert falls rot): gieriges reverse water-filling mit D_j(b) = σ_j² κ_b
  spalten_varw  wie spalten_var, aber Fehlergewicht σ_j² · mittlere quadratische Gewichtsspalte (zeigt, ob die Gewichtsenergie zusätzlich trägt)
  zeilen        Bits 0..4 je AUSGANGSZEILE nach ihrem Ausgangsbeitrag Σ_j W_ij² σ_j²
  κ_b = (1; 0,3634; 0,1188; 0,0374; 0,0115) (Gauß-Quantisierungsfehler bei b = 0..4 Bit). Gruppen gleicher Bitbreite erhalten je Zeile eine eigene Skala.
BYTES (gleich für alle Arme, mit Abzug): Gewichtsbits + Zeilenskala je Bitgruppe (fp16) + Bias fp16 + Vorzeichendiagonale (Rotation) + Bitkarte 3 Bit je Kanal bzw. Zeile; das Budget der Zuteilung ist so gewählt,
  dass Bytes(Arm) <= Bytes(gleich). Ausgewiesen als "bytes".
GEGENSPIELER (gelernt, gleiche Bytes): fp16-MLP kleiner Breite h-h (fp16-Bytes <= Bytes gleich_rot), gleiche Epochen trainiert, auf fp16 gerundet.
AMPEL (Einzelseed, vorläufig): Gewinn = Val-Acc(spalten_var_rot) − max(gleich_rot, klein_fp16), Prozentpunkte, Minimum über beide Längen, Schwelle 0,5 pp. Zusatzfelder: ampel_zeilen (zeilen_rot), ampel_roh (spalten_var_roh gegen gleich_roh).
Lies zuerst: arme[<länge>] gleich_rot gegen spalten_var_rot, dann spalten_var_roh gegen gleich_roh, dann bits_hist (verteilen sich die Bits überhaupt ungleich?)."""
from __future__ import annotations
import heapq
from _z1 import *

KAPPA = (1.0, 0.3634, 0.1188, 0.0374, 0.0115)
MAXB = 4


def greedy_bits(d, summe):
    """Verteilt `summe` Bits auf Posten mit Gewicht d (0..MAXB je Posten), maximiert Fehlerabnahme Σ d·(κ_b − κ_{b+1}); (Heap, Grenzgewinn fällt mit b)."""
    n = len(d)
    b = [0] * n
    h = [(-float(d[i]) * (KAPPA[0] - KAPPA[1]), i) for i in range(n)]
    heapq.heapify(h)
    for _ in range(int(summe)):
        if not h:
            break
        _, i = heapq.heappop(h)
        b[i] += 1
        if b[i] < MAXB:
            heapq.heappush(h, (-float(d[i]) * (KAPPA[b[i]] - KAPPA[b[i] + 1]), i))
    return torch.tensor(b)


def quant_spaltenbits(Wr, bits_s):
    Wq = torch.zeros_like(Wr)
    for b in range(1, MAXB + 1):
        m = bits_s == b
        if m.any():
            Wq[:, m] = quant_gitter(Wr[:, m], b)
    return Wq


def quant_zeilenbits(Wr, bits_z):
    Wq = torch.zeros_like(Wr)
    for b in range(1, MAXB + 1):
        m = bits_z == b
        if m.any():
            Wq[m] = quant_gitter(Wr[m], b)
    return Wq


def bytes_spalten(o, ip, bits_s, rot):
    gruppen = len({int(b) for b in bits_s.tolist() if b > 0})
    return o * float(bits_s.sum()) / 8 + 2 * o * gruppen + 2 * o + (ip / 8 if rot else 0) + 3 * ip / 8


def bytes_zeilen(o, ip, bits_z, rot):
    return float(bits_z.sum()) * ip / 8 + 2 * o + 2 * o + (ip / 8 if rot else 0) + 3 * o / 8


def lagen_eingaenge(netz, X):
    xs, x = [], X
    for k, l in enumerate(netz.ls):
        xs.append(x)
        x = l(x)
        if k < len(netz.ls) - 1:
            x = F.relu(x)
    return xs


def baue_arme(netz, Xk, seed, rot, arten):
    """Gibt {art: (kb.Netz, bytes_gesamt, bits_hist je Schicht)} für alle `arten` mit/ohne Rotation."""
    with torch.no_grad():
        eing = lagen_eingaenge(netz, Xk)
    res = {a: ([], 0.0, []) for a in arten}
    for k, (l, x) in enumerate(zip(netz.ls, eing)):
        q = QATLin(l, 2, rot, seed * 101 + k + 1)
        xr = F.pad(x, (0, q.pad)) if q.pad else x
        if rot:
            xr = rot_vor(xr, q.s)
        sig = xr.pow(2).mean(0)
        Wr = q.V.detach()
        o, ip = Wr.shape
        cj = Wr.pow(2).mean(0)
        zeilen_beitrag = (Wr.pow(2) * sig[None, :]).sum(1)
        for a in arten:
            if a == "gleich":
                Wq, by, hist = quant_gitter(Wr, 2), kb.bytes_q_linear(o, l.in_features, 2, rot, pad=rot), {"2": ip}
            elif a in ("spalten_var", "spalten_varw"):
                S = int(2 * ip - 48 - 3 * ip / o)
                bs = greedy_bits(sig if a == "spalten_var" else sig * cj, S)
                Wq, by = quant_spaltenbits(Wr, bs), bytes_spalten(o, ip, bs, rot)
                hist = {str(b): int((bs == b).sum()) for b in range(MAXB + 1)}
            else:
                S = int(2 * o - 3 * o / ip)
                bz = greedy_bits(zeilen_beitrag, S)
                Wq, by = quant_zeilenbits(Wr, bz), bytes_zeilen(o, ip, bz, rot)
                hist = {str(b): int((bz == b).sum()) for b in range(MAXB + 1)}
            q2 = QATLin(l, 2, rot, seed * 101 + k + 1)
            q2.fest = Wq
            res[a][0].append(q2)
            res[a] = (res[a][0], res[a][1] + by, res[a][2] + [{"fehler_rel": float(((Wq - Wr) ** 2).sum() / (Wr ** 2).sum()), "bits": hist, "mittel_bit": sum(int(b) * c for b, c in hist.items()) / (ip if a.startswith("spalten") or a == "gleich" else o)}])
    return {a: (kb.Netz(ls).eval(), by, h) for a, (ls, by, h) in res.items()}


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 4) if schnell else (6, 24)
    xtr, ytr, xva, yva = mlp_daten(schnell)
    Xk = xtr[:5000]
    arten = ("gleich", "spalten_var", "spalten_varw", "zeilen")
    out = {"epochen": eps, "arme": {}, "kosten": {}}
    for ep in eps:
        t = time.time()
        torch.manual_seed(seed)
        n = kb.baue_bild([512, 512])
        mlp_train(n, xtr, ytr, seed, ep, 1e-3)
        a0 = acc(n, xva, yva)
        A = out["arme"][ep] = {"fp32": {"acc": a0, "bytes_fp16": kb.bytes_fp16_linear(n)}}
        for rot in (True, False):
            for art, (q, by, hist) in baue_arme(n, Xk, seed, rot, arten).items():
                A[f"{art}_{'rot' if rot else 'roh'}"] = {"acc": acc(q, xva, yva), "bytes": by, "schichten": hist}
        h = mlp_gleiche_bytes_breite(A["gleich_rot"]["bytes"])
        torch.manual_seed(seed + 1)
        kl = kb.baue_bild([h, h])
        mlp_train(kl, xtr, ytr, seed, ep, 1e-3)
        kl = fp16_runden(kl)
        A["klein_fp16"] = {"acc": acc(kl, xva, yva), "bytes": kb.bytes_fp16_linear(kl), "breite": h}
        out["kosten"][ep] = {"macs_gross": mlp_macs([512, 512]), "macs_klein": mlp_macs([h, h]), "param_gross": 784 * 512 + 512 * 512 + 5120, "param_klein": zaehle_param(kl)}
        print(f"Z1_bitzuteilung s{seed} ep{ep} fp32 {a0:.4f} ({time.time() - t:.0f}s)", flush=True)
        for k, v in A.items():
            if k != "fp32":
                print(f"   {k:20s} acc {v['acc']:.4f}  Bytes {v['bytes']:.0f}", flush=True)
    g_haupt, g_zeilen, g_roh = [], [], []
    for ep in eps:
        A = out["arme"][ep]
        gegner = max(A["gleich_rot"]["acc"], A["klein_fp16"]["acc"])
        g_haupt.append(100 * (A["spalten_var_rot"]["acc"] - gegner))
        g_zeilen.append(100 * (A["zeilen_rot"]["acc"] - gegner))
        g_roh.append(100 * (A["spalten_var_roh"]["acc"] - A["gleich_roh"]["acc"]))
    out["ampel"] = ampel(min(g_haupt), 0.5, "pp", "spalten_var_rot gegen max(gleich_rot, klein_fp16), Minimum über beide Längen")
    out["ampel_zeilen"] = ampel(min(g_zeilen), 0.5, "pp", "zeilen_rot gegen max(gleich_rot, klein_fp16)")
    out["ampel_roh"] = ampel(min(g_roh), 0.5, "pp", "spalten_var_roh gegen gleich_roh (ohne Rotation)")
    out["gewinn_pp_je_laenge"] = {"spalten_var_rot": g_haupt, "zeilen_rot": g_zeilen, "spalten_var_roh": g_roh}
    print("Ampel", out["ampel"]["farbe"], "Zeilen", out["ampel_zeilen"]["farbe"], "roh", out["ampel_roh"]["farbe"], out["gewinn_pp_je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z1_bitzuteilung", main)
