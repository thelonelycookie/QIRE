"""Z1_destill_rot -- Distillation in ein rotiertes 2-Bit-Schülermodell (C02; Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: Hilft Distillation (KD) eines fp-Lehrers dem QAT-Schüler mit Rotation (2 Bit, echte 4 Stufen) MEHR als den Gegenspielern, und bleibt das bei vierfacher Trainingslänge so?
VORAB-P (vor dem Lauf): 50 % grün (Quelle Z1_kombinator C02). Hauptrisiko: KD hilft jedem Schüler gleich, und der Rotationsvorteil schrumpft im QAT (Latentgewichte passen sich dem Gitter an).
  Teilwahrscheinlichkeiten: rot_kd > rot_hart bei beiden Längen 65 %; rot_kd > klein_kd (fp16, gleiche Bytes) bei beiden Längen 55 %; rot_kd > roh_kd 45 %.
LEHRER: MNIST-MLP 784-512-512-10 fp32, 30 Epochen Adam 1e-3 (--schnell: 3), Logits auf den 55 000 Trainingsbildern einmal vorberechnet.
SCHÜLER (Architektur identisch zum Lehrer außer "klein"; alle zwei Trainingslängen: EP_KURZ und 4 x EP_KURZ Epochen, Adam, 5 % Aufwärmen, Kosinus):
  rot_kd / rot_hart      QAT 2 Bit, Gewichte in rotierter Basis (Vorzeichen*Hadamard, Eingang 784->1024), Start = Lehrergewichte rotiert, lr 3e-4; KD = 0,9*T²*KL(T=2) + 0,1*CE; hart = nur CE
  roh_kd / roh_hart      dasselbe ohne Rotation
  rot_kd_neu / rot_hart_neu   wie rot_*, aber Zufallsstart (lr 1e-3): QAT ohne Lehrer-Vorwissen im Startgewicht
  klein_kd / klein_hart  fp32-Training, dann auf fp16 gerundet; Breite h-h so, dass fp16-Bytes <= Bytes des 2-Bit-Schülers (GLEICHE BYTES, KD hilft diesem Gegenspieler gleich)
  ptq_rot                Lehrer ohne Nachtraining rotiert + 2 Bit gerundet (Referenz, 0 Trainingsschritte)
GEGENSPIELER (fiesester je Länge): max(rot_hart, klein_kd, klein_hart). Ampelfeld: Gewinn = rot_kd − fiesester Gegenspieler (Prozentpunkte Val-Genauigkeit), Minimum über beide Längen.
BYTES: kb.bytes_q_linear (2 Bit/Gewicht + Zeilenskala + Bias fp16 + Vorzeichendiagonale bei Rotation, Eingang 1024), Gegenspieler fp16. FLOPs/Training: Schüler 3 x MACs je Beispiel,
  Lehrer-Vorwärts einmal je Bild (vorberechnet) = MACs_lehrer x 55 000 einmalig, ausgewiesen. Rauschen: Val 5 000 Bilder, sd ~ 0,25 pp je Zahl.
Lies zuerst: ampel, dann arme[lang] rot_kd gegen rot_hart und klein_kd; dann roh_kd (Rotation im QAT noch nötig?)."""
from __future__ import annotations
from _z1 import *

EP_L, LR_Q, LR_NEU, LR_FP = 30, 3e-4, 1e-3, 1e-3


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    ep_l, ep_k = (3, 1) if schnell else (EP_L, 4)
    laengen = {"kurz": ep_k, "lang": 4 * ep_k}
    xtr, ytr, xva, yva = mlp_daten(schnell)
    torch.manual_seed(seed)
    lehrer = kb.baue_bild([512, 512])
    t0 = time.time()
    mlp_train(lehrer, xtr, ytr, seed, ep_l, LR_FP)
    zl = logits(lehrer, xtr)
    out = {"ep_lehrer": ep_l, "laengen_epochen": laengen, "acc_lehrer": acc(lehrer, xva, yva), "bytes_lehrer_fp16": kb.bytes_fp16_linear(lehrer),
           "macs_lehrer": mlp_macs([512, 512]), "lehrer_vorwaerts_macs_einmalig": mlp_macs([512, 512]) * len(xtr), "sek_lehrer": time.time() - t0, "arme": {}}
    print(f"Z1_destill_rot s{seed} Lehrer {out['acc_lehrer']:.4f} ({out['sek_lehrer']:.0f}s)", flush=True)
    ref = einfrieren(ersetze_linear(lehrer, 2, True, seed))
    b2 = bytes_q(ref)
    h = mlp_gleiche_bytes_breite(b2)
    out.update({"bytes_2bit_schueler": b2, "breite_klein": h, "bytes_klein_fp16": kb.bytes_fp16_linear(kb.baue_bild([h, h])), "macs_klein": mlp_macs([h, h])})
    out["arme"]["ptq_rot"] = {"acc": acc(ref, xva, yva), "bytes": b2, "gewichtsfehler_rel": float(np.mean([m.rel_fehler() for m in qlins(ref)])), "epochen": 0}
    print(f"  ptq_rot {out['arme']['ptq_rot']['acc']:.4f}  Bytes 2Bit {b2:.0f}  klein h={h}", flush=True)

    def schueler(art, rot, kd, neu, sd):
        torch.manual_seed(sd)
        if art == "klein":
            return kb.baue_bild([h, h])
        basis = kb.baue_bild([512, 512]) if neu else lehrer
        return ersetze_linear(basis, 2, rot, sd)

    plan = [("rot_kd", "q", True, True, False), ("rot_hart", "q", True, False, False), ("roh_kd", "q", False, True, False), ("roh_hart", "q", False, False, False),
            ("rot_kd_neu", "q", True, True, True), ("rot_hart_neu", "q", True, False, True), ("klein_kd", "klein", False, True, False), ("klein_hart", "klein", False, False, False)]
    for lname, ep in laengen.items():
        out["arme"][lname] = {}
        for name, art, rot, kd, neu in plan:
            t = time.time()
            s = schueler(art, rot, kd, neu, seed * 13 + ep)
            lr = LR_FP if art == "klein" else (LR_NEU if neu else LR_Q)
            mlp_train(s, xtr, ytr, seed * 13 + ep + 1, ep, lr, lehrer=zl if kd else None)
            if art == "klein":
                s = fp16_runden(s)
                by, macs = kb.bytes_fp16_linear(s), mlp_macs([h, h])
            else:
                einfrieren(s)
                by, macs = bytes_q(s), mlp_macs([512, 512])
            a = acc(s, xva, yva)
            out["arme"][lname][name] = {"acc": a, "bytes": by, "macs_je_bild": macs, "trainings_macs": 3 * macs * len(xtr) * ep, "epochen": ep, "lr": lr, "sekunden": time.time() - t}
            print(f"  {lname:5s} {name:13s} acc {a:.4f}  Bytes {by:.0f}  ({time.time() - t:.0f}s)", flush=True)

    gewinne, gegner = {}, {}
    for lname in laengen:
        A = out["arme"][lname]
        g = {k: A[k]["acc"] for k in ("rot_hart", "klein_kd", "klein_hart")}
        gegner[lname] = max(g, key=g.get)
        gewinne[lname] = 100 * (A["rot_kd"]["acc"] - g[gegner[lname]])
    out["fiesester_gegner"] = gegner
    out["gewinn_pp_je_laenge"] = gewinne
    out["ampel"] = ampel(min(gewinne.values()), 0.5, "pp", "rot_kd gegen max(rot_hart, klein_kd, klein_hart), Minimum über beide Längen")
    out["zusatz_pp"] = {l: {"kd_hilft_rot (rot_kd - rot_hart)": 100 * (out["arme"][l]["rot_kd"]["acc"] - out["arme"][l]["rot_hart"]["acc"]),
                            "rotation_im_qat (rot_kd - roh_kd)": 100 * (out["arme"][l]["rot_kd"]["acc"] - out["arme"][l]["roh_kd"]["acc"]),
                            "kd_hilft_klein (klein_kd - klein_hart)": 100 * (out["arme"][l]["klein_kd"]["acc"] - out["arme"][l]["klein_hart"]["acc"])} for l in laengen}
    print("Ampel", out["ampel"]["farbe"], out["gewinn_pp_je_laenge"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z1_destill_rot", main)
