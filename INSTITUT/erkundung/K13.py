"""K13 Grob-zu-fein-Curriculum (Kontextlänge) -- ERKUNDUNG, kein Befund.
Frage: Gibt ein Kontext-Curriculum (kurz -> lang) bei FESTEM Trainings-FLOP-Budget bessere Byte-LM-bpc als fester Kontext?
Aufbau: Byte-LM d=128, 4 Lagen, 4 Köpfe (_lm.py), Tokens/Schritt konstant 1024 (Batch = 1024/Kontext), Lernrate fest (Kosinus über FLOP-Anteil).
Budget B = Trainings-FLOPs von T Schritten bei festem Kontext 128 (T=4000), gezählt je Schritt mit flops_train_token(d, ctx) (3x Vorwärts, Zählung wie I-045-v4/v5).
Jeder Arm läuft, bis seine kumulierten FLOPs B erreichen (Arme mit kürzerem Kontext machen daher etwas mehr Schritte).
ARME: f32, f64, f128, f256 (fester Kontext = Kontrollen), W64_256 (64 -> 128 -> 256 zu je 1/3 der FLOPs), S32_256 (32 -> 64 -> 128 -> 256 zu je 1/4).
MASS: Val-bpc (dev, 65536 Bytes) bei Auswertungs-Kontext 32/64/128/256; Kopfzahl = 128 (und 256). Param, Schritte, FLOPs ausgewiesen.
Kontrolle: fester Kontext 32/64 (gleiche FLOPs) und fest 128/256. Das Curriculum zählt nur, wenn es den BESTEN festen Arm schlägt.
Aufruf: python3 erkundung/K13.py --seed 950 [--schnell]   (aus INSTITUT/)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lm import *

D, LR, T_VOLL = 128, 1e-3, 4000
ARME = {"f32": [(32, 1.0)], "f64": [(64, 1.0)], "f128": [(128, 1.0)], "f256": [(256, 1.0)],
        "W64_256": [(64, 1 / 3), (128, 1 / 3), (256, 1 / 3)],
        "S32_256": [(32, .25), (64, .25), (128, .25), (256, .25)]}


def plan(stufen, budget, d=D):
    """Liste (ctx, lr-Anteil) je Schritt; Stufen verbrauchen je ihren FLOP-Anteil, bis das Budget erreicht ist."""
    ctxe, fracs, cum = [], [], 0.0
    for c, anteil in stufen:
        ende = cum + anteil * budget
        fl = flops_train_token(d, 4, c) * 1024
        while cum < ende - 1e-6:
            ctxe.append(c)
            fracs.append(cum / budget)
            cum += fl
    return ctxe, fracs


def main(seed: int, schnell: bool = False) -> dict:
    T = 60 if schnell else T_VOLL
    tr, va, te = daten(klein=schnell)
    if schnell:
        va = va[:8192]
    budget = T * flops_train_token(D, 4, 128) * 1024
    out = {"id": "K13", "seed": seed, "schnell": schnell, "lr": LR, "d": D, "T_voll": T, "budget_flops": budget, "arme": {}}
    for arm, stufen in ARME.items():
        t0 = time.time()
        ctxe, fracs = plan(stufen, budget)
        netz = LM(D, seed=seed)
        opt = torch.optim.Adam(netz.parameters(), LR)
        g = torch.Generator().manual_seed(seed + 5)
        r = lauf_schritte(netz, opt, tr, g, ctxe, LR, fracs)
        fl = sum(flops_train_token(D, 4, c) * 1024 for c in ctxe)
        e = {"schritte": len(ctxe), "flops": fl, "param": n_param(netz), "param_ohne_emb": n_param(netz, True), "instabil": r["instabil"],
             "train_verlust_bpc": r["train_verlust"], "sek": time.time() - t0}
        for c in (32, 64, 128, 256):
            e[f"val_bpc_ctx{c}"] = None if r["instabil"] else bpc(netz, va, ctx=c)
        out["arme"][arm] = e
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
    fest = {a: out["arme"][a]["val_bpc_ctx128"] for a in ("f32", "f64", "f128", "f256") if out["arme"][a]["val_bpc_ctx128"] is not None}
    bestfest = min(fest, key=fest.get) if fest else None
    out["bester_fester_arm"] = bestfest
    out["vergleich_ctx128"] = {a: (out["arme"][a]["val_bpc_ctx128"] - fest[bestfest]) if bestfest and out["arme"][a]["val_bpc_ctx128"] is not None else None
                               for a in ("W64_256", "S32_256")}
    out["lesart"] = "Differenz = Curriculum minus bester fester Arm bei gleichen FLOPs (negativ = Curriculum besser). Ein Seed, Tuning-Seed, Erkundung."
    return out


if __name__ == "__main__":
    cli("K13", main, __doc__)
