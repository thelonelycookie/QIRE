"""Z1_ptq_laenge -- Flacht die Rotation den PTQ-Schaden über die Trainingslänge ab? (Idee 5 aus Z1_grenzen; Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: Je länger ein Netz trainiert wird, desto stärker schadet Nachher-Quantisierung (PTQ). Wächst der Schaden mit Rotation (Vorzeichen*Hadamard) langsamer mit der Trainingslänge als ohne?
  Oder schrumpft nur der Rotationsvorteil (Frühregime)? Das entscheidet, ob der R1-Befund (Rotation + 2 Bit schlägt klein fp16) bei langem Training trägt.
VORAB-P (vor dem Lauf): 40 % grün. Teil: Schaden wächst mit Länge ohne Rotation schneller als mit (Steigung roh > rot) 60 %; Rotation hilft bei der längsten Länge noch >= Schwelle 50 %; MLP und LM gleiche Richtung 55 %.
MODELLE (je 3 Trainingslängen, Verhältnis 1 : 4 : 16):
  MLP   MNIST 784-512-512-10, Epochen {3, 12, 48} (--schnell {1, 2, 4}), Adam 1e-3, Kosinus. Maß: Schaden = Val-Genauigkeit fp32 minus quantisiert, in Prozentpunkten.
  LM    Byte-LM (UD_German-GSD) d=128, 4 Lagen, FFN 4d, Kontext 128, Schritte {300, 1200, 4800} (--schnell {20, 40, 80}), lr 2e-3. Maß: Schaden = Val-bpc quantisiert minus fp32.
QUANTISIERUNG (PTQ, kein Nachtraining): alle nn.Linear (MLP: alle drei; LM: qkv, proj, FFN a/b; Einbettung/Position/LayerNorm bleiben fp16, für alle Arme gleich), echte 2^b Stufen (b = 2, 3),
  Zeilenskala per MSE-Suche, "rot" = Vorzeichen*Hadamard auf dem Eingang (MNIST-Eingang 784->1024), "roh" = ohne Drehung.
GEGENSPIELER (gleiche Bytes, GELERNT, gleiche Länge): MLP fp16 mit Breite h-h, fp16-Bytes <= Bytes des rotierten Modells (je Bitbreite); LM fp16 mit kleinerem d (Vielfaches von 16), Bytes einschl. Einbettung <= rotiertes Modell.
AUSGABE: schaden[<modell>][<b>bit][<roh|rot>][<länge>], rotationsgewinn = Schaden roh − Schaden rot je Länge, steigung_je_verdopplung (Kleinste Quadrate über log2 der Länge) für roh, rot und Differenz,
  vs_klein = Qualität(rot-quantisiert) − Qualität(klein fp16) je Länge. Bytes, Parameter und MACs im Feld "kosten".
AMPEL (Einzelseed, vorläufig): "ampel" = steigung_roh − steigung_rot beim 2-Bit-MLP (Schwelle 0,2 pp je Verdopplung; positiv = Rotation flacht ab); "ampel_lm" dasselbe in bpc (Schwelle 0,005);
  "ampel_gegner" = min über Längen von vs_klein beim 2-Bit-MLP (Schwelle 0,5 pp; positiv = rotiertes Großmodell schlägt klein-fp16 gleicher Bytes auch bei der längsten Länge).
Lies zuerst: rotationsgewinn je Länge (wächst oder schrumpft?), dann ampel_gegner."""
from __future__ import annotations
from _z1 import *

LM_D = 128


def lm_bauen(d):
    return LMX([FFNDichtX(d, 4 * d) for _ in range(4)], d=d)


def lm_bytes_fp16(d):
    n = lm_bauen(d)
    return kb.bytes_fp16_linear(n) + kb.bytes_rest_fp16(n)


def lm_breite_gleiche_bytes(ziel):
    best = 16
    for d in range(16, LM_D + 1, 16):
        if lm_bytes_fp16(d) <= ziel:
            best = d
    return best


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 2, 4) if schnell else (3, 12, 48)
    Ts = (20, 40, 80) if schnell else (300, 1200, 4800)
    xtr, ytr, xva, yva = mlp_daten(schnell)
    out = {"mlp_epochen": eps, "lm_schritte": Ts, "kosten": {}, "roh": {"mlp": {}, "lm": {}}, "schaden": {"mlp": {}, "lm": {}}, "klein": {"mlp": {}, "lm": {}}}
    # ---------------------------------------------------------------- MLP
    for ep in eps:
        t = time.time()
        torch.manual_seed(seed)
        n = kb.baue_bild([512, 512])
        mlp_train(n, xtr, ytr, seed, ep, 1e-3)
        a0 = acc(n, xva, yva)
        out["roh"]["mlp"][ep] = {"acc_fp32": a0}
        for b in (2, 3):
            qs = {m: einfrieren(ersetze_linear(n, b, m == "rot", seed)) for m in ("roh", "rot")}
            for m, q in qs.items():
                out["schaden"]["mlp"].setdefault(f"{b}bit", {}).setdefault(m, {})[ep] = 100 * (a0 - acc(q, xva, yva))
            by = bytes_q(qs["rot"])
            h = mlp_gleiche_bytes_breite(by)
            torch.manual_seed(seed + 1)
            kl = kb.baue_bild([h, h])
            mlp_train(kl, xtr, ytr, seed, ep, 1e-3)
            kl = fp16_runden(kl)
            out["klein"]["mlp"].setdefault(f"{b}bit", {})[ep] = {"acc": acc(kl, xva, yva), "acc_rot_quant": acc(qs["rot"], xva, yva), "breite": h, "bytes": kb.bytes_fp16_linear(kl)}
            out["kosten"].setdefault("mlp", {})[f"{b}bit"] = {"bytes_rot": by, "bytes_roh": bytes_q(qs["roh"]), "bytes_fp16_gross": kb.bytes_fp16_linear(n), "macs_gross": mlp_macs([512, 512]), "macs_klein": mlp_macs([h, h])}
        print(f"Z1_ptq_laenge s{seed} MLP ep{ep}: fp32 {a0:.4f}  Schaden 2bit roh/rot {out['schaden']['mlp']['2bit']['roh'][ep]:.2f}/{out['schaden']['mlp']['2bit']['rot'][ep]:.2f} pp  "
              f"3bit {out['schaden']['mlp']['3bit']['roh'][ep]:.2f}/{out['schaden']['mlp']['3bit']['rot'][ep]:.2f}  ({time.time() - t:.0f}s)", flush=True)
    # ---------------------------------------------------------------- LM
    cache_klein = {}
    for T in Ts:
        t = time.time()
        torch.manual_seed(seed)
        n = lm_bauen(LM_D)
        r = lm_trainiere(n, seed, T, schnell=schnell)
        v0 = r["bpc"]
        out["roh"]["lm"][T] = {"bpc_fp32": v0, "instabil": r["instabil"]}
        rest = kb.bytes_rest_fp16(n)
        for b in (2, 3):
            qs = {m: einfrieren(ersetze_linear(n, b, m == "rot", seed)) for m in ("roh", "rot")}
            vq = {m: lm_val(q) for m, q in qs.items()}
            for m in qs:
                out["schaden"]["lm"].setdefault(f"{b}bit", {}).setdefault(m, {})[T] = vq[m] - v0
            by = bytes_q(qs["rot"]) + rest
            d = lm_breite_gleiche_bytes(by)
            if (d, T) not in cache_klein:               # gleiche Breite bei 2 und 3 Bit: nur einmal trainieren
                torch.manual_seed(seed + 1)
                kl = lm_bauen(d)
                lm_trainiere(kl, seed, T, schnell=schnell)
                kl = fp16_runden(kl)
                cache_klein[(d, T)] = (kl, lm_val(kl))
            kl, bk = cache_klein[(d, T)]
            out["klein"]["lm"].setdefault(f"{b}bit", {})[T] = {"bpc": bk, "bpc_rot_quant": vq["rot"], "d": d, "bytes": lm_bytes_fp16(d)}
            out["kosten"].setdefault("lm", {})[f"{b}bit"] = {"bytes_rot": by, "bytes_roh": bytes_q(qs["roh"]) + rest, "bytes_fp16_gross": lm_bytes_fp16(LM_D), "param_gross": zaehle_param(n),
                                                          "param_klein": zaehle_param(kl), "macs_gross": lm_macs(LM_D, 4, 8 * LM_D * LM_D), "macs_klein": lm_macs(d, 4, 8 * d * d)}
        print(f"Z1_ptq_laenge s{seed} LM T{T}: fp32 {v0:.4f} bpc  Schaden 2bit roh/rot {out['schaden']['lm']['2bit']['roh'][T]:.4f}/{out['schaden']['lm']['2bit']['rot'][T]:.4f}  "
              f"3bit {out['schaden']['lm']['3bit']['roh'][T]:.4f}/{out['schaden']['lm']['3bit']['rot'][T]:.4f}  ({time.time() - t:.0f}s)", flush=True)
    # ---------------------------------------------------------------- Auswertung im Skript
    ausw = {}
    for mod, langen in (("mlp", eps), ("lm", Ts)):
        ausw[mod] = {}
        for bk, sd in out["schaden"][mod].items():
            roh = [sd["roh"][L] for L in langen]
            rot = [sd["rot"][L] for L in langen]
            ausw[mod][bk] = {"rotationsgewinn": {L: sd["roh"][L] - sd["rot"][L] for L in langen},
                             "steigung_je_verdopplung": {"roh": steigung(langen, roh), "rot": steigung(langen, rot)}}
            ausw[mod][bk]["steigung_je_verdopplung"]["roh_minus_rot"] = ausw[mod][bk]["steigung_je_verdopplung"]["roh"] - ausw[mod][bk]["steigung_je_verdopplung"]["rot"]
            kl = out["klein"][mod][bk]
            ausw[mod][bk]["vs_klein"] = {L: (100 * (kl[L]["acc_rot_quant"] - kl[L]["acc"]) if mod == "mlp" else kl[L]["bpc"] - kl[L]["bpc_rot_quant"]) for L in langen}
    out["auswertung"] = ausw
    out["ampel"] = ampel(ausw["mlp"]["2bit"]["steigung_je_verdopplung"]["roh_minus_rot"], 0.2, "pp je Verdopplung", "Steigung Schaden roh − rot, MLP 2 Bit")
    out["ampel_lm"] = ampel(ausw["lm"]["2bit"]["steigung_je_verdopplung"]["roh_minus_rot"], 0.005, "bpc je Verdopplung", "Steigung Schaden roh − rot, LM 2 Bit")
    out["ampel_gegner"] = ampel(min(ausw["mlp"]["2bit"]["vs_klein"].values()), 0.5, "pp", "rot-quantisiertes Großmodell − klein fp16 gleicher Bytes, MLP 2 Bit, Minimum über Längen")
    print("Ampel", out["ampel"]["farbe"], "LM", out["ampel_lm"]["farbe"], "Gegner", out["ampel_gegner"]["farbe"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z1_ptq_laenge", main)
