"""Z2_rauschformung -- Idee F1 aus Z2_ideen (Audio: Sigma-Delta-Rauschformung): Nach der Rotation den Rundungsfehler spaltenweise mit der Eingangskovarianz auf die noch offenen Spalten zurückführen (GPTQ/OBQ-artig).
Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH. Ehrlich: Das IST GPTQ [G]; Wert = ersetzt die schwache Rundungs-Baseline des Archivs und prüft den Gegner "klein fp16 gleicher Bytes".
FRAGE: Schlägt "Rotation + Fehlerrückführung" bei echten 4 Stufen (2 Bit; zusätzlich 3 Bit = 8 Stufen) das klein-fp16-Netz gleicher Bytes bei beiden Trainingslängen, und wie viel bringt die Rückführung gegenüber einfachem Runden?
ARME (je Modell, je Bitbreite, je Trainingslänge): runden_roh, runden_rot (Z1-Stand), gptq_roh (Kontrolle: GPTQ ohne Rotation), gptq_rot (F1), klein (fp16, gelernt, Bytes <= Bytes des rotierten Modells; Breite wie Z1_ptq_laenge).
  "rot" = Vorzeichen*Hadamard auf dem Eingang (MNIST 784->1024). Zeilenskala bei allen Armen identisch (MSE-Suche vor der Rückführung), Rückführung rundet auf diesem festen Gitter. Die Skalen/Bytes sind für alle quantisierten Arme gleich.
  Kalibrierung: MLP 256 Trainingsbilder, LM 64 Trainingsfenster x 128 Bytes; Eingangskovarianz der fp-Schicht (nicht der schon quantisierten Vorgänger), Dämpfung 1 % der mittleren Diagonale.
MODELLE/LÄNGEN: MLP MNIST 784-512-512-10, Epochen {3, 12} (--schnell {1, 2}); Byte-LM d=128, 4 Lagen, Schritte {600, 2400} (--schnell {40, 160}). Maß: Genauigkeit (MLP) bzw. Val-bpc (LM). Val-Bytes: Z2_VAL (Standard 32768).
VORAB-P (vor dem Lauf): gptq_rot schlägt klein-fp16 bei beiden Längen (MLP 2 Bit) 40 %; gptq_rot besser als runden_rot 85 %; gptq_rot besser als gptq_roh 55 %; LM gptq_rot schlägt klein bei beiden Längen 35 %.
AMPEL (Einzelseed, vorläufig): "ampel" = min über Längen von acc(gptq_rot) − acc(klein), MLP 2 Bit, in pp (Schwelle 0,5); "ampel_lm" = min über Längen von bpc(klein) − bpc(gptq_rot), LM 2 Bit (Schwelle 0,01);
  "ampel_rueckfuehrung" = min über Längen von acc(gptq_rot) − acc(runden_rot), MLP 2 Bit (Schwelle 0,5 pp).
Lies zuerst: tabelle[<modell>][<bit>][<länge>], dann ampel."""
from __future__ import annotations
from _z2 import *
from Z1_ptq_laenge import lm_bauen, lm_bytes_fp16, lm_breite_gleiche_bytes, LM_D


def kalib_eingaenge(netz, fwd):
    """Eingaben aller nn.Linear (in modules()-Reihenfolge) bei einem Kalibrierlauf: Liste von (N x i)-Tensoren."""
    lins = [m for m in netz.modules() if isinstance(m, nn.Linear)]
    buf = [[] for _ in lins]
    hs = [m.register_forward_hook(lambda mod, inp, o, k=k: buf[k].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))) for k, m in enumerate(lins)]
    netz.eval()
    with torch.no_grad():
        fwd(netz)
    for h in hs:
        h.remove()
    return [torch.cat(b) for b in buf]


def gptq_modell(netz, bits, rot, seed, eingaenge):
    q = ersetze_linear(netz, bits, rot, seed)
    for qm, X in zip(qlins(q), eingaenge):
        if qm.pad:
            X = F.pad(X, (0, qm.pad))
        if qm.rot:
            X = rot_vor(X, qm.s)
        Hk = X.T @ X / len(X)
        V = qm.V.detach()
        _, s = quant_gitter(V, bits, skala=True)
        qm.fest = gptq(V, Hk, bits, s)
    q.eval()
    return q


def lm_kalib(seed, schnell):
    tr = ss.daten(schnell)[0]
    g = torch.Generator().manual_seed(seed + 77)
    i = torch.randint(0, len(tr) - KONTEXT - 1, (64,), generator=g)
    x = tr[i[:, None] + torch.arange(KONTEXT)].long()
    return lambda n: n(x)


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    eps = (1, 2) if schnell else (3, 12)
    Ts = (40, 160) if schnell else (600, 2400)
    xtr, ytr, xva, yva = mlp_daten(schnell)
    out = {"mlp_epochen": eps, "lm_schritte": Ts, "val_bytes": VAL_N, "tabelle": {"mlp": {}, "lm": {}}, "kosten": {}}
    arme = lambda n, bits, ein, mess: {
        "runden_roh": mess(einfrieren(ersetze_linear(n, bits, False, seed))), "runden_rot": mess(einfrieren(ersetze_linear(n, bits, True, seed))),
        "gptq_roh": mess(gptq_modell(n, bits, False, seed, ein)), "gptq_rot": mess(gptq_modell(n, bits, True, seed, ein))}
    # ---------------------------------------------------------------- MLP
    mess = lambda q: acc(q, xva, yva)
    for ep in eps:
        t = time.time()
        torch.manual_seed(seed)
        n = kb.baue_bild([512, 512])
        mlp_train(n, xtr, ytr, seed, ep, 1e-3)
        ein = kalib_eingaenge(n, lambda m: m(xtr[:256]))
        a0 = acc(n, xva, yva)
        for bits in (2, 3):
            tab = arme(n, bits, ein, mess)
            tab["fp32"] = a0
            by = bytes_q(einfrieren(ersetze_linear(n, bits, True, seed)))
            h = mlp_gleiche_bytes_breite(by)
            torch.manual_seed(seed + 1)
            kl = kb.baue_bild([h, h])
            mlp_train(kl, xtr, ytr, seed, ep, 1e-3)
            tab["klein"], tab["klein_breite"], tab["bytes_rot"], tab["bytes_klein"] = acc(fp16_runden(kl), xva, yva), h, by, kb.bytes_fp16_linear(kl)
            out["tabelle"]["mlp"].setdefault(f"{bits}bit", {})[ep] = tab
        d = out["tabelle"]["mlp"]
        print(f"Z2_rauschformung s{seed} MLP ep{ep}: fp32 {a0:.4f} | 2bit roh/rot Runden {d['2bit'][ep]['runden_roh']:.4f}/{d['2bit'][ep]['runden_rot']:.4f}  GPTQ roh/rot {d['2bit'][ep]['gptq_roh']:.4f}/{d['2bit'][ep]['gptq_rot']:.4f}  klein {d['2bit'][ep]['klein']:.4f} (h={d['2bit'][ep]['klein_breite']}) ({time.time() - t:.0f}s)", flush=True)
    # ---------------------------------------------------------------- LM
    cache_klein = {}
    for T in Ts:
        t = time.time()
        torch.manual_seed(seed)
        n = lm_bauen(LM_D)
        lm_trainiere(n, seed, T, schnell=schnell)
        n.eval()
        v0 = val_bpc(n, schnell)
        ein = kalib_eingaenge(n, lm_kalib(seed, schnell))
        mess = lambda q: val_bpc(q, schnell)
        rest = kb.bytes_rest_fp16(n)
        for bits in (2, 3):
            tab = arme(n, bits, ein, mess)
            tab["fp32"] = v0
            by = bytes_q(einfrieren(ersetze_linear(n, bits, True, seed))) + rest
            d = lm_breite_gleiche_bytes(by)
            if (d, T) not in cache_klein:
                torch.manual_seed(seed + 1)
                kl = lm_bauen(d)
                lm_trainiere(kl, seed, T, schnell=schnell)
                cache_klein[(d, T)] = val_bpc(fp16_runden(kl), schnell)
            tab["klein"], tab["klein_d"], tab["bytes_rot"], tab["bytes_klein"] = cache_klein[(d, T)], d, by, lm_bytes_fp16(d)
            out["tabelle"]["lm"].setdefault(f"{bits}bit", {})[T] = tab
        d = out["tabelle"]["lm"]["2bit"][T]
        print(f"Z2_rauschformung s{seed} LM T{T}: fp32 {v0:.4f} | 2bit Runden roh/rot {d['runden_roh']:.4f}/{d['runden_rot']:.4f}  GPTQ roh/rot {d['gptq_roh']:.4f}/{d['gptq_rot']:.4f}  klein {d['klein']:.4f} (d={d['klein_d']}) ({time.time() - t:.0f}s)", flush=True)
    # ---------------------------------------------------------------- Ampeln
    m2, l2 = out["tabelle"]["mlp"]["2bit"], out["tabelle"]["lm"]["2bit"]
    out["ampel"] = ampel(min(100 * (m2[e]["gptq_rot"] - m2[e]["klein"]) for e in eps), 0.5, "pp", "gptq_rot − klein fp16 gleicher Bytes, MLP 2 Bit, Minimum über Längen")
    out["ampel_lm"] = ampel(min(l2[T]["klein"] - l2[T]["gptq_rot"] for T in Ts), 0.01, "bpc", "bpc(klein fp16) − bpc(gptq_rot), LM 2 Bit, Minimum über Längen")
    out["ampel_rueckfuehrung"] = ampel(min(100 * (m2[e]["gptq_rot"] - m2[e]["runden_rot"]) for e in eps), 0.5, "pp", "gptq_rot − runden_rot, MLP 2 Bit, Minimum über Längen")
    print("Ampel Gegner", out["ampel"]["farbe"], "LM", out["ampel_lm"]["farbe"], "Rückführung", out["ampel_rueckfuehrung"]["farbe"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z2_rauschformung", main)
