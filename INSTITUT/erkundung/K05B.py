"""K05B -- Zusatzarme zu K05: KIVI-Arm und Rotation+Restfenster (Erkundung Welle 3, 28.10.2026; kein Befund, nicht in LABORBUCH).
K05.py bleibt unverändert; Bausteine (lm_fwd, att_klassen, quant_vek, Arm, bits_je_wert, masken, bewerte, arm_quant, arm_fp) werden importiert.
Gleiches Netz (kb.LM d=128, 4 Lagen, Kopfdim 64, normal trainiert: lr 2e-3, T=4000, schnell klein/250), gleiche Test-Auswertung, gleiche Bit-Zählung (mittlere Cache-Bits je Wert über alle Positionen, fp16 = 16).
NEU gegenüber K05 (laut f2-Bericht fehlen KIVI und Fenster+Rotation):
  kivi_b{b}_G{G}  KIVI (Liu et al. 2024): Schlüssel je KANAL (min/max über Gruppen von G Token), Werte je TOKEN (min/max über die 64 Kanäle des Kopfes), asymmetrisch gleichförmig b Bit,
                  Skala+Nullpunkt fp16 (32 Bit je Gruppe). Restfenster: Token der aktuellen, noch nicht vollen Gruppe (j//G == i//G) bleiben fp16 (KIVI-Residual R = G).
                  Bits je Wert: Schlüssel b+32/G, Werte b+32/64, gemittelt über K und V; plus fp16-Anteil laut Maske. Alle b in 1..4, G in {4,8,16,32,64} werden gerechnet.
                  Bei Kontext 128 dominiert das fp16-Fenster: die Bits je Wert liegen bei >= 4 (siehe bits_je_wert). Deshalb zusätzlich kivi_ohneFenster_LECK_*: kein Fenster, aber Gruppenstatistik schaut in die Zukunft der Gruppe (optimistisch, nur Bit-Buchhaltung).
  hadfenster_b{b}@z  Rotation (Zufalls-Hadamard 64, MSE-Skala je Vektor wie Arm `had`) mit b Bit + fp16-Gleitfenster der letzten R Token; R so, dass die Bits je Wert dem Ziel z am nächsten kommen (b+0,25 < z-0,05).
ANKER aus K05 (gleiche Funktionen): fp, had@z, roh@z für z = 1,25 / 2,25 / 3,25.
FAZIT je Ziel: nächstliegende KIVI-Konfiguration (Abstand der Bits je Wert zu z) und beste hadfenster-Konfiguration, jeweils minus fp und minus had. Wichtig: die Bits je Wert der KIVI-Konfigurationen liegen NICHT genau auf dem Ziel (`bits_je_wert` je Arm mitlesen).
Aufruf: python3 erkundung/K05B.py --seed 950 [--schnell]"""
import time

import _hilfe_k0508 as H
import torch
import K05 as K

kb, ss = H.kb, H.ss
ZIELE = K.ZIELE
KIVI_B, KIVI_G = (1, 2, 3, 4), (4, 8, 16, 32, 64)


def _rund16(x):
    return x.half().float()


def asym(x, dim, bits):
    """Gleichförmige asymmetrische Quantisierung, min/max über `dim` (keepdim), Skala und Nullpunkt in fp16."""
    lo = _rund16(x.amin(dim, keepdim=True))
    sk = _rund16(((x.amax(dim, keepdim=True) - x.amin(dim, keepdim=True)) / (2 ** bits - 1)).clamp_min(1e-8))
    return ((x - lo) / sk).round().clamp(0, 2 ** bits - 1) * sk + lo


def kivi_k(x, b, G):
    B, Hh, T, d = x.shape
    return asym(x.reshape(B, Hh, T // G, G, d), 3, b).reshape(B, Hh, T, d)


def kivi_v(x, b):
    return asym(x, 3, b)


def arm_kivi(T, b, G, dh=64):
    i, j, c = K.masken(T)
    fp = c & ((j // G) == (i // G))
    q = c & ~fp
    kos = [(fp, 16.0), (q, b + (32.0 / G + 32.0 / dh) / 2)]
    return K.Arm("kivi", lambda l, k, v: [(kivi_k(k, b, G), kivi_v(v, b), q), (k, v, fp)], kos, {"b": b, "G": G})


def arm_kivi_leck(T, b, G, dh=64):
    """KIVI ohne fp16-Fenster: auch die laufende Gruppe ist quantisiert, ihre Min/Max-Statistik enthält aber KÜNFTIGE Token der Gruppe (Leck, optimistisch!).
    Nur zur Bit-Buchhaltung bei kurzem Kontext; kein zulässiger Arm für Aussagen."""
    _, _, c = K.masken(T)
    return K.Arm("kivi_leck", lambda l, k, v: [(kivi_k(k, b, G), kivi_v(v, b), c)], [(c, b + (32.0 / G + 32.0 / dh) / 2)], {"b": b, "G": G, "leck": True})


def arm_hadfenster(T, b, R, seed, dh=64):
    i, j, c = K.masken(T)
    fp = c & ((i - j) < R)
    q = c & ~fp

    def fn(l, k, v):
        Qk, Qv = kb.rot_matrix(dh, seed * 977 + l * 2), kb.rot_matrix(dh, seed * 977 + l * 2 + 1)
        return [(K.quant_vek(k, b, Qk), K.quant_vek(v, b, Qv), q), (k, v, fp)]

    return K.Arm("hadfenster", fn, [(fp, 16.0), (q, b + 16.0 / dh)], {"b": b, "R": R})


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    klein = schnell
    T = K.T_SCHNELL if schnell else K.T_VOLL
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    netz = kb.LM(128, 4)
    inf = kb.trainiere_lm(netz, seed, K.LR, T, klein=klein)
    t_train = time.time() - t0
    _, _, te = ss.daten(klein)
    ctx = ss.KONTEXT
    bpc_ref = ss.bpc_auswerten(netz, te)
    arme = {"fp": K.arm_fp(ctx)}
    for z in ZIELE:
        b = int(z)
        arme[f"had@{z}"] = K.arm_quant("had", ctx, b, True, False, None, seed)
        arme[f"roh@{z}"] = K.arm_quant("roh", ctx, b, False, False, None, seed)
        for bb in (1, 2, 3):
            if bb + 0.25 < z - 0.05:
                R = K.passe(ctx, z, lambda r_, bb=bb: arm_hadfenster(ctx, bb, r_, seed).kosten, range(1, ctx))
                arme[f"hadfenster_b{bb}@{z}"] = arm_hadfenster(ctx, bb, R, seed)
    for b in KIVI_B:
        for G in KIVI_G:
            arme[f"kivi_b{b}_G{G}"] = arm_kivi(ctx, b, G)
    for b in (1, 2, 3):
        for G in (32, 64):
            arme[f"kivi_ohneFenster_LECK_b{b}_G{G}"] = arm_kivi_leck(ctx, b, G)
    zeilen = {}
    for name, arm in arme.items():
        zeilen[name] = {"bpc": K.bewerte(netz, te, arm), "bits_je_wert": K.bits_je_wert(arm.kosten, ctx), "par": arm.info}
    fp = zeilen["fp"]["bpc"]
    fazit = {}
    for z in ZIELE:
        had = zeilen[f"had@{z}"]["bpc"]
        kv = min((n for n in zeilen if n.startswith("kivi") and "LECK" not in n), key=lambda n: abs(zeilen[n]["bits_je_wert"] - z))
        hf = [n for n in zeilen if n.startswith("hadfenster") and n.endswith(f"@{z}")]
        f_ = {"had_minus_fp": had - fp, "kivi_naechste": kv, "kivi_bits_je_wert": zeilen[kv]["bits_je_wert"], "kivi_minus_fp": zeilen[kv]["bpc"] - fp, "kivi_minus_had": zeilen[kv]["bpc"] - had}
        lk = min((n for n in zeilen if "LECK" in n), key=lambda n: abs(zeilen[n]["bits_je_wert"] - z))
        f_.update({"kivi_leck_naechste": lk, "kivi_leck_bits_je_wert": zeilen[lk]["bits_je_wert"], "kivi_leck_minus_fp": zeilen[lk]["bpc"] - fp})
        if hf:
            bh = min(hf, key=lambda n: zeilen[n]["bpc"])
            f_.update({"hadfenster_beste": bh, "hadfenster_minus_fp": zeilen[bh]["bpc"] - fp, "hadfenster_minus_had": zeilen[bh]["bpc"] - had})
        fazit[str(z)] = f_
    r = {"kid": "K05B", "seed": seed, "schnell": schnell, "T": T, "lr": K.LR, "klein": klein,
         "modell": {"d": 128, "lagen": 4, "kopfdim": 64, "parameter": H.n_param(netz), "kv_fp16_bytes_je_token_lage": 2 * 128 * 2},
         "flops_hinweis": "Attention- und Linear-FLOPs unveraendert; Rotation 64·log2(64)=384 Additionen je KV-Vektor (had, hadfenster); KIVI: Min/Max + Rundung je Vektor/Gruppe; Cache-Bytes siehe bits_je_wert (16 = fp16)",
         "bpc_ref_eigene_auswertung": bpc_ref, "pruefung_fp_gleich": abs(fp - bpc_ref), "train_info": inf, "arme": zeilen, "fazit": fazit,
         "sekunden_training": t_train, "sekunden_auswertung": time.time() - t0 - t_train}
    r["zusammenfassung"] = "\n".join(f"Ziel {z}: had {fazit[str(z)]['had_minus_fp']:+.3f}, KIVI ({fazit[str(z)]['kivi_naechste']}, {fazit[str(z)]['kivi_bits_je_wert']:.2f} Bit) {fazit[str(z)]['kivi_minus_fp']:+.3f}"
                                     + (f", hadfenster ({fazit[str(z)]['hadfenster_beste']}) {fazit[str(z)]['hadfenster_minus_fp']:+.3f}" if "hadfenster_beste" in fazit[str(z)] else "") for z in ZIELE)
    return r


if __name__ == "__main__":
    H.cli("K05B", main)
