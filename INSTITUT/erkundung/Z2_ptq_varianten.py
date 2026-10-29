"""Z2_ptq_varianten -- Trägt "Rotation flacht den PTQ-Schaden über die Trainingslänge ab" auch bei echten 4+ Stufen, im KV-Cache und in der FFN-Mitte? (Ideen M5, M1, M2, M4 aus Z2_ideen; Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
STUFEN-BEFUND (geprüft an Z1_ptq_laenge.py, LM-Teil): "2 Bit" = quant_gitter(bits=2) = 2^2 = 4 ECHTE Stufen (Mittelpunktsgitter ±0,5 s, ±1,5 s), nicht ternär; 3 Bit = 8 Stufen. Ternär kommt dort nicht vor.
  Der Vorbehalt "ternär" (I-048) betrifft andere Läufe; hier wird ternär (L=3) trotzdem als Arm mitgeführt, damit die Stufenachse 3/4/8/16 vollständig ist.
FRAGE: Schaden(T) = bpc(quantisiert) − bpc(fp32) eines Byte-LM (d=128, 4 Lagen, FFN 4d, Kontext 128, lr 2e-3) bei 4 Trainingslängen T ∈ {300, 600, 1200, 2400} (--schnell: {20,40,80,160}).
  Wächst er mit Rotation langsamer als ohne? Steigung je Verdopplung (Kleinste Quadrate über log2 T), Potenzfit a·T^b, Vorhersage für 2·T_max (= 4800) zur späteren Prüfung (Vorab-Eintrag, nicht gerechnet).
ZIELE (je ein Arm-Block): "gewichte" (alle nn.Linear: qkv, proj, FFN a/b; Einbettung/Position/LayerNorm fp), "kv" (K und V je Kopf, Länge 64, je Token), "ffn_gelu" (Aktivierung nach GELU vor der Down-Projektion, je Token über 512),
  "ffn_z2" (dasselbe in einem eigens trainierten Netz mit z²-Aktivierung).
STUFEN L ∈ {3 (ternär), 4 (2 Bit), 8 (3 Bit), 16 (4 Bit)}. TRANSFORM: "roh", "zufall" (Haar-Orthogonal, dichte Matrix), "hadamard" (Vorzeichen*Hadamard wie Z1). Quantisierer überall gleich: symmetrisches Gleichgitter, Skala je Zeile (Gewicht: je Ausgang; Aktivierung: je Token/Kopf) per MSE-Suche
  = schließt Clipping ein (Kontrolle für "Perzentil-Clipping hilft genauso", M2). Zusatzkontrolle nur Ziel kv: "roh_kanal" (KIVI-artig: K Skala je Kanal über das Fenster, V je Token; K-Skala sieht das ganze 128er-Fenster -> leicht optimistisch).
  Nicht untersucht: gemeinsame Fehler mehrerer Ziele, echte Cache-Bytes (Skalen), Kernel-Zeit. Val-Bytes: Z2_VAL (Standard 32768).
KONTROLLE Norm (M4): Gewichts-RMS je Länge im Feld "gewichts_rms"; Schaden relativ zur Norm wird nicht gerechnet, nur Potenzfit je Arm (fit_a, fit_b) und die Differenz der Exponenten.
VORAB-P (vor dem Lauf): M5 60 % (Effekt bleibt bei L=4), 35 % (bleibt bei L=8); M1 55 %; M2 55 % (gelu), z² 45 %; M4 b_rot < b_roh 60 %.
AMPEL (Einzelseed, vorläufig): je Ziel bei L=4 (2 Bit): gewinn = steigung_roh − steigung_hadamard (bpc je Verdopplung), Schwelle 0,005 (wie Z1_ptq_laenge). Weitere L und "zufall" stehen in der Auswertung.
Lies zuerst: auswertung[ziel][L][transform].steigung_je_verdopplung, dann ampel_*."""
from __future__ import annotations
from _z2 import *

STUFEN = (3, 4, 8, 16)


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    Ts = (20, 40, 80, 160) if schnell else (300, 600, 1200, 2400)
    out = {"Ts": Ts, "stufen": STUFEN, "val_bytes": VAL_N, "bpc_fp32": {"gelu": {}, "z2": {}}, "instabil": {}, "gewichts_rms": {},
           "schaden": {"gewichte": {}, "kv": {}, "ffn_gelu": {}, "ffn_z2": {}}}
    for T in Ts:
        for akt in ("gelu", "z2"):
            t = time.time()
            torch.manual_seed(seed)
            n = lm_q_bauen(128, akt)
            r = lm_trainiere(n, seed, T, schnell=schnell)
            n.eval()
            v0 = val_bpc(n, schnell)
            out["bpc_fp32"][akt][T], out["instabil"][f"{akt}_{T}"] = v0, r["instabil"]
            out["gewichts_rms"][f"{akt}_{T}"] = float(torch.sqrt(torch.stack([(m.weight ** 2).mean() for m in n.modules() if isinstance(m, nn.Linear)]).mean()))
            if r["instabil"]:
                print(f"Z2_ptq_varianten s{seed} {akt} T{T}: INSTABIL, übersprungen", flush=True)
                continue
            for L in STUFEN:
                for art in ROT_ARTEN:
                    if akt == "gelu":
                        q = gewichte_quant(n, L, art, seed)
                        out["schaden"]["gewichte"].setdefault(L, {}).setdefault(art, {})[T] = val_bpc(q, schnell) - v0
                        Q = rot_q(64, art, seed + 11)
                        haken_setzen(n, kv={"L": L, "Q": Q})
                        out["schaden"]["kv"].setdefault(L, {}).setdefault(art, {})[T] = val_bpc(n, schnell) - v0
                    Qf = rot_q(512, art, seed + 13)
                    haken_setzen(n, mitte={"L": L, "Q": Qf})
                    out["schaden"][f"ffn_{akt}"].setdefault(L, {}).setdefault(art, {})[T] = val_bpc(n, schnell) - v0
                    haken_setzen(n)
                if akt == "gelu":
                    haken_setzen(n, kv={"L": L, "Q": None, "kanal": True})
                    out["schaden"]["kv"][L].setdefault("roh_kanal", {})[T] = val_bpc(n, schnell) - v0
                    haken_setzen(n)
            print(f"Z2_ptq_varianten s{seed} {akt} T{T}: fp32 {v0:.4f} bpc ({time.time() - t:.0f}s)", flush=True)
            if akt == "gelu":
                print("   L=4 Schaden gewichte roh/zuf/had {:.4f}/{:.4f}/{:.4f}  kv {:.4f}/{:.4f}/{:.4f}  ffn {:.4f}/{:.4f}/{:.4f}".format(
                    *[out["schaden"][z][4][a][T] for z, a in (("gewichte", "roh"), ("gewichte", "zufall"), ("gewichte", "hadamard"), ("kv", "roh"), ("kv", "zufall"), ("kv", "hadamard"),
                                                           ("ffn_gelu", "roh"), ("ffn_gelu", "zufall"), ("ffn_gelu", "hadamard"))]), flush=True)
    # ---------------------------------------------------------------- Auswertung
    ausw = {}
    for z, d1 in out["schaden"].items():
        ausw[z] = {}
        Tz = [T for T in Ts if not out["instabil"].get(f"{'z2' if z == 'ffn_z2' else 'gelu'}_{T}")]
        for L, d2 in d1.items():
            ausw[z][L] = {art: zeile_auswerten(Tz, [d3[T] for T in Tz]) for art, d3 in d2.items() if all(T in d3 for T in Tz)}
            if "roh" in ausw[z][L]:
                for art in ("zufall", "hadamard", "roh_kanal"):
                    if art in ausw[z][L]:
                        ausw[z][L][art]["steigung_roh_minus_diese"] = ausw[z][L]["roh"]["steigung_je_verdopplung"] - ausw[z][L][art]["steigung_je_verdopplung"]
                        ausw[z][L][art]["exponent_b_roh_minus_diese"] = ausw[z][L]["roh"]["fit_b"] - ausw[z][L][art]["fit_b"]
    out["auswertung"] = ausw
    for z in ausw:
        a = ausw[z].get(4, {}).get("hadamard", {})
        out[f"ampel_{z}"] = ampel(a.get("steigung_roh_minus_diese", float("nan")) if a else float("nan"), 0.005, "bpc je Verdopplung", f"Steigung Schaden roh − hadamard, {z}, L=4 (2 Bit)")
    print("Ampel " + "  ".join(f"{z} {out['ampel_' + z]['farbe']}" for z in ausw), flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z2_ptq_varianten", main)
