"""Z2_kv_treppe -- Idee L22 aus Z2_ideen (Dekohärenz x KV-Cache, "Kohärenzzeit"): Die Bits je Cache-Eintrag fallen mit dem Alter des Tokens (Treppe), bei gleichem mittlerem Bitbudget.
Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH. Bekannt: Rezenzfenster + niedrige Bits (KIVI), Mixed-Precision-KV (MiKV) [G, vor Zitat prüfen].
FRAGE: Schlägt eine Alters-Treppe bei gleichem mittlerem Bitbudget (<= 2 Bit je K/V-Wert, gemittelt über Alter 0..127) die gleichmäßige 2-Bit-Ablage und die aufmerksamkeitsbasierte Zuteilung (L08)? Alles NACH Rotation (Vorzeichen*Hadamard über den 64er-Kopf).
SETUP: Byte-LM d=128, 4 Lagen, Kontext 128, Trainingslängen {600, 2400} (--schnell {40,160}), gelu-FFN, fp32-Gewichte; nur K und V werden nachträglich quantisiert (PTQ, kein Nachtraining), je Token und Kopf, Skala per MSE-Suche.
  Zuteilung gilt je (Anfrage i, Schlüssel j) -- also mit dem Alter i−j zum Zeitpunkt der Anfrage (ein Cache, der Einträge mit der Zeit herunterrechnet; Näherung: Herunterrechnen geht vom fp-Wert aus, nicht von der Zwischenstufe).
ARME (Bitbreiten 16 = unverändert, sonst 2^b Stufen; Treppen geben (Alters-Obergrenze, Bits); Mittelwert über Alter 0..127 wird im Ergebnis gemeldet):
  gleich_rot / gleich_roh: alle Einträge 2 Bit (Mittel 2,00; etwas MEHR als die Treppen -> Nachteil für die Treppe); roh = ohne Rotation (Rotationskontrolle).
  treppeA_rot / treppeA_roh: Alter 0-7: 8 Bit, 8-15: 4, 16-55: 2, 56-127: 1 Bit (Mittel 1,94).
  treppeB_rot: Alter 0-7: fp16, sonst 1 Bit (Mittel 1,94) = Rezenzfenster mit 1-Bit-Rest.
  fenster_weg: Alter 0-15 fp16, älter gestrichen (Mittel 2,00) = Rezenzfenster allein.
  aufm_rot (L08, Kontrolle): dieselbe Bit-Tabelle wie treppeA, aber nach Rang der bisher (Anfragen < i) erhaltenen Aufmerksamkeit des Schlüssels aus dem fp-Lauf (H2O-artig), die 4 jüngsten Einträge als geschützt (Rang 0-3).
  Nicht untersucht: echte Cache-Bytes (Skalen je Token), Umsetzbarkeit des Herunterrechnens, andere Treppenformen (nur diese zwei vorab festgelegt, kein Gitter über Treppen = keine Auswahl am Val-Satz). Val-Bytes: Z2_VAL (Standard 32768).
VORAB-P (vor dem Lauf): treppeA_rot besser als gleich_rot bei beiden Längen 30 %; treppeA besser als aufm_rot 40 %; treppeB besser als gleich_rot 35 %; Rotation hilft (gleich_rot < gleich_roh) 85 %.
AMPEL (Einzelseed, vorläufig): "ampel" = min über Längen von bpc(gleich_rot) − bpc(treppeA_rot) (Schwelle 0,01 bpc; positiv = Treppe besser); "ampel_gegen_aufm" = min über Längen von bpc(aufm_rot) − bpc(treppeA_rot) (0,01);
  "ampel_B" = min über Längen von bpc(gleich_rot) − bpc(treppeB_rot) (0,01); "ampel_rotation" = min über Längen von bpc(gleich_roh) − bpc(gleich_rot) (0,01).
Lies zuerst: bpc[<länge>][<arm>] und mittel_bits, dann ampel."""
from __future__ import annotations
from _z2 import *

TAB_A = [(8, 8), (16, 4), (56, 2), (128, 1)]
TAB_B = [(8, 16), (128, 1)]
TAB_W = [(16, 16), (128, 0)]
TAB_G = [(128, 2)]


def mittel_bits(tab, n=128):
    s, a0 = 0, 0
    for g, b in tab:
        s += (min(g, n) - a0) * b
        a0 = min(g, n)
    return s / n


def klasse(tab, wert):
    """wert (Alter oder Rang, ganzzahliger Tensor) -> Bits je Eintrag aus der Treppe."""
    b = torch.full_like(wert, tab[-1][1])
    for g, bits in reversed(tab):
        b = torch.where(wert < g, torch.full_like(wert, bits), b)
    return b


def treppe_aufmerksamkeit(tab, Q, modus):
    def f(att, q, k, v):
        B, H, T, hd = q.shape
        i = torch.arange(T)[:, None]
        j = torch.arange(T)[None, :]
        kaus = (j <= i)
        Sfp = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        Sm = Sfp.masked_fill(~kaus, float("-inf"))
        if modus == "alter":
            bits = klasse(tab, (i - j).clamp_min(0)).expand(B, H, T, T)
        else:      # Rang nach bisher erhaltener Aufmerksamkeit (Anfragen < i), jüngste 4 geschützt
            P = torch.softmax(Sm, -1)
            cum = torch.cumsum(P, -2) - P                            # Summe über Anfragen i' < i
            sc = cum + 1e-6 * j.float() / T                         # Gleichstand -> neuere Einträge vorn
            sc = torch.where(i - j < 4, torch.full_like(sc, 1e9), sc).masked_fill(~kaus, float("-inf"))
            rang = sc.argsort(-1, descending=True).argsort(-1)
            bits = klasse(tab, rang)
        S = torch.full_like(Sfp, float("-inf"))
        sel = []
        for b in sorted(set(bt for _, bt in tab)):
            m = (bits == b) & kaus
            if b == 0:
                continue
            L = 65536 if b >= 16 else 2 ** b
            Kb, Vb = quant_rot(k, L, Q), quant_rot(v, L, Q)
            S = torch.where(m, (q @ Kb.transpose(-1, -2)) / math.sqrt(hd), S)
            sel.append((m, Vb))
        P = torch.softmax(S, -1)
        return sum((P * m) @ Vb for m, Vb in sel)
    return f


def arme(seed):
    Qh = rot_q(64, "hadamard", seed + 11)
    return {"gleich_rot": ("kv", 4, Qh, TAB_G), "gleich_roh": ("kv", 4, None, TAB_G), "treppeA_rot": ("t", None, Qh, TAB_A), "treppeA_roh": ("t", None, None, TAB_A),
            "treppeB_rot": ("t", None, Qh, TAB_B), "fenster_weg": ("t", None, Qh, TAB_W), "aufm_rot": ("a", None, Qh, TAB_A)}


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    Ts = (40, 160) if schnell else (600, 2400)
    out = {"Ts": Ts, "val_bytes": VAL_N, "bpc": {}, "mittel_bits": {n: mittel_bits(a[3]) for n, a in arme(seed).items()}}
    for T in Ts:
        t = time.time()
        torch.manual_seed(seed)
        n = lm_q_bauen(128, "gelu")
        r = lm_trainiere(n, seed, T, schnell=schnell)
        n.eval()
        res = {"fp32": val_bpc(n, schnell), "instabil": r["instabil"]}
        for name, (art, L, Q, tab) in arme(seed).items():
            if art == "kv":
                haken_setzen(n, kv={"L": L, "Q": Q})
            else:
                haken_setzen(n)
                f = treppe_aufmerksamkeit(tab, Q, "alter" if art == "t" else "rang")
                for b in n.bl:
                    b.att.treppe = f
            res[name] = val_bpc(n, schnell)
        haken_setzen(n)
        out["bpc"][T] = res
        print(f"Z2_kv_treppe s{seed} T{T}: " + "  ".join(f"{k} {v:.4f}" for k, v in res.items() if k != "instabil") + f"  ({time.time() - t:.0f}s)", flush=True)
    b = out["bpc"]
    out["ampel"] = ampel(min(b[T]["gleich_rot"] - b[T]["treppeA_rot"] for T in Ts), 0.01, "bpc", "bpc(gleich_rot) − bpc(treppeA_rot), min über Längen")
    out["ampel_gegen_aufm"] = ampel(min(b[T]["aufm_rot"] - b[T]["treppeA_rot"] for T in Ts), 0.01, "bpc", "bpc(aufm_rot) − bpc(treppeA_rot), min über Längen")
    out["ampel_B"] = ampel(min(b[T]["gleich_rot"] - b[T]["treppeB_rot"] for T in Ts), 0.01, "bpc", "bpc(gleich_rot) − bpc(treppeB_rot), min über Längen")
    out["ampel_rotation"] = ampel(min(b[T]["gleich_roh"] - b[T]["gleich_rot"] for T in Ts), 0.01, "bpc", "bpc(gleich_roh) − bpc(gleich_rot), min über Längen")
    print("Ampel A", out["ampel"]["farbe"], "gegen aufm", out["ampel_gegen_aufm"]["farbe"], "B", out["ampel_B"]["farbe"], "Rotation", out["ampel_rotation"]["farbe"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z2_kv_treppe", main)
