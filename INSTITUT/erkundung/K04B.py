"""K04B -- Zusatzarme zu K04: dicht1024 als Pflichtarm + Niedrigrang-W1 (Erkundung Welle 3, 28.10.2026; kein Befund, nicht in LABORBUCH).
K04.py bleibt unverändert; Bausteine (EingefrorenFFN, bau, bytes_ffn) werden importiert. Daten, Seeds, Training, Lernratenwahl je Klasse und Zählung wie K04.
ARME:
  dicht512        Referenz (Bytes B, FLOPs 2·d·512)               [Anker]
  zufall_s1024    eingefrorenes R aus Seed, s+b1+W2 gelernt       [Anker, Hauptarm von K04]
  dicht1024       W1, W2 gelernt, f=1024: gleiche FLOPs wie zufall_s1024, doppelte Bytes  (K04-Pflichtarm laut f2-Bericht)
  nr1024_r1       W1 = U V mit Rang 1 (f+d Zahlen), f=1024, W2 voll: Bytes ≈ zufall_s1024 (Differenz 128 Parameter je Lage) = härteste Kontrolle „gleiche Bytes, gleiche Breite“
  nr1024_r8       wie oben, Rang 8 (Zusatz: +~7 Tsd. Parameter je Lage, ≈ +5 % Bytes) zur Einordnung der Rangabhängigkeit
FLOPs: MACs der FFN je Token und Lage: dicht 2·d·f; Niedrigrang r·(d+f) + d·f; zufall_s1024 2·d·f (R voll gerechnet).
Lernrate je Klasse (dicht / eingefroren / niedrigrang) einmal aus {1e-3,4e-3} nach 400 Schritten (Val-bpc), dann 4000 Schritte (--schnell 40). Val-bpc, nie Test.
Lies zuerst: nr1024_r1 - zufall_s1024 (gleiche Bytes), nr1024_r1 - dicht512, dicht1024 - zufall_s1024, zufall_s1024 - dicht512."""
from __future__ import annotations
from _basis import *
import K04 as K


class FFNNiedrigrang(nn.Module):
    def __init__(self, d, f, r):
        super().__init__()
        self.v = nn.Linear(d, r, bias=False)
        self.u = nn.Linear(r, f)
        self.b = nn.Linear(f, d)

    def forward(self, x):
        return self.b(F.gelu(self.u(self.v(x))))


def bytes_nr(r, d=128, f=1024, lagen=4):
    return 2 * lagen * (r * (d + f) + f + f * d + d)


def bau(arm, seed, d=128, lagen=4):
    if arm.startswith("nr1024"):
        r = int(arm.split("_r")[1])
        torch.manual_seed(seed)
        return LMX([FFNNiedrigrang(d, 1024, r) for _ in range(lagen)], d)
    return K.bau(arm, seed, d, lagen)


ARME = ("dicht512", "zufall_s1024", "dicht1024", "nr1024_r1", "nr1024_r8")


def main(seed: int, schnell: bool = False) -> dict:
    T = 40 if schnell else 4000
    T_lr = 10 if schnell else 400
    out = {"T": T, "arme": {}}
    lrs = {}
    for arm in ARME:
        klasse = "dicht" if arm.startswith("dicht") else ("niedrigrang" if arm.startswith("nr") else "eingefroren")
        if klasse not in lrs:
            lrs[klasse] = lm_lr_waehlen(lambda: bau(arm, seed), seed, schnell, T_kurz=T_lr)
        lr, tab = lrs[klasse]
        netz = bau(arm, seed)
        r = lm_trainiere(netz, seed, T, lr=lr, schnell=schnell)
        f = 1024 if arm.endswith("1024") or arm.startswith("nr") else 512
        if arm.startswith("nr"):
            rg = int(arm.split("_r")[1])
            by, macs = bytes_nr(rg), rg * (128 + f) + 128 * f
        else:
            by, macs = K.bytes_ffn(arm), 2 * 128 * f
        r.update({"lr": lr, "lr_tabelle": {str(k): v for k, v in tab.items()}, "ffn_bytes": by, "ffn_macs_je_lage": macs,
                  "param_trainierbar": zaehle_param(netz), "param_gesamt_inkl_seed_R": zaehle_param(netz, False) + (4 * f * 128 if arm.startswith("zufall") else 0)})
        out["arme"][arm] = r
        print(f"K04B s{seed} {arm}: bpc {r['bpc']:.4f} lr {lr} FFN-Bytes {r['ffn_bytes']} MACs/Lage {macs} {r['sekunden']/60:.1f} min", flush=True)
    a = out["arme"]
    b = lambda k: a[k]["bpc"]
    out["differenzen_bpc"] = {"gleiche_Bytes (zufall_s1024-dicht512)": b("zufall_s1024") - b("dicht512"),
                              "R_gelernt (dicht1024-zufall_s1024)": b("dicht1024") - b("zufall_s1024"),
                              "niedrigrang_gleiche_Bytes (nr1024_r1-zufall_s1024)": b("nr1024_r1") - b("zufall_s1024"),
                              "niedrigrang_r1_vs_dicht512": b("nr1024_r1") - b("dicht512"),
                              "niedrigrang_r8_vs_dicht512": b("nr1024_r8") - b("dicht512"),
                              "dicht1024_vs_dicht512": b("dicht1024") - b("dicht512")}
    return out


if __name__ == "__main__":
    rahmen("K04B", main)
