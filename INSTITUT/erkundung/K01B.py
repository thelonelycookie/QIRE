"""K01B -- Zusatzarme zu K01: gelernte Gegenspieler zum Vorzeichen-Schlüssel (Erkundung Welle 3, 28.10.2026; kein Befund, nicht in LABORBUCH).
K01.py bleibt unverändert; dieses Skript importiert dessen Bausteine (LMGeteilt, GeteiltFFN, zaehlung) und nutzt dieselben Daten, Seeds, Trainingsregeln und Zählung.
FRAGE (f2-Bericht, K01 „umbauen“): Holen feste Zufalls-Schlüssel (0 Byte) etwas zurück, was die billigste GELERNTE Alternative nicht auch zurückholt?
ARME (alle: EINE FFN f=512 für alle Lagen, Attention/Einbettung je Lage eigen wie in K01):
  geteilt_o     ALBERT-artige reine Teilung (identisch zu K01-Arm geteilt_o; als Anker mitgerechnet)
  geteilt_diag  Teilung + GELERNTE Lagen-Diagonalen: Skala auf Eingang (d), Zwischenschicht (f), Ausgang (d) je Lage, Start 1  (2d+f Zahlen je Lage)
  geteilt_diagb wie geteilt_diag, zusätzlich gelernter Bias auf der Zwischenschicht je Lage (f weitere Zahlen je Lage)
  geteilt_lora  Teilung + Niedrigrang-Zusatz je Lage (Relaxed-Recursive-Stil): W1_l = W1 + U_l V_l, Rang 1 (f+d Zahlen je Lage), U_l startet bei 0
  geteilt_k     (nur mit K01B_MIT_K=1) der feste Vorzeichen-Schlüssel aus K01, als Vergleich im selben Lauf
Zusatzbytes der gelernten Arme: 2 B (fp16) je Zusatzzahl, in `ffn_bytes` bereits enthalten (`ffn_bytes_basis` = geteilt_o). K01B_LAGEN=8 rechnet 8 statt 4 Lagen (Zusatz laut Bericht).
TRAINING/MESSUNG: wie K01 (4000 Schritte, --schnell 60, Adam lr 2e-3 Kosinus, Val-bpc). Lies zuerst: geteilt_diag - geteilt_o, geteilt_lora - geteilt_o, dann gegen K01-Wert geteilt_k und klein."""
from __future__ import annotations
import math
from _basis import *
import K01 as K

LAGEN = int(os.environ.get("K01B_LAGEN", "4"))


class GeteiltGelernt(nn.Module):
    """Gemeinsame FFN a (d->f), b (f->d); je Lage gelernte Diagonalen bzw. Niedrigrang-Zusatz."""
    braucht_lage = True

    def __init__(self, d, f, lagen, art):
        super().__init__()
        self.art = art
        self.a, self.b = nn.Linear(d, f), nn.Linear(f, d)
        if art in ("diag", "diagb"):
            self.s_ein, self.s_mit, self.s_aus = (nn.Parameter(torch.ones(lagen, n)) for n in (d, f, d))
            if art == "diagb":
                self.b_mit = nn.Parameter(torch.zeros(lagen, f))
        elif art == "lora":
            self.V = nn.Parameter(torch.randn(lagen, 1, d) / math.sqrt(d))
            self.U = nn.Parameter(torch.zeros(lagen, f, 1))

    def forward(self, x, lage):
        if self.art in ("diag", "diagb"):
            h = self.a(x * self.s_ein[lage]) * self.s_mit[lage]
            if self.art == "diagb":
                h = h + self.b_mit[lage]
            return self.b(F.gelu(h)) * self.s_aus[lage]
        h = self.a(x)
        if self.art == "lora":
            h = h + (x @ self.V[lage].T) @ self.U[lage].T
        return self.b(F.gelu(h))


def zusatz_zahlen(arm, d=128, f=512, lagen=LAGEN):
    return {"geteilt_diag": lagen * (2 * d + f), "geteilt_diagb": lagen * (2 * d + 2 * f), "geteilt_lora": lagen * (f + d)}.get(arm, 0)


def bau(arm, seed, d=128, f=512, lagen=LAGEN):
    if arm in ("geteilt_o", "geteilt_k"):
        return K.bauer(arm, seed, d, lagen)()
    torch.manual_seed(seed)
    return K.LMGeteilt(GeteiltGelernt(d, f, lagen, arm.split("_")[1]), d, lagen)


def main(seed: int, schnell: bool = False) -> dict:
    T = 60 if schnell else 4000
    arme = ["geteilt_o", "geteilt_diag", "geteilt_diagb", "geteilt_lora"] + (["geteilt_k"] if os.environ.get("K01B_MIT_K") == "1" else [])
    out = {"T": T, "lr": LM_LR, "lagen": LAGEN, "arme": {}}
    for arm in arme:
        netz = bau(arm, seed)
        r = lm_trainiere(netz, seed, T, lr=LM_LR, schnell=schnell)
        z = K.zaehlung("geteilt_o" if arm != "geteilt_k" else arm, lagen=LAGEN)
        zz = zusatz_zahlen(arm)
        r.update(z)
        r["ffn_bytes_basis"] = z["ffn_bytes"]
        r["ffn_bytes"] = z["ffn_bytes"] + 2 * zz
        r["zusatz_zahlen_gelernt"] = zz
        # Zusatz-Multiplikationen/MACs je Lage und Token (grob): diag 2d+f, diagb 2d+2f, lora (f+d) + d·... Rang-1: d+f
        r["zusatz_macs_je_lage"] = {"geteilt_diag": 2 * 128 + 512, "geteilt_diagb": 2 * 128 + 2 * 512, "geteilt_lora": 128 + 512}.get(arm, 0)
        r["param_trainierbar"] = zaehle_param(netz)
        out["arme"][arm] = r
        print(f"K01B s{seed} {arm}: bpc {r['bpc']:.4f}  FFN-Bytes {r['ffn_bytes']} (+{2*zz})  {r['sekunden']/60:.1f} min", flush=True)
    a = out["arme"]
    out["differenzen_bpc"] = {f"{k}_minus_geteilt_o": a[k]["bpc"] - a["geteilt_o"]["bpc"] for k in a if k != "geteilt_o"}
    out["hinweis"] = "Vergleich mit K01 (gleicher Seed, gleiche Regeln): klein (gleiche Bytes), geteilt_k, normal stehen in roh/K01_s<seed>.json."
    return out


if __name__ == "__main__":
    rahmen("K01B", main)
