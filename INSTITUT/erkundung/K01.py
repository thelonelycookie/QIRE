"""K01 -- Schlüssel-Superposition über Lagen (Erkundung Welle 2, 28.10.2026; kein Befund, nicht in LABORBUCH).
FRAGE: Alle 4 Lagen teilen EINE FFN-Matrix; jede Lage liest sie durch eigene feste Vorzeichen-Diagonalen (und Hadamard-Drehung) anders.
  Spart das Bytes ohne Qualitätsverlust? Ist der Schlüssel nötig (Kontrolle: Teilung ohne Schlüssel)? Schlägt es ein kleines dichtes Modell gleicher Bytes?
MODELL: Byte-LM d=128, 4 Lagen, 4 Köpfe (kb.koepfe), Kontext 128, gebundene Ausgabe; nur die FFN unterscheidet sich (GELU, Bias, Pre-LN, Attention/Einbettung je Lage eigen und gleich).
ARME (FFN-Bytes = 2 B/Gewicht+Bias fp16, über alle 4 Lagen):
  normal       je Lage eigene FFN f=512                                (Referenz, 4 Matrixpaare)
  klein        je Lage eigene FFN f=128                                (dichtes Modell gleicher Bytes wie geteilt f=512)
  geteilt_o    eine FFN f=512 für alle Lagen, ohne Schlüssel           (Kontrolle: nur Teilung)
  geteilt_k    eine FFN f=512, Lage l liest sie durch Schlüssel K_l    (Vorzeichen auf Eingang d, Zwischen f, Ausgang d + Hadamard auf Ein-/Ausgang)
  breit_o/k    wie geteilt_o/k, aber f=1024 (Bytes = halb so viele wie normal; FLOPs 2x der FFN)
SCHLÜSSEL: aus dem Seed erzeugt, 0 Bytes (zusätzlich ausgewiesen: bei Speicherung (2d+f)/8 Byte je Lage). Hadamard schnell gezählt (d·log2 d Additionen je Drehung).
TRAINING: 4000 Schritte (--schnell: 60), Adam lr 2e-3 (Kosinus, 5 % Aufwärmen, Clip 1,0), Batch 8x128. Gemessen: Val-bpc (ss.VAL_BYTES), nie Test.
AUSGABE: roh/K01_s<seed>.json mit bpc, FFN-Bytes, Parameter, MACs je Token, Sekunden je Arm. Lies zuerst: geteilt_k - geteilt_o (Schlüsselwirkung), geteilt_k - klein (gleiche Bytes), breit_k - normal."""
from __future__ import annotations
import math
from _basis import *


class GeteiltFFN(nn.Module):
    """Eine gemeinsame FFN-Matrix; Schlüssel je Lage (feste Puffer)."""
    braucht_lage = True

    def __init__(self, d, f, lagen, seed, schluessel=True):
        super().__init__()
        self.a, self.b = nn.Linear(d, f), nn.Linear(f, d)
        self.sk = schluessel
        if schluessel:
            g = torch.Generator().manual_seed(seed * 977 + 3)
            sg = lambda n: (torch.randint(0, 2, (lagen, n), generator=g) * 2 - 1).float()
            self.register_buffer("s_ein", sg(d)); self.register_buffer("s_mit", sg(f)); self.register_buffer("s_aus", sg(d))
            self.register_buffer("H", kb.hadamard(d))

    def forward(self, x, lage):
        if not self.sk:
            return self.b(F.gelu(self.a(x)))
        u = (x * self.s_ein[lage]) @ self.H                     # Drehung des Eingangs
        h = F.gelu(self.a(u) * self.s_mit[lage])               # Vorzeichen auf der Zwischenschicht
        return (self.b(h) * self.s_aus[lage]) @ self.H


class LMGeteilt(LMX):
    """LMX mit EINER FFN, die in allen Lagen aufgerufen wird (Parameter nur einmal vorhanden)."""

    def __init__(self, ffn, d=128, lagen=4):
        super().__init__([_Leer() for _ in range(lagen)], d=d)
        self.gemeinsam = ffn
        for b in self.bl:
            b.ffn = _Zeiger(ffn)


class _Leer(nn.Module):
    def forward(self, x):
        return x


class _Zeiger(nn.Module):
    """Ruft die gemeinsame FFN auf, ohne sie als Kind zu registrieren (Parameter zählen einmal)."""
    braucht_lage = True

    def __init__(self, ziel):
        super().__init__()
        object.__setattr__(self, "ziel", ziel)

    def forward(self, x, lage):
        return self.ziel(x, lage)


def bauer(arm, seed, d=128, lagen=4):
    def bau():
        torch.manual_seed(seed)
        if arm == "normal":
            return LMX([FFNDichtX(d, 512) for _ in range(lagen)], d)
        if arm == "klein":
            return LMX([FFNDichtX(d, 128) for _ in range(lagen)], d)
        f = 1024 if arm.startswith("breit") else 512
        return LMGeteilt(GeteiltFFN(d, f, lagen, seed, schluessel=arm.endswith("_k")), d, lagen)
    return bau


def zaehlung(arm, d=128, lagen=4):
    f = {"normal": 512, "klein": 128}.get(arm, 1024 if arm.startswith("breit") else 512)
    paare = lagen if arm in ("normal", "klein") else 1
    ffn_bytes = 2 * paare * (2 * d * f + f + d)
    macs = 2 * d * f
    extra_add = 0
    if arm.endswith("_k"):
        extra_add = 2 * d * int(math.log2(d)) + 2 * d + f           # 2 schnelle Hadamard + Vorzeichen (als Additionen/Mult.)
    return {"ffn_bytes": ffn_bytes, "ffn_macs_je_lage": macs, "schluessel_zusatz_ops_je_lage": extra_add,
            "macs_gesamt_je_token": lm_macs(d, lagen, macs), "schluessel_bytes_falls_gespeichert": lagen * (2 * d + f) // 8 if arm.endswith("_k") else 0}


ARME = ("normal", "klein", "geteilt_o", "geteilt_k", "breit_o", "breit_k")


def main(seed: int, schnell: bool = False) -> dict:
    T = 60 if schnell else 4000
    out = {"T": T, "lr": LM_LR, "arme": {}}
    for arm in ARME:
        netz = bauer(arm, seed)()
        r = lm_trainiere(netz, seed, T, lr=LM_LR, schnell=schnell)
        r.update(zaehlung(arm)); r["param_trainierbar"] = zaehle_param(netz)
        out["arme"][arm] = r
        print(f"K01 s{seed} {arm}: bpc {r['bpc']:.4f}  FFN-Bytes {r['ffn_bytes']}  {r['sekunden']/60:.1f} min", flush=True)
    a = out["arme"]
    out["differenzen_bpc"] = {"schluessel_schmal (geteilt_k-geteilt_o)": a["geteilt_k"]["bpc"] - a["geteilt_o"]["bpc"],
                              "schluessel_breit (breit_k-breit_o)": a["breit_k"]["bpc"] - a["breit_o"]["bpc"],
                              "geteilt_k_vs_klein (gleiche Bytes)": a["geteilt_k"]["bpc"] - a["klein"]["bpc"],
                              "geteilt_k_vs_normal": a["geteilt_k"]["bpc"] - a["normal"]["bpc"],
                              "breit_k_vs_normal": a["breit_k"]["bpc"] - a["normal"]["bpc"]}
    return out


if __name__ == "__main__":
    rahmen("K01", main)
