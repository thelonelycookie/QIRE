"""K04 -- Eingefrorene Seed-Aufwärtsschicht (Erkundung Welle 2, 28.10.2026; kein Befund, nicht in LABORBUCH).
FRAGE: Die erste FFN-Matrix R (f x d) kommt aus einem Seed (0 Bytes, nie trainiert); gelernt werden nur Zeilenskalen s (f), Bias und die zweite Matrix W2 (d x f).
  Bei GLEICHEN Bytes baut man dafür doppelt so breit (f=1024 statt 512). Schlägt das das dichte, voll gelernte f=512-Modell?
MODELL: Byte-LM d=128, 4 Lagen, 4 Köpfe, Kontext 128, gebundene Ausgabe; nur die FFN unterscheidet sich (GELU): h = gelu(s ⊙ (R x) + b1), y = W2 h + b2.
ARME (FFN-Bytes fp16 über alle Lagen; FLOPs als MACs der FFN je Token und Lage):
  dicht512     W1, W2 gelernt, f=512                            (Referenz: Bytes B, FLOPs 2·d·512)
  zufall_s1024 R Gauß aus Seed, s+b1+W2 gelernt, f=1024         (Hauptarm, Bytes ≈ B, FLOPs 2x)
  zufall_o1024 wie oben ohne Zeilenskalen s                     (Kontrolle: ohne Skalen, nur Bias)
  had_s1024    R = 8 Blöcke H128·diag(Vorzeichen_j) (Seed), mit Skalen (Kontrolle: strukturiertes R statt Gauß)
  Zusatzarme nur mit K04_ZUSATZ=1 (Zeitbudget): dicht1024 (W1 gelernt, f=1024: gleiche FLOPs wie Hauptarm, doppelte Bytes), zufall_s512 (gleiche FLOPs wie dicht512, halbe Bytes)
TRAINING: 4000 Schritte (--schnell: 40), Adam lr 2e-3 (Kosinus, 5 % Aufwärmen, Clip 1,0), Batch 8x128, Val-bpc, nie Test. Lernrate je Klasse (dicht / eingefroren) einmal aus {1e-3, 4e-3} nach 400 Schritten (Val-bpc) gewählt, dann volles Training.
Lies zuerst: zufall_s1024 - dicht512 (gleiche Bytes), zufall_s1024 - zufall_o1024 (Skalen), zufall_s1024 - had_s1024, (mit K04_ZUSATZ) zufall_s512 - dicht512, dicht1024 - zufall_s1024."""
from __future__ import annotations
from _basis import *


class EingefrorenFFN(nn.Module):
    def __init__(self, d, f, seed, art="zufall", skalen=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed * 7 + 123)
        if art == "zufall":
            R = torch.randn(f, d, generator=g) / math.sqrt(d)
        else:                                                    # Hadamard-Blöcke mit Vorzeichen
            H = kb.hadamard(d)
            R = torch.cat([H * (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float()[None, :] for _ in range(f // d)])
        self.register_buffer("R", R)
        self.s = nn.Parameter(torch.ones(f)) if skalen else None
        self.b1 = nn.Parameter(torch.zeros(f))
        self.b = nn.Linear(f, d)

    def forward(self, x):
        h = x @ self.R.T
        if self.s is not None:
            h = h * self.s
        return self.b(F.gelu(h + self.b1))


def bytes_ffn(arm, d=128, lagen=4):
    f = 1024 if arm.endswith("1024") else 512
    if arm.startswith("dicht"):
        return 2 * lagen * (2 * d * f + f + d)
    return 2 * lagen * (d * f + d + f + (f if "_s" in arm else 0))


def bau(arm, seed, d=128, lagen=4):
    torch.manual_seed(seed)
    f = 1024 if arm.endswith("1024") else 512
    if arm.startswith("dicht"):
        return LMX([FFNDichtX(d, f) for _ in range(lagen)], d)
    art = "had" if arm.startswith("had") else "zufall"
    return LMX([EingefrorenFFN(d, f, seed * 10 + l, art, skalen="_s" in arm) for l in range(lagen)], d)


ARME = ("dicht512", "zufall_s1024", "zufall_o1024", "had_s1024")
ZUSATZ = ("dicht1024", "zufall_s512")          # nur mit Umgebungsvariable K04_ZUSATZ=1 (Zeitbudget: je +10 bzw. +6 min)


def main(seed: int, schnell: bool = False) -> dict:
    T = 40 if schnell else 4000
    T_lr = 10 if schnell else 400
    out = {"T": T, "arme": {}}
    lrs = {}
    for arm in ARME + (ZUSATZ if os.environ.get("K04_ZUSATZ") == "1" else ()):
        klasse = "dicht" if arm.startswith("dicht") else "eingefroren"      # Lernrate je Klasse einmal gewählt (Zeitbudget)
        if klasse not in lrs:
            lrs[klasse] = lm_lr_waehlen(lambda: bau(arm, seed), seed, schnell, T_kurz=T_lr)
        lr, tab = lrs[klasse]
        netz = bau(arm, seed)
        r = lm_trainiere(netz, seed, T, lr=lr, schnell=schnell)
        f = 1024 if arm.endswith("1024") else 512
        r.update({"lr": lr, "lr_tabelle": {str(k): v for k, v in tab.items()}, "ffn_bytes": bytes_ffn(arm), "ffn_macs_je_lage": 2 * 128 * f,
                  "param_trainierbar": zaehle_param(netz), "param_gesamt_inkl_seed_R": zaehle_param(netz, False) + (0 if arm.startswith("dicht") else 4 * f * 128)})
        out["arme"][arm] = r
        print(f"K04 s{seed} {arm}: bpc {r['bpc']:.4f} lr {lr} FFN-Bytes {r['ffn_bytes']} {r['sekunden']/60:.1f} min", flush=True)
    a = out["arme"]
    out["differenzen_bpc"] = {"gleiche_Bytes (zufall_s1024-dicht512)": a["zufall_s1024"]["bpc"] - a["dicht512"]["bpc"],
                              "Skalen (zufall_s1024-zufall_o1024)": a["zufall_s1024"]["bpc"] - a["zufall_o1024"]["bpc"],
                              "R_Hadamard (had_s1024-zufall_s1024)": a["had_s1024"]["bpc"] - a["zufall_s1024"]["bpc"],
                              }
    if "dicht1024" in a:
        out["differenzen_bpc"]["R_gelernt (dicht1024-zufall_s1024)"] = a["dicht1024"]["bpc"] - a["zufall_s1024"]["bpc"]
        out["differenzen_bpc"]["gleiche_FLOPs (zufall_s512-dicht512)"] = a["zufall_s512"]["bpc"] - a["dicht512"]["bpc"]
    return out


if __name__ == "__main__":
    rahmen("K04", main)
