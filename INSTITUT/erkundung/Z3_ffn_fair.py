"""Z3_ffn_fair -- Fairer Vergleich in der FFN-Mitte: Rotation gegen die STÄRKSTEN Quantisierer OHNE Rotation (Erkundung 29.10.2026; KEIN Befund, nicht in LABORBUCH).
ANLASS: In Z2_ptq_varianten sank der Schaden durch Rotation riesig (Raven, 20 Seeds: roh 1,5 bpc, rot 0,05 bpc), aber "roh" mit 4 Stufen war viel schlechter als "roh" mit 3 Stufen.
  Verdacht: das symmetrische Gleichgitter ohne Nullstufe bzw. die Max-/MSE-Skala je Token benachteiligt "roh" unfair (Ausreißerkanäle). Z3 prüft das mit den stärksten bekannten Gegnern.
AUFGABE (wie Z2): Byte-LM d=128, 4 Lagen, FFN 4d (Breite 512), Kontext 128, Aktivierung GELU und z²; Aktivierung vor der Down-Projektion wird je Token quantisiert (Gewichte fp32).
  4 Trainingslängen T ∈ {300, 600, 1200, 2400} (SCHNELL=1 / --schnell: {20,40,80,160}); Schaden = bpc(quantisiert) − bpc(fp32), Val-Bytes Z2_VAL (Standard 32768).
BITS b ∈ {2, 3, 4}  ->  L = 2^b echte Stufen (2 Bit = 4 Stufen, wie Z2).
ARME OHNE Rotation (Gegner):
  sym         : symmetrisches Mittelpunktsgitter, Skala je Token per grober MSE-Suche (8 Kandidaten 0,3..1 x Max) = Z2-"roh"
  asym        : (a) asymmetrisch mit ganzzahligem Nullpunkt, reines min/max je Token
  klip_sym    : (c) wie sym, aber feine MSE-Suche der Clipping-Skala (24 Kandidaten 0,15..1)
  klip_asym   : (a)+(c) asymmetrisch + feine MSE-Clipping-Suche
  kanal       : (b) Skala je Token UND je Kanal: statische Kanalskala c_j = RMS_j der Kalibrierung, danach sym je Token
  smooth      : (d) SmoothQuant: c_j = max|x_j|^0,5 / max|W_down[:,j]|^0,5 (Kalibrierung), die Gegenskala wird in W_down gefaltet (hier äquivalent), dann sym je Token
  ausr        : (e) LLM.int8-artig: die 8 Kanäle mit größtem Kalibrier-Max bleiben fp16, Rest sym je Token (Bytes zählen!)
  gruppe32    : Skala (+ Nullpunkt) je Gruppe von 32 Kanälen je Token, asym + feine Klip-Suche (Bytes zählen!)
  alle        : smooth + ausr + asym + feine Klip-Suche kombiniert (stärkster zusammengesetzter Gegner)
ARME MIT Rotation: art ∈ {hadamard (Vorzeichen*Hadamard), zufall (Haar-Orthogonal, dicht)} x Skalenvariante ∈ {sym, asym, klip_sym, klip_asym, gruppe32} (Zufall nur sym, klip_asym, gruppe32 wegen Laufzeit)  (Namen "had_sym", "zuf_klip_asym" ...).
KOSTEN je Token (Vektor der Breite 512), im Feld "kosten": bits_pro_token (Nutzlast + Skalen/Nullpunkte fp16 + fp16-Ausreißer), eff_bit_pro_wert, statische_bytes_je_lage (Kanalskalen/Ausreißerindizes),
  flops_rotation (Hadamard n·log2 n = 4608, dicht n² = 262144; die Rückdrehung ist in W_down gefaltet), flops_quant (grob: Kandidaten x 3n für die Skalensuche).
AMPEL (Einzelseed, vorläufig), je Aktivierung und Bits: je Länge T bester Rotationsarm (kleinster Schaden) gegen besten Nicht-Rotationsarm; gewinn = Minimum über T von (Schaden_beste_NichtRot − Schaden_beste_Rot) in bpc; Schwelle 0,005 (wie Z1/Z2).
  Zweite Ampel "bytefair": nur Nicht-Rotationsarme mit bits_pro_token <= dem des besten Rotationsarms der Länge T (sonst nan). Auswahl des besten Arms je T auf Val-Schaden (leicht optimistisch für beide Seiten gleich).
VORAB-P (vor dem Lauf): Rotation bleibt grün (Gewinn > 0,005 bpc, Minimum über T) gegen den besten Gegner: 2 Bit GELU 55 %, 3 Bit 35 %, 4 Bit 20 %; z² 2 Bit 45 %. Der beste Gegner ist "alle" oder "smooth" (60 %).
  Z2-Verdacht "unfaires roh" bestätigt (asym/ausr/smooth verbessern sym bei 4 Stufen um > 50 %): 70 %.
Nicht untersucht: Gewichte/KV, Kernelzeit, GPTQ. Kalibrierung: 8 Trainings-Batches (8x128) nach dem Training, gleiche Daten für alle Arme.
Lies zuerst: ampel_<akt>_b<bits>, dann schaden[akt][bits][arm][T]."""
from __future__ import annotations
from _z2 import *

BITS = (2, 3, 4)
N = 512
KAN_AUSR = 8


# ------------------------------------------------------------------ Quantisierer-Kern
def _q_sym(x, L, nf, fmin):
    """Mittelpunktsgitter (L gerade) / Nullgitter (L ungerade), Skala je Zeile (letzte Achse), MSE-Suche über nf Kandidaten fmin..1 x Max."""
    gerade = L % 2 == 0
    hi = L / 2 - 0.5 if gerade else (L - 1) / 2
    norm = L / 2 if gerade else hi
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best = bq = None
    for f in np.linspace(fmin, 1.0, nf):
        s = amax * float(f) / norm
        q = ((x / s - 0.5).round() + 0.5).clamp(-hi, hi) * s if gerade else (x / s).round().clamp(-hi, hi) * s
        e = ((x - q) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bq = e, q
        else:
            m = e < best
            best, bq = torch.where(m, e, best), torch.where(m, q, bq)
    return bq


def _q_asym(x, L, nf, fmin):
    """Asymmetrisch mit ganzzahligem Nullpunkt (0 exakt darstellbar), Bereich [min,max] x f je Zeile; nf=1: reines min/max."""
    mn = x.amin(-1, keepdim=True).clamp(max=0.0)
    mx = x.amax(-1, keepdim=True).clamp(min=0.0)
    best = bq = None
    for f in (np.linspace(fmin, 1.0, nf) if nf > 1 else [1.0]):
        lo, hi = mn * float(f), mx * float(f)
        s = ((hi - lo) / (L - 1)).clamp_min(1e-12)
        z = (-lo / s).round()
        q = ((x / s).round() + z).clamp(0, L - 1)
        q = (q - z) * s
        e = ((x - q) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bq = e, q
        else:
            m = e < best
            best, bq = torch.where(m, e, best), torch.where(m, q, bq)
    return bq


def quant_zeile(x, L, modus, g=None):
    """modus: 'sym' (8 Kand. 0,3..1) | 'asym' (min/max) | 'klip_sym' (24 Kand. 0,15..1) | 'klip_asym'. g: Gruppengröße (Skala je Gruppe) oder None."""
    f = {"sym": lambda t: _q_sym(t, L, 8, 0.3), "asym": lambda t: _q_asym(t, L, 1, 1.0),
         "klip_sym": lambda t: _q_sym(t, L, 24, 0.15), "klip_asym": lambda t: _q_asym(t, L, 24, 0.15)}[modus]
    if g is None:
        return f(x)
    sh = x.shape
    return f(x.reshape(*sh[:-1], sh[-1] // g, g)).reshape(sh)


# ------------------------------------------------------------------ Arme
def arm_liste():
    """[(name, spec)]; spec: rot (None|'hadamard'|'zufall'), modus, g, kanal (None|'rms'|'smooth'), ausr (Anzahl fp16-Kanäle), nf_kosten."""
    A = [("sym", dict(modus="sym")), ("asym", dict(modus="asym")), ("klip_sym", dict(modus="klip_sym")), ("klip_asym", dict(modus="klip_asym")),
         ("kanal", dict(modus="sym", kanal="rms")), ("smooth", dict(modus="sym", kanal="smooth")), ("ausr", dict(modus="sym", ausr=KAN_AUSR)),
         ("gruppe32", dict(modus="klip_asym", g=32)),
         ("alle", dict(modus="klip_asym", kanal="smooth", ausr=KAN_AUSR))]
    for art, kurz, ms in (("hadamard", "had", ("sym", "asym", "klip_sym", "klip_asym")), ("zufall", "zuf", ("sym", "klip_asym"))):   # Zufall nur 2 Skalenvarianten (Laufzeit)
        for m in ms:
            A.append((f"{kurz}_{m}", dict(modus=m, rot=art)))
        A.append((f"{kurz}_gruppe32", dict(modus="klip_asym", g=32, rot=art)))
    return A


NF = {"sym": 8, "asym": 1, "klip_sym": 24, "klip_asym": 24}


def kosten(spec, b):
    """Bits und FLOPs je Token (Vektor Breite 512). Skala/Nullpunkt fp16 (asym: Skala 16 + Nullpunkt b Bit)."""
    g = spec.get("g")
    ngr = N // g if g else 1
    k = spec.get("ausr", 0)
    sk = ngr * (16 + (b if spec["modus"].endswith("asym") else 0))
    bits = (N - k) * b + k * 16 + sk
    rot = spec.get("rot")
    return {"bits_pro_token": bits, "eff_bit_pro_wert": bits / N, "flops_rotation": N * 9 if rot == "hadamard" else (N * N if rot == "zufall" else 0),
            "flops_quant": NF[spec["modus"]] * 3 * N, "statische_bytes_je_lage": (2 * N if spec.get("kanal") else 0) + (2 * k if k else 0)}


def arm_bauen(spec, L, Q, kal):
    """Gibt f(a, lage) -> quantisierte Aktivierung (Rückdrehung bzw. Gegenskala einberechnet)."""
    g, modus = spec.get("g"), spec["modus"]

    def f(a, lage):
        if Q is not None:
            return quant_zeile(a @ Q.T, L, modus, g) @ Q
        k = kal[lage]
        c = None
        if spec.get("kanal") == "rms":
            c = k["rms"]
        elif spec.get("kanal") == "smooth":
            c = k["smooth"]
        x = a if c is None else a / c
        n = spec.get("ausr", 0)
        if n:
            m = torch.zeros(N, dtype=torch.bool)
            m[(k["amax"] / (c if c is not None else 1.0)).topk(n).indices] = True
            q = quant_zeile(x.masked_fill(m, 0.0), L, modus, g)
            q = torch.where(m, x.half().float(), q)
        else:
            q = quant_zeile(x, L, modus, g)
        return q if c is None else q * c
    return f


# ------------------------------------------------------------------ Netz mit Haken
class FFNZ3(FFNQ):
    def __init__(self, d, f, akt="gelu"):
        super().__init__(d, f, akt)
        self.lage = 0
        self.rec = None

    def forward(self, x):
        a = self.akt(self.a(x))
        if self.rec is not None:
            self.rec.append(a.detach().reshape(-1, a.shape[-1]))
        if self.mitte is not None:
            a = self.mitte(a, self.lage)
        return self.b(a)


def lm_z3_bauen(d=128, akt="gelu", lagen=4):
    n = LMX([FFNZ3(d, 4 * d, akt) for _ in range(lagen)], d=d)
    for i, b in enumerate(n.bl):
        b.att = AttnQ(d, kb.koepfe(d))
        b.ffn.lage = i
    return n


@torch.no_grad()
def kalibrieren(n, seed, schnell):
    tr = ss.daten(schnell)[0]
    g = torch.Generator().manual_seed(seed + 77)
    for b in n.bl:
        b.ffn.rec, b.ffn.mitte = [], None
    n.eval()
    for _ in range(8):
        i = torch.randint(0, len(tr) - 130, (8,), generator=g)
        x = tr[i[:, None] + torch.arange(128)].long()
        n(x)
    kal = []
    for b in n.bl:
        A = torch.cat(b.ffn.rec)
        b.ffn.rec = None
        amax = A.abs().amax(0).clamp_min(1e-6)
        rms = A.pow(2).mean(0).sqrt().clamp_min(1e-6)
        wm = b.ffn.b.weight.abs().amax(0).clamp_min(1e-6)
        sm = amax.pow(0.5) / wm.pow(0.5)
        kal.append({"amax": amax, "rms": rms / rms.mean(), "smooth": sm / sm.mean()})
    return kal


def auswahl_ampel(S, arme_rot, arme_nr, kosten_b, Ts, bytefair=False):
    gew = []
    for T in Ts:
        r = min(arme_rot, key=lambda a: S[a][T])
        kand = [a for a in arme_nr if (not bytefair or kosten_b[a]["bits_pro_token"] <= kosten_b[r]["bits_pro_token"])]
        if not kand:
            return float("nan"), None
        nr = min(kand, key=lambda a: S[a][T])
        gew.append((S[nr][T] - S[r][T], r, nr))
    m = min(gew, key=lambda t: t[0])
    return float(m[0]), {"je_T": [{"T": int(T), "gewinn": float(x[0]), "beste_rot": x[1], "beste_nichtrot": x[2]} for T, x in zip(Ts, gew)]}


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    Ts = (20, 40, 80, 160) if schnell else (300, 600, 1200, 2400)
    arme = arm_liste()
    rot_namen = [a for a, s in arme if s.get("rot")]
    nr_namen = [a for a, s in arme if not s.get("rot")]
    out = {"Ts": Ts, "bits": BITS, "val_bytes": VAL_N, "bpc_fp32": {"gelu": {}, "z2": {}}, "instabil": {}, "schaden": {"gelu": {}, "z2": {}},
           "arme": [a for a, _ in arme], "rotationsarme": rot_namen, "kosten": {str(b): {a: kosten(s, b) for a, s in arme} for b in BITS}}
    for akt in ("gelu", "z2"):
        for b in BITS:
            out["schaden"][akt][b] = {a: {} for a, _ in arme}
    for T in Ts:
        for akt in ("gelu", "z2"):
            t = time.time()
            torch.manual_seed(seed)
            n = lm_z3_bauen(128, akt)
            r = lm_trainiere(n, seed, T, schnell=schnell)
            n.eval()
            v0 = val_bpc(n, schnell)
            out["bpc_fp32"][akt][T], out["instabil"][f"{akt}_{T}"] = v0, r["instabil"]
            if r["instabil"]:
                print(f"Z3_ffn_fair s{seed} {akt} T{T}: INSTABIL, übersprungen", flush=True)
                continue
            kal = kalibrieren(n, seed, schnell)
            Qs = {"hadamard": rot_q(N, "hadamard", seed + 13), "zufall": rot_q(N, "zufall", seed + 13)}
            for b in BITS:
                L = 2 ** b
                for a, s in arme:
                    f = arm_bauen(s, L, Qs.get(s.get("rot")), kal)
                    for blk in n.bl:
                        blk.ffn.mitte = f
                    with torch.no_grad():
                        out["schaden"][akt][b][a][T] = val_bpc(n, schnell) - v0
                for blk in n.bl:
                    blk.ffn.mitte = None
            print(f"Z3_ffn_fair s{seed} {akt} T{T}: fp32 {v0:.4f} bpc ({time.time() - t:.0f}s)", flush=True)
            for b in BITS:
                S = out["schaden"][akt][b]
                bn = min(nr_namen, key=lambda a: S[a][T])
                br = min(rot_namen, key=lambda a: S[a][T])
                print(f"   {b} Bit: sym {S['sym'][T]:.4f}  bester Nicht-Rot {bn} {S[bn][T]:.4f}  bester Rot {br} {S[br][T]:.4f}", flush=True)
    # ---------------------------------------------------------------- Auswertung
    out["auswertung"] = {}
    for akt in ("gelu", "z2"):
        Tz = [T for T in Ts if not out["instabil"].get(f"{akt}_{T}")]
        out["auswertung"][akt] = {}
        for b in BITS:
            S = out["schaden"][akt][b]
            out["auswertung"][akt][b] = {a: zeile_auswerten(Tz, [S[a][T] for T in Tz]) for a in S if all(T in S[a] for T in Tz)} if len(Tz) >= 2 else {}
            if len(Tz) < 1:
                continue
            kb_ = out["kosten"][str(b)]
            for variante, bf in (("", False), ("_bytefair", True)):
                gw, det = auswahl_ampel(S, rot_namen, nr_namen, kb_, Tz, bf)
                am = ampel(gw, 0.005, "bpc (Minimum über T)", f"bester Rotationsarm vs. bester Nicht-Rotationsarm, {akt}, {b} Bit" + (", nur Gegner mit <= Bits" if bf else ""))
                am["details"] = det
                out[f"ampel_{akt}_b{b}{variante}"] = am
    print("Ampel " + "  ".join(f"{akt}/{b}b {out[f'ampel_{akt}_b{b}']['farbe']}({out[f'ampel_{akt}_b{b}']['gewinn']:.3f})" for akt in ("gelu", "z2") for b in BITS
                               if f"ampel_{akt}_b{b}" in out), flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z3_ffn_fair", main)
