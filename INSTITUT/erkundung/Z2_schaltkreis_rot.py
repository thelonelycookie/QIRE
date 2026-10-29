"""Z2_schaltkreis_rot -- Gelernte Schaltkreis-Rotation (Butterfly aus 2x2-Givens, wie Qubit-Gatter) vor 2-Bit-Quantisierung (Erkundung Welle 3, 28.10.2026; KEIN Befund, nicht in LABORBUCH).
FRAGE: Schlägt eine quanten-inspirierte, GELERNTE Rotation (log2(n) Ziegelschichten aus Givens-Rotationen, Winkel per Adam auf den Ausgabe-Quantisierungsfehler gelernt) vor einer
  2-Bit-Quantisierung mit echten 4 Stufen die Zufalls-Orthogonale und die Zufalls-Vorzeichen+Hadamard-Rotation, wenn die Rotationskosten (Bytes UND FLOPs) fair mitgezählt werden?
VORAB-P (vor dem Lauf): Ampel Genauigkeit grün (Arm 5 oder 6 schlägt Arm 2 UND Arm 4 bei gleichen Bytes um >= 0,5 pp, Minimum über beide Längen) 8 %; (Genauigkeit am MLP liegt bei 2 Bit eng beieinander, Einzelseed rauscht);
  Ampel Ausgabefehler grün (>= 5 % weniger Ausgabefehler als der Bessere der beiden Gegner) 30 %; Schaltkreis(fp16-Winkel) hat kleineren Ausgabefehler als Vorzeichen+Hadamard OHNE Bytes-Ausgleich 60 %;
  voll gelernte Cayley-Rotation (Obergrenze) hat kleineren Ausgabefehler als Schaltkreis 85 %; Zufalls-Orthogonale schlägt Vorzeichen+Hadamard im Ausgabefehler 20 % (bei einem Bruchteil der Kosten nein);
  Schaltkreis mit 2-3 Bit Winkeln verliert gegenüber fp16-Winkeln weniger als 10 % Ausgabefehler 55 %; Byte-LM-Teil: Ampel bpc grün (>= 0,005 bpc) 10 %.
MODELL: MNIST-MLP 784-512-512-10 fp32 (Adam 1e-3, Kosinus), ZWEI Trainingslängen EP = 3 und 12 Epochen (--schnell: 1 und 4); PTQ der Gewichte ohne Nachtraining, Val 5 000 Bilder. Daneben (falls Z2_LM an, Standard an)
  ein kleines Byte-LM aus _bytelm.py (d = 128, 4 Lagen, Gelu-FFN 4d): nur die FFN-Gewichte (8 Matrizen) werden quantisiert, ZWEI Trainingslängen T = 800 und 2400 Schritte (--schnell: 100 und 300).
QUANTISIERER (alle Arme identisch): Zeilen-Gleichgitter mit 4 Stufen (±0,5 s, ±1,5 s), Zeilenskala per MSE-Suche (kein Ternär); Rotation wirkt auf die EINGANGSachse: W' = W R^T, Eingang x' = R x.
ZIELFUNKTION der gelernten Rotationen: relativer Ausgabefehler  tr(E S E^T) / tr(W S W^T),  E = Q(W R^T) R - W,  S = Eingangskovarianz (zweites Moment) der Schicht auf 4 000 Trainingsbildern bzw. 16 Trainingsbatches;
  Quantisierer mit Straight-Through; der beste Zwischenstand (inkl. Startwert) wird behalten. Gemeldet wird derselbe Fehler auf der Val-Kovarianz (ausgabefehler_rel) und auf der Kalibrierkovarianz (kal_fehler).
ARME (alle vor derselben 2-Bit-Quantisierung):
  1 ohne            keine Rotation (Eingang 784 nativ)
  2 orth_zufall     dichte Zufalls-Orthogonale (QR), n nativ (784/512), 2 n² Bytes (fp16), n² MACs
  3 hadamard        feste Hadamard (ohne Vorzeichen), auf 1024 aufgefüllt, 0 Bytes, n log2 n Additionen
  4 vorz_hadamard   Zufalls-Vorzeichen + Hadamard, n Bit, n log2 n + n Additionen
  4g vorz_had_gb_X  GLEICHE-BYTES-Gegner zu Arm X in {5, 6}: wie 4, aber die Mehrbytes von X (Winkelbytes minus n/8) kaufen 3 statt 2 Bit für die Zeilen mit größtem Ausgabefehler (n/8 Bytes je Zeile)
  5 schaltkreis     log2(n) Ziegelschichten, Schicht k koppelt i und i xor 2^k, n/2 Winkel je Schicht, Start π/4 + 0,2·N(0,1) (Hadamard-nah, Seed), fp16-Winkel (2 Byte), 2 n log2 n MACs (4 je Givens)
  6 schaltkreis_w1/w2/w3  wie 5, Winkel auf 2^b Stufen {(k+½)π/2^b} gerundet (b = 1,2,3; Straight-Through-QAT), ceil(Winkel·b/8) Bytes
  7 cayley_voll     voll gelernte orthogonale Matrix R = Cayley(A) R0 (R0 = Vorzeichen-Hadamard bzw. bei 784 Zufalls-Orthogonale), n² Parameter, 2 n² Bytes, n² MACs (Obergrenze, SpinQuant-artig)
  8 kronecker       Kronecker-Produkt kleiner orthogonaler Faktoren (Cayley je Faktor, "Qubit-Tensorprodukt"): 784 = 4·4·7·7, 512 = 8·8·8, 128 = 4·4·8; 2 Σ b² Bytes, n Σ b MACs
BYTES: o·n·2/8 + Skala fp16 + Bias fp16 (je Zeile 4 Byte) + Rotationsbytes (+ Mehrzeilen-Bit bei 4g); n = Spaltenzahl der rotierten Matrix. FLOPs = 2 MACs je Eingang (Gewichte + Rotation). fp16 der Rotation wird nur
  gezählt, nicht simuliert (Rotation läuft in fp32). Ehrlichkeit: Die Wirkung ist klassisch (I-038); gezählt wird nur der Aufwand der Rotation, nicht deren Einfluss auf Aktivierungen/Hardware.
AMPEL (Einzelseed, vorläufig): der beste Schaltkreis-Arm (5, 6w1-w3) wird nach dem KALIBRIERfehler gewählt (nicht nach Val). Gewinn_Genauigkeit = Acc(Arm) − max(Acc(orth_zufall), Acc(vorz_had_gb_Arm)) in pp, Minimum über beide Längen, Schwelle 0,5 pp.
  Gewinn_Fehler = 100·(1 − Fehler(Arm) / min(Fehler(orth_zufall), Fehler(vorz_had_gb_Arm))) in %, Minimum über beide Längen, Schwelle 5 %. LM: bpc-Differenz (niedriger = besser), Schwelle 0,005.
Lies zuerst: je Länge arme[...] ausgabefehler_rel + bytes + macs, dann ampel_fehler/ampel, dann Cayley (Obergrenze) und Kronecker."""
from __future__ import annotations
from _z1 import *
import _bytelm as BL

STEPS = 200
SCH = ("schaltkreis", "schaltkreis_w1", "schaltkreis_w2", "schaltkreis_w3")
KRON = {784: (4, 4, 7, 7), 512: (8, 8, 8), 128: (4, 4, 8), 1024: (4, 4, 4, 4, 4), 256: (4, 4, 4, 4)}
LR = {"schaltkreis": 0.02, "cayley": 3e-4, "kron": 5e-3}
LM_AN = os.environ.get("Z2_LM", "1") not in ("", "0")


def pow2(i):
    return 1 << (i - 1).bit_length()


def orth(n, seed):
    return kb.orth_matrix(n, seed)


# ------------------------------------------------------------------ Rotationen: matrix() = R mit x' = R x
class RotFest(nn.Module):
    def __init__(self, R, byte, macs):
        super().__init__()
        self.register_buffer("R", R)
        self.byte, self.macs = float(byte), float(macs)

    def matrix(self):
        return self.R


class Butterfly(nn.Module):
    def __init__(self, n, seed, wbits=None):
        super().__init__()
        self.n, self.K, self.wbits = n, int(math.log2(n)), wbits
        g = torch.Generator().manual_seed(seed * 31 + 7)
        self.th = nn.Parameter(math.pi / 4 + 0.2 * torch.randn(self.K, n // 2, generator=g))
        self.byte = math.ceil(self.K * n / 2 * wbits / 8) if wbits else 2.0 * self.K * n / 2
        self.macs = 2.0 * n * self.K

    def winkel(self):
        th = self.th
        if self.wbits:
            L = 2 ** self.wbits
            snap = ((th * L / math.pi - 0.5).round() + 0.5) * math.pi / L
            th = th + (snap - th).detach()
        return th

    def matrix(self):
        n, Y, th = self.n, torch.eye(self.n), self.winkel()
        for k in range(self.K):
            h = 2 ** k
            Y = Y.reshape(n // (2 * h), 2, h, n)
            a, b = Y[:, 0], Y[:, 1]
            c, s = torch.cos(th[k]).reshape(n // (2 * h), h, 1), torch.sin(th[k]).reshape(n // (2 * h), h, 1)
            Y = torch.stack([c * a - s * b, s * a + c * b], 1).reshape(n, n)
        return Y


def cayley(P):
    n = P.shape[0]
    A = P - P.T
    I = torch.eye(n)
    return torch.linalg.solve(I + A, I - A)


class CayleyVoll(nn.Module):
    def __init__(self, n, R0):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(n, n))
        self.register_buffer("R0", R0)
        self.byte, self.macs = 2.0 * n * n, float(n * n)

    def matrix(self):
        return cayley(self.P) @ self.R0


class Kron(nn.Module):
    def __init__(self, faktoren, seed):
        super().__init__()
        self.fs = faktoren
        self.P = nn.ParameterList([nn.Parameter(torch.zeros(b, b)) for b in faktoren])
        for k, b in enumerate(faktoren):
            self.register_buffer(f"Q{k}", orth(b, seed * 7 + k))
        self.byte, self.macs = 2.0 * sum(b * b for b in faktoren), float(math.prod(faktoren) * sum(faktoren))

    def matrix(self):
        R = None
        for k, p in enumerate(self.P):
            Rk = cayley(p) @ getattr(self, f"Q{k}")
            R = Rk if R is None else torch.kron(R, Rk)
        return R


def sign_had(n, seed):
    s = vorzeichen(n, seed)
    return fwht(torch.diag(s)).T.contiguous()


def baue_rot(art, i, sd):
    """Gibt (Rotationsmodul, n) für Eingangsbreite i zurück."""
    n2 = pow2(i)
    if art == "ohne":
        return RotFest(torch.eye(i), 0, 0), i
    if art == "orth_zufall":
        return RotFest(orth(i, sd), 2.0 * i * i, i * i), i
    if art == "hadamard":
        return RotFest(fwht(torch.eye(n2)).T.contiguous(), 0, n2 * math.log2(n2)), n2
    if art in ("vorz_hadamard", "vorz_had_gb"):
        return RotFest(sign_had(n2, sd), n2 / 8, n2 * math.log2(n2) + n2), n2
    if art == "schaltkreis":
        return Butterfly(n2, sd), n2
    if art.startswith("schaltkreis_w"):
        return Butterfly(n2, sd, int(art[-1])), n2
    if art == "cayley_voll":
        return CayleyVoll(i, sign_had(i, sd) if i == pow2(i) else orth(i, sd)), i
    if art == "kronecker":
        return Kron(KRON[i], sd), i
    raise ValueError(art)


# ------------------------------------------------------------------ Quantisierung, Fehler, Lernen
def quant2(Wr, mehr=None):
    q = quant_gitter(Wr, 2)
    if mehr is not None and len(mehr):
        q3 = quant_gitter(Wr[mehr], 3)
        q = q.clone()
        q[mehr] = q3
    return q


def fehler_abs(E, S):
    return float(((E @ S) * E).sum())


def verlust(rot, Wp, S, den):
    R = rot.matrix()
    Wr = Wp @ R.T
    q = quant_gitter(Wr.detach(), 2, nf=6)
    Qs = Wr + (q - Wr).detach()
    E = Qs @ R - Wp
    return ((E @ S) * E).sum() / den


def lerne(rot, Wp, S, steps, lr):
    den = float(((Wp @ S) * Wp).sum())
    opt = torch.optim.Adam(rot.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps), eta_min=lr * 0.1)
    best, snap = float("inf"), None
    for t in range(steps + 1):
        l = verlust(rot, Wp, S, den)
        if float(l) < best:
            best, snap = float(l), {k: v.detach().clone() for k, v in rot.state_dict().items()}
        if t < steps:
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sch.step()
    rot.load_state_dict(snap)
    return best


def kovarianz(X, n):
    X = X.reshape(-1, X.shape[-1])
    X = F.pad(X, (0, n - X.shape[1])) if n > X.shape[1] else X
    return (X.T @ X) / len(X)


def deckung(x):
    return x if isinstance(x, (int, float)) else float(x)


def ptq_arme(Ws, Scal, Sval, bewerte, seed, steps, log=print):
    """Ws: Liste fp32-Gewichte (o,i); Scal/Sval: Listen von Eingangs-Zweitmomenten (i,i) (unaufgefüllt).
    bewerte(Liste effektiver Gewichte) -> dict (acc/bpc/...). Gibt {arm: Ergebnis} zurück."""
    arme, rots = {}, {}
    nL = len(Ws)

    def eff_und_fehler(arm, mehr_je_schicht=None):
        eff, e_val, e_kal, e_gew, b_gew, b_rot, mac_rot, mac_gew, rest = [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
        ws = den_v = den_k = nw = 0.0
        for l in range(nL):
            W, rot, n = Ws[l], rots[arm][l], rots[arm][l].matrix().shape[0]
            o, i = W.shape
            Wp = F.pad(W, (0, n - i))
            Sv, Sc = kovarianz_pad(Sval[l], n), kovarianz_pad(Scal[l], n)
            with torch.no_grad():
                R = rot.matrix()
                Wr = Wp @ R.T
                mehr = None if mehr_je_schicht is None else mehr_je_schicht[l]
                if mehr == "auto":                                    # Zeilen mit dem größten Fehler nehmen (Anzahl k = auto_k[l])
                    q0 = quant_gitter(Wr, 2)
                    ez = ((((q0 @ R) - Wp) @ Sv) * ((q0 @ R) - Wp)).sum(1)
                    mehr = torch.argsort(ez, descending=True)[:auto_k[arm][l]]
                q = quant2(Wr, mehr)
                E = q @ R - Wp
                eff.append((q @ R)[:, :i].clone())
                e_val += fehler_abs(E, Sv)
                e_kal += fehler_abs(E, Sc)
                den_v += fehler_abs(Wp, Sv)
                den_k += fehler_abs(Wp, Sc)
                e_gew += float((E ** 2).sum())
                nw += float((Wp ** 2).sum())
                b_gew += o * n * 2 / 8 + 4 * o + (0 if mehr is None else len(mehr) * n / 8)
                b_rot += rot.byte
                mac_rot += rot.macs
                mac_gew += o * n
        return eff, {"ausgabefehler_rel": e_val / den_v, "kal_fehler": e_kal / den_k, "gewichtsfehler_rel": e_gew / nw,
                     "bytes": b_gew + b_rot, "bytes_rotation": b_rot, "bytes_gewichte": b_gew, "macs_gewichte": mac_gew, "macs_rotation": mac_rot,
                     "macs_gesamt": mac_gew + mac_rot, "flops_gesamt": 2 * (mac_gew + mac_rot)}

    auto_k = {}
    arten = ["ohne", "orth_zufall", "hadamard", "vorz_hadamard", "schaltkreis", "schaltkreis_w1", "schaltkreis_w2", "schaltkreis_w3", "cayley_voll", "kronecker"]
    for arm in arten:
        t = time.time()
        rots[arm] = []
        lernzeit = 0.0
        for l in range(nL):
            i = Ws[l].shape[1]
            rot, n = baue_rot(arm, i, seed * 101 + l + 1)
            if arm in ("schaltkreis", "schaltkreis_w1", "schaltkreis_w2", "schaltkreis_w3", "cayley_voll", "kronecker"):
                schl = "schaltkreis" if arm.startswith("schaltkreis") else ("cayley" if arm == "cayley_voll" else "kron")
                tl = time.time()
                lerne(rot, F.pad(Ws[l], (0, n - i)), kovarianz_pad(Scal[l], n), steps, LR[schl])
                lernzeit += time.time() - tl
            rot.eval()
            rots[arm].append(rot)
        eff, m = eff_und_fehler(arm)
        m.update(bewerte(eff))
        m["sek_lernen"], m["sek_gesamt"] = lernzeit, time.time() - t
        arme[arm] = m
        log(f"   {arm:15s} {fmt(m)}  Fehler {m['ausgabefehler_rel']:.4f}  Bytes {m['bytes']:.0f}  MACs {m['macs_gesamt']:.0f}  ({m['sek_gesamt']:.0f}s, lernen {lernzeit:.0f}s)")
    # Gleiche-Bytes-Gegner zu Schaltkreis-Armen
    for arm in SCH:
        name = "vorz_had_gb_" + arm
        rots[name] = rots["vorz_hadamard"]
        k = []
        for l in range(nL):
            n = rots["vorz_hadamard"][l].matrix().shape[0]
            extra = rots[arm][l].byte - rots["vorz_hadamard"][l].byte
            k.append(int(max(0, min(Ws[l].shape[0], extra // (n / 8)))))
        auto_k[name] = k
        t = time.time()
        eff, m = eff_und_fehler(name, ["auto"] * nL)
        m.update(bewerte(eff))
        m["zeilen_3bit"], m["sek_gesamt"] = k, time.time() - t
        arme[name] = m
        log(f"   {name:24s} {fmt(m)}  Fehler {m['ausgabefehler_rel']:.4f}  Bytes {m['bytes']:.0f}  3-Bit-Zeilen {k}")
    return arme


def kovarianz_pad(S, n):
    i = S.shape[0]
    return F.pad(S, (0, n - i, 0, n - i)) if n > i else S


def fmt(m):
    return f"acc {m['acc']:.4f}" if "acc" in m else f"bpc {m['bpc']:.4f}"


def ampel_paket(A, metrik, hoeher_besser, schwelle_m, laengen):
    """A: {laenge: arme}. Gibt Ampelstrukturen zurück (beste Schaltkreis-Arm nach Kalibrierfehler je Länge; Gewinn = Minimum über Längen)."""
    je_arm, kal = {}, {a: 0.0 for a in SCH}
    for a in SCH:
        g_m, g_e = [], []
        for L in laengen:
            X = A[L]
            gegner_m = [X["orth_zufall"][metrik], X["vorz_had_gb_" + a][metrik]]
            d = (X[a][metrik] - max(gegner_m)) if hoeher_besser else (min(gegner_m) - X[a][metrik])
            g_m.append(100 * d if metrik == "acc" else d)
            g_e.append(100 * (1 - X[a]["ausgabefehler_rel"] / min(X["orth_zufall"]["ausgabefehler_rel"], X["vorz_had_gb_" + a]["ausgabefehler_rel"])))
            kal[a] += X[a]["kal_fehler"]
        je_arm[a] = {"gewinn_metrik_je_laenge": g_m, "gewinn_fehler_prozent_je_laenge": g_e}
    best = min(kal, key=kal.get)
    einheit = "pp" if metrik == "acc" else "bpc"
    return {"gewaehlter_arm_nach_kalibrierfehler": best, "je_arm": je_arm,
            "ampel": ampel(min(je_arm[best]["gewinn_metrik_je_laenge"]), schwelle_m, einheit, f"{best} gegen max(orth_zufall, vorz_had_gb_{best}), Minimum über beide Längen"),
            "ampel_fehler": ampel(min(je_arm[best]["gewinn_fehler_prozent_je_laenge"]), 5.0, "% weniger Ausgabefehler", f"{best} gegen min(Fehler orth_zufall, vorz_had_gb_{best}), Minimum über beide Längen")}


# ------------------------------------------------------------------ MLP
def mlp_teil(seed, schnell):
    eps = (1, 4) if schnell else (3, 12)
    steps = 60 if schnell else STEPS
    xtr, ytr, xva, yva = mlp_daten(schnell)
    res, A = {}, {}
    for ep in eps:
        t0 = time.time()
        torch.manual_seed(seed)
        netz = kb.baue_bild([512, 512])
        mlp_train(netz, xtr, ytr, seed, ep, 1e-3)
        a0 = acc(netz, xva, yva)
        with torch.no_grad():
            z0 = netz(xva)
        print(f"Z2 MLP s{seed} ep{ep} fp32 {a0:.4f}", flush=True)
        # Kovarianzen der Schicht-Eingänge (Kalibrierung: Train, Bewertung: Val)
        def eingaenge(X):
            out, h = [], X
            with torch.no_grad():
                for k, l in enumerate(netz.ls):
                    out.append(h)
                    h = l(h)
                    if k < len(netz.ls) - 1:
                        h = F.relu(h)
            return out
        Xk = eingaenge(xtr[:4000])
        Xv = eingaenge(xva)
        Scal = [kovarianz(x, x.shape[1]) for x in Xk]
        Sval = [kovarianz(x, x.shape[1]) for x in Xv]
        Ws = [l.weight.data.clone() for l in netz.ls]

        def bewerte(eff):
            q = copy.deepcopy(netz)
            for l, W in zip(q.ls, eff):
                l.weight.data = W.clone()
            q.eval()
            with torch.no_grad():
                zq = q(xva)
            return {"acc": acc(q, xva, yva), "logit_fehler_rel": float((zq - z0).norm() / z0.norm())}

        arme = ptq_arme(Ws, Scal, Sval, bewerte, seed, steps)
        A[ep] = arme
        res[ep] = {"acc_fp32": a0, "bytes_fp16": kb.bytes_fp16_linear(netz), "arme": arme, "sek_laenge": time.time() - t0}
        print(f"Z2 MLP s{seed} ep{ep} fertig {time.time() - t0:.0f}s", flush=True)
    ap = ampel_paket(A, "acc", True, 0.5, eps)
    return {"epochen": eps, "schritte_lernen": steps, "laengen": res, **ap}


# ------------------------------------------------------------------ Byte-LM
def lm_teil(seed, schnell):
    Ts = (100, 300) if schnell else (800, 2400)
    steps = 60 if schnell else STEPS
    tr, va, _ = BL.daten(schnell)
    d = 128
    res, A = {}, {}
    for T in Ts:
        t0 = time.time()
        torch.manual_seed(seed)
        netz = BL.LM(d, lambda dd, i: BL.FFNDicht(dd, 4 * dd, "gelu"))
        opt = torch.optim.Adam(netz.parameters(), lr=2e-3)
        warm = max(1, T // 20)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, T - warm))))
        g = torch.Generator().manual_seed(seed + 5)
        offs = torch.arange(BL.KONTEXT)
        netz.train()
        for _ in range(T):
            i = torch.randint(0, len(tr) - BL.KONTEXT - 1, (BL.BATCH,), generator=g)
            l = netz.nll(tr[i[:, None] + offs].long(), tr[i[:, None] + offs + 1].long()).mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(netz.parameters(), 1.0)
            opt.step()
            sch.step()
        netz.eval()
        vv = va[:BL.VAL_BYTES]
        b0 = BL.bpc(netz, vv)
        lins = [m for b in netz.bl for m in (b.ffn.a, b.ffn.b)]
        # Eingänge der FFN-Matrizen: Kalibrierung aus Trainings-, Bewertung aus Validierungsbytes
        def sammle(text, nb):
            buf = [[] for _ in lins]
            hs = [m.register_forward_hook(lambda mod, inp, out, k=k: buf[k].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))) for k, m in enumerate(lins)]
            gg = torch.Generator().manual_seed(seed + 77)
            with torch.no_grad():
                for _ in range(nb):
                    i = torch.randint(0, len(text) - BL.KONTEXT - 1, (BL.BATCH,), generator=gg)
                    netz.nll(text[i[:, None] + offs].long(), text[i[:, None] + offs + 1].long())
            for h in hs:
                h.remove()
            return [kovarianz(torch.cat(b), b[0].shape[1]) for b in buf]
        Scal, Sval = sammle(tr, 16), sammle(vv, 16)
        Ws = [m.weight.data.clone() for m in lins]
        print(f"Z2 LM s{seed} T{T} fp32 bpc {b0:.4f}  (Training {time.time() - t0:.0f}s)", flush=True)

        def bewerte(eff):
            q = copy.deepcopy(netz)
            ql = [m for b in q.bl for m in (b.ffn.a, b.ffn.b)]
            for m, W in zip(ql, eff):
                m.weight.data = W.clone()
            return {"bpc": BL.bpc(q, vv)}

        arme = ptq_arme(Ws, Scal, Sval, bewerte, seed, steps)
        A[T] = arme
        res[T] = {"bpc_fp32": b0, "sek_laenge": time.time() - t0, "arme": arme,
                  "hinweis": "nur FFN-Matrizen (8) quantisiert und gezählt; Attention, Einbettung, Normen bleiben fp32"}
    ap = ampel_paket(A, "bpc", False, 0.005, Ts)
    return {"T": Ts, "schritte_lernen": steps, "laengen": res, **ap}


def main(seed: int, schnell: bool = False) -> dict:
    schnell = schnell or SCHNELL
    out = {"mlp": mlp_teil(seed, schnell)}
    m = out["mlp"]
    out["ampel"], out["ampel_fehler"] = m["ampel"], m["ampel_fehler"]
    print("Z2 MLP Ampel", m["ampel"]["farbe"], m["ampel"]["gewinn"], "| Fehler-Ampel", m["ampel_fehler"]["farbe"], m["ampel_fehler"]["gewinn"], "| Arm", m["gewaehlter_arm_nach_kalibrierfehler"], flush=True)
    if LM_AN:
        out["lm"] = lm_teil(seed, schnell)
        l = out["lm"]
        out["ampel_lm"], out["ampel_lm_fehler"] = l["ampel"], l["ampel_fehler"]
        print("Z2 LM Ampel", l["ampel"]["farbe"], l["ampel"]["gewinn"], "| Fehler-Ampel", l["ampel_fehler"]["farbe"], l["ampel_fehler"]["gewinn"], flush=True)
    return out


if __name__ == "__main__":
    rahmen("Z2_schaltkreis_rot", main)
