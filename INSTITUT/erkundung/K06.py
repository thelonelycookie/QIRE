"""K06 (Erkundung, kein Befund): Haar/Walsh-Basis fuer Gewichte, GLEICHE BYTES.
Trainiertes MNIST- und Fashion-MLP 784-512-512-10 (kb.baue_bild, 30 Epochen Adam, lr 1e-3, Seed = Argument; schnell: 2000 Bilder, 3 Epochen), danach OHNE Nachtraining
Gewichte je Schicht nach einem Byte-Budget  Bt = o·1024·b/8 + 2·o + 1024/8  (b = 2 oder 3 Bit; Konvention von I-048: Eingang 784 -> 1024 aufgefuellt, 2 B Skala je Zeile, Rotationsvektor erlaubt)
ersetzt. Alle Arme duerfen hoechstens Bt Bytes (Bias fp16 in allen Armen gleich, nicht gezaehlt). Ausgewiesen: tatsaechliche Bytes, relativer Fehler, Test-Genauigkeit.
Zwei Umfaenge: "alle" (3 Schichten ersetzt) und "W1" (nur Eingangsschicht, Rest fp32).
Arme:
  roh, had, walsh   Zeilenweise MSE-Skala-Rundung (kb.quant_rows) im Originalraum / mit Zufalls-Vorzeichen-Hadamard (I-048-Arm) / mit Hadamard ohne Vorzeichen
  orth              Zufalls-Orthogonale (Haar-verteilt) + quant_rows (Kontrolle)
  haar_unif         Haar-Basis, Zeilenskala (2 B), je Haar-Stufe eine Matrixskala (MSE), alle Stufen b Bit    (Kontrolle fuer Bitverteilung)
  haar_alloc        wie haar_unif, aber Bits je Stufe nach Varianz verteilt (gierig, gleiche Summe n·b)      <- Haupt-Arm "Haar-Bitallokation"
  orth_alloc        Zufalls-Orthogonale, Bits nach EIGENER Stufen-Varianz (Kontrolle: Basis ohne Struktur)
  orth_profil       Zufalls-Orthogonale mit dem Bit-Profil der Haar-Basis (Kontrolle: gleiche Bitverteilung)
  topm16_{haar,walsh,betrag,orth}   je Zeile die m betragsgroessten Koeffizienten (fp16-Wert + 10-Bit-Index), m aus dem Budget
  topm4_{...}       wie oben mit 4-Bit-Werten (Zeilenskala), mehr Koeffizienten
  svd               Niedrigrang fp16 (Truncated-SVD) mit gleichen Bytes
  haar2d_unif/alloc, topm16_haar2d   nur W1: 2D-Haar auf dem 28x28-Bild (in 32x32 eingebettet)
Flops: Inferenz-MACs unveraendert (dichte Auswertung der rekonstruierten Matrix); Zusatzoperationen der schnellen Transformation werden je Arm ausgewiesen.
Aufruf: python3 erkundung/K06.py --seed 950 [--schnell]"""
import math
import time

import _hilfe_k0508 as H
import torch
import torch.nn.functional as F

kb = H.kb
LR, EP_VOLL, EP_SCHNELL = 1e-3, 30, 3
BITS = (2, 3)
NPAD = 1024


# ------------------------------------------------------------------ Basen
def haar(n):
    M = torch.ones(1, 1)
    while M.shape[0] < n:
        k = M.shape[0]
        M = torch.cat([torch.kron(M, torch.tensor([[1.0, 1.0]])), torch.kron(torch.eye(k), torch.tensor([[1.0, -1.0]]))], 0) / math.sqrt(2)
    return M


def haar_stufen(n):
    """Stufe je Zeile der Haar-Matrix: 0 = Mittel, j = Zeilen [2^(j-1), 2^j)."""
    return torch.tensor([0] + [int(math.floor(math.log2(r))) + 1 for r in range(1, n)])


def basis(art, n, seed, i_real=None):
    """(Q, Stufen, Zusatz-Additionen je Eingangsvektor). Zeilen von Q = Basisvektoren; Koeffizienten c = w @ Q.T."""
    if art == "haar":
        return haar(n), haar_stufen(n), 2 * n
    if art == "haar2d":
        h = haar(32)
        s = haar_stufen(32)
        return torch.kron(h, h), torch.maximum(s[:, None], s[None, :]).reshape(-1), 4 * 32 * 32
    if art == "walsh":
        return kb.hadamard(n), None, n * int(math.log2(n))
    if art == "had":
        return kb.rot_matrix(n, seed), None, n * int(math.log2(n))
    if art == "orth":
        return kb.orth_matrix(n, seed + 99), None, n * n
    if art == "betrag":
        return torch.eye(n), None, 0
    raise ValueError(art)


def einbetten(W, art):
    """Spalten auf Basisbreite: 1D auf 1024/512 auffuellen, 2D: 784 -> 32x32 (Zeile r, Spalte c)."""
    o, i = W.shape
    if art == "haar2d":
        X = torch.zeros(o, 32, 32)
        X[:, :28, :28] = W.reshape(o, 28, 28)
        return X.reshape(o, 1024)
    n = NPAD if i == 784 else i
    return F.pad(W, (0, n - i)) if n != i else W


def auspacken(C, art, i):
    if art == "haar2d":
        return C.reshape(-1, 32, 32)[:, :28, :28].reshape(-1, 784)
    return C[:, :i]


# ------------------------------------------------------------------ Quantisierer
def quant_stufe(X, b):
    """Matrix-globale MSE-Skala (13 Clip-Werte) fuer b Bit; b=0 -> 0, b=1 -> Vorzeichen·Mittel."""
    if b == 0 or X.numel() == 0:
        return torch.zeros_like(X)
    if b == 1:
        return torch.sign(X) * X.abs().mean()
    qmax = 2 ** (b - 1) - 1
    amax = X.abs().max().clamp_min(1e-12)
    best, bq = None, None
    for f in torch.linspace(0.3, 1.0, 15):
        s = amax * f / qmax
        Xq = (X / s).round().clamp(-qmax, qmax) * s
        e = float(((X - Xq) ** 2).sum())
        if best is None or e < best:
            best, bq = e, Xq
    return bq


def verteile(var, groesse, summe, bmax=8):
    """Gierig: Bits je Stufe; Gewinn je Bit ~ var·4^-b. Summe der Bits je Zeile = sum(groesse·b) <= summe."""
    b = [0] * len(var)
    rest = summe
    while True:
        kand = [l for l in range(len(var)) if b[l] < bmax and groesse[l] <= rest]
        if not kand:
            break
        l = max(kand, key=lambda l: var[l] * 4.0 ** (-b[l]))
        b[l] += 1
        rest -= groesse[l]
    return b


def quant_alloc(C, stufen, bits_je_stufe):
    """Zeilen auf Effektivwert normiert (Zeilenskala 2 B), je Stufe Matrixskala, bits_je_stufe Bit."""
    rms = (C.norm(dim=1, keepdim=True) / math.sqrt(C.shape[1])).half().float().clamp_min(1e-12)
    N = C / rms
    out = torch.zeros_like(N)
    for l, b in enumerate(bits_je_stufe):
        idx = stufen == l
        if bool(idx.any()):
            out[:, idx] = quant_stufe(N[:, idx], b)
    return out * rms


def stufen_var(C, stufen):
    rms = (C.norm(dim=1, keepdim=True) / math.sqrt(C.shape[1])).clamp_min(1e-12)
    N = C / rms
    L = int(stufen.max()) + 1
    return [float((N[:, stufen == l] ** 2).mean()) for l in range(L)], [int((stufen == l).sum()) for l in range(L)]


def topm(C, m, wertbits):
    idx = C.abs().topk(m, dim=1).indices
    K = torch.zeros_like(C)
    K.scatter_(1, idx, C.gather(1, idx))
    if wertbits == 16:
        return K.half().float()
    Kq = kb.quant_rows(K, wertbits)
    return torch.where(K != 0, Kq, torch.zeros(()))


# ------------------------------------------------------------------ Ein Arm fuer eine Matrix
def ersetze(W, art, b, seed):
    """Gibt (W_hat, Bytes, Zusatzoperationen) zurueck. W: (o, i)."""
    o, i = W.shape
    n = NPAD if i == 784 else i
    Bt = kb.bytes_q_linear(o, i, b, True, pad=True)           # inkl. 2 B Skala je Zeile und n/8 B Vorzeichen (Biase nicht, siehe oben)
    Bt -= 2 * o                                                  # bytes_q_linear zaehlt Skala + Bias; Bias gleich in allen Armen -> raus
    typ, _, bas = art.partition(":")
    if typ == "roh":
        What = kb.quant_rows(W, b)
        return What, o * i * b / 8 + 2 * o, 0
    if typ in ("had", "walsh", "orth"):
        Q, _, add = basis(typ, n, seed)
        C = einbetten(W, typ) @ Q.T
        return auspacken(kb.quant_rows(C, b) @ Q, typ, i), o * n * b / 8 + 2 * o + (n / 8 if typ == "had" else 0), add
    if typ.startswith("haar") or typ in ("orth_alloc", "orth_profil", "haar2d_unif", "haar2d_alloc"):
        zweid = typ.startswith("haar2d")
        if typ.startswith("haar"):
            Q, st, add = basis("haar2d" if zweid else "haar", n, seed)
        else:
            Q, _, add = basis("orth", n, seed)
            st = None
        C = einbetten(W, "haar2d" if zweid else "haar") @ Q.T
        nn_ = C.shape[1]
        if st is None:                                           # orth: gleiche Stufengroessen wie Haar (nur Gruppierung), Bitprofil aus Haar
            st = haar_stufen(nn_)
        summe = nn_ * b
        if typ.endswith("unif"):
            bits = [b] * (int(st.max()) + 1)
        elif typ == "orth_profil":
            Qh, sth, _ = basis("haar", n, seed)
            Ch = einbetten(W, "haar") @ Qh.T
            bits = verteile(*stufen_var(Ch, sth), summe)
        else:
            bits = verteile(*stufen_var(C, st), summe)
        Chat = quant_alloc(C, st, bits)
        bytes_ = sum(bits[l] * int((st == l).sum()) for l in range(len(bits))) / 8 * o + 2 * o + 2 * len(bits) + len(bits)
        return auspacken(Chat @ Q, "haar2d" if zweid else "haar", i), bytes_, add
    if typ.startswith("topm"):
        wb = int(typ[4:])
        Q, _, add = basis(bas if bas != "haar2d" else "haar2d", n, seed)
        C = einbetten(W, bas) @ Q.T
        nn_ = C.shape[1]
        zeile = n * b + 16                                       # Bits je Zeile: Gewichtsbudget + Skalenplatz
        ib = math.ceil(math.log2(nn_))
        m = int((zeile - (16 if wb != 16 else 0)) // (wb + ib))
        Chat = topm(C, m, wb)
        return auspacken(Chat @ Q, bas, i), o * (m * (wb + ib) + (16 if wb != 16 else 0)) / 8, add
    if typ == "svd":
        gew = (Bt - 0) * 8                                       # Bits fuer Gewichte (inkl. Skalenplatz)
        r = max(1, int(gew // (16 * (o + i))))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(r, len(S))
        Ur, Vr = (U[:, :r] * S[:r]).half().float(), Vh[:r].half().float()
        return Ur @ Vr, 2 * r * (o + i), 0
    raise ValueError(art)


def netz_mit(netz, ds, arm, b, scope, seed):
    """Kopie von netz mit ersetzten Schichten; liefert (Netz, Bytes, Zusatzops, relative Fehler je Schicht)."""
    import copy
    n2 = copy.deepcopy(netz)
    by, add, fehl = 0.0, 0, []
    for k, l in enumerate(n2.ls):
        if scope == "W1" and k > 0:
            continue
        W = l.weight.data
        # Arme mit 2D-Basis gelten nur fuer die Eingangsschicht
        a = arm
        if k > 0:
            a = arm.replace("haar2d_", "haar_").replace("haar2d", "haar")
        What, bb, ad = ersetze(W, a, b, seed * 101 + k)
        fehl.append(float((W - What).norm() / W.norm()))
        l.weight.data = What.contiguous()
        by += bb
        add += ad
    return n2, by, add, fehl


def arme_liste(scope):
    a = ["roh", "had", "walsh", "orth", "haar_unif", "haar_alloc", "orth_alloc", "orth_profil", "svd"]
    for t in ("topm16", "topm4"):
        a += [f"{t}:{x}" for x in ("haar", "walsh", "betrag", "orth")]
    if scope == "W1":
        a += ["haar2d_unif", "haar2d_alloc", "topm16:haar2d", "topm4:haar2d"]
    return a


def genauigkeit(netz, ds, klein):
    return kb.bewerte_bild(netz, ds, klein, "test")


def eine_ds(ds, seed, schnell):
    klein = schnell
    torch.manual_seed(seed)
    netz = kb.baue_bild([512, 512])
    t0 = time.time()
    inf = kb.trainiere_bild(netz, ds, seed, LR, EP_SCHNELL if schnell else EP_VOLL, klein=klein)
    t_tr = time.time() - t0
    acc_fp = genauigkeit(netz, ds, klein)
    zeilen = {}
    for scope in ("alle", "W1"):
        for b in BITS:
            for arm in arme_liste(scope):
                if scope == "alle" and "2d" in arm:
                    continue
                n2, by, add, fehl = netz_mit(netz, ds, arm, b, scope, seed)
                zeilen[f"{scope}|{b}|{arm}"] = {"acc": genauigkeit(n2, ds, klein), "bytes": by, "zusatz_ops": add, "rel_fehler": fehl}
    # Bytes-Vorgabe je Umfang/Bit (Summe der Budgets)
    ziel = {f"{s}|{b}": float(sum(kb.bytes_q_linear(o, i, b, True, pad=True) - 2 * o for k, (o, i) in enumerate(((512, 784), (512, 512), (10, 512))) if s == "alle" or k == 0)) for s in ("alle", "W1") for b in BITS}
    return {"acc_fp": acc_fp, "train_info": inf, "sekunden_training": t_tr, "ziel_bytes": ziel, "zeilen": zeilen}


def main(seed: int, schnell: bool = False) -> dict:
    t0 = time.time()
    r = {"kid": "K06", "seed": seed, "schnell": schnell, "lr": LR, "epochen": EP_SCHNELL if schnell else EP_VOLL, "modell": "MLP 784-512-512-10",
         "parameter": 784 * 512 + 512 + 512 * 512 + 512 + 5120 + 10, "macs_inferenz": 784 * 512 + 512 * 512 + 5120, "datensaetze": {}}
    for ds in ("mnist", "fashion"):
        r["datensaetze"][ds] = eine_ds(ds, seed, schnell)
    # Zusammenfassung: bester Basis-Arm gegen beste Kontrolle je Zelle
    zus = []
    for ds, d in r["datensaetze"].items():
        for scope in ("alle", "W1"):
            for b in BITS:
                z = {k.split("|")[2]: v["acc"] for k, v in d["zeilen"].items() if k.startswith(f"{scope}|{b}|")}
                kandidaten = {k: v for k, v in z.items() if k in ("haar_alloc", "haar2d_alloc", "walsh") or (k.startswith("topm") and ("haar" in k or "walsh" in k))}
                kontrollen = {k: v for k, v in z.items() if k in ("roh", "had", "orth", "svd", "haar_unif") or (k.startswith("topm") and ("betrag" in k or "orth" in k))}
                kb_ = max(kandidaten, key=kandidaten.get)
                ko_ = max(kontrollen, key=kontrollen.get)
                zus.append(f"{ds} {scope} {b}b: fp {d['acc_fp']:.4f} | bester Haar/Walsh-Arm {kb_} {z[kb_]:.4f} | beste Kontrolle {ko_} {z[ko_]:.4f}")
    r["zusammenfassung"] = "\n".join(zus)
    r["sekunden"] = time.time() - t0
    return r


if __name__ == "__main__":
    H.cli("K06", main)
