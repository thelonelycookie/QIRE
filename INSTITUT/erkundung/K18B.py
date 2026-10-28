"""K18B -- Zusatzarme zu K18: n-Gramm- und Prompt-Lookup-Entwurf (Erkundung Welle 3, 28.10.2026; kein Befund, nicht in LABORBUCH).
K18.py bleibt unverändert; train_lm, uebereinstimmung, spekulativ, Konstant, KID-Muster werden importiert. Gleiches Ziel (Byte-LM d=128, f=512, 4 Lagen, T=4000, schnell 40, lr 2e-3, gleicher Seed),
gleiche Maße (Greedy-Übereinstimmung über den ganzen Test, Stichproben-Akzeptanz sum min(p,q), spekulative Simulation gamma=4: 128 Starts à 64 Byte), nur Teil 1 (Akzeptanz). KEINE Zeitmessung.
ENTWÜRFE (alle ohne Gradienten, aus den Trainingsbytes gezählt bzw. aus dem Kontext):
  prompt_lookup   0 Bytes: letzte 3 Byte im bisherigen Kontext suchen (jüngster früherer Treffer), das darauf folgende Byte vorschlagen; sonst häufigstes Byte
  ngramm2/3/4     Zählmodell mit Kontext 1/2/3 Byte, Witten-Bell-Interpolation mit allen niedrigeren Ordnungen; Tabelle aus ALLEN Trainingsbytes (voll)
  ngramm3_gb/4_gb dieselben Modelle, Tabelle so beschnitten (kleinste Zählwerte weg, globale Schwelle), dass die Tabellenbytes <= Bytes des 2-Bit-Zwillings sind (GLEICHE BYTES)
  zwilling_had    Anker aus K18 (2 Bit + Hadamard, kb.quantisiere) -- gleiche Zeilenwerte wie in roh/K18_s<seed>.json
  unigramm        Anker: häufigstes Byte immer
Tabellenbytes: je Eintrag Kontext (k Byte) + nächstes Byte + Zähler (1 Byte, logarithmisch, ~9 % Auflösung) = k+2 Byte; Ordnung 1 (Unigramm) 256·2. Das Modell ist also NICHT parameterfrei; nur prompt_lookup ist es.
Zusatz: akzeptiert_pro_kb = mittlere akzeptierte Länge je KiB Entwurfsbytes (Speicher), Aussage nur über Akzeptanz pro Byte, nicht über Zeit. Speed-up-Obergrenze bei Entwurfskosten c=0: token_je_ziel_durchgang.
Lies zuerst: spekulativ.akzeptiert_mittel und greedy_uebereinstimmung der n-Gramm-Entwürfe gegen zwilling_had (Bytes beachten) und prompt_lookup."""
from __future__ import annotations
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hilfe_k17_20 as H
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import abteilungen.struktur_sprache as ss
import abteilungen.kompression_bytes as kb
import K18 as K

KID = "K18B"
GAMMA = K.GAMMA


class NGramm(nn.Module):
    """Interpolierter Byte-n-Gramm-Entwurf; forward(idx) gibt log-Wahrscheinlichkeiten (B,T,256) für das Byte NACH jeder Position."""

    def __init__(self, train, N, min_zaehler=1):
        super().__init__()
        self.N = N
        t = train.numpy().astype(np.int64)
        self.tab = []                                    # je Kontextlänge k = 1..N-1: (sortierte Schlüssel ctx*256+nxt, Zähler)
        for k in range(1, N):
            ctx = np.zeros(len(t) - k, dtype=np.int64)
            for j in range(k):
                ctx = ctx * 256 + t[j:len(t) - k + j]
            key, cnt = np.unique(ctx * 256 + t[k:], return_counts=True)
            m = cnt >= min_zaehler
            q = np.minimum(np.round(8 * np.log2(1 + cnt[m])), 255)                    # 1-Byte-Logzähler (Auflösung ~9 %, bis 2^32)
            self.tab.append((key[m], (2.0 ** (q / 8) - 1).astype(np.float32)))
        u = np.bincount(t, minlength=256).astype(np.float64)
        self.uni = torch.tensor((u + 1) / (u + 1).sum(), dtype=torch.float32)
        self.min_zaehler = min_zaehler

    def bytes(self):
        return 512 + sum(len(k) * (i + 1 + 2) for i, (k, _) in enumerate(self.tab))

    def _ml(self, k, ctx):
        """Zähler-Matrix (M,256), Summe und Zahl der Nachfolger je Kontext."""
        key, cnt = self.tab[k - 1]
        uq, inv = np.unique(ctx, return_inverse=True)
        lo = np.searchsorted(key, uq * 256)
        hi = np.searchsorted(key, uq * 256 + 256)
        ln = hi - lo
        rows = np.zeros((len(uq), 256), dtype=np.float32)
        if ln.sum() > 0:
            r = np.repeat(np.arange(len(uq)), ln)
            pos = np.repeat(lo, ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
            rows[r, (key[pos] & 255).astype(np.int64)] = cnt[pos]
        return rows[inv], ln[inv].astype(np.float32)

    def forward(self, idx):
        B, T = idx.shape
        x = idx.numpy().astype(np.int64)
        p = np.broadcast_to(self.uni.numpy(), (B * T, 256)).copy()
        for k in range(1, self.N):
            ctx = np.zeros((B, T), dtype=np.int64)
            ok = np.zeros((B, T), dtype=bool)
            ok[:, k - 1:] = True
            for j in range(k):                                           # Kontext = x[t-k+1..t]
                sh = np.zeros((B, T), dtype=np.int64)
                sh[:, k - 1 - j:] = x[:, :T - (k - 1 - j)] if k - 1 - j > 0 else x
                ctx = ctx * 256 + sh
            ctx, okf = ctx.reshape(-1), ok.reshape(-1)
            idxs = np.nonzero(okf)[0]
            rows, nf = self._ml(k, ctx[idxs])
            tot = rows.sum(1, keepdims=True)
            lam = tot / (tot + nf[:, None] + 1e-9)
            ml = rows / np.maximum(tot, 1)
            neu = np.where(tot > 0, lam * ml + (1 - lam) * p[idxs], p[idxs])
            p[idxs] = neu
        return torch.log(torch.from_numpy(p).clamp_min(1e-9)).reshape(B, T, 256)


class PromptLookup(nn.Module):
    def __init__(self, byte, n=3):
        super().__init__()
        self.byte, self.n = int(byte), n

    def forward(self, idx):
        B, T = idx.shape
        n = self.n
        nxt = torch.full((B, T), self.byte, dtype=torch.long)
        if T > n:
            W = T - n + 1
            win = idx.unfold(1, n, 1)                                    # (B,W,n): Fenster t endet bei Position t+n-1
            eq = (win[:, :, None, :] == win[:, None, :, :]).all(-1)      # (B,W,W)
            fr = torch.arange(W)
            gueltig = eq & (fr[None, :] < fr[:, None])[None]             # früherer Treffer s < t
            sc = torch.where(gueltig, fr[None, None, :] + 1, torch.zeros((), dtype=torch.long)).amax(-1)   # jüngster Treffer+1, 0 = keiner
            s = (sc - 1).clamp_min(0)
            folge = torch.gather(idx, 1, (s + n).clamp(max=T - 1))       # Byte nach dem früheren Treffer
            nxt[:, n - 1:] = torch.where(sc > 0, folge, nxt[:, n - 1:])
        lg = torch.full((B, T, 256), -10.0)
        lg.scatter_(2, nxt[..., None], 10.0)
        return lg


def bpc_entwurf(m, text):
    n = (len(text) - 1) // ss.KONTEXT
    x = text[:n * ss.KONTEXT].long().reshape(n, ss.KONTEXT)
    y = text[1:n * ss.KONTEXT + 1].long().reshape(n, ss.KONTEXT)
    s = 0.0
    with torch.no_grad():
        for i in range(0, n, 16):
            s += float(F.cross_entropy(m(x[i:i + 16]).reshape(-1, 256), y[i:i + 16].reshape(-1), reduction="sum"))
    return s / (n * ss.KONTEXT) / np.log(2)


def beschneide(train, N, bytes_ziel):
    """Kleinste globale Zählschwelle, bei der die Tabelle in bytes_ziel passt."""
    for th in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 255):
        m = NGramm(train, N, th)
        if m.bytes() <= bytes_ziel:
            return m, th
    return m, th


def main(seed: int, schnell: bool = False) -> dict:
    H.pruefe_seed(seed)
    torch.set_num_threads(1)
    t0 = time.time()
    T = 40 if schnell else 4000
    r = {"kid": KID, "seed": seed, "schnell": schnell, "T": T, "gamma": GAMMA}
    tr, va, te = ss.daten(schnell)
    ziel, st = K.train_lm(128, 512, seed, T, 2e-3, schnell)
    ziel.eval()
    r["ziel"] = {"bpc_test": kb.bewerte_lm(ziel, schnell, "test"), "parameter": int(sum(p.numel() for p in ziel.parameters())),
                 "bytes_linear_fp16": kb.bytes_fp16_linear(ziel), "bytes_rest_fp16": kb.bytes_rest_fp16(ziel), "sekunden_training": st["sekunden"]}
    zw = kb.quantisiere(ziel, 2, "had", seed)
    zw.eval()
    zw_bytes = float(sum(q.bytes() for q in zw.modules() if isinstance(q, kb.QLinear)))
    haeufig = int(torch.bincount(tr.long(), minlength=256).argmax())
    entw = {"prompt_lookup": (PromptLookup(haeufig), 0.0, {})}
    for N in (2, 3, 4):
        m = NGramm(tr, N)
        entw[f"ngramm{N}"] = (m, float(m.bytes()), {"eintraege": [len(k) for k, _ in m.tab]})
    for N in (3, 4):
        m, th = beschneide(tr, N, zw_bytes)
        entw[f"ngramm{N}_gb"] = (m, float(m.bytes()), {"zaehl_schwelle": th, "eintraege": [len(k) for k, _ in m.tab]})
    entw["zwilling_had"] = (zw, zw_bytes, {})
    entw["unigramm"] = (K.Konstant(haeufig), 0.0, {})
    r["entwuerfe"] = {}
    for k, (m, by, info) in entw.items():
        m.eval()
        t1 = time.time()
        gm, sm = K.uebereinstimmung(ziel, m, te)
        z = {"greedy_uebereinstimmung": gm, "stichprobe_akzeptanz": sm, "spekulativ": K.spekulativ(ziel, m, te, starts=32 if schnell else 128, seed=seed),
             "bytes_entwurf": by, "bytes_zwilling_referenz": zw_bytes}
        z.update(info)
        z["akzeptiert_pro_kib"] = z["spekulativ"]["akzeptiert_mittel"] / (by / 1024) if by > 0 else None
        z["speedup_obergrenze_c0"] = z["spekulativ"]["token_je_ziel_durchgang"]
        if isinstance(m, NGramm):
            z["bpc_test"] = bpc_entwurf(m, te)
        elif k == "zwilling_had":
            z["bpc_test"] = kb.bewerte_lm(m, schnell, "test")
        else:
            z["bpc_test"] = None
        z["sekunden"] = time.time() - t1
        r["entwuerfe"][k] = z
        print(f"{KID} s{seed} {k}: greedy {gm:.3f}  akz {z['spekulativ']['akzeptiert_mittel']:.3f}  Bytes {by:.0f}  {z['sekunden']:.0f}s", flush=True)
    r["hinweis"] = "Erkundung, kein Befund. Nur Akzeptanz (Teil 1 von K18); keine Zeitmessung. n-Gramm-Tabellen sind Speicher (Bytes ausgewiesen), nur prompt_lookup ist parameterfrei."
    r["sekunden_gesamt"] = time.time() - t0
    H.speichere(KID, seed, r, schnell)
    return r


if __name__ == "__main__":
    H.cli(KID, main)
