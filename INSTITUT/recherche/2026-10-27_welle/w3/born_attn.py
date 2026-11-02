"""w3: Born-Regel-Attention (Erkundung, Tuning-Seeds 920-924). Byte-LM UD_German-GSD, 1 Thread."""
import math, sys, time, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
D = "/home/user/QIRE/INSTITUT/daten/"
def lade(t):
    z = [l[9:].rstrip("\n") for l in open(D+f"de_gsd_{t}.conllu", encoding="utf-8") if l.startswith("# text = ")]
    return torch.from_numpy(np.frombuffer(("\n".join(z)+"\n").encode(), np.uint8).copy()).long()
TR, VA, TE = lade("train"), lade("dev"), lade("test")
TE = TE[:65536]; VA = VA[:32768]

def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)

class Attn(nn.Module):
    def __init__(s, d, h, art, m=8, seed=0):
        super().__init__()
        s.h, s.hd, s.art, s.d = h, d//h, art, d
        s.qkv = nn.Linear(d, 3*d); s.o = nn.Linear(d, d)
        s.m = m
        if art == "hada":   # feste Zufalls-Hadamard-Projektion hd -> m (Vorzeichen D, H, feste Zeilenauswahl)
            g = torch.Generator().manual_seed(1234+seed)
            sg = torch.randint(0, 2, (s.hd,), generator=g).float()*2-1
            idx = torch.randperm(s.hd, generator=g)[:m]
            P = (hadamard(s.hd)*sg)[idx] * math.sqrt(s.hd/m)   # m x hd
            s.register_buffer("P", P)
    def feat_dim(s):
        return {"soft": 0, "born": s.hd**2, "elu": s.hd, "hada": s.m**2}[s.art]
    def phi(s, x):
        if s.art == "elu": return F.elu(x)+1
        if s.art == "hada": x = x @ s.P.T
        return (x.unsqueeze(-1)*x.unsqueeze(-2)).flatten(-2)       # x (x) x
    def kern(s, q, k):   # nichtnegativer Kern der Paare, q,k: B,H,N,hd -> B,H,N,N
        if s.art == "born": return (q@k.transpose(-1, -2))**2
        if s.art == "hada": return ((q@s.P.T)@(k@s.P.T).transpose(-1, -2))**2
        if s.art == "elu": return F.elu(q).add(1)@(F.elu(k)+1).transpose(-1, -2)
    def forward(s, x, mode="par", chunk=64):
        B, N, _ = x.shape
        q, k, v = s.qkv(x).view(B, N, 3, s.h, s.hd).permute(2, 0, 3, 1, 4)
        if s.art == "soft":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif mode == "par":     # O(N^2): mathematisch identisch, schneller beim Training
            A = s.kern(q, k).tril()
            y = (A@v)/(A.sum(-1, keepdim=True)+1e-6)
        else:                   # O(N*F): kausal per Praefixsumme (Chunks)
            fq, fk = s.phi(q), s.phi(k)
            S = torch.zeros(B, s.h, fq.shape[-1], s.hd+1)
            vv = torch.cat([v, torch.ones_like(v[..., :1])], -1)    # letzte Spalte = Nenner
            msk = torch.ones(chunk, chunk).tril()
            outs = []
            for a in range(0, N, chunk):
                qc, kc, vc = fq[:, :, a:a+chunk], fk[:, :, a:a+chunk], vv[:, :, a:a+chunk]
                c = qc.shape[2]
                o = qc@S + ((qc@kc.transpose(-1, -2))*msk[:c, :c])@vc
                outs.append(o); S = S + kc.transpose(-1, -2)@vc
            o = torch.cat(outs, 2); y = o[..., :-1]/(o[..., -1:]+1e-6)
        return s.o(y.transpose(1, 2).reshape(B, N, s.d))

class Block(nn.Module):
    def __init__(s, d, h, art, seed):
        super().__init__()
        s.l1, s.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.a = Attn(d, h, art, seed=seed)
        s.f = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, **kw):
        x = x+s.a(s.l1(x), **kw); return x+s.f(s.l2(x))

class LM(nn.Module):
    def __init__(s, art, d=64, h=4, L=2, ctx=512, seed=0):
        super().__init__()
        s.emb = nn.Embedding(256, d); s.pos = nn.Embedding(ctx, d)
        s.bl = nn.ModuleList([Block(d, h, art, seed) for _ in range(L)]); s.ln = nn.LayerNorm(d)
        s.d, s.h, s.L, s.art = d, h, L, art
    def forward(s, x, **kw):
        z = s.emb(x)+s.pos(torch.arange(x.shape[1]))
        for b in s.bl: z = b(z, **kw)
        return s.ln(z)@s.emb.weight.T

def flops_token(m, N):
    """Vorwaerts-FLOPs/Token (2 je MAC), ohne LayerNorm/Aktivierungen; Attention mit mittlerer Kausallaenge N/2 bzw. Zustandsform."""
    d, hd, h = m.d, m.d//m.h, m.h
    lin = 2*(3*d*d + d*d + 8*d*d)           # qkv, o, ffn
    a = m.bl[0].a; Fd = a.feat_dim()
    if m.art == "soft": att = h*(2*hd*N/2*2)         # QK + AV
    else:               att = h*(2*Fd*(hd+1)*2 + Fd)  # Zustand aktualisieren + lesen + Merkmal
    return m.L*(lin+att) + 2*d*256, m.L*att

def batch(daten, B, N, g):
    i = torch.randint(0, len(daten)-N-1, (B,), generator=g)
    return torch.stack([daten[j:j+N+1] for j in i.tolist()])

@torch.no_grad()
def bpc(m, daten, N, bs=8):
    m.eval(); n = (len(daten)-1)//N
    X = daten[:n*N].view(n, N); Y = daten[1:n*N+1].view(n, N); t = 0.
    for a in range(0, n, bs):
        lg = m(X[a:a+bs]); t += F.cross_entropy(lg.reshape(-1, 256), Y[a:a+bs].reshape(-1), reduction="sum").item()
    m.train(); return t/(n*N)/math.log(2)

def train(art, N, seed, steps, lr, tokens=1024):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = LM(art, ctx=N, seed=seed); opt = torch.optim.Adam(m.parameters(), lr=lr)
    B = tokens//N; t0 = time.time()
    for s in range(steps):
        f = min(1, (s+1)/(0.05*steps)) * (0.5*(1+math.cos(math.pi*s/steps)) if s > 0.05*steps else 1)
        for p in opt.param_groups: p["lr"] = lr*f
        b = batch(TR, B, N, g); lg = m(b[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m, time.time()-t0

def zeit(m, N, B=2, rep=5, mode="par"):
    x = torch.randint(0, 256, (B, N)); m.eval(); ts = []
    with torch.no_grad():
        m(x, mode=mode)
        for _ in range(rep):
            t = time.perf_counter(); m(x, mode=mode); ts.append(time.perf_counter()-t)
    return float(np.median(ts))/(B*N)*1e6      # us/Token

if __name__ == "__main__":
    cmd = sys.argv[1]
    ARME = ["soft", "born", "elu", "hada"]
    if cmd == "test":     # Aequivalenz par == Chunk-Praefixsumme, Kausalitaet
        for art in ARME[1:]:
            m = LM(art, ctx=256).eval(); x = torch.randint(0, 256, (2, 200))
            with torch.no_grad():
                d = (m(x, mode="par")-m(x, mode="seq")).abs().max().item()
                x2 = x.clone(); x2[:, 150:] = 0
                c = (m(x, mode="seq")[:, :150]-m(x2, mode="seq")[:, :150]).abs().max().item()
            print(art, "par-vs-seq", d, "Kausalitaet", c)
        for art in ARME: print(art, "Params", sum(p.numel() for p in LM(art).parameters()), flops_token(LM(art), 128))
    if cmd == "zeit":     # Laufzeit/FLOPs je Kontextlaenge, nur Vorwaerts
        res = {}
        for N in (128, 512, 2048, 4096):
            for art in ARME:
                m = LM(art, ctx=N); fl, fa = flops_token(m, N)
                res[f"{art}_{N}"] = dict(us_seq=zeit(m, N, mode="seq", rep=3 if N > 1000 else 5), us_par=zeit(m, N, mode="par", rep=3 if N > 1000 else 5) if (art == "soft" or N <= 2048) else None, flops=fl, flops_att=fa)
                print(art, N, res[f"{art}_{N}"], flush=True)
        json.dump(res, open(os.path.dirname(__file__)+"/zeit.json", "w"), indent=1)
    if cmd == "lauf":     # bpc: Argumente: ctx steps lr seeds-komma arme-komma val(0/1)
        N, steps, lr = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
        seeds = [int(x) for x in sys.argv[5].split(",")]; arme = sys.argv[6].split(","); val = sys.argv[7] == "1"
        for art in arme:
            for sd in seeds:
                m, t = train(art, N, sd, steps, lr)
                r = dict(art=art, N=N, seed=sd, steps=steps, lr=lr, bpc=bpc(m, VA if val else TE, N), val=val, trainzeit=t)
                print(json.dumps(r), flush=True)
                open(os.path.dirname(__file__)+"/ergebnisse.jsonl", "a").write(json.dumps(r)+"\n")
