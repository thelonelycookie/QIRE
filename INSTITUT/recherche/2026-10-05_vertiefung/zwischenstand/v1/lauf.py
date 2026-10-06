import sys; sys.path.insert(0,"."); from kern import *
def pool(x,g): return F.adaptive_avg_pool2d(x.reshape(-1,1,16,16),g).reshape(len(x),-1)
def tanh_net(Atr,ytr,Ate,yte,h,seed):
    def train(A,y,lr,steps):
        gen=torch.Generator().manual_seed(seed); d=A.shape[1]
        W1=(torch.randn(d,h,generator=gen)/d**.5).requires_grad_(); b1=torch.zeros(h,requires_grad=True)
        W2=(torch.randn(h,10,generator=gen)/h**.5).requires_grad_(); b2=torch.zeros(10,requires_grad=True)
        o=torch.optim.Adam([W1,b1,W2,b2],lr=lr)
        for _ in range(steps):
            o.zero_grad(); F.cross_entropy(torch.tanh(A@W1+b1)@W2+b2,y).backward(); o.step()
        return lambda B: (torch.tanh(B@W1+b1)@W2+b2).detach()
    a,v=val_split(len(ytr),seed)
    lr=max([0.02,0.06],key=lambda l:acc(train(Atr[a],ytr[a],l,1500),Atr[v],ytr[v]))
    return acc(train(Atr,ytr,lr,1500),Ate,yte)
def run(seed):
    x,y,xt,yt=daten("mnist",seed); rng=np.random.default_rng(seed); R=[]
    def add(name,fam,real,bits,a,**kw): R.append(dict(seed=seed,name=name,fam=fam,real=real,bits=bits,acc=a,**kw))
    # ---- klassisch
    for g in (2,3,4,5):
        A,B=pool(x,g),pool(xt,g); a,l=head_w(A,y,B,yt,seed); add(f"pool{g}x{g}+softmax","klass",g*g*10+10,0,a,l2=l)
    add("pool5x5->3tanh->10","klass",25*3+3+30+10,0,tanh_net(pool(x,5),y,pool(xt,5),yt,3,seed))
    add("pool4x4->4tanh->10","klass",16*4+4+40+10,0,tanh_net(pool(x,4),y,pool(xt,4),yt,4,seed))
    idx=None
    for m in (5,9,10,20):
        s=topm(x,y,m); a,l=head_w(x[:,s],y,xt[:,s],yt,seed); add(f"pixel-Fisher-top{m}","klass",m*10+10,8*m,a,l2=l)
    # ---- Clifford-Zelle
    cfg=[("identitaet","16x16","keine")]
    for blk in ("2x2","4x4","8x8","16x16"):
        for kind in ("bent","zufallsform","keine"):
            if blk=="2x2" and kind=="zufallsform": continue   # identisch zu bent (Rang 2 hat nur eine Form)
            if blk=="8x8" and kind!="bent": continue
            cfg.append((blk,blk,kind))
    cfg.append(("16x16","16x16","zufallszeichen"))
    for name,blk,kind in cfg:
        if name=="identitaet":
            P,Pt=x**2/ (x**2).sum(1,keepdim=True).clamp_min(1e-8),xt**2/(xt**2).sum(1,keepdim=True).clamp_min(1e-8); mk="keine"
        else:
            m=mask_vec(kind,blk,rng); V=V_of(blk); P,Pt=merkmale(x,V,m),merkmale(xt,V,m)
        k=len(BLOCK[blk][0])*2 if name!="identitaet" else 0
        mbits={"bent":0,"keine":0,"zufallsform":k*(k-1)//2,"zufallszeichen":256}.get(kind,0)
        tag=f"{name}|{kind}"; Ff,Fft=16*P.sqrt(),16*Pt.sqrt()
        grp=gruppe_daten(Ff,y); 
        # H1: Gruppierung + Bias
        G,Gt=gruppen_summen(P,grp),gruppen_summen(Pt,grp); a,t=head_bias(G,y,Gt,yt,seed)
        add(tag+"|H1 daten-Gruppen+Bias","zelle",10,mbits+int(round(256*math.log2(10))),a,tau=t)
        gi=torch.arange(256)*10//256; G,Gt=gruppen_summen(P,gi),gruppen_summen(Pt,gi); a,t=head_bias(G,y,Gt,yt,seed)
        add(tag+"|H1i Index-Gruppen+Bias","zelle",10,mbits,a,tau=t)
        G,Gt=gruppen_summen(P,grp),gruppen_summen(Pt,grp); a,l=head_w(G,y,Gt,yt,seed)
        add(tag+"|H2 Gruppen+10x10","zelle",110,mbits+int(round(256*math.log2(10))),a,l2=l)
        for mm in (5,10,20):
            s=topm(Ff,y,mm); a,l=head_w(Ff[:,s],y,Fft[:,s],yt,seed); add(tag+f"|H3 Fisher-top{mm}","zelle",mm*10+10,mbits+8*mm,a,l2=l)
        if blk=="16x16" and kind in ("bent","keine"):
            add(tag+"|Voll 256x10","zelle",2570,mbits,acc(fit_softmax(Ff,y,1e-2),Fft,yt),l2=1e-2)
    return R
if __name__=="__main__":
    seeds=[int(s) for s in sys.argv[1:]]
    for s in seeds:
        t=time.time(); R=run(s); json.dump(R,open(f"erg_{s}.json","w")); print(s,len(R),round(time.time()-t),flush=True)
