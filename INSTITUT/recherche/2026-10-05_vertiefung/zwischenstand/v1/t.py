import sys,time; sys.path.insert(0,"."); from kern import *
x,y,xt,yt=daten("mnist",900)
t=time.time(); V=V_of("16x16"); m=mask_vec("bent","16x16",np.random.default_rng(0)); P=merkmale(x,V,m); print("feat",time.time()-t)
Ff=16*P.sqrt(); t=time.time(); f=fit_softmax(Ff,y,1e-2); print("voll fit",time.time()-t)
t=time.time(); s=topm(Ff,y,10); f=fit_softmax(Ff[:,s],y,1e-2); print("top10 fit",time.time()-t)
G=gruppen_summen(P,gruppe_daten(Ff,y)); t=time.time(); f=fit_softmax(G,y,0,tau_bias=5.); print("bias fit",time.time()-t)
t=time.time(); G=gruppen_summen(P,gruppe_daten(Ff,y)); print("gruppen",time.time()-t)
