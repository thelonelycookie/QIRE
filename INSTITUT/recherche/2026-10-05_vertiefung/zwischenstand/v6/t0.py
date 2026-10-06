from common import *
t=time.time(); xtr,ytr,xte,yte=daten("mnist",980,1000,2000); print(xtr.shape,time.time()-t)
U=bent_U(8,8); print(U.shape, float((U@U.T-torch.eye(256)).abs().max()))
U4=bent_U(4,4); print(float((U4@U4.T-torch.eye(256)).abs().max()))
t=time.time(); c,s,med,nb=clifford_suche(xtr,ytr,980,steps=100,politur=20); print(s,med,nb,time.time()-t)
for nm,UU in [("bent8",U),("bent4",U4),("search",unitaer_clifford(kanonisch(c),N))]:
    print(nm, ridge_loo(born(xtr,UU),ytr), kopf_logistisch(born(xtr,UU),ytr,born(xte,UU),yte,l2=1e-2)[0])
