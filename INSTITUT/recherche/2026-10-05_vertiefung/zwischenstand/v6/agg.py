import json,numpy as np,glob
R=json.load(open("geo_mnist_980_981_982_983_984.json")); S=list(R.values())
ms=lambda a:f"{np.mean(a):.2f}±{np.std(a,ddof=1):.2f}"
C=["identitaet","walsh_voll","bent8","bent4block","zufall_clifford","suche_clifford","suche_k4","generisch_k32"]
print("g bei lam=1e-3 / lam=1e-1")
for c in C:
    print(c,"|",*[ms([s[c]["g"][k]["0.001"] for s in S])+" / "+ms([s[c]["g"][k]["0.1"] for s in S]) for k in ["quadratisch","rbf_med","rbf_eng","linear"]])
print("nmse,acc_kopf,acc_krrKQ")
for c in C:
    print(c,"|",*[f"{np.mean([s[c]['rkhs_nmse'][k] for s in S]):.3f}" for k in ["quadratisch","rbf_med","rbf_eng","linear"]],"|",ms([100*s[c]["acc_kopf"] for s in S]),ms([100*s[c]["acc_krr_KQ"] for s in S]))
print("klass krr acc",{k:ms([100*s["acc_klassisch_krr"][k] for s in S]) for k in S[0]["acc_klassisch_krr"]})
print("loo",ms([100*s["loo_suche"] for s in S]),ms([100*s["loo_zufallsmedian"] for s in S]),ms([100*s["loo_k4"] for s in S]))
# paired diffs of g quad
for a,b in [("suche_k4","suche_clifford"),("generisch_k32","suche_clifford"),("suche_clifford","zufall_clifford")]:
    d=[s[a]["g"]["quadratisch"]["0.001"]-s[b]["g"]["quadratisch"]["0.001"] for s in S]; print(a,b,ms(d))
# surr
for f in glob.glob("surr_n8_0_1_2_4_8_*.json"):
    D=json.load(open(f))
    for k in [0,1,2,4,8]:
        for p in ["spread","layer0","qubit0"]:
            r=[d for d in D if d["k"]==k and d["place"]==p]
            if r and k>0: print("n8 k",k,p,len(r),"supp",ms([d["support"] for d in r]),"n99",ms([d["n0.99"] for d in r]),"n90",ms([d["n0.9"] for d in r]),"eff",ms([d["eff_rang"] for d in r]),"3^k-1",3**k-1)
D=[json.loads(l) for l in open("surr16.log")]
for k in [1,2,4]:
    r=[d for d in D if d["k"]==k]; print("n16 k",k,len(r),[ (d["support"],d["n0.9"],d["n0.99"],d["n0.999"]) for d in r])
