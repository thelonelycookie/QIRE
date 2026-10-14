import math
L=4
print("Kostenmodell (Flops je Bild und Winkelsatz, volle 2^n-Born-Auslese), L=4")
print("SV_gen = 3 L n N ; SV_clif = (L n + 2k) N ; Surr_amort = 2^(k+1) N ; Surr_frisch = 2^k L n N + 2^(k+1) N")
for n in (8,16,32):
    N=2**n
    print(f"\nn={n}, N=2^{n}={N:.3g}")
    be={}
    for name,f in [("amort_vs_SVclif",lambda k:(2**(k+1))<= (L*n+2*k)),("amort_vs_SVgen",lambda k:(2**(k+1))<=3*L*n),
                   ("frisch_vs_SVclif",lambda k:(2**k*L*n+2**(k+1))<=(L*n+2*k)),("frisch_vs_SVgen",lambda k:(2**k*L*n+2**(k+1))<=3*L*n)]:
        ks=[k for k in range(0,40) if f(k)]
        be[name]=max(ks) if ks else None
    print(" groesstes k, bei dem das exakte Surrogat noch NICHT teurer ist:",be)
    print("  k | 3^k Frequenzen | 2^k Zweige | Zahlen im datenunabh. Surrogat (3^k N N(N+1)/2, n<=16) | Zahlen Zweige-Form (2^k * N*N/…)")
    for k in (0,1,2,4,8,16,32):
        q=3**k*N*(N*(N+1)//2) if n<=16 else float('nan')
        print(f"  {k:2d} | {3**k:.3g} | {2**k:.3g} | {q:.3g}")
    # Abtast-Auslese: SV = (L n)*N ; Stabilisatorsumme: 2^k * c n^2 (c=1) je Amplitude/Stichprobe
    ks=[k for k in range(0,60) if 2**k*n*n <= L*n*N]
    print(" Abtast-/Einzelamplitude: Surrogat 2^k n^2 <= SV L n 2^n  bis k =",max(ks))
