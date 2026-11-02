# Richter A: Aussicht auf H1-Bestätigung (gleiche FLOPs/Bytes, nach genug Training)

Paarweise (i<j), Gewinner zuerst; aus konsistenter Gesamtordnung abgeleitet.

K01>K02
K03>K01
K01>K04
K05>K01
K01>K06
K07>K01
K01>K08
K01>K09
K01>K10
K01>K11
K01>K12
K13>K01
K14>K01
K01>K15
K01>K16
K01>K17
K01>K18
K19>K01
K01>K20
K03>K02
K02>K04
K05>K02
K02>K06
K07>K02
K02>K08
K02>K09
K02>K10
K02>K11
K02>K12
K13>K02
K14>K02
K02>K15
K02>K16
K02>K17
K18>K02
K19>K02
K02>K20
K03>K04
K05>K03
K03>K06
K03>K07
K03>K08
K03>K09
K03>K10
K03>K11
K03>K12
K03>K13
K03>K14
K03>K15
K03>K16
K03>K17
K03>K18
K03>K19
K03>K20
K05>K04
K06>K04
K07>K04
K04>K08
K09>K04
K04>K10
K04>K11
K12>K04
K13>K04
K14>K04
K04>K15
K04>K16
K04>K17
K18>K04
K19>K04
K04>K20
K05>K06
K05>K07
K05>K08
K05>K09
K05>K10
K05>K11
K05>K12
K05>K13
K05>K14
K05>K15
K05>K16
K05>K17
K05>K18
K05>K19
K05>K20
K07>K06
K06>K08
K06>K09
K06>K10
K06>K11
K06>K12
K13>K06
K14>K06
K06>K15
K06>K16
K06>K17
K18>K06
K19>K06
K06>K20
K07>K08
K07>K09
K07>K10
K07>K11
K07>K12
K13>K07
K14>K07
K07>K15
K07>K16
K07>K17
K07>K18
K19>K07
K07>K20
K09>K08
K10>K08
K11>K08
K12>K08
K13>K08
K14>K08
K08>K15
K08>K16
K08>K17
K18>K08
K19>K08
K20>K08
K09>K10
K09>K11
K12>K09
K13>K09
K14>K09
K09>K15
K09>K16
K09>K17
K18>K09
K19>K09
K09>K20
K11>K10
K12>K10
K13>K10
K14>K10
K10>K15
K10>K16
K10>K17
K18>K10
K19>K10
K20>K10
K12>K11
K13>K11
K14>K11
K11>K15
K11>K16
K11>K17
K18>K11
K19>K11
K20>K11
K13>K12
K14>K12
K12>K15
K12>K16
K12>K17
K18>K12
K19>K12
K12>K20
K14>K13
K13>K15
K13>K16
K13>K17
K13>K18
K19>K13
K13>K20
K14>K15
K14>K16
K14>K17
K14>K18
K19>K14
K14>K20
K15>K16
K15>K17
K18>K15
K19>K15
K20>K15
K16>K17
K18>K16
K19>K16
K20>K16
K18>K17
K19>K17
K20>K17
K19>K18
K18>K20
K19>K20

## Rangliste
1. K05: Rotation vor KV-Quantisierung ist belegt (Ausreißerglättung), Vorteil gegen Rundung ohne Rotation bei gleichen Cache-Bytes sehr wahrscheinlich; nachtrainierfrei, kein Trainingsende-Problem.
2. K03: Vektor-Quantisierung nach Rotation hat echten Granulargewinn gegenüber Skalar, I-048-Arm ist der harte Gegner, Vorteil klein aber wahrscheinlich.
3. K19: Lokalität ist ein starker, belegter Prior; bei kleinem n und FLOP-gleich schlägt sie dicht fast sicher, schrumpft bei viel Daten, aber MNIST-Regime bleibt.
4. K14: Wachstum (Net2Net/Stapeln) spart belegt Trainings-FLOPs; bei langem Training schrumpft der Vorteil, kurz aber robust.
5. K13: Kontextlängen-Warmup spart bekannt FLOPs; Gewinn verschwindet teils bei langem Training, Kontrolle fest 64 gefährlich.
6. K07: Niedrigbit-Zustände funktionieren, Rotation bringt gegen 8-Bit-Block-Adam wohl wenig Extra; Bytes-Vorteil gegen fp32 aber trivial bestätigbar.
7. K01: Lagenteilung mit Schlüsseln: gleiche Bytes verbreitert ist Gegner; Teilung kostet meist Qualität, Schlüssel helfen etwas.
8. K18: Entwurfsmodell/Akzeptanz wahrscheinlich messbar, aber Bytes-zu-Zeit auf CPU unsicher und H1 unscharf.
9. K02: Schlüssel gegen ohne Schlüssel gewinnt wohl, gegen drei getrennte Modelle gleicher Bytes eher Gleichstand.
10. K06: Zufalls-Orthogonale macht dasselbe (R1), Haar-Vorteil nur bei Eingangsschicht wahrscheinlich.
11. K12: bpc gleichauf ist keine Bestätigung; FLOP-Gewinn des Kopfs klein bei d=128.
12. K09: Normierung verschiebt Lernraten-Abhängigkeit, LayerNorm-Kontrolle dürfte gleich viel erreichen.
13. K04: Eingefrorene Zufalls-W1 verliert bei langem Training gegen gelernte gleiche Bytes (I-044/045-Lehre).
14. K20: Beschränkung hilft beim Speichern, ReLU6 und GELU bei 4 Bit sind starke Gegner.
15. K11: Fliegen-Hash schlägt dichten Kopf gleicher FLOPs selten.
16. K10: Born-Stichprobe schlägt Top-S/dicht schmal laut Vorab-P nicht; Varianzkosten.
17. K08: Feste Hadamard-Rotation im Optimierer ist kein Eigenbasis-Gewinn wie SOAP, Vorteil unwahrscheinlich.
18. K15: Dropout getunt ist starker Gegner, Rotation gibt kaum Neues.
19. K16: Tiefe plus Skip bringt Fastfood hinter dichtes Skip-Modell gleicher Operationen (I-045-Lehre).
20. K17: Zufallsmerkmale plus Ridge verlieren gegen trainierten MLP gleicher Inferenz-FLOPs.
