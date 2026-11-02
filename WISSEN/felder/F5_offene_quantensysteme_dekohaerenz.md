# F5: Offene Quantensysteme/Dekohärenz

Stand Welle 1, Agent a1 (≈ 10 Suchen, F5 und F1 teils gemeinsam). Nur Abstracts/Suchauszüge, kein Volltext.

## Landkarte
- **NN für Lindblad (Richtung Physik):** RBM-Dichtematrix [arXiv:1902.05131]. Überträgt nichts auf klassische NN.
- **Dissipative Quanten-NN:** Beer-Perzeptron (CPTP-Schichten), Trainierbarkeit bestritten [arXiv:2005.12458]; dissipativer Klassifikator [arXiv:2307.12293].
- **Rauschen schadet:** Noise-induced Barren Plateaus [arXiv:2007.14384], [arXiv:2402.08475].
- **Rauschen nützt:** Quanten-Reservoir-Computing mit Rauschen/Dämpfung [arXiv:2207.07924], [arXiv:2409.07886]; Gegenstimme: Kohärenz hilft monoton [arXiv:2409.17734], [arXiv:2504.17837].
- **Klassisch übertragen: Rauschen als Regularisierer:** Gewichtsrauschen schätzt Hessian-Spur, flache Minima [arXiv:2306.08553]; Rauschinjektion für Niedrigbit [arXiv:2311.14114], [arXiv:2608.20988].
- **Hardware-Rauschen (analog CIM):** rauschbewusstes Training [arXiv:2508.11940], QAT-Robustheit [arXiv:2501.14531].
- Lindblad/Dekohärenz wurde nach diesen Funden nicht als Methode auf klassische NN übertragen (nur Metapher); keine direkte Übertragung gefunden.

## Stärkste Baselines/Gegner für QIRE
- Rauschinjektion/Noise-aware QAT [arXiv:2508.11940], [arXiv:2311.14114].
- Clipping + RandBET [arXiv:2006.13977].
- Quantisierung mit Gruppenskalen (Z3).

## Wichtige Zahlen (ungeprüft)
- Entkoppelter STE: bis +5,3 % Genauigkeit, 2,2x schneller [arXiv:2508.11940].
- Jacobi-Rauschen: bis +9 % / +37 % bei W4A4 [arXiv:2608.20988].
- Noise Stability: bis +2,4 % [arXiv:2306.08553].

## Quantenbezug
- Echt quanten: Lindblad, CPTP-Kanäle, Reservoir-Experimente auf Hardware.
- Analogie: Dekohärenz als „Rauschen, das Information in Basis-Diagonale schiebt“, Born-Quadrat als Messung (Dephasierung). Die Born-Auslese in QIRE ist eine Dephasierungs-Messung, klassisch simulierbar (I-038).
- Hebel: Rauschtraining als billige „Dekohärenz-Regularisierung“; passt zu Z3 (Rauschtyp R1, Rotation macht Rauschen fast additiv).

## Bezug zu eigenen Befunden
- Z3: Rotation macht Quantisierungsrauschen additiv (Steigung 0,10 vs. roh 0,91), bringt aber keine Qualität gegen Gruppenskalen. Das passt zur Literatur: Rauschstruktur ist nicht entscheidend, Gegner gewinnen anders.
- Mögliche Spannung: Rauschtraining-Gewinne (bis +9 %) stammen aus schwachen Baselines; unsere fairen Gegner sprechen für Vorsicht.

## Lücken/offene Fragen
- Keine Arbeit gefunden zu Lindblad-Generatoren als Schichten klassischer NN.
- Kein Vergleich Rauschinjektion gegen Rotation bei gleichem Bit-Budget gefunden.
- Kein Volltext gelesen.

## Runde 2 (Welle 3, c3; nur Abstracts, Zahlen ungeprüft)
- **Stochastische Rundung im Training:** BF16 mit SR schlägt Mischpräzision, impliziter Regularisierungseffekt unter Adam [arXiv:2502.20566]. SR ist der naheliegende Gegner zu „Rauschtraining".
- **Rauschinjektion vor PTQ:** mittelwertfreies Differential-Rauschen erzeugt flache Minima [arXiv:2511.01462]; Standard-Injektion verzerrt das Ziel. Ergänzt [arXiv:2306.08553].
- **Analog-Hardware:** Lipschitz-Regularisierung + Kompensation [arXiv:2211.14917]. Keine Arbeit gefunden, die Hadamard-Rotation unter analogem Gewichtsrauschen untersucht.
- **Dephasierungs-Schicht:** keine Arbeit zu Lindblad-/Dephasierungs-Schichten in klassischen NN gefunden. Nächster klassischer Nachbar: Dropout, mit Theorie in [arXiv:2505.07792]. Bestätigt die langweiligste Erklärung (Dämpfung/Dropout).
- **SDC:** Fehler im Training können wie Rauschen wirken oder Gradienten zerstören [arXiv:2604.00726]; Randthema.

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1, a1 (F1/F5 gemeinsam, ≈ 10 Suchen) | siehe Welle 1 | hoch |
| Welle 3, c3 Runde 2 (F5, gemeinsame Suchen) | 5 (F5) | mittel, Funde bestätigen eher bekannte Linien (Regularisierung, Dropout) |

## Schlüsse und neue Wege (Agent)
1. **Rauschtraining (R1-Typ) in QIRE-Trainings als Regularisierer testen**, Gegner Standard-QAT. Beleg: [arXiv:2306.08553], [arXiv:2311.14114]. Langweiligste Erklärung: übliche Regularisierung, kein Quanteneffekt. Quantenbezug: Analogie (Dekohärenz).
2. **Lindblad-artige Schicht** (Mischung + Dephasierung = Diagonale abdämpfen) als formale Brücke zur Born-Auslese. Beleg: [arXiv:2307.12293]. Langweiligste Erklärung: entspricht Dämpfung/Dropout. Quantenbezug: echte Formalisierung, klassisch simulierbar.
3. **Nicht behaupten**, dass Rauschen hilft, ohne Gegner mit Clipping/Gruppenskalen. Beleg: [arXiv:2409.17734] widerspricht [arXiv:2207.07924].
