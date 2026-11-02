# F2: Renormierungsgruppe/Multiskala

*Welle 1, a2. Nur Abstracts gelesen (arXiv-API bestätigt), kein Volltext. Alle Zahlen ungeprüft.*

## Landkarte
- **RG und tiefes Lernen (Theorie):** Tiefe entspricht Skala [arXiv:1301.3124]; jüngster RG-Rahmen für Lernkurven mit Potenzgesetz-Daten [arXiv:2510.25553]. Liefert Deutung, keine Baseline.
- **MERA/Tensornetze als Schicht:** MERA statt vollverbundener CNN-Schichten, besser als Tensor-Train [arXiv:1711.03357]; Übersicht Tensornetze in ML [arXiv:2604.14287].
- **Wavelet = MERA:** Daubechies-Wavelets liefern ein analytisches MERA [arXiv:1602.01166]; wMERA für Feldtheorien [arXiv:2404.11715]. Echte Quantenphysik-Brücke zwischen F2 und F3.
- **Wavelet-Netze:** Streunetze (Wavelet, Betrag, Mittelung, ohne Lernen) [arXiv:1203.1513, arXiv:1312.5940]; Wavelet-artige Sprachmodelle: WaveletGPT [arXiv:2409.12924], LMWT [arXiv:2504.08801], HRT [arXiv:2509.20581] (die beiden letzten Evidenz dünn).
- **Hierarchische Sprachmodelle:** Hourglass [arXiv:2110.13711], H-Net mit dynamischem Chunking [arXiv:2507.07955].
- Wer gegen wen: feste Wavelet-Mischer (WaveletGPT, Streunetz) gegen gelernte Hierarchien (Hourglass, H-Net) gegen Tensornetz-Schichten (MERA).

## Stärkste Baselines/Gegner für QIRE
- Hourglass und H-Net: gleiches Rechenbudget, klassisch, starke Sprachmodell-Evidenz.
- MERA-Schicht [arXiv:1711.03357]: direkter quanten-inspirierter Gegner zur Hadamard-Mischung.
- WaveletGPT: parameterfreier fester Mehrskalen-Mischer, fast 2-fach schneller bis gleiche Güte.

## Wichtige Zahlen (mit geprüft/ungeprüft)
- WaveletGPT: gleiche Vortrainingsgüte fast 2-fach schneller (ungeprüft).
- MERA-CNN: laut Suchauszug etwa 14000-fache Parameterreduktion, <1 % Verlust (ungeprüft, nicht im Abstract).
- H-Net: Byte-Ebene, 1 Stufe schlägt BPE-Transformer bei gleichem Aufwand; Training etwa 2-fach langsamer (ungeprüft).

## Quantenbezug
- **Echt quanten:** MERA ist ein Quantenzustands-Ansatz; Wavelet-MERA-Beweis [arXiv:1602.01166] ist exakte Physik. Die ML-Nutzung (MERA-Schicht) ist aber klassisch gerechnet.
- **Analogie:** RG-als-Tiefe, Wavelet-Mehrskalen in Sprachmodellen.
- **Hebel für QIRE:** feste, billige Mehrskalen-Mischung (Hadamard ist die Haar-nahe Mischung auf allen Skalen) mit Betrag/Quadrat-Auslese ähnelt dem Streunetz; "quanten-inspiriert" lässt sich über Wavelet-MERA-Brücke sauber begründen.

## Bezug zu eigenen Befunden (I-/Z-IDs)
- I-038 (klassische Wirkung): stützt, Streunetz und WaveletGPT erklären Mischung + Betrag ohne Quanten.
- Z3 (Rotation ≈ gleichwertig): neutral; Mehrskalen-Mischer wurden nicht gegen Hadamard getestet.

## Runde 2 (Welle 3, c4; nur Abstracts, Zahlen ungeprüft)
- **Hybride Streunetze:** festes Streunetz als Frontend [arXiv:1703.08961]; lernbare Wavelet-Parameter, Tight-Frame-Start hilft [arXiv:2107.09539]. Fest vs. gelernt: gelernte Parameter gewinnen bei wenig Daten (43,6 % vs 38,8 %, ungeprüft).
- **Haar-DWT als fester Mischer:** WaveMix [arXiv:2203.03689]. Im Suchlauf kein Vergleich Haar-Pyramide gegen Hadamard im selben Modell gefunden (Lücke bleibt). Ein 2026er WNet-Preprint (kein arXiv-Eintrag gefunden, nicht aufgenommen) argumentiert, 2-Tap-Haar-Mischer verbinden Token nie über feste Blöcke hinaus.
- H-Net/Hourglass: keine neue Quelle; Suche nach Alternativbegriffen lieferte nichts Neues.

## Lücken/offene Fragen
- Kein Volltext, kein Vergleich Hadamard vs. Haar-Wavelet-Pyramide bei gleichen Parametern in der Literatur gefunden.
- HRT/LMWT: Aussagekraft unklar (kleine Gruppen, wenige Belege).

## Sättigung: Runde | neue Quellen | Anteil neu
| Runde | neue Quellen | Anteil neu |
|---|---|---|
| Welle 1, a2 (F2, 7 Suchen) | 13 (F2) | hoch, Feld nicht gesättigt |
| Welle 3, c4 Runde 2 (F2, 3 Suchen) | 3 neue (F2) | mittel; Streunetz-Hybride neu, Haar-vs-Hadamard weiter Lücke |

## Schlüsse und neue Wege (Agent)
1. **Streunetz-Linie als Rahmen für QIRE:** Mischung + Betrag ist strukturell ein Streunetz mit Hadamard statt Wavelet. Belege: [arXiv:1203.1513, arXiv:1312.5940, arXiv:2409.12924]. Langweiligste Erklärung: reine Faltungs-/Mischungs-Nichtlinearität, nichts Quantenhaftes. Quantenbezug: nur Analogie; Wavelet-MERA [arXiv:1602.01166] ist die echte Brücke.
2. **Test: Haar-Pyramide statt Hadamard** (gleiche Parameter, gleiche Auslese). Belege: WaveletGPT, Evenbly-White. Langweiligste Erklärung: kein Unterschied (Z3-Muster). Quantenbezug: wMERA/MERA-Struktur als Begründung, Wirkung klassisch.
3. **Ehrlicher Gegner einplanen:** MERA-Schicht und Hourglass in die Vergleichstabelle, sonst fehlt der stärkste klassische/quanten-inspirierte Gegner. Langweiligste Erklärung: Hourglass gewinnt durch Rechenverteilung, nicht Mischung.
4. **(R2) Hadamard-Frontend gegen fest-vs-gelernt-Befund prüfen:** Parametric Scattering zeigt, dass gelernte Wavelet-Parameter bei wenig Daten helfen [arXiv:2107.09539]. Idee: Hadamard mit lernbarer Diagonale (Fastfood-artig) ist schon dieser Mittelweg. Langweiligste Erklärung: bessere Initialisierung, nicht Struktur. Quantenbezug: Analogie.