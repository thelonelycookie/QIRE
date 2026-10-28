# f1: Faden "Born-Quadrat als Aktivierung" (Welle 3, 28.10.2026)

Nichts gerechnet außer paarweise Differenzen aus vorhandenen Roh-JSONs (I-045-v2/v3/v4). Wirkung klassisch (I-038).

## 1. Kurzfassung
- Der z²-Vorteil ist **nicht konsistent**. Er hängt von Training und Breite ab und **kippt** bei langem Training und d = 512 (dicht).
- Dicht, Byte-LM, z² minus GELU (bpc, negativ = z² besser, n = 8, I-045-v3):
  d = 256: T1000 −0,045 · T4000 −0,118 · T16000 −0,013.
  d = 512: T1000 −0,189 · T4000 −0,081 · **T16000 +0,047** (z² sicher schlechter, 0/8 Seeds besser).
- Gegen ReLU² (Primer) ist reines z² bei d = 512, T = 16000 um **+0,040** schlechter. ReLU² ≈ GELU (+0,008).
  Wichtig: Der Gewinn kommt aus dem **Gate** (Vorzeichen) von ReLU², nicht aus dem reinen z².
- Zwei starke Vorbehalte, die jede Aussage "z² verliert" derzeit untergraben: (a) Lernrate nur bei T = 4000 gesucht (grobes Gitter, Faktor 3) und für T = 16000 übernommen, (b) T = 16000 sind 10 Epochen auf 1,6 MB, also Überanpassung möglich (Train-bpc fehlt).
- Gegenbefund: I-047 (TT-FFN, d = 512, T = 16000, Raven): z² schlägt GELU um **0,040 bpc**. Rohdaten fehlen im Repo, nur die Zusammenfassung aus dem Auftrag.
- Der saubere Lauf (Abschnitt 5, "I-050") entscheidet es mit Lernrate je (Arm, d, T), Normierungs- und Vorzeichenkontrollen, Train-bpc, 8 Seeds.

## 2. Fundstellen (Zahlen aus LABORBUCH und JSON)

### 2.1 Sprachmodell (Byte-LM, UD_German-GSD, 4 Lagen, FFN 4d, Batch 8×128, Adam, Kosinus; Test-bpc)
| Lauf | Bedingungen | Seeds | Befund |
|---|---|---|---|
| I-045 (Nebenarm) | d = 256, T = 1000 | 5 (nicht konfirmatorisch) | Sz (Fastfood+z²) 3,410 vs D_gelu 3,516 |
| I-045-v2 | d = 256, T = 1000, 10 Seeds | 2410–2419 | D_gelu 3,520 · D_relu2 3,485 · **D_z2 3,478** · Sz 3,403 · S_gelu 3,599 · L_z2 3,563. z²−GELU −0,043 (±0,006), relu2−GELU −0,035, z²−relu2 −0,008 (9/10 Seeds) |
| I-045-v3 | d ∈ {256, 512}, T ∈ {1000, 4000, 16000} | 2600–2607 (n = 8) | Tabelle unten |
| I-045-v4 | d = 512, FFN-FLOP-gleich (h = 580), T = 4000/16000 | 2620–2627 | Df_z2 − Df_gelu: T4000 −0,094 (±0,019), **T16000 +0,011** (±0,008, p = 0,007); Df_gelu 2,086 gegen Df_z2 2,097 |
| I-047 | TT-FFN χ = 4, d = 512, T = 16000 | 2680–2687 | H3: TT_z2 besser als TT_gelu um 0,040 bpc (nur Auftragsnotiz, JSON nicht im Repo) |

I-045-v3, Differenzen je Seed gepaart (neu berechnet aus `ergebnisse/I-045-v3__20261026_052958.json`), negativ = erstgenannter besser:
| d | T | z²−GELU | ReLU²−GELU | z²−ReLU² |
|---|---|---|---|---|
| 256 | 1000 | −0,045 (8/8) | −0,034 | −0,011 |
| 256 | 4000 | **−0,118** (8/8) | −0,084 | −0,034 |
| 256 | 16000 | −0,013 (8/8) | −0,020 | +0,007 (1/8 besser) |
| 512 | 1000 | **−0,189** (8/8) | −0,153 | −0,036 |
| 512 | 4000 | −0,081 (8/8) | −0,059 | −0,023 |
| 512 | 16000 | **+0,047** (0/8) | +0,008 | **+0,040** (0/8) |
Absolutwerte d = 512, T = 16000: D_gelu 2,106 · D_relu2 2,113 · D_z2 2,153 · (Sz 2,177). Bei d = 256, T = 16000: relu2 2,096 · z² 2,103 · gelu 2,116.
Lernraten (alle dichten Arme in v3 und v4): 1,67e-4 (= 5e-4/3, gitterverlängert), Gitter 5e-4/1,5e-3/4,5e-3 (Faktor 3), nur T = 4000, ein Tuning-Seed (2698/2699), "nicht am Rand" gemeldet.

### 2.2 Klassifikation (Born² gegen |z| und ReLU)
- I-038 (Fastfood/Haar, MNIST/Fashion 16×16, 20 Seeds 2000–2019): Born² gegen beste klassische Nichtlinearität |z|: MNIST **+0,35 pp** [0,27; 0,43], Fashion 0,00. Quadrat gegen linear: +2,21 / +1,27 pp.
- I-039-v2 (Seeds 2100–2119): Der Vorsprung kommt aus Mischung + Gruppierung, nicht aus dem Quadrat; b_ff 80,26 % gegen |z|-Variante 80,65 % (−0,39 pp, gleichwertig).
- I-041 (32×32, Seeds 2060–2069): Born-Vorsprung gegen |z| wächst nicht (MNIST HU −0,07, Fashion schrumpft −0,57).
- I-042 (N = 1024/4096/16384, Seeds 2200–2204, Fastfood fest, Rohpixel, 1 Schicht): z² (B) / |z| (C) / ReLU (D):
  CIFAR-10: N1024 40,3 / 39,3 / 46,1 · N4096 45,8 / 44,6 / 49,7 · N16384 49,1 / 48,7 / 52,3.
  Fashion32: N1024 87,4 / 87,1 / 86,8 · N4096 88,6 / 88,6 / 88,1 · N16384 89,1 / 89,5 / 88,5.
  Urteile H2: z² gegen |z| CIFAR +0,34 (BORN_BESSER), Fashion −0,36 (KLASSISCH_BESSER); z² gegen ReLU CIFAR −3,18, Fashion +0,57. "Kein Gewinner" (Laborbuch). Der z²-minus-|z|-Abstand schrumpft auf CIFAR mit N (+1,0 → +1,2 → +0,3).
- I-044 / I-044-v2: Ks (Fastfood+z², Kopf) 74,3 % Spitze nach 4 Epochen; Nichtlinearität in {z², |z|, ReLU} auf Tuning-Seed gewählt (z² im Code fest). I-044-v3: Vorsprung von Ks gegen linearen Kopf schrumpft mit Epochen (+0,37 → −0,77 → −0,51 pp). Kein reiner Aktivierungsvergleich.
- T-ternaer (Seeds 310–319): ternaer_quad − ternaer = **+6,2 pp** (Quadrat gegen keine Nichtlinearität, nicht gegen GELU).
- r3 (Erkundung, MNIST/Fashion 16×16, 20 Epochen, 10k Train, Seeds 980–984, n = 5): MNIST 5k reell z² 94,57 gegen ReLU 93,74 (+0,83), 18k 96,38 gegen 96,13 (+0,25); Fashion 5k +0,64, 18k +0,13 (innerhalb sd). Bei gleichen FLOPs (sq_F gegen relu_F): MNIST +0,58 / +0,37, Fashion −0,11 / −0,09. Vorteil schrumpft mit Datenmenge/Training.
- I-045-v2/v3/Primer-Referenz: relu² (Primer) liegt in v3 immer zwischen GELU und z² oder besser.

## 3. Konsistenz und Trends
1. **Training:** Klassifikation und LM zeigen dasselbe Muster. Der z²-Vorsprung ist am größten bei knappem Training (T = 1000–4000, 4 Epochen, 5k Daten) und schrumpft danach: LM d = 512: −0,189 → −0,081 → +0,047. Das ist die Signatur "Optimierungsvorteil im Frühregime" (wie I-044-v3, I-049).
2. **Breite:** Bei T = 1000 wächst der Vorsprung mit d (−0,045 → −0,189). Bei T = 16000 kippt er mit d (−0,013 → +0,047). Das stützt "größeres Modell braucht länger Training, bis GELU aufholt" oder "z² verliert Kapazität/Stabilität in der Breite", beides nicht getrennt.
3. **Datenmenge:** Nur indirekt (r3: Vorsprung schrumpft von 5k zu 18k). Im LM nie variiert (I-045-v5 testet es, noch nicht ausgewertet).
4. **Vorzeichen/Gate:** ReLU² ≥ z² bei langem Training. Reines z² ist symmetrisch (verliert das Vorzeichen, keine Sparsität); die Primer-Literatur nutzt gerade das Gate.
5. **Struktur:** Mit Fastfood schlägt z² bei T = 1000 (Sz), bei langem Training nicht (v3/v4). In TT (I-047) bleibt z² > GELU auch bei T = 16000. Das ist offen und spricht dafür, dass z² dort wirkt, wo die Mischung linear stark eingeschränkt ist.
6. **Lernrate ist Störfaktor:** z² ist unbeschränkt (quartisches Wachstum der Gewichtsgradienten); die optimale Rate hängt wahrscheinlich stärker von T und d ab als bei GELU. Die Rate wurde nur bei T = 4000 gesucht, d. h. z² war bei T = 4000 "fair" und bei T = 16000 möglicherweise zu hoch/niedrig. Das ist der wichtigste offene Punkt.
7. **Überanpassung:** T = 16000 = 10,15 Epochen. Ob z² schneller überanpasst (hohe Kapazität, kein Beschnitt der Aktivierung), ist ohne Train-bpc nicht entscheidbar.

## 4. Literatur (kurz, 2 WebSearches; Rest aus vorhandenen Berichten w11, 2026-10-07_literatur_struktur_quadrat)
- **Primer** (So et al. 2021, NeurIPS, arXiv 2109.08668): Squared ReLU ist die wirksamste Änderung; Gewinn wächst mit Rechenbudget (bis 1,9 Mrd. Parameter, etwa ein Drittel der Trainingsrechnung). Gegen unser Muster: dort wächst der Vorteil mit Skalierung, hier schrumpft er (aber ReLU², nicht reines z², und ein hochoptimierter Aufbau mit Lernratensuche).
- **ReLU² Wins** (arXiv 2402.03804) und BitNet b1.58 2B4T (2504.12285): ReLU² als Standard, Argument ist Sparsität/Hardware, nicht reine Güte (aus w11 übernommen, nicht neu geprüft).
- **Quadratische Neuronen / Polynomnetze:** Π-Nets (Chrysos et al.), Fan et al. "On Expressivity and Trainability of Quadratic Networks" (arXiv 2110.06081): Quadratische Netze sind ausdrucksstärker, aber **leichter trainings-kollabierend** (Gegenmittel ReLinear). Polynom-*Aktivierung* ist etwas anderes als polynomiales *Neuron*: erstere bleibt stückweise linear in Entscheidungsgrenzen. Passt zu Punkt 6.
- Einordnung: reines z² als FFN-Aktivierung ist eine Primer-Variante ohne Gate; kein neuer Befund erwartbar, nur eine saubere Kalibrierung. Neuheit ist laut Malte nicht nötig.
- Alle Zitate vor Verwendung in der Arbeit am Original prüfen.

## 5. Vorschlag: Lauf I-050 "Aktivierung bei gleichen FLOPs, langes Training"
Entscheidet: **"Born-Quadrat z² schlägt GELU/ReLU² bei gleichen FLOPs, auch nach langem Training."** Alle Arme haben identische Matmul-FLOPs und Parameter (nur Aktivierung verschieden), also "gleiche FLOPs" ist exakt (Aktivierung selbst vernachlässigt, im Ergebnis ausgewiesen).

**Modul:** Profil/Erweiterung von `abteilungen/struktur_sprache.py` (wie v3/v4), Agenda-Kennung I-050, Daten/Split/Messung/Checkpoint/Wiederaufnahme unverändert (UD_German-GSD, Byte, 4 Lagen, FFN 4d, Batch 8×128).

**Arme (6, nur FFN-Aktivierung, dicht, Breite 4d):**
| Arm | Aktivierung | Rolle |
|---|---|---|
| gelu | GELU | Standard-Gegner (H1) |
| relu2 | relu(z)² | Primer-Gegner (H2) |
| **z2** | z² | geprüfter Arm (Born-Quadrat) |
| z2n | z² von parameterfrei normiertem z (LayerNorm ohne Parameter vor der Aktivierung) | Kontrolle Stabilität/Normierung |
| abs | \|z\| | Kontrolle "klassisch statt Quadrat" (Betrag ohne Quadrat) |
| zabsz | z·\|z\| (vorzeichentreues Quadrat) | Kontrolle "Vorzeichenverlust"; trennt Symmetrie von Quadrat |

**Bedingungen:** d ∈ {256, 512}, T ∈ {2000, 8000, 16000}. Optional dritte Breite d = 128 (billig, zeigt die Breitenachse; +≈ 40 Kernstunden). Pflicht: 8 Seeds. Train-bpc (erste 64 kB der benutzten Trainingsbytes) für jede Aufgabe, Lücke = Test − Train. T = 16000 bleibt die längste Dauer, weil mehr Training auf 1,6 MB nur noch Wiederholung ist; "langes Training" heißt hier 10 Epochen. Optional (Entscheidung Malte): bei Zugang zu größerem deutschem Korpus eine Zelle mit T = 32000 ohne Wiederholung.

**Lernrate (entscheidend für Fairness):** je (Arm, d, T) **auf voller Schrittzahl T** auf einem Tuning-Seed, Gitter Faktor 2 (5 Punkte: 5e-5 … 8e-4, beidseitig verlängerbar, mit `gittersuche_stabil`). Rand-Raten ausweisen. Das beseitigt Vorbehalt (a). Ein zweiter Tuning-Seed für die Hauptzelle (512, 16000) erhöht die Robustheit; ohne ihn Vorbehalt notieren.

**Hypothesen (gepaart, einseitig, n = 8, Test-bpc, Holm über H1a–H2b):**
- H1a / H1b: z2 < gelu bei (d = 256 / 512, T = 16000). Urteil BORN_BESSER | H0 (gelu sicher besser) | HU.
- H2a / H2b: z2 < relu2, ebenso. Hauptaussage "Born-Quadrat schlägt GELU/ReLU²" ist nur dann gezeigt, wenn **H1b und H2b beide BORN_BESSER** sind (d = 512, T = 16000 als Hauptzelle; H1a/H2a Zusatz in Holm).
- Nicht in Holm (zweiseitig, α = 0,05): **H3** Trend Δ = bpc(z2) − bpc(gelu): Δ(T = 16000) − Δ(T = 2000) je Seed gepaart: WAECHST_BEI_TRAINING (Vorteil schrumpft) | SCHRUMPFT | HU, getrennt je d. **H4** Breite: Δ(d = 512) − Δ(d = 256) bei T = 16000. **H5** Normierung: z2n − z2 bei T = 16000, d = 512 (trägt ein Stabilitätsproblem?). **H6** Vorzeichen: zabsz − z2 und relu2 − z2 (Symmetrieverlust?). **H7** Überanpassung: Lücke(z2) − Lücke(gelu) bei T = 16000.
- Beschreibend: alle 36 Zellen (Arm × d × T) mit Test/Val/Train-bpc, Lernraten und Rand-Flags, Divergenzrate, Aktivierungsstatistik (Kurtosis/Maximalbetrag der Zwischenwerte am Ende) für z2 vs z2n, Zeit je Schritt.

**Negativkontrolle:** z2 und gelu, d = 256, 1000 Schritte, permutierte Trainingsbytes, 2 Seeds, bestanden bei Mittel ≥ 0,9 × Unigramm-Entropie (4,743). Divergenz-/Kollaps-Regeln wie v4 (Seed bleibt in den Zahlen, betroffene Hypothese = HU).

**Seeds:** 8 Mess-Seeds + 2 NK + 1–2 Tuning-Seeds, frisch mit `werkzeuge/seeds_pruefen.py --neu`, vor dem Lauf in `seeds.json` eintragen (Tuning-Endziffern ≥ 90). Keine Zahl aus v3/v4 darf in die Kriterien einfließen.

**Erwartung vorab (ehrlich, kein Befund):** H1b/H2b eher **H0 oder HU** (≈ 20 % BORN_BESSER), weil z² bei v3 (512, 16000) um 0,047 hinter GELU lag. Falls die Lernratensuche je T den Rückstand schließt, wird es HU/gleichwertig; ein Sieg wäre die Überraschung. H3 eher "Vorteil schrumpft". H6 vermutlich "relu2 besser als z2" (Gate). Jeder Ausgang ist berichtbar und schließt die Frage "Quadrat vs. Gate" sauber ab.

**Laufzeit Raven (Schätzung aus v3/v4-Messwerten, 1 Thread; d = 256: 0,32 s/Schritt, d = 512: ≈ 1,0 s/Schritt):**
- Messung: 6 Arme × 8 Seeds × (2000 + 8000 + 16000 = 26.000 Schritte) × (0,32 + 1,0) ≈ 1,65 · 10⁶ s ≈ **458 Kernstunden**; mit d = 128 (≈ 0,15 s) zusätzlich ≈ 50.
- Tuning: 6 Arme × 2 d × 3 T × ≈ 5 Raten, volle Schrittzahl: ≈ 6 · 26.000 · 5 · 1,32 ≈ **286 Kernstunden**, das Tuning bestimmt die Wandzeit (Runde = längste Aufgabe).
- Gesamt ≈ **750 Kernstunden**; bei 72 Prozessen und Aufschlag 1,3 etwa **16–20 h Wandzeit** (Tuning ≈ 9 h in 2 Runden à 4,6 h, Messung ≈ 8 h). Knapp unter dem 24-h-Limit; Wiederaufnahme je Aufgabe und Checkpoint alle 2000 Schritte sind Pflicht. Kürzungsoptionen: Tuning auf 4 Raten (Faktor 2 um den v3-Wert) → ≈ −20 %; 6 statt 8 Seeds → Messung −25 %; nur Hauptzelle d = 512 voll, d = 256 mit T ∈ {2000, 16000} → ≈ −30 %.
- Vor dem Start auf Raven: Vortest (Aktivierungsformeln, FLOP-/Parametergleichheit exakt, Trainierbarkeit aller 6 Arme, Urteilszweige erzwungen, Wiederaufnahme), Rauchtest nur auf Tuning-Seed. Freigabe durch Malte nötig (Regel: vor jedem Lauf).

**Billiger Vorversuch (Container, keine Freigabe nötig, Tuning-Seed):** Nur die Frage, ob Lernrate je T das Kippen von v3 (512, 16000) schließt: gelu und z2, d = 256 oder 512, T = 4000 und 16000, Raten Faktor 2, ein Tuning-Seed, nur Tuning-Seed-Validierung. Wenn auch hier z2 hinter gelu bleibt, ist der Kippeffekt kein Ratenartefakt und I-050 sollte auf die Hauptzelle verschlankt werden.

## 6. Offene Punkte / was fehlt
- I-047-Roh-JSON und I-049/I-048 sind nicht im Repo (nur Zusammenfassung); H3-Zahl 0,040 stammt aus dem Auftrag. Bitte `I-047__*.json` nach `ergebnisse/` holen, um die TT-Differenz je Seed zu prüfen.
- Kein Train-bpc je Arm in v3/v4 vorhanden (Überanpassungsfrage offen; I-045-v5 liefert es für gelu und Df_z2).
- Lernrate in v3/v4 für alle dichten Arme gleich (1,67e-4); das kann am Gitter liegen (Faktor 3, eine Seed), nicht an der Aktivierung.
- Rohdaten gerechnet: `ergebnisse/I-045-v2__20261007_112736.json`, `I-045-v3__20261026_052958.json`, `I-045-v4__20261027_033627.json`, `I-042__20261006_133726.json`, `I-044__20261006_155444.json`, r3: `recherche/2026-10-27_welle/r3_bericht.md`.

Sources:
- [Primer (arXiv 2109.08668)](https://arxiv.org/pdf/2109.08668)
- [On Expressivity and Trainability of Quadratic Networks (arXiv 2110.06081)](https://arxiv.org/pdf/2110.06081)
