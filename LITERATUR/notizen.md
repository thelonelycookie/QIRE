# Literaturnotizen QIRE (R5.3)

*Stand 2026-10-14. Erzeugt von `LITERATUR/erzeuge.py` aus `LITERATUR/quellen_daten.py` (nicht von Hand ändern, sondern dort ergänzen und neu erzeugen).*
*Quelle ausschließlich Repo-Texte: `INSTITUT/recherche/*`, `INSTITUT/NEUHEIT_BORN_MAGIE.md`, Handover/LABORBUCH, `A/sources/literature_cards/`. Nichts wurde im Internet nachgeschlagen. Wo etwas fehlt, steht **unvollständig**; Titel mit `*` hat der Katalogisierer ergänzt und sind vor dem Zitieren zu prüfen.*

**Ehrliche Grenze:** Das meiste sind Suchauszüge/Abstracts. Nur Einträge mit Lesestatus „Volltext“ hat das Institut im PDF gelesen. „nicht gefunden“ in einer Neuheitsprüfung heißt: in wenigen Suchen nichts gefunden, kein Beweis.

## Dateien
| Datei | Inhalt |
|---|---|
| `literatur.bib` | BibTeX aller 275 Arbeiten (Key = `nachname+jahr+kurz`), `note`-Feld mit Lesestatus, Lücken, QIRE-Bezug |
| `notizen.md` | diese Datei: Prüfungsübersicht, Begriffe, alle Arbeiten (alphabetisch nach Key) |
| `neuheit_index.json` | maschinenlesbarer Index für die Ideen-KI (Format unten) |
| `quellen_daten.py` / `erzeuge.py` | Datenquelle und Generator (`python LITERATUR/erzeuge.py [--pruefen]`) |

## Format `neuheit_index.json` (für Lückenkarte und Neuheitsfilter der Ideen-KI)
```json
{
  "format": "qire-neuheit-index", "version": 1, "stand": "...",
  "urteil_skala": { "<urteil>": "Bedeutung", ... },
  "begriffe": {
    "<begriff-schluessel>": {
      "beschreibung": "Klartext", "synonyme": ["..."],
      "arbeiten": ["<bib-key>", ...],        // alle Arbeiten, die den Begriff berühren
      "pruefungen": ["N-B", ...],            // Neuheitsprüfungen, in denen der Begriff vorkommt
      "neuheitsurteile": ["teilweise_bekannt", ...],
      "status": "bekannt|teilweise_bekannt|gegenbefund|neue_verbindung|nicht_gefunden|offen|ungeprueft"
    }
  },
  "neuheitspruefungen": [ { "id", "datum", "datei", "aussage", "urteil", "begriffe", "arbeiten", "hinweis", "grenzen" } ],
  "arbeiten": { "<bib-key>": { "arxiv", "autoren", "jahr", "titel", "status", "kernaussage", "qire_bezug", "neuheit_hinweis", "begriffe", "unvollstaendig", "fundstellen" } }
}
```
**Benutzung durch die Ideen-KI**
- *Neuheitsfilter:* Idee auf Begriffe abbilden (Schlüssel oder `synonyme`). Hat ein Begriff `status` `bekannt` oder `teilweise_bekannt`, dann `arbeiten` und `neuheitspruefungen[].hinweis` in die Karte als Abgrenzung schreiben und die Neuheitsbewertung deckeln (kein „erstmals“). Bei `nicht_gefunden`/`neue_verbindung` normale Bewertung, aber „plausibel“ statt „neu“. `gegenbefund` muss in der Karte diskutiert werden. Schon geprüfte Aussagen (`neuheitspruefungen[].aussage`) nicht erneut als Recherche beauftragen.
- *Lückenkarte:* Zellen = Paare von Begriffen. Ein Paar ist **gefüllt**, wenn eine Arbeit beide Begriffe in `begriffe` trägt; **leer**, wenn keine Arbeit beide trägt (Bonus für leere Felder, aber „leer“ = nur „in unseren Notizen nicht belegt“). Begriffe mit wenigen Arbeiten und `status: ungeprueft` sind Suchlücken der Literaturarbeit, nicht Neuland.
- Der Index wird von `erzeuge.py` neu erzeugt; nie von Hand editieren.

Lesestatus: **Volltext** (PDF gelesen) · **Abstract** (Abstract gelesen) · **Auszug** (Suchtreffer/Snippet) · **nur Titel**.

## 1. Neuheitsprüfungen des Instituts (nicht wiederholen)
| ID | Datum | Aussage | Urteil | Datei |
|---|---|---|---|---|
| N-A | 2026-10-07 | Born-Auslese (\|psi\|^2) vs. lineare Auslese, gleiche Schaltung, gleiche Parameterzahl | **teilweise_bekannt** | `INSTITUT/NEUHEIT_BORN_MAGIE.md` |
| N-B | 2026-10-07 | Magie als Kompressionsmass: wenige Nicht-Clifford-Winkel tragen den Nutzen, Rest auf k*pi/2 runden (nach dem Training) | **teilweise_bekannt** | `INSTITUT/NEUHEIT_BORN_MAGIE.md` |
| N-M7 | 2026-09-28 | Magie-Budget-Linie (Vorgabe 7): Magie-Konzentration, Hybrid-Kaskade, Magie vs. Datenmenge, Magiebedarf-Praediktor | **nicht_gefunden** | `INSTITUT/recherche/2026-09-28_neuheit_magie.md` |
| N-C | 2026-09-29 | Reine Clifford-Schaltung per diskreter Suche als fertiger Bildklassifikator (Endmodell), gleichwertig zum voll trainierten VQC | **nicht_gefunden** | `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md` |
| N-I036v2 | 2026-10-05 | Clifford-Klassifikator = quadrierter lokaler Block-Walsh ueber vorzeichenmaskierten Pixeln (Maske vs. Blockwahl) | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md` |
| N-I036v3 | 2026-10-05 | Rolle der quadratischen Vorzeichenmaske (Rang, Seed-Transfer) in Clifford-Klassifikatoren | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_neuheit_I-036-v3.md` |
| N-STRUKTUR | 2026-10-06 | Strukturierte (Hadamard*Diagonal) Mischung + Quadrat als Born-artige Auslese | **bekannt** | `INSTITUT/recherche/2026-10-06_literatur_struktur.md` |
| N-VOLLTEXT1 | 2026-10-07 | Volltextpruefung: Deep Fried Convnets, Fastfood, OPU, Primer, Monarch, Pixelated Butterfly; I-040-v2 H4 | **neue_verbindung** | `INSTITUT/recherche/2026-10-07_volltext_arxiv.md` |
| N-VOLLTEXT2 | 2026-10-07 | A Struktur bei wenigen Epochen; B Pareto exakte Genauigkeit x Bit/Schuss; C Clifford-Zelle vs BitNet-ternaer; D sqrt(L)-Gesetz fuer Adapterlagen | **neue_verbindung** | `INSTITUT/recherche/2026-10-07_volltext_2.md` |
| N-QUADRAT-ADAPTER | 2026-10-07 | Strukturierte Mischung + Quadrat als Adapter-Zweig (I-045-v2) | **teilweise_bekannt** | `INSTITUT/recherche/2026-10-07_literatur_struktur_quadrat.md` |
| N-BREIT01 | 2026-10-05 | Informationsausbeute je Schuss relativ zu log2 K als Guetemass fuer Born-Klassifikatoren; Clifford-Maske auf I(C;X) optimieren | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/01_holevo.md` |
| N-BREIT02 | 2026-10-05 | Magie-Budget k als Regler fuer Surrogat-Genauigkeit; Clifford/Magie-Strukturen und Parametereffizienz; Var(dL) vs (n,k); Rang <-> Verschraenkung <-> Trainingsgewinn | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md` |
| N-BREIT03 | 2026-10-05 | Datensatz-optimierter Clifford am Lader (T-Zahl des Ladens), T-Budget als Trainingsgrenze am Lader, Walsh-Raum-Laden | **teilweise_bekannt** | `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md` |
| N-BREIT04 | 2026-10-05 | Magie-Konzentration (M50/M90), Magiebedarf vs Anzahl Klassen, Accuracy vs T-Count Pareto, Clifford-Rang vs Nicht-Clifford-Zahl | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/04_magie.md` |
| N-BREIT05 | 2026-10-05 | Bit-Budget-Pareto statt Parameterzahl; geteilte Maske / niedriger Rang; gelernte Zuordnung gruppierter Basiszustaende | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md` |
| N-BREIT06 | 2026-10-05 | Skalierungsgesetz 'Verlustgewinn gegen Breite' mit Exponent bei log-d-Parametern (QIRE-Adapter) | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md` |
| N-BREIT07 | 2026-10-05 | Adapter mit log-Breite-Parametern; diskret/Clifford im LLM-Adapter; Born-Regel als Attention; Adapter-Skalierung | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md` |
| N-BREIT08 | 2026-10-05 | Clifford-Klassifikator + CDR; Kostenmodell trainiertes Clifford+k-T-Modell (~100 Parameter); Schussrauschen; trainierter Rang vs Hardwarekosten | **nicht_gefunden** | `INSTITUT/recherche/2026-10-05_breit/08_hardware.md` |
| N-BREIT09 | 2026-10-05 | Graphzustands-Kernel, MBQC-QML, Bent-Phase vor WHT, Quadratform-Rang als ML-Praediktor | **offen** | `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md` |
| N-BREIT10 | 2026-10-05 | Red-Team: Benchmarking, Dequantisierung, Etikett quantum-inspired, klassische Nachbarn von Projekt A | **gegenbefund** | `INSTITUT/recherche/2026-10-05_breit/10_red_team.md` |
| N-BREIT11 | 2026-10-05 | Klassische Effizienz-Nachbarn: Fastfood, Deep Fried, ACDC, QuaRot, SpinQuant | **bekannt** | `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md` |

### Details je Prüfung

**N-A: Born-Auslese (|psi|^2) vs. lineare Auslese, gleiche Schaltung, gleiche Parameterzahl** — Urteil: *teilweise_bekannt* (2026-10-07)
- Hinweis: Prinzip als Ablation in Klassifikationskoepfen/Sequenzmodellen vorhanden, gemischte Evidenz (Kalibrierung: linear besser). Neu nur die kontrollierte Ablation an Butterfly-Schicht (Ry/Rz + CNOT-Ring) bei gleicher Parameterzahl. Keine Prioritaet fuer das Born-Prinzip beanspruchen.
- Grenzen: Nur Suchtreffer-Zusammenfassungen; Vorwaertszitate (Semantic Scholar) nicht geprueft.
- Arbeiten: `cvnn_kopf2026`, `wellenfunktion_sprache2026`, `quantum_shortcut2026`, `qnbm`, `unitary_rnn2015`

**N-B: Magie als Kompressionsmass: wenige Nicht-Clifford-Winkel tragen den Nutzen, Rest auf k*pi/2 runden (nach dem Training)** — Urteil: *teilweise_bekannt* (2026-10-07)
- Hinweis: Einzelbausteine bekannt (Clifford als Initialisierung, T-Zahl als Compilerkosten, Pruning auf 0). Post-hoc-Clifford-Rundung einer trainierten, klassisch gerechneten Schicht als Kompressionsmass: nicht gefunden. Opus-Anmerkung: Magie ist fuer die Quantenausfuehrung reale Kostengroesse (Q2, T ~ 31 k), fuer die klassische Butterfly-Ausfuehrung nur 'multiplikationsfrei' - beide Lesarten trennen.
- Grenzen: Negativbefund unter eingeschraenkter Suche; Zitierungen der Nachbarn nicht geprueft.
- Arbeiten: `ravi2022cafqa`, `cheng2022clifford_init`, `stabilizer_bootstrap2024`, `kulshrestha2024qadaprune`, `kissinger2020t_reduktion`, `t_count_numerisch2026`, `nqs_nonstab_regime2026`, `maronese2026expressibility`, `kyriienko2025diskret`, `xu2025butterflyquant`

**N-M7: Magie-Budget-Linie (Vorgabe 7): Magie-Konzentration, Hybrid-Kaskade, Magie vs. Datenmenge, Magiebedarf-Praediktor** — Urteil: *nicht_gefunden* (2026-09-28)
- Hinweis: Magie-Konzentration, Magie vs. Datenmenge, Magiebedarf-Praediktor: nichts Vergleichbares gefunden (Einzelaussage). Hybrid-Kaskade: teilweise publiziert als Mechanismus-Genre (BranchyNet 2016, partial QEC), Kombination 'Stufe nach T-Zahl' nicht gefunden - als Kombination bekannter Bausteine kennzeichnen. Abgrenzung zu CAFQA/NCC-WS in der Arbeit explizit.
- Grenzen: Nur Abstracts/Snippets; NeurIPS/ICML-QML-Workshops schlecht indexiert; zweite Runde dblp/Google Scholar mit 'magic concentration' und 'loss-ranked Clifford rounding' offen.
- Arbeiten: `quantum_resources2024`, `nqs_nonstab2025`, `stabilizer_klassifikator2025`, `zx_parametrisiert2024`, `clifford_perturbation_em2024`, `rotations_hadamard2024`, `ncc_ws2023`, `vqa_eft2024`, `adaptive_pruning_qfim2024`, `sre_ml2025`, `sre_gnn2025`, `no_reference_free2026`, `anticoncentration2025`, `mnisq2026`

**N-C: Reine Clifford-Schaltung per diskreter Suche als fertiger Bildklassifikator (Endmodell), gleichwertig zum voll trainierten VQC** — Urteil: *nicht_gefunden* (2026-09-29)
- Hinweis: Keine Arbeit gefunden, die (a) reine Clifford-Schaltung (b) per diskreter Suche (c) als Endmodell (d) fuer Bildklassifikation (e) mit Vergleich zum voll trainierten VQC zeigt. Ehrliche Formulierung: klassisch simulierbar (Gottesman-Knill), KEIN Quantenvorteil; Amplitudenkodierung ist der teure Nicht-Clifford-Teil. Vorsicht: 'Clifford' auch als Clifford-ALGEBREN.
- Grenzen: Wenige Suchanfragen, nur Abstracts.
- Arbeiten: `ravi2022cafqa`, `cheng2022clifford_init`, `meyer2025lce`, `bravyi2019lowrank`, `lipardi2026magic_qas`, `delorenzis2025qelm`, `bowles2024better`, `bermejo2024qcnn`, `cerezo2025bp_sim`, `landman2023rff`, `schuld2021encoding`, `gottesman1998`, `aaronson2004chp`, `leone2022sre`, `ivaki2025reservoir`, `maronese2026expressibility`, `hadamard_muster_mnist2024`, `lu2023qasbench`

**N-I036v2: Clifford-Klassifikator = quadrierter lokaler Block-Walsh ueber vorzeichenmaskierten Pixeln (Maske vs. Blockwahl)** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Neuheit plausibel (neue Zusammenhaenge: Clifford-QML <-> maskierter Block-Walsh). Bekannt: WHT als NN-Schicht, Clifford-Simulierbarkeit, Clifford-Start fuer VQAs.
- Grenzen: Drei Suchanfragen, nur Snippets.
- Arbeiten: `pan2021wht`, `wht_cnn2025`, `cmps2024`, `i036_v2_html`, `i036_v2_pdf`, `i036_v2_2605`, `i036_v2_2608`

**N-I036v3: Rolle der quadratischen Vorzeichenmaske (Rang, Seed-Transfer) in Clifford-Klassifikatoren** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Keine Arbeit zur Rolle der quadratischen Phase in Clifford-Klassifikatoren gefunden. Zu zitieren: Dehaene & De Moor 2003; gelernte Phasenmasken in optischer Klassifikation (2604.25480, vor der Arbeit lesen).
- Grenzen: Zwei Suchanfragen, nur Snippets.
- Arbeiten: `dehaene2003clifford`, `ncc_phase_2022`, `stab_lernen_2609_10974`, `stab_lernen_2607_02444`, `materialisierung2026`, `single_photon_cs2026`

**N-STRUKTUR: Strukturierte (Hadamard*Diagonal) Mischung + Quadrat als Born-artige Auslese** — Urteil: *bekannt* (2026-10-06)
- Hinweis: Bekannt: Struktur-Mischung, Zufallsprojektion + |.|^2, Quadrat als Aktivierung, Polynomkern-Naeherung, Surrogate. Moeglicherweise neu (nur 'nicht gefunden'): kontrollierter gepaarter Vergleich Quadrat vs |z| vs ReLU nach DERSELBEN festen Mischung (I-038 H-G, I-039-v2); Befund, dass der Vorsprung aus Mischung + Gruppierung kommt; Skalierung 16->32 (I-041); I-040-v2 H4. Vertretbar: 'bekannte Bausteine, neu kombiniert und ehrlich vermessen'. Nicht vertretbar: 'erstmals', 'neue Architektur'.
- Grenzen: Volltext erst in 2026-10-07_volltext_arxiv.md nachgeholt.
- Arbeiten: `le2013fastfood`, `yu2016sorf`, `yang2015deepfried`, `moczulski2016acdc`, `sindhwani2015structured`, `dao2019butterfly`, `kaleidoscope2020`, `dao2022monarch`, `chen2022pixelfly`, `qiu2024compute`, `gonon2025kronecker_sparse`, `saade2016opu`, `ohana2020opu_kerne`, `karnick`, `livni2014`, `so2021primer`, `born_text_2022`, `stoudenmire2016`, `landman2023rff`, `qelm_bilder2024`, `delorenzis2025qelm`, `shofar2022`, `fewshots2024`, `samplingnoise2024`

**N-VOLLTEXT1: Volltextpruefung: Deep Fried Convnets, Fastfood, OPU, Primer, Monarch, Pixelated Butterfly; I-040-v2 H4** — Urteil: *neue_verbindung* (2026-10-07)
- Hinweis: I-044-v2 berechtigt (Deep Fried: weder linearer noch budgetgleicher Gegner). Quadrat gegen |z| in Saade/Ohana/Fastfood/Primer NICHT gemacht -> eigene Ablation eigenstaendig. Fastfood-Zeitgewinn ab n~1.024 extern gestuetzt (Pixelfly Fussn. 4). I-040-v2 H4: neue Verbindung, nicht neu; Sbaiz 2021 und Hu 2023 zwingend zitieren; H2 (schussbewusster Kopf) hat Vorlaeufer (Yomo, ShofaR).
- Grenzen: Volltexte aus arXiv-PDFs gelesen.
- Arbeiten: `yang2015deepfried`, `le2013fastfood`, `saade2016opu`, `ohana2020opu_kerne`, `so2021primer`, `dao2022monarch`, `chen2022pixelfly`, `sbaiz2021singlephoton`, `hu2023eigentasks`, `recio2024singleshot`, `yomo2025`, `shofar2022`, `firstphoton2024`

**N-VOLLTEXT2: A Struktur bei wenigen Epochen; B Pareto exakte Genauigkeit x Bit/Schuss; C Clifford-Zelle vs BitNet-ternaer; D sqrt(L)-Gesetz fuer Adapterlagen** — Urteil: *neue_verbindung* (2026-10-07)
- Hinweis: Alle vier Teile 'neue Verbindung' (Bausteine einzeln bekannt, direkter Vergleich/Gesetz nicht gefunden). D: bekannt sind fruehe Saettigung (Quantum-PEFT ab L=3, BOFT) und PEFT-Exponenten ~0 (Zhang 2024).
- Grenzen: Volltext-Zeilen mit [P] (Paper) / [S] (Schluss) gekennzeichnet.
- Arbeiten: `qiu2024compute`, `qiu2025linear`, `sbaiz2021singlephoton`, `hu2023eigentasks`, `banchi2021qib`, `ma2023photonen`, `ilias2026einzelschuss`, `bitnet158`, `xu2025butterflyquant`, `pan2021wht`, `clifford_equivariant2023`, `hadamard_klassisch_hadanet2019`, `hadamard_nn_2018`, `larocca2021lie`, `oft2023`, `zhang2024scaling_peft`, `rslora2023`, `lora_miub2025`, `zhang2022gradient_skalierung`, `ghorbani2021kernel`, `ba2022onestep`, `dao2022monarch`, `chen2022pixelfly`

**N-QUADRAT-ADAPTER: Strukturierte Mischung + Quadrat als Adapter-Zweig (I-045-v2)** — Urteil: *teilweise_bekannt* (2026-10-07)
- Hinweis: Naechster Nachbar Needle 2 'Hadamard MLP' (Projektseite, SiLU statt Quadrat): Idee Hadamard-Mischung/Nichtlinearitaet/Rueckmischung/3 Diagonalen nicht neu. Quadratischer Adapterzweig nicht gefunden; Neuheit des Bausteins nicht behaupten, Needle-2-Naehe nennen.
- Grenzen: 6 Suchanfragen, nur Titel/Zusammenfassungen; Needle ist keine Peer-Review-Quelle.
- Arbeiten: `needle2`, `le2013fastfood`, `yang2015deepfried`, `sindhwani2015structured`, `dao2022monarch`, `so2021primer`, `relu2_sparsitaet2024`, `hadamard_adapter2024`

**N-BREIT01: Informationsausbeute je Schuss relativ zu log2 K als Guetemass fuer Born-Klassifikatoren; Clifford-Maske auf I(C;X) optimieren** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Information pro Schuss bei Quantum Decision Trees und Margen-Schuss-Schranken gefunden; Fano/Holevo einzeln Standard. Als Kennzahl fuer Born-Klassifikatoren: nicht gefunden.
- Grenzen: Nur Auszuege.
- Arbeiten: `qdt_info_schuss2025`, `yomo2025`, `shofar2022`, `recio2024singleshot`, `amplitude_limits2025`, `audenaert2007chernoff`, `huang2020shadows`, `aaronson2017shadow`, `brassard2000amplitude`, `buhrman2001fingerprint`, `chen2022speicher`, `measure_first2023`, `fano_2019`, `bennett_wiesner_superdense`

**N-BREIT02: Magie-Budget k als Regler fuer Surrogat-Genauigkeit; Clifford/Magie-Strukturen und Parametereffizienz; Var(dL) vs (n,k); Rang <-> Verschraenkung <-> Trainingsgewinn** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Kein anerkanntes Beispiel 'trainierbar + beweisbar nicht simulierbar + nuetzlich auf klassischen Daten' bekannt. Fourier-Surrogate und Pauli-Propagation bekannt; Magie-Budget als Regler, Var(dL) vs (n,k), Rang<->Verschraenkung<->Trainingsgewinn: nicht gefunden.
- Grenzen: Websuche-Budget erschoepft; fast alles nur Auszuege.
- Arbeiten: `cerezo2025bp_sim`, `mcclean2018bp`, `schreiber2023surrogat`, `shin2024tn_dequant`, `gilfuster2025`, `bowles2024better`, `goh2025gsim`, `rudolph2023lowesa`, `angrisani2025pauli`, `heyraud2023`, `zhang2024lernbarkeit_clifford`, `wang2026lernbar_nicht_simulierbar`, `mbl_hea2024`, `mbl_variational2024`, `larocca2025uebersicht`, `schuld_killoran2022`, `tang2019`, `chia2022sampling`, `cotler2021revisiting`, `entanglement_geometry_2607_17872`, `bravyi2016gosset`

**N-BREIT03: Datensatz-optimierter Clifford am Lader (T-Zahl des Ladens), T-Budget als Trainingsgrenze am Lader, Walsh-Raum-Laden** — Urteil: *teilweise_bekannt* (2026-10-05)
- Hinweis: Walsh-Raum als Laderaum NICHT neu (Boosari 2609.28760, Zylberman-Debbasch). Neu hoechstens: Kopplung mit Winkelrundung aufs Clifford-Gitter / T-Budget-Rechnung eines Klassifikators. West 2023 optimiert Tiefe, nicht T.
- Grenzen: Snippet-basiert; Gosset-Kothari/Luo-Li-Volltext offen.
- Arbeiten: `boosari_walsh2026`, `walsh_2307_08384`, `stab_qram2025`, `cmps2024`, `west2023approx`, `gosset_kothari`, `optimal_t_sparsity2026`, `luo_li_sparse`, `shende2006`, `zustandspraep_cnot_unter2021`, `dilip2022mps`, `holmes_matsuura`, `fourier_mps2024`, `jaques_rattew_qram`, `perezsalinas2020reupload`

**N-BREIT04: Magie-Konzentration (M50/M90), Magiebedarf vs Anzahl Klassen, Accuracy vs T-Count Pareto, Clifford-Rang vs Nicht-Clifford-Zahl** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Begriff 'Magie-Konzentration einer Aufgabe (M50/M90)' nicht gefunden. QRC-Arbeit 2607.12035 ('Magie unverzichtbar') steht QIRE-Befund MNIST gegenueber; Ivaki: Optimum nahe Universalitaet, keine Konzentration. 'T-Gatter ~ CNOT-Kosten' gilt nur fuer fehlerkorrigierte Rechner mit Cultivation.
- Grenzen: Nur Auszuege.
- Arbeiten: `leone2022sre`, `magic_robustness_1807_10296`, `sre_nullitaet2026`, `sre_xor_fwht2025`, `sre_saettigung2025`, `bravyi2016gosset`, `shallow_magic_depth2025`, `begusic2023clifford_perturbation`, `ross_selinger2014`, `kliuchnikov2023synthese`, `nn_synthese2025`, `gidney2024cultivation`, `gidney2025cultivation_surface`, `gidney2025rsa`, `almost_ft_qml2025`, `grewal2025t_dotiert`, `wang2026lernbar_nicht_simulierbar`, `ivaki2025reservoir`, `qrc_magie2026`, `nonstab_optimierung2025`, `lipardi2026magic_qas`, `magie_vorteil2024`, `magie_vorteil_zurueck2025`, `gu2024pseudomagie`, `meyer2025lce`, `sim_verrauschte_magie2026`

**N-BREIT05: Bit-Budget-Pareto statt Parameterzahl; geteilte Maske / niedriger Rang; gelernte Zuordnung gruppierter Basiszustaende** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Gesamt-Bit-Vergleich QML vs. klassisch (Softmax/MLP/TM): nur Blier&Ollivier fuer Codelaenge und Binaer/Ternaer-Arbeiten fuer Bit pro Gewicht; kein QML-Vergleich auf Gesamtbitbasis gefunden.
- Grenzen: Suchbudget erschoepft, ca. 15 Suchen; vor Veroeffentlichung nachsuchen.
- Arbeiten: `chen2015hashednets`, `courbariaux2016`, `twn2016`, `blier2018`, `frankle2019lottery`, `li2018intrinsic`, `aghajanyan2021`, `tsetlin_fuzzy2023`, `tsetlin_2025`, `stoudenmire2016`

**N-BREIT06: Skalierungsgesetz 'Verlustgewinn gegen Breite' mit Exponent bei log-d-Parametern (QIRE-Adapter)** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Quantum-PEFT: log-Parameterzahl; QuanTA: Qualitaetsskalierung; BOFT: d log d. Gesetz mit explizitem Exponenten bei log-d-Parametern: nicht gefunden (Snippet-Ebene).
- Grenzen: Nur Auszuege.
- Arbeiten: `quantum_peft2025`, `quanta2024`, `quic2025`, `boft2023`, `krona2022`, `kronlora2025`, `hoffmann2022chinchilla`, `besiroglu2024chinchilla_repl`, `caballero2022bnsl`, `caro2022generalisierung`, `scaling_hybrid_qnn2026`, `ohno2026lie_scaling`, `rydberggpt2026`, `rydberggpt_methode2024`, `dao_monarch_mixer2023`, `finetuning_gesetze2025`

**N-BREIT07: Adapter mit log-Breite-Parametern; diskret/Clifford im LLM-Adapter; Born-Regel als Attention; Adapter-Skalierung** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: Adapter mit log-Breite-Parametern nicht gefunden (QPA: Qubits log2 m, aber Parameter ueber MLP; QuanTA: Parameter nicht log). Diskret/Clifford in LLM-Adaptern: nicht gefunden (alle trainieren kontinuierlich). 'Born als Attention': Neuheit unsicher (Quixer, 2603.03318).
- Grenzen: Suche begrenzt.
- Arbeiten: `qpa2025`, `qtha2025`, `quanta2024`, `hyperq2026`, `quixer2024`, `duneau2024`, `quantum_llm_ft2025`, `hyquT2025`, `lambeq2021`, `lambeq_gen2_2025`, `compositional_text2024`, `compactifai2024`, `tensorized_embedding2019`, `tensorgpt2023`, `tensorslm2025`, `lm_tensor_trains2024`, `quantum_self_attention2026`, `quantum_attention_vit2024`, `qattention_survey2025`, `qpsan2026`, `cayley_adapter2026`

**N-BREIT08: Clifford-Klassifikator + CDR; Kostenmodell trainiertes Clifford+k-T-Modell (~100 Parameter); Schussrauschen; trainierter Rang vs Hardwarekosten** — Urteil: *nicht_gefunden* (2026-10-05)
- Hinweis: CDR nutzt Clifford-NAHE Trainingsschaltungen fuer nicht-Clifford-Ziel; CAFQA/Clapton Clifford fuer VQE-Initialisierung. Als Klassifikator-Modell + CDR: nichts gefunden. Schussrauschen: als Methodik rahmen, nicht als Entdeckung.
- Grenzen: Nur ca. 16 Suchanfragen, Budget erschoepft.
- Arbeiten: `cdr2021`, `cdr_extrapolation2025`, `clapton2024`, `clifford_twirling2024`, `cafqa_ion2024`, `mckay2016zgates`, `kodierung_uebersicht2026`, `amplitude_approx_flach2021`, `shot_kodierung2026`, `hw_2606_21686`

**N-BREIT09: Graphzustands-Kernel, MBQC-QML, Bent-Phase vor WHT, Quadratform-Rang als ML-Praediktor** — Urteil: *offen* (2026-10-05)
- Hinweis: Neuheit NICHT bestaetigt, nur 'nicht gefunden bei Suche' (Suche duenn); Bent-Phase vor WHT als Merkmalsschicht nicht gefunden; Rang als ML-Praediktor: nur Hein et al. als Theorie.
- Grenzen: 6 Anfragen, einige vom Budget abgewiesen.
- Arbeiten: `mbqml2024`, `mbqcnn2024`, `resource_kernel2025`, `hein2003schmidt`, `schmidt_mass2000`, `dehaene2003clifford`, `graph_0812_4625`

**N-BREIT10: Red-Team: Benchmarking, Dequantisierung, Etikett quantum-inspired, klassische Nachbarn von Projekt A** — Urteil: *gegenbefund* (2026-10-05)
- Hinweis: KronA und BOFT sind direkte klassische Konkurrenten zur 'QIRE-Kronecker-Drehung mit log2(Breite) Winkeln'. Dequantisierungsrahmen und Benchmarks stuetzen 'Wirkung klassisch'.
- Grenzen: Nur Auszuege ueber Sekundaerseiten.
- Arbeiten: `bowles2024better`, `tang2019`, `chia2022sampling`, `cotler2021revisiting`, `quantum_inspired_survey2023`, `quantum_inspired_label2026`, `krona2022`, `boft2023`, `redteam_2604_18837`, `redteam_2608_15617`, `redteam_2304_04932`, `redteam_2111_09079`, `redteam_2508_00230`

**N-BREIT11: Klassische Effizienz-Nachbarn: Fastfood, Deep Fried, ACDC, QuaRot, SpinQuant** — Urteil: *bekannt* (2026-10-05)
- Hinweis: Siehe N-STRUKTUR.
- Grenzen: IDs mit [?] markiert.
- Arbeiten: `le2013fastfood`, `yang2015deepfried`, `moczulski2016acdc`, `quarot2024`, `spinquant2024`

## 2. Begriffe (Kurzfassung des Index)

| Begriff | Status | Arbeiten | Prüfungen |
|---|---|---|---|
| `adapter-peft` | teilweise_bekannt | 21 | N-VOLLTEXT2, N-QUADRAT-ADAPTER, N-BREIT06, N-BREIT07, N-BREIT10 |
| `amplitudenkodierung` | nicht_gefunden | 6 | N-BREIT08 |
| `amplitudenschaetzung` | ungeprueft | 1 | – |
| `approximatives-laden` | teilweise_bekannt | 4 | N-BREIT03 |
| `architektursuche` | ungeprueft | 5 | – |
| `barren-plateaus` | ungeprueft | 2 | – |
| `benchmark-qml` | gegenbefund | 5 | N-BREIT10 |
| `born-auslese` | teilweise_bekannt | 13 | N-A, N-BREIT07 |
| `butterfly` | ungeprueft | 6 | – |
| `clifford-endmodell` | nicht_gefunden | 0 | N-C, N-I036v2 |
| `clifford-initialisierung` | teilweise_bekannt | 8 | N-B, N-I036v2 |
| `clifford-naehe` | nicht_gefunden | 7 | N-BREIT08 |
| `clifford-orakel` | teilweise_bekannt | 1 | N-BREIT03 |
| `clifford-quadratform` | nicht_gefunden | 5 | N-I036v3, N-BREIT09 |
| `clifford-simulation` | ungeprueft | 4 | – |
| `clifford-suche-diskret` | nicht_gefunden | 3 | N-C |
| `datenkodierung` | ungeprueft | 5 | – |
| `datensaetze-qml` | ungeprueft | 1 | – |
| `dequantisierung` | gegenbefund | 19 | N-BREIT02, N-BREIT10 |
| `diskrete-winkel` | ungeprueft | 1 | – |
| `energieeffizienz` | ungeprueft | 1 | – |
| `expressibilitaet` | ungeprueft | 1 | – |
| `fehlerminderung` | nicht_gefunden | 6 | N-BREIT08 |
| `fourier-modell` | ungeprueft | 2 | – |
| `gate-synthese` | ungeprueft | 3 | – |
| `generalisierung-qml` | ungeprueft | 4 | – |
| `gf2` | nicht_gefunden | 1 | N-I036v3 |
| `gottesman-knill` | nicht_gefunden | 2 | N-C |
| `graphzustaende` | offen | 7 | N-BREIT09 |
| `hadamard-produkt-verwechslung` | ungeprueft | 3 | – |
| `hadamard-walsh` | bekannt | 13 | N-I036v2, N-STRUKTUR, N-QUADRAT-ADAPTER, N-BREIT03, N-BREIT11 |
| `hardware-fehlerkorrektur` | ungeprueft | 1 | – |
| `hardware-rz-virtuell` | nicht_gefunden | 1 | N-BREIT08 |
| `holevo` | nicht_gefunden | 8 | N-BREIT01 |
| `hybrid-kaskade` | nicht_gefunden | 1 | N-M7 |
| `hyperparametersuche` | ungeprueft | 1 | – |
| `information-bottleneck` | ungeprueft | 2 | – |
| `information-pro-schuss` | neue_verbindung | 5 | N-VOLLTEXT1, N-VOLLTEXT2, N-BREIT01 |
| `initialisierung` | ungeprueft | 1 | – |
| `intrinsische-dimension` | ungeprueft | 2 | – |
| `klassifikationskopf` | teilweise_bekannt | 2 | N-A |
| `klassische-simulierbarkeit` | nicht_gefunden | 13 | N-C, N-BREIT02 |
| `kompression` | ungeprueft | 1 | – |
| `kronecker` | gegenbefund | 2 | N-BREIT10 |
| `lernbarkeit-stabilizer` | ungeprueft | 1 | – |
| `lie-algebra` | neue_verbindung | 3 | N-VOLLTEXT2, N-BREIT02, N-BREIT06 |
| `magic-state-kosten` | ungeprueft | 6 | – |
| `magie-dosis` | nicht_gefunden | 2 | N-M7, N-BREIT02, N-BREIT04 |
| `magie-kompression` | teilweise_bekannt | 3 | N-B |
| `magie-konzentration` | nicht_gefunden | 3 | N-M7, N-BREIT04 |
| `magie-mass` | ungeprueft | 3 | – |
| `magie-suchsignal` | ungeprueft | 1 | – |
| `magie-tiefe` | ungeprueft | 1 | – |
| `magiebedarf-praediktor` | nicht_gefunden | 2 | N-M7, N-BREIT04 |
| `marge` | ungeprueft | 2 | – |
| `mbqc` | offen | 2 | N-BREIT09 |
| `mdl` | nicht_gefunden | 1 | N-BREIT05 |
| `mixture-of-experts` | ungeprueft | 1 | – |
| `monarch` | ungeprueft | 2 | – |
| `multiplikationsfrei` | ungeprueft | 2 | – |
| `namensverwechslung` | ungeprueft | 1 | – |
| `neuheitspruefung` | ungeprueft | 1 | – |
| `nichtlinearitaet` | ungeprueft | 1 | – |
| `nichtstabilizerness-nqs` | ungeprueft | 3 | – |
| `nichtstabilizerness-qml` | ungeprueft | 10 | – |
| `overparametrisierung` | ungeprueft | 1 | – |
| `parametereffizienz` | nicht_gefunden | 9 | N-BREIT05 |
| `pareto` | ungeprueft | 1 | – |
| `pauli-propagation` | ungeprueft | 2 | – |
| `phasenmaske` | nicht_gefunden | 1 | N-I036v3 |
| `polynomkern` | bekannt | 3 | N-STRUKTUR |
| `pruning-klassisch` | ungeprueft | 2 | – |
| `pruning-pqc` | teilweise_bekannt | 3 | N-B |
| `pseudomagie` | ungeprueft | 1 | – |
| `qaoa` | ungeprueft | 1 | – |
| `qml` | ungeprueft | 2 | – |
| `qml-bilder` | ungeprueft | 3 | – |
| `qram` | ungeprueft | 2 | – |
| `quadrat-aktivierung` | bekannt | 5 | N-STRUKTUR, N-VOLLTEXT1, N-QUADRAT-ADAPTER |
| `quanten-nlp` | nicht_gefunden | 13 | N-BREIT07 |
| `quanten-peft` | nicht_gefunden | 6 | N-BREIT06, N-BREIT07 |
| `quanteninspiriert` | gegenbefund | 4 | N-BREIT10 |
| `quantenspeicher` | ungeprueft | 1 | – |
| `quantenvorteil` | ungeprueft | 5 | – |
| `quantisierung` | ungeprueft | 1 | – |
| `quantisierung-ternaer` | bekannt | 9 | N-VOLLTEXT2, N-BREIT05, N-BREIT11 |
| `reservoir-computing` | ungeprueft | 6 | – |
| `ressourcentheorie` | ungeprueft | 1 | – |
| `schmidt-rang` | offen | 2 | N-BREIT09 |
| `schusszahl` | neue_verbindung | 15 | N-VOLLTEXT1, N-BREIT01 |
| `shadow-tomography` | ungeprueft | 3 | – |
| `simulated-annealing` | ungeprueft | 2 | – |
| `simulation-gpu` | ungeprueft | 1 | – |
| `skalierung` | ungeprueft | 4 | – |
| `skalierungsgesetze` | neue_verbindung | 12 | N-VOLLTEXT2, N-BREIT06 |
| `sprachmodell` | ungeprueft | 8 | – |
| `sre` | nicht_gefunden | 10 | N-M7, N-BREIT04 |
| `stabilizer-rang` | ungeprueft | 2 | – |
| `statistik-methodik` | ungeprueft | 3 | – |
| `strukturierte-schichten` | bekannt | 16 | N-STRUKTUR, N-VOLLTEXT1, N-VOLLTEXT2, N-QUADRAT-ADAPTER, N-BREIT11 |
| `t-zaehlung` | teilweise_bekannt | 11 | N-B, N-BREIT03, N-BREIT04 |
| `tensornetz` | ungeprueft | 9 | – |
| `tensornetz-kompression` | nicht_gefunden | 6 | N-BREIT07 |
| `trainierbarkeit` | nicht_gefunden | 12 | N-BREIT02 |
| `transformer-ffn` | ungeprueft | 2 | – |
| `tsetlin` | nicht_gefunden | 2 | N-BREIT05 |
| `unitaer` | ungeprueft | 6 | – |
| `unitaer-training` | ungeprueft | 1 | – |
| `verschraenkung` | ungeprueft | 1 | – |
| `vqe` | ungeprueft | 1 | – |
| `zufallsmerkmale` | ungeprueft | 6 | – |
| `zufallsmerkmale-opu` | bekannt | 2 | N-STRUKTUR |
| `zustandspraeparation` | teilweise_bekannt | 17 | N-BREIT03 |
| `zx-kalkuel` | ungeprueft | 2 | – |

## 3. Arbeiten (alphabetisch nach Key)

Felder: arXiv · Autoren · Jahr · Lesestatus · Kernaussage · Bezug zu QIRE-IDs · Neuheitsprüfung-Ergebnis · Fundstellen (Datei:Zeile).

### `aaronson2004chp` — Improved simulation of stabilizer circuits
- **arXiv:** quant-ph/0406196 · **Autoren:** Aaronson, Gottesman · **Jahr:** 2004 · **Venue:** Phys. Rev. A 70, 052328 (2004)
- **Lesestatus:** Abstract
- **Kernaussage:** CHP-Simulation fuer tausende Qubits; Stabilizer-Simulation ist parity-L-vollstaendig.
- **QIRE-Bezug:** Linie Clifford
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:57`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:133`
- **Begriffe:** gottesman-knill, klassische-simulierbarkeit

### `aaronson2017shadow` — Shadow Tomography of Quantum States \*
- **arXiv:** 1711.01053 · **Autoren:** Aaronson · **Jahr:** 2017
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Õ(eps^-4 log^4 M log D) Kopien fuer M Messungen. (Titel aus Wissen)
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:28`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:121`
- **Begriffe:** shadow-tomography

### `adaptive_pruning_qfim2024` — Adaptive pruning algorithm using a quantum Fisher information matrix for PQCs
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** Quantum Machine Intelligence 2024 (Springer, doi 10.1007/s42484-024-00221-9)
- **Lesestatus:** Auszug · **unvollständig:** Autoren, arXiv-ID
- **Kernaussage:** Entfernt Winkel nahe 0 mittels Quantum-Fisher-Information.
- **QIRE-Bezug:** I-001
- **Neuheitsprüfung:** Andere Auswahlregel (nicht Verlustanstieg-Ranking auf Clifford-Gitter).
- **Begriffe:** pruning-pqc

### `aghajanyan2021` — Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning \*
- **arXiv:** 2012.13255 · **Autoren:** Aghajanyan et al. · **Jahr:** 2021
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Hunderte bis Tausende; groessere Modelle niedriger.
- **QIRE-Bezug:** Parametereffizienz, Adapter
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:18`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:97`
- **Begriffe:** parametereffizienz, intrinsische-dimension, adapter-peft

### `almost_ft_qml2025` — Almost fault-tolerant QML
- **arXiv:** 2507.18954 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Nur CNOTs korrigieren, parametrisierte Einqubit-Gatter unkorrigiert; von Millionen auf wenige Tausend Qubits.
- **QIRE-Bezug:** Hardware-Argument
- **Neuheitsprüfung:** Konkurrierender Weg, das Magie-Problem im QML zu umgehen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:94`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:97`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:268`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:302`
- **Begriffe:** hardware-fehlerkorrektur

### `amplitude_approx_flach2021` — (Approximative Amplitudenkodierung in flachen parametrisierten Schaltungen)
- **arXiv:** 2103.13211 · **Autoren:** unvollständig · **Jahr:** 2021
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Alternative zu exakter Amplitudenkodierung.
- **QIRE-Bezug:** Datenladen, Hardware
- **Neuheitsprüfung:** Nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:39`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:103`
- **Begriffe:** amplitudenkodierung, approximatives-laden

### `amplitude_limits2025` — Limitations of Amplitude Encoding on Quantum Classification
- **arXiv:** 2503.01545 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Vollstaendiges Zuruecklesen der Eingabe widerspraeche Holevo.
- **QIRE-Bezug:** Holevo/Kodierung
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:20`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:43`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:133`
- **Begriffe:** holevo, amplitudenkodierung

### `amplitude_vs_winkel2024` — (Vergleich Amplituden- vs. Winkelkodierung auf MNIST)
- **arXiv:** 2410.09121 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Amplitudenkodierung oft besser bei weniger Qubits (0,891 vs 0,780 binaer), aber tiefere Schaltungen.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:46`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:117`
- **Begriffe:** datenkodierung, amplitudenkodierung

### `angrisani2025pauli` — (Pauli-Propagation, klassische Schaetzung)
- **arXiv:** 2409.01706 · **Autoren:** Angrisani et al. · **Jahr:** 2025 · **Venue:** PRL 2025
- **Lesestatus:** Auszug · **unvollständig:** Titel
- **Kernaussage:** Klassisch schaetzbar fuer die meisten zufaelligen unstrukturierten Schaltungen mit einzelqubit-invarianter Verteilung.
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:35`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:116`
- **Begriffe:** dequantisierung, pauli-propagation

### `anticoncentration2025` — Anticoncentration in Clifford Circuits and Beyond: Pseudo-Magic States
- **arXiv:** 2502.20455 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Magie-Konzentrationsverteilungen in Zufallsschaltungen.
- **QIRE-Bezug:** Magie-Konzentration
- **Neuheitsprüfung:** Andere Systemklasse.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:117`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:192`
- **Begriffe:** magie-konzentration

### `araujo2021approx` — (approximative Zustandspraeparation)
- **arXiv:** unvollständig (keine ID) · **Autoren:** Araujo · **Jahr:** 2021
- **Lesestatus:** nur Titel · **unvollständig:** Titel, Venue, arXiv-ID
- **Kernaussage:** Approximative Zustandspraeparation.
- **QIRE-Bezug:** UEBERGABE
- **Neuheitsprüfung:** Als bekannt eingestuft.
- **Begriffe:** approximatives-laden

### `audenaert2007chernoff` — Discriminating states: the quantum Chernoff bound \*
- **arXiv:** quant-ph/0610027 · **Autoren:** Audenaert et al. · **Jahr:** 2007 · **Venue:** PRL 98, 160501
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Fehler zweier Klassenzustaende faellt wie exp(-M xi) mit der Quanten-Chernoff-Information. (Titel aus Wissen)
- **QIRE-Bezug:** Holevo/Schusszahl
- **Neuheitsprüfung:** Schuss-Grenze M ~ ln(1/delta)/xi.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:42`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:135`
- **Begriffe:** holevo, schusszahl

### `ba2022onestep` — High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation \*
- **arXiv:** 2205.01445 · **Autoren:** Ba u. a. · **Jahr:** 2022
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Ein Gradientenschritt verbessert Merkmale gegenueber festen Zufallsmerkmalen. (Titel aus Wissen, Recherche: 'One Gradient Step')
- **QIRE-Bezug:** Zufallsmerkmale vs. Training
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:49`
- **Begriffe:** zufallsmerkmale

### `banchi2021qib` — Generalization in quantum machine learning: a quantum information standpoint \*
- **arXiv:** 2102.08991 · **Autoren:** Banchi, Pereira, Pirandola · **Jahr:** 2021 · **Venue:** PRX Quantum 2021 (Titel aus Wissen)
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Variationaler Quanten-Informationsflaschenhals L_IB = I(X:Z) - beta I(C:Z); Schuesse nur als Sonderfall.
- **QIRE-Bezug:** I-040-v2 (beta/lambda-Reihe)
- **Neuheitsprüfung:** Volltext: methodisches Vorbild, andere Achse.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:98`
- **Begriffe:** information-bottleneck, schusszahl

### `begusic2023clifford_perturbation` — Clifford Perturbation Theory
- **arXiv:** 2306.04797 · **Autoren:** Begusic, Chan (laut Recherche) · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Winkel nahe Clifford, Pauli-Pfade abschneiden.
- **QIRE-Bezug:** I-001
- **Neuheitsprüfung:** Genau die Winkelnaehe nach dem Runden.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:30`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:128`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:36`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:117`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:61` …
- **Begriffe:** clifford-naehe, klassische-simulierbarkeit

### `bennett_wiesner_superdense` — Superdense coding
- **arXiv:** 0708.3062 · **Autoren:** Bennett, Wiesner (1992) · **Jahr:** 1992
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** 2 Bit je gesendetem Qubit mit Verschraenkung.
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** arXiv-ID 0708.3062 als Fundstelle, Zuordnung nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:31`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:125`
- **Begriffe:** holevo

### `bermejo2024qcnn` — QCNNs are (effectively) classically simulable
- **arXiv:** 2408.12739 · **Autoren:** Bermejo et al. · **Jahr:** 2025 · **Venue:** PRX Quantum
- **Lesestatus:** Abstract
- **Kernaussage:** Klassische Surrogate erreichen oder uebertreffen QCNNs; 'locally easy' Datensaetze.
- **QIRE-Bezug:** Kernaussage, Einordnung
- **Neuheitsprüfung:** Warnung zur Benchmarkwahl.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:43`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:137`
- **Begriffe:** klassische-simulierbarkeit, benchmark-qml, dequantisierung

### `besiroglu2024chinchilla_repl` — Chinchilla Scaling: A replication attempt
- **arXiv:** 2404.10102 · **Autoren:** Besiroglu et al. · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Drei Probleme am Fit: zu enge Konfidenzintervalle, Optimierer vor Konvergenz gestoppt, gerundete Parameter.
- **QIRE-Bezug:** Skalierung (Fits fragil)
- **Neuheitsprüfung:** Lehre: Fits an wenigen Punkten fragil.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:18`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:128`
- **Begriffe:** skalierungsgesetze

### `bitnet158` — BitNet b1.58
- **arXiv:** 2402.17764 · **Autoren:** Ma u. a. · **Jahr:** 2024
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Ternaere Gewichte {-1,0,+1}*gamma.
- **QIRE-Bezug:** T-ternaer, IDEEN
- **Neuheitsprüfung:** Pflichtzitat; Vergleich Clifford-Zelle gegen BitNet-ternaer.
- **Fundstellen:** `INSTITUT/IDEEN.md:102`, `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md:131`, `INSTITUT/recherche/2026-10-07_volltext_2.md:124`
- **Begriffe:** quantisierung-ternaer, multiplikationsfrei

### `blier2018` — The Description Length of Deep Learning Models \*
- **arXiv:** 1802.07044 · **Autoren:** Blier, Ollivier · **Jahr:** 2018 · **Venue:** NeurIPS 2018
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Beste prequentielle Codelaenge fuer MNIST 4,10 kbit (Compression-Ratio 0,021).
- **QIRE-Bezug:** Bit-Budget-Pareto (05_parametereffizienz 4.1)
- **Neuheitsprüfung:** Parameterunabhaengige Vergleichsgroesse.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:17`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:98`
- **Begriffe:** parametereffizienz, mdl

### `boft2023` — BOFT (Orthogonal Butterfly Fine-Tuning) \*
- **arXiv:** 2311.06243 · **Autoren:** Liu et al. · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Orthogonale Butterfly-Faktorisierung, O(d log d) Parameter; schlaegt LoRA/OFT bei NLU und Mathe.
- **QIRE-Bezug:** T-006, Adapter
- **Neuheitsprüfung:** Naechste klassische Verwandte des QIRE-Adapters (d log d statt log d Parameter).
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:27`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:133`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:30`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:206`, `INSTITUT/recherche/2026-10-07_volltext_2.md:171`
- **Begriffe:** adapter-peft, butterfly, unitaer

### `boosari_walsh2026` — Walsh-Transform Realization of Dense-to-Sparse Quantum State Preparation
- **arXiv:** 2609.28760 · **Autoren:** Boosari · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Daten in den Walsh-Raum, nur groesste Koeffizienten laden, danach eine Hadamard-Lage ohne CNOT zurueck.
- **QIRE-Bezug:** Datenladen (Walsh-Raum), I-005/I-036
- **Neuheitsprüfung:** Walsh-Raum als Laderaum ist NICHT neu; nur die Kopplung mit Magie-Budget/Clifford-Orbit-Suche bleibt.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:28`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:112`
- **Begriffe:** zustandspraeparation, hadamard-walsh, approximatives-laden

### `born_nichtstab2026` — (Born-Maschinen und Nichtstabilitaet)
- **arXiv:** 2609.32025 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Treffer, nicht gelesen.
- **QIRE-Bezug:** Hardware/Magie
- **Neuheitsprüfung:** Nicht gelesen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:103`
- **Begriffe:** born-auslese, nichtstabilizerness-qml

### `born_text_2022` — Text Classification with Born's Rule
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** 2022 · **Venue:** NeurIPS 2022
- **Lesestatus:** Auszug · **unvollständig:** Autoren, arXiv-ID
- **Kernaussage:** Text als Ueberlagerung von Woertern; Uebergangswahrscheinlichkeit nach Born-Regel; einbettbar in NN.
- **QIRE-Bezug:** Aussage A (Name 'Born')
- **Neuheitsprüfung:** Gleicher Name, andere Konstruktion.
- **Begriffe:** born-auslese

### `bowles2024better` — Better than classical? The subtle art of benchmarking quantum machine learning models \*
- **arXiv:** 2403.07059 · **Autoren:** Bowles, Ahmed, Schuld · **Jahr:** 2024
- **Lesestatus:** Abstract · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** 12 QML-Modelle x 6 binaere Aufgaben (160 Datensaetze): klassische Out-of-the-box-Modelle gewinnen; Entfernen der Verschraenkung schadet oft nicht.
- **QIRE-Bezug:** FORSCHUNGSMETHODIK, Kernaussage, Clifford-Einordnung
- **Neuheitsprüfung:** Kontext/Warnung zur Benchmarkwahl.
- **Fundstellen:** `INSTITUT/FORSCHUNGSMETHODIK.md:40`, `INSTITUT/FORSCHUNGSMETHODIK.md:57`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:42`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:136`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:44` …
- **Begriffe:** benchmark-qml, quantenvorteil, klassische-simulierbarkeit

### `brassard2000amplitude` — Quantum Amplitude Amplification and Estimation (Titel aus Wissen) \*
- **arXiv:** quant-ph/0005055 · **Autoren:** Brassard, Hoyer, Mosca, Tapp · **Jahr:** 2000
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Fehler eps mit O(1/eps) Orakelaufrufen statt O(1/eps^2) Schuessen.
- **QIRE-Bezug:** Holevo/Schusszahl
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:33`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:127`
- **Begriffe:** amplitudenschaetzung, schusszahl

### `bravyi2016gosset` — Improved classical simulation of quantum circuits dominated by Clifford gates \*
- **arXiv:** 1601.07601 · **Autoren:** Bravyi, Gosset · **Jahr:** 2016 · **Venue:** Phys. Rev. Lett. 116, 250501 (2016)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Laufzeit poly + 2^(0,23 t) (Naeherung) bzw. 2^(0,5 t) (exakt) mit t = T-Zahl. (Titel aus Wissen des Katalogisierers, in der Recherche nur Autoren/Zeitschrift)
- **QIRE-Bezug:** C-hardware, Magie-Budget, I-036
- **Neuheitsprüfung:** Standardquelle fuer 'Kosten wachsen mit der T-Zahl'.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_hardware_demo.md:31`, `INSTITUT/recherche/2026-09-29_hardware_demo.md:81`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:104`
- **Begriffe:** klassische-simulierbarkeit, stabilizer-rang, t-zaehlung

### `bravyi2019lowrank` — Simulation of quantum circuits by low-rank stabilizer decompositions
- **arXiv:** 1808.00128 · **Autoren:** Bravyi et al. · **Jahr:** 2019
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Kosten wachsen exponentiell nur in der Zahl der Nicht-Clifford-Gatter.
- **QIRE-Bezug:** Magie-Budget
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:30`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:127`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:117`
- **Begriffe:** klassische-simulierbarkeit, stabilizer-rang

### `bravyi_kitaev2005` — (Universal quantum computation with ideal Clifford gates and noisy ancillas)
- **arXiv:** quant-ph/0403025 · **Autoren:** Bravyi, Kitaev · **Jahr:** 2005
- **Lesestatus:** nur Titel · **unvollständig:** Titel, Venue
- **Kernaussage:** Magic-State-Destillation; Primaerquelle fuer 'T-Gatter verbraucht einen destillierten Magic State' (nicht abgerufen).
- **QIRE-Bezug:** Hardware-Argument
- **Neuheitsprüfung:** Nicht gelesen.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_hardware_demo.md:49`
- **Begriffe:** magic-state-kosten

### `buhrman2001fingerprint` — Quantum Fingerprinting \*
- **arXiv:** quant-ph/0102001 · **Autoren:** Buhrman, Cleve, Watrous, de Wolf · **Jahr:** 2001 · **Venue:** PRL 87, 167902
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Gleichheitstest mit O(log n) Qubits. (Titel aus Wissen)
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:32`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:126`
- **Begriffe:** holevo

### `caballero2022bnsl` — Broken Neural Scaling Laws
- **arXiv:** 2210.14891 · **Autoren:** Caballero et al. · **Jahr:** 2022
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Gebrochenes Potenzgesetz modelliert Knicke, Double Descent.
- **QIRE-Bezug:** Skalierung
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:19`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:129`
- **Begriffe:** skalierungsgesetze

### `cafqa_folge_2308_11616` — (Folgearbeit zu CAFQA, Titel in der Recherche nicht genannt)
- **arXiv:** 2308.11616 · **Autoren:** unvollständig · **Jahr:** unvollständig
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Jahr, Titel, Venue
- **Kernaussage:** Laut ideen.json vom Institut notiert; Inhalt in den Recherchedateien nicht beschrieben.
- **QIRE-Bezug:** I-001
- **Neuheitsprüfung:** Als CAFQA-Folgearbeit gefuehrt; keine eigene Neuheitsbewertung.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:49`
- **Begriffe:** clifford-initialisierung

### `cafqa_ion2024` — (CAFQA auf Ionenfalle)
- **arXiv:** 2408.06482 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** CAFQA auf Ionenfalle demonstriert.
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:28`
- **Begriffe:** clifford-initialisierung

### `caro2022generalisierung` — Generalization in quantum machine learning from few training data \*
- **arXiv:** 2111.05292 · **Autoren:** Caro et al. · **Jahr:** 2022 · **Venue:** Nat. Commun. 2022
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Generalisierungsfehler <~ sqrt(T/N), T trainierbare Gatter, N Trainingsdaten. (Titel aus Wissen)
- **QIRE-Bezug:** Skalierung (Kopf dominiert die Schranke)
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:37`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:140`
- **Begriffe:** generalisierung-qml, skalierung

### `cayley_adapter2026` — Cayley Unitary Adapters
- **arXiv:** 2605.05914 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Abstract · **unvollständig:** Autoren, Venue
- **Kernaussage:** Cayley-Adapter in gefrorenen LLM-Projektionen, Zwei-Qubit-Bloecke.
- **QIRE-Bezug:** Aussage A, T-Linie (Adapter)
- **Neuheitsprüfung:** Nur Abstract bestaetigt; Details nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:29`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:124`, `INSTITUT/recherche/2026-10-07_volltext_2.md:169`
- **Begriffe:** adapter-peft, unitaer, born-auslese

### `cdr2021` — Error mitigation with Clifford quantum-circuit data (CDR) \*
- **arXiv:** 2005.10189 · **Autoren:** Czarnik, Arrasmith, Coles, Cincio · **Jahr:** 2021 · **Venue:** Quantum 2021 (Titel aus Wissen)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Regressionsmodell auf Clifford-nahen Trainingsschaltungen fuer Fehlerminderung.
- **QIRE-Bezug:** Hardware (C-hardware)
- **Neuheitsprüfung:** Clifford-NAH fuer nicht-Clifford-Ziel; Klassifikator + CDR nicht gefunden.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:33`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:102`
- **Begriffe:** fehlerminderung, clifford-naehe

### `cdr_extrapolation2025` — (CDR mit Non-Clifford-Extrapolation und Energy Sampling)
- **arXiv:** 2511.03556 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Erweiterung von CDR.
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:33`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:102`
- **Begriffe:** fehlerminderung

### `cerezo2025bp_sim` — Does provable absence of barren plateaus imply classical simulability?
- **arXiv:** 2312.09121 · **Autoren:** Cerezo et al. · **Jahr:** 2025 · **Venue:** Nat. Commun. 2025
- **Lesestatus:** Abstract
- **Kernaussage:** Trainierbarkeit und klassische Simulierbarkeit haengen oft zusammen (Fall fuer Fall, kein allgemeiner Satz).
- **QIRE-Bezug:** Kernaussage, Einordnung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:44`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:138`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:36`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:130`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:25` …
- **Begriffe:** klassische-simulierbarkeit, trainierbarkeit, dequantisierung

### `chen2015hashednets` — HashedNets
- **arXiv:** 1504.04788 · **Autoren:** Chen et al. · **Jahr:** 2015
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Bei 1/64-Kompression 1,99 % Fehler auf MNIST vs. 2,69 % bei gleich grossen Standardnetzen.
- **QIRE-Bezug:** Parametereffizienz (Zaehlfrage)
- **Neuheitsprüfung:** Die Hashfunktion zaehlt dort als frei.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:20`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:100`
- **Begriffe:** parametereffizienz, kompression

### `chen2022pixelfly` — Pixelated Butterfly: Simple and Efficient Sparse Training for Neural Network Models \*
- **arXiv:** 2112.00029 · **Autoren:** Chen, Dao u. a. · **Jahr:** 2022 · **Venue:** ICLR 2022
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Fussnote 4: Butterfly-Multiplikation erst ab n ca. 1.024 schneller als dicht; Pixelfly bis 2,3x schneller.
- **QIRE-Bezug:** I-041 (FWHT schneller ab N=1.024)
- **Neuheitsprüfung:** Volltext: externe Bestaetigung der Schwelle n ca. 1.024.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:47`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:56`
- **Begriffe:** strukturierte-schichten, butterfly

### `chen2022speicher` — Exponential separations between learning with and without quantum memory \*
- **arXiv:** 2111.05881 · **Autoren:** Chen, Cotler, Huang, Li · **Jahr:** 2022 · **Venue:** FOCS 2022 (Titel aus Wissen)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Ohne >= n Qubit Speicher Omega(2^((n-k)/3)) Kopien; mit Speicher O(n).
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:29`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:122`
- **Begriffe:** quantenspeicher

### `cheng2022clifford_init` — Clifford Circuit Initialisation for Variational Quantum Algorithms
- **arXiv:** 2207.01539 · **Autoren:** Cheng, Khosla, Self, Lin, Li, Medina, Kim · **Jahr:** 2022
- **Lesestatus:** Volltext · **unvollständig:** Venue
- **Kernaussage:** Simulated Annealing ueber Clifford-Parameterpunkte als Vor-Optimierung fuer VQAs; Domaene Quantenchemie; Autoren nennen Skalierungsgrenzen.
- **QIRE-Bezug:** I-005, I-036 (methodisch naechste Suche: SA ueber Cliffords), B
- **Neuheitsprüfung:** Clifford als Start, nicht als Endmodell; keine Magiezaehlung, keine Kompression.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:52`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:114`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:53`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:189`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:27` …
- **Begriffe:** clifford-initialisierung, clifford-suche-diskret, simulated-annealing

### `chia2022sampling` — Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning
- **arXiv:** 1910.06151 · **Autoren:** Chia, Gilyen, Li, Lin, Tang, Wang · **Jahr:** 2022 · **Venue:** J. ACM 69(5) (2022)
- **Lesestatus:** Auszug
- **Kernaussage:** Ganze Klasse von QML-Beschleunigungen faellt unter Sampling-Zugriff klassisch nur polynomiell langsamer.
- **QIRE-Bezug:** Einordnung
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:36`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:129`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:40`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:121`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:36` …
- **Begriffe:** dequantisierung

### `clapton2024` — Clapton
- **arXiv:** 2406.15721 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Clifford-unterstuetzte Problemtransformation fuer VQAs.
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Clifford als Initialisierung/Rauschmodell.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:34`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:102`
- **Begriffe:** clifford-initialisierung, fehlerminderung

### `clifford_equivariant2023` — Clifford Group Equivariant Neural Networks
- **arXiv:** 2305.11141 · **Autoren:** Ruhe u. a. · **Jahr:** 2023
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Clifford-ALGEBREN/-Gruppe (Multivektoren, O(n)/E(n)-Aequivarianz).
- **QIRE-Bezug:** Namensverwechslung
- **Neuheitsprüfung:** Nichts mit Clifford-Quantengattern; im Text ausdruecklich abgrenzen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:128`
- **Begriffe:** namensverwechslung

### `clifford_perturbation_em2024` — Clifford Perturbation Approximation for Quantum Error Mitigation
- **arXiv:** 2412.09518 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Winkel nahe an Clifford gehalten; Zweck Fehlerminderung.
- **QIRE-Bezug:** I-001
- **Neuheitsprüfung:** Anderer Zweck (Fehlerminderung).
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:35`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:162`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:185`
- **Begriffe:** clifford-naehe, fehlerminderung

### `clifford_twirling2024` — (Clifford-Twirling / Pauli-Fehlerpropagation)
- **arXiv:** 2412.01311 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe Hardware-Recherche.
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:34`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:102`
- **Begriffe:** fehlerminderung

### `cmps2024` — Clifford-augmentierte MPS
- **arXiv:** 2412.17209 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Cliffords zur Verschraenkungsreduktion.
- **QIRE-Bezug:** I-036-v2, Datenladen
- **Neuheitsprüfung:** Autoren nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:70`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:119`, `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md:36`
- **Begriffe:** tensornetz, clifford-simulation

### `compactifai2024` — CompactifAI
- **arXiv:** 2401.14109 · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** Multiverse
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** MPO-Zerlegung; LLaMA-2 7B auf 30 % der Groesse, > 90 % Genauigkeit (Firmenangaben teils Marketing).
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:35`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:128`
- **Begriffe:** tensornetz-kompression, sprachmodell

### `compositional_text2024` — Quantum Algorithms for Compositional Text Processing
- **arXiv:** 2408.06061 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Begleitpapier.
- **QIRE-Bezug:** Quanten-NLP
- **Neuheitsprüfung:** Nur Titel gesehen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:19`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:117`
- **Begriffe:** quanten-nlp

### `cotler2021revisiting` — Revisiting dequantization and quantum advantage in learning tasks \*
- **arXiv:** 2112.00811 · **Autoren:** Cotler, Huang, McClean · **Jahr:** 2021
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** SQ-Zugang moeglicherweise zu maechtiger Rohstoff; Vergleich evtl. unfair. (Titel aus Wissen)
- **QIRE-Bezug:** Einordnung
- **Neuheitsprüfung:** Gegenposition zu Tang.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:40`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:121`
- **Begriffe:** dequantisierung

### `courbariaux2016` — BinaryConnect / BNN
- **arXiv:** 1602.02830 · **Autoren:** unvollständig · **Jahr:** 2016
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** BinaryConnect 98,82 %, BNN 98,60 % auf MNIST.
- **QIRE-Bezug:** Parametereffizienz
- **Neuheitsprüfung:** Spart Bit pro Gewicht, nicht Gewichtsanzahl.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:21`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:101`
- **Begriffe:** quantisierung-ternaer, parametereffizienz

### `cvnn_kopf2026` — Complex-Valued Unitary Representations as Classification Heads for Improved Uncertainty Quantification in Deep Neural Networks
- **arXiv:** 2602.15283 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Komplexer Hilbertraum, Cayley-Unitaere; Auslese per Magnitude-Softmax oder Born-Regel; Magnitude-Softmax besser kalibriert (2,4x geringerer ECE auf CIFAR-10).
- **QIRE-Bezug:** Aussage A (Born vs. linear), Q-020 P4
- **Neuheitsprüfung:** Gegenbefund: Born NICHT besser, bei Kalibrierung schlechter; in der Diskussion von Q-020 P4 aufgreifen.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:21`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:112`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:121`
- **Begriffe:** born-auslese, klassifikationskopf, unitaer

### `dao2019butterfly` — Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations \*
- **arXiv:** 1903.05895 · **Autoren:** Dao u. a. · **Jahr:** 2019 · **Venue:** ICML 2019
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Butterfly-Faktorisierung lernt schnelle Transformationen.
- **QIRE-Bezug:** Butterfly-Linie
- **Neuheitsprüfung:** Pflichtzitat fuer Butterfly-Faktorisierung.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:127`
- **Begriffe:** butterfly, strukturierte-schichten

### `dao2022monarch` — Monarch: Expressive Structured Matrices for Efficient and Accurate Training \*
- **arXiv:** 2204.00595 · **Autoren:** Dao u. a. · **Jahr:** 2022 · **Venue:** ICML 2022
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Produkt zweier Blockdiagonalen; 1,7-2,0x schnelleres Training bei ViT/GPT-2 ohne Qualitaetsverlust; keine Schwellenbreite angegeben.
- **QIRE-Bezug:** I-041 (Zeitgewinn), T-Linie
- **Neuheitsprüfung:** Volltext: Schwellenbreite nicht zitierbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:46`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:56`
- **Begriffe:** strukturierte-schichten, butterfly, monarch

### `dao_monarch_mixer2023` — Monarch Mixer
- **arXiv:** 2310.12109 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Gleicht GPT-artige Transformer bei 360 M Parametern; BERT-Guete mit bis zu 27 % weniger Parametern.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:26`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:132`
- **Begriffe:** strukturierte-schichten, monarch

### `dehaene2003clifford` — Clifford group, stabilizer states, and linear and quadratic operations over GF(2) \*
- **arXiv:** quant-ph/0304125 · **Autoren:** Dehaene, De Moor · **Jahr:** 2003 · **Venue:** Phys. Rev. A 68, 042318 (2003)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Clifford-Gruppe ~ Gruppe aus linearen/quadratischen Operationen ueber GF(2) (Bruhat-/Quadratform-Darstellung). (Titel aus Wissen, in der Recherche nur Kurzbeschreibung)
- **QIRE-Bezug:** I-036-v3 (quadratische Vorzeichenmaske)
- **Neuheitsprüfung:** Pflichtzitat fuer die Quadratform-Darstellung der Clifford-Gruppe.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:42`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:45`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:257`
- **Begriffe:** clifford-quadratform, graphzustaende, gf2

### `delorenzis2025qelm` — Entanglement and Classical Simulability in Quantum Extreme Learning Machines
- **arXiv:** 2509.06873 · **Autoren:** De Lorenzis et al. · **Jahr:** 2025
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** MNIST, Fashion-MNIST, CIFAR-10; moderate Verschraenkung, klassisch simulierbar, nuetzliche Merkmale.
- **QIRE-Bezug:** I-036, Clifford-Zelle als QELM-Analogie
- **Neuheitsprüfung:** Literaturanker (Vergleich auf denselben Datensaetzen).
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:41`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:135`, `INSTITUT/recherche/2026-10-06_literatur_struktur.md:35`
- **Begriffe:** reservoir-computing, klassische-simulierbarkeit, qml-bilder

### `dequant_stoc2022` — (Chia/Gilyen u. a., Gharibian/Le Gall STOC 2022: QSVT-Dequantisierung)
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** unvollständig · **Venue:** STOC 2022
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Jahr, Titel, arXiv-ID
- **Kernaussage:** Dequantisierungsrahmen.
- **QIRE-Bezug:** Einordnung
- **Neuheitsprüfung:** Nur im Text erwaehnt.
- **Begriffe:** dequantisierung

### `dilip2022mps` — (MPS-Korrespondenz fuer Zustandspraeparation)
- **arXiv:** 2204.11170 · **Autoren:** Dilip, Liu, Smith, Pollmann · **Jahr:** 2022 · **Venue:** PRR 4, 043007
- **Lesestatus:** Auszug · **unvollständig:** Titel
- **Kernaussage:** 11 Qubit mit Tensornetz-Niveau.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:25`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:110`
- **Begriffe:** zustandspraeparation, tensornetz

### `duneau2024` — Scalable and interpretable QNLP: an implementation on trapped ions
- **arXiv:** 2409.08777 · **Autoren:** Duneau et al. · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** QDisCoCirc, Fragebeantwortung auf handgebauten Spielzeugdaten auf Quantinuum H1-1.
- **QIRE-Bezug:** Quanten-NLP
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:18`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:116`
- **Begriffe:** quanten-nlp

### `eat2023` — Energy-Aware Training (EAT)
- **arXiv:** 2307.00368 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Energieverbrauch von DL-Modellen senken.
- **QIRE-Bezug:** A/sources (energy_aware_training)
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/energy_aware_training.md:39`
- **Begriffe:** energieeffizienz

### `elivagar2024` — Elivagar
- **arXiv:** 2401.09393 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Effiziente Schaltungssuche fuer Klassifikation.
- **QIRE-Bezug:** I-005
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:35`
- **Begriffe:** architektursuche

### `entanglement_geometry_2607_17872` — Entanglement geometry separates circuit cutting, classical hardness, and trainability
- **arXiv:** 2607.17872 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Nur Titel gesehen.
- **QIRE-Bezug:** Rang <-> Verschraenkung (Etappe 2)
- **Neuheitsprüfung:** Vor Etappe 2 pruefen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:88`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:125`
- **Begriffe:** verschraenkung, trainierbarkeit

### `fano_2019` — (Fano-Ungleichung; Quelle in der Recherche nur als Link)
- **arXiv:** 1901.00555 · **Autoren:** unvollständig · **Jahr:** 2019
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Fano: I(C;C^) >= log2 K - h2(Pe) - Pe log2(K-1).
- **QIRE-Bezug:** Holevo/Fano
- **Neuheitsprüfung:** Standardwissen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:40`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:136`
- **Begriffe:** holevo, information-pro-schuss

### `farhi2014qaoa` — A Quantum Approximate Optimization Algorithm
- **arXiv:** 1411.4028 · **Autoren:** Farhi et al. · **Jahr:** 2014
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** QAOA.
- **QIRE-Bezug:** A/sources (qaoa_qubo)
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/qaoa_qubo.md:19`
- **Begriffe:** qaoa

### `fewshots2024` — Few measurement shots challenge generalization
- **arXiv:** 2411.06600 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Wenige Schuesse = dominante Fehlerquelle.
- **QIRE-Bezug:** I-040
- **Neuheitsprüfung:** Auszug.
- **Begriffe:** schusszahl, generalisierung-qml

### `finetuning_gesetze2025` — (zitiert Zhang et al. 2024: L(D) = A/D^lambda + E fuer LoRA)
- **arXiv:** 2512.14237 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Fine-Tuning-Gesetze.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:21`
- **Begriffe:** skalierungsgesetze, adapter-peft

### `firstphoton2024` — First Photon ML (Experiment zu Sbaiz)
- **arXiv:** 2410.17471 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Volltext · **unvollständig:** Autoren, Venue
- **Kernaussage:** Experiment: 31,0 % gegen 24,57 % klassisch.
- **QIRE-Bezug:** I-040-v2
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:58`
- **Begriffe:** information-pro-schuss, schusszahl

### `fourier_mps2024` — (Fourier-Moden-MPS)
- **arXiv:** 2311.07666 · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** Quantum 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel
- **Kernaussage:** Bilddaten mit schnell abfallendem Fourier-Spektrum haben kleinen Schmidt-Rang.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Autoren nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:26`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:111`
- **Begriffe:** zustandspraeparation, tensornetz

### `frankle2019lottery` — The Lottery Ticket Hypothesis \*
- **arXiv:** 1803.03635 · **Autoren:** Frankle, Carbin · **Jahr:** 2019 · **Venue:** ICLR 2019
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** LeNet-300-100 auf MNIST mit 1,8 % der Gewichte ca. 97 %.
- **QIRE-Bezug:** Parametereffizienz
- **Neuheitsprüfung:** Topologie (Indexbits) wird meist nicht mitgezaehlt.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:19`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:99`
- **Begriffe:** parametereffizienz, pruning-klassisch

### `frqi_neqr_2025` — (FRQI/NEQR-Vergleich fuer Klassifikation)
- **arXiv:** 2507.22039 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** FRQI (Le et al.), NEQR (Zhang et al. 2013).
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:31`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:113`
- **Begriffe:** datenkodierung

### `ghorbani2021kernel` — When do neural networks outperform kernel methods?
- **arXiv:** 2006.13409 · **Autoren:** Ghorbani u. a. · **Jahr:** 2021
- **Lesestatus:** Volltext · **unvollständig:** Venue
- **Kernaussage:** Fashion-MNIST: NN, RF, NT, RKHS vergleichbar; mit Rauschen verschlechtern sich RF/NT/RKHS schneller.
- **QIRE-Bezug:** Feste Zufallsmerkmale vs. trainierte Schicht
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:50`
- **Begriffe:** zufallsmerkmale

### `gidney2024cultivation` — Magic state cultivation: growing T states as cheap as CNOT gates
- **arXiv:** 2409.17595 · **Autoren:** Gidney, Shutty, Jones · **Jahr:** 2024
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** T-Zustaende innerhalb eines Oberflaechencode-Patches; Fehlerraten um 1e-9; Raumzeit-Kosten etwa Faktor 10 unter frueheren Verfahren.
- **QIRE-Bezug:** Hardware-Argument 'T ist teuer'
- **Neuheitsprüfung:** Relativiert den T-Preis (sinkt, verschwindet nicht).
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_hardware_demo.md:50`, `INSTITUT/recherche/2026-09-29_hardware_demo.md:82`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:82`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:85`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:286` …
- **Begriffe:** magic-state-kosten, t-zaehlung

### `gidney2025cultivation_surface` — Cultivation on the surface code
- **arXiv:** 2502.01743 · **Autoren:** Gidney · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Folgearbeit zur Oberflaechencode-Integration der Cultivation.
- **QIRE-Bezug:** Hardware-Argument
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:86`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:287`
- **Begriffe:** magic-state-kosten

### `gidney2025rsa` — (RSA-2048 mit < 1 Mio. verrauschter Qubits in < 1 Woche; Titel nicht genannt)
- **arXiv:** 2505.15917 · **Autoren:** Gidney · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Titel, Venue
- **Kernaussage:** RSA-2048 mit weniger als 1 Mio. verrauschter Qubits in unter einer Woche, u. a. durch Cultivation.
- **QIRE-Bezug:** Hardware-Argument
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:87`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:89`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:288`
- **Begriffe:** magic-state-kosten

### `gilfuster2025` — (Trainierbarkeit und Dequantisierung; Titel nicht genannt)
- **arXiv:** 2406.07072 · **Autoren:** Gil-Fuster, Gyurik, Perez-Salinas, Dunjko · **Jahr:** 2025 · **Venue:** ICLR 2025
- **Lesestatus:** Auszug · **unvollständig:** Titel
- **Kernaussage:** Formale Definition von Trainierbarkeit und Dequantisierung; ob erstes letzteres impliziert.
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Ergebnis nicht gelesen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:28`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:113`
- **Begriffe:** dequantisierung, trainierbarkeit

### `goh2025gsim` — (g-sim: Simulation ueber die Lie-Algebra)
- **arXiv:** 2308.01432 · **Autoren:** Goh, Larocca, Cincio, Cerezo, Sauvage · **Jahr:** 2025 · **Venue:** Phys. Rev. Research 7, 033266
- **Lesestatus:** Auszug · **unvollständig:** Titel
- **Kernaussage:** Effizient, wenn DLA-Dimension polynomiell; Magiezustaende als Anfangszustand moeglich.
- **QIRE-Bezug:** Lie-Algebra/T-006
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:33`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:115`
- **Begriffe:** lie-algebra, klassische-simulierbarkeit

### `gonon2025kronecker_sparse` — Fast Inference with Kronecker-Sparse Matrices
- **arXiv:** unvollständig (keine ID) · **Autoren:** Gonon et al. · **Jahr:** 2025 · **Venue:** ICML 2025
- **Lesestatus:** Auszug · **unvollständig:** arXiv-ID
- **Kernaussage:** Butterfly/Monarch-Muster: Median 1,4x schneller (FP32), nicht immer schneller als dicht.
- **QIRE-Bezug:** I-041
- **Neuheitsprüfung:** Mittlere Naehe ('ab welcher Breite').
- **Begriffe:** strukturierte-schichten

### `gosset_kothari` — Quantum State Preparation with Optimal T-Count
- **arXiv:** 2411.04790 · **Autoren:** Gosset, Kothari · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** eps-Naeherung beliebiger n-Qubit-Zustaende mit O(sqrt(2^n log(1/eps))) T-Gattern, untere Schranke gleich.
- **QIRE-Bezug:** Datenladen (T-Zahl des Ladens)
- **Neuheitsprüfung:** Auszug; Volltextlesen offen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:19`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:105`
- **Begriffe:** zustandspraeparation, t-zaehlung

### `gottesman1998` — The Heisenberg representation of quantum computers
- **arXiv:** quant-ph/9807006 · **Autoren:** Gottesman · **Jahr:** 1998
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Gottesman-Knill: Stabilizer-Schaltungen klassisch effizient simulierbar.
- **QIRE-Bezug:** Linie Clifford, ehrliche Formulierung
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:56`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:132`
- **Begriffe:** gottesman-knill, klassische-simulierbarkeit

### `gpu_sim2025` — (GPU-Simulation: cuQuantum, PennyLane-Lightning-GPU u. a.)
- **arXiv:** 2509.04955 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe LABORBUCH.
- **QIRE-Bezug:** Simulation
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/LABORBUCH.md:771`
- **Begriffe:** simulation-gpu

### `graph_0812_4625` — (Graphzustaende; Titel nicht genannt)
- **arXiv:** 0812.4625 · **Autoren:** unvollständig · **Jahr:** 2008
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Quelle in 09_graphzustaende.
- **QIRE-Bezug:** Graphzustaende (09)
- **Neuheitsprüfung:** Titel fehlt.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:39`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:52`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:256`
- **Begriffe:** graphzustaende

### `grewal2025t_dotiert` — (Lernen von t-dotierten Stabilizerzustaenden; Titel nicht genannt)
- **arXiv:** unvollständig (keine ID) · **Autoren:** Grewal, Iyer, Kretschmer, Liang · **Jahr:** 2025 · **Venue:** Quantum 9, 1907 (2025); Einzelkopie-Variante Quantum 8, 1250 (2024)
- **Lesestatus:** Auszug · **unvollständig:** Titel, arXiv-ID
- **Kernaussage:** Lernzeit poly(n, 2^t, 1/eps) fuer Zustaende aus Clifford + t Nicht-Clifford-Gattern.
- **QIRE-Bezug:** Magiebedarf M50
- **Neuheitsprüfung:** Naechster Verwandter von 'Magie bestimmt Lernkosten', aber nicht dasselbe wie M50.
- **Begriffe:** nichtstabilizerness-qml, lernbarkeit-stabilizer

### `gu2024pseudomagie` — Pseudomagic quantum states \*
- **arXiv:** 2308.16228 · **Autoren:** Gu, Leone, Ghosh, Eisert, Yelin, Quek · **Jahr:** 2024 · **Venue:** Phys. Rev. Lett. 132, 210602 (2024)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Zustaende mit wenig Magie, fuer beschraenkte Beobachter nicht von maximaler Magie unterscheidbar; Magie nicht effizient messbar. (Titel aus Zitat abgeleitet, in der Recherche nicht woertlich genannt)
- **QIRE-Bezug:** I-002, I-035
- **Neuheitsprüfung:** Betrifft rechenbeschraenkte Unterscheidung, nicht exakt berechnete SRE kleiner Zellen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:133`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:293`
- **Begriffe:** sre, pseudomagie

### `hadamard_adapter2024` — Hadamard-Adapter
- **arXiv:** 2407.11033 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Elementweise Skalierung.
- **QIRE-Bezug:** I-045-v2
- **Neuheitsprüfung:** 'Multiplikativ' meint Gewichtsprodukte, nicht Quadrat der Aktivierung.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_literatur_struktur_quadrat.md:7`
- **Begriffe:** adapter-peft, hadamard-produkt-verwechslung

### `hadamard_klassisch_hadanet2019` — HadaNets
- **arXiv:** 1905.10759 · **Autoren:** unvollständig · **Jahr:** 2019
- **Lesestatus:** Abstract · **unvollständig:** Autoren, Venue
- **Kernaussage:** Hadamard-PRODUKT (elementweise), nicht Hadamard-Transformation.
- **QIRE-Bezug:** T-ternaer
- **Neuheitsprüfung:** Nur Namensgleichheit; hoechstens Randzitat.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:129`
- **Begriffe:** hadamard-produkt-verwechslung

### `hadamard_muster_mnist2024` — (Klassifikation aus Hadamard-Basis-Messungen)
- **arXiv:** 2407.12506 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** MNIST/Fashion aus 64 Hadamard-Mustern: klassisch ca. 96/84 %, QNN 95/81 %.
- **QIRE-Bezug:** I-036
- **Neuheitsprüfung:** Zahlen ungeprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:90`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:143`
- **Begriffe:** hadamard-walsh, qml-bilder

### `hadamard_nn_2018` — Energy Efficient Hadamard Neural Networks
- **arXiv:** 1805.05421 · **Autoren:** unvollständig · **Jahr:** 2018
- **Lesestatus:** Abstract · **unvollständig:** Autoren, Venue
- **Kernaussage:** Hadamard-Produkt Gewicht = FP x binaer.
- **QIRE-Bezug:** T-ternaer
- **Neuheitsprüfung:** Nicht verwechseln.
- **Begriffe:** hadamard-produkt-verwechslung

### `han2016deepcompression` — Deep Compression: ... Pruning, Trained Quantization and Huffman Coding \*
- **arXiv:** 1510.00149 · **Autoren:** Han et al. · **Jahr:** 2016 · **Venue:** ICLR 2016
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Pruning + Quantisierung + Huffman.
- **QIRE-Bezug:** A/sources (pruning)
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/pruning.md:14`
- **Begriffe:** pruning-klassisch, quantisierung-ternaer

### `hein2003schmidt` — Multiparty entanglement in graph states \*
- **arXiv:** quant-ph/0307130 · **Autoren:** Hein et al. (laut Recherche) · **Jahr:** 2003
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Schmidt-Rang/-Mass fuer jede Bipartition A|B bei Graphzustaenden. (Titel aus Wissen, unvollstaendig belegt)
- **QIRE-Bezug:** Rang <-> Verschraenkung, V-kompass
- **Neuheitsprüfung:** Standardwissen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:33`
- **Begriffe:** graphzustaende, schmidt-rang

### `heyraud2023` — (Gradientenvarianz zufaellig initialisierter Schaltungen ueber Clifford-Abbildung)
- **arXiv:** unvollständig (keine ID) · **Autoren:** Heyraud et al. · **Jahr:** 2023 · **Venue:** PRX Quantum 4, 040335 (doi 10.1103/PRXQuantum.4.040335)
- **Lesestatus:** Auszug · **unvollständig:** Titel, arXiv-ID
- **Kernaussage:** Gradientenvarianz exakt ueber Abbildung auf Clifford-Schaltungen berechenbar.
- **QIRE-Bezug:** Magie-Dosis/Trainierbarkeit
- **Neuheitsprüfung:** Hintergrund.
- **Begriffe:** trainierbarkeit, clifford-naehe

### `hoffmann2022chinchilla` — Training Compute-Optimal Large Language Models (Chinchilla) \*
- **arXiv:** 2203.15556 · **Autoren:** Hoffmann et al. · **Jahr:** 2022
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Ueber 400 Modelle; Modellgroesse und Tokenzahl gleich skalieren.
- **QIRE-Bezug:** Skalierung (T-006)
- **Neuheitsprüfung:** Nur Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:17`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:127`
- **Begriffe:** skalierungsgesetze

### `holevo_n_bit2026` — (Holevo-Grenze und Amplitudenkodierung)
- **arXiv:** 2606.26312 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** n Qubit liefern pro Kopie hoechstens n klassische Bit.
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:47`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:118`
- **Begriffe:** holevo, amplitudenkodierung

### `holmes_matsuura` — (Glatte Funktionen als MPS)
- **arXiv:** 2303.01562 · **Autoren:** Holmes, Matsuura · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Titel, Venue
- **Kernaussage:** Eine Lage Zwei-Qubit-Gatter, bis 20 Qubit auf Ionenfalle.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:26`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:111`
- **Begriffe:** zustandspraeparation, tensornetz

### `hu2023eigentasks` — Tackling Sampling Noise in Physical Systems for Machine Learning Applications: Fundamental Limits and Eigentasks \*
- **arXiv:** 2307.16083 · **Autoren:** Hu, Angelatos, Khan u. a. · **Jahr:** 2023 · **Venue:** Phys. Rev. X 13 (2023)
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Fuer 2-Designs gilt C_T = K(S+1)/(S+K): maximale Expressibilitaet => Kapazitaet verschwindet bei festem S. Schaltung wird nicht trainiert. (Titel aus Wissen)
- **QIRE-Bezug:** I-040-v2 H4 (Haar schadet bei wenigen Schuessen)
- **Neuheitsprüfung:** Volltext: Theorie zu H4; Mass ist Kapazitaet, nicht I(Klasse;1 Schuss). Zwingend zitieren.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:92`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:57`
- **Begriffe:** information-pro-schuss, schusszahl, reservoir-computing

### `huang2020shadows` — Predicting many properties of a quantum system from very few measurements (Classical Shadows) \*
- **arXiv:** 2106.10190 · **Autoren:** Huang, Kueng, Preskill · **Jahr:** 2020 · **Venue:** Nature Phys. 16, 1050 (2020)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** M Eigenschaften mit O(log M) Zufallsmessungen. (arXiv-ID-Zuordnung unsicher: Recherche nennt 2412.03381 und 2106.10190)
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:27`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:120`
- **Begriffe:** shadow-tomography

### `hw_2606_21686` — (in 08_hardware bei CDR genannt)
- **arXiv:** 2606.21686 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Endlich-Schuss-Analyse CDR vs. ?
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:33`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:102`
- **Begriffe:** fehlerminderung

### `hw_2607_grosse` — (in 01_holevo bei Schusszahl vs. Marge genannt)
- **arXiv:** 2608.24891 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 01_holevo 2.3.
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:41`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:137`
- **Begriffe:** schusszahl

### `hyperband2018` — Hyperband
- **arXiv:** 1603.06560 · **Autoren:** Li et al. · **Jahr:** 2018 · **Venue:** JMLR 2018
- **Lesestatus:** Auszug
- **Kernaussage:** Bandit-basierte Hyperparametersuche.
- **QIRE-Bezug:** A/sources (hyperband)
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/hyperband.md:18`
- **Begriffe:** hyperparametersuche

### `hyperq2026` — HyperQ - Circuit Hypernetworks
- **arXiv:** 2609.24657 · **Autoren:** Wang et al. (Montreal/Mila) · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** IQP-Schaltungen exakt klassisch auswertbar; 1,1-Mrd.-LM, 64 Qubits +4,71 ggue. Backbone, +3,67 ggue. LoRA.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Naechster Verwandter zu Clifford/diskret im LLM-Adapter.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:30`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:125`
- **Begriffe:** adapter-peft, quanten-nlp, klassische-simulierbarkeit

### `hyquT2025` — Hybrid Quantum Transformer (HyQuT)
- **arXiv:** 2511.10653 · **Autoren:** Kong · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** VQCs im Transformer bei 8M/150M Parametern; 10 Qubits/80 Gatter ersetzen ca. 10 % der Parameter.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Simulation unklar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:25`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:120`
- **Begriffe:** quanten-nlp, adapter-peft

### `i036_v2_2605` — (Quelle der Neuheitspruefung I-036-v2; CRiSP 2026 evtl. dieselbe)
- **arXiv:** 2605.23138 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Clifford-Initialisierung von VQAs (PRA 111, 062413, 2025; CRiSP 2026) - Zuordnung ungeklaert.
- **QIRE-Bezug:** I-036-v2
- **Neuheitsprüfung:** Clifford als Start.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md:34`
- **Begriffe:** clifford-initialisierung

### `i036_v2_2608` — (Quelle der Neuheitspruefung I-036-v2)
- **arXiv:** 2608.01653 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Titel nicht genannt.
- **QIRE-Bezug:** I-036-v2
- **Neuheitsprüfung:** Nur als Quelle gelistet.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md:36`
- **Begriffe:** clifford-simulation

### `i036_v2_html` — (Quelle der Neuheitspruefung I-036-v2)
- **arXiv:** 2408.03351 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Titel nicht genannt.
- **QIRE-Bezug:** I-036-v2
- **Neuheitsprüfung:** Nur als Quelle gelistet.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md:33`
- **Begriffe:** clifford-simulation

### `i036_v2_pdf` — (Quelle der Neuheitspruefung I-036-v2)
- **arXiv:** 2303.15665 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Titel nicht genannt.
- **QIRE-Bezug:** I-036-v2
- **Neuheitsprüfung:** Nur als Quelle gelistet.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md:33`
- **Begriffe:** clifford-simulation

### `ilias2026einzelschuss` — (Training bei fester Schusszahl)
- **arXiv:** 2512.20492 · **Autoren:** Ilias, Hu, Vives, Tuereci · **Jahr:** 2026
- **Lesestatus:** Volltext · **unvollständig:** Titel, Venue
- **Kernaussage:** Schaltung bei festem S auf das Aufgabenrisiko trainiert; fuer S->inf optimierte Parameter bleiben bei kleinem Budget nicht optimal; Bayessche Metrologie.
- **QIRE-Bezug:** I-040-v2
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:93`
- **Begriffe:** schusszahl

### `ivaki2025reservoir` — Optimal quantum reservoir learning in proximity to universality
- **arXiv:** 2510.18623 · **Autoren:** Ivaki, Karjula, Ala-Nissila · **Jahr:** 2025 · **Venue:** PRA 2025/26 (laut Recherche)
- **Lesestatus:** Abstract
- **Kernaussage:** Anteil p der Clifford-Gatter durch nicht-stabilisierende ersetzt; Leistung korreliert mit Verschraenkungsspektrum und langreichweitiger Magie; Optimum nahe Universalitaet.
- **QIRE-Bezug:** Magie-Dosis, I-036
- **Neuheitsprüfung:** Optimum-Befund, keine Konzentration; Reservoir-Aufgaben, nicht Bildklassifikation.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:78`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:141`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:109`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:112`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:128` …
- **Begriffe:** nichtstabilizerness-qml, reservoir-computing, magie-dosis

### `jaques_rattew_qram` — QRAM: A Survey and Critique
- **arXiv:** 2305.10310 · **Autoren:** Jaques, Rattew · **Jahr:** 2025 · **Venue:** Quantum 9, 1922 (2025)
- **Lesestatus:** Auszug
- **Kernaussage:** Vorteile schrumpfen, wenn Steuerhardware und klassische Parallelitaet mitgerechnet werden.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:37`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:114`
- **Begriffe:** qram

### `kaleidoscope2020` — Kaleidoscope matrices \*
- **arXiv:** unvollständig (keine ID) · **Autoren:** Dao et al. · **Jahr:** 2020 · **Venue:** ICLR 2020
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, arXiv-ID
- **Kernaussage:** K-Matrizen fassen jede strukturierte Matrix; ShuffleNet +5 % ImageNet.
- **QIRE-Bezug:** Strukturierte Schichten
- **Neuheitsprüfung:** Hohe Naehe.
- **Begriffe:** strukturierte-schichten, butterfly

### `karnick` — Random features for dot product kernels / Improved Random Features for Dot Product Kernels; TensorSketch/TensorSRHT
- **arXiv:** unvollständig (keine ID) · **Autoren:** Kar, Karnick; Wacker et al. · **Jahr:** unvollständig · **Venue:** JMLR (Wacker et al.)
- **Lesestatus:** Auszug · **unvollständig:** Jahr, arXiv-ID
- **Kernaussage:** Zufallsmerkmale fuer Polynomkerne, strukturierte Sketches.
- **QIRE-Bezug:** Quadrat = Grad-2-Polynomkern
- **Neuheitsprüfung:** Hohe Naehe.
- **Begriffe:** polynomkern, zufallsmerkmale

### `kissinger2020t_reduktion` — Reducing the number of non-Clifford gates in quantum circuits
- **arXiv:** unvollständig (keine ID) · **Autoren:** Kissinger, van de Wetering · **Jahr:** 2020 · **Venue:** Phys. Rev. A 102 (2020)
- **Lesestatus:** Auszug · **unvollständig:** arXiv-ID
- **Kernaussage:** T-Zahl-Reduktion im ZX-Kalkuel, inkl. Parameter-Elimination fuer variationale Schaltungen; aequivalenzerhaltend.
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Compiler-Perspektive; nicht lernend/approximativ.
- **Begriffe:** t-zaehlung, zx-kalkuel

### `kliuchnikov2023synthese` — (Shorter quantum circuits via single-qubit gate approximation; Titel nicht genannt)
- **arXiv:** 2203.10064 · **Autoren:** Kliuchnikov, Lauter, Minko, Paetznick, Petit · **Jahr:** 2023 · **Venue:** Quantum 7, 1208 (2023)
- **Lesestatus:** Auszug · **unvollständig:** Titel
- **Kernaussage:** Sequenzlaenge Faktor 7/9, gemischte Kanaele sparen ca. Faktor 2.
- **QIRE-Bezug:** T-Zaehlung
- **Neuheitsprüfung:** Konstanten nicht bestaetigt.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:72`
- **Begriffe:** t-zaehlung, gate-synthese

### `kodierung_uebersicht2026` — (Merkmalskodierung Uebersicht)
- **arXiv:** 2606.05387 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Mottonen-artig ca. 980 CNOT; Untergrenze ca. 123; ca. 510 CNOT bei D=256 mit Fidelity ca. 0,21 bei 3e-3.
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Auszug (Umfrage).
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:37`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:103`
- **Begriffe:** amplitudenkodierung, zustandspraeparation

### `krona2022` — KronA
- **arXiv:** 2212.10650 · **Autoren:** Edalati et al. · **Jahr:** 2022
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Kronecker-Adapter.
- **QIRE-Bezug:** T-006 (QIRE-Kronecker-Drehung), Adapter
- **Neuheitsprüfung:** Direkter klassischer Konkurrent zur QIRE-Kronecker-Drehung mit log2(Breite) Winkeln.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:28`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:134`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:30`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:206`
- **Begriffe:** adapter-peft, kronecker, quanten-peft

### `kronlora2025` — Kron-LoRA / LoKr
- **arXiv:** 2508.01961 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Bis 4x weniger Parameter als LoRA bei aehnlicher Ausdrucksstaerke.
- **QIRE-Bezug:** Adapter
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:28`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:134`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:206`
- **Begriffe:** adapter-peft, kronecker

### `kulshrestha2024qadaprune` — QAdaPrune: Adaptive Parameter Pruning For Training Variational Quantum Circuits
- **arXiv:** 2408.13352 · **Autoren:** Kulshrestha et al. · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Adaptives Parameter-Pruning (Parameter nahe 0 entfernen) fuer VQCs mit vergleichbarer Leistung.
- **QIRE-Bezug:** I-001 (Clifford-Rundung nach Verlustanstieg)
- **Neuheitsprüfung:** Pruning, nicht Clifford-Rundung; Auswahlregel Naehe zu 0.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:61`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:103`
- **Begriffe:** pruning-pqc, magie-kompression

### `kyriienko2025diskret` — Advantage for Discrete Variational Quantum Algorithms in Circuit Recompilation
- **arXiv:** 2510.01154 · **Autoren:** Kyriienko, Umeano, Holmes · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Diskrete Winkel in VQAs, hohe Magie; Kompilierung.
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Kompilierung statt ML-Kompression.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:66`
- **Begriffe:** diskrete-winkel, t-zaehlung

### `lambeq2021` — lambeq
- **arXiv:** 2110.04236 · **Autoren:** unvollständig · **Jahr:** 2021 · **Venue:** Quantinuum
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** Python-Bibliothek, Saetze (DisCoCat) -> Quantenschaltungen.
- **QIRE-Bezug:** Quanten-NLP
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:16`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:117`
- **Begriffe:** quanten-nlp

### `lambeq_gen2_2025` — Efficient Generation of Parameterised Quantum Circuits from Large Texts
- **arXiv:** 2505.13208 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** lambeq Gen II, Text-Ebene.
- **QIRE-Bezug:** Quanten-NLP
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:17`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:117`
- **Begriffe:** quanten-nlp

### `landman2023rff` — Classically approximating variational quantum machine learning with random Fourier features \*
- **arXiv:** 2210.13200 · **Autoren:** Landman et al. · **Jahr:** 2023 · **Venue:** ICLR 2023 (laut Recherche)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** VQCs mit Hamilton-Kodierung klassisch per Random Fourier Features approximierbar.
- **QIRE-Bezug:** I-038 (Wirkung klassisch)
- **Neuheitsprüfung:** Stuetzt 'Wirkung klassisch'.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:45`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:140`
- **Begriffe:** dequantisierung, zufallsmerkmale

### `larocca2021lie` — (Saettigung beim Rang der dynamischen Lie-Algebra)
- **arXiv:** 2109.11676 · **Autoren:** Larocca u. a. · **Jahr:** 2021
- **Lesestatus:** Volltext · **unvollständig:** Titel, Venue
- **Kernaussage:** Kritische Parameterzahl M_c <= dim(g); QFIM-/Hessian-Rang saettigen bei dim(g)-1; Berechnungs-Phasenuebergang der Trainierbarkeit.
- **QIRE-Bezug:** T-006 / Lie-Skalierung (v5)
- **Neuheitsprüfung:** Volltext-Zeile: Paper.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:174`
- **Begriffe:** lie-algebra, overparametrisierung

### `larocca2025uebersicht` — (Uebersicht ueber Barren Plateaus)
- **arXiv:** 2405.00781 · **Autoren:** Larocca et al. · **Jahr:** 2025 · **Venue:** Nature Reviews Physics (s42254-025-00813-9)
- **Lesestatus:** nur Titel · **unvollständig:** Titel
- **Kernaussage:** Uebersichtsartikel.
- **QIRE-Bezug:** Trainierbarkeit
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:108`
- **Begriffe:** trainierbarkeit, barren-plateaus

### `le2013fastfood` — Fastfood - Approximating Kernel Expansions in Loglinear Time
- **arXiv:** 1408.3060 · **Autoren:** Le, Sarlos, Smola · **Jahr:** 2013 · **Venue:** ICML 2013
- **Lesestatus:** Volltext
- **Kernaussage:** Hadamard*Diagonal ersetzt Gauss-Matrix; O(n log d) Zeit, O(n) Speicher; Polynomkerne nur ueber Legendre-Entwicklung (Grad 10, UCI-Regression); CIFAR-10 mit RBF.
- **QIRE-Bezug:** I-038 (Fastfood-Mischung), I-041, I-044-v2, T-ternaer
- **Neuheitsprüfung:** Volltext gelesen: Polynomkern Grad 2 mit Hadamard auf MNIST NICHT gemessen. Strukturierte Mischung ist bekannt, nicht als eigener Befund verkaufen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md:125`, `INSTITUT/recherche/2026-10-06_literatur_struktur.md:18`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:55`
- **Begriffe:** strukturierte-schichten, hadamard-walsh, zufallsmerkmale, polynomkern

### `leone2022sre` — Stabilizer Renyi entropy
- **arXiv:** 2106.12587 · **Autoren:** Leone, Oliviero, Hamma · **Jahr:** 2022 · **Venue:** Phys. Rev. Lett. 128, 050402 (2022)
- **Lesestatus:** Abstract
- **Kernaussage:** Gut berechenbares Magiemass ohne Minimierung; Monoton der Ressourcentheorie.
- **QIRE-Bezug:** I-002, I-035, Magie-Messung
- **Neuheitsprüfung:** Standardquelle fuer SRE.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:59`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:134`
- **Begriffe:** sre, magie-mass

### `li2018intrinsic` — Measuring the Intrinsic Dimension of Objective Landscapes \*
- **arXiv:** 1804.08838 · **Autoren:** Li, Farkhoor, Liu, Yosinski · **Jahr:** 2018 · **Venue:** ICLR 2018
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** MLP 784-200-200-10 erreicht 90 % mit d_int ca. 750.
- **QIRE-Bezug:** Parametereffizienz
- **Neuheitsprüfung:** Nur Suchauszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:18`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:96`
- **Begriffe:** parametereffizienz, intrinsische-dimension

### `lipardi2026magic_qas` — Magic-Informed Quantum Architecture Search
- **arXiv:** 2605.03932 · **Autoren:** Lipardi et al. · **Jahr:** 2026
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** MCTS + GNN steuern die Magie (SRE) von Kandidatenschaltungen; getestet auf Grundzustandsenergien/Zustandsapproximation.
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Magie als Suchsignal, nicht Kosten/Nutzen je Aufgabe; nicht auf Klassifikation getestet.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:36`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:131`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:121`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:123`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:297`
- **Begriffe:** magie-suchsignal, architektursuche, sre

### `litinski2019` — (Magic State Distillation: Not as Costly as You Think / Game of Surface Codes; Titel in der Recherche nicht genannt)
- **arXiv:** 1905.06903 · **Autoren:** Litinski · **Jahr:** 2019
- **Lesestatus:** nur Titel · **unvollständig:** Titel, Venue
- **Kernaussage:** Primaerquelle fuer T-Kosten fehlertoleranter Rechner (nicht abgerufen).
- **QIRE-Bezug:** Hardware-Argument
- **Neuheitsprüfung:** Nicht gelesen.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_hardware_demo.md:49`
- **Begriffe:** magic-state-kosten

### `livni2014` — On the Computational Efficiency of Training Neural Networks
- **arXiv:** unvollständig (keine ID) · **Autoren:** Livni et al. · **Jahr:** 2014 · **Venue:** NeurIPS 2014
- **Lesestatus:** Auszug · **unvollständig:** arXiv-ID
- **Kernaussage:** Netze mit x^2 als Aktivierung (Polynomnetze) so ausdrucksstark wie Schwellennetze, effizient lernbar.
- **QIRE-Bezug:** Quadrat-Aktivierung
- **Neuheitsprüfung:** Mittlere Naehe.
- **Begriffe:** quadrat-aktivierung

### `lm_tensor_trains2024` — Language Modeling Using Tensor Trains
- **arXiv:** 2405.04590 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** TT-Sprachmodell, Details unklar.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:38`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:129`
- **Begriffe:** tensornetz-kompression, sprachmodell

### `lora_miub2025` — MIUB-Skalierungsgesetz fuer LoRA
- **arXiv:** 2501.03152 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Im PDF nur Zusammenfassung, keine Exponentenzahl.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:152`, `INSTITUT/recherche/2026-10-07_volltext_2.md:177`
- **Begriffe:** adapter-peft, skalierungsgesetze

### `lu2023qasbench` — QAS-Bench
- **arXiv:** unvollständig (keine ID) · **Autoren:** Lu et al. · **Jahr:** 2023 · **Venue:** ICML 2023 (PMLR 202)
- **Lesestatus:** Auszug · **unvollständig:** arXiv-ID
- **Kernaussage:** Vergleicht Brute Force, SA, GA, RL, differenzierbare Suche; SA skaliert schwach.
- **QIRE-Bezug:** I-005, I-036
- **Neuheitsprüfung:** Auszug.
- **Begriffe:** architektursuche, simulated-annealing

### `luo_li_sparse` — Sparse Quantum State Preparation (Luo-Li)
- **arXiv:** 2608.00414 · **Autoren:** Luo, Li · **Jahr:** unvollständig
- **Lesestatus:** nur Titel · **unvollständig:** Jahr, Venue
- **Kernaussage:** T-Zahl ~ sqrt(s) fuer s-duenne Zustaende (ID-Zuordnung unsicher).
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Zuordnung ungeprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:19`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:106`
- **Begriffe:** zustandspraeparation

### `ma2023photonen` — (Stochastisches Training fuer wenige Photonen)
- **arXiv:** 2307.15712 · **Autoren:** Ma, Wang, Laydevant, Wright, McMahon · **Jahr:** 2023
- **Lesestatus:** Volltext · **unvollständig:** Titel, Venue
- **Kernaussage:** MNIST 98 % bei 0,038 Photonen je MAC; Training modelliert Photodetektions-Statistik.
- **QIRE-Bezug:** I-040-v2
- **Neuheitsprüfung:** 'Fuer das Rauschmodell trainieren' ist bekannt.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:96`
- **Begriffe:** schusszahl

### `magic_robustness_1807_10296` — (Robustheit der Magie / Mana / Stabilizer-Polytop; Zuordnung unklar)
- **arXiv:** 1807.10296 · **Autoren:** unvollständig · **Jahr:** 2018
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Howard & Campbell 2017; Veitch et al.; Heinrich & Gross 2019 - ar5iv-Link nennt 1807.10296.
- **QIRE-Bezug:** Magie-Ressourcentheorie
- **Neuheitsprüfung:** Zuordnung zu einer der drei Arbeiten ungeklaert.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:43`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:290`
- **Begriffe:** magie-mass, ressourcentheorie

### `magie_vorteil2024` — Unconditional quantum magic advantage in shallow circuit computation
- **arXiv:** 2402.12246 · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** Nat. Commun. 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** Magie noetig fuer Vorteil bei bestimmten Aufgaben.
- **QIRE-Bezug:** B, Einordnung
- **Neuheitsprüfung:** Theorie, andere Frage.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:126`
- **Begriffe:** nichtstabilizerness-qml, quantenvorteil

### `magie_vorteil_zurueck2025` — Quantenvorteil bis auf Magie zurueckverfolgt (Gottesman-Knill-Grenze der Ein-Weg-Kommunikation; exakter Titel in der Recherche nicht angegeben)
- **arXiv:** 2506.19369 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Quantenvorteil auf Magie zurueckgefuehrt.
- **QIRE-Bezug:** B, Einordnung
- **Neuheitsprüfung:** Theorie.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:125`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:127`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:304`
- **Begriffe:** nichtstabilizerness-qml, quantenvorteil

### `maronese2026expressibility` — High-expressibility Quantum Neural Networks using only classical resources
- **arXiv:** 2506.13605 · **Autoren:** Maronese, Ferrari, Vandelli, Dragoni · **Jahr:** 2026 · **Venue:** Quantum Mach. Intell. 2026
- **Lesestatus:** Abstract
- **Kernaussage:** Clifford-erweiterte MPS naehern die Haar-Verteilung schneller an (Verschraenkung und Magie).
- **QIRE-Bezug:** B, Magie-Dosis
- **Neuheitsprüfung:** Expressibilitaet statt Kompression; keine Klassifikationsergebnisse.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:65`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:79`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:142`
- **Begriffe:** nichtstabilizerness-qml, expressibilitaet, tensornetz

### `martyniuk2024qas` — Quantum Architecture Search: A Survey
- **arXiv:** 2406.06210 · **Autoren:** Martyniuk et al. · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Survey.
- **QIRE-Bezug:** I-005
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:34`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:130`
- **Begriffe:** architektursuche

### `materialisierung2026` — (Materialisierung, Stabilizer)
- **arXiv:** 2604.15405 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Materialisierung.
- **QIRE-Bezug:** I-036-v3
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v3.md:31`
- **Begriffe:** clifford-quadratform

### `matryoshka_quant2025` — Matryoshka Quantization
- **arXiv:** 2502.06786 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** IDEEN.md: Richardson auf die Logits.
- **QIRE-Bezug:** IDEEN
- **Neuheitsprüfung:** Nur Verweis.
- **Fundstellen:** `INSTITUT/IDEEN.md:42`
- **Begriffe:** quantisierung-ternaer

### `mbl_hea2024` — Hardware-efficient ansatz without barren plateaus in any depth
- **arXiv:** 2403.04844 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Beruht nach Sekundaerauszug auf Vielteilchen-Lokalisierung (MBL).
- **QIRE-Bezug:** Trainierbar vs. simulierbar
- **Neuheitsprüfung:** Gegenbeispiel-Kandidat; Aussage nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:8`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:26`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:96`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:111`
- **Begriffe:** trainierbarkeit

### `mbl_variational2024` — Exploiting MBL for scalable variational simulation
- **arXiv:** 2404.17560 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Verwandt mit 2403.04844.
- **QIRE-Bezug:** Trainierbar vs. simulierbar
- **Neuheitsprüfung:** Nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:26`
- **Begriffe:** trainierbarkeit

### `mbqcnn2024` — Measurement-based quantum CNN
- **arXiv:** 2412.08207 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Deep Learning durch Messung.
- **QIRE-Bezug:** Graphzustaende (09)
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:74`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:75`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:260`
- **Begriffe:** graphzustaende, mbqc

### `mbqml2024` — Measurement-based quantum machine learning
- **arXiv:** 2405.08319 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** MBQC-QML.
- **QIRE-Bezug:** Graphzustaende (09)
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:70`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:73`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:259`
- **Begriffe:** graphzustaende, mbqc

### `mcclean2018bp` — Barren plateaus in quantum neural network training landscapes
- **arXiv:** unvollständig (keine ID) · **Autoren:** McClean et al. · **Jahr:** 2018 · **Venue:** Nat. Commun. 9, 4812
- **Lesestatus:** Auszug · **unvollständig:** arXiv-ID
- **Kernaussage:** Gradientenvarianz tiefer zufaelliger PQCs faellt exponentiell mit der Qubitzahl.
- **QIRE-Bezug:** Trainierbarkeit
- **Neuheitsprüfung:** Standardquelle.
- **Begriffe:** trainierbarkeit, barren-plateaus

### `mckay2016zgates` — Efficient Z-Gates for Quantum Computing
- **arXiv:** 1612.00858 · **Autoren:** McKay et al. · **Jahr:** 2016
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Rz auf IBM/Transmon-Hardware virtuell (Phasenverschiebung, Dauer null).
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Belegt (Auszug).
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_hardware_demo.md:47`, `INSTITUT/recherche/2026-09-29_hardware_demo.md:80`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:44`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:104`
- **Begriffe:** hardware-rz-virtuell

### `measure_first2023` — (Measure-first-Grenze)
- **arXiv:** 2311.12618 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Feste Messstrategie hat in konstruierten Aufgaben exponentielle Nachteile.
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Gilt nicht direkt (Unitaere wird trainiert).
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:44`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:134`
- **Begriffe:** holevo

### `meyer2025lce` — Trainability of Quantum Models Beyond Known Classical Simulability
- **arXiv:** 2507.06344 · **Autoren:** Meyer, Scala, Tacchino, Lucchi · **Jahr:** 2025
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Taylor-Surrogat nahe Clifford-Schaltungen und 'Linear Clifford Encoder' (einige Clifford-Gatter im Ansatz halten Gradienten konstant).
- **QIRE-Bezug:** Magie-Dosis, Clifford + wenige Nicht-Clifford
- **Neuheitsprüfung:** Bruecke Clifford plus wenige Nicht-Clifford.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:29`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:126`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:137`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:143`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:298`
- **Begriffe:** clifford-naehe, trainierbarkeit, dequantisierung

### `mnisq2026` — MNISQ: A Large-Scale Quantum Circuit Dataset
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** 2026 · **Venue:** Nature Scientific Data (doi 10.1038/s41597-026-07493-9)
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, arXiv-ID
- **Kernaussage:** Grosser Quantenschaltungs-Datensatz.
- **QIRE-Bezug:** I-024
- **Neuheitsprüfung:** Ohne Magie-Bezug.
- **Begriffe:** datensaetze-qml

### `moczulski2016acdc` — ACDC: A Structured Efficient Linear Layer
- **arXiv:** 1511.05946 · **Autoren:** Moczulski, M. et al. · **Jahr:** 2016 · **Venue:** ICLR 2016
- **Lesestatus:** Auszug
- **Kernaussage:** Diagonal*DCT*Diagonal*DCT^-1, O(N) Parameter; verschachtelt mit ReLU in CNNs.
- **QIRE-Bezug:** I-038, I-039-v2
- **Neuheitsprüfung:** Hohe Naehe (Ihre genannte Familie).
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md:127`, `INSTITUT/recherche/2026-10-07_volltext_2.md:48`
- **Begriffe:** strukturierte-schichten

### `moe2017` — The Sparsely-Gated Mixture-of-Experts Layer
- **arXiv:** 1701.06538 · **Autoren:** Shazeer et al. · **Jahr:** 2017
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** MoE-Schicht.
- **QIRE-Bezug:** A/sources (mixture_of_experts)
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/mixture_of_experts.md:19`
- **Begriffe:** mixture-of-experts

### `mps_adaptiv2024` — (konstante Tiefe mit adaptiven Schaltungen)
- **arXiv:** 2404.16083 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Titel/Auszug.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:27`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:111`
- **Begriffe:** zustandspraeparation

### `mps_logtiefe2023` — (log-Tiefe-MPS-Praeparation)
- **arXiv:** 2307.01696 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Titel/Auszug.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:27`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:111`
- **Begriffe:** zustandspraeparation

### `natural_quantization2025` — Natural Quantization of NN
- **arXiv:** 2503.15482 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Randtreffer.
- **QIRE-Bezug:** T-ternaer
- **Neuheitsprüfung:** Randtreffer.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:130`
- **Begriffe:** quantisierung-ternaer

### `ncc_phase_2022` — Lernen von Phasenzustaenden
- **arXiv:** 2208.07851 · **Autoren:** unvollständig · **Jahr:** 2022
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Mathematik/Lernen von Phasenzustaenden.
- **QIRE-Bezug:** I-036-v3
- **Neuheitsprüfung:** Quadratische Form dort Darstellung bzw. Lernziel.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v3.md:30`
- **Begriffe:** graphzustaende, clifford-quadratform

### `ncc_ws2023` — Warm Starting Variational Quantum Algorithms with Near Clifford Circuits (NCC-WS)
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** 2023 · **Venue:** Electronics 12(2):347 (MDPI)
- **Lesestatus:** Auszug · **unvollständig:** Autoren, arXiv-ID
- **Kernaussage:** Pretraining mit Nahe-Clifford-Schaltungen zur Initialisierung von PQCs.
- **QIRE-Bezug:** B, I-001
- **Neuheitsprüfung:** Genre 'Clifford als Start'; HTTP 403, nur Snippet gelesen.
- **Begriffe:** clifford-initialisierung

### `needle2` — Needle 2 'Hadamard MLP'
- **arXiv:** unvollständig (keine ID) · **Autoren:** cactus-compute (Projektseiten) · **Jahr:** unvollständig · **Venue:** Projektseiten, keine begutachtete Arbeit
- **Lesestatus:** Auszug · **unvollständig:** Jahr, arXiv-ID
- **Kernaussage:** Form D3 (.) H(act(H(D1 (.) x) (.) D2)), drei trainierte Diagonalen, feste Walsh-Hadamard-Matrizen; Aktivierung SiLU.
- **QIRE-Bezug:** I-045, ff_gelu
- **Neuheitsprüfung:** Naechster Nachbar; Idee 'Hadamard-Mischung, Nichtlinearitaet, Rueckmischung, 3 Diagonalen' nicht neu.
- **Begriffe:** strukturierte-schichten, hadamard-walsh, adapter-peft

### `nn_synthese2025` — (Neuronale Netze fuer naeherungsweise Clifford+T-Synthese)
- **arXiv:** 2504.15990 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** ML-gestuetzte Synthese.
- **QIRE-Bezug:** T-Zaehlung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:78`
- **Begriffe:** gate-synthese

### `no_reference_free2026` — No Reference-Free Generalization in Quantum Machine Learning
- **arXiv:** 2606.22331 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Volltext · **unvollständig:** Autoren, Venue
- **Kernaussage:** Fehlender Referenzrahmen im Hilbertraum bei zu wenig Trainingszustaenden; kein Magie-Bezug.
- **QIRE-Bezug:** I-035
- **Neuheitsprüfung:** Nicht einschlaegig; bestaetigt eher die Luecke.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:104`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:120`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:195`
- **Begriffe:** generalisierung-qml

### `nonstab_optimierung2025` — Role of Nonstabilizerness in Quantum Optimization
- **arXiv:** 2505.17185 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Magie in Quantenoptimierung.
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Magie als Suchkriterium.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:120`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:123`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:296`
- **Begriffe:** nichtstabilizerness-qml

### `nqs_magie2024` — Retrieving non-stabilizerness with Neural Networks
- **arXiv:** 2403.00919 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Magie von Neural Quantum States.
- **QIRE-Bezug:** B, LABORBUCH (Neuheit I-002/I-035)
- **Neuheitsprüfung:** Verwandt, andere Frage (Vielteilchen-Wellenfunktionen).
- **Fundstellen:** `INSTITUT/LABORBUCH.md:788`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:64`
- **Begriffe:** nichtstabilizerness-nqs

### `nqs_nonstab2025` — Non-stabilizerness of Neural Quantum States
- **arXiv:** 2502.09725 · **Autoren:** unvollständig · **Jahr:** 2025 · **Venue:** Phys. Rev. Research (laut Recherche)
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** Monte-Carlo-Verfahren zur Magiemessung in beliebigen variationellen Wellenfunktionen (NQS).
- **QIRE-Bezug:** B, Magie-Konzentration
- **Neuheitsprüfung:** Magie-Konzentrationsverteilungen in Vielteilchensystemen, kein Bild-Klassifikator.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:64`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:18`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:117`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:182`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:39` …
- **Begriffe:** nichtstabilizerness-nqs, magie-konzentration

### `nqs_nonstab_regime2026` — Neural Quantum States in Non-Stabilizer Regimes
- **arXiv:** 2603.28646 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Nicht-Stabilizerness bestimmt Kompression/Darstellungseffizienz von RBMs (laut Snippet).
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Naechstes 'Magie <-> Kompression'-Motiv, aber fuer Vielteilchen-Wellenfunktionen, nicht Netzschichten.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:64`
- **Begriffe:** nichtstabilizerness-nqs, magie-kompression

### `oft2023` — OFT (Orthogonal Finetuning)
- **arXiv:** 2306.07280 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Volltext · **unvollständig:** Autoren, Venue
- **Kernaussage:** ADE20K: r=2/4/8/16 -> mIoU 27,18/27,06/24,09/21,0; Saettigung bei Orthogonalbloecken.
- **QIRE-Bezug:** T-006 (Lagen-Saettigung)
- **Neuheitsprüfung:** Volltext: kein Exponent.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:172`
- **Begriffe:** adapter-peft, unitaer

### `ohana2020opu_kerne` — Kernel computations from large-scale random features obtained by Optical Processing Units \*
- **arXiv:** 1910.09880 · **Autoren:** Ohana u. a. · **Jahr:** 2020 · **Venue:** ICASSP 2020 (Titel aus Wissen)
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Optische RF = |Ux|^2, Bezug zum Polynomkern; Vergleich nur mit RBF-RF und linear.
- **QIRE-Bezug:** I-038, I-039-v2
- **Neuheitsprüfung:** Volltext: keine ReLU-/|.|-Ablation, keine Hadamard-Struktur.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:55`
- **Begriffe:** zufallsmerkmale-opu, quadrat-aktivierung, polynomkern

### `ohno2026lie_scaling` — Scaling Behavior of PQCs, Lie-algebraic
- **arXiv:** 2608.20603 · **Autoren:** Ohno · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** DLA-Dimension und Observablen-Orbit saettigen schnell mit der Parameterzahl.
- **QIRE-Bezug:** T-006 / v5
- **Neuheitsprüfung:** Effektive Dimension statt nominale Parameter als Skalierungsachse.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:36`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:117`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:139`
- **Begriffe:** skalierungsgesetze, lie-algebra

### `opennovelty2026` — OpenNovelty
- **arXiv:** 2601.01576 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Werkzeug zur Neuheitspruefung (laut FORSCHUNGSMETHODIK).
- **QIRE-Bezug:** Neuheitspruefung
- **Neuheitsprüfung:** Methodik.
- **Fundstellen:** `INSTITUT/FORSCHUNGSMETHODIK.md:55`
- **Begriffe:** statistik-methodik, neuheitspruefung

### `optimal_t_sparsity2026` — Optimal T Counts under Sparsity
- **arXiv:** 2607.28260 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** T-Zahl bei duenn besetzten Zustaenden (Luo-Li-Linie).
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Volltextlesen offen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:19`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:93`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:106`
- **Begriffe:** zustandspraeparation, t-zaehlung

### `pan2021wht` — (Schnelle Walsh-Hadamard-Schicht, multiplikationsfrei; Titel nicht genannt)
- **arXiv:** 2104.07085 · **Autoren:** Pan u. a. · **Jahr:** 2021
- **Lesestatus:** Abstract · **unvollständig:** Titel, Venue
- **Kernaussage:** FWHT + Smooth-Thresholding ersetzt 1x1-Faltungen in MobileNet-V2 (2,270 M -> 540 K Parameter, CIFAR-10 95,21 % -> 92,98 %); multiplikationsfreie 2x2-Hadamard-Operatoren.
- **QIRE-Bezug:** I-036-v2, T-ternaer
- **Neuheitsprüfung:** Pflichtzitat; naechster Nachbar fuer 'Hadamard-Schicht als multiplikationsfreier Baustein' (feste +-1-Butterfly, keine Quantenschaltung).
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:126`
- **Begriffe:** hadamard-walsh, multiplikationsfrei, strukturierte-schichten

### `pareto_nas2018` — Pareto-Optimal Device-Aware Neural Architectures
- **arXiv:** 1808.09830 · **Autoren:** unvollständig · **Jahr:** 2018
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Pareto-Optimierung.
- **QIRE-Bezug:** A/sources (pareto_optimization)
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/pareto_optimization.md:18`
- **Begriffe:** pareto, architektursuche

### `perezsalinas2020reupload` — Data re-uploading for a universal quantum classifier \*
- **arXiv:** 1907.02085 · **Autoren:** Perez-Salinas et al. · **Jahr:** 2020 · **Venue:** Quantum 4, 226
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Ein Qubit genuegt als universeller Klassifikator bei mehrfachem Umladen. (Titel aus Wissen)
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:43`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:115`
- **Begriffe:** datenkodierung

### `pitfalls_ml2020` — Pitfalls in Machine Learning Research (laut FORSCHUNGSMETHODIK 'Pitfalls in ML Research')
- **arXiv:** 2011.02832 · **Autoren:** unvollständig · **Jahr:** 2020
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Methodikquelle.
- **QIRE-Bezug:** FORSCHUNGSMETHODIK
- **Neuheitsprüfung:** Methodik.
- **Fundstellen:** `INSTITUT/FORSCHUNGSMETHODIK.md:56`
- **Begriffe:** statistik-methodik

### `polarquant2026` — PolarQuant
- **arXiv:** 2603.29078 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** IDEEN.md.
- **QIRE-Bezug:** IDEEN
- **Neuheitsprüfung:** Nur Verweis.
- **Fundstellen:** `INSTITUT/IDEEN.md:29`
- **Begriffe:** quantisierung-ternaer

### `qattention_survey2025` — (Survey Quanten-Attention)
- **arXiv:** 2504.03192 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Survey.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Nur Titel.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:31`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:119`
- **Begriffe:** quanten-nlp

### `qdt_info_schuss2025` — (Information pro Schuss in Quantum Decision Trees)
- **arXiv:** 2502.11412 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Information pro Schuss als Messgroesse.
- **QIRE-Bezug:** I-040
- **Neuheitsprüfung:** Klassifikator-'Wirkungsgrad' nicht gefunden.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:45`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:138`
- **Begriffe:** information-pro-schuss

### `qelm_bilder2024` — Quanten-ELM fuer Bilder (Titel nicht woertlich genannt)
- **arXiv:** 2409.00998 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Festes Quantenreservoir + klassischer Auslesekopf verbessert Klassifikation; Messwahrscheinlichkeiten als Merkmale.
- **QIRE-Bezug:** Kernaussage
- **Neuheitsprüfung:** Mittlere Naehe.
- **Fundstellen:** `INSTITUT/recherche/2026-10-06_literatur_struktur.md:35`
- **Begriffe:** reservoir-computing, qml-bilder

### `qiu2024compute` — Compute Better Spent: Replacing Dense Layers with Structured Matrices
- **arXiv:** 2406.06248 · **Autoren:** Qiu u. a. · **Jahr:** 2024 · **Venue:** ICML 2024
- **Lesestatus:** Volltext
- **Kernaussage:** Bei gleichem FLOP-Budget: Monarch gleich/besser als dicht, gleicher Exponent; BTT groesster Exponent.
- **QIRE-Bezug:** I-044-v2 (Epochen/Struktur)
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:44`
- **Begriffe:** strukturierte-schichten, skalierung

### `qiu2025linear` — Searching for efficient linear layers over a continuous space of structured matrices
- **arXiv:** 2410.02117 · **Autoren:** Qiu u. a. · **Jahr:** 2025
- **Lesestatus:** Volltext · **unvollständig:** Venue
- **Kernaussage:** Existing structured matrices do not significantly outperform dense in the compute-optimal setting.
- **QIRE-Bezug:** I-044-v2
- **Neuheitsprüfung:** Volltext: erwartet, dass der Abstand Ks-Klin mit mehr Epochen schrumpft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:45`
- **Begriffe:** strukturierte-schichten, skalierung

### `qml_advantages2025` — QML advantages (Titel laut FORSCHUNGSMETHODIK)
- **arXiv:** 2504.15964 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Methodikquelle.
- **QIRE-Bezug:** FORSCHUNGSMETHODIK
- **Neuheitsprüfung:** Methodik.
- **Fundstellen:** `INSTITUT/FORSCHUNGSMETHODIK.md:57`
- **Begriffe:** statistik-methodik, quantenvorteil

### `qnbm` — Introducing Non-Linearity into Quantum Generative Models (Quantum Neuron Born Machine)
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** unvollständig · **Venue:** Papers with Code
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Jahr, arXiv-ID
- **Kernaussage:** 3x kleinerer Fehler bei aehnlicher Parameterzahl, QNBM vs. QCBM.
- **QIRE-Bezug:** Aussage A
- **Neuheitsprüfung:** Vergleich nicht Born vs. linear.
- **Begriffe:** born-auslese, nichtlinearitaet

### `qpa2025` — QPA - Quantum Parameter Adaptation
- **arXiv:** 2410.09846 · **Autoren:** unvollständig · **Jahr:** 2025 · **Venue:** ICLR 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** QNN erzeugt LoRA/DoRA-Gewichte ueber klassischen MLP; GPT-2: 52,06 % der LoRA-Parameter (+0,75 %); Inferenz klassisch.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Nahe: Qubit-Zahl log2 m, aber Parameter ueber MLP.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:28`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:123`
- **Begriffe:** adapter-peft, quanten-peft, quanten-nlp

### `qpsan2026` — QPSAN (Bild)
- **arXiv:** 2605.25365 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Nur Titel.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Nur Titel.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:31`
- **Begriffe:** quanten-nlp

### `qrc_magie2026` — Quantum magic and non-commutativity as computational resources in QRC
- **arXiv:** 2607.12035 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Bedingte T-Gatter regelbar; mehr Abstand vom Clifford-Punkt = bessere Leistung, korreliert mit Magie ('unverzichtbare Ressource').
- **QIRE-Bezug:** I-036, Magie-Dosis
- **Neuheitsprüfung:** Gegenbefund zu QIRE (MNIST: Clifford-Suche = volles Training); denkbare Erklaerung: QRC braucht Magie als Merkmalserzeuger.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:113`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:116`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:161`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:268`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:295`
- **Begriffe:** nichtstabilizerness-qml, reservoir-computing, magie-dosis

### `qtha2025` — QTHA - Quantum-Enhanced LLM Efficient Fine Tuning
- **arXiv:** 2503.12790 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** -76 % trainierbare Parameter gegenueber LoRA; Simulation unklar.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:27`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:122`
- **Begriffe:** adapter-peft, quanten-peft

### `quanta2024` — QuanTA
- **arXiv:** 2406.00132 · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** NeurIPS 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** Quantenschaltkreis-inspirierte Tensoren, hochrangig.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Engste Konkurrenz/Vorarbeit zum QIRE-Adapter.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:30`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:136`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:36`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:126`, `INSTITUT/recherche/2026-10-07_volltext_2.md:170`
- **Begriffe:** adapter-peft, quanten-peft

### `quantum_attention_vit2024` — Quantum Attention for ViT in HEP
- **arXiv:** 2411.13520 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Nur Titel.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Nur Titel.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:31`
- **Begriffe:** quanten-nlp

### `quantum_inspired_label2026` — (Quantum-inspired/QML Begriffsverwendung)
- **arXiv:** 2604.14287 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Nur Snippet gesehen.
- **QIRE-Bezug:** Etikett
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:28`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:204`
- **Begriffe:** quanteninspiriert

### `quantum_inspired_survey2023` — (Survey zu quantum-inspired/Tensor-Netz-ML)
- **arXiv:** 2308.11269 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Survey; Begriff 'quantum-inspired' wird uneinheitlich genutzt.
- **QIRE-Bezug:** Etikett 'quantum-inspired'
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:28`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:204`
- **Begriffe:** quanteninspiriert, tensornetz

### `quantum_llm_ft2025` — Quantum LLM Fine-Tuning
- **arXiv:** 2504.08732 · **Autoren:** Kim, Mei, Girotto, Yamada, Roetteler · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Satz-Transformer + VQC-Kopf, bis +3,14 % Genauigkeit.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Klein.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:26`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:121`
- **Begriffe:** adapter-peft, quanten-nlp

### `quantum_peft2025` — Quantum-PEFT
- **arXiv:** 2503.05431 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Pauli-Parametrisierung, Parameterzahl logarithmisch in der Dimension, Rang voll; Saettigung ab L=3.
- **QIRE-Bezug:** T-006, QIRE-Adapter
- **Neuheitsprüfung:** Engste Konkurrenz/Vorarbeit zum QIRE-Adapter; Born-Quadrat-Alleinstellung widerlegt, Alleinstellung ueber Messung/Kontrollen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:30`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:136`, `INSTITUT/recherche/2026-10-07_volltext_2.md:168`
- **Begriffe:** adapter-peft, quanten-peft, born-auslese

### `quantum_resources2024` — Quantum resources of quantum and classical variational methods
- **arXiv:** 2409.13008 · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** Mach. Learn.: Sci. Technol. (IOP, laut Recherche)
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** Verbindet Nicht-Stabilizerness mit Aufgabenleistung bei Quanten-Reservoir-Computing (QELM).
- **QIRE-Bezug:** I-035, I-002, Magie-Konzentration
- **Neuheitsprüfung:** Einschlaegigste Quelle 'Magie <-> Leistung', aber Reservoir statt ueberwachter Bildklassifikation (dem Institut schon bekannt).
- **Fundstellen:** `INSTITUT/LABORBUCH.md:788`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:13`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:91`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:120`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:181`
- **Begriffe:** nichtstabilizerness-qml, reservoir-computing, magie-konzentration

### `quantum_self_attention2026` — Quantum-Inspired Self-Attention in a LLM
- **arXiv:** 2603.03318 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Nur Titel.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Born als Attention evtl. schon angelegt.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:31`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:87`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:130`
- **Begriffe:** quanten-nlp, born-auslese

### `quantum_shortcut2026` — The Quantum Shortcut: Complex Phase-State Dynamics Reduce the Optimization Steps of Sequence Models
- **arXiv:** 2608.14691 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Komplexe Phasenzustands-Dynamik fuer Sequenzmodelle (Born-Regel-Ausgabe).
- **QIRE-Bezug:** Aussage A
- **Neuheitsprüfung:** Nur Titel/Kontext, Inhalt nicht verifiziert.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:29`
- **Begriffe:** born-auslese, sprachmodell

### `quarot2024` — QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- **arXiv:** 2404.00456 · **Autoren:** Ashkboos, S. et al. · **Jahr:** 2024 · **Venue:** NeurIPS 2024
- **Lesestatus:** nur Titel
- **Kernaussage:** Rotation vor Quantisierung.
- **QIRE-Bezug:** T-ternaer
- **Neuheitsprüfung:** Aus 11_hardware_effizienz_klassisch (mit [?]).
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md:128`
- **Begriffe:** quantisierung-ternaer, hadamard-walsh

### `quic2025` — QuIC
- **arXiv:** 2502.06916 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** < 0,02 % Speicher.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Engste Konkurrenz/Vorarbeit.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:30`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:136`
- **Begriffe:** adapter-peft, quanten-peft

### `quixer2024` — Quixer
- **arXiv:** 2406.04305 · **Autoren:** Khatri, Matos, Coopmans, Clark · **Jahr:** 2024 · **Venue:** Quantinuum
- **Lesestatus:** Auszug
- **Kernaussage:** Token-Superposition per LCU + QSVT; Penn Treebank; verglichen mit LSTM/Transformer/FNet.
- **QIRE-Bezug:** T-Linie, Born als Attention
- **Neuheitsprüfung:** Naechste Naehe zu 'Born-Regel als Attention'.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:22`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:115`
- **Begriffe:** quanten-nlp, sprachmodell, born-auslese

### `ravi2022cafqa` — CAFQA: A classical simulation bootstrap for variational quantum algorithms (Clifford Ansatz For Quantum Accuracy)
- **arXiv:** 2202.12924 · **Autoren:** Ravi, Gokhale et al. (weitere Autoren in der Recherche nicht verifiziert) · **Jahr:** 2022
- **Lesestatus:** Abstract · **unvollständig:** Venue
- **Kernaussage:** Reiner Clifford-Ansatz, Parameter per Bayes-Optimierung im diskreten Clifford-Raum, klassisch simuliert; Startpunkt fuer VQE (ca. 99 % der Korrelationsenergie ueber Hartree-Fock, 2,5x schnellere Konvergenz).
- **QIRE-Bezug:** B (Magie als Kompression), I-001, I-005, I-036, Q-020
- **Neuheitsprüfung:** Naechster Nachbar von QIRE-B, aber Clifford als INITIALISIERUNG vor dem Training (Ziel VQE-Energie); QIRE rundet NACH dem Training und nutzt das Ergebnis als Endmodell/Kompressionsmass. Abgrenzung in der Arbeit Pflicht (Verwechslungsgefahr).
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:48`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:114`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:48`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:188`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:26` …
- **Begriffe:** clifford-initialisierung, clifford-suche-diskret, magie-kompression, vqe

### `recio2024singleshot` — Single-shot quantum machine learning
- **arXiv:** 2406.13812 · **Autoren:** Recio-Armengol, Eisert, Meyer · **Jahr:** 2024
- **Lesestatus:** Volltext · **unvollständig:** Venue
- **Kernaussage:** Definition 'single-shot'; Schranke ueber Unterscheidbarkeit (Holevo-Helstrom); generisch single-shot UND trainierbar geht nicht.
- **QIRE-Bezug:** I-040-v2
- **Neuheitsprüfung:** Volltext: theoretischer Rahmen, nicht dieselbe Messung.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:41`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:137`, `INSTITUT/recherche/2026-10-07_volltext_2.md:94`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:57`
- **Begriffe:** schusszahl, holevo

### `redteam_2111_09079` — (in 10_red_team bei Dequantisierung genannt)
- **arXiv:** 2111.09079 · **Autoren:** unvollständig · **Jahr:** 2021
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 10_red_team 2.6.
- **QIRE-Bezug:** Red Team
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:26`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:202`
- **Begriffe:** dequantisierung

### `redteam_2304_04932` — (in 10_red_team bei Dequantisierung genannt)
- **arXiv:** 2304.04932 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 10_red_team 2.6.
- **QIRE-Bezug:** Red Team
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:26`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:202`
- **Begriffe:** dequantisierung

### `redteam_2508_00230` — (in 10_red_team 2.8 genannt)
- **arXiv:** 2508.00230 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 10_red_team 2.8.
- **QIRE-Bezug:** Red Team
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:30`
- **Begriffe:** adapter-peft

### `redteam_2604_18837` — (in 10_red_team bei Benchmarking genannt)
- **arXiv:** 2604.18837 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 10_red_team 2.2.
- **QIRE-Bezug:** Red Team
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:18`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:205`
- **Begriffe:** benchmark-qml

### `redteam_2608_15617` — (in 10_red_team bei Benchmarking genannt)
- **arXiv:** 2608.15617 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 10_red_team 2.2.
- **QIRE-Bezug:** Red Team
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:18`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:205`
- **Begriffe:** benchmark-qml

### `ref_1905_01426` — (in der Naehe von Stoudenmire&Schwab genannt)
- **arXiv:** 1905.01426 · **Autoren:** unvollständig · **Jahr:** 2019
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 05_parametereffizienz.
- **QIRE-Bezug:** Parametereffizienz
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:25`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:102`
- **Begriffe:** tensornetz

### `ref_2504_10595` — (in der Recherche in der Naehe von Holmes-Matsuura genannt)
- **arXiv:** 2504.10595 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Siehe 03_datenladen.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Nicht zuordenbar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:26`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:111`
- **Begriffe:** zustandspraeparation

### `relu2_sparsitaet2024` — ReLU^2 fuer Sparsitaet (Titel nicht woertlich genannt)
- **arXiv:** 2402.03804 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Quadrat als Aktivierung in dichten FFN.
- **QIRE-Bezug:** I-045-v2
- **Neuheitsprüfung:** Wechselwirkung 'strukturierte Mischung x Quadrat' nicht gefunden.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_literatur_struktur_quadrat.md:6`
- **Begriffe:** quadrat-aktivierung, transformer-ffn

### `resource_kernel2025` — A Resource Efficient Quantum Kernel
- **arXiv:** 2507.03689 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Titel passt zu Graphzustands-Kernel; nur Snippet.
- **QIRE-Bezug:** Graphzustaende (09)
- **Neuheitsprüfung:** Nicht gelesen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:82`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:263`
- **Begriffe:** graphzustaende

### `rff_kritik_quantum2025` — (Kritik/Grenzen von RFF-Dequantisierung)
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** unvollständig · **Venue:** Quantum 2025 (q-2025-02-20-1640)
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Jahr, Titel, arXiv-ID
- **Kernaussage:** Kritik/Grenzen der RFF-Naeherung variationaler QML.
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Hintergrund.
- **Begriffe:** dequantisierung

### `ross_selinger2014` — Optimal ancilla-free Clifford+T approximation of z-rotations (gridsynth) \*
- **arXiv:** 1403.2975 · **Autoren:** Ross, Selinger · **Jahr:** 2014
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Ancillafrei, T-Zahl 3*log2(1/eps) + O(loglog(1/eps)).
- **QIRE-Bezug:** C-hardware (T ~ 31 k), I-001
- **Neuheitsprüfung:** Standardquelle fuer Rz-Synthese.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:69`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:71`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:284`
- **Begriffe:** t-zaehlung, gate-synthese

### `rotations_hadamard2024` — Optimal number of parametrized rotations and Hadamard gates ...
- **arXiv:** 2407.07846 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Parameterreduktion in PQCs (Gatterzahl-Minimierung).
- **QIRE-Bezug:** I-001
- **Neuheitsprüfung:** Anderer Fokus, nicht Magie-Konzentration.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:38`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:186`
- **Begriffe:** pruning-pqc

### `rslora2023` — rsLoRA
- **arXiv:** 2312.03732 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Skalenfaktor alpha/sqrt(r) statt alpha/r stabilisiert hoehere Raenge.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:152`, `INSTITUT/recherche/2026-10-07_volltext_2.md:176`
- **Begriffe:** adapter-peft, skalierungsgesetze

### `rudolph2023lowesa` — (LOWESA: Pauli-Propagations-Surrogat)
- **arXiv:** 2308.09109 · **Autoren:** Rudolph, Fontana, Holmes, Cincio · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Titel, Venue
- **Kernaussage:** Rekonstruiert Erwartungswert-Landschaften (127-Qubit-Ising, Minuten bis Stunden auf Laptop).
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:34`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:116`
- **Begriffe:** dequantisierung, pauli-propagation

### `rydberggpt2026` — Do Quantum Models Scale Like LLMs? (RydbergGPT)
- **arXiv:** 2609.20912 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Transformer auf Rydberg-Messdaten; Potenzgesetz nahe kritischem Punkt.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Transformer auf Quantendaten, nicht Quantenmodell.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:34`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:138`
- **Begriffe:** skalierungsgesetze

### `rydberggpt_methode2024` — RydbergGPT
- **arXiv:** 2405.21052 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Transformer auf Rydberg-Atomen.
- **QIRE-Bezug:** T-006
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:138`
- **Begriffe:** skalierungsgesetze

### `saade2016opu` — Random Projections through Multiple Optical Scattering: Approximating kernels at the speed of light \*
- **arXiv:** 1510.06664 · **Autoren:** Saade u. a. · **Jahr:** 2016 · **Venue:** ICASSP 2016 (Titel aus Wissen)
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Zufallsprojektion durch optische Streuung, Messung |y|^2; MNIST: Kern 1,31 %, N=10.000 Projektionen ca. 2 %, linear 12 %.
- **QIRE-Bezug:** I-038, I-039-v2 (Quadrat)
- **Neuheitsprüfung:** Volltext: |.|^2 als Merkmal bekannt, aber kein Vergleich mit ReLU/|.| und keine strukturierte Matrix.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:55`
- **Begriffe:** zufallsmerkmale-opu, quadrat-aktivierung, born-auslese

### `salek2018qib` — (Quanten-Informationsflaschenhals)
- **arXiv:** 1810.03644 · **Autoren:** Salek u. a. · **Jahr:** 2018
- **Lesestatus:** Auszug · **unvollständig:** Titel, Venue
- **Kernaussage:** Quanten-IB als klassisch-quantenmechanischer Kanal.
- **QIRE-Bezug:** I-040-v2
- **Neuheitsprüfung:** Nur Titel/Zusammenfassung gelesen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:99`
- **Begriffe:** information-bottleneck

### `samplingnoise2024` — Generalization Error with Sampling Noise
- **arXiv:** 2410.14654 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Generalisierung leidet bei wenigen Schuessen.
- **QIRE-Bezug:** I-040
- **Neuheitsprüfung:** Auszug.
- **Begriffe:** schusszahl, generalisierung-qml

### `sbaiz2021singlephoton` — Single-Photon Image Classification
- **arXiv:** 2008.05859 · **Autoren:** Sbaiz, Fischbacher u. a. · **Jahr:** 2021 · **Venue:** ICLR 2021
- **Lesestatus:** Volltext
- **Kernaussage:** I(Klasse; 1 Schuss) auf MNIST gemessen: ohne Interferenz 22,96 % / 1,20 Bit; trainierte Unitaere 41,27 % / 2,04 Bit; ein Ziel (Einzelschuss-Kreuzentropie).
- **QIRE-Bezug:** I-040-v2 H4, Pareto Genauigkeit x Bit/Schuss
- **Neuheitsprüfung:** Volltext: zwingend zitieren. Messgroesse bekannt; Pareto-/Strafterm nicht gefunden.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:91`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:57`
- **Begriffe:** information-pro-schuss, schusszahl, unitaer-training

### `scaling_hybrid_qnn2026` — Scaling Laws for Hybrid QNNs
- **arXiv:** 2604.06007 · **Autoren:** Vyskubov et al. · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Mehr Qubits verbessert meist, mehr Quantenlagen nicht zuverlaessig.
- **QIRE-Bezug:** T-005/T-006
- **Neuheitsprüfung:** Gegenpunkt zu T-005.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:35`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:117`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:139`
- **Begriffe:** skalierungsgesetze, qml

### `schmidt_mass2000` — (Schmidt-Mass; Titel in der Recherche nicht genannt)
- **arXiv:** quant-ph/0007081 · **Autoren:** unvollständig · **Jahr:** 2000
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Schmidt-Mass fuer Verschraenkung.
- **QIRE-Bezug:** V-kompass
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:38`, `INSTITUT/recherche/2026-10-05_breit/09_graphzustaende.md:255`
- **Begriffe:** schmidt-rang

### `schreiber2023surrogat` — Classical surrogates for quantum learning models \*
- **arXiv:** 2206.11740 · **Autoren:** Schreiber, Eisert, Meyer · **Jahr:** 2023 · **Venue:** PRL 2023
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Re-uploading-Modelle sind abgeschnittene Fourier-Reihen mit klassischen Surrogaten. (Titel aus Wissen, in der Recherche nicht genannt)
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:41`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:122`
- **Begriffe:** dequantisierung, fourier-modell

### `schuld2021encoding` — Effect of data encoding on the expressive power of variational quantum machine learning models \*
- **arXiv:** 2008.08605 · **Autoren:** Schuld, Sweke, Meyer · **Jahr:** 2021 · **Venue:** Phys. Rev. A 103, 032430
- **Lesestatus:** Abstract · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Modelle als Fourier-Reihen; Kodierung bestimmt das Spektrum. (Amplitudenkodierung: polynomial in den Eingaben [EIGEN-Einordnung].)
- **QIRE-Bezug:** Clifford-Zelle (quadratische Merkmale)
- **Neuheitsprüfung:** Standardquelle.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:46`, `INSTITUT/recherche/2026-09-29_clifford_klassifikatoren.md:139`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:42`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:115`
- **Begriffe:** dequantisierung, datenkodierung, fourier-modell

### `schuld_killoran2022` — Is quantum advantage the right goal for quantum machine learning? \*
- **arXiv:** 2203.01340 · **Autoren:** Schuld, Killoran · **Jahr:** 2022 · **Venue:** PRX Quantum 3, 030101
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Aufruf, 'Quantenvorteil' als dominantes Ziel zu hinterfragen. (Titel aus Wissen)
- **QIRE-Bezug:** Einordnung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:43`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:123`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:16`, `INSTITUT/recherche/2026-10-05_breit/10_red_team.md:194`
- **Begriffe:** quantenvorteil, benchmark-qml

### `shadows_2412_03381` — (in der Recherche zusammen mit Classical Shadows genannt)
- **arXiv:** 2412.03381 · **Autoren:** unvollständig · **Jahr:** unvollständig
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Jahr, Titel, Venue
- **Kernaussage:** Siehe huang2020shadows.
- **QIRE-Bezug:** Holevo
- **Neuheitsprüfung:** Zuordnung unklar.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:27`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:120`
- **Begriffe:** shadow-tomography

### `shallow_magic_depth2025` — Classical Simulability of Quantum Circuits with Shallow Magic Depth
- **arXiv:** 2409.13809 · **Autoren:** unvollständig · **Jahr:** 2025 · **Venue:** PRX Quantum 6, 010337 (2025)
- **Lesestatus:** Auszug · **unvollständig:** Autoren
- **Kernaussage:** Wo die Magie sitzt (Verteilung ueber Tiefe) bestimmt die Simulierbarkeit.
- **QIRE-Bezug:** Magie-Budget
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:36`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:117`
- **Begriffe:** klassische-simulierbarkeit, magie-tiefe

### `shende2006` — Synthesis of quantum logic circuits \*
- **arXiv:** quant-ph/0406176 · **Autoren:** Shende, Bullock, Markov · **Jahr:** 2006 · **Venue:** IEEE TCAD
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** Zustandspraeparation 'asymptotisch optimal'. (Titel aus Wissen)
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:17`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:103`
- **Begriffe:** zustandspraeparation

### `shin2024tn_dequant` — Dequantizing quantum machine learning models using tensor networks \*
- **arXiv:** 2307.06937 · **Autoren:** Shin, Teo, Jeong · **Jahr:** 2024 · **Venue:** Phys. Rev. Research 6, 023218
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** VQML-Modelle sind Teilklasse von MPS-Modellen mit eingeschraenkten Koeffizienten. (Titel aus Wissen)
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:42`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:122`
- **Begriffe:** dequantisierung, tensornetz

### `shofar2022` — Shot-frugal and robust quantum kernel classifiers
- **arXiv:** 2210.06971 · **Autoren:** unvollständig · **Jahr:** 2022
- **Lesestatus:** Volltext · **unvollständig:** Autoren, Venue
- **Kernaussage:** Schusszahl bei Kernklassifikatoren senken; Fehlerschranken ueber Margenfehler.
- **QIRE-Bezug:** I-040-v2 H2
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:41`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:137`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:58`
- **Begriffe:** schusszahl, marge

### `shot_kodierung2026` — (Shot-basierte Kodierung)
- **arXiv:** 2604.06135 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Alternative Kodierung.
- **QIRE-Bezug:** Hardware
- **Neuheitsprüfung:** Nicht geprueft.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:39`, `INSTITUT/recherche/2026-10-05_breit/08_hardware.md:103`
- **Begriffe:** datenkodierung

### `sim_verrauschte_magie2026` — Klassische Simulation mit verrauschten Magiezustaenden
- **arXiv:** 2601.10111 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Klassische Simulation mit verrauschten Magiezustaenden.
- **QIRE-Bezug:** Simulierbarkeit
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:66`
- **Begriffe:** klassische-simulierbarkeit, clifford-naehe

### `sindhwani2015structured` — Structured Transforms for Small-Footprint Deep Learning \*
- **arXiv:** unvollständig (keine ID) · **Autoren:** Sindhwani et al. · **Jahr:** 2015 · **Venue:** NeurIPS 2015 (Titel aus Wissen)
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert, arXiv-ID
- **Kernaussage:** Toeplitz-artige Matrizen, >3,5x Kompression bei Schluesselworterkennung fast ohne Verlust.
- **QIRE-Bezug:** Strukturierte Schichten
- **Neuheitsprüfung:** Mittlere Naehe (Sprache).
- **Begriffe:** strukturierte-schichten

### `single_photon_cs2026` — Quantum Compressed Sensing Enables Image Classification with a Single Photon
- **arXiv:** 2604.25480 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Gelernte Phasenmaske (SLM) + Hadamard-artige Messung zur Bildklassifikation.
- **QIRE-Bezug:** I-036-v3 (naechster Nachbar)
- **Neuheitsprüfung:** Optisch und kontinuierlich, nicht diskret-Clifford, kein Rang; vor der Arbeit lesen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v3.md:32`, `INSTITUT/recherche/2026-10-07_volltext_2.md:97`
- **Begriffe:** hadamard-walsh, phasenmaske, schusszahl

### `so2021primer` — Primer: Searching for Efficient Transformers for Language Modeling \*
- **arXiv:** 2109.08668 · **Autoren:** So u. a. · **Jahr:** 2021
- **Lesestatus:** Volltext · **unvollständig:** Titel unverifiziert, Venue
- **Kernaussage:** Quadriertes ReLU im Transformer-FFN; groesster Einzelgewinn.
- **QIRE-Bezug:** I-039-v2, I-045-v2, Quadrat-Aktivierung
- **Neuheitsprüfung:** Volltext: kein Vergleich Quadrat gegen |x| und kein reines x^2; stuetzt 'Quadrat hilft gegen ReLU', nicht 'Quadrat schlaegt |x|'.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:56`
- **Begriffe:** quadrat-aktivierung, transformer-ffn

### `spinquant2024` — SpinQuant: LLM Quantization with Learned Rotations
- **arXiv:** 2405.16406 · **Autoren:** Liu, Z. et al. · **Jahr:** 2024
- **Lesestatus:** nur Titel · **unvollständig:** Venue
- **Kernaussage:** Gelernte Rotationen.
- **QIRE-Bezug:** T-ternaer
- **Neuheitsprüfung:** Aus 11_hardware_effizienz_klassisch (mit [?]).
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md:129`
- **Begriffe:** quantisierung-ternaer

### `sre_gnn2025` — Nonstabilizerness Estimation using Graph Neural Networks
- **arXiv:** 2511.23224 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** SRE-Schaetzung per GNN aus Schaltungsgraph.
- **QIRE-Bezug:** I-002, I-035
- **Neuheitsprüfung:** Wie 2509.16799.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:102`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:120`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:194`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:47`
- **Begriffe:** sre, magiebedarf-praediktor

### `sre_ml2025` — A Study on Stabilizer Renyi Entropy Estimation using Machine Learning
- **arXiv:** 2509.16799 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** SRE-Schaetzung per Random Forest/SVR an zufaelligen/strukturierten Schaltungen.
- **QIRE-Bezug:** I-002, I-035
- **Neuheitsprüfung:** Schaetzt SRE; QIRE fragt, ob SRE den Magiebedarf VORHERSAGT.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:97`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:120`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:193`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:47`
- **Begriffe:** sre, magiebedarf-praediktor

### `sre_nullitaet2026` — Stabilizer Statistical Mechanics
- **arXiv:** 2608.14798 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Stabilizer-Nullitaet: ganzzahliges, additives Monoton, untere Schranke fuer die T-Zahl.
- **QIRE-Bezug:** Magie-Messung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:45`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:46`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:305`
- **Begriffe:** sre, magie-mass, t-zaehlung

### `sre_qubitgrenze_2601_07824` — (SRE nur bis ca. 15 Qubits direkt berechenbar)
- **arXiv:** 2601.07824 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Hessematrix waechst mit T-Count, Saettigung bei ca. 10 (ungeklaert).
- **QIRE-Bezug:** Magie-Messung
- **Neuheitsprüfung:** Nicht verifiziert.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:39`, `INSTITUT/recherche/2026-10-05_breit/06_skalierung.md:142`
- **Begriffe:** sre, skalierungsgesetze

### `sre_saettigung2025` — (Saettigung der SRE als Komplexitaetsmass)
- **arXiv:** 2406.04190 · **Autoren:** unvollständig · **Jahr:** 2025 · **Venue:** Quantum 2025 (q-2025-07-21-1801)
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel
- **Kernaussage:** SRE-Saettigung als Komplexitaetsmass.
- **QIRE-Bezug:** Magie-Messung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:51`
- **Begriffe:** sre

### `sre_xor_fwht2025` — (schneller exakter XOR-FWHT-Algorithmus fuer SRE; Titel nicht genannt)
- **arXiv:** 2512.24685 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Schneller exakter Algorithmus zur SRE-Berechnung ueber FWHT.
- **QIRE-Bezug:** Magie-Messung, I-036
- **Neuheitsprüfung:** Walsh-Hadamard-Struktur der Clifford-Zelle ist die von diesen Algorithmen genutzte Struktur [EIGEN].
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/04_magie.md:48`
- **Begriffe:** sre, hadamard-walsh

### `stab_lernen_2607_02444` — (Stabilizer-Lernen)
- **arXiv:** 2607.02444 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Stabilizer-Lernen.
- **QIRE-Bezug:** I-036-v3
- **Neuheitsprüfung:** Quadratische Form als Lernziel.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v3.md:31`
- **Begriffe:** clifford-quadratform

### `stab_lernen_2609_10974` — (Stabilizer-Lernen)
- **arXiv:** 2609.10974 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Stabilizer-Lernen.
- **QIRE-Bezug:** I-036-v3
- **Neuheitsprüfung:** Quadratische Form als Lernziel.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v3.md:31`
- **Begriffe:** clifford-quadratform

### `stab_qram2025` — Stab-QRAM
- **arXiv:** 2509.26494 · **Autoren:** Li et al. · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Clifford-only Orakel fuer affine Boolesche Daten ueber F2, ancillafrei, T-Zahl 0.
- **QIRE-Bezug:** Datenladen, I-005
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:38`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:114`
- **Begriffe:** qram, clifford-orakel

### `stabilizer_bootstrap2024` — The Stabilizer Bootstrap of Quantum Machine Learning with up to 10000 qubits
- **arXiv:** 2412.11356 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Stabilizer-Bootstrap fuer Quantum Neural Networks: Optimierung im Clifford-Raum vor dem Quantenlauf; Simulation bis 10000 Qubits; Verbesserung je nach Observablen und Datenmenge.
- **QIRE-Bezug:** B, I-001
- **Neuheitsprüfung:** Inhaltlich am naechsten an Aussage B (QML + Clifford), aber Bootstrap/Initialisierung, keine Rundung einer trainierten Schicht.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:56`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:114`
- **Begriffe:** clifford-initialisierung, clifford-suche-diskret, qml

### `stabilizer_klassifikator2025` — A hybrid variational quantum circuit approach for stabilizer states classifiers
- **arXiv:** 2511.09430 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Klassifikation von Zustaenden nach Magie-Gehalt (CNN erkennt Nicht-Stabilizer-Zustaende).
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Magie als Klassenmerkmal, nicht Magie in einem trainierten Klassifikator.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:23`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:183`
- **Begriffe:** nichtstabilizerness-qml

### `stoudenmire2016` — Supervised Learning with Quantum-Inspired Tensor Networks \*
- **arXiv:** 1605.05775 · **Autoren:** Stoudenmire, Schwab · **Jahr:** 2016 · **Venue:** NeurIPS 2016
- **Lesestatus:** Auszug · **unvollständig:** Titel unverifiziert
- **Kernaussage:** MPS-Klassifikator auf MNIST: Bonddimension m=10 ca. 5 %, m=20 ca. 2 %, m=120 0,97 % Fehler. (Titel aus Wissen)
- **QIRE-Bezug:** Parametereffizienz (MPS nicht parameterklein)
- **Neuheitsprüfung:** Quantum-inspired Tensornetz-Klassifikator.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:25`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:102`
- **Begriffe:** tensornetz, quanteninspiriert, born-auslese

### `surrogat_skalierbar2025` — (Skalierbarkeit klassischer Surrogate fuer Quantenmodelle)
- **arXiv:** 2508.06131 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Skalierbarkeit der Surrogate.
- **QIRE-Bezug:** Dequantisierung
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:41`
- **Begriffe:** dequantisierung

### `t_count_numerisch2026` — T Count as a Numerically Solvable Minimization Problem
- **arXiv:** 2603.25101 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** T-Zahl-Minimierung; Rundung von Rz-Winkeln auf Clifford/T mit Binaersuche unter Synthesegrenze.
- **QIRE-Bezug:** B
- **Neuheitsprüfung:** Compiler-Perspektive, nicht ML-Leistung.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:63`
- **Begriffe:** t-zaehlung

### `tang2019` — A quantum-inspired classical algorithm for recommendation systems
- **arXiv:** unvollständig (keine ID) · **Autoren:** Tang · **Jahr:** 2019
- **Lesestatus:** Auszug · **unvollständig:** Venue, arXiv-ID
- **Kernaussage:** Dequantisierung: klassischer Algorithmus mit Sampling-Zugang, nur polynomiell langsamer.
- **QIRE-Bezug:** Einordnung 'quantum-inspired'
- **Neuheitsprüfung:** Standardquelle.
- **Begriffe:** dequantisierung, quanteninspiriert

### `tensorgpt2023` — TensorGPT
- **arXiv:** 2307.00526 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Bis 38,4x Embedding-Kompression.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:38`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:129`
- **Begriffe:** tensornetz-kompression, sprachmodell

### `tensorized_embedding2019` — Tensorized Embedding Layers
- **arXiv:** 1901.10787 · **Autoren:** unvollständig · **Jahr:** 2019
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** WikiText-103, 3,8x Kompression, +1,3 Test-Perplexity.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:38`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:129`
- **Begriffe:** tensornetz-kompression, sprachmodell

### `tensorslm2025` — TensorSLM
- **arXiv:** 2506.13514 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Tensor-Train-Sprachmodell.
- **QIRE-Bezug:** T-Linie
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:38`, `INSTITUT/recherche/2026-10-05_breit/07_quanten_sprachmodelle.md:129`
- **Begriffe:** tensornetz-kompression, sprachmodell

### `tn_review_2023` — Tensor Decomposition for Model Reduction in Neural Networks: A Review
- **arXiv:** 2304.13539 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Venue
- **Kernaussage:** Review (literature card tensor_networks).
- **QIRE-Bezug:** A/sources
- **Neuheitsprüfung:** Literaturkarte.
- **Fundstellen:** `A/sources/literature_cards/tensor_networks.md:16`
- **Begriffe:** tensornetz-kompression

### `tsetlin_2025` — (Tsetlin-Maschine, Literale/Klausel mit Pruning)
- **arXiv:** 2508.08350 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Literale/Klausel < 5-13 mit Pruning.
- **QIRE-Bezug:** Bit-Budget-Vergleich
- **Neuheitsprüfung:** Anderes Papier, Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:22`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:34`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:103`
- **Begriffe:** parametereffizienz, tsetlin

### `tsetlin_fuzzy2023` — Fuzzy-Pattern-Tsetlin-Maschine
- **arXiv:** 2301.08190 · **Autoren:** unvollständig · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** 98,56 % auf MNIST mit 20 Klauseln.
- **QIRE-Bezug:** Bit-Budget-Vergleich
- **Neuheitsprüfung:** Klassische Bit-Zaehlweise.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:22`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:103`
- **Begriffe:** parametereffizienz, tsetlin

### `twn2016` — Ternary Weight Networks
- **arXiv:** 1605.04711 · **Autoren:** unvollständig · **Jahr:** 2016
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** TWN 99,35 % auf MNIST.
- **QIRE-Bezug:** Parametereffizienz, T-ternaer
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:21`, `INSTITUT/recherche/2026-10-05_breit/05_parametereffizienz.md:101`
- **Begriffe:** quantisierung-ternaer, parametereffizienz

### `unitary_rnn2015` — Unitary Evolution Recurrent Neural Networks
- **arXiv:** 1511.06464 · **Autoren:** unvollständig · **Jahr:** 2015
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Lineare Auslese (Re/Im konkateniert).
- **QIRE-Bezug:** Aussage A
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:35`
- **Begriffe:** unitaer, born-auslese

### `vqa_eft2024` — Variational Quantum Algorithms in the era of Early Fault Tolerance
- **arXiv:** unvollständig (keine ID) · **Autoren:** unvollständig · **Jahr:** 2024 · **Venue:** ACM/ISCA 2024 (DOI 10.1145/3695053.3731112)
- **Lesestatus:** Auszug · **unvollständig:** Autoren, arXiv-ID
- **Kernaussage:** Partial QEC korrigiert Clifford-Teile exakt, Rz(theta) per Magic-State-Injection.
- **QIRE-Bezug:** I-015, I-034 (Hybrid-Kaskade)
- **Neuheitsprüfung:** Verwandtes Konzept 'Clifford billig, Magie teuer' im Fehlerkorrektur-Kontext; Abgrenzung der Kaskade noetig.
- **Begriffe:** magic-state-kosten, hybrid-kaskade

### `walsh_2307_08384` — (Walsh-Raum-Laden; in 03_datenladen neben Boosari verlinkt, moeglicherweise Zylberman-Debbasch - Zuordnung ungeprueft)
- **arXiv:** 2307.08384 · **Autoren:** unvollständig · **Jahr:** unvollständig
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Jahr, Titel, Venue
- **Kernaussage:** Walsh-Reihen fuer Zustandspraeparation (laut Zusammenhang).
- **QIRE-Bezug:** Datenladen (Walsh-Raum)
- **Neuheitsprüfung:** Walsh-Raum als Laderaum ist nicht neu.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:28`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:112`
- **Begriffe:** zustandspraeparation, hadamard-walsh

### `wang2026lernbar_nicht_simulierbar` — Learnable yet not simulable: a quantum resource theory of learning models
- **arXiv:** 2608.02325 · **Autoren:** Wang, Du, Tao · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Dynamische Stabilizer-Entropie (DSE); Schaltungsfamilien, die ein Surrogat lernt, aber nicht aus der Beschreibung emulierbar sind.
- **QIRE-Bezug:** Trainierbar vs. simulierbar
- **Neuheitsprüfung:** Praezisest neue Form 'lernbar, aber nicht simulierbar'.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:29`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:114`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:107`, `INSTITUT/recherche/2026-10-05_breit/04_magie.md:268`
- **Begriffe:** trainierbarkeit, dequantisierung, sre

### `wellenfunktion_sprache2026` — Deep Sequence Modeling with Quantum Dynamics: Language as a Wave Function
- **arXiv:** 2602.22255 · **Autoren:** unvollständig · **Jahr:** 2026
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** Streng unitaere Dynamik (Cayley), Token-Wahrscheinlichkeit per Born-Regel; Born-Ausgabe haengt von N^2 Merkmalen ab, linear von N.
- **QIRE-Bezug:** Aussage A, T-Linie
- **Neuheitsprüfung:** Sprachmodell mit Wellenfunktion, Ablation Born vs. linear laut Snippet; keine Butterfly-Struktur.
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:25`, `INSTITUT/NEUHEIT_BORN_MAGIE.md:112`
- **Begriffe:** born-auslese, sprachmodell, unitaer

### `west2023approx` — Drastic Circuit Depth Reductions with Preserved Adversarial Robustness by Approximate Encoding for Quantum Machine Learning
- **arXiv:** 2309.09424 · **Autoren:** West et al. · **Jahr:** 2023
- **Lesestatus:** Auszug · **unvollständig:** Venue
- **Kernaussage:** Variationale, genetische, MPS-basierte Verfahren; Schaltungen zwei Groessenordnungen flacher; MNIST/Fashion, IBM-Hardware.
- **QIRE-Bezug:** UEBERGABE (approximatives Laden), Replikation
- **Neuheitsprüfung:** 'Im Kern schon publiziert' - als Replikation kennzeichnen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:23`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:108`
- **Begriffe:** approximatives-laden, amplitudenkodierung, zustandspraeparation

### `wht_cnn2025` — (WHT-Schichten in CNNs)
- **arXiv:** 2506.16418 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** WHT-Schichten in CNNs / Hadamard-Domain-Quantisierung.
- **QIRE-Bezug:** I-036-v2
- **Neuheitsprüfung:** Feste Transformationen, keine gesuchte Vorzeichenmaske mit quadrierter Auslese.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_neuheit_I-036-v2.md:36`
- **Begriffe:** hadamard-walsh, strukturierte-schichten

### `xu2025butterflyquant` — ButterflyQuant
- **arXiv:** 2509.09679 · **Autoren:** Xu u. a. · **Jahr:** 2025
- **Lesestatus:** Volltext · **unvollständig:** Venue
- **Kernaussage:** Lernbare Givens-Butterfly (n/2*log2 n stetige Winkel) ersetzt die feste Hadamard-Rotation als Vorverarbeitung fuer 2-Bit-Gewichtsquantisierung von LLaMA-2; Winkel bleiben stetig; Hadamard als Spezialfall.
- **QIRE-Bezug:** T-Linie (Butterfly), B, T-ternaer
- **Neuheitsprüfung:** Keine Clifford-/Magie-Betrachtung; Pflichtzitat (Volltext gelesen).
- **Fundstellen:** `INSTITUT/NEUHEIT_BORN_MAGIE.md:67`, `INSTITUT/recherche/2026-10-07_volltext_2.md:123`
- **Begriffe:** butterfly, quantisierung, hadamard-walsh

### `yang2015deepfried` — Deep Fried Convnets
- **arXiv:** 1412.7149 · **Autoren:** Yang u. a. · **Jahr:** 2015 · **Venue:** ICCV 2015
- **Lesestatus:** Volltext
- **Kernaussage:** Adaptive Fastfood ersetzt FC-Schichten (ImageNet gleiche Guete mit ca. halb so vielen Parametern); MNIST: Referenz Caffe-LeNet 430.500 Param.
- **QIRE-Bezug:** I-044-v2 (fairer Kopfvergleich), Idee 'Fastfood statt FC-Kopf'
- **Neuheitsprüfung:** Volltext: kein linearer, kein budgetgleicher Gegner; Weight Decay nicht berichtet; Nichtlinearitaet nicht spezifiziert, kein Quadrat. 'Fastfood statt FC-Kopf' ist bekannt: Replikation.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/11_hardware_effizienz_klassisch.md:126`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:4`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:55`
- **Begriffe:** strukturierte-schichten, hadamard-walsh, klassifikationskopf

### `yomo2025` — You Only Measure Once
- **arXiv:** 2509.20090 · **Autoren:** unvollständig · **Jahr:** 2025
- **Lesestatus:** Volltext · **unvollständig:** Autoren, Venue
- **Kernaussage:** Klassen <-> Gruppen von Basiszustaenden, Training auf scharfe Verteilungen fuer 1 Schuss; Schusszahl ~ 1/Marge^2.
- **QIRE-Bezug:** I-040-v2 H2
- **Neuheitsprüfung:** Training auf Schussrauschen ist bekannt; H2 nicht als neu verkaufen.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:41`, `INSTITUT/recherche/2026-10-05_breit/01_holevo.md:137`, `INSTITUT/recherche/2026-10-07_volltext_2.md:95`, `INSTITUT/recherche/2026-10-07_volltext_arxiv.md:58`
- **Begriffe:** schusszahl, marge

### `yu2016sorf` — Orthogonal Random Features (SORF)
- **arXiv:** unvollständig (keine ID) · **Autoren:** Yu et al. · **Jahr:** 2016 · **Venue:** NeurIPS 2016
- **Lesestatus:** Auszug · **unvollständig:** arXiv-ID
- **Kernaussage:** Orthogonale statt Gauss-Matrix senkt Fehler; SORF = binaere Diagonale x Walsh-Hadamard, O(d log d).
- **QIRE-Bezug:** I-038 H-C (Haar vs. Fastfood)
- **Neuheitsprüfung:** Hohe Naehe.
- **Begriffe:** zufallsmerkmale, hadamard-walsh, strukturierte-schichten

### `zhang2022gradient_skalierung` — (Anfangsvarianz ~ 1/L haelt Gradienten polynomial)
- **arXiv:** 2203.09376 · **Autoren:** Zhang u. a. · **Jahr:** 2022 · **Venue:** NeurIPS 2022
- **Lesestatus:** Auszug · **unvollständig:** Titel
- **Kernaussage:** Anfangsvarianz der Winkel ~ 1/L (Streuung ~ 1/sqrt(L)) hält den Gradienten polynomial statt exponentiell klein.
- **QIRE-Bezug:** T-006 (sqrt(L)-Gesetz)
- **Neuheitsprüfung:** Initialisierung, kein Gewinngesetz.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:151`, `INSTITUT/recherche/2026-10-07_volltext_2.md:175`
- **Begriffe:** trainierbarkeit, skalierung, initialisierung

### `zhang2024lernbarkeit_clifford` — (Lernbarkeit ueber Landschaftsfluktuation, per Clifford-Sampling schaetzbar)
- **arXiv:** 2406.11805 · **Autoren:** Zhang, Zhu, Wang · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Titel, Venue
- **Kernaussage:** Lernbarkeit per Clifford-Sampling klassisch schaetzbar.
- **QIRE-Bezug:** Trainierbarkeit
- **Neuheitsprüfung:** Hintergrund.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:22`, `INSTITUT/recherche/2026-10-05_breit/02_trainierbar_simulierbar.md:119`
- **Begriffe:** trainierbarkeit, clifford-naehe

### `zhang2024scaling_peft` — When Scaling Meets LLM Finetuning
- **arXiv:** 2402.17193 · **Autoren:** Zhang u. a. · **Jahr:** 2024
- **Lesestatus:** Volltext · **unvollständig:** Venue
- **Kernaussage:** Multiplikatives Gesetz; PEFT-Exponenten winzig (|alpha| << 1e-2): Skalieren der PEFT-Parameter ineffektiv.
- **QIRE-Bezug:** T-006 (sqrt(L)-Gesetz)
- **Neuheitsprüfung:** Volltext.
- **Fundstellen:** `INSTITUT/recherche/2026-10-07_volltext_2.md:173`
- **Begriffe:** adapter-peft, skalierungsgesetze

### `zhang_yung_yuan` — (Tiefe/Ancilla-Tradeoff bei Zustandspraeparation)
- **arXiv:** 2108.06150 · **Autoren:** Zhang, Yung, Yuan · **Jahr:** 2021
- **Lesestatus:** Auszug · **unvollständig:** Titel, Venue
- **Kernaussage:** Tiefe Omega(max{n, 2^n/(m+n)}) mit m Hilfsqubits.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:18`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:104`
- **Begriffe:** zustandspraeparation

### `zustandspraep_cnot_unter2021` — (Untere CNOT-Schranke fuer Zustandspraeparation)
- **arXiv:** 2111.03132 · **Autoren:** unvollständig · **Jahr:** 2021
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Mindestens floor((2^n-n-1)/2) CNOTs; 10 Qubit exakt = 2047 CNOTs.
- **QIRE-Bezug:** Datenladen, C-hardware
- **Neuheitsprüfung:** Primaerquelle pruefen.
- **Fundstellen:** `INSTITUT/recherche/2026-09-29_hardware_demo.md:41`, `INSTITUT/recherche/2026-09-29_hardware_demo.md:83`
- **Begriffe:** zustandspraeparation

### `zustandspraep_tiefe2022` — (Zustandspraeparation, Tiefe)
- **arXiv:** 2202.11302 · **Autoren:** unvollständig · **Jahr:** 2022
- **Lesestatus:** nur Titel · **unvollständig:** Autoren, Titel, Venue
- **Kernaussage:** Von Recherche zusammen mit 2108.06150 genannt.
- **QIRE-Bezug:** Datenladen
- **Neuheitsprüfung:** Auszug.
- **Fundstellen:** `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:18`, `INSTITUT/recherche/2026-10-05_breit/03_datenladen.md:104`
- **Begriffe:** zustandspraeparation

### `zx_parametrisiert2024` — Optimal compilation of parametrised quantum circuits
- **arXiv:** 2401.12877 · **Autoren:** unvollständig · **Jahr:** 2024
- **Lesestatus:** Auszug · **unvollständig:** Autoren, Venue
- **Kernaussage:** ZX-Vereinfachung fuer parametrisierte Schaltungen; Elimination redundanter freier Parameter.
- **QIRE-Bezug:** I-001
- **Neuheitsprüfung:** Kein nachtraegliches Runden gelernter Winkel auf pi/2.
- **Fundstellen:** `INSTITUT/recherche/2026-09-28_neuheit_magie.md:31`, `INSTITUT/recherche/2026-09-28_neuheit_magie.md:184`
- **Begriffe:** t-zaehlung, zx-kalkuel
