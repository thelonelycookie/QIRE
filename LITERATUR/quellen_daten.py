# -*- coding: utf-8 -*-
"""Kuratierte Literaturdaten fuer QIRE (R5.3). Einzige Quelle der Wahrheit fuer
literatur.bib, notizen.md und neuheit_index.json (erzeugt von erzeuge.py).

REGELN
- Nur Angaben, die in INSTITUT/recherche/*, NEUHEIT_BORN_MAGIE.md, den Handovern
  oder A/sources/literature_cards/ stehen. Was dort fehlt: None (= unvollstaendig).
- status: "Volltext" (Institut hat das PDF gelesen), "Abstract" (Abstract gelesen),
  "Auszug" (nur Suchtreffer/Snippet), "nur Titel".
- qire: QIRE-Bezug (Befund-/Ideen-IDs, Linien). neuheit: Ergebnis der
  Neuheitspruefung zu dieser Arbeit (Rolle als Nachbar/Vorlaeufer/Gegenbefund).
- begriffe: Schlagwoerter (Schluessel in neuheit_index.json).
"""

# Felder: key, arxiv, autoren, jahr, titel, venue, kern, qire, neuheit, status, begriffe
_E = []


def A(key, arxiv, autoren, jahr, titel, venue, kern, qire, neuheit, status, begriffe):
    _E.append(dict(key=key, arxiv=arxiv, autoren=autoren, jahr=jahr, titel=titel, venue=venue,
                   kern=kern, qire=qire, neuheit=neuheit, status=status, begriffe=begriffe))


# ---------------------------------------------------------------- Clifford / Magie (Linie B, I-001, I-036)
A("ravi2022cafqa", "2202.12924", "Ravi, Gokhale et al. (weitere Autoren in der Recherche nicht verifiziert)", 2022,
  "CAFQA: A classical simulation bootstrap for variational quantum algorithms (Clifford Ansatz For Quantum Accuracy)", None,
  "Reiner Clifford-Ansatz, Parameter per Bayes-Optimierung im diskreten Clifford-Raum, klassisch simuliert; Startpunkt fuer VQE (ca. 99 % der Korrelationsenergie ueber Hartree-Fock, 2,5x schnellere Konvergenz).",
  "B (Magie als Kompression), I-001, I-005, I-036, Q-020",
  "Naechster Nachbar von QIRE-B, aber Clifford als INITIALISIERUNG vor dem Training (Ziel VQE-Energie); QIRE rundet NACH dem Training und nutzt das Ergebnis als Endmodell/Kompressionsmass. Abgrenzung in der Arbeit Pflicht (Verwechslungsgefahr).",
  "Abstract", ["clifford-initialisierung", "clifford-suche-diskret", "magie-kompression", "vqe"])
A("lange2023cafqa_folge", "2308.11616", None, None, "(Folgearbeit zu CAFQA, Titel in der Recherche nicht genannt)", None,
  "Laut ideen.json vom Institut notiert; Inhalt in den Recherchedateien nicht beschrieben.", "I-001",
  "Als CAFQA-Folgearbeit gefuehrt; keine eigene Neuheitsbewertung.", "nur Titel", ["clifford-initialisierung"])
A("cheng2022clifford_init", "2207.01539", "Cheng, Khosla, Self, Lin, Li, Medina, Kim", 2022,
  "Clifford Circuit Initialisation for Variational Quantum Algorithms", None,
  "Simulated Annealing ueber Clifford-Parameterpunkte als Vor-Optimierung fuer VQAs; Domaene Quantenchemie; Autoren nennen Skalierungsgrenzen.",
  "I-005, I-036 (methodisch naechste Suche: SA ueber Cliffords), B",
  "Clifford als Start, nicht als Endmodell; keine Magiezaehlung, keine Kompression.", "Volltext",
  ["clifford-initialisierung", "clifford-suche-diskret", "simulated-annealing"])
A("stabilizer_bootstrap2024", "2412.11356", None, 2024, "The Stabilizer Bootstrap of Quantum Machine Learning with up to 10000 qubits", None,
  "Stabilizer-Bootstrap fuer Quantum Neural Networks: Optimierung im Clifford-Raum vor dem Quantenlauf; Simulation bis 10000 Qubits; Verbesserung je nach Observablen und Datenmenge.",
  "B, I-001", "Inhaltlich am naechsten an Aussage B (QML + Clifford), aber Bootstrap/Initialisierung, keine Rundung einer trainierten Schicht.",
  "Auszug", ["clifford-initialisierung", "clifford-suche-diskret", "qml"])
A("kulshrestha2024qadaprune", "2408.13352", "Kulshrestha et al.", 2024, "QAdaPrune: Adaptive Parameter Pruning For Training Variational Quantum Circuits", None,
  "Adaptives Parameter-Pruning (Parameter nahe 0 entfernen) fuer VQCs mit vergleichbarer Leistung.", "I-001 (Clifford-Rundung nach Verlustanstieg)",
  "Pruning, nicht Clifford-Rundung; Auswahlregel Naehe zu 0.", "Auszug", ["pruning-pqc", "magie-kompression"])
A("kissinger2020t_reduktion", None, "Kissinger, van de Wetering", 2020, "Reducing the number of non-Clifford gates in quantum circuits", "Phys. Rev. A 102 (2020)",
  "T-Zahl-Reduktion im ZX-Kalkuel, inkl. Parameter-Elimination fuer variationale Schaltungen; aequivalenzerhaltend.", "B",
  "Compiler-Perspektive; nicht lernend/approximativ.", "Auszug", ["t-zaehlung", "zx-kalkuel"])
A("t_count_numerisch2026", "2603.25101", None, 2026, "T Count as a Numerically Solvable Minimization Problem", None,
  "T-Zahl-Minimierung; Rundung von Rz-Winkeln auf Clifford/T mit Binaersuche unter Synthesegrenze.", "B", "Compiler-Perspektive, nicht ML-Leistung.", "Auszug", ["t-zaehlung"])
A("nqs_magie2024", "2403.00919", None, 2024, "Retrieving non-stabilizerness with Neural Networks", None,
  "Magie von Neural Quantum States.", "B, LABORBUCH (Neuheit I-002/I-035)", "Verwandt, andere Frage (Vielteilchen-Wellenfunktionen).", "Auszug", ["nichtstabilizerness-nqs"])
A("nqs_nonstab2025", "2502.09725", None, 2025, "Non-stabilizerness of Neural Quantum States", "Phys. Rev. Research (laut Recherche)",
  "Monte-Carlo-Verfahren zur Magiemessung in beliebigen variationellen Wellenfunktionen (NQS).", "B, Magie-Konzentration",
  "Magie-Konzentrationsverteilungen in Vielteilchensystemen, kein Bild-Klassifikator.", "Auszug", ["nichtstabilizerness-nqs", "magie-konzentration"])
A("nqs_nonstab_regime2026", "2603.28646", None, 2026, "Neural Quantum States in Non-Stabilizer Regimes", None,
  "Nicht-Stabilizerness bestimmt Kompression/Darstellungseffizienz von RBMs (laut Snippet).", "B",
  "Naechstes 'Magie <-> Kompression'-Motiv, aber fuer Vielteilchen-Wellenfunktionen, nicht Netzschichten.", "Auszug", ["nichtstabilizerness-nqs", "magie-kompression"])
A("maronese2026expressibility", "2506.13605", "Maronese, Ferrari, Vandelli, Dragoni", 2026, "High-expressibility Quantum Neural Networks using only classical resources", "Quantum Mach. Intell. 2026",
  "Clifford-erweiterte MPS naehern die Haar-Verteilung schneller an (Verschraenkung und Magie).", "B, Magie-Dosis",
  "Expressibilitaet statt Kompression; keine Klassifikationsergebnisse.", "Abstract", ["nichtstabilizerness-qml", "expressibilitaet", "tensornetz"])
A("kyriienko2025diskret", "2510.01154", "Kyriienko, Umeano, Holmes", 2025, "Advantage for Discrete Variational Quantum Algorithms in Circuit Recompilation", None,
  "Diskrete Winkel in VQAs, hohe Magie; Kompilierung.", "B", "Kompilierung statt ML-Kompression.", "Auszug", ["diskrete-winkel", "t-zaehlung"])
A("xu2025butterflyquant", "2509.09679", "Xu u. a.", 2025, "ButterflyQuant", None,
  "Lernbare Givens-Butterfly (n/2*log2 n stetige Winkel) ersetzt die feste Hadamard-Rotation als Vorverarbeitung fuer 2-Bit-Gewichtsquantisierung von LLaMA-2; Winkel bleiben stetig; Hadamard als Spezialfall.", "T-Linie (Butterfly), B, T-ternaer",
  "Keine Clifford-/Magie-Betrachtung; Pflichtzitat (Volltext gelesen).", "Volltext", ["butterfly", "quantisierung", "hadamard-walsh"])
A("quantum_resources2024", "2409.13008", None, 2024, "Quantum resources of quantum and classical variational methods", "Mach. Learn.: Sci. Technol. (IOP, laut Recherche)",
  "Verbindet Nicht-Stabilizerness mit Aufgabenleistung bei Quanten-Reservoir-Computing (QELM).", "I-035, I-002, Magie-Konzentration",
  "Einschlaegigste Quelle 'Magie <-> Leistung', aber Reservoir statt ueberwachter Bildklassifikation (dem Institut schon bekannt).", "Auszug", ["nichtstabilizerness-qml", "reservoir-computing", "magie-konzentration"])
A("stabilizer_klassifikator2025", "2511.09430", None, 2025, "A hybrid variational quantum circuit approach for stabilizer states classifiers", None,
  "Klassifikation von Zustaenden nach Magie-Gehalt (CNN erkennt Nicht-Stabilizer-Zustaende).", "B", "Magie als Klassenmerkmal, nicht Magie in einem trainierten Klassifikator.", "Auszug", ["nichtstabilizerness-qml"])
A("zx_parametrisiert2024", "2401.12877", None, 2024, "Optimal compilation of parametrised quantum circuits", None,
  "ZX-Vereinfachung fuer parametrisierte Schaltungen; Elimination redundanter freier Parameter.", "I-001", "Kein nachtraegliches Runden gelernter Winkel auf pi/2.", "Auszug", ["t-zaehlung", "zx-kalkuel"])
A("clifford_perturbation_em2024", "2412.09518", None, 2024, "Clifford Perturbation Approximation for Quantum Error Mitigation", None,
  "Winkel nahe an Clifford gehalten; Zweck Fehlerminderung.", "I-001", "Anderer Zweck (Fehlerminderung).", "Auszug", ["clifford-naehe", "fehlerminderung"])
A("rotations_hadamard2024", "2407.07846", None, 2024, "Optimal number of parametrized rotations and Hadamard gates ...", None,
  "Parameterreduktion in PQCs (Gatterzahl-Minimierung).", "I-001", "Anderer Fokus, nicht Magie-Konzentration.", "Auszug", ["pruning-pqc"])
A("ncc_ws2023", None, None, 2023, "Warm Starting Variational Quantum Algorithms with Near Clifford Circuits (NCC-WS)", "Electronics 12(2):347 (MDPI)",
  "Pretraining mit Nahe-Clifford-Schaltungen zur Initialisierung von PQCs.", "B, I-001", "Genre 'Clifford als Start'; HTTP 403, nur Snippet gelesen.", "Auszug", ["clifford-initialisierung"])
A("vqa_eft2024", None, None, 2024, "Variational Quantum Algorithms in the era of Early Fault Tolerance", "ACM/ISCA 2024 (DOI 10.1145/3695053.3731112)",
  "Partial QEC korrigiert Clifford-Teile exakt, Rz(theta) per Magic-State-Injection.", "I-015, I-034 (Hybrid-Kaskade)",
  "Verwandtes Konzept 'Clifford billig, Magie teuer' im Fehlerkorrektur-Kontext; Abgrenzung der Kaskade noetig.", "Auszug", ["magic-state-kosten", "hybrid-kaskade"])
A("adaptive_pruning_qfim2024", None, None, 2024, "Adaptive pruning algorithm using a quantum Fisher information matrix for PQCs", "Quantum Machine Intelligence 2024 (Springer, doi 10.1007/s42484-024-00221-9)",
  "Entfernt Winkel nahe 0 mittels Quantum-Fisher-Information.", "I-001", "Andere Auswahlregel (nicht Verlustanstieg-Ranking auf Clifford-Gitter).", "Auszug", ["pruning-pqc"])
A("anticoncentration2025", "2502.20455", None, 2025, "Anticoncentration in Clifford Circuits and Beyond: Pseudo-Magic States", None,
  "Magie-Konzentrationsverteilungen in Zufallsschaltungen.", "Magie-Konzentration", "Andere Systemklasse.", "Auszug", ["magie-konzentration"])
A("sre_ml2025", "2509.16799", None, 2025, "A Study on Stabilizer Renyi Entropy Estimation using Machine Learning", None,
  "SRE-Schaetzung per Random Forest/SVR an zufaelligen/strukturierten Schaltungen.", "I-002, I-035", "Schaetzt SRE; QIRE fragt, ob SRE den Magiebedarf VORHERSAGT.", "Auszug", ["sre", "magiebedarf-praediktor"])
A("sre_gnn2025", "2511.23224", None, 2025, "Nonstabilizerness Estimation using Graph Neural Networks", None,
  "SRE-Schaetzung per GNN aus Schaltungsgraph.", "I-002, I-035", "Wie 2509.16799.", "Auszug", ["sre", "magiebedarf-praediktor"])
A("no_reference_free2026", "2606.22331", None, 2026, "No Reference-Free Generalization in Quantum Machine Learning", None,
  "Fehlender Referenzrahmen im Hilbertraum bei zu wenig Trainingszustaenden; kein Magie-Bezug.", "I-035", "Nicht einschlaegig; bestaetigt eher die Luecke.", "Volltext", ["generalisierung-qml"])
A("mnisq2026", None, None, 2026, "MNISQ: A Large-Scale Quantum Circuit Dataset", "Nature Scientific Data (doi 10.1038/s41597-026-07493-9)",
  "Grosser Quantenschaltungs-Datensatz.", "I-024", "Ohne Magie-Bezug.", "nur Titel", ["datensaetze-qml"])
A("lipardi2026magic_qas", "2605.03932", "Lipardi et al.", 2026, "Magic-Informed Quantum Architecture Search", None,
  "MCTS + GNN steuern die Magie (SRE) von Kandidatenschaltungen; getestet auf Grundzustandsenergien/Zustandsapproximation.", "B",
  "Magie als Suchsignal, nicht Kosten/Nutzen je Aufgabe; nicht auf Klassifikation getestet.", "Abstract", ["magie-suchsignal", "architektursuche", "sre"])
A("nonstab_optimierung2025", "2505.17185", None, 2025, "Role of Nonstabilizerness in Quantum Optimization", None,
  "Magie in Quantenoptimierung.", "B", "Magie als Suchkriterium.", "Auszug", ["nichtstabilizerness-qml"])
A("qrc_magie2026", "2607.12035", None, 2026, "Quantum magic and non-commutativity as computational resources in QRC", None,
  "Bedingte T-Gatter regelbar; mehr Abstand vom Clifford-Punkt = bessere Leistung, korreliert mit Magie ('unverzichtbare Ressource').", "I-036, Magie-Dosis",
  "Gegenbefund zu QIRE (MNIST: Clifford-Suche = volles Training); denkbare Erklaerung: QRC braucht Magie als Merkmalserzeuger.", "Auszug", ["nichtstabilizerness-qml", "reservoir-computing", "magie-dosis"])
A("magie_vorteil2024", "2402.12246", None, 2024, "Unconditional quantum magic advantage in shallow circuit computation", "Nat. Commun. 2024",
  "Magie noetig fuer Vorteil bei bestimmten Aufgaben.", "B, Einordnung", "Theorie, andere Frage.", "Auszug", ["nichtstabilizerness-qml", "quantenvorteil"])
A("magie_vorteil_zurueck2025", "2506.19369", None, 2025, "Quantenvorteil bis auf Magie zurueckverfolgt (Gottesman-Knill-Grenze der Ein-Weg-Kommunikation; exakter Titel in der Recherche nicht angegeben)", None,
  "Quantenvorteil auf Magie zurueckgefuehrt.", "B, Einordnung", "Theorie.", "Auszug", ["nichtstabilizerness-qml", "quantenvorteil"])
A("gu2024pseudomagie", "2308.16228", "Gu, Leone, Ghosh, Eisert, Yelin, Quek", 2024, "Pseudomagic quantum states", "Phys. Rev. Lett. 132, 210602 (2024)",
  "Zustaende mit wenig Magie, fuer beschraenkte Beobachter nicht von maximaler Magie unterscheidbar; Magie nicht effizient messbar. (Titel aus Zitat abgeleitet, in der Recherche nicht woertlich genannt)", "I-002, I-035",
  "Betrifft rechenbeschraenkte Unterscheidung, nicht exakt berechnete SRE kleiner Zellen.", "Auszug", ["sre", "pseudomagie"])
A("leone2022sre", "2106.12587", "Leone, Oliviero, Hamma", 2022, "Stabilizer Renyi entropy", "Phys. Rev. Lett. 128, 050402 (2022)",
  "Gut berechenbares Magiemass ohne Minimierung; Monoton der Ressourcentheorie.", "I-002, I-035, Magie-Messung", "Standardquelle fuer SRE.", "Abstract", ["sre", "magie-mass"])
A("sre_nullitaet2026", "2608.14798", None, 2026, "Stabilizer Statistical Mechanics", None,
  "Stabilizer-Nullitaet: ganzzahliges, additives Monoton, untere Schranke fuer die T-Zahl.", "Magie-Messung", "Hintergrund.", "Auszug", ["sre", "magie-mass", "t-zaehlung"])
A("sre_xor_fwht2025", "2512.24685", None, 2025, "(schneller exakter XOR-FWHT-Algorithmus fuer SRE; Titel nicht genannt)", None,
  "Schneller exakter Algorithmus zur SRE-Berechnung ueber FWHT.", "Magie-Messung, I-036",
  "Walsh-Hadamard-Struktur der Clifford-Zelle ist die von diesen Algorithmen genutzte Struktur [EIGEN].", "Auszug", ["sre", "hadamard-walsh"])
A("sre_saettigung2025", "2406.04190", None, 2025, "(Saettigung der SRE als Komplexitaetsmass)", "Quantum 2025 (q-2025-07-21-1801)",
  "SRE-Saettigung als Komplexitaetsmass.", "Magie-Messung", "Hintergrund.", "Auszug", ["sre"])
A("sim_verrauschte_magie2026", "2601.10111", None, 2026, "Klassische Simulation mit verrauschten Magiezustaenden", None,
  "Klassische Simulation mit verrauschten Magiezustaenden.", "Simulierbarkeit", "Hintergrund.", "nur Titel", ["klassische-simulierbarkeit", "clifford-naehe"])
A("bravyi2016gosset", "1601.07601", "Bravyi, Gosset", 2016, "Improved classical simulation of quantum circuits dominated by Clifford gates", "Phys. Rev. Lett. 116, 250501 (2016)",
  "Laufzeit poly + 2^(0,23 t) (Naeherung) bzw. 2^(0,5 t) (exakt) mit t = T-Zahl. (Titel aus Wissen des Katalogisierers, in der Recherche nur Autoren/Zeitschrift)", "C-hardware, Magie-Budget, I-036",
  "Standardquelle fuer 'Kosten wachsen mit der T-Zahl'.", "Auszug", ["klassische-simulierbarkeit", "stabilizer-rang", "t-zaehlung"])
A("bravyi2019lowrank", "1808.00128", "Bravyi et al.", 2019, "Simulation of quantum circuits by low-rank stabilizer decompositions", None,
  "Kosten wachsen exponentiell nur in der Zahl der Nicht-Clifford-Gatter.", "Magie-Budget", "Standardquelle.", "Abstract", ["klassische-simulierbarkeit", "stabilizer-rang"])
A("begusic2023clifford_perturbation", "2306.04797", "Begusic, Chan (laut Recherche)", 2023, "Clifford Perturbation Theory", None,
  "Winkel nahe Clifford, Pauli-Pfade abschneiden.", "I-001", "Genau die Winkelnaehe nach dem Runden.", "Auszug", ["clifford-naehe", "klassische-simulierbarkeit"])
A("shallow_magic_depth2025", "2409.13809", None, 2025, "Classical Simulability of Quantum Circuits with Shallow Magic Depth", "PRX Quantum 6, 010337 (2025)",
  "Wo die Magie sitzt (Verteilung ueber Tiefe) bestimmt die Simulierbarkeit.", "Magie-Budget", "Hintergrund.", "Auszug", ["klassische-simulierbarkeit", "magie-tiefe"])
A("gottesman1998", "quant-ph/9807006", "Gottesman", 1998, "The Heisenberg representation of quantum computers", None,
  "Gottesman-Knill: Stabilizer-Schaltungen klassisch effizient simulierbar.", "Linie Clifford, ehrliche Formulierung", "Standardquelle.", "Abstract", ["gottesman-knill", "klassische-simulierbarkeit"])
A("aaronson2004chp", "quant-ph/0406196", "Aaronson, Gottesman", 2004, "Improved simulation of stabilizer circuits", "Phys. Rev. A 70, 052328 (2004)",
  "CHP-Simulation fuer tausende Qubits; Stabilizer-Simulation ist parity-L-vollstaendig.", "Linie Clifford", "Standardquelle.", "Abstract", ["gottesman-knill", "klassische-simulierbarkeit"])
A("dehaene2003clifford", "quant-ph/0304125", "Dehaene, De Moor", 2003, "Clifford group, stabilizer states, and linear and quadratic operations over GF(2)", "Phys. Rev. A 68, 042318 (2003)",
  "Clifford-Gruppe ~ Gruppe aus linearen/quadratischen Operationen ueber GF(2) (Bruhat-/Quadratform-Darstellung). (Titel aus Wissen, in der Recherche nur Kurzbeschreibung)", "I-036-v3 (quadratische Vorzeichenmaske)",
  "Pflichtzitat fuer die Quadratform-Darstellung der Clifford-Gruppe.", "Auszug", ["clifford-quadratform", "graphzustaende", "gf2"])
A("hein2003schmidt", "quant-ph/0307130", "Hein et al. (laut Recherche)", 2004, "Multiparty entanglement in graph states", None,
  "Schmidt-Rang/-Mass fuer jede Bipartition A|B bei Graphzustaenden. (Titel aus Wissen, unvollstaendig belegt)", "Rang <-> Verschraenkung, V-kompass", "Standardwissen.", "Auszug", ["graphzustaende", "schmidt-rang"])
A("schmidt_mass2000", "quant-ph/0007081", None, 2000, "(Schmidt-Mass; Titel in der Recherche nicht genannt)", None,
  "Schmidt-Mass fuer Verschraenkung.", "V-kompass", "Hintergrund.", "nur Titel", ["schmidt-rang"])
A("gidney2024cultivation", "2409.17595", "Gidney, Shutty, Jones", 2024, "Magic state cultivation: growing T states as cheap as CNOT gates", None,
  "T-Zustaende innerhalb eines Oberflaechencode-Patches; Fehlerraten um 1e-9; Raumzeit-Kosten etwa Faktor 10 unter frueheren Verfahren.", "Hardware-Argument 'T ist teuer'", "Relativiert den T-Preis (sinkt, verschwindet nicht).", "Abstract", ["magic-state-kosten", "t-zaehlung"])
A("gidney2025cultivation_surface", "2502.01743", "Gidney", 2025, "Cultivation on the surface code", None,
  "Folgearbeit zur Oberflaechencode-Integration der Cultivation.", "Hardware-Argument", "Hintergrund.", "Auszug", ["magic-state-kosten"])
A("gidney2025rsa", "2505.15917", "Gidney", 2025, "(RSA-2048 mit < 1 Mio. verrauschter Qubits in < 1 Woche; Titel nicht genannt)", None,
  "RSA-2048 mit weniger als 1 Mio. verrauschter Qubits in unter einer Woche, u. a. durch Cultivation.", "Hardware-Argument", "Hintergrund.", "Auszug", ["magic-state-kosten"])
A("litinski2019", "1905.06903", "Litinski", 2019, "(Magic State Distillation: Not as Costly as You Think / Game of Surface Codes; Titel in der Recherche nicht genannt)", None,
  "Primaerquelle fuer T-Kosten fehlertoleranter Rechner (nicht abgerufen).", "Hardware-Argument", "Nicht gelesen.", "nur Titel", ["magic-state-kosten"])
A("bravyi_kitaev2005", "quant-ph/0403025", "Bravyi, Kitaev", 2005, "(Universal quantum computation with ideal Clifford gates and noisy ancillas)", None,
  "Magic-State-Destillation; Primaerquelle fuer 'T-Gatter verbraucht einen destillierten Magic State' (nicht abgerufen).", "Hardware-Argument", "Nicht gelesen.", "nur Titel", ["magic-state-kosten"])
A("ross_selinger2014", "1403.2975", "Ross, Selinger", 2014, "Optimal ancilla-free Clifford+T approximation of z-rotations (gridsynth)", None,
  "Ancillafrei, T-Zahl 3*log2(1/eps) + O(loglog(1/eps)).", "C-hardware (T ~ 31 k), I-001", "Standardquelle fuer Rz-Synthese.", "Auszug", ["t-zaehlung", "gate-synthese"])
A("kliuchnikov2023synthese", "2203.10064", "Kliuchnikov, Lauter, Minko, Paetznick, Petit", 2023, "(Shorter quantum circuits via single-qubit gate approximation; Titel nicht genannt)", "Quantum 7, 1208 (2023)",
  "Sequenzlaenge Faktor 7/9, gemischte Kanaele sparen ca. Faktor 2.", "T-Zaehlung", "Konstanten nicht bestaetigt.", "Auszug", ["t-zaehlung", "gate-synthese"])
A("nn_synthese2025", "2504.15990", None, 2025, "(Neuronale Netze fuer naeherungsweise Clifford+T-Synthese)", None,
  "ML-gestuetzte Synthese.", "T-Zaehlung", "Hintergrund.", "nur Titel", ["gate-synthese"])
A("almost_ft_qml2025", "2507.18954", None, 2025, "Almost fault-tolerant QML", None,
  "Nur CNOTs korrigieren, parametrisierte Einqubit-Gatter unkorrigiert; von Millionen auf wenige Tausend Qubits.", "Hardware-Argument", "Konkurrierender Weg, das Magie-Problem im QML zu umgehen.", "Auszug", ["hardware-fehlerkorrektur"])
A("meyer2025lce", "2507.06344", "Meyer, Scala, Tacchino, Lucchi", 2025, "Trainability of Quantum Models Beyond Known Classical Simulability", None,
  "Taylor-Surrogat nahe Clifford-Schaltungen und 'Linear Clifford Encoder' (einige Clifford-Gatter im Ansatz halten Gradienten konstant).", "Magie-Dosis, Clifford + wenige Nicht-Clifford",
  "Bruecke Clifford plus wenige Nicht-Clifford.", "Abstract", ["clifford-naehe", "trainierbarkeit", "dequantisierung"])
A("ivaki2025reservoir", "2510.18623", "Ivaki, Karjula, Ala-Nissila", 2025, "Optimal quantum reservoir learning in proximity to universality", "PRA 2025/26 (laut Recherche)",
  "Anteil p der Clifford-Gatter durch nicht-stabilisierende ersetzt; Leistung korreliert mit Verschraenkungsspektrum und langreichweitiger Magie; Optimum nahe Universalitaet.", "Magie-Dosis, I-036",
  "Optimum-Befund, keine Konzentration; Reservoir-Aufgaben, nicht Bildklassifikation.", "Abstract", ["nichtstabilizerness-qml", "reservoir-computing", "magie-dosis"])
A("grewal2025t_dotiert", None, "Grewal, Iyer, Kretschmer, Liang", 2025, "(Lernen von t-dotierten Stabilizerzustaenden; Titel nicht genannt)", "Quantum 9, 1907 (2025); Einzelkopie-Variante Quantum 8, 1250 (2024)",
  "Lernzeit poly(n, 2^t, 1/eps) fuer Zustaende aus Clifford + t Nicht-Clifford-Gattern.", "Magiebedarf M50", "Naechster Verwandter von 'Magie bestimmt Lernkosten', aber nicht dasselbe wie M50.", "Auszug", ["nichtstabilizerness-qml", "lernbarkeit-stabilizer"])
A("wang2026lernbar_nicht_simulierbar", "2608.02325", "Wang, Du, Tao", 2026, "Learnable yet not simulable: a quantum resource theory of learning models", None,
  "Dynamische Stabilizer-Entropie (DSE); Schaltungsfamilien, die ein Surrogat lernt, aber nicht aus der Beschreibung emulierbar sind.", "Trainierbar vs. simulierbar", "Praezisest neue Form 'lernbar, aber nicht simulierbar'.", "Auszug", ["trainierbarkeit", "dequantisierung", "sre"])
A("ghosh_kopplung_ct2025", "2607.17872", None, 2026, "Entanglement geometry separates circuit cutting, classical hardness, and trainability", None,
  "Nur Titel gesehen.", "Rang <-> Verschraenkung (Etappe 2)", "Vor Etappe 2 pruefen.", "nur Titel", ["verschraenkung", "trainierbarkeit"])

# ---------------------------------------------------------------- Dequantisierung / Simulierbarkeit / Benchmarks
A("bowles2024better", "2403.07059", "Bowles, Ahmed, Schuld", 2024, "Better than classical? The subtle art of benchmarking quantum machine learning models", None,
  "12 QML-Modelle x 6 binaere Aufgaben (160 Datensaetze): klassische Out-of-the-box-Modelle gewinnen; Entfernen der Verschraenkung schadet oft nicht.", "FORSCHUNGSMETHODIK, Kernaussage, Clifford-Einordnung",
  "Kontext/Warnung zur Benchmarkwahl.", "Abstract", ["benchmark-qml", "quantenvorteil", "klassische-simulierbarkeit"])
A("bermejo2024qcnn", "2408.12739", "Bermejo et al.", 2025, "QCNNs are (effectively) classically simulable", "PRX Quantum",
  "Klassische Surrogate erreichen oder uebertreffen QCNNs; 'locally easy' Datensaetze.", "Kernaussage, Einordnung", "Warnung zur Benchmarkwahl.", "Abstract", ["klassische-simulierbarkeit", "benchmark-qml", "dequantisierung"])
A("cerezo2025bp_sim", "2312.09121", "Cerezo et al.", 2025, "Does provable absence of barren plateaus imply classical simulability?", "Nat. Commun. 2025",
  "Trainierbarkeit und klassische Simulierbarkeit haengen oft zusammen (Fall fuer Fall, kein allgemeiner Satz).", "Kernaussage, Einordnung", "Hintergrund.", "Abstract", ["klassische-simulierbarkeit", "trainierbarkeit", "dequantisierung"])
A("delorenzis2025qelm", "2509.06873", "De Lorenzis et al.", 2025, "Entanglement and Classical Simulability in Quantum Extreme Learning Machines", None,
  "MNIST, Fashion-MNIST, CIFAR-10; moderate Verschraenkung, klassisch simulierbar, nuetzliche Merkmale.", "I-036, Clifford-Zelle als QELM-Analogie", "Literaturanker (Vergleich auf denselben Datensaetzen).", "Abstract", ["reservoir-computing", "klassische-simulierbarkeit", "qml-bilder"])
A("qelm_bilder2024", "2409.00998", None, 2024, "Quanten-ELM fuer Bilder (Titel nicht woertlich genannt)", None,
  "Festes Quantenreservoir + klassischer Auslesekopf verbessert Klassifikation; Messwahrscheinlichkeiten als Merkmale.", "Kernaussage", "Mittlere Naehe.", "Auszug", ["reservoir-computing", "qml-bilder"])
A("schuld2021encoding", "2008.08605", "Schuld, Sweke, Meyer", 2021, "Effect of data encoding on the expressive power of variational quantum machine learning models", "Phys. Rev. A 103, 032430",
  "Modelle als Fourier-Reihen; Kodierung bestimmt das Spektrum. (Amplitudenkodierung: polynomial in den Eingaben [EIGEN-Einordnung].)", "Clifford-Zelle (quadratische Merkmale)", "Standardquelle.", "Abstract", ["dequantisierung", "datenkodierung", "fourier-modell"])
A("landman2023rff", "2210.13200", "Landman et al.", 2023, "Classically approximating variational quantum machine learning with random Fourier features", "ICLR 2023 (laut Recherche)",
  "VQCs mit Hamilton-Kodierung klassisch per Random Fourier Features approximierbar.", "I-038 (Wirkung klassisch)", "Stuetzt 'Wirkung klassisch'.", "Auszug", ["dequantisierung", "zufallsmerkmale"])
A("schreiber2023surrogat", "2206.11740", "Schreiber, Eisert, Meyer", 2023, "Classical surrogates for quantum learning models", "PRL 2023",
  "Re-uploading-Modelle sind abgeschnittene Fourier-Reihen mit klassischen Surrogaten. (Titel aus Wissen, in der Recherche nicht genannt)", "Dequantisierung", "Hintergrund.", "Auszug", ["dequantisierung", "fourier-modell"])
A("surrogat_skalierbar2025", "2508.06131", None, 2025, "(Skalierbarkeit klassischer Surrogate fuer Quantenmodelle)", None,
  "Skalierbarkeit der Surrogate.", "Dequantisierung", "Hintergrund.", "nur Titel", ["dequantisierung"])
A("shin2024tn_dequant", "2307.06937", "Shin, Teo, Jeong", 2024, "Dequantizing quantum machine learning models using tensor networks", "Phys. Rev. Research 6, 023218",
  "VQML-Modelle sind Teilklasse von MPS-Modellen mit eingeschraenkten Koeffizienten. (Titel aus Wissen)", "Dequantisierung", "Hintergrund.", "Auszug", ["dequantisierung", "tensornetz"])
A("gilfuster2025", "2406.07072", "Gil-Fuster, Gyurik, Perez-Salinas, Dunjko", 2025, "(Trainierbarkeit und Dequantisierung; Titel nicht genannt)", "ICLR 2025",
  "Formale Definition von Trainierbarkeit und Dequantisierung; ob erstes letzteres impliziert.", "Dequantisierung", "Ergebnis nicht gelesen.", "Auszug", ["dequantisierung", "trainierbarkeit"])
A("goh2025gsim", "2308.01432", "Goh, Larocca, Cincio, Cerezo, Sauvage", 2025, "(g-sim: Simulation ueber die Lie-Algebra)", "Phys. Rev. Research 7, 033266",
  "Effizient, wenn DLA-Dimension polynomiell; Magiezustaende als Anfangszustand moeglich.", "Lie-Algebra/T-006", "Hintergrund.", "Auszug", ["lie-algebra", "klassische-simulierbarkeit"])
A("rudolph2023lowesa", "2308.09109", "Rudolph, Fontana, Holmes, Cincio", 2023, "(LOWESA: Pauli-Propagations-Surrogat)", None,
  "Rekonstruiert Erwartungswert-Landschaften (127-Qubit-Ising, Minuten bis Stunden auf Laptop).", "Dequantisierung", "Hintergrund.", "Auszug", ["dequantisierung", "pauli-propagation"])
A("angrisani2025pauli", "2409.01706", "Angrisani et al.", 2025, "(Pauli-Propagation, klassische Schaetzung)", "PRL 2025",
  "Klassisch schaetzbar fuer die meisten zufaelligen unstrukturierten Schaltungen mit einzelqubit-invarianter Verteilung.", "Dequantisierung", "Hintergrund.", "Auszug", ["dequantisierung", "pauli-propagation"])
A("heyraud2023", None, "Heyraud et al.", 2023, "(Gradientenvarianz zufaellig initialisierter Schaltungen ueber Clifford-Abbildung)", "PRX Quantum 4, 040335 (doi 10.1103/PRXQuantum.4.040335)",
  "Gradientenvarianz exakt ueber Abbildung auf Clifford-Schaltungen berechenbar.", "Magie-Dosis/Trainierbarkeit", "Hintergrund.", "Auszug", ["trainierbarkeit", "clifford-naehe"])
A("zhang2024lernbarkeit_clifford", "2406.11805", "Zhang, Zhu, Wang", 2024, "(Lernbarkeit ueber Landschaftsfluktuation, per Clifford-Sampling schaetzbar)", None,
  "Lernbarkeit per Clifford-Sampling klassisch schaetzbar.", "Trainierbarkeit", "Hintergrund.", "Auszug", ["trainierbarkeit", "clifford-naehe"])
A("mbl_hea2024", "2403.04844", None, 2024, "Hardware-efficient ansatz without barren plateaus in any depth", None,
  "Beruht nach Sekundaerauszug auf Vielteilchen-Lokalisierung (MBL).", "Trainierbar vs. simulierbar", "Gegenbeispiel-Kandidat; Aussage nicht geprueft.", "Auszug", ["trainierbarkeit"])
A("mbl_variational2024", "2404.17560", None, 2024, "Exploiting MBL for scalable variational simulation", None,
  "Verwandt mit 2403.04844.", "Trainierbar vs. simulierbar", "Nicht geprueft.", "nur Titel", ["trainierbarkeit"])
A("mcclean2018bp", None, "McClean et al.", 2018, "Barren plateaus in quantum neural network training landscapes", "Nat. Commun. 9, 4812",
  "Gradientenvarianz tiefer zufaelliger PQCs faellt exponentiell mit der Qubitzahl.", "Trainierbarkeit", "Standardquelle.", "Auszug", ["trainierbarkeit", "barren-plateaus"])
A("schuld_killoran2022", "2203.01340", "Schuld, Killoran", 2022, "Is quantum advantage the right goal for quantum machine learning?", "PRX Quantum 3, 030101",
  "Aufruf, 'Quantenvorteil' als dominantes Ziel zu hinterfragen. (Titel aus Wissen)", "Einordnung", "Hintergrund.", "Auszug", ["quantenvorteil", "benchmark-qml"])
A("larocca2025uebersicht", "2405.00781", "Larocca et al.", 2025, "(Uebersicht ueber Barren Plateaus)", "Nature Reviews Physics (s42254-025-00813-9)",
  "Uebersichtsartikel.", "Trainierbarkeit", "Hintergrund.", "nur Titel", ["trainierbarkeit", "barren-plateaus"])
A("larocca2021lie", "2109.11676", "Larocca u. a.", 2021, "(Saettigung beim Rang der dynamischen Lie-Algebra)", None,
  "Kritische Parameterzahl M_c <= dim(g); QFIM-/Hessian-Rang saettigen bei dim(g)-1; Berechnungs-Phasenuebergang der Trainierbarkeit.", "T-006 / Lie-Skalierung (v5)", "Volltext-Zeile: Paper.", "Volltext", ["lie-algebra", "overparametrisierung"])
A("caro2022generalisierung", "2111.05292", "Caro et al.", 2022, "Generalization in quantum machine learning from few training data", "Nat. Commun. 2022",
  "Generalisierungsfehler <~ sqrt(T/N), T trainierbare Gatter, N Trainingsdaten. (Titel aus Wissen)", "Skalierung (Kopf dominiert die Schranke)", "Hintergrund.", "Auszug", ["generalisierung-qml", "skalierung"])
A("zhang2022gradient_skalierung", "2203.09376", "Zhang u. a.", 2022, "(Anfangsvarianz ~ 1/L haelt Gradienten polynomial)", "NeurIPS 2022",
  "Anfangsvarianz der Winkel ~ 1/L (Streuung ~ 1/sqrt(L)) hält den Gradienten polynomial statt exponentiell klein.", "T-006 (sqrt(L)-Gesetz)", "Initialisierung, kein Gewinngesetz.", "Auszug", ["trainierbarkeit", "skalierung", "initialisierung"])
A("holmes_ent_cl_rff", None, None, None, "(Kritik/Grenzen von RFF-Dequantisierung)", "Quantum 2025 (q-2025-02-20-1640)",
  "Kritik/Grenzen der RFF-Naeherung variationaler QML.", "Dequantisierung", "Hintergrund.", "nur Titel", ["dequantisierung"])
A("tang2019", None, "Tang", 2019, "A quantum-inspired classical algorithm for recommendation systems", None,
  "Dequantisierung: klassischer Algorithmus mit Sampling-Zugang, nur polynomiell langsamer.", "Einordnung 'quantum-inspired'", "Standardquelle.", "Auszug", ["dequantisierung", "quanteninspiriert"])
A("chia2022sampling", "1910.06151", "Chia, Gilyen, Li, Lin, Tang, Wang", 2022, "Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning", "J. ACM 69(5) (2022)",
  "Ganze Klasse von QML-Beschleunigungen faellt unter Sampling-Zugriff klassisch nur polynomiell langsamer.", "Einordnung", "Standardquelle.", "Auszug", ["dequantisierung"])
A("cotler2021revisiting", "2112.00811", "Cotler, Huang, McClean", 2021, "Revisiting dequantization and quantum advantage in learning tasks", None,
  "SQ-Zugang moeglicherweise zu maechtiger Rohstoff; Vergleich evtl. unfair. (Titel aus Wissen)", "Einordnung", "Gegenposition zu Tang.", "Auszug", ["dequantisierung"])
A("quantum_inspired_survey2023", "2308.11269", None, 2023, "(Survey zu quantum-inspired/Tensor-Netz-ML)", None,
  "Survey; Begriff 'quantum-inspired' wird uneinheitlich genutzt.", "Etikett 'quantum-inspired'", "Hintergrund.", "nur Titel", ["quanteninspiriert", "tensornetz"])
A("quantum_inspired_label2026", "2604.14287", None, 2026, "(Quantum-inspired/QML Begriffsverwendung)", None,
  "Nur Snippet gesehen.", "Etikett", "Hintergrund.", "nur Titel", ["quanteninspiriert"])
A("rfk_kritik", None, None, None, "(Chia/Gilyen u. a., Gharibian/Le Gall STOC 2022: QSVT-Dequantisierung)", "STOC 2022",
  "Dequantisierungsrahmen.", "Einordnung", "Nur im Text erwaehnt.", "nur Titel", ["dequantisierung"])
A("pitfalls_ml2020", "2011.02832", None, 2020, "Pitfalls in Machine Learning Research (laut FORSCHUNGSMETHODIK 'Pitfalls in ML Research')", None,
  "Methodikquelle.", "FORSCHUNGSMETHODIK", "Methodik.", "nur Titel", ["statistik-methodik"])
A("qml_advantages2025", "2504.15964", None, 2025, "QML advantages (Titel laut FORSCHUNGSMETHODIK)", None,
  "Methodikquelle.", "FORSCHUNGSMETHODIK", "Methodik.", "nur Titel", ["statistik-methodik", "quantenvorteil"])
A("opennovelty2026", "2601.01576", None, 2026, "OpenNovelty", None,
  "Werkzeug zur Neuheitspruefung (laut FORSCHUNGSMETHODIK).", "Neuheitspruefung", "Methodik.", "nur Titel", ["statistik-methodik", "neuheitspruefung"])

# ---------------------------------------------------------------- Born-Auslese, Quadrat, strukturierte Schichten
A("cvnn_kopf2026", "2602.15283", None, 2026, "Complex-Valued Unitary Representations as Classification Heads for Improved Uncertainty Quantification in Deep Neural Networks", None,
  "Komplexer Hilbertraum, Cayley-Unitaere; Auslese per Magnitude-Softmax oder Born-Regel; Magnitude-Softmax besser kalibriert (2,4x geringerer ECE auf CIFAR-10).", "Aussage A (Born vs. linear), Q-020 P4",
  "Gegenbefund: Born NICHT besser, bei Kalibrierung schlechter; in der Diskussion von Q-020 P4 aufgreifen.", "Auszug", ["born-auslese", "klassifikationskopf", "unitaer"])
A("wellenfunktion_sprache2026", "2602.22255", None, 2026, "Deep Sequence Modeling with Quantum Dynamics: Language as a Wave Function", None,
  "Streng unitaere Dynamik (Cayley), Token-Wahrscheinlichkeit per Born-Regel; Born-Ausgabe haengt von N^2 Merkmalen ab, linear von N.", "Aussage A, T-Linie",
  "Sprachmodell mit Wellenfunktion, Ablation Born vs. linear laut Snippet; keine Butterfly-Struktur.", "Auszug", ["born-auslese", "sprachmodell", "unitaer"])
A("quantum_shortcut2026", "2608.14691", None, 2026, "The Quantum Shortcut: Complex Phase-State Dynamics Reduce the Optimization Steps of Sequence Models", None,
  "Komplexe Phasenzustands-Dynamik fuer Sequenzmodelle (Born-Regel-Ausgabe).", "Aussage A", "Nur Titel/Kontext, Inhalt nicht verifiziert.", "nur Titel", ["born-auslese", "sprachmodell"])
A("qnbm", None, None, None, "Introducing Non-Linearity into Quantum Generative Models (Quantum Neuron Born Machine)", "Papers with Code",
  "3x kleinerer Fehler bei aehnlicher Parameterzahl, QNBM vs. QCBM.", "Aussage A", "Vergleich nicht Born vs. linear.", "Auszug", ["born-auslese", "nichtlinearitaet"])
A("unitary_rnn2015", "1511.06464", None, 2015, "Unitary Evolution Recurrent Neural Networks", None,
  "Lineare Auslese (Re/Im konkateniert).", "Aussage A", "Hintergrund.", "Auszug", ["unitaer", "born-auslese"])
A("cayley_adapter2026", "2605.05914", None, 2026, "Cayley Unitary Adapters", None,
  "Cayley-Adapter in gefrorenen LLM-Projektionen, Zwei-Qubit-Bloecke.", "Aussage A, T-Linie (Adapter)", "Nur Abstract bestaetigt; Details nicht geprueft.", "Abstract", ["adapter-peft", "unitaer", "born-auslese"])
A("le2013fastfood", "1408.3060", "Le, Sarlos, Smola", 2013, "Fastfood - Approximating Kernel Expansions in Loglinear Time", "ICML 2013",
  "Hadamard*Diagonal ersetzt Gauss-Matrix; O(n log d) Zeit, O(n) Speicher; Polynomkerne nur ueber Legendre-Entwicklung (Grad 10, UCI-Regression); CIFAR-10 mit RBF.", "I-038 (Fastfood-Mischung), I-041, I-044-v2, T-ternaer",
  "Volltext gelesen: Polynomkern Grad 2 mit Hadamard auf MNIST NICHT gemessen. Strukturierte Mischung ist bekannt, nicht als eigener Befund verkaufen.", "Volltext", ["strukturierte-schichten", "hadamard-walsh", "zufallsmerkmale", "polynomkern"])
A("yang2015deepfried", "1412.7149", "Yang u. a.", 2015, "Deep Fried Convnets", "ICCV 2015",
  "Adaptive Fastfood ersetzt FC-Schichten (ImageNet gleiche Guete mit ca. halb so vielen Parametern); MNIST: Referenz Caffe-LeNet 430.500 Param.", "I-044-v2 (fairer Kopfvergleich), Idee 'Fastfood statt FC-Kopf'",
  "Volltext: kein linearer, kein budgetgleicher Gegner; Weight Decay nicht berichtet; Nichtlinearitaet nicht spezifiziert, kein Quadrat. 'Fastfood statt FC-Kopf' ist bekannt: Replikation.", "Volltext", ["strukturierte-schichten", "hadamard-walsh", "klassifikationskopf"])
A("saade2016opu", "1510.06664", "Saade u. a.", 2016, "Random Projections through Multiple Optical Scattering: Approximating kernels at the speed of light", "ICASSP 2016 (Titel aus Wissen)",
  "Zufallsprojektion durch optische Streuung, Messung |y|^2; MNIST: Kern 1,31 %, N=10.000 Projektionen ca. 2 %, linear 12 %.", "I-038, I-039-v2 (Quadrat)",
  "Volltext: |.|^2 als Merkmal bekannt, aber kein Vergleich mit ReLU/|.| und keine strukturierte Matrix.", "Volltext", ["zufallsmerkmale-opu", "quadrat-aktivierung", "born-auslese"])
A("ohana2020opu_kerne", "1910.09880", "Ohana u. a.", 2020, "Kernel computations from large-scale random features obtained by Optical Processing Units", "ICASSP 2020 (Titel aus Wissen)",
  "Optische RF = |Ux|^2, Bezug zum Polynomkern; Vergleich nur mit RBF-RF und linear.", "I-038, I-039-v2", "Volltext: keine ReLU-/|.|-Ablation, keine Hadamard-Struktur.", "Volltext", ["zufallsmerkmale-opu", "quadrat-aktivierung", "polynomkern"])
A("so2021primer", "2109.08668", "So u. a.", 2021, "Primer: Searching for Efficient Transformers for Language Modeling", None,
  "Quadriertes ReLU im Transformer-FFN; groesster Einzelgewinn.", "I-039-v2, I-045-v2, Quadrat-Aktivierung",
  "Volltext: kein Vergleich Quadrat gegen |x| und kein reines x^2; stuetzt 'Quadrat hilft gegen ReLU', nicht 'Quadrat schlaegt |x|'.", "Volltext", ["quadrat-aktivierung", "transformer-ffn"])
A("dao2022monarch", "2204.00595", "Dao u. a.", 2022, "Monarch: Expressive Structured Matrices for Efficient and Accurate Training", "ICML 2022",
  "Produkt zweier Blockdiagonalen; 1,7-2,0x schnelleres Training bei ViT/GPT-2 ohne Qualitaetsverlust; keine Schwellenbreite angegeben.", "I-041 (Zeitgewinn), T-Linie",
  "Volltext: Schwellenbreite nicht zitierbar.", "Volltext", ["strukturierte-schichten", "butterfly", "monarch"])
A("chen2022pixelfly", "2112.00029", "Chen, Dao u. a.", 2022, "Pixelated Butterfly: Simple and Efficient Sparse Training for Neural Network Models", "ICLR 2022",
  "Fussnote 4: Butterfly-Multiplikation erst ab n ca. 1.024 schneller als dicht; Pixelfly bis 2,3x schneller.", "I-041 (FWHT schneller ab N=1.024)", "Volltext: externe Bestaetigung der Schwelle n ca. 1.024.", "Volltext", ["strukturierte-schichten", "butterfly"])
A("dao2019butterfly", "1903.05895", "Dao u. a.", 2019, "Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations", "ICML 2019",
  "Butterfly-Faktorisierung lernt schnelle Transformationen.", "Butterfly-Linie", "Pflichtzitat fuer Butterfly-Faktorisierung.", "Auszug", ["butterfly", "strukturierte-schichten"])
A("moczulski2016acdc", "1511.05946", "Moczulski, M. et al.", 2016, "ACDC: A Structured Efficient Linear Layer", "ICLR 2016",
  "Diagonal*DCT*Diagonal*DCT^-1, O(N) Parameter; verschachtelt mit ReLU in CNNs.", "I-038, I-039-v2", "Hohe Naehe (Ihre genannte Familie).", "Auszug", ["strukturierte-schichten"])
A("sindhwani2015structured", None, "Sindhwani et al.", 2015, "Structured Transforms for Small-Footprint Deep Learning", "NeurIPS 2015 (Titel aus Wissen)",
  "Toeplitz-artige Matrizen, >3,5x Kompression bei Schluesselworterkennung fast ohne Verlust.", "Strukturierte Schichten", "Mittlere Naehe (Sprache).", "Auszug", ["strukturierte-schichten"])
A("yu2016sorf", None, "Yu et al.", 2016, "Orthogonal Random Features (SORF)", "NeurIPS 2016",
  "Orthogonale statt Gauss-Matrix senkt Fehler; SORF = binaere Diagonale x Walsh-Hadamard, O(d log d).", "I-038 H-C (Haar vs. Fastfood)", "Hohe Naehe.", "Auszug", ["zufallsmerkmale", "hadamard-walsh", "strukturierte-schichten"])
A("kaleidoscope2020", None, "Dao et al.", 2020, "Kaleidoscope matrices", "ICLR 2020",
  "K-Matrizen fassen jede strukturierte Matrix; ShuffleNet +5 % ImageNet.", "Strukturierte Schichten", "Hohe Naehe.", "Auszug", ["strukturierte-schichten", "butterfly"])
A("qiu2024compute", "2406.06248", "Qiu u. a.", 2024, "Compute Better Spent: Replacing Dense Layers with Structured Matrices", "ICML 2024",
  "Bei gleichem FLOP-Budget: Monarch gleich/besser als dicht, gleicher Exponent; BTT groesster Exponent.", "I-044-v2 (Epochen/Struktur)", "Volltext.", "Volltext", ["strukturierte-schichten", "skalierung"])
A("qiu2025linear", "2410.02117", "Qiu u. a.", 2025, "Searching for efficient linear layers over a continuous space of structured matrices", None,
  "Existing structured matrices do not significantly outperform dense in the compute-optimal setting.", "I-044-v2", "Volltext: erwartet, dass der Abstand Ks-Klin mit mehr Epochen schrumpft.", "Volltext", ["strukturierte-schichten", "skalierung"])
A("gonon2025kronecker_sparse", None, "Gonon et al.", 2025, "Fast Inference with Kronecker-Sparse Matrices", "ICML 2025",
  "Butterfly/Monarch-Muster: Median 1,4x schneller (FP32), nicht immer schneller als dicht.", "I-041", "Mittlere Naehe ('ab welcher Breite').", "Auszug", ["strukturierte-schichten"])
A("dao_monarch_mixer2023", "2310.12109", None, 2023, "Monarch Mixer", None,
  "Gleicht GPT-artige Transformer bei 360 M Parametern; BERT-Guete mit bis zu 27 % weniger Parametern.", "T-Linie", "Hintergrund.", "Auszug", ["strukturierte-schichten", "monarch"])
A("hadamard_klassisch_hadanet2019", "1905.10759", None, 2019, "HadaNets", None, "Hadamard-PRODUKT (elementweise), nicht Hadamard-Transformation.", "T-ternaer", "Nur Namensgleichheit; hoechstens Randzitat.", "Abstract", ["hadamard-produkt-verwechslung"])
A("hadamard_nn_2018", "1805.05421", None, 2018, "Energy Efficient Hadamard Neural Networks", None, "Hadamard-Produkt Gewicht = FP x binaer.", "T-ternaer", "Nicht verwechseln.", "Abstract", ["hadamard-produkt-verwechslung"])
A("pan2021wht", "2104.07085", "Pan u. a.", 2021, "(Schnelle Walsh-Hadamard-Schicht, multiplikationsfrei; Titel nicht genannt)", None,
  "FWHT + Smooth-Thresholding ersetzt 1x1-Faltungen in MobileNet-V2 (2,270 M -> 540 K Parameter, CIFAR-10 95,21 % -> 92,98 %); multiplikationsfreie 2x2-Hadamard-Operatoren.", "I-036-v2, T-ternaer",
  "Pflichtzitat; naechster Nachbar fuer 'Hadamard-Schicht als multiplikationsfreier Baustein' (feste +-1-Butterfly, keine Quantenschaltung).", "Abstract", ["hadamard-walsh", "multiplikationsfrei", "strukturierte-schichten"])
A("wht_cnn2025", "2506.16418", None, 2025, "(WHT-Schichten in CNNs)", None,
  "WHT-Schichten in CNNs / Hadamard-Domain-Quantisierung.", "I-036-v2", "Feste Transformationen, keine gesuchte Vorzeichenmaske mit quadrierter Auslese.", "Auszug", ["hadamard-walsh", "strukturierte-schichten"])
A("bitnet158", "2402.17764", "Ma u. a.", 2024, "BitNet b1.58", None,
  "Ternaere Gewichte {-1,0,+1}*gamma.", "T-ternaer, IDEEN", "Pflichtzitat; Vergleich Clifford-Zelle gegen BitNet-ternaer.", "Abstract", ["quantisierung-ternaer", "multiplikationsfrei"])
A("relu2_sparsitaet2024", "2402.03804", None, 2024, "ReLU^2 fuer Sparsitaet (Titel nicht woertlich genannt)", None,
  "Quadrat als Aktivierung in dichten FFN.", "I-045-v2", "Wechselwirkung 'strukturierte Mischung x Quadrat' nicht gefunden.", "Auszug", ["quadrat-aktivierung", "transformer-ffn"])
A("hadamard_adapter2024", "2407.11033", None, 2024, "Hadamard-Adapter", None,
  "Elementweise Skalierung.", "I-045-v2", "'Multiplikativ' meint Gewichtsprodukte, nicht Quadrat der Aktivierung.", "Auszug", ["adapter-peft", "hadamard-produkt-verwechslung"])
A("clifford_equivariant2023", "2305.11141", "Ruhe u. a.", 2023, "Clifford Group Equivariant Neural Networks", None,
  "Clifford-ALGEBREN/-Gruppe (Multivektoren, O(n)/E(n)-Aequivarianz).", "Namensverwechslung", "Nichts mit Clifford-Quantengattern; im Text ausdruecklich abgrenzen.", "Abstract", ["namensverwechslung"])
A("ghorbani2021kernel", "2006.13409", "Ghorbani u. a.", 2021, "When do neural networks outperform kernel methods?", None,
  "Fashion-MNIST: NN, RF, NT, RKHS vergleichbar; mit Rauschen verschlechtern sich RF/NT/RKHS schneller.", "Feste Zufallsmerkmale vs. trainierte Schicht", "Volltext.", "Volltext", ["zufallsmerkmale"])
A("ba2022onestep", "2205.01445", "Ba u. a.", 2022, "High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation", None,
  "Ein Gradientenschritt verbessert Merkmale gegenueber festen Zufallsmerkmalen. (Titel aus Wissen, Recherche: 'One Gradient Step')", "Zufallsmerkmale vs. Training", "Volltext.", "Volltext", ["zufallsmerkmale"])
A("born_text_2022", None, None, 2022, "Text Classification with Born's Rule", "NeurIPS 2022",
  "Text als Ueberlagerung von Woertern; Uebergangswahrscheinlichkeit nach Born-Regel; einbettbar in NN.", "Aussage A (Name 'Born')", "Gleicher Name, andere Konstruktion.", "Auszug", ["born-auslese"])
A("stoudenmire2016", "1605.05775", "Stoudenmire, Schwab", 2016, "Supervised Learning with Quantum-Inspired Tensor Networks", "NeurIPS 2016",
  "MPS-Klassifikator auf MNIST: Bonddimension m=10 ca. 5 %, m=20 ca. 2 %, m=120 0,97 % Fehler. (Titel aus Wissen)", "Parametereffizienz (MPS nicht parameterklein)", "Quantum-inspired Tensornetz-Klassifikator.", "Auszug", ["tensornetz", "quanteninspiriert", "born-auslese"])
A("livni2014", None, "Livni et al.", 2014, "On the Computational Efficiency of Training Neural Networks", "NeurIPS 2014",
  "Netze mit x^2 als Aktivierung (Polynomnetze) so ausdrucksstark wie Schwellennetze, effizient lernbar.", "Quadrat-Aktivierung", "Mittlere Naehe.", "Auszug", ["quadrat-aktivierung"])
A("karnick", None, "Kar, Karnick; Wacker et al.", None, "Random features for dot product kernels / Improved Random Features for Dot Product Kernels; TensorSketch/TensorSRHT", "JMLR (Wacker et al.)",
  "Zufallsmerkmale fuer Polynomkerne, strukturierte Sketches.", "Quadrat = Grad-2-Polynomkern", "Hohe Naehe.", "Auszug", ["polynomkern", "zufallsmerkmale"])
A("needle2", None, "cactus-compute (Projektseiten)", None, "Needle 2 'Hadamard MLP'", "Projektseiten, keine begutachtete Arbeit",
  "Form D3 (.) H(act(H(D1 (.) x) (.) D2)), drei trainierte Diagonalen, feste Walsh-Hadamard-Matrizen; Aktivierung SiLU.", "I-045, ff_gelu",
  "Naechster Nachbar; Idee 'Hadamard-Mischung, Nichtlinearitaet, Rueckmischung, 3 Diagonalen' nicht neu.", "Auszug", ["strukturierte-schichten", "hadamard-walsh", "adapter-peft"])

# ---------------------------------------------------------------- Schuesse / Information / Holevo
A("sbaiz2021singlephoton", "2008.05859", "Sbaiz, Fischbacher u. a.", 2021, "Single-Photon Image Classification", "ICLR 2021",
  "I(Klasse; 1 Schuss) auf MNIST gemessen: ohne Interferenz 22,96 % / 1,20 Bit; trainierte Unitaere 41,27 % / 2,04 Bit; ein Ziel (Einzelschuss-Kreuzentropie).", "I-040-v2 H4, Pareto Genauigkeit x Bit/Schuss",
  "Volltext: zwingend zitieren. Messgroesse bekannt; Pareto-/Strafterm nicht gefunden.", "Volltext", ["information-pro-schuss", "schusszahl", "unitaer-training"])
A("hu2023eigentasks", "2307.16083", "Hu, Angelatos, Khan u. a.", 2023, "Tackling Sampling Noise in Physical Systems for Machine Learning Applications: Fundamental Limits and Eigentasks", "Phys. Rev. X 13 (2023)",
  "Fuer 2-Designs gilt C_T = K(S+1)/(S+K): maximale Expressibilitaet => Kapazitaet verschwindet bei festem S. Schaltung wird nicht trainiert. (Titel aus Wissen)", "I-040-v2 H4 (Haar schadet bei wenigen Schuessen)",
  "Volltext: Theorie zu H4; Mass ist Kapazitaet, nicht I(Klasse;1 Schuss). Zwingend zitieren.", "Volltext", ["information-pro-schuss", "schusszahl", "reservoir-computing"])
A("recio2024singleshot", "2406.13812", "Recio-Armengol, Eisert, Meyer", 2024, "Single-shot quantum machine learning", None,
  "Definition 'single-shot'; Schranke ueber Unterscheidbarkeit (Holevo-Helstrom); generisch single-shot UND trainierbar geht nicht.", "I-040-v2", "Volltext: theoretischer Rahmen, nicht dieselbe Messung.", "Volltext", ["schusszahl", "holevo"])
A("yomo2025", "2509.20090", None, 2025, "You Only Measure Once", None,
  "Klassen <-> Gruppen von Basiszustaenden, Training auf scharfe Verteilungen fuer 1 Schuss; Schusszahl ~ 1/Marge^2.", "I-040-v2 H2", "Training auf Schussrauschen ist bekannt; H2 nicht als neu verkaufen.", "Volltext", ["schusszahl", "marge"])
A("shofar2022", "2210.06971", None, 2022, "Shot-frugal and robust quantum kernel classifiers", None,
  "Schusszahl bei Kernklassifikatoren senken; Fehlerschranken ueber Margenfehler.", "I-040-v2 H2", "Volltext.", "Volltext", ["schusszahl", "marge"])
A("firstphoton2024", "2410.17471", None, 2024, "First Photon ML (Experiment zu Sbaiz)", None,
  "Experiment: 31,0 % gegen 24,57 % klassisch.", "I-040-v2", "Volltext.", "Volltext", ["information-pro-schuss", "schusszahl"])
A("banchi2021qib", "2102.08991", "Banchi, Pereira, Pirandola", 2021, "Generalization in quantum machine learning: a quantum information standpoint", "PRX Quantum 2021 (Titel aus Wissen)",
  "Variationaler Quanten-Informationsflaschenhals L_IB = I(X:Z) - beta I(C:Z); Schuesse nur als Sonderfall.", "I-040-v2 (beta/lambda-Reihe)", "Volltext: methodisches Vorbild, andere Achse.", "Volltext", ["information-bottleneck", "schusszahl"])
A("salek2018qib", "1810.03644", "Salek u. a.", 2018, "(Quanten-Informationsflaschenhals)", None,
  "Quanten-IB als klassisch-quantenmechanischer Kanal.", "I-040-v2", "Nur Titel/Zusammenfassung gelesen.", "Auszug", ["information-bottleneck"])
A("ma2023photonen", "2307.15712", "Ma, Wang, Laydevant, Wright, McMahon", 2023, "(Stochastisches Training fuer wenige Photonen)", None,
  "MNIST 98 % bei 0,038 Photonen je MAC; Training modelliert Photodetektions-Statistik.", "I-040-v2", "'Fuer das Rauschmodell trainieren' ist bekannt.", "Volltext", ["schusszahl"])
A("ilias2026einzelschuss", "2512.20492", "Ilias, Hu, Vives, Tuereci", 2026, "(Training bei fester Schusszahl)", None,
  "Schaltung bei festem S auf das Aufgabenrisiko trainiert; fuer S->inf optimierte Parameter bleiben bei kleinem Budget nicht optimal; Bayessche Metrologie.", "I-040-v2", "Volltext.", "Volltext", ["schusszahl"])
A("fewshots2024", "2411.06600", None, 2024, "Few measurement shots challenge generalization", None,
  "Wenige Schuesse = dominante Fehlerquelle.", "I-040", "Auszug.", "Auszug", ["schusszahl", "generalisierung-qml"])
A("samplingnoise2024", "2410.14654", None, 2024, "Generalization Error with Sampling Noise", None,
  "Generalisierung leidet bei wenigen Schuessen.", "I-040", "Auszug.", "Auszug", ["schusszahl", "generalisierung-qml"])
A("qdt_info_schuss2025", "2502.11412", None, 2025, "(Information pro Schuss in Quantum Decision Trees)", None,
  "Information pro Schuss als Messgroesse.", "I-040", "Klassifikator-'Wirkungsgrad' nicht gefunden.", "Auszug", ["information-pro-schuss"])
A("amplitude_limits2025", "2503.01545", None, 2025, "Limitations of Amplitude Encoding on Quantum Classification", None,
  "Vollstaendiges Zuruecklesen der Eingabe widerspraeche Holevo.", "Holevo/Kodierung", "Auszug.", "Auszug", ["holevo", "amplitudenkodierung"])
A("holevo_n_bit2026", "2606.26312", None, 2026, "(Holevo-Grenze und Amplitudenkodierung)", None,
  "n Qubit liefern pro Kopie hoechstens n klassische Bit.", "Holevo", "Auszug.", "Auszug", ["holevo", "amplitudenkodierung"])
A("fano_2019", "1901.00555", None, 2019, "(Fano-Ungleichung; Quelle in der Recherche nur als Link)", None, "Fano: I(C;C^) >= log2 K - h2(Pe) - Pe log2(K-1).", "Holevo/Fano", "Standardwissen.", "nur Titel", ["holevo", "information-pro-schuss"])
A("audenaert2007chernoff", "quant-ph/0610027", "Audenaert et al.", 2007, "Discriminating states: the quantum Chernoff bound", "PRL 98, 160501",
  "Fehler zweier Klassenzustaende faellt wie exp(-M xi) mit der Quanten-Chernoff-Information. (Titel aus Wissen)", "Holevo/Schusszahl", "Schuss-Grenze M ~ ln(1/delta)/xi.", "Auszug", ["holevo", "schusszahl"])
A("measure_first2023", "2311.12618", None, 2023, "(Measure-first-Grenze)", None, "Feste Messstrategie hat in konstruierten Aufgaben exponentielle Nachteile.", "Holevo", "Gilt nicht direkt (Unitaere wird trainiert).", "Auszug", ["holevo"])
A("brassard2000amplitude", "quant-ph/0005055", "Brassard, Hoyer, Mosca, Tapp", 2000, "Quantum Amplitude Amplification and Estimation (Titel aus Wissen)", None,
  "Fehler eps mit O(1/eps) Orakelaufrufen statt O(1/eps^2) Schuessen.", "Holevo/Schusszahl", "Standardquelle.", "Auszug", ["amplitudenschaetzung", "schusszahl"])
A("buhrman2001fingerprint", "quant-ph/0102001", "Buhrman, Cleve, Watrous, de Wolf", 2001, "Quantum Fingerprinting", "PRL 87, 167902",
  "Gleichheitstest mit O(log n) Qubits. (Titel aus Wissen)", "Holevo", "Standardquelle.", "Auszug", ["holevo"])
A("aaronson2018shadow", "1711.01053", "Aaronson", 2018, "Shadow Tomography of Quantum States", None,
  "Õ(eps^-4 log^4 M log D) Kopien fuer M Messungen. (Titel aus Wissen)", "Holevo", "Auszug.", "Auszug", ["shadow-tomography"])
A("huang2020shadows", "2106.10190", "Huang, Kueng, Preskill", 2020, "Predicting many properties of a quantum system from very few measurements (Classical Shadows)", "Nature Phys. 16, 1050 (2020)",
  "M Eigenschaften mit O(log M) Zufallsmessungen. (arXiv-ID-Zuordnung unsicher: Recherche nennt 2412.03381 und 2106.10190)", "Holevo", "Auszug.", "Auszug", ["shadow-tomography"])
A("shadows_2412_03381", "2412.03381", None, None, "(in der Recherche zusammen mit Classical Shadows genannt)", None, "Siehe huang2020shadows.", "Holevo", "Zuordnung unklar.", "nur Titel", ["shadow-tomography"])
A("chen2022speicher", "2111.05881", "Chen, Cotler, Huang, Li", 2022, "Exponential separations between learning with and without quantum memory", "FOCS 2022 (Titel aus Wissen)",
  "Ohne >= n Qubit Speicher Omega(2^((n-k)/3)) Kopien; mit Speicher O(n).", "Holevo", "Auszug.", "Auszug", ["quantenspeicher"])
A("bennett_wiesner_superdense", "0708.3062", "Bennett, Wiesner (1992)", 1992, "Superdense coding", None, "2 Bit je gesendetem Qubit mit Verschraenkung.", "Holevo", "arXiv-ID 0708.3062 als Fundstelle, Zuordnung nicht geprueft.", "Auszug", ["holevo"])

# ---------------------------------------------------------------- Daten laden / Zustandspraeparation
A("west2023approx", "2309.09424", "West et al.", 2023, "Drastic Circuit Depth Reductions with Preserved Adversarial Robustness by Approximate Encoding for Quantum Machine Learning", None,
  "Variationale, genetische, MPS-basierte Verfahren; Schaltungen zwei Groessenordnungen flacher; MNIST/Fashion, IBM-Hardware.", "UEBERGABE (approximatives Laden), Replikation",
  "'Im Kern schon publiziert' - als Replikation kennzeichnen.", "Auszug", ["approximatives-laden", "amplitudenkodierung", "zustandspraeparation"])
A("araujo2021approx", None, "Araujo", 2021, "(approximative Zustandspraeparation)", None, "Approximative Zustandspraeparation.", "UEBERGABE", "Als bekannt eingestuft.", "nur Titel", ["approximatives-laden"])
A("perezsalinas2020reupload", "1907.02085", "Perez-Salinas et al.", 2020, "Data re-uploading for a universal quantum classifier", "Quantum 4, 226",
  "Ein Qubit genuegt als universeller Klassifikator bei mehrfachem Umladen. (Titel aus Wissen)", "Datenladen", "Auszug.", "Auszug", ["datenkodierung"])
A("zhang_yung_yuan", "2108.06150", "Zhang, Yung, Yuan", 2021, "(Tiefe/Ancilla-Tradeoff bei Zustandspraeparation)", None,
  "Tiefe Omega(max{n, 2^n/(m+n)}) mit m Hilfsqubits.", "Datenladen", "Auszug.", "Auszug", ["zustandspraeparation"])
A("zustandspraep_tiefe2022", "2202.11302", None, 2022, "(Zustandspraeparation, Tiefe)", None, "Von Recherche zusammen mit 2108.06150 genannt.", "Datenladen", "Auszug.", "nur Titel", ["zustandspraeparation"])
A("shende2006", "quant-ph/0406176", "Shende, Bullock, Markov", 2006, "Synthesis of quantum logic circuits", "IEEE TCAD",
  "Zustandspraeparation 'asymptotisch optimal'. (Titel aus Wissen)", "Datenladen", "Auszug.", "Auszug", ["zustandspraeparation"])
A("zustandspraep_cnot_unter2021", "2111.03132", None, 2021, "(Untere CNOT-Schranke fuer Zustandspraeparation)", None,
  "Mindestens floor((2^n-n-1)/2) CNOTs; 10 Qubit exakt = 2047 CNOTs.", "Datenladen, C-hardware", "Primaerquelle pruefen.", "Auszug", ["zustandspraeparation"])
A("gosset_kothari", "2411.04790", "Gosset, Kothari", 2024, "Quantum State Preparation with Optimal T-Count", None,
  "eps-Naeherung beliebiger n-Qubit-Zustaende mit O(sqrt(2^n log(1/eps))) T-Gattern, untere Schranke gleich.", "Datenladen (T-Zahl des Ladens)", "Auszug; Volltextlesen offen.", "Auszug", ["zustandspraeparation", "t-zaehlung"])
A("optimal_t_sparsity2026", "2607.28260", None, 2026, "Optimal T Counts under Sparsity", None, "T-Zahl bei duenn besetzten Zustaenden (Luo-Li-Linie).", "Datenladen", "Volltextlesen offen.", "nur Titel", ["zustandspraeparation", "t-zaehlung"])
A("luo_li_sparse", "2608.00414", "Luo, Li", None, "Sparse Quantum State Preparation (Luo-Li)", None, "T-Zahl ~ sqrt(s) fuer s-duenne Zustaende (ID-Zuordnung unsicher).", "Datenladen", "Zuordnung ungeprueft.", "nur Titel", ["zustandspraeparation"])
A("holmes_matsuura", "2303.01562", "Holmes, Matsuura", 2023, "(Glatte Funktionen als MPS)", None, "Eine Lage Zwei-Qubit-Gatter, bis 20 Qubit auf Ionenfalle.", "Datenladen", "Auszug.", "Auszug", ["zustandspraeparation", "tensornetz"])
A("fourier_mps2024", "2311.07666", None, 2024, "(Fourier-Moden-MPS)", "Quantum 2024", "Bilddaten mit schnell abfallendem Fourier-Spektrum haben kleinen Schmidt-Rang.", "Datenladen", "Autoren nicht geprueft.", "Auszug", ["zustandspraeparation", "tensornetz"])
A("dilip2022mps", "2204.11170", "Dilip, Liu, Smith, Pollmann", 2022, "(MPS-Korrespondenz fuer Zustandspraeparation)", "PRR 4, 043007", "11 Qubit mit Tensornetz-Niveau.", "Datenladen", "Auszug.", "Auszug", ["zustandspraeparation", "tensornetz"])
A("mps_logtiefe2023", "2307.01696", None, 2023, "(log-Tiefe-MPS-Praeparation)", None, "Titel/Auszug.", "Datenladen", "Auszug.", "nur Titel", ["zustandspraeparation"])
A("mps_adaptiv2024", "2404.16083", None, 2024, "(konstante Tiefe mit adaptiven Schaltungen)", None, "Titel/Auszug.", "Datenladen", "Auszug.", "nur Titel", ["zustandspraeparation"])
A("boosari_walsh2026", "2609.28760", "Boosari", 2026, "Walsh-Transform Realization of Dense-to-Sparse Quantum State Preparation", None,
  "Daten in den Walsh-Raum, nur groesste Koeffizienten laden, danach eine Hadamard-Lage ohne CNOT zurueck.", "Datenladen (Walsh-Raum), I-005/I-036",
  "Walsh-Raum als Laderaum ist NICHT neu; nur die Kopplung mit Magie-Budget/Clifford-Orbit-Suche bleibt.", "Auszug", ["zustandspraeparation", "hadamard-walsh", "approximatives-laden"])
A("stab_qram2025", "2509.26494", "Li et al.", 2025, "Stab-QRAM", None, "Clifford-only Orakel fuer affine Boolesche Daten ueber F2, ancillafrei, T-Zahl 0.", "Datenladen, I-005", "Auszug.", "Auszug", ["qram", "clifford-orakel"])
A("cmps2024", "2412.17209", None, 2024, "Clifford-augmentierte MPS", None, "Cliffords zur Verschraenkungsreduktion.", "I-036-v2, Datenladen", "Autoren nicht geprueft.", "Auszug", ["tensornetz", "clifford-simulation"])
A("jaques_rattew_qram", "2305.10310", "Jaques, Rattew", 2025, "QRAM: A Survey and Critique", "Quantum 9, 1922 (2025)", "Vorteile schrumpfen, wenn Steuerhardware und klassische Parallelitaet mitgerechnet werden.", "Datenladen", "Auszug.", "Auszug", ["qram"])
A("amplitude_vs_winkel2024", "2410.09121", None, 2024, "(Vergleich Amplituden- vs. Winkelkodierung auf MNIST)", None, "Amplitudenkodierung oft besser bei weniger Qubits (0,891 vs 0,780 binaer), aber tiefere Schaltungen.", "Datenladen", "Auszug.", "Auszug", ["datenkodierung", "amplitudenkodierung"])
A("frqi_neqr_2025", "2507.22039", None, 2025, "(FRQI/NEQR-Vergleich fuer Klassifikation)", None, "FRQI (Le et al.), NEQR (Zhang et al. 2013).", "Datenladen", "Auszug.", "Auszug", ["datenkodierung"])
A("amplitude_approx_flach2021", "2103.13211", None, 2021, "(Approximative Amplitudenkodierung in flachen parametrisierten Schaltungen)", None, "Alternative zu exakter Amplitudenkodierung.", "Datenladen, Hardware", "Nicht geprueft.", "nur Titel", ["amplitudenkodierung", "approximatives-laden"])
A("kodierung_uebersicht2026", "2606.05387", None, 2026, "(Merkmalskodierung Uebersicht)", None, "Mottonen-artig ca. 980 CNOT; Untergrenze ca. 123; ca. 510 CNOT bei D=256 mit Fidelity ca. 0,21 bei 3e-3.", "Hardware", "Auszug (Umfrage).", "Auszug", ["amplitudenkodierung", "zustandspraeparation"])
A("shot_kodierung2026", "2604.06135", None, 2026, "(Shot-basierte Kodierung)", None, "Alternative Kodierung.", "Hardware", "Nicht geprueft.", "nur Titel", ["datenkodierung"])
A("born_nichtstab2026", "2609.32025", None, 2026, "(Born-Maschinen und Nichtstabilitaet)", None, "Treffer, nicht gelesen.", "Hardware/Magie", "Nicht gelesen.", "nur Titel", ["born-auslese", "nichtstabilizerness-qml"])
A("cdr2021", "2005.10189", "Czarnik, Arrasmith, Coles, Cincio", 2021, "Error mitigation with Clifford quantum-circuit data (CDR)", "Quantum 2021 (Titel aus Wissen)", "Regressionsmodell auf Clifford-nahen Trainingsschaltungen fuer Fehlerminderung.", "Hardware (C-hardware)", "Clifford-NAH fuer nicht-Clifford-Ziel; Klassifikator + CDR nicht gefunden.", "Auszug", ["fehlerminderung", "clifford-naehe"])
A("cdr_extrapolation2025", "2511.03556", None, 2025, "(CDR mit Non-Clifford-Extrapolation und Energy Sampling)", None, "Erweiterung von CDR.", "Hardware", "Auszug.", "Auszug", ["fehlerminderung"])
A("clapton2024", "2406.15721", None, 2024, "Clapton", None, "Clifford-unterstuetzte Problemtransformation fuer VQAs.", "Hardware", "Clifford als Initialisierung/Rauschmodell.", "Auszug", ["clifford-initialisierung", "fehlerminderung"])
A("clifford_twirling2024", "2412.01311", None, 2024, "(Clifford-Twirling / Pauli-Fehlerpropagation)", None, "Siehe Hardware-Recherche.", "Hardware", "Auszug.", "Auszug", ["fehlerminderung"])
A("cafqa_ion2024", "2408.06482", None, 2024, "(CAFQA auf Ionenfalle)", None, "CAFQA auf Ionenfalle demonstriert.", "Hardware", "Auszug.", "Auszug", ["clifford-initialisierung"])
A("mckay2016zgates", "1612.00858", "McKay et al.", 2016, "Efficient Z-Gates for Quantum Computing", None, "Rz auf IBM/Transmon-Hardware virtuell (Phasenverschiebung, Dauer null).", "Hardware", "Belegt (Auszug).", "Abstract", ["hardware-rz-virtuell"])
A("gpu_sim2025", "2509.04955", None, 2025, "(GPU-Simulation: cuQuantum, PennyLane-Lightning-GPU u. a.)", None, "Siehe LABORBUCH.", "Simulation", "Auszug.", "nur Titel", ["simulation-gpu"])

# ---------------------------------------------------------------- QAS, Rechenaufwand
A("lu2023qasbench", None, "Lu et al.", 2023, "QAS-Bench", "ICML 2023 (PMLR 202)", "Vergleicht Brute Force, SA, GA, RL, differenzierbare Suche; SA skaliert schwach.", "I-005, I-036", "Auszug.", "Auszug", ["architektursuche", "simulated-annealing"])
A("martyniuk2024qas", "2406.06210", "Martyniuk et al.", 2024, "Quantum Architecture Search: A Survey", None, "Survey.", "I-005", "Auszug.", "Auszug", ["architektursuche"])
A("elivagar2024", "2401.09393", None, 2024, "Elivagar", None, "Effiziente Schaltungssuche fuer Klassifikation.", "I-005", "Auszug.", "Auszug", ["architektursuche"])
A("hadamard_muster_mnist2024", "2407.12506", None, 2024, "(Klassifikation aus Hadamard-Basis-Messungen)", None, "MNIST/Fashion aus 64 Hadamard-Mustern: klassisch ca. 96/84 %, QNN 95/81 %.", "I-036", "Zahlen ungeprueft.", "Auszug", ["hadamard-walsh", "qml-bilder"])
A("ncc_phase_2022", "2208.07851", None, 2022, "Lernen von Phasenzustaenden", None, "Mathematik/Lernen von Phasenzustaenden.", "I-036-v3", "Quadratische Form dort Darstellung bzw. Lernziel.", "Auszug", ["graphzustaende", "clifford-quadratform"])
A("stab_lernen_2609_10974", "2609.10974", None, 2026, "(Stabilizer-Lernen)", None, "Stabilizer-Lernen.", "I-036-v3", "Quadratische Form als Lernziel.", "Auszug", ["clifford-quadratform"])
A("stab_lernen_2607_02444", "2607.02444", None, 2026, "(Stabilizer-Lernen)", None, "Stabilizer-Lernen.", "I-036-v3", "Quadratische Form als Lernziel.", "Auszug", ["clifford-quadratform"])
A("materialisierung2026", "2604.15405", None, 2026, "(Materialisierung, Stabilizer)", None, "Materialisierung.", "I-036-v3", "Auszug.", "Auszug", ["clifford-quadratform"])
A("single_photon_cs2026", "2604.25480", None, 2026, "Quantum Compressed Sensing Enables Image Classification with a Single Photon", None,
  "Gelernte Phasenmaske (SLM) + Hadamard-artige Messung zur Bildklassifikation.", "I-036-v3 (naechster Nachbar)", "Optisch und kontinuierlich, nicht diskret-Clifford, kein Rang; vor der Arbeit lesen.", "Auszug", ["hadamard-walsh", "phasenmaske", "schusszahl"])
A("i036_v2_html", "2408.03351", None, 2024, "(Quelle der Neuheitspruefung I-036-v2)", None, "Titel nicht genannt.", "I-036-v2", "Nur als Quelle gelistet.", "nur Titel", ["clifford-simulation"])
A("i036_v2_pdf", "2303.15665", None, 2023, "(Quelle der Neuheitspruefung I-036-v2)", None, "Titel nicht genannt.", "I-036-v2", "Nur als Quelle gelistet.", "nur Titel", ["clifford-simulation"])
A("i036_v2_2605", "2605.23138", None, 2026, "(Quelle der Neuheitspruefung I-036-v2; CRiSP 2026 evtl. dieselbe)", None, "Clifford-Initialisierung von VQAs (PRA 111, 062413, 2025; CRiSP 2026) - Zuordnung ungeklaert.", "I-036-v2", "Clifford als Start.", "Auszug", ["clifford-initialisierung"])
A("i036_v2_2608", "2608.01653", None, 2026, "(Quelle der Neuheitspruefung I-036-v2)", None, "Titel nicht genannt.", "I-036-v2", "Nur als Quelle gelistet.", "nur Titel", ["clifford-simulation"])

# ---------------------------------------------------------------- Parametereffizienz / Skalierung / Adapter / Sprache
A("chen2015hashednets", "1504.04788", "Chen et al.", 2015, "HashedNets", None, "Bei 1/64-Kompression 1,99 % Fehler auf MNIST vs. 2,69 % bei gleich grossen Standardnetzen.", "Parametereffizienz (Zaehlfrage)", "Die Hashfunktion zaehlt dort als frei.", "Auszug", ["parametereffizienz", "kompression"])
A("courbariaux2016", "1602.02830", None, 2016, "BinaryConnect / BNN", None, "BinaryConnect 98,82 %, BNN 98,60 % auf MNIST.", "Parametereffizienz", "Spart Bit pro Gewicht, nicht Gewichtsanzahl.", "Auszug", ["quantisierung-ternaer", "parametereffizienz"])
A("twn2016", "1605.04711", None, 2016, "Ternary Weight Networks", None, "TWN 99,35 % auf MNIST.", "Parametereffizienz, T-ternaer", "Auszug.", "Auszug", ["quantisierung-ternaer", "parametereffizienz"])
A("blier2018", "1802.07044", "Blier, Ollivier", 2018, "The Description Length of Deep Learning Models", "NeurIPS 2018", "Beste prequentielle Codelaenge fuer MNIST 4,10 kbit (Compression-Ratio 0,021).", "Bit-Budget-Pareto (05_parametereffizienz 4.1)", "Parameterunabhaengige Vergleichsgroesse.", "Auszug", ["parametereffizienz", "mdl"])
A("frankle2019lottery", "1803.03635", "Frankle, Carbin", 2019, "The Lottery Ticket Hypothesis", "ICLR 2019", "LeNet-300-100 auf MNIST mit 1,8 % der Gewichte ca. 97 %.", "Parametereffizienz", "Topologie (Indexbits) wird meist nicht mitgezaehlt.", "Auszug", ["parametereffizienz", "pruning-klassisch"])
A("li2018intrinsic", "1804.08838", "Li, Farkhoor, Liu, Yosinski", 2018, "Measuring the Intrinsic Dimension of Objective Landscapes", "ICLR 2018", "MLP 784-200-200-10 erreicht 90 % mit d_int ca. 750.", "Parametereffizienz", "Nur Suchauszug.", "Auszug", ["parametereffizienz", "intrinsische-dimension"])
A("aghajanyan2021", "2012.13255", "Aghajanyan et al.", 2021, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", None, "Hunderte bis Tausende; groessere Modelle niedriger.", "Parametereffizienz, Adapter", "Auszug.", "Auszug", ["parametereffizienz", "intrinsische-dimension", "adapter-peft"])
A("tsetlin_fuzzy2023", "2301.08190", None, 2023, "Fuzzy-Pattern-Tsetlin-Maschine", None, "98,56 % auf MNIST mit 20 Klauseln.", "Bit-Budget-Vergleich", "Klassische Bit-Zaehlweise.", "Auszug", ["parametereffizienz", "tsetlin"])
A("tsetlin_2025", "2508.08350", None, 2025, "(Tsetlin-Maschine, Literale/Klausel mit Pruning)", None, "Literale/Klausel < 5-13 mit Pruning.", "Bit-Budget-Vergleich", "Anderes Papier, Auszug.", "Auszug", ["parametereffizienz", "tsetlin"])
A("hoffmann2022chinchilla", "2203.15556", "Hoffmann et al.", 2022, "Training Compute-Optimal Large Language Models (Chinchilla)", None, "Ueber 400 Modelle; Modellgroesse und Tokenzahl gleich skalieren.", "Skalierung (T-006)", "Nur Auszug.", "Auszug", ["skalierungsgesetze"])
A("besiroglu2024chinchilla_repl", "2404.10102", "Besiroglu et al.", 2024, "Chinchilla Scaling: A replication attempt", None, "Drei Probleme am Fit: zu enge Konfidenzintervalle, Optimierer vor Konvergenz gestoppt, gerundete Parameter.", "Skalierung (Fits fragil)", "Lehre: Fits an wenigen Punkten fragil.", "Auszug", ["skalierungsgesetze"])
A("caballero2022bnsl", "2210.14891", "Caballero et al.", 2022, "Broken Neural Scaling Laws", None, "Gebrochenes Potenzgesetz modelliert Knicke, Double Descent.", "Skalierung", "Auszug.", "Auszug", ["skalierungsgesetze"])
A("krona2022", "2212.10650", "Edalati et al.", 2022, "KronA", None, "Kronecker-Adapter.", "T-006 (QIRE-Kronecker-Drehung), Adapter", "Direkter klassischer Konkurrent zur QIRE-Kronecker-Drehung mit log2(Breite) Winkeln.", "Auszug", ["adapter-peft", "kronecker", "quanten-peft"])
A("kronlora2025", "2508.01961", None, 2025, "Kron-LoRA / LoKr", None, "Bis 4x weniger Parameter als LoRA bei aehnlicher Ausdrucksstaerke.", "Adapter", "Auszug.", "Auszug", ["adapter-peft", "kronecker"])
A("boft2023", "2311.06243", "Liu et al.", 2023, "BOFT (Orthogonal Butterfly Fine-Tuning)", None, "Orthogonale Butterfly-Faktorisierung, O(d log d) Parameter; schlaegt LoRA/OFT bei NLU und Mathe.", "T-006, Adapter", "Naechste klassische Verwandte des QIRE-Adapters (d log d statt log d Parameter).", "Auszug", ["adapter-peft", "butterfly", "unitaer"])
A("oft2023", "2306.07280", None, 2023, "OFT (Orthogonal Finetuning)", None, "ADE20K: r=2/4/8/16 -> mIoU 27,18/27,06/24,09/21,0; Saettigung bei Orthogonalbloecken.", "T-006 (Lagen-Saettigung)", "Volltext: kein Exponent.", "Volltext", ["adapter-peft", "unitaer"])
A("quantum_peft2025", "2503.05431", None, 2025, "Quantum-PEFT", None, "Pauli-Parametrisierung, Parameterzahl logarithmisch in der Dimension, Rang voll; Saettigung ab L=3.", "T-006, QIRE-Adapter", "Engste Konkurrenz/Vorarbeit zum QIRE-Adapter; Born-Quadrat-Alleinstellung widerlegt, Alleinstellung ueber Messung/Kontrollen.", "Auszug", ["adapter-peft", "quanten-peft", "born-auslese"])
A("quanta2024", "2406.00132", None, 2024, "QuanTA", "NeurIPS 2024", "Quantenschaltkreis-inspirierte Tensoren, hochrangig.", "T-006", "Engste Konkurrenz/Vorarbeit zum QIRE-Adapter.", "Auszug", ["adapter-peft", "quanten-peft"])
A("quic2025", "2502.06916", None, 2025, "QuIC", None, "< 0,02 % Speicher.", "T-006", "Engste Konkurrenz/Vorarbeit.", "Auszug", ["adapter-peft", "quanten-peft"])
A("zhang2024scaling_peft", "2402.17193", "Zhang u. a.", 2024, "When Scaling Meets LLM Finetuning", None, "Multiplikatives Gesetz; PEFT-Exponenten winzig (|alpha| << 1e-2): Skalieren der PEFT-Parameter ineffektiv.", "T-006 (sqrt(L)-Gesetz)", "Volltext.", "Volltext", ["adapter-peft", "skalierungsgesetze"])
A("rslora2023", "2312.03732", None, 2023, "rsLoRA", None, "Skalenfaktor alpha/sqrt(r) statt alpha/r stabilisiert hoehere Raenge.", "T-006", "Auszug.", "Auszug", ["adapter-peft", "skalierungsgesetze"])
A("lora_miub2025", "2501.03152", None, 2025, "MIUB-Skalierungsgesetz fuer LoRA", None, "Im PDF nur Zusammenfassung, keine Exponentenzahl.", "T-006", "Auszug.", "Auszug", ["adapter-peft", "skalierungsgesetze"])
A("finetuning_gesetze2025", "2512.14237", None, 2025, "(zitiert Zhang et al. 2024: L(D) = A/D^lambda + E fuer LoRA)", None, "Fine-Tuning-Gesetze.", "T-006", "Auszug.", "Auszug", ["skalierungsgesetze", "adapter-peft"])
A("scaling_hybrid_qnn2026", "2604.06007", "Vyskubov et al.", 2026, "Scaling Laws for Hybrid QNNs", None, "Mehr Qubits verbessert meist, mehr Quantenlagen nicht zuverlaessig.", "T-005/T-006", "Gegenpunkt zu T-005.", "Auszug", ["skalierungsgesetze", "qml"])
A("ohno2026lie_scaling", "2608.20603", "Ohno", 2026, "Scaling Behavior of PQCs, Lie-algebraic", None, "DLA-Dimension und Observablen-Orbit saettigen schnell mit der Parameterzahl.", "T-006 / v5", "Effektive Dimension statt nominale Parameter als Skalierungsachse.", "Auszug", ["skalierungsgesetze", "lie-algebra"])
A("rydberggpt2026", "2609.20912", None, 2026, "Do Quantum Models Scale Like LLMs? (RydbergGPT)", None, "Transformer auf Rydberg-Messdaten; Potenzgesetz nahe kritischem Punkt.", "T-006", "Transformer auf Quantendaten, nicht Quantenmodell.", "Auszug", ["skalierungsgesetze"])
A("rydberggpt_methode2024", "2405.21052", None, 2024, "RydbergGPT", None, "Transformer auf Rydberg-Atomen.", "T-006", "Auszug.", "Auszug", ["skalierungsgesetze"])
A("pqc_skalierung_2506", "2601.07824", None, 2026, "(SRE nur bis ca. 15 Qubits direkt berechenbar)", None, "Hessematrix waechst mit T-Count, Saettigung bei ca. 10 (ungeklaert).", "Magie-Messung", "Nicht verifiziert.", "Auszug", ["sre", "skalierungsgesetze"])
A("natural_quantization2025", "2503.15482", None, 2025, "Natural Quantization of NN", None, "Randtreffer.", "T-ternaer", "Randtreffer.", "nur Titel", ["quantisierung-ternaer"])
A("matryoshka_quant2025", "2502.06786", None, 2025, "Matryoshka Quantization", None, "IDEEN.md: Richardson auf die Logits.", "IDEEN", "Nur Verweis.", "nur Titel", ["quantisierung-ternaer"])
A("polarquant2026", "2603.29078", None, 2026, "PolarQuant", None, "IDEEN.md.", "IDEEN", "Nur Verweis.", "nur Titel", ["quantisierung-ternaer"])
A("quarot2024", "2404.00456", "Ashkboos, S. et al.", 2024, "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", "NeurIPS 2024", "Rotation vor Quantisierung.", "T-ternaer", "Aus 11_hardware_effizienz_klassisch (mit [?]).", "nur Titel", ["quantisierung-ternaer", "hadamard-walsh"])
A("spinquant2024", "2405.16406", "Liu, Z. et al.", 2024, "SpinQuant: LLM Quantization with Learned Rotations", None, "Gelernte Rotationen.", "T-ternaer", "Aus 11_hardware_effizienz_klassisch (mit [?]).", "nur Titel", ["quantisierung-ternaer"])
A("fastfood_variants_dummy", None, None, None, "(keine Eintraege)", None, "-", "-", "-", "nur Titel", [])
# Entfernen des Platzhalters
_E.pop()

# TensorNetz / Quanten-Sprachmodelle
A("tensorized_embedding2019", "1901.10787", None, 2019, "Tensorized Embedding Layers", None, "WikiText-103, 3,8x Kompression, +1,3 Test-Perplexity.", "T-Linie", "Auszug.", "Auszug", ["tensornetz-kompression", "sprachmodell"])
A("tensorgpt2023", "2307.00526", None, 2023, "TensorGPT", None, "Bis 38,4x Embedding-Kompression.", "T-Linie", "Auszug.", "Auszug", ["tensornetz-kompression", "sprachmodell"])
A("tensorslm2025", "2506.13514", None, 2025, "TensorSLM", None, "Tensor-Train-Sprachmodell.", "T-Linie", "Auszug.", "Auszug", ["tensornetz-kompression", "sprachmodell"])
A("lm_tensor_trains2024", "2405.04590", None, 2024, "Language Modeling Using Tensor Trains", None, "TT-Sprachmodell, Details unklar.", "T-Linie", "Auszug.", "Auszug", ["tensornetz-kompression", "sprachmodell"])
A("compactifai2024", "2401.14109", None, 2024, "CompactifAI", "Multiverse", "MPO-Zerlegung; LLaMA-2 7B auf 30 % der Groesse, > 90 % Genauigkeit (Firmenangaben teils Marketing).", "T-Linie", "Auszug.", "Auszug", ["tensornetz-kompression", "sprachmodell"])
A("lambeq2021", "2110.04236", None, 2021, "lambeq", "Quantinuum", "Python-Bibliothek, Saetze (DisCoCat) -> Quantenschaltungen.", "Quanten-NLP", "Auszug.", "Auszug", ["quanten-nlp"])
A("lambeq_gen2_2025", "2505.13208", None, 2025, "Efficient Generation of Parameterised Quantum Circuits from Large Texts", None, "lambeq Gen II, Text-Ebene.", "Quanten-NLP", "Auszug.", "Auszug", ["quanten-nlp"])
A("compositional_text2024", "2408.06061", None, 2024, "Quantum Algorithms for Compositional Text Processing", None, "Begleitpapier.", "Quanten-NLP", "Nur Titel gesehen.", "nur Titel", ["quanten-nlp"])
A("quixer2024", "2406.04305", "Khatri, Matos, Coopmans, Clark", 2024, "Quixer", "Quantinuum", "Token-Superposition per LCU + QSVT; Penn Treebank; verglichen mit LSTM/Transformer/FNet.", "T-Linie, Born als Attention", "Naechste Naehe zu 'Born-Regel als Attention'.", "Auszug", ["quanten-nlp", "sprachmodell", "born-auslese"])
A("duneau2024", "2409.08777", "Duneau et al.", 2024, "Scalable and interpretable QNLP: an implementation on trapped ions", None, "QDisCoCirc, Fragebeantwortung auf handgebauten Spielzeugdaten auf Quantinuum H1-1.", "Quanten-NLP", "Auszug.", "Auszug", ["quanten-nlp"])
A("qpa2025", "2410.09846", None, 2025, "QPA - Quantum Parameter Adaptation", "ICLR 2025", "QNN erzeugt LoRA/DoRA-Gewichte ueber klassischen MLP; GPT-2: 52,06 % der LoRA-Parameter (+0,75 %); Inferenz klassisch.", "T-006", "Nahe: Qubit-Zahl log2 m, aber Parameter ueber MLP.", "Auszug", ["adapter-peft", "quanten-peft", "quanten-nlp"])
A("qtha2025", "2503.12790", None, 2025, "QTHA - Quantum-Enhanced LLM Efficient Fine Tuning", None, "-76 % trainierbare Parameter gegenueber LoRA; Simulation unklar.", "T-006", "Auszug.", "Auszug", ["adapter-peft", "quanten-peft"])
A("quantum_llm_ft2025", "2504.08732", "Kim, Mei, Girotto, Yamada, Roetteler", 2025, "Quantum LLM Fine-Tuning", None, "Satz-Transformer + VQC-Kopf, bis +3,14 % Genauigkeit.", "T-Linie", "Klein.", "Auszug", ["adapter-peft", "quanten-nlp"])
A("hyquT2025", "2511.10653", "Kong", 2025, "Hybrid Quantum Transformer (HyQuT)", None, "VQCs im Transformer bei 8M/150M Parametern; 10 Qubits/80 Gatter ersetzen ca. 10 % der Parameter.", "T-Linie", "Simulation unklar.", "Auszug", ["quanten-nlp", "adapter-peft"])
A("hyperq2026", "2609.24657", "Wang et al. (Montreal/Mila)", 2026, "HyperQ - Circuit Hypernetworks", None, "IQP-Schaltungen exakt klassisch auswertbar; 1,1-Mrd.-LM, 64 Qubits +4,71 ggue. Backbone, +3,67 ggue. LoRA.", "T-006", "Naechster Verwandter zu Clifford/diskret im LLM-Adapter.", "Auszug", ["adapter-peft", "quanten-nlp", "klassische-simulierbarkeit"])
A("quantum_self_attention2026", "2603.03318", None, 2026, "Quantum-Inspired Self-Attention in a LLM", None, "Nur Titel.", "T-Linie", "Born als Attention evtl. schon angelegt.", "nur Titel", ["quanten-nlp", "born-auslese"])
A("quantum_attention_vit2024", "2411.13520", None, 2024, "Quantum Attention for ViT in HEP", None, "Nur Titel.", "T-Linie", "Nur Titel.", "nur Titel", ["quanten-nlp"])
A("qattention_survey2025", "2504.03192", None, 2025, "(Survey Quanten-Attention)", None, "Survey.", "T-Linie", "Nur Titel.", "nur Titel", ["quanten-nlp"])
A("qpsan2026", "2605.25365", None, 2026, "QPSAN (Bild)", None, "Nur Titel.", "T-Linie", "Nur Titel.", "nur Titel", ["quanten-nlp"])
A("tn_review_2023", "2304.13539", None, 2023, "Tensor Decomposition for Model Reduction in Neural Networks: A Review", None, "Review (literature card tensor_networks).", "A/sources", "Literaturkarte.", "nur Titel", ["tensornetz-kompression"])
A("han2016deepcompression", "1510.00149", "Han et al.", 2016, "Deep Compression: ... Pruning, Trained Quantization and Huffman Coding", "ICLR 2016", "Pruning + Quantisierung + Huffman.", "A/sources (pruning)", "Literaturkarte.", "Auszug", ["pruning-klassisch", "quantisierung-ternaer"])
A("hyperband2018", "1603.06560", "Li et al.", 2018, "Hyperband", "JMLR 2018", "Bandit-basierte Hyperparametersuche.", "A/sources (hyperband)", "Literaturkarte.", "Auszug", ["hyperparametersuche"])
A("moe2017", "1701.06538", "Shazeer et al.", 2017, "The Sparsely-Gated Mixture-of-Experts Layer", None, "MoE-Schicht.", "A/sources (mixture_of_experts)", "Literaturkarte.", "Auszug", ["mixture-of-experts"])
A("farhi2014qaoa", "1411.4028", "Farhi et al.", 2014, "A Quantum Approximate Optimization Algorithm", None, "QAOA.", "A/sources (qaoa_qubo)", "Literaturkarte.", "Auszug", ["qaoa"])
A("eat2023", "2307.00368", None, 2023, "Energy-Aware Training (EAT)", None, "Energieverbrauch von DL-Modellen senken.", "A/sources (energy_aware_training)", "Literaturkarte.", "Auszug", ["energieeffizienz"])
A("pareto_nas2018", "1808.09830", None, 2018, "Pareto-Optimal Device-Aware Neural Architectures", None, "Pareto-Optimierung.", "A/sources (pareto_optimization)", "Literaturkarte.", "Auszug", ["pareto", "architektursuche"])

# Graphzustaende / MBQC
A("mbqml2024", "2405.08319", None, 2024, "Measurement-based quantum machine learning", None, "MBQC-QML.", "Graphzustaende (09)", "Auszug.", "Auszug", ["graphzustaende", "mbqc"])
A("mbqcnn2024", "2412.08207", None, 2024, "Measurement-based quantum CNN", None, "Deep Learning durch Messung.", "Graphzustaende (09)", "Auszug.", "Auszug", ["graphzustaende", "mbqc"])
A("resource_kernel2025", "2507.03689", None, 2025, "A Resource Efficient Quantum Kernel", None, "Titel passt zu Graphzustands-Kernel; nur Snippet.", "Graphzustaende (09)", "Nicht gelesen.", "nur Titel", ["graphzustaende"])
A("graph_0812_4625", "0812.4625", None, 2008, "(Graphzustaende; Titel nicht genannt)", None, "Quelle in 09_graphzustaende.", "Graphzustaende (09)", "Titel fehlt.", "nur Titel", ["graphzustaende"])
A("magic_robustness_1807_10296", "1807.10296", None, 2018, "(Robustheit der Magie / Mana / Stabilizer-Polytop; Zuordnung unklar)", None, "Howard & Campbell 2017; Veitch et al.; Heinrich & Gross 2019 - ar5iv-Link nennt 1807.10296.", "Magie-Ressourcentheorie", "Zuordnung zu einer der drei Arbeiten ungeklaert.", "Auszug", ["magie-mass", "ressourcentheorie"])
A("grosse_unklar_2504_10595", "2504.10595", None, 2025, "(in der Recherche in der Naehe von Holmes-Matsuura genannt)", None, "Siehe 03_datenladen.", "Datenladen", "Nicht zuordenbar.", "nur Titel", ["zustandspraeparation"])
A("stoudenmire_umfeld_1905_01426", "1905.01426", None, 2019, "(in der Naehe von Stoudenmire&Schwab genannt)", None, "Siehe 05_parametereffizienz.", "Parametereffizienz", "Nicht zuordenbar.", "nur Titel", ["tensornetz"])
A("saade_cluster_2008_15617", "2608.15617", None, 2026, "(in 10_red_team bei Benchmarking genannt)", None, "Siehe 10_red_team 2.2.", "Red Team", "Nicht zuordenbar.", "nur Titel", ["benchmark-qml"])
A("redteam_2604_18837", "2604.18837", None, 2026, "(in 10_red_team bei Benchmarking genannt)", None, "Siehe 10_red_team 2.2.", "Red Team", "Nicht zuordenbar.", "nur Titel", ["benchmark-qml"])
A("redteam_2304_04932", "2304.04932", None, 2023, "(in 10_red_team bei Dequantisierung genannt)", None, "Siehe 10_red_team 2.6.", "Red Team", "Nicht zuordenbar.", "nur Titel", ["dequantisierung"])
A("redteam_2111_09079", "2111.09079", None, 2021, "(in 10_red_team bei Dequantisierung genannt)", None, "Siehe 10_red_team 2.6.", "Red Team", "Nicht zuordenbar.", "nur Titel", ["dequantisierung"])
A("redteam_2508_00230", "2508.00230", None, 2025, "(in 10_red_team 2.8 genannt)", None, "Siehe 10_red_team 2.8.", "Red Team", "Nicht zuordenbar.", "nur Titel", ["adapter-peft"])
A("hw_2606_21686", "2606.21686", None, 2026, "(in 08_hardware bei CDR genannt)", None, "Endlich-Schuss-Analyse CDR vs. ?", "Hardware", "Nicht zuordenbar.", "nur Titel", ["fehlerminderung"])
A("hw_2607_grosse", "2608.24891", None, 2026, "(in 01_holevo bei Schusszahl vs. Marge genannt)", None, "Siehe 01_holevo 2.3.", "Holevo", "Nicht zuordenbar.", "nur Titel", ["schusszahl"])

ALLE = _E
