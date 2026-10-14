# Jury-Übungsset (R4.2)

*Stand 14.10.2026. Die 27 Fragen stammen wörtlich im Sinn aus `UEBERARBEITUNG_2026-09-23/BERICHT/RECHERCHE_2026-09-28/07_schriftliche_arbeit_jury.md` §4 (dort: Ableitung aus den Jurykriterien, keine offiziell veröffentlichte Liste). Kurzform der Quelle dort: „Recherche 07“.*
*Antworten sind Kurzfassungen (je ca. 30 Sekunden) und folgen `CLAUDE.md`: ehrlich „quanten-inspiriert“, nicht „Quantenvorteil“. Zahlen stehen mit Quelle da. Alles mit „zu prüfen“ ist unsicher oder muss Malte in eigenen Worten bestätigen. Malte muss jede Antwort selbst erklären können (Eigenanteil, Kriterium Nr. 1).*

Abkürzungen: LB = `INSTITUT/LABORBUCH.md`, KA = `INSTITUT/KERNAUSSAGE_2026-10-07.md`, UE = `INSTITUT/UEBERGABE.md`, HO = `INSTITUT/HANDOVER_2026-10-06.md`, ERG = `INSTITUT/ergebnisse/`, HA = `UEBERARBEITUNG_2026-09-23/A/HANDOFF_AKTUELL.md`, FB = `INSTITUT/FEHLERBUCH.md`.

---

## Teil 1: Die 27 Fragen

### Eigenanteil und KI-Nutzung
**1. Was hast du selbst entwickelt, was kommt aus Bibliotheken?**
Eigen: Versuchsdesign, Vorregistrierung, Kontrollen, Auswertung, die Module unter `INSTITUT/abteilungen/`. Aus Bibliotheken: PyTorch, NumPy, SciPy, scikit-learn, Qiskit (`CLAUDE.md` „Technik“). Die Mischungsverfahren (Fastfood, Haar, SORF) sind aus der Literatur und werden zitiert. *Zu ergänzen:* konkrete Funktionen, die Malte am Code zeigen kann. Beleg: KA §1 Satz 3; `INSTITUT/recherche/2026-10-06_literatur_struktur.md`.

**2. Welche Teile hat ein KI-Werkzeug geschrieben? Kannst du sie erklären?**
Der Arbeitsablauf ist offen: „Opus plant, Sonnet-Agenten führen aus“ (UE §2 Nr. 8); Module und Routinearbeit entstanden mit KI-Hilfe, Planung und Urteile prüft Malte. Das gehört in „Unterstützungsleistungen“ der Arbeit. **Zu prüfen:** Wortlaut des KI-Leitbilds von Jugend forscht (Recherche 04/07). **Vorbereitung:** Malte erklärt Kernfunktionen (Fastfood-Mischung, TOST, Holm) ohne Hilfe. Beleg: UE §2; `CLAUDE.md`.

**3. Wie bist du auf die Idee gekommen? Was hast du gelernt?**
Ausgangsziel: ressourceneffizientere KI mithilfe von Quantenphysik (UE §1). Gelernt: Der „Quanten“-Teil trägt nichts Messbares, die Struktur schon (I-038). **Persönliche Geschichte muss Malte selbst formulieren.** Beleg: UE §1; LB 06.10.2026 I-038.

**4. Warum diese Fragestellung, was war schwer?**
Schwer waren Fairness der Kontrollen und Fehlerquellen: geerbter Kopf, entartete Kontrollen, Lernrate je Arm (FB; UE §3). Beleg: UE §3 „Neu gelernt“; FB.

### Quanten
**5. Ist das echtes Quantencomputing? Was ist „quanten“, was nur inspiriert?**
Nein. Alles läuft klassisch auf CPU/GPU. Inspiriert sind: Hadamard/Walsh-Mischung, Born-artige Quadrat-Auslese |z|², Clifford-Begriffe. Die Wirkung ist klassisch erklärbar (CLAUDE.md; I-038). Beleg: KA §1; LB 06.10.2026 I-038.

**6. Was ist Qubit, Superposition, Verschränkung, und kommt das im Modell vor?**
Die Zelle simuliert Qubits (Amplitudenkodierung, Ry-Drehungen, CNOT-Ring, Born-Auslese). Verschränkung trägt nichts bei: QIRE gelernt ≡ Zufallswinkel (I-038 H-B: +0,22 / +0,07 pp, GLEICHWERTIG) und ≡ feste Mischung (H-A). Es ist eine andere Parametrisierung, kein Vorteil. *Grenze:* „ohne CNOT“ verliert 1,3 / 2,0 pp (Erkundung v3), Fastfood holt das ohne Quantenstruktur wieder auf. Beleg: `ERG/I-038__20261006_065007.json`; HO §2 (v3).

**7. Warum Clifford, was heißt Gottesman-Knill für „Quantenvorteil“?**
Clifford-Schaltungen sind klassisch effizient simulierbar (Gottesman-Knill), also kein Quantenvorteil. Ich behaupte ausdrücklich **keinen**. Clifford diente als Ressourcenzählung (T-Gatter, „Magie“), nicht als Quantennutzen (KA §3). Beleg: KA §1, §3; UE §4.

**8. Würde das auf echter Quantenhardware laufen? Rauschen?**
Nicht untersucht. Die Amplitudenkodierung braucht 247–377 CZ-Gatter; Rz ist auf NISQ-Hardware virtuell, „0 T“ hilft dort kaum; das Datenladen ist der Engpass (7.636 T = 7,7× die Schaltung, C-kodierung). Auslese braucht etwa 550 Schüsse/Bild für 90 % (v4, Erkundung). Beleg: HO §2; UE §4 (C-kodierung); `ERG/C-kodierung__20260924_192627.json`.

**9. Warum nicht Physik als Fachgebiet?**
Weil der Befund eine Algorithmen-/Effizienzaussage ist, keine Quantenexperiment-Aussage; reale Quantenhardware wird nicht eingesetzt. **Mit Wettbewerbsleitung abstimmen.** Beleg: `GLIEDERUNG.md` §2; Recherche 04 „Fachgebietswahl“.

### Methodik und wissenschaftliche Exzellenz
**10. Wie fair ist der Vergleich (16 Winkel gegen 67.848-Parameter-MLP)?**
Dieser Vergleich ist **zurückgenommen**: Die 67.848-Kontrolle lag hinter einem linearen Modell (HA „Nachtrag 23. September“). In der heutigen Arbeit werden die Hauptvergleiche gegen faire Kontrollen geführt: klassische Nichtlinearitäten (ReLU, tanh, |z|), gleiches Bitbudget (I-039-v2), linearer und budgetgleicher Kopf (I-044-v2), Lernrate je Arm auf eigenen Tuning-Seeds. Beleg: HA Z. 59–63, 204; LB 06.10.2026 I-039-v2; LB 07.10.2026 I-044-v2.

**11. Wie viele Seeds, welche Streuung, signifikant oder Zufall?**
Meist 10–20 Seeds (z. B. 20 Seeds 2000–2019 in I-038), manche 15. Tests: exakter Wilcoxon, Holm-Korrektur, 90-%-KI, TOST für Gleichwertigkeit. Ältere Läufe I-042/I-044 hatten nur 5 Seeds und wurden mit 15 Seeds wiederholt (I-042-r, I-044-r). Beleg: LB 07.10.2026 (I-042-r, I-044-r); UE §3; KA §3 „Offene Offenlegungen“.

**12. Was heißt „vorregistriert“, wie beweist du, dass nichts nachträglich geändert wurde?**
Kriterien, Seeds und Auswertung stehen vor dem Lauf im Code und in einem Laborbuch-Eintrag „VORREGISTRIERUNG“, committet vor den Zahlen (git-Historie mit Zeitstempeln). Das ist keine externe Registrierung (z. B. OSF). Abweichungen sind offengelegt, z. B. I-039-Rauchtest auf Mess-Seed 2020. Beleg: UE §3; LB 06.10.2026 (I-038, I-039 Vorregistrierung); Recherche 07 §2.

**13. Was wurde ausprobiert und nicht berichtet?**
Nichts wird verschwiegen: alle Läufe stehen in `agenda.json` (55 Punkte, davon 47 „erledigt“), alle Rohdaten in `INSTITUT/ergebnisse/`, jeder Fehlschlag im FEHLERBUCH. Verworfene Ideen: z. B. Z-zne auf Papier. Beleg: `INSTITUT/agenda.json`; FB; `JUGEND_FORSCHT/dashboard/dashboard.html`.

**14. Wurde auf dem Testsatz abgestimmt? Data Leakage?**
Lernraten werden auf eigenen Tuning-Seeds gewählt, getrennt von Mess-Seeds (Protokollregel; Rauchtests nur auf Tuning-Seeds). Ausnahme offengelegt: I-039-Rauchtest. **Zu prüfen:** ob Trainings-/Validierungs-/Testaufteilung je Modul in der Arbeit einheitlich beschrieben werden kann. Beleg: UE §3; `CLAUDE.md`; LB 06.10.2026 I-039.

**15. MNIST/Fashion sind kleine, gelöste Benchmarks. Trägt das auf größere Aufgaben?**
Teilweise nein, und das steht in der Arbeit: auf CIFAR-10 kostet die feste Struktur −6,3 pp (I-042-r H1, 15 Seeds). Sie trägt dort, wo die Eingabe schon gute Merkmale hat (CNN-Kopf, Fashion). Alle Bilder sind 16×16 bzw. resampeltes 32×32. Beleg: `ERG/I-042-r__20261007_095606.json`; KA §3.

**16. Warum ist Walsh bei wenig Daten besser, ab wann kippt es? Mechanismus bewiesen?**
*Aus Projekt A (Q-019), im Institut nicht neu geprüft:* Der Mechanismus ist **Vermutung**, nicht bewiesen. Bewiesen ist nur der Satz „Rang 2h ⇒ flaches Walsh-Spektrum“ (v2/09). Heutige Linie: Mischung trägt, nicht Walsh speziell (Fastfood ≡ Haar²). Beleg: KA §2 (Satz bewiesen); `UEBERARBEITUNG_2026-09-23/BERICHT/RECHERCHE_2026-09-28/02_q019_walsh_datenknappheit.md` (Inhalt nicht neu gelesen, **zu prüfen**).

**17. Warum reicht Clifford auf MNIST, aber nicht auf Fashion?**
I-005-v2 (20 Seeds): MNIST GLEICHWERTIG (0 Magie), Fashion VOLL_BESSER um 1,31 pp; 4 Magiewinkel (≈ 123 T) schließen etwa die halbe Lücke (I-005-v3). Ursache nicht belegt; später zeigt I-036/I-036-v2, dass der Clifford-Nutzen großteils lokale Hadamard-Mischung plus abgestimmte Maske ist. Beleg: UE §0; `ERG/I-005-v2__20260928_203403.json`; `ERG/I-005-v3__20260929_162054.json`.

### Chatbot-Adapter
**18. Was macht der Adapter, auf welchem Basismodell? Was ist dein Beitrag?**
QIRE-Adapter auf einem vortrainierten Sprachmodell (LLäMmlein-120M, laut Handover „T-004/T-005 liefen mit 120M auf CPU“). Eigen: Adapter-Entwurf und Messungen; das Basismodell ist fremd (Lizenz RAIL-M, research-only). Beleg: HO §6; `recherche/2026-10-06_chat_sweep/C.md`.

**19. Wie misst du „besser“? Vergleich mit LoRA bei gleicher Parameterzahl?**
T-005: Verlust fällt streng monoton mit der Lagenzahl; L10 erreicht 55 % des LoRA-Gewinns mit 1/16 der Parameter; QIRE+LoRA ergänzt nicht (UE §0). Der Vergleich L = 20/40 gegen LoRA (T-006) ist **noch nicht ausgewertet** (KA §4). Beleg: UE §0; KA §4.

**20. Wo versagt er? Beispiele für Fehlantworten.**
**Lücke:** Fehlbeispiele sind im Institut nicht zusammengestellt. *Aufgabe:* aus Projekt-A-Läufen (`UEBERARBEITUNG_2026-09-23/A/`) 3–5 Beispiele heraussuchen. Zu prüfen.

**21. Warum 192 Parameter, Zufall oder Suche?**
`qire_kron` mit 192 Parametern war der als PRIMÄR vorregistrierte Arm in T-004 (Projekt A, PROJEKTTAGEBUCH Z. 5022). **Warum genau 192, ist hier nicht belegt: zu prüfen**, bevor Malte antwortet. Beleg: `UEBERARBEITUNG_2026-09-23/A/PROJEKTTAGEBUCH.md` (Z. 5022).

### Ressourceneffizienz und Relevanz
**22. Wie definierst du Ressourceneffizienz? Energie/Zeit gemessen?**
Gemessen werden: reelle Parameter, formale und strenge Strukturbits (I-039: 110 Parameter, +882 / 3.046 Bit), Operationsverhältnis (89× bei N = 1.024) und CPU-Zeit (schneller ab N = 1.024 bei Batch 1). **Energie wurde nicht gemessen.** Beleg: `ERG/I-039__20261006_051108.json`; `ERG/I-041__20261006_100927.json`; KA §2.

**23. Wer würde das nutzen? Was ist neu gegenüber bestehenden Methoden?**
Ehrlich: strukturierte Mischungen und |·|² sind bekannt (Fastfood 2013, SORF 2016, OPU-Merkmale, Primer 2021). Beitrag: kontrollierter, vorregistrierter Vergleich mit Negativ-/Positivkontrollen und die Messung der Information pro Schuss (I-040-v2, I-046: Literaturlücke plausibel, nicht bewiesen). Beleg: KA §1 Satz 3; `recherche/2026-10-06_literatur_struktur.md`; LB 07.10.2026 I-046.

**24. Wichtigster offener Punkt, nächster Schritt?**
T-006-Auswertung, Fastfood-Schicht in einem Sprachmodell (I-043 auf PC, I-045-v2 Replikation), MPI-Skalierung ≥ 1 Mrd. Parameter, Volltext-Neuheitsprüfung. Beleg: KA §4.

### Selbstkritik
**25. Größte Schwäche?**
(1) Kleine Benchmarks. (2) Wirkung klassisch erklärbar, kein Quantenvorteil. (3) Der Vorsprung des Strukturkopfes ist ein Trainingsbudget-Effekt (I-044-v3: bei 24 Epochen −0,51 pp gegen linearen Kopf). (4) Literatur nur aus Suchauszügen. Beleg: KA §3, §4; `ERG/I-044-v3__20261007_153149.json`.

**26. Welches Ergebnis hat dich überrascht oder deine Erwartung widerlegt?**
Der Haar-Befund: eine zufällige Haar-Orthogonale + Quadrat ist gleichauf mit gelernter QIRE-Zelle (0,938 / 0,834 gegen 0,938 / 0,837; Erkundung v3), danach I-038 H-A bis H-C vorregistriert bestätigt. Zweitens: der „+5,61 pp“-Vorsprung aus I-039 war kein Born-Effekt (I-039-v2). Drittens: I-035, Magiebedarf nicht aus der Generalisierungslücke vorhersagbar (H0). Beleg: HO §2; LB 06.10.2026; UE §4.

**27. Was würdest du mit mehr Zeit oder Rechenleistung ändern?**
Größere Modelle (MPI), mehr Seeds und Epochen, echte Sprachmodell-Tests, Energiemessung, Volltext-Neuheitsprüfung, bei gleichem Bit-/Parameterbudget gegen Fastfood mit trainierten Diagonalen (I-046-Folgefrage). Beleg: KA §4; LB 07.10.2026 I-046.

---

## Teil 2: 15 zusätzliche kritische Fragen (Red Team)

**Z1. Ist das nicht einfach klassisch?**
Ja, die Wirkung ist klassisch erklärbar (I-038). Genau deshalb heißt es „quanten-inspiriert“. Beleg: CLAUDE.md; KA §1.

**Z2. Dequantisierung: Gibt es überhaupt etwas, das ein klassischer Rechner nicht auch könnte?**
Die Born-Merkmale sind ein 256-dimensionaler Unterraum der quadratischen Formen; der QIRE-Kern ist für jedes Magie-Budget dequantisiert, klassische RBF-/Quadrat-Kerne sind bei 1.000 Trainingsbildern besser (94,4 / 93,7 % gegen 82–91 %). *Erkundung, nicht vorregistriert, Fashion fehlt.* Beleg: HO §2 (v6); `recherche/2026-10-05_vertiefung/v6_dequantisierung.md`.

**Z3. „Haar = QIRE“ heißt, deine Zelle ist überflüssig?**
Als Quantenelement ja: gelernte Winkel tragen nichts Messbares (I-038 H-A: +0,18 / +0,10 pp, TOST ±0,5 pp). Praktischer Gewinn liegt in der Struktur (Fastfood, 0 gelernte Mischparameter, O(N log N)). Beleg: `ERG/I-038__20261006_065007.json`.

**Z4. Warum nicht einfach ReLU?**
Gemessen: Quadrat gegen |z| MNIST +0,35 pp, Fashion 0,00 (I-038 H-G); I-039-v2 GLEICHWERTIG; I-042 wechselt je Aufgabe. Quadrat ist **nicht** die bessere Nichtlinearität. Beleg: KA §3.

**Z5. Primer kennt „squared ReLU“ schon. Wo ist das neu?**
Nicht neu, so in der Arbeit. Neuer Teil: Sz (Fastfood + z²) schlägt D_z2 (dicht + z²) um 0,075 bpc (Wechselwirkung); in kleinem Modell, ein Datensatz, Literaturprüfung dazu offen. Beleg: LB 07.10.2026 I-045-v2 (Satz zu Literatur am Ende abgeschnitten, **zu prüfen**).

**Z6. Ist „110 Parameter“ nicht geschönt, wenn 882 bis 3.046 Strukturbits dazukommen?**
Beide Zahlen werden berichtet (formal 882, streng 3.046). Der faire Bitvergleich I-039-v2 zeigt: Mit gleichem Bitbudget und |z| erreicht ein klassisches Modell dasselbe (80,65 gegen 80,26 %). Die 110 Parameter zeigen Struktur, keinen Born-Vorteil. Beleg: LB 06.10.2026 I-039, I-039-v2.

**Z7. Ihr habt mehrere Datensätze und Hypothesen getestet. Multiples Testen, Forking Paths?**
Holm-Korrektur über die angemeldete Familie, Kriterien vor dem Lauf, Gegenrichtungen werden veröffentlicht (viele H0/HU: Dashboard zeigt 35 H0 und 54 HU Zeilen neben 100 H1-Zeilen, heuristisch gezählt). Beleg: UE §3; `JUGEND_FORSCHT/dashboard/dashboard.html`.

**Z8. Wie sicher ist „GLEICHWERTIG“ (TOST ±0,5 pp)? Warum genau ±0,5?**
Die Grenze wurde vor dem Lauf festgelegt (relativ zum Effekt); sie ist eine Setzung. Aussage gilt nur für diese Grenze. **Begründung der Grenze: Malte aus LB 06.10.2026 I-038 Vorregistrierung nachlesen und in eigenen Worten formulieren.** Beleg: LB 06.10.2026; FORSCHUNGSMETHODIK.md.

**Z9. Kontrollen: Woher weißt du, dass sie überhaupt etwas messen können?**
Negativkontrolle (Zufallslabel ≈ 10 %) und Positivkontrolle in jedem Lauf; z. B. I-040 PK-1 versagte (0,868 < 0,88) und machte H2/H3 ungültig, daher I-040-v2. Beleg: `ERG/I-040__20261006_070724.json`; LB 06.10.2026.

**Z10. Ein Teil der Ergebnisse (Etappe 2, v1–v6) ist „Erkundung“. Verkaufst du die als Befunde?**
Nein. Sie sind als Erkundung/Hinweis gekennzeichnet (KA §2). Nur vorregistrierte Läufe zählen als Befund. Beleg: HO §2 „Erkundung, keine Befunde“.

**Z11. Der Zielkonflikt Genauigkeit/Information pro Schuss: nur weil die gelernte Matrix 32.640 Parameter hat?**
Das ist die offene Einschränkung (LB 07.10.2026 I-046): Ein Teil des Vorteils ist „mehr Parameter“; die budgetgleiche Front mit Fastfood-Diagonalen steht aus. Beleg: LB 07.10.2026 I-046.

**Z12. Ihr nutzt ein fremdes Sprachmodell. Was ist dann „euer“ Ergebnis?**
Der Adapter und seine Messung (T-005). Das Basismodell ist fremd (RAIL-M, research-only). Zusätzlich I-045 in einem selbst trainierten kleinen Modell (4 Lagen, d = 256). Beleg: HO §6; LB 07.10.2026.

**Z13. Ist die Ideen-KI nicht der eigentliche Erfinder und du nur Prüfer?**
Stand 14.10.2026: Die Ideen-KI ist nur geplant, wartet auf Freigabe, schlägt nur vor (kein Commit, kein Lauf). Falls sie in der Arbeit erscheint: „auf QIRE spezialisiertes, feinjustiertes Basismodell“, Urteile und Läufe entscheidet Malte (Plan §7). Beleg: `IDEEN_KI/PLAN_2026-10-13.md` §0, §7.

**Z14. Reproduzierbarkeit: Läuft das bei anderen?**
Rohdaten, Seeds, Module und Wiederaufnahme sind im Repo (`INSTITUT/ergebnisse/`, `abteilungen/`), Datensätze lädt `labor/gemeinsam.py`; Läufe liefen auf Container und PC. Ein öffentliches Repo mit DOI ist geplant (`VEROEFFENTLICHUNG.md`). *Zu prüfen:* unabhängiger Neustart in sauberer Umgebung. Beleg: `CLAUDE.md` „Technik“.

**Z15. Was, wenn ein Juror sagt: Das ist Fastfood von 2013, mit neuem Namen?**
Zustimmen und abgrenzen: Fastfood ist Grundlage und zitiert (`CLAUDE.md`: Neuheit nicht nötig, Bekanntes darf Grundlage sein). Der Beitrag ist die kontrollierte, vorregistrierte Messung samt Grenzen (CIFAR, Budgeteffekt, Information pro Schuss). Beleg: KA §1; `recherche/2026-10-06_literatur_struktur.md`.

---

## Teil 3: Offene Lücken für Malte (vor dem Gespräch schließen)
1. Fragen 2, 3, 16, 20, 21: Teile hängen von Maltes eigenem Wissen oder Projekt-A-Material ab.
2. Frage 8/14: Hardware- und Datenaufteilungsdetails je Modul bündeln.
3. T-006 und I-043 fehlen noch (Fragen 19, 24).
4. Neue Fragen kann die Ideen-KI später erzeugen (`IDEEN_KI/PLAN_2026-10-13.md` §8 R4.2); nur nach Freigabe.
