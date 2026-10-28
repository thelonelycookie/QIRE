# f3: Inkubation (28.10.2026) — verworfene Ideen im Licht der vier H0 und des z²-Signals

Nichts gerechnet, kein git. Gelesen: 00_AUFTRAG, KREATIVITAET, SYNTHESE (27.10.), TURNIER, konv_kandidaten (A–E), KERNAUSSAGE, IDEEN.md, ideen.json (22 verworfen, 13 weitergeleitet), FEHLERBUCH, LABORBUCH (I-044-v3, I-045-v2/v3/v4, I-047/I-049/I-045-v5 Vorregistrierungen), w9, r3.
**Quellenlage:** Die Ergebnisse von I-049, I-047, I-046-v2 und T-ternaer-butterfly stehen noch nicht im LABORBUCH meiner Kopie. Ich übernehme ihre Zahlen nur aus dem Auftrag (Raven, 28.10.). Was ich daraus ableite, ist unter diesem Vorbehalt zu lesen.

## 1. Kurzfassung
- **Das „Frühregime-Gesetz“ ist ein Muster, noch kein Gesetz.** Belegt: ein Vorzeichenwechsel des Strukturvorteils mit mehr Training in drei unabhängigen Linien (CNN-Kopf I-044-v3, Sprachmodell-FFN I-045-v3, Hourglass I-049 gegen den Rauchtest R2). Nicht belegt: die Form „∝ 1/Training“ (je Linie nur 2–3 Punkte, der Vorteil wird negativ statt null) und die Ursache (Rechenbudget, Epochen oder Lernrate, siehe §3).
- **Vier H0 sind keine einheitliche Gruppe.** I-049 und I-045-v3/I-044-v3 zeigen die zeitliche Umkehr. I-047 zeigt nur den Endstand. I-046-v2 und T-ternär-butterfly sind anderer Art (Budget und Optimierbarkeit). Siehe §2.
- **z² > GELU ist selbst früh-gebunden, nur nicht überall.** Dichter Gewinn z² gegenüber GELU: d=256: +0,045 / +0,118 / +0,013 bpc (T = 1000 / 4000 / 16000); d=512: +0,188 / +0,081 / **−0,047**. Schmale dichte FFN (v4, T = 16000): −0,011. Nur im Tensor-Train (I-047, T = 16000) bleibt +0,040. Lesart: z² hilft, wo Kapazität oder Training knapp sind; das wäre dasselbe Gesetz eine Ebene tiefer, und es ist bekannt (Primer/squared ReLU).
- **Wiederbelebung lohnt für drei Linien:** (a) Gewichtsteilung über Lagen, Schlüssel-Superposition (I-017 → K01), (b) Cache-Bytes mit Rotation (I-027 → K05), (c) Struktur als Anlauf mit Übergabe (I-016/I-021/I-007 → K13/K14). Alle drei nur mit Lernkurve statt Endpunkt. Streichen bleibt: TT, Born-Attention, Phasen-Gewichte, Frühausstieg, ZNE/PEC, TT-Einbettung.
- **Der eine Lauf:** „Vorteilskurve“ (Arbeitsname I-050): Sprachmodell d = 256, vier Arme aus I-045-v5, acht T-Werte von 250 bis 32000, 8 Seeds, Lernrate je (Arm, T), Kosinus je T, Trainings-FLOPs als x-Achse, Anpassung L(C) = L∞ + A·C^−α je Arm. Etwa 200–260 Kernstunden, ein Raven-Tag (Schätzung, §5).

## 2. Was die vier H0 zusammen sagen
| Lauf | Was verlor | Wann | Art des Scheiterns |
|---|---|---|---|
| I-049 | Hourglass + LRU, −0,055 bpc gegen flach (T = 16000) | nach Rauchtest +0,33 bpc bei 190 Schritten (R2: 3,372 gegen 3,703) | zeitliche Umkehr, vorab von den Berichten als wahrscheinlich genannt |
| I-047 | Tensor-Train-FFN, etwa −0,06 gegen dicht/Niedrigrang | Endstand | Struktur gibt Parameter, aber bei gleichen FLOPs zu wenig Freiheitsgrade (χ = 4 war vom FLOP-Budget erzwungen) |
| I-046-v2 | strukturierte Drehung mit 384 Parametern besser als id/Haar? Nein | kein Zeitbezug | Budgetgrenze: ein winziges Budget trägt die Front nicht |
| T-ternär-butterfly | ternärer Butterfly kollabiert | kein Zeitbezug | Optimierbarkeit; offen, ob Latentgewichte/STE fehlten (Vortest/Code prüfen) |

Gemeinsam ist den ersten beiden und den älteren Linien (I-044-v3, I-045-v3/v4): **wer von Grund auf mit gleichen FLOPs gegen eine dichte Matrix antritt, verliert bei ausreichendem Training.** Dichte Matmuls sind der härteste Gegner, weil sie die Rechnung pro FLOP am besten nutzen und Hardware-optimiert sind. Struktur „verkauft“ Parameter und Bytes, nicht FLOPs.

**Wo der Vorteil überlebt**, alles außerhalb von „gleiche FLOPs, von Grund auf, langes Training“:
- Bytes eines **fertig trainierten** Modells (R1: Rotation + 2 Bit, +0,4 pp bei gleichen Bytes; I-048 läuft). Das Gesetz greift hier nicht, weil nicht trainiert wird.
- Winzige Budgets (I-039: 110 Parameter; T-ternär: 64 Bit gegen 103.904 Bit, +1,8 pp). Das ist die Spitze des Frühregimes.
- Knappe Trainingsschritte (Adapter, T-006: bei 300 Schritten 2,5× je Parameter besser als LoRA).
- Langes Training vermutlich nur in Kombination mit z² in engen Modellen (TT +0,040).

**Ist das ein Befund für die Arbeit? Ja, in vorsichtiger Form.** Tragfähige Aussage: „Quanten-inspirierte Struktur verschiebt die Lernkurve nach links (schnelleres Lernen), nicht nach unten (besserer Endstand). Wo das Budget vor dem Kreuzungspunkt C* endet, gewinnt sie.“ Der saubere Beleg ist R2 → I-049: derselbe Vergleich, Rauchtest +0,33 bpc, vorregistriert −0,055 bpc. Das ist zugleich ein Methodenbefund (Rauchtests mit 190–600 Schritten sind systematisch zugunsten neuer Struktur verzerrt).

## 3. Warum „Strukturvorteil ∝ 1/Training“ zu stark ist
1. **Form:** In I-045-v3 (d = 256, Sz gegen D_gelu: +0,112 / +0,067 / −0,093) wird der Vorteil **negativ**. Ein reines 1/C-Gesetz geht gegen null, nicht darunter. Passender ist ein Zweitermmodell: Struktur hat größeres A (schneller) und größeres L∞ (höherer Boden). Δ(C) = a·C^−α − b mit Kreuzung bei C*. Das ist die übliche Skalierungsgesetz-Lage (Inductive Bias gewinnt klein, verliert groß; Literatur aus dem Gedächtnis: Kaplan 2020, Tay u. a. 2022, ViT gegen CNN 2020; vor Verwendung prüfen).
2. **Nicht monoton:** d = 512, Sz gegen D_z2: −0,342 / −0,376 / −0,024. Sz liegt dort nie vorn, und die Lücke schrumpft nur zuletzt. W2 in w9 (Sz wird mit d schlechter) und der Vorbehalt „Lernrate nur bei T = 4000 gesucht“ gelten: Der Verdacht auf Lernrate-Artefakt ist offen. Mit I-045-v5 (Lernrate je T) entscheidet sich das.
3. **Drei Störgrößen sind mit „Training“ vermengt:** (a) Rechen-FLOPs, (b) Epochen auf 1,6 MB (T = 16000 sind 10 Epochen, bei I-049 T = 16000 sogar etwa 20), also Überanpassung statt „zu viel Training“, (c) Lernrate. Auf 1,6 MB ist der Boden datenbegrenzt, beide Arme laufen auf denselben Boden zu. Ein „Schrumpfen“ des Vorteils kann deshalb rein ein Bodeneffekt sein.
4. **Einheiten:** CNN-Kopf in pp, LM in bpc. Vergleichbar wird es erst über den relativen Überschuss (L − L∞)/L∞ je Arm aus der Anpassung.
5. **Datenlage:** I-044-v3: 3 Punkte, I-045-v3: 3 pro (Arm, d), I-049: 3 geplant (nur der Endpunkt im Auftrag), I-047: Endpunkt. Die vorhandenen Punkte sagen **Vorzeichenwechsel zwischen etwa 4 und 12 Epochen bzw. zwischen 4000 und 16000 Schritten** (lineare Interpolation in log T: Sz gegen D_gelu, d = 256, Nullstelle bei T ≈ 7000, etwa 7 Mio Bytes). Mehr nicht.

## 4. Wiederbelebung: welche verworfene Idee lohnt, und warum
Maßstab: (i) neues Wissen ändert die Begründung, (ii) es gibt einen billigen Test mit fieser Kontrolle, (iii) die Idee verlangt nicht, dass Struktur beim Endstand gewinnt.

### 4.1 Ja
| Alt | Neu | Warum jetzt | Test (Tuning-Seeds) | Vorab-P |
|---|---|---|---|---|
| **I-017** Trotter-Winkelteilung über Lagen („spart nur 32 Zahlen, kein Engpass“) | **K01** Schlüssel-Superposition, Turnier 2. | Im LM sind es Millionen Gewichte, also ein echter Byte-Hebel; FLOPs bleiben gleich. Zugleich der Kandidat mit dem größten Frühregime-Risiko (weniger freie Parameter lernen früh schnell, Boden höher) | Pflicht: Lernkurve T = 600 / 2400 / 9600, Kontrolle „geteilt ohne Schlüssel“ und „kleineres dichtes Modell gleicher Bytes“ | 30 % bei langem T (K-Bericht: 35 % kurz) |
| **I-027** Dekohärenz → KV-Cache („nur Analogie, H2O/StreamingLLM“) | **K05** Cache-Bytes, Rotation + 1–2 Bit, Haar-Alterung | Ein Cache wird erst zur Inferenzzeit gepackt: **das Gesetz greift nicht**, der Hebel liegt auf der Byte-Achse, die nach den vier H0 als einzige überlebt. Rotation hat in R1 schon gegen die Zufalls-Orthogonale gewonnen; es ist dieselbe Kontrolle | wie K05; Pflicht: Vergleich mit KIVI/Kanalskala ohne Rotation | 40–50 % |
| **I-016 / I-021** MERA/RG-Encoder, **I-007** adaptive Trotter-Tiefe | **K13** Kontext-Curriculum, **K14** wachsendes Modell, Hourglass-Anlauf | Das Gesetz sagt: Struktur ist früh billig und gut. Man nutzt sie deshalb **nur früh** und übergibt dann an dichte Rechnung (Anlauf mit Übergabe). Im besten Fall liegt man auf der unteren Hüllkurve beider Lernkurven. Gewinn nur im Gebiet C < C*, plus Übergabekosten; ehrlich als begrenzt einplanen | Erst die Kurve (§5). Dann: Sz-Anlauf 20 % der Schritte, dann dichte FFN (funktionserhaltende Verbreiterung wie K14), gleiche Gesamt-FLOPs | 25 % |

### 4.2 Bedingt (abhängig von I-045-v5 / I-048)
- **Sz als Anlauf (SYNTHESE-„rot“ nach v4):** Wenn I-045-v5 H1 zeigt, dass Sz bei **ideal gleichen Operationen** (Di_z2, h = 42 / 46) nicht verliert, ist das der einzige Rest an FLOP-Argument. Dann ist Sz Kandidat für den Anlauf in 4.1. Fällt H1, ist Sz endgültig Speicherlösung.
- **Born-Normierung K09 und Hill K20:** Das z²-Signal ist ein Aktivierungseffekt (I-045-v2 H3, R3, I-047 H3). Wenn z² bei knapper Kapazität bleibt, sind Normierung und Beschränkung (Bytes der Aktivierung, 3–4 Bit) die logischen Nachfolger. Einen eigenen Lauf nicht vorher.
- **T-ternär-butterfly:** Kollaps kann Optimierung sein. Die Linie nur öffnen, wenn der Arm mit Latentgewichten (STE) trainiert wurde; sonst ist es „QAT fehlt“ und gehört zu E3 (QAT-Paket).

### 4.3 Nein (bleibt verworfen, jetzt mit mehr Gewicht)
- **Tensor-Train / MPS (SYNTHESE rot, IDEEN Nr. 6, V-kompass):** I-047 bestätigt. V-kompass sagte schon: Niedrigrang-SVD schlägt jede TT-Ordnung, der Rang kommt von der Aufgabe.
- **Born-Attention (w3):** Der Rauchtest war im Frühregime, also dort, wo neue Struktur am besten aussieht, und fand trotzdem keinen Vorteil. Das ist ein stärkeres Nein als „zu kurz trainiert“.
- **Phasen-Gewichte (SYNTHESE Rang 4, gestrichen):** R3: reell z² schlägt Nur-Phase bei gleichen FLOPs, Zufallsphasen = gelernte. Der Rest ist z² plus doppelte FLOPs. Nur als Speicherzahl (Winkel statt Zahl) halten, kein eigener Test.
- **Frühes Aussteigen / Zeno (IDEEN „nicht aufgenommen“, w6):** bleibt Early-Exit, kein Quantenbezug.
- **I-022 ZNE, I-023 PEC, I-025/I-030 Adiabatik, I-031 Imaginärzeit, I-029 Lieb-Robinson:** Papier-Widerlegung oder reine Analogie; das neue Wissen ändert nichts.
- **I-014 / I-013 / I-018 TT- und Tensor-Einbettung:** auf Byte-Ebene ist die Einbettung 256 × d, kein Engpass. Erst wieder bei Token-Vokabular (Chatbot), dann als Übernahme (word2ket), nicht als Befund.
- **I-009 Pauli-Propagation, I-011 Quantenlauf, I-012 Schatten:** keine Wirkung auf Training oder Bytes; Hardware-Fragen.

## 5. Der eine Lauf, der das Gesetz quantitativ belegt: „Vorteilskurve“
**Frage:** Existiert ein Kreuzungspunkt C*, wo liegt er in Trainings-FLOPs, und wandert er mit Epochen (Daten) oder nicht (Rechnen)?
**Warum ein Lauf genügt:** I-044-v3, I-045-v3, I-049 liefern schon 10–15 Punkte, aber auf verschiedenen Rastern, mit Lernrate bei nur einem T, Einheiten und Gegnern gemischt. Der Lauf legt ein gemeinsames Raster über **ein** Modellpaar und füllt die Lücken.

**Aufbau (Entwurf, Wiederverwendung des I-045-v5-Moduls):**
- Byte-LM, d = 256, 4 Lagen, Kontext 128, Batch 8 × 128, Adam, 5 % Aufwärmen, Kosinus **je T** (jeder T eigener Lauf, nicht eine Kurve abgeschnitten).
- Arme: **Sz** (Fastfood + z², Struktur), **Di_z2** (ideal gleiche Operationen), **Df_gelu** oder **Df_z2** (gleiche Matmul-FLOPs), **D_gelu** (volle Breite). Dazu D_z2 für die Aktivierungsfrage (Δ_Aktivierung(C) = D_gelu − D_z2 aus denselben Läufen, kostet nichts).
- **T ∈ {250, 500, 1000, 2000, 4000, 8000, 16000, 32000}** (acht Werte, log2-Raster; ¼ bis 20 Epochen auf voller Datenmenge). Seeds 8. Lernrate **je (Arm, T)** auf dem Tuning-Seed.
- **Trainings-FLOPs:** C = 3 · F_vorwärts(Gesamtmodell, beide Zählungen: matmul und ideal) · Bytes. F aus dem Code, nicht von Hand (Vortest wie in I-045-v5).
- **Kontrollen gegen die Störgrößen:** (1) halbe Datenmenge für T ∈ {4000, 16000, 32000} (trennt Epochen von FLOPs); (2) Train-bpc je Lauf; (3) falls möglich ein Zusatzarm auf frischen Daten ohne Wiederholung (> 33 MB Text, das braucht den 2,9-GB-Korpus von Maltes PC; auf Raven nur, wenn er dorthin darf).
- **Hypothesen (vorab):** H1 Kreuzung existiert: Δ(C_min) > 0 und Δ(C_max) < 0 für Sz gegen Df, gepaart je Seed. H2 Δ fällt monoton in log C (Spearman über acht T, je Seed, Mittel). H3 C* aus der Anpassung L(C) = L∞ + A·C^−α je Arm, mit Bootstrap-KI über Seeds; Vorhersage: C* ≈ 7 Mio Bytes bei d = 256 (aus §3). H4 C* in Epochen oder in FLOPs? Verschiebt die halbe Datenmenge C* in Epochen, ist es ein Datenregime; bleibt es in FLOPs, ein Rechenregime. H5 (Zusatz, nicht Holm): Aktivierungsvorteil Δ_Akt(C) folgt demselben Muster.
- **Urteilsprosa im Voraus:** Kreuzung belegt → „Strukturvorteil nur unterhalb C*“, mit C* als Zahl. Keine Kreuzung → Sz schlechter ab T = 250 (Lernrate-Artefakt?) oder Sz gleichwertig (dann FLOP-Vorteil real).
- **Kosten (Schätzung, aus den Schrittzeiten der I-045-v5-Vorregistrierung, d = 256: Sz 0,32, Df 0,20, Di 0,15, D_gelu 0,32 s):** 63 750 Schritte je (Arm, Seed) ≈ 4,4 h Kern; 4 Arme × 8 Seeds ≈ 140 Kernstunden, Tuning und Halbdaten dazu, mit Aufschlag etwa 200–260 Kernstunden. Längste Aufgabe 32 000 Schritte ≈ 3 h (mit Aufschlag), Wandzeit mit Tuning in zwei Runden etwa 8–10 h auf 72 Prozessen. **Passt in ein Raven-Fenster; nach Maltes Freigabe.**
- **Vorab-P (grob):** H1 Kreuzung existiert 75 %; H2 monoton 60 %; H4 Rechenregime (C* bleibt in FLOPs) 45 %, Datenregime 40 %, unentscheidbar 15 %.
- **Reihenfolge:** erst I-045-v5 und I-048 abwarten (dort läuft die Lernrate je T und die ideale Zählung; v5 liefert bei T = 4000 / 16000 schon zwei Punkte der Kurve).

### 5.1 Gratis davor (keine Freigabe nötig)
- **Vorteilskurve v0 aus Bestand:** Aus den JSON von I-044-v3, I-045-v3/v4, I-049, I-047 eine Tabelle Δ gegen C bauen (Trainings-FLOPs und Epochen), alle Linien in eine Abbildung, relative Skala. Ohne Rechnen. Das zeigt sofort, ob die Linien auf eine Kurve fallen, und welche Lücken der Lauf schließen muss.
- **Billigere Alternative zum Lauf, nur zum Üben:** eine Lernkurve mit WSD-Plan (konstante Rate, Abzweig mit Abkühlen bei acht Schrittzahlen) statt acht getrennter Kosinus-Läufe. Spart etwa Faktor 3, aber ersetzt nicht die Lernrate je T; als Rauchtest auf Tuning-Seeds im Container machbar.

## 6. Regeln, die aus dem Gesetz folgen (sofort anwendbar)
1. **Rauchtest-Regel „zwei Punkte“:** Von-Grund-auf-Ideen (K01, K04, K12, K13, K14, K16, K17) werden bei zwei Längen getestet (z. B. 600 und 2400 Schritte). Weiter nur, wenn der Vorteil mit T nicht um mehr als die Hälfte schrumpft. Die 600-Schritt-Rauchtests der 20 Kandidaten sind sonst alle systematisch zugunsten der Neuheit verzerrt (Beleg: R2 → I-049).
2. **Bytes-Ideen zuerst:** K03, K05, K06, K07 und die I-048-Familie sind gegen das Gesetz immun (fertig trainiertes Modell, Nachbehandlung). Kein Aufwand für Lernrate je T nötig.
3. **K04 (eingefrorene Seed-Aufwärtsschicht)** ist der klassische Frühregime-Kandidat (zufällige Merkmale sehen früh gut aus); Vorab-P auf etwa 20 % senken und nur mit Zwei-Punkt-Regel testen.
4. **Der Arbeit-Satz:** „Wir messen den Strukturvorteil als Funktion des Trainingsbudgets, nicht als Zahl.“ Die ehrliche Standardantwort an die Jury auf „Was bringt QIRE?“ lautet dann: Bytes (ohne Training), Winzigbudgets und Anlauf; nicht: besserer Endstand bei gleichen FLOPs.

## 7. Offene Zweifel
- Das z²-Signal ist **nicht** durchgängig: d = 512, T = 16000 liegt D_gelu vorn (2,106 gegen 2,153); relu² (2,113) schlägt z² (2,153) dort ebenfalls. Der Auftrag nennt „ein konsistentes Signal“; in den Daten ist es konsistent bei kurzem und mittlerem T, nicht am Ende. Lernrate-Artefakt bei T = 16000 (Raten nur bei T = 4000 gesucht) ist wahrscheinlich; v5 klärt es nur für die FFN-Arme.
- Die vier H0 sind in meiner Kopie nicht im LABORBUCH. Zahlen aus dem Auftrag, ohne Konfidenzintervalle.
- Alle Literaturnamen in §3 aus dem Gedächtnis; vor Zitat am Original prüfen.
- Dieser Bericht hat selbst nur Daten von Byte-LM auf 1,6 MB Text und einer CNN-Kopf-Aufgabe. Ob das Gesetz bei Milliarden-Modellen gilt, kann nur MPI (offen) beantworten.
