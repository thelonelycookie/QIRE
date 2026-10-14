# Übergabe — Stand 23. September 2026

Kalt lesbar. Ersetzt `HANDOFF_PROJEKT_A.md` (6. August) als Einstieg; jenes
bleibt für den Verlauf bis dahin gültig.

*Abschnitt 00 am 23. September neu (Gesamtreview). Abschnitt 0 ist Stand
30. August, der Rest Stand 27. August abends — mit Nachträgen vom
23. September, die als solche markiert sind.*

---

## 000. Stand 28. September (neuester Abschnitt, zuerst lesen)

- **Q-020 ist gerechnet.** Alle drei Hypothesen halten, jeweils 15/15 Seeds,
  und `q_audit` ist grün:
  - PRIMÄR: 16 Winkel schlagen das faire ReLU-MLP, +0,0113 auf dem
    offiziellen MNIST-Testsatz. Der Mechanismus ist die klassische
    Kronecker-Drehung (`kron` ist gleichwertig).
  - Verschränkung und Tiefe tragen ohne Encoder: +0,34 bzw. +0,33. Klassisch
    geht es dort aber besser (1,000).
  - Die Kurzfassung ist gefüllt. Details im Tagebuch, 28.09.
- **T-002:** Der Zusatzarm `butterfly_ffn_zentriert` läuft standardmäßig mit.
  Der Bedarf liegt bei 74–93 GPU-h (die alte Angabe „30–35 h“ war falsch).
- **Q-019W gerechnet, alle H1:**
  - Butterfly (40) schlägt dense (65.536) bei N = 200 um +0,31.
  - N\* ≈ 3.100.
  - Butterfly ≈ feste WHT (0 Parameter).
  - Die Teilung der Winkel ist der Grund.
  - Die P3-Kontrolle `lowrank` war ein Strohmann und wurde vor dem Lauf durch
    `givens_zufall` ersetzt.
  - Details im Tagebuch, 28.09. Die Kurzfassung ist ergänzt (zweite KI-Art).
- **Institut:** I-005-v2 (Clifford-Suche gegen volles Training, MNIST +
  Fashion, TOST, 20 Seeds) läuft seit dem 28.09. abends.
- **Neuer Chatbot-Weg T-004**, Entwurf `ENTWURF_T004_CHATBOT_ADAPTER.md`:
  QIRE-Kronecker-Drehung mit 11 Winkeln je Matrix als Adapter auf
  LLäMmlein-1B (Breite 2048 = 2¹¹), im Vergleich mit LoRA/BOFT, auf Kaggle.
  Offen: `transformers`/`peft` installieren und das Modell herunterladen
  (Freigabe Malte).
- **Recherchen:** `../BERICHT/RECHERCHE_2026-09-28/` (01–05) und
  `../../INSTITUT/recherche/`. Malte will **keine Abgrenzung** von fremden
  Arbeiten. Fremde Ideen mitbenutzen ist erwünscht, nur im
  Quellenverzeichnis nennen.
- **Jugend forscht:** höchstens 15 Seiten, kein Anhang, KI-Nutzung angeben
  (Recherche 04).
- **Institut:** I-004-mnist ist ausgewertet (R-1 H1: ein Training im
  Clifford-Gerüst genügt). I-005 lief am 28.09.

## 00. Das Erste (23. September): der Gesamtreview und was daraus folgt

Zwischen dem 30. August und dem 23. September ist **kein Lauf** gerechnet
worden (keine neuen Rohdaten). Am 23. September wurde alles — Code, Rohdaten,
`q_audit`, `q_summary`, Tagebuch, Dokumente — gegengelesen. Grundlage:
`../BERICHT/00_REVIEW_BEFUNDE.md`. Tagebucheintrag: „23. September —
Gesamtreview: was hält, was nicht“.

**Die fünf wichtigsten Befunde des Reviews:**

1. **Die Kontrolle der Kernaussage ist zu schwach.** `classical_norm_deep`
   (67.848 Parameter, 0,8407) liegt in einer Stichprobe hinter einem rein
   linearen Modell (0,8544 gegen 0,8873; ReLU-Netz mit gleicher Engstelle
   0,9246). **Die Stichprobe ist KEIN Befund** — 3 Seeds, nicht vorregistriert,
   Lernrate auf denselben Seeds gewählt. Sie reicht aber, um „+0,0932 gegen
   67.848“ nicht mehr als Kernaussage zu führen. Was trägt: 16 Winkel liegen vor
   dem 2.048-Parameter-Quadratmodell. Fairer Nachtest: **Q-020 (a)**.
2. **Verschränkung und Tiefe sind nicht belegt.** Beide TOST-Aussagen stammen
   aus Q-003-replication mit trainierbarem Encoder, und die Tiefe war dort die
   Positivkontrolle — durchgefallen (`q_summary`: „jede
   Gleichwertigkeitsaussage aus diesem Lauf steht nicht“). Nachtest **Q-020 (b)**
   im Q-018-Aufbau ohne Encoder.
3. **T-002 hat vier Fehler**, die vor jedem GPU-Lauf behoben sein müssen
   (Abschnitt 7). Die Wiederaufnahme ist *nicht* sicher, wie hier bisher stand.
4. **B-014: Korrektur nur im Tagebuch.** `intrinsic_dimension.py` rechnete
   weiter pseudorepliziert; das automatische Urteil widerspricht den
   Zellmitteln. `qire_ai.audit` prüft B-014 gar nicht.
5. **Q-019:** die `untied`-Entscheidung ist gefallen (27. Aug.); die
   Primärhypothese ist fast per Konstruktion wahr (Abschnitt 6).

**Parallel in Arbeit (Überarbeitung 23. Sep.):** Code-Korrekturen an
`sprachmodell.py`, `q_summary.py`, `q_audit.py`, `unitarity_depth.py`,
`intrinsic_dimension.py`, `qire_ai/audit.py`, neues `kernaussage.py`;
Dokumente `ENTWURF_Q020.md` und `LITERATUR_UND_ALLEINSTELLUNG.md`. Was davon
fertig ist, steht in `../BERICHT/`.

**Fristen:** Anmeldung **30. November 2026**. Schriftliche Arbeit **Mitte Januar
2027** (eine Landesseite nennt den 17.01.2027 — prüfen). GPU-Ergebnisse müssen
erst im Januar in der Arbeit stehen.

**Neue Kurzfassung:** `KURZFASSUNG_ENTWURF_v2.md`, mit **[OFFEN: …]** für Q-020
und T-002.

---

## 0. (Stand 30. August) Q-011 ist im dritten Anlauf abgebrochen

**Stand 30. August.** Der dritte Q-011-Lauf lief nicht durch. Kein
Python-Prozess mehr aktiv, das Log endet am **27. August, 23:58** bei
**mnist Tiefe 16, Seed 1810 (10 von 15)**. Es gibt keinen Abbruchgrund im Log —
der Prozess ist mit der Sitzung oder dem Rechner gestorben, nicht an einem
Fehler. `fashion_mnist` wurde nie erreicht.

Gerettet ist trotzdem einiges, weil nach jeder Tiefe gesichert wurde:

    results/experiments/Q-011__20260827_230017.json   225 Messzeilen

Darin stecken die Tiefen **1, 2, 3, 4, 8 vollständig** auf mnist, mit dem
richtigen (vollen) Suchbudget. Nur Tiefe 16 fehlt, und mit ihr der ganze
zweite Datensatz.

**Was die Tuning-Abstände zeigen** — Abstand `orthogonal` minus `spectral`
in der Lernratensuche über drei Tuning-Seeds:

| Tiefe | 1 | 2 | 3 | 4 | 8 | 16 (unvollständig) |
|---|---|---|---|---|---|---|
| Abstand | +0,0049 | +0,0014 | +0,0044 | +0,0050 | +0,0067 | **+0,0232** |

Bei Tiefe 16 fällt `spectral` erstmals deutlich ab. Das ist genau die
Richtung, die die Hypothese vorhersagt — **aber es sind Tuning-Zahlen aus drei
Seeds, kein Messergebnis.** Sie gehören nicht zitiert und sind hier nur
notiert, damit die Beobachtung nicht verlorengeht.

**Die Entscheidung, die ansteht:** `unitarity_depth.py` hat **keine
Wiederaufnahme**. Ein Neustart beginnt bei Tiefe 1 und kostet die vollen
10–14 Stunden noch einmal. Drei Möglichkeiten, in dieser Reihenfolge sinnvoll:

1. **Wiederaufnahme einbauen** und nur Tiefe 16 + fashion_mnist nachrechnen.
   Das Muster steht in `sprachmodell.py` (`weiter=True` überspringt fertige
   Arm/Seed-Paare) und ist dort erprobt. Kostet ~1 Stunde Arbeit, spart ~10
   Stunden Rechnen. **Das ist der empfohlene Weg.**
2. Ganz neu starten, diesmal auf dem Server statt auf dem Arbeitsrechner.
3. Q-011 zurückstellen und stattdessen Q-019 (`zeitreihen.py`) bauen —
   der Entwurf steht, das Modul ist noch nicht geschrieben.

**Vor jedem Neustart:** die Gitterverlängerung in `unitarity_depth.py`
verlängert nur nach **oben**. Regel 1(b) verlangt beide Richtungen;
`zeitreihen.py` macht es richtig und ist die Vorlage.

**Ein bekannter Nebeneffekt, damit er keine halbe Stunde kostet:** Bei mnist
Tiefe 8 und 16 liefert `plain` an *allen vier* Lernraten exakt 0,1000 —
Rateniveau bei zehn Klassen. Der unbeschränkte Arm ist dort kollabiert. Weil
alle Werte gleich sind, wählt `max()` die erste Rate und `lr_edge` meldet
`plain` am unteren Rand. **Das ist Kollaps, nicht Untertunierung**, und es ist
unschädlich: `plain` geht nur in die Auflösungsprüfung bei der *kleinsten*
Tiefe ein. Bis Tiefe 4 lagen alle Raten sauber im Inneren (durchweg 0,003).

Die beiden früheren Abbrüche sind im Tagebuch begründet und waren beide
richtig. Der zweite hat den Fehler aufgedeckt, der jede Aussage des Laufs
entwertet hätte: Die Lernratensuche lief mit halbem Budget, wählte dadurch
systematisch zu hohe Raten, und **in fünf von fünf Tiefen gewann ausnahmslos
der Arm mit der kleineren Rate** — um ±0,013, während der gesuchte Effekt bei
+0,005 liegt. Die alten Rohdaten liegen unter
`results/experiments/vorlaeufe_q011/` und sind **nicht verwertbar**.

**Nicht parallel rechnen lassen.** 6 physische Kerne (Ryzen 5 3600), 16 GB.
Immer nur ein schwerer Lauf.

---

## 1. Womit anfangen

    .\py.cmd -m quantum_sim.q_summary    # Belegstand Q/T-Serie
    .\py.cmd -m quantum_sim.q_audit      # Zahlen gegen Rohdaten (106, Stand 30. Aug.)
    .\py.cmd -m qire_ai.audit            # B-Serie -- läuft nur im Archiv (B-Rohdaten fehlen hier)

*(Nachtrag 23. Sep.: Die frühere Angabe `.venv/Scripts/python.exe` stimmte für
diesen Ordner nicht — die Umgebung liegt außerhalb, siehe `CLAUDE.md` §3 bzw.
`py.cmd`. `qire_ai.audit` fällt hier erwartungsgemäß durch, weil die
B-Rohdaten im Archiv liegen.)*

Dann lesen, in dieser Reihenfolge:

1. **`../BERICHT/00_REVIEW_BEFUNDE.md`** — der Gesamtreview vom 23. September
2. **`STAND_UND_OFFENES.md`** — Stichpunktfassung beider Teilprojekte, bester Überblick
3. **`PROJEKTTAGEBUCH.md`**, der Eintrag vom **23. September** und die Einträge vom **27. und 30. August** (ganz unten vor „Zwischenstand")
4. **`KURZFASSUNG_ENTWURF_v2.md`** — die neue Argumentationslinie
5. **`ENTWURF_Q020.md`** — der nächste Lauf (faire Referenz, Verschränkung × Tiefe)
6. **`ENTWURF_T_SERIE_KERNEL.md`** — Roadmap der T-Serie
7. **`ENTWURF_ZEITREIHEN.md`** — Q-019; das Modul ist inzwischen gebaut und geprüft, aber nicht gelaufen

---

## 2. Ziel und Termin

**Ressourceneffizientere KI mithilfe von Quantenphysik.** Bis zum
**30. November** sollen ein **Chatbot** und eine **zweite KI-Art** stehen. Das
Ziel ist gesetzt und wird nicht neu verhandelt. *(Nachtrag 23. Sep.: Die
schriftliche Arbeit ist erst Mitte Januar 2027 fällig; zur Anmeldung reicht
der Stand mit Platzhaltern.)*

Der zentrale Konflikt, mit Zahlen: Die Struktur gewinnt ab **16.384**
Dimensionen, trainierbar ist auf dieser Hardware **256**. Faktor 64. Deshalb
wird **Qualität bei trainierbarer Breite** gemessen und **Effizienz bei ihrer
eigenen Breite**, getrennt berichtet. „Der Chatbot läuft dank der Struktur
schneller" wäre falsch und darf nicht behauptet werden.

---

## 3. Belegstand in einem Absatz

**Projekt B ist abgeschlossen.** Strukturierte Schichten ersetzen keine dichte —
und B-014 erklärt warum: Der Effektivrang wird von der Aufgabe gesetzt, nicht
von der Architektur; die Aufgabe braucht rund sieben Richtungen, der Rang war
nie die bindende Beschränkung.

**Projekt A gilt in Teilen.** ~~16–80 Schaltungsparameter schlagen 67.848
(+0,0932 / +0,0599). Verschränkung und Tiefe tragen nichts bei (TOST).~~
*(Zurückgenommen 23. Sep.: Die 67.848-Kontrolle ist schwächer als ein lineares
Modell — Stichprobe, kein Befund —, und die Verschränkungs-/Tiefenaussagen
stammen aus einem Lauf mit trainierbarem Encoder und durchgefallener
Positivkontrolle.)* **Stand 23. Sep.:** 16 Winkel liegen auf MNIST vor dem
2.048-Parameter-Quadratmodell (nicht vorregistriert); der faire Vergleich
folgt in Q-020. Q-015: die Schaltung schlägt die volle globale Drehung bei
408-fach weniger Freiheitsgraden (MNIST; Fashion zurückgezogen) —
Trainierbarkeit, nicht Ausdruckskraft. Q-018: das Training der Winkel trägt
+0,4077 bei, sobald kein Encoder aushilft.

**T-Serie:** T-000 kalibriert, T-001 mit Urteil T0 (der Umbau bringt 1,6×,
verschiebt den Übergangspunkt aber nicht), T-002 gebaut und vorregistriert, aber
der Lauf ging verloren.

Nicht verwertbar: Q-009 (drei Seeds), Q-014 (Auflösungsprüfung durchgefallen).

---

## 4. Was am 27. August passiert ist — sechs Tagebucheinträge

1. **Q-011 vor dem Start geprüft: fünf Fehler.** scipys asymptotischer Spearman
   (p 0,0000 statt 0,0417), vier Tiefen konnten den Primärtest nach Holm nicht
   bestehen, fehlender Zweig für fallenden Trend, *die eigene Korrektur davon war
   ebenfalls falsch*, und — der schlimmste — `if res_ok is False` griff bei einem
   NumPy-Wahrheitswert nicht: die Auflösungsprüfung wurde berechnet, ausgegeben
   und beim Urteil ignoriert.
2. **Q-011 nach 40 Minuten abgebrochen und neu gestartet**, weil bei Tiefe 2 der
   Mechanismus-Arm am Gitterrand saß und der Kontrollarm nicht.
3. **B-014s p-Werte sind Pseudoreplikation.** Korreliert wurde über 240
   Einzelmessungen statt über 24 Gitterzellen. **Die Befunde halten**, die
   p-Werte waren um bis zu 168 Größenordnungen zu klein. Korrigiert.
4. **B-012 geprüft und sauber** — Permutationstest bestätigt p<0,0001.
5. **Machtanalyse: Regel 2 war zu schwach.** Auf Fashion-MNIST ist die Streuung
   viermal so hoch wie auf MNIST; B-007s dortige Effekte lagen *unter* der
   Nachweisgrenze. „Repliziert nicht" hieß dort „nicht messbar", nicht „nicht
   vorhanden". Für dieselbe Empfindlichkeit wären dort 130 statt 15 Seeds nötig.
6. **Zweiter Q-011-Abbruch: die Lernrate war der ganze Effekt.** Die Suche lief
   mit halbem Budget und wählte deshalb systematisch zu hohe Raten. Der Artefakt
   beträgt ±0,013 — das Dreifache des gesuchten Effekts. Belegt zugleich den
   Lernraten-Verdacht, der am 10. August zu Q-014 notiert wurde.
7. **Nachts: dasselbe halbe Suchbudget steckt in elf Modulen.** Gefunden nicht
   durch gezielte Suche, sondern beim Lesen von `global_orthogonal.py` als
   *Vorlage* für Q-019. Fünf betroffene Läufe haben Rohdaten. **Q-015 auf
   Fashion-MNIST (+0,0201) ist zurückgezogen** — nur das 1,5-fache des
   Artefakts, und der Kontrollarm saß auf dem unteren Gitterrand. Die
   MNIST-Zahlen halten: Q-015 (+0,0674, beide Arme innen, 5,2×),
   Q-003-replication (+0,0932, `hybrid` am oberen Rand, also eher untertrieben,
   7,2×) und Q-018 (+0,4077, Primärpaar innen, 31×). **Q-003 auf Fashion
   (+0,0599) ist dagegen aufgebläht** — dort sitzt der Kontrollarm am unteren
   Rand; hält bei 4,6×, aber nur mit Vorbehalt zitierbar.
   Maßgeblich ist **an welchem Gitterrand** ein Arm sitzt, nicht wer die
   kleinere Rate hat — meine erste Fassung dieser Regel war zu grob und ist im
   Tagebuch korrigiert.

---

## 5. Die Protokollregeln — Regel 1 und 2 sind neu gefasst

1. **Lernrate je Arm — drei Teile.** (a) Eigene Tuning-Seeds je Arm. (b) Das
   Gitter muss das Optimum **enthalten**, und die automatische Verlängerung
   (`ERWEITERUNGEN`) gilt in **beide** Richtungen. Ein Arm am unteren Rand ist
   genauso untertuniert wie einer am oberen — nur fällt es weniger auf, weil die
   Verzerrung dort oft *für* die eigene Hypothese läuft. `unitarity_depth.py`
   verlängert nur nach oben; `zeitreihen.py` in beide. (c) **Die Suche muss
   dasselbe Budget haben wie der Messlauf.** Ein reduziertes Budget ist kein
   Sparmodell, sondern ein anderer Versuch — es verschiebt das Optimum
   systematisch nach oben.
2. **Genug Seeds — zweifach prüfen.** (a) Ist Signifikanz rechnerisch erreichbar?
   Bei n Seeds ist 2/2ⁿ der kleinstmögliche Wilcoxon-Wert, nach Holm über k
   Vergleiche entsprechend mehr. (b) **Ist der erwartete Effekt bei der
   vorhandenen Streuung nachweisbar?** Streuung aus einem Vorlauf schätzen. Ist
   sie es nicht, ist der Lauf ein Nullbefund mit Ansage.
3. **„nicht signifikant" ist nicht „gleich"** — TOST benutzen.
4. **Holm nur über die angemeldete Familie.**
5. **Rohdaten nach jedem Abschnitt sichern**, nicht am Ende — und über
   Maschinengrenzen hinweg (Abschnitt 7).
6. **Positivkontrolle einbauen und prüfen, dass sie feuern kann.**
7. **Kriterien vor dem Lauf ins Skript.** Jede Änderung daran gehört ins
   Tagebuch, *bevor* Zahlen entstehen.

Und zwei Regeln aus dieser Woche:

> **Eine Korrektur ist kein Beleg, sondern eine neue Behauptung.** Sie gehört
> genauso gegen synthetische Daten geprüft wie das Original.

> **Wird ein Fehler gefunden, wird nach seiner Klasse gesucht.** Die
> B-014-Pseudoreplikation kam aus dem Durchsuchen des ganzen Codes nach dem
> Q-011-Muster, nicht aus einer Prüfung von B-014.

> **Wer eine Datei als Vorlage benutzt, prüft sie zuerst gegen die aktuellen
> Protokollregeln.** Sie ist geschrieben worden, als die Regel noch nicht galt.
> So wurde das halbe Suchbudget in zehn Modulen gefunden — und so hatte Q-011
> den scipy-Spearman geerbt.

Und zwei Regeln aus dem Review vom 23. September:

> **Eine Kontrolle muss selbst gegen eine triviale Referenz (linear) geprüft
> werden.** Die „starke klassische Referenz“ von Q-003-replication
> (`classical_norm_deep`, 67.848 Parameter) lag in einer Stichprobe hinter
> einem rein linearen Modell. Geprüft hatte ich die Kontrolle immer nur gegen
> den Mechanismus, nie gegen etwas Einfacheres. Ein Abstand zu einer gelähmten
> Kontrolle misst deren Schwäche, nicht die Stärke des Mechanismus.

> **Korrekturen müssen im Code landen, nicht nur im Tagebuch.** Beleg: B-014.
> Die Pseudoreplikation wurde am 27. August im Tagebuch korrigiert, aber
> `intrinsic_dimension.py` rechnete weiter über 240 Einzelmessungen — wer die
> Auswertung neu gestartet hätte, hätte wieder p=3·10⁻¹⁸⁷ bekommen. Eine
> Korrektur ist erst erledigt, wenn der Code das Richtige ausgibt und `q_audit`
> bzw. `qire_ai.audit` die Zahl prüft.

Dazu, als Anwendung von Regel 6: **Eine durchgefallene Positivkontrolle
entwertet jede Gleichwertigkeitsaussage desselben Laufs** — auch dann, wenn
man eine Erklärung hat, warum sie durchfallen durfte. Genau das ist am
5. August mit „Tiefe trägt nichts bei“ passiert.

**Der Vortest ist Pflicht.** `analyse()` auf synthetischen `Recorder`-Daten
laufen lassen und **jeden** Ausgang einzeln erzwingen, auch den unerwarteten. Er
hat bisher neun Fehler gefunden, jedes Mal vor dem Lauf.

---

## 6. Was als Nächstes ansteht

**Neu, Stand 23. September:**

| | |
|---|---|
| **zuerst** | Code-Korrekturen der Überarbeitung prüfen (`../BERICHT/`), dann **Q-020 vorregistrieren** (Entwurf `ENTWURF_Q020.md`), Malte gibt frei, Vorregistrierung **vor** dem Lauf ins Tagebuch |
| **dann** | Q-020 rechnen (CPU oder GPU, siehe Entwurf) |
| **vor T-002** | die vier Fehler aus Abschnitt 7 behoben und per Probe gegengeprüft; Architekturentscheidung `ButterflyFFN` von Malte |
| **Q-019** | Weichenstellung von Malte: so laufen lassen (4,6–5,5 h) oder umbauen (Datenknappheit / Walsh-Hadamard) |
| **Q-011** | Wiederaufnahme + Gitterverlängerung in beide Richtungen, dann Tiefe 16 und Fashion nachrechnen |
| **30. Nov.** | Anmeldung mit `KURZFASSUNG_ENTWURF_v2.md` |
| **bis Mitte Jan. 2027** | schriftliche Arbeit, Literatur (`LITERATUR_UND_ALLEINSTELLUNG.md`), `B-003`-Doppelvergabe |

*Alte Tabelle, Stand 27. August (überholt: Q-011 ist abgebrochen, Q-019
braucht mit dem fünften Arm 4,6–5,5 h):*

| | |
|---|---|
| **jetzt** | Q-011 zu Ende laufen lassen, auswerten, ins Tagebuch und `q_audit`. Dabei zuerst `lr_edge_*` prüfen |
| **dann** | **Q-019 laufen lassen** — Modul steht, geprüft, ~2,5–3 h. Erst die Weichenstellung unten entscheiden |
| **bei Guthaben** | T-002 auf RunPod: `bash server_setup/runpod_t002.sh` |
| **ab Woche 12** | schriftliche Arbeit, Literaturkarten, `B-003`-Doppelvergabe |

### Q-019 ist gebaut und geprüft, aber nicht gelaufen

`quantum_sim/zeitreihen.py`. Vier Arme (`butterfly` 40 Parameter, `dense`
65.536, `fft_fest` 0, `narrow` 32.768), zwei Datenbedingungen (periodisch und
zeitlich permutiert), 15 Seeds.

    python -m quantum_sim.zeitreihen vortest     # Vortest + Gittertest, Pflicht
    python -m quantum_sim.zeitreihen saettigung  # Schwierigkeit
    python -m quantum_sim.zeitreihen macht       # Regel 2(b), fehlt noch
    python -m quantum_sim.zeitreihen             # der Lauf, ~2,5-3 h

Bestanden: Vortest (8 Szenarien × 5 Ziehungen, Marke **und** Zweig),
Gitterverlängerung in beide Richtungen (5/5), Bauteile (Fourierbasis
orthonormal 4,7·10⁻⁸, `butterfly` und `fft_fest` normerhaltend, 40 Parameter).

**Der Vorlauf hat der Erzählung die Grundlage genommen** — Einzelheiten im
Tagebucheintrag „Q-019 gebaut". Kurz: Die Kontrolle trägt spektakulär
(`butterfly` und `fft_fest` fallen auf permutierter Zeitachse auf Rateniveau,
`dense` und `narrow` nicht), aber `butterfly` (0,686) schlägt `dense` (0,913)
auf periodischen Signalen **nicht** — und die echte Fouriertransformation
(0,804) auch nicht. P2 wird also fallen, der Primärtest halten; der Lauf landet
im Zweig „HU — geteiltes Ergebnis".

~~**Die Weichenstellung, die nur Malte treffen kann:** so laufen lassen, oder
vorher einen fünften Arm `untied` (ungeteilte Butterfly, 5.120 Parameter)
ergänzen. Nur die ungeteilte Variante *kann* die FFT darstellen; sie würde
zeigen, ob die Gewichtsteilung der Preis ist. Als Arm **außerhalb** der
angemeldeten Familie geführt bleibt Regel 4 sauber. Kosten: rund eine Stunde
mehr.~~

**Nachtrag 23. September — diese Weichenstellung ist gefallen**, und zwar
schon am 27. August nachts (Tagebuch „Q-019: der fünfte Arm, und zwei Fehler in
meinem eigenen Urteilsbaum“): `untied` ist eingebaut, außerhalb der Familie,
rein beschreibend. Der Absatz oben war veraltet. Zwei Korrekturen daran:

- „Nur die ungeteilte Variante *kann* die FFT darstellen“ ist **offen**, nicht
  belegt — die Positivkontrolle von `fourier_erreichbarkeit()` hat gezeigt,
  dass das Anpassungsverfahren nicht konvergiert.
- Die Laufzeit ist mit fünf Armen **4,6 h, mit Reserve bis 5,5 h** — nicht
  2,5–3 h plus eine Stunde.

**Entwurfsschwächen aus dem Review (§4), vor dem Lauf zu bedenken:**

- Die Primärhypothese (Wechselwirkung periodisch × permutiert) ist **fast per
  Konstruktion wahr**: Eine feste Permutation ist eine orthogonale lineare
  Abbildung, die eine dichte Schicht mitlernt, eine Butterfly nicht. Das
  beweist Ordnungsabhängigkeit, keinen Vorteil.
- P2 (butterfly schlägt dense auf periodisch) fällt laut Vorlauf (0,686
  gegen 0,913).
- ⟨Z_q⟩ liefert nur 8 Zahlen aus 256; nur 3 davon tragen Bandinformation.
- Chancen auf echten Vorteil: (a) **wenig Trainingsdaten** (Lernkurve
  N=50…6.000; 40 gegen 65.536 Parameter als Induktionsbias); (b)
  **Walsh-Hadamard statt Fourier** — die geteilte Butterfly ohne CNOT ist ein
  Kronecker-Produkt von 2×2-Drehungen und enthält die Walsh-Hadamard-
  Transformation exakt; eine Aufgabe mit Rechteck-/Schaltsignalen wäre der Fall,
  in dem 8–40 Parameter die ideale Basis exakt darstellen.
- Q-019 und T-000 fehlen in `q_summary.QUESTIONS` (Regel aus `CLAUDE.md` §6).

**Die Weichenstellung, die jetzt bei Malte liegt:** Q-019 so laufen lassen
(Ergebnis „HU — geteilt“ absehbar, trotzdem ehrlich berichtbar) oder auf (a)/(b)
umbauen — dann mit neuer Vorregistrierung.

Bewusst aufgegeben: Q-010, Q-012, Q-016, Q-017 als eigenständige Läufe.

---

## 7. Infrastruktur

**RunPod: kein Guthaben.** Der Pod vom 11./12. August wurde terminiert, bevor
die Ergebnisse geholt waren — 22 Stunden Rechenzeit verloren, Ursache
wahrscheinlich leeres Konto. Rund 300 € Fördergeld sind beantragt, aber noch
nicht da.

Wenn wieder ein Pod läuft, drei Dinge sind Pflicht:

- **Netzwerk-Volume**, kein pod-gebundener Datenträger — das war die Ursache
- **RTX 4090 auf Community Cloud**, 0,34 $/h statt 0,74 $ (T-002 kostet ~10 $)
- Ergebnisse **nach jedem Seed** herunterladen

~~`quantum_sim/sprachmodell.py` nimmt einen abgebrochenen Lauf inzwischen **wieder
auf** — bereits gerechnete (Arm, Seed)-Paare werden übersprungen. Getestet.~~

**Nachtrag 23. September — T-002 hat vier Fehler, die vor jedem GPU-Lauf
behoben sein müssen** (Review §3; Korrektur in `sprachmodell.py` ist Teil der
Überarbeitung, Stand in `../BERICHT/`):

1. `LR_SUCHE_SCHRITTE = 1000`, Messlauf 13.000 (`runpod_t002.sh`) bzw. 20.000
   (CLI-Standard) → Regel 1(c) verletzt, schlimmer als das halbe Budget vom
   27. August; das Modul fehlte in der Liste der elf. Gitterverlängerung nur nach
   oben → Regel 1(b).
2. Die Wiederaufnahme liest nur die **neueste** T-002-Datei
   (`result_store.latest`). `runpod_start.sh` fährt vor jedem Lauf die Probe
   (60 Schritte) → danach ist die Probe die neueste Datei, und der Hauptlauf
   fängt **still bei null** an. Bei parallelen Cluster-Jobs bricht das Prinzip
   ganz. „Getestet“ oben galt nur für den Einzelfall ohne Probe.
3. Schrittzahl nicht vorregistriert: 13.000 gegen 20.000.
4. `trainiere_arm` speichert `"lr": LR` (Rückfallwert 3e-4) statt der benutzten
   Rate.

**Architektur-Einwand (Entscheidung Malte, vor dem Lauf ins Tagebuch):**
`ButterflyFFN` gibt `|Amplitude|² · 256 · scale` aus — nichtnegativ, feste
Summe, ein skalarer Gewinn. Der FFN-Block kann damit nichts vom Residualstrom
abziehen; ein Ergebnis T0 ist sehr wahrscheinlich (Probe: −0,32 nats nach 60
Schritten). `butterfly_qkv` lag in der Probe fast gleichauf (−0,0045 nats) und
ist der realistischere „Quanten-Chatbot“. Vorschlag: zusätzlicher Arm außerhalb
der Familie mit zentrierter Ausgabe + Gewinn je Kanal (256 Parameter).

**Rechenbedarf:** 32 Läufe (4 Arme × 8 Seeds) + Lernratensuche. Butterfly ≈
0,334 s/Schritt auf RTX 4090 bei Batch 64 → ~72 min bei 13.000 Schritten; mit
korrekter Suche ~30–35 GPU-h auf einer Karte, parallel auf 8 GPUs ein
Nachmittag. Butterfly ist auf der GPU startaufwandbegrenzt → mehr GPUs helfen,
eine schnellere kaum.

**SSH-Schlüssel** für Pods: `~/.ssh/runpod_qire` (ohne Passwort, weil die Shell
nicht interaktiv ist). Öffentlicher Teil muss in den RunPod-Kontoeinstellungen
hinterlegt sein.

**Mini-PC geplant** (Minisforum UM890 Pro, 8 Kerne, 64 GB) als Dauerläufer für
CPU-Arbeit, damit der Hauptrechner frei bleibt. Noch nicht gekauft. Achtung:
Speicherkrise 2026, DDR5 kostet das Vierfache von 2025 — fertig konfiguriert
kaufen ist derzeit billiger als Barebone plus Einzelteile.

**Korpus für T-002 liegt lokal vollständig** unter `results/korpus/`
(`train.npy` 795 MB, 397 Mio. Token). Muss nicht neu erzeugt werden.
*(Nachtrag 23. Sep.: gemeint ist `results/korpus/` im **Archivordner**; in
diesem Ordner liegen nur `bpe_4096.json` und `val.npy`, siehe
`DATEN_EXTERN.md`.)*

**Nachträge vom 23. September:**

- **Kein Git.** Für ein Projekt, dessen Wert an Vorregistrierung hängt, ist ein
  Commit-Zeitstempel *vor* den Rohdaten der stärkste Beleg. Empfehlung:
  `git init` + privates Repo (Entscheidung Malte).
- **`ibm_quantum_token.txt` liegt im Klartext im Archivordner.** Nach der
  PC-Infektion (seit Januar 2026) gelten Zugangsdaten als kompromittiert →
  Token bei IBM neu erzeugen, Datei entfernen. Das macht Malte selbst.
- **IBM-Hardware-Rohdaten (Counts) sind nirgends gespeichert**, nur die
  Zusammenfassung in `demo/artifacts/hardware_evidence.json`. Über die Job-IDs
  herunterladen, solange IBM sie hält.
- **Speicher:** Die Systemplatte hat wieder **141 GB frei** (frühere Angabe
  „99 % voll, 29 GB frei“ ist überholt).
- **Standdemo** liegt jetzt unter `demo/` in diesem Ordner (vorher nur im
  Archiv).
- RunPod-Guthaben und Fördergeld: Stand 27. August; aktuellen Stand bei Malte
  erfragen.

---

## 8. Entscheidungen, die nur Malte treffen kann

*(Bis 27. August drei, seit dem Review vom 23. September neun.)*

- **Q-020 freigeben** — Arme, Seeds, Schwellen, Rechner; Vorregistrierung vor
  dem Lauf ins Tagebuch
- **T-002-Architektur** — `ButterflyFFN` so lassen oder zusätzlicher zentrierter
  Arm außerhalb der Familie
- **Q-019** — so laufen lassen oder umbauen (Datenknappheit / Walsh-Hadamard)
- **Git** — `git init` + privates Repo
- **IBM-Token** neu erzeugen und Klartextdatei im Archiv entfernen
- **Aufteilung A/B für die Anmeldung** — bis zu drei Projekte erlaubt
- **`B-003` ist doppelt vergeben** — DMRG-Rangzuweisung und Blockschicht-
  Vergleich. Die Rohdaten gehören dem zweiten (im B-Handoff ist der DMRG-Lauf
  bereits als B-015 geführt)
- **Echter Lesestand der Literatur** — die Karten sind pauschal als ungelesen
  markiert, tatsächlich sind rund drei Viertel gelesen
- **Frist der schriftlichen Arbeit** beim eigenen Landeswettbewerb prüfen

---

## 9. Umgangston

Deutschsprachig, auch im Code und in den Kommentaren. Das Projekt lebt von
Ehrlichkeit gegenüber den eigenen Zahlen: Zurückgenommene Aussagen bleiben mit
Begründung stehen, statt gelöscht zu werden. Ein durchgefallener Test ist ein
Ergebnis, kein Rückschlag. Wer nur die erwarteten Ausgänge kodiert, verliert den
interessanten.
