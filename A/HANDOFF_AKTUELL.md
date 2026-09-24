# Übergabe — Stand 30. August 2026

Kalt lesbar. Ersetzt `HANDOFF_PROJEKT_A.md` (6. August) als Einstieg; jenes
bleibt für den Verlauf bis dahin gültig.

*Abschnitt 0 am 30. August nachgeführt: Q-011 ist abgebrochen, nicht mehr am
Rechnen. Der Rest des Dokuments ist Stand 27. August, abends.*

---

## 0. Das Erste: Q-011 ist im dritten Anlauf abgebrochen

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

    .venv/Scripts/python.exe -m quantum_sim.q_summary    # Belegstand Q/T-Serie
    .venv/Scripts/python.exe -m quantum_sim.q_audit      # 106 Zahlen gegen Rohdaten
    .venv/Scripts/python.exe -m qire_ai.audit            # dasselbe für die B-Serie

Dann lesen, in dieser Reihenfolge:

1. **`STAND_UND_OFFENES.md`** — Stichpunktfassung beider Teilprojekte, bester Überblick
2. **`PROJEKTTAGEBUCH.md`**, die sechs Einträge vom **27. August** (ganz unten vor „Zwischenstand")
3. **`ENTWURF_T_SERIE_KERNEL.md`** — Roadmap bis zum 30. November
4. **`ENTWURF_ZEITREIHEN.md`** — nächster geplanter Lauf, noch nicht gebaut

---

## 2. Ziel und Termin

**Ressourceneffizientere KI mithilfe von Quantenphysik.** Bis zum
**30. November** sollen ein **Chatbot** und eine **zweite KI-Art** stehen. Das
Ziel ist gesetzt und wird nicht neu verhandelt.

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

**Projekt A gilt in Teilen.** 16–80 Schaltungsparameter schlagen 67.848
(+0,0932 / +0,0599). Verschränkung und Tiefe tragen nichts bei (TOST). Q-015:
die Schaltung schlägt die volle globale Drehung bei 408-fach weniger
Freiheitsgraden — Trainierbarkeit, nicht Ausdruckskraft. Q-018: das Training der
Winkel trägt +0,4077 bei, sobald kein Encoder aushilft.

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

**Der Vortest ist Pflicht.** `analyse()` auf synthetischen `Recorder`-Daten
laufen lassen und **jeden** Ausgang einzeln erzwingen, auch den unerwarteten. Er
hat bisher neun Fehler gefunden, jedes Mal vor dem Lauf.

---

## 6. Was als Nächstes ansteht

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

**Die Weichenstellung, die nur Malte treffen kann:** so laufen lassen, oder
vorher einen fünften Arm `untied` (ungeteilte Butterfly, 5.120 Parameter)
ergänzen. Nur die ungeteilte Variante *kann* die FFT darstellen; sie würde
zeigen, ob die Gewichtsteilung der Preis ist. Als Arm **außerhalb** der
angemeldeten Familie geführt bleibt Regel 4 sauber. Kosten: rund eine Stunde
mehr.

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

`quantum_sim/sprachmodell.py` nimmt einen abgebrochenen Lauf inzwischen **wieder
auf** — bereits gerechnete (Arm, Seed)-Paare werden übersprungen. Getestet.

**SSH-Schlüssel** für Pods: `~/.ssh/runpod_qire` (ohne Passwort, weil die Shell
nicht interaktiv ist). Öffentlicher Teil muss in den RunPod-Kontoeinstellungen
hinterlegt sein.

**Mini-PC geplant** (Minisforum UM890 Pro, 8 Kerne, 64 GB) als Dauerläufer für
CPU-Arbeit, damit der Hauptrechner frei bleibt. Noch nicht gekauft. Achtung:
Speicherkrise 2026, DDR5 kostet das Vierfache von 2025 — fertig konfiguriert
kaufen ist derzeit billiger als Barebone plus Einzelteile.

**Korpus für T-002 liegt lokal vollständig** unter `results/korpus/`
(`train.npy` 795 MB, 397 Mio. Token). Muss nicht neu erzeugt werden.

---

## 8. Drei Entscheidungen, die nur Malte treffen kann

- **Aufteilung A/B für die Anmeldung** — bis zu drei Projekte erlaubt
- **`B-003` ist doppelt vergeben** — DMRG-Rangzuweisung und Blockschicht-
  Vergleich. Die Rohdaten gehören dem zweiten
- **Echter Lesestand der Literatur** — die Karten sind pauschal als ungelesen
  markiert, tatsächlich sind rund drei Viertel gelesen

---

## 9. Umgangston

Deutschsprachig, auch im Code und in den Kommentaren. Das Projekt lebt von
Ehrlichkeit gegenüber den eigenen Zahlen: Zurückgenommene Aussagen bleiben mit
Begründung stehen, statt gelöscht zu werden. Ein durchgefallener Test ist ein
Ergebnis, kein Rückschlag. Wer nur die erwarteten Ausgänge kodiert, verliert den
interessanten.
