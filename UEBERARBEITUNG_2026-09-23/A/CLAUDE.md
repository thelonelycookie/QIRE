# Projekt A — QIRE (Jugend forscht)

Dieser Ordner ist **vollständig und eigenständig**. Alles, was Projekt A
braucht, liegt hier: Dokumente, Code, Rohdaten, Serverskripte. Wer diesen
Ordner zugewiesen bekommt, braucht nichts weiter — außer dem Python-Interpreter
(Abschnitt 3) und, für einen T-002-Lauf, dem großen Korpus (`DATEN_EXTERN.md`).

Alles ist auf Deutsch: Code, Kommentare, Dokumente, Variablennamen. Das bitte
beibehalten.

---

## 1. Ziel und Termin

**Ressourceneffizientere KI mithilfe von Quantenphysik.** Bis zum
**30. November 2026** sollen ein **Chatbot** und eine **zweite KI-Art** stehen.
Das Ziel ist gesetzt und wird nicht neu verhandelt. Malte macht das Projekt
allein.

**Fristen:** Anmeldung **30. November 2026**. Schriftliche Arbeit **Mitte
Januar 2027** (eine Landesseite nennt den 17.01.2027 — beim eigenen
Landeswettbewerb prüfen). GPU-Ergebnisse (Q-020, T-002) müssen erst im Januar
in der Arbeit stehen, nicht in der Anmeldung.

Der zentrale Konflikt, mit Zahlen: Die Struktur gewinnt ab **16.384**
Dimensionen, trainierbar ist auf dieser Hardware **256**. Faktor 64. Deshalb
wird **Qualität bei trainierbarer Breite** gemessen und **Effizienz bei ihrer
eigenen Breite**, und beides **getrennt berichtet**. „Der Chatbot läuft dank der
Struktur schneller" wäre falsch und darf nicht behauptet werden.

**Projekt B ist abgeschlossen** (strukturierte Schichten ersetzen keine dichte;
B-014 erklärt warum). Es liegt in `../B/` und wird hier nur zitiert.

---

## 2. Womit du anfängst

```bash
"$VENV" -m quantum_sim.q_summary    # Belegstand der Q/T-Serie
"$VENV" -m quantum_sim.q_audit      # Zahlen gegen die Rohdaten nachrechnen
```

(`"$VENV"` = der Interpreter aus Abschnitt 3; unter Windows kürzer `.\py.cmd`.)

Dann lesen, in dieser Reihenfolge:

1. **`../../INSTITUT/HANDOVER_*.md`** (das neueste) — aktueller Gesamtstand
   beider Linien (Institut und Projekt A), laufende Läufe, offene Entscheidungen.
2. **`../../INSTITUT/UEBERGABE.md` §0–3** — Maltes Vorgaben und Protokollregeln.
3. **`../../INSTITUT/PLAN_*_gesamt.md`** (das neueste) — der gültige Plan.
4. **`HANDOFF_AKTUELL.md`** — Einstieg Projekt A (Chatbot-Adapter, Q/T-Serie).
5. **`PROJEKTTAGEBUCH.md`** — maßgeblich für Projekt A, jede Zahl mit Herleitung.
6. Vorregistrierungen `ENTWURF_*.md` nur bei Bedarf (z. B. `ENTWURF_T006_SKALIERUNG.md`).

`BEFUNDE_UND_RICHTUNGEN.md` ist der schnellste Überblick, wenn wenig Zeit ist:
~60 Stichworteinträge über Befunde, Theorie, Läufe und Fehlerquellen.
Historisch (nicht mehr als Stand lesen): `ROADMAP.md`, `HANDOFF_PROJEKT_A.md`,
`FORSCHUNGSIDEEN.md`, `KURZFASSUNG_ENTWURF.md`, `RESULTS.md`,
`SCIENTIFIC_LIMITS.md`.

Den aktuellen Stand nie hier eintragen (er veraltet), sondern im neuesten
`INSTITUT/HANDOVER_*.md`.

---

## 3. Umgebung

Die virtuelle Umgebung liegt **außerhalb** dieses Ordners (1,3 GB):

```
C:/Users/Malte/Downloads/qire_ai_console_v1_0_remote_research_platform/qire_ai_console_v1_0/.venv/Scripts/python.exe
```

Python 3.11.9. Aus diesem Ordner heraus aufgerufen importiert sie den Code von
**hier** — alle Pfade lösen über `Path(__file__).parent.parent` auf, nicht über
das Arbeitsverzeichnis. Nachprüfbar mit `python -c "import quantum_sim;
print(quantum_sim.__file__)"`. `UMGEBUNG.md` erklärt, wie die Umgebung neu
gebaut wird, falls der alte Ordner verschwindet.

**Hardware: Ryzen 5 3600, 6 physische Kerne, 16 GB.** Immer nur **ein**
schwerer Lauf gleichzeitig. Ein Lauf über Nacht überlebt das Sitzungsende
nicht zuverlässig — genau daran ist Q-011 dreimal gescheitert.

**Geld (RunPod, Hardware-Zugänge) und große Downloads nur nach Rückfrage bei
Malte.** Skripte dafür liegen in `server_setup/`. **Vor jedem T-002-Lauf**
müssen die vier Fehler aus `HANDOFF_AKTUELL.md` Abschnitt 7 behoben sein
(Lernratensuche mit zu kleinem Budget, Wiederaufnahme liest die Probe statt des
Hauptlaufs, Schrittzahl nicht festgelegt, falsche Lernrate gespeichert).

Git: Repo `thelonelycookie/QIRE` auf GitHub. Vorregistrierungen werden vor
den Rohdaten committet (Zeitstempel).

---

## 4. Die Protokollregeln

Vollständig mit Begründung in `HANDOFF_AKTUELL.md` Abschnitt 5. Kurz:

1. **Lernrate je Arm**, eigene Tuning-Seeds, Gitter muss das Optimum
   *enthalten* und in **beide** Richtungen verlängern — und die Suche muss
   **dasselbe Budget** haben wie der Messlauf.
2. **Genug Seeds** — ist Signifikanz rechnerisch erreichbar, und ist der
   erwartete Effekt bei der vorhandenen Streuung nachweisbar?
3. **„nicht signifikant" ist nicht „gleich"** — TOST.
4. **Holm nur über die angemeldete Familie.**
5. **Rohdaten nach jedem Abschnitt sichern**, nicht am Ende.
6. **Positivkontrolle einbauen** und prüfen, dass sie feuern *kann*.
7. **Kriterien vor dem Lauf ins Skript.**

Dazu drei Regeln aus Fehlern dieses Sommers:

> **Eine Korrektur ist kein Beleg, sondern eine neue Behauptung.** Sie gehört
> genauso gegen synthetische Daten geprüft wie das Original.

> **Wird ein Fehler gefunden, wird nach seiner Klasse gesucht** — nicht nur die
> eine Stelle repariert.

> **Wer eine Datei als Vorlage benutzt, prüft sie zuerst gegen die aktuellen
> Protokollregeln.** Sie ist geschrieben worden, als die Regel noch nicht galt.

Und zwei Regeln aus dem Review vom 23. September:

> **Eine Kontrolle muss selbst gegen eine triviale Referenz (linear) geprüft
> werden.** Die 67.848-Parameter-Kontrolle von Q-003-replication lag in einer
> Stichprobe hinter einem rein linearen Modell; der Abstand +0,0932 maß vor
> allem ihre Schwäche. Praktisch: ein linearer Vergleichsarm im selben Lauf
> oder ein Vorlauf, der die Kontrolle gegen linear stellt — vor der
> Vorregistrierung.

> **Korrekturen müssen im Code landen, nicht nur im Tagebuch.** Beleg: B-014 —
> die Pseudoreplikation war am 27. August im Tagebuch korrigiert, aber
> `intrinsic_dimension.py` rechnete weiter über 240 Einzelmessungen. Eine
> Korrektur ist erst erledigt, wenn der Code das Richtige ausgibt und ein
> Prüfwerkzeug die Zahl nachrechnet.

Und als Anwendung von Regel 6: **Fällt die Positivkontrolle durch, steht
keine Gleichwertigkeitsaussage aus demselben Lauf** — auch nicht mit einer
nachgeschobenen Erklärung, warum sie durchfallen durfte (so geschehen mit
„Tiefe trägt nichts bei“ am 5. August).

---

## 5. Der Vortest ist Pflicht

**Vor jedem mehrstündigen Lauf:** `analyse()` des Moduls auf synthetischen
`Recorder`-Daten laufen lassen und **jeden** vorregistrierten Ausgang einzeln
erzwingen, auch den unerwarteten, plus eine fehlschlagende Auflösungsprüfung.

Er hat bisher **neun** Fehler gefunden, jedes Mal vor dem Lauf. Ohne ihn wären
mehrfach zweistellige Stundenzahlen verbrannt worden.

Vier Dinge, die dabei schon schiefgingen und deshalb mitzuprüfen sind:

- Die synthetischen Daten so bauen, dass die beiden Kandidatenmetriken sich
  **nicht konstant** unterscheiden — sonst zeigt kein gepaarter Test, welches
  Feld benutzt wurde.
- **Jedes Szenario über ~5 Rauschziehungen** laufen lassen. Ein Urteil, das mit
  der Ziehung kippt, ist kein Urteil.
- Auf den **Verzweigungstext** prüfen, nicht nur auf die Marke H1/H0/HU —
  mehrere Ausgänge teilen sich eine Marke.
- Der Datengenerator selbst kann der Fehler sein. Bei Q-019 bekam der
  Basisarm kein eigenes Rauschen; der gemeinsame Seed-Anteil kürzt sich in der
  gepaarten Differenz weg.

---

## 6. Wenn eine Zahl entsteht

Gilt für **Q/T-Läufe von Projekt A**. Institutsläufe (I-…, C-…) gehen
stattdessen ins `INSTITUT/LABORBUCH.md` und `INSTITUT/agenda.json`.

Sie gehört **sofort** an drei Stellen:

1. ins `PROJEKTTAGEBUCH.md`, mit Herleitung,
2. in `quantum_sim/q_audit.py`, damit sie gegen die Rohdaten nachgerechnet wird,
3. in `quantum_sim/q_summary.QUESTIONS`, sonst taucht der fertige Lauf im
   Bericht gar nicht auf. Genau das ist mit Q-015 passiert.

Änderungen an einer Vorregistrierung gehören ins Tagebuch, **bevor** Zahlen
entstehen — sonst sind sie von nachträglichem Tuning nicht zu unterscheiden.

---

## 7. Arbeitsweise

- **Opus plant, Sonnet-Agenten führen aus** (Recherche, Modulbau, Rechnungen);
  Opus nur, wo Sonnet nicht reicht. **Vor jedem neuen vorregistrierten Lauf
  auf Maltes Zustimmung warten.**
- **Selbstständig arbeiten, nur bei wichtigen Entscheidungen fragen.** Eine
  wichtige Entscheidung ist: Richtung des Projekts, Geld, ein Lauf über viele
  Stunden, das Verwerfen eines Befunds. Nicht: welcher Dateiname, welche
  Bibliothek, wie ein Test aufgebaut wird.
- **Nichts behaupten, was nicht in den Rohdaten steht.** `q_audit` ist die
  Instanz, nicht das Gedächtnis.
- Ein durchgefallener Auflösungstest ist **kein** Nullbefund. Er heißt, dass
  der Aufbau nichts hätte finden können.
- Malte trifft Richtungsentscheidungen auch gegen meine Empfehlung (deutscher
  Korpus, beide Butterfly-Platzierungen). Das ist in Ordnung und wird umgesetzt,
  nicht neu aufgerollt.

---

## 8. Was hier bewusst **nicht** liegt

Der 2,9-GB-Korpus und die Projekt-B-Rohdaten. Wo sie liegen und wie sie sich
neu erzeugen lassen: **`DATEN_EXTERN.md`**. Bitte den Archivordner nicht
löschen, ohne das gelesen zu haben.
