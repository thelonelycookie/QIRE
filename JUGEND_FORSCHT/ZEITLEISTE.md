# Zeitleiste bis zum Bundesfinale (R4.3)

*Stand 14.10.2026 (Mittwoch). Alle Daten mit „zu prüfen“ sind unbestätigt. Quellen: `CLAUDE.md` (Anmeldung 30.11.2026, Arbeit Mitte Januar 2027), `UEBERARBEITUNG_2026-09-23/BERICHT/RECHERCHE_2026-09-28/04_jugend_forscht_2027.md` und `07_schriftliche_arbeit_jury.md` (kurz „Recherche 04/07“), `INSTITUT/HANDOVER_2026-10-06.md` §6, `IDEEN_KI/PLAN_2026-10-13.md`. Wochentage und Tagesabstände sind gerechnet.*

## 1. Feste und unsichere Eckdaten
| Datum | Ereignis | Sicherheit | Quelle |
|---|---|---|---|
| Mi 14.10.2026 | heute (Tag 0) | – | – |
| **Mo 30.11.2026** | Anmeldeschluss Runde 2027 (Tag 47) | Jahresregel gesichert, Datum nicht wörtlich bestätigt | Recherche 04 §1, Recherche 07 §3 |
| Mitte Januar 2027 | Abgabe schriftliche Arbeit als PDF über das Online-System (Jufo WV) | **zu prüfen.** Offizielle FAQ: „immer im Januar“. Vorjahresmuster Sachsen-Anhalt: Mi 15.01.2025, 23:59 Uhr. Wettbewerbsleitung meldet den Termin laut Landesseite bis Ende Dezember | Recherche 07 §3 |
| Februar 2027 (Annahme) | Regionalwettbewerb (Halle oder Stendal/Greppin) | **zu prüfen.** 2026 war Halle am 17.02.2026; Termin 2027 nicht veröffentlicht | Recherche 04 §1 |
| April 2027 (Annahme) | Landeswettbewerb Sachsen-Anhalt | **zu prüfen.** 2026 war 09.–10.04.2026; 2027 nicht veröffentlicht | Recherche 04 §1 |
| **Do 27. bis So 30.05.2027** | Bundesfinale Würzburg (Tag 225 bis 228) | mittel (Suchsnippet Jugend forscht BW) | Recherche 07 §3; Chat-Durchsicht (HANDOVER §6) |

**Zu beobachten:** Die drei Termin-Seiten aus Recherche 04 (Regionalwettbewerb Halle, Landeswettbewerb Sachsen-Anhalt, Seite „schriftliche Arbeit“) ab sofort alle zwei Wochen, im Dezember wöchentlich.

## 2. Planungsprinzip
Rückwärts vom Abgabetermin, mit je einer Woche Puffer vor den zwei harten Fristen:
- Anmeldung: intern fertig **Mo 23.11.2026** (Puffer 7 Tage).
- Arbeit: intern fertig **Fr 08.01.2027**; früheste plausible Abgabe 10./15.01.2027 (Puffer 7 Tage plus Reserve, falls der Termin eher am 10.01. liegt: **zu prüfen**).
- Schreibzeit: Leitfaden empfiehlt mindestens 4 Wochen, Auswertung plus Schreiben mindestens die Hälfte des Gesamtaufwands (Recherche 07 §1.1); Schreiben beginnt spätestens Anfang Dezember (Recherche 07 §3). Hier: Beginn der Schreibphase **Mo 16.11.2026**, also 7+ Wochen.
- Regel: **Rechenstopp für die Arbeit am Sa 12.12.2026.** Spätere Läufe sind nur Zusatz am Stand (Leitfaden erlaubt neue Ergebnisse zusätzlich am Stand, Recherche 07 §1.1).
- Alle neuen vorregistrierten Läufe nur mit Maltes Freigabe (`CLAUDE.md`).

## 3. Phasen und Meilensteine

### Phase A: Grundlage und Entscheidungen (14.10. bis 25.10.)
| Datum | Meilenstein | Verantwortlich |
|---|---|---|
| bis Fr 16.10. | Malte beantwortet die Entscheidungen aus `IDEEN_KI/PLAN_2026-10-13.md` §10 (Plan, Merge, Archiv, Veröffentlichung jetzt oder später) | Malte |
| bis Mo 19.10. | T-006-Auswertung (Rohdaten Kern liegen laut `KERNAUSSAGE` §4 vor; Freigabe nötig, falls Rechnung) | Opus plant, Sonnet rechnet |
| bis So 25.10. | Fachgebiet mit Wettbewerbsleitung abstimmen (`GLIEDERUNG.md` §2), Titel wählen (3 Varianten, `KERNAUSSAGE` §5) | Malte |
| bis So 25.10. | R0 aus dem Plan (Branches nach `main`, `START.md`, Verschlanken): Voraussetzung für die öffentliche Version | Sonnet-Agenten, Opus prüft |

### Phase B: letzte Rechnungen (26.10. bis 12.12.)
| Zeitraum | Inhalt | Hinweis |
|---|---|---|
| 26.10. bis 15.11. | Kandidaten (je mit Freigabe): I-043 (Fastfood in FFN/Adapter, PC), Replikation von I-045-v2 in größerem Modell, fairer Bitvergleich, T-ternaer-mehr (20 Seeds), budgetgleiche I-046-Front | alle aus `KERNAUSSAGE` §4 und `agenda.json`; Dauer je Lauf **zu prüfen** (Laufzeiten stehen im jeweiligen Laborbuch-Eintrag) |
| 16.11. bis 12.12. | nur noch Läufe, die ein bereits geschriebenes Kapitel schließen | Rechenstopp 12.12. |
| MPI-Zugang | Details offen (HANDOVER §6); falls er vor dem 12.12. kommt, höchstens ein einzelner Skalierungslauf | **zu prüfen** |

### Phase C: Anmeldung (bis 23.11. intern, 30.11. hart)
| Datum | Meilenstein |
|---|---|
| bis Fr 06.11. | Anmeldedaten sammeln: Titel, Fachgebiet, Kurzbeschreibung (genaue Pflichtfelder **zu prüfen** auf der Anmeldeseite) |
| bis Mo 16.11. | Entwurf Kurzbeschreibung aus `KERNAUSSAGE` §1, ehrlich „quanten-inspiriert“ |
| **Mo 23.11.** | Anmeldung abgeschickt (Puffer bis 30.11.) |
| Mo 30.11. | harte Frist |

### Phase D: Schreibphase (16.11. bis 08.01.)
| Datum | Meilenstein |
|---|---|
| ab Mo 16.11. | Kapitel 1 bis 4 (Fragestellung, Grundlagen, Literatur, Methode) schreiben; hängen nicht von neuen Läufen ab |
| bis So 29.11. | Abbildungen aus `INSTITUT/labor/abbildungen.py` (R2.2, ein anderer Agent baut es); erster PDF-Build mit Größenmessung, Ziel < 5 MB |
| 30.11. bis 12.12. | Ergebnis-Kapitel 5 bis 10 (siehe `GLIEDERUNG.md` §4) |
| Sa 12.12. | **Entscheidung Zusatzkapitel Ideen-KI** und Rechenstopp |
| bis So 20.12. | Vollentwurf (15 Seiten gezählt, Seitenbudget aus `GLIEDERUNG.md` §3) |
| 21.12. bis So 03.01. | Korrektur und Fremdlesen (z. B. Lehrkraft, Eltern; Weihnachten/Ferientermine **zu prüfen**); Zahlen-Audit: jede Zahl gegen Ergebnis-JSON (R3.6 im Plan) |
| Mo 04.01. bis Mi 06.01. | PDF-Endbau, Größe < 5 MB prüfen (Limit 5 MB oder 30 MB widersprüchlich, im Portal **zu prüfen**), Formalia-Check (Rand, Schrift, 15 Seiten, kein Anhang, keine Weblinks im Text) |
| **Fr 08.01.2027** | Arbeit intern fertig (Puffer bis zur Abgabe) |
| ca. 10.–15.01.2027 | Abgabe, Datum **zu prüfen**; KI-Nutzung als Unterstützungsleistung im Portal angeben (Recherche 04 §2) |

### Optionales Zusatzkapitel: Ideen-KI-Befund
- Voraussetzung: Maltes Freigabe des Plans, Stufe S1 abgenommen (Stopp < 5 min gemessen, 1 Nacht ohne Absturz, ≥ 10 Karten, Malte bewertet 20 Ideen blind; `IDEEN_KI/PLAN_2026-10-13.md` §6).
- Messgrößen laut Plan: Trefferquote (Anteil H1 unter getesteten KI-Ideen gegen Malte/Claude), Blindnote, Neuheitsquote, Ideen/Stunde.
- Zeitfenster: Bau und M0 nach R0, Betriebsnächte Ende Oktober bis Mitte Dezember; Entscheidung am **12.12.**. Reicht die Datenlage nicht (Trefferquote braucht getestete Ideen, die selbst Läufe brauchen), bleibt es bei einem Satz im Ausblick. Zahlen dafür gibt es heute nicht (`IDEEN_KI/` enthält nur den Plan).

### Veröffentlichung (siehe `VEROEFFENTLICHUNG.md`)
Vorbereiten (Aufräumen, Lizenzen, `CITATION.cff`) im Dezember, **Veröffentlichung und DOI erst nach Maltes ausdrücklicher Zustimmung**, Vorschlag: nach Abgabe der Arbeit (ca. 15.–31.01.2027), spätestens vor dem Landeswettbewerb.

## 4. Nach der Abgabe
| Zeitraum | Meilenstein |
|---|---|
| Mitte Januar bis Regionalwettbewerb | Kurzpräsentation (max. 10 Minuten, Recherche 07 §4) schreiben, Jury-Übung mit `JURY_UEBUNG.md` (27 plus 15 Fragen) und Lehrkraft als Probejury |
| vor dem Regionalwettbewerb (Termin zu prüfen) | Stand: Poster, Laptop mit Dashboard (`dashboard/dashboard.html`), ausgedruckte Versuchsdokumentation (laut Leitfaden am Stand erlaubt), 5-Minuten-Version bereit |
| Regionalwettbewerb | **zu prüfen** (Annahme Februar) |
| Landeswettbewerb | **zu prüfen** (Annahme April) |
| bis Mitte Mai | neue Befunde seit Abgabe aufnehmen (T-006, I-043, MPI) als Zusatzblatt am Stand; Poster/Slides aktualisieren |
| **27.–30.05.2027** | Bundesfinale Würzburg |

## 5. Risiken und Puffer
| Risiko | Gegenmaßnahme |
|---|---|
| Abgabe liegt früher als 15.01. | interner Stichtag 08.01.; ab 02.12. wöchentlich Landesseite prüfen |
| Nutzungslimit knapp (CLAUDE.md, Handover) | Rechenstopp 12.12.; Schreiben unabhängig von Rechnung beginnen |
| Läufe brechen ab (I-036 ging im Container verloren, `UEBERGABE` §0) | Wiederaufnahme je Seed, Zwischenstände committen |
| Seitenbudget 15 Seiten reicht nicht | Magie-/Clifford-Linie nur als Absatz; Rohdaten an den Stand |
| PDF zu groß | Abbildungen als Vektor-PDF oder 150 bis 200 dpi, erster Build bis 29.11. |
| Unklare Anmeldeformalitäten | Wettbewerbsleitung Halle früh kontaktieren (Kontakt Recherche 04) |
| Neuheitsprüfung nur aus Suchauszügen | im November Volltext der 3 engsten Arbeiten lesen (`KERNAUSSAGE` §4) |
