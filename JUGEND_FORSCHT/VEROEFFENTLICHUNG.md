# Öffentliche Version des Repos: Plan (R4.4)

*Stand 14.10.2026. **Nur Planung. Nichts davon wird ohne Maltes ausdrückliche Zustimmung ausgeführt** (`IDEEN_KI/PLAN_2026-10-13.md` §8 R4.4 und Entscheidung 7 in §10). Es wurde kein Repo angelegt, nichts veröffentlicht, kein Push, kein Commit. Rechtliche Angaben (Lizenzen, Zenodo-Verfahren) stammen aus allgemeinem Wissen und sind alle **zu prüfen**, bevor Malte sie in einer Veröffentlichung verwendet.*

## 1. Ziel
Ein aufgeräumtes, zitierbares Repo (Code, Rohdaten der Läufe, Laborbuch, Arbeit) mit DOI über Zenodo. Es soll beim Jurygespräch ausgelegt/verlinkt werden können (Recherche 07 §2: Code und Protokolle gehören an den Stand und ins öffentliche Repo, nicht in die Arbeit), und es soll Maltes Namen tragen (UE §2 Nr. 5, `INSTITUT/UEBERGABE.md`).

## 2. Ist-Zustand des privaten Repos (gemessen am 14.10.2026)
| Messgröße | Wert | Befehl/Quelle |
|---|---|---|
| Remote | `github.com/thelonelycookie/QIRE` (privat/öffentlich **zu prüfen**) | `git remote -v` |
| Dateien unter Versionskontrolle | 1.472 | `git ls-files \| wc -l` |
| Arbeitsverzeichnis | 54 MB, davon `.git` 18 MB | `du -sh` |
| `INSTITUT/ergebnisse/` | 16 MB; 188 getrackte `_zwischen_*`-Dateien, zusammen etwa 1,5 MB | `du`, `git ls-files` |
| `UEBERARBEITUNG_2026-09-23/` | 24 MB | `du -sh` |
| getrackte `.pt`-Modelldateien | 73 Stück, etwa 4,5 MB | `git ls-files` |
| getrackte `.log`-Dateien | 71 | `git ls-files` |
| doppelte `val.npy` | je 3,9 MB in `A/results/korpus/` und `UEBERARBEITUNG_2026-09-23/A/results/korpus/` (Plan R0.3) | `git ls-files` |
| größte Ergebnis-JSON | ca. 3,9 MB (`I-034-fashion__20260925_075431.json`) | `du` |
| Commit-Autoren | `Claude <noreply@anthropic.com>` 141 Commits, `Malte Liebscher <liebschermalte@gmail.com>` 42, `Malte <liebschermalte@gmail.com>` 7, `thelonelycookie <liebschermalte@gmail.com>` 32, `Malte Liebscher <noreply@anthropic.com>` 2 | `git log --format` |
| Commit-Nachrichten mit `Co-Authored-By` | 134 | `git log` |
| `LICENSE`-Datei | fehlt | `ls` |

Die Größe ist kein Hindernis (alles weit unter Zenodos Grenzen, **zu prüfen**). Das Problem sind Inhalt und Historie.

## 3. Was raus muss oder geprüft werden
| Punkt | Befund | Maßnahme |
|---|---|---|
| **Zugangsdaten** | `.gitignore` schließt `*token*.txt`, `.env`, `*.pem` aus. Der IBM-Token wird aus Umgebungsvariable oder Token-Datei gelesen (`A/quantum_sim/qiskit_bridge.py` `_load_token`). `git ls-files` listet keine Datei mit „token“/„.env“/„credentials“. Das reicht nicht als Beweis. | vor Veröffentlichung Geheimnis-Scan über **gesamte Historie** (z. B. gitleaks/trufflehog; GitHub Secret Scanning, `mcp__github__run_secret_scanning` steht in der Werkzeugliste); Treffer-Token sofort widerrufen, falls je eingecheckt; Ergebnis dokumentieren |
| **Private Chats** | `INSTITUT/recherche/2026-10-06_chat_sweep/` (A.md, B.md, C.md, 00_ZUSAMMENFASSUNG.md) sind Auszüge aus Chat-Sitzungen; `INSTITUT/HANDOVER_2026-10-06.md` und `chat_sweep/A.md` nennen Session-IDs und PC-interne Details (Prozess-IDs, Ryzen/16-GB-PC). | aus der öffentlichen Version entfernen oder auf Zusammenfassung reduzieren; Malte entscheidet |
| **Persönliche Kontakte Dritter** | `UEBERARBEITUNG_2026-09-23/BERICHT/RECHERCHE_2026-09-28/04_jugend_forscht_2027.md` enthält E-Mail und Telefonnummer der Wettbewerbsleitung, Namen von Ansprechpersonen. Weitere E-Mail-artige Muster in 4 `zwischenstand/*.py`-Dateien (vermutlich harmlos, **prüfen**). | Kontaktdaten schwärzen oder Datei weglassen |
| **MPI-Treffen** | `UEBERARBEITUNG_2026-09-23/MPI_TREFFEN/` (Gesprächsleitfaden, Cluster-Anleitung, Skripte `t002_*.sh/sbatch`) betrifft Dritte und evtl. Zugangsdetails | nicht veröffentlichen, solange MPI nicht zugestimmt hat; **zu prüfen** |
| **Werkstatt / Betrieb** | `WERKSTATT/` (Agentenrollen, Ruhezeit, `worker-settings.json`, PowerShell-Skripte), `institut.cmd`, `.claude-flow/` | weglassen; interner Betrieb, nicht Teil der Wissenschaft |
| **Rohdaten-Größe** | `_zwischen_*` (Wiederaufnahmezustand), `.log`, doppelte `val.npy`, `.pt`-Checkpoints, Datenarchive (`INSTITUT/daten/` ist bereits gitignored) | `_zwischen_*` und `.log` nicht übernehmen; `.pt` nur die für `demo` nötigen; Ergebnis-JSONs behalten (sie sind die Belege für jede Zahl) |
| **Historie mit KI-Autorenzeile** | 141 Commits von „Claude“, 134 mit `Co-Authored-By`; `CLAUDE.md` verlangt Commits unter Maltes Namen und „KI im Repo-Inhalt nicht erwähnen“, ältere Commits bleiben unverändert (HANDOVER §6). Zugleich verlangt Jugend forscht, KI-Nutzung als Unterstützungsleistung offenzulegen (Recherche 04 §2), und das Repo erwähnt Opus/Sonnet-Agenten an vielen Stellen. | **Entscheidung Malte nötig**, siehe §4 |
| **E-Mail in Commit-Metadaten** | die private Adresse steht in allen Autorenzeilen | bei einer Veröffentlichung der Historie sichtbar; bei Snapshot-Variante nicht (dort GitHub-noreply-Adresse nutzen) |
| **Modell-/Datenlizenzen** | LLäMmlein-120M: RAIL-M, research-only; „öffentlich gehosteter Chat wäre Distribution (Lizenz vorher prüfen)“ (`chat_sweep/C.md`) | keine Modellgewichte hochladen; nur Verweis |
| **Dritttexte** | Literatur-Auszüge in `recherche/` (Zitate aus PDFs) | nur kurze Zitate mit Quelle; keine Volltexte |

## 4. Entscheidung: Historie oder Snapshot
| | A: Historie veröffentlichen | B: frisches Repo mit Snapshot (empfohlen) |
|---|---|---|
| Vorregistrierungsbeweis (Commit vor Zahlen) | bleibt als Git-Historie erhalten | geht verloren, außer Zeitstempel im Laborbuch und ein Archiv der alten Historie (privat) wird auf Anfrage gezeigt |
| Autorenzeilen „Claude“ | öffentlich | nicht sichtbar |
| private E-Mail in Metadaten | öffentlich | nicht sichtbar |
| Geheimnisrisiko | hoch (ganze Historie scannen, ggf. Rewrite) | niedrig |
| Aufwand | hoch | niedrig |

Empfehlung: **B** für das öffentliche Repo, und die Vorregistrierung stattdessen belegen durch (1) Laborbuch-Einträge mit Datum, (2) die private Original-Historie als Nachweis beim Jurygespräch auf dem Laptop. Alternativ zu B: **Historie behalten**, dann braucht es einen Rewrite, der der Zusage „ältere Commits bleiben unverändert“ (HANDOVER §6) widerspricht; nicht ohne Maltes Zustimmung. **Zu prüfen:** KI-Transparenz im README (Abschnitt „Entstehung und Unterstützungsleistungen“), damit die Offenlegung wie in der Arbeit konsistent bleibt. Das weicht von der Zeile „KI im Repo-Inhalt nicht erwähnen“ in `CLAUDE.md` ab, deshalb entscheidet Malte.

## 5. Zielstruktur der öffentlichen Version (Vorschlag)
```
qire/
  README.md           Kurzfassung, ehrliche Aussage „quanten-inspiriert, kein Quantenvorteil“, Reproduktionsanleitung, Unterstützungsleistungen
  LICENSE             MIT oder Apache-2.0 (Code)
  LICENSE-DATA.md     CC BY-4.0 für Texte/Rohdaten + Hinweis auf Datensatzlizenzen
  CITATION.cff        aus CITATION.cff.entwurf
  .zenodo.json        optional (Metadaten für Zenodo; sonst aus CITATION.cff, zu prüfen)
  arbeit/             schriftliche Arbeit (PDF, < 5 MB) + LaTeX/Word-Quelle
  code/               INSTITUT/labor/, INSTITUT/abteilungen/, direktor.py, Projekt-A-Kern (A/quantum_sim)
  ergebnisse/         Ergebnis-JSONs (ohne _zwischen_*) + agenda.json
  laborbuch/          LABORBUCH.md, FEHLERBUCH.md, FORSCHUNGSMETHODIK.md
  dashboard/          erzeuge.py + dashboard.html
  recherche/          nur Literaturliste, keine Chats
  daten/README.md     wie man MNIST/Fashion/UCR selbst lädt (keine Daten im Repo)
```

## 6. Lizenzvorschlag
| Teil | Vorschlag | Begründung / Prüfpunkt |
|---|---|---|
| Code | **MIT** (einfach) oder **Apache-2.0** (mit Patentklausel) | Beides erlaubt Nachnutzung und Zitierbarkeit. Empfehlung MIT, weil Maltes Ziel Sichtbarkeit ist; Wahl trifft Malte. Alle Dateikopfzeilen/Abhängigkeiten auf Kompatibilität prüfen |
| Text, Laborbuch, Arbeit, Abbildungen | **CC BY-4.0** | Namensnennung bei Weiterverwendung |
| Ergebnis-JSONs (eigene Messdaten) | **CC BY-4.0** | eigene Daten; Datenbank-/Urheberrecht für Messwerte **zu prüfen** |
| Datensätze | **nicht mitliefern** | siehe Tabelle unten |
| Modellgewichte | nicht mitliefern | LLäMmlein: RAIL-M research-only (`chat_sweep/C.md`) |

### Datensatz-Lizenzen (alle aus Gedächtnis, **zu prüfen**)
Die Datensätze stehen nicht im Repo (`INSTITUT/daten/` ist gitignored, `labor/gemeinsam.py` lädt sie, `CLAUDE.md` „Technik“). So soll es bleiben.
| Datensatz | Nutzung im Projekt | Lizenzlage (unbestätigt) | Pflicht |
|---|---|---|---|
| MNIST | Hauptdatensatz (I-038 bis I-046) | Herkunft LeCun/Cortes/Burges; eine ausdrückliche Lizenz auf der Quellseite ist mir nicht sicher bekannt; Spiegelungen nennen unterschiedliche Lizenzen | Quelle zitieren; Lizenz der benutzten Spiegelung nachlesen |
| Fashion-MNIST | Hauptdatensatz | Zalando Research; nach meiner Erinnerung MIT-Lizenz | Quelle (Xiao et al. 2017) zitieren; Lizenz im GitHub-Repo prüfen |
| CIFAR-10 | I-042, I-044 | Krizhevsky 2009; kein formaler Lizenztext bekannt | Technischen Report zitieren |
| UCR (TwoPatterns, Wafer) | Q-019W-Replikation geplant, in `chat_sweep/00_ZUSAMMENFASSUNG.md` als „UCR-Lizenz ungeprüft“ geführt | Nutzungsbedingungen des UCR-Archivs unbekannt | **Vor Nutzung und Veröffentlichung prüfen**; Zitierpflicht lesen |
| Textkorpus (Projekt A, Deutsch) | Chatbot-Adapter | Herkunft in `UEBERARBEITUNG_2026-09-23/A/quantum_sim/korpus_deutsch.py` nachschlagen | **zu prüfen** |

## 7. Zenodo-Verknüpfung und DOI (Ablauf)
Alles unten ist das übliche Verfahren aus allgemeinem Wissen, **Details zu prüfen** auf zenodo.org und docs.github.com:
1. Öffentliches GitHub-Repo anlegen (Snapshot-Variante, §4).
2. Zenodo-Konto mit GitHub anmelden (Anmeldung über GitHub-OAuth) und im Bereich „GitHub“ den Schalter für dieses Repo einschalten. Das muss vor dem Release passieren.
3. `CITATION.cff` (und optional `.zenodo.json`) im Repo prüfen: Autorname, Lizenz, Titel, Schlagwörter. Zenodo übernimmt die Metadaten daraus bzw. aus dem Formular (Verhalten **zu prüfen**).
4. Auf GitHub ein Release erstellen (Tag z. B. `v1.0-jugend-forscht-2027`). Zenodo archiviert den Stand automatisch und vergibt eine **Versions-DOI** und eine **Konzept-DOI** (gilt für alle Versionen).
5. DOI in README, `CITATION.cff` (Feld `doi`) und in die Arbeit (Quellenverzeichnis, ohne Weblink im Fließtext, siehe Recherche 07 §2) eintragen; Badge optional.
6. Weitere Versionen (z. B. nach dem Bundesfinale) als neue Releases; die Konzept-DOI bleibt gleich.
Beachten: Ein veröffentlichter Zenodo-Datensatz kann nicht einfach zurückgezogen werden; Namen und Beschreibung vorher prüfen.

## 8. Zeitpunkt
- Vorbereitung (Aufräumen, Geheimnis-Scan, Lizenzen prüfen, README, Reproduktionstest): Dezember 2026, parallel zur Schreibphase (`ZEITLEISTE.md`).
- **Veröffentlichung und DOI: nach Abgabe der schriftlichen Arbeit** (Abgabe ca. 10.–15.01.2027, **zu prüfen**), Vorschlag zweite Januarhälfte, spätestens vor dem Landeswettbewerb. Gründe: (1) Die DOI kann in die gedruckten/ausgelegten Unterlagen für Regional- und Landeswettbewerb. (2) Bis zur Abgabe bleibt die Arbeit änderbar. (3) Der Quellenstand der Arbeit und das Release stimmen überein.
- **Alternative früh:** Wenn die Jury-Kriterien Vorregistrierung als Pluspunkt werten (Recherche 07 §1.2), kann das Repo früher veröffentlicht werden, dann aber mit Aufräumen vor dem 30.11. **Entscheidung Malte.**

## 9. Checkliste
- [ ] Malte hat Veröffentlichung ausdrücklich freigegeben (Plan §10 Nr. 7)
- [ ] Variante A/B gewählt (Historie oder Snapshot)
- [ ] Geheimnis-Scan der gesamten Historie, Ergebnis dokumentiert; ggf. Token widerrufen
- [ ] Private Chats, Session-IDs, PC-Details, MPI-Material, Werkstatt entfernt
- [ ] Kontaktdaten Dritter geschwärzt (Recherche 04)
- [ ] `_zwischen_*`, `.log`, doppelte `val.npy`, nicht benötigte `.pt` entfernt
- [ ] Datensatzlizenzen MNIST, Fashion-MNIST, CIFAR-10, UCR geprüft und im README zitiert
- [ ] LLäMmlein-Lizenz geprüft; keine Gewichte im Repo
- [ ] Lizenzwahl getroffen (Code MIT/Apache-2.0, Text/Daten CC BY-4.0), `LICENSE` angelegt
- [ ] README: ehrliche Aussage „quanten-inspiriert, kein Quantenvorteil“, Reproduktionsanleitung, Unterstützungsleistungen
- [ ] `CITATION.cff` mit echtem Namen (Platzhalter ersetzt), Titel passend zur Arbeit
- [ ] Reproduktionstest in sauberer Umgebung (`pip install torch numpy scipy scikit-learn qiskit`, Dashboard erzeugen, ein Schnelllauf)
- [ ] Zahlen-Audit: jede Zahl im README gegen Ergebnis-JSON
- [ ] Zenodo-Konto verknüpft, Repo aktiviert
- [ ] Release erstellt, DOI erhalten, in README und Arbeit eingetragen
- [ ] Alten privaten Stand als Archiv sichern (Nachweis Vorregistrierung)
