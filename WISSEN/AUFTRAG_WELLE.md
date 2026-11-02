# Auftrag für Recherche-Agenten (gilt für jede Welle)

Projekt QIRE (Jugend forscht): quanten-inspirierte, ressourceneffiziente KI. Alles auf Deutsch. Kein git.

## Vorher lesen
- `WISSEN/README.md` und `WISSEN/werkzeuge/schema.json`
- Deine Feld-Datei(en) `WISSEN/felder/<ID>_*.md` und alle Einträge deines Feldes in `WISSEN/quellen.json` (nur lesen). Das zeigt dir, was schon bekannt ist.
- Für den Bezug zu eigenen Befunden: `INSTITUT/recherche/zyklus/Z3_ergebnisse_raven.md` (Rotation gegen faire Gegner ≈ gleichwertig) und `INSTITUT/KERNAUSSAGE_2026-10-07.md`.

## Suchen
- ≤ 12 Websuchen je Feld. Reihenfolge: Übersichtsarbeiten, dann Neuestes (2025–2026), dann die stärkste Baseline bzw. den stärksten Gegner für QIRE.
- Bei arXiv-Funden Titel und Abstract über `http://export.arxiv.org/api/query?id_list=…` bestätigen (3 s Pause). Nichts erfinden. Was nicht bestätigt ist, wird nicht eingetragen.
- Erst ab 3 verlässlichen Quellen zählt etwas als „Konsens“.
- PDFs, die du ganz liest, kommen nach `WISSEN/volltexte/` (gitignored). Nur dann gilt `volltext_gelesen=true`, und nur dann sind Zahlen `geprueft=true`.

## Schreiben (nur diese Dateien)
1. `WISSEN/neu/<welle>_<agent>.json`: eine JSON-Liste NEUER Einträge im Schema, keine IDs, die schon in `quellen.json` stehen. Setze `welle` und `gefunden_am` (heute) und `herkunft` = „Welle <n>, <agent>“.
2. Deine Feld-Datei(en) ausfüllen bzw. ergänzen: Landkarte (Hauptlinien, wer gegen wen), stärkste Baselines/Gegner für QIRE, wichtige Zahlen mit geprüft/ungeprüft, **Quantenbezug** (was ist echt quanten, was nur Analogie, was ist der quanten-inspirierte Hebel für QIRE), Bezug zu eigenen Befunden, Lücken. Zitate als `[arXiv:…]`.
3. Am Ende der Feld-Datei kommt ein Abschnitt „Schlüsse und neue Wege (Agent)“ mit 2–5 Punkten. Jeder Punkt nennt den Schluss, die Belege, die langweiligste Erklärung und den Quantenbezug. Diese Punkte sind Vorschläge, Opus prüft sie.

## Schlussnachricht (≤ 8 Zeilen)
Zahl neuer Quellen, davon Volltext, der wichtigste Fund, ein möglicher Widerspruch zu eigenen Befunden, die stärkste Idee für QIRE.
