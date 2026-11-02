# WISSEN: Wissensbasis für QIRE

**Zweck:** alles, was für QIRE (quanten-inspirierte, ressourceneffiziente KI) relevant ist, einheitlich erfassen, Schlüsse ziehen, Sättigung messen. Plan: `INSTITUT/PLAN_2026-11-02_wissensbasis.md`.

## Ablage
- `quellen.json`: Liste aller Quellen, einzige Quelle der Wahrheit
- `felder/<ID>_<name>.md`: 22 Themenfelder (K1–K8, N1–N8, F1–F6)
- `schluesse/`: abgeleitete Schlüsse (S-xxx.md)
- `volltexte/`: heruntergeladene Volltexte
- `werkzeuge/`: `schema.json`, `pruefen.py`
- `INDEX.md`: automatisch erzeugte Übersicht

## Schema (Kurzform, Details in `werkzeuge/schema.json`)
`id` (arXiv:xxxx.xxxxx | doi:… | url:…), `titel`, `jahr` (int|null), `autoren`, `feld[]`, `kernaussage`,
`zahlen[{wert, kontext, geprueft}]`, `baseline_fuer_qire`, `quantenbezug` (direkt|analogie|keiner),
`bezug_qire_befund[]`, `stuetzt_widerspricht` (stuetzt|widerspricht|neutral|unklar),
`volltext_gelesen`, `gefunden_am` (YYYY-MM-DD), `welle`, `herkunft`.

## Befehle
```
python3 WISSEN/werkzeuge/pruefen.py                  # prüfen (Exit 1 bei Fehlern)
python3 WISSEN/werkzeuge/pruefen.py --saettigung 2   # Anteil neu je Feld in Welle 2, < 10 % markiert
python3 WISSEN/werkzeuge/pruefen.py --index          # INDEX.md neu schreiben
python3 -m pytest tests/test_wissen.py -q            # Tests
```
