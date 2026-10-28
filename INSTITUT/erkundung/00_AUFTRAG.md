# Erkundungspaket Welle 2 (28.10.2026) — Rauchtests auf Raven

Freigabe Malte 28.10.: „alles machen, alles Vorgeschlagene darf getestet werden“. Das sind **Erkundungen, keine Befunde**: nichts in LABORBUCH, agenda oder seeds.json, kein direktor-Lauf.

## Regeln
- Lies `konv_kandidaten.md` (hier), `INSTITUT/recherche/2026-10-27_welle/SYNTHESE.md` ist auf Branch `origin/welle-2026-10-27-ergebnisse` (`git show origin/welle-2026-10-27-ergebnisse:INSTITUT/recherche/2026-10-27_welle/SYNTHESE.md`).
- Neu (28.10., Raven): **I-049 H0** (Hourglass+LRU bei T=16000 um 0,055 bpc schlechter als flach; Vorteil war Frühregime). **I-047 H0** (TT-FFN 0,06 bpc schlechter; Born-Quadrat +0,040 bpc besser als GELU). I-048 läuft gerade.
- Je Kandidat ein Skript `INSTITUT/erkundung/<KID>.py` mit Funktion `main(seed:int) -> dict` und `if __name__=="__main__"` (Argument `--seed`, `--schnell`). Es speichert `INSTITUT/erkundung/roh/<KID>_s<seed>.json`. Wiederverwenden: `labor/gemeinsam.py` (Daten), `abteilungen/struktur_sprache.py` (Byte-LM), `abteilungen/kompression_bytes.py` (Quantisierung, Bytes-Zählung).
- **Tuning-Seeds 950–959** (alle Kandidaten teilen sie, erlaubt). Nie Mess-Seeds.
- Rechenbudget auf Raven: **je Seed ≤ 45 min auf 1 Kern**. Lieber ordentlich trainieren (Lehre I-044-v3/I-045-v3/I-049: Kurzläufe täuschen) – Byte-LM mindestens 4000 Schritte bei d=128 oder MLP volle Epochen.
- **Jeder Test braucht die härteste klassische Kontrolle** (gleiche FLOPs bzw. gleiche Bytes, Zufallsvariante). Parameter, FLOPs, Bytes ausweisen.
- `--schnell` muss im Container in ≤ 2 min laufen (Rauchtest auf Seed 959). Führe ihn aus, er muss fehlerfrei sein.
- Kein git. Nur deine Dateien schreiben. Daten liegen in `INSTITUT/daten/`.
- Schlussnachricht ≤ 8 Zeilen: Dateien, `--schnell` ok?, geschätzte Minuten je Seed auf Raven.
