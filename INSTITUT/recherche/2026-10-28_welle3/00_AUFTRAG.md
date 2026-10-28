# Welle 3 (28.10.2026) — Zuspitzen, Killer, Inkubation

Alles Deutsch. Nichts rechnen (außer ausdrücklich erlaubt), kein git. Arbeitsordner /home/user/QIRE_welle.
Lies zuerst: INSTITUT/KREATIVITAET_2026-10-27.md, recherche/2026-10-27_welle/SYNTHESE.md, recherche/2026-10-28_welle2/TURNIER.md und konv_kandidaten.md, KERNAUSSAGE_2026-10-07.md.

## Neue Lage (Raven, 28.10.)
- **I-049 H0:** Hourglass+LRU bei T=16000 um 0,055 bpc schlechter als flach; der Rauchtest-Vorteil war Frühregime.
- **I-047 H0:** Tensor-Train-FFN etwa 0,06 bpc schlechter als dicht/Niedrigrang. **Aber H3: Born-Quadrat z² schlägt GELU um 0,040 bpc.** Ähnlich schon I-045-v3 (D_z2 oft besser als D_gelu) und R3 (reell z² stark).
- **I-046-v2 H0:** bei 384 Parametern keine strukturierte Drehung besser als id und Haar.
- **T-ternaer-butterfly H0:** ternärer Butterfly-Arm kollabiert im Training.
- Läuft: I-048 (Rotation + Niedrigbit bei gleichen Bytes), I-045-v5, Erkundungspaket K01–K20/E01–E09 (Tuning-Seeds).
- Bekannter Vorbehalt: quant_rows(bits=2) ist ternär (1,58 Bit), gezählt 2 Bit.

## Ausgabe
`<kürzel>_bericht.md` im Ordner 2026-10-28_welle3, ≤ 200 Zeilen. Schlussnachricht ≤ 8 Zeilen.
