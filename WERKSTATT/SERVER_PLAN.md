# Heimserver für QIRE und spätere Forschungsprojekte

Stand: 12.10.2026. Preise sind **Schätzungen** aus Angebotspreisen; Speicher und
Grafikkarten sind 2026 wegen Speicherknappheit stark überteuert. Vor dem Kauf
auf Geizhals/Kleinanzeigen prüfen.

## Rahmen
- Komplett neues Gerät, der Arbeits-PC bleibt unverändert.
- Steht am Router, per LAN angebunden, läuft 24/7, kein Spielen.
- Ubuntu Server ohne Bildschirm, Fernzugriff über SSH/Tailscale.
  Linux erlaubt eine echte Sandbox für die Worker (Ordnergrenze).

## Stufe 1 — günstig: ca. 1.700–2.300 €
| Teil | Vorschlag | ca. € |
|---|---|---|
| CPU | Ryzen 5 9600 (6 Kerne, 65 W) | 200 |
| Mainboard | B650 (AM5) | 150 |
| RAM | 64 GB DDR5 (2×32) | 460–800 |
| GPU | RTX 5060 Ti 16 GB neu | 680–850 |
| SSD | 1 TB NVMe | 80–120 |
| Netzteil | 650 W, 80+ Gold | 90 |
| Gehäuse + Kühler | gedämmtes Midi-Tower + Tower-Kühler | 130 |
Modelle bis ~14B. Leise, sparsam.

## Stufe 2 — Preis-Leistung (Empfehlung): ca. 2.400–3.100 €
| Teil | Vorschlag | ca. € |
|---|---|---|
| CPU | Ryzen 7 9700X (8 Kerne, 65 W) | 300 |
| Mainboard | B650/X670 mit zweitem PCIe-x16-Steckplatz (Erweiterung) | 200–250 |
| RAM | 64 GB DDR5 (2×32), später auf 128 GB erweiterbar | 460–800 |
| GPU | **RTX 3090 24 GB gebraucht** | 1.000–1.200 |
| SSD | 2 TB NVMe | 150–250 |
| Netzteil | 1000 W, 80+ Gold (Reserve für zweite GPU) | 150 |
| Gehäuse | gedämmt, groß (z. B. Fractal Define 7) | 180 |
| Kühler | Tower-Luftkühler | 40 |
Modelle bis 32B mit großem Kontext, Simulationen nebenher.

## Stufe 3 — darüber
**3a: Mini-PC Ryzen AI Max+ 395, 128 GB gemeinsamer Speicher — ca. 2.300–2.900 €**
- Sehr leise, ~10–20 W im Leerlauf (Strom ~40–60 €/Jahr statt ~150–220 €).
- 128 GB für Modelle **und** Simulationen.
- Nachteil: Speicherbandbreite ~256 GB/s gegen ~936 GB/s der RTX 3090, also
  große Modelle grob 2–3× langsamer. AMD statt Nvidia (ROCm/Vulkan statt CUDA),
  nicht erweiterbar.

**3b: Stufe 2 mit Ryzen 9, 128 GB RAM, zwei RTX 3090 (48 GB VRAM) — ca. 3.800–4.800 €**
- Modelle bis ~70B. Lauter, stromhungrig (Leerlauf ~80–100 W).

## Laufende Kosten
24/7 bei 35 ct/kWh: Stufe 1/2 ca. 150–220 €/Jahr im Leerlauf, mehr unter Last.

## Quellen
- RTX 3090 gebraucht: kleinanzeigen.de (Angebote ~1.000–1.450 €, Okt. 2026)
- RTX 5060 Ti 16 GB: bestvaluegpu.com, shattered.io (~680–850 €)
- DDR5 64 GB: geizhals.de, guenstiger.de (~460–1.300 €)
- Ryzen AI Max+ 395 Mini-PCs: borncity.com, ebay.de (~2.270–2.900 €)
