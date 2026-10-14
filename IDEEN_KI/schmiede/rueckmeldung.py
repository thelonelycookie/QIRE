"""Urteile (H1/H0/HU) aus INSTITUT/agenda.json + ergebnisse/ zu Karten zuordnen,
Belohnung an die Banditen, Few-shot-Speicher, Blindnoten-CSV."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

BELOHNUNG = {"H1": 1.0, "HU": 0.3, "H0": 0.2}   # H0 ist informativ, aber kein Treffer


def _lies(p: Path, std):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return std


def urteil_zu_punkt(punkt_id: str, ergebnisse: Path) -> Optional[str]:
    """Gesamturteil aus den Ergebnisdateien <ID>__*.json: irgendein H1 -> H1; sonst alle H0 -> H0; sonst HU."""
    urteile = []
    for f in sorted(ergebnisse.glob(f"{punkt_id}__*.json")):
        d = _lies(f, {})
        u = d.get("urteil")
        if isinstance(u, dict):
            urteile += [str(v)[:2].upper() for v in u.values()]
    urteile = [u for u in urteile if u in ("H1", "H0", "HU")]
    if not urteile:
        return None
    if "H1" in urteile:
        return "H1"
    return "H0" if all(u == "H0" for u in urteile) else "HU"


def ordne_zu(agenda: Path, ergebnisse: Path) -> list[dict]:
    """Findet Agenda-Punkte mit Feld `quelle_ideen_ki` (Karten-ID) und deren Urteil."""
    punkte = _lies(agenda, {}).get("punkte", [])
    out = []
    for p in punkte:
        kid = p.get("quelle_ideen_ki")
        if not kid:
            continue
        u = urteil_zu_punkt(p["id"], ergebnisse)
        if u:
            out.append({"karte": kid, "punkt": p["id"], "urteil": u})
    return out


def _alle_karten(ausgabe: Path) -> dict[str, dict]:
    karten = {}
    for f in ausgabe.glob("karten_*.json"):
        for k in _lies(f, {}).get("karten", []):
            karten[k["id"]] = k
    return karten


class Rueckmeldung:
    def __init__(self, daten: Path, ausgabe: Path, bandit):
        self.daten, self.ausgabe, self.bandit = daten, ausgabe, bandit
        self.stand_pfad = daten / "rueckmeldung_stand.json"
        self.beispiele_pfad = daten / "beispiele.jsonl"
        self.stand = _lies(self.stand_pfad, {"urteile": {}, "noten": {}})

    def _sichere_stand(self):
        self.daten.mkdir(parents=True, exist_ok=True)
        tmp = self.stand_pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.stand, indent=1), encoding="utf-8")
        os.replace(tmp, self.stand_pfad)

    def _beispiel(self, karte: dict, urteil: Optional[str], note: Optional[float]):
        self.daten.mkdir(parents=True, exist_ok=True)
        with open(self.beispiele_pfad, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": karte["id"], "idee": karte["frage"], "kanal": karte["kanal"],
                                "urteil": urteil, "note": note, "rubrik": karte.get("rubrik")},
                               ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def verarbeite_urteile(self, agenda: Path, ergebnisse: Path) -> int:
        karten, n = _alle_karten(self.ausgabe), 0
        for z in ordne_zu(agenda, ergebnisse):
            k = karten.get(z["karte"])
            if not k or self.stand["urteile"].get(z["karte"]) == z["urteil"]:
                continue
            self.bandit.aktualisiere(k["kanal"], k["operator"], BELOHNUNG[z["urteil"]], gewicht=3.0)
            self.stand["urteile"][z["karte"]] = z["urteil"]
            self._beispiel(k, z["urteil"], None)
            n += 1
        self._sichere_stand()
        return n

    # ---- Blindnoten
    def erzeuge_vorlage(self, pfad: Optional[Path] = None, n: int = 20) -> Path:
        """CSV fuer Maltes Blindnoten (ohne Score/Operator, damit blind)."""
        pfad = pfad or self.ausgabe / "blindnoten.csv"
        if pfad.exists():
            return pfad   # nie bestehende Noten ueberschreiben
        karten = sorted(_alle_karten(self.ausgabe).values(), key=lambda k: k["id"])[:n]
        self.ausgabe.mkdir(parents=True, exist_ok=True)
        with open(pfad, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["karten_id", "idee", "note_1_bis_5", "kommentar"])
            for k in karten:
                w.writerow([k["id"], k["frage"], "", ""])
        return pfad

    def lies_blindnoten(self, pfad: Optional[Path] = None) -> int:
        pfad = pfad or self.ausgabe / "blindnoten.csv"
        if not pfad.exists():
            return 0
        karten, n = _alle_karten(self.ausgabe), 0
        with open(pfad, encoding="utf-8", newline="") as f:
            for z in csv.DictReader(f, delimiter=";"):
                try:
                    note = float(str(z.get("note_1_bis_5", "")).replace(",", "."))
                except ValueError:
                    continue
                kid = z.get("karten_id", "")
                k = karten.get(kid)
                if not k or not 1 <= note <= 5 or self.stand["noten"].get(kid) == note:
                    continue
                self.bandit.aktualisiere(k["kanal"], k["operator"], (note - 1) / 4.0, gewicht=2.0)
                self.stand["noten"][kid] = note
                self._beispiel(k, None, note)
                n += 1
        self._sichere_stand()
        return n

    def fewshot(self, kanal: Optional[str] = None, k: int = 3) -> str:
        """Textblock mit je den besten und schlechtesten Beispielen (aus Urteilen/Noten)."""
        if not self.beispiele_pfad.exists():
            return ""
        bsp = []
        for z in self.beispiele_pfad.read_text(encoding="utf-8").splitlines():
            try:
                bsp.append(json.loads(z))
            except Exception:
                pass
        def wert(b):
            return BELOHNUNG[b["urteil"]] if b.get("urteil") else (b["note"] - 1) / 4 if b.get("note") else None
        bsp = [b for b in bsp if wert(b) is not None and (kanal is None or b.get("kanal") == kanal)]
        if not bsp:
            return ""
        bsp.sort(key=wert)
        gut, schlecht = bsp[-k:], bsp[:k]
        z = ["Beispiele aus Maltes/Claudes Urteilen:"]
        z += [f"- GUT ({b.get('urteil') or b.get('note')}): {b['idee']}" for b in gut]
        z += [f"- SCHWACH ({b.get('urteil') or b.get('note')}): {b['idee']}" for b in schlecht if b not in gut]
        return "\n".join(z)
