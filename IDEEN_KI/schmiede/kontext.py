"""RAG ueber Repo-Dateien: in Abschnitte zerlegen, Index cachen, passende Ausschnitte
im Token-Budget liefern. Nie das ganze Repo."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

MAX_ABSCHNITT = 900          # Zeichen
_WORT = re.compile(r"[a-zA-ZÄÖÜäöüß0-9_]{3,}")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORT.findall(text)]


def schaetze_token(text: str) -> int:
    return max(1, int(len(text) / 3.5))


@dataclass
class Abschnitt:
    quelle: str
    titel: str
    text: str


def _teile(quelle: str, titel: str, text: str) -> list[Abschnitt]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_ABSCHNITT:
        return [Abschnitt(quelle, titel, text)]
    out, cur = [], ""
    for absatz in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(absatz) > MAX_ABSCHNITT:
            out.append(Abschnitt(quelle, titel, cur.strip()))
            cur = ""
        while len(absatz) > MAX_ABSCHNITT:
            out.append(Abschnitt(quelle, titel, absatz[:MAX_ABSCHNITT]))
            absatz = absatz[MAX_ABSCHNITT:]
        cur += absatz + "\n\n"
    if cur.strip():
        out.append(Abschnitt(quelle, titel, cur.strip()))
    return out


def zerlege_markdown(quelle: str, text: str) -> list[Abschnitt]:
    out, titel, puffer = [], "(Kopf)", []
    for zeile in text.splitlines():
        if re.match(r"^#{1,4}\s", zeile):
            out += _teile(quelle, titel, "\n".join(puffer))
            titel, puffer = zeile.lstrip("# ").strip(), []
        else:
            puffer.append(zeile)
    out += _teile(quelle, titel, "\n".join(puffer))
    return out


def befundtabelle(text: str) -> list[Abschnitt]:
    """Zeilen der Tabelle in UEBERGABE.md Abschnitt 4 -> je ein Abschnitt."""
    m = re.search(r"^## 4\..*?(?=^## \d)", text, re.S | re.M)
    if not m:
        return []
    out = []
    for z in m.group(0).splitlines():
        if z.startswith("|") and not re.match(r"^\|[-\s|]+$", z) and "Befund" not in z[:12]:
            zellen = [c.strip() for c in z.strip("|").split("|")]
            out.append(Abschnitt("UEBERGABE.md#4", "Befund", " -- ".join(zellen)))
    return out


def zerlege_ideen_json(pfad: Path) -> list[Abschnitt]:
    try:
        d = json.loads(pfad.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for k in d.get("karten", []):
        t = (f"{k.get('id','')}: {k.get('titel','')} | Konzept: {k.get('konzept','')} | "
             f"Status: {k.get('status','')} | {k.get('grund','') or k.get('neuheit','')}")
        out.append(Abschnitt("ideen.json", k.get("id", ""), t[:MAX_ABSCHNITT]))
    return out


def zerlege_agenda(pfad: Path) -> list[Abschnitt]:
    try:
        d = json.loads(pfad.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [Abschnitt("agenda.json", p.get("id", ""),
                      f"{p.get('id','')} [{p.get('status','')}] {p.get('frage','')} {p.get('notiz','')}"[:MAX_ABSCHNITT])
            for p in d.get("punkte", [])]


class KontextIndex:
    def __init__(self, abschnitte: list[Abschnitt]):
        self.abschnitte = abschnitte
        self.tf = [Counter(tokens(a.titel + " " + a.text)) for a in abschnitte]
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = max(1, len(abschnitte))
        self.idf = {w: math.log(1 + n / d) for w, d in df.items()}
        self.norm = [math.sqrt(sum((v * self.idf[w]) ** 2 for w, v in c.items())) or 1.0 for c in self.tf]

    def suche(self, anfrage: str, token_budget: int = 1200, max_treffer: int = 12) -> list[Abschnitt]:
        q = Counter(tokens(anfrage))
        if not q:
            return []
        wertung = []
        for i, c in enumerate(self.tf):
            s = sum(q[w] * c[w] * self.idf.get(w, 0) ** 2 for w in q if w in c) / self.norm[i]
            if s > 0:
                wertung.append((s, i))
        wertung.sort(reverse=True)
        out, rest = [], token_budget
        for _, i in wertung[:max_treffer * 3]:
            a = self.abschnitte[i]
            kosten = schaetze_token(a.text) + 8
            if kosten > rest:
                continue
            out.append(a)
            rest -= kosten
            if len(out) >= max_treffer:
                break
        return out

    def speichern(self, pfad: Path, signatur: dict) -> None:
        pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signatur": signatur, "abschnitte": [asdict(a) for a in self.abschnitte]},
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(pfad)


def quellen(repo: Path, ki_ordner: Path) -> list[Path]:
    inst = repo / "INSTITUT"
    kand = [inst / "UEBERGABE.md", repo / "START.md", inst / "START.md", inst / "FEHLERBUCH.md",
            inst / "agenda.json", inst / "ideen.json"]
    kand += sorted((inst / "recherche").glob("*.md")) if (inst / "recherche").exists() else []
    lit = ki_ordner / "literatur"
    if lit.exists():
        kand += sorted(lit.rglob("*.md")) + sorted(lit.rglob("*.txt"))
    return [p for p in kand if p.exists()]


def lade_abschnitte(pfad: Path) -> list[Abschnitt]:
    name = pfad.name
    if name == "agenda.json":
        return zerlege_agenda(pfad)
    if name == "ideen.json":
        return zerlege_ideen_json(pfad)
    text = pfad.read_text(encoding="utf-8", errors="replace")
    if name == "UEBERGABE.md":
        return befundtabelle(text) + zerlege_markdown(name, text)
    return zerlege_markdown(name, text)


def baue_index(repo: Path, ki_ordner: Path, cache: Path | None = None) -> KontextIndex:
    """Baut den Index; nutzt den Cache, wenn Dateien (Name, Groesse, mtime) unveraendert sind."""
    dateien = quellen(repo, ki_ordner)
    sig = {str(p): [p.stat().st_size, int(p.stat().st_mtime)] for p in dateien}
    if cache and cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("signatur") == sig:
                return KontextIndex([Abschnitt(**a) for a in d["abschnitte"]])
        except Exception:
            pass
    abschnitte: list[Abschnitt] = []
    for p in dateien:
        try:
            abschnitte += lade_abschnitte(p)
        except Exception:
            continue
    idx = KontextIndex(abschnitte)
    if cache:
        idx.speichern(cache, sig)
    return idx


def formatiere(abschnitte: list[Abschnitt]) -> str:
    return "\n".join(f"- [{a.quelle}/{a.titel}] {a.text}" for a in abschnitte)
