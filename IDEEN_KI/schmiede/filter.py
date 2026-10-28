"""Duplikatfilter (Embedding-Aehnlichkeit, Fallback TF-IDF in numpy) + Luckenkarte."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

_DIM = 2048


def _woerter(text: str) -> list[str]:
    w = re.findall(r"\w{3,}", text.lower())
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


def _slot(wort: str) -> int:
    return int(hashlib.md5(wort.encode("utf-8")).hexdigest()[:8], 16) % _DIM


class AehnlichkeitsIndex:
    """Naechster-Nachbar-Suche ueber alle bisherigen Ideentexte.
    Backend: Embeddings des LLM (falls vorhanden), sonst gehashtes TF-IDF."""

    def __init__(self, llm=None, schwelle_embedding: float = 0.88, schwelle_tfidf: float = 0.70):
        self.llm = llm
        self.s_emb, self.s_tf = schwelle_embedding, schwelle_tfidf
        self.ids: list[str] = []
        self.texte: list[str] = []
        self._tf: list[np.ndarray] = []     # rohe Zaehlvektoren (float32, _DIM)
        self._df = np.zeros(_DIM, dtype=np.float32)
        self._emb: list[np.ndarray] = []
        self._emb_ok: Optional[bool] = None  # None = unbekannt

    # ---- intern
    def _embed(self, texte: list[str]) -> Optional[np.ndarray]:
        if self.llm is None or self._emb_ok is False:
            return None
        try:
            e = self.llm.embed(texte)
        except Exception:
            e = None
        if e is None:
            self._emb_ok = False
            return None
        self._emb_ok = True
        e = np.asarray(e, dtype=np.float32)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.where(n == 0, 1, n)

    @staticmethod
    def _zaehl(text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        for w in _woerter(text):
            v[_slot(w)] += 1
        return v

    # ---- Schnittstelle
    def hinzufuegen(self, ideen_id: str, text: str) -> None:
        self.ids.append(ideen_id)
        self.texte.append(text)
        v = self._zaehl(text)
        self._tf.append(v)
        self._df += (v > 0)
        e = self._embed([text])
        if e is not None:
            self._emb.append(e[0])

    def naechster(self, text: str) -> tuple[float, Optional[str], str]:
        """(Aehnlichkeit, id des naechsten, Verfahren)."""
        if not self.ids:
            return 0.0, None, "leer"
        e = self._embed([text]) if self._emb_ok is not False else None
        if e is not None and len(self._emb) == len(self.ids):
            sims = np.stack(self._emb) @ e[0]
            i = int(np.argmax(sims))
            return float(sims[i]), self.ids[i], "embedding"
        qv = self._zaehl(text)
        n = len(self._tf) + 1
        idf = np.log((1 + n) / (1 + self._df)) + 1.0
        M = np.log1p(np.stack(self._tf)) * idf
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        q = np.log1p(qv) * idf
        q /= max(float(np.linalg.norm(q)), 1e-9)
        sims = M @ q
        i = int(np.argmax(sims))
        return float(sims[i]), self.ids[i], "tfidf"

    def ist_duplikat(self, text: str) -> tuple[bool, float, Optional[str]]:
        s, i, verfahren = self.naechster(text)
        schwelle = self.s_emb if verfahren == "embedding" else self.s_tf
        return s >= schwelle, s, i


class Lueckenkarte:
    """Matrix Konzept x Konzept: wie oft wurde das Paar behandelt? Leere Felder bekommen Bonus."""

    def __init__(self, konzepte_a: list[str], konzepte_b: list[str], pfad: Optional[Path] = None):
        self.a, self.b, self.pfad = konzepte_a, konzepte_b, pfad
        self.zaehler: dict[str, int] = {}
        if pfad and pfad.exists():
            try:
                self.zaehler = json.loads(pfad.read_text(encoding="utf-8"))
            except Exception:
                self.zaehler = {}

    @staticmethod
    def _k(x: str, y: str) -> str:
        return x + "||" + y

    def zaehle(self, x: str, y: str, n: int = 1) -> None:
        self.zaehler[self._k(x, y)] = self.zaehler.get(self._k(x, y), 0) + n

    def anzahl(self, x: str, y: str) -> int:
        return self.zaehler.get(self._k(x, y), 0)

    def bonus(self, x: str, y: str) -> float:
        return 1.0 / (1.0 + self.anzahl(x, y))

    def fuelle_aus_texten(self, texte: list[str]) -> None:
        """Zaehlt Paare, deren Stichwoerter gemeinsam in einem Text vorkommen (Repo/Literatur)."""
        def kern(k):  # erstes bedeutungstragendes Wort
            return re.split(r"[\s/(]", k.strip())[0].lower()
        ka = {k: kern(k) for k in self.a}
        kb = {k: kern(k) for k in self.b}
        for t in texte:
            tl = t.lower()
            ha = [k for k, w in ka.items() if w and w in tl]
            hb = [k for k, w in kb.items() if w and w in tl]
            for x in ha:
                for y in hb:
                    self.zaehle(x, y)

    def ziehe_paar(self, rng, versuche: int = 8) -> tuple[str, str]:
        """Bevorzugt leere/seltene Paare: aus mehreren Zufallspaaren das mit dem groessten Bonus."""
        best = None
        for _ in range(versuche):
            x, y = rng.choice(self.a), rng.choice(self.b)
            b = self.bonus(x, y)
            if best is None or b > best[0]:
                best = (b, x, y)
        return best[1], best[2]

    def speichern(self) -> None:
        if not self.pfad:
            return
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pfad.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.zaehler, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.pfad)
