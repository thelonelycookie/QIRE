"""Schlanke Schnittstelle zum llama.cpp-Server (OpenAI-kompatibel) + Attrappe fuer Tests."""
from __future__ import annotations

import hashlib
import re
import time
from typing import Optional

import numpy as np

MAX_TIMEOUT_S = 90.0   # harte Obergrenze je Aufruf (Stopp-Garantie < 5 min)


class LLMFehler(RuntimeError):
    pass


class LLM:
    """Basisklasse. chat() liefert Text, embed() liefert (n, d)-Array oder None."""

    def chat(self, system: str, user: str, temperatur: float = 0.7,
             max_tokens: int = 256, timeout: float = MAX_TIMEOUT_S) -> str:
        raise NotImplementedError

    def embed(self, texte: list[str]) -> Optional[np.ndarray]:
        return None


class LlamaServerLLM(LLM):
    def __init__(self, basis_url: str = "http://127.0.0.1:8080", modell: str = "lokal",
                 embed_url: Optional[str] = None, timeout: float = MAX_TIMEOUT_S):
        import requests  # spaet importieren: Tests ohne Netz brauchen es nicht
        self._r = requests
        self.url = basis_url.rstrip("/")
        self.embed_url = (embed_url or "").rstrip("/") or None
        self.modell = modell
        self.timeout = min(float(timeout), MAX_TIMEOUT_S)
        self.letzte_timings: dict = {}

    def chat(self, system, user, temperatur=0.7, max_tokens=256, timeout=MAX_TIMEOUT_S):
        t = min(float(timeout), self.timeout, MAX_TIMEOUT_S)
        nutz = {"model": self.modell, "temperature": temperatur, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        try:
            r = self._r.post(self.url + "/v1/chat/completions", json=nutz, timeout=t)
            r.raise_for_status()
            d = r.json()
            self.letzte_timings = d.get("timings", {}) or {}
            return d["choices"][0]["message"]["content"] or ""
        except Exception as e:  # Netz, Timeout, JSON ...
            raise LLMFehler(f"chat fehlgeschlagen: {e}") from e

    def embed(self, texte):
        basis = self.embed_url
        if not basis:
            return None
        try:
            r = self._r.post(basis + "/v1/embeddings", json={"input": texte}, timeout=min(60, self.timeout))
            r.raise_for_status()
            daten = r.json()["data"]
            return np.array([x["embedding"] for x in daten], dtype=np.float32)
        except Exception:
            try:  # aelterer Endpunkt /embedding
                vs = []
                for t in texte:
                    r = self._r.post(basis + "/embedding", json={"content": t}, timeout=min(60, self.timeout))
                    r.raise_for_status()
                    e = r.json()
                    e = e[0]["embedding"] if isinstance(e, list) else e["embedding"]
                    vs.append(np.array(e, dtype=np.float32).reshape(-1) if not isinstance(e[0], list)
                              else np.array(e[0], dtype=np.float32))
                return np.stack(vs)
            except Exception:
                return None


class AttrappenLLM(LLM):
    """Deterministisch, ohne Netz. Antwortet je nach Rolle im Systemprompt
    (Marker 'ROLLE:ERFINDER' usw.). Zaehlt Aufrufe; optional Verzoegerung."""

    def __init__(self, verzoegerung: float = 0.0, embeddings: bool = False,
                 kaputt_jedes: int = 0):
        self.verzoegerung = verzoegerung
        self.mit_embeddings = embeddings
        self.kaputt_jedes = kaputt_jedes   # jeder n-te Aufruf wirft LLMFehler
        self.aufrufe = 0

    @staticmethod
    def _h(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)

    def chat(self, system, user, temperatur=0.7, max_tokens=256, timeout=MAX_TIMEOUT_S):
        self.aufrufe += 1
        if self.kaputt_jedes and self.aufrufe % self.kaputt_jedes == 0:
            raise LLMFehler("Attrappe: simulierter Fehler")
        if self.verzoegerung:
            time.sleep(self.verzoegerung)
        h = self._h(system + "|" + user + f"|{self.aufrufe}")
        if "ROLLE:ERFINDER" in system:
            return (f"Idee {h % 100000}: Teste, ob Winkelmuster {h % 17} die Magie-Konzentration "
                    f"bei Ziffern senkt; Kontrolle: Zufallswinkel {h % 13}.")
        if "ROLLE:PHYSIK" in system:
            return '{"verstoesse": [], "gedankenexperiment": false, "begruendung": "keine Verletzung erkennbar"}'
        if "ROLLE:RUBRIK" in system:
            z = [1 + (h >> (3 * i)) % 5 for i in range(4)]
            return ('```json\n{"neuheit": %d, "pruefbarkeit": %d, "erkenntniswert": %d, '
                    '"quantenstufe": "Q%d", "ehrlichkeit": %d, "begruendung": "Attrappe"}\n```'
                    % (z[0], z[1], z[2], 1 + z[3] % 3, z[3]))
        if "ROLLE:KARTE" in system:
            return ('{"titel": "Attrappen-Karte %d", "hypothese": "H1: Effekt vorhanden", '
                    '"kontrolle": "Zufallsbaseline und entartete Kontrolle", '
                    '"erwarteter_effekt": "+1 pp, KI schliesst 0 aus", '
                    '"abbruch": "Abbruch, wenn Effekt < 0,1 pp bei 3 Seeds", '
                    '"neuheitsbegruendung": "Attrappe", "billigster_test": "10 min CPU", '
                    '"konzept": "Magie", "engpass": "Energie", "quantenstufe": "Q2"}' % (h % 1000))
        return "Attrappe"

    def embed(self, texte):
        if not self.mit_embeddings:
            return None
        return np.stack([_hash_vektor(t, 128) for t in texte])


def _hash_vektor(text: str, dim: int = 256) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v
