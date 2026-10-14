# -*- coding: utf-8 -*-
"""Gemeinsame GPU-Sperre (R5.1): Routine-Worker und Ideen-KI laden nie gleichzeitig ein Modell.

Nur Standardbibliothek, damit die Datei ohne Abhaengigkeit von der Ideen-KI kopiert/importiert werden kann
(Schnittstelle und Protokoll: GPU_SPERRE.md).

Dateien (alle in WERKSTATT/, per WERKSTATT/.gitignore nie committet):
    gpu.lock        Besitzer der GPU (JSON, atomar angelegt mit O_EXCL, Herzschlag alle ~30 s)
    gpu.anfrage     Vorrangflagge "ich will die GPU" (JSON); Ideen-KI holt sich die Sperre dann NICHT neu
    gpu.vram_frei   Quittung der Ideen-KI: Modell entladen, VRAM ist frei (nur bei vram_frei=True)

Veraltet (herrenlos) ist eine Sperre, wenn: Besitzer-PID auf demselben Rechner nicht mehr lebt, ODER der
Herzschlag aelter als tot_nach_s ist, ODER ablauf ueberschritten ist. Veraltete Sperren werden uebernommen.
"""
from __future__ import annotations

import json
import os
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

STANDARD_DATEI = Path(__file__).resolve().parent.parent / "gpu.lock"
TOT_NACH_S = 180          # Herzschlag aelter -> herrenlos
HERZSCHLAG_S = 30
MAX_HALTEN_S = 600        # Obergrenze, danach gilt die Sperre als abgelaufen (Worker-Aufruf <= 90 s)


def jetzt() -> float:
    return time.time()


def pid_lebt(pid: int) -> bool:
    """Lebt der Prozess? (POSIX: kill 0, Windows: OpenProcess)"""
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - nur Windows
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lies(pfad: Path) -> Optional[dict]:
    try:
        d = json.loads(Path(pfad).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _schreibe_atomar(pfad: Path, d: dict) -> None:
    tmp = pfad.with_name(pfad.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, pfad)


def ist_veraltet(d: Optional[dict], pfad: Path, tot_nach_s: float = TOT_NACH_S, uhr=jetzt) -> bool:
    """Herrenlose Sperre? d=None (leer/kaputt) gilt erst nach 10 s Dateialter als veraltet
    (der Besitzer koennte gerade schreiben)."""
    t = uhr()
    if d is None:
        try:
            return t - Path(pfad).stat().st_mtime > 10
        except OSError:
            return False
    if d.get("host") == platform.node() and not pid_lebt(int(d.get("pid", 0))):
        return True
    if t - float(d.get("herzschlag", d.get("start", 0))) > tot_nach_s:
        return True
    if d.get("ablauf") and t > float(d["ablauf"]):
        return True
    return False


def lese_sperre(datei: Path = STANDARD_DATEI, tot_nach_s: float = TOT_NACH_S) -> Optional[dict]:
    """Aktuelle Sperre (dict, mit Feld 'veraltet') oder None, wenn frei. Fuer Status/Ideen-KI."""
    datei = Path(datei)
    if not datei.exists():
        return None
    d = _lies(datei)
    veraltet = ist_veraltet(d, datei, tot_nach_s)
    return dict(d or {"besitzer": "unbekannt"}, veraltet=veraltet)


def anfrage_offen(datei: Path = STANDARD_DATEI, uhr=jetzt) -> Optional[dict]:
    """Offene Vorrang-Anfrage (gpu.anfrage neben gpu.lock) oder None. Abgelaufene/herrenlose Anfragen zaehlen nicht."""
    p = Path(datei).with_name("gpu.anfrage")
    d = _lies(p) if p.exists() else None
    if not d:
        return None
    if d.get("host") == platform.node() and not pid_lebt(int(d.get("pid", 0))):
        return None
    if d.get("ablauf") and uhr() > float(d["ablauf"]):
        return None
    return d


class GpuSperre:
    """Sperre mit Besitzername. Benutzung:

        s = GpuSperre("routine-worker", "Lauf-Log zusammenfassen")
        if s.erwerbe(warte_max_s=900):
            try:  ... LLM-Aufrufe ...
            finally: s.freigeben()
    oder:  with GpuSperre(...) as s: (wirft SperreBelegt, wenn nach warte_max_s nicht frei)
    """

    def __init__(self, besitzer: str, zweck: str = "", datei: Path = STANDARD_DATEI, tot_nach_s: float = TOT_NACH_S,
                 max_halten_s: float = MAX_HALTEN_S, herzschlag_s: float = HERZSCHLAG_S, uhr=jetzt):
        self.besitzer, self.zweck = besitzer, zweck
        self.datei = Path(datei)
        self.tot_nach_s, self.max_halten_s, self.herzschlag_s = tot_nach_s, max_halten_s, herzschlag_s
        self.uhr = uhr
        self.token: Optional[str] = None
        self._stopp = threading.Event()
        self._faden: Optional[threading.Thread] = None

    # ---- Grundoperationen
    def _inhalt(self) -> dict:
        t = self.uhr()
        return {"besitzer": self.besitzer, "zweck": self.zweck, "pid": os.getpid(), "host": platform.node(),
                "start": t, "herzschlag": t, "ablauf": t + self.max_halten_s, "token": self.token}

    def versuche(self) -> bool:
        """Einmal versuchen. True = Sperre gehoert uns."""
        self.datei.parent.mkdir(parents=True, exist_ok=True)
        self.token = uuid.uuid4().hex
        for _ in range(3):
            try:
                fd = os.open(str(self.datei), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                d = _lies(self.datei)
                if not ist_veraltet(d, self.datei, self.tot_nach_s, self.uhr):
                    return False
                self._raeume_veraltete_ab(d)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._inhalt(), f, ensure_ascii=False, indent=1)
            return True
        return False

    def _raeume_veraltete_ab(self, beurteilt: Optional[dict]) -> None:
        """Veraltete Sperre atomar wegbewegen (nur wenn es noch dieselbe ist, sonst zurueck)."""
        weg = self.datei.with_name(f"gpu.lock.veraltet.{int(self.uhr())}.{os.getpid()}")
        try:
            os.replace(self.datei, weg)
        except OSError:
            return
        jetzt_d = _lies(weg)
        if (beurteilt or {}).get("token") != (jetzt_d or {}).get("token"):
            # Zwischenzeitlich hat jemand anderes eine frische Sperre angelegt -> zurueckstellen, falls Platz frei
            try:
                os.link(weg, self.datei)
            except OSError:
                pass
        try:
            os.remove(weg)
        except OSError:
            pass

    def erwerbe(self, warte_max_s: float = 0, anfrage: bool = True, vram_frei: bool = False,
                pause_s: float = 2.0, stopp=lambda: False) -> bool:
        """Wartet bis zu warte_max_s auf die Sperre. anfrage=True legt gpu.anfrage (Vorrang) an, damit die
        Ideen-KI nach ihrem laufenden Schritt nicht neu zugreift. vram_frei=True wartet zusaetzlich auf die
        Quittung gpu.vram_frei (Ideen-KI hat ihr Modell entladen)."""
        ende = self.uhr() + warte_max_s
        angefragt = False
        try:
            if anfrage:
                self._schreibe_anfrage(vram_frei)
                angefragt = True
            while True:
                if self.versuche():
                    if not vram_frei or self._vram_quittiert():
                        self._start_herzschlag()
                        return True
                    self._gib_frei_ohne_anfrage()  # zu frueh: Ideen-KI hat noch nicht entladen
                if stopp() or self.uhr() >= ende:
                    break
                time.sleep(min(pause_s, max(0.05, ende - self.uhr())))
        except BaseException:
            if angefragt:
                self._loesche_anfrage()
            raise
        if angefragt:
            self._loesche_anfrage()
        return False

    def freigeben(self) -> None:
        """Gibt frei (nur wenn die Sperre noch uns gehoert) und nimmt Anfrage/Quittung weg."""
        self._stopp.set()
        if self._faden:
            self._faden.join(timeout=2)
            self._faden = None
        self._gib_frei_ohne_anfrage()
        self._loesche_anfrage()
        try:
            self.datei.with_name("gpu.vram_frei").unlink()
        except OSError:
            pass

    def _gib_frei_ohne_anfrage(self) -> None:
        d = _lies(self.datei)
        if d and d.get("token") == self.token:
            try:
                self.datei.unlink()
            except OSError:
                pass

    # ---- Herzschlag
    def herzschlag(self) -> None:
        d = _lies(self.datei)
        if d and d.get("token") == self.token:
            d["herzschlag"] = self.uhr()
            _schreibe_atomar(self.datei, d)

    def _start_herzschlag(self) -> None:
        self._stopp.clear()

        def lauf():
            while not self._stopp.wait(self.herzschlag_s):
                try:
                    self.herzschlag()
                except OSError:
                    pass
        self._faden = threading.Thread(target=lauf, daemon=True)
        self._faden.start()

    # ---- Anfrage / Quittung
    def _schreibe_anfrage(self, vram_frei: bool) -> None:
        t = self.uhr()
        _schreibe_atomar(self.datei.with_name("gpu.anfrage"),
                         {"von": self.besitzer, "zweck": self.zweck, "pid": os.getpid(), "host": platform.node(),
                          "zeit": t, "ablauf": t + 1800, "vram_frei": vram_frei})

    def _loesche_anfrage(self) -> None:
        p = self.datei.with_name("gpu.anfrage")
        d = _lies(p)
        if d and d.get("pid") == os.getpid():
            try:
                p.unlink()
            except OSError:
                pass

    def _vram_quittiert(self) -> bool:
        d = _lies(self.datei.with_name("gpu.vram_frei"))
        return bool(d) and self.uhr() - float(d.get("zeit", 0)) < 3600

    # ---- with-Statement
    def __enter__(self):
        if not self.erwerbe(warte_max_s=0):
            raise SperreBelegt(str(lese_sperre(self.datei)))
        return self

    def __exit__(self, *a):
        self.freigeben()


class SperreBelegt(RuntimeError):
    pass
