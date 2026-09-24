from __future__ import annotations

"""Interaktive Stand-Demo: Ziffer zeichnen, drei Modelle vergleichen.

Der Besucher zeichnet eine Ziffer und sieht sofort, was die drei Modelle daraus
machen. Die Gegenueberstellung ist selbst die Aussage der Arbeit:

  Quantenschaltung (Simulation)  -- zieht mit dem klassischen Modell mit
  Quantenschaltung (Hardware)    -- faellt ab, lief aber echt auf ibm_fez
  klassisch                      -- die Kontrolle

Alles laeuft offline. Am Stand ist kein Internet und keine Wartezeit noetig;
die Belege der echten Hardware-Laeufe (Job-IDs) werden mit angezeigt und sind
ueber quantum.ibm.com nachpruefbar.

Bedienung
  linke Maustaste  zeichnen
  rechte Maustaste radieren
  Leertaste        Feld loeschen

Start: python -m demo.demo_app
"""

import json
import math
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from quantum_sim.hardware_model import HardwareReadyModel
from quantum_sim.quantum_layer import HybridModel

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

CANVAS = 336          # 28 * 12, damit die Verkleinerung glatt aufgeht
CELL = CANVAS // 28
BRUSH = 1.6           # Pinselradius in 28x28-Zellen

BG = "#12141a"
FG = "#e8eaf0"
MUTED = "#8b93a7"
ACCENT_Q = "#5ac8fa"
ACCENT_H = "#ffb340"
ACCENT_C = "#98d8a0"


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Quanten-KI: Ziffernerkennung — Jugend forscht")
        root.configure(bg=BG)

        self.grid = np.zeros((28, 28), dtype=np.float32)
        self.bundle = self._load_models()

        self._build_ui()
        self._predict()

    # ---------------------------------------------------------------- Laden
    def _load_models(self) -> Dict:
        path = ARTIFACT_DIR / "demo_models.pt"
        if not path.exists():
            raise SystemExit(
                f"Modelldatei fehlt: {path}\n"
                "Bitte zuerst 'python -m demo.train_demo_model' ausfuehren."
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)

        q_sim = HybridModel(blob["input_dim"], blob["sim_qubits"], blob["sim_layers"],
                            blob["num_classes"], entangle=True)
        q_sim.load_state_dict(blob["quantum_sim_state"]); q_sim.eval()

        q_hw = HardwareReadyModel(blob["input_dim"], blob["hw_qubits"],
                                  blob["hw_layers"], blob["num_classes"])
        q_hw.load_state_dict(blob["quantum_hw_state"]); q_hw.eval()

        from .train_demo_model import MatchedClassical
        classical = MatchedClassical(blob["input_dim"], blob["classical_width"],
                                     blob["num_classes"], blob["classical_depth"])
        classical.load_state_dict(blob["classical_state"]); classical.eval()

        evidence_path = ARTIFACT_DIR / "hardware_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}

        return {"q_sim": q_sim, "q_hw": q_hw, "classical": classical,
                "accuracy": blob["accuracy"], "parameters": blob["parameters"],
                "evidence": evidence}

    # ------------------------------------------------------------ Oberflaeche
    def _build_ui(self):
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, padx=16, pady=16, sticky="n")

        tk.Label(left, text="Ziffer hier zeichnen", bg=BG, fg=FG,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS,
                                bg="#000000", highlightthickness=1,
                                highlightbackground="#2a2f3a", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", lambda e: self._paint(e, 1.0))
        self.canvas.bind("<Button-1>", lambda e: self._paint(e, 1.0))
        self.canvas.bind("<B3-Motion>", lambda e: self._paint(e, 0.0))
        self.canvas.bind("<Button-3>", lambda e: self._paint(e, 0.0))
        self.root.bind("<space>", lambda e: self._clear())

        tk.Label(left, text="linke Taste zeichnen  ·  rechte Taste radieren  ·  Leertaste loeschen",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))

        tk.Button(left, text="Loeschen", command=self._clear, bg="#232833", fg=FG,
                  relief="flat", font=("Segoe UI", 10), padx=14, pady=6,
                  activebackground="#2e3442", activeforeground=FG).pack(anchor="w", pady=10)

        right = tk.Frame(self.root, bg=BG)
        right.grid(row=0, column=1, padx=(0, 20), pady=16, sticky="n")

        tk.Label(right, text="Was die drei Modelle erkennen", bg=BG, fg=FG,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 10))

        self.panels = {}
        for key, title, subtitle, colour in (
            ("q_sim", "Quantenschaltung (Simulation)",
             "8 Qubits · 256-dim Zustandsraum", ACCENT_Q),
            ("q_hw", "Quantenschaltung (hardwaretauglich)",
             "10 Qubits · lief echt auf ibm_fez", ACCENT_H),
            ("classical", "Klassische Schicht",
             "gleiche Breite · Kontrolle", ACCENT_C),
        ):
            self.panels[key] = self._make_panel(right, title, subtitle, colour)

        self._build_evidence(right)

    def _make_panel(self, parent, title: str, subtitle: str, colour: str) -> Dict:
        frame = tk.Frame(parent, bg="#1a1e26", highlightthickness=1,
                         highlightbackground="#2a2f3a")
        frame.pack(fill="x", pady=5)
        inner = tk.Frame(frame, bg="#1a1e26")
        inner.pack(fill="x", padx=12, pady=10)

        head = tk.Frame(inner, bg="#1a1e26")
        head.pack(fill="x")
        tk.Label(head, text=title, bg="#1a1e26", fg=colour,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        acc = tk.Label(head, text="", bg="#1a1e26", fg=MUTED, font=("Segoe UI", 9))
        acc.pack(side="right")

        tk.Label(inner, text=subtitle, bg="#1a1e26", fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")

        body = tk.Frame(inner, bg="#1a1e26")
        body.pack(fill="x", pady=(6, 0))
        digit = tk.Label(body, text="—", bg="#1a1e26", fg=FG, font=("Segoe UI", 34, "bold"),
                         width=2)
        digit.pack(side="left", padx=(0, 12))
        bars = tk.Canvas(body, width=280, height=64, bg="#1a1e26", highlightthickness=0)
        bars.pack(side="left")
        return {"digit": digit, "bars": bars, "acc": acc, "colour": colour}

    def _build_evidence(self, parent):
        ev = self.bundle["evidence"]
        if not ev:
            return
        frame = tk.Frame(parent, bg="#1a1e26", highlightthickness=1,
                         highlightbackground="#2a2f3a")
        frame.pack(fill="x", pady=(10, 0))
        inner = tk.Frame(frame, bg="#1a1e26")
        inner.pack(fill="x", padx=12, pady=10)
        tk.Label(inner, text="Belege der Laeufe auf echter Quantenhardware",
                 bg="#1a1e26", fg=FG, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(inner, text=f"Prozessor {ev['backend']} ({ev['processor_qubits']} Qubits)",
                 bg="#1a1e26", fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        for run in ev.get("runs", []):
            text = (f"{run['label']}:  Job {run['job_id']}   "
                    f"Uebereinstimmung {run['agreement']:.0%}   "
                    f"Signalstaerke {run['damping']:.0%}")
            tk.Label(inner, text=text, bg="#1a1e26", fg=MUTED,
                     font=("Consolas", 8)).pack(anchor="w")
        tk.Label(inner, text="Job-IDs pruefbar auf quantum.ibm.com",
                 bg="#1a1e26", fg=MUTED, font=("Segoe UI", 8, "italic")).pack(anchor="w", pady=(4, 0))

    # -------------------------------------------------------------- Zeichnen
    def _paint(self, event, value: float):
        cx, cy = event.x / CELL, event.y / CELL
        radius = int(math.ceil(BRUSH))
        for gy in range(max(0, int(cy) - radius), min(28, int(cy) + radius + 1)):
            for gx in range(max(0, int(cx) - radius), min(28, int(cx) + radius + 1)):
                dist = math.hypot(gx + 0.5 - cx, gy + 0.5 - cy)
                if dist <= BRUSH:
                    strength = 1.0 if dist <= BRUSH * 0.55 else 0.55
                    if value > 0:
                        self.grid[gy, gx] = max(self.grid[gy, gx], strength)
                    else:
                        self.grid[gy, gx] = 0.0
        self._redraw()
        self._predict()

    def _redraw(self):
        self.canvas.delete("px")
        for gy in range(28):
            for gx in range(28):
                v = self.grid[gy, gx]
                if v > 0.01:
                    shade = int(v * 255)
                    colour = f"#{shade:02x}{shade:02x}{shade:02x}"
                    self.canvas.create_rectangle(
                        gx * CELL, gy * CELL, (gx + 1) * CELL, (gy + 1) * CELL,
                        fill=colour, outline="", tags="px")

    def _clear(self):
        self.grid[:] = 0.0
        self._redraw()
        self._predict()

    # ------------------------------------------------------------ Vorhersage
    def _centred_input(self) -> Optional[torch.Tensor]:
        """MNIST-Ziffern sind zentriert und groessennormiert. Ohne dieselbe
        Vorverarbeitung passt eine frei gezeichnete Ziffer nicht zum Training."""
        if self.grid.sum() < 1e-3:
            return None
        ys, xs = np.nonzero(self.grid > 0.05)
        top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
        crop = self.grid[top:bottom + 1, left:right + 1]

        height, width = crop.shape
        scale = 20.0 / max(height, width)          # MNIST: Ziffer in 20x20-Box
        new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
        ys_idx = (np.linspace(0, height - 1, new_h)).astype(int)
        xs_idx = (np.linspace(0, width - 1, new_w)).astype(int)
        resized = crop[np.ix_(ys_idx, xs_idx)]

        out = np.zeros((28, 28), dtype=np.float32)
        oy, ox = (28 - new_h) // 2, (28 - new_w) // 2
        out[oy:oy + new_h, ox:ox + new_w] = resized
        return torch.tensor(out.reshape(1, -1), dtype=torch.float32)

    def _predict(self):
        x = self._centred_input()
        acc = self.bundle["accuracy"]
        params = self.bundle["parameters"]
        for key in ("q_sim", "q_hw", "classical"):
            panel = self.panels[key]
            panel["acc"].config(text=f"{acc[key]*100:.1f}% · {params[key]} Parameter")
            if x is None:
                panel["digit"].config(text="—")
                panel["bars"].delete("all")
                continue
            with torch.no_grad():
                logits = self.bundle[key](x)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
            panel["digit"].config(text=str(int(probs.argmax())))
            self._draw_bars(panel["bars"], probs, panel["colour"])

    def _draw_bars(self, canvas: tk.Canvas, probs: np.ndarray, colour: str):
        canvas.delete("all")
        width, height = 280, 64
        slot = width / 10
        for d in range(10):
            h = float(probs[d]) * (height - 14)
            x0 = d * slot + 3
            x1 = (d + 1) * slot - 3
            canvas.create_rectangle(x0, height - 12 - h, x1, height - 12,
                                    fill=colour if probs[d] > 0.05 else "#2a2f3a",
                                    outline="")
            canvas.create_text((x0 + x1) / 2, height - 5, text=str(d),
                               fill=MUTED, font=("Segoe UI", 7))


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
