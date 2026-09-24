# MPI-Treffen — alles an einem Ort

Stand 24. September 2026. Reihenfolge = Reihenfolge im Gespräch.

| Ordner / Datei | Wofür | Ausdrucken? |
|---|---|---|
| `01_Gespraech/MPI_GESPRAECH.pdf` (auch .docx, .md) | Dein Leitfaden: was du machst, was du beantragst, Fragen an sie, Fragen von ihnen mit Antworten, Glossar | **ja** (7 Seiten) |
| `02_Cluster_und_Rechenbedarf/Cluster_Anleitung.docx` (README.md) | Wie T-002 auf ihrem Rechner läuft (SLURM-Job-Array), Annahmen, Fragenliste für die Technik | ja, für die IT-Leute |
| `02_Cluster_und_Rechenbedarf/T-002_Stand_und_Entscheidungen.md` | Rechenbedarf im Detail (59–74 GPU-h ohne, 74–93 mit Zusatzarm), offene Entscheidungen | nein, zum Nachschlagen |
| `02_…/t002_zelle.sbatch`, `t002_steuerung.sh` | die fertigen Cluster-Skripte | nein |
| `03_Literatur_und_Alleinstellung/…pdf` (auch .docx, .md) | Einordnung in die Forschung, was an deiner Arbeit neu ist, Zukunftsrichtungen | optional (8 Seiten) |
| `04_Demo/Demo_starten.bat` | Doppelklick startet die Stand-Demo (Ziffer zeichnen) — **nur auf diesem PC** | – |
| `04_Demo/Demo_Anleitung.md` | Bedienung und Sätze für Gespräche am Stand | nein |
| `05_Code_fuer_den_Cluster.zip` | der Code, den der Cluster braucht (0,5 MB) | – |
| `06_Fuer_den_Cluster_Daten/` | Korpus: `train.npy` (758 MB), `val.npy`, `bpe_4096.json` — falls die Rechenknoten kein Internet haben | – |

## Zum Mitnehmen

- Ausdruck von `01_Gespraech/MPI_GESPRAECH.pdf`
- USB-Stick mit diesem ganzen Ordner (~765 MB) — dann kannst du Code und Daten direkt übergeben
- Laptop, falls du die Demo zeigen willst: `Demo_starten.bat` läuft nur auf diesem PC. Für einen
  Laptop müsste dort Python mit PyTorch installiert werden (siehe `UMGEBUNG.md` im Zip).

## Drei Dinge, die sich seit gestern geändert haben

1. **A und B sind getrennt.** A (Quanten) ist das Hauptprojekt, B nur Nebenbefund (Abschnitt 2.7).
2. **Rechenbedarf T-002 ist ~60–75 GPU-Stunden** statt 30–35 — weil die Lernratensuche jetzt mit
   vollem Budget läuft, wie dein eigenes Protokoll es verlangt. Auf 8 GPUs gleichzeitig ~7–11 h.
3. **Die Schaltung selbst ist schon veröffentlicht** (Quantum-PEFT, ICLR 2025). Dein Beitrag ist die
   Prüfmethodik, die Born-Regel-Zerlegung und der GPU-Bündler — Abschnitt 7 im Leitfaden.
