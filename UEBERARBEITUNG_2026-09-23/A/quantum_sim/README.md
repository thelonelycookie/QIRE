# quantum_sim

Eigener, von Grund auf selbst geschriebener Quantenschaltkreis-Simulator
(kein Wrapper um Qiskit/PennyLane) — neues, eigenständiges Kapitel neben der
klassischen QIRE-Simulation in `qire_ai/`. Ziel: echtes Verständnis der
Quantenmechanik hinter Quantencomputing aufbauen (Zustandsvektoren, unitäre
Gates, Born-Regel), bevor mit echter Quantenhardware verglichen wird.

## Funktionsweise

Ein Zustand von n Qubits sind 2^n komplexe Amplituden. Hier gespeichert als
Tensor der Form `(2,)*n` (Achse i = Qubit i), nicht als flacher Vektor. Ein
Gate auf k Qubits ist eine 2^k×2^k unitäre Matrix; sie wird per
`np.tensordot` nur auf die betroffenen Achsen angewendet (Standardtechnik
echter Simulatoren, vermeidet den Bau einer vollen 2^n×2^n Matrix).

```python
from quantum_sim import Circuit

# Bell-Zustand: maximal verschränktes Paar
c = Circuit(2).h(0).cnot(0, 1)
c.print_state()          # |00>: 0.7071  |11>: 0.7071
c.measure_all(shots=1000) # {'00': ~500, '11': ~500}
```

## Validierung

`python -m quantum_sim.validate` prüft gegen bekannte, theoretisch berechenbare
Schaltungen (Bell-Zustand, GHZ-Zustand, Superposition, Toffoli, Rz(2π),
Stichproben-Häufigkeiten) — alle 9 Checks bestehen (Stand 2026-08-01).

## Bekannte Grenzen

- **Kein Quantenvorteil möglich**: läuft auf normaler Hardware, rechnet die
  Quantenmathematik exakt nach, kann selbst keine Beschleunigung zeigen.
- **Exponentielle Skalierung**: 2^n Amplituden. Auf einem normalen Laptop
  praktikabel bis ca. 20-22 Qubits (siehe Messung unten), danach Speicher-
  /Zeitbedarf schnell zu groß.

  | Qubits | Zeit (H auf jedes Qubit) | Speicher |
  |---|---|---|
  | 10 | 0.3 ms | <0.1 MB |
  | 15 | 11 ms | 0.5 MB |
  | 18 | 45 ms | 4.2 MB |
  | 20 | 220 ms | 16.8 MB |
  | 22 | 1140 ms | 67 MB |

- **Qubit-Reihenfolge-Konvention**: Qubit 0 = Tensor-Achse 0 (linkestes Bit in
  `bitstring()`). Andere Frameworks (z. B. Qiskit) drucken Qubit 0 oft als
  *rechtestes* Bit — beim späteren Vergleich mit echter Hardware unbedingt
  explizit gegenprüfen, nicht annehmen.
- Kein Rauschen/Dekohärenz modelliert (ideal, nicht wie echte Hardware).

## Qiskit-Cross-Validierung (erledigt)

`python -m quantum_sim.qiskit_bridge` baut Bell- und GHZ-Zustand zusätzlich
in Qiskit nach und vergleicht die Wahrscheinlichkeiten gegen diesen eigenen
Simulator — zwei unabhängige Implementierungen stimmen überein. Dabei fiel
direkt der oben erwähnte Konventionsunterschied auf: Qiskit meldet denselben
Zustand mit gespiegelter Bit-Reihenfolge (Qubit 0 rechts statt links). Wird
in `qiskit_bridge.to_quantum_sim_order()` korrigiert, nicht einfach ignoriert.

## Echte IBM-Quantenhardware (nächster Schritt, braucht deinen eigenen Account)

`qiskit_bridge.run_on_real_hardware()` ist vorbereitet, aber noch nicht
ausgeführt — das braucht deinen **eigenen** IBM-Quantum-Account (kostenlos,
10 Freiminuten/Monat auf echten 100+-Qubit-Prozessoren):

1. Account anlegen auf https://quantum.ibm.com (das musst du selbst tun)
2. API-Token im Dashboard kopieren
3. Als Umgebungsvariable setzen (nicht in eine Datei schreiben, die geteilt/committet wird):
   ```bash
   export IBM_QUANTUM_TOKEN="dein-token-hier"
   ```
4. Dann: `python -c "from quantum_sim.qiskit_bridge import *; print(run_on_real_hardware(bell_state_qiskit()))"`

Das verbraucht echte (kontingentierte) Quantenrechenzeit — deshalb nicht
automatisch ausgeführt, sondern erst wenn du bereit bist.
