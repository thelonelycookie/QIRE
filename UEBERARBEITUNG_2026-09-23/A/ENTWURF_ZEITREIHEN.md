# Entwurf: Q-019 — Zeitreihen, die zweite KI-Art

Angelegt 27. August 2026, während Q-011 läuft. **Nichts hier ist gemessen.**

---

## Warum ausgerechnet Zeitreihen

Alle bisherigen Anwendungen des Projekts fragen: *Ist die Struktur genauso gut
wie eine dichte Schicht, nur billiger?* Die Antwort war bestenfalls
„gleichwertig". Zeitreihen sind die einzige Anwendung in
`BEFUNDE_UND_RICHTUNGEN.md`, bei der ein **inhaltlicher** Vorteil zu erwarten
ist:

> Die Butterfly-Struktur *ist* die schnelle Fouriertransformation. Für
> periodische Signale ist das die natürliche Basis, und dort sollte die
> Architektur nicht nur gleich gut, sondern besser sein als eine dichte Schicht.

> **Nachtrag 27. August, nachts — dieser Satz ist falsch.** Die Butterfly-Schicht hat **geteilte** Winkel (40), die schnelle Fouriertransformation braucht je Paar einen **eigenen** Drehfaktor (5.120). Gemeinsam ist beiden das **Verbindungsmuster** und damit die Kostenordnung O(d log d) — nicht die Transformation. Im Q-019-Vorlauf erreicht `butterfly` 0,686, die echte Fouriertransformation 0,804. Der Satz bleibt hier stehen, weil er den Verlauf dokumentiert.


Das ist eine stärkere Behauptung als alles bisher Geprüfte — und deshalb auch
leichter zu widerlegen.

## Die Aufgabe

Synthetische Signale der Länge d = 256 = 2⁸: Summe weniger Sinusanteile mit
zufälliger Frequenz, Phase und Amplitude, dazu Rauschen. Die Klasse ist das
**Frequenzband, in dem der stärkste Anteil liegt** (8 Bänder, 8 Klassen).

Warum diese Aufgabe: Sie ist im Frequenzraum trivial und im Zeitraum schwer.
Ein Modell, das die richtige Basis mitbringt, muss fast nichts lernen; eines
ohne sie muss die Fouriertransformation aus den Daten rekonstruieren.

## Die Arme

| Arm | Mittelschicht | Parameter |
|---|---|---|
| `butterfly` | orthogonale Butterfly + quadratische Auslese | **40** |
| `dense` | dichte 256×256 + gleiche Auslese | 65.536 |
| `fft_fest` | **echte** FFT, untrainiert, + gleiche Auslese | 0 |
| `narrow` | dichte 256→64→256 | 32.768 |

`fft_fest` ist neu und wichtig: Wenn die Butterfly-Struktur ihren Vorteil
daraus zieht, dass sie *fourierähnlich* ist, dann muss die echte FFT mindestens
so gut sein. Ist sie es nicht, liegt der Vorteil woanders — und die
Fourier-Erzählung wäre falsch.

## Die Kontrolle, die den Befund erst tragfähig macht

Dieselben Signale, aber mit einer **festen zufälligen Vertauschung der
Zeitachse**. Alle Werte bleiben, die Klasse bleibt, die Periodizität ist
zerstört.

Wenn der Vorteil der Butterfly-Struktur strukturell ist, **muss er dort
verschwinden.** Bleibt er, kommt er nicht von der Fourier-Verwandtschaft,
sondern von etwas anderem — und die Deutung wäre hinfällig.

Das ist exakt die Kontrolle, mit der B-004 Monarchs Vorteil eingeordnet hat:
+0,0115 auf Bildern, verschwunden bei permutierten Pixeln (p=0,23).

## Vorregistrierung

Balancierte Genauigkeit, Lernrate je Arm auf eigenen Tuning-Seeds, 15 Seeds,
beide Datenbedingungen.

> **PRIMÄR — die Wechselwirkung.** Der Abstand (butterfly − dense) ist auf
> periodischen Signalen **größer** als auf permutierten. Wilcoxon über die
> seedweise gepaarten Differenzen der Differenzen.
>
> **P2** — auf periodischen Signalen ist (butterfly − dense) positiv.
>
> Holm über genau diese beiden.

Der Primärtest ist bewusst die Wechselwirkung und nicht der einfache
Vergleich. „Butterfly schlägt dicht" allein wäre auch mit einem
Rechenartefakt erklärbar; „Butterfly schlägt dicht **nur dort, wo die Struktur
zu den Daten passt**" ist es nicht.

**H1** beide halten → die Struktur bringt einen inhaltlichen Vorteil, und
zwar nachweislich aus der Passung zur Datenstruktur. Das wäre der erste
Befund des Projekts, der über „gleichwertig, aber billiger" hinausgeht.

**H1+** zusätzlich ist `butterfly` nicht schlechter als `fft_fest` → die
gelernte Näherung erreicht die echte Transformation.

**H0** kein Wechselwirkungseffekt → der Vorteil ist nicht strukturell, und die
Fourier-Erzählung fällt.

**HU** weder noch → erreichte Schranke berichten.

**AUFLÖSUNGSPRÜFUNG:** `dense` muss `narrow` auf periodischen Signalen
schlagen. Sieht der Aufbau diesen groben Kapazitätsunterschied nicht, kann er
den feineren Strukturunterschied erst recht nicht sehen.

**BODENPRÜFUNG:** Mindestens ein Arm muss deutlich über Rateniveau (0,125 bei
8 Klassen) liegen. Sonst misst die Zelle Trainingsversagen.

## Was vor dem Lauf zu prüfen ist

1. Vortest auf synthetischen Auswertungsdaten: **jeder** Ausgang muss feuern,
   einschließlich „Wechselwirkung geht in die falsche Richtung".
2. Sättigungsprüfung über **alle** Arme — die Aufgabe darf für keinen trivial
   sein. Bei `fft_fest` ist das eine reale Gefahr: Wenn die echte FFT die
   Aufgabe perfekt löst, trennt nichts mehr.
3. Erreichbarkeit der Signifikanz: 15 Seeds → kleinstmögliches Wilcoxon-p
   6,1·10⁻⁵, nach Holm über zwei 1,2·10⁻⁴. Reicht.
4. Kein `stats.spearmanr` im Auswertungspfad — hier wird nichts über wenige
   Stützstellen korreliert, also entfällt die Fehlerklasse aus Q-011.

---

## Nachtrag vom 27. August, nachts — beim Bauen geändert

Das Modul steht als `quantum_sim/zeitreihen.py`. Der Entwurf oben bleibt
unverändert stehen; hier steht, was beim Bauen anders wurde und warum. Alles
davon ist **vor** dem Lauf entstanden, es existieren noch keine Messzahlen.

**1. Ein Ausgang fehlte — gefunden vom Vortest.** Wenn die Wechselwirkung hält,
aber P2 fällt, landete der Lauf im Zweig „weder Wechselwirkung noch
Gleichwertigkeit belegt" — obwohl die Wechselwirkung gerade belegt *ist*. Das
ist dieselbe Fehlerklasse wie der fehlende H0−-Zweig in Q-011: ein Urteil, das
den eigenen Daten widerspricht. Der Fall ist jetzt ein eigener Zweig und
inhaltlich der interessanteste Nebenausgang: Er heißt, die Struktur *gewinnt*
auf periodischen Signalen nicht, sondern *verliert* auf permutierten.

**2. H0 ist nur bei kleiner Streuung erreichbar — ausgerechnet, nicht gehofft.**
TOST gegen ±0,005 besteht bei 15 Seeds nur, wenn die seedweise Streuung der
Wechselwirkung unter

    sd = 0,005 · √15 / 1,7613 = 0,0110

liegt. Ist sie größer, **kann** H0 nicht feuern, und der Lauf liefert nur H1,
H0− oder HU. Das ist kein Mangel, aber es muss vorher dastehen — sonst wird ein
unerreichbarer Ausgang hinterher als „nicht eingetreten" gelesen.

**3. Punkt 3 des Entwurfs deckte nur die halbe Regel 2 ab.** Er prüft, ob
Signifikanz *rechnerisch* erreichbar ist (2/2¹⁵ = 6,1·10⁻⁵). Seit dem
27. August verlangt Regel 2 zusätzlich, ob der *erwartete Effekt bei der
vorhandenen Streuung* nachweisbar ist. Dafür gibt es jetzt `macht()`: ein
Vorlauf über fünf Seeds, aus dem die Streuung der Wechselwirkung geschätzt und
die Nachweisgrenze bei 80 % Macht berechnet wird. B-007 ist genau an diesem
zweiten Teil gescheitert, drei Wochen lang unbemerkt.

**4. Der Vortest prüft den Zweig, nicht die Marke.** Vier verschiedene Ausgänge
heißen „HU" (Auflösung, Boden, geteilt, unentschieden). Würden zwei davon
verschmelzen, sähe ein Vortest, der nur auf die Marke schaut, nichts. Er
vergleicht deshalb zusätzlich den Urteilstext.

**5. Jedes Szenario läuft über fünf Rauschziehungen.** Der erste Entwurf des
Vortests fiel bei Szenario G durch — aber nicht, weil die Auswertung falsch
war, sondern weil der *Prüfstand* falsch war: `dense` bekam gar kein eigenes
Rauschen, und der gemeinsame Seed-Anteil kürzt sich in der gepaarten Differenz
weg. Das „schwache, verrauschte" Szenario war in Wahrheit ein starkes, sauberes.
Ein falsch gebauter Prüfstand ist genauso schädlich wie ein falsch gebauter
Test. Bei Szenario F zeigte sich danach, dass das Urteil in zwei von fünf
Ziehungen kippte — grenzwertig, also kein Urteil. Beides ist behoben.

**6. Nyquist ist aus den Signalfrequenzen genommen.** Die Frequenzen laufen
1..127 statt 1..128. Grund: In der reellen Fourierbasis liegt der
Nyquist-Anteil auf Index 1 und würde von der Auslese Band 0 zugeschlagen statt
Band 7. Das betrifft 1 von 128 Frequenzen — klein, aber ein Etikettenfehler.
Band 7 hat dadurch 15 statt 16 Frequenzen; gemessen wird balancierte
Genauigkeit, das trägt das.

**7. Eine Asymmetrie, die benannt gehört.** `butterfly` und `fft_fest` sind
normerhaltend, `dense` und `narrow` nicht. Der lineare Kopf hebt jeden festen
Maßstab auf, aber die Gradienten sind unterschiedlich groß, und das wirkt wie
eine andere Lernrate. Aufgefangen wird das durch die Lernratensuche je Arm
(Regel 1) — ohne sie wäre der Vergleich verzerrt, und zwar unbekannt in welche
Richtung.

**8. Die Lernratensuche läuft mit vollem Budget und verlängert in beide
Richtungen.** Ersteres ist Regel 1(c) seit heute Nachmittag. Letzteres ist die
Ergänzung von heute Nacht: `unitarity_depth.py` verlängert nur nach oben, und
genau ein Arm am *unteren* Rand ist es, der Q-015 auf Fashion-MNIST entwertet.

### Stand

- [x] Modul gebaut
- [x] Vortest bestanden — 8 Szenarien × 5 Ziehungen, Marke **und** Zweig geprüft
- [x] Bauteile verifiziert: Fourierbasis orthonormal (4,7·10⁻⁸), butterfly und
      fft_fest normerhaltend, Butterfly hat exakt 40 Parameter
- [x] Sättigungsprüfung gefahren — nichts gesättigt, Auflösungsabstand
      `dense` − `narrow` = +0,128, Kontrolle trägt (butterfly und fft_fest
      fallen auf permutiert auf Rateniveau)
- [x] Gitterverlängerung in beide Richtungen getestet (5/5)
- [ ] Machtanalyse (Regel 2b) — `python -m quantum_sim.zeitreihen macht`
- [ ] Lauf (~2,5–3 h, gehört hinter Q-011)
