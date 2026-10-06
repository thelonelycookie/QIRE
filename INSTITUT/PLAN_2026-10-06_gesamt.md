# Gesamtplan v2 — Neuausrichtung nach Etappe 1 und 2 (06.10.2026)

*Verfasser: Direktor (Opus). Grundlage: `recherche/2026-10-05_breit/01–12` und `recherche/2026-10-05_vertiefung/v1–v6`. Ersetzt `PLAN_2026-10-05_effizienz.md`.*
*Status: **freigegeben am 06.10.** (Malte: „nach eigenem Ermessen anfangen“). E1–E7 gemäß Empfehlung. Maltes Ziel: ein quanten-inspirierter Vorteil, also ein Quantenprinzip, das klassische Standardlösungen bei gleichem Budget schlägt.*

---

## 0. Kurzfassung (für Malte, 1 Minute)

1. **v3 ändert die Geschichte.** Eine *zufällige* orthogonale Matrix mit anschließendem Quadrat, ohne einen einzigen gelernten Winkel, ist so gut wie die trainierte QIRE-Zelle (MNIST 0,938 / 0,938; Fashion 0,834 / 0,837). Ebenso gut sind QIRE mit Zufallswinkeln, Fastfood (Hadamard × Vorzeichen) und die Clifford-Suche. **Der Gewinn kommt aus „dichter Mischung vor einem Quadrat“. Das ist klassisch: Es braucht keine Verschränkung und keine gelernten Winkel.**
2. **v6 bestätigt das mathematisch.** Jede Born-Auslese ist ein 256-dimensionaler Unterraum der quadratischen Formen in x/‖x‖. Für jedes Magie-Budget k ist der Kern damit im Sinne von Huang et al. dequantisiert (g ≈ 3,8, unabhängig von k).
3. **Folge:** Die Linie „Clifford-Suche / Magiewinkel / Maskenrang“ (I-005…I-036-v3) ist sauber gemessen. Als *Quantenaussage* trägt sie aber nicht mehr. Ehrlich lassen sich drei Dinge sagen:
   - (a) eine **Struktur- und Effizienzaussage**: dichte, billige Mischung plus Born-Quadrat, O(N log N), 0 gelernte Mischparameter;
   - (b) eine **Messaussage**: wie viel Information pro Schuss eine Born-Auslese tatsächlich liefert (Holevo-Wirkungsgrad);
   - (c) eine **Skalierungsaussage** zum log-d-Adapter: Wo sättigt er, und warum (Lie-Algebra)?
4. **Vorschlag:** Vier vorregistrierte Läufe in dieser Reihenfolge:
   - **P1** Gleichwertigkeitsbefund (Pflicht);
   - **P2** 100-Parameter-MNIST;
   - **P3** Adapter-Sättigung (PC);
   - **P4** Holevo-Wirkungsgrad mit Schussrauschen.
   
   Dazu kommen zwei kleinere Optionen (P5 Skalierung n = 10, P6 kopffreie Zelle).
5. **Deine Entscheidungen** stehen in §5. Die wichtigste (E1): Akzeptierst du, dass die Arbeit ehrlich „quanten-inspiriert“ wird und nicht „Quantenvorteil“ behauptet? Davon hängt ab, wie viel noch in die Clifford-Linie fließt.

---

## 1. Ehrliche Neuausrichtung

### 1.1 Was nach Etappe 2 steht (Erkundung, noch nicht vorregistriert)

| Frage | Antwort | Quelle |
|---|---|---|
| Woher kommt der QIRE-Gewinn? | Dichte orthogonale Mischung plus Quadrat. Haar ≈ QIRE ≈ Zufallswinkel ≈ Fastfood ≈ Suche (MNIST). Gelernte volle Orthogonale (Cayley 35k P.) bringt +1,5 pp. | v3 |
| Trägt der CNOT (Verschränkung)? | Ohne CNOT −1,3 / −2,0 pp. Fastfood ohne jede Quantenstruktur holt das aber zurück. Der CNOT wirkt also als **Mischung**, nicht als Verschränkung. | v3 |
| Ehrliche Referenz | Pixel-Softmax 0,910 / 0,818, nicht „Identität“ 0,900. Vorsprung von Mischung plus Quadrat: **+2,9 pp MNIST, +1,6 pp Fashion**. | v3 |
| Rang / Bent | Der Rangeffekt zeigt sich nur global (k = 8). Die Streuung innerhalb eines Rangs ist so groß wie der Effekt. Eine feste Bent-Form liegt etwa 2 pp unter der Suche. Satz „Rang 2h ⇒ flaches Walsh-Spektrum“ bewiesen. | v2, 09 |
| 100 Parameter | ≤ 100 P.: 65–70 % (mit 250–850 Bit Struktur). 110–118 P.: Zelle 76,4 % gegen klassisch 75,1 %, quadrierte Pixel ohne WHT 76,3 %. Kein Clifford-Vorsprung erkennbar. | v1, 05 |
| Holevo | χ der Klassenzustände = 0,98 Bit. Identität nutzt 37 % davon pro Schuss, Suche nur 16 %. 90 % Genauigkeit brauchen ≈ 550 Schüsse. Holevo bindet nicht, die Messbasis verschenkt viel. | v4, 01 |
| Quantenhaftigkeit | Dequantisiert für jedes k; klassische RBF-/Quadrat-Kerne schlagen QIRE bei n = 1.000. | v6, 02 |
| QLM-Skalierung | Nicht unendlich: Datenladen Θ(N), Auslese 1/ε², Trainierbarkeit gegen Simulierbarkeit. Die Sättigung des Adapters bestimmt die Lie-Algebra. | 06, 02, **v5** |
| Effizienz auf CPU | Die WHT-Zelle hat ~28× weniger Rechenoperationen, ist auf CPU aber 5–17× **langsamer** als BLAS. Ein Gewinn zeigt sich erst mit eigenem Kernel oder ASIC. Der Kopf kostet mehr als die Zelle. | 11 |

### 1.2 Neue Kernaussage (Entwurf, wird durch P1 belegt oder verworfen)

> „Eine Born-Auslese, also das Quadrat nach einer dichten unitären Mischung, ist ein starkes, parameterfreies Merkmal. Eine feste, zufällige oder strukturierte Mischung (Hadamard × Vorzeichen, O(N log N), 256 Bit) genügt; gelernte Winkel, Verschränkung und Magie tragen auf MNIST/Fashion nichts Messbares bei. Gegenüber Softmax auf Pixeln mit gleich vielen Kopfparametern bringt das +2,9 / +1,6 pp. Die Quantenphysik liefert die Idee und die Sprache (Born-Regel, Clifford-Gruppe, Holevo), aber keinen Rechenvorteil.“

**Was sich ändert:**
- Die Q-Stufen der bisherigen QIRE-Befunde werden ehrlich auf **Q2/Q1** gesetzt: exakte QM im Simulator, deren Wirkung klassisch erklärbar ist.
- „0 T-Gatter genügen“ bleibt als **Ressourcenaussage für fehlertolerante Hardware** stehen, nicht als Aussage über Quantennutzen.

**Was erhalten bleibt (und neu sein könnte):**
1. Der **Holevo-/Fano-Wirkungsgrad** als Gütemaß einer Born-Auslese. Die Literatursuche dazu ist noch dünn.
2. Der **log-d-Adapter** mit erklärter Sättigung.
3. Der **Satz Rang ⇒ Walsh-Flachheit** als Erklärbaustein.
4. Die **Kontrollbatterie** selbst: Viele QML-Arbeiten haben sie nicht.

### 1.3 Was nicht weiterverfolgt wird (vorerst)

- I-037 (Rang-Dosiskurve): Seit v2/v3 ist ihr Erklärwert gering. Status → „zurückgestellt“.
- Weitere Masken-/Bent-Suche als Hauptlinie.
- Hardware-Lauf mit Amplitudenkodierung (247–377 CZ). Höchstens als Prinzipbeweis, siehe E5.

---

## 2. Vorregistrierbare Läufe (priorisiert)

Für alle Läufe gilt:
- §3 der `UEBERGABE.md`: Lernrate je Arm mit `gittersuche_stabil`, Kriterien im Code vor dem Lauf, Vortest erzwingt jeden Urteilszweig, TOST für „gleich“, Holm nur über die angemeldete Familie, Rohdaten sofort sichern, Wiederaufnahme je Seed.
- **Seeds:** neue Blöcke ab 2000. Nicht verwendet werden 900–999 (Erkundung) und 1000 (bereits benutzt).
- Daten: 16×16, je Seed 6.000 Trainingsbilder und 5.000 offizielle Testbilder (wie v1–v4).
- **Parameterzählung (verbindlich ab jetzt):**
  - Trainierte reelle Zahlen,
  - **plus** Beschreibungslänge der festen/diskreten Wahl in Bit (Maske, Gruppierung, Seed-Auswahl),
  - **plus** ein Hinweis, falls eine zufällige feste Matrix verwendet wird. Die Matrix ist „gratis“ nur, wenn ihr Seed nicht ausgewählt wurde.

### P1 · I-038 „Struktur-Äquivalenz“ (Pflichtbefund) — Prio 1

- **Frage:** Ist QIRE gleichwertig zu klassischer dichter Mischung plus Quadrat?
- **Arme:**
  - Pixel-Softmax;
  - Identität²;
  - Haar²;
  - Haar ohne Quadrat (linear);
  - Fastfood-2 (H·D₂·H·D₁)²;
  - QIRE L = 4 mit Zufallswinkeln;
  - QIRE L = 4 gelernt;
  - QIRE ohne CNOT gelernt;
  - Clifford-Suche (gepaart neu gerechnet);
  - Cayley gelernt.
  
  Alle Arme bekommen denselben Kopf (2.570 P.).
- **Hypothesen (Familie für Holm: H-A…H-F):**
  - H-A: QIRE gelernt ≡ Haar² (TOST ±0,5 pp).
  - H-B: QIRE gelernt ≡ QIRE Zufallswinkel (TOST ±0,5 pp).
  - H-C: Fastfood-2 ≡ Haar² (TOST ±0,5 pp).
  - H-D: Haar² − Pixel-Softmax > 0 (Superiorität; erwartet +2,9 / +1,6 pp).
  - H-E: Quadrat trägt, also Haar² − Haar linear > 0.
  - H-F: Suche ≤ Haar² auf Fashion (einseitig).
  - **H-G (06.10., Maltes Ziel „quanten-inspirierter Vorteil“):** Haar² ≥ beste klassische Nichtlinearität nach derselben Mischung (ReLU, tanh, |z|, z² ohne Normierung) bzw. MLP mit gleich vielen Parametern. Der beste klassische Arm wird auf dem Tuning-Seed gewählt.
  - Cayley − QIRE wird nur berichtet, mit Parameterzählung.
- **Positivkontrolle, die feuern kann:** H-D und H-E (in v3 jeweils > 2 pp).
- **Negativkontrolle:** ein Zufallslabel-Arm (Gil-Fuster), erwartet ≈ 10 %.
- **Daten / Seeds:** MNIST und Fashion; 20 Seeds 2000–2019; Tuning-Seed 2099.
- **Laufzeit / Ort:** ≈ 60–90 min Container (4 Prozesse), mit Wiederaufnahme. Die Clifford-Suche kostet ≈ 6 min je Seed und Datensatz. Wird sie als Arm aufgenommen: +4 h. Option: Suche nur auf 10 Seeds oder **auf den PC** auslagern (E3).
- **Wert:** Das ist der ehrliche Pflichtbefund für die schriftliche Arbeit. Er schützt vor dem stärksten Jury-Einwand.

### P2 · I-039 „MNIST mit ~100 Parametern“ — Prio 1

- **Ziel (vorregistriert):** ≥ 70 % bei ≤ 110 trainierten reellen Parametern. Strukturbits werden getrennt berichtet.
- **Arme (gleiches Budget, Raten je Arm getunt):**
  - (a) Born-Gruppierung nach Haar- oder Fastfood-Mischung, nur 10 Bias, Gruppierung datengewählt;
  - (b) Gruppensummen plus 10×10-Kopf (110 P.);
  - (c) quadrierte Pixel ohne Mischung, gleicher Kopf (Ablation);
  - (d) klassisch: 5×5 → 3 tanh → 10 (118 P.);
  - (e) Fisher-Top-9-Pixel plus Softmax (100 P.).
  - Neu (f), **Klassenregister-Auslese:** 256 Born-Ausgänge auf 10 Klassen per fester Partition, *ohne* Kopf (Idee A2 aus 12), also 0 reelle Parameter plus Partitionsbits.
- **Hypothesen:**
  - H1: Bester Born-Arm ≥ 70 % (absolut, einseitig gegen 70).
  - H2: Born-Arm (b) ≡ oder > klassisch (d) bei gleicher Zahl reeller Parameter (TOST ±1 pp / Superiorität).
  - H3: Mischung trägt, also (b) − (c) > 0.
  - **Bits pro Prozentpunkt** wird als zweites Maß berichtet.
- **Seeds:** 2020–2039 (20). **Laufzeit / Ort:** ≈ 30–45 min Container.
- **Realistische Erwartung (v1):** H1 knapp ja (70–76 %), H2 gleich, H3 offen. 80 % mit 100 P. ist nicht zu erwarten; das wird vorher offen gesagt.

### P3 · T-006 „Adapter-Sättigung“ (QLM-Skalierung) — Prio 1, **PC**

- **Frage:** Wächst der Gewinn des log-d-Adapters (T-005: L^0,55 bis L = 10) weiter, oder sättigt er? Und liegt die Sättigung dort, wo die Lie-Algebra bzw. der Jacobi-Rang (v5) sie vorhersagt?
- **Arme:** QIRE-Adapter L ∈ {1, 2, 5, 10, 20, 40} auf LLäMmlein-120M wie T-005. Kontrollen:
  - Kronecker-Drehung ohne CNOT, gleiches L (Mischungskontrolle);
  - LoRA r = 1 als Referenz;
  - Zufallswinkel, eingefroren.
- **Hypothesen (vor dem Lauf festgelegt):**
  - H1: Der Gewinn bei L = 40 ist höchstens 10 % größer als bei L = 20, also Sättigung.
  - H2: Ein Modell „Potenzgesetz + Plateau“ passt besser als ein reines Potenzgesetz (ΔAIC > 4).
  - H3: Das Plateau liegt dort, wo der Jacobi-Rang (v5) aufhört zu wachsen.
- **Seeds:** 3 Seeds 2040–2042 je L. **Laufzeit / Ort:** PC, geschätzt 8–15 h insgesamt; die genaue Zahl folgt aus den T-005-Zeiten vor der Freigabe. Grund für den PC: Modell und Daten liegen dort, die Läufe sind lang.
- **Wert:** Direkte, ehrliche Antwort auf „Ist ein QLM unendlich skalierbar?“: **Nein. Hier sättigt es, und das ist der Grund.**

### P4 · I-040 „Holevo-Wirkungsgrad“ — Prio 2

- **Frage:** Wie viel der Holevo-Größe χ nutzt eine Born-Auslese pro Schuss, und was kostet das an Genauigkeit unter Schussrauschen?
- **Arme:** Identität; Haar; Fastfood; Bent-4×4; Suche. Dazu ein **schussbewusster Kopf**, trainiert auf S ∈ {10, 30, 100, 300, 1000} Schüssen, und als Kontrolle ein Kopf, der auf exakten Wahrscheinlichkeiten trainiert und auf Schüssen getestet wird.
- **Hypothesen:**
  - H1: I(Klasse; 1 Schuss) der Identität > Haar und > Suche (gepaart, Wilcoxon).
  - H2: Ein schussbewusstes Training schlägt das exakte Training bei S ≤ 100.
  - H3: Ein höherer Wirkungsgrad η = I/χ sagt den Schussbedarf für 90 % voraus (Rangkorrelation über die Arme).
- **Seeds:** 2050–2059 (10; Informationsgrößen sind fast deterministisch). **Laufzeit / Ort:** ≈ 30–60 min Container.
- **Wert:** Beantwortet „Holevo umgehen?“ ehrlich. Umgehen lässt sich die Grenze nicht. Sie bindet aber auch nicht; bindend ist die Messbasis, und die lässt sich messen und verbessern. Ein möglicherweise neues Gütemaß. Die Neuheitsprüfung steht noch aus (E4).

### P5 · I-041 „Effizienz im Großen: Skalierung n = 8 → 10“ — Prio 2 (optional)

- **Frage:** Bleibt „Fastfood² ≈ Haar² > Pixel“ bei 32×32 (n = 10, 1.024 Merkmale) bestehen? Und wie verhalten sich Rechenoperationen und gemessene Zeit, wenn N wächst? Zum Vergleich: T-001 fand den Übergang bei N ≈ 4–16k.
- **Arme:** Pixel-Softmax, Haar², Fastfood-2². Dazu ein **Zählmodell plus Zeitmessung**: numpy-FWHT gegen BLAS, N = 256 … 16.384.
- **Seeds:** 2060–2069. **Laufzeit / Ort:** ≈ 1–2 h Container (Haar 1024² ist noch billig).
- **Wert:** Antwort auf „Effizienz im Großen“ ohne Quantenrhetorik. Auf CPU erwarten wir ehrlich: keinen Zeitgewinn unter N ≈ 4k.

### P6 · Ideen ohne Lauf (Prio 3, nur nach Literaturprüfung)

- A15 **Matchgate-Zelle**: Quantenzelle, die polynomiell in n skaliert; die bessere Antwort auf „skalierbar?“. Sie muss sich aber gegen „orthogonal + Quadrat“ (P1) behaupten.
- A9 **D4-/XOR-Äquivarianz** im Clifford-Raum.
- 11 **Hadamard-Rotation für Quantisierung** (QuaRot-Linie). Billig, aber bekanntes Genre: höchstens als Werkzeug kennzeichnen.

---

## 3. Reihenfolge und Zeitplan

| Schritt | Was | Wer | Ort | Dauer |
|---|---|---|---|---|
| 0 | Freigabe dieses Plans, Entscheidungen §5 | Malte | — | — |
| 1 | Module P1, P2 bauen und vortesten (alle Urteilszweige) | Sonnet | Container | ~1 h |
| 2 | Vorregistrierung P1, P2 committen → **Freigabe Lauf** → rechnen | Sonnet / Opus | Container | ~2 h |
| 3 | P3 vorbereiten (Laufzeit aus T-005, Skript für `institut.cmd`) → Freigabe → Malte startet | Sonnet / Malte | PC | 8–15 h |
| 4 | P4 (und ggf. P5) analog | Sonnet | Container | ~1–3 h |
| 5 | LABORBUCH, Kernaussage, Q-Stufen in `UEBERGABE.md` §0 neu | Opus | — | — |

Ziel: P1–P4 bis etwa 20.10. fertig. Damit bleibt Luft bis zur Anmeldung am 30.11.

---

## 4. Risiken

- **P1 kann H-A widerlegen.** Ist QIRE doch messbar besser als Haar, ist das *gut*: Dann trägt die Struktur doch etwas. Beide Ausgänge sind berichtbar.
- **P3:** Sättigt der Adapter nicht bis L = 40, ist das ebenfalls berichtbar: „bis L = 40 keine Sättigung, Skalierungsexponent …“. Es zieht aber längere Läufe nach sich.
- **Neuheit:** Fast alle Literaturangaben stammen aus Suchauszügen. Bevor etwas als neu formuliert wird (Holevo-Wirkungsgrad, log-d-Adapter), braucht es eine echte Websuche (E4).
- **Container-Abbrüche:** Wiederaufnahme je Seed ist Pflicht, Zwischenstände werden committet.

---

## 5. Entscheidungen für Malte

- **E1 Rahmung.** Ich empfehle, „quanten-inspirierte Struktur plus ehrliche Messung“ als Kernaussage zu akzeptieren und die Clifford-/Magielinie nur noch als Ressourcenaussage (T-Kosten) zu führen.
  - Alternative: weiter nach echter Quantenressource suchen (Matchgate, Quantendaten). Das ist riskanter und hat wenig Aussicht bis November.
- **E2 Freigabe der Läufe.** Welche von P1–P5 (empfohlen: P1, P2, P3, P4; P5 optional)?
- **E3 Clifford-Suche in P1.** Als gepaarter Arm (+4 h Container bzw. PC) oder die I-036-v3-Werte ungepaart übernehmen? Empfehlung: gepaart auf 10 Seeds.
- **E4 Websuche.** Das Limit `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` in der Umgebung anheben, damit vor P4/P3 eine echte Neuheitsprüfung möglich ist?
- **E5 Hardware.** IBM-Prinzipbeweis (10 min/Monat) jetzt einplanen oder streichen? Empfehlung: streichen bzw. erst im November, nur als Demo.
- **E6 Rechenort.** Lange Läufe (P3, ggf. Suche) auf deinem PC: Bist du bereit, sie über `institut.cmd` zu starten?
- **E7 100-Parameter-Ziel.** Ist „≥ 70 % bei ≤ 110 P. mit ehrlicher Bitzählung“ als Ziel in Ordnung, statt 80 %+?
