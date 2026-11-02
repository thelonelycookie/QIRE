#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Überführt den Literaturbestand in WISSEN/quellen.json (Welle 0).

Reihenfolge: (1) LITERATUR/quellen_daten.py + neuheit_index.json (+ literatur.bib für DOI/URL),
(2) recherche/zyklus/STROM.md, REAKTOR.md, ARCHIV.md, (3) übrige recherche/**/*.md.
Nichts wird erfunden: Was in den Dateien fehlt, bleibt leer/null/"unklar".
Aufruf: python3 WISSEN/werkzeuge/ueberfuehren.py
"""
import glob, json, os, re, sys, difflib
from collections import OrderedDict, defaultdict

WURZEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REC = os.path.join(WURZEL, "INSTITUT", "recherche")
sys.path.insert(0, os.path.join(WURZEL, "LITERATUR"))
import quellen_daten as qd  # noqa: E402

STANDARD_DATUM = "2026-11-02"
STAT = {"zusammengefuehrt": 0, "id_treffer": 0, "titel_treffer": 0}

# ------------------------------------------------------------------ Hilfen
def rel(p):
    return os.path.relpath(p, WURZEL)

def datei_datum(pfad):
    """Bestes Datum zur Datei: aus Pfad (YYYY-MM-DD), sonst DD.MM.2026 im Kopf, sonst Standard."""
    m = re.search(r"(20\d\d-\d\d-\d\d)", pfad)
    if m:
        return m.group(1)
    try:
        kopf = open(pfad, encoding="utf-8").read(1500)
    except OSError:
        return STANDARD_DATUM
    m = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(2026)\b", kopf)
    if m:
        return "%s-%02d-%02d" % (m.group(3), int(m.group(2)), int(m.group(1)))
    return STANDARD_DATUM

def norm_titel(t):
    t = re.sub(r"\(?arxiv[: ]*[\w./-]+\)?", "", t.lower())
    return re.sub(r"[^a-z0-9äöüß]", "", t)

def jahr_aus_id(i):
    m = re.match(r"arXiv:(\d\d)(\d\d)\.", i)
    if m and 7 <= int(m.group(1)) <= 26 and 1 <= int(m.group(2)) <= 12:
        return 2000 + int(m.group(1))
    m = re.match(r"arXiv:[a-z-]+/(\d\d)(\d\d)\d+", i)
    if m:
        y = int(m.group(1))
        return (1900 if y > 50 else 2000) + y
    return None

def md_rein(s):
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = s.replace("**", "").replace("`", "").replace("\\|", "/")
    s = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", s)
    s = re.sub(r"\s*\|\s*", "; ", s).strip("; ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def saetze(s, n=2, maxl=320):
    teile = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"(])", s)
    out = " ".join(teile[:n]).strip()
    if len(out) > maxl:
        out = out[:maxl].rsplit(" ", 1)[0] + " ..."
    return out

QUANT = re.compile(r"quant(?!isier|ization|ized)|qubit|born|clifford|verschr|stabilizer|magie|magic|shot|holevo|schüsse|vqe|\bpqc", re.I)
QUANT_TITEL = re.compile(r"quant(?!isier|ization|ized|ile)|qubit|born |born-|clifford|stabilizer|magic|holevo|shadow|vqe|variational|\bqml\b|lambeq|dequantis|rydberg|schrödinger|schrodinger", re.I)
ANALOG = re.compile(r"hadamard|walsh|tensor|\bmps\b|\bmpo\b|matrix product|butterfly|unitär|unitar|unitary|orthogon|kronecker|rotation|rotate|givens|cayley|random feature|fastfood|monarch|renormal|wavelet|fourier|interferenz", re.I)

def quantenbezug(titel, text):
    if QUANT_TITEL.search(titel):
        return "direkt"
    if QUANT.search(titel + " " + text) and re.search(r"quanten-inspir|quantum-inspired|inspir", titel + text, re.I):
        return "analogie"
    if ANALOG.search(titel):
        return "analogie"
    return "keiner"

FELDER = OrderedDict([
 ("K1", r"quantiz|quantisier|\bptq\b|\bqat\b|gptq|awq|\brtn\b|quip|low-?bit|niedrigbit|gewichtsquant|weight quant"),
 ("K2", r"kv[- ]?cache|\bkv\b|aktivierungsquant|activation quant|\bw\d+a\d+|smoothquant|kivi|a4\b|aktivierungs"),
 ("K3", r"hadamard|rotation|rotat|butterfly|givens|orthogon|unitär|unitar|cayley|quarot|spinquant|walsh|fwht|\bboft\b|\boft\b"),
 ("K4", r"codebook|codebuch|lattice|gitter|vector quant|vektorquant|lloyd|trellis|qtip|aqlm|quip#|k-means|\be8\b|polar ?quant|turboquant"),
 ("K5", r"tensor|\bmps\b|\bmpo\b|matrix product|compactifai|dmrg|tensorgpt|\btt[- ]"),
 ("K6", r"quanten-inspir|quantum-inspired|dequantis|clifford|magie|magic|stabilizer|born[- ]|\bqml\b|quantum machine|variational|\bvqe\b|\bpqc|holevo|shadow|graph.?state|qubit|quantum|quanten|gottesman|qnlp|lambeq|quixer|tang"),
 ("K7", r"fastfood|monarch|structured matri|strukturiert|kronecker|pixelfly|kaleidoscope|hadamard-?mlp|circulant|toeplitz|butterfly|low-?rank|niedrigrang|\blora\b|adapter|\bpeft\b|structured transform|sorf|krona"),
 ("K8", r"bitnet|tern[aä]r|1\.58|paretoq|binar|binär|1-bit|xnor|\bbnn\b|sub-?1-bit"),
 ("N1", r"scaling law|skalengesetz|skalierung|chinchilla|scaling|skalen"),
 ("N2", r"mamba|\bssm\b|state[- ]space|\bmoe\b|mixture of experts|\blru\b|rwkv|linear attention|recurrent|\brnn\b|hybrid|transformer"),
 ("N3", r"distill"),
 ("N4", r"sparsit|sparse|pruning|prune|lottery|2:4|beschneid"),
 ("N5", r"energie|energy|hardware|fpga|accelerator|beschleuniger|tops/w|\bedge\b|\bcpu\b|photonic|\bopu\b|optical|optik|latenz|latency"),
 ("N6", r"random feature|zufalls|\bseed|random projection|reservoir|\belm\b|\brff\b|\bvera\b|\bnola\b|pranc|supermask|random basis|extreme learning|eigentask"),
 ("N7", r"relu|gelu|aktivierungsfunktion|activation function|primer|squared|swiglu|polynom|quadratic neuron|quadratisch"),
 ("N8", r"rate[- ]distortion|entropy|entropie|shannon|information|holevo|informations|bound|grenze|lower bound"),
 ("F1", r"error correct|fehlerkorrektur|coding theory|codierung|ldpc|stabilizer code|syndrom|error-correcting"),
 ("F2", r"renormali|multiskal|multiscale|rg flow|hierarch"),
 ("F3", r"wavelet|compressed sensing|signal|fourier|\bdct\b|\bfft\b|frame theory|rahmen"),
 ("F4", r"spik|neurobio|dendrit|biolog|sparse cod|brain|hebb|gehirn"),
 ("F5", r"open quantum|dekoh|decoher|lindblad|offene quanten"),
 ("F6", r"monte.?carlo|tensor.?network simulation|simulierbar|simulation|\bdmrg\b|mcmc"),
])
FELD_RE = {k: re.compile(v, re.I) for k, v in FELDER.items()}

def felder(titel, text, begriffe=()):
    s = titel + " " + " ".join(begriffe)
    treffer = [k for k, r in FELD_RE.items() if r.search(s)]
    if not treffer:
        treffer = [k for k, r in FELD_RE.items() if r.search(text)][:3]
    return treffer

BEZ_RE = re.compile(r"\b(I-\d{3}(?:-v\d)?|Z[1-4][a-c]?|Q-\d{3}|T-\d{3}|T-ternaer|Z\d-\d\d)\b")
def bezug_ids(text):
    return sorted(set(m.group(1) for m in BEZ_RE.finditer(text)))

def sw(text):
    t = text.lower()
    if re.search(r"widerspr(?!icht nichts)|gegenbefund|widerlegt|relativiert|bedroht|teilwiderspruch", t) and not re.search(r"widerspricht nichts|kein widerspruch", t):
        return "widerspricht"
    if re.search(r"stützt|stuetzt|stützen|bestätigt|belegt", t):
        return "stuetzt"
    if re.search(r"irrelevant|nur abgrenzung|neutral|themenfern", t):
        return "neutral"
    return "unklar"

ZAHL_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?\s?(?:%|pp|x|×|Bit|bit|Qubits?|Parameter|Mio|Mrd|B\b|M\b|PPL|TOPS/W|bpc)")
def zahlen_aus_text(text, geprueft, quelle):
    out = []
    for s in re.split(r"(?<=[.;])\s+", text):
        ms = ZAHL_RE.findall(s)
        if ms:
            out.append({"wert": "; ".join(m.strip() for m in ms[:4]), "kontext": s.strip()[:200] + " [" + quelle + "]", "geprueft": bool(geprueft)})
    return out[:3]

def neu_eintrag(i, titel=""):
    return {"id": i, "titel": titel, "jahr": None, "autoren": "", "feld": [], "kernaussage": "",
            "zahlen": [], "baseline_fuer_qire": None, "quantenbezug": "keiner", "bezug_qire_befund": [],
            "stuetzt_widerspricht": "unklar", "volltext_gelesen": False, "gefunden_am": STANDARD_DATUM,
            "welle": 0, "herkunft": ""}

# ------------------------------------------------------------------ Speicher
EINTR = OrderedDict()      # id -> Eintrag
TITELIDX = {}              # normierter Titel -> id
TPRIO = {}
HK = defaultdict(set)      # id -> Menge Herkunftsdateien

def finde(i, titel):
    if i in EINTR:
        return i
    nt = norm_titel(titel)
    if len(nt) >= 18:
        if nt in TITELIDX:
            return TITELIDX[nt]
        for k, v in TITELIDX.items():
            if len(k) >= 25 and (k.startswith(nt) or nt.startswith(k)) and min(len(k), len(nt)) >= 25:
                return v
    return None

PLATZ = "Titel in den Quellen nicht genannt"
def eintragen(i, titel, quelle, **kw):
    """Legt an oder führt zusammen. kw: autoren, jahr, kern, feldtext, begriffe, zahlen, baseline, qb, bezug, sw, volltext, datum, titelsicher"""
    vorh = finde(i, titel) if titel and titel != PLATZ else (i if i in EINTR else None)
    if vorh is not None and quelle not in HK[vorh] and HK[vorh]:
        STAT["id_treffer" if vorh == i else "titel_treffer"] += 1
    neu = vorh is None
    if neu:
        e = neu_eintrag(i, titel or PLATZ)
        EINTR[i] = e
        TPRIO[i] = kw.get("tprio", 0)
        vorh = i
    else:
        e = EINTR[vorh]
        if not neu and quelle not in HK[vorh] and kw.get("zaehle", True) and HK[vorh]:
            STAT["zusammengefuehrt"] += 1 if kw.get("struktur", False) else 0
        # Titel nur ersetzen, wenn bisher Platzhalter
        if titel and titel != PLATZ and (e["titel"] == PLATZ or kw.get("tprio", 0) > TPRIO.get(vorh, 0)):
            e["titel"] = titel
            TPRIO[vorh] = kw.get("tprio", 0)
        # ID aufwerten: bib:/url: -> arXiv:
        if i.startswith("arXiv:") and not e["id"].startswith("arXiv:"):
            alt = e["id"]
            e["id"] = i
            EINTR[i] = EINTR.pop(alt)
            HK[i] |= HK.pop(alt, set())
            for k, v in list(TITELIDX.items()):
                if v == alt:
                    TITELIDX[k] = i
            vorh = i
    HK[vorh].add(quelle)
    if titel and titel != PLATZ:
        TITELIDX.setdefault(norm_titel(titel), vorh)
    if kw.get("autoren") and not e["autoren"]:
        e["autoren"] = kw["autoren"]
    if kw.get("jahr") and not e["jahr"]:
        e["jahr"] = kw["jahr"]
    if kw.get("kern"):
        if not e["kernaussage"]:
            e["kernaussage"] = kw["kern"]
        elif kw.get("kern_ergaenze") and kw["kern"] not in e["kernaussage"] and len(e["kernaussage"]) < 220:
            e["kernaussage"] += " " + kw["kern"]
    for z in kw.get("zahlen", []):
        if z not in e["zahlen"]:
            e["zahlen"].append(z)
    if kw.get("baseline") and not e["baseline_fuer_qire"]:
        e["baseline_fuer_qire"] = kw["baseline"]
    for b in kw.get("bezug", []):
        if b not in e["bezug_qire_befund"]:
            e["bezug_qire_befund"].append(b)
    if kw.get("sw") and kw["sw"] != "unklar" and e["stuetzt_widerspricht"] == "unklar":
        e["stuetzt_widerspricht"] = kw["sw"]
    if kw.get("volltext"):
        e["volltext_gelesen"] = True
    d = kw.get("datum")
    if d and (e["gefunden_am"] == STANDARD_DATUM or (kw.get("datum_frueher") and d < e["gefunden_am"])):
        e["gefunden_am"] = d
    e.setdefault("_text", "")
    e["_text"] += " " + (kw.get("feldtext") or "") + " " + " ".join(kw.get("begriffe", []))
    e.setdefault("_begr", [])
    e["_begr"] += list(kw.get("begriffe", []))
    return vorh

# ------------------------------------------------------------------ 1. Bestand LITERATUR
def lade_bib_links():
    t = open(os.path.join(WURZEL, "LITERATUR", "literatur.bib"), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", t, re.S):
        k, body = m.group(1), m.group(2)
        d = re.search(r"doi[ :]*(10\.\d{4,9}/[^\s)}]+)", body)
        u = re.search(r"url\s*=\s*\{(https?://[^}]+)\}", body)
        out[k] = {"doi": d.group(1).rstrip(".,") if d else None,
                  "url": u.group(1) if u and "arxiv.org" not in u.group(1) else None}
    return out

def schritt1():
    idx = json.load(open(os.path.join(WURZEL, "LITERATUR", "neuheit_index.json"), encoding="utf-8"))["arbeiten"]
    bib = lade_bib_links()
    qd_src = "LITERATUR/quellen_daten.py; LITERATUR/neuheit_index.json; LITERATUR/literatur.bib; LITERATUR/notizen.md"
    for e in qd._E:
        k = e["key"]
        if e["arxiv"]:
            i = "arXiv:" + e["arxiv"]
        elif bib.get(k, {}).get("doi"):
            i = "doi:" + bib[k]["doi"]
        elif bib.get(k, {}).get("url"):
            i = "url:" + bib[k]["url"]
        else:
            i = "bib:" + k
        ix = idx.get(k, {})
        vollt = e["status"] == "Volltext"
        kern = (e["kern"] or "").strip()
        tx = " ".join(filter(None, [e["kern"], e["qire"], e["neuheit"], e["titel"]]))
        fund = ix.get("fundstellen", [])
        daten = [re.search(r"(20\d\d-\d\d-\d\d)", f).group(1) for f in fund if re.search(r"(20\d\d-\d\d-\d\d)", f)]
        datum = min(daten) if daten else STANDARD_DATUM
        basel = None
        if re.search(r"baseline|gegner|standardquelle|vergleich", (e["neuheit"] or "") + " " + (e["qire"] or ""), re.I):
            basel = (e["neuheit"] or e["qire"])[:300]
        autoren = e["autoren"] or ""
        if re.search(r"unvollst|nicht verifiziert", autoren):
            autoren = ""
        titel = e["titel"]
        eintragen(i, titel, qd_src, tprio=5, autoren=autoren, jahr=e["jahr"], kern=kern,
                  zahlen=zahlen_aus_text(kern, vollt, "Bestand LITERATUR, Status %s" % e["status"]),
                  baseline=basel, bezug=bezug_ids(e["qire"] or "") + bezug_ids(e["neuheit"] or ""),
                  sw=("widerspricht" if re.search(r"gegenbefund|widerspr|relativiert", (e["neuheit"] or ""), re.I)
                      else "unklar"),
                  volltext=vollt, datum=datum, feldtext=tx, begriffe=e["begriffe"], struktur=False)
        EINTR[i if i in EINTR else TITELIDX[norm_titel(titel)]]["_qb_text"] = tx
    # Hinweis: aus Fundstellen weitere Herkunft (nur Dateien mit Namen)
    for e in qd._E:
        k = e["key"]
        i = ("arXiv:" + e["arxiv"]) if e["arxiv"] else None
        if i is None:
            i = next((x for x, v in EINTR.items() if x.endswith(k) and x.startswith("bib:")), None) \
                or ("doi:" + bib[k]["doi"] if bib.get(k, {}).get("doi") else None) \
                or ("url:" + bib[k]["url"] if bib.get(k, {}).get("url") else None)
        if i in EINTR:
            for f in idx.get(k, {}).get("fundstellen", []):
                HK[i].add(f.split(":")[0])

# ------------------------------------------------------------------ 2./3. Markdown
ID_NEU = re.compile(r"(?<![\d.])((?:0[7-9]|1\d|2[0-6])(?:0[1-9]|1[0-2])\.\d{4,5})(?:v\d+)?(?![\d])")
ID_ALT = re.compile(r"arxiv[:/ .a-z]*?((?:quant-ph|cs\.[A-Z]{2}|math-ph|cond-mat|hep-th|stat\.ML)/\d{7})", re.I)

def ids_in_zeile(z):
    gefunden = []
    low = z.lower()
    for m in ID_NEU.finditer(z):
        vor = z[max(0, m.start() - 40):m.start()].lower()
        if "arxiv" in vor or "arxiv" in low[:m.start()] and re.search(r"[(\[/ :]$", z[:m.start()]) or "arxiv.org" in low:
            gefunden.append(("arXiv:" + m.group(1), m.start(), m.end()))
    for m in ID_ALT.finditer(z):
        gefunden.append(("arXiv:" + m.group(1), m.start(), m.end()))
    return gefunden

def titel_fuer(z, i, s, e):
    """Titel aus Zeile z für ID i (Position s..e); None, wenn nicht ermittelbar."""
    num = i.split(":", 1)[1]
    m = re.search(r"title\s*=\s*\{(.+?)\}\s*,\s*(?:author|booktitle|journal|year|note)", z)
    if m:
        return re.sub(r"[{}]", "", m.group(1)).strip(), 3
    for lm in re.finditer(r"\[([^\]]+)\]\(([^)]*)\)", z):
        if num in lm.group(2):
            t = lm.group(1)
            t = re.sub(r",?\s*\(?arxiv[: ]*[\d./v-]+\)?", "", t, flags=re.I).strip(" ,;")
            if len(t) >= 4:
                return t, 2
    z2 = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", z)
    pos = z2.find(num)
    kand = [(m.start(), m.group(1)) for m in re.finditer(r"\*\*([^*]{4,160})\*\*", z2)]
    kand += [(m.start(), m.group(1)) for m in re.finditer(r"„([^“]{8,200})“", z2)]
    kand += [(m.start(), m.group(1)) for m in re.finditer(r"\"([^\"]{8,200})\"", z2)]
    if kand and pos >= 0:
        vorher = [k for k in kand if k[0] <= pos]
        k = max(vorher, key=lambda x: x[0]) if vorher else min(kand, key=lambda x: x[0])
        if abs(k[0] - pos) < 120:
            return k[1].strip(" ,.:;"), 1
    return None

def autoren_fuer(z):
    m = re.search(r"author\s*=\s*\{(.+?)\}\s*,\s*(?:booktitle|journal|year|note|title)", z)
    if m:
        return re.sub(r"[{}]", "", m.group(1)).replace(" and ", ", ").strip()
    return ""

def jahr_fuer(z):
    m = re.search(r"year\s*=\s*\{(\d{4})\}", z)
    return int(m.group(1)) if m else None

def md_scan(pfad, volltext=False, prio=0):
    quelle = rel(pfad)
    datum = datei_datum(pfad)
    welle = 0
    zeilen = open(pfad, encoding="utf-8").read().split("\n")
    belege = defaultdict(lambda: {"titel": None, "autoren": "", "jahr": None, "ctx": [], "tabelle": False})
    for z in zeilen:
        if not z.strip():
            continue
        for i, s, e in ids_in_zeile(z):
            b = belege[i]
            tt = titel_fuer(z, i, s, e)
            if tt and tt[1] > b.get("tprio", 0):
                b["titel"], b["tprio"] = tt
            b["autoren"] = b["autoren"] or autoren_fuer(z)
            b["jahr"] = b["jahr"] or jahr_fuer(z)
            b["ctx"].append(z)
        # Nicht-arXiv: fette Titel + URL/DOI in Listenzeilen
        if re.match(r"\s*(?:[-*]|\d+\.|\|)", z) and "arxiv" not in z.lower():
            m = re.search(r"\*\*([^*]{12,200})\*\*", z)
            u = re.search(r"(https?://[^\s)\]|;,]+)", z)
            d = re.search(r"\b(10\.\d{4,9}/[^\s,;)\]|]+)", z)
            if m and (u or d) and not m.group(1).rstrip().endswith(":") and not re.match(r"\d+(\.\d+)*\s", m.group(1)):
                if d:
                    i = "doi:" + d.group(1).rstrip(".")
                else:
                    i = "url:" + u.group(1).rstrip(".")
                b = belege[i]
                b["titel"], b["tprio"] = m.group(1).strip(" ,.:;"), 1
                b["ctx"].append(z)
    for i, b in belege.items():
        ctx = " ".join(md_rein(c) for c in b["ctx"][:3])
        titel = b["titel"] or PLATZ
        rest = md_rein(b["ctx"][0])
        if titel != PLATZ:
            rest = rest.replace(titel, "", 1)
        rest = re.sub(r"\(?\barxiv[: ]*[\d./v-]+\)?", "", rest, flags=re.I)
        rest = re.sub(r"^[\s;:,.\-–—()]+", "", rest).strip()
        rest = re.sub(r"^[^()]{0,30}\)[:;,.]?\s*", "", rest).strip()
        rest = re.sub(r"\b(?:title|author|booktitle|journal|year|note)\s*=\s*\{[^}]*\}\s*,?", "", rest)
        kern = saetze(rest) if len(rest) >= 30 else ""
        if not kern:
            kern = "In %s genannt; Inhalt dort nicht näher beschrieben." % os.path.basename(quelle)
        else:
            kern = kern + " (Kontext aus %s)" % os.path.basename(quelle)
        sw_ = sw(ctx)
        eintragen(i, titel, quelle, tprio=b.get("tprio", 0), autoren=b["autoren"], jahr=b["jahr"] or jahr_aus_id(i),
                  kern=kern, feldtext=ctx, bezug=bezug_ids(ctx), sw=sw_, volltext=volltext,
                  datum=datum, struktur=False, kern_ergaenze=False)
        # Kern nicht-informativ ersetzen, falls später besserer Kontext
        e = EINTR[i if i in EINTR else TITELIDX.get(norm_titel(titel), i)]
        if e["kernaussage"].startswith("In ") and "nicht näher beschrieben" in e["kernaussage"] and "nicht näher" not in kern:
            e["kernaussage"] = kern
        if e["jahr"] is None:
            e["jahr"] = jahr_aus_id(e["id"])

# STROM
def zellen(z):
    c = [x.strip() for x in z.strip().strip("|").split("|")]
    return c

def schritt_strom():
    pfad = os.path.join(REC, "zyklus", "STROM.md")
    quelle = rel(pfad)
    lauf_datum = "2026-10-29"
    for z in open(pfad, encoding="utf-8").read().split("\n"):
        m = re.match(r"## (\d\d)\.(\d\d)\. Lauf", z)
        if m:
            lauf_datum = "2026-%s-%s" % (m.group(2), m.group(1))
        if not z.startswith("|") or z.startswith("|---") or z.startswith("| Datum"):
            continue
        c = zellen(z)
        if len(c) != 5:
            continue
        datum_roh, titel, idz, zahl, bezug = c
        ids = re.findall(r"(?<![\d.])((?:0[7-9]|1\d|2[0-6])(?:0[1-9]|1[0-2])\.\d{4,5})", idz + " " + titel)
        ids = list(dict.fromkeys(ids))
        md = re.match(r"(20\d\d)-(\d\d)", datum_roh)
        jahr = int(md.group(1)) if md else None
        if not md:
            my = re.match(r"(20\d\d)", datum_roh)
            jahr = int(my.group(1)) if my else None
        zl = []
        if zahl and not re.match(r"^(keine|—|-)\b", zahl.lower()):
            zl = [{"wert": zahl, "kontext": "%s (Späher-Snippet, STROM %s)" % (titel, lauf_datum), "geprueft": False}]
        bez = bezug_ids(bezug)
        kern = "%s Quelle: Späher-Fund, nur Snippet." % saetze(md_rein(bezug), 2, 300).rstrip(".;, ") + ""
        kern = kern.replace(" Quelle: Späher", ". Quelle: Späher")
        sw_ = sw(bezug)
        basel = bezug if re.search(r"gegner|baseline|messlatte", bezug, re.I) else None
        if len(ids) >= 1:
            titel_teile = [t.strip() for t in re.split(r";", titel)] if len(ids) > 1 else [titel]
            if len(titel_teile) != len(ids):
                titel_teile = [titel] * len(ids)
            for i, t in zip(ids, titel_teile):
                t = re.sub(r"\s*\(\d{4}\.\d{4,5}\)", "", t).strip()
                eintragen("arXiv:" + i, t, quelle, tprio=4, jahr=jahr_aus_id("arXiv:" + i) or jahr, kern=kern, zahlen=zl,
                          baseline=basel, bezug=bez, sw=sw_, datum=lauf_datum, feldtext=titel + " " + bezug,
                          struktur=True)
            continue
        # keine arXiv-ID
        link = idz.strip()
        mu = re.search(r"((?:https?://)?[\w.-]+\.[a-z]{2,}/[^\s;,)]+)", link)
        if mu:
            u = mu.group(1)
            ident = "url:" + (u if u.startswith("http") else "https://" + u)
        else:
            ident = "bib:strom_" + re.sub(r"[^a-z0-9]+", "_", titel.lower())[:40].strip("_")
        eintragen(ident, titel, quelle, jahr=jahr, kern=kern, zahlen=zl, baseline=basel, bezug=bez, sw=sw_,
                  datum=lauf_datum, feldtext=titel + " " + bezug, struktur=True)

def schritt_reaktor():
    pfad = os.path.join(REC, "zyklus", "REAKTOR.md")
    quelle = rel(pfad)
    datum = "2026-10-29"
    for z in open(pfad, encoding="utf-8").read().split("\n"):
        if not z.startswith("|") or z.startswith("|---") or z.startswith("| Fund"):
            continue
        c = zellen(z)
        if len(c) != 3:
            continue
        ids = ID_NEU.findall(c[0])
        wirkung, gegner = md_rein(c[1]), md_rein(c[2])
        basel = None
        if gegner and not re.match(r"^(keiner|kein Gegner)", gegner, re.I):
            basel = gegner[:300]
        s = sw(wirkung)
        if re.match(r"nur abgrenzung|neutral", wirkung, re.I):
            s = "neutral"
        elif re.match(r"st[üu]tz", wirkung, re.I):
            s = "stuetzt"
        elif re.match(r"(teil)?widerspr", wirkung, re.I):
            s = "widerspricht"
        for i in ids:
            iid = "arXiv:" + i
            if iid not in EINTR:
                continue
            eintragen(iid, "", quelle, baseline=basel, bezug=bezug_ids(c[1] + c[2]), sw=s,
                      kern=None, feldtext=wirkung + " " + gegner, datum=datum, struktur=True)
            e = EINTR[iid]
            if s != "unklar":
                e["stuetzt_widerspricht"] = s
            if basel:
                e["baseline_fuer_qire"] = basel

def schritt_archiv():
    pfad = os.path.join(REC, "zyklus", "ARCHIV.md")
    md_scan(pfad)

# ------------------------------------------------------------------ Hauptlauf
def haupt():
    schritt1()
    schritt_strom()
    schritt_reaktor()
    schritt_archiv()
    alle = sorted(glob.glob(os.path.join(REC, "**", "*.md"), recursive=True))
    bereits = {os.path.join(REC, "zyklus", n) for n in ("STROM.md", "REAKTOR.md", "ARCHIV.md")}

    def rang(p):
        r = rel(p)
        for n, key in enumerate(("2026-10-05_breit", "2026-10-05_vertiefung", "chat_sweep", "volltext", "2026-10-27_welle/", "2026-10-28_welle2", "2026-10-28_welle3")):
            if key in r:
                return (n, r)
        return (9, r)
    for p in sorted(alle, key=rang):
        if p in bereits:
            continue
        md_scan(p, volltext=("volltext" in os.path.basename(p)))
    # Abschluss
    out = []
    for i, e in EINTR.items():
        text = e.pop("_text", "") + " " + e["titel"] + " " + e["kernaussage"]
        begr = e.pop("_begr", [])
        e.pop("_qb_text", None)
        e["feld"] = felder(e["titel"], text, begr)
        e["quantenbezug"] = quantenbezug(e["titel"], text)
        if not e["feld"] and e["quantenbezug"] == "direkt":
            e["feld"] = ["K6"]
        e["herkunft"] = "; ".join(sorted(HK[i]))
        if e["jahr"] is None:
            e["jahr"] = jahr_aus_id(e["id"])
        e["bezug_qire_befund"] = sorted(set(e["bezug_qire_befund"]))
        # Zahlen entdoppeln
        seen, z2 = set(), []
        for z in e["zahlen"]:
            k = (z["wert"], z["kontext"])
            if k not in seen:
                seen.add(k); z2.append(z)
        e["zahlen"] = z2
        if not e["kernaussage"]:
            e["kernaussage"] = "Inhalt in den Quellen nicht beschrieben."
        e["id"] = i
        out.append(e)
    FELDREIHE = ["id", "titel", "jahr", "autoren", "feld", "kernaussage", "zahlen", "baseline_fuer_qire",
                 "quantenbezug", "bezug_qire_befund", "stuetzt_widerspricht", "volltext_gelesen",
                 "gefunden_am", "welle", "herkunft"]
    out = [OrderedDict((k, e[k]) for k in FELDREIHE) for e in out]
    ziel = os.path.join(WURZEL, "WISSEN", "quellen.json")
    with open(ziel, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("Einträge:", len(out), STAT)

if __name__ == "__main__":
    haupt()
