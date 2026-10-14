#!/usr/bin/env python3
"""Befund-Dashboard (R4.5): erzeugt eine einzelne statische dashboard.html.

Liest INSTITUT/agenda.json und INSTITUT/ergebnisse/*.json (jeweils der neueste Lauf je ID),
optional IDEEN_KI/status.json und IDEEN_KI/ausgabe/karten_*.json.
Kein externes JavaScript. Alle Zahlen stammen woertlich aus den Urteils-Strings der
Ergebnis-JSONs; das Dashboard rechnet nichts nach.

Aufruf:  python3 JUGEND_FORSCHT/dashboard/erzeuge.py [ausgabe.html]
"""
import glob
import html
import json
import os
import re
import sys
from datetime import datetime

WURZEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INST = os.path.join(WURZEL, "INSTITUT")
KI_DIR = os.path.join(WURZEL, "IDEEN_KI")

# Heuristische Einordnung des ersten Wortes im Urteils-String (Rohlabel bleibt sichtbar).
BEST = {"H1", "GLEICHWERTIG", "NICHT_UNTERLEGEN", "HAAR_BESSER", "BORN_BESSER", "QUADRAT_TRAEGT",
        "E_DOMINIERT", "SUCHE_BESSER", "VORSPRUNG_WAECHST", "STRUKTUR_HILFT", "SZ_BESSER", "AKTIVIERUNG",
        "UNTERSCHIED", "ZIELKONFLIKT_TRAINIERBAR", "INFO_UEBER_FEST", "GENAUIGKEIT_UEBER_FEST",
        "B_SCHNELLER", "IDENTITAET_BESTE", "SCHUSSBEWUSST_BESSER", "MISCHUNG_TRAEGT", "UEBERLEGEN", "B+"}
NEG = {"H0", "SUCHE_NICHT_BESSER", "VOLL_BESSER", "ZUFALL_BESSER", "KLASSISCH_BESSER", "SCHRUMPFT"}
KONTR = {"BESTANDEN", "PK_OK", "NK_OK", "DURCHGEFALLEN"}


def klasse(schluessel, text):
    wort = re.split(r"[\s(:,]", text.strip(), maxsplit=1)[0] if text else ""
    s = schluessel.upper()
    if wort in KONTR or s.startswith(("NK", "PK")):
        return "Kontrolle", wort
    if wort.lower().startswith("beschreibend") or s.startswith("B-") or s == "B":
        return "beschreibend", wort
    if wort == "HU":
        return "HU", wort
    if wort in NEG:
        return "H0", wort
    if wort in BEST:
        return "H1", wort
    return "sonst", wort


def parse(text):
    """Zieht Effekt, KI, n, p woertlich aus dem Urteils-String (nur Anzeige)."""
    d = re.search(r"\bd\s*([+\-−]?[\d.,]+)\s*pp", text)
    ki = re.search(r"KI\s*\[([^\]]+)\]", text)
    n = re.search(r"\bn\s*=?\s*(\d+)", text)
    p = re.search(r"p_(?:Holm|TOST|zweiseitig)?\s*([\d.,eE+\-]+)", text)
    return {
        "effekt": (d.group(1) + " pp") if d else "",
        "ki": ("[" + ki.group(1) + "]") if ki else "",
        "n": n.group(1) if n else "",
        "p": p.group(0) if p else "",
    }


def lade_ergebnisse():
    neueste = {}
    for f in sorted(glob.glob(os.path.join(INST, "ergebnisse", "*.json"))):
        name = os.path.basename(f)
        if name.startswith("_zwischen") or "__" not in name:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:  # defekte Datei nicht verschweigen
            print("WARNUNG: nicht lesbar:", name, e, file=sys.stderr)
            continue
        id_, stempel = name[:-5].split("__", 1)
        neueste[id_] = (stempel, name, d)  # sortiert: spaetere Stempel ueberschreiben
    return neueste


def baue_daten():
    with open(os.path.join(INST, "agenda.json"), encoding="utf-8") as fh:
        agenda = json.load(fh)["punkte"]
    amap = {p["id"]: p for p in agenda}
    ergebnisse = lade_ergebnisse()
    zeilen = []
    ohne_urteil = []
    for id_, (stempel, name, d) in sorted(ergebnisse.items()):
        u = d.get("urteil")
        linie = amap.get(id_, {}).get("abteilung") or d.get("abteilung", "?")
        datum = "%s.%s.%s" % (stempel[6:8], stempel[4:6], stempel[0:4]) if len(stempel) >= 8 else ""
        if not isinstance(u, dict):
            ohne_urteil.append(id_)
            continue
        for k, v in u.items():
            text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            kl, label = klasse(k, text)
            z = {"id": id_, "linie": linie, "datum": datum, "hyp": k, "klasse": kl,
                 "label": label, "text": text[:400], "quelle": "INSTITUT/ergebnisse/" + name}
            z.update(parse(text))
            zeilen.append(z)
    agenda_tab = [{"id": p["id"], "linie": p.get("abteilung", ""), "status": p["status"],
                   "wartet_auf": p.get("wartet_auf", ""), "ort": p.get("ort", ""),
                   "frage": p.get("frage", "")[:220]} for p in agenda]
    # Ideen-KI (optional)
    ki = {"status": None, "karten": [], "hinweis": ""}
    sp = os.path.join(KI_DIR, "status.json")
    if os.path.exists(sp):
        try:
            with open(sp, encoding="utf-8") as fh:
                ki["status"] = json.load(fh)
        except Exception as e:
            ki["hinweis"] = "status.json nicht lesbar: %s" % e
    for f in sorted(glob.glob(os.path.join(KI_DIR, "ausgabe", "karten_*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                dat = json.load(fh)
            liste = dat if isinstance(dat, list) else dat.get("karten", [dat])
            for kk in liste:
                ki["karten"].append({"datei": os.path.basename(f), "karte": kk})
        except Exception as e:
            ki["hinweis"] += " %s nicht lesbar: %s;" % (os.path.basename(f), e)
    if ki["status"] is None and not ki["karten"] and not ki["hinweis"]:
        ki["hinweis"] = ("Noch keine Daten: IDEEN_KI/status.json und ausgabe/karten_*.json existieren nicht. "
                         "Laut IDEEN_KI/PLAN_2026-10-13.md wartet die Ideen-KI auf Maltes Freigabe.")
    return {"erzeugt": datetime.now().strftime("%d.%m.%Y %H:%M"), "zeilen": zeilen,
            "agenda": agenda_tab, "ki": ki, "ohne_urteil": ohne_urteil,
            "anzahl_ergebnisdateien": len(ergebnisse)}


VORLAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QIRE Befund-Dashboard</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#666;--kar:#fff;--rand:#d8d8dc;--h1:#1b7f3b;--h0:#b3261e;--hu:#8a6d00;--ko:#4a5a8a;--be:#777}
@media(prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--kar:#202024;--rand:#3a3a40;--h1:#5fd486;--h0:#ff8a80;--hu:#e6c35c;--ko:#9fb3ee;--be:#999}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin-inline:auto}
h1{margin:.2em 0}h2{margin-top:1.6em;border-bottom:1px solid var(--rand);padding-bottom:.2em}
.mut{color:var(--mut);font-size:.9em}.karten{display:flex;gap:10px;flex-wrap:wrap}
.k{background:var(--kar);border:1px solid var(--rand);border-radius:8px;padding:8px 14px;min-width:90px}.k b{font-size:1.6em;display:block}
.bar{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}select,input{background:var(--kar);color:var(--fg);border:1px solid var(--rand);border-radius:6px;padding:6px 8px;font:inherit}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--kar)}th,td{border-bottom:1px solid var(--rand);padding:6px 8px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--kar);cursor:pointer;white-space:nowrap}
.b{display:inline-block;border-radius:10px;padding:1px 9px;font-size:.85em;font-weight:600;border:1px solid currentColor}
.H1{color:var(--h1)}.H0{color:var(--h0)}.HU{color:var(--hu)}.Kontrolle{color:var(--ko)}.beschreibend,.sonst{color:var(--be)}
code{font-size:.85em}.ehrlich{border-left:4px solid var(--hu);padding:6px 12px;background:var(--kar)}
</style></head><body>
<h1>QIRE Befund-Dashboard</h1>
<p class="mut">Erzeugt __ERZEUGT__ aus INSTITUT/agenda.json und INSTITUT/ergebnisse/*.json (__NDATEIEN__ Ergebnisdateien, jeweils neuester Lauf je ID). Zahlen stehen woertlich so in den Urteils-Strings der Rohdaten.</p>
<p class="ehrlich">Ehrliche Rahmung: Die Wirkung von QIRE ist klassisch erklaerbar (I-038). Gemeint ist ein <b>quanten-inspirierter</b> Vorteil bei Parametern und Rechenaufwand, kein Quantenvorteil. Die Einordnung H1 / H0 / HU unten ist eine automatische Heuristik auf dem ersten Wort des Urteils; massgeblich ist das Rohlabel und der Urteilstext.</p>
<div class="karten" id="kacheln"></div>
<h2>Befunde</h2>
<div class="bar">
<label>Linie <select id="fl"></select></label>
<label>Urteil <select id="fu"><option value="">alle</option><option>H1</option><option>H0</option><option>HU</option><option>Kontrolle</option><option>beschreibend</option><option>sonst</option></select></label>
<label>Suche <input id="fs" placeholder="ID, Hypothese, Text"></label>
<label><input type="checkbox" id="fk"> Kontrollen und Beschreibendes ausblenden</label>
</div>
<div class="tw"><table id="tab"><thead><tr>
<th data-k="id">ID</th><th data-k="linie">Linie</th><th data-k="hyp">Hypothese</th><th data-k="klasse">Urteil</th><th data-k="label">Rohlabel</th><th data-k="effekt">Effekt</th><th data-k="ki">KI (90 %)</th><th data-k="n">n</th><th data-k="p">p</th><th data-k="datum">Datum</th></tr></thead><tbody></tbody></table></div>
<p class="mut" id="zaehler"></p>
<h2>Agenda (Status aller Punkte)</h2>
<div class="bar"><label>Status <select id="fa"></select></label></div>
<div class="tw"><table id="ag"><thead><tr><th>ID</th><th>Linie</th><th>Status</th><th>wartet auf</th><th>Ort</th><th>Frage</th></tr></thead><tbody></tbody></table></div>
<h2>Ideen-KI</h2>
<div id="ki"></div>
<script id="daten" type="application/json">__DATEN__</script>
<script>
const D=JSON.parse(document.getElementById('daten').textContent);
const $=s=>document.querySelector(s);
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let sk='id',sr=1;
function kacheln(){const c={};D.zeilen.forEach(z=>c[z.klasse]=(c[z.klasse]||0)+1);
 const ids=new Set(D.zeilen.map(z=>z.id));
 let h=`<div class="k"><b>${ids.size}</b>Laeufe mit Urteil</div>`;
 ['H1','H0','HU','Kontrolle','beschreibend','sonst'].forEach(k=>{if(c[k])h+=`<div class="k"><b class="${k}">${c[k]}</b>${k}</div>`});
 h+=`<div class="k"><b>${D.agenda.length}</b>Agenda-Punkte</div>`;$('#kacheln').innerHTML=h}
function fillSel(el,vals){el.innerHTML='<option value="">alle</option>'+vals.map(v=>`<option>${esc(v)}</option>`).join('')}
function tab(){const l=$('#fl').value,u=$('#fu').value,s=$('#fs').value.toLowerCase(),k=$('#fk').checked;
 let r=D.zeilen.filter(z=>(!l||z.linie===l)&&(!u||z.klasse===u)&&(!k||!['Kontrolle','beschreibend'].includes(z.klasse))&&(!s||(z.id+' '+z.hyp+' '+z.text).toLowerCase().includes(s)));
 r.sort((a,b)=>String(a[sk]).localeCompare(String(b[sk]),'de',{numeric:true})*sr);
 $('#tab tbody').innerHTML=r.map(z=>`<tr title="${esc(z.text)}"><td><b>${esc(z.id)}</b></td><td>${esc(z.linie)}</td><td>${esc(z.hyp)}</td><td><span class="b ${z.klasse}">${z.klasse}</span></td><td><code>${esc(z.label)}</code></td><td>${esc(z.effekt)}</td><td>${esc(z.ki)}</td><td>${esc(z.n)}</td><td>${esc(z.p)}</td><td>${esc(z.datum)}</td></tr>`).join('');
 $('#zaehler').textContent=r.length+' von '+D.zeilen.length+' Zeilen. Mauszeiger auf einer Zeile zeigt den Urteilstext; Quelle je Zeile: '+(r[0]?r[0].quelle:'-')+' (Beispiel).'}
function ag(){const s=$('#fa').value;$('#ag tbody').innerHTML=D.agenda.filter(a=>!s||a.status===s).map(a=>`<tr><td><b>${esc(a.id)}</b></td><td>${esc(a.linie)}</td><td>${esc(a.status)}</td><td>${esc(a.wartet_auf)}</td><td>${esc(a.ort)}</td><td>${esc(a.frage)}</td></tr>`).join('')}
function ki(){const k=D.ki;let h='';
 if(k.hinweis)h+=`<p class="ehrlich">${esc(k.hinweis)}</p>`;
 if(k.status){h+=`<h3>Status (status.json)</h3><pre>${esc(JSON.stringify(k.status,null,1))}</pre>`}
 if(k.karten.length){h+=`<h3>Karten (${k.karten.length})</h3><div class="tw"><table><thead><tr><th>Datei</th><th>Karte</th></tr></thead><tbody>`+k.karten.map(c=>`<tr><td>${esc(c.datei)}</td><td><pre style="white-space:pre-wrap;margin:0">${esc(JSON.stringify(c.karte,null,1))}</pre></td></tr>`).join('')+'</tbody></table></div>'}
 $('#ki').innerHTML=h}
fillSel($('#fl'),[...new Set(D.zeilen.map(z=>z.linie))].sort());
fillSel($('#fa'),[...new Set(D.agenda.map(a=>a.status))].sort());
['fl','fu','fs','fk'].forEach(i=>$('#'+i).addEventListener('input',tab));$('#fa').addEventListener('input',ag);
document.querySelectorAll('#tab th').forEach(t=>t.addEventListener('click',()=>{const k=t.dataset.k;sr=(sk===k)?-sr:1;sk=k;tab()}));
kacheln();tab();ag();ki();
</script></body></html>
"""


def main():
    ziel = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "dashboard.html")
    daten = baue_daten()
    js = json.dumps(daten, ensure_ascii=False).replace("</", "<\\/")
    seite = (VORLAGE.replace("__ERZEUGT__", html.escape(daten["erzeugt"]))
             .replace("__NDATEIEN__", str(daten["anzahl_ergebnisdateien"]))
             .replace("__DATEN__", js))
    with open(ziel, "w", encoding="utf-8") as fh:
        fh.write(seite)
    print("geschrieben:", ziel, "|", len(daten["zeilen"]), "Zeilen,", len(daten["agenda"]), "Agenda-Punkte,",
          "ohne Urteils-dict:", daten["ohne_urteil"])


if __name__ == "__main__":
    main()
