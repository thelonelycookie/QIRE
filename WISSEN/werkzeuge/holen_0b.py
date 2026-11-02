#!/usr/bin/env python3
"""Holt arXiv-Metadaten aller arXiv-Einträge der quellen.json und cached sie in volltexte/arxiv_abstracts.json."""
import json, re, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
W = Path(__file__).resolve().parent.parent
CACHE = W/"volltexte"/"arxiv_abstracts.json"
NS = {"a": "http://www.w3.org/2005/Atom"}
def basis(i):
    return re.sub(r"v\d+$", "", i)
def main():
    q = json.load(open(W/"quellen.json"))
    ids = [e["id"][6:] for e in q if e["id"].startswith("arXiv:")]
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    todo = [i for i in ids if i not in cache]
    for s in range(0, len(todo), 50):
        blk = todo[s:s+50]
        url = "http://export.arxiv.org/api/query?id_list=%s&max_results=50" % ",".join(blk)
        for versuch in range(4):
            try:
                x = urllib.request.urlopen(url, timeout=60).read(); break
            except Exception as ex:
                print("Fehler", ex); time.sleep(10)
        else:
            continue
        root = ET.fromstring(x)
        for en in root.findall("a:entry", NS):
            eid = basis(en.find("a:id", NS).text.split("/abs/")[-1])
            t = " ".join((en.find("a:title", NS).text or "").split())
            if t == "Error" or "api/errors" in en.find("a:id", NS).text: continue
            cache[eid] = {
                "titel": t,
                "abstract": " ".join((en.find("a:summary", NS).text or "").split()),
                "autoren": [a.find("a:name", NS).text for a in en.findall("a:author", NS)],
                "jahr": int(en.find("a:published", NS).text[:4]),
            }
        print(s, len(cache)); time.sleep(3)
    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=0)
    miss = [i for i in ids if i not in cache and basis(i) not in cache]
    print("fehlend", miss)
main()
