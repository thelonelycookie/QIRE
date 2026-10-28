#!/usr/bin/env python3
"""Auswertung des Erkundungspakets Welle 2 (K01-K20, E01-E09) -- Erkundung, kein Befund, nicht in LABORBUCH.

Liest roh/<KID>_s95x.json (je Tuning-Seed eine Datei), berechnet je Kandidat
  1. den HAUPTVERGLEICH: Idee gegen die HAERTESTE Kontrolle, gepaart ueber Seeds
     (je Seed und Zelle: Gewinn = Idee besser als beste Kontrolle; Zellen werden je Seed gemittelt;
      ueber Seeds: Mittel, sd (ddof=1), Vorzeichen-Zaehlung plus/n, t = Mittel/(sd/sqrt(n))),
  2. die AMPEL (feste Regel unten),
  3. den BRIER-Wert der Vorab-P aus konv_kandidaten.md (Ereignis "gruen"), gruppiert nach Denkwerkzeug k1-k5.
Ausgabe: Markdown auf stdout (und mit --out in eine Datei).

AMPEL (fest, vor dem Blick auf Daten; Gewinn > 0 heisst: Idee besser als die haerteste Kontrolle; Einheit bpc oder Prozentpunkte):
  gruen: Mittel > Schwelle  UND  plus >= ceil(0,8 n)  UND  (n < 3 oder t >= 2)
  gelb : Mittel > 0         UND  plus >= ceil(0,6 n)            (und nicht gruen)
  rot  : sonst
  grau : weniger als MIN_N Seeds (Standard 3) -> keine Ampel, nicht im Brier.
  Schwelle (Mindesteffekt): 0,010 bpc bzw. 0,5 Prozentpunkte (je Vergleich in der Tabelle SPEC einstellbar).
  "Haerteste Kontrolle" = je Seed und Zelle die beste der aufgefuehrten Kontrollen (konservativ).

BRIER: p = Vorab-P (fuehrende Prozentzahl der P-Spalte), o = 1 wenn Ampel gruen, sonst 0; Brier = Mittel (p-o)^2.
  Zusatz: Variante "vorsichtig" mit der kleinsten Prozentzahl der P-Zelle (z. B. "45 % (gegen I-048-Arm 30 %)" -> 30 %).
  Referenzen: p = 0,5 (Brier 0,25) und p = beobachtete Gruen-Rate (Brier-Skill-Wert dazu).
  Gruppen: Denkwerkzeug aus der Herkunft (k1 Morphologischer Kasten, k2 Fehlerbuch-Bergbau, k3 Umkehrung,
  k4 Zufallsimpuls, k5 Perspektiven). "erste Herkunft" ordnet jeden Kandidaten genau einer Gruppe zu,
  "alle Herkuenfte" zaehlt ihn in jeder genannten Gruppe.

HAUPTVERGLEICHS-SCHLUESSEL: Tabelle SPEC weiter unten (je KID: Idee, Kontrollen, JSON-Pfade, Richtung, Einheit, Schwelle).
  E-Kandidaten heissen wie in konv_kandidaten.md (E1...E12); ihre Dateien heissen E01...E09 (E4, E8, E10, E11, E12 sind
  Auswertungen anderer Dateien: E4/E10/E11 aus E01, E8 aus E07, E12 aus E06).

Aufruf (aus INSTITUT/erkundung/):
  python3 auswertung.py                      # alle Kandidaten, roh/<KID>_s95?.json
  python3 auswertung.py --nur K01,K05 --out auswertung.md
  python3 auswertung.py --schnell --min-n 1  # NUR FORMATTEST mit den Schnell-JSONs (nicht aussagekraeftig)
"""
from __future__ import annotations

import argparse, glob, json, math, re, statistics, sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
MIN_N = 3
SCHWELLE_BPC, SCHWELLE_PP = 0.010, 0.5
DENKWERKZEUG = {"k1": "Morphologischer Kasten", "k2": "Fehlerbuch-Bergbau", "k3": "Umkehrung",
                "k4": "Zufallsimpuls (fremde Paper)", "k5": "Perspektiven"}


# ------------------------------------------------------------------ Hilfen fuer JSON-Pfade
def g(d, *pfad):
    """Wert am Pfad oder None (fehlende Schluessel, Nicht-Zahlen am Ende, Nicht-Dicts unterwegs -> None)."""
    for k in pfad:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return float(d) if isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d) else None


def schl(d, *pfad):
    """Schluessel des Dicts am Pfad (leer, wenn es keins ist)."""
    for k in pfad:
        if not isinstance(d, dict) or k not in d:
            return []
        d = d[k]
    return list(d) if isinstance(d, dict) else []


def zelle(name, praefix, idee, kontrollen, metrik):
    """Zelle = (Name, Pfad der Idee, [(Kontrollname, Pfad), ...]); Pfad = praefix + (Arm,) + metrik."""
    return (name, tuple(praefix) + (idee,) + tuple(metrik), [(c, tuple(praefix) + (c,) + tuple(metrik)) for c in kontrollen])


def V(idee, kontrolle, zellen, richtung, einheit, schwelle=None):
    """Ein Vergleich. richtung 'min' (kleiner ist besser, bpc) oder 'max' (Genauigkeit, Akzeptanz); einheit 'bpc' oder 'pp'."""
    return dict(idee=idee, kontrolle=kontrolle, zellen=zellen, richtung=richtung, einheit=einheit,
                schwelle=schwelle if schwelle is not None else (SCHWELLE_BPC if einheit == "bpc" else SCHWELLE_PP))


# ------------------------------------------------------------------ SPEC: Hauptvergleichs-Schluessel je KID
# Jede Zellenfunktion bekommt das JSON eines Seeds und liefert die Zellen. Fehlt eine Zelle/Kontrolle, wird sie uebersprungen.
def _k05(d):
    bits = sorted({k.split("@")[1] for k in schl(d, "arme") if "@" in k and float(k.split("@")[1]) <= 2.25})
    return [zelle(f"{b} Bit", ("arme",), f"had@{b}", [f"roh@{b}", f"roh_kanal@{b}", f"pca@{b}"], ("bpc",)) for b in bits]


def _k06(d):
    return [zelle(f"{ds} {cfg}", ("datensaetze", ds, "zeilen"), f"{cfg}|haar_alloc",
                  [f"{cfg}|haar_unif", f"{cfg}|orth_alloc", f"{cfg}|had"], ("acc",))
            for ds in schl(d, "datensaetze") for cfg in ("alle|2", "W1|2")]


def _k07(d):
    af = [a for a in schl(d, "arme") if a.startswith("adafactor")]
    return [zelle("4 Bit", ("arme",), "b4rot", ["b4roh"] + af, ("test_bpc",)),
            zelle("2 Bit", ("arme",), "b2rot", ["b2roh"] + af, ("test_bpc",))]


def dims(d):
    """K09/K10/K12 speichern flache Schluessel 'd64/gelu' unter 'zellen': liefert ['d64', 'd128', ...]."""
    return sorted({k.split("/")[0] for k in schl(d, "zellen")})


def fz(dim, idee, kontrollen, metrik=("bpc_test",)):
    """Zelle fuer flache Schluessel 'dim/arm'."""
    return (dim, ("zellen", f"{dim}/{idee}") + tuple(metrik), [(c, ("zellen", f"{dim}/{c}") + tuple(metrik)) for c in kontrollen])


def _k09(d):
    return [(f"{dm} {i}",) + fz(dm, i, ["gelu", "z2", "z2ln"])[1:] for dm in dims(d) for i in ("z2bn32", "z2bn8")]


def _k10(d):
    return [(f"{dm} {s}",) + fz(dm, f"born_{s}", [f"unif_{s}", f"top_{s}", f"schmal_{s}"])[1:] for dm in dims(d) for s in ("S4", "S8")]


def _k14(d):
    return [zelle(T, ("laeufe", T), "G_iso", ["D128_eqF", "G_naiv", "D64_eqF"], ("val_bpc",))
            for T in schl(d, "laeufe") if T.startswith("T")]


def _k15a(d):
    br, bd = d.get("A_bytelm", {}).get("bestes_rot"), d.get("A_bytelm", {}).get("bestes_drop")
    if not br or not bd:
        return []
    return [zelle("Byte-LM", ("A_bytelm", "arme"), br, [bd, "keine_0.0"], ("val_bpc",))]


def _k15b(d):
    out = []
    for ds in schl(d, "B_mlp"):
        sel = d["B_mlp"][ds].get("gewaehlt_auf_val", {})
        if "rot" in sel and "drop" in sel and "keine" in sel:
            out.append(zelle(ds, ("B_mlp", ds, "alle"), sel["rot"], [sel["drop"], sel["keine"]], ("test",)))
    return out


def _k17(d):
    return [zelle(f"{ds} {n}", (ds, n), "ff_z2", ["ff_relu", "mlp_flops_gleich"], ("acc_test",))
            for ds in schl(d) if ds.startswith("A_") for n in schl(d, ds) if isinstance(d[ds][n], dict)]


def _k17b(d):
    return [("Suche " + t, (b, t, "suche", "acc_test"), [("beste_zufall", (b, t, "beste_zufall", "acc_test")),
                                                         ("start_draw", (b, t, "start_draw", "acc_test"))])
            for b in schl(d) if b.startswith("B_suche") for t in ("kurz", "lang")]


def _k19(idee, kontrollen):
    return lambda d: [zelle(c, ("ergebnis", c), idee, kontrollen, ("acc_test",)) for c in schl(d, "ergebnis")]


def _e1_arme(d, ds, bits):
    """Kontrollen von E1: nur Arme im Byte-Budget (Gruppenskalen ohne Rotation, Zeilenskala ohne Rotation)."""
    arme = schl(d, "ergebnisse", ds, bits, "arme")
    return [a for a in arme if (a.startswith("roh_g") or a == "roh_zeile") and d["ergebnisse"][ds][bits]["arme"][a].get("im_budget", True)]


def _e01_zellen(idee, kontrollen_fn, extra=()):
    def f(d):
        out = []
        for ds in schl(d, "ergebnisse"):
            for b in schl(d, "ergebnisse", ds):
                if not b.startswith("bits"):
                    continue
                ks = kontrollen_fn(d, ds, b)
                z = zelle(f"{ds} {b}", ("ergebnisse", ds, b, "arme"), idee, ks, ("test", "mittel"))
                z[2].extend((x, ("ergebnisse", ds, b, x, "test")) for x in extra)
                out.append(z)
        return out
    return f


def _e02(d):
    out = []
    for ds in schl(d, "ergebnisse"):
        for b in schl(d, "ergebnisse", ds):
            if b.startswith("bits"):
                p = ("ergebnisse", ds, b)
                out.append((f"{ds} {b}", p + ("gross_quant", "had", "test"),
                            [("klein_flopgleich", p + ("klein", "flopgleich", "test")), ("roh", p + ("gross_quant", "roh", "test")),
                             ("orth", p + ("gross_quant", "orth", "test"))]))
    return out


def _e03(idee, kontrolle):
    return lambda d: [(idee, ("arme", idee, "test"), [(kontrolle, ("arme", kontrolle, "test")), ("klein_fp16", ("klein_fp16", "test"))])]


def _e05(d):
    out = []
    for w in schl(d, "breiten"):
        for b in schl(d, "breiten", w):
            if b.startswith("bits"):
                p = ("breiten", w, b)
                out.append((f"h{w} {b}", p + ("had", "test"), [("roh", p + ("roh", "test")), ("klein_fp16", p + ("klein_fp16", "test"))]))
    return out


def _e06(bits, idee, kontrollen):
    return lambda d: [(bits, (bits, "arme", idee, "test"),
                       [(k, (bits, "arme", k, "test")) for k in kontrollen] + [("klein_fp16", (bits, "klein_fp16", "test"))])]


def _e07(d):
    return [(f"{b} {k}", ("E7", b, "auswahl", k, "kurtosis_top", "test"), [("zufall", ("E7", b, "auswahl", k, "zufall", "mittel"))])
            for b in schl(d, "E7") for k in schl(d, "E7", b, "auswahl")]


def _e08(d):
    return [(f"{K} {b}", ("E8", K, f"nur_W1_topk_{b}", "test"),
             [("nirgends_topk", ("E8", K, f"nirgends_topk_{b}", "test")), ("ueberall_topk", ("E8", K, f"ueberall_topk_nur_genauigkeit_{b}", "test"))])
            for K in schl(d, "E8") for b in ("b2", "b3")]


def _e09(d):
    return [(f"{b} Bit", ("arme", f"had{b}", "test"), [("fp_bestval", ("arme", "fp_bestval", "test")), (f"rausch_had{b}", ("arme", f"rausch_had{b}", "test"))])
            for b in (3, 4)]


A, M = ("arme",), ("bpc",)
SPEC = {
    "K01": dict(titel="Schluessel-Superposition ueber Lagen", haupt=V(
        "geteilt_k (geteilte FFN + Vorzeichenschluessel)", "klein (dicht, gleiche Bytes), geteilt_o (ohne Schluessel)",
        lambda d: [zelle("-", A, "geteilt_k", ["klein", "geteilt_o"], M)], "min", "bpc"),
        neben=[V("breit_k (verbreitert, Schluessel)", "breit_o, normal", lambda d: [zelle("-", A, "breit_k", ["breit_o", "normal"], M)], "min", "bpc")]),
    "K02": dict(titel="Aufgaben-Superposition", haupt=V(
        "geteilt_k", "getrennt_B (gleiche Bytes), geteilt_o (ohne Schluessel)",
        lambda d: [zelle("-", A, "geteilt_k", ["getrennt_B", "geteilt_o"], ("acc_mittel",))], "max", "pp"),
        neben=[V("geteilt_k", "nur geteilt_o (Schluessel allein)", lambda d: [zelle("-", A, "geteilt_k", ["geteilt_o"], ("acc_mittel",))], "max", "pp")]),
    "K03": dict(titel="Vektor-Codebuch ohne Speicher", haupt=V(
        "zufall_rot (Seed-Codebuch, rotiert)", "skalar2_rot (I-048-Arm), skalar4_rot, e8_rot",
        lambda d: [zelle("-", A, "zufall_rot", ["skalar2_rot", "skalar4_rot", "e8_rot"], ("acc",))], "max", "pp"),
        neben=[V("rm_rot", "wie Haupt", lambda d: [zelle("-", A, "rm_rot", ["skalar2_rot", "skalar4_rot", "e8_rot"], ("acc",))], "max", "pp"),
               V("hex_rot", "wie Haupt", lambda d: [zelle("-", A, "hex_rot", ["skalar2_rot", "skalar4_rot", "e8_rot"], ("acc",))], "max", "pp")]),
    "K04": dict(titel="Eingefrorene Seed-Aufwaertsschicht", haupt=V(
        "zufall_s1024 (Zufalls-W1 + Zeilenskalen, breiter)", "dicht512 (gleiche Bytes), zufall_o1024 (ohne Skalen)",
        lambda d: [zelle("-", A, "zufall_s1024", ["dicht512", "zufall_o1024"], M)], "min", "bpc"),
        neben=[V("had_s1024 (R = Hadamard)", "zufall_s1024", lambda d: [zelle("-", A, "had_s1024", ["zufall_s1024"], M)], "min", "bpc")]),
    "K05": dict(titel="Cache-Bytes: Rotation, 1-Bit-Skizze, Haar-Alterung", haupt=V(
        "had@b (Rotation, b <= 2,25 Bit)", "roh@b, roh_kanal@b, pca@b", _k05, "min", "bpc"),
        neben=[V("alter_haar@2.25", "fenster, zufallfall, pca (2,25 Bit)",
                 lambda d: [zelle("2.25 Bit", A, "alter_haar@2.25", ["fenster@2.25", "zufallfall@2.25", "pca@2.25"], M)], "min", "bpc")]),
    "K06": dict(titel="Haar/Walsh-Basis fuer Gewichte", haupt=V(
        "haar_alloc (Haar + Bitallokation), 2 Bit, alle/W1", "haar_unif, orth_alloc, had", _k06, "max", "pp")),
    "K07": dict(titel="Optimiererzustand rotiert, 2-4 Bit", haupt=V(
        "b4rot / b2rot (rotierter Adam-Zustand)", "b4roh/b2roh, Adafactor", _k07, "min", "bpc")),
    "K08": dict(titel="Adam in gedrehter Basis", haupt=V(
        "had (Hadamard-Rotation des Gradienten)", "adam, vz (nur Vorzeichen), orth (Zufalls-Orthogonale)",
        lambda d: [zelle("-", A, "had", ["adam", "vz", "orth"], ("test_bpc",))], "min", "bpc")),
    "K09": dict(titel="Born-Normierung der FFN-Mitte", haupt=V(
        "z2bn32 / z2bn8 (z^2 + Born-Norm)", "gelu, z2, z2ln", _k09, "min", "bpc")),
    "K10": dict(titel="Born-Stichprobe der FFN-Spalten", haupt=V(
        "born_S (Stichprobe ~ Aktivierung^2), S4 und S8", "unif_S, top_S, schmal_S (FLOP-gleich dicht)", _k10, "min", "bpc")),
    "K11": dict(titel="Fliegen-Hash-Kopf", haupt=V(
        "fly_top (Top-k der Hadamard-Expansion)", "fly_zufallsspalten, fly_zufallsprobe, dicht_flopgleich",
        lambda d: [zelle(z, ("zellen", z), "fly_top", ["fly_zufallsspalten", "fly_zufallsprobe", "dicht_flopgleich"], ("acc_test",))
                   for z in schl(d, "zellen")], "max", "pp")),
    "K12": dict(titel="Bit-Baum / Hadamard-Code als Ausgabekopf", haupt=V(
        "bittree (8 binaere Entscheidungen)", "voll (Softmax), rang32 (gelernter Rang-32-Kopf)",
        lambda d: [fz(dm, "bittree", ["voll", "rang32"]) for dm in dims(d)], "min", "bpc"),
        neben=[V("had32 (festes Codebuch)", "rand32, rang32",
                 lambda d: [fz(dm, "had32", ["rand32", "rang32"]) for dm in dims(d)], "min", "bpc")]),
    "K13": dict(titel="Grob-zu-fein-Curriculum (Kontextlaenge)", haupt=V(
        "W64_256 (64->128->256)", "f32, f64, f128, f256 (fester Kontext, gleiche FLOPs)",
        lambda d: [zelle("ctx128", A, "W64_256", ["f32", "f64", "f128", "f256"], ("val_bpc_ctx128",))], "min", "bpc"),
        neben=[V("S32_256 (32->64->128->256)", "wie Haupt",
                 lambda d: [zelle("ctx128", A, "S32_256", ["f32", "f64", "f128", "f256"], ("val_bpc_ctx128",))], "min", "bpc")]),
    "K14": dict(titel="Wachsendes Modell", haupt=V(
        "G_iso (funktionserhaltend verbreitert)", "D128_eqF (gleiche FLOPs), G_naiv, D64_eqF", _k14, "min", "bpc")),
    "K15": dict(titel="Rotiertes Dropout und Schuss-Rauschen", haupt=V(
        "bestes rotiertes Dropout (Byte-LM, Val-bpc)", "bestes normales Dropout, kein Dropout", _k15a, "min", "bpc"),
        neben=[V("rotiertes Dropout (MLP, Test-Genauigkeit)", "normales Dropout, keins", _k15b, "max", "pp")]),
    "K16": dict(titel="Trotter-Stapel: Fastfood mit Skip", haupt=V(
        "S3skip", "S1, S3skip_zufall, Dicht1_F, Dicht3skip",
        lambda d: [zelle(ds, ("daten", ds, "arme"), "S3skip", ["S1", "S3skip_zufall", "Dicht1_F", "Dicht3skip"], ("test",))
                   for ds in schl(d, "daten")], "max", "pp")),
    "K17": dict(titel="Zufallsmerkmale + Ridge-Kopf", haupt=V(
        "ff_z2 (Fastfood^2 + Ridge)", "ff_relu, mlp_flops_gleich", _k17, "max", "pp"),
        neben=[V("Bit-Flip-Suche", "beste Zufalls-Draws, Start-Draw", _k17b, "max", "pp")]),
    "K18": dict(titel="2-Bit-Zwilling als Entwurf", haupt=V(
        "zwilling_had (Akzeptanzrate)", "zwilling_roh, klein_dicht",
        lambda d: [zelle("-", ("entwuerfe",), "zwilling_had", ["zwilling_roh", "klein_dicht"], ("stichprobe_akzeptanz",))], "max", "pp")),
    "K19": dict(titel="Lokale Bloecke auf der Pixelseite", haupt=V(
        "z16 (Z-Kurve, 4x4-Kacheln)", "perm, dicht_flops, dicht_param", _k19("z16", ["perm", "dicht_flops", "dicht_param"]), "max", "pp"),
        neben=[V("z64", "wie Haupt", _k19("z64", ["perm", "dicht_flops", "dicht_param"]), "max", "pp"),
               V("streif", "wie Haupt", _k19("streif", ["perm", "dicht_flops", "dicht_param"]), "max", "pp"),
               V("kron (Kronecker-Filter)", "lr, dicht_kronmacs", _k19("kron", ["lr", "dicht_kronmacs"]), "max", "pp")]),
    "K20": dict(titel="Beschraenkte Schwellen-Aktivierung (Hill)", haupt=V(
        "hill (bei 4 Bit trainiert)", "gelu, z2, relu6",
        lambda d: [zelle("-", A, "hill", ["gelu", "z2", "relu6"], ("bpc_test_trainierte_bits",))], "min", "bpc")),
    # --- Ergaenzungen (konv_kandidaten.md Abschnitt C); datei = Dateikuerzel, falls anders als die ID
    "E1": dict(datei="E01", titel="Gruppenskalen statt Rotation", haupt=V(
        "rotv_zeile (Rotation + Zeilenskala)", "roh_g16..g128, roh_zeile (im Budget), klein_fp16",
        _e01_zellen("rotv_zeile", _e1_arme, ("klein_fp16",)), "max", "pp")),
    "E2": dict(datei="E02", titel="Trainings-FLOP-Gleichheit", haupt=V(
        "gross_quant had (Rotation)", "klein flopgleich, roh, orth", _e02, "max", "pp")),
    "E3": dict(datei="E03", titel="QAT-Paket", haupt=V(
        "qat_had", "qat_roh, klein_fp16", _e03("qat_had", "qat_roh"), "max", "pp"),
        neben=[V("zeno_had", "zeno_roh, klein_fp16", _e03("zeno_had", "zeno_roh"), "max", "pp"),
               V("lin_had", "lin_roh, klein_fp16", _e03("lin_had", "lin_roh"), "max", "pp")]),
    "E4": dict(datei="E01", titel="Block-Hadamard", haupt=V(
        "rotb32_g32", "roh_g32 (Blockskala ohne Rotation)", _e01_zellen("rotb32_g32", lambda d, ds, b: ["roh_g32"]), "max", "pp"),
        neben=[V("rotb8_g8", "roh_g8", _e01_zellen("rotb8_g8", lambda d, ds, b: ["roh_g8"]), "max", "pp"),
               V("rotb128_g128", "roh_g128", _e01_zellen("rotb128_g128", lambda d, ds, b: ["roh_g128"]), "max", "pp")]),
    "E5": dict(datei="E05", titel="Breitenskalierung", haupt=V("had (Rotation)", "roh, klein_fp16 (gleiche Bytes)", _e05, "max", "pp")),
    "E6": dict(datei="E06", titel="Nur Skalen/Bias nachtrainieren", haupt=V(
        "skala_had (2 Bit)", "skala_roh, ste_had, klein_fp16", _e06("bits2", "skala_had", ["skala_roh", "ste_had"]), "max", "pp")),
    "E7": dict(datei="E07", titel="Kurtosis-Gesetz, selektive Rotation", haupt=V(
        "Rotation der Kurtosis-Top-k-Schichten", "Zufallsauswahl gleicher Groesse", _e07, "min", "bpc")),
    "E8": dict(datei="E07", titel="Rotation nur auf der dichten Seite", haupt=V(
        "nur_W1_topk", "nirgends_topk, ueberall_topk", _e08, "min", "bpc")),
    "E9": dict(datei="E09", titel="Quantisierung als Regularisierer", haupt=V(
        "had3/had4 (rotiert quantisiert)", "fp_bestval, Gauss-Gewichtsrauschen gleicher Staerke", _e09, "min", "bpc")),
    "E10": dict(datei="E01", titel="Entropiecodierung nach Rotation", haupt=V(
        "ent_rotv", "ent_roh", _e01_zellen("ent_rotv", lambda d, ds, b: ["ent_roh"]), "max", "pp")),
    "E11": dict(datei="E01", titel="Subtraktiver Dither", haupt=V(
        "dither_rotv_sub", "dither_rotv_nosub, _det, _stoch",
        _e01_zellen("dither_rotv_sub", lambda d, ds, b: ["dither_rotv_nosub", "dither_rotv_det", "dither_rotv_stoch"]), "max", "pp")),
    "E12": dict(datei="E06", titel="1-Bit-Netz mit Skalen-Nachtraining", haupt=V(
        "skala_had (1 Bit)", "skala_roh, ptq_roh, klein_fp16", _e06("bits1", "skala_had", ["skala_roh", "ptq_roh"]), "max", "pp")),
}


# ------------------------------------------------------------------ Laden
def kid_sortierschluessel(k):
    return (k[0], int(re.sub(r"\D", "", k) or 0))


def dateien(roh: Path, datei: str, schnell: bool):
    """Gibt {seed: Pfad}. Normal: roh/<KID>_s95?.json; --schnell: roh/<KID>*schnell*.json bzw. roh/schnell/<KID>_*.json (sonst wie normal)."""
    muster = [f"{datei}_s9[0-9][0-9].json"]
    if schnell:
        muster = [f"{datei}*schnell*.json", f"schnell/{datei}_s9[0-9][0-9].json"]
    out = {}
    for m in muster:
        for p in sorted(glob.glob(str(roh / m))):
            s = re.search(r"s(9\d\d)", Path(p).name)
            seed = int(s.group(1)) if s else None
            if seed is None:
                try:
                    seed = int(json.load(open(p)).get("seed", json.load(open(p)).get("_seed")))
                except Exception:
                    continue
            out[seed] = p
    return out


def lade(pfad):
    try:
        return json.load(open(pfad))
    except Exception as e:  # beschaedigte Datei: melden, ueberspringen
        print(f"WARNUNG: {pfad} nicht lesbar ({e})", file=sys.stderr)
        return None


# ------------------------------------------------------------------ Rechnen
def gewinn_seed(d, v):
    """(Gewinn, haerteste_Kontrolle(n), Zellenzahl) fuer einen Seed. Gewinn in bpc bzw. Prozentpunkten, positiv = Idee besser."""
    skala = 100.0 if v["einheit"] == "pp" else 1.0
    gs, namen = [], []
    for zelle_ in v["zellen"](d):
        name, ip, kontrollen = zelle_
        iw = g(d, *ip)
        kw = [(k, g(d, *p)) for k, p in kontrollen]
        kw = [(k, x) for k, x in kw if x is not None]
        if iw is None or not kw:
            continue
        k, best = (min if v["richtung"] == "min" else max)(kw, key=lambda t: t[1])
        gs.append((best - iw if v["richtung"] == "min" else iw - best) * skala)
        namen.append(k)
    if not gs:
        return None, [], 0
    return sum(gs) / len(gs), namen, len(gs)


def vergleiche(kid, v, roh, schnell, min_n):
    datei = SPEC[kid].get("datei", kid)
    fs = dateien(roh, datei, schnell)
    werte, haerteste, zellen_n = {}, [], []
    for seed, p in sorted(fs.items()):
        d = lade(p)
        if d is None:
            continue
        w, namen, nz = gewinn_seed(d, v)
        if w is not None:
            werte[seed] = w
            haerteste += namen
            zellen_n.append(nz)
    return auswerten(werte, haerteste, zellen_n, v, min_n, len(fs))


def auswerten(werte, haerteste, zellen_n, v, min_n, n_dateien):
    xs = list(werte.values())
    n = len(xs)
    r = dict(n=n, n_dateien=n_dateien, werte=werte, einheit=v["einheit"], schwelle=v["schwelle"], zellen=(max(zellen_n) if zellen_n else 0))
    if n == 0:
        r.update(mittel=None, sd=None, plus=0, t=None, ampel="grau", haerteste="-")
        return r
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else None
    plus = sum(1 for x in xs if x > 0)
    t = (m / (sd / math.sqrt(n))) if sd not in (None, 0) else (None if sd is None else (math.inf if m > 0 else -math.inf if m < 0 else 0.0))
    if n < min_n:
        ampel = "grau"
    elif m > v["schwelle"] and plus >= math.ceil(0.8 * n) and (n < 3 or (t is not None and t >= 2)):
        ampel = "gruen"
    elif m > 0 and plus >= math.ceil(0.6 * n):
        ampel = "gelb"
    else:
        ampel = "rot"
    h = max(set(haerteste), key=haerteste.count) if haerteste else "-"
    r.update(mittel=m, sd=sd, plus=plus, t=t, ampel=ampel, haerteste=h)
    return r


# ------------------------------------------------------------------ konv_kandidaten.md
def lese_konv(pfad: Path):
    """{ID: dict(p_lead, p_min, herkunft[list], titel)} aus den Tabellen (Zeilen '| K01 |' bzw. '| E1 |')."""
    out = {}
    for z in pfad.read_text(encoding="utf-8").splitlines():
        if not z.startswith("|"):
            continue
        c = [x.strip() for x in z.strip().strip("|").split("|")]
        if len(c) < 5 or not re.fullmatch(r"[KE]\d+", c[0]):
            continue
        pz = c[-2]                       # vorletzte Spalte = Vorab-P (letzte = Jury)
        ps = [int(x) for x in re.findall(r"(\d+)\s*%", pz)]
        if not ps:
            continue
        her = []
        for k in re.findall(r"k([1-5])#", c[2]):
            if f"k{k}" not in her:
                her.append(f"k{k}")
        out[c[0]] = dict(p_lead=ps[0] / 100, p_min=min(ps) / 100, herkunft=her, titel=c[1], pzelle=pz)
    return out


# ------------------------------------------------------------------ Ausgabe
def fmt(x, nd=3, vz=True):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:+.{nd}f}" if vz else f"{x:.{nd}f}"


AMPEL_ZEICHEN = {"gruen": "GRUEN", "gelb": "GELB", "rot": "ROT", "grau": "grau"}


def zeile_vgl(kid, r, v, extra=""):
    e = "bpc" if r["einheit"] == "bpc" else "pp"
    nd = 3 if e == "bpc" else 2
    return (f"| {kid} | {v['idee']} | {v['kontrolle']} | {e} | {r['n']} | {fmt(r['mittel'], nd)} | {fmt(r['sd'], nd, False)} | "
            f"{r['plus']}/{r['n']} | {fmt(r['t'], 1)} | {r['haerteste']} | {AMPEL_ZEICHEN[r['ampel']]}{extra} |")


def brier(paare):
    return statistics.fmean((p - o) ** 2 for p, o in paare) if paare else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roh", default=str(HIER / "roh"))
    ap.add_argument("--konv", default=str(HIER / "konv_kandidaten.md"))
    ap.add_argument("--nur", default="", help="Komma-Liste, z. B. K01,K05,E2")
    ap.add_argument("--schnell", action="store_true", help="Schnell-JSONs lesen (NUR Formattest)")
    ap.add_argument("--min-n", type=int, default=MIN_N, help="Mindestzahl Seeds fuer eine Ampel (Standard 3)")
    ap.add_argument("--out", default="")
    o = ap.parse_args()
    roh = Path(o.roh)
    konv = lese_konv(Path(o.konv)) if Path(o.konv).exists() else {}
    wahl = [k.strip() for k in o.nur.split(",") if k.strip()] or sorted(SPEC, key=kid_sortierschluessel)

    L = ["# Auswertung Erkundungspaket Welle 2" + (" -- FORMATTEST mit Schnell-JSONs, nicht aussagekraeftig" if o.schnell else ""), ""]
    L += ["Erkundung auf Tuning-Seeds, kein Befund. Gewinn = Idee besser als die haerteste Kontrolle (je Seed und Zelle die beste Kontrolle), "
          "gepaart ueber Seeds; bpc (kleiner ist besser) bzw. Prozentpunkte. "
          f"Ampel: gruen = Mittel > Schwelle (0,010 bpc / 0,5 pp), plus >= 80 % der Seeds, t >= 2; gelb = Mittel > 0 und plus >= 60 %; rot sonst; "
          f"grau = weniger als {o.min_n} Seeds.", ""]

    kopf = "| ID | Idee | haerteste Kontrolle(n) | Einheit | n | Mittel | sd | plus/n | t | meist haerteste | Ampel |\n|---|---|---|---|---|---|---|---|---|---|---|"
    haupt, neben, tab = [], [], {}
    for kid in wahl:
        if kid not in SPEC:
            print(f"WARNUNG: {kid} nicht in SPEC", file=sys.stderr)
            continue
        s = SPEC[kid]
        r = vergleiche(kid, s["haupt"], roh, o.schnell, o.min_n)
        tab[kid] = r
        p = konv.get(kid, {})
        extra = f" | {int(round(p['p_lead'] * 100))} %" if p else " | ?"
        haupt.append(zeile_vgl(kid, r, s["haupt"], extra))
        for i, nv in enumerate(s.get("neben", [])):
            rn = vergleiche(kid, nv, roh, o.schnell, o.min_n)
            neben.append(zeile_vgl(f"{kid}.{i + 1}", rn, nv))
    kopf_h = kopf.replace("| Ampel |", "| Ampel | Vorab-P |") + "---|"
    kopf_h = kopf_h.replace("|---|---|---|---|---|---|---|---|---|---|---|---|", "|---|---|---|---|---|---|---|---|---|---|---|---|")
    L += ["## 1. Hauptvergleich je Kandidat", "", kopf_h, *haupt, ""]
    # Dateiabdeckung
    L += ["Seeds je Kandidat (Datei-Seeds): " + ", ".join(f"{k} {tab[k]['n']}/{tab[k]['n_dateien']}" for k in tab) + "  (ausgewertet/Dateien)", ""]
    if neben:
        L += ["## 2. Nebenvergleiche (nicht im Brier)", "", kopf, *neben, ""]

    # Brier
    zeilen, paare_lead, paare_min, gruppen_erste, gruppen_alle = [], [], [], {}, {}
    for kid, r in tab.items():
        p = konv.get(kid)
        if not p or r["ampel"] == "grau":
            continue
        ob = 1.0 if r["ampel"] == "gruen" else 0.0
        paare_lead.append((p["p_lead"], ob))
        paare_min.append((p["p_min"], ob))
        zeilen.append(f"| {kid} | {p['titel']} | {','.join(p['herkunft'])} | {p['pzelle']} | {int(ob)} | {(p['p_lead'] - ob) ** 2:.3f} | {(p['p_min'] - ob) ** 2:.3f} | {AMPEL_ZEICHEN[r['ampel']]} |")
        if p["herkunft"]:
            gruppen_erste.setdefault(p["herkunft"][0], []).append((p["p_lead"], p["p_min"], ob))
            for h in p["herkunft"]:
                gruppen_alle.setdefault(h, []).append((p["p_lead"], p["p_min"], ob))
    L += ["## 3. Brier-Wert der Vorab-P (Ereignis: Ampel gruen)", ""]
    if not paare_lead:
        L += ["Keine Kandidaten mit Ampel (zu wenige Seeds) -- kein Brier-Wert.", ""]
    else:
        rate = statistics.fmean(o_ for _, o_ in paare_lead)
        b, bm = brier(paare_lead), brier(paare_min)
        b_ref, b_rate = 0.25, brier([(rate, o_) for _, o_ in paare_lead])
        L += [f"Kandidaten im Brier: {len(paare_lead)}; beobachtete Gruen-Rate {rate:.2f}; mittlere Vorab-P {statistics.fmean(p for p, _ in paare_lead):.2f}.", "",
              f"- Brier (fuehrende P): **{b:.3f}**  | vorsichtige P (kleinste Zahl der Zelle): {bm:.3f}",
              f"- Referenz p = 0,5: 0,250  | Referenz p = Gruen-Rate: {b_rate:.3f}",
              f"- Brier-Skill gegen p = Gruen-Rate: {(1 - b / b_rate) if b_rate else float('nan'):+.2f}  (positiv = Vorab-P informativer als die Basisrate)", ""]
        L += ["| ID | Titel | Herkunft | Vorab-P | gruen | (p-o)^2 | (p_vors.-o)^2 | Ampel |", "|---|---|---|---|---|---|---|---|", *zeilen, ""]
        for titel, gr in (("erste Herkunft (jeder Kandidat genau einmal)", gruppen_erste), ("alle Herkuenfte (Mehrfachzaehlung)", gruppen_alle)):
            L += [f"### Nach Denkwerkzeug -- {titel}", "", "| Gruppe | Denkwerkzeug | n | mittlere P | Gruen-Anteil | Brier | Brier vorsichtig |", "|---|---|---|---|---|---|---|"]
            for k in sorted(gr):
                a = gr[k]
                L.append(f"| {k} | {DENKWERKZEUG[k]} | {len(a)} | {statistics.fmean(x[0] for x in a):.2f} | {statistics.fmean(x[2] for x in a):.2f} | "
                         f"{brier([(x[0], x[2]) for x in a]):.3f} | {brier([(x[1], x[2]) for x in a]):.3f} |")
            L.append("")
        L += ["Lesart: Brier < 0,25 besser als Raten mit 50 %; mittlere P ueber Gruen-Anteil = Gruppe war zu optimistisch (Kalibrierung). "
              "Bei n <= 6 je Gruppe nur grobe Tendenz.", ""]
    ausgabe = "\n".join(L)
    print(ausgabe)
    if o.out:
        Path(o.out).write_text(ausgabe + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
