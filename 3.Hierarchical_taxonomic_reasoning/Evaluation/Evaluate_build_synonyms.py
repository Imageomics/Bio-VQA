#!/usr/bin/env python3
"""
build_synonyms.py
=================
Download the GBIF Backbone Taxonomy and build a normalized
   synonym_name -> accepted_name
lookup, saved as a pickle for rescore_soft.py.

This replaces the hand-written LEGACY_CLASS dict: instead of enumerating
Reptilia/Chondrichthyes/etc. by hand, every deprecated name (Magnoliophyta,
Osteichthyes, Filicales, ...) resolves to its currently-accepted name via
GBIF's own taxonomicStatus / acceptedNameUsageID columns.

Output: synonyms.pkl  ->  {"map": {norm_synonym: norm_accepted}, "accepted": set(norm_accepted)}

Run:
    python build_synonyms.py                    # downloads ~600MB zip, builds lookup
    python build_synonyms.py --backbone /path/to/backbone.zip   # use a local copy
    python build_synonyms.py --out synonyms.pkl
"""

import os
import re
import sys
import csv
import sys as _sys
import pickle

# GBIF Taxon.tsv has description fields far larger than the 131072 default
try:
    csv.field_size_limit(_sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)
import zipfile
import argparse
import unicodedata
import urllib.request

BACKBONE_URL = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def norm(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ").replace("_", " ")
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()


def download(url, dest):
    print(f"[dl] {url}\n     -> {dest}")

    def hook(block, bsize, total):
        if total > 0:
            pct = min(100, block * bsize * 100 // total)
            if block % 200 == 0 or pct == 100:
                sys.stdout.write(f"\r     {pct:3d}%  "
                                 f"{block * bsize // (1024*1024):,} MB")
                sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, hook)
    print("\n[dl] done")


def find_taxon_tsv(zf):
    for n in zf.namelist():
        base = n.split("/")[-1].lower()
        if base in ("taxon.tsv", "taxon.txt"):
            return n
    raise SystemExit(f"[err] no Taxon.tsv in zip; members: {zf.namelist()[:20]}")


def build(backbone_zip, out_path):
    with zipfile.ZipFile(backbone_zip) as zf:
        member = find_taxon_tsv(zf)
        print(f"[build] reading {member}")

        # pass 1: taxonID -> accepted canonical name (for id-based resolution)
        # pass 2: synonym canonical -> accepted canonical
        # GBIF Taxon.tsv is tab-separated with a header row.
        id2name = {}
        rows = []
        with zf.open(member) as fh:
            text = (line.decode("utf-8", "replace") for line in fh)
            rdr = csv.DictReader(text, delimiter="\t")
            cols = rdr.fieldnames or []
            # tolerate GBIF's column-name variants
            def col(*names):
                for nm in names:
                    if nm in cols:
                        return nm
                return None
            c_id = col("taxonID", "id")
            c_name = col("canonicalName", "scientificName")
            c_status = col("taxonomicStatus")
            c_acc = col("acceptedNameUsageID", "acceptedNameUsageId", "acceptedKey")
            if not (c_id and c_name and c_status):
                raise SystemExit(f"[err] unexpected columns: {cols[:15]}")
            print(f"[build] columns: id={c_id} name={c_name} "
                  f"status={c_status} accepted_id={c_acc}")

            n = 0
            for r in rdr:
                n += 1
                tid = r.get(c_id, "").strip()
                nm = norm(r.get(c_name, ""))
                st = (r.get(c_status, "") or "").strip().lower()
                acc = (r.get(c_acc, "") or "").strip() if c_acc else ""
                if tid and nm:
                    id2name[tid] = nm
                rows.append((nm, st, acc))
                if n % 1_000_000 == 0:
                    print(f"[build]   {n:,} rows")
        print(f"[build] {n:,} rows total, {len(id2name):,} names with ids")

        syn = {}
        accepted = set()
        n_syn = 0
        for nm, st, acc in rows:
            if not nm:
                continue
            is_syn = "synonym" in st or st in ("doubtful", "misapplied")
            if is_syn and acc:
                target = id2name.get(acc)
                if target and target != nm:
                    syn[nm] = target
                    n_syn += 1
            elif st in ("accepted", "valid"):
                accepted.add(nm)

        print(f"[build] synonym->accepted pairs : {len(syn):,}")
        print(f"[build] accepted names          : {len(accepted):,}")

        with open(out_path, "wb") as fh:
            pickle.dump({"map": syn, "accepted": accepted}, fh)
        print(f"[build] wrote {out_path}")

        # sanity: the names we care about
        for probe in ["magnoliophyta", "reptilia", "osteichthyes",
                      "chondrichthyes", "anthophyta", "coniferophyta"]:
            tgt = syn.get(probe)
            print(f"   probe  {probe:16s} -> {tgt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None,
                    help="local backbone.zip; if omitted, download it")
    ap.add_argument("--out", default="synonyms.pkl")
    ap.add_argument("--workdir", default=".")
    args = ap.parse_args()

    zip_path = args.backbone
    if zip_path is None:
        zip_path = os.path.join(args.workdir, "backbone.zip")
        if not os.path.exists(zip_path):
            download(BACKBONE_URL, zip_path)
        else:
            print(f"[dl] reusing existing {zip_path}")

    build(zip_path, args.out)


if __name__ == "__main__":
    main()