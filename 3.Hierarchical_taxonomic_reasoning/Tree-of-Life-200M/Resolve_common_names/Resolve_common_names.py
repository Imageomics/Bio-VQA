"""
Stage 1 + 2 of the common-name pipeline.

Stage 1  manifests -> English/ASCII/wordlist triage
    reports/resolve_common_name_english_ascii.csv   resolved here
    reports/fix_it.txt                              human-readable report
    reports/taxa_to_resolve.txt                     machine handoff to stage 2

Stage 2  unresolved taxa -> iNaturalist / GBIF / Wikipedia
    reports/resolve_common_name_inat_wiki.csv

Then you fill the remainder by hand into resolve_common_name_manual_check.csv
(columns: taxa, resolved_common_name, source) and run
finalize_resolve_common_names.py.

is_english cascade
  gbif        language code in the GBIF backbone       (authoritative, --gbif)
  genus       the name is just the genus               -> yes
  non_ascii   contains non-ASCII letters               -> no
  wordlist    share of dictionary tokens: >=0.5 yes, 0.0 no, between unknown

Eponyms are language-neutral. The NAME IS NEVER ALTERED -- "Peters' dwarf
frog" is stored and emitted in full. The eponym token is only skipped when
counting dictionary words, so the verdict rests on the rest of the name:
"Peters' dwarf frog" passes on "dwarf frog", while "Nordmann's vargspindel"
fails on "vargspindel" instead of being auto-passed by the apostrophe.

Usage
-----
  python resolve_common_names_pipeline.py BASE_DIR \\
      --gbif 2022-11-23 --out-dir reports/ --contact tes@testgmail.com

  --gbif takes a dated GBIF backbone version (downloaded and cached on first
  use), or 'current', or a URL, or a path to an already-extracted dir/.zip.
  Prefer a dated version: 'current' moves, so the answer key would not be
  reproducible.

  # stage 1 only, no network
  python resolve_common_names_pipeline.py BASE_DIR --skip-api
"""


import argparse
import csv
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

# --------------------------------------------------------------------------
# Stage 1 -- language triage
# --------------------------------------------------------------------------

ENGLISH_CODES = {"en", "eng", "english"}


def lang_primary(code):  
    return (code or "").strip().lower().replace("_", "-").split("-")[0]

MANIFEST_COLS = {"scientific_name", "common", "uuid"}

PUNCT_MAP = {0x2019: "'", 0x2018: "'", 0x00B4: "'", 0x0060: "'", 0x201B: "'",
             0x02BC: "'", 0x2032: "'", 0x2013: "-", 0x2014: "-", 0x2212: "-",
             0x201C: '"', 0x201D: '"', 0x00A0: " "}

EPONYM = re.compile(r"^[A-Za-z]{2,}(?:'s|s')$")

COMPOUND_BLOCK = {"berg", "lod", "den", "gran", "mos", "sten", "dun", "kop",
                  "fla", "bar", "hag", "sam", "kus"}

NAME_FIXES = {
    "longjawed orb weavers": "Long-jawed orb weavers",
    "longjawed orb weaver": "Long-jawed orb weaver",
}


def norm(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(s.casefold().split())


def depunct(s):
    return (s or "").translate(PUNCT_MAP)


def has_non_ascii(s):
    return any(ord(c) > 127 and unicodedata.category(c).startswith("L")
               for c in depunct(s))


def apply_fix(name):
    return NAME_FIXES.get(norm(name), name)


def read_manifests(base_dir):
    sci_names = defaultdict(Counter)
    sci_files = defaultdict(set)
    scanned, n_files, n_fixed = [], 0, 0
    for root, _, files in os.walk(base_dir):
        for name in sorted(files):
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, newline="", encoding="utf-8",
                          errors="replace") as fh:
                    header = next(csv.reader(fh), [])
            except (OSError, StopIteration):
                continue
            if not MANIFEST_COLS <= {c.strip() for c in header}:
                continue
            n_files += 1
            n_rows, seen = 0, set()
            with open(path, newline="", encoding="utf-8",
                      errors="replace") as fh:
                for row in csv.DictReader(fh):
                    sci = (row.get("scientific_name") or "").strip()
                    common = (row.get("common") or "").strip()
                    if not (sci and common):
                        continue
                    fixed = apply_fix(common)
                    if fixed != common:
                        n_fixed += 1
                    sci_names[sci][fixed] += 1
                    sci_files[sci].add(name)
                    seen.add(sci)
                    n_rows += 1
            scanned.append((name, n_rows, len(seen)))
            print("file\t%-32s %8d rows %7d taxa" % (name, n_rows, len(seen)))
    return sci_names, sci_files, scanned, n_files, n_fixed


GBIF_BASE = "https://hosted-datasets.gbif.org/datasets/backbone"
VERSION_RE = re.compile(r"^(?:current|\d{4}-\d{2}-\d{2})$")


def gbif_url(spec):
    """Map a --gbif spec to a download URL, or None if it is a local path."""
    if os.path.exists(spec):
        return None
    if spec.startswith(("http://", "https://")):
        return spec
    if VERSION_RE.match(spec):
        return "%s/%s/backbone.zip" % (GBIF_BASE, spec)
    raise SystemExit(
        "--gbif %r is not an existing path, a URL, or a version.\n"
        "\n"
        "If you meant a local file, check the path -- %r was not found\n"
        "relative to the current directory (%s).\n"
        "\n"
        "To download instead, pass a dated version and the script will fetch\n"
        "and cache it for you:\n"
        "    --gbif 2022-11-23\n"
        "\n"
        "Browse available versions at %s"
        % (spec, spec, os.getcwd(), GBIF_BASE))


def download(url, dest, ua):
    """Stream a URL to dest, writing to dest.part first."""
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    print("downloading %s" % url)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done, chunk = 0, 1 << 20
        with open(part, "wb") as fh:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if total:
                    sys.stdout.write("\r  %6.1f%%  %.2f / %.2f GB"
                                     % (100.0 * done / total,
                                        done / 1e9, total / 1e9))
                else:
                    sys.stdout.write("\r  %.2f GB" % (done / 1e9))
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(part, dest)


def ensure_gbif(spec, cache_dir, ua):
    """Return a path load_languages() can read, downloading if needed."""
    url = gbif_url(spec)
    if url is None:
        return spec

    version = spec if VERSION_RE.match(spec) else "custom"
    vdir = os.path.join(cache_dir, version)
    tsv = os.path.join(vdir, "VernacularName.tsv")
    if os.path.exists(tsv):
        print("using cached %s" % tsv)
        return vdir

    os.makedirs(vdir, exist_ok=True)
    zpath = os.path.join(vdir, "backbone.zip")
    if not os.path.exists(zpath):
        download(url, zpath, ua)

    with zipfile.ZipFile(zpath) as zf:
        member = next((n for n in zf.namelist()
                       if os.path.basename(n) == "VernacularName.tsv"), None)
        if member is None:
            raise SystemExit("no VernacularName.tsv inside %s" % zpath)
        print("extracting %s" % member)
        with zf.open(member) as src, open(tsv, "wb") as dst:
            while True:
                buf = src.read(1 << 20)
                if not buf:
                    break
                dst.write(buf)
    print("extracted %s (%.2f GB)  -- backbone.zip may now be deleted"
          % (tsv, os.path.getsize(tsv) / 1e9))
    return vdir


def load_languages(path):
    def stream():
        if os.path.isdir(path):
            full = os.path.join(path, "VernacularName.tsv")
            if not os.path.exists(full):
                raise SystemExit("missing VernacularName.tsv in %s" % path)
            return open(full, newline="", encoding="utf-8", errors="replace")
        zf = zipfile.ZipFile(path)
        members = {os.path.basename(n): n for n in zf.namelist()}
        if "VernacularName.tsv" not in members:
            raise SystemExit("missing VernacularName.tsv in %s" % path)
        return io.TextIOWrapper(zf.open(members["VernacularName.tsv"]),
                                encoding="utf-8", errors="replace")

    langs = defaultdict(set)
    with stream() as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            name = (row.get("vernacularName") or "").strip()
            if name:
                langs[norm(name)].add(lang_primary(row.get("language")))
    return langs


def load_wordlist(path=None):
    if path:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return {w.strip().casefold() for w in fh if w.strip()}
    try:
        from english_words import get_english_words_set
        return get_english_words_set(["web2"], lower=True)
    except Exception:
        pass
    for cand in ("/usr/share/dict/words", "/usr/dict/words"):
        if os.path.exists(cand):
            with open(cand, encoding="utf-8", errors="replace") as fh:
                return {w.strip().casefold() for w in fh if w.strip()}
    return None


def token_is_english(tok, words, compound):
    t = tok.casefold()
    if t in words:
        return True
    for suf in ("s", "es", "ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            b = t[:-len(suf)]
            if b in words:
                return True
            if suf in ("ing", "ed"):
                if b + "e" in words:
                    return True
                if len(b) > 1 and b[-1] == b[-2] and b[:-1] in words:
                    return True
    if t.endswith("ies") and len(t) > 4 and t[:-3] + "y" in words:
        return True
    if not compound:
        return False
    for i in range(3, len(t) - 2):
        a, b = t[:i], t[i:]
        if a in COMPOUND_BLOCK or b in COMPOUND_BLOCK:
            continue
        if token_is_english(a, words, False) and token_is_english(b, words, False):
            return True
    return False


def english_fraction(name, words, compound):

    toks = [t for t in re.split(r"[^A-Za-z']+", depunct(name))
            if len(t.strip("'")) > 1]
    kept = [t for t in toks if not EPONYM.match(t)]
    if not kept:
        return None            # nothing but eponyms -> undecidable
    return sum(token_is_english(t.strip("'"), words, compound)
               for t in kept) / len(kept)


def english_verdict(name, langs, words, compound, sci=None, tally=None):

    codes = langs.get(norm(name)) if langs is not None else None
    has_en = bool(codes) and any(c in ENGLISH_CODES for c in codes)
    has_foreign = bool(codes) and any(c and c not in ENGLISH_CODES for c in codes)
    if tally is not None and codes:
        tally["gbif_matched"] += 1
        if has_en:
            tally["gbif_english"] += 1
        elif has_foreign:
            tally["gbif_foreign"] += 1
        else:
            tally["gbif_blank_lang"] += 1

    if has_en:
        return "yes", "gbif"
    if sci and norm(name) == norm(sci).split(" ")[0]:
        return "yes", "genus"
    if has_non_ascii(name):
        return "no", "non_ascii"
    if words:
        frac = english_fraction(name, words, compound)
        if frac is not None:
            if frac >= 0.5:
                return "yes", "wordlist"
            if has_foreign:
                return "no", "gbif"
            if frac == 0.0:
                return "no", "wordlist"
            return "unknown", "wordlist"
    if has_foreign:
        return "no", "gbif"
    return "unknown", "none"


def pick(entries):
    """entries: [(form, n_rows, verdict, evidence)] -> (form, evidence, how)"""
    for want in ("yes", "unknown"):
        pool = [e for e in entries if e[2] == want]
        if pool:
            pool.sort(key=lambda e: (-e[1], e[0]))
            return pool[0][0], pool[0][3], want
    return "", "", "drop"


def fmt_tuple(forms):
    return "[" + ", ".join('"%s"' % f.replace('"', "'") for f in forms) + "]"


# --------------------------------------------------------------------------
# Stage 2 -- API resolution
# --------------------------------------------------------------------------

INAT_API = "https://api.inaturalist.org/v1/taxa"
GBIF_MATCH = "https://api.gbif.org/v1/species/match"
GBIF_SPECIES = "https://api.gbif.org/v1/species"
WIKI_API = "https://en.wikipedia.org/w/api.php"

OK_RANKS = {"species", "subspecies", "variety", "form", "hybrid"}
SCHEMA = 4

BAD_SUBSTRINGS = ("species", "genus", "family", "subfamily", "order ",
                  "http", "sensu ", " auct")
BAD_SUFFIXES = (" group", " complex", " section", " subgenus", " clade",
                " aggregate", " agg.", " sensu lato", " s.l.", " s. lat.")


def fetch_json(url, params, ua, timeout=30, retries=3):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}" if params else url,
                                 headers={"User-Agent": ua})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 ** attempt)
    return None


def clean_candidate(cand, *scinames):
    if not cand:
        return None
    cand = re.sub(r"\s+", " ", str(cand)).strip(" \t\n'\"()[]")
    if not cand:
        return None
    low = cand.lower()
    if len(cand) > 60 or len(cand.split()) > 6:
        return None
    if any(low.endswith(s) for s in BAD_SUFFIXES):
        return None
    if any(b in low for b in BAD_SUBSTRINGS):
        return None
    if not re.search(r"[A-Za-z]{3}", cand):
        return None
    if not re.fullmatch(r"[A-Za-z0-9 '\-\.\u00C0-\u024F]+", cand):
        return None
    for sci in scinames:
        if not sci:
            continue
        toks = sci.lower().split()
        for part in [sci.lower()] + toks[1:]:
            if len(part) > 3 and part in low:
                return None
    return cand


def inat_lookup(sciname, ua):
    data = fetch_json(INAT_API, {"q": sciname, "per_page": 30, "locale": "en",
                                 "is_active": "true"}, ua)
    if data is None:
        return None, None
    target = sciname.strip().lower()
    for res in data.get("results", []):
        if res.get("name", "").strip().lower() != target:
            continue
        if (res.get("rank") or "").lower() not in OK_RANKS:
            continue
        cand = clean_candidate(res.get("preferred_common_name"), sciname)
        if cand:
            return cand, res.get("id")
    return None, None


def gbif_match(sciname, ua):
    data = fetch_json(GBIF_MATCH, {"name": sciname, "strict": "false"}, ua)
    if data is None or data.get("matchType") in (None, "NONE"):
        return None, None
    accepted = None
    if data.get("synonym"):
        acc = data.get("species") or data.get("scientificName")
        if acc:
            accepted = " ".join(acc.split()[:2])
    return data.get("usageKey"), accepted


def gbif_vernacular(key, sciname, ua):
    if not key:
        return None
    data = fetch_json(f"{GBIF_SPECIES}/{key}/vernacularNames", {"limit": 200}, ua)
    if data is None:
        return None
    for rec in data.get("results", []):
        if (rec.get("language") or "").lower() not in ("eng", "en"):
            continue
        cand = clean_candidate(rec.get("vernacularName"), sciname)
        if cand:
            return cand
    return None


LEAD_PATTERNS = [
    r"^SCINAME,\s+(?:also\s+)?(?:commonly\s+|sometimes\s+|variously\s+)?"
    r"(?:known|called|referred\s+to)\s+as\s+(?:the\s+)?(.+?)\s*[,.]",
    r"^SCINAME,\s+the\s+(.+?),\s+is\b",
    r"^The\s+(.+?)\s*\(\s*SCINAME\b",
    r"^SCINAME\s+is\s+a\s+species\s+of\s+[a-z\- ]+?\s+"
    r"(?:commonly\s+)?known\s+as\s+(?:the\s+)?(.+?)\s*[,.]",
    r"^SCINAME\s+is\s+(?:also\s+)?(?:commonly\s+|sometimes\s+)?"
    r"known\s+as\s+(?:the\s+)?(.+?)\s*[,.]",
]
INTRO_PATTERNS = [
    r"\bCommon\s+names?\s+(?:include|includes|are|is)\s+(?:the\s+)?"
    r"(.+?)\s*(?:\band\b|[,.;])",
    r"\bIt\s+is\s+commonly\s+known\s+as\s+(?:the\s+)?(.+?)\s*[,.;]",
]


def wikipedia_lookup(sciname, ua):
    data = fetch_json(WIKI_API,
                      {"action": "query", "format": "json", "formatversion": "2",
                       "redirects": "1", "titles": sciname, "prop": "extracts",
                       "exintro": "1", "explaintext": "1"}, ua)
    if data is None:
        return None, None
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None, None
    title = pages[0].get("title", "")
    extract = pages[0].get("extract", "") or ""

    if title.strip().lower() != sciname.strip().lower():
        cand = clean_candidate(title, sciname)
        if cand:
            return cand, title

    lead = extract.strip().split("\n")[0]
    lead = re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", lead)[0]
    masked = lead.replace(sciname, "SCINAME")
    for pat in LEAD_PATTERNS:
        m = re.search(pat, masked, flags=re.IGNORECASE)
        if m:
            cand = clean_candidate(m.group(1), sciname)
            if cand:
                return cand, title

    intro = re.sub(r"\s+", " ", extract).replace(sciname, "SCINAME")
    for pat in INTRO_PATTERNS:
        m = re.search(pat, intro, flags=re.IGNORECASE)
        if m:
            cand = clean_candidate(m.group(1), sciname)
            if cand:
                return cand, title
    return None, title


def api_resolve(sciname, ua):
    key, accepted = gbif_match(sciname, ua)

    def probe(name, k):
        i, iid = inat_lookup(name, ua)
        g = gbif_vernacular(k, name, ua)
        w, wt = wikipedia_lookup(name, ua)
        return {"inat": i, "inat_id": iid, "gbif": g, "gbif_key": k,
                "wiki": w, "wiki_title": wt}

    res = probe(sciname, key)
    if accepted and accepted.lower() != sciname.lower() \
            and not (res["inat"] or res["gbif"] or res["wiki"]):
        k2, _ = gbif_match(accepted, ua)
        res2 = probe(accepted, k2)
        if res2["inat"] or res2["gbif"] or res2["wiki"]:
            res = res2

    hits = [(res["inat"], "inat"), (res["gbif"], "gbif"), (res["wiki"], "wiki")]
    hits = [(c, s) for c, s in hits if c]
    resolved = hits[0][0] if hits else ""

    urls = []
    if resolved:
        r = resolved.lower()
        if (res["inat"] or "").lower() == r and res["inat_id"]:
            urls.append(f"https://www.inaturalist.org/taxa/{res['inat_id']}")
        if (res["wiki"] or "").lower() == r and res["wiki_title"]:
            t = urllib.parse.quote(str(res["wiki_title"]).replace(" ", "_"))
            urls.append(f"https://en.wikipedia.org/wiki/{t}")
        if (res["gbif"] or "").lower() == r and res["gbif_key"]:
            urls.append(f"https://www.gbif.org/species/{res['gbif_key']}")

    return {"taxa": sciname,
            "inat_common_name": res["inat"] or "",
            "wiki_common_name": res["wiki"] or "",
            "gbif_common_name": res["gbif"] or "",
            "resolved_common_name": resolved,
            "source": " ; ".join(urls),
            "_schema": SCHEMA}


def load_cache(path):
    cache, stale = {}, 0
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                    t = rec["taxa"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if rec.get("_schema") != SCHEMA:
                    stale += 1
                    cache.pop(t, None)
                    continue
                cache[t] = rec
    if stale:
        print("ignoring %d cache record(s) from an older schema" % stale)
    return cache


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--gbif",
                    help="GBIF backbone: an existing dir/.zip, a URL, or a "
                         "version to download such as 2022-11-23 or 'current'")
    ap.add_argument("--gbif-cache", default=None,
                    help="where downloads land (default: <out-dir>/gbif)")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--wordlist")
    ap.add_argument("--no-wordlist", action="store_true")
    ap.add_argument("--no-compound", action="store_true")
    ap.add_argument("--skip-api", action="store_true",
                    help="stage 1 only; no network")
    ap.add_argument("--contact", default="")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    p_ascii = os.path.join(od, "resolve_common_name_english_ascii.csv")
    p_fixit = os.path.join(od, "taxa_to_resolve_details.txt")
    p_todo = os.path.join(od, "taxa_to_resolve.txt")
    p_api = os.path.join(od, "resolve_common_name_inat_wiki.csv")
    p_cache = os.path.join(od, ".api_cache.jsonl")


    gbif_path = None

    if args.gbif:
        gcache = args.gbif_cache or os.path.join(od, "gbif")
        gbif_path = ensure_gbif(args.gbif, gcache, "biovqa_common_names/1.0")

    # ---------------- stage 1 ----------------
    sci_names, sci_files, scanned, n_files, n_fixed = read_manifests(args.base_dir)
    if not sci_names:
        raise SystemExit("no manifest CSVs found under %s" % args.base_dir)
    print("manifests: %d   taxa: %d   name fixes: %d"
          % (n_files, len(sci_names), n_fixed))

    langs = load_languages(gbif_path) if gbif_path else None
    words = None if args.no_wordlist else load_wordlist(args.wordlist)
    print("evidence: gbif=%s  wordlist=%s  compound=%s"
          % ("%d forms" % len(langs) if langs else "no",
             "%d words" % len(words) if words else "no",
             not args.no_compound))

    stats = Counter()
    ascii_rows, na_taxa, forms_by_taxon = [], [], {}

    for sci in sorted(sci_names):
        groups = defaultdict(Counter)
        for form, n in sci_names[sci].items():
            groups[norm(form)][form] += n
        entries = []
        for key in groups:
            form = sorted(groups[key].items(),
                          key=lambda kv: (-kv[1], kv[0]))[0][0]
            v, ev = english_verdict(form, langs, words,
                                    not args.no_compound, sci, stats)
            entries.append((form, sum(groups[key].values()), v, ev))
        entries.sort(key=lambda e: (-e[1], e[0]))
        forms_by_taxon[sci] = [e[0] for e in entries]

        resolved, evidence, how = pick(entries)
        stats["taxa_seen"] += 1
        stats["resolved_" + how] += 1
        if how == "drop":
            na_taxa.append((sci, entries, sorted(sci_files.get(sci, ()))))
        else:
            ascii_rows.append([sci, fmt_tuple(forms_by_taxon[sci]),
                               resolved, evidence])

    with open(p_ascii, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["taxa", "common_name", "resolved_common_name", "evidence"])
        w.writerows(ascii_rows)

    with open(p_todo, "w", encoding="utf-8") as fh:
        for sci, _, _ in na_taxa:
            fh.write(sci + "\n")

    with open(p_fixit, "w", encoding="utf-8") as fh:
        n_multi = sum(1 for _, e, _ in na_taxa if len(e) > 1)
        fh.write("# taxa with no English common name from stage 1: %d\n"
                 "#   %d with several names, none English\n"
                 "#   %d with a single non-English name\n\n"
                 % (len(na_taxa), n_multi, len(na_taxa) - n_multi))
        for name, nr, nt in sorted(scanned):
            fh.write("file\t%-32s %8d rows %7d taxa\n" % (name, nr, nt))
        fh.write("\n")
        by_file = defaultdict(list)
        for sci, entries, files in na_taxa:
            by_file["; ".join(files) or "(unknown)"].append((sci, entries))
        for fname in sorted(by_file):
            fh.write("=" * 66 + "\nfile\t%s\n" % fname + "=" * 66 + "\n\n")
            for sci, entries in by_file[fname]:
                fh.write("%s   [%d name%s]\n"
                         % (sci, len(entries), "" if len(entries) == 1 else "s"))
                for form, nr, v, ev in entries:
                    fh.write("    %-42s %6d rows   %s / %s\n"
                             % (form, nr, v, ev))
                fh.write("\n")

    print("-" * 52)
    for k in ("taxa_seen", "resolved_yes", "resolved_unknown", "resolved_drop"):
        print("  %-24s %10d" % (k, stats[k]))
    if langs is not None:
        print("  %-24s %10s" % ("-- gbif evidence --", ""))
        for k in ("gbif_matched", "gbif_english", "gbif_foreign",
                  "gbif_blank_lang"):
            print("  %-24s %10d" % (k, stats[k]))
    print("-" * 52)
    print("wrote %s (%d taxa)" % (p_ascii, len(ascii_rows)))
    print("wrote %s (%d taxa)" % (p_fixit, len(na_taxa)))
    print("wrote %s (%d taxa)" % (p_todo, len(na_taxa)))
    if stats["resolved_unknown"]:
        print("NOTE: %d taxa resolved on an 'unknown' verdict -- spot-check "
              "these in the CSV (evidence=wordlist)" % stats["resolved_unknown"])
    if langs is None:
        print("NOTE: no --gbif; verdicts are wordlist triage only")

    # ---------------- stage 2 ----------------
    if args.skip_api:
        print("\n--skip-api: stopping before stage 2")
        return

    ua = "biovqa_common_names/1.0"
    if args.contact:
        ua += " (%s)" % args.contact

    cache = {} if args.refresh else load_cache(p_cache)
    todo = [sci for sci, _, _ in na_taxa if sci not in cache]
    print("\nstage 2: %d taxa | %d cached | %d to fetch"
          % (len(na_taxa), len(na_taxa) - len(todo), len(todo)))

    with open(p_cache, "a", encoding="utf-8") as cf:
        for i, sci in enumerate(todo, 1):
            rec = api_resolve(sci, ua)
            cache[sci] = rec
            cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cf.flush()
            print("[%d/%d] %s -> %s" % (i, len(todo), sci,
                                        rec["resolved_common_name"] or "(none)"))
            if i < len(todo):
                time.sleep(args.sleep)

    hit = 0
    with open(p_api, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["taxa", "common_name", "inat_common_name",
                    "wiki_common_name", "resolved_common_name", "source"])
        for sci, _, _ in na_taxa:
            r = cache.get(sci, {})
            if r.get("resolved_common_name"):
                hit += 1
            w.writerow([sci, fmt_tuple(forms_by_taxon.get(sci, [])),
                        r.get("inat_common_name", ""),
                        r.get("wiki_common_name", ""),
                        r.get("resolved_common_name", ""),
                        r.get("source", "")])

    print("\nwrote %s" % p_api)
    print("  resolved   %d/%d" % (hit, len(na_taxa)))
    print("  remaining  %d  <- fill these into "
          "resolve_common_name_manual_check.csv" % (len(na_taxa) - hit))


if __name__ == "__main__":
    main()