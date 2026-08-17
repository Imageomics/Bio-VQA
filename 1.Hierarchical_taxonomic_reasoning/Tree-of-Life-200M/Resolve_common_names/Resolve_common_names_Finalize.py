#!/usr/bin/env python3
"""
Merge the three resolution CSVs into one answer key.

Inputs (all optional except the first)
  resolve_common_name_english_ascii.csv   taxa, common_name,
                                          resolved_common_name, evidence
  resolve_common_name_inat_wiki.csv       taxa, common_name, inat_common_name,
                                          wiki_common_name,
                                          resolved_common_name, source
  resolve_common_name_manual_check.csv    taxa, resolved_common_name, source

Precedence, highest first
  1  manual    hand-curated with a cited source
  2  inat_wiki iNaturalist / GBIF / Wikipedia
  3  ascii     stage-1 language triage (source = the evidence code)

Outputs
  final_common_names.csv        taxa, resolved_common_name, source
  excluded_no_english_name.csv  taxa left with no name -- drop these items
  name_conflicts.csv            taxa where the tiers disagree

Conventions
  * A manual row whose resolved_common_name is literally N/A is an explicit
    exclusion: the taxon is dropped and lower tiers are NOT consulted. Use it
    when you have checked and no English name exists.
  * A manual row left blank falls through to the lower tiers. The count of
    those is reported so a half-finished sheet is visible rather than silent.

Conflicts
  A conflict is any taxon where two tiers supply materially different names,
  or where one tier says N/A while another supplies a name. Comparison ignores
  case, spacing and punctuation, so "Four-spot Orbweaver" and "four-spot
  orb-weaver" do NOT count as a conflict -- only real disagreements are shown.
  Higher precedence silently wins, so these are printed to make that visible.

Usage
-----
  python finalize_resolve_common_names.py --dir reports/
  python finalize_resolve_common_names.py --ascii a.csv --api b.csv --manual m.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

NA = {"n/a", "na", "none", "-"}
TIERS = ["manual", "inat_wiki", "ascii"]


def cmpkey(s):
    """Comparison key: case, spacing and punctuation insensitive."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def write_optional(path, header, rows, label):
    """Write path only if there is something to report.

    An empty report is worse than no report: a stale file left over from a
    previous run reads as a current finding. So remove it when it no longer
    applies.
    """
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("wrote %s (%d rows, %s)" % (path, len(rows), label))
        return True
    if os.path.exists(path):
        os.remove(path)
        print("removed stale %s (no %s this run)" % (path, label))
    return False


def read_csv(path, required=("taxa",)):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        cols = {c.strip() for c in (rdr.fieldnames or [])}
        missing = [c for c in required if c not in cols]
        if missing:
            sys.exit("%s is missing column(s): %s" % (path, ", ".join(missing)))
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()}
                for r in rdr]


def index(rows, name_col, source_col):
    """taxa -> (name, source). Later duplicate rows win, with a warning."""
    out, dupes = {}, 0
    for r in rows:
        t = r.get("taxa", "")
        if not t:
            continue
        if t in out:
            dupes += 1
        out[t] = (r.get(name_col, ""), r.get(source_col, ""))
    return out, dupes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="directory holding the three CSVs (default names)")
    ap.add_argument("--ascii", default=None)
    ap.add_argument("--api", default=None)
    ap.add_argument("--manual", default=None)
    ap.add_argument("-o", "--out", default=None,
                    help="output CSV (default: <dir>/final_common_names.csv)")
    ap.add_argument("--show-conflicts", type=int, default=25,
                    help="how many conflicts to print (0 = none; the CSV "
                         "always holds them all)")
    args = ap.parse_args()

    d = args.dir or "."
    
    p_ascii = args.ascii or os.path.join(d, "resolve_common_name_english_ascii.csv")
    p_api = args.api or os.path.join(d, "resolve_common_name_inat_wiki.csv")
    p_manual = args.manual or os.path.join(d, "resolve_common_name_manual.csv")
    p_out = args.out or os.path.join(d, "common_names_resolved.csv")
    outdir = os.path.dirname(p_out) or "."
    p_drop = os.path.join(outdir, "excluded_no_english_name.csv")
    p_conf = os.path.join(outdir, "name_conflicts.csv")

    if not os.path.exists(p_ascii):
        sys.exit("required input not found: %s" % p_ascii)
    for label, p in (("inat_wiki", p_api), ("manual", p_manual)):
        if not os.path.exists(p):
            print("note: no %s input at %s -- skipping that tier" % (label, p))

    ascii_rows = read_csv(p_ascii, ("taxa", "resolved_common_name", "evidence"))
    api_rows = read_csv(p_api, ("taxa", "resolved_common_name", "source"))
    man_rows = read_csv(p_manual, ("taxa", "resolved_common_name", "source"))

    ascii_ix, d1 = index(ascii_rows, "resolved_common_name", "evidence")
    api_ix, d2 = index(api_rows, "resolved_common_name", "source")
    man_ix, d3 = index(man_rows, "resolved_common_name", "source")
    by_tier = {"manual": man_ix, "inat_wiki": api_ix, "ascii": ascii_ix}

    for label, n in (("ascii", d1), ("inat_wiki", d2), ("manual", d3)):
        if n:
            print("warning: %d duplicate taxa row(s) in %s; last one wins"
                  % (n, label))

    all_taxa = sorted(set(ascii_ix) | set(api_ix) | set(man_ix))
    orphans = sorted(set(man_ix) - set(ascii_ix) - set(api_ix))

    stats = Counter()
    out_rows, dropped, blank_manual = [], [], []
    chosen = {}

    for t in all_taxa:
        name, source, tier = "", "", ""

        m_name, m_src = man_ix.get(t, ("", ""))
        if t in man_ix:
            if m_name.lower() in NA:
                stats["manual_explicit_drop"] += 1
                chosen[t] = "manual (N/A)"
                dropped.append([t, "manual N/A: %s" % (m_src or "no source")])
                continue
            if m_name:
                name, source, tier = m_name, m_src, "manual"
            else:
                blank_manual.append(t)

        if not name:
            a_name, a_src = api_ix.get(t, ("", ""))
            if a_name:
                name, source, tier = a_name, a_src, "inat_wiki"

        if not name:
            s_name, s_ev = ascii_ix.get(t, ("", ""))
            if s_name:
                name, source, tier = s_name, s_ev, "ascii"

        if name:
            stats[tier] += 1
            chosen[t] = tier
            out_rows.append([t, name, source])
        else:
            stats["unresolved"] += 1
            dropped.append([t, "no name from any tier"])

    # ---------------- conflicts ----------------
    conflicts = []
    for t in all_taxa:
        present = []
        for tier in TIERS:
            n, src = by_tier[tier].get(t, ("", ""))
            if n:
                present.append((tier, n, src))
        if len(present) < 2:
            continue
        named = [(tr, n, s) for tr, n, s in present if n.lower() not in NA]
        na_tiers = [tr for tr, n, _ in present if n.lower() in NA]
        keys = {cmpkey(n) for _, n, _ in named}
        if len(keys) > 1 or (na_tiers and named):
            kind = "excluded_vs_named" if (na_tiers and named) else "different_name"
            conflicts.append((t, kind, present))

    with open(p_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["taxa", "resolved_common_name", "source"])
        w.writerows(out_rows)

    conf_rows = [[t, kind, chosen.get(t, ""), tier, n, src]
                 for t, kind, present in conflicts
                 for tier, n, src in present]

    wrote_drop = write_optional(p_drop, ["taxa", "reason"], dropped,
                                "excluded taxa")
    wrote_conf = write_optional(
        p_conf, ["taxa", "conflict_kind", "winning_tier", "tier", "name",
                 "source"], conf_rows, "conflicts")

    total = len(all_taxa)
    print("-" * 60)
    print("  %-26s %10d" % ("taxa total", total))
    for k in TIERS:
        print("  %-26s %10d" % ("from " + k, stats[k]))
    print("  %-26s %10d" % ("excluded (manual N/A)", stats["manual_explicit_drop"]))
    print("  %-26s %10d" % ("excluded (no name)", stats["unresolved"]))
    print("  %-26s %10d" % ("conflicts between tiers", len(conflicts)))
    print("-" * 60)
    kept = len(out_rows)
    print("  coverage %d/%d (%.1f%%)" % (kept, total, 100.0 * kept / max(total, 1)))

    if conflicts and args.show_conflicts:
        shown = conflicts[:args.show_conflicts]
        print("\nconflicts (%d shown of %d; full list in %s)"
              % (len(shown), len(conflicts), os.path.basename(p_conf)))
        for t, kind, present in shown:
            print("  %s   [%s]" % (t, kind))
            for tier, n, src in present:
                mark = "<- used" if tier == chosen.get(t, "").split(" ")[0] else ""
                print("      %-10s %-38s %s" % (tier, n, mark))
        if len(conflicts) > len(shown):
            print("  ... %d more in %s" % (len(conflicts) - len(shown), p_conf))

    missing_src = sum(1 for r in out_rows if not r[2])
    if missing_src:
        print("\nwarning: %d row(s) have an empty source column" % missing_src)
    if blank_manual:
        print("warning: %d manual row(s) had a blank resolved_common_name and "
              "fell through to a lower tier; put N/A to force exclusion"
              % len(blank_manual))
        for t in blank_manual[:10]:
            print("           %s" % t)
    if orphans:
        print("warning: %d manual taxa not present in stage 1 (typo?)"
              % len(orphans))
        for t in orphans[:10]:
            print("           %s" % t)

    print("\nwrote %s (%d rows)" % (p_out, kept))
    if not wrote_drop:
        print("no exclusions -- %s not written"
              % os.path.basename(p_drop))
    if not wrote_conf:
        print("no conflicts  -- %s not written"
              % os.path.basename(p_conf))


if __name__ == "__main__":
    main()