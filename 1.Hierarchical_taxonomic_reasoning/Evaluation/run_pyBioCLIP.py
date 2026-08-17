#!/usr/bin/env python3
"""
pyBioCLIP.py
============
BioCLIP-2 specialist baseline for BioVQA C1, via pybioclip.

    https://github.com/Imageomics/pybioclip
    https://imageomics.github.io/pybioclip/python-tutorial/

pybioclip's default model IS BioCLIP 2, paired with its TreeOfLife-200M text
embeddings - the same catalog your ground truth comes from. It owns the prompt
format, tokenizer and batching, so none of that is reimplemented here.

Protocol
--------
BioCLIP-2 is a CLIP model, not a VQA model, so it cannot "answer questions":

  1. TreeOfLifeClassifier.predict(paths, Rank.SPECIES, k=1) gives ONE species
     per image.
  2. Its returned dict already carries the full lineage:
        kingdom, phylum, class, order, family, genus,
        species_epithet, species (binomial), common_name, score
     so all 8 open answers are DERIVED from that single prediction.
  3. MCQ rows use CustomLabelsClassifier over the union of that group's option
     strings; each item then argmaxes over its OWN four options. Softmax is
     monotonic, so restricting to a subset preserves the ranking.

--label_space : the three baselines, and why all three matter
-------------------------------------------------------------
  kingdom  (DEFAULT)  Restrict to the kingdom the QUESTION states.
                      Your prompts say "the PLANT shown in this image" or
                      "the ANIMAL shown in this image". The VLMs are handed that
                      word for free; BioCLIP-2 sees only pixels. Without this
                      filter the comparison is rigged AGAINST BioCLIP-2, and you
                      get failures like an image of Pinus strobus classified as
                      Marmara fasciella - the "White pine barkminer moth", an
                      insect named after the tree it eats. This is the FAIR
                      head-to-head: both sides now know it is a plant.

  tol                 The entire TreeOfLife-200M, no hint at all. Harder than
                      what the VLMs faced. Useful as an ablation: the
                      kingdom-vs-tol gap measures how much of the specialist's
                      advantage is just being told the kingdom.

  group               Restricted to that group's GT species. A CEILING - a
                      closed candidate set the VLMs never had. Upper bound only,
                      never a head-to-head.

Each writes to its OWN model directory, so run all three and Evaluate.py reports
them as three models side by side:

    <Group>/BioCLIP2_kingdom/... -> column bioclip2_kingdom_answer
    <Group>/BioCLIP2/...         -> column bioclip2_answer
    <Group>/BioCLIP2_group/...   -> column bioclip2_group_answer

Chunks are written as they finish and skipped on re-run -> resumable.

Usage
-----
    export BIOVQA_BASE_DIR=/scratch/.../Bio-VQA/3-Hierarchical-Taxonomic-Reasoning

    python pyBioCLIP.py --group Amphibians-10K                    # kingdom (fair)
    python pyBioCLIP.py --all
    python pyBioCLIP.py --all --label_space tol                   # no hint
    python pyBioCLIP.py --all --label_space group                 # ceiling

    # or without the env var:
    python pyBioCLIP.py --base_dir /path/to/3-Hierarchical-Taxonomic-Reasoning --all

Requires: pip install pybioclip
"""

import os
import re
import json
import inspect
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# No hardcoded BASE_DIR. It comes from --base_dir or $BIOVQA_BASE_DIR.
ENV_BASE_DIR = "BIOVQA_BASE_DIR"

# label_space -> (column prefix, model directory)
TAGS = {
    "kingdom": ("bioclip2_kingdom", "BioCLIP2/BioCLIP2_kingdom"),
    "tol":     ("bioclip2",         "BioCLIP2/BioCLIP2_tol"),
    "group":   ("bioclip2_group",   "BioCLIP2/BioCLIP2_group"),
}
CHUNK_IMAGES = 1000          # also bounds the MCQ score dict

# "... ? Options: A, B, C, D"  ->  the "Options:" lead-in is NOT an option.
_OPT_LEADIN = re.compile(r"^\s*(?:options?|choices?)\s*(?:are)?\s*[:\-\u2013]?\s*",
                         re.IGNORECASE)
# "(A) ", "[2] ", "C) ", "3. "  ->  enumerator, not part of the label.
# Deliberately does NOT strip "B. bufo" (letter + dot), which is a real name.
_ENUM = re.compile(r"^\s*(?:\(\s*[A-Za-z0-9]\s*\)|\[\s*[A-Za-z0-9]\s*\]"
                   r"|[A-Za-z][\)\]]|\d+[\).:])\s+")


def detect(question):
    """-> (asked_rank, is_mcq).  Mirrors Evaluate.py exactly."""
    q = question.lower()
    mcq = "options" in q
    for rank, pat in [("scientific_name", "scientific name"),
                      ("common_name", "common name"),
                      ("species", "species epithet"), ("species", "species"),
                      ("phylum", "phylum"), ("class", "class"),
                      ("order", "order"), ("family", "family"),
                      ("genus", "genus")]:
        if pat in q:
            return rank, mcq
    return "common_name", mcq


def mcq_options(question, raw=False):
    """
    Option strings from the tail of an MCQ question.

    `raw=True` reproduces the old behaviour (naive comma split), which leaked
    the "Options:" lead-in into the FIRST option of every question - so option A
    was scored as a different string than the GT and could never be marked
    correct. Keep raw=True only if Evaluate.py expects that exact form.
    """
    tail = question.rsplit("?", 1)[-1]
    if raw:
        return [o.strip() for o in tail.split(",") if o.strip()]
    tail = _OPT_LEADIN.sub("", tail, count=1)
    out = []
    for o in tail.split(","):
        o = _ENUM.sub("", o).strip().strip(".;")
        if o:
            out.append(o)
    return out


def canonical_csv(gdir):
    for f in sorted(os.listdir(gdir)):
        p = os.path.join(gdir, f)
        if f.endswith("-QA_difficulty.csv") and os.path.isfile(p):
            return p
    return None


def gt_binomials(df):
    """Each image's GT binomial, rebuilt from that image's own answers."""
    gt = defaultdict(dict)
    for uid, v in zip(df["uuid"], df["qa_pairs"]):
        d = json.loads(v) if isinstance(v, str) else v
        r, m = detect(d["Q"])
        if not m:
            gt[uid][r] = d["A"]
    out = {}
    for uid, g in gt.items():
        sci = g.get("scientific_name")
        if not sci and "genus" in g and "species" in g:
            sci = f"{g['genus']} {g['species']}"
        if sci:
            out[uid] = str(sci).strip()
    return out


def question_kingdoms(df):
    """
    The kingdom the QUESTION hands the model for free.
    Your prompts read "the plant shown in this image" / "the animal shown in
    this image". The VLMs get that word; BioCLIP-2 must be told too, or the
    comparison is not like-for-like.
    """
    found = set()
    for v in df["qa_pairs"]:
        q = (json.loads(v) if isinstance(v, str) else v)["Q"].lower()
        if re.search(r"\bplants?\b", q):
            found.add("Plantae")
        if re.search(r"\banimals?\b", q):
            found.add("Animalia")
        if re.search(r"\bfung(us|i)\b", q):
            found.add("Fungi")
    return found


# pybioclip's prediction dict -> the answer for each question type.
PRED_KEY = {
    "phylum": "phylum", "class": "class", "order": "order",
    "family": "family", "genus": "genus",
    "species": "species_epithet",      # bare epithet
    "scientific_name": "species",      # binomial
    "common_name": "common_name",
}
CATEGORY = {
    "phylum": "phylum", "class": "class", "order": "order",
    "family": "family", "genus": "genus", "species": "species",
    "scientific_name": "scientific name", "common_name": "common_name",
}


def supported_kwargs(fn, **want):
    """
    Keep only the kwargs THIS installed pybioclip accepts. `callback` exists in
    recent releases but not older ones; passing it to an old version raises
    TypeError, which previously got swallowed by the per-image fallback and
    silently disabled batching (~64x slower while looking like it worked).
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in want.items() if k in params}


def safe_predict(paths, tag, call_batch, call_one):
    """
    Batched predict, with a per-image fallback for UNREADABLE IMAGES ONLY.
    TypeError means the call signature is wrong - a bug here, not a bad JPEG -
    so re-raise it rather than degrade to 1-image batches.
    """
    if not paths:
        return []
    try:
        return call_batch(paths)
    except TypeError:
        raise
    except Exception as e:
        print(f"      [warn] {tag} batch failed ({e}); isolating bad images")
        out = []
        for p in paths:
            try:
                out.extend(call_one(p))
            except Exception:
                print(f"      [warn] unreadable: {p}")
        return out


# --------------------------------------------------------------------------- #
def run_group(group, args, device):
    from bioclip import Rank
    from bioclip.predict import TreeOfLifeClassifier, CustomLabelsClassifier

    tag, dirname = TAGS[args.label_space]
    gdir = os.path.join(args.base_dir, group)
    canon = canonical_csv(gdir)
    if canon is None:
        print(f"[skip] {group}: no canonical CSV")
        return
    outdir = os.path.join(gdir, dirname)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(canon).replace(".csv", "")
    merged = os.path.join(outdir, f"{base}_{tag}.csv")

    df = pd.read_csv(canon)
    print(f"[{group}] {len(df):,} rows, {df.uuid.nunique():,} images")

    tol = TreeOfLifeClassifier(device=device)     # default model IS BioCLIP 2
    _p = inspect.signature(tol.predict).parameters
    print(f"[{group}] predict() accepts: {', '.join(_p)}")
    if "batch_size" not in _p:
        print(f"[{group}] [warn] this pybioclip has no batch_size; will be slow")

    ld = tol.get_label_data()

    # ---- label space ---------------------------------------------------- #
    if args.label_space == "kingdom":
        want = question_kingdoms(df)
        have = set(ld["kingdom"].dropna().unique())
        keep = sorted(want & have)
        if keep:
            tol.apply_filter(tol.create_taxa_filter(Rank.KINGDOM, keep))
            n = int(ld["kingdom"].isin(keep).sum())
            print(f"[{group}] KINGDOM HINT from the question -> {keep}  "
                  f"({n:,} of {len(ld):,} ToL taxa kept)")
            print(f"[{group}]   the VLM prompt says this word too; this is the "
                  f"like-for-like comparison")
        else:
            print(f"[{group}] [warn] no kingdom word found in the questions "
                  f"(wanted {want or 'none'}, ToL has {sorted(have)[:5]}...); "
                  f"staying open-set")
    elif args.label_space == "group":
        gtb = gt_binomials(df)
        known = set(ld["species"])
        keep = sorted({s for s in gtb.values() if s in known})
        miss = len(set(gtb.values())) - len(keep)
        if keep:
            tol.apply_filter(tol.create_taxa_filter(Rank.SPECIES, keep))
            print(f"[{group}] FILTERED to {len(keep):,} GT species "
                  f"({miss} not in ToL) -- this is a CEILING, not a fair fight")
        else:
            print(f"[{group}] no GT species in ToL; staying open-set")
    else:
        print(f"[{group}] label space = full TreeOfLife-200M, NO hint "
              f"({len(ld):,} taxa). Harder than what the VLMs faced.")

    # ---- MCQ classifier ------------------------------------------------- #
    mcq_clf = None
    if not args.skip_mcq:
        u = set()
        sample = None
        for v in df["qa_pairs"]:
            d = json.loads(v) if isinstance(v, str) else v
            if detect(d["Q"])[1]:
                opts = mcq_options(d["Q"], raw=args.raw_mcq_options)
                if sample is None:
                    sample = opts
                u.update(opts)
        if u:
            print(f"[{group}] MCQ: {len(u):,} distinct option strings")
            print(f"[{group}]   parsed sample -> {sample}")
            print(f"[{group}]   (if these do not look like clean labels, "
                  f"check mcq_options() against your question format)")
            mcq_clf = CustomLabelsClassifier(sorted(u), device=device)

    # ---- images --------------------------------------------------------- #
    uimgs = (df[["uuid", "local_path"]].drop_duplicates("uuid")
             .to_records(index=False).tolist())
    rows_by_uuid = defaultdict(list)
    for i, u in enumerate(df["uuid"]):
        rows_by_uuid[u].append(i)

    chunks = [uimgs[i:i + CHUNK_IMAGES]
              for i in range(0, len(uimgs), CHUNK_IMAGES)]
    parts, phyla = [], Counter()
    for ci, ch in enumerate(chunks):
        cpath = os.path.join(outdir, f"{base}_chunk{ci}_{tag}.csv")
        if os.path.exists(cpath) and not args.overwrite:
            print(f"   chunk{ci}: exists, skip")
            parts.append(cpath)
            continue
        print(f"   chunk{ci}: {len(ch):,} images")

        paths, uids = [], []
        for uid, rel in ch:
            p = rel if os.path.isabs(rel) else os.path.join(gdir, rel)
            if os.path.exists(p):
                paths.append(p)
                uids.append(uid)
            else:
                print(f"      [warn] missing: {p}")
        path2uuid = dict(zip(paths, uids))

        def _cb(done, total):
            if done % (args.batch * 5) < args.batch:
                print(f"      tol {done:,}/{total:,}", flush=True)

        tol_kw = supported_kwargs(tol.predict, batch_size=args.batch, callback=_cb)
        preds = safe_predict(
            paths, "tol",
            lambda ps: tol.predict(ps, Rank.SPECIES, k=1, **tol_kw),
            lambda p: tol.predict([p], Rank.SPECIES, k=1))
        by_uuid = {}
        for p in preds:
            u = path2uuid.get(p.get("file_name"))
            if u is not None and u not in by_uuid:      # k=1, keep the top hit
                by_uuid[u] = p
        for p in by_uuid.values():
            phyla[p.get("phylum", "?")] += 1

        mscore = defaultdict(dict)
        if mcq_clf is not None:
            good = [p for p in paths if path2uuid.get(p) in by_uuid]
            mcq_kw = supported_kwargs(mcq_clf.predict, batch_size=args.batch)
            mp = safe_predict(
                good, "mcq",
                lambda ps: mcq_clf.predict(ps, k=None, **mcq_kw),
                lambda p: mcq_clf.predict([p], k=None))
            for r in mp:
                u = path2uuid.get(r.get("file_name"))
                if u is not None:
                    mscore[u][r["classification"]] = r["score"]

        out = []
        for uid, _rel in ch:
            rec = by_uuid.get(uid)
            for ri in rows_by_uuid[uid]:
                r = df.iloc[ri]
                d = (json.loads(r["qa_pairs"])
                     if isinstance(r["qa_pairs"], str) else r["qa_pairs"])
                asked, is_mcq = detect(d["Q"])
                if is_mcq:
                    sc = mscore.get(uid, {})
                    opts = [o for o in mcq_options(d["Q"],
                                                   raw=args.raw_mcq_options)
                            if o in sc]
                    ans = max(opts, key=lambda o: sc[o]) if opts else ""
                    cat = "common_name_MCQ"
                elif rec is None:
                    ans, cat = "", CATEGORY[asked]
                else:
                    ans = rec.get(PRED_KEY[asked], "") or ""
                    cat = CATEGORY[asked]
                row = r.to_dict()
                row[f"{tag}_answer"] = ans
                row["category"] = cat
                row[f"{tag}_score"] = (rec or {}).get("score", np.nan)
                out.append(row)
        pd.DataFrame(out).to_csv(cpath, index=False)
        parts.append(cpath)

    if phyla:
        tot = sum(phyla.values())
        print(f"[{group}] predicted phyla (freshly computed chunks only): "
              + "  ".join(f"{k}={v/tot:.1%}" for k, v in phyla.most_common(4)))
        print(f"[{group}]   (if this does not match the group, the model is "
              f"searching the wrong kingdom)")

    full = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    full.to_csv(merged, index=False)
    print(f"[{group}] -> {merged}  ({len(full):,} rows)")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--base_dir", default=os.environ.get(ENV_BASE_DIR),
                    help=f"Root directory holding the group subdirectories "
                         f"(e.g. .../3-Hierarchical-Taxonomic-Reasoning). "
                         f"Falls back to ${ENV_BASE_DIR}.")
    ap.add_argument("--group_suffix", default="-10K",
                    help="Subdirectories with this suffix are treated as groups.")
    ap.add_argument("--group", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--label_space", choices=["kingdom", "tol", "group"],
                    default="kingdom",
                    help="kingdom = restrict to the kingdom the question states "
                         "(FAIR: the VLM prompt says 'plant'/'animal' too); "
                         "tol = whole TreeOfLife-200M, no hint; "
                         "group = that group's GT species, a CEILING")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / cuda:0. Default: auto-detect.")
    ap.add_argument("--skip_mcq", action="store_true")
    ap.add_argument("--raw_mcq_options", action="store_true",
                    help="Old naive comma split, which leaves the 'Options:' "
                         "lead-in glued to the first option. Use only if "
                         "Evaluate.py expects that exact string form.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.base_dir:
        ap.error(f"--base_dir is required (or set ${ENV_BASE_DIR})")
    args.base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    if not os.path.isdir(args.base_dir):
        raise SystemExit(f"ERROR: --base_dir does not exist: {args.base_dir}")

    if not args.group and not args.all:
        ap.error("pass --group <NAME> for one group, or --all for every group")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag, dirname = TAGS[args.label_space]
    print(f"[init] base_dir={args.base_dir}")
    print(f"[init] device={device}  label_space={args.label_space}")
    print(f"[init] writing -> <Group>/{dirname}/  column {tag}_answer")
    print(f"[init] pybioclip default model = BioCLIP 2 + TreeOfLife-200M")

    avail = sorted(d for d in os.listdir(args.base_dir)
                   if d.endswith(args.group_suffix)
                   and os.path.isdir(os.path.join(args.base_dir, d)))
    if not avail:
        raise SystemExit(f"ERROR: no *{args.group_suffix} directories under "
                         f"{args.base_dir}")

    if args.group and not args.all:
        if args.group not in avail:
            raise SystemExit(
                f"ERROR: --group {args.group!r} is not a group directory.\n"
                f"  Looked in : {args.base_dir}\n"
                f"  Available : {', '.join(avail)}\n"
                f"  If this looks like a number (e.g. a unix GID), the launching\n"
                f"  shell script set the group from the bash special variable\n"
                f"  GROUPS. Rename it, and remember sbatch snapshots the script\n"
                f"  at submit time -- scancel and resubmit.")
        groups = [args.group]
    else:
        groups = avail

    for g in groups:
        run_group(g, args, device)

    print("\n[done] now re-run Evaluate.py; it picks this up automatically.")


if __name__ == "__main__":
    main()