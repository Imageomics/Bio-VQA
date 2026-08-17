#!/usr/bin/env python3
"""
prefetch_bioclip2.py
====================
Download the BioCLIP-2 weights + TreeOfLife-200M embeddings ONCE, into HF_HOME,
and verify the cache is intact. Run this BEFORE submitting the array job.

Why this exists
---------------
A 10-task SLURM array all starts at the same instant. Every task calls
TreeOfLifeClassifier(), every task finds an empty HF_HOME, and all ten begin
downloading open_clip_config.json to the same path on Lustre. huggingface_hub
guards this with file locks, but advisory locks are unreliable on parallel
filesystems, so the writes interleave and you get a config containing the same
JSON object twice:

    json.decoder.JSONDecodeError: Extra data: line 32 column 2 (char 534)

Downloading once, serially, removes the race entirely. The array job then runs
with HF_HUB_OFFLINE=1 so no task can ever re-download.

Usage
-----
    export HF_HOME=/fs/scratch/PAS2136/rayees/hf_cache
    python prefetch_bioclip2.py            # verify, download if needed
    python prefetch_bioclip2.py --clean    # wipe a corrupted cache first
    python prefetch_bioclip2.py --check    # verify only, download nothing
"""

import os
import sys
import json
import glob
import shutil
import argparse


def bad_json(hf_home):
    """Every .json under the cache that will not parse."""
    out = []
    for p in glob.glob(os.path.join(hf_home, "**", "*.json"), recursive=True):
        if os.path.islink(p) and not os.path.exists(p):
            out.append((p, "broken symlink"))
            continue
        try:
            with open(p) as f:
                json.load(f)
        except Exception as e:
            out.append((p, str(e)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="delete HF_HOME before downloading")
    ap.add_argument("--check", action="store_true",
                    help="only verify; do not download")
    args = ap.parse_args()

    hf = os.environ.get("HF_HOME")
    if not hf:
        sys.exit("ERROR: HF_HOME is not set.\n"
                 "  export HF_HOME=/fs/scratch/PAS2136/$USER/hf_cache")
    print(f"[cache] HF_HOME = {hf}")

    if args.clean and os.path.isdir(hf):
        print(f"[cache] removing {hf}")
        shutil.rmtree(hf)

    os.makedirs(hf, exist_ok=True)

    # ---- verify whatever is already there ------------------------------- #
    bad = bad_json(hf)
    if bad:
        print(f"\n[cache] {len(bad)} CORRUPT file(s):")
        for p, e in bad[:10]:
            print(f"   {p}\n      {e}")
        print("\n[cache] this is the concurrent-download race. Re-run with --clean:")
        print("   python prefetch_bioclip2.py --clean")
        sys.exit(1)
    print(f"[cache] json files present and parseable: "
          f"{len(glob.glob(os.path.join(hf, '**', '*.json'), recursive=True))}")

    if args.check:
        print("[cache] --check only; done.")
        return

    # ---- download, serially, in this one process ------------------------ #
    print("\n[fetch] loading BioCLIP-2 + TreeOfLife-200M embeddings ...")
    print("        (first run pulls ~2-3 GB; later runs are instant)")
    import torch
    from bioclip.predict import TreeOfLifeClassifier

    clf = TreeOfLifeClassifier(device="cpu")     # cpu: this is a download, not compute
    ld = clf.get_label_data()
    print(f"[fetch] ToL label data: {len(ld):,} taxa")
    print(f"[fetch] columns: {list(ld.columns)}")
    n_species = ld["species"].nunique() if "species" in ld.columns else -1
    print(f"[fetch] distinct species: {n_species:,}")

    # ---- verify again, now that everything is on disk ------------------- #
    bad = bad_json(hf)
    if bad:
        print(f"\n[cache] STILL corrupt after download ({len(bad)} files):")
        for p, e in bad[:10]:
            print(f"   {p}\n      {e}")
        sys.exit(1)

    print("\n[ok] cache is warm and valid.")
    print("     Now submit the array; it runs with HF_HUB_OFFLINE=1 so no task")
    print("     will try to download anything:")
    print("        sbatch run_bioclip2.sh")


if __name__ == "__main__":
    main()