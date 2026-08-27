#!/usr/bin/env python3
"""
Audit Bio-VQA Markdown footers.

Classifies every .md/.MD file under ROOT into three buckets:

  OK      - has the anonymized footer: "Part of the <b>Bio-VQA</b> benchmark"
  OTHER   - has a <sub> footer block, but not that text (inspect these)
  NONE    - has no <sub> footer block at all

Run after anonymize_footers.py to confirm nothing was missed and to see which
files never had a footer in the first place.

Usage:
    python audit_footers.py
    python audit_footers.py --root /path/to/repo
"""

import argparse
import sys
from pathlib import Path
import re

ROOT = Path("/users/PAS2136/rayees/Bio-VQA_github/Bio-VQA")

TARGET = "Part of the <b>Bio-VQA</b> benchmark"
SUB_BLOCK = re.compile(r"<sub>.*?</sub>", re.DOTALL | re.IGNORECASE)


def classify(path: Path):
    """Return (bucket, list_of_sub_blocks)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "UNREADABLE", []

    blocks = SUB_BLOCK.findall(text)
    if not blocks:
        return "NONE", []
    if any(TARGET in b for b in blocks):
        return "OK", blocks
    return "OTHER", blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Root not found: {root}")

    md_files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".md"
        and "__MACOSX" not in p.parts
        and ".git" not in p.parts
    )

    buckets = {"OK": [], "OTHER": [], "NONE": [], "UNREADABLE": []}
    for path in md_files:
        bucket, blocks = classify(path)
        buckets[bucket].append((path.relative_to(root), blocks))

    print(f"Scanning {root}")
    print(f"Markdown files found: {len(md_files)}\n")

    print(f"=== OK — anonymized footer present ({len(buckets['OK'])}) ===")
    for rel, _ in buckets["OK"]:
        print(f"  {rel}")

    print(f"\n=== OTHER — has a <sub> block, but not the target footer "
          f"({len(buckets['OTHER'])}) ===")
    if not buckets["OTHER"]:
        print("  (none)")
    for rel, blocks in buckets["OTHER"]:
        print(f"  {rel}")
        for b in blocks:
            print(f"      {' '.join(b.split())}")

    print(f"\n=== NONE — no footer at all ({len(buckets['NONE'])}) ===")
    for rel, _ in buckets["NONE"]:
        print(f"  {rel}")

    if buckets["UNREADABLE"]:
        print(f"\n=== UNREADABLE ({len(buckets['UNREADABLE'])}) ===")
        for rel, _ in buckets["UNREADABLE"]:
            print(f"  {rel}")

    print(f"\nSummary: OK={len(buckets['OK'])}  "
          f"OTHER={len(buckets['OTHER'])}  "
          f"NONE={len(buckets['NONE'])}")


if __name__ == "__main__":
    main()