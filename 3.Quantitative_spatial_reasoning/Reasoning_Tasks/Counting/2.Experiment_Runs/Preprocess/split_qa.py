#!/usr/bin/env python3
"""
Combine the three Counting tiers into one manifest and shard it for SLURM.

  python split_qa.py \\
      --task_root     /path/to/1-Spatial-Visual-Reasoning-tasks/Counting \\
      --datasets_root /path/to/1-Spatial-Visual-Reasoning-datasets \\
      --run_dir       .            # default: this script's own directory

Nothing is hardcoded. --run_dir defaults to the directory containing this
file; the two data roots must be supplied.

WHY ONE COMBINED MANIFEST
The model is loaded once per array task, so folding the tiers together turns
three model loads into one. Rows are interleaved, so every chunk holds the same
tier mix and one failed task cannot wipe out a whole tier.

TIER NAMING
The directories are Easy / Medium / Hard, but those labels describe the wrong
thing. All three ask for a count; what changes is what the model must do to
find the target:

  Single      one photograph, no panel to locate
  Cued        two photographs of DIFFERENT species. The question names the
              species, so the species identifies the panel and the word
              "left" is never load-bearing.
  Positional  two photographs of the SAME species. The species is
              uninformative, so the panel must be located from "left"/"right".

Cued and Positional are both two-panel and go through the same resize, so
Cued -> Positional isolates positional indexing with resolution held constant.
That contrast is the point of the tier ladder, and it is why Cued is not
"regenerated to remove the shortcut" - the shortcut is the control condition.

COLUMNS ADDED
  row_id         restores original order at merge time
  tier           single | cued | positional
  n_panels       1 or 2 - drives the panel clause in the prompt, nothing else
  panel          left | right | single, parsed from the question
  animal_term    bees, beetles, animals, ... parsed from the question
  source_dataset recovered per tier (see recover_dataset)
  image_abspath  resolved here, once. Chunk CSVs live in chunks/chunkNN/ while
                 images stay under the task or datasets root, so a relative
                 path would be unresolvable from every chunk.
  question       panel wording normalised to "photograph" (see below)
  question_original   never modified

WHY "photograph"
"How many bees are there in the left image?" is ambiguous when the whole canvas
is also an image: a model can read it as the left REGION of one scene. Saying
"the left photograph", together with the panel clause in the prompt, removes
that reading. The original string is kept in question_original.
"""

import argparse
import os
import re
import shutil

import pandas as pd

TIERS = [("single", "Easy", 1), ("cued", "Medium", 2),
         ("positional", "Hard", 2)]

# animal term -> source dataset. "zebras" is ambiguous and resolved per panel.
TERM2DS = {
    "bees": "Bees", "beetles": "Beetles-500", "fish": "Fish",
    "camels": "Camel-500", "cattle": "Cattle-500", "sheep": "Sheep-500",
    "seals": "Seal-500", "kiangs": "Kiang-500",
    "animals": "GrevysZebra", "grevys zebras": "GrevysZebra",
    "zebras": None,
}
GZ_ONLY = {"animals", "grevys zebras"}   # emitted only by GrevysZebra

Q_RE = re.compile(
    r"how many (.+?) are there in (?:the (left|right)|this) (?:image|photograph)\?",
    re.I)


def parse_question(q):
    """-> (animal_term, panel). panel is left | right | single."""
    m = Q_RE.match(str(q).strip())
    if not m:
        return None, None
    term = re.sub(r"\s*\(wild asses\)", "", m.group(1).strip().lower())
    where = m.group(2)
    return term, (where.lower() if where else "single")


def normalise_question(q):
    """Panel reference -> 'photograph'; drop the kiang gloss, which is the only
    explanatory parenthetical in the whole set and therefore the only question
    carrying extra semantic support."""
    s = str(q).strip()
    s = s.replace(" in the left image?", " in the left photograph?")
    s = s.replace(" in the right image?", " in the right photograph?")
    return re.sub(r"\s*\(wild asses\)", "", s)


def recover_dataset(df, tier):
    if tier == "single":
        # image_path looks like "Bees/Images/xxx.jpg"
        return df["image_path"].astype(str).str.strip().str.lstrip("/") \
                 .str.split("/").str[0]

    if tier == "positional":
        # Hard filenames embed it: concat_{dataset}_...  Both panels match.
        known = sorted(set(v for v in TERM2DS.values() if v) | {"Zebra-500"},
                       key=len, reverse=True)

        def from_name(n):
            body = str(n)
            if body.startswith("concat_"):
                body = body[len("concat_"):]
            for ds in known:
                if body.startswith(ds + "_"):
                    return ds
            return None

        out = df["image"].map(from_name)
        return out.fillna(df["animal_term"].map(TERM2DS))

    # Cued: filename carries no dataset, so map from the term. "zebras" is
    # ambiguous between Zebra-500 and GrevysZebra; a GrevysZebra panel always
    # also emits "animals" and "grevys zebras", so if neither appears on that
    # panel then "zebras" must be Zebra-500.
    ds = df["animal_term"].map(TERM2DS)
    gz_panels = set(map(tuple, df.loc[df["animal_term"].isin(GZ_ONLY),
                                      ["image", "panel"]].values))
    amb = ds.isna()
    if amb.any():
        ds.loc[amb] = ["GrevysZebra" if (i, p) in gz_panels else "Zebra-500"
                       for i, p in df.loc[amb, ["image", "panel"]].values]
    return ds


def load_tier(task_root, datasets_root, tier, folder, n_panels):
    d = os.path.join(task_root, folder)
    csv_path = os.path.join(d, "qa.csv")
    if not os.path.isfile(csv_path):
        print("[skip] %s: no qa.csv at %s" % (tier, csv_path))
        return None

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].astype(str)

    # Single tier paths ("Bees/Images/x.jpg") resolve against the DATASETS
    # root; the two-panel tiers ("Images/concat_x.jpg") against the task dir.
    base = datasets_root if n_panels == 1 else d

    def abspath(rel):
        p = str(rel).strip().lstrip("/")
        for cand in (os.path.join(base, p),
                     os.path.join(d, "Images", os.path.basename(p)),
                     os.path.join(datasets_root, p)):
            if os.path.exists(cand):
                return cand
        return os.path.join(base, p)

    df["image_abspath"] = df["image_path"].map(abspath)

    parsed = df["question"].map(parse_question)
    df["animal_term"] = [p[0] for p in parsed]
    df["panel"] = [p[1] for p in parsed]
    unparsed = int(df["animal_term"].isna().sum())
    if unparsed:
        print("  !! %d questions did not match the expected pattern" % unparsed)
        print("     e.g. %r" % df.loc[df["animal_term"].isna(),
                                      "question"].iloc[0])

    df["source_dataset"] = recover_dataset(df, tier)
    df["question_original"] = df["question"].astype(str)
    df["question"] = df["question_original"].map(normalise_question)
    df["answer"] = pd.to_numeric(df["answer"], errors="coerce").astype("Int64")
    df["tier"] = tier
    df["n_panels"] = n_panels

    missing = int((~df["image_abspath"].map(os.path.exists)).sum())
    print("[ok]   %-11s %6d rows  %5d images  missing=%d  bad_answer=%d"
          % (tier, len(df), df["image"].nunique(), missing,
             int(df["answer"].isna().sum())))
    if missing:
        print("       first missing: %s"
              % df.loc[~df["image_abspath"].map(os.path.exists),
                       "image_abspath"].iloc[0])
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", required=True,
                    help="directory holding Easy/ Medium/ Hard/")
    ap.add_argument("--datasets_root", required=True,
                    help="directory holding Bees/ Beetles-500/ ...")
    ap.add_argument("--run_dir", default=here)
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    chunk_root = os.path.join(args.run_dir, "chunks")
    if os.path.isdir(chunk_root):
        if not args.force:
            raise SystemExit(
                "chunks/ already exists at %s\n"
                "  Re-splitting would orphan results already written there.\n"
                "  Use --force to wipe and rebuild." % chunk_root)
        shutil.rmtree(chunk_root)

    frames = [f for f in (load_tier(args.task_root, args.datasets_root,
                                    t, folder, n)
                          for t, folder, n in TIERS) if f is not None]
    if not frames:
        raise SystemExit("no tiers found under " + args.task_root)

    keep = ["tier", "n_panels", "panel", "source_dataset", "animal_term",
            "image", "image_abspath", "question", "question_original", "answer"]
    df = pd.concat([f[keep] for f in frames], ignore_index=True)
    df.insert(0, "row_id", range(len(df)))

    print()
    print("total %d rows" % len(df))
    print(df.groupby("tier").size().to_string())
    print()
    print("rows per source dataset:")
    print(pd.crosstab(df["source_dataset"], df["tier"]).to_string())
    print()

    os.makedirs(chunk_root, exist_ok=True)
    for i in range(args.n_chunks):
        sub = df.iloc[i::args.n_chunks].copy()
        d = os.path.join(chunk_root, "chunk%02d" % i)
        os.makedirs(d, exist_ok=True)
        sub.to_csv(os.path.join(d, "qa.csv"), index=False)
        print("chunk%02d  %6d rows  %s"
              % (i, len(sub), dict(sub["tier"].value_counts())))

    total = sum(len(df.iloc[i::args.n_chunks]) for i in range(args.n_chunks))
    assert total == len(df), "row count mismatch"
    print()
    print("OK: %d rows across %d chunks, no row lost or duplicated."
          % (total, args.n_chunks))
    print("next:  python %s" % os.path.join(args.run_dir, "stats.py"))


if __name__ == "__main__":
    main()
