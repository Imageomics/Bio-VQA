import os
import re
import argparse
import pandas as pd

ANNOT_PREFIX = "Images_Annotated/"

# Annotated files are named "{source_stem}_{5-hex-uuid}.jpg"
UUID_SUFFIX = re.compile(r"_[0-9a-f]{5}$", re.IGNORECASE)

# MCQ answers look like "(C) crown"
LETTER_KEY = re.compile(r"^\(([A-E])\)\s*(.*)$", re.DOTALL)


def stem_of(path):
    """images/001.X/Y_0046_18.jpg -> Y_0046_18"""
    return os.path.splitext(os.path.basename(str(path)))[0]


def build_species_index(df):

    is_annot = df["filename"].astype(str).str.startswith(ANNOT_PREFIX)

    # Species is reliable ONLY for source rows: images/{species}/{file}
    src_species = (
        df.loc[~is_annot, "filename"].astype(str).str.split("/").str[1]
    )

    src_stems = df.loc[~is_annot, "filename"].map(stem_of)
    stem_to_species = dict(zip(src_stems, src_species))
    
    raw_annot = df.loc[is_annot, "filename"].map(stem_of)
    stripped = raw_annot.str.replace(UUID_SUFFIX, "", regex=True)
    annot_stems = stripped.where(stripped.isin(stem_to_species), raw_annot)

    source_stem = pd.Series(index=df.index, dtype=object)
    source_stem.loc[~is_annot] = src_stems
    source_stem.loc[is_annot] = annot_stems

    species = source_stem.map(stem_to_species).fillna("Unresolved")
    n_unmatched = int((species == "Unresolved").sum())

    return species, is_annot, source_stem, n_unmatched


def split_answer(ans):
    """'(C) crown' -> ('C', 'crown');  'Phoebastria' -> (None, 'Phoebastria')"""
    m = LETTER_KEY.match(str(ans).strip())
    return (m.group(1), m.group(2).strip()) if m else (None, str(ans).strip())


def pct(n, d):
    return (n / d * 100) if d else 0.0


def generate_cub_stats(csv_path, out_name="data-stats.txt"):
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    output_file = os.path.join(base_dir, out_name)

    print(f"[*] Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Error loading CSV: {e}")
        return

    if df.empty:
        print("[!] CSV is empty; nothing to report.")
        return

    required = {"filename", "question", "answer", "category"}
    missing = required - set(df.columns)
    if missing:
        print(f"[!] Missing required columns: {sorted(missing)}")
        return

    species, is_annot, source_stem, n_unmatched = build_species_index(df)
    df = df.assign(species=species, is_annotated=is_annot, source_stem=source_stem)

    df[["answer_letter", "answer_text"]] = df["answer"].apply(
        lambda a: pd.Series(split_answer(a))
    )

    total_qa = len(df)
    src = df[~df.is_annotated]
    ann = df[df.is_annotated]

    n_src_img = src["filename"].nunique()
    n_ann_img = ann["filename"].nunique()
    n_photographs = df["source_stem"].nunique()   # underlying distinct photos
    n_species = df.loc[df.species != "Unresolved", "species"].nunique()

    with open(output_file, "w", encoding="utf-8") as f:
        w = f.write
        w("=" * 69 + "\n")
        w("             CUB-200 FINE-GRAINED VQA DATASET STATISTICS\n")
        w("=" * 69 + "\n\n")

        # ------------------------------------------------------------------
        w("[1] HIGH-LEVEL METRICS\n")
        w("-" * 65 + "\n")
        w(f"Total QA Pairs                  : {total_qa:,}\n")
        w(f"Distinct Photographs            : {n_photographs:,}\n")
        w(f"  - Source image files          : {n_src_img:,}\n")
        w(f"  - Marker-annotated renders    : {n_ann_img:,}\n")
        w(f"Total Image Files Referenced    : {n_src_img + n_ann_img:,}\n")
        if n_photographs:
            w(f"Avg Questions per Photograph    : {total_qa / n_photographs:.2f}\n")
        if n_src_img:
            w(f"Avg Questions per Source Image  : {len(src) / n_src_img:.2f}\n")
        q_per_src = src.groupby("filename").size()
        if len(q_per_src):
            w(f"Max Questions on a Source Image : {q_per_src.max()}\n")
            w(f"Min Questions on a Source Image : {q_per_src.min()}\n")
        if len(ann):
            q_per_ann = ann.groupby("filename").size()
            w(f"Questions per Annotated Image   : {q_per_ann.min()}-{q_per_ann.max()}\n")
        w("\n")

        # ------------------------------------------------------------------
        w("[2] BIRD SPECIES / CLASS METRICS\n")
        w("-" * 65 + "\n")
        w(f"Total Unique Bird Species       : {n_species:,}\n")
        if n_species:
            w(f"Average Photographs per Species : {n_photographs / n_species:.2f}\n")
            w(f"Average Questions per Species   : {total_qa / n_species:.2f}\n")
            per_sp = df.groupby("species").size()
            w(f"Min / Max Questions per Species : {per_sp.min():,} / {per_sp.max():,}\n")
        if n_unmatched:
            w(f"[!] Rows with unresolved species : {n_unmatched:,}\n")
        w("\n")

        # ------------------------------------------------------------------
        w("[3] QUESTION STRUCTURAL ANALYSIS\n")
        w("-" * 65 + "\n")
        mc_mask = df["question"].str.contains("Options:", na=False)
        mc_count = int(mc_mask.sum())
        open_count = total_qa - mc_count
        w(f"Multiple-Choice Questions       : {mc_count:,} ({pct(mc_count, total_qa):.1f}%)\n")
        w(f"Open-Ended Questions            : {open_count:,} ({pct(open_count, total_qa):.1f}%)\n")
        yes_no = int(df["answer_text"].str.lower().isin(["yes", "no"]).sum())
        w(f"Binary (Yes/No) Answers         : {yes_no:,} ({pct(yes_no, total_qa):.1f}%)\n")
        w(f"Visually-Prompted (marker) Items: {len(ann):,} ({pct(len(ann), total_qa):.1f}%)\n")

        # Answer-key balance: a skewed letter distribution is a leakage channel.
        letters = df.loc[mc_mask, "answer_letter"].value_counts().sort_index()
        if len(letters):
            w("Answer-Key Distribution (MCQ)   : ")
            w(", ".join(f"{k}={pct(v, mc_count):.1f}%" for k, v in letters.items()))
            w("\n")
        w("\n")

        # ------------------------------------------------------------------
        w("[4] DETAILED CATEGORY BREAKDOWN\n")
        w("-" * 65 + "\n")
        for cat, count in df["category"].value_counts().items():
            cat_df = df[df["category"] == cat]
            cat_mc = int(cat_df["question"].str.contains("Options:", na=False).sum())
            w(f"\u27a4 {cat}:\n")
            w(f"   \u2022 Total Questions : {count:,} ({pct(count, total_qa):.1f}%)\n")
            w(f"   \u2022 Unique Stems    : {cat_df['question'].nunique():,}\n")
            w(f"   \u2022 Unique Answers  : {cat_df['answer_text'].nunique():,}\n")
            w(f"   \u2022 Multiple-Choice : {cat_mc:,}\n")
            w(f"   \u2022 Open-Ended      : {count - cat_mc:,}\n")
            w(f"   \u2022 Photographs     : {cat_df['source_stem'].nunique():,}\n\n")

        w("=" * 69 + "\n")

    print(f"[+] Successfully generated detailed CUB-200 VQA stats at: {output_file}")
    if n_unmatched:
        print(f"[!] {n_unmatched:,} rows could not be matched to a species.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate detailed stats for the CUB-200 QA dataset."
    )
    p.add_argument("--csv_path", type=str, required=True,
                   help="Absolute path to the QA.csv file")
    p.add_argument("--out_name", type=str, default="data-stats.txt",
                   help="Output filename (written beside the CSV)")
    args = p.parse_args()
    generate_cub_stats(args.csv_path, args.out_name)