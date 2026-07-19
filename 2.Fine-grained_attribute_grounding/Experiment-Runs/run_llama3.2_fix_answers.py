import argparse, re, sys
from collections import defaultdict
import pandas as pd

PREAMBLE = re.compile(
    r"^(?:the\s+\w+\s+of\s+the\s+bird\s+(?:in\s+th\w+\s+image\s+)?is"
    r"|the\s+bird\s+(?:in\s+th\w+\s+image\s+)?is|this\s+is|it\s+is"
    r"|scientific\s+name|common\s+name|genus|answer)"
    r"\s*[:\-]?\s*(?:an?\s+)?", re.I)


def clean_text(s):
    s = re.sub(r"[*`#]", "", str(s))          # kill markdown, keep the middle
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(3):                        # peel "The genus of the bird is X."
        t = PREAMBLE.sub("", s, count=1).strip()
        if t == s or not t:
            break
        s = t
    return s.strip(" .;,:")                   # kill the trailing period


def norm(s):
    """'Black-footed Albatross' / \"Bewick's Wren\" -> 'black footed albatross'"""
    s = re.sub(r"'s\b", "", str(s).lower())
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", s).split()) + " "


def parse_options(q):
    m = re.search(r"options\s*:\s*(.+)$", str(q), re.I | re.S)
    if not m:
        return {}
    items = re.findall(r"\(([A-Za-z])\)\s*(.*?)(?=\s*,\s*\([A-Za-z]\)|\s*$)",
                       m.group(1), re.S)
    return {L.upper(): re.sub(r"\s+", " ", b).strip(" .;,") for L, b in items}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--pred_col", default=None)
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    pred_col = a.pred_col or next(
        (c for c in df.columns
         if c.lower().endswith("_answer") and c.lower() != "answer"), None)
    if not pred_col:
        sys.exit("no *_answer prediction column found")

    # vocab: labels ever used as GT for this exact (non-MCQ) question
    vocab = defaultdict(set)
    for q, gt in zip(df["question"], df["answer"]):
        if "options:" not in str(q).lower() and pd.notna(gt):
            vocab[str(q)].add(str(gt).strip())
    vocab = {q: sorted(v, key=lambda x: -len(x)) for q, v in vocab.items()}

    out, method = [], []
    for q, raw in zip(df["question"], df[pred_col]):
        if pd.isna(raw):
            out.append(""); method.append("empty"); continue

        txt = clean_text(raw)
        opts = parse_options(q)

        if opts:                                       # ---- MCQ
            m = re.search(r"\(\s*([A-Za-z])\s*\)", txt)
            L = m.group(1).upper() if m else None
            if L in opts:
                out.append(f"({L}) {opts[L]}"); method.append("letter")
            else:
                out.append(txt); method.append("no_letter")
        else:                                          # ---- naming
            n = norm(txt)
            hit = next((lab for lab in vocab.get(str(q), []) if norm(lab) in n), None)
            out.append(hit or txt)
            method.append("vocab" if hit else "unmatched")


    gt = df["answer"].astype(str).str.strip()
    clean_series = pd.Series(out, index=df.index)
    method_series = pd.Series(method, index=df.index)

    print(f"rows            : {len(df)}")
    print(f"EM raw          : {(df[pred_col].astype(str).str.strip() == gt).mean():.2%}")
    print(f"EM clean        : {(clean_series.astype(str).str.strip() == gt).mean():.2%}")
    print(method_series.value_counts().to_string())

    bad_mask = method_series.isin(["no_letter", "unmatched"])

    if bad_mask.any():
    
        print(f"\nneeds a look (n={int(bad_mask.sum())}):")
        for _, r in df[bad_mask].head(10).iterrows():
            print(f"  gt={r['answer']!r:<32} pred={str(r[pred_col])[:60]!r}")


    df[pred_col] = clean_series
    df.to_csv(a.csv, index=False)
    print(f"\nupdated column {pred_col!r} in place, saved: {a.csv}")


if __name__ == "__main__":
    main()