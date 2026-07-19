from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import evaluate_soft as S


DEFAULT_CATS = ["Relative Salience", "Morphological Attributes", "Anatomical Localization"]

DEFAULT_BANDS = [("low", 0.0, 0.2), ("mid", 0.4, 0.6), ("high", 0.8, 1.0001)]

# per-model CAGS outputs now live in  base_dir/<model>/Evaluation/
EVAL_SUBDIR = "Evaluation"
# the CAGS leaderboard now lives in  base_dir/Results/Eval - CAGS/
CAGS_RESULTS_SUBDIR = os.path.join("Results", "Eval - CAGS")

INSTRUCTIONS = """\
HOW TO RATE
-----------
For each row you see: a bird photo, a question (with answer options), the
CORRECT answer, and a MODEL's answer. Judge ONLY how correct the MODEL'S answer
is, using the image and the correct answer as your reference.

Rating scale (fill your column with 0, 1, or 2):
  2 = Right          the model's answer is correct / equivalent to the correct answer
  1 = Partially right the model's answer is not correct but is CLOSE -- e.g. the right
                     body region but wrong property, an adjacent color, an off-by-one
                     size, one of two correct parts, a near-miss location
  0 = Wrong          the model's answer is unrelated / clearly incorrect / far off

Rate independently. Do not confer. If you truly cannot tell from the image,
leave blank (do not guess).

Open the image at the path in 'image_path'. Judge against what you can see.
"""


def load_pool(base_dir, scored_name, images_root, cats, exclude_no_parse, allowed=None):
    rows = []
    S.OUTPUT_BASENAMES = {"eval_soft_scored.csv", "eval_soft_summary.csv", "eval_soft_leaderboard.csv", scored_name}
    for model_name, csv_path, pred_col in S.discover_models(base_dir):
        if allowed is not None and model_name not in allowed:
            continue
        d = os.path.dirname(csv_path)
        # eval_soft_scored.csv is written by evaluate_soft into the model's Evaluation/ subdir
        spath = os.path.join(d, EVAL_SUBDIR, scored_name)
        if not os.path.exists(spath):
            print(f"[warn] {model_name}: {scored_name} missing in {EVAL_SUBDIR}/ "
                  f"-> skipping (run SoftEval first)")
            continue
        cons = pd.read_csv(csv_path)
        sc = pd.read_csv(spath)
        if len(cons) != len(sc):
            print(f"[warn] {model_name}: length mismatch cons={len(cons)} "
                  f"scored={len(sc)} -> skipping")
            continue
        cons = cons.reset_index(drop=True)
        sc = sc.reset_index(drop=True)
        part = pd.DataFrame({
            "model": model_name,
            "image_path": cons["filename"].map(
                lambda f: os.path.join(images_root, str(f))),
            "question": cons["question"],
            "gold_answer": cons["answer"],
            "model_answer": cons[pred_col],
            "category": sc["category"],
            "family": sc["family"],
            "cags": sc["cags"],
            "exact": sc["exact"],
            "resolve_status": sc.get("resolve_status", "n/a"),
        })
        part = part[part["category"].isin(cats)]
        if exclude_no_parse:
            part = part[part["resolve_status"] != "no_parse"]
        rows.append(part)
    if not rows:
        sys.exit("[error] no model produced a joinable (consolidated + scored) pair")
    return pd.concat(rows, ignore_index=True)


def assign_band(cags, bands):
    for name, lo, hi in bands:
        if lo <= cags < hi:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="dataset root (REQUIRED)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--images_root", default=None,
                    help="prepended to the 'filename' field (default = base_dir)")
    ap.add_argument("--scored_name", default="eval_soft_scored.csv")
    ap.add_argument("--per_cell", type=int, default=10,
                    help="items per (category x band) cell")
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--cap_per_model", type=int, default=0,
                    help="max items one model may contribute per cell (0 = no cap)")
    ap.add_argument("--exclude_no_parse", action="store_true",
                    help="drop unparseable predictions (default: keep them)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model dir names to include "
                         "(e.g. 'Qwen3VL_32B,Qwen3VL_30B,InternVL3_38B'). "
                         "Overrides --top_models.")
    ap.add_argument("--top_models", type=int, default=0,
                    help="include only the top-N models by CAGS, read from the "
                         "leaderboard CSV under Results/Eval - CAGS (0 = all models)")
    ap.add_argument("--leaderboard_name", default="eval_soft_leaderboard.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"[start] manual-review sample | base_dir = {args.base_dir}", flush=True)

    # ---- resolve the allowed-model set ----
    allowed = None
    if args.models:
        allowed = {m.strip() for m in args.models.split(",") if m.strip()}
        print(f"[models] explicit whitelist ({len(allowed)}): {sorted(allowed)}")
    elif args.top_models > 0:
        lb_path = os.path.join(args.base_dir, CAGS_RESULTS_SUBDIR, args.leaderboard_name)
        if not os.path.exists(lb_path):
            sys.exit(f"[error] --top_models needs the leaderboard at {lb_path} "
                     f"(run SoftEval first) or pass --models explicitly")
        lb = pd.read_csv(lb_path)
        sort_col = "CAGS" if "CAGS" in lb.columns else lb.columns[3]
        lb = lb.sort_values(sort_col, ascending=False)
        allowed = set(lb["model"].head(args.top_models).tolist())
        print(f"[models] top-{args.top_models} by {sort_col}: {list(allowed)}")

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(base_dir, "Results", "Manual Review")
    images_root = args.images_root or base_dir
    os.makedirs(out_dir, exist_ok=True)
    cats = DEFAULT_CATS
    bands = DEFAULT_BANDS

    print(f"[pool] scanning {base_dir}")
    pool = load_pool(base_dir, args.scored_name, images_root, cats,
                     args.exclude_no_parse, allowed=allowed)
    pool["band"] = pool["cags"].map(lambda c: assign_band(float(c), bands))
    pool = pool[pool["band"].notna()].reset_index(drop=True)
    print(f"[pool] {len(pool)} candidate items across {pool['model'].nunique()} models")

    rng = np.random.default_rng(args.seed)
    picks = []
    print("\n[cells] target = %d per cell" % args.per_cell)
    for cat in cats:
        for bname, lo, hi in bands:
            cell = pool[(pool.category == cat) & (pool.band == bname)]
            if len(cell) == 0:
                print(f"  {cat:<26} {bname:<4}: EMPTY (0 available)")
                continue
            if args.cap_per_model > 0:
                keep_idx = []
                for _, gidx in cell.groupby("model", sort=False).groups.items():
                    gidx = pd.Index(gidx)
                    m = min(len(gidx), args.cap_per_model)
                    keep_idx.extend(pd.Series(gidx).sample(m, random_state=args.seed).tolist())
                cell = cell.loc[keep_idx]
            k = min(args.per_cell, len(cell))
            sel = cell.sample(k, random_state=int(rng.integers(1e9)))
            picks.append(sel)
            print(f"  {cat:<26} {bname:<4}: picked {k:>3} of {len(cell):>7}  "
                  f"(models: {sel['model'].nunique()})")

    sample = pd.concat(picks, ignore_index=True)
    # shuffle so band/category/model isn't guessable from row order
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "item_id", [f"IT{ i:04d}" for i in range(1, len(sample) + 1)])

    # blind annotator sheet
    sheet_cols = ["item_id", "image_path", "question", "gold_answer", "model_answer"]
    sheet = sample[sheet_cols].copy()
    for r in range(1, args.raters + 1):
        sheet[f"rater{r}"] = ""    # 0 / 1 / 2

    # hidden key
    key = sample[["item_id", "model", "category", "family",
                  "cags", "exact", "band", "resolve_status"]].copy()

    sheet_path = os.path.join(out_dir, "annotation_sheet.csv")
    key_path = os.path.join(out_dir, "annotation_key.csv")
    instr_path = os.path.join(out_dir, "INSTRUCTIONS.txt")
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(key_path, index=False)
    open(instr_path, "w").write(INSTRUCTIONS)

    print(f"\n[summary] {len(sample)} items | {args.raters} rater columns")
    print("  band distribution :", dict(sample.band.value_counts()))
    print("  category dist      :", dict(sample.category.value_counts()))
    print("  model spread       :", dict(sample.model.value_counts()))
    print("  EM=1 fraction      : %.3f (should be ~1/3, concentrated in 'high')"
          % sample.exact.mean())
    print(f"\n[write] {sheet_path}\n[write] {key_path}\n[write] {instr_path}")


if __name__ == "__main__":
    main()