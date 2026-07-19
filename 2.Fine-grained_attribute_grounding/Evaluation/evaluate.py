"""
evaluate.py
===========
Exact-match ("hard") evaluation for the CUB-200 fine-grained VQA component.

The headline signal is accuracy and the per-family / per-category breakdown --
these carry the real information for this benchmark. Macro precision/recall/F1
were intentionally dropped: with hundreds of open-vocabulary gold answers those
class-averaged scores collapse to near-noise and mislead more than they inform.

For every model directory under base_dir this computes strict correctness and
reports it from as many angles as a reviewer might ask for:

  - overall accuracy, plus MCQ-only vs open-ended-only accuracy
  - accuracy per category
  - accuracy per question family (routed with the same logic evaluate_soft uses)
  - accuracy per (category x family)
  - MCQ format diagnostics: parse rate, resolve-status counts,
    letter-vs-text mismatch rate
  - MCQ position-bias diagnostics: predicted-letter distribution and
    accuracy broken down by the gold option's letter

Outputs
-------
Per model, into  base_dir/<model_dir>/Evaluation/ :
    eval_stats.txt          human-readable report
    eval_rows.csv           one row per question (gold, pred, correct, status)
    eval_summary.csv        tidy long-form of every table (slice, key, n, acc)

Merged across all models, into  base_dir/Results/Eval - Exact Match/ :
    eval_stats_ALL.txt      leaderboard + every per-model report
    eval_leaderboard.csv    one row per model (accuracy, splits, parse rate ...)
    eval_by_category_ALL.csv    model x category accuracy matrix
    eval_by_family_ALL.csv      model x family accuracy matrix
    eval_format_ALL.csv         model x format/position-bias diagnostics

Model discovery, option parsing, gold parsing, prediction resolution and the
question-family router are reused from evaluate_soft so the two evaluators stay
consistent (same models, same families, same parse decisions).
"""

from __future__ import annotations
import argparse
import os
import re
import string
import sys
from collections import Counter

import numpy as np
import pandas as pd

import evaluate_soft as S


# --------------------------------------------------------------------------
# normalisation for the hard string comparison
# --------------------------------------------------------------------------

_SEP = re.compile(r"[-_/\\]")
_PUNC_TBL = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def norm_hard(s):
    """Aggressive normaliser: lower-case, split hyphen/slash/underscore,
    drop punctuation, collapse whitespace. Two strings are an exact match iff
    their norm_hard() forms are identical."""
    s = str(s).strip().lower()
    s = _SEP.sub(" ", s)
    s = s.translate(_PUNC_TBL)
    s = _WS.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------
# per-row evaluation
# --------------------------------------------------------------------------

def eval_row(question, answer, pred_raw):
    options = S.parse_options(str(question))
    fam = S.question_family(str(question), options)
    is_mcq = bool(options)

    g_let, g_txt = S.parse_gold(answer, options)
    p_let, p_txt, status = S.resolve_prediction(pred_raw, options, prefer="letter")

    if is_mcq and g_let is not None:
        correct = (p_let is not None) and (p_let == g_let)
    else:
        correct = (p_txt is not None) and (norm_hard(p_txt) == norm_hard(g_txt))

    return {
        "family": fam,
        "is_mcq": is_mcq,
        "gold_letter": g_let,
        "gold_text": g_txt,
        "pred_letter": p_let,
        "pred_text": p_txt,
        "resolve_status": status,
        "correct": bool(correct),
    }


# --------------------------------------------------------------------------
# metric helpers
# --------------------------------------------------------------------------

def acc(sub):
    return round(100.0 * sub["correct"].mean(), 2) if len(sub) else 0.0


def table_by(rows, col):
    out = []
    for key, g in rows.groupby(col, dropna=False):
        out.append({col: key, "n": len(g), "acc": acc(g)})
    df = pd.DataFrame(out).sort_values("acc", ascending=False)
    return df.reset_index(drop=True)


def table_cat_family(rows):
    out = []
    for (cat, fam), g in rows.groupby(["category", "family"], dropna=False):
        out.append({"category": cat, "family": fam, "n": len(g), "acc": acc(g)})
    df = pd.DataFrame(out).sort_values(["category", "acc"], ascending=[True, False])
    return df.reset_index(drop=True)


def format_diagnostics(rows):
    """MCQ-only: parse rate, resolve-status mix, position bias."""
    mcq = rows[rows["is_mcq"]]
    n = len(mcq)
    if n == 0:
        return {}, pd.DataFrame(), pd.DataFrame()

    status_counts = mcq["resolve_status"].value_counts().to_dict()
    parse_rate = round(100.0 * (mcq["resolve_status"] != "no_parse").mean(), 2)
    mismatch_rate = round(100.0 * (mcq["resolve_status"] == "letter_text_mismatch").mean(), 2)

    pred_letters = mcq["pred_letter"].dropna()
    pred_dist = (pred_letters.value_counts(normalize=True) * 100).round(2)
    chose_A = round(float(pred_dist.get("A", 0.0)), 2)

    # accuracy by the letter of the gold answer -> exposes "always picks A"
    acc_by_gold = (mcq.dropna(subset=["gold_letter"])
                      .groupby("gold_letter")["correct"].mean().mul(100).round(2))

    summary = {
        "mcq_n": n,
        "parse_rate": parse_rate,
        "letter_text_mismatch_rate": mismatch_rate,
        "chose_A_pct": chose_A,
        "n_no_parse": int(status_counts.get("no_parse", 0)),
    }
    pred_dist_df = pred_dist.rename_axis("pred_letter").reset_index(name="pct")
    acc_by_gold_df = acc_by_gold.rename_axis("gold_letter").reset_index(name="acc")
    return summary, pred_dist_df, acc_by_gold_df


# --------------------------------------------------------------------------
# per-model driver
# --------------------------------------------------------------------------

def evaluate_model(model_name, csv_path, pred_col, q_col, a_col, eval_dir):
    df = pd.read_csv(csv_path)
    for c in (q_col, a_col, pred_col):
        if c not in df.columns:
            print(f"  [skip] {model_name}: missing column '{c}'")
            return None

    recs = [eval_row(q, a, p) for q, a, p in zip(df[q_col], df[a_col], df[pred_col])]
    rows = pd.DataFrame(recs)
    rows["category"] = df.get("category", pd.Series(["?"] * len(df))).astype(str).str.strip().values
    for extra in ("filename", "image_id"):
        if extra in df.columns:
            rows[extra] = df[extra].values

    total = len(rows)
    correct = int(rows["correct"].sum())
    accuracy = acc(rows)

    mcq = rows[rows["is_mcq"]]
    opn = rows[~rows["is_mcq"]]
    mcq_acc = acc(mcq)
    open_acc = acc(opn)

    by_cat = table_by(rows, "category")
    by_fam = table_by(rows, "family")
    by_cf = table_cat_family(rows)
    fmt, pred_dist_df, acc_by_gold_df = format_diagnostics(rows)

    # ---- write per-model CSVs ----
    rows_out_cols = [c for c in ["filename", "image_id", "category", "family",
                                 "is_mcq", "gold_letter", "gold_text",
                                 "pred_letter", "pred_text", "resolve_status",
                                 "correct"] if c in rows.columns]
    rows[rows_out_cols].to_csv(os.path.join(eval_dir, "eval_rows.csv"), index=False)

    summary_long = pd.concat([
        pd.DataFrame([{"slice": "overall", "key": "ALL", "n": total, "acc": accuracy}]),
        by_cat.rename(columns={"category": "key"}).assign(slice="category")[["slice", "key", "n", "acc"]],
        by_fam.rename(columns={"family": "key"}).assign(slice="family")[["slice", "key", "n", "acc"]],
    ], ignore_index=True)
    summary_long.to_csv(os.path.join(eval_dir, "eval_summary.csv"), index=False)

    # ---- build the text report ----
    L = ["=" * 80, model_name, "=" * 80,
         f"CSV                : {os.path.basename(csv_path)}",
         f"Prediction column  : {pred_col}",
         f"Questions          : {total}",
         f"Correct            : {correct}",
         "",
         f"Accuracy (all)     : {accuracy:.2f}%",
         f"  MCQ only         : {mcq_acc:.2f}%   (n={len(mcq)})",
         f"  Open-ended only  : {open_acc:.2f}%   (n={len(opn)})",
         ""]

    if fmt:
        L += ["MCQ FORMAT / POSITION DIAGNOSTICS", "-" * 80,
              f"Parse rate                 : {fmt['parse_rate']:.2f}%   "
              f"(unparseable: {fmt['n_no_parse']})",
              f"Letter/text mismatch rate  : {fmt['letter_text_mismatch_rate']:.2f}%",
              f"Chose option (A)           : {fmt['chose_A_pct']:.2f}%   "
              f"(a heavy skew toward one letter flags position bias)",
              "",
              "Predicted-letter distribution (%):",
              pred_dist_df.to_string(index=False) if len(pred_dist_df) else "n/a",
              "",
              "Accuracy by gold-answer letter (%):",
              acc_by_gold_df.to_string(index=False) if len(acc_by_gold_df) else "n/a",
              ""]

    L += ["PER CATEGORY", "-" * 80,
          by_cat.to_string(index=False) if len(by_cat) else "No category information.",
          "",
          "PER QUESTION FAMILY", "-" * 80,
          by_fam.to_string(index=False) if len(by_fam) else "No family information.",
          "",
          "PER (CATEGORY x FAMILY)", "-" * 80,
          by_cf.to_string(index=False) if len(by_cf) else "n/a"]

    report = "\n".join(L)
    with open(os.path.join(eval_dir, "eval_stats.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    result = {
        "model": model_name,
        "n": total,
        "accuracy": accuracy,
        "mcq_acc": mcq_acc,
        "open_acc": open_acc,
    }
    if fmt:
        result.update({k: fmt[k] for k in
                       ("parse_rate", "letter_text_mismatch_rate", "chose_A_pct")})

    return {
        "result": result,
        "report": report,
        "by_cat": by_cat.set_index("category")["acc"],
        "by_fam": by_fam.set_index("family")["acc"],
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="root holding one sub-directory per model (REQUIRED)")
    ap.add_argument("--q_col", default="question")
    ap.add_argument("--a_col", default="answer")
    ap.add_argument("--skip_done", action="store_true",
                    help="skip a model if its Evaluation/eval_stats.txt exists")
    args = ap.parse_args()

    base_dir = args.base_dir
    print(f"[start] evaluate | base_dir = {base_dir}", flush=True)
    if not os.path.isdir(base_dir):
        sys.exit(f"[error] base_dir does not exist: {base_dir}")

    # never let discovery pick up our own or the soft-eval outputs
    S.OUTPUT_BASENAMES = {
        "eval_soft_scored.csv", "eval_soft_summary.csv", "eval_soft_leaderboard.csv",
        "eval_rows.csv", "eval_summary.csv",
    }

    overall_dir = os.path.join(base_dir, "Results", "Eval - Exact Match")
    os.makedirs(overall_dir, exist_ok=True)

    print("=" * 80)
    print(f"Exact-match evaluation over: {base_dir}")
    print("=" * 80)

    board, reports = [], []
    cat_matrix, fam_matrix, fmt_rows = {}, {}, []
    n_models = 0

    for model_name, csv_path, pred_col in S.discover_models(base_dir):
        n_models += 1
        eval_dir = os.path.join(os.path.dirname(csv_path), "Evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        if args.skip_done and os.path.exists(os.path.join(eval_dir, "eval_stats.txt")):
            print(f"[skip_done] {model_name}")
            continue

        print(f"Processing: {model_name}  (pred: {pred_col})")
        out = evaluate_model(model_name, csv_path, pred_col,
                             args.q_col, args.a_col, eval_dir)
        if out is None:
            continue

        board.append(out["result"])
        reports.append(out["report"])
        cat_matrix[model_name] = out["by_cat"]
        fam_matrix[model_name] = out["by_fam"]
        r = out["result"]
        fmt_rows.append({k: r.get(k) for k in
                         ["model", "n", "accuracy", "mcq_acc", "open_acc",
                          "parse_rate", "letter_text_mismatch_rate", "chose_A_pct"]})

    if not board:
        print("[done] no model results found.")
        return

    # ---- merged leaderboard ----
    lb = (pd.DataFrame(board)
            .sort_values("accuracy", ascending=False)
            .reset_index(drop=True))
    lb.to_csv(os.path.join(overall_dir, "eval_leaderboard.csv"), index=False)

    # ---- model x category / family accuracy matrices ----
    cat_df = pd.DataFrame(cat_matrix).T.sort_index()
    fam_df = pd.DataFrame(fam_matrix).T.sort_index()
    cat_df.to_csv(os.path.join(overall_dir, "eval_by_category_ALL.csv"))
    fam_df.to_csv(os.path.join(overall_dir, "eval_by_family_ALL.csv"))

    # ---- format / bias matrix ----
    pd.DataFrame(fmt_rows).to_csv(
        os.path.join(overall_dir, "eval_format_ALL.csv"), index=False)

    # ---- combined text file ----
    all_path = os.path.join(overall_dir, "eval_stats_ALL.txt")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write("EXACT-MATCH EVALUATION - MERGED REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write("WHAT THIS IS\n")
        f.write("  Strict correctness: a prediction counts only if it equals the gold answer\n")
        f.write("  after normalisation (MCQ scored on the chosen letter). Accuracy and the\n")
        f.write("  per-family / per-category breakdown carry the signal for this benchmark;\n")
        f.write("  macro P/R/F1 was dropped as uninformative for open-vocabulary answers.\n\n")
        f.write("COLUMNS\n")
        f.write("  accuracy  overall exact-match accuracy (%).\n")
        f.write("  mcq_acc   accuracy on multiple-choice items only.\n")
        f.write("  open_acc  accuracy on open-ended items only (the recognition-vs-generation\n")
        f.write("            gap shows up as mcq_acc >> open_acc).\n")
        f.write("  parse_rate / letter_text_mismatch_rate / chose_A_pct  MCQ format & position\n")
        f.write("            -bias diagnostics; a high chose_A_pct flags letter-position bias.\n\n")
        f.write("FILES  (paths relative to base_dir)\n")
        f.write("  <model>/Evaluation/eval_rows.csv     one row per question (gold, pred, status, correct).\n")
        f.write("  <model>/Evaluation/eval_summary.csv  tidy overall/family/category accuracy.\n")
        f.write("  <model>/Evaluation/eval_stats.txt    per-model report.\n")
        f.write("  Results/Eval - Exact Match/eval_leaderboard.csv     one row per model.\n")
        f.write("  Results/Eval - Exact Match/eval_by_category_ALL.csv model x category accuracy.\n")
        f.write("  Results/Eval - Exact Match/eval_by_family_ALL.csv   model x family accuracy.\n")
        f.write("  Results/Eval - Exact Match/eval_format_ALL.csv      model x format/bias diagnostics.\n\n")
        f.write("=" * 100 + "\n")
        f.write("OVERALL MODEL COMPARISON  (exact match)\n")
        f.write("=" * 100 + "\n\n")
        f.write(lb.to_string(index=False))
        f.write("\n\nACCURACY BY CATEGORY (rows = model)\n")
        f.write("-" * 100 + "\n")
        f.write(cat_df.round(2).to_string())
        f.write("\n\nACCURACY BY QUESTION FAMILY (rows = model)\n")
        f.write("-" * 100 + "\n")
        f.write(fam_df.round(2).to_string())
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("INDIVIDUAL MODEL REPORTS\n")
        f.write("=" * 100 + "\n\n")
        f.write(("\n\n" + "=" * 100 + "\n\n").join(reports))

    print("\n" + "=" * 80)
    print("LEADERBOARD (exact match, sorted by accuracy)")
    print("=" * 80)
    print(lb.to_string(index=False))
    print(f"\n[write] {all_path}")
    print(f"[done] {n_models} model dir(s) processed.")


if __name__ == "__main__":
    main()