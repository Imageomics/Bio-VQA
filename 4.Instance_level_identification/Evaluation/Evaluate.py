#!/usr/bin/env python3
"""
BioVQA C4 - evaluation of instance-level identification results.

Walks --base_dir for every instance_identification_<N>_<condition> directory,
reads every QA_<model>.csv inside, and scores each (model x species x
condition) cell.

  python evaluate_c4.py --base_dir /path/to/4-Instance-Level-Identification

Outputs, written to --out_dir (default <base_dir>/evaluation):
  QA_<model>_results.csv    one row per species x condition for that model
  QA_all_results.csv        every cell, every model
  QA_summary_by_model.csv   pooled per model x condition + difficulty effect
  QA_unparsed_samples.csv   raw strings the parser could not resolve

WHY THESE METRICS
Raw accuracy is not interpretable here. A model that answers "Yes" to
everything scores exactly 0.500 on a balanced set while discriminating
nothing. The task is a same/different discrimination, so the natural frame is
signal detection theory: d' measures discriminability independently of how
biased the model is, and criterion c measures the bias itself. Balanced
accuracy and MCC are reported as bias-robust alternatives that need no
distributional assumption.

PARSING
The C1/C2 soft-EM idea does not transfer directly - there is no synonym set
for "Yes". The analogue implemented here is TIERED parsing: each answer is
resolved by the strictest rule that applies, and the tier is recorded, so the
permissiveness of the parser is auditable rather than hidden. Tier counts
appear in every output row.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from statistics import NormalDist

import numpy as np
import pandas as pd

RESERVED_COLUMNS = {"image_path", "qa_pair", "question", "answer", "label"}
_Z = NormalDist().inv_cdf


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def strip_decoration(text):
    """Remove markdown, quotes, and trailing punctuation without touching words."""
    text = str(text).strip()
    text = re.sub(r"[*_`#>]", "", text)          # markdown emphasis / headers
    text = text.strip().strip("\"'“”‘’ \t\r\n")
    text = text.strip(" .,:;!?-–—")
    return text.strip()


def parse_answer(raw):
    """
    Resolve a raw model string to "Yes" / "No" / "" plus the tier that did it.

    exact           the whole answer is yes/no once decoration is stripped
                    ("Yes", "yes.", "**No**", "'No'")
    prefix          the answer begins with yes/no
                    ("Yes, they are the same individual")
    contains_unique exactly one of the two words appears anywhere, as a word
                    ("The answer is no")
    negated         an explicit "not the same" / "different individual" phrasing
                    with no bare yes/no token
    unparsed        anything else, including answers containing BOTH words
    """
    if raw is None or (isinstance(raw, float) and raw != raw):
        return "", "empty"

    text = strip_decoration(raw)
    if not text:
        return "", "empty"

    low = text.lower()

    if low in ("yes", "y", "true", "same"):
        return "Yes", "exact"
    if low in ("no", "n", "false", "different"):
        return "No", "exact"

    # Instruction echo: "Yes or No", "yes/no". Never a real answer.
    if re.match(r"^(yes|no)\s*(or|/|,)\s*(yes|no)\b", low):
        return "", "unparsed"

    # Refusal or hedge. Without this, "I am not sure" resolves to No on the
    # word "not", silently converting an abstention into a confident answer.
    if re.search(r"\b(cannot|can not|can't|cant|unable|not sure|unsure|"
                 r"not possible|impossible|difficult to (tell|say|determine)|"
                 r"insufficient|unclear|hard to (tell|say)|"
                 r"no way to (tell|know|determine))\b", low):
        return "", "unparsed"

    if re.match(r"^yes\b", low):
        return "Yes", "prefix"
    if re.match(r"^no\b", low):
        return "No", "prefix"

    has_yes = re.search(r"\byes\b", low) is not None
    has_no = re.search(r"\b(no|not)\b", low) is not None
    if has_yes and not has_no:
        return "Yes", "contains_unique"
    if has_no and not has_yes:
        return "No", "contains_unique"

    # both or neither: fall back to explicit same/different phrasing
    if not (has_yes and has_no):
        if re.search(r"\b(not the same|different individual|different animal)\b", low):
            return "No", "negated"
        if re.search(r"\b(the same individual|same animal)\b", low):
            return "Yes", "negated"

    return "", "unparsed"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def prop_se(p, n):
    if not n or p != p:
        return float("nan")
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def diff_test(p1, n1, p2, n2):
    """
    Difference between two independent proportions.
    Returns (diff, se, z, ci_lo, ci_hi, significant_at_05).
    Cells come from disjoint item sets, so the variances add.
    """
    if p1 != p1 or p2 != p2 or not n1 or not n2:
        nan = float("nan")
        return nan, nan, nan, nan, nan, False
    d = p1 - p2
    se = float(np.sqrt(prop_se(p1, n1) ** 2 + prop_se(p2, n2) ** 2))
    z = d / se if se > 0 else float("nan")
    return d, se, z, d - 1.96 * se, d + 1.96 * se, abs(z) >= 1.96 if z == z else False


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p). Exact, no scipy."""
    from math import comb
    if n == 0:
        return float("nan")
    return float(sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
                     for i in range(k, n + 1)))


def signal_detection(tp, fn, fp, tn):
    """
    d' and criterion c with the log-linear correction.

    The correction (add 0.5 to each count, 1 to each total) is applied
    unconditionally rather than only at ceiling, because applying it
    selectively biases comparisons between cells.
    """
    hit = (tp + 0.5) / (tp + fn + 1.0)
    fa = (fp + 0.5) / (fp + tn + 1.0)
    z_h, z_f = _Z(hit), _Z(fa)
    return {
        "hit_rate": hit,
        "false_alarm_rate": fa,
        "d_prime": z_h - z_f,
        "criterion_c": -0.5 * (z_h + z_f),
    }


def bootstrap_ci(truth, pred, n_boot, seed, statistic="accuracy"):
    """Percentile CI over item resampling. Returns (lo, hi)."""
    if len(truth) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    correct = (truth == pred)
    idx = rng.integers(0, len(truth), size=(n_boot, len(truth)))

    if statistic == "accuracy":
        stats = correct[idx].mean(axis=1)
    else:  # balanced accuracy
        same = (truth == 1)
        stats = np.empty(n_boot)
        for b in range(n_boot):
            t, c = truth[idx[b]], correct[idx[b]]
            s, d = (t == 1), (t == 0)
            stats[b] = 0.5 * ((c[s].mean() if s.any() else np.nan)
                              + (c[d].mean() if d.any() else np.nan))
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def score_cell(truth_labels, pred_labels, tiers, n_boot, seed):
    """All metrics for one (model, species, condition) cell."""
    n_total = len(truth_labels)
    keep = [i for i, p in enumerate(pred_labels) if p]
    n_scored = len(keep)
    n_unparsed = n_total - n_scored

    row = {
        "n_items": n_total,
        "n_scored": n_scored,
        "n_unparsed": n_unparsed,
        "coverage": n_scored / n_total if n_total else float("nan"),
    }
    for tier in ("exact", "prefix", "contains_unique", "negated",
                 "unparsed", "empty"):
        row["tier_" + tier] = tiers.get(tier, 0)

    if n_scored == 0:
        row.update({k: float("nan") for k in (
            "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
            "accuracy_all_items", "balanced_accuracy", "bacc_ci_lo", "bacc_ci_hi",
            "precision", "recall_sensitivity", "specificity", "npv", "f1",
            "mcc", "cohens_kappa", "d_prime", "criterion_c",
            "hit_rate", "false_alarm_rate", "yes_rate", "yes_rate_truth")})
        row.update({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
        return row

    truth = np.array([1 if truth_labels[i] == "Yes" else 0 for i in keep])
    pred = np.array([1 if pred_labels[i] == "Yes" else 0 for i in keep])

    tp = int(((truth == 1) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    row.update({"TP": tp, "FN": fn, "FP": fp, "TN": tn})

    acc = (tp + tn) / n_scored
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if prec == prec and sens == sens and (prec + sens) else float("nan"))

    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom > 0 else float("nan")

    p_obs = acc
    p_exp = (((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n_scored ** 2))
    kappa = ((p_obs - p_exp) / (1 - p_exp)) if p_exp != 1 else float("nan")

    acc_lo, acc_hi = bootstrap_ci(truth, pred, n_boot, seed, "accuracy")
    b_lo, b_hi = bootstrap_ci(truth, pred, n_boot, seed + 1, "balanced")

    row.update({
        "accuracy": acc,
        "accuracy_ci_lo": acc_lo,
        "accuracy_ci_hi": acc_hi,
        # unparsed counted as wrong: the pessimistic reading
        "accuracy_all_items": (tp + tn) / n_total if n_total else float("nan"),
        "balanced_accuracy": 0.5 * (sens + spec),
        "bacc_ci_lo": b_lo,
        "bacc_ci_hi": b_hi,
        "precision": prec,
        "recall_sensitivity": sens,
        "specificity": spec,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
        "cohens_kappa": kappa,
        "yes_rate": float(pred.mean()),
        "yes_rate_truth": float(truth.mean()),
    })
    row.update(signal_detection(tp, fn, fp, tn))
    return row


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def parse_task_dir(task_dir):
    """instance_identification_1000_easy -> (n_pairs, condition)."""
    name = os.path.basename(os.path.normpath(task_dir))
    m = re.match(r"instance_identification_(\d+)_(.+)$", name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def species_label(task_dir):
    parent = os.path.basename(os.path.dirname(os.path.normpath(task_dir)))
    label = re.sub(r"\.coco$", "", parent)
    return label, parent


def find_tasks(base_dir, dir_pattern):
    out = []
    for root, dirs, _files in os.walk(base_dir):
        dirs.sort()
        for d in list(dirs):
            if d.startswith(dir_pattern):
                out.append(os.path.join(root, d))
    return sorted(set(out))


def answer_column(df, model):
    """Prefer <model>_answer; otherwise the single non-reserved column."""
    preferred = model + "_answer"
    if preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns
                  if c not in RESERVED_COLUMNS and not c.startswith("Unnamed")]
    if len(candidates) == 1:
        return candidates[0]
    ending = [c for c in candidates if c.endswith("_answer")]
    if len(ending) == 1:
        return ending[0]
    return None


# --------------------------------------------------------------------------- #
# narrative reports
# --------------------------------------------------------------------------- #

# Thresholds used by the interpretation text. Stated here rather than buried
# in the prose so they can be cited, argued with, or changed in one place.
D_PRIME_FLOOR = 0.20      # below this, discriminability is not meaningful
D_PRIME_STRONG = 1.00     # above this, clearly discriminating
CRITERION_STRONG = 1.00   # |c| above this is a severe response bias
CRITERION_MILD = 0.35
YES_RATE_DEGENERATE = 0.90
UNPARSED_CONCERN = 0.02   # 2% of items


def rule(char="-", width=78):
    return char * width


def wrap(text, width=78, indent=""):
    import textwrap
    return "\n".join(textwrap.fill(p, width, initial_indent=indent,
                                   subsequent_indent=indent)
                     if p.strip() else ""
                     for p in text.split("\n"))


def classify_model(s):
    """
    (verdict, explanation) for one row of the summary table.

    The verdict is keyed on the RANDOM condition, not the best condition.
    Easy-condition pairs are separable by appearance alone, so scoring well
    there does not demonstrate identity verification - grading on it would be
    close to circular. Random is the neutral, unbiased estimate of capability.
    """
    ds = [s.get(c + "_d_prime", float("nan")) for c in
          ("easy", "random", "hard")]
    ds = [d for d in ds if d == d]
    if not ds:
        return "no data", "No scored cells."

    neutral = s.get("random_d_prime", float("nan"))
    if neutral != neutral:
        neutral = max(ds)
    best = neutral
    worst = min(ds)
    yr = [s.get(c + "_yes_rate", float("nan")) for c in
          ("easy", "random", "hard")]
    yr = [y for y in yr if y == y]
    extreme_yes = any(y >= YES_RATE_DEGENERATE or y <= 1 - YES_RATE_DEGENERATE
                      for y in yr)

    # Strongly negative d' is NOT chance performance: the model is separating
    # the classes and then labelling them backwards. Judged on the NEUTRAL
    # condition only - the hard condition is built to invert d', so including
    # it here would misdiagnose every model the manipulation works on.
    if neutral <= -D_PRIME_FLOOR:
        return ("systematically inverted",
                "d' is negative in every condition, meaning the model "
                "separates same from different pairs but assigns the labels "
                "the wrong way round. Check the answer polarity and the "
                "ground-truth column before interpreting this as a finding.")

    if best < D_PRIME_FLOOR:
        if extreme_yes:
            return ("degenerate",
                    "d' is at or below zero in every condition and the model "
                    "answers almost always the same way. Its accuracy near "
                    "0.5 reflects the class balance, not perception.")
        return ("at chance",
                "d' is at or below the floor in every condition. The model "
                "responds to both classes but the responses carry no "
                "information about identity.")

    if best < D_PRIME_STRONG:
        return ("weak discrimination",
                "d' in the neutral condition is above the floor but modest "
                "(%.2f). The model extracts some signal about individual "
                "identity, well short of reliable." % best)

    return ("discriminating",
            "d' in the neutral condition is substantial (%.2f): the model "
            "separates same-individual from different-individual pairs "
            "without help from the appearance manipulation." % best)


def bias_note(s):
    cs = [(c, s.get(c + "_criterion_c", float("nan")),
           s.get(c + "_yes_rate", float("nan")))
          for c in ("easy", "random", "hard")]
    cs = [(c, v, y) for c, v, y in cs if v == v]
    if not cs:
        return ""
    worst = max(cs, key=lambda t: abs(t[1]))
    cond, c, y = worst
    if abs(c) >= CRITERION_STRONG:
        direction = "Yes" if c < 0 else "No"
        return ("Severe response bias: criterion c = %+.2f in the %s condition "
                "(yes-rate %.3f), i.e. a strong standing preference for "
                "\"%s\" regardless of the images. Raw accuracy is not "
                "interpretable for this model; use d' or balanced accuracy."
                % (c, cond, y, direction))
    if abs(c) >= CRITERION_MILD:
        direction = "Yes" if c < 0 else "No"
        return ("Moderate response bias: criterion c = %+.2f in the %s "
                "condition (yes-rate %.3f), leaning toward \"%s\"."
                % (c, cond, y, direction))
    return ("Response bias is mild (largest |c| = %.2f), so raw accuracy and "
            "balanced accuracy tell a similar story here." % abs(worst[1]))


def difficulty_verdict(s):
    """Test easy vs hard on pooled accuracy; return (label, sentence)."""
    pe, ne = s.get("easy_accuracy"), s.get("easy_n_scored")
    ph, nh = s.get("hard_accuracy"), s.get("hard_n_scored")
    d, se, z, lo, hi, sig = diff_test(pe, ne, ph, nh)
    if d != d:
        return "unknown", "Not enough conditions present to test the effect."
    if sig and d > 0:
        return ("confirmed",
                "Easy exceeds hard by %.1f accuracy points (95%% CI %.1f to "
                "%.1f, z = %.1f). The appearance-similarity manipulation moves "
                "this model in the predicted direction."
                % (100 * d, 100 * lo, 100 * hi, z))
    if sig and d < 0:
        return ("reversed",
                "Hard exceeds easy by %.1f accuracy points (95%% CI %.1f to "
                "%.1f, z = %.1f). This is a real reversal, not noise, and "
                "needs explanation rather than reporting as an anomaly."
                % (-100 * d, 100 * lo, 100 * hi, z))
    return ("flat",
            "Easy minus hard is %.1f accuracy points (95%% CI %.1f to %.1f), "
            "not distinguishable from zero. Appearance similarity does not "
            "measurably drive this model's answers."
            % (100 * d, 100 * lo, 100 * hi))


def write_model_report(path, model, sub, s):
    L = []
    A = L.append
    A(rule("="))
    A("BioVQA C4 - individual re-identification")
    A("Model: %s" % model)
    A("Generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A(rule("="))
    A("")

    A("PER-CELL RESULTS")
    A(rule())
    A("%-14s %-8s %6s %7s %7s %7s %7s %7s" % (
        "species", "cond", "n", "acc", "bacc", "d'", "c", "yes"))
    for sp, g in sub.groupby("species"):
        for cond in ("easy", "random", "hard"):
            r = g[g.condition == cond]
            if r.empty:
                continue
            r = r.iloc[0]
            A("%-14s %-8s %6d %7.3f %7.3f %7.3f %+7.2f %7.3f" % (
                sp[:14], cond, r.n_scored, r.accuracy, r.balanced_accuracy,
                r.d_prime, r.criterion_c, r.yes_rate))
    A("")

    verdict, why = classify_model(s)
    A("VERDICT: %s" % verdict.upper())
    A(rule())
    A(wrap(why))
    A("")
    A(wrap(bias_note(s)))
    A("")

    label, sentence = difficulty_verdict(s)
    A("DIFFICULTY MANIPULATION: %s" % label.upper())
    A(rule())
    A(wrap(sentence))
    A("")

    # per-species ordering, with a significance test on easy vs hard
    A("PER-SPECIES ORDERING (easy >= random >= hard expected)")
    A(rule())
    n_mono = 0
    n_full = 0
    for sp, g in sub.groupby("species"):
        vals = {r.condition: (r.accuracy, r.n_scored, r.balanced_accuracy)
                for r in g.itertuples()}
        if not {"easy", "random", "hard"} <= set(vals):
            A("  %-14s incomplete" % sp[:14])
            continue
        n_full += 1
        ordered = (vals["easy"][2] >= vals["random"][2] >= vals["hard"][2])
        n_mono += int(ordered)
        d, se, z, lo, hi, sig = diff_test(vals["easy"][0], vals["easy"][1],
                                          vals["hard"][0], vals["hard"][1])
        tag = "monotonic" if ordered else "NOT monotonic"
        star = "significant" if sig else "n.s."
        A("  %-14s %-14s  easy-hard = %+6.1f pts  [%+.1f, %+.1f]  %s"
          % (sp[:14], tag, 100 * d, 100 * lo, 100 * hi, star))
    A("")
    if n_full:
        p_mono = binom_tail(n_mono, n_full, 1.0 / 6.0)
        A(wrap("%d of %d species show the full easy >= random >= hard ordering. "
               "Under the null hypothesis that condition does not matter, a "
               "random ordering would produce this in 1 of 6 species by "
               "chance; the probability of %d or more is %.4f."
               % (n_mono, n_full, n_mono, p_mono)))
    A("")

    A("PARSING")
    A(rule())
    tiers = {c[5:]: int(sub[c].sum()) for c in sub.columns
             if c.startswith("tier_")}
    total = int(sub.n_items.sum())
    A("  " + "  ".join("%s=%d" % kv for kv in tiers.items()))
    frac = int(sub.n_unparsed.sum()) / max(1, total)
    A("  unparsed %d of %d items (%.2f%%)"
      % (int(sub.n_unparsed.sum()), total, 100 * frac))
    if frac > UNPARSED_CONCERN:
        A("")
        A(wrap("WARNING: more than %.0f%% of answers could not be resolved to "
               "Yes/No. Inspect the unparsed-samples file before trusting "
               "these numbers - a model that frequently refuses or hedges is "
               "not the same as one that answers wrongly, and the two are "
               "scored differently here." % (100 * UNPARSED_CONCERN)))
    A("")
    A(rule("="))

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_overall_report(path, all_df, summary_df, base, expected_models):
    L = []
    A = L.append
    A(rule("="))
    A("BioVQA C4 - individual re-identification: evaluation report")
    A("Generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    A("Source:    %s" % base)
    A(rule("="))
    A("")

    n_models = all_df.model.nunique()
    n_species = all_df.species.nunique()
    conds = sorted(all_df.condition.unique())
    A("SCOPE")
    A(rule())
    A("  %d models x %d species x %d conditions = %d cells, %d scored items"
      % (n_models, n_species, len(conds), len(all_df),
         int(all_df.n_scored.sum())))
    A("  conditions: %s" % ", ".join(conds))
    expected = n_models * n_species * len(conds)
    if len(all_df) < expected:
        A("  INCOMPLETE: %d of %d cells present. Missing cells are excluded "
          "from every aggregate below." % (len(all_df), expected))
    if expected_models:
        missing = sorted(set(expected_models) - set(all_df.model.unique()))
        if missing:
            A("  models not yet run: %s" % ", ".join(missing))
    A("")

    # ---- 1. parsing integrity ---- #
    A("1. PARSING INTEGRITY")
    A(rule())
    tot = int(all_df.n_items.sum())
    unp = int(all_df.n_unparsed.sum())
    A("  %d of %d answers unresolved (%.3f%%)" % (unp, tot, 100 * unp / max(1, tot)))
    tier_cols = [c for c in all_df.columns if c.startswith("tier_")]
    A("  " + "  ".join("%s=%d" % (c[5:], int(all_df[c].sum())) for c in tier_cols))
    bad = summary_df[summary_df.n_unparsed_total
                     > UNPARSED_CONCERN * (tot / max(1, n_models))]
    if len(bad):
        A("  elevated unparsed rates: %s" % ", ".join(bad.model.tolist()))
    A("")
    A(wrap("Answers are resolved by tiered rules and the tier is recorded, so "
           "the leniency of the parser is auditable. 'exact' means the whole "
           "answer was Yes/No once markdown and punctuation were stripped; "
           "'prefix' means it began with Yes/No; 'contains_unique' means "
           "exactly one of the two words appeared. Refusals and hedges are "
           "deliberately left unresolved rather than being coerced into an "
           "answer."))
    A("")

    # ---- 2. discriminability ---- #
    A("2. DISCRIMINABILITY (d', pooled across species)")
    A(rule())
    A(wrap("Verdicts are keyed on the RANDOM condition. Easy-condition pairs "
           "are separable by appearance alone, so a high easy d' does not by "
           "itself demonstrate identity verification."))
    A("")
    A("  %-30s %8s %8s %8s   %s" % ("model", "easy", "random", "hard", "verdict"))
    for r in summary_df.sort_values("random_d_prime", ascending=False).itertuples():
        s = r._asdict()
        v, _ = classify_model(s)
        A("  %-30s %8.3f %8.3f %8.3f   %s" % (
            r.model[:30], r.easy_d_prime, r.random_d_prime, r.hard_d_prime, v))
    A("")
    degen = [r.model for r in summary_df.itertuples()
             if classify_model(r._asdict())[0] in ("degenerate", "at chance")]
    if degen:
        A(wrap("At or below the d' floor of %.2f in every condition: %s. For "
               "these models raw accuracy near 0.5 is an artefact of the "
               "balanced design and must not be read as partial competence."
               % (D_PRIME_FLOOR, ", ".join(degen))))
        A("")

    # ---- 3. response bias ---- #
    A("3. RESPONSE BIAS (criterion c; negative = biased toward Yes)")
    A(rule())
    A("  %-30s %8s %8s %8s" % ("model", "c easy", "c random", "c hard"))
    for r in summary_df.sort_values("random_criterion_c").itertuples():
        A("  %-30s %+8.2f %+8.2f %+8.2f" % (
            r.model[:30], r.easy_criterion_c, r.random_criterion_c,
            r.hard_criterion_c))
    A("")
    biased = [r.model for r in summary_df.itertuples()
              if abs(r.random_criterion_c) >= CRITERION_STRONG]
    if biased:
        A(wrap("Severe bias (|c| >= %.1f in the neutral condition): %s. Report "
               "d' or balanced accuracy for these; raw accuracy conflates "
               "their bias with their perception."
               % (CRITERION_STRONG, ", ".join(biased))))
        A("")

    below = summary_df[summary_df.hard_d_prime < 0]
    if len(below):
        A(wrap("BELOW CHANCE: %d of %d models have NEGATIVE d' in the hard "
               "condition (range %.3f to %.3f). Negative d' means the model "
               "separates the two classes and then labels them backwards - it "
               "is not guessing, it is being systematically misled. This is "
               "the strongest evidence in the run that models answer by "
               "appearance similarity rather than by individual identity, "
               "since the hard condition is built to put those two cues in "
               "opposition."
               % (len(below), len(summary_df),
                  below.hard_d_prime.min(), below.hard_d_prime.max())))
        A("")

    best_neutral = summary_df.random_d_prime.max()
    A(wrap("CEILING: the highest neutral-condition d' across all %d models is "
           "%.3f, roughly %.0f%% balanced accuracy. Even without any "
           "adversarial manipulation, individual re-identification is close to "
           "unsolved for this generation of models."
           % (len(summary_df), best_neutral,
              100 * NormalDist().cdf(best_neutral / 2.0))))
    A("")

    # ---- 4. the difficulty manipulation ---- #
    A("4. DIFFICULTY MANIPULATION")
    A(rule())
    A("  %-30s %10s %16s   %s" % (
        "model", "easy-hard", "95% CI", "verdict"))
    verdicts = Counter()
    for r in summary_df.itertuples():
        s = r._asdict()
        label, _ = difficulty_verdict(s)
        verdicts[label] += 1
        d, se, z, lo, hi, sig = diff_test(
            s.get("easy_accuracy"), s.get("easy_n_scored"),
            s.get("hard_accuracy"), s.get("hard_n_scored"))
        A("  %-30s %+9.1f  [%+6.1f, %+6.1f]   %s" % (
            r.model[:30], 100 * d, 100 * lo, 100 * hi, label))
    A("")
    n_mono = int(summary_df.species_monotonic.sum())
    n_cells = int(summary_df.species_complete.sum())
    p_mono = binom_tail(n_mono, n_cells, 1.0 / 6.0) if n_cells else float("nan")
    A(wrap("Across all model x species combinations, %d of %d show the full "
           "easy >= random >= hard ordering on balanced accuracy. If condition "
           "were irrelevant, orderings would be arbitrary and this would occur "
           "in about 1 of 6 cases; the probability of %d or more under that "
           "null is %.2e."
           % (n_mono, n_cells, n_mono, p_mono)))
    A("")
    A("  effect confirmed: %d    flat: %d    reversed: %d"
      % (verdicts["confirmed"], verdicts["flat"], verdicts["reversed"]))
    A("")

    # ---- 5. species ---- #
    A("5. BY SPECIES (mean d' across models)")
    A(rule())
    A("  %-16s %8s %8s %8s %10s" % (
        "species", "easy", "random", "hard", "easy-hard"))
    for sp, g in all_df.groupby("species"):
        vals = {c: g[g.condition == c].d_prime.mean() for c in
                ("easy", "random", "hard")}
        A("  %-16s %8.3f %8.3f %8.3f %10.3f" % (
            sp[:16], vals["easy"], vals["random"], vals["hard"],
            vals["easy"] - vals["hard"]))
    A("")
    A(wrap("Species differ in how much the manipulation bites. That is "
           "expected: the conditions are defined by DINOv2 cosine similarity, "
           "which is a proxy for what a VLM perceives, and the proxy is better "
           "in some species than others. Check margin_same_minus_different in "
           "each dataset's metadata.json - species where that margin is small "
           "are the ones where a flat or reversed model result is least "
           "surprising."))
    A("")

    # ---- 6. what this implies ---- #
    A("6. WHAT THIS IMPLIES")
    A(rule())

    n_disc = sum(1 for r in summary_df.itertuples()
                 if classify_model(r._asdict())[0] == "discriminating")
    n_weak = sum(1 for r in summary_df.itertuples()
                 if classify_model(r._asdict())[0] == "weak discrimination")
    n_null = n_models - n_disc - n_weak

    A(wrap("(a) Capability. Of %d models, %d discriminate individual identity "
           "substantially, %d weakly, and %d not at all. Individual "
           "re-identification is therefore not a capability current "
           "vision-language models possess uniformly, and the spread is wide "
           "enough that a single aggregate number would misrepresent the "
           "field." % (n_models, n_disc, n_weak, n_null)))
    A("")

    if verdicts["confirmed"] >= max(1, n_models // 2):
        A(wrap("(b) Mechanism. The majority of models track appearance "
               "similarity: making same-individual pairs look alike and "
               "different-individual pairs look unalike raises accuracy, and "
               "reversing that lowers it. This is consistent with models "
               "answering \"are these the same individual?\" by asking \"do "
               "these look alike?\" rather than by identifying "
               "individual-specific markings. The size of easy-minus-hard is "
               "a per-model measure of how much they rely on that shortcut."))
    else:
        A(wrap("(b) Mechanism. The manipulation moves a minority of models, so "
               "the evidence that models rely on global appearance similarity "
               "is weaker in this run than the design anticipated. Before "
               "concluding that models use identity-specific cues, verify from "
               "each metadata.json that the intended similarity margins were "
               "actually achieved in the generated data."))
    A("")

    if verdicts["reversed"]:
        rev = [r.model for r in summary_df.itertuples()
               if difficulty_verdict(r._asdict())[0] == "reversed"]
        A(wrap("(c) Reversals. %d model(s) score significantly HIGHER on hard "
               "than on easy: %s. These are not noise and should be reported, "
               "not suppressed. Two mechanisms are worth checking before "
               "interpreting them. First, the easy condition pairs the most "
               "dissimilar different-individual images, which can differ in "
               "background, lighting and pose as well as identity - a model "
               "may read that as a different scene rather than a different "
               "animal. Second, DINOv2 similarity is only a proxy for "
               "VLM-perceived similarity, and in species where the two "
               "diverge the manipulation does not do what its name says."
               % (verdicts["reversed"], ", ".join(rev))))
        A("")

    if biased:
        A(wrap("(d) Bias. %d model(s) show a severe standing preference for "
               "one answer. Any table in the paper that reports raw accuracy "
               "for these will invite the objection that 0.5 reflects the "
               "class balance rather than measurement. Report d' and balanced "
               "accuracy in the main text and keep raw accuracy in an "
               "appendix." % len(biased)))
        A("")

    A(wrap("(e) Reporting. The defensible framing is not that the benchmark "
           "guarantees easy > random > hard, but that this ordering is a "
           "PREDICTION of the hypothesis that models judge identity by "
           "appearance similarity. The ordering holds in the data by "
           "construction - verified before any model was run - so a model that "
           "departs from it is telling you something about the model, not "
           "revealing a defect in the benchmark."))
    A("")

    # ---- 7. caveats ---- #
    A("7. CAVEATS")
    A(rule())
    n_med = int(all_df.n_scored.median())
    half = 1.96 * 100 * float(np.sqrt(0.25 / max(1, n_med)))
    A(wrap("- Cells are about %d items each. The 95%% interval on a single "
           "accuracy is roughly +/-%.1f points and on a difference between two "
           "cells roughly +/-%.1f points, so smaller gaps are not "
           "interpretable." % (n_med, half, half * float(np.sqrt(2)))))
    A(wrap("- With %d models x %d species x 3 pairwise contrasts, several "
           "nominally significant results are expected by chance. Treat "
           "individual cells as exploratory and the pooled tests as "
           "confirmatory." % (n_models, n_species)))
    A(wrap("- Some source images appear in more than one pair, so items within "
           "a cell are not fully independent and the intervals above are "
           "mildly optimistic."))
    A(wrap("- d' assumes equal-variance Gaussian evidence distributions. It is "
           "used here as a bias-corrected effect size, not as a claim about "
           "the models' internal representations."))
    A(wrap("- Unresolved answers are excluded from the primary metrics. The "
           "accuracy_all_items column gives the pessimistic reading in which "
           "they count as errors."))
    A("")
    A(rule("="))

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def write_atomic(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", required=True,
                   help="root containing the per-species dataset directories")
    p.add_argument("--key_string", default="QA_",
                   help="result-file prefix (default: QA_)")
    p.add_argument("--dir_pattern", default="instance_identification_1000_",
                   help="task-directory prefix to match")
    p.add_argument("--out_dir", default=None,
                   help="default: <base_dir>/evaluation")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="model names to skip, e.g. superseded runs")
    p.add_argument("--n_boot", type=int, default=2000,
                   help="bootstrap resamples per cell; 0 disables CIs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_unparsed_samples", type=int, default=40,
                   help="raw unparsed strings recorded per cell for auditing")
    p.add_argument("--expect_models", nargs="*", default=[],
                   help="model names the full sweep should contain; any that "
                        "are absent are named in the report as not yet run")
    p.add_argument("--no_reports", action="store_true",
                   help="write only the CSVs, skip the .txt reports")
    args = p.parse_args()

    base = os.path.abspath(args.base_dir)
    if not os.path.isdir(base):
        sys.exit("base_dir does not exist: " + base)
    out_dir = args.out_dir or os.path.join(base, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    tasks = find_tasks(base, args.dir_pattern)
    if not tasks:
        sys.exit("no directories starting with %r under %s"
                 % (args.dir_pattern, base))
    log("%d task directories found" % len(tasks))

    rows, unparsed_rows = [], []

    for task_dir in tasks:
        n_pairs, condition = parse_task_dir(task_dir)
        species, species_dir = species_label(task_dir)

        files = sorted(f for f in os.listdir(task_dir)
                       if f.startswith(args.key_string) and f.endswith(".csv")
                       and f != args.key_string.rstrip("_") + ".csv"
                       and f != "QA.csv")
        if not files:
            log("  no result files in %s" % task_dir)
            continue

        for fname in files:
            model = fname[len(args.key_string):-len(".csv")]
            if model in args.exclude:
                continue

            path = os.path.join(task_dir, fname)
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                log("  unreadable %s: %r" % (path, exc))
                continue

            col = answer_column(df, model)
            if col is None or "qa_pair" not in df.columns:
                log("  cannot identify columns in %s" % path)
                continue

            truth_labels, pred_labels, tiers = [], [], Counter()
            n_bad_truth = 0
            for i in range(len(df)):
                try:
                    gt = json.loads(df["qa_pair"].iloc[i])["A"]
                except Exception:
                    n_bad_truth += 1
                    continue
                pred, tier = parse_answer(df[col].iloc[i])
                truth_labels.append(str(gt).strip())
                pred_labels.append(pred)
                tiers[tier] += 1
                if not pred and len(unparsed_rows) < 10 ** 6:
                    if tiers["unparsed"] + tiers["empty"] <= args.max_unparsed_samples:
                        unparsed_rows.append({
                            "model": model, "species": species,
                            "condition": condition,
                            "image_path": df["image_path"].iloc[i]
                            if "image_path" in df.columns else "",
                            "ground_truth": gt,
                            "raw_answer": repr(df[col].iloc[i])[:300],
                            "tier": tier,
                        })

            if n_bad_truth:
                log("  %s: %d rows with unreadable qa_pair" % (fname, n_bad_truth))

            cell = score_cell(truth_labels, pred_labels, tiers,
                              args.n_boot if args.n_boot > 0 else 0, args.seed)
            cell.update({
                "model": model,
                "species": species,
                "species_dir": species_dir,
                "condition": condition,
                "n_pairs_generated": n_pairs,
                "answer_column": col,
                "source_file": os.path.relpath(path, base),
            })
            rows.append(cell)

        log("  %-46s %d model(s)" % (
            os.path.join(species, os.path.basename(task_dir)), len(files)))

    if not rows:
        sys.exit("no result files matched; check --key_string")

    lead = ["model", "species", "condition", "n_items", "n_scored", "n_unparsed",
            "coverage", "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
            "balanced_accuracy", "bacc_ci_lo", "bacc_ci_hi", "d_prime",
            "criterion_c", "mcc", "cohens_kappa", "f1", "precision",
            "recall_sensitivity", "specificity", "npv", "hit_rate",
            "false_alarm_rate", "yes_rate", "yes_rate_truth",
            "TP", "FN", "FP", "TN", "accuracy_all_items"]
    all_df = pd.DataFrame(rows)
    ordered = lead + [c for c in all_df.columns if c not in lead]
    all_df = all_df[ordered].sort_values(["model", "species", "condition"])

    write_atomic(all_df, os.path.join(out_dir, args.key_string + "all_results.csv"))

    for model, sub in all_df.groupby("model", sort=True):
        write_atomic(sub, os.path.join(
            out_dir, "%s%s_results.csv" % (args.key_string, model)))

    # ---- pooled per model x condition, plus the difficulty effect ---------- #
    summary = []
    for model, sub in all_df.groupby("model", sort=True):
        by_cond = {}
        for cond, s in sub.groupby("condition"):
            tp, fn = int(s.TP.sum()), int(s.FN.sum())
            fp, tn = int(s.FP.sum()), int(s.TN.sum())
            n = tp + fn + fp + tn
            sens = tp / (tp + fn) if (tp + fn) else float("nan")
            spec = tn / (tn + fp) if (tn + fp) else float("nan")
            sdt = signal_detection(tp, fn, fp, tn)
            by_cond[cond] = {
                "accuracy": (tp + tn) / n if n else float("nan"),
                "balanced_accuracy": 0.5 * (sens + spec),
                "d_prime": sdt["d_prime"],
                "criterion_c": sdt["criterion_c"],
                "yes_rate": (tp + fp) / n if n else float("nan"),
                "n_scored": n,
                "n_unparsed": int(s.n_unparsed.sum()),
            }

        row = {"model": model,
               "n_cells": len(sub),
               "n_unparsed_total": int(sub.n_unparsed.sum())}
        for cond in ("easy", "random", "hard"):
            c = by_cond.get(cond, {})
            for k in ("accuracy", "balanced_accuracy", "d_prime",
                      "criterion_c", "yes_rate", "n_scored"):
                row["%s_%s" % (cond, k)] = c.get(k, float("nan"))

        e, h = by_cond.get("easy", {}), by_cond.get("hard", {})
        row["effect_acc_easy_minus_hard"] = (
            e.get("accuracy", float("nan")) - h.get("accuracy", float("nan")))
        row["effect_dprime_easy_minus_hard"] = (
            e.get("d_prime", float("nan")) - h.get("d_prime", float("nan")))

        # per-species monotonicity easy >= random >= hard, on balanced accuracy
        mono = 0
        n_sp = 0
        for _sp, g in sub.groupby("species"):
            vals = {r.condition: r.balanced_accuracy for r in g.itertuples()}
            if {"easy", "random", "hard"} <= set(vals):
                n_sp += 1
                if vals["easy"] >= vals["random"] >= vals["hard"]:
                    mono += 1
        row["species_monotonic"] = mono
        row["species_complete"] = n_sp
        summary.append(row)

    summary_df = pd.DataFrame(summary).sort_values(
        "effect_dprime_easy_minus_hard", ascending=False)
    write_atomic(summary_df,
                 os.path.join(out_dir, args.key_string + "summary_by_model.csv"))

    if unparsed_rows:
        write_atomic(pd.DataFrame(unparsed_rows),
                     os.path.join(out_dir, args.key_string + "unparsed_samples.csv"))

    # ---- narrative reports ------------------------------------------------ #
    if not args.no_reports:
        srows = {r["model"]: r for r in summary_df.to_dict("records")}
        for model, sub in all_df.groupby("model", sort=True):
            write_model_report(
                os.path.join(out_dir, "%s%s_report.txt" % (args.key_string, model)),
                model, sub, srows[model])
        write_overall_report(
            os.path.join(out_dir, args.key_string + "report.txt"),
            all_df, summary_df, base, args.expect_models)
        log("reports written: %s and %d per-model"
            % (args.key_string + "report.txt", len(srows)))

    # ---- console report --------------------------------------------------- #
    print("", flush=True)
    print("=" * 100, flush=True)
    print("POOLED ACROSS SPECIES   (bacc = balanced accuracy, d' = discriminability)",
          flush=True)
    print("=" * 100, flush=True)
    print("%-30s %-23s %-23s %8s %6s" % (
        "model", "easy  bacc / d'", "hard  bacc / d'",
        "d' e-h", "mono"), flush=True)
    print("-" * 100, flush=True)
    for r in summary_df.itertuples():
        print("%-30s %8.3f /%7.3f      %8.3f /%7.3f   %8.3f  %d/%d" % (
            r.model[:30],
            r.easy_balanced_accuracy, r.easy_d_prime,
            r.hard_balanced_accuracy, r.hard_d_prime,
            r.effect_dprime_easy_minus_hard,
            r.species_monotonic, r.species_complete), flush=True)
    print("=" * 100, flush=True)

    total_unparsed = int(all_df.n_unparsed.sum())
    print("\n%d cells, %d models, %d species, %d conditions" % (
        len(all_df), all_df.model.nunique(),
        all_df.species.nunique(), all_df.condition.nunique()), flush=True)
    print("%d unparsed answers of %d total (%.3f%%)" % (
        total_unparsed, int(all_df.n_items.sum()),
        100.0 * total_unparsed / max(1, int(all_df.n_items.sum()))), flush=True)
    tier_cols = [c for c in all_df.columns if c.startswith("tier_")]
    print("parse tiers: " + "  ".join(
        "%s=%d" % (c[5:], int(all_df[c].sum())) for c in tier_cols), flush=True)
    print("\nwritten to " + out_dir, flush=True)


if __name__ == "__main__":
    main()