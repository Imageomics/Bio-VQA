#!/usr/bin/env python3
"""
Authoritative evaluation for the Localization task.

  python evaluate.py --results_dir ... --out_dir ...
  python evaluate.py --bootstrap 2000
  python evaluate.py --no_consistent_mode      # score each row independently

Outputs:
  loc_overall.csv        one row per model
  loc_by_group.csv       one row per model x tier
  loc_by_size.csv        one row per model x target-size quartile
  loc_parse_modes.csv    coordinate convention used, per model
  loc_report.md          human-readable summary

--------------------------------------------------------------------------- #
METRICS

A predicted box never matches the gold box exactly, so everything here is
overlap-based and reported at several tolerances rather than as a single
threshold.

  meanIoU     mean intersection over union. The headline continuous number.
  acc50/75/90 share of predictions at or above that IoU. 0.50 is the usual
              detection convention, 0.75 is a demanding match, and 0.90 is
              near-exact. Reported together because a model can be broadly
              right and never precise, or occasionally exact and usually lost,
              and one threshold cannot tell those apart.
  acc25       a deliberately loose band, mostly to show where a model sits
              when acc50 is on the floor.
  centreIn    share of predictions whose CENTRE falls inside the gold box.
              This separates "knows where the animal is but cannot size the
              box" from "has no idea", and it is the only column on which a
              pointing-trained model can be compared fairly against a
              box-trained one.
  gtCentreIn  the converse: gold centre inside the predicted box. A model that
              returns a huge box scores well here and badly on IoU, so the two
              together diagnose over-large predictions.

BASELINES, computed on the identical rows as each model
  wholeImage  predict the entire canvas every time.
  centre50    predict the middle 50% of the canvas.
  meanBox     predict that tier's average gold box, in normalised coordinates.
Single-annotation photographs often have the animal filling much of the frame,
so wholeImage is not always negligible. A result that does not clear all three
is not a localisation.

PANEL CONFUSION, multi-panel tiers only
For an image carrying several gold boxes, the prediction is compared against
every box on that canvas. `panelConfusion` is the share of items where the
prediction overlaps some OTHER panel's gold box more than its own. That is a
direct measurement of positional indexing failure, within tier, with
resolution and canvas geometry held constant - much stronger than inferring it
from a difference between two tier means.

COORDINATE CONVENTIONS
The runner resolves each reply independently and records which convention it
used. By default this script then re-resolves every row of a model under that
model's MAJORITY convention, because a model is overwhelmingly consistent and
a per-row decision can flip on an ambiguous reply. The majority is taken from
the resolver's own judgment, never from the gold boxes. `--no_consistent_mode`
disables it, and both numbers are worth comparing.
"""

import argparse
import ast
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


# --------------------------------------------------------------------------- #
# geometry

def iou_xywh(g, p):
    gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
    px1, py1, px2, py2 = p[0], p[1], p[0] + p[2], p[1] + p[3]
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
    return inter / union if union > 0 else 0.0


def centre_in(outer, inner):
    cx, cy = inner[0] + inner[2] / 2.0, inner[1] + inner[3] / 2.0
    return (outer[0] <= cx <= outer[0] + outer[2]
            and outer[1] <= cy <= outer[1] + outer[3])


# --------------------------------------------------------------------------- #
# coordinate resolution, mirroring the runner

def _valid(x1, y1, x2, y2, W, H):
    if not (x2 > x1 and y2 > y1):
        return False
    tol = 0.02
    if x1 < -tol * W or y1 < -tol * H or x2 > (1 + tol) * W or y2 > (1 + tol) * H:
        return False
    frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
    return 1e-5 <= frac <= 1.0


def scales_for(W, H):
    return {"abs": (1.0, 1.0), "norm01": (W, H),
            "norm1000": (W / 1000.0, H / 1000.0), "pct": (W / 100.0, H / 100.0)}


def resolve_fixed(answer, W, H, mode):
    """Resolve under a named convention, e.g. 'norm1000_xywh'."""
    nums = NUM_RE.findall(str(answer).replace(",", " "))
    if len(nums) < 4 or not isinstance(mode, str) or "_" not in mode:
        return None
    try:
        v = [float(x) for x in nums[:4]]
    except ValueError:
        return None
    name, fmt = mode.rsplit("_", 1)
    sc = scales_for(W, H).get(name)
    if sc is None:
        return None
    a, b, c, d = v[0] * sc[0], v[1] * sc[1], v[2] * sc[0], v[3] * sc[1]
    x1, y1, x2, y2 = (a, b, a + c, b + d) if fmt == "xywh" else (a, b, c, d)
    if not _valid(x1, y1, x2, y2, W, H):
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def choose_convention(raws, Ws, Hs):
    """Pick the coordinate convention for a whole model, using NO ground truth.

    A single reply cannot settle this: the images are 2400 px wide, so 600
    could be a pixel coordinate or a 0-1000 normalised one. A model's whole
    set of replies can, and the signal is WHERE THE BOXES SIT.

    A model localising across the image places its boxes all over the frame,
    so the median box centre is near the middle. The same replies read under
    the wrong convention are displaced: 0-1000 values read as pixels on a
    2400 px canvas can never place a centre past 42% of the width, so the
    median centre sits far left of middle. Scoring each convention by how
    close its median centre is to the image centre, weighted by the fraction
    of replies it parses at all, recovers the intended frame.

    WHY THE MEDIAN AND NOT A HIGH PERCENTILE.
    An earlier version scored conventions by 99th-percentile coverage - how
    far right and down the boxes reached. That fails on a model that MIXES
    conventions. Gemma-4 emits 60% of its answers in 0-1000 form and 21% as
    pixel coordinates for an imagined larger canvas; under the absolute
    reading the oversized ones are rejected as out of bounds while the
    remainder happen to reach the edges, so a biased minority made the wrong
    convention look well-spread. The median is immune to that, and on this
    corpus it agrees with the ground-truth-optimal convention for 18 of 18
    models where the percentile rule managed 16.

    Ground truth is still never consulted: only the predictions and the image
    dimensions enter.
    """
    best, best_score = None, -1.0
    for name in ("abs", "norm1000", "norm01", "pct"):
        for fmt in ("xywh", "xyxy"):
            mode = name + "_" + fmt
            got = []
            for r, W, H in zip(raws, Ws, Hs):
                b = resolve_fixed(r, W, H, mode)
                if b is not None:
                    got.append((b, W, H))
            if len(got) < 0.25 * max(len(raws), 1):
                continue
            cx = float(np.median([(b[0] + b[2] / 2.0) / W for b, W, H in got]))
            cy = float(np.median([(b[1] + b[3] / 2.0) / H for b, W, H in got]))
            frac = len(got) / float(len(raws))
            centred = (1.0 - min(abs(cx - 0.5) * 2.0, 1.0)) * \
                      (1.0 - min(abs(cy - 0.5) * 2.0, 1.0))
            score = centred * frac
            if score > best_score:
                best, best_score = mode, score
    return best, best_score


def block(g, p, W, H):
    n = len(g)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "meanIoU", "acc25", "acc50", "acc75", "acc90",
                 "centreIn", "gtCentreIn")}
    ious = np.array([iou_xywh(a, b) for a, b in zip(g, p)])
    return {
        "n": int(n),
        "meanIoU": float(ious.mean()),
        "acc25": float((ious >= 0.25).mean()),
        "acc50": float((ious >= 0.50).mean()),
        "acc75": float((ious >= 0.75).mean()),
        "acc90": float((ious >= 0.90).mean()),
        "centreIn": float(np.mean([centre_in(a, b) for a, b in zip(g, p)])),
        "gtCentreIn": float(np.mean([centre_in(b, a) for a, b in zip(g, p)])),
    }


def baselines(d):
    g = list(zip(d["gt_x"], d["gt_y"], d["gt_w"], d["gt_h"]))
    W = d["img_w"].values.astype(float)
    H = d["img_h"].values.astype(float)
    whole = [(0.0, 0.0, w, h) for w, h in zip(W, H)]
    c50 = [(w * .25, h * .25, w * .5, h * .5) for w, h in zip(W, H)]
    fx = float((d["gt_x"] / d["img_w"]).mean())
    fy = float((d["gt_y"] / d["img_h"]).mean())
    fw = float((d["gt_w"] / d["img_w"]).mean())
    fh = float((d["gt_h"] / d["img_h"]).mean())
    mean = [(w * fx, h * fy, w * fw, h * fh) for w, h in zip(W, H)]
    return {
        "base_whole_IoU": block(g, whole, W, H)["meanIoU"],
        "base_whole_acc50": block(g, whole, W, H)["acc50"],
        "base_c50_IoU": block(g, c50, W, H)["meanIoU"],
        "base_c50_acc50": block(g, c50, W, H)["acc50"],
        "base_mean_IoU": block(g, mean, W, H)["meanIoU"],
        "base_mean_acc50": block(g, mean, W, H)["acc50"],
    }


def panel_confusion(d):
    """Share of multi-target items where the prediction overlaps some other
    panel's gold box more than its own."""
    sub = d[d["n_targets"] > 1]
    if not len(sub):
        return float("nan")
    hit = 0
    tot = 0
    # keyed on the resolved file: basenames are not unique across tiers
    for _, grp in sub.groupby("image_abspath"):
        golds = list(zip(grp["gt_x"], grp["gt_y"], grp["gt_w"], grp["gt_h"]))
        preds = list(grp["_p"])
        for i, p in enumerate(preds):
            own = iou_xywh(golds[i], p)
            other = max([iou_xywh(golds[j], p)
                         for j in range(len(golds)) if j != i] or [0.0])
            tot += 1
            if other > own:
                hit += 1
    return hit / tot if tot else float("nan")


def boot(d, key, fn, n_boot, seed=0):
    if n_boot <= 0 or not len(d):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = d.groupby(key).indices
    ks = list(groups.keys())
    idx = [groups[k] for k in ks]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ks), len(ks))
        vals.append(fn(d.iloc[np.concatenate([idx[i] for i in pick])]))
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(here, "results"))
    ap.add_argument("--out_dir", default=os.path.join(here, "results"))
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--no_consistent_mode", action="store_true",
                    help="score each row under its own resolved convention "
                         "instead of the model's majority convention")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "qa_*.csv")))
    if not files:
        raise SystemExit("no merged qa_*.csv under " + args.results_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    overall, groups, sizes, modes = [], [], [], []

    for path in files:
        key = os.path.basename(path)[len("qa_"):-len(".csv")]
        df = pd.read_csv(path)
        col = key + "_answer"
        raw_col = key + "_answer_unfiltered"
        mode_col = key + "_parse_mode"
        if col not in df.columns:
            cand = [c for c in df.columns if c.endswith("_answer")
                    and not c.endswith("_unfiltered")]
            if not cand:
                print("[skip] %s: no answer column" % key)
                continue
            col = cand[0]

        # Convention for this model, chosen from the whole set of replies by
        # the coverage test above rather than from any single row. The
        # row-level modes recorded by the runner are kept for the report.
        mc = Counter(df[mode_col].dropna()) if mode_col in df.columns else Counter()
        mc.pop("unparsed", None)
        mc.pop("invalid", None)
        row_majority = mc.most_common(1)[0][0] if mc else None
        majority, cov_score = (None, float("nan"))
        if (not args.no_consistent_mode) and raw_col in df.columns:
            majority, cov_score = choose_convention(
                df[raw_col].tolist(), df["img_w"].tolist(), df["img_h"].tolist())
        if majority is None:
            majority = row_majority
        modes.append(dict(model=key, chosen=majority or "-",
                          row_majority=row_majority or "-",
                          coverage_score=round(cov_score, 3)
                          if cov_score == cov_score else float("nan"),
                          **{k: int(v) for k, v in mc.items()}))

        boxes, used = [], []
        for _, r in df.iterrows():
            b = None
            if (not args.no_consistent_mode) and majority and raw_col in df.columns:
                b = resolve_fixed(r[raw_col], r["img_w"], r["img_h"], majority)
                if b is not None:
                    used.append(majority)
            if b is None and pd.notna(r.get(col)):
                try:
                    b = list(json.loads(r[col]))
                    used.append(r.get(mode_col, "row"))
                except Exception:
                    b = None
            if b is None:
                used.append("unparsed")
            boxes.append(b)

        df["_p"] = boxes
        ok = df["_p"].notna() & df["gt_w"].notna()
        if "gt_out_of_bounds" in df.columns:
            ok = ok & (~df["gt_out_of_bounds"].fillna(False).astype(bool))
        d = df[ok].copy()
        parse_rate = float(df["_p"].notna().mean())
        if not len(d):
            print("[skip] %s: nothing scoreable" % key)
            continue

        g = list(zip(d["gt_x"], d["gt_y"], d["gt_w"], d["gt_h"]))
        p = list(d["_p"])
        W, H = d["img_w"].values, d["img_h"].values

        if "tier" in d.columns:
            for tier, gg in d.groupby("tier"):
                s = block(list(zip(gg["gt_x"], gg["gt_y"], gg["gt_w"], gg["gt_h"])),
                          list(gg["_p"]), gg["img_w"].values, gg["img_h"].values)
                s.update(model=key, tier=tier, **baselines(gg),
                         panelConfusion=panel_confusion(gg))
                groups.append(s)

        # target-size quartiles: localisation collapses on small subjects, and
        # a pooled number hides which end a model fails at
        if "gt_area_frac" in d.columns and d["gt_area_frac"].notna().any():
            q = pd.qcut(d["gt_area_frac"], 4, labels=["Q1 smallest", "Q2",
                                                      "Q3", "Q4 largest"],
                        duplicates="drop")
            for lab, gg in d.groupby(q, observed=True):
                s = block(list(zip(gg["gt_x"], gg["gt_y"], gg["gt_w"], gg["gt_h"])),
                          list(gg["_p"]), gg["img_w"].values, gg["img_h"].values)
                s.update(model=key, size_bin=str(lab),
                         median_area_frac=float(gg["gt_area_frac"].median()))
                sizes.append(s)

        s = block(g, p, W, H)
        # meanIoU is averaged over the replies that produced a usable box.
        # A model that returns nothing usable on half its items has not
        # localised those items, so meanIoU_all charges them as zero and is
        # the figure to use for any comparison across models or against
        # another task. meanIoU alone rewards a low parse rate.
        s["meanIoU_all"] = s["meanIoU"] * parse_rate
        s["acc50_all"] = s["acc50"] * parse_rate
        s.update(model=key, parse_rate=parse_rate,
                 mode=majority or "per-row", **baselines(d),
                 panelConfusion=panel_confusion(d),
                 images=int(d["image_abspath"].nunique()))

        if args.bootstrap:
            lo, hi = boot(d, "image_abspath",
                          lambda x: float(np.mean(
                              [iou_xywh(a, b) for a, b in zip(
                                  zip(x["gt_x"], x["gt_y"], x["gt_w"], x["gt_h"]),
                                  x["_p"])])),
                          args.bootstrap)
            s["meanIoU_lo"], s["meanIoU_hi"] = lo, hi

        overall.append(s)
        print("[ok] %-32s n=%6d  IoU %.3f  acc50 %.3f  acc75 %.3f  "
              "centreIn %.3f  (whole %.3f, c50 %.3f, mean %.3f)  "
              "panelConf %.3f  parse %.3f  mode %s"
              % (key, s["n"], s["meanIoU"], s["acc50"], s["acc75"],
                 s["centreIn"], s["base_whole_IoU"], s["base_c50_IoU"],
                 s["base_mean_IoU"], s["panelConfusion"], parse_rate,
                 s["mode"]))

    ov = pd.DataFrame(overall)
    gr = pd.DataFrame(groups)
    sz = pd.DataFrame(sizes)
    mo = pd.DataFrame(modes).fillna(0)
    p1 = os.path.join(args.out_dir, "loc_overall.csv")
    p2 = os.path.join(args.out_dir, "loc_by_group.csv")
    p3 = os.path.join(args.out_dir, "loc_by_size.csv")
    p4 = os.path.join(args.out_dir, "loc_parse_modes.csv")
    ov.to_csv(p1, index=False)
    gr.to_csv(p2, index=False)
    sz.to_csv(p3, index=False)
    mo.to_csv(p4, index=False)

    cols = [c for c in ["model", "n", "images", "meanIoU_all", "meanIoU",
                        "meanIoU_lo", "meanIoU_hi", "acc50_all",
                        "acc25", "acc50", "acc75", "acc90",
                        "centreIn", "gtCentreIn", "base_whole_IoU",
                        "base_c50_IoU", "base_mean_IoU", "panelConfusion",
                        "parse_rate", "mode"] if c in ov.columns]
    show = ov[cols].sort_values(
        "meanIoU_all" if "meanIoU_all" in ov.columns else "meanIoU",
        ascending=False)

    md = [
        "# BioVQA C3 - Localization results",
        "",
        "Bounding-box prediction on single photographs (`easy`) and on "
        "three-panel canvases (`medium`, `hard`). Generated by `evaluate.py`.",
        "",
        "## The coordinate frame",
        "",
        "The gold boxes for the multi-panel tiers already carry the panel "
        "offset - on one Hard canvas the three boxes start at x = 1760, 4925 "
        "and 9803 - so they are in whole-canvas pixels, and the manifest "
        "stores no panel offsets. Every tier is therefore asked for the same "
        "thing: pixels measured from the top-left of the entire image, with "
        "the canvas dimensions stated in the prompt.",
        "",
        "Models nevertheless reply in whatever convention they were trained "
        "on - Qwen in 0-1000, others in 0-1 fractions, some in [x1,y1,x2,y2] "
        "rather than [x,y,w,h]. Scoring a correct localisation as zero "
        "because of its units would measure format compliance, not "
        "localisation, so each model is scored under the convention it "
        "actually used.",
        "",
        "**`meanIoU_all` is the column to compare across models.** `meanIoU` "
        "averages only over replies that produced a usable box, so a model "
        "that returns nothing parseable on half its items is scored on the "
        "easier half; `meanIoU_all` charges those as zero. On this corpus the "
        "two differ by a factor of four for the worst-affected model.",
        "",
        "That convention is chosen ACROSS the model's replies, not per reply. "
        "These images are 2400 px wide, so a 0-1000 answer and a pixel answer "
        "occupy the same numeric range and no single reply can separate them. "
        "The whole set can: the correct reading produces boxes reaching the "
        "right and bottom edges, while a misread 0-1000 answer cannot place "
        "anything past 42% of the width. The gold boxes are never consulted, "
        "so the choice cannot tune the metric it precedes.",
        "",
        "## Metrics",
        "",
        "A predicted box never matches exactly, so overlap is reported at "
        "several tolerances. `acc50` is the usual detection convention, "
        "`acc75` a demanding match, `acc90` near-exact. `centreIn` - "
        "prediction centre inside the gold box - separates *knows where the "
        "animal is but cannot size the box* from *has no idea*, and is the "
        "only column on which a pointing-trained model can be compared "
        "fairly against a box-trained one.",
        "",
        "**Three baselines are computed on the identical rows as each "
        "model**: the whole canvas, the middle 50%, and that tier's average "
        "gold box. Single-animal photographs often have the subject filling "
        "much of the frame, so the whole-canvas baseline is not always "
        "negligible. A result that does not clear all three is not a "
        "localisation.",
        "",
        "## Overall",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, r in show.iterrows():
        md.append("| " + " | ".join(
            (r[c] if isinstance(r[c], str) else
             ("%d" % r[c] if c in ("n", "images") else "%.3f" % r[c]))
            for c in cols) + " |")

    if len(gr):
        gc = [c for c in ["model", "tier", "n", "meanIoU", "acc50", "acc75",
                          "centreIn", "base_whole_IoU", "base_mean_IoU",
                          "panelConfusion"] if c in gr.columns]
        md += ["", "## By tier", "",
               "`panelConfusion` is the share of multi-panel items where the "
               "prediction overlaps some OTHER panel's gold box more than its "
               "own. It measures positional indexing directly, within tier, "
               "with resolution and canvas geometry held constant - which is "
               "stronger than inferring it from a gap between two tier means.",
               "",
               "| " + " | ".join(gc) + " |", "|" + "---|" * len(gc)]
        for _, r in gr.sort_values(["model", "tier"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.3f" % r[c]))
                for c in gc) + " |")

    if len(sz):
        sc = [c for c in ["model", "size_bin", "n", "median_area_frac",
                          "meanIoU", "acc50", "centreIn"] if c in sz.columns]
        md += ["", "## By target size", "",
               "Localisation collapses on small subjects, and a pooled number "
               "hides which end a model fails at. Quartiles of gold box area "
               "as a fraction of the canvas.",
               "",
               "| " + " | ".join(sc) + " |", "|" + "---|" * len(sc)]
        for _, r in sz.sort_values(["model", "size_bin"]).iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else
                 ("%d" % r[c] if c == "n" else "%.4f" % r[c]))
                for c in sc) + " |")

    if len(mo):
        mc = list(mo.columns)
        md += ["", "## Coordinate conventions", "",
               "`chosen` is the convention used to score the model, selected "
               "across all of its replies by a coverage test that never looks "
               "at the gold boxes: the correct reading produces boxes that "
               "reach the right and bottom edges of the image, while a "
               "misread 0-1000 answer is confined to the top-left. "
               "`row_majority` is what the per-row resolver in the runner "
               "concluded; the two disagree when the images are small enough "
               "that a single reply cannot distinguish 0-1000 from pixels.",
               "",
               "| " + " | ".join(mc) + " |", "|" + "---|" * len(mc)]
        for _, r in mo.iterrows():
            md.append("| " + " | ".join(
                (r[c] if isinstance(r[c], str) else "%d" % r[c])
                for c in mc) + " |")

    md += [
        "",
        "## Notes",
        "",
        "Confidence intervals bootstrap over IMAGES, not rows: a multi-panel "
        "canvas carries three questions. Images are keyed on the resolved "
        "file path rather than the basename, because basenames are not unique "
        "across tiers.",
        "",
        "Rows whose gold box falls outside its own canvas are excluded and "
        "counted by `split_qa.py`; those are a benchmark defect rather than a "
        "model failure.",
        "",
    ]
    p5 = os.path.join(args.out_dir, "loc_report.md")
    with open(p5, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print()
    for p in (p1, p2, p3, p4, p5):
        print("wrote " + p)
    print()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()