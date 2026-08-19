#!/usr/bin/env python3
"""
Relative-Position QA generation (BioVQA C3), distractor condition, Beetles-500.

Controlled specimen-tray setting: constant working distance and near-uniform
specimen scale, so that referent identification and direction labelling are
both well posed. Restricted to a single dataset deliberately — in field imagery
(drone/telephoto wildlife) targets are frequently a few dozen pixels across and
pairs mix a foreground subject with a background speck, which makes both the
"which box holds an animal" step and the centre-to-centre direction label
indefensible.

Every emitted item carries exactly six boxes:
    red    x2 : 1 on the target specimen, 1 on empty space
    blue   x2 : 1 on the target specimen, 1 on empty space
    green  x1 : on a non-target specimen
    cyan   x1 : on a non-target specimen

All six boxes are size-matched: the largest/smallest area ratio across the six
is bounded by --max_size_ratio, and empty boxes are sized from the median
annotated box in the image. Empty red/blue placement is solved so that each of
the three incorrect referent pairings yields a distinct listed wrong option.
Pairs that cannot satisfy the full six-box composition, or the size bound, are
rejected rather than emitted degraded.

Budget: 1000 pairs from 500 images, 2 per image, with fill passes if some
images yield fewer.
"""

import os
import re
import json
import math
import random
import argparse
import itertools
import statistics
from collections import Counter

import pandas as pd
from PIL import Image, ImageOps, ImageDraw

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

COLORS = {
    "red":   (225, 25, 25),
    "blue":  (30, 60, 220),
    "green": (0, 190, 60),
    "cyan":  (0, 220, 230),
}
NEUTRAL_COLORS = ["green", "cyan"]

MIN_ANIMALS = 4          # 2 targets + 2 neutral-coloured specimens
N_BOXES = 6              # constant, by construction

DIRS = ["right", "bottom-right", "bottom", "bottom-left",
        "left", "top-left", "top", "top-right"]

SCHEMAS = [
    ("xyxy",   ("x_min", "y_min", "x_max", "y_max")),
    ("xyxy",   ("xmin", "ymin", "xmax", "ymax")),
    ("cxcywh", ("x_center", "y_center", "width", "height")),
    ("xywh",   ("bbox_x", "bbox_y", "bbox_w", "bbox_h")),
    ("xywh",   ("x", "y", "w", "h")),
]

LABEL_COLS = ["class_name", "label", "species", "scientific_name", "category"]


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------

def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def half_diag(b):
    return 0.5 * math.hypot(b[2] - b[0], b[3] - b[1])


def direction_of(c1, c2, margin_deg):
    """Direction of c1 with respect to c2 (image coords, y grows down).
    None if within margin_deg of an octant boundary."""
    dx, dy = c1[0] - c2[0], c1[1] - c2[1]
    if dx == 0 and dy == 0:
        return None
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    idx = int(((ang + 22.5) % 360.0) // 45.0)
    delta = abs((ang - idx * 45.0 + 180.0) % 360.0 - 180.0)
    if delta > 22.5 - margin_deg:
        return None
    return DIRS[idx]


def opposite(d):
    return DIRS[(DIRS.index(d) + 4) % 8]


def neighbours(d):
    i = DIRS.index(d)
    return {DIRS[(i - 1) % 8], DIRS[(i + 1) % 8]}


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    return inter / float(area(a) + area(b) - inter + 1e-7)


def overlaps(a, b, mx=0.0, my=0.0):
    return not (a[2] + mx < b[0] or b[2] + mx < a[0] or
                a[3] + my < b[1] or b[3] + my < a[1])


def size_ratio(boxes):
    a = [area(b) for b in boxes]
    return max(a) / max(min(a), 1e-7)


# ----------------------------------------------------------------------------
# coordinate resolution
# ----------------------------------------------------------------------------

def box_from(kind, vals, W, H, norm):
    a, b, c, d = (float(v) for v in vals)
    if norm:
        a, c = a * W, c * W
        b, d = b * H, d * H
    if kind == "xyxy":
        return [a, b, c, d]
    if kind == "xywh":
        return [a, b, a + c, b + d]
    return [a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0]


def valid_box(bx, W, H, tol=0.02):
    if not all(math.isfinite(v) for v in bx):
        return False
    x1, y1, x2, y2 = bx
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return False
    if x1 < -tol * W or y1 < -tol * H or x2 > (1 + tol) * W or y2 > (1 + tol) * H:
        return False
    if (x2 - x1) * (y2 - y1) > 0.85 * W * H:
        return False
    return True


def image_sizes(path):
    with Image.open(path) as im:
        raw = im.size
        tr = ImageOps.exif_transpose(im).size
    return raw, tr


def calibrate(ds_path, df, rng, sample_imgs=120, sample_rows=1500):
    present = [(k, c) for k, c in SCHEMAS if all(col in df.columns for col in c)]
    if not present:
        return None, "no recognised bbox columns"

    imgs = df["img_id"].drop_duplicates().tolist()
    rng.shuffle(imgs)
    sizes = {}
    for iid in imgs:
        p = os.path.join(ds_path, "Images", iid)
        if not os.path.exists(p):
            continue
        try:
            sizes[iid] = image_sizes(p)
        except Exception:
            continue
        if len(sizes) >= sample_imgs:
            break
    if not sizes:
        return None, "no readable images under Images/"

    sub = df[df["img_id"].isin(sizes)]
    if len(sub) > sample_rows:
        sub = sub.sample(sample_rows, random_state=0)

    best = None
    for kind, cols in present:
        for norm in (True, False):
            for exif in (True, False):
                ok = tot = 0
                for _, r in sub.iterrows():
                    raw, tr = sizes[r["img_id"]]
                    W, H = tr if exif else raw
                    tot += 1
                    try:
                        b = box_from(kind, [r[c] for c in cols], W, H, norm)
                    except Exception:
                        continue
                    if valid_box(b, W, H):
                        ok += 1
                score = ok / max(tot, 1)
                if best is None or score > best["score"]:
                    best = {"kind": kind, "cols": cols, "norm": norm,
                            "exif": exif, "score": score}
    return best, ""


# ----------------------------------------------------------------------------
# dataset loading
# ----------------------------------------------------------------------------

def load_beetles(ds_path):
    csv_path = os.path.join(ds_path, "Detections", "Detections.csv")
    if not os.path.exists(csv_path):
        return None, "missing Detections/Detections.csv"
    raw = pd.read_csv(csv_path)
    rows = []
    for _, r in raw.iterrows():
        try:
            coords = json.loads(r["coordinates"])
        except Exception:
            continue
        for c in coords:
            rows.append({"img_id": os.path.basename(str(r["image_name"])),
                         "x_min": c["x_min"], "y_min": c["y_min"],
                         "x_max": c["x_max"], "y_max": c["y_max"],
                         "label": "beetle"})
    if not rows:
        return None, "no parseable detections"
    return pd.DataFrame(rows), ""


def row_label(r, fallback="beetle"):
    for c in LABEL_COLS:
        if c in r.index:
            v = r[c]
            if isinstance(v, str) and v.strip():
                return v.replace("_", " ").strip()
    return fallback


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------

def draw_box(draw, box, rgb, W, H):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(3, int(0.003 * max(W, H)))
    draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=(0, 0, 0), width=1)
    draw.rectangle([x1, y1, x2, y2], outline=rgb, width=w)


# ----------------------------------------------------------------------------
# empty-space sampling (size-matched)
# ----------------------------------------------------------------------------

def sample_free_box(W, H, occupied, wh, rng, margin, tries=150):
    mx, my = margin * W, margin * H
    bw, bh = wh
    if bw >= W * 0.9 or bh >= H * 0.9:
        return None
    for _ in range(tries):
        x1 = rng.uniform(0.01 * W, max(0.01 * W, W - bw - 0.01 * W))
        y1 = rng.uniform(0.01 * H, max(0.01 * H, H - bh - 0.01 * H))
        cand = [x1, y1, x1 + bw, y1 + bh]
        if any(overlaps(cand, b, mx, my) for b in occupied):
            continue
        return cand
    return None


def build_free_pool(W, H, real_boxes, ref_wh, rng, margin, jitter, n=120):
    """Empty regions sized from the median annotated box (so distractors match
    specimen scale), biased away from annotated specimens so that any
    undetected specimen is less likely to fall inside a supposedly empty box."""
    pool = []
    for _ in range(n):
        bw = ref_wh[0] * rng.uniform(1 - jitter, 1 + jitter)
        bh = ref_wh[1] * rng.uniform(1 - jitter, 1 + jitter)
        cands = [sample_free_box(W, H, real_boxes + pool, (bw, bh), rng, margin)
                 for _ in range(6)]
        cands = [c for c in cands if c]
        if cands:
            pool.append(max(cands, key=lambda c: min(
                math.dist(center(c), center(rb)) for rb in real_boxes)))
    return pool


def solve_distractors(pool, rt, bt, ans, margin_deg, rng, tries=600):
    """Pick an empty red box and an empty blue box so that all three incorrect
    referent pairings give directions that are distinct and neither equal nor
    adjacent to the true answer."""
    bad = {ans} | neighbours(ans)
    rt_c, bt_c = center(rt), center(bt)

    cand_r, cand_b = [], []
    for b in pool:
        d = direction_of(center(b), bt_c, margin_deg)
        if d is not None and d not in bad:
            cand_r.append((b, d))
        d = direction_of(rt_c, center(b), margin_deg)
        if d is not None and d not in bad:
            cand_b.append((b, d))
    if not cand_r or not cand_b:
        return None

    best = None
    for _ in range(tries):
        rd, d1 = rng.choice(cand_r)
        bd, d2 = rng.choice(cand_b)
        if rd is bd or overlaps(rd, bd):
            continue
        d3 = direction_of(center(rd), center(bd), margin_deg)
        if d3 is None or d3 in bad:
            continue
        distinct = len({d1, d2, d3})
        if best is None or distinct > best[0]:
            best = (distinct, rd, bd, [d1, d2, d3])
        if distinct == 3:
            break
    if best is None:
        return None
    return best[1], best[2], best[3]


def pick_neutrals(spare_real, used, rng, max_ratio, k=2):
    """k non-target specimens, non-overlapping and size-compatible with the
    boxes already placed."""
    order = list(spare_real)
    rng.shuffle(order)
    out = []
    for bx in order:
        if any(overlaps(bx, u) for u in used + out):
            continue
        if size_ratio(used + out + [bx]) > max_ratio:
            continue
        out.append(bx)
        if len(out) == k:
            return out
    return None


# ----------------------------------------------------------------------------
# question construction
# ----------------------------------------------------------------------------

def fill_options(ans, forced, rng):
    seen, out = set(), []
    for o in [ans] + list(forced):
        if o not in seen:
            seen.add(o)
            out.append(o)
    pool = [d for d in DIRS if d not in seen and d not in neighbours(ans)]
    rng.shuffle(pool)
    while len(out) < 4 and pool:
        out.append(pool.pop())
    out = out[:4]
    rng.shuffle(out)
    return out


QUESTION = (
    "This image is marked with six boxes: two red, two blue, one green and one "
    "cyan. Exactly one red box and exactly one blue box contain a specimen; "
    "the other red box and the other blue box are empty. The green and cyan "
    "boxes are irrelevant to the question. Where is the specimen inside the red "
    "box located with respect to the specimen inside the blue box? "
    "Options: {opts}")


# ----------------------------------------------------------------------------
# per-image processing
# ----------------------------------------------------------------------------

def prepare_image(ds_path, img_id, group, cfg, args, rej):
    """Returns (base_image, real, pool) or None."""
    img_path = os.path.join(ds_path, "Images", str(img_id))
    if not os.path.exists(img_path):
        rej["no_image"] += 1
        return None
    try:
        with Image.open(img_path) as im0:
            im0 = ImageOps.exif_transpose(im0) if cfg["exif"] else im0
            base = im0.convert("RGB")
    except Exception:
        rej["error"] += 1
        return None
    W, H = base.size

    real = []
    for _, r in group.iterrows():
        try:
            b = box_from(cfg["kind"], [r[c] for c in cfg["cols"]], W, H, cfg["norm"])
        except Exception:
            rej["invalid_box"] += 1
            continue
        if not valid_box(b, W, H):
            rej["invalid_box"] += 1
            continue
        if area(b) < args.min_box_frac * W * H:
            rej["box_too_small"] += 1
            continue
        b = [max(0.0, b[0]), max(0.0, b[1]), min(W - 1.0, b[2]), min(H - 1.0, b[3])]
        real.append({"box": b, "label": row_label(r)})
    if len(real) < MIN_ANIMALS:
        rej["few_animals"] += 1
        return None

    real_boxes = [x["box"] for x in real]
    ref_wh = (statistics.median(b[2] - b[0] for b in real_boxes),
              statistics.median(b[3] - b[1] for b in real_boxes))
    pool = build_free_pool(
        W, H, real_boxes, ref_wh,
        random.Random(f"{img_id}|pool|{args.seed}"),
        args.free_margin, args.size_jitter)
    return base, real, pool


def candidate_pairs(real, dir_count, args, rej, crng):
    """Geometrically valid pairs, red/blue assigned to balance the answer
    distribution, ordered so that rarer directions are taken first."""
    cands = []
    for i, j in itertools.combinations(range(len(real)), 2):
        bi, bj = real[i]["box"], real[j]["box"]
        if size_ratio([bi, bj]) > args.max_size_ratio:
            rej["size_mismatch"] += 1
            continue
        if iou(bi, bj) > args.max_iou:
            rej["overlap"] += 1
            continue
        if math.dist(center(bi), center(bj)) < args.min_sep * (
                half_diag(bi) + half_diag(bj)):
            rej["too_close"] += 1
            continue
        d = direction_of(center(bi), center(bj), args.margin_deg)
        if d is None:
            rej["boundary"] += 1
            continue
        if dir_count[d] <= dir_count[opposite(d)]:
            cands.append((i, j, d))
        else:
            cands.append((j, i, opposite(d)))
    cands.sort(key=lambda t: (dir_count[t[2]], crng.random()))
    return cands


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def process(args):
    ds = args.dataset
    ds_path = os.path.join(args.base_dir, ds)
    img_out = os.path.join(args.output_dir, "Images")
    os.makedirs(img_out, exist_ok=True)

    rng = random.Random(f"{ds}|{args.seed}")
    df, err = load_beetles(ds_path)
    if df is None:
        raise SystemExit(f"[{ds}] FATAL: {err}")

    cfg, err = calibrate(ds_path, df, rng)
    if cfg is None:
        raise SystemExit(f"[{ds}] FATAL: {err}")
    print(f"[{ds}] schema={cfg['kind']}{cfg['cols']} normalized={cfg['norm']} "
          f"exif={cfg['exif']} fit={cfg['score']:.3f}")

    rej = Counter()
    dir_count = Counter()
    qa = []
    per_img = Counter()          # emitted pairs per image
    used_pairs = set()           # (img_id, i, j) already emitted

    grouped = list(df.groupby("img_id"))
    rej["few_animals"] += sum(1 for _, g in grouped if len(g) < MIN_ANIMALS)
    groups = [(k, g) for k, g in grouped if len(g) >= MIN_ANIMALS]
    rng.shuffle(groups)
    if args.max_images:
        groups = groups[:args.max_images]
    print(f"[{ds}] eligible images: {len(groups)}")

    cap = args.pairs_per_image
    while len(qa) < args.target_pairs and cap <= args.pairs_per_image_cap:
        made_progress = False
        for img_id, group in groups:
            if len(qa) >= args.target_pairs:
                break
            if per_img[img_id] >= cap:
                continue

            prepped = prepare_image(ds_path, img_id, group, cfg, args, rej)
            if prepped is None:
                per_img[img_id] = cap          # do not retry next round
                continue
            base, real, pool = prepped
            W, H = base.size

            crng = random.Random(f"{img_id}|combos|{args.seed}")
            cands = candidate_pairs(real, dir_count, args, rej, crng)

            for i, j, ans in cands:
                if per_img[img_id] >= cap or len(qa) >= args.target_pairs:
                    break
                key = (img_id, min(i, j), max(i, j))
                if key in used_pairs:
                    continue
                prng = random.Random(f"{img_id}|{key[1]}|{key[2]}|{args.seed}")

                a, b = real[i], real[j]
                rt, bt = a["box"], b["box"]

                sol = solve_distractors(pool, rt, bt, ans, args.margin_deg, prng)
                if sol is None:
                    rej["no_distractor"] += 1
                    continue
                rd, bd, wrong = sol
                if size_ratio([rt, bt, rd, bd]) > args.max_size_ratio:
                    rej["size_mismatch"] += 1
                    continue

                spare = [x["box"] for k, x in enumerate(real) if k not in (i, j)]
                neutrals = pick_neutrals(spare, [rt, bt, rd, bd], prng,
                                         args.max_size_ratio, k=len(NEUTRAL_COLORS))
                if neutrals is None:
                    rej["no_neutrals"] += 1
                    continue

                placed = [(rt, "red"), (rd, "red"), (bt, "blue"), (bd, "blue")]
                placed += list(zip(neutrals, NEUTRAL_COLORS))
                assert len(placed) == N_BOXES, "six-box composition violated"
                ratio = size_ratio([p[0] for p in placed])
                assert ratio <= args.max_size_ratio + 1e-6, "size bound violated"

                im = base.copy()
                d = ImageDraw.Draw(im)
                order = list(placed)
                prng.shuffle(order)
                for bx, c in order:
                    draw_box(d, bx, COLORS[c], W, H)

                opts = fill_options(ans, wrong, prng)
                stem = f"{ds}_{os.path.splitext(str(img_id))[0]}_pair{per_img[img_id]}"
                name = f"{stem}_dist.jpg"
                im.save(os.path.join(img_out, name), quality=95)

                qa.append({
                    "image": name, "image_path": f"Images/{name}",
                    "dataset": ds, "condition": "distractor", "pair_id": stem,
                    "source_image": img_id,
                    "question": QUESTION.format(opts=", ".join(opts)),
                    "answer": ans, "options": "|".join(opts), "n_boxes": N_BOXES,
                    "size_ratio": round(ratio, 3),
                    "red_label": a["label"], "blue_label": b["label"],
                    "boxes": json.dumps({
                        "red_target": [round(v, 1) for v in rt],
                        "red_empty": [round(v, 1) for v in rd],
                        "blue_target": [round(v, 1) for v in bt],
                        "blue_empty": [round(v, 1) for v in bd],
                        "green": [round(v, 1) for v in neutrals[0]],
                        "cyan": [round(v, 1) for v in neutrals[1]],
                        "wrong_pairings": wrong}),
                })
                dir_count[ans] += 1
                used_pairs.add(key)
                per_img[img_id] += 1
                made_progress = True

        if len(qa) >= args.target_pairs:
            break
        if not made_progress and cap >= args.pairs_per_image_cap:
            break
        cap += 1
        print(f"[{ds}] {len(qa)} pairs after cap={cap - 1}; raising to {cap}")

    if qa:
        out = pd.DataFrame(qa)
        out.to_csv(os.path.join(args.output_dir, "qa.csv"), index=False)
        vc = out.answer.value_counts()
        print("\nn_boxes:", out.n_boxes.value_counts().to_dict())
        print("images used:", out.source_image.nunique())
        print("pairs per image:", out.source_image.value_counts()
                                    .value_counts().to_dict())
        print("size_ratio:", out.size_ratio.describe()[["mean", "50%", "max"]]
                                .round(3).to_dict())
        print("answers:", vc.to_dict())
        print("majority-class baseline:", round(vc.iloc[0] / len(out), 4))

    diag = {"dataset": ds, "schema": cfg["kind"], "columns": " ".join(cfg["cols"]),
            "normalized": cfg["norm"], "exif_transpose": cfg["exif"],
            "calibration_fit": round(cfg["score"], 4),
            "eligible_images": len(groups), "images_used": len(per_img),
            "pairs": len(qa), "final_cap": cap,
            "status": "ok" if len(qa) >= args.target_pairs else "short"}
    diag.update({f"rej_{k}": v for k, v in sorted(rej.items())})
    pd.DataFrame([diag]).to_csv(
        os.path.join(args.output_dir, "diagnostics.csv"), index=False)
    print("\nrejections:", dict(sorted(rej.items())))
    print(f"wrote {len(qa)} questions to {args.output_dir}/qa.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="Beetles-500")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--target_pairs", type=int, default=1000)
    p.add_argument("--pairs_per_image", type=int, default=2)
    p.add_argument("--pairs_per_image_cap", type=int, default=5,
                   help="raise per-image allowance up to this if short of target")
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    p.add_argument("--max_size_ratio", type=float, default=2.0,
                   help="max area ratio between largest and smallest of the six boxes")
    p.add_argument("--min_box_frac", type=float, default=0.0005,
                   help="drop annotated boxes below this fraction of image area")
    p.add_argument("--size_jitter", type=float, default=0.10,
                   help="empty-box size jitter around the median annotated box")
    p.add_argument("--max_iou", type=float, default=0.05)
    p.add_argument("--min_sep", type=float, default=0.60)
    p.add_argument("--margin_deg", type=float, default=8.0)
    p.add_argument("--free_margin", type=float, default=0.015)
    args = p.parse_args()
    process(args)