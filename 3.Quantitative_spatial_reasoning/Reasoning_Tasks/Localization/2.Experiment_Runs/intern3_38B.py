#!/usr/bin/env python3
"""
BioVQA C3 - Localization (bounding-box prediction, Easy + Medium + Hard).
Model: InternVL3-38B

  python intern3_38B.py --qa_csv chunks/chunk00/qa.csv --out_dir chunks/chunk00

No path is hardcoded anywhere in this file: every location arrives as a
command-line argument.

Everything outside the MODEL BLOCK is identical across all eighteen scripts
and must stay that way - it is the controlled part of the experiment.
"""

import argparse
import ast
import json
import os
import re
import time

import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================================== #
# EXPERIMENT CONSTANTS - identical in all eighteen scripts
# =========================================================================== #

# THE COORDINATE FRAME IS THE WHOLE ANSWER HERE.
#
# The ground truth for the multi-panel tiers already has the panel offset
# added: on one Medium canvas the three boxes start at x = 137, 2743 and 5320,
# and on a Hard canvas at x = 1760, 4925 and 9803. The gold boxes are
# therefore in WHOLE-CANVAS pixels, and the manifest stores no panel offsets
# or panel widths, so a panel-relative answer could not be mapped back even if
# we wanted it. Every tier is asked for the same thing: pixels measured from
# the top-left corner of the entire image shown.
#
# The canvas dimensions are injected per row. Without them "in pixels" is
# meaningless to a model that has no idea how large the image is, and the
# resolver downstream would have nothing to normalise against.
PANEL_FRAME = (
    "This image consists of three separate photographs placed side by side.\n"
)

INSTRUCTION_HEAD = (
    "\n\n"
    "Respond with four numbers in this exact format and nothing else:\n"
    "[x, y, width, height]\n"
    "where x and y are the pixel coordinates of the top-left corner of the "
    "box, measured from the top-left corner of the WHOLE image described "
    "above, and width and height are the size of the box in pixels.\n"
    "Do not use fractions, percentages or normalised coordinates. Do not "
    "explain.\n"
    "Example: [1240, 88, 540, 410]"
)


def build_prompt(question, n_panels, width, height):
    frame = PANEL_FRAME if int(n_panels) > 1 else ""
    frame += ("The whole image is %d pixels wide and %d pixels tall.\n\n"
              % (int(width), int(height)))
    return frame + str(question).strip() + INSTRUCTION_HEAD


# =========================================================================== #
# MODEL BLOCK - InternVL3-38B
# =========================================================================== #

MODEL_ID = os.environ.get("MODEL_ID", "OpenGVLab/InternVL3-38B-Instruct")
MODEL_KEY = "internvl3_38B-Instruct"
DEFAULT_BATCH_SIZE = 2
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 48))
VISION_BUDGET = "internvl max_num_tiles=12"

# RESOLUTION
# The multi-panel canvases here are enormous - three photographs side by side
# reach 12,000 px wide - and the target is often a small animal inside one
# panel. The budget is raised accordingly. It cannot be raised without limit:
# at some point the vision tower emits more tokens than the model can attend
# over, so this is a compromise, and the figure is recorded in every run's
# summary.json so results are never compared across different budgets.
MAX_NUM_TILES = 12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_tiles(image_file, input_size=448, max_num=12):
    import torch
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Shim for the tied-weights check that trips some InternVL remote-code
    # revisions. Harmless where unnecessary, so failure to apply is not fatal.
    try:
        from transformers.modeling_utils import PreTrainedModel
        PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
    except Exception as exc:
        print("  note: tied-weights shim not applied (%r)" % (exc,), flush=True)

    print("Loading " + MODEL_ID + "...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        device_map="auto",
    ).eval()
    print("  loaded class: " + type(model).__name__, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False
    )
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return model, tokenizer, generation_config


def run_batch(handle, batch):
    import torch
    model, tokenizer, generation_config = handle

    tiles, num_patches_list, questions = [], [], []
    for path, prompt in batch:
        pv = load_tiles(path, max_num=MAX_NUM_TILES).to(torch.bfloat16)
        tiles.append(pv)
        num_patches_list.append(pv.size(0))
        questions.append("<image>\n" + prompt)

    pixel_values = torch.cat(tiles, dim=0).to(model.device)

    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config,
    )
    return [str(r).strip() for r in responses]


# =========================================================================== #
# HARNESS - identical in all eighteen scripts
# =========================================================================== #

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def _valid(x1, y1, x2, y2, W, H):
    """True if this candidate box is a physically possible answer."""
    if not (x2 > x1 and y2 > y1):
        return False
    tol = 0.02
    if x1 < -tol * W or y1 < -tol * H or x2 > (1 + tol) * W or y2 > (1 + tol) * H:
        return False
    frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
    return 1e-5 <= frac <= 1.0


def resolve_box(answer, W, H):
    """-> ((x, y, w, h) in absolute pixels, interpretation used).

    Models do not agree on a coordinate convention. Qwen emits 0-1000
    normalised values, several models emit 0-1 fractions, some emit
    [x1,y1,x2,y2] instead of [x,y,w,h], and some obey the prompt and emit
    absolute pixels. Trying the conventions in a principled order and keeping
    the FIRST that yields a valid in-bounds box recovers the intent, instead
    of scoring a correct localisation as zero because of its units.

    The order is decided by magnitude, because that is the only evidence
    available in a single reply:

      all values <= 1        can only be fractions
      all values <= 1000 and small relative to the canvas
                             a 0-1000 answer is far likelier than a real
                             prediction confined to the top-left corner of a
                             12,000-pixel-wide canvas
      otherwise              take the prompt at its word and read pixels

    FIRST VALID WINS, deliberately. An earlier version scored candidates and
    kept the best, with a mild preference for smaller boxes - which handed
    every Qwen-style 0-1000 answer to the absolute reading, because a box in
    the corner scores "better" than the correct one. Ordering must decide,
    not scoring.

    Some ambiguity is irreducible: 600 on a 4000-pixel canvas could be either
    convention. The prompt states the canvas size and forbids normalised
    coordinates, and the chosen mode is recorded per row so evaluate.py can
    re-resolve a model under its own majority convention and report both.

    Nothing is repaired: no clipping to bounds, no aspect correction, no
    nearest-plausible box. A wrong box stays wrong.
    """
    nums = NUM_RE.findall(str(answer).replace(",", " "))
    if len(nums) < 4:
        return None, "unparsed"
    try:
        v = [float(x) for x in nums[:4]]
    except ValueError:
        return None, "unparsed"

    ABS = ("abs", 1.0, 1.0)
    N01 = ("norm01", W, H)
    N1K = ("norm1000", W / 1000.0, H / 1000.0)
    PCT = ("pct", W / 100.0, H / 100.0)

    mx = max(abs(t) for t in v)
    if mx <= 1.02:
        order = [N01, PCT, N1K, ABS]
    elif mx <= 1000.0 and mx < 0.25 * max(W, H):
        order = [N1K, PCT, ABS, N01]
    elif mx <= 100.0:
        order = [PCT, N1K, ABS, N01]
    else:
        order = [ABS, N1K, PCT, N01]

    for name, sx, sy in order:
        a, b, c, d = v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy
        for fmt, (x1, y1, x2, y2) in (("xywh", (a, b, a + c, b + d)),
                                      ("xyxy", (a, b, c, d))):
            if _valid(x1, y1, x2, y2, W, H):
                return (round(x1, 2), round(y1, 2),
                        round(x2 - x1, 2), round(y2 - y1, 2)), name + "_" + fmt
    return None, "invalid"


def run_all(handle, qa_csv, out_dir, batch_size, overwrite, limit):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qa_" + MODEL_KEY + ".csv")
    json_path = os.path.join(out_dir, "qa_" + MODEL_KEY + ".summary.json")

    if os.path.exists(out_path) and not overwrite:
        log("already done, skipping: " + out_path)
        return

    df = pd.read_csv(qa_csv)
    if limit:
        df = df.iloc[:limit].copy()

    raw_col = MODEL_KEY + "_answer_unfiltered"
    col = MODEL_KEY + "_answer"
    mode_col = MODEL_KEY + "_parse_mode"
    df[raw_col] = pd.NA
    df[col] = pd.NA
    df[mode_col] = pd.NA

    # Row POSITION travels with each item, so answers are written back to the
    # row they came from. A skipped row can never shift everything after it.
    work, missing = [], 0
    for pos in range(len(df)):
        row = df.iloc[pos]
        try:
            p = str(row["image_abspath"]).strip()
            if not p or not os.path.exists(p):
                missing += 1
                continue
            work.append((pos, p, build_prompt(row["question"], row["n_panels"],
                                              row["img_w"], row["img_h"])))
        except Exception as exc:
            log("  row %d unusable: %r" % (pos, exc))

    if missing:
        log("WARNING: %d images missing on disk" % missing)
    if not work:
        raise SystemExit(
            "\nABORTING: not a single image resolved on disk.\n"
            "  Re-run split_qa.py, which writes an absolute image_abspath.\n")

    log("%d items, batch size %d" % (len(work), batch_size))
    raw_idx = df.columns.get_loc(raw_col)
    col_idx = df.columns.get_loc(col)
    mode_idx = df.columns.get_loc(mode_col)
    done, failed_batches, sanity_checked = 0, 0, False
    recent = []
    t0 = time.time()

    for start in range(0, len(work), batch_size):
        chunk = work[start:start + batch_size]
        try:
            answers = run_batch(handle, [(p, q) for _, p, q in chunk])
        except Exception as exc:
            failed_batches += 1
            log("  batch at %d failed: %r" % (start, exc))
            continue
        for (pos, _, _), answer in zip(chunk, answers):
            df.iat[pos, raw_idx] = answer
            box, mode = resolve_box(answer, df.iat[pos, df.columns.get_loc("img_w")],
                                    df.iat[pos, df.columns.get_loc("img_h")])
            df.iat[pos, col_idx] = pd.NA if box is None else json.dumps(list(box))
            df.iat[pos, mode_idx] = mode
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. A model whose first answers contain no four numbers at all
        # is misconfigured - wrong class, broken chat template, dead env.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if all(len(NUM_RE.findall(str(a).replace(",", " "))) < 4
                   for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers contained four "
                    "numbers.\n  raw sample: %r\n" % (done, recent[:6]))

        if done % 200 < batch_size:
            rate = done / max(time.time() - t0, 1e-6)
            log("  %d/%d  %.2f q/s  eta %.0f min"
                % (done, len(work), rate,
                   (len(work) - done) / max(rate, 1e-6) / 60))

    n_batches = (len(work) + batch_size - 1) // batch_size
    if failed_batches > n_batches // 2:
        raise SystemExit(
            "\nABORTING: %d of %d batches failed. Not writing a partial CSV.\n"
            % (failed_batches, n_batches))

    df.to_csv(out_path, index=False)
    log("saved " + out_path)

    # ---- run record ------------------------------------------------------ #
    def iou(g, p):
        gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
        px1, py1, px2, py2 = p[0], p[1], p[0] + p[2], p[1] + p[3]
        ix1, iy1 = max(gx1, px1), max(gy1, py1)
        ix2, iy2 = min(gx2, px2), min(gy2, py2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
        return inter / union if union > 0 else 0.0

    ious = []
    for _, r in df.iterrows():
        if pd.isna(r[col]):
            continue
        try:
            g = [float(r["gt_x"]), float(r["gt_y"]),
                 float(r["gt_w"]), float(r["gt_h"])]
            ious.append(iou(g, json.loads(r[col])))
        except Exception:
            pass
    n = len(ious)
    mi = float(sum(ious) / n) if n else float("nan")
    a50 = float(sum(1 for t in ious if t >= 0.5) / n) if n else float("nan")

    record = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "task": "localization",
        "qa_csv": os.path.abspath(qa_csv),
        "out_csv": os.path.abspath(out_path),
        "rows": int(len(df)),
        "scored": n,
        "unparsed": int(len(df) - df[col].notna().sum()),
        "parse_modes": {k: int(v) for k, v in
                        df[mode_col].value_counts().items()},
        "missing_images": missing,
        "failed_batches": failed_batches,
        "batch_size": int(batch_size),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "vision_budget": VISION_BUDGET,
        "prompt_panel_frame": PANEL_FRAME,
        "prompt_instruction": INSTRUCTION_HEAD,
        "indicative_meanIoU": mi,
        "indicative_acc50": a50,
        "seconds": round(time.time() - t0, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log("saved " + json_path)
    log("indicative mean IoU %.3f | Acc@0.5 %.3f | unparsed %d | missing %d | "
        "failed batches %d  (evaluate.py is authoritative)"
        % (mi, a50, record["unparsed"], missing, failed_batches))
    log("parse modes: %r" % (record["parse_modes"],))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_csv", required=True)
    parser.add_argument("--out_dir", default="",
                        help="default: the directory holding --qa_csv")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N rows (smoke test)")
    args = parser.parse_args()

    if not os.path.isfile(args.qa_csv):
        raise SystemExit("qa.csv not found: " + args.qa_csv)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.qa_csv))

    log("model  " + MODEL_ID)
    log("key    " + MODEL_KEY)
    log("qa     " + args.qa_csv)
    log("vision " + VISION_BUDGET)

    print("-" * 78, flush=True)
    print("PROMPT, single-panel tier:", flush=True)
    print(build_prompt("Provide bounding box coordinates for the subject "
                       "animal (zebra) in this image.", 1, 4000, 3000),
          flush=True)
    print("", flush=True)
    print("PROMPT, three-panel tier:", flush=True)
    print(build_prompt("Provide the bounding box coordinates for the main "
                       "subject (zebra) in the middle image, in [x, y, w, h] "
                       "format.", 3, 12595, 3500), flush=True)
    print("-" * 78, flush=True)

    handle = load_model()
    run_all(handle, args.qa_csv, out_dir, args.batch_size,
            args.overwrite, args.limit)


if __name__ == "__main__":
    main()
