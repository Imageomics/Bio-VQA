#!/usr/bin/env python3
"""
BioVQA C3 - Viewpoint (Single + Cued + Positional tiers, one pass).
Model: InternVL3.5-8B

  python intern3_5_8B.py --qa_csv chunks/chunk00/qa.csv --out_dir chunks/chunk00

No path is hardcoded anywhere in this file: every location arrives as a
command-line argument.

Everything outside the MODEL BLOCK is identical across all eighteen scripts
and must stay that way - it is the controlled part of the experiment.
"""

import argparse
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

# The gold answers are raw Wildbook viewpoint tokens. Asking an open question
# against a closed, unstated vocabulary measures whether a model can guess the
# annotation scheme, not whether it perceives viewpoint - models answer "side
# view" or "profile" and score zero. The accepted answers are therefore
# enumerated in the prompt, exactly as stored.
#
# Override if stats.py reports gold labels outside this set:
#   export VIEWPOINT_LABELS="front,back,left,right,frontleft,frontright,..."
VIEWPOINT_LABELS = [
    s.strip().lower() for s in os.environ.get(
        "VIEWPOINT_LABELS",
        "front,back,left,right,frontleft,frontright,backleft,backright,up,down"
    ).split(",") if s.strip()
]

# Panel framing is added only for the multi-panel tiers. Apart from this one
# clause the scaffolding is identical across tiers, so the tier comparison is
# not contaminated by a prompt difference.
PANEL_FRAME = (
    "This image consists of three separate photographs placed side by side.\n\n"
)

INSTRUCTION_HEAD = (
    "\n\n"
    "Answer with the direction the animal is facing, relative to the camera.\n"
    "Choose exactly one of these accepted answers:\n"
)

INSTRUCTION_TAIL = (
    "\n"
    "Reply with one of those options, written exactly as above in lower case, "
    "and nothing else. No explanation, no punctuation.\n"
    "For example, if the animal faces away from the camera and to the left, "
    "reply: backleft"
)


def options_block():
    return "  " + ", ".join(VIEWPOINT_LABELS)


def build_prompt(question, n_panels):
    frame = PANEL_FRAME if int(n_panels) > 1 else ""
    return (frame + str(question).strip()
            + INSTRUCTION_HEAD + options_block() + INSTRUCTION_TAIL)


# =========================================================================== #
# MODEL BLOCK - InternVL3.5-8B
# =========================================================================== #

MODEL_ID = os.environ.get("MODEL_ID", "OpenGVLab/InternVL3_5-8B-Instruct")
MODEL_KEY = "internvl3_5_8B-Instruct"
DEFAULT_BATCH_SIZE = 8
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 16))

# Tiles per image. 12 is the InternVL default; 6 is used across the whole
# InternVL family here so the tiling budget is not a hidden variable between
# model sizes.
MAX_NUM_TILES = 6
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

# Longest first, so "frontleft" is never matched as "front".
LABELS_BY_LEN = sorted(VIEWPOINT_LABELS, key=len, reverse=True)


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def canon(text):
    """Lower-case, drop everything that is not a letter. Turns 'front-left',
    'Front Left' and 'front_left' all into 'frontleft'."""
    return re.sub(r"[^a-z]", "", str(text).lower())


def resolve_label(answer):
    """Longest accepted label contained in the raw answer, or None.

    The raw string is preserved in <KEY>_answer_unfiltered; this resolved
    label goes in <KEY>_answer, so nothing is lost and evaluate.py never has
    to re-parse free text."""
    t = canon(answer)
    if not t:
        return None
    for lab in LABELS_BY_LEN:
        if lab in t:
            return lab
    return None


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
    if "n_panels" not in df.columns:
        df["n_panels"] = 1

    raw_col = MODEL_KEY + "_answer_unfiltered"
    col = MODEL_KEY + "_answer"
    df[raw_col] = pd.NA
    df[col] = pd.NA

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
            work.append((pos, p, build_prompt(row["question"], row["n_panels"])))
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
            v = resolve_label(answer)
            df.iat[pos, col_idx] = pd.NA if v is None else v
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. If not one of the first answers contains an accepted
        # label, the model is misconfigured or the chat template is broken.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if all(resolve_label(a) is None for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers contained an "
                    "accepted viewpoint label.\n  raw sample: %r\n"
                    "  accepted: %s\n"
                    % (done, recent[:6], ", ".join(VIEWPOINT_LABELS)))

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
    # Written to disk, not only to the SLURM .out, so the run is reproducible
    # and auditable after the logs are rotated away.
    gold = df["answer"].map(canon)
    pred = df[col].astype("object")
    ok = gold.isin(VIEWPOINT_LABELS) & pred.notna()
    n = int(ok.sum())
    em = float((gold[ok] == pred[ok]).mean()) if n else float("nan")

    record = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "task": "viewpoint",
        "qa_csv": os.path.abspath(qa_csv),
        "out_csv": os.path.abspath(out_path),
        "rows": int(len(df)),
        "scored": n,
        "unparsed": int(len(df) - pred.notna().sum()),
        "gold_off_list": int((~gold.isin(VIEWPOINT_LABELS)).sum()),
        "missing_images": missing,
        "failed_batches": failed_batches,
        "batch_size": int(batch_size),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "accepted_labels": VIEWPOINT_LABELS,
        "prompt_head": INSTRUCTION_HEAD,
        "prompt_tail": INSTRUCTION_TAIL,
        "prompt_panel_frame": PANEL_FRAME,
        "indicative_EM": em,
        "seconds": round(time.time() - t0, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log("saved " + json_path)
    log("indicative EM %.3f | unparsed %d | missing %d | failed batches %d  "
        "(evaluate.py is authoritative)"
        % (em, record["unparsed"], missing, failed_batches))


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

    print("-" * 78, flush=True)
    print("ACCEPTED ANSWERS (identical across all models):", flush=True)
    print(options_block(), flush=True)
    print("", flush=True)
    print("PROMPT, single-panel tier:", flush=True)
    print(build_prompt("What is the viewpoint of the animal "
                       "(red bounding box) shown in this image?", 1), flush=True)
    print("-" * 78, flush=True)

    handle = load_model()
    run_all(handle, args.qa_csv, out_dir, args.batch_size,
            args.overwrite, args.limit)


if __name__ == "__main__":
    main()
