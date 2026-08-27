#!/usr/bin/env python3
"""
BioVQA C3 - Counting-Relative (2x2 grid, Same + Different conditions).
Model: Qwen2.5-VL-7B-Instruct

  python qwenVL2_5_7B.py --qa_csv chunks/chunk00/qa.csv --out_dir chunks/chunk00

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

# The source question says "which of these 4 images", but the canvas handed to
# the model is also an image, so "image" is overloaded. The frame names the
# grid explicitly and restates the letter-to-quadrant mapping, which is the
# only thing a model needs in order to answer in the requested format.
#
# The scale sentence matters: the four panels are pasted at different
# magnifications, so a model comparing apparent animal size rather than count
# would be answering a different question. Saying so costs nothing and removes
# one interpretation of an otherwise ambiguous comparison.
FRAME = (
    "This image is a 2x2 grid of four separate photographs:\n"
    "  Image A - top-left      Image B - top-right\n"
    "  Image C - bottom-left   Image D - bottom-right\n\n"
)

INSTRUCTION = (
    "\n\n"
    "Count the animals in each of the four photographs and compare the "
    "totals. The photographs are shown at different scales, so compare how "
    "many animals there are, not how large they appear.\n"
    "Exactly one photograph is the answer. Choose it even if the difference "
    "is small.\n"
    "Answer immediately with the letter. Do not describe the photographs and "
    "do not explain your reasoning.\n"
    "Respond with one letter and nothing else: A, B, C or D."
)


def build_prompt(question):
    return FRAME + str(question).strip() + INSTRUCTION


# =========================================================================== #
# MODEL BLOCK - Qwen2.5-VL-7B-Instruct
# =========================================================================== #

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
MODEL_KEY = "qwenVL2_5_7B"
DEFAULT_BATCH_SIZE = 4
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 32))
VISION_BUDGET = "qwen max_pixels=2560*28*28"

# RESOLUTION
# Four photographs share one canvas, so each panel receives roughly a quarter
# of the pixel budget - and the animals being counted are often small aerial
# subjects to begin with. The budget is raised toward native resolution as in
# the other beetle-plate tasks, applied identically to every model.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 2560 * 28 * 28   # ~2.0 MP


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Auto resolves the architecture from the checkpoint config rather than us
    # asserting dense vs MoE by hand - hardcoding the class is what silently
    # destroyed an earlier run by randomly initialising mismatched layers.
    print("Loading " + MODEL_ID + " with SDPA...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    print("  loaded class: " + type(model).__name__, flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    processor.tokenizer.padding_side = "left"
    return model, processor


def run_batch(handle, batch):
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = handle

    messages_batch = [
        [{"role": "user", "content": [
            {"type": "image", "image": "file://" + os.path.abspath(path)},
            {"type": "text", "text": prompt},
        ]}]
        for path, prompt in batch
    ]

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs

    inputs = processor(**kwargs).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    return [x.strip() for x in decoded]


# =========================================================================== #
# HARNESS - identical in all eighteen scripts
# =========================================================================== #

# Answer extraction, in priority order.
#
# Two traps this has to avoid. First, "Image" contains a lower-case 'a', so a
# case-insensitive [ABCD] search returns A for every option label - the same
# failure that made every "Beetle X" answer parse as B in the relative-size
# task. Every pattern below is therefore anchored or case-sensitive.
#
# Second, a model that narrates enumerates the panels before committing
# ("Image A has three, Image B has seven ... so the answer is B"), so a
# first-match rule would return A. Every pattern takes the LAST match.
#
# Models also answer with the quadrant name instead of the letter, which is a
# correct answer in the wrong format, so positions are mapped rather than
# discarded.
POSITIONS = {"topleft": "A", "topright": "B",
             "bottomleft": "C", "bottomright": "D"}

ANSWER_MARK_RE = re.compile(
    r"(?:answer|choice|selection|pick|most|least|fewest|greatest|largest)\b"
    r"[^A-Za-z0-9]{0,12}(?:is\s+)?(?:image\s*)?\**\s*([ABCD])(?![A-Za-z])",
    re.I)   # re.I is required: "**Answer:** B" is capitalised, and without it
            # the marker never matches and the fallback returns the last
            # "Image X" from the enumeration instead of the conclusion.
LEAD_RE = re.compile(r"^\W*(?:image\s+)?([ABCD])(?![A-Za-z])", re.I)
IMAGE_RE = re.compile(r"[Ii]mage\s*\**\s*([ABCD])(?![A-Za-z])")
POS_RE = re.compile(r"\b(top|bottom)[\s\-_]*(left|right)\b", re.I)
LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def resolve_letter(answer):
    """The chosen quadrant letter, or None.

    The raw string is preserved in <KEY>_answer_unfiltered; this resolved
    letter goes in <KEY>_answer, so nothing is lost and evaluate.py never has
    to re-parse free text."""
    s = str(answer)
    if not s.strip():
        return None

    m = list(ANSWER_MARK_RE.finditer(s))
    if m:
        return m[-1].group(1).upper()

    m = LEAD_RE.match(s)
    if m:
        return m.group(1).upper()

    m = list(IMAGE_RE.finditer(s))
    if m:
        return m[-1].group(1).upper()

    m = list(POS_RE.finditer(s))
    if m:
        key = (m[-1].group(1) + m[-1].group(2)).lower()
        if key in POSITIONS:
            return POSITIONS[key]

    m = list(LETTER_RE.finditer(s))
    if m:
        return m[-1].group(1)
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
            work.append((pos, p, build_prompt(row["question"])))
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
            v = resolve_letter(answer)
            df.iat[pos, col_idx] = pd.NA if v is None else v
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. A model whose first answers contain no option letter is
        # misconfigured - wrong class, broken chat template, dead environment.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if all(resolve_letter(a) is None for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers contained an "
                    "option letter.\n  raw sample: %r\n" % (done, recent[:6]))

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
    gold = df["answer_letter"].astype(str).str.upper()
    pred = df[col].astype(str).str.upper()
    ok = gold.isin(list("ABCD")) & pred.isin(list("ABCD"))
    n = int(ok.sum())
    acc = float((gold[ok] == pred[ok]).mean()) if n else float("nan")
    top = pred[pred.isin(list("ABCD"))].mode()
    top = str(top.iloc[0]) if len(top) else "-"
    unparsed = int(len(df) - pred.isin(list("ABCD")).sum())

    record = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "task": "counting_relative",
        "qa_csv": os.path.abspath(qa_csv),
        "out_csv": os.path.abspath(out_path),
        "rows": int(len(df)),
        "scored": n,
        "unparsed": unparsed,
        "missing_images": missing,
        "failed_batches": failed_batches,
        "batch_size": int(batch_size),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "vision_budget": VISION_BUDGET,
        "prompt_frame": FRAME,
        "prompt_instruction": INSTRUCTION,
        "indicative_accuracy": acc,
        "indicative_pred_top": top,
        "seconds": round(time.time() - t0, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log("saved " + json_path)
    log("indicative accuracy %.3f (chance 0.250) | most frequent answer %s | "
        "unparsed %d | missing %d | failed batches %d  "
        "(evaluate.py is authoritative)"
        % (acc, top, unparsed, missing, failed_batches))


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
    print("PROMPT:", flush=True)
    print(build_prompt(
        "Which of these 4 images contains most number of bees: Image A "
        "(top-left), Image B (top-right), Image C (bottom-left), Image D "
        "(bottom-right)?"), flush=True)
    print("-" * 78, flush=True)

    handle = load_model()
    run_all(handle, args.qa_csv, out_dir, args.batch_size,
            args.overwrite, args.limit)


if __name__ == "__main__":
    main()
