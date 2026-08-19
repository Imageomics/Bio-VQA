#!/usr/bin/env python3
"""
BioVQA C3 - Morphometry (scale-bar referenced absolute measurement).
Model: LLaVA-NeXT Vicuna-7B

  python llava_next_vicuna_7B.py --qa_csv chunks/chunk00/qa.csv --out_dir chunks/chunk00

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

# The source question says "with respect to the scalebar ... in cm?" but never
# states how long the scale bar is. Without that, the ratio is computable and
# the answer in centimetres is not, and a model can only produce a number by
# recalling typical beetle dimensions - which is precisely the shortcut this
# task exists to rule out. The frame below supplies the missing constant, the
# procedure, and an explicit instruction not to answer from prior knowledge.
FRAME = (
    "This image shows a plate of pinned beetle specimens, photographed at a "
    "single scale.\n"
    "A BLUE line marks the scale bar. The scale bar is exactly 1 cm long.\n"
    "A RED line marks the measurement asked about below.\n\n"
)

INSTRUCTION = (
    "\n\n"
    "Compare the pixel length of the red line to the pixel length of the blue "
    "scale bar. Because the scale bar is exactly 1 cm long, the red line's "
    "length in centimetres equals that ratio: if the red line is half as long "
    "as the blue line the answer is 0.50 cm, and if it is twice as long the "
    "answer is 2.00 cm.\n"
    "Measure the two lines as they appear in this image. Do not answer from "
    "what you know about typical beetle sizes.\n"
    "Respond with one number rounded to two decimal places followed by cm, "
    "and nothing else. No explanation, no range, no other units.\n"
    "Example: 0.87 cm"
)


def build_prompt(question):
    return FRAME + str(question).strip() + INSTRUCTION


# =========================================================================== #
# MODEL BLOCK - LLaVA-NeXT Vicuna-7B
# =========================================================================== #
# apply_chat_template reads the template from the checkpoint, so the Mistral
# and Vicuna variants share this block unchanged.

MODEL_ID = os.environ.get("MODEL_ID", "llava-hf/llava-v1.6-vicuna-7b-hf")
MODEL_KEY = "llava_next_vicuna_7B"
DEFAULT_BATCH_SIZE = 8
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 12))
VISION_BUDGET = "model default"


def load_model():
    import torch
    from transformers import LlavaNextForConditionalGeneration, AutoProcessor

    print("Loading " + MODEL_ID + " with SDPA...", flush=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    print("  loaded class: " + type(model).__name__, flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


def run_batch(handle, batch):
    import torch
    model, processor = handle

    prompts, images = [], []
    for path, prompt in batch:
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        prompts.append(processor.apply_chat_template(
            conversation, add_generation_prompt=True))
        images.append(Image.open(path).convert("RGB"))

    inputs = processor(images=images, text=prompts, padding=True,
                       return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False)

    trimmed = generated[:, inputs["input_ids"].shape[-1]:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    return [x.strip() for x in decoded]


# =========================================================================== #
# HARNESS - identical in all eighteen scripts
# =========================================================================== #

# First number, with an optional unit. Handles "0.87 cm", "8.7mm", ".63 cm",
# "approximately 1.1 centimetres" and a bare "0.87".
NUM_RE = re.compile(
    r"([-+]?\d*\.?\d+)\s*(mm|millimet\w*|cm|centimet\w*)?", re.I)


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def resolve_cm(answer):
    """Measurement in centimetres, or None.

    Millimetres are converted rather than rejected: several models answer
    "8.7 mm" for a 0.87 cm line, which is correct physics in the wrong unit,
    and scoring that as 8.7 would be our bug, not theirs. A bare number is
    read as centimetres, which is what the prompt asks for.

    The raw string is preserved in <KEY>_answer_unfiltered; this resolved
    float goes in <KEY>_answer, so nothing is lost and evaluate.py never has
    to re-parse free text."""
    m = NUM_RE.search(str(answer).replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    if unit.startswith("mm") or unit.startswith("millimet"):
        v = v / 10.0
    return v


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
            v = resolve_cm(answer)
            df.iat[pos, col_idx] = pd.NA if v is None else v
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. A model whose first answers contain no number at all is
        # misconfigured - wrong class, broken chat template, dead environment.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if all(resolve_cm(a) is None for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers contained a "
                    "number.\n  raw sample: %r\n" % (done, recent[:6]))

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
    truth = pd.to_numeric(df["answer_cm"], errors="coerce")
    pred = pd.to_numeric(df[col], errors="coerce")
    ok = truth.notna() & pred.notna() & (pred > 0)
    n = int(ok.sum())
    rel = ((pred[ok] - truth[ok]).abs() / truth[ok]) if n else None

    record = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "task": "morphometry",
        "qa_csv": os.path.abspath(qa_csv),
        "out_csv": os.path.abspath(out_path),
        "rows": int(len(df)),
        "scored": n,
        "unparsed": int(len(df) - pred.notna().sum()),
        "missing_images": missing,
        "failed_batches": failed_batches,
        "batch_size": int(batch_size),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "vision_budget": VISION_BUDGET,
        "prompt_frame": FRAME,
        "prompt_instruction": INSTRUCTION,
        "indicative_mean_rel_err": float(rel.mean()) if n else float("nan"),
        "indicative_acc20": float((rel <= 0.20).mean()) if n else float("nan"),
        "seconds": round(time.time() - t0, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log("saved " + json_path)
    log("indicative mean rel err %.3f | acc@20%% %.3f | unparsed %d | "
        "missing %d | failed batches %d  (evaluate.py is authoritative)"
        % (record["indicative_mean_rel_err"], record["indicative_acc20"],
           record["unparsed"], missing, failed_batches))


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
    print(build_prompt("What is the measurement of the elytra length (marked "
                       "with red line) with respect to the scalebar (marked "
                       "with blue line) in cm?"), flush=True)
    print("-" * 78, flush=True)

    handle = load_model()
    run_all(handle, args.qa_csv, out_dir, args.batch_size,
            args.overwrite, args.limit)


if __name__ == "__main__":
    main()
