#!/usr/bin/env python3
"""
BioVQA C3 - OCR (specimen barcode label transcription).
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

# The accession format is stated because it is a real property of the label,
# not a hint: every code is the letter A followed by eleven digits. Saying so
# removes format guessing from what is being measured and makes an unparseable
# answer genuinely a reading failure rather than a formatting one. It does NOT
# give the answer away - the informative characters are the trailing digits,
# and stats.py reports exactly how many of them vary across the corpus.
#
# The prompt deliberately does not repeat the "A0..." hint twice; the source
# question already carries it.
FRAME = (
    "This image shows a specimen label with a barcode and a printed accession "
    "code beneath it.\n\n"
)

INSTRUCTION = (
    "\n\n"
    "Read the printed characters underneath the barcode. The code is always "
    "the letter A followed by eleven digits, twelve characters in total, with "
    "no spaces or punctuation.\n"
    "Transcribe exactly what is printed. Do not guess a plausible code and do "
    "not correct what you see.\n"
    "Respond with the twelve-character code and nothing else.\n"
    "Example: A00000012345"
)


def build_prompt(question):
    return FRAME + str(question).strip() + INSTRUCTION


# =========================================================================== #
# MODEL BLOCK - Qwen2.5-VL-7B-Instruct
# =========================================================================== #

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
MODEL_KEY = "qwenVL2_5_7B"
DEFAULT_BATCH_SIZE = 4
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 24))
VISION_BUDGET = "qwen max_pixels=3840*28*28"

# RESOLUTION
# Small printed characters are the entire task, so the vision budget is raised
# to roughly native resolution as in the morphometry task. Downsampling a label
# crop is the one thing guaranteed to turn a reading task into a guessing task.
# Applied identically to every model.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 3840 * 28 * 28   # ~3.0 MP, near native


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

# Canonical form first: A followed by digits. Falls back to a bare digit run so
# a model that drops the leading letter is still scored on what it read.
CODE_RE = re.compile(r"[A-Za-z][\s\-_.]*\d[\d\s\-_.]{6,}")
DIGITS_RE = re.compile(r"\d[\d\s\-_.]{6,}")


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def resolve_code(answer):
    """The transcribed accession code, or None.

    Strips the separators models insert into long digit strings ("A 0000 0032
    929") and uppercases the letter, because neither is a reading error. Keeps
    everything else verbatim: no length padding, no prefix repair, no nearest
    valid code. A wrong character must stay wrong or the error rate becomes
    meaningless.

    The raw string is preserved in <KEY>_answer_unfiltered; this resolved code
    goes in <KEY>_answer."""
    s = str(answer)
    m = CODE_RE.search(s)
    if m:
        return re.sub(r"[\s\-_.]", "", m.group()).upper()
    m = DIGITS_RE.search(s)
    if m:
        return re.sub(r"[\s\-_.]", "", m.group())
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
            v = resolve_code(answer)
            df.iat[pos, col_idx] = pd.NA if v is None else v
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. A model whose first answers contain no digit run at all is
        # misconfigured - wrong class, broken chat template, dead environment.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if all(resolve_code(a) is None for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers contained a "
                    "code-like string.\n  raw sample: %r\n"
                    % (done, recent[:6]))

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
    gold = df["answer"].astype(str).str.upper()
    pred = df[col].astype(str).str.upper().replace({"NAN": None})
    ok = pred.notna() & (pred != "None")
    n = int(ok.sum())
    em = float((gold[ok] == pred[ok]).mean()) if n else float("nan")
    wellformed = float(pred[ok].str.fullmatch(r"A\d{11}").mean()) if n else float("nan")

    record = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "task": "ocr",
        "qa_csv": os.path.abspath(qa_csv),
        "out_csv": os.path.abspath(out_path),
        "rows": int(len(df)),
        "scored": n,
        "unparsed": int(len(df) - n),
        "missing_images": missing,
        "failed_batches": failed_batches,
        "batch_size": int(batch_size),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "vision_budget": VISION_BUDGET,
        "prompt_frame": FRAME,
        "prompt_instruction": INSTRUCTION,
        "indicative_EM": em,
        "indicative_wellformed": wellformed,
        "seconds": round(time.time() - t0, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log("saved " + json_path)
    log("indicative EM %.3f | well-formed %.3f | unparsed %d | missing %d | "
        "failed batches %d  (evaluate.py is authoritative)"
        % (em, wellformed, record["unparsed"], missing, failed_batches))


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
    print(build_prompt("What is the text under the barcode (starting with "
                       "A0...) shown in this image?"), flush=True)
    print("-" * 78, flush=True)

    handle = load_model()
    run_all(handle, args.qa_csv, out_dir, args.batch_size,
            args.overwrite, args.limit)


if __name__ == "__main__":
    main()
