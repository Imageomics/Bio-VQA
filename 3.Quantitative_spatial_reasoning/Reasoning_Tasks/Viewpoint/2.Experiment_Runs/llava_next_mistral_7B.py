#!/usr/bin/env python3
"""
BioVQA C3 - Viewpoint (Single + Cued + Positional tiers, one pass).
Model: LLaVA-NeXT Mistral-7B

  python llava_next_mistral_7B.py --qa_csv chunks/chunk00/qa.csv --out_dir chunks/chunk00

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
# MODEL BLOCK - LLaVA-NeXT Mistral-7B
# =========================================================================== #
# apply_chat_template reads the template from the checkpoint, so the Mistral
# and Vicuna variants share this block unchanged.

MODEL_ID = os.environ.get("MODEL_ID", "llava-hf/llava-v1.6-mistral-7b-hf")
MODEL_KEY = "llava_next_mistral_7B"
DEFAULT_BATCH_SIZE = 16
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 16))


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
