#!/usr/bin/env python3
"""
BioVQA C3 - Counting (Single + Cued + Positional tiers, one pass).
Model: Phi-3.5-vision-instruct

  python phi_3_5_vision.py --qa_csv chunks/chunk00/qa.csv --out_dir chunks/chunk00

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

# Panel framing is added only for the two-panel tiers. Apart from this one
# clause the scaffolding is identical across tiers, so the tier comparison is
# not contaminated by a prompt difference. split_qa.py also rewrites "the left
# image" to "the left photograph" in the question itself, because "image" is
# otherwise overloaded: the canvas is an image too.
PANEL_FRAME = (
    "This image consists of two separate photographs placed side by side.\n"
    "Count only within the photograph named in the question and ignore the "
    "other one.\n\n"
)

# Every clause removes a specific failure mode seen in counting VQA: hedged
# ranges ("about 20-25"), refusals on dense scenes, unit suffixes, and
# restatement of the question.
INSTRUCTION = (
    "\n\n"
    "Count every individual, including any that are partially visible or cut "
    "off at the edge of the photograph.\n"
    "Give a single best estimate even if you are uncertain.\n"
    "Respond with one integer and nothing else: no words, no units, no "
    "ranges.\n"
    "If you count seven, respond: 7"
)


def build_prompt(question, n_panels):
    frame = PANEL_FRAME if int(n_panels) > 1 else ""
    return frame + str(question).strip() + INSTRUCTION


# =========================================================================== #
# MODEL BLOCK - Phi-3.5-vision-instruct
# =========================================================================== #

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/Phi-3.5-vision-instruct")
MODEL_KEY = "phi35_vision"
# Phi-3-vision does not batch cleanly, so run_batch generates sequentially.
# Batch size only controls progress-log cadence here.
DEFAULT_BATCH_SIZE = 4
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 8))
NUM_CROPS = int(os.environ.get("NUM_CROPS", 16))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPL", "eager")


def apply_cache_shims():
    """
    Phi-3.5-vision's remote modelling code was written against an OLD
    transformers Cache API. Newer transformers progressively removed those
    methods, so generate() dies with:
        'DynamicCache' object has no attribute 'seen_tokens'
        'DynamicCache' object has no attribute 'get_usable_length'
    These shims re-attach the legacy names on top of the current API.
    """
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

        if not hasattr(DynamicCache, "get_max_length"):
            if hasattr(DynamicCache, "get_max_cache_shape"):
                DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
            else:
                DynamicCache.get_max_length = lambda self: None

        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):
                if hasattr(self, "get_max_cache_shape"):
                    max_length = self.get_max_cache_shape()
                elif hasattr(self, "get_max_length"):
                    max_length = self.get_max_length()
                else:
                    max_length = None
                previous_seq_length = self.get_seq_length(layer_idx)
                if (max_length is not None
                        and previous_seq_length + new_seq_length > max_length):
                    return max_length - new_seq_length
                return previous_seq_length
            DynamicCache.get_usable_length = _get_usable_length

        print("  DynamicCache compatibility shims applied", flush=True)
    except Exception as exc:
        print("  [WARN] could not apply DynamicCache shim: %r" % (exc,), flush=True)


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    apply_cache_shims()

    print("Loading " + MODEL_ID + " (trust_remote_code, attn="
          + ATTN_IMPLEMENTATION + ")...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        _attn_implementation=ATTN_IMPLEMENTATION,
    ).eval()
    print("  loaded class: " + type(model).__name__, flush=True)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        num_crops=NUM_CROPS,
    )
    return model, processor


def run_batch(handle, batch):
    import torch
    model, processor = handle

    answers = []
    for path, prompt in batch:
        pil_image = Image.open(path).convert("RGB")

        # Phi-3 vision image placeholder format
        messages = [{"role": "user", "content": "<|image_1|>\n" + prompt}]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(text, [pil_image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        answers.append(processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip())

    return answers


# =========================================================================== #
# HARNESS - identical in all eighteen scripts
# =========================================================================== #

INT_RE = re.compile(r"-?\d+")


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def resolve_int(answer):
    """First integer anywhere in the raw string, or None.

    Models routinely wrap the number: "There are 13 bees in this image".
    The raw string is preserved in <KEY>_answer_unfiltered; this resolved
    integer goes in <KEY>_answer, so nothing is lost and evaluate.py never
    has to re-parse free text."""
    m = INT_RE.search(str(answer).replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
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
            v = resolve_int(answer)
            df.iat[pos, col_idx] = pd.NA if v is None else v
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. A model whose first answers contain no integer at all is
        # misconfigured - wrong class, broken chat template, dead environment.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if all(resolve_int(a) is None for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers contained an "
                    "integer.\n  raw sample: %r\n" % (done, recent[:6]))

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
    truth = pd.to_numeric(df["answer"], errors="coerce")
    pred = pd.to_numeric(df[col], errors="coerce")
    ok = truth.notna() & pred.notna()
    n = int(ok.sum())
    unparsed = int(len(df) - pred.notna().sum())
    em = float((truth[ok] == pred[ok]).mean()) if n else float("nan")
    mae = float((truth[ok] - pred[ok]).abs().mean()) if n else float("nan")

    record = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "task": "counting",
        "qa_csv": os.path.abspath(qa_csv),
        "out_csv": os.path.abspath(out_path),
        "rows": int(len(df)),
        "scored": n,
        "unparsed": unparsed,
        "missing_images": missing,
        "failed_batches": failed_batches,
        "batch_size": int(batch_size),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "prompt_instruction": INSTRUCTION,
        "prompt_panel_frame": PANEL_FRAME,
        "indicative_EM": em,
        "indicative_MAE": mae,
        "seconds": round(time.time() - t0, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log("saved " + json_path)
    log("indicative EM %.3f | MAE %.2f | unparsed %d | missing %d | "
        "failed batches %d  (evaluate.py is authoritative)"
        % (em, mae, unparsed, missing, failed_batches))


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
    print("PROMPT, single-panel tier:", flush=True)
    print(build_prompt("How many bees are there in this image?", 1), flush=True)
    print("", flush=True)
    print("PROMPT, two-panel tiers:", flush=True)
    print(build_prompt("How many bees are there in the left photograph?", 2),
          flush=True)
    print("-" * 78, flush=True)

    handle = load_model()
    run_all(handle, args.qa_csv, out_dir, args.batch_size,
            args.overwrite, args.limit)


if __name__ == "__main__":
    main()
