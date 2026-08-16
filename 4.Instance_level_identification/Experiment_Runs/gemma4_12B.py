#!/usr/bin/env python3
"""
BioVQA C4 - instance-level identification inference.
Model: Gemma-4-12B-it

Runs one model over every (dataset x difficulty) QA.csv in a single process,
loading the model once. Self-contained: no shared imports between models.

  python run_gemma4_12b.py --data_root /path/to/4-Instance-Level-Identification

Everything below the MODEL BLOCK is identical across all five model scripts and
must stay that way - it is the controlled part of the experiment.
"""

import argparse
import json
import os
import time

import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================================== #
# EXPERIMENT CONSTANTS - identical in all five scripts
# =========================================================================== #

INSTRUCTION = (
    "\n"
    "* The image shows two photographs placed side by side.\n"
    "* Each photograph may contain more than one animal - consider only the "
    "main subject in the foreground.\n"
    "* Judge only from what is visible in the two photographs.\n"
    "* Answer with one word: Yes or No. Do not explain."
)

DATASETS = [
    "BelugaID-2022/beluga.coco",
    "HyenaID-2022/hyena.coco",
    "LeopardID-2022/leopard.coco",
    "WhalesharkID/whaleshark.coco",
    "GrevysZebra",
]

DIFFICULTIES = ["easy", "hard", "random"]


def build_prompt(question):
    return str(question).strip() + INSTRUCTION



# =========================================================================== #
# MODEL BLOCK - Gemma-4-12B-it
# =========================================================================== #

MODEL_ID = "google/gemma-4-12B-it"
MODEL_KEY = "gemma4_12B"
DEFAULT_BATCH_SIZE = 2
MAX_NEW_TOKENS = 15


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading " + MODEL_ID + " with SDPA...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


def run_batch(handle, batch):
    import torch
    model, processor = handle

    messages_batch = [
        [{"role": "user", "content": [
            {"type": "image", "image": os.path.abspath(path)},
            {"type": "text", "text": prompt},
        ]}]
        for path, prompt in batch
    ]

    inputs = processor.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False)

    trimmed = generated[:, input_len:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
    return [x.strip() for x in decoded]


# =========================================================================== #
# HARNESS - identical in all five scripts
# =========================================================================== #

def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + str(msg), flush=True)


def normalize(answer):
    """Yes/No extraction for the on-screen summary only. The raw model string
    is what gets written to disk; scoring happens in your evaluation script."""
    text = str(answer).strip().lower()
    for ch in "*#>`\"' \t":
        text = text.lstrip(ch)
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    return ""


def build_image_path(base_dir, local_path):
    local_path = str(local_path).strip()
    if os.path.isabs(local_path):
        return local_path
    return os.path.join(base_dir, local_path)


def collect_tasks(data_root, num, difficulties):
    """Expand data_root x DATASETS x difficulties into task directories."""
    found, missing = [], []
    total = num * 2
    for rel in DATASETS:
        for diff in difficulties:
            task_dir = os.path.join(data_root, rel,
                                    "instance_identification_%d_%s" % (total, diff))
            if os.path.isfile(os.path.join(task_dir, "QA.csv")):
                found.append(task_dir)
            else:
                missing.append(task_dir)
    return found, missing


def run_task(handle, task_dir, batch_size, overwrite, limit):
    """Run one QA.csv. Returns a summary dict, or None if skipped."""
    csv_path = os.path.join(task_dir, "QA.csv")
    out_path = os.path.splitext(csv_path)[0] + "_" + MODEL_KEY + ".csv"
    label = os.path.basename(os.path.normpath(task_dir))
    species = os.path.basename(os.path.dirname(os.path.normpath(task_dir)))

    if os.path.exists(out_path) and not overwrite:
        log("  already done, skipping: " + out_path)
        return None

    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[:limit].copy()

    col = MODEL_KEY + "_answer"
    df[col] = pd.NA

    # Row POSITION travels with each item, so answers are written back to the
    # row they came from. A skipped row can never shift everything after it.
    work, missing = [], 0
    for pos in range(len(df)):
        row = df.iloc[pos]
        try:
            image_path = build_image_path(task_dir, row["image_path"])
            if not os.path.exists(image_path):
                missing += 1
                continue
            qa = json.loads(row["qa_pair"])
            work.append((pos, image_path, build_prompt(qa["Q"])))
        except Exception as exc:
            log("  row %d unusable: %r" % (pos, exc))

    if missing:
        log("  warning: %d images missing on disk" % missing)
    if not work:
        log("  nothing to run")
        return None

    log("  %d items, batch size %d" % (len(work), batch_size))
    col_idx = df.columns.get_loc(col)
    done, failed_batches, sanity_checked = 0, 0, False
    recent = []

    for start in range(0, len(work), batch_size):
        chunk = work[start:start + batch_size]
        try:
            answers = run_batch(handle, [(p, q) for _, p, q in chunk])
        except Exception as exc:
            failed_batches += 1
            log("  batch at %d failed: %r" % (start, exc))
            continue
        for (pos, _, _), answer in zip(chunk, answers):
            df.iat[pos, col_idx] = answer
        recent.extend(answers)
        done += len(chunk)

        if start == 0:
            log("  first raw outputs: %r" % (answers,))

        # Fail fast. A model whose first answers are all unparseable is
        # misconfigured - wrong model class, broken chat template, randomly
        # initialised weights. Stop now instead of writing 15,000 rows of
        # garbage over the next several hours.
        if not sanity_checked and done >= min(20, len(work)):
            sanity_checked = True
            if not any(normalize(a) for a in recent):
                raise SystemExit(
                    "\nABORTING: none of the first %d answers parsed as Yes/No.\n"
                    "  raw sample: %r\n"
                    "  The model is almost certainly misconfigured. Check the\n"
                    "  transformers LOAD REPORT above for MISSING weights, and\n"
                    "  confirm the model class matches the checkpoint.\n"
                    % (done, recent[:6]))

        if done % 200 < batch_size:
            log("  %d/%d" % (done, len(work)))

    df.to_csv(out_path, index=False)

    truth = [json.loads(q)["A"] for q in df["qa_pair"]]
    preds = [normalize(a) for a in df[col]]
    scored = [(t, p) for t, p in zip(truth, preds) if p]
    n = len(scored)
    acc = sum(1 for t, p in scored if t == p) / n if n else float("nan")
    yes = sum(1 for _, p in scored if p == "Yes") / n if n else float("nan")

    log("  saved " + out_path)
    log("  accuracy %.3f | yes-rate %.3f | unparsed %d | missing %d | "
        "failed batches %d" % (acc, yes, len(preds) - n, missing, failed_batches))

    if n == 0:
        raise SystemExit(
            "\nABORTING: not one answer in this task parsed as Yes/No.\n"
            "  raw sample: %r\n"
            "  The model is almost certainly misconfigured. Check the\n"
            "  transformers LOAD REPORT for MISSING weights, and confirm the\n"
            "  model class matches the checkpoint (dense vs MoE).\n"
            % (list(df[col].head(6)),))

    return {"species": species, "task": label, "n": n,
            "accuracy": acc, "yes_rate": yes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/fs/ess/PAS2136/Rayeed/Bio-VQA/"
                                "4-Instance-Level-Identification")
    parser.add_argument("--difficulties", nargs="+", default=DIFFICULTIES,
                        choices=DIFFICULTIES)
    parser.add_argument("--num", type=int, default=500,
                        help="pairs per class used at generation time")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N rows of each task (smoke test)")
    args = parser.parse_args()

    tasks, missing = collect_tasks(args.data_root, args.num, args.difficulties)

    log("model  " + MODEL_ID)
    log("key    " + MODEL_KEY)
    log("tasks  %d found, %d not generated" % (len(tasks), len(missing)))
    for m in missing:
        log("       missing: " + m)
    if not tasks:
        raise SystemExit("No QA.csv found under " + args.data_root)

    print("-" * 78, flush=True)
    print("PROMPT SUFFIX (identical across all models and conditions):", flush=True)
    print(INSTRUCTION, flush=True)
    print("-" * 78, flush=True)

    handle = load_model()

    summaries = []
    for i, task_dir in enumerate(tasks, 1):
        log("[%d/%d] %s" % (i, len(tasks), task_dir))
        try:
            result = run_task(handle, task_dir, args.batch_size,
                              args.overwrite, args.limit)
        except Exception as exc:
            log("  TASK FAILED: %r" % (exc,))
            continue
        if result:
            summaries.append(result)

    print("", flush=True)
    print("=" * 78, flush=True)
    print("SUMMARY  " + MODEL_KEY, flush=True)
    print("=" * 78, flush=True)
    print("%-22s%-34s%6s%9s%9s" %
          ("species", "condition", "n", "acc", "yes"), flush=True)
    for s in summaries:
        print("%-22s%-34s%6d%9.3f%9.3f" %
              (s["species"], s["task"], s["n"], s["accuracy"], s["yes_rate"]),
              flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()