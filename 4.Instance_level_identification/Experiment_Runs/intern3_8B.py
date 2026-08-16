#!/usr/bin/env python3
"""
BioVQA C4 - instance-level identification inference.
Model: InternVL3-8B

Runs one model over every (dataset x difficulty) QA.csv in a single process,
loading the model once. Self-contained: no shared imports between models.

  python intern3_8B.py --data_root /path/to/4-Instance-Level-Identification

Everything below the MODEL BLOCK is identical across all five model scripts and
must stay that way - it is the controlled part of the experiment.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

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
# MODEL BLOCK - InternVL3-8B
# =========================================================================== #

# Repo name is overridable without editing this file, because OpenGVLab has
# shipped both "-Instruct" and bare variants:
#   MODEL_ID=OpenGVLab/InternVL3-8B sbatch intern3_8B.sh
MODEL_ID = os.environ.get("MODEL_ID", "OpenGVLab/InternVL3-8B-Instruct")
MODEL_KEY = "internvl3_8B-Instruct"
DEFAULT_BATCH_SIZE = 4
MAX_NEW_TOKENS = 15

# Tiles per image. 12 is the InternVL default; 6 is kept here to match the 38B
# script, so the InternVL family is internally consistent and the tiling budget
# is not a hidden variable between model sizes.
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
    """
    The full DATASETS x difficulties grid in fixed order, each entry tagged with
    whether its QA.csv exists.

    Indexing the whole grid rather than only the generated tasks is deliberate:
    --task_index must mean the same thing on every submission, so a condition
    that has not been generated yet cannot silently shift every later index and
    make a SLURM array write the wrong species into the wrong file.
    """
    grid = []
    total = num * 2
    for rel in DATASETS:
        for diff in difficulties:
            task_dir = os.path.join(data_root, rel,
                                    "instance_identification_%d_%s" % (total, diff))
            grid.append((task_dir, os.path.isfile(os.path.join(task_dir, "QA.csv"))))
    return grid


IO_PREFETCH_BATCHES = 3


def warm_file(path):
    """Pull a file into the OS page cache. Runs on background threads so that
    Lustre read latency for the NEXT batches overlaps GPU work on this one."""
    try:
        with open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
    except Exception:
        pass


def run_task(handle, task_dir, batch_size, overwrite, limit, io_workers=4):
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

    log("  %d items, batch size %d, io workers %d"
        % (len(work), batch_size, io_workers))
    col_idx = df.columns.get_loc(col)
    done, failed_batches, sanity_checked = 0, 0, False
    recent = []

    pool = ThreadPoolExecutor(max_workers=io_workers) if io_workers > 0 else None
    warm_ptr = 0

    for start in range(0, len(work), batch_size):
        # queue reads a few batches ahead of where the GPU currently is
        if pool is not None:
            target = min(len(work),
                         start + batch_size * (1 + IO_PREFETCH_BATCHES))
            while warm_ptr < target:
                pool.submit(warm_file, work[warm_ptr][1])
                warm_ptr += 1

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

    if pool is not None:
        pool.shutdown(wait=False)

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
    parser.add_argument("--io_workers", type=int, default=4,
                        help="background threads prefetching images into page "
                             "cache; 0 disables")
    parser.add_argument("--task_index", type=int, default=-1,
                        help="run only the Nth task of the grid (0-based); "
                             "-1 runs all. Use with a SLURM array.")
    args = parser.parse_args()

    grid = collect_tasks(args.data_root, args.num, args.difficulties)

    log("model  " + MODEL_ID)
    log("key    " + MODEL_KEY)

    if args.task_index >= 0:
        if args.task_index >= len(grid):
            raise SystemExit("--task_index %d out of range (grid has %d entries)"
                             % (args.task_index, len(grid)))
        selected = [grid[args.task_index]]
        log("task   index %d of %d: %s"
            % (args.task_index, len(grid), selected[0][0]))
    else:
        selected = grid
        log("grid   %d entries" % len(grid))

    tasks = [d for d, ok in selected if ok]
    missing = [d for d, ok in selected if not ok]

    log("tasks  %d found, %d not generated" % (len(tasks), len(missing)))
    for m in missing:
        log("       missing: " + m)
    if not tasks:
        if args.task_index >= 0:
            raise SystemExit(0)   # nothing to do for this array element
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
                              args.overwrite, args.limit, args.io_workers)
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
