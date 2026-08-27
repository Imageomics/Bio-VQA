#!/usr/bin/env python3
"""
Bio-VQA / Spatial-Visual-Reasoning -- Optical Character Recognition (OCR) task.

For each NEON beetle tray image:
  1. ask Moondream to detect "Barcode"
  2. draw the detected box onto the image (visual hint for the OCR question)
  3. save the plotted image to  <output_dir>/Barcodes/<stem>.jpg
  4. append a row to          <output_dir>/Barcodes_task_<id>.csv

The per-task CSVs are awk-merged into <output_dir>/Barcodes.csv by the SBATCH
script, and qa.csv is built from that.

Ground truth for the OCR answer is the NEON specimen ID, which is the image
filename stem -- not a model read. Moondream is used only for localisation.

Output filenames MUST stay <stem>.jpg: qa.csv is frozen (models already run).

"""

import argparse
import csv
import os
import sys
import time

from PIL import Image, ImageDraw
from tqdm import tqdm

import moondream as md

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
BOX_COLOR = (255, 0, 0)  # RGB red (PIL)
CSV_FIELDS = ["image", "barcode", "barcode_bbox", "request_id"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def list_images(image_dir):
    names = [f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)]
    return sorted(names)


def to_pixels(obj, w, h):
    """Moondream cloud returns normalised coords; fall through if already absolute."""
    try:
        vals = [float(obj["x_min"]), float(obj["y_min"]),
                float(obj["x_max"]), float(obj["y_max"])]
    except (KeyError, TypeError, ValueError):
        return None

    if max(vals) <= 1.0:
        x0, y0, x1, y1 = vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h
    else:
        x0, y0, x1, y1 = vals

    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def pad_box(box, w, h, frac):
    if frac <= 0:
        return box
    x0, y0, x1, y1 = box
    dw, dh = int((x1 - x0) * frac), int((y1 - y0) * frac)
    return (max(0, x0 - dw), max(0, y0 - dh),
            min(w - 1, x1 + dw), min(h - 1, y1 + dh))


def detect_with_retry(model, image, prompt, retries, backoff):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return model.detect(image, prompt)
        except Exception as e:  # network / rate-limit / transient API errors
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
    raise last_err


def already_done(csv_path):
    """Resume support: image paths already recorded in this task's CSV."""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("image"):
                done.add(row["image"])
    return done


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Moondream barcode detection + plotting for the OCR task")
    ap.add_argument("--base_dir", required=True,
                    help="Dataset root containing the image subdir")
    ap.add_argument("--images_subdir", default="group_images_resized",
                    help="Image folder under base_dir; also the prefix written to the CSV")
    ap.add_argument("--output_dir", required=True,
                    help="Task root, e.g. '.../Optical Character Recognition - OCR'")
    ap.add_argument("--images_out_subdir", default="Barcodes",
                    help="Where plotted images go under output_dir. Must match qa.csv.")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--num_tasks", type=int, default=1)
    ap.add_argument("--prompt", default="Barcode", help="Moondream detect prompt")
    ap.add_argument("--thickness", type=int, default=0,
                    help="Box line width in px. 0 = scale to image size.")
    ap.add_argument("--box_pad_frac", type=float, default=0.0,
                    help="Grow the box by this fraction. Moondream boxes the bars only; "
                         "raise this if you want the printed text inside the box. "
                         "Changing it changes the images, so re-run inference if you do.")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0, help="Base seconds for exponential backoff")
    ap.add_argument("--sleep", type=float, default=0.0, help="Fixed delay between API calls")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N images (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip images already present in this task's CSV")
    args = ap.parse_args()

    api_key = os.environ.get("MOONDREAM_API_KEY")
    if not api_key:
        sys.exit("MOONDREAM_API_KEY is not set in the environment.")

    image_dir = os.path.join(args.base_dir, args.images_subdir)
    if not os.path.isdir(image_dir):
        sys.exit(f"Image directory not found: {image_dir}")

    out_img_dir = os.path.join(args.output_dir, args.images_out_subdir)
    os.makedirs(out_img_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"Barcodes_task_{args.task_id}.csv")

    # Strided shard so every array task gets a comparable mix of file sizes.
    all_names = list_images(image_dir)
    shard = all_names[args.task_id::args.num_tasks]
    if args.limit:
        shard = shard[:args.limit]

    done = already_done(csv_path) if args.resume else set()
    write_header = not (args.resume and os.path.exists(csv_path))
    mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"

    model = md.vl(api_key=api_key)

    n_ok = 0
    skipped = {"no_detection": 0, "bad_box": 0, "unreadable": 0, "api_error": 0, "resumed": 0}

    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        desc = f"task {args.task_id}/{args.num_tasks}"
        for name in tqdm(shard, desc=desc, unit="img"):
            rel_path = os.path.join(args.images_subdir, name)
            if rel_path in done:
                skipped["resumed"] += 1
                continue

            stem = os.path.splitext(name)[0]

            try:
                image = Image.open(os.path.join(image_dir, name)).convert("RGB")
            except Exception:
                skipped["unreadable"] += 1
                continue

            w, h = image.size

            try:
                result = detect_with_retry(model, image, args.prompt, args.retries, args.backoff)
            except Exception as e:
                skipped["api_error"] += 1
                tqdm.write(f"[api] {name}: {type(e).__name__}: {e}")
                continue

            objects = result.get("objects", []) or []
            request_id = result.get("request_id", "") or ""

            # Keep the largest detection if Moondream returned several.
            best, best_area = None, -1
            for obj in objects:
                px = to_pixels(obj, w, h)
                if px is None:
                    continue
                area = (px[2] - px[0]) * (px[3] - px[1])
                if area > best_area:
                    best, best_area = px, area

            if best is None:
                skipped["no_detection" if not objects else "bad_box"] += 1
                continue

            box = pad_box(best, w, h, args.box_pad_frac)
            x0, y0, x1, y1 = box

            width = args.thickness if args.thickness > 0 else max(3, round(min(w, h) * 0.006))
            draw = ImageDraw.Draw(image)
            draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=width)

            image.save(os.path.join(out_img_dir, f"{stem}.jpg"),
                       "JPEG", quality=args.jpeg_quality)

            # Schema matches the existing Barcodes.csv so the awk merge still works.
            writer.writerow({
                "image": rel_path,
                "barcode": stem,
                "barcode_bbox": str([{"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}]),
                "request_id": request_id,
            })
            f.flush()  # survive an 8h wall-time kill
            n_ok += 1

            if args.sleep:
                time.sleep(args.sleep)

    print(f"\n[task {args.task_id}] wrote {n_ok} rows -> {csv_path}")
    print(f"[task {args.task_id}] plotted images -> {out_img_dir}")
    print(f"[task {args.task_id}] shard size {len(shard)} | skipped {skipped}")


if __name__ == "__main__":
    main()